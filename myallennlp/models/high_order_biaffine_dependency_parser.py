from typing import Dict, Optional, Tuple, Any, List
import logging
import copy

from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules import Dropout
import numpy

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding, InputVariationalDropout
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, Activation
from allennlp.nn.util import get_text_field_mask, get_range_vector
from allennlp.nn.util import get_device_of, masked_log_softmax, get_lengths_from_binary_sequence_mask,masked_softmax
from allennlp.nn.chu_liu_edmonds import decode_mst
from allennlp.training.metrics import AttachmentScores
from myallennlp.metric import IterativeAttachmentScores
from myallennlp.modules.reparametrization import masked_cross_entropy, masked_gumbel_softmax,inplace_masked_gumbel_softmax,sample_gumbel
from myallennlp.modules.refiner import ScoreBasedRefiner
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

POS_TO_IGNORE = {'``', "''", ':', ',', '.', 'PU', 'PUNCT', 'SYM'}

@Model.register("high_order_parser")
class HighOrderBiaffineDependencyParser(Model):
    """
    This dependency parser follows the model of
    ` Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)
    <https://arxiv.org/abs/1611.01734>`_ .

    Word representations are generated using a bidirectional LSTM,
    followed by separate biaffine classifiers for pairs of words,
    predicting whether a directed arc exists between the two words
    and the dependency label the arc should have. Decoding can either
    be done greedily, or the optimial Minimum Spanning Tree can be
    decoded using Edmond's algorithm by viewing the dependency tree as
    a MST on a fully connected graph, where nodes are words and edges
    are scored dependency arcs.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use to generate representations
        of tokens.
    tag_representation_dim : ``int``, required.
        The dimension of the MLPs used for dependency tag prediction.
    arc_representation_dim : ``int``, required.
        The dimension of the MLPs used for head arc prediction.
    tag_feedforward : ``FeedForward``, optional, (default = None).
        The feedforward network used to produce tag representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    arc_feedforward : ``FeedForward``, optional, (default = None).
        The feedforward network used to produce arc representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    pos_tag_embedding : ``Embedding``, optional.
        Used to embed the ``pos_tags`` ``SequenceLabelField`` we get as input to the model.
    use_mst_decoding_for_validation : ``bool``, optional (default = True).
        Whether to use Edmond's algorithm to find the optimal minimum spanning tree during validation.
        If false, decoding is greedy.
    dropout : ``float``, optional, (default = 0.0)
        The variational dropout applied to the output of the encoder and MLP layers.
    input_dropout : ``float``, optional, (default = 0.0)
        The dropout applied to the embedded text input.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 tag_representation_dim: int,
                 arc_representation_dim: int,
                 refiner: Seq2SeqEncoder,
                 tag_feedforward: FeedForward = None,
                 arc_feedforward: FeedForward = None,
                 pos_tag_embedding: Embedding = None,
                 use_mst_decoding_for_validation: bool = True,
                 gumbel_t:float=1.0,
                 last_only:bool=False,
                 learn_score:bool=True,
                 stational_loss:bool = False,
                 dropout: float = 0.0,
                 relaxed_head:bool=False,
                 input_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(HighOrderBiaffineDependencyParser, self).__init__(vocab, regularizer)
        self.gumbel_t = gumbel_t
        self.learn_score = learn_score
        self.relaxed_head = relaxed_head
        self.last_only = last_only
        self.refiner = refiner
        self.stational_loss = stational_loss
        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        encoder_dim = encoder.get_output_dim()
        self.head_arc_feedforward = arc_feedforward or \
                                        FeedForward(encoder_dim, 1,
                                                    arc_representation_dim,
                                                    Activation.by_name("elu")())
        self.child_arc_feedforward = copy.deepcopy(self.head_arc_feedforward)

        self.arc_attention = BilinearMatrixAttention(arc_representation_dim,
                                                     arc_representation_dim,
                                                     use_input_biases=True)

        num_labels = self.vocab.get_vocab_size("head_tags")

        self.head_tag_feedforward = tag_feedforward or \
                                        FeedForward(encoder_dim, 1,
                                                    tag_representation_dim,
                                                    Activation.by_name("elu")())
        self.child_tag_feedforward = copy.deepcopy(self.head_tag_feedforward)

        self.tag_bilinear = torch.nn.modules.Bilinear(tag_representation_dim,
                                                      tag_representation_dim,
                                                      num_labels)
        self.refiner.set_tagger(self.tag_bilinear)

        self._num_labels = num_labels
        self._pos_tag_embedding = pos_tag_embedding or None
        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)
        self._head_sentinel = torch.nn.Parameter(torch.randn([1, 1, encoder.get_output_dim()]))

        representation_dim = text_field_embedder.get_output_dim()
        if pos_tag_embedding is not None:
            representation_dim += pos_tag_embedding.get_output_dim()

        check_dimensions_match(representation_dim, encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")

        check_dimensions_match(tag_representation_dim, self.head_tag_feedforward.get_output_dim(),
                               "tag representation dim", "tag feedforward output dim")
        check_dimensions_match(arc_representation_dim, self.head_arc_feedforward.get_output_dim(),
                               "arc representation dim", "arc feedforward output dim")

        self.use_mst_decoding_for_validation = use_mst_decoding_for_validation

        tags = self.vocab.get_token_to_index_vocabulary("pos")
        punctuation_tag_indices = {tag: index for tag, index in tags.items() if tag in POS_TO_IGNORE}
        self._pos_to_ignore = set(punctuation_tag_indices.values())
        logger.info(f"Found POS tags correspoding to the following punctuation : {punctuation_tag_indices}. "
                    "Ignoring words with these POS tags for evaluation.")

        self._attachment_scores = IterativeAttachmentScores()
        initializer(self)


    @overrides
    def forward(self,  # type: ignore
                words: Dict[str, torch.LongTensor],
                pos_tags: torch.LongTensor,
                metadata: List[Dict[str, Any]],
                head_tags: torch.LongTensor = None,
                head_indices: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        words : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, sequence_length)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        pos_tags : ``torch.LongTensor``, required.
            The output of a ``SequenceLabelField`` containing POS tags.
            POS tags are required regardless of whether they are used in the model,
            because they are used to filter the evaluation metric to only consider
            heads of words which are not punctuation.
        head_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels for the arcs
            in the dependency parse. Has shape ``(batch_size, sequence_length)``.
        head_indices : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer indices denoting the parent of every
            word in the dependency parse. Has shape ``(batch_size, sequence_length)``.

        Returns
        -------
        An output dictionary consisting of:
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised.
        arc_loss : ``torch.FloatTensor``
            The loss contribution from the unlabeled arcs.
        loss : ``torch.FloatTensor``, optional
            The loss contribution from predicting the dependency
            tags for the gold arcs.
        heads : ``torch.FloatTensor``
            The predicted head indices for each word. A tensor
            of shape (batch_size, sequence_length).
        head_types : ``torch.FloatTensor``
            The predicted head types for each arc. A tensor
            of shape (batch_size, sequence_length).
        mask : ``torch.LongTensor``
            A mask denoting the padded elements in the batch.
        """
        embedded_text_input = self.text_field_embedder(words)
        if pos_tags is not None and self._pos_tag_embedding is not None:
            embedded_pos_tags = self._pos_tag_embedding(pos_tags)
            embedded_text_input = torch.cat([embedded_text_input, embedded_pos_tags], -1)
        elif self._pos_tag_embedding is not None:
            raise ConfigurationError("Model uses a POS embedding, but no POS tags were passed.")

        mask = get_text_field_mask(words)
        embedded_text_input = self._input_dropout(embedded_text_input)
        encoded_text = self.encoder(embedded_text_input, mask)

        batch_size, _, encoding_dim = encoded_text.size()

        head_sentinel = self._head_sentinel.expand(batch_size, 1, encoding_dim)
        # Concatenate the head sentinel onto the sentence representation.
        encoded_text = torch.cat([head_sentinel, encoded_text], 1)
        mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
        if head_indices is not None:
            head_indices = torch.cat([head_indices.new_zeros(batch_size, 1), head_indices], 1)
        if head_tags is not None:
            head_tags = torch.cat([head_tags.new_zeros(batch_size, 1), head_tags], 1)
        float_mask = mask.float()
        encoded_text = self._dropout(encoded_text)

        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc_representation = self._dropout(self.head_arc_feedforward(encoded_text))
        child_arc_representation = self._dropout(self.child_arc_feedforward(encoded_text))

        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag_representation = self._dropout(self.head_tag_feedforward(encoded_text))
        child_tag_representation = self._dropout(self.child_tag_feedforward(encoded_text))
        # shape (batch_size, sequence_length, sequence_length)
        attended_arcs = self.arc_attention(head_arc_representation,
                                           child_arc_representation)

        high_order_weights = self.refiner.get_high_order_weights(encoded_text,mask)

        minus_inf = -1e8
        minus_mask = (1 - float_mask) * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        attended_arcs = masked_log_softmax(attended_arcs, mask=mask.unsqueeze(-1))
        # Mask padded tokens, because we only want to consider actual words as heads.
        if self.training:
            relaxed_head = masked_gumbel_softmax(attended_arcs, tau=self.gumbel_t,
                                                 mask=mask.unsqueeze(-1))
        else:
            relaxed_head = masked_softmax(attended_arcs, mask=mask.unsqueeze(-1))

        if self.relaxed_head:

            head_tag_logits = self._get_head_tags_with_relaxed(head_tag_representation,
                                                               child_tag_representation,
                                                               relaxed_head)
        else:
            head_tag_logits = self._get_head_tags(head_tag_representation,
                                                  child_tag_representation,
                                                  head_indices)

        relaxed_head_tags = masked_softmax(head_tag_logits, mask.unsqueeze(-1), dim=2)

        if self.training or not self.use_mst_decoding_for_validation:
            attended_arcs_list, relaxed_head_list ,head_tag_logits_list,relaxed_head_tags_list = self.refiner(head_tag_representation,child_tag_representation,attended_arcs,relaxed_head,head_tag_logits, relaxed_head_tags,mask,high_order_weights = high_order_weights,head_indices=head_indices,
                                                                  head_tags = head_tags)
            if  self.training and self.last_only:
                attended_arcs_list = [attended_arcs_list[-1]]
                relaxed_head_list = [relaxed_head_list[-1]]
            predicted_heads_list = []
            predicted_head_tags_list = []

            for relaxed_head_tags,relaxed_head in zip(relaxed_head_tags_list,relaxed_head_list):
                # Given the greedily predicted heads, decode their dependency tags.
                # shape (batch_size, sequence_length, num_head_tags)

                _, predicted_heads = relaxed_head.max(2)
                _, predicted_head_tags = relaxed_head_tags.max(2)
                predicted_heads_list.append(predicted_heads)
                predicted_head_tags_list.append(predicted_head_tags)
        else:
            attended_arcs_list, relaxed_head_list ,head_tag_logits_list,relaxed_head_tags_list = self.refiner(head_tag_representation,child_tag_representation,attended_arcs,relaxed_head,head_tag_logits, relaxed_head_tags,mask,high_order_weights = high_order_weights)
            predicted_heads_list = []
            predicted_head_tags_list = []

            for attended_arcs,relaxed_head in zip(attended_arcs_list,relaxed_head_list):

                predicted_heads, predicted_head_tags = self._mst_decode(head_tag_representation,
                                                                        child_tag_representation,
                                                                        attended_arcs,
                                                                        mask)
                predicted_heads_list.append(predicted_heads)
                predicted_head_tags_list.append(predicted_head_tags)


        loss = 0
        for i, (attended_arcs,relaxed_head,head_tag_logits,relaxed_head_tags,predicted_heads,predicted_head_tags) in enumerate(zip(attended_arcs_list,relaxed_head_list,head_tag_logits_list,relaxed_head_tags_list,predicted_heads_list,predicted_head_tags_list)):
            if head_indices is not None and head_tags is not None :
                if self.training:
                    loss =loss + 1/(len(attended_arcs_list)-i) * self._max_margin_loss(attended_arcs=attended_arcs,
                                                            head_indices=head_indices,
                                                            head_tags=head_tags,
                                                            relaxed_head=relaxed_head,
                                                            head_tag_logits=head_tag_logits,
                                                            relaxed_head_tags=relaxed_head_tags,
                                                            mask=mask,
                                                            high_order_weights=high_order_weights)
                else:
                    arc_nll, tag_nll = self._construct_loss(head_tag_representation=head_tag_representation,
                                                            child_tag_representation=child_tag_representation,
                                                            attended_arcs=attended_arcs,
                                                            head_indices=predicted_heads.long(),
                                                            head_tags=predicted_head_tags.long(),
                                                            mask=mask)
                    loss = loss+ 1/(len(attended_arcs_list)-i) * (arc_nll + tag_nll)
                evaluation_mask = self._get_mask_for_eval(mask[:, 1:], pos_tags)
                # We calculate attatchment scores for the whole sentence
                # but excluding the symbolic ROOT token at the start,
                # which is why we start from the second element in the sequence.
                self._attachment_scores(predicted_heads[:, 1:],
                                        predicted_head_tags[:, 1:],
                                        head_indices[:, 1:],
                                        head_tags[:, 1:],
                                        evaluation_mask,
                                        n_iteration = i,
                                        refine_lr=self.refiner.get_lr()  )
            else:
                arc_nll, tag_nll = self._construct_loss(head_tag_representation=head_tag_representation,
                                                        child_tag_representation=child_tag_representation,
                                                        attended_arcs=attended_arcs,
                                                        head_indices=predicted_heads.long(),
                                                        head_tags=predicted_head_tags.long(),
                                                        mask=mask)
                loss = loss + arc_nll + tag_nll

        output_dict = {
                "heads": predicted_heads,
                "head_tags": predicted_head_tags,
                "loss": loss,
                "mask": mask,
                "words": [meta["words"] for meta in metadata],
                "pos": [meta["pos"] for meta in metadata]
                }

        return output_dict


    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        head_tags = output_dict.pop("head_tags").cpu().detach().numpy()
        heads = output_dict.pop("heads").cpu().detach().numpy()
        mask = output_dict.pop("mask")
        lengths = get_lengths_from_binary_sequence_mask(mask)
        head_tag_labels = []
        head_indices = []
        for instance_heads, instance_tags, length in zip(heads, head_tags, lengths):
            instance_heads = list(instance_heads[1:length])
            instance_tags = instance_tags[1:length]
            labels = [self.vocab.get_token_from_index(label, "head_tags")
                      for label in instance_tags]
            head_tag_labels.append(labels)
            head_indices.append(instance_heads)

        output_dict["predicted_dependencies"] = head_tag_labels
        output_dict["predicted_heads"] = head_indices
        return output_dict

    def _construct_loss(self,
                        head_tag_representation: torch.Tensor,
                        child_tag_representation: torch.Tensor,
                        attended_arcs: torch.Tensor,
                        head_indices: torch.Tensor,
                        head_tags: torch.Tensor,
                        mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachements of a given word to all other words.
        head_indices : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The indices of the heads for every word.
        head_tags : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The dependency labels of the heads for every word.
        mask : ``torch.Tensor``, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        Returns
        -------
        arc_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc loss.
        tag_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc tag loss.
        """
        float_mask = mask.float()
        batch_size, sequence_length, _ = attended_arcs.size()
        # shape (batch_size, 1)
        range_vector = get_range_vector(batch_size, get_device_of(attended_arcs)).unsqueeze(1)
        # shape (batch_size, sequence_length, sequence_length)
        normalised_arc_logits = masked_log_softmax(attended_arcs,
                                                   mask) * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)

        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(head_tag_representation, child_tag_representation, head_indices)
        normalised_head_tag_logits = masked_log_softmax(head_tag_logits,
                                                        mask.unsqueeze(-1)) * float_mask.unsqueeze(-1)
        # index matrix with shape (batch, sequence_length)
        timestep_index = get_range_vector(sequence_length, get_device_of(attended_arcs))
        child_index = timestep_index.view(1, sequence_length).expand(batch_size, sequence_length).long()
        # shape (batch_size, sequence_length)
        arc_loss = normalised_arc_logits[range_vector, child_index, head_indices]
        tag_loss = normalised_head_tag_logits[range_vector, child_index, head_tags]
        # We don't care about predictions for the symbolic ROOT token's head,
        # so we remove it from the loss.
        arc_loss = arc_loss[:, 1:]
        tag_loss = tag_loss[:, 1:]

        # The number of valid positions is equal to the number of unmasked elements minus
        # 1 per sequence in the batch, to account for the symbolic HEAD token.
        valid_positions = mask.sum() - batch_size

        arc_nll = -arc_loss.sum() / valid_positions.float()
        tag_nll = -tag_loss.sum() / valid_positions.float()
        return arc_nll, tag_nll

    def _output_distance(self,
                        attended_arcs: torch.Tensor,
                        relaxed_head_tag:torch.Tensor,
                        head_indices: torch.Tensor,
                        head_tags: torch.Tensor,
                        mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachements of a given word to all other words.
        head_indices : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The indices of the heads for every word.
        head_tags : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The dependency labels of the heads for every word.
        mask : ``torch.Tensor``, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        Returns
        -------
        arc_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc loss.
        tag_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc tag loss.
        """
        float_mask = mask.float()
        batch_size, sequence_length, _ = attended_arcs.size()

        # shape (batch_size, 1)
        range_vector = get_range_vector(batch_size, get_device_of(attended_arcs)).unsqueeze(1)
        # shape (batch_size, sequence_length, sequence_length)
        normalised_arc_logits =  masked_log_softmax(attended_arcs,
                                                   mask) * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)

        # shape (batch_size, sequence_length, num_head_tags)
        normalised_head_tag_logits =(relaxed_head_tag+1e-10).log()*  float_mask.unsqueeze(-1)

        # index matrix with shape (batch, sequence_length)
        timestep_index = get_range_vector(sequence_length, get_device_of(attended_arcs))
        child_index = timestep_index.view(1, sequence_length).expand(batch_size, sequence_length).long()
        # shape (batch_size, sequence_length)
        arc_loss = normalised_arc_logits[range_vector, child_index, head_indices]
        tag_loss = normalised_head_tag_logits[range_vector, child_index, head_tags]
        # We don't care about predictions for the symbolic ROOT token's head,
        # so we remove it from the loss.
        arc_loss = arc_loss[:, 1:]
        tag_loss = tag_loss[:, 1:]

        # The number of valid positions is equal to the number of unmasked elements minus
        # 1 per sequence in the batch, to account for the symbolic HEAD token.

       # print ("relaxed_head",relaxed_head.size())
      #  print ("relaxed_head_tag",relaxed_head_tag.size())
     #   print ("arc_loss",arc_loss.size())
    #    print ("tag_loss",tag_loss.size())
        arc_nll = -arc_loss.unsqueeze(2)
        tag_nll = -tag_loss.unsqueeze(2)
        return arc_nll ,tag_nll

    def _max_margin_loss(self,
                        attended_arcs: torch.Tensor,
                        head_indices: torch.Tensor,
                        head_tags: torch.Tensor,
                         relaxed_head: torch.Tensor,
                         head_tag_logits: torch.Tensor,
                         relaxed_head_tags:torch.Tensor,
                        mask: torch.Tensor,
                         high_order_weights:torch.Tensor=None) -> torch.Tensor:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.

        Parameters
        ----------
        head_tag_logits : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_num),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachements of a given word to all other words.
        head_indices : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The indices of the heads for every word.
        head_tags : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The dependency labels of the heads for every word.
        relaxed_head : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length).
            The indices of the heads for every word.
        relaxed_head_tags : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, num_head_tags).
            The dependency labels of the heads for every word.
        mask : ``torch.Tensor``, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.
        high_order_features : ``torch.Tensor``, required.
            A mask of shape (batch_size, sequence_length,sequence_length,num_relations), denoting unpadded
            elements in the sequence.

        Returns
        -------
        arc_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc loss.
        tag_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc tag loss.
        """
        batch_size, sequence_length, _ = attended_arcs.size()

        float_mask = mask.float()
        soft_head_indices = torch.zeros(batch_size,sequence_length,sequence_length,device=head_indices.device)
        soft_head_indices.scatter_(2, head_indices.unsqueeze(2), 1)
        soft_head_indices = soft_head_indices* float_mask.unsqueeze(2)* float_mask.unsqueeze(1)

        soft_head_tags = torch.zeros(batch_size,sequence_length,self._num_labels,device=head_tags.device)
        soft_head_tags.scatter_(2, head_tags.unsqueeze(2), 1)
        soft_head_tags = soft_head_tags * float_mask.unsqueeze(2)

        gold_arc_nll, gold_tag_nll,gold_sibling_score , gold_grand_pa_score = self._score_per_instance(head_tag_logits=head_tag_logits,
                                                                  attended_arcs=attended_arcs,
                                                                  relaxed_head=soft_head_indices,
                                                                  relaxed_head_tags=soft_head_tags,
                                                                  mask=mask,
                                                                  high_order_weights=high_order_weights)
        arc_nll, tag_nll, sibling_score , grand_pa_score = self._score_per_instance(head_tag_logits=head_tag_logits,
                                                    attended_arcs=attended_arcs,
                                                    relaxed_head=relaxed_head,
                                                    relaxed_head_tags=relaxed_head_tags,
                                                    mask=mask,
                                                    high_order_weights=high_order_weights)
        delta_arc,delta_tag = self._output_distance(attended_arcs,relaxed_head_tags,head_indices,head_tags,mask)

        sibling_delta, grand_pa_delta = self.refiner._high_order_output_distance(relaxed_head,relaxed_head_tags,soft_head_indices,soft_head_tags,mask)
        # The number of valid positions is equal to the number of unmasked elements minus
        # 1 per sequence in the batch, to account for the symbolic HEAD token.
        # shape ( batch_size, 1 )
        valid_positions = mask.sum() -batch_size

        sibling_loss = torch.clamp(-gold_sibling_score+sibling_score+sibling_delta, min=0).sum()/valid_positions.float()
        grand_pa_loss = torch.clamp(-gold_grand_pa_score+grand_pa_score+grand_pa_delta, min=0).sum()/valid_positions.float()
        if self.stational_loss:
            stational_loss = self.refiner.stational_score(soft_head_indices,soft_head_tags,attended_arcs,high_order_weights,mask)
            stational_loss = stational_loss.sum() / valid_positions.float()
        else:
            stational_loss = 0
        arc_nll = torch.clamp(-gold_arc_nll + arc_nll + delta_arc, min=0).sum() / valid_positions.float()
        tag_nll = torch.clamp(-gold_tag_nll + tag_nll + delta_tag, min=0).sum() / valid_positions.float()
        if self.learn_score:
            return arc_nll +  tag_nll + sibling_loss+grand_pa_loss + stational_loss
        else:
            return  arc_nll + tag_nll +stational_loss





    def _get_head_tags_with_relaxed(self,
                       head_tag_representation: torch.Tensor,
                       child_tag_representation: torch.Tensor,
                        relaxed_head: torch.Tensor) -> torch.Tensor:
        """
        Decodes the head tags given the head and child tag representations
        and a tensor of head indices to compute tags for. Note that these are
        either gold or predicted heads, depending on whether this function is
        being called to compute the loss, or if it's being called during inference.

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        relaxed_head : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length,sequence_length). The distribution of the heads
            for every word. It is assumed to be masked already

        Returns
        -------
        head_tag_logits : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length, num_head_tags),
            representing logits for predicting a distribution over tags
            for each arc.
        """


        # shape (batch_size, sequence_length, tag_representation_dim)
        selected_head_tag_representations = torch.matmul(relaxed_head,head_tag_representation)
      #  selected_head_tag_representations = selected_head_tag_representations.contiguous()
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self.tag_bilinear(selected_head_tag_representations,
                                            child_tag_representation)
        return head_tag_logits

    def _score_per_instance(self,
                                head_tag_logits: torch.Tensor,
                        attended_arcs: torch.Tensor,
                        relaxed_head: torch.Tensor,
                        relaxed_head_tags: torch.Tensor,
                        mask: torch.Tensor,
                        high_order_weights:torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.

        Parameters
        ----------
        head_tag_logits : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_num),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachements of a given word to all other words.
        relaxed_head : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length).
            The indices of the heads for every word.
        relaxed_head_tags : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, num_head_tags).
            The dependency labels of the heads for every word.
        mask : ``torch.Tensor``, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        Returns
        -------
        arc_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc loss.
        tag_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc tag loss.
        """
        float_mask = mask.float()
        batch_size, sequence_length, _ = attended_arcs.size()

        # shape (batch_size, sequence_length, sequence_length)
        arc_nll = relaxed_head * attended_arcs * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)

        tag_nll =  relaxed_head_tags * head_tag_logits * float_mask.unsqueeze(2)


        # We don't care about predictions for the symbolic ROOT token's head,
        # so we remove it from the loss.
        # batch, length
        arc_nll = arc_nll[:, 1:]
        tag_nll = tag_nll[:, 1:]
        sibling_score, grand_pa_score = self.refiner._score_per_instance(attended_arcs,relaxed_head,relaxed_head_tags,mask,high_order_weights)
        return arc_nll, tag_nll ,sibling_score , grand_pa_score

    def _greedy_decode(self,
                       head_tag_representation: torch.Tensor,
                       child_tag_representation: torch.Tensor,
                       attended_arcs: torch.Tensor,
                       mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions by decoding the unlabeled arcs
        independently for each word and then again, predicting the head tags of
        these greedily chosen arcs indpendently. Note that this method of decoding
        is not guaranteed to produce trees (i.e. there maybe be multiple roots,
        or cycles when children are attached to their parents).

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachements of a given word to all other words.

        Returns
        -------
        heads : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the greedily decoded heads of each word.
        """
        # Mask the diagonal, because the head of a word can't be itself.
        attended_arcs = attended_arcs + torch.diag(attended_arcs.new(mask.size(1)).fill_(-numpy.inf))
        # Mask padded tokens, because we only want to consider actual words as heads.
        if mask is not None:
            minus_mask = (1 - mask).byte().unsqueeze(2)
            attended_arcs.masked_fill_(minus_mask, -numpy.inf)

        # Compute the heads greedily.
        # shape (batch_size, sequence_length)
        _, heads = attended_arcs.max(dim=2)

        # Given the greedily predicted heads, decode their dependency tags.
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(head_tag_representation,
                                              child_tag_representation,
                                              heads)
        _, head_tags = head_tag_logits.max(dim=2)
        return heads, head_tags

    def _relaxed_decode(self,
                       attended_arcs: torch.Tensor,
                       mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions by decoding the unlabeled arcs
        independently for each word and then again, predicting the head tags of
        these greedily chosen arcs indpendently. Note that this method of decoding
        is not guaranteed to produce trees (i.e. there maybe be multiple roots,
        or cycles when children are attached to their parents).

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachements of a given word to all other words.

        Returns
        -------
        heads : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the greedily decoded heads of each word.
        """
        # Mask the diagonal, because the head of a word can't be itself.
        attended_arcs = attended_arcs + torch.diag(attended_arcs.new(mask.size(1)).fill_(-numpy.inf))
        # Mask padded tokens, because we only want to consider actual words as heads.
        if mask is not None:
            minus_mask = (1 - mask).byte().unsqueeze(2)
            attended_arcs.masked_fill_(minus_mask, -numpy.inf)

            relaxed_head = masked_gumbel_softmax(attended_arcs,tau=self.gumbel_t,mask = mask)
            relaxed_head.masked_fill_(minus_mask, 0)


        return relaxed_head
    def _mst_decode(self,
                    head_tag_representation: torch.Tensor,
                    child_tag_representation: torch.Tensor,
                    attended_arcs: torch.Tensor,
                    mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions using the Edmonds' Algorithm
        for finding minimum spanning trees on directed graphs. Nodes in the
        graph are the words in the sentence, and between each pair of nodes,
        there is an edge in each direction, where the weight of the edge corresponds
        to the most likely dependency label probability for that arc. The MST is
        then generated from this directed graph.

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachements of a given word to all other words.

        Returns
        -------
        heads : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the optimally decoded heads of each word.
        """
        batch_size, sequence_length, tag_representation_dim = head_tag_representation.size()

        lengths = mask.data.sum(dim=1).long().cpu().numpy()

        expanded_shape = [batch_size, sequence_length, sequence_length, tag_representation_dim]
        head_tag_representation = head_tag_representation.unsqueeze(2)
        head_tag_representation = head_tag_representation.expand(*expanded_shape).contiguous()
        child_tag_representation = child_tag_representation.unsqueeze(1)
        child_tag_representation = child_tag_representation.expand(*expanded_shape).contiguous()
        # Shape (batch_size, sequence_length, sequence_length, num_head_tags)
        pairwise_head_logits = self.tag_bilinear(head_tag_representation, child_tag_representation)

        # Note that this log_softmax is over the tag dimension, and we don't consider pairs
        # of tags which are invalid (e.g are a pair which includes a padded element) anyway below.
        # Shape (batch, num_labels,sequence_length, sequence_length)
        normalized_pairwise_head_logits = F.log_softmax(pairwise_head_logits, dim=3).permute(0, 3, 1, 2)

        # Mask padded tokens, because we only want to consider actual words as heads.
        minus_inf = -1e8
        minus_mask = (1 - mask.float()) * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # Shape (batch_size, sequence_length, sequence_length)
        normalized_arc_logits = F.log_softmax(attended_arcs, dim=2).transpose(1, 2)

        # Shape (batch_size, num_head_tags, sequence_length, sequence_length)
        # This energy tensor expresses the following relation:
        # energy[i,j] = "Score that i is the head of j". In this
        # case, we have heads pointing to their children.
        batch_energy = torch.exp(normalized_arc_logits.unsqueeze(1) + normalized_pairwise_head_logits)
        return self._run_mst_decoding(batch_energy, lengths)

    @staticmethod
    def _run_mst_decoding(batch_energy: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        head_tags = []
        for energy, length in zip(batch_energy.detach().cpu(), lengths):
            scores, tag_ids = energy.max(dim=0)
            # Although we need to include the root node so that the MST includes it,
            # we do not want any word to be the parent of the root node.
            # Here, we enforce this by setting the scores for all word -> ROOT edges
            # edges to be 0.
            scores[0, :] = 0
            # Decode the heads. Because we modify the scores to prevent
            # adding in word -> ROOT edges, we need to find the labels ourselves.
            instance_heads, _ = decode_mst(scores.numpy(), length, has_labels=False)

            # Find the labels which correspond to the edges in the max spanning tree.
            instance_head_tags = []
            for child, parent in enumerate(instance_heads):
                instance_head_tags.append(tag_ids[parent, child].item())
            # We don't care what the head or tag is for the root token, but by default it's
            # not necesarily the same in the batched vs unbatched case, which is annoying.
            # Here we'll just set them to zero.
            instance_heads[0] = 0
            instance_head_tags[0] = 0
            heads.append(instance_heads)
            head_tags.append(instance_head_tags)
        return torch.from_numpy(numpy.stack(heads)), torch.from_numpy(numpy.stack(head_tags))

    def _get_head_tags(self,
                       head_tag_representation: torch.Tensor,
                       child_tag_representation: torch.Tensor,
                       head_indices: torch.Tensor) -> torch.Tensor:
        """
        Decodes the head tags given the head and child tag representations
        and a tensor of head indices to compute tags for. Note that these are
        either gold or predicted heads, depending on whether this function is
        being called to compute the loss, or if it's being called during inference.

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        head_indices : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length). The indices of the heads
            for every word.

        Returns
        -------
        head_tag_logits : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length, num_head_tags),
            representing logits for predicting a distribution over tags
            for each arc.
        """
        batch_size = head_tag_representation.size(0)
        # shape (batch_size,)
        range_vector = get_range_vector(batch_size, get_device_of(head_tag_representation)).unsqueeze(1)

        # This next statement is quite a complex piece of indexing, which you really
        # need to read the docs to understand. See here:
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
        # In effect, we are selecting the indices corresponding to the heads of each word from the
        # sequence length dimension for each element in the batch.

        # shape (batch_size, sequence_length, tag_representation_dim)
        selected_head_tag_representations = head_tag_representation[range_vector, head_indices]
        selected_head_tag_representations = selected_head_tag_representations.contiguous()
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self.tag_bilinear(selected_head_tag_representations,
                                            child_tag_representation)
        return head_tag_logits

    def _get_mask_for_eval(self,
                           mask: torch.LongTensor,
                           pos_tags: torch.LongTensor) -> torch.LongTensor:
        """
        Dependency evaluation excludes words are punctuation.
        Here, we create a new mask to exclude word indices which
        have a "punctuation-like" part of speech tag.

        Parameters
        ----------
        mask : ``torch.LongTensor``, required.
            The original mask.
        pos_tags : ``torch.LongTensor``, required.
            The pos tags for the sequence.

        Returns
        -------
        A new mask, where any indices equal to labels
        we should be ignoring are masked.
        """
        new_mask = mask.detach()
        for label in self._pos_to_ignore:
            label_mask = pos_tags.eq(label).long()
            new_mask = new_mask * (1 - label_mask)
        return new_mask

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._attachment_scores.get_metric(reset,self.training)
