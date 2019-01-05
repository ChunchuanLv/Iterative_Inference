from typing import Dict, Optional, Tuple, Any, List
import logging
import copy

from overrides import overrides
import torch
from torch.nn.modules import Dropout
import numpy

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding, InputVariationalDropout
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, Activation
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from myallennlp.metric import IterativeLabeledF1Measure
from myallennlp.modules.refiner.srl_score_based_refiner import SRLScoreBasedRefiner
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from allennlp.nn.util import masked_softmax, weighted_sum

from myallennlp.dataset_readers.MultiCandidatesSequence import MultiCandidatesSequence
from myallennlp.dataset_readers.conll2009 import NEGATIVE_PRED
@Model.register("srl_graph_parser")
class SRLGraphParser(Model):
    """
    A Parser for arbitrary graph stuctures.

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
        The dimension of the MLPs used for arc tag prediction.
    arc_representation_dim : ``int``, required.
        The dimension of the MLPs used for arc prediction.
    tag_feedforward : ``FeedForward``, optional, (default = None).
        The feedforward network used to produce tag representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    arc_feedforward : ``FeedForward``, optional, (default = None).
        The feedforward network used to produce arc representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    pos_tag_embedding : ``Embedding``, optional.
        Used to embed the ``pos_tags`` ``SequenceLabelField`` we get as input to the model.
    dropout : ``float``, optional, (default = 0.0)
        The variational dropout applied to the output of the encoder and MLP layers.
    input_dropout : ``float``, optional, (default = 0.0)
        The dropout applied to the embedded text input.
    edge_prediction_threshold : ``int``, optional (default = 0.5)
        The probability at which to consider a scored edge to be 'present'
        in the decoded graph. Must be between 0 and 1.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 arc_representation_dim: int,
                 tag_representation_dim: int,
                 refine_representation_dim: int=0,
                 refiner: Seq2SeqEncoder = None,
                 arc_feedforward: FeedForward = None,
                 train_score:bool=True,
                 tag_feedforward: FeedForward = None,
                 pos_tag_embedding: Embedding = None,
                 pred_embedding: Embedding = None,
                 dropout: float = 0.0,
                 input_dropout: float = 0.0,
                 edge_prediction_threshold: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SRLGraphParser, self).__init__(vocab, regularizer)
        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.train_score = train_score
        self.edge_prediction_threshold = edge_prediction_threshold
        if not 0 < edge_prediction_threshold < 1:
            raise ConfigurationError(f"edge_prediction_threshold must be between "
                                     f"0 and 1 (exclusive) but found {edge_prediction_threshold}.")

        encoder_dim = encoder.get_output_dim()



        num_labels = self.vocab.get_vocab_size("tags")
        print ("num_labels",num_labels)
        self.refiner = refiner
        if self.refiner:
            self.refiner.set_score_mlp(n_tags=num_labels,extra_dim=refine_representation_dim)
            self.predicte_feedforward = FeedForward(encoder_dim, 1,
                                                    refine_representation_dim,
                                                    Activation.by_name("elu")())
            self.arguement_feedforward = FeedForward(encoder_dim, 1,
                                                    refine_representation_dim,
                                                    Activation.by_name("elu")())

        self.head_arc_feedforward = arc_feedforward or \
                                        FeedForward(encoder_dim, 1,
                                                    arc_representation_dim,
                                                    Activation.by_name("elu")())
        self.child_arc_feedforward = copy.deepcopy(self.head_arc_feedforward)

        self.arc_attention = BilinearMatrixAttention(arc_representation_dim,
                                                     arc_representation_dim,
                                                     use_input_biases=True)


        self.head_tag_feedforward = tag_feedforward or \
                                        FeedForward(encoder_dim, 1,
                                                    tag_representation_dim,
                                                    Activation.by_name("elu")())
        self.child_tag_feedforward = copy.deepcopy(self.head_tag_feedforward)

        self.tag_bilinear = BilinearMatrixAttention(tag_representation_dim,
                                                    tag_representation_dim,
                                                    label_dim=num_labels,
                                                    use_input_biases=True)

        self._pos_tag_embedding = pos_tag_embedding or None
        self._pred_embedding = pred_embedding or None
        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)

        representation_dim = text_field_embedder.get_output_dim()
        if pos_tag_embedding is not None:
            representation_dim += pos_tag_embedding.get_output_dim()

        check_dimensions_match(representation_dim, encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        check_dimensions_match(tag_representation_dim, self.head_tag_feedforward.get_output_dim(),
                               "tag representation dim", "tag feedforward output dim")

        check_dimensions_match(arc_representation_dim, self.head_arc_feedforward.get_output_dim(),
                               "arc representation dim", "arc feedforward output dim")
        self._labelled_f1 = IterativeLabeledF1Measure(negative_label=0,negative_pred=vocab.get_token_index(NEGATIVE_PRED, "pred"))
        self._tag_loss = torch.nn.CrossEntropyLoss(reduction="none") # ,ignore_index=-1
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                verb_indicator: torch.LongTensor,
                pos_tags: torch.LongTensor = None,
                pred_candidates: torch.LongTensor = None,
                pred_indexes: torch.LongTensor = None,
                preds: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                arc_tags: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``.
        verb_indicator: torch.LongTensor, required.
            An integer ``SequenceFeatureField`` representation of the position of the verb
            in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be
            all zeros, in the case that the sentence has no verbal predicate.
        pos_tags : ``torch.LongTensor``, optional, (default = None).
            The output of a ``SequenceLabelField`` containing POS tags.
        arc_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer indices denoting the parent of every
            word in the dependency parse. Has shape ``(batch_size, sequence_length, sequence_length)``.
        pred_candidates : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer indices denoting the parent of every
            word in the dependency parse. Has shape ``(batch_size, sequence_length, batch_max_senses)``.

        Returns
        -------
        An output dictionary.
        """

        arc_tags = arc_tags.long()
        embedded_text_input = self.text_field_embedder(tokens)
        # shape (batch_size, sequence_length, batch_max_senses , pred_dim)
        embedded_candidate_preds = self._pred_embedding(pred_candidates)

        # shape (batch_size, sequence_length, batch_max_senses )
        pred_mask = (pred_candidates > 0).float()

        if pos_tags is not None and self._pos_tag_embedding is not None:
            embedded_pos_tags = self._pos_tag_embedding(pos_tags)
            embedded_text_input = torch.cat([embedded_text_input, embedded_pos_tags], -1)
        elif self._pos_tag_embedding is not None:
            raise ConfigurationError("Model uses a POS embedding, but no POS tags were passed.")

        mask = get_text_field_mask(tokens)
        embedded_text_input = self._input_dropout(embedded_text_input)
        encoded_text = self.encoder(embedded_text_input, mask)

        float_mask = mask.float()
        # shape (batch_size, sequence_length, hidden_dim)
        encoded_text = self._dropout(encoded_text)


        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc_representation = self._dropout(self.head_arc_feedforward(encoded_text))
        child_arc_representation = self._dropout(self.child_arc_feedforward(encoded_text))

        # shape (batch ,1 ,sequence_length, 1 )
        predicate_mask = verb_indicator.float().unsqueeze(1).unsqueeze(-1)


        # shape (batch_size, sequence_length, sequence_length,1)
        arc_logits = self.arc_attention(head_arc_representation,
                                        child_arc_representation).unsqueeze(-1) #+ (1-predicate_mask)*1e9
        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag_representation = self._dropout(self.head_tag_feedforward(encoded_text))
        child_tag_representation = self._dropout(self.child_tag_feedforward(encoded_text))
        # shape (batch_size, num_tags, sequence_length, sequence_length)
        arc_tag_logits = self.tag_bilinear(head_tag_representation,
                                           child_tag_representation)


        # Switch to (batch_size, sequence_length, refine_representation_dim)
        predict_representation = self._dropout(self.predicte_feedforward(encoded_text)) * float_mask.unsqueeze(-1)


        #(batch_size, sequence_length, max_sense)
        pred_logits = embedded_candidate_preds.matmul(predict_representation.unsqueeze(-1)).squeeze(-1)

        if self.training is False:
            arc_logits = arc_logits + (1-predicate_mask)*1e9
            pred_logits = pred_logits - (1-pred_mask)*1e9


        # Switch to (batch_size, sequence_length, sequence_length, num_tags)
        arc_tag_logits = arc_tag_logits.permute(0, 2, 3, 1)
        arc_tag_logits = torch.cat([arc_logits, arc_tag_logits], dim=-1).contiguous()


        arc_tag_probs, arc_probs, pred_probs = self._greedy_decode(arc_tag_logits,pred_logits)
        if self.refiner.iterations > 0 :

            argument_representation = self._dropout(self.arguement_feedforward(encoded_text)) * float_mask.unsqueeze(-1)
            arc_tag_probs_list,arc_probs_list,intermediates_list = self.refiner(predict_representation,
                                                                                argument_representation,
                                                                                arc_tag_logits,
                                                                                arc_tag_probs,
                                                                                arc_logits,
                                                                                arc_probs,
                                                                                verb_indicator,
                                                                                mask,
                                                                                arc_tags)
        else:
            arc_tag_probs_list = [arc_tag_probs]
            arc_probs_list = [arc_probs]
            intermediates_list = [{}]
            argument_representation = None
        output_dict = {
                "arc_tag_probs": arc_tag_probs_list[-1],
                "mask": mask,
                "tokens": [meta["tokens"] for meta in metadata],
                }

        output_dict["loss"] = 0
        output_dict["tag_loss"] = 0

        if self.refiner :
            for i,(arc_tag_probs,arc_probs,intermediates) in enumerate(zip(arc_tag_probs_list,arc_probs_list,intermediates_list)):
                if arc_tags is not None:
                    loss = self._max_margin_loss(arc_logits=arc_logits,
                                                         arc_probs=arc_probs,
                                                         arc_tag_logits=arc_tag_logits,
                                                    arc_tag_probs=arc_tag_probs,
                                                            arc_tags=arc_tags,
                                                    predict_representation=predict_representation,
                                                 argument_representation=argument_representation,
                                                 intermediates=intermediates,
                                                        verb_indicator = verb_indicator,
                                                    pred_probs=pred_probs,
                                                        pred_logits=pred_logits,
                                                        pred_indexes=pred_indexes,
                                                        pred_mask=pred_mask,
                                                            mask=mask)

                    output_dict["loss"] = output_dict["loss"] + + 1/(len(arc_tag_probs_list)-i) *loss
                    tag_mask = float_mask.unsqueeze(1) * float_mask.unsqueeze(2)
                    # We stack scores here because the f1 measure expects a
                    # distribution, rather than a single value.
               #     arc_tag_probs = torch.cat([one_minus_arc_probs, arc_tag_probs*arc_probs], dim=-1)

                    self._labelled_f1(arc_tag_probs ,arc_tags+1, tag_mask,pred_probs,pred_candidates,preds,n_iteration=i)
        else:

            for i,(arc_tag_probs,arc_probs) in enumerate(arc_tag_probs_list):
                if arc_tags is not None:
                    tag_nll = self._construct_loss(arc_tag_logits=arc_tag_logits,
                                                            arc_tags=arc_tags,
                                                            mask=mask)

                    output_dict["loss"] = output_dict["loss"] + tag_nll
                    output_dict["tag_loss"] =  output_dict["tag_loss"] + tag_nll

                    # Make the arc tags not have negative values anywhere
                    # (by default, no edge is indicated with -1).

                    tag_mask = float_mask.unsqueeze(1) * float_mask.unsqueeze(2)
                    # shape (batch ,sequence_length ,sequence_length )
                    predicate_mask = float_mask.unsqueeze(1) * float_mask.unsqueeze(2) * verb_indicator.float().unsqueeze(1)
                    # We stack scores here because the f1 measure expects a
                    # distribution, rather than a single value.
                    self._labelled_f1(arc_tag_probs, arc_tags+1, predicate_mask,n_iteration=i)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        arc_tag_probs = output_dict["arc_tag_probs"].cpu().detach().numpy()
        arc_probs = output_dict["arc_probs"].cpu().detach().numpy()
        mask = output_dict["mask"]
        lengths = get_lengths_from_binary_sequence_mask(mask)
        arcs = []
        arc_tags = []
        for instance_arc_probs, instance_arc_tag_probs, length in zip(arc_probs, arc_tag_probs, lengths):

            arc_matrix = instance_arc_probs > self.edge_prediction_threshold
            edges = []
            edge_tags = []
            for i in range(length):
                for j in range(length):
                    if arc_matrix[i, j] == 1:
                        edges.append((i, j))
                        tag = instance_arc_tag_probs[i, j].argmax(-1)
                        edge_tags.append(self.vocab.get_token_from_index(tag, "labels"))
            arcs.append(edges)
            arc_tags.append(edge_tags)

        output_dict["arcs"] = arcs
        output_dict["arc_tags"] = arc_tags
        return output_dict
    def _max_margin_loss(self,arc_logits,
                          arc_probs,
                         arc_tag_logits,
                            arc_tag_probs,
                          arc_tags,
                            predict_representation,
                         argument_representation,
                         intermediates,
                        verb_indicator ,
                         pred_probs,
                         pred_logits,
                         pred_indexes,
                         pred_mask,
                            mask):
        '''pred_probs: (batch_size, sequence_length, max_senses)'''

        float_mask = mask.float().unsqueeze(-1)
        # shape (batch ,sequence_length ,sequence_length, 1 )
        predicate_mask = float_mask.unsqueeze(1) * float_mask.unsqueeze(2) * verb_indicator.float().unsqueeze(1).unsqueeze(-1)


        valid_positions = predicate_mask.sum()


        # shape (batch ,sequence_length )

        pred_nll = -(torch.log(pred_probs+1e-6)).gather(-1,pred_indexes.unsqueeze(-1)).squeeze(-1) * verb_indicator.float()
        #   log_logits = torch.nn.functional.log_softmax(pred_logits.squeeze(-1), dim=-1)
        #  pred_nll = -log_logits.gather(-1,pred_indexes.unsqueeze(-1)).squeeze(-1) * verb_indicator.float()
        pred_nll = pred_nll.sum() /valid_positions

        batch_size, sequence_length = mask.size()

        arc_indices = (arc_tags != -1).float().unsqueeze(-1)

        soft_tags = torch.zeros(batch_size, sequence_length,sequence_length,arc_tag_probs.size(-1), device=arc_tag_probs.device)
        soft_tags.scatter_(3, arc_tags.unsqueeze(3)+1, 1)

        # shape (batch ,sequence_length,sequence_length ,1)
     #   delta_tag =  - ((arc_tag_probs+1e-6).log() * soft_tags).sum(3,keepdim=True)
        delta_tag = self._tag_loss(arc_tag_probs.permute(0,3,1,2),arc_tags+1)

        # shape (batch ,sequence_length,sequence_length ,1 )
        delta_tag = delta_tag.unsqueeze(-1)  * predicate_mask


        # shape (batch ,sequence_length,sequence_length ,1)
     #   delta_arc = 1-arc_probs*arc_indices # torch.nn.functional.binary_cross_entropy(arc_probs, arc_indices, reduction='none') #1-arc_probs*arc_indices #
    #    delta_arc = torch.nn.functional.binary_cross_entropy(arc_probs, arc_indices, reduction='none')
   #     delta_arc = delta_arc* float_mask.unsqueeze(1)* float_mask.unsqueeze(2)

        # shape (batch ,sequence_length,sequence_length ,1)
        delta = delta_tag #+ delta_arc
        #    delta = delta.detach()

        #local_score (batch ,sequence_length,sequence_length,n_labels )
        #predict_score (batch ,sequence_length,1 )
        #argument_score (batch ,sequence_length,1 )
        if self.train_score:
            tag_score,predict_score = self.refiner.get_score_per_feature(predict_representation,
                                                                                   argument_representation,
                    arc_tag_logits,
                    arc_tag_probs,
                    arc_probs,
                    mask,
                    intermediates)
            gold_tag_score,gold_predict_score = self.refiner.get_score_per_feature(predict_representation,
                                                                                                   argument_representation,
                    arc_tag_logits,
                    soft_tags,
                    arc_indices,
                    mask,
                    intermediates)


            tag_nll = torch.clamp((-gold_tag_score + tag_score)* predicate_mask  + delta_tag, min=0).sum() / valid_positions
      #      arc_nll = torch.clamp((-gold_arc_score + arc_score )* predicate_mask+ delta_arc, min=0).sum() / valid_positions
            predict_nll = torch.clamp(-gold_predict_score + predict_score + delta.sum(1), min=0).sum() / valid_positions
      #      argument_nll = torch.clamp(-gold_argument_score + argument_score + delta.sum(2), min=0).sum() / valid_positions

            return pred_nll + tag_nll+predict_nll#+argument_nll

        else:

        #    print ("soft_tags",soft_tags.size())
         #   print ("arc_tag_probs",arc_tag_probs.size())
         #   print ("arc_tag_logits",arc_tag_logits.size())
         #   print ("delta_tag",delta_tag.size())
            tag_nll = torch.clamp((-soft_tags + arc_tag_probs)*arc_tag_logits* predicate_mask  + delta_tag, min=0).sum() / valid_positions.float()
      #      arc_nll = torch.clamp((-arc_indices + arc_probs)*arc_logits* predicate_mask  + delta_arc, min=0).sum() / valid_positions.float()
            return   tag_nll + pred_nll #+ arc_nll
    def _construct_loss(self,arc_tag_logits: torch.Tensor,
                        arc_tags: torch.Tensor,
                        mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for an adjacency matrix.

        Parameters
        ----------
        arc_scores : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate a
            binary classification decision for whether an edge is present between two words.
        arc_tag_logits : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length, num_tags) used to generate
            a distribution over edge tags for a given edge.
        arc_tags : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length).
            The labels for every arc.
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
        # Make the arc tags not have negative values anywhere
        # (by default, no edge is indicated with -1).
        # We want the mask for the tags to only include the unmasked words
        # and we only care about the loss with respect to the gold arcs.
        tag_mask = float_mask.unsqueeze(1) * float_mask.unsqueeze(2)

        batch_size, sequence_length, _, num_tags = arc_tag_logits.size()
        original_shape = [batch_size, sequence_length, sequence_length]
        reshaped_logits = arc_tag_logits.view(-1, num_tags)
        reshaped_tags = arc_tags.view(-1)
        tag_nll = self._tag_loss(reshaped_logits, reshaped_tags.long()).view(original_shape) * tag_mask

        valid_positions = tag_mask.sum()

        tag_nll = tag_nll.sum() / valid_positions.float()
        return tag_nll

    @staticmethod
    def _greedy_decode(arc_tag_logits: torch.Tensor,
                        pred_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions by decoding the unlabeled arcs
        independently for each word and then again, predicting the head tags of
        these greedily chosen arcs indpendently.

        Parameters
        ----------
        arc_scores : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachements of a given word to all other words.
        arc_tag_logits : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length, num_tags) used to
            generate a distribution over tags for each arc.
        mask : ``torch.Tensor``, required.
            A mask of shape (batch_size, sequence_length).

        Returns
        -------
        arc_probs : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length, sequence_length) representing the
            probability of an arc being present for this edge.
        arc_tag_probs : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length, sequence_length, sequence_length)
            representing the distribution over edge tags for a given edge.
        """
        # Mask the diagonal, because we don't self edges.
        # shape (batch_size, sequence_length, sequence_length, num_tags)
    #    arc_tag_logits = arc_tag_logits + inf_diagonal_mask.unsqueeze(0).unsqueeze(-1)
        # Mask padded tokens, because we only want to consider actual word -> word edges.
     #   minus_mask = (1 - mask).byte().unsqueeze(2)
       # arc_tag_logits.masked_fill_(minus_mask.unsqueeze(-1), -numpy.inf)
        # shape (batch_size, sequence_length, sequence_length, num_tags)



        # shape (batch_size, sequence_length, max_sense)
        pred_probs = torch.nn.functional.softmax(pred_logits.squeeze(-1), dim=-1)

        # shape (batch_size, sequence_length, sequence_length,n_tags)
        arc_tag_probs = torch.nn.functional.softmax(arc_tag_logits, dim=-1)

        # shape (batch_size, sequence_length, sequence_length,1)
        arc_probs = 1-arc_tag_probs[:,:,:,0]#arc_scores.sigmoid()

        return  arc_tag_probs,arc_probs.unsqueeze(-1),pred_probs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        return  self._labelled_f1.get_metric(reset,training=self.training)
