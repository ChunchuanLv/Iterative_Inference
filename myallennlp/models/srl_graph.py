from typing import Dict, Optional, Tuple, Any, List
import logging
import copy

from overrides import overrides
import torch
from torch.nn.modules import Dropout
import numpy

import gc
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
                 refiner: Seq2SeqEncoder = None,
                 arc_feedforward: FeedForward = None,
                 train_score: float = 1.0,
                 train_gold: float = 0.1,
                 tag_feedforward: FeedForward = None,
                 pos_tag_embedding: Embedding = None,
                 dep_tag_embedding: Embedding = None,
                 predicate_embedding: Embedding = None,
                 dropout: float = 0.0,
                 detach:bool = False,
                 input_dropout: float = 0.0,
                 edge_prediction_threshold: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SRLGraphParser, self).__init__(vocab, regularizer)
        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.train_score = train_score
        self.detach = detach
        self.train_gold = train_gold
        self.edge_prediction_threshold = edge_prediction_threshold
        if not 0 < edge_prediction_threshold < 1:
            raise ConfigurationError(f"edge_prediction_threshold must be between "
                                     f"0 and 1 (exclusive) but found {edge_prediction_threshold}.")
     #   print ("predicates",self.vocab._index_to_token["predicates"])
     #   print ("tags",self.vocab._index_to_token["tags"])
        encoder_dim = encoder.get_output_dim()

        num_labels = self.vocab.get_vocab_size("tags")
        print("num_labels", num_labels)

        node_dim = predicate_embedding.get_output_dim()
        self.refiner = refiner
        if self.refiner:
            self.refiner.set_scorer(n_tags=num_labels, input_node_dim=node_dim)
            self.predicte_feedforward = FeedForward(encoder_dim, 1,
                                                    node_dim,
                                                    Activation.by_name("elu")())
            self.extra_feedforward = FeedForward(encoder_dim, 1,
                                                 self.refiner.node_dim,
                                                 Activation.by_name("elu")())

        self.arg_arc_feedforward = arc_feedforward or \
                                   FeedForward(encoder_dim, 1,
                                               arc_representation_dim,
                                               Activation.by_name("elu")())
        self.pred_arc_feedforward = copy.deepcopy(self.arg_arc_feedforward)

        self.arc_attention = BilinearMatrixAttention(arc_representation_dim,
                                                     arc_representation_dim,
                                                     use_input_biases=True)

        self.arg_tag_feedforward = tag_feedforward or \
                                   FeedForward(encoder_dim, 1,
                                               tag_representation_dim,
                                               Activation.by_name("elu")())
        self.pred_tag_feedforward = copy.deepcopy(self.arg_tag_feedforward)

        self.tag_bilinear = BilinearMatrixAttention(tag_representation_dim,
                                                    tag_representation_dim,
                                                    label_dim=num_labels,
                                                    use_input_biases=True) #,activation=Activation.by_name("tanh")()

        self._pos_tag_embedding = pos_tag_embedding or None
        self._dep_tag_embedding = dep_tag_embedding or None
        self._pred_embedding = predicate_embedding or None
        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)


     #   check_dimensions_match(representation_dim, encoder.get_input_dim(), "text field embedding dim", "encoder input dim")

        self._labelled_f1 = IterativeLabeledF1Measure(negative_label=0, negative_pred=0,
                                                      selected_metrics=["f1", "pred_f1", "lab_f1","scores"])
        self._tag_loss = torch.nn.NLLLoss(reduction="none")  # ,ignore_index=-1
        self._sense_loss = torch.nn.NLLLoss(reduction="none")  # ,ignore_index=-1
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                pos_tags: torch.LongTensor,
                dep_tags: torch.LongTensor ,
                predicate_candidates: torch.LongTensor = None,
                predicate_indexes: torch.LongTensor = None,
                sense_indexes: torch.LongTensor = None,
                predicates: torch.LongTensor = None,
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
            word in the dependency parse. Has shape ``(batch_size, predicates_len, batch_max_senses)``.

        predicate_indexes:  shape (batch_size, predicates_len)

        Returns
        -------
        An output dictionary.
        """
      #  torch.cuda.empty_cache()
        arc_tags = arc_tags.long()
        embedded_text_input = self.text_field_embedder(tokens)

        # shape (batch_size, predicates_len, batch_max_senses , pred_dim)
        embedded_candidate_preds = self._pred_embedding(predicate_candidates)

        # shape (batch_size, predicates_len, batch_max_senses )
        sense_mask = (predicate_candidates > 0).float()


        predicate_indexes = predicate_indexes.long()

        embedded_pos_tags = self._pos_tag_embedding(pos_tags)
        embedded_dep_tags = self._dep_tag_embedding(dep_tags)
        embedded_text_input = torch.cat([embedded_text_input, embedded_pos_tags,embedded_dep_tags], -1)

        mask = get_text_field_mask(tokens)

        batch_size, sequence_length = mask.size()

        embedded_text_input = self._input_dropout(embedded_text_input)

        float_mask = mask.float()

        predicate_mask = (predicate_indexes != -1).float()
        graph_mask = predicate_mask.unsqueeze(-1).unsqueeze(1)* float_mask.unsqueeze(-1).unsqueeze(2)

        # shape (batch_size, sequence_length, hidden_dim)
        encoded_text = self._dropout(self.encoder(embedded_text_input, mask))

        if self.detach:
            extra_representation = self._dropout(self.extra_feedforward(encoded_text.detach()))
        else:
            extra_representation = self._dropout(self.extra_feedforward(encoded_text))


        padding_for_predicate = torch.zeros(size=[batch_size, 1, encoded_text.size(-1)], device=encoded_text.device)

        # shape (batch_size, predicates_len, hidden_dim)
        encoded_text_for_predicate = torch.cat([padding_for_predicate, encoded_text], dim=1)

    #    print ("paded encoded_text_for_predicate",encoded_text_for_predicate.size())
    #    print("encoded_text_for_predicate", encoded_text_for_predicate.size())

  #      print("predicate_indexes", predicate_indexes.size())
        index_size = list(predicate_indexes.size())+[encoded_text.size(-1)]

   #     print("index_size", index_size)
  #      print("predicate_indexes", predicate_indexes.size())
        effective_predicate_indexes =  (predicate_indexes.unsqueeze(-1) + 1).expand(index_size)
        encoded_text_for_predicate = encoded_text_for_predicate.gather(dim=1, index = effective_predicate_indexes)


    #    print ("selected encoded_text_for_predicate",encoded_text_for_predicate.size())
        # shape (batch_size, sequence_length, arc_representation_dim)
        arg_arc_representation = self._dropout(self.arg_arc_feedforward(encoded_text))

        # shape (batch_size, predicates_len, arc_representation_dim)
        pred_arc_representation = self._dropout(self.pred_arc_feedforward(encoded_text_for_predicate))

        # shape (batch_size, sequence_length, predicates_len,1)
        arc_logits = self.arc_attention(arg_arc_representation,
                                        pred_arc_representation).unsqueeze(-1)  # + (1-predicate_mask)*1e9

        # shape (batch_size, sequence_length, tag_representation_dim)
        arg_tag_representation = self._dropout(self.arg_tag_feedforward(encoded_text))

        # shape (batch_size, predicates_len, arc_representation_dim)
        pred_tag_representation = self._dropout(self.pred_tag_feedforward(encoded_text_for_predicate))

        # shape (batch_size, num_tags, sequence_length, predicates_len)
        arc_tag_logits = self.tag_bilinear(arg_tag_representation,
                                           pred_tag_representation)

        # Switch to (batch_size, predicates_len, refine_representation_dim)
        predicate_representation = self._dropout(self.predicte_feedforward(encoded_text_for_predicate))

        # (batch_size, predicates_len, max_sense)
        sense_logits = embedded_candidate_preds.matmul(predicate_representation.unsqueeze(-1)).squeeze(-1)

        if self.training is False:
            arc_logits = arc_logits + (1 - predicate_mask.unsqueeze(-1).unsqueeze(1)) * 1e9
            sense_logits = sense_logits - (1 - sense_mask) * 1e9

        # Switch to (batch_size, sequence_length, predicates_len, num_tags)
        arc_tag_logits = arc_tag_logits.permute(0, 2, 3, 1)
        arc_tag_logits = torch.cat([arc_logits, arc_tag_logits], dim=-1).contiguous()

        output_dict = {
            "tokens": [meta["tokens"] for meta in metadata],
        }

        output_dict["loss"] = 0

        soft_tags = torch.zeros(size=arc_tag_logits.size(), device=arc_tag_logits.device)
        soft_tags.scatter_(3, arc_tags.unsqueeze(3) + 1, 1)

    #    print ("sense_logits",sense_logits.size(),sense_logits)
    #    print ("sense_indexes",sense_indexes.size(),sense_indexes)
        soft_index = torch.zeros(size=sense_logits.size(), device=sense_logits.device)
        soft_index.scatter_(2, sense_indexes.unsqueeze(2), 1)

        if self.refiner.iterations > 0:
            lists, c_lists , gold_results = self.refiner(predicate_representation,
                                               extra_representation,
                                               arc_tag_logits,
                                               sense_logits,
                                               embedded_candidate_preds,
                                               graph_mask,
                                               sense_mask,
                                               predicate_mask,
                                               soft_tags,
                                               soft_index)
            gold_scores, gold_sense_logits, gold_sense_probs, gold_arc_tag_logits, gold_arc_tag_probs = gold_results
            if self.train_gold:
                output_dict["loss"] = self.train_gold * self._max_margin_loss(gold_arc_tag_logits,
                                                            gold_arc_tag_probs,
                                                            arc_tags,
                                                            soft_tags,
                                                            gold_sense_logits,
                                                            gold_sense_probs,
                                                            sense_indexes,
                                                            soft_index,
                                                            gold_scores,
                                                            gold_scores,
                                                            graph_mask,
                                                            sense_mask,
                                                            predicate_mask,
                                                            is_gold=True)
                self._labelled_f1(gold_arc_tag_probs, arc_tags + 1,  graph_mask.squeeze(-1), gold_sense_probs, predicate_candidates,
                                  predicates, gold_scores,n_iteration= 0)
            for i, (arc_tag_logits, arc_tag_probs, sense_logits, sense_probs, scores) \
                    in enumerate(zip(*c_lists)):
                if arc_tags is not None:
                    loss = self._max_margin_loss(arc_tag_logits,
                                                 arc_tag_probs,
                                                 arc_tags,
                                                 soft_tags,
                                                 sense_logits,
                                                 sense_probs,
                                                 sense_indexes,
                                                 soft_index,
                                                 scores,
                                                 gold_scores,
                                                 graph_mask,
                                                 sense_mask,
                                                 predicate_mask,
                                                 is_gold=True)

                    output_dict["loss"] = output_dict["loss"] +  loss
                    # We stack scores here because the f1 measure expects a
                    # distribution, rather than a single value.
                    #     arc_tag_probs = torch.cat([one_minus_arc_probs, arc_tag_probs*arc_probs], dim=-1)

                    self._labelled_f1(arc_tag_probs, arc_tags + 1, graph_mask.squeeze(-1), sense_probs, predicate_candidates,
                                      predicates, scores, n_iteration= -i - 1)
            for i, (arc_tag_logits, arc_tag_probs, sense_logits, sense_probs, scores) \
                    in enumerate(zip(*lists)):
                if arc_tags is not None:
                    loss = self._max_margin_loss(arc_tag_logits,
                                                 arc_tag_probs,
                                                 arc_tags,
                                                 soft_tags,
                                                 sense_logits,
                                                 sense_probs,
                                                 sense_indexes,
                                                 soft_index,
                                                 scores,
                                                 gold_scores ,
                                                 graph_mask,
                                                 sense_mask,
                                                 predicate_mask)

                    output_dict["arc_tag_probs"+str(i)] = arc_tag_probs
                    output_dict["sense_probs"+str(i)] = sense_probs

                    output_dict["loss"] = output_dict["loss"] +  1 / (len(lists) - i + 1) * loss
                    # We stack scores here because the f1 measure expects a
                    # distribution, rather than a single value.
                    #     arc_tag_probs = torch.cat([one_minus_arc_probs, arc_tag_probs*arc_probs], dim=-1)

                    self._labelled_f1(arc_tag_probs, arc_tags + 1, graph_mask.squeeze(-1), sense_probs, predicate_candidates,
                                      predicates, scores,n_iteration=i +1)
        else:
            arc_tag_probs, sense_probs = self._greedy_decode(arc_tag_logits, sense_logits)

            loss = self._max_margin_loss(arc_tag_logits,
                                         arc_tag_probs,
                                         arc_tags,
                                         soft_tags,
                                         sense_logits,
                                         sense_probs,
                                         sense_indexes,
                                         soft_index,
                                         None,
                                         None,
                                         graph_mask,
                                         sense_mask,
                                         predicate_mask)

            output_dict["loss"] = output_dict["loss"] +  loss
            # We stack scores here because the f1 measure expects a
            # distribution, rather than a single value.
            #     arc_tag_probs = torch.cat([one_minus_arc_probs, arc_tag_probs*arc_probs], dim=-1)

            self._labelled_f1(arc_tag_probs, arc_tags + 1, graph_mask.squeeze(-1),
                              sense_probs, predicate_candidates,
                              predicates, n_iteration=0)

        output_dict["loss"] = loss

        # output_dict["additional"] = d
        #  print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$forward added additional$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        return output_dict


    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        for i in range(10):
            if "arc_tag_probs"+str(i) in output_dict:
                output_dict["predicted_arc_tags"+str(i) ] = output_dict["arc_tag_probs"+str(i) ].argmax(-1)- 1


                output_dict["sense_argmax"+str(i) ] = output_dict["sense_probs"+str(i) ].argmax(-1)

        return output_dict

    def _max_margin_loss(self, arc_tag_logits,
                         arc_tag_probs,
                         arc_tags,
                         soft_tags,
                         sense_logits,
                         sense_probs,
                         sense_indexes,
                         soft_index,
                         scores,
                         gold_scores,
                         graph_mask,
                         sense_mask,
                         predicate_mask,
                         is_gold=False):
        '''pred_probs: (batch_size, sequence_length, max_senses)'''

        valid_positions = graph_mask.sum().float()

        # shape (batch ,predicate_length ,1)
     #   delta_sense = self._sense_loss(torch.nn.functional.log_softmax(sense_logits, dim=-1).permute(0, 2, 1),sense_indexes)
        delta_sense = self._sense_loss((sense_probs+1e-6).log().permute(0, 2, 1),
                                       sense_indexes).unsqueeze(-1)

        #   delta_tag =  - ((arc_tag_probs+1e-6).log() * soft_tags).sum(3,keepdim=True)
        #   delta_tag = self._tag_loss(arc_tag_logits.permute(0,3,1,2),arc_tags+1)
      #  delta_tag = self._tag_loss(torch.nn.functional.log_softmax(arc_tag_logits, dim=-1).permute(0, 3, 1, 2), arc_tags + 1)

        # shape (batch ,sequence_length,sequence_length ,1)
        delta_tag = self._tag_loss((arc_tag_probs+1e-6).log().permute(0, 3, 1, 2),
                                   arc_tags + 1).unsqueeze(-1)


        if self.train_score and not is_gold and gold_scores is not None:
            if False:
                print ("soft_tags",soft_tags.size())
                print ("arc_tag_probs",arc_tag_probs.size())
                print ("arc_tag_logits",arc_tag_logits.size())
                print ("delta_tag",delta_tag.size())
                print ("graph_mask",graph_mask.size())
            tag_nll = torch.clamp(((-soft_tags + arc_tag_probs) * arc_tag_logits + delta_tag) * graph_mask,
                                  min=0).sum() / valid_positions

            sense_nll = torch.clamp(((-soft_index + sense_probs) * sense_logits + delta_sense) * sense_mask,
                                    min=0).sum() / valid_positions
            score_nll = torch.clamp(((-gold_scores + scores) + delta_tag + delta_sense.sum(-1,keepdim=True).unsqueeze(1)) * graph_mask,
                                    min=0).sum() / valid_positions

            return sense_nll + tag_nll + self.train_score * score_nll
        elif is_gold:
            tag_nll = torch.clamp(((-soft_tags + arc_tag_probs) * arc_tag_logits + delta_tag) * graph_mask,
                                  min=0).sum() / valid_positions

            sense_nll = torch.clamp(((-soft_index + sense_probs) * sense_logits + delta_sense) * sense_mask,
                                    min=0).sum() / valid_positions
            score_nll = -scores.sum() / valid_positions

            return sense_nll + tag_nll + self.train_score * score_nll

        else:
            tag_nll = torch.clamp(((-soft_tags + arc_tag_probs) * arc_tag_logits + delta_tag) * graph_mask,
                                  min=0).sum() / valid_positions

            sense_nll = torch.clamp(((-soft_index + sense_probs) * sense_logits + delta_sense) * sense_mask,
                                    min=0).sum() / valid_positions
            return tag_nll + sense_nll  # + arc_nll

    def _construct_loss(self, arc_tag_logits,
                        arc_tag_probs,
                        arc_tags,
                        verb_indicator,
                        pred_probs,
                        pred_indexes,
                        mask):
        '''pred_probs: (batch_size, sequence_length, max_senses)'''

        float_mask = mask.float().unsqueeze(-1)
        # shape (batch ,sequence_length ,sequence_length, 1 )
        arg_tag_mask = float_mask.unsqueeze(1) * float_mask.unsqueeze(2) * verb_indicator.float().unsqueeze(
            1).unsqueeze(-1)

        valid_positions = arg_tag_mask.sum()

        # shape (batch ,sequence_length )

        pred_nll = -(torch.log(pred_probs + 1e-6)).gather(-1, pred_indexes.unsqueeze(-1)).squeeze(
            -1) * verb_indicator.float()
        #   log_logits = torch.nn.functional.log_softmax(pred_logits.squeeze(-1), dim=-1)
        #  pred_nll = -log_logits.gather(-1,pred_indexes.unsqueeze(-1)).squeeze(-1) * verb_indicator.float()
        pred_nll = pred_nll.sum() / valid_positions

        batch_size, sequence_length = mask.size()

        soft_tags = torch.zeros(batch_size, sequence_length, sequence_length, arc_tag_probs.size(-1),
                                device=arc_tag_probs.device)
        soft_tags.scatter_(3, arc_tags.unsqueeze(3) + 1, 1)

        # shape (batch ,sequence_length,sequence_length ,1)
        delta_tag = self._tag_loss(torch.nn.functional.log_softmax(arc_tag_logits, dim=-1).permute(0, 3, 1, 2),
                                   arc_tags + 1)

        # shape (batch ,sequence_length,sequence_length ,1 )
        delta_tag = delta_tag.unsqueeze(-1)

        if self.max_margin:
            tag_nll = torch.clamp(((-soft_tags + arc_tag_probs) * arc_tag_logits + delta_tag) * arg_tag_mask,
                                  min=0).sum() / valid_positions.float()
        else:
            tag_nll = (delta_tag * arg_tag_mask).sum() / valid_positions.float()
        return tag_nll + pred_nll  # + arc_nll

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

        return arc_tag_probs, pred_probs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        return self._labelled_f1.get_metric(reset, training=self.training)
