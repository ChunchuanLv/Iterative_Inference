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
from allennlp.modules.seq2seq_encoders import PassThroughEncoder
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, Activation
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from myallennlp.metric import IterativeLabeledF1Measure
from myallennlp.modules.refiner.srl_score_based_refiner import SRLScoreBasedRefiner

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
from itertools import chain
from allennlp.models.archival import load_archive
from allennlp.nn.util import masked_softmax, weighted_sum

from myallennlp.dataset_readers.MultiCandidatesSequence import MultiCandidatesSequence
from myallennlp.modules.reparametrization.gumbel_softmax import hard, _sample_gumbel, inplace_masked_gumbel_softmax


from myallennlp.models.srl_graph_base import SRLGraphParserBase
@Model.register("srl_graph_parser_refine")
class SRLGraphParserRefine(Model):
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
                 base_model_archive: str,
                 refiner: Seq2SeqEncoder ,
                 encoder: Seq2SeqEncoder = None,
                 train_score: float = 1.0,
                 train_gold: float = 0.1,
                 dropout:float = 0.3,
                 normalize_logits:bool=False,
                 distance:str = "kl",
                 refine_epoch: int = -1,
                 detach:bool = False,
                 finetune:str="no",
                 detach_partial:int = 0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SRLGraphParserRefine, self).__init__(vocab, regularizer)
        self.train_score = train_score
        self.refine_epoch = refine_epoch
        self.finetune = finetune
        self.detach = detach
        self.train_gold = train_gold
        self.detach_partial = detach_partial
        self.distance = distance
        base_model:SRLGraphParserBase = load_archive(base_model_archive).model

        num_labels = self.vocab.get_vocab_size("tags")
        if normalize_logits:
            self.normalize_logits = torch.nn.LayerNorm(num_labels+1,elementwise_affine=False)
        else:
            self.normalize_logits = None
        print("num_labels", num_labels)

        node_dim = base_model._pred_embedding.get_output_dim()
        self.refiner = refiner
        self.refiner.set_scorer(n_tags=num_labels, input_node_dim=node_dim)
        self._dropout = InputVariationalDropout(dropout)

        self.encoder = encoder

     #   check_dimensions_match(representation_dim, encoder.get_input_dim(), "text field embedding dim", "encoder input dim")

        self._labelled_f1 = IterativeLabeledF1Measure(negative_label=0, negative_pred=0,
                                                      selected_metrics=["F", "l_F","p_F","h_S"])
        self._tag_loss = torch.nn.NLLLoss(reduction="none")  # ,ignore_index=-1
        self._sense_loss = torch.nn.NLLLoss(reduction="none")  # ,ignore_index=-1
        initializer(self)
        if  self.finetune == "no":
            for param in base_model.parameters():
                param.requires_grad = False

        elif self.finetune == "predicate":
            for param in base_model.parameters():
                param.requires_grad = False

            for param in base_model._pred_embedding.parameters():
                param.requires_grad = True
            for param in base_model.predicte_feedforward.parameters():
                param.requires_grad = True
        elif self.finetune == "predicate_emb":
            for param in base_model.parameters():
                param.requires_grad = False
            for param in base_model._pred_embedding.parameters():
                param.requires_grad = True
        elif self.finetune == "logits":
            for param in base_model.parameters():
                param.requires_grad = False

            for param in base_model._pred_embedding.parameters():
                param.requires_grad = True
            for param in base_model.predicte_feedforward.parameters():
                param.requires_grad = True
            for param in base_model.tag_bilinear.parameters():
                param.requires_grad = True
            for param in base_model.arc_attention.parameters():
                param.requires_grad = True
        else:
            assert self.finetune == "all", ("finetune is set as "+self.finetune+" need to be one of no, all, predicate, logits")

        encoder_dim = base_model.encoder.get_output_dim() if self.encoder is  None else self.encoder.get_output_dim()
        self.extra_feedforward = FeedForward(encoder_dim, 1,
                                             self.refiner.node_dim,
                                             Activation.by_name("elu")())

        if self.encoder is not None:
            self.predicte_feedforward = FeedForward(encoder_dim, 1,
                    node_dim,
                    Activation.by_name("elu")())


        self.base_model = base_model

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                pos_tags: torch.LongTensor,
                dep_tags: torch.LongTensor ,
                predicate_candidates: torch.LongTensor = None,
                epoch: int = None,
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

     #   if self.refine_epoch > -1 and epoch is not None and epoch >  self.refine_epoch:
     #       self.freeze_initial()


        input_dict = self.base_model(tokens,
                                        pos_tags,
                                        dep_tags,
                                        predicate_candidates,
                                        epoch,
                                        predicate_indexes,
                                        sense_indexes,
                                        predicates,
                                        metadata,
                                        arc_tags)
        arc_tags = arc_tags.long()

        # shape (batch_size, predicates_len, batch_max_senses , pred_dim)
        embedded_candidate_preds = input_dict["embedded_candidate_preds"]
        input_sense_logits  = input_dict["sense_logits"]
        input_arc_tag_logits = input_dict["arc_tag_logits"]
        arc_tag_probs = input_dict["arc_tag_probs"]
        sense_probs = input_dict["sense_probs"]
        embedded_text_input = input_dict["embedded_text_input"]
        if self.normalize_logits :
            input_arc_tag_logits = self.normalize_logits(input_arc_tag_logits)
       #     input_arc_tag_logits = input_arc_tag_logits - input_arc_tag_logits.mean(-1).unsqueeze(-1)
        # shape (batch_size, predicates_len, batch_max_senses )
        sense_mask = (predicate_candidates > 0).float()


        predicate_indexes = predicate_indexes.long()


        mask = get_text_field_mask(tokens)



        batch_size, sequence_length = mask.size()

        float_mask = mask.float()

        predicate_mask = (predicate_indexes > -1).float()
        graph_mask = predicate_mask.unsqueeze(-1).unsqueeze(1)* float_mask.unsqueeze(-1).unsqueeze(2)
        if self.encoder is not None:
            # shape (batch_size, sequence_length, hidden_dim)
            if isinstance(self.encoder, FeedForward):
                encoded_text = self._dropout(self.encoder(embedded_text_input))
            else:
                encoded_text = self._dropout(self.encoder(embedded_text_input, mask))

            padding_for_predicate = torch.zeros(size=[batch_size, 1, encoded_text.size(-1)], device=encoded_text.device)

            # shape (batch_size, predicates_len, hidden_dim)
            encoded_text_for_predicate = torch.cat([padding_for_predicate, encoded_text], dim=1)

            #    print ("paded encoded_text_for_predicate",encoded_text_for_predicate.size())
            #    print("encoded_text_for_predicate", encoded_text_for_predicate.size())

            #      print("predicate_indexes", predicate_indexes.size())
            index_size = list(predicate_indexes.size()) + [encoded_text.size(-1)]

            #     print("index_size", index_size)
            #      print("predicate_indexes", predicate_indexes.size())
            effective_predicate_indexes = (predicate_indexes.unsqueeze(-1) + 1).expand(index_size)
            encoded_text_for_predicate = encoded_text_for_predicate.gather(dim=1, index=effective_predicate_indexes)


            # Switch to (batch_size, predicates_len, refine_representation_dim)
            predicate_representation = self._dropout(self.predicte_feedforward(encoded_text_for_predicate))
        else:
            predicate_representation = input_dict["predicate_representation"]
            encoded_text = input_dict["encoded_text"]

        extra_representation = self._dropout(self.extra_feedforward(encoded_text))


        output_dict = {
            "tokens": [meta["tokens"] for meta in metadata],
        }


        soft_tags = torch.zeros(size=input_arc_tag_logits.size(), device=input_arc_tag_logits.device)
        soft_tags.scatter_(3, arc_tags.unsqueeze(3) + 1, 1) * graph_mask

    #    print ("sense_logits",sense_logits.size(),sense_logits)
    #    print ("sense_indexes",sense_indexes.size(),sense_indexes)
        soft_index = torch.zeros(size=input_sense_logits.size(), device=input_sense_logits.device)
        soft_index.scatter_(2, sense_indexes.unsqueeze(2), 1) * sense_mask


        output_dict["loss"] =  0

        lists, c_lists , gold_results = self.refiner(predicate_representation,
                                           extra_representation,
                                            input_arc_tag_logits,
                                                     input_sense_logits,
                                           embedded_candidate_preds,
                                           graph_mask,
                                           sense_mask,
                                           predicate_mask,
                                           soft_tags,
                                           soft_index,
                                                 arc_tag_probs,
                                                 sense_probs)
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
        output_dict["arc_tag_probs_g"] = gold_arc_tag_probs
        output_dict["sense_probs_g"] = gold_sense_probs
        self._labelled_f1(gold_arc_tag_probs, arc_tags + 1,  graph_mask.squeeze(-1), gold_sense_probs, predicate_candidates,
                          predicates, gold_scores, soft_tags*input_arc_tag_logits,n_iteration= 0)
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
                                             is_corrupt=True)

                output_dict["loss"] = output_dict["loss"] +   loss
                # We stack scores here because the f1 measure expects a
                # distribution, rather than a single value.
                #     arc_tag_probs = torch.cat([one_minus_arc_probs, arc_tag_probs*arc_probs], dim=-1)

                self._labelled_f1(arc_tag_probs, arc_tags + 1, graph_mask.squeeze(-1), sense_probs, predicate_candidates,
                                  predicates, scores, arc_tag_probs*input_arc_tag_logits,n_iteration= -i - 1)
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

                output_dict["loss"] = output_dict["loss"] + loss
                # We stack scores here because the f1 measure expects a
                # distribution, rather than a single value.
                #     arc_tag_probs = torch.cat([one_minus_arc_probs, arc_tag_probs*arc_probs], dim=-1)

                self._labelled_f1(arc_tag_probs, arc_tags + 1, graph_mask.squeeze(-1), sense_probs, predicate_candidates,
                                  predicates, scores, arc_tag_probs*input_arc_tag_logits,n_iteration=i + 1)


        # output_dict["additional"] = d
        #  print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$forward added additional$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        output_dict["predicate_mask"] = predicate_mask
        output_dict["sense_mask"] = sense_mask

        return output_dict


    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        for i in range(10):
            if "arc_tag_probs"+str(i) in output_dict:
                output_dict["predicted_arc_tags"+str(i) ] = output_dict["arc_tag_probs"+str(i) ].argmax(-1)- 1


                output_dict["sense_argmax"+str(i) ] = output_dict["sense_probs"+str(i) ].argmax(-1)
        if "arc_tag_probs_g" in output_dict:
            output_dict["predicted_arc_tags_g" ] = output_dict["arc_tag_probs_g" ].argmax(-1)- 1


            output_dict["sense_argmax_g"] = output_dict["sense_probs_g" ].argmax(-1)

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
                         is_gold=False,
                         is_corrupt=False):
        '''pred_probs: (batch_size, sequence_length, max_senses)'''

        valid_positions = graph_mask.sum().float()

        # shape (batch ,predicate_length ,1)
      #  delta_tag = self._tag_loss((arc_tag_probs+1e-6).log().permute(0, 3, 1, 2), arc_tags + 1).unsqueeze(-1)


        if not is_gold and  gold_scores is not None and scores is not None:
            if self.distance == "kl":
                delta_sense = self._sense_loss(torch.nn.functional.log_softmax(sense_logits, dim=-1).permute(0, 2, 1),sense_indexes).unsqueeze(-1)
                #    delta_sense = self._sense_loss((sense_probs+1e-6).log().permute(0, 2, 1), sense_indexes).unsqueeze(-1)

                # shape (batch ,sequence_length,sequence_length ,1)
                delta_tag = self._tag_loss(torch.nn.functional.log_softmax(arc_tag_logits, dim=-1).permute(0, 3, 1, 2), arc_tags + 1).unsqueeze(-1)
            elif self.distance == "l1_through":
                delta_sense = torch.abs(sense_probs-soft_index).sum(-1,keepdim=True)
                #    delta_sense = self._sense_loss((sense_probs+1e-6).log().permute(0, 2, 1), sense_indexes).unsqueeze(-1)

                # shape (batch ,sequence_length,sequence_length ,1)
                delta_tag = torch.abs(arc_tag_probs-soft_tags).sum(-1,keepdim=True)

            else:

                delta_sense = torch.abs(sense_probs-soft_index).sum(-1,keepdim=True).detach()
                #    delta_sense = self._sense_loss((sense_probs+1e-6).log().permute(0, 2, 1), sense_indexes).unsqueeze(-1)

                # shape (batch ,sequence_length,sequence_length ,1)
                delta_tag = torch.abs(arc_tag_probs-soft_tags).sum(-1,keepdim=True).detach()
        # shape (batch ,sequence_length,sequence_length ,1)
            tag_nll = torch.clamp(((-soft_tags + arc_tag_probs) * arc_tag_logits + delta_tag) * graph_mask,
                                  min=0).sum() / valid_positions

            sense_nll = torch.clamp(((-soft_index + sense_probs) * sense_logits + delta_sense) * sense_mask,
                                    min=0).sum() / valid_positions
            nll =  sense_nll + tag_nll
            if self.train_score:
                score_nll = torch.clamp((self.train_score *(-gold_scores + scores) + delta_tag + delta_sense.sum(-1,keepdim=True).unsqueeze(1)) * graph_mask,
                                        min=0).sum() / valid_positions
                nll = nll +  score_nll
            return nll
        else :

            if self.distance == "kl":
                delta_sense = self._sense_loss(torch.nn.functional.log_softmax(sense_logits, dim=-1).permute(0, 2, 1),sense_indexes).unsqueeze(-1)
                #    delta_sense = self._sense_loss((sense_probs+1e-6).log().permute(0, 2, 1), sense_indexes).unsqueeze(-1)

                # shape (batch ,sequence_length,sequence_length ,1)
                delta_tag = self._tag_loss(torch.nn.functional.log_softmax(arc_tag_logits, dim=-1).permute(0, 3, 1, 2), arc_tags + 1).unsqueeze(-1)
            elif self.distance == "l1_through":
                delta_sense = torch.abs(sense_probs-soft_index).sum(-1,keepdim=True)
                #    delta_sense = self._sense_loss((sense_probs+1e-6).log().permute(0, 2, 1), sense_indexes).unsqueeze(-1)

                # shape (batch ,sequence_length,sequence_length ,1)
                delta_tag = torch.abs(arc_tag_probs-soft_tags).sum(-1,keepdim=True)
            else: #assume l1


                delta_sense = torch.abs(sense_probs-soft_index).sum(-1,keepdim=True).detach()
                #    delta_sense = self._sense_loss((sense_probs+1e-6).log().permute(0, 2, 1), sense_indexes).unsqueeze(-1)

                # shape (batch ,sequence_length,sequence_length ,1)
                delta_tag = torch.abs(arc_tag_probs-soft_tags).sum(-1,keepdim=True).detach()


            tag_nll = torch.clamp(((-soft_tags + arc_tag_probs) * arc_tag_logits + delta_tag) * graph_mask,
                                  min=0).sum() / valid_positions

            sense_nll = torch.clamp(((-soft_index + sense_probs) * sense_logits + delta_sense) * sense_mask,
                                    min=0).sum() / valid_positions
            return tag_nll + sense_nll

    def _construct_loss(self, arc_tag_logits,
                         arc_tag_probs,
                         arc_tags,
                         soft_tags,
                         sense_logits,
                         sense_probs,
                         sense_indexes,
                         soft_index,
                         graph_mask,
                         sense_mask):
        '''pred_probs: (batch_size, sequence_length, max_senses)'''

        valid_positions = graph_mask.sum().float()

        delta_sense = self._sense_loss((sense_probs+1e-6).log().permute(0, 2, 1), sense_indexes).unsqueeze(-1)


        # shape (batch ,sequence_length,sequence_length ,1)
        delta_tag = self._tag_loss((arc_tag_probs+1e-6).log().permute(0, 3, 1, 2), arc_tags + 1).unsqueeze(-1)

        tag_nll = torch.clamp(((-soft_tags + arc_tag_probs) * arc_tag_logits + delta_tag) * graph_mask,
                              min=0).sum() / valid_positions

        sense_nll = torch.clamp(((-soft_index + sense_probs) * sense_logits + delta_sense) * sense_mask,
                                min=0).sum() / valid_positions
        return tag_nll + sense_nll  # + arc_nll

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
