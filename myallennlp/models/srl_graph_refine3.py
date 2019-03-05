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


@Model.register("srl_graph_parser_refine3")
class SRLGraphParserRefine2(Model):
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
                 refiner: Seq2SeqEncoder,
                 rep_dim:int=0,
                 encoder: Seq2SeqEncoder = None,
                 joint_train:bool = False,
                 re_estimate_logits:bool = False,
                 train_score: float = 1.0,
                 dropout: float = 0.3,
                 normalize_logits: bool = False,
                 distance: str = "kl",
                 finetune: str = "no",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SRLGraphParserRefine2, self).__init__(vocab, regularizer)
        self.train_score = train_score
        self.finetune = finetune
        self.re_estimate_logits = re_estimate_logits
        self.distance = distance
        base_model: SRLGraphParserBase = load_archive(base_model_archive).model

        base_model.gumbel_t = refiner.gumbel_t
        self.joint_train = joint_train
        if self.joint_train: assert self.finetune == "no" and self.re_estimate_logits is False
        self.encoder = None if encoder is None else encoder
        assert self.encoder is not None, "have not implemented reuse for now"

        num_labels = self.vocab.get_vocab_size("tags")
        if normalize_logits:
            self.normalize_logits = torch.nn.LayerNorm(num_labels + 1, elementwise_affine=False)
        else:
            self.normalize_logits = None
        print("num_labels", num_labels)

        node_dim = base_model._pred_embedding.get_output_dim()

        encoder_dim = self.encoder.get_output_dim() if self.encoder else base_model.encoder.get_output_dim()

        self.rep_dim = rep_dim
        if rep_dim :
            self.predicte_rep_feedforward = FeedForward(encoder_dim, 1,
                                                    rep_dim,
                                                    Activation.by_name("elu")())
            self.argument_rep_feedforward = FeedForward(encoder_dim, 1,
                                                        rep_dim,
                                                    Activation.by_name("elu")())
        else:
            rep_dim = encoder_dim
        self.refiner = refiner
        self.refiner.set_scorer(n_tags=num_labels, node_dim=node_dim, hidden_dim=rep_dim)
        self._dropout = InputVariationalDropout(dropout)

        #   check_dimensions_match(representation_dim, encoder.get_input_dim(), "text field embedding dim", "encoder input dim")

        self._labelled_f1 = IterativeLabeledF1Measure(negative_label=0, negative_pred=0,
                                                      selected_metrics=["F", "l_F", "p_F", "h_S"])
        self._tag_loss = torch.nn.NLLLoss(reduction="none")  # ,ignore_index=-1
        self._sense_loss = torch.nn.NLLLoss(reduction="none")  # ,ignore_index=-1
        initializer(self)
        if self.finetune == "no":
            if self.re_estimate_logits:
                self.arg_arc_feedforward = copy.deepcopy(base_model.arg_arc_feedforward)
                self.pred_arc_feedforward = copy.deepcopy(base_model.pred_arc_feedforward)

                self.arc_attention = copy.deepcopy(base_model.arc_attention)

                self.arg_tag_feedforward =  copy.deepcopy(base_model.arg_tag_feedforward)
                self.pred_tag_feedforward =  copy.deepcopy(base_model.pred_tag_feedforward)

                self.tag_bilinear =  copy.deepcopy(base_model.tag_bilinear)
                self.predicte_feedforward =  copy.deepcopy(base_model.predicte_feedforward)
                self._pred_embedding = copy.deepcopy(base_model._pred_embedding)
            for param in base_model.parameters():
                param.requires_grad = False
            if not self.re_estimate_logits:
                self._pred_embedding = copy.deepcopy(base_model._pred_embedding)
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
            assert self.finetune == "all", (
                    "finetune is set as " + self.finetune + " need to be one of no, all, predicate, logits")

        self.base_model = base_model

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                pos_tags: torch.LongTensor,
                dep_tags: torch.LongTensor,
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


        # shape (batch_size, predicates_len, batch_max_senses , pred_dim)
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
        if self.finetune == "no":
            embedded_candidate_preds = self._pred_embedding(predicate_candidates)
            input_sense_logits  = input_dict["sense_logits"].detach()
            input_arc_tag_logits = input_dict["arc_tag_logits"].detach()
            input_arc_tag_probs = input_dict["arc_tag_probs"].detach()
            input_sense_probs = input_dict["sense_probs"].detach()
            embedded_text_input = input_dict["embedded_text_input"].detach()
            predicate_representation = input_dict["predicate_representation"].detach()
            encoded_text =  input_dict["encoded_text"].detach()
            encoded_text_for_predicate =  input_dict["encoded_text_for_predicate"].detach()
        else:
            embedded_candidate_preds = input_dict["embedded_candidate_preds"]
            input_sense_logits  = input_dict["sense_logits"]
            input_arc_tag_logits = input_dict["arc_tag_logits"]
            input_arc_tag_probs = input_dict["arc_tag_probs"]
            input_sense_probs = input_dict["sense_probs"]
            embedded_text_input = input_dict["embedded_text_input"]
            predicate_representation = input_dict["predicate_representation"]
            encoded_text =  input_dict["encoded_text"]
            encoded_text_for_predicate =  input_dict["encoded_text_for_predicate"]

   #     print ("fresh predicate_representation",predicate_representation.size())
        # shape (batch_size, predicates_len, batch_max_senses )
        sense_mask = (predicate_candidates > 0).float()

        if self.normalize_logits:
            input_arc_tag_logits = self.normalize_logits(input_arc_tag_logits)
        #     input_arc_tag_logits = input_arc_tag_logits - input_arc_tag_logits.mean(-1).unsqueeze(-1)

        predicate_indexes = predicate_indexes.long()

        mask = get_text_field_mask(tokens)

        batch_size, sequence_length = mask.size()

        float_mask = mask.float()

        predicate_mask = (predicate_indexes > -1).float()
        graph_mask = predicate_mask.unsqueeze(-1).unsqueeze(1) * float_mask.unsqueeze(-1).unsqueeze(2)


        local_arc_tag_logits = None
        local_sense_logits = None
        if self.re_estimate_logits:

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
            local_sense_logits = embedded_candidate_preds.matmul(predicate_representation.unsqueeze(-1)).squeeze(-1)
            if self.training is False:
                arc_logits = arc_logits + (1 - predicate_mask.unsqueeze(-1).unsqueeze(1)) * 1e9
                local_sense_logits = local_sense_logits - (1 - sense_mask) * 1e9

            # Switch to (batch_size, sequence_length, predicates_len, num_tags)
            arc_tag_logits = arc_tag_logits.permute(0, 2, 3, 1)
            local_arc_tag_logits = torch.cat([arc_logits, arc_tag_logits], dim=-1).contiguous()



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


        output_dict = {
            "tokens": [meta["tokens"] for meta in metadata],
        }

        soft_tags = torch.zeros(size=input_arc_tag_logits.size(), device=input_arc_tag_logits.device)
        soft_tags.scatter_(3, arc_tags.unsqueeze(3) + 1, 1) * graph_mask

        #    print ("sense_logits",sense_logits.size(),sense_logits)
        #    print ("sense_indexes",sense_indexes.size(),sense_indexes)
        soft_index = torch.zeros(size=input_sense_logits.size(), device=input_sense_logits.device)
        soft_index.scatter_(2, sense_indexes.unsqueeze(2), 1) * sense_mask

        if self.training and self.re_estimate_logits:
            local_arc_tag_logits = local_arc_tag_logits + self.refiner.gumbel_t * (
                _sample_gumbel(local_arc_tag_logits.size(), out=local_arc_tag_logits.new()) - soft_tags)
            local_sense_logits = local_sense_logits + self.refiner.gumbel_t * (
                            _sample_gumbel(local_sense_logits.size(), out=local_sense_logits.new()) - soft_index)

        if self.joint_train:

            input_sense_logits  = input_dict["sense_logits"].detach()
            input_arc_tag_logits = input_dict["arc_tag_logits"].detach()
            input_arc_tag_probs = input_dict["arc_tag_probs"].detach()
            input_sense_probs = input_dict["sense_probs"].detach()

        output_dict["loss"] = 0

        if self.rep_dim :
            encoded_text_for_predicate = self._dropout(self.predicte_rep_feedforward(encoded_text_for_predicate))
            encoded_text = self._dropout(self.argument_rep_feedforward(encoded_text))

      #  print ("before feed predicate_representation",predicate_representation.size())
        lists, c_lists, gold_results = self.refiner(encoded_text_for_predicate,
                                                    encoded_text,
                                                    predicate_representation,
                                                    embedded_candidate_preds,
                                                    graph_mask,
                                                    sense_mask,
                                                    predicate_mask,
                                                    input_arc_tag_logits,
                                                    input_sense_logits,
                                                    input_arc_tag_probs,
                                                    input_sense_probs,
                                                    soft_tags,
                                                    soft_index,
                                                    local_arc_tag_logits,
                                                    local_sense_logits)
        gold_score_nodes, gold_score_edges, gold_sense_logits, gold_sense_probs, gold_arc_tag_logits, gold_arc_tag_probs = gold_results
        output_dict["arc_tag_probs_g"] = gold_arc_tag_probs
        output_dict["sense_probs_g"] = gold_sense_probs

        gold_scores = gold_score_nodes.unsqueeze(1) + gold_score_edges
        self._labelled_f1(gold_arc_tag_probs, arc_tags + 1, graph_mask.squeeze(-1), gold_sense_probs,
                          predicate_candidates,
                          predicates, gold_scores, soft_tags * input_arc_tag_logits, n_iteration=0)
        for i, (arc_tag_logits, arc_tag_probs, sense_logits, sense_probs, score_nodes, score_edges) \
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
                                             score_nodes,
                                             score_edges,
                                             gold_score_nodes,
                                             gold_score_edges,
                                             graph_mask,
                                             sense_mask,
                                             predicate_mask)

                output_dict["loss"] = output_dict["loss"] + loss
                # We stack scores here because the f1 measure expects a
                # distribution, rather than a single value.
                #     arc_tag_probs = torch.cat([one_minus_arc_probs, arc_tag_probs*arc_probs], dim=-1)

                scores = score_nodes.unsqueeze(1).cpu() + score_edges.cpu()
                self._labelled_f1(arc_tag_probs, arc_tags + 1, graph_mask.squeeze(-1), sense_probs,
                                  predicate_candidates,
                                  predicates, scores, arc_tag_probs.cpu() * input_arc_tag_logits.cpu(), n_iteration=-i - 1)

        for i, (arc_tag_logits, arc_tag_probs, sense_logits, sense_probs, score_nodes, score_edges) \
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
                                             score_nodes,
                                             score_edges,
                                             gold_score_nodes,
                                             gold_score_edges,
                                             graph_mask,
                                             sense_mask,
                                             predicate_mask)

                output_dict["arc_tag_probs" + str(i)] = arc_tag_probs
                output_dict["sense_probs" + str(i)] = sense_probs

                output_dict["loss"] = output_dict["loss"] + loss
                # We stack scores here because the f1 measure expects a
                # distribution, rather than a single value.
                #     arc_tag_probs = torch.cat([one_minus_arc_probs, arc_tag_probs*arc_probs], dim=-1)

                scores = score_nodes.unsqueeze(1).cpu() + score_edges.cpu() if score_nodes is not None else None
                self._labelled_f1(arc_tag_probs, arc_tags + 1, graph_mask.squeeze(-1), sense_probs,
                                  predicate_candidates,
                                  predicates, scores, arc_tag_probs.cpu() * input_arc_tag_logits.cpu(), n_iteration=i + 1)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        for i in range(10):
            if "arc_tag_probs" + str(i) in output_dict:
                output_dict["predicted_arc_tags" + str(i)] = output_dict["arc_tag_probs" + str(i)].argmax(-1) - 1

                output_dict["sense_argmax" + str(i)] = output_dict["sense_probs" + str(i)].argmax(-1)
        if "arc_tag_probs_g" in output_dict:
            output_dict["predicted_arc_tags_g"] = output_dict["arc_tag_probs_g"].argmax(-1) - 1

            output_dict["sense_argmax_g"] = output_dict["sense_probs_g"].argmax(-1)

        if "arc_tag_probs" in output_dict:
            output_dict["predicted_arc_tags"] = output_dict["arc_tag_probs"].argmax(-1) - 1

            output_dict["sense_argmax"] = output_dict["sense_probs"].argmax(-1)

        return output_dict

    def _max_margin_loss(self, arc_tag_logits,
                         arc_tag_probs,
                         arc_tags,
                         soft_tags,
                         sense_logits,
                         sense_probs,
                         sense_indexes,
                         soft_index,
                         score_nodes,
                         score_edges,
                         gold_score_nodes,
                         gold_score_edges,
                         graph_mask,
                         sense_mask,
                         predicate_mask):
        '''pred_probs: (batch_size, sequence_length, max_senses)'''

        valid_positions = graph_mask.sum().float()

        # shape (batch ,predicate_length ,1)
        #  delta_tag = self._tag_loss((arc_tag_probs+1e-6).log().permute(0, 3, 1, 2), arc_tags + 1).unsqueeze(-1)

        if gold_score_nodes is not None and score_nodes is not None and self.train_score:
            if self.distance == "kl":
                delta_sense = self._sense_loss(torch.nn.functional.log_softmax(sense_logits, dim=-1).permute(0, 2, 1),
                                               sense_indexes).unsqueeze(-1)
                #    delta_sense = self._sense_loss((sense_probs+1e-6).log().permute(0, 2, 1), sense_indexes).unsqueeze(-1)

                # shape (batch ,sequence_length,sequence_length ,1)
                delta_tag = self._tag_loss(torch.nn.functional.log_softmax(arc_tag_logits, dim=-1).permute(0, 3, 1, 2),
                                           arc_tags + 1).unsqueeze(-1)
            elif self.distance == "l1_through":
                delta_sense = torch.abs(sense_probs - soft_index).sum(-1, keepdim=True)
                #    delta_sense = self._sense_loss((sense_probs+1e-6).log().permute(0, 2, 1), sense_indexes).unsqueeze(-1)

                # shape (batch ,sequence_length,sequence_length ,1)
                delta_tag = torch.abs(arc_tag_probs - soft_tags).sum(-1, keepdim=True)

            else:

                delta_sense = torch.abs(sense_probs - soft_index).sum(-1, keepdim=True).detach()
                #    delta_sense = self._sense_loss((sense_probs+1e-6).log().permute(0, 2, 1), sense_indexes).unsqueeze(-1)

                # shape (batch ,sequence_length,sequence_length ,1)
                delta_tag = torch.abs(arc_tag_probs - soft_tags).sum(-1, keepdim=True).detach()
            # shape (batch ,sequence_length,sequence_length ,1)

            node_score_nll = torch.clamp((self.train_score * (-gold_score_nodes + score_nodes) + delta_sense) * predicate_mask.unsqueeze(-1),
                                         min=0).sum() / valid_positions
            edge_node_score_nll = torch.clamp(
                (self.train_score * (-gold_score_edges + score_edges) + delta_tag.sum(-1, keepdim=True)) * graph_mask,
                min=0).sum() / valid_positions

            nll =  node_score_nll + edge_node_score_nll

            return nll
        else:

            if self.distance == "kl":
                delta_sense = self._sense_loss(torch.nn.functional.log_softmax(sense_logits, dim=-1).permute(0, 2, 1),
                                               sense_indexes).unsqueeze(-1)
                #    delta_sense = self._sense_loss((sense_probs+1e-6).log().permute(0, 2, 1), sense_indexes).unsqueeze(-1)

                # shape (batch ,sequence_length,sequence_length ,1)
                delta_tag = self._tag_loss(torch.nn.functional.log_softmax(arc_tag_logits, dim=-1).permute(0, 3, 1, 2),
                                           arc_tags + 1).unsqueeze(-1)
            elif self.distance == "l1_through":
                delta_sense = torch.abs(sense_probs - soft_index).sum(-1, keepdim=True)
                #    delta_sense = self._sense_loss((sense_probs+1e-6).log().permute(0, 2, 1), sense_indexes).unsqueeze(-1)

                # shape (batch ,sequence_length,sequence_length ,1)
                delta_tag = torch.abs(arc_tag_probs - soft_tags).sum(-1, keepdim=True)
            else:  # assume l1

                delta_sense = torch.abs(sense_probs - soft_index).sum(-1, keepdim=True).detach()
                #    delta_sense = self._sense_loss((sense_probs+1e-6).log().permute(0, 2, 1), sense_indexes).unsqueeze(-1)

                # shape (batch ,sequence_length,sequence_length ,1)
                delta_tag = torch.abs(arc_tag_probs - soft_tags).sum(-1, keepdim=True).detach()

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

        delta_sense = self._sense_loss((sense_probs + 1e-6).log().permute(0, 2, 1), sense_indexes).unsqueeze(-1)

        # shape (batch ,sequence_length,sequence_length ,1)
        delta_tag = self._tag_loss((arc_tag_probs + 1e-6).log().permute(0, 3, 1, 2), arc_tags + 1).unsqueeze(-1)

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
        pred_probs = torch.nn.functional.softmax(pred_logits, dim=-1)

        # shape (batch_size, sequence_length, sequence_length,n_tags)
        arc_tag_probs = torch.nn.functional.softmax(arc_tag_logits, dim=-1)

        return arc_tag_probs, pred_probs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        return self._labelled_f1.get_metric(reset, training=self.training)
