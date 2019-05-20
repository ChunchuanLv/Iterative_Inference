from overrides import overrides
import torch
from torch.nn import Dropout
import torch.nn.functional as F

import copy
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules import InputVariationalDropout, Embedding
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn.activations import Activation
from allennlp.nn import InitializerApplicator, RegularizerApplicator

from allennlp.nn.util import get_device_of, masked_log_softmax, get_lengths_from_binary_sequence_mask, masked_softmax
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
import numpy
from allennlp.nn.util import masked_softmax, weighted_sum
from allennlp.nn.util import add_positional_features
from typing import List, Tuple, Dict
from myallennlp.modules.refiner.mlp_forward_backward import MLPForwardBackward
from myallennlp.modules.refiner.graph_auto_encoder import GraphAutoEncoder
from myallennlp.modules.reparametrization.gumbel_softmax import hard, _sample_gumbel, inplace_masked_gumbel_softmax
from myallennlp.modules.refiner.refiner import SRLRefiner

from myallennlp.modules.sparsemax import sparse_max
from torch.nn.modules import Linear
@Seq2SeqEncoder.register("direct_refiner")
class DirectSRLRefiner(SRLRefiner):
    # pylint: disable=line-too-long
    """
    Implements a stacked self-attention encoder similar to the Transformer
    architecture in `Attention is all you Need
    <https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .

    This encoder combines 3 layers in a 'block':

    1. A 2 layer FeedForward network.
    2. Multi-headed self attention, which uses 2 learnt linear projections
       to perform a dot-product similarity between every pair of elements
       scaled by the square root of the sequence length.
    3. Layer Normalisation.

    These are then stacked into ``num_layers`` layers.

    Parameters
    ----------
    input_dim : ``int``, required.
        The input dimension of the encoder.
    hidden_dim : ``int``, required.
        The hidden dimension used for the _input_ to self attention layers
        and the _output_ from the feedforward layers.
    projection_dim : ``int``, required.
        The dimension of the linear projections for the self-attention layers.
    feedforward_hidden_dim : ``int``, required.
        The middle dimension of the FeedForward network. The input and output
        dimensions are fixed to ensure sizes match up for the self attention layers.
    num_layers : ``int``, required.
        The number of stacked self attention -> feedfoward -> layer normalisation blocks.
    num_attention_heads : ``int``, required.
        The number of attention heads to use per layer.
    use_positional_encoding: ``bool``, optional, (default = True)
        Whether to add sinusoidal frequencies to the input tensor. This is strongly recommended,
        as without this feature, the self attention layers have no idea of absolute or relative
        position (as they are just computing pairwise similarity between vectors of elements),
        which can be important features for many tasks.
    dropout_prob : ``float``, optional, (default = 0.1)
        The dropout probability for the feedforward network.
    residual_dropout_prob : ``float``, optional, (default = 0.2)
        The dropout probability for the residual connections.
    attention_dropout_prob : ``float``, optional, (default = 0.1)
        The dropout probability for the attention distributions in each attention layer.
    """




    def initialize_network(self, n_tags: int, sense_dim: int, rep_dim: int):
        self.n_tags = n_tags

        self._arc_tag_arg_enc = Linear(rep_dim, self.hidden_dim)

        if self.use_predicate_rep:
            self._arc_tag_pred_enc = Linear(rep_dim, self.hidden_dim)

        if self.graph_type != 2:
            self._arc_tag_sense_enc = Linear(sense_dim, self.hidden_dim)

        if self.graph_type == 1:
            self._arc_tag_tags_enc = Linear(n_tags+1, self.hidden_dim)
        elif self.graph_type == 2:
            self._arc_tag_tags_enc = Linear(n_tags+1, self.hidden_dim)
        else:
            self._arc_tag_tags_enc = Linear(2*n_tags+1, self.hidden_dim)

        if self.weight_tie:
            self.arc_tag_refiner = lambda x: x.matmul(self._arc_tag_tags_enc.weight[:,:n_tags+1])

            if self.graph_type != 2:
                self.predicate_linear = Linear(rep_dim+n_tags+sense_dim,self.hidden_dim)
            else:
                self.predicate_linear = Linear(rep_dim+sense_dim,self.hidden_dim)

            self.predicte_refiner = lambda x: self._dropout(self.activation(self.predicate_linear(x)))\
                    .matmul(self.predicate_linear.weight[:,:sense_dim])
        else:
            self.arc_tag_refiner =  FeedForward(self.hidden_dim,1,
                                                n_tags+1,Activation.by_name("linear")(),dropout=self.dropout)

            self.predicte_refiner = FeedForward(rep_dim+n_tags+sense_dim, 2,
                [self.hidden_dim]+[sense_dim],
                [self.activation]+[Activation.by_name("linear")()],dropout=self.dropout)
    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return True

    def one_iteration(self,
                      predicate_rep,
                      input_sense_logits,
                      argument_rep,
                      input_arc_tag_logits,
                      embedded_candidate_preds,
                      graph_mask,
                      sense_mask,
                      predicate_mask,
                      arc_tag_probs,
                      sense_probs,
                      soft_tags,
                      soft_index,
                      step_size=1,
                      sense_rep = None):

        all_edges = (arc_tag_probs * graph_mask).sum(1)
        all_active_edges = all_edges[:, :, 1:]

        # shape (batch_size, predicates_len, node_dim)
        predicate_emb =  self._dropout((sense_probs.unsqueeze(2).matmul(embedded_candidate_preds)).squeeze(
            2) * predicate_mask.unsqueeze(-1))

        if self.graph_type == 3:
            predicate_representation = self._dropout(
                self.predicte_refiner(torch.cat([predicate_rep,all_active_edges], dim=-1)))
        elif self.graph_type == 2:
            predicate_representation = self._dropout(
                self.predicte_refiner(torch.cat([predicate_emb,predicate_rep], dim=-1)))
        else :
            predicate_representation = self._dropout(
                self.predicte_refiner(torch.cat([predicate_emb,predicate_rep,all_active_edges], dim=-1)))

        # (batch_size, predicates_len, max_sense)
        sense_logits = embedded_candidate_preds.matmul(predicate_representation.unsqueeze(-1)).squeeze(-1)



        all_other_edges = (all_edges.unsqueeze(1) - arc_tag_probs )* graph_mask
        all_other_edges = all_other_edges[:,:,:,1:]

        encoded_arg_enc = self._arc_tag_arg_enc(argument_rep)

        if self.use_predicate_rep:
            encoded_pred_enc = self._arc_tag_pred_enc(predicate_rep)


        if self.graph_type == 1:
            tag_input_date = arc_tag_probs
        elif self.graph_type == 2:
            tag_input_date = arc_tag_probs
        else:
            tag_input_date = torch.cat([arc_tag_probs,all_other_edges ],dim=-1)

        tag_enc = self._arc_tag_tags_enc(tag_input_date)

        if self.use_predicate_rep:
            linear_added = tag_enc + encoded_arg_enc.unsqueeze(2).expand_as(tag_enc) + encoded_pred_enc.unsqueeze(1).expand_as(tag_enc)
        else:
            linear_added = tag_enc + encoded_arg_enc.unsqueeze(2).expand_as(tag_enc)

        if self.graph_type != 2:
            predicate_emb_enc = self._arc_tag_sense_enc(predicate_emb)
            linear_added = linear_added + predicate_emb_enc.unsqueeze(1).expand_as(tag_enc)


        arc_tag_logits = self.arc_tag_refiner(self._dropout(self.activation(linear_added)))


        sense_logits = sense_logits + input_sense_logits
        arc_tag_logits = arc_tag_logits + input_arc_tag_logits


        if self.training and self.gumbel_t:
            arc_tag_logits = arc_tag_logits + self.gumbel_t * _sample_gumbel(arc_tag_logits.size(),
                                                                             out=arc_tag_logits.new())
            sense_logits = sense_logits + 5 * self.gumbel_t * _sample_gumbel(sense_logits.size(),
                                                                         out=sense_logits.new())
        if self.training and self.subtract_gold and soft_tags is not None:
            arc_tag_logits = arc_tag_logits + self.subtract_gold * (- soft_tags)
            sense_logits = sense_logits + self.subtract_gold * (- soft_index)

        arc_tag_probs, sense_probs = self.decode(arc_tag_logits,sense_logits,graph_mask,sense_mask,soft_tags,soft_index,arc_tag_probs,sense_probs,step_size)

        return arc_tag_logits, arc_tag_probs, sense_logits, sense_probs, None, None