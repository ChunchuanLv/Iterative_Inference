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



    def __init__(self,
                 iterations: int = 1,
                 dropout: float = 0.3,
                 stright_through: bool = False,
                 hidden_dim: int = 40,
                 corruption_rate: float = 0.1,
                 corruption_iterations: int = 1,
                 testing_iterations: int = 5,
                 gumbel_t: float = 0.0,
                 weight_tie:bool=False,
                 activation:str="sigmoid") -> None:
        super(DirectSRLRefiner, self).__init__()
        self.gumbel_t = gumbel_t
        self.stright_through = stright_through
        self.iterations = iterations
        self.hidden_dim = hidden_dim
        self._dropout = Dropout(dropout)
        self.testing_iterations = testing_iterations
        self.dropout = dropout
        self.corruption_rate = corruption_rate
        self._corrupt_mask = lambda x: torch.bernoulli(
            x.data.new(x.data.size()[:-1]).fill_(1 - self.corruption_rate)).unsqueeze(-1)
        self.corruption_iterations = corruption_iterations
        self.weight_tie = weight_tie
        self.activation =  Activation.by_name(activation)()
    def initialize_network(self, n_tags: int, sense_dim: int, rep_dim: int):

        self._arc_tag_arg_enc = Linear(rep_dim, self.hidden_dim)
        self._arc_tag_predicate_enc = Linear(sense_dim, self.hidden_dim)
        self._arc_tag_tags_enc = Linear(2*n_tags+1, self.hidden_dim)


        if self.weight_tie:
            self.arc_tag_refiner = lambda x: x.matmul(self._arc_tag_tags_enc.weight[:,:n_tags+1])

            self.predicate_linear = Linear(rep_dim+n_tags+sense_dim,self.hidden_dim)
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
                      sense_probs):

        all_edges = (arc_tag_probs * graph_mask).sum(1)
        all_active_edges = all_edges[:, :, 1:]

        # shape (batch_size, predicates_len, node_dim)
        predicate_emb = (sense_probs.unsqueeze(2).matmul(embedded_candidate_preds)).squeeze(
            2) * predicate_mask.unsqueeze(-1)

        predicate_representation = self._dropout(
            self.predicte_refiner(torch.cat([predicate_emb,predicate_rep,all_active_edges], dim=-1)))

        # (batch_size, predicates_len, max_sense)
        sense_logits = embedded_candidate_preds.matmul(predicate_representation.unsqueeze(-1)).squeeze(-1)



        all_other_edges = (all_edges.unsqueeze(1) - arc_tag_probs )* graph_mask
        all_other_edges = all_other_edges[:,:,:,1:]

        encoded_text_enc = self._arc_tag_arg_enc(argument_rep)
        predicate_emb_enc = self._arc_tag_predicate_enc(predicate_emb)

        tag_input_date = torch.cat([arc_tag_probs,all_other_edges ],dim=-1)
        tag_enc = self._arc_tag_tags_enc(tag_input_date)

        linear_added = tag_enc + encoded_text_enc.unsqueeze(2).expand_as(tag_enc)+ predicate_emb_enc.unsqueeze(1).expand_as(tag_enc)

        arc_tag_logits = self.arc_tag_refiner(self.activation(self._dropout(linear_added)))


        sense_logits = sense_logits + input_sense_logits
        arc_tag_logits = arc_tag_logits + input_arc_tag_logits

        if self.training:
            arc_tag_logits = arc_tag_logits + self.gumbel_t * (
                        _sample_gumbel(arc_tag_logits.size(), out=arc_tag_logits.new()))
            sense_logits = sense_logits + self.gumbel_t * (
                        _sample_gumbel(sense_logits.size(), out=sense_logits.new()) )


        arc_tag_probs = torch.nn.functional.softmax(arc_tag_logits, dim=-1)*graph_mask
        sense_probs = torch.nn.functional.softmax(sense_logits, dim=-1)*sense_mask



        return arc_tag_logits, arc_tag_probs, sense_logits, sense_probs, None, None