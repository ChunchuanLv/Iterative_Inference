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
from myallennlp.modules.refiner.graph_auto_encoder2 import GraphAutoEncoder2
from myallennlp.modules.refiner.graph_auto_encoder3 import GraphAutoEncoder3
from myallennlp.modules.reparametrization.gumbel_softmax import hard, _sample_gumbel, inplace_masked_gumbel_softmax


@Seq2SeqEncoder.register("spens_refiner")
class SRLRefiner(Seq2SeqEncoder):
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
                 denoise: bool = False,
                 global_gating: bool = False,
                 gating: bool = False,
                 dropout_local: bool = True,
                 detach_type: str = "no",
                 gradient_pass:bool=False,
                 score_dim: int = 40,
                 graph_encoder:int=2,
                 corrupt_input: bool = False,
                 corruption_rate: float = 0.1,
                 corruption_iterations: int = 1,
                 testing_iterations: int = 5,
                 gumbel_t: float = 0.0) -> None:
        super(SRLRefiner, self).__init__()
        self.gumbel_t = gumbel_t
        self.detach_type = detach_type
        self.gradient_pass = gradient_pass
        self.denoise = denoise
        self.gating = gating
        self.graph_encoder = graph_encoder
        self.dropout_local = dropout_local
        self.global_gating = global_gating
        self.stright_through = stright_through
        self.iterations = iterations
        self.score_dim = score_dim
        self._dropout = Dropout(dropout)
        self.testing_iterations = testing_iterations
        self.dropout = dropout
        self.corruption_rate = corruption_rate
        self._corrupt_mask = lambda x: torch.bernoulli(
            x.data.new(x.data.size()[:-1]).fill_(1 - self.corruption_rate)).unsqueeze(-1)
        self.corrupt_input = corrupt_input
        self.corruption_iterations = corruption_iterations

    def initialize_network(self, n_tags: int, sense_dim: int, rep_dim: int):
        self.graph_scorer = GraphAutoEncoder(node_dim=sense_dim, edge_dim=n_tags + 1, rep_dim=rep_dim,
                                              score_dim=self.score_dim, dropout=self.dropout)
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
                      predicate_hidden,
                      input_sense_logits,
                      argument_hidden,
                      input_arc_tag_logits,
                      predicate_representation,
                      embedded_candidate_preds,
                      graph_mask,
                      sense_mask,
                      predicate_mask,
                      soft_tags,
                      soft_index,
                      arc_tag_probs,
                      sense_probs):

    #    print ("predicate_representation",predicate_representation.size())
        predicate_emb = ((sense_probs.unsqueeze(2).matmul(embedded_candidate_preds)).squeeze(
            2) + predicate_representation) * predicate_mask.unsqueeze(-1)
        score_nodes, score_edges, grad_to_predicate_emb, grad_to_arc_tag_probs = self.graph_scorer(predicate_emb,
                                                                                                   arc_tag_probs * graph_mask,
                                                                                                   predicate_hidden,
                                                                                                   argument_hidden,
                                                                                                   graph_mask)
        grad_to_sense_probs = embedded_candidate_preds.matmul(grad_to_predicate_emb.unsqueeze(-1)).squeeze(-1)

        input_arc_tag_logits = input_arc_tag_logits + grad_to_arc_tag_probs
        input_sense_logits = input_sense_logits + grad_to_sense_probs


        arc_tag_logits = input_arc_tag_logits
        sense_logits = input_sense_logits
        if self.gumbel_t and self.training:
            arc_tag_logits = arc_tag_logits + self.gumbel_t * (
                        _sample_gumbel(arc_tag_logits.size(), out=arc_tag_logits.new()))
            sense_logits = sense_logits + self.gumbel_t * (
                        _sample_gumbel(sense_logits.size(), out=sense_logits.new()) )

        arc_tag_probs_soft = torch.nn.functional.softmax(arc_tag_logits, dim=-1)

        sense_probs_soft = torch.nn.functional.softmax(sense_logits, dim=-1)

        arc_tag_probs = hard(arc_tag_probs_soft, graph_mask) if self.stright_through else arc_tag_probs_soft
        sense_probs = hard(sense_probs_soft, sense_mask) if self.stright_through  else sense_probs_soft

        return input_arc_tag_logits, arc_tag_probs, input_sense_logits, sense_probs, score_nodes, score_edges

