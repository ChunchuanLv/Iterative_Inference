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
                 hidden_dim: int = 40,
                 corruption_rate: float = 0.1,
                 corruption_iterations: int = 1,
                 testing_iterations: int = 5,
                 gumbel_t: float = 0.0) -> None:
        super(SRLRefiner, self).__init__()
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

    def initialize_network(self, n_tags: int, sense_dim: int, rep_dim: int):
        pass

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
        pass


    @overrides
    def forward(self, predicate_hidden,
                argument_hidden,
                embedded_candidate_preds,
                graph_mask,
                sense_mask,
                predicate_mask,
                input_arc_tag_logits,
                input_sense_logits,
                arc_tag_probs_soft,
                sense_probs_soft,
                soft_tags=None,
                soft_index=None,
                compute_gold=False):  # pylint: disable=arguments-differ
        '''

        :param predict_representation:
        :param argument_representation:
        :param input_arc_tag_logits:
        :param arc_tag_probs:
        :param mask:
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.
        :param arc_tags:
        :param pre_dropout_mask:
        :param embedded_candidate_preds: (batch_size, sequence_length, max_senses, node_dim)
        :return:
        '''
        # shape (batch_size, predicates_len, batch_max_senses )  sense_mask

        # scores = scores + (arc_tag_probs * input_arc_tag_logits).sum(-1, keepdim=True)/scores.size(-1)*graph_mask

  #      print ("refiner2 in predicate_representation",predicate_representation.size())

        arc_tag_probs = hard(arc_tag_probs_soft, graph_mask) if self.stright_through else arc_tag_probs_soft
        sense_probs = hard(sense_probs_soft, sense_mask) if self.stright_through else sense_probs_soft

        arc_tag_logits_list = [input_arc_tag_logits]
        arc_tag_probs_list = [arc_tag_probs]
        sense_logits_list = [input_sense_logits]
        sense_probs_list = [sense_probs]
        node_scores_list = []
        edge_scores_list = []


        iterations = self.iterations if self.training else self.testing_iterations
        #      arg_intermediates_list = []
        #  predicate_representation = predicate_representation.detach()
        for i in range(iterations):

            arc_tag_logits, arc_tag_probs, sense_logits, sense_probs, score_nodes, score_edges = self.one_iteration(
                predicate_hidden,
                input_sense_logits,
                argument_hidden,
                input_arc_tag_logits,
                embedded_candidate_preds,
                graph_mask,
                sense_mask,
                predicate_mask,
                arc_tag_probs,
                sense_probs
            )

            arc_tag_logits_list.append(arc_tag_logits)
            sense_logits_list.append(sense_logits)
            arc_tag_probs_list.append(arc_tag_probs)
            sense_probs_list.append(sense_probs)
            node_scores_list.append(score_nodes)
            edge_scores_list.append(score_edges)

        node_scores_list.append(None)
        edge_scores_list.append(None)

        c_arc_tag_logits_list = []
        c_arc_tag_probs_list = []
        c_sense_logits_list = []
        c_sense_probs_list = []
        c_node_scores_list = []
        c_edge_scores_list = []

        if soft_tags is not None and compute_gold:
            gold_arc_tag_logits, gold_arc_tag_probs, gold_sense_logits, gold_sense_probs, gold_score_nodes, gold_score_edges = self.one_iteration(
                        predicate_hidden,
                        input_sense_logits,
                        argument_hidden,
                        input_arc_tag_logits,
                        embedded_candidate_preds,
                        graph_mask,
                        sense_mask,
                        predicate_mask,
                        soft_tags,
                        soft_index
                )

            gold_results = [gold_score_nodes, gold_score_edges, gold_sense_logits, gold_sense_probs,
                            gold_arc_tag_logits, gold_arc_tag_probs]
        else:
            gold_results = [None] * 6

        iterations = self.corruption_iterations if self.training else 1
        if self.corruption_rate and soft_tags is not None and self.training:
            for i in range(iterations):
                c_soft_tags = self.corrupt_one_hot(soft_tags, graph_mask)
                c_soft_index = self.corrupt_index(soft_index, sense_mask)
                gold_arc_tag_logits, gold_arc_tag_probs, gold_sense_logits, gold_sense_probs, score_nodes, score_edges = self.one_iteration(
                        predicate_hidden,
                        input_sense_logits,
                        argument_hidden,
                        input_arc_tag_logits,
                        embedded_candidate_preds,
                        graph_mask,
                        sense_mask,
                        predicate_mask,
                        c_soft_tags,
                        c_soft_index
                         )

                c_arc_tag_logits_list.append(gold_arc_tag_logits)
                c_arc_tag_probs_list.append(gold_arc_tag_probs)
                c_sense_logits_list.append(gold_sense_logits)
                c_sense_probs_list.append(gold_sense_probs)
                c_node_scores_list.append(score_nodes)
                c_edge_scores_list.append(score_edges)

        return (arc_tag_logits_list, arc_tag_probs_list, sense_logits_list, sense_probs_list, node_scores_list,
                edge_scores_list), \
               (
               c_arc_tag_logits_list, c_arc_tag_probs_list, c_sense_logits_list, c_sense_probs_list, c_node_scores_list,
               c_edge_scores_list), gold_results

    def corrupt_one_hot(self, gold, mask, sample=None):
        corrupt_mask = self._corrupt_mask(mask)
        #  corrupt_mask =1
        if sample is None:
            sample = mask * torch.distributions.one_hot_categorical.OneHotCategorical(
                logits=torch.zeros_like(gold)).sample()

        return (gold * corrupt_mask + (1 - corrupt_mask) * sample) * mask

    def corrupt_index(self, soft_index, sense_mask, sample=None):
        sense_mask = sense_mask + 1e-6
        mask_sum = sense_mask.sum(dim=-1, keepdim=True)

        # batch pre_len 1
        corrupt_mask = self._corrupt_mask(mask_sum)

        if sample is None:
            probs = sense_mask / mask_sum
            sample = torch.distributions.one_hot_categorical.OneHotCategorical(probs=probs).sample()

        out = sense_mask * (soft_index * corrupt_mask + (1 - corrupt_mask) * sample)
        ''' 
        print ("soft_index",soft_index.size())
        print ("sense_mask",sense_mask.size())
        print ("mask_sum",mask_sum.size())
        print ("probs",probs.size())
        print ("corrupt_mask",corrupt_mask.size())
        print ("random_sample",random_sample.size())
        print ("out",out.size()) '''

        return out
