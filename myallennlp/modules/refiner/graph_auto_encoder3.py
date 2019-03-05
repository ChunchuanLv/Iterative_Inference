from overrides import overrides
import torch, gc
from torch.nn import Dropout, Linear
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from allennlp.nn.util import masked_softmax, weighted_sum, masked_log_softmax, masked_normalize
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

from allennlp.nn import InitializerApplicator, RegularizerApplicator, Activation
from allennlp.modules import FeedForward
from myallennlp.modules.reparametrization.gumbel_softmax import masked_gumbel_softmax

from myallennlp.auto_grad.my_auto_grad import *
import copy


class GraphAutoEncoder3(Seq2SeqEncoder):
    # pylint: disable=line-too-long
    """
    This class implements the key-value scaled dot product attention mechanism
    detailed in the paper `Attention is all you Need
    <https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .

    The attention mechanism is a weighted sum of a projection V of the inputs, with respect
    to the scaled, normalised dot product of Q and K, which are also both linear projections
    of the input. This procedure is repeated for each attention head, using different parameters.

    Parameters
    ----------
    num_heads : ``int``, required.
        The number of attention heads to use.
    input_dim : ``int``, required.
        The size of the last dimension of the input tensor.
    attention_dim ``int``, required.
        The total dimension of the query and key projections which comprise the
        dot product attention function. Must be divisible by ``num_heads``.
    values_dim : ``int``, required.
        The total dimension which the input is projected to for representing the values,
        which are combined using the attention. Must be divisible by ``num_heads``.
    output_projection_dim : ``int``, optional (default = None)
        The dimensionality of the final output projection. If this is not passed
        explicitly, the projection has size `input_size`.
    attention_dropout_prob : ``float``, optional (default = 0.1).
        The dropout probability applied to the normalised attention
        distributions.
    """

    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int,
                 score_dim: int,
                 dropout: float = 0.3) -> None:
        super(GraphAutoEncoder3, self).__init__()
        self.dropout = dropout
        self._dropout = Dropout(dropout)
        self._node_dim = node_dim
        self._edge_dim = edge_dim
        self._score_dim = score_dim
        self._hidden_dim = hidden_dim

        self.h = nn.functional.softplus

        self.sigmoid = nn.functional.sigmoid

        #   self._message_combine = Linear(hidden_dim, score_dim)

        self._node_score = Linear(edge_dim - 1 + node_dim + hidden_dim, score_dim)

        self._edge_score = Linear(2 * edge_dim - 1, score_dim)
        self._edge_hidden = Linear( hidden_dim, score_dim)
        self._edge_node = Linear( node_dim, score_dim)
        #    self._pred_mess = Linear(hidden_dim, hidden_dim)
        #     self._arg_mess = Linear(hidden_dim, hidden_dim)

        self._dropout = Dropout(dropout)

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return True

    def get_computation_graph(self,
                              nodes: torch.Tensor,
                              edges: torch.Tensor,
                              predicate_rep: torch.Tensor,
                              argument_rep: torch.Tensor,
                              graph_mask: torch.Tensor = None):
        '''

        nodes : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps,node_dim)
        extra_nodes : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps,node_dim)
        edges : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, timesteps,edge_dim)
        argue_rep : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps , extra_dim)
        predicate_rep : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, extra_dim)
        graph : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, timesteps)
        '''

        def wrap_with_dropout(f):
            def inner_wrap(f, x):
                linear = f(x)
                out = self.h(linear)
                out_mask = self._dropout_mask(out)
                return [out * out_mask, out_mask, linear]

            return lambda x: inner_wrap(f, x)

        graph = ComputationGraph(["linear_node_scores", "linear_edge_scores"], ["nodes", "edges"] )

        # edge_rep (batch_size, pre_len,node_dim)
        graph.add_node(ComputationNode("all_edges", ["edges"], lambda edges: (edges * graph_mask).sum(1, keepdim=True),
                                       lambda grad, all_edges, edges: grad.expand_as(edges) * graph_mask))

        def node_score_forward(all_edges, nodes):
            all_edges = all_edges[:, :, :, 1:]
            cated = torch.cat([all_edges.squeeze(1), nodes, predicate_rep], dim=-1)
            return self._node_score(cated)

        def node_score_backward(grad, score, all_edges, nodes):
            # batch_size , pre_len , catted_dim
            backed_grad = grad.matmul(self._node_score.weight)

            grad_to_all_edges = backed_grad[:, :, :(all_edges.size(-1) - 1)].unsqueeze(1)

            padded_grad_toall_edges = torch.cat(
                [torch.zeros_like(grad_to_all_edges[:, :, :, 0]).unsqueeze(-1), grad_to_all_edges], dim=-1) * graph_mask

            grad_to_nodes = backed_grad[:, :, grad_to_all_edges.size(-1):grad_to_all_edges.size(-1) + nodes.size(-1)]

            return [padded_grad_toall_edges, grad_to_nodes]

        # A tensor of shape (batch_size, timesteps, pre_len ,hidden_dim)
        graph.add_node(ComputationNode("linear_node_scores", ["all_edges", "nodes"],
                                       node_score_forward,
                                       node_score_backward))

        def edge_score_forward(edges, all_edges, nodes):
            negative_edges = (all_edges - edges) * graph_mask
            negative_edges = negative_edges[:, :, :, 1:]

            hidden_mapped = self._edge_hidden(argument_rep)
            node_mapped = self._edge_node(nodes)
            cated = torch.cat([edges, negative_edges], dim=-1)
            return self._edge_score(cated) + node_mapped.unsqueeze(1).repeat(1, edges.size(1), 1, 1) + hidden_mapped.unsqueeze(2).repeat(1, 1, edges.size(2), 1)

        def edge_score_backward(grad, score, edges, all_edges, nodes):
            # batch_size , seq_len, pre_len , catted_dim
            backed_grad = grad.matmul(self._edge_score.weight)

            grad_to_edges = backed_grad[:, :, :, :edges.size(-1)] * graph_mask

            current = edges.size(-1)

            grad_to_negative_edges = backed_grad[:, :, :, current:]


            padded_grad_to_negative_edges = torch.cat(
                [torch.zeros_like(grad_to_negative_edges[:, :, :, 0]).unsqueeze(-1), grad_to_negative_edges],
                dim=-1) * graph_mask
            grad_to_edges = grad_to_edges - padded_grad_to_negative_edges

            grad_to_all_edges = padded_grad_to_negative_edges.sum(1, keepdim=True)

            grad_to_nodes = (grad.matmul(self._edge_node.weight) * graph_mask).sum(1)

            return [grad_to_edges, grad_to_all_edges, grad_to_nodes]

        # A tensor of shape (batch_size, timesteps, pre_len ,hidden_dim)
        graph.add_node(ComputationNode("linear_edge_scores", ["edges", "all_edges", "nodes"],
                                       edge_score_forward,
                                       edge_score_backward))

        graph.forward([nodes, edges])
        return graph

    def score(self, graph: ComputationGraph, graph_mask):
        # shape  (batch_size, timesteps, timesteps , score_dim)
        linear_edge_scores = graph.get_tensor_by_name("linear_edge_scores")
        linear_node_scores = graph.get_tensor_by_name("linear_node_scores")
        return self.h(linear_node_scores) * graph_mask[:, 0], self.h(linear_edge_scores) * graph_mask

    def score_gradient(self, graph: ComputationGraph, graph_mask: torch.Tensor):
        linear_edge_scores = graph.get_tensor_by_name("linear_edge_scores")
        linear_node_scores = graph.get_tensor_by_name("linear_node_scores")

        return self._dropout(self.sigmoid(linear_node_scores)) * graph_mask[:, 0], \
               self._dropout( self.sigmoid(linear_edge_scores) )* graph_mask

    def get_score_and_graph(self,
                            nodes: torch.Tensor,
                            edges: torch.Tensor,
                            predicate_rep: torch.Tensor, argument_rep: torch.Tensor,
                            graph_mask: torch.Tensor = None):
        '''

        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, timesteps, input_dim)
        '''
        graph = self.get_computation_graph(nodes, edges, predicate_rep, argument_rep, graph_mask)

        return self.score(graph, graph_mask), graph

    @overrides
    def forward(self,
                nodes: torch.Tensor,
                edges: torch.Tensor,
                predicate_rep: torch.Tensor,
                argument_rep: torch.Tensor,
                graph_mask: torch.Tensor = None,
                get_grads: bool = True) -> torch.FloatTensor:
        """
        Parameters
        ----------
        nodes : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size,  pre_length, input_dim)
        predicate_rep : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size,  pre_length, input_dim)
        graph : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, timesteps,1)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).

        Returns
        -------
        gradient w.r.t to score m1 Relu m2 input
        """

        scores, graph = self.get_score_and_graph(nodes, edges, predicate_rep, argument_rep, graph_mask)
        grads = self.score_gradient(graph, graph_mask)
        score_nodes, score_edges = scores
        grad_to_nodes, grad_to_edges = graph.backward(grads)
        del graph
        return score_nodes, score_edges, grad_to_nodes, grad_to_edges


def main():
    '''
    node_dim: int,
                 edge_dim: int,
                 hidden_dim: int,
                 score_dim: int,

    :return:
    '''
    node_dim = 2
    edge_dim = 7
    hidden_dim = 15
    score_dim = 10

    batch_size = 2
    graph_size = 3
    mlpfbb = GraphAutoEncoder3(node_dim, edge_dim, hidden_dim, score_dim)
    mlpfbb.eval()
    nodes = torch.rand(batch_size, graph_size, node_dim, requires_grad=True)
    argument_rep = torch.rand(batch_size, graph_size * 2, hidden_dim)
    predicate_rep = torch.rand(batch_size, graph_size, hidden_dim)
    edges = torch.rand(batch_size, graph_size * 2, graph_size, edge_dim, requires_grad=True)

    sizes = [[6, 3], [4, 2]]
    graph_mask = torch.zeros(size=edges.size()[:-1])

    for i, [s0, s1] in enumerate(sizes):
        data_t = torch.ones(s0, s1)
        graph_mask[i].narrow(0, 0, s0).narrow(1, 0, s1).copy_(data_t)
    graph_mask = graph_mask.unsqueeze(-1)
    graph = None

    score_nodes, score_edges, grad_to_nodes, grad_to_edges = mlpfbb(nodes, edges, predicate_rep, argument_rep,
                                                                    graph_mask)

    l = score_nodes.sum() + score_edges.sum()
    l.backward()

    print("gradient_to_edge", grad_to_edges)
    print("edges.grad", edges.grad)

    print("diff", (grad_to_edges - edges.grad).pow(2).sum())

    print("grad_to_nodes", grad_to_nodes)
    print("nodes.grad", nodes.grad)

    print("diff", (grad_to_nodes - nodes.grad).pow(2).sum())

    print("grad_to_edges[0]", grad_to_edges[1][:, :, 2])


if __name__ == "__main__":
    main()
