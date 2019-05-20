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


class GraphAutoEncoder2(Seq2SeqEncoder):
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
                 sense_dim: int,
                 n_tags: int,
                 rep_dim: int,
                 score_dim: int,
                 dropout: float = 0.3,
                 use_predicate_rep = True) -> None:
        super(GraphAutoEncoder2, self).__init__()
        self.dropout = dropout
        self._dropout = Dropout(dropout)
        self._node_dim = sense_dim
        self._score_dim = score_dim
        self._hidden_dim = rep_dim
        self.use_predicate_rep = use_predicate_rep
        self.h = nn.functional.softplus

        self.sigmoid = nn.functional.sigmoid

        self._arc_tag_arg_enc = Linear(rep_dim, score_dim)
        if self.use_predicate_rep :
            self._arc_tag_pred_enc = Linear(rep_dim, score_dim)

        self._arc_tag_sense_enc = Linear(sense_dim, score_dim)
        self._arc_tag_tags_enc = Linear(2*n_tags+1, score_dim)

        self._arg_score_weight= Linear(score_dim, 1)

        self._predicte_score= Linear(rep_dim + n_tags  + sense_dim, score_dim)

        self._predicte_score_weight = Linear(score_dim, 1)

        self._dropout = Dropout(dropout)

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return True

    def get_computation_graph(self,
                              predicate_emb: torch.Tensor,
                              arc_tag_probs: torch.Tensor,
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

        def dropout_after(f):
            def inner_wrap(f,x):
                out = f(x)
                out_mask = self._dropout_mask(out)
                return [out*out_mask,out_mask]

            return lambda x: inner_wrap(f,x)

        graph = ComputationGraph(["predicate_scores", "role_scores"], ["predicate_emb", "arc_tag_probs"] )

        # edge_rep (batch_size, pre_len,node_dim)
        graph.add_node(ComputationNode("all_tag_probs", ["arc_tag_probs"], lambda arc_tag_probs: (arc_tag_probs * graph_mask).sum(1, keepdim=True),
                                       lambda grad, all_tag_probs, arc_tag_probs: grad.expand_as(arc_tag_probs) * graph_mask))

        def node_score_forward(all_edges, predicate_emb):
            all_edges = all_edges[:, :, :, 1:]
            cated = torch.cat([all_edges.squeeze(1), predicate_emb, predicate_rep], dim=-1)

            linear = self._predicte_score(cated)
            non_linear = self.h(linear)
            dropped =  non_linear * self._predicte_score_weight.weight.unsqueeze(0)
            return [dropped, linear]


        def node_score_backward(grad, score,  linear, all_edges, predicate_emb):
            grad = grad * self._predicte_score_weight.weight.unsqueeze(0)  * self.sigmoid(linear)
            # batch_size , pre_len , catted_dim
            backed_grad = grad.matmul(self._predicte_score.weight)

            grad_to_all_edges = backed_grad[:, :, :(all_edges.size(-1) - 1)].unsqueeze(1)

            padded_grad_toall_edges = torch.cat(
                [torch.zeros_like(grad_to_all_edges[:, :, :, 0]).unsqueeze(-1), grad_to_all_edges], dim=-1) * graph_mask

            grad_to_predicate_emb = backed_grad[:, :, grad_to_all_edges.size(-1):grad_to_all_edges.size(-1) + predicate_emb.size(-1)]

            return [padded_grad_toall_edges, grad_to_predicate_emb]

        # A tensor of shape (batch_size, timesteps, pre_len ,hidden_dim)
        graph.add_node(ComputationNode("predicate_scores", ["all_tag_probs", "predicate_emb"],
                                       node_score_forward,
                                       node_score_backward))


        def edge_score_forward(arc_tag_probs, all_tag_probs, predicate_emb):
            other_tags = (all_tag_probs - arc_tag_probs) * graph_mask
            other_tags = other_tags[:, :, :, 1:]

            encoded_arg_enc = self._arc_tag_arg_enc(argument_rep)
            if self.use_predicate_rep:
                encoded_pred_enc = self._arc_tag_pred_enc(predicate_rep)

            predicate_emb_enc = self._arc_tag_sense_enc(predicate_emb)

            tag_input_date = torch.cat([arc_tag_probs, other_tags], dim=-1)
            tag_enc = self._arc_tag_tags_enc(tag_input_date)

            if self.use_predicate_rep:
                linear_added = tag_enc + encoded_arg_enc.unsqueeze(2).expand_as(tag_enc) + predicate_emb_enc.unsqueeze(
                    1).expand_as(tag_enc) + encoded_pred_enc.unsqueeze(1).expand_as(tag_enc)
            else:
                linear_added = tag_enc + encoded_arg_enc.unsqueeze(2).expand_as(tag_enc) + predicate_emb_enc.unsqueeze(
                    1).expand_as(tag_enc)
            non_linear = self.h(linear_added)
            dropped =  non_linear * self._arg_score_weight.weight.unsqueeze(0).unsqueeze(0)
            return [dropped, linear_added]

        def edge_score_backward(grad, role_hidden, linear, arc_tag_probs, all_tag_probs, predicate_emb):
            grad = grad * self._arg_score_weight.weight.unsqueeze(0).unsqueeze(0)  * self.sigmoid(linear)
            # batch_size , seq_len, pre_len , catted_dim
            grad_to_tags = grad.matmul(self._arc_tag_tags_enc.weight)

            current = arc_tag_probs.size(-1)

            grad_to_arc_tag_probs = grad_to_tags[:, :, :, :current] * graph_mask

            grad_to_other_tags = grad_to_tags[:, :, :, current:]* graph_mask


            padded_grad_to_other_tags = torch.cat(
                [torch.zeros_like(grad_to_other_tags[:, :, :, 0]).unsqueeze(-1), grad_to_other_tags],
                dim=-1) * graph_mask

            grad_to_arc_tag_probs = grad_to_arc_tag_probs - padded_grad_to_other_tags

            grad_to_all_tag_probs = padded_grad_to_other_tags.sum(1, keepdim=True)

            grad_to_predicate_emb = (grad.matmul(self._arc_tag_sense_enc.weight) * graph_mask).sum(1)

            return [grad_to_arc_tag_probs, grad_to_all_tag_probs, grad_to_predicate_emb]

        # A tensor of shape (batch_size, timesteps, pre_len ,hidden_dim)
        graph.add_node(ComputationNode("role_scores", ["arc_tag_probs", "all_tag_probs", "predicate_emb"],
                                       edge_score_forward,
                                       edge_score_backward))





        graph.forward([predicate_emb, arc_tag_probs])
        return graph

    def score_gradient(self, graph: ComputationGraph, graph_mask: torch.Tensor):
        predicate_scores = graph.get_tensor_by_name("predicate_scores")
        role_scores = graph.get_tensor_by_name("role_scores")

        return torch.ones_like(predicate_scores)*  graph_mask[:,0], torch.ones_like(role_scores)*  graph_mask

    def get_score_and_graph(self,
                            predicate_emb: torch.Tensor,
                            arc_tag_probs: torch.Tensor,
                            predicate_rep: torch.Tensor, argument_rep: torch.Tensor,
                            graph_mask: torch.Tensor = None):
        '''

        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, timesteps, input_dim)
        '''
        graph = self.get_computation_graph(predicate_emb, arc_tag_probs, predicate_rep, argument_rep, graph_mask)

        return (graph.get_tensor_by_name("predicate_scores")*  graph_mask[:,0], graph.get_tensor_by_name("role_scores")*  graph_mask), graph


    @overrides
    def forward(self,
                predicate_emb: torch.Tensor,
                arc_tag_probs: torch.Tensor,
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

        scores, graph = self.get_score_and_graph(predicate_emb, arc_tag_probs, predicate_rep, argument_rep, graph_mask)
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

                 sense_dim: int,
                 n_tags: int,
                 rep_dim: int,
                 score_dim: int,
                 dropout: float = 0.3,
                 use_predicate_rep = True
    :return:
    '''
    node_dim = 2
    edge_dim = 7
    hidden_dim = 15
    score_dim = 10

    batch_size = 2
    graph_size = 3
    mlpfbb = GraphAutoEncoder2(sense_dim=node_dim, n_tags=edge_dim-1,rep_dim= hidden_dim, score_dim=score_dim,dropout=0.3,use_predicate_rep=False)
   # mlpfbb.eval()
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
