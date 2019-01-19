from overrides import overrides
import torch,gc
from torch.nn import Dropout, Linear
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from allennlp.nn.util import masked_softmax, weighted_sum,masked_log_softmax,masked_normalize
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

from allennlp.nn import InitializerApplicator, RegularizerApplicator, Activation
from allennlp.modules import FeedForward
from myallennlp.modules.reparametrization.gumbel_softmax import masked_gumbel_softmax

from myallennlp.auto_grad.my_auto_grad import *
import copy


class GraphAutoEncoder(Seq2SeqEncoder):
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
                 detach:bool=True,
                 dropout: float = 0.3) -> None:
        super(GraphAutoEncoder, self).__init__()
        self.dropout = dropout
        self._dropout = Dropout(dropout)
        self._node_dim  = node_dim
        self._edge_dim  = edge_dim
        self._hidden_dim = hidden_dim
        self._edge_embed = Linear(edge_dim, node_dim)

        self.h = nn.functional.softplus

        self.sigmoid = nn.functional.sigmoid

        self._edge_mix = Linear(node_dim,
                                hidden_dim)

        self._message_combine = Linear(hidden_dim, score_dim)

        self._detach = detach
        self._edge_embed_to_score_m = Linear(node_dim, score_dim)


        self._pred_mess = Linear(hidden_dim, hidden_dim)
        self._arg_mess = Linear(hidden_dim, hidden_dim)

        self._dropout_mask = lambda x : torch.bernoulli(x.data.new(x.data.size()).fill_(1 - self.dropout)) if self.training else 1

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return True

    def get_computation_graph(self,
                nodes: torch.Tensor,
                extra_nodes: torch.Tensor,
                edges:torch.Tensor,
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
            def inner_wrap(f,x):
                linear = f(x)
                out = self.h(linear)
                out_mask = self._dropout_mask(out)
                return [out*out_mask,out_mask,linear]

            return lambda x: inner_wrap(f,x)



        graph = ComputationGraph("linear_score",["nodes","edges"],["extra_nodes"])


        def tmp( grad, edge_rep,edges):
            return grad.matmul(self._edge_embed.weight)
        def tmp_f( edges):
            return self._edge_embed(edges)
        # edge_rep (batch_size, timesteps, pre_len,node_dim)
        graph.add_node(ComputationNode("edge_rep",["edges"],tmp_f,tmp))

        def edge_node_back(grad, edge_node_rep,edge_rep,nodes,extra_nodes):
            masked_grad = grad*graph_mask
            return  [masked_grad ,masked_grad.sum(1) ]

        # edge_node_rep (batch_size, timesteps, pre_len,node_dim)
        graph.add_node(ComputationNode("edge_node_rep",["edge_rep","nodes"],
                                       lambda edge_rep,nodes,extra_nodes: (edge_rep + nodes.unsqueeze(1) +  extra_nodes.unsqueeze(2))*graph_mask,
                                       edge_node_back,extra_input_names=["extra_nodes"]))


        # A tensor of shape (batch_size, timesteps, pre_len ,hidden_dim)
        graph.add_node(ComputationNode("edge_mix",["edge_node_rep"],
                                       wrap_with_dropout(self._edge_mix),
                                       lambda grad,edge_mix, out_mask,linear,edge_node_rep: [(grad*out_mask* self.sigmoid(linear) ).matmul(self._edge_mix.weight) ]))


        # A tensor of shape (batch_size , timesteps , 1,hidden_dim)
        graph.add_node(ComputationNode("arg_message",["edge_mix"],
                                       wrap_with_dropout(lambda x: self._arg_mess(x.sum(2,keepdim=True))),
                                       lambda grad,arg_message, out_mask, linear,edge_mix: [((grad*out_mask*self.sigmoid(linear)).matmul(self._arg_mess.weight)).expand_as(edge_mix) ]))


        # A tensor of shape (batch_size, 1, pre_len,hidden_dim)
        graph.add_node(ComputationNode("pred_message",["edge_mix"],
                                       wrap_with_dropout(lambda x: self._pred_mess(x.sum(1,keepdim=True))),
                                       lambda grad,_pred_mess, out_mask, linear,edge_mix: [((grad*out_mask*self.sigmoid(linear)).matmul(self._pred_mess.weight)).expand_as(edge_mix) ]))



        # A tensor of shape (batch_size,  timesteps, pre_len, hidden_dim)
        graph.add_node(ComputationNode("linear_combined_message",["arg_message","pred_message","edge_mix"],
                                       lambda arg_message,pred_message,edge_mix: arg_message+pred_message + edge_mix,
                                       lambda grad,linear_combined_message, arg_message,pred_message,edge_mix: [grad.sum(2,keepdim=True),grad.sum(1,keepdim=True), grad ]))



        # A tensor of shape (batch_size, timesteps, timesteps,score_dim)
        graph.add_node(ComputationNode("combined_message",["linear_combined_message"],
                                       wrap_with_dropout(self._message_combine),
                                       lambda grad, combined_message,out_mask,linear,linear_combined_message: (grad*out_mask*self.sigmoid(linear)).matmul(self._message_combine.weight)))


        # A tensor of shape (batch_size, timesteps, timesteps,score_dim)
        graph.add_node(ComputationNode("linear_score",["combined_message","edge_rep"],
                                       lambda combined_message,edge_rep: combined_message + self._edge_embed_to_score_m(edge_rep),
                                       lambda grad,linear_score, combined_message,edge_rep: [grad,grad.matmul(self._edge_embed_to_score_m.weight) ]))

        graph.forward([nodes,edges],[extra_nodes])

        return graph


    def score(self,graph:ComputationGraph,graph_mask):

        #shape  (batch_size, timesteps, timesteps , score_dim)
        linear_score = graph.get_tensor_by_name("linear_score")
        return self.h(linear_score)*graph_mask

    def score_gradient(self,graph:ComputationGraph,graph_mask:torch.Tensor):


        linear_score = graph.get_tensor_by_name("linear_score")

        return self._dropout(self.sigmoid(linear_score))*graph_mask

    def get_score_and_graph(self,
                nodes: torch.Tensor,
                extra_nodes: torch.Tensor,
                edges:torch.Tensor,
                graph_mask: torch.Tensor = None,
                graph:ComputationGraph = None):
        '''

        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, timesteps, input_dim)
        '''
        if graph is None :
            graph = self.get_computation_graph(nodes,extra_nodes,edges,graph_mask)

        return  self.score(graph,graph_mask),graph
    @overrides
    def forward(self,
                nodes: torch.Tensor,
                extra_nodes: torch.Tensor,
                edges:torch.Tensor,
                graph_mask: torch.Tensor = None,
                graph: ComputationGraph = None,
                get_grads:bool=True) -> torch.FloatTensor:
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, length,  length, input_dim)
        graph : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, timesteps,1)
        extra_inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size,  length, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).

        Returns
        -------
        gradient w.r.t to score m1 Relu m2 input
        """
        if graph is None :
            graph = self.get_computation_graph(nodes,extra_nodes,edges,graph_mask)

        score,graph = self.get_score_and_graph(nodes,extra_nodes,edges,graph_mask,graph)
        grad_to_score = self.score_gradient(graph,graph_mask)

        grad_to_nodes,grad_to_edges = graph.backward(grad_to_score)
        del graph
        return score,  grad_to_nodes,grad_to_edges


def main():
    '''
    node_dim: int,
                 edge_dim: int,
                 hidden_dim: int,
                 score_dim: int,

    :return:
    '''
    node_dim = 2
    edge_dim = 3
    hidden_dim = 15
    score_dim = 10

    batch_size = 2
    graph_size = 3
    mlpfbb = GraphAutoEncoder(node_dim,edge_dim,hidden_dim,score_dim)

    nodes = torch.rand(batch_size,graph_size,node_dim,requires_grad=True)
    extra_nodes = torch.rand(batch_size,graph_size*2,node_dim)
    edges = torch.rand(batch_size,graph_size*2,graph_size,edge_dim,requires_grad=True)

    graph_mask =  torch.rand(size=edges.size()[:-1]).unsqueeze(-1)

    graph = None

    score, grad_to_nodes, grad_to_edges = mlpfbb(nodes,extra_nodes,edges,graph_mask)



    l = score.sum()
    l.backward()

    print ("gradient_to_edge",grad_to_edges)
    print ("edges.grad",edges.grad)

    print("diff",(grad_to_edges-edges.grad).pow(2).sum())



    print ("grad_to_nodes",grad_to_nodes)
    print ("nodes.grad",nodes.grad)

    print("diff",(grad_to_nodes-nodes.grad).pow(2).sum())

    print("grad_to_edges[0]",grad_to_edges[1][:,:,2])




if __name__ == "__main__":
    main()
