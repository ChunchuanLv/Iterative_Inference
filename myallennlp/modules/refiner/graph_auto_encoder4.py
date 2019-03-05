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


class GraphAutoEncoder4(Seq2SeqEncoder):
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
                 input_node_dim:int,
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int,
                 score_dim: int,
                 full_back:bool=False,
                 mask_null:bool=False,
                 dropout: float = 0.3) -> None:
        super(GraphAutoEncoder4, self).__init__()
        self.dropout = dropout
        self._dropout = Dropout(dropout)
        self._input_node_dim  = input_node_dim
        self._node_dim  = node_dim
        self._edge_dim  = edge_dim
        self._score_dim  = score_dim
        self.full_back = full_back
        self._hidden_dim = hidden_dim
        self._edge_embed = Linear(edge_dim, node_dim)

        self._negative_edge_embed = Linear(edge_dim - 1, node_dim)

        self._node_embed = Linear(input_node_dim, node_dim)
        self.h = nn.functional.softplus

        self.sigmoid = nn.functional.sigmoid

      #  self._edge_mix_hidden  = Linear(node_dim, hidden_dim)
        self._edge_mix= Linear(node_dim,
                                score_dim)

     #   self._message_combine = Linear(hidden_dim, score_dim)

        self.mask_null = mask_null
        self._edge_embed_to_score_m = Linear(node_dim, score_dim)


        self._predicate_to_score_m = Linear(input_node_dim, score_dim)


    #    self._pred_mess = Linear(hidden_dim, hidden_dim)
   #     self._arg_mess = Linear(hidden_dim, hidden_dim)

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
        if self.mask_null:
            graph_mask = graph_mask * (1 - edges[:,:,:,0]).unsqueeze(-1).detach()
        graph = ComputationGraph("linear_score",["input_nodes","edges"],["extra_nodes"])

        # A tensor of shape (batch_size, timesteps, pre_len ,hidden_dim)
        graph.add_node(ComputationNode("nodes",["input_nodes"],
                                       self._node_embed,
                                       lambda grad,nodes, input_nodes: [grad.matmul(self._node_embed.weight) ]))




        def tmp( grad, edge_rep,edges):

            grad = grad.matmul(self._edge_embed.weight)
         #   grad = torch.cat([torch.zeros_like(grad[:,:,:,0]).unsqueeze(-1),grad[:,:,:,1:]],dim=-1)
            return grad

        def tmp_f( edges):
      #      edges = torch.cat([torch.zeros_like(edges[:,:,:,0]).unsqueeze(-1),edges[:,:,:,1:]],dim=-1)
            return self._edge_embed(edges)


        def negative(edges):

            all_edges = (edges* graph_mask).sum(1,keepdim=True)

            negative_edges = (all_edges - edges )* graph_mask
            negative_edges = negative_edges[:,:,:,1:]
            if not self.full_back:
                negative_edges = negative_edges.detach()
            out , out_mask, linear =  wrap_with_dropout(self._negative_edge_embed) (negative_edges)

         #   print ("out",out.size())
         #   print ("graph_mask",graph_mask.size())
            return [out,out_mask,linear]


        def negative_back(grad,negative_rep,out_mask,linear,edges):
            if not self.full_back:
                return [torch.zeros_like(edges)]
            else:
                grad = (grad*out_mask* self.sigmoid(linear) ).matmul(self._negative_edge_embed.weight)
                grad = torch.cat([torch.zeros_like(grad[:, :, :, 0]).unsqueeze(-1), grad], dim=-1)
                grad = grad * graph_mask
                grad = (grad.sum(1,keepdim=True) - grad )* graph_mask#-grad
                return [grad]


        # edge_rep (batch_size, timesteps, pre_len,node_dim)
        graph.add_node(ComputationNode("edge_rep",["edges"],tmp_f,tmp))

     #   # edge_rep (batch_size, timesteps, pre_len,node_dim)
        graph.add_node(ComputationNode("negative_edge",["edges"],negative,negative_back))

        def edge_node_back(grad, edge_node_rep,negative_edge,nodes,extra_nodes):
            masked_grad = grad*graph_mask
            return  [ masked_grad, masked_grad.sum(1) ]


        # edge_node_rep (batch_size, timesteps, pre_len,node_dim)
        graph.add_node(ComputationNode("edge_node_rep",["negative_edge","nodes"],
                                       lambda negative_edge,nodes,extra_nodes: ( negative_edge + nodes.unsqueeze(1) +  extra_nodes.unsqueeze(2))*graph_mask,
                                       edge_node_back,extra_input_names=["extra_nodes"]))


        # A tensor of shape (batch_size, timesteps, pre_len ,hidden_dim)
        graph.add_node(ComputationNode("edge_mix",["edge_node_rep"],
                                       wrap_with_dropout(self._edge_mix),
                                       lambda grad,edge_mix_hidden, out_mask,linear,edge_node_rep: [(grad *graph_mask *out_mask* self.sigmoid(linear) ).matmul(self._edge_mix.weight) ]))











        def linear_score_back(grad, linear_score, edge_mix, edge_rep, input_nodes):
            masked_grad = grad * graph_mask

            grad_to_edge_rep = masked_grad.matmul(self._edge_embed_to_score_m.weight)

            grad_to_input_nodes = masked_grad.sum(1).matmul(self._predicate_to_score_m.weight)

            return [masked_grad, grad_to_edge_rep, grad_to_input_nodes]

        # A tensor of shape (batch_size, timesteps, timesteps,score_dim )
        graph.add_node(ComputationNode("linear_score",["edge_mix","edge_rep","input_nodes"],
                                       lambda edge_mix,edge_rep,input_nodes: edge_mix + self._edge_embed_to_score_m(edge_rep)+self._predicate_to_score_m(input_nodes).unsqueeze(1),
                                       linear_score_back))

    #    print ("graph order",graph.get_backward_order())





        graph.forward([nodes,edges],[extra_nodes])
        return graph


    def score(self,graph:ComputationGraph,graph_mask):

        #shape  (batch_size, timesteps, timesteps , score_dim)
        linear_score = graph.get_tensor_by_name("linear_score")
        return self.h(linear_score)*graph_mask

    def score_gradient(self,graph:ComputationGraph,graph_mask:torch.Tensor):


        linear_score = graph.get_tensor_by_name("linear_score")

        return self.sigmoid(linear_score)*graph_mask

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
                predicates_emb: torch.Tensor,
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
            graph = self.get_computation_graph(predicates_emb,extra_nodes,edges,graph_mask)

        score,graph = self.get_score_and_graph(nodes,extra_nodes,edges,graph_mask,graph)
        grad_to_score = self.score_gradient(graph,graph_mask)

        grad_to_nodes,grad_to_edges = graph.backward(grad_to_score)
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
    input_node_dim = 2
    edge_dim = 3
    hidden_dim = 15
    score_dim = 10

    batch_size = 2
    graph_size = 3
    mlpfbb = GraphAutoEncoder(input_node_dim,node_dim,edge_dim,hidden_dim,score_dim)
 #   mlpfbb.eval()
    nodes = torch.rand(batch_size,graph_size,node_dim,requires_grad=True)
    extra_nodes = torch.rand(batch_size,graph_size*2,node_dim)
    edges = torch.rand(batch_size,graph_size*2,graph_size,edge_dim,requires_grad=True)


    sizes = [[6,3],[4,2]]
    graph_mask =  torch.zeros(size=edges.size()[:-1])

    for i , [s0,s1] in enumerate(sizes):
        data_t = torch.ones(s0,s1)
        graph_mask[i].narrow(0, 0, s0).narrow(1, 0, s1).copy_(data_t)
    graph_mask = graph_mask.unsqueeze(-1)
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
