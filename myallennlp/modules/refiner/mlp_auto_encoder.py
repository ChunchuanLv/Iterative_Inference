from overrides import overrides
import torch
from torch.nn import Dropout, Linear
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from allennlp.nn.util import masked_softmax, weighted_sum,masked_log_softmax,masked_normalize
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

from myallennlp.modules.reparametrization.gumbel_softmax import masked_gumbel_softmax

class MLPAutoEncoder(Seq2SeqEncoder):
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
                 input_dim: int,
                 extra_input_dim: int,
                 hidden_dim: int,
                 sum_dim:int=1,
                 leak:float = 0.1,
                 dropout: float = 0.3) -> None:
        super(MLPAutoEncoder, self).__init__()
        self.sum_dim = sum_dim
        self.dropout = dropout
        self._input_dim  = input_dim+extra_input_dim
        self._output_dim = input_dim
        self.m1 = Linear(input_dim+extra_input_dim, hidden_dim)
        self.h =  nn.Softplus()
        self.leak = torch.nn.Parameter(torch.FloatTensor([leak/(1-leak)]).log(),requires_grad=False)
    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return True

    def get_intermediates(self,
                inputs: torch.Tensor,
                graph:torch.Tensor,
                argue_rep: torch.Tensor,
                predicate_rep: torch.Tensor,):
        '''

        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, timesteps, input_dim)
        argue_rep : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps , extra_dim)
        predicate_rep : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, extra_dim)
        graph : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, timesteps)
        '''
        intermediates = {}
        leak =  self.leak.sigmoid()
    #    print ("inputs",inputs.size())
    #    print ("graph",graph.size())
        inputs = inputs*(graph*(1-leak)+leak)
        inputs = inputs.sum(dim=self.sum_dim)
        combined_inputs = torch.cat([inputs,predicate_rep],dim=-1)
        intermediates["combined_inputs"] = combined_inputs

        h1 = intermediates.setdefault("h1",self.m1(combined_inputs))
        if self.training:
            dropout_mask = intermediates.setdefault("dropout_mask", torch.bernoulli(h1.data.new(h1.data.size()).fill_(1-self.dropout)))
        else:
            dropout_mask = 1
            intermediates["dropout_mask"] = 1

        intermediates.setdefault("h1_dropped",h1*dropout_mask)



        return intermediates


    def score(self,beofre_h,intermediates):

        #shape  (batch_size, timesteps, hidden_dim)
        score =  intermediates.setdefault("score",self.h(beofre_h) )

        return score,intermediates

    def score_gradient(self,beofre_h,intermediates={}):
        gradient =  intermediates.setdefault("gate",beofre_h.sigmoid())

        return gradient,intermediates

    def get_score(self,
                inputs: torch.Tensor,
                graph:torch.Tensor,
                argue_rep: torch.Tensor,
                predicate_rep: torch.Tensor,
                intermediates:Dict = {}):
        '''

        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, timesteps, input_dim)
        '''
        if intermediates is None or len(intermediates)==0:
            intermediates = self.get_intermediates(inputs,graph,argue_rep,predicate_rep)

        return  self.score(intermediates["h1_dropped"],intermediates)
    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                graph:torch.Tensor,
                argue_rep: torch.Tensor,
                predicate_rep: torch.Tensor,
                intermediates:Dict={}) -> torch.FloatTensor:
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
        if intermediates is None or len(intermediates)==0:
            intermediates = self.get_intermediates(inputs,graph,argue_rep,predicate_rep)

        leak = self.leak.sigmoid()
        # shape  (batch_size, timesteps, hidden_dim)
        gradient_to_h, intermediates = self.score_gradient(intermediates["h1_dropped"],intermediates)

        dropout_mask = intermediates["dropout_mask"]

        #shape  (batch_size, timesteps, hidden_dim)
        gradient_to_h = gradient_to_h *dropout_mask

        #shape  (batch_size, timesteps, 1, input_dim) or  (batch_size , 1 , timesteps, input_dim)
        gradient_to_summed_input = gradient_to_h.matmul(self.m1.weight[:,:inputs.size(-1)]).unsqueeze(self.sum_dim)

        gradient_to_input = gradient_to_summed_input*(graph*(1-leak)+leak)

        gradient_to_graph = (gradient_to_summed_input * inputs).sum(dim=-1,keepdim=True)*(1-leak)
        return gradient_to_input,gradient_to_graph,intermediates


def main():
    mlpfbb = MLPAutoEncoder(2,3,15)

    input = torch.rand(1,3,3,2)
    graph = torch.rand(1,3,3,1)
    input.requires_grad = True
    graph.requires_grad = True
    extra = torch.rand(1,3,3)
    extra2 = torch.rand(1,3,3)
    intermediates = None

    score ,intermediates= mlpfbb.get_score(input,graph,extra,extra2,intermediates=intermediates)
    gradient_to_input,gradient_to_graph,intermediates = mlpfbb(input,graph,extra,extra2,intermediates=intermediates)



    l = score.sum()
    l.backward()

    print ("gradient_to_input",gradient_to_input)
    print ("input.grad",input.grad)

    print("diff",(gradient_to_input-input.grad).pow(2).sum())



    print ("gradient_to_graph",gradient_to_graph)
    print ("graph.grad",graph.grad)

    print("diff",(gradient_to_graph-graph.grad).pow(2).sum())

if __name__ == "__main__":
    main()
