from overrides import overrides
import torch
from torch.nn import Dropout, Linear
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from allennlp.nn.util import masked_softmax, weighted_sum,masked_log_softmax,masked_normalize
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

from myallennlp.modules.reparametrization.gumbel_softmax import masked_gumbel_softmax

class MLPForwardBackward(Seq2SeqEncoder):
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
                 max:int = 10,
                 dropout: float = 0.3) -> None:
        super(MLPForwardBackward, self).__init__()
        self.sum_dim = sum_dim
        self.dropout = dropout
        self._input_dim  = input_dim+extra_input_dim
        self._output_dim = input_dim
        self.m1 = Linear(input_dim+extra_input_dim, hidden_dim)
        self.m2 = Linear( hidden_dim,1,bias=False)
        self.elu = torch.nn.functional.elu
        self.max = max
    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return True

    def get_intermediates(self,
                inputs: torch.Tensor,
                extra_inputs: torch.Tensor):
        intermediates = {}

        inputs = inputs.sum(dim=self.sum_dim)
        combined_inputs = torch.cat([inputs,extra_inputs],dim=-1)
        intermediates["combined_inputs"] = combined_inputs

        h1 = intermediates.setdefault("h1",self.m1(combined_inputs))
        if self.training:
            dropout_mask = intermediates.setdefault("dropout_mask", torch.bernoulli(h1.data.new(h1.data.size()).fill_(1-self.dropout)))
        else:
            dropout_mask = 1
            intermediates["dropout_mask"] = 1
        h1_dropped = intermediates.setdefault("h1_dropped",h1*dropout_mask)

        h1_non_linear = intermediates.setdefault("h1_non_linear",self.elu (h1_dropped))

        intermediates["h2"] = self.m2 (h1_non_linear )
        return intermediates

    def get_score(self,
                inputs: torch.Tensor,
                extra_inputs: torch.Tensor,
                intermediates:Dict = {}):
        '''

        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, timesteps, input_dim)
        '''
        if intermediates is None or len(intermediates)==0:
            intermediates = self.get_intermediates(inputs,extra_inputs)

        h2 = intermediates["h2"]
        score = h2.clamp(max=self.max)
        return  score,intermediates
    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                extra_inputs: torch.Tensor,
                intermediates:Dict={}) -> torch.FloatTensor:
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, length,  length, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).

        Returns
        -------
        gradient w.r.t to score m1 Relu m2 input
        """
        if intermediates is None or len(intermediates)==0:
            intermediates = self.get_intermediates(inputs,extra_inputs)

        # shape  (batch_size, timesteps, 1)
        gradient_to_h2 = (intermediates["h2"] <self.max).float()
    #    gradient_to_h2 = gradient_to_h2.unsqueeze(1)

        # shape  (batch_size, timesteps, hidden_dim) weight is (1,hidden_dim) vector
        gradient_to_h = gradient_to_h2   * self.m2.weight.unsqueeze(0)

        h1_dropped = intermediates["h1_dropped"]
        dropout_mask = intermediates["dropout_mask"]
        sign = (torch.sign(h1_dropped)+1)/2

        #shape  (batch_size, timesteps, hidden_dim)
        gate = (sign + (1-sign) * torch.exp(h1_dropped))*dropout_mask

        #shape  (batch_size, timesteps, hidden_dim)
        gradient_to_h = gate * gradient_to_h

        #shape  (batch_size, timesteps, input_dim)
        gradient_to_input = (gradient_to_h.matmul(self.m1.weight[:,:inputs.size(-1)]) ).unsqueeze(self.sum_dim).expand_as(inputs)
        return gradient_to_input,intermediates


def main():
    mlpfbb = MLPForwardBackward(2,3,15,1,10,0.1)

    input = torch.rand(1,3,3,2)

    input.requires_grad = True
    extra = torch.rand(1,3,3)
    intermediates = None
    gradient_to_input,intermediates = mlpfbb(input,extra,intermediates=intermediates)
    score ,intermediates= mlpfbb.get_score(input,extra,intermediates=intermediates)



    l = score.sum()
    l.backward()

    print ("gradient_to_input",gradient_to_input)
    print ("input.grad",input.grad)

    print("diff",(gradient_to_input-input.grad).pow(2).sum())

if __name__ == "__main__":
    main()
