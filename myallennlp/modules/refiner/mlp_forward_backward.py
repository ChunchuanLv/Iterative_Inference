from overrides import overrides
import torch
from torch.nn import Dropout, Linear
import torch.nn as nn
import torch.nn.functional as F

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
                 dropout: float = 0.3) -> None:
        super(MLPForwardBackward, self).__init__()

        self.dropout = dropout
        self._input_dim  = input_dim+extra_input_dim
        self._output_dim = input_dim
        self.m1 = Linear(input_dim+extra_input_dim, hidden_dim)
        self.m2 = Linear( hidden_dim,1,bias=False)
        self.elu = torch.nn.functional.elu
    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return True


    def get_score(self,
                inputs: torch.Tensor,
                extra_inputs: torch.Tensor,
                mask: torch.LongTensor = None):
        '''

        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        '''
        inputs = torch.cat([inputs,extra_inputs],dim=-1)

        h1 = self.m1(inputs)
        if self.training:
            dropout_mask = torch.bernoulli(h1.data.new(h1.data.size()).fill_(1-self.dropout))
        else:
            dropout_mask = 1
        h1 = h1*dropout_mask
        if mask:
            float_mask = mask.float()
            return  self.m2 ( self.elu(h1))*float_mask.unsqueeze(-1),dropout_mask

        return  self.m2 ( self.elu (h1)),dropout_mask
    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                extra_inputs: torch.Tensor,
                mask: torch.LongTensor = None,
                dropout_mask:torch.Tensor=None,) -> torch.FloatTensor:
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).

        Returns
        -------
        gradient w.r.t to score m1 Relu m2 input
        """
        combined_inputs = torch.cat([inputs,extra_inputs],dim=-1)
        h1 = self.m1(combined_inputs)
        if dropout_mask is None:
            if self.training:
                dropout_mask = torch.bernoulli(h1.data.new(h1.data.size()).fill_(1 - self.dropout))
            else:
                dropout_mask = 1
        h1 = h1*dropout_mask
        sign = (torch.sign(h1)+1)/2

        #shape  (batch_size, timesteps, hidden_dim)
        gate = (sign + (1-sign) * torch.exp(h1))*dropout_mask

        #shape  (batch_size, timesteps, hidden_dim)
        gradient_to_h = gate * self.m2.weight.unsqueeze(0)

        #shape  (batch_size, timesteps, input_dim)
        gradient_to_input = gradient_to_h.matmul(self.m1.weight[:,:inputs.size(-1)])
        return gradient_to_input


def main():
    mlpfbb = MLPForwardBackward(2,3,15,0.1)

    input = torch.rand(2,2)

    input.requires_grad = True
    extra = torch.rand(2,3)

    score ,dropout_mask= mlpfbb.get_score(input,extra)

    gradient_to_input = mlpfbb(input,extra,dropout_mask=dropout_mask)

    l = score.sum()
    l.backward()

    print ("gradient_to_input",gradient_to_input)
    print ("input.grad",input.grad)

    print("diff",(gradient_to_input-input.grad).pow(2).sum())

if __name__ == "__main__":
    main()
