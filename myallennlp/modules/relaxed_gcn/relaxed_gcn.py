from overrides import overrides
import torch
from torch.nn import Dropout, Linear
import torch.nn as nn
import torch.nn.functional as F

from allennlp.modules.layer_norm import LayerNorm
from allennlp.nn.util import masked_softmax, weighted_sum,masked_log_softmax,masked_normalize
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

from myallennlp.modules.reparametrization.gumbel_softmax import masked_gumbel_softmax

@Seq2SeqEncoder.register("relaxed_gcn")
class RelaxedGCN(Seq2SeqEncoder):
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
                 num_heads: int,
                 input_dim: int,
                 values_dim: int,
                 attention_dropout_prob:float,
                 residual:bool=False,
                 residual_dropout_prob:float=0) -> None:
        super(RelaxedGCN, self).__init__()

        self._num_heads = num_heads
        self._input_dim = input_dim
        self._output_dim =  input_dim
        self._values_dim = values_dim
        self._residual = residual
        self._attention_dropout = Dropout(attention_dropout_prob)
        if residual:

            self.layer_norm = LayerNorm(input_dim)
            self.dropout = Dropout(residual_dropout_prob)

        self._combined_projection = Linear(input_dim, values_dim)
        self._output_projection = Linear(values_dim, self._output_dim) if values_dim != self._output_dim else lambda x: x

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim


    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                multi_headed_attention: torch.Tensor,
                mask: torch.LongTensor = None) -> torch.FloatTensor:
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        multi_headed_attention : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, num_heads, timesteps, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).

        Returns
        -------
        A tensor of shape (batch_size, timesteps, output_projection_dim),
        where output_projection_dim = input_dim by default.
        """
        num_heads = self._num_heads
        batch_size, timesteps, _ = inputs.size()
        multi_headed_attention = self._attention_dropout(multi_headed_attention).view(batch_size * num_heads, timesteps, timesteps)
        if mask is None:
            mask = inputs.new_ones(batch_size, timesteps)


        assert torch.isnan(inputs).sum() == 0, ("inputs",inputs)


        # Shape (batch_size, timesteps, 2 * attention_dim + values_dim)
        values = self._combined_projection(inputs)

        # Shape (num_heads * batch_size, timesteps, values_dim / num_heads)
        values_per_head = values.view(batch_size, timesteps, num_heads, int(self._values_dim/num_heads))
        values_per_head = values_per_head.transpose(1, 2).contiguous()
        values_per_head = values_per_head.view(batch_size * num_heads, timesteps, int(self._values_dim/num_heads))

        outputs = weighted_sum(values_per_head, multi_headed_attention)

        # Reshape back to original shape (batch_size, timesteps, values_dim)
        # shape (batch_size, num_heads, timesteps, values_dim/num_heads)
        outputs = outputs.view(batch_size, num_heads, timesteps, int(self._values_dim / num_heads))
        # shape (batch_size, timesteps, num_heads, values_dim/num_heads)
        outputs = outputs.transpose(1, 2).contiguous()
        # shape (batch_size, timesteps, values_dim)
        outputs = outputs.view(batch_size, timesteps, self._values_dim)

        # Project back to original input size.
        # shape (batch_size, timesteps, input_size)
        outputs = self._output_projection(outputs)
        if self._residual:
            outputs = self.layer_norm( self.dropout(outputs )+ inputs)
        return outputs
