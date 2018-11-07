from typing import List

from overrides import overrides
import torch
from torch.nn import Dropout

from allennlp.modules.feedforward import FeedForward
from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules.seq2seq_encoders.multi_head_self_attention import MultiHeadSelfAttention
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn.activations import Activation
from allennlp.nn.util import add_positional_features
from myallennlp.modules.my_multi_head_self_attention_refiner import MultiHeadAttention


@Seq2SeqEncoder.register("my_stacked_self_attention_refiner")
class MyStackedSelfAttentionRefinment(Seq2SeqEncoder):
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
                 head_dim: int,
                 child_dim: int,
                 projection_dim:int,
                 feedforward_hidden_dim: int,
                 num_iterations: int,
                 num_attention_heads: int = 1,
                 use_positional_encoding: bool = False,
                 dropout_prob: float = 0.1,
                 residual_dropout_prob: float = 0.2,
                 attention_dropout_prob: float = 0.1) -> None:
        super(MyStackedSelfAttentionRefinment, self).__init__()
        self.head_dim = head_dim
        self.child_dim = child_dim
        input_dim = head_dim + child_dim
        self._use_positional_encoding = use_positional_encoding

        feedfoward_input_dim = input_dim
        self.num_iterations = num_iterations
        self.feedforward = FeedForward(feedfoward_input_dim,
                                     activations=[Activation.by_name('elu')(),
                                                  Activation.by_name('linear')()],
                                     hidden_dims=[feedforward_hidden_dim, head_dim + child_dim],
                                     num_layers=2,
                                     dropout=dropout_prob)


        self.feedforward_layer_norm = LayerNorm(self.feedforward.get_output_dim())
        self.attention = MultiHeadAttention(n_head = num_attention_heads, head_dim=head_dim, child_dim=child_dim, hidden_dim=projection_dim,
                                            dropout=attention_dropout_prob)


        self.dropout = Dropout(residual_dropout_prob)
        self._input_dim = input_dim
        self._output_dim = self.attention.get_output_dim()

    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return False

    @overrides
    def forward(self, heads: torch.Tensor,
                children:torch.Tensor,
                attended_arcs:torch.Tensor,
                mask: torch.Tensor): # pylint: disable=arguments-differ
        '''


        :param    heads : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, head_dim):
        :param children:
        :param attended_arcs:
        :param mask:
        :return:
        '''
        heads_list = []
        child_list = []
        arcs_list = []
        heads_list.append(heads)
        child_list.append(children)
        arcs_list.append(attended_arcs)

        inputs = torch.cat([heads,children],dim=2)
        if self._use_positional_encoding:
            output = add_positional_features(inputs)
        else:
            output = inputs
        for i in range(self.num_iterations):
            cached_input = output
            # Project output of attention encoder through a feedforward
            # network and back to the input size for the next layer.
            # shape (batch_size, timesteps, input_size)
            feedforward_output = self.feedforward(output)
            feedforward_output = self.dropout(feedforward_output)
            if feedforward_output.size() == cached_input.size():
                # First layer might have the wrong size for highway
                # layers, so we exclude it here.
                feedforward_output = self.feedforward_layer_norm(feedforward_output + cached_input)
            # shape (batch_size, sequence_length, hidden_dim)
            attention_output,attended_arcs = self.attention(heads,children,feedforward_output,attended_arcs, mask)

            output = attention_output
      #      heads = output[:,:,:self.head_dim]
      #      children = output[:,:,self.head_dim:]


            heads_list.append(output[:,:,:self.head_dim])
            child_list.append(output[:,:,self.head_dim:])
            arcs_list.append(attended_arcs)
        return heads_list, child_list, arcs_list
