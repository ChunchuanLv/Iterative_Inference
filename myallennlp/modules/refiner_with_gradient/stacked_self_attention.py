from overrides import overrides
import torch
from torch.nn import Dropout
import torch.nn.functional as F

from allennlp.modules.feedforward import FeedForward
from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn.activations import Activation

from allennlp.nn import InitializerApplicator, RegularizerApplicator

from allennlp.nn.util import masked_softmax, weighted_sum
from allennlp.nn.util import add_positional_features
from myallennlp.modules.refiner.multi_head_self_attention import SelfAttentionRefinment


@Seq2SeqEncoder.register("gradient_refiner")
class GradientRefinment(Seq2SeqEncoder):
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
                 input_dim: int,
                 hidden_dim: int,
                 projection_dim: int,
                 feedforward_hidden_dim: int,
                 num_attention_heads: int,
                 use_positional_encoding: bool = True,
                 gating:bool = True,
                 dropout_prob: float = 0.1,
                 residual_dropout_prob: float = 0.2,
                 attention_dropout_prob: float = 0.1) -> None:
        super(GradientRefinment, self).__init__()
        self._use_positional_encoding = use_positional_encoding

        self.feedforward = FeedForward(input_dim,
                                     activations=[Activation.by_name('elu')(),
                                                  Activation.by_name('linear')()],
                                     hidden_dims=[feedforward_hidden_dim, hidden_dim],
                                     num_layers=2,
                                     dropout=dropout_prob)

        self.feedforward_layer_norm = LayerNorm(self.feedforward.get_output_dim())
        self.attention = SelfAttentionRefinment(num_heads=num_attention_heads,
                                                    input_dim=hidden_dim,
                                                    attention_dim=projection_dim,
                                                    values_dim=projection_dim,
                                                    output_projection_dim = input_dim,
                                                         gating=gating)

        self.layer_norm = LayerNorm(self.attention.get_output_dim())

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

        def _iterative_decode(self,
                              attended_arcs: torch.Tensor,
                              high_order_features: torch.Tensor,
                              mask: torch.Tensor,
                              iterations: int = 1,
                              head_indices: torch.Tensor = None,
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
            '''

            attended_arcs : ``torch.Tensor``, required.
                A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
                a distribution over attachements of a given word to all other words.
            head_indices : ``torch.Tensor``, required.
                A tensor of shape (batch_size, sequence_length).
                The indices of the heads for every word.
            head_tags : ``torch.Tensor``, required.
                A tensor of shape (batch_size, sequence_length).
                The dependency labels of the heads for every word.
            mask : ``torch.Tensor``, required.
                A mask of shape (batch_size, sequence_length), denoting unpadded
                elements in the sequence.

            return refined_attended_arcs actually relaxed_head.log()
    '''

            def mask_attended_arcs(attended_arcs):
                attended_arcs = attended_arcs + torch.diag(attended_arcs.new(mask.size(1)).fill_(-numpy.inf))
                attended_arcs.masked_fill_(minus_mask, -numpy.inf)
                return attended_arcs

            def get_high_order_delta_y_gradient():

                if head_indices is not None:
                    sibling = torch.matmul(relaxed_head, relaxed_head.transpose(1, 2))

                    grand_pa = torch.matmul(relaxed_head, relaxed_head)

                    sgn_sibling = torch.sign(sibling - gold_sibling) * float_mask.unsqueeze(1) * float_mask.unsqueeze(2)
                    sgn_grand_pa = torch.sign(grand_pa - gold_grand_pa) * float_mask.unsqueeze(
                        1) * float_mask.unsqueeze(2)

                    gradient_to_head = (torch.matmul(sgn_sibling, relaxed_head)
                                        + torch.matmul(sgn_sibling.transpose(1, 2), relaxed_head)
                                        + torch.matmul(sgn_grand_pa, relaxed_head.transpose(1, 2))
                                        + torch.matmul(relaxed_head.transpose(1, 2), sgn_grand_pa))

                    approximate_gradient_to_attended_arcs = gradient_to_head * relaxed_head
                    gradient_to_attended_arcs = approximate_gradient_to_attended_arcs - approximate_gradient_to_attended_arcs.sum(
                        2, keepdim=True) * relaxed_head * float_mask.unsqueeze(1) * float_mask.unsqueeze(2)
                    return gradient_to_attended_arcs
                else:
                    return 0

            def get_delta_y_gradient():

                if head_indices is not None:
                    return (relaxed_head - soft_head) * float_mask.unsqueeze(1) * float_mask.unsqueeze(2)
                else:
                    return 0

            float_mask = mask.float()
            minus_mask = (1 - mask).byte().unsqueeze(2)
            batch_size, sequence_length = float_mask.size()
            if head_indices is not None:
                soft_head = torch.zeros(batch_size, sequence_length, sequence_length, device=head_indices.device)
                soft_head.scatter_(2, head_indices.unsqueeze(2), 1)
                soft_head = soft_head * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)
                gold_sibling = torch.matmul(soft_head, soft_head.transpose(1, 2))
                gold_grand_pa = torch.matmul(soft_head, soft_head)
            input_attended_arcs = attended_arcs
            # Mask padded tokens, because we only want to consider actual words as heads.
            if self.training:
                relaxed_head = masked_gumbel_softmax(mask_attended_arcs(attended_arcs), tau=self.gumbel_t,
                                                     mask=mask.unsqueeze(-1))
            else:
                relaxed_head = masked_softmax(mask_attended_arcs(attended_arcs), mask=mask.unsqueeze(-1))
            relaxed_head = relaxed_head * float_mask.unsqueeze(1) * float_mask.unsqueeze(2)
            relaxed_head.masked_fill_(minus_mask, 0)

            if self.high_order_weight:

                sibling_weights = high_order_features[:, 0]
                grand_pa_weights = high_order_features[:, 1]
                lr = self.iterative_lr
                for i in range(iterations):

                    attended_arcs = attended_arcs \
                                    + lr * (input_attended_arcs + get_delta_y_gradient() + self.high_order_weight * (
                                torch.matmul(grand_pa_weights, relaxed_head.transpose(1, 2)) \
                                + torch.matmul(sibling_weights, relaxed_head)) + get_high_order_delta_y_gradient())

                    attended_arcs = masked_log_softmax(attended_arcs,
                                                       mask) * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)
                    if self.gumbel_testing_inplace:
                        noise, relaxed_head = inplace_masked_gumbel_softmax(mask_attended_arcs(attended_arcs),
                                                                            tau=self.gumbel_t, mask=mask.unsqueeze(-1))
                        attended_arcs = attended_arcs + lr * noise * float_mask.unsqueeze(1) * float_mask.unsqueeze(2)
                    elif self.training or self.gumbel_testing:
                        relaxed_head = masked_gumbel_softmax(mask_attended_arcs(attended_arcs), tau=self.gumbel_t,
                                                             mask=mask.unsqueeze(-1))
                    else:
                        relaxed_head = masked_softmax(mask_attended_arcs(attended_arcs), mask=mask.unsqueeze(-1))
                    relaxed_head = relaxed_head * float_mask.unsqueeze(1) * float_mask.unsqueeze(2)
                    relaxed_head.masked_fill_(minus_mask, 0)
                    lr = lr * self.cool_down

            return attended_arcs, relaxed_head
    @overrides
    def forward(self, input: torch.Tensor,
                attended_arcs:torch.Tensor,
                mask: torch.Tensor,
                num_iterations:int): # pylint: disable=arguments-differ
        '''


        :param    heads : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, head_dim):
        :param children:
        :param attended_arcs:
        :param mask:
        :return:
        '''
        assert torch.isnan(input).sum() == 0, ("refiner input",input)
        data_list = []
        attended_arcs_list = []

        data_list.append(input)
        attended_arcs_list.append(attended_arcs)

        if self._use_positional_encoding:
            output = add_positional_features(input)
        else:
            output = input
        for i in range(num_iterations):
            cached_input = output
            # Project output of attention encoder through a feedforward
            # network and back to the input size for the next layer.
            # shape (batch_size, timesteps, input_size)
            assert torch.isnan(output).sum() == 0, ("output",output)
            feedforward_output = self.feedforward(output)
            assert torch.isnan(feedforward_output).sum() == 0, ("feedforward_output",feedforward_output)
            feedforward_output = self.dropout(feedforward_output)
            if feedforward_output.size() == cached_input.size():
                # First layer might have the wrong size for highway
                # layers, so we exclude it here.
                feedforward_output = self.feedforward_layer_norm(feedforward_output + cached_input)
                assert torch.isnan(feedforward_output).sum() == 0, ("normed_feedforward_output",feedforward_output)
            # shape (batch_size, sequence_length, hidden_dim)
            attention_output,attended_arcs = self.attention(feedforward_output,attended_arcs, mask)
            assert torch.isnan(attention_output).sum() == 0, ("attention_output",attention_output)

            output =  self.layer_norm(self.dropout(attention_output) + feedforward_output)

            data_list.append(output)
            attended_arcs_list.append(  attended_arcs)
        return data_list, attended_arcs_list
