from overrides import overrides
import torch
from torch.nn import Dropout
import torch.nn.functional as F

import copy
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules import InputVariationalDropout
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn.activations import Activation
from allennlp.nn import InitializerApplicator, RegularizerApplicator

from allennlp.nn.util import get_device_of, masked_log_softmax, get_lengths_from_binary_sequence_mask,masked_softmax
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
import numpy
from allennlp.nn.util import masked_softmax, weighted_sum
from allennlp.nn.util import add_positional_features
from typing import List ,Tuple,Dict
from myallennlp.modules.refiner.mlp_forward_backward import MLPForwardBackward
from myallennlp.modules.refiner.mlp_auto_encoder import MLPAutoEncoder
from myallennlp.modules.reparametrization.gumbel_softmax import hard, _sample_gumbel,inplace_masked_gumbel_softmax

@Seq2SeqEncoder.register("srl_score_refiner")
class SRLScoreBasedRefiner(Seq2SeqEncoder):
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
                 iterations: int = 1,
                 dropout:float=0.3,
                 stright_through:bool=False,
                 denoise:bool=False,
                 hidden_dim:int = 300,
                 detach:bool=False,
                 initial_loss:bool = True,
                 gumbel_t:float = 1.0) -> None:
        super(SRLScoreBasedRefiner, self).__init__()
        self.gumbel_t = gumbel_t
        self.detach = detach
        self.denoise = denoise
        self.stright_through = stright_through
        self.iterations = iterations
        self.hidden_dim = hidden_dim
        self._dropout = Dropout(dropout)
        self.initial_loss = initial_loss

    def set_score_mlp(self,n_tags:int,
                 extra_dim: int):
        self.predicte_mlp = MLPAutoEncoder(n_tags + 1,extra_dim,self.hidden_dim,sum_dim=1,dropout=self._dropout.p )
   #     self.argument_mlp = MLPAutoEncoder(n_tags + 1,extra_dim,self.hidden_dim,sum_dim = 2,dropout=self._dropout.p,max = self.max )

    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return True


    def get_score_per_feature(self,predict_representation:torch.Tensor,
                              argument_representation:torch.Tensor,
                arc_tag_logits:torch.Tensor,
                arc_tag_relaxed:torch.Tensor,
                arc_probs:torch.Tensor,
                mask: torch.Tensor,
                intermediates:Dict={}) -> Tuple[torch.Tensor]:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.

        Parameters
        ----------
        arc_tag_relaxed : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, ,sequence_length, tag_num),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        arc_tag_logits : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, ,sequence_length, tag_num),
            a distribution over attachements of a given word to all other words.
        predict_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, pre_dim).
            The indices of the heads for every word.
        argument_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, arg_dim).
            The dependency labels of the heads for every word.
        mask : ``torch.Tensor``, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        Returns
        -------
        arc_scores : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length),
        tag_score : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length, tag_num),
        predicate_score : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, 1),
        argument_score : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, 1),
        """

        float_mask = mask.float().unsqueeze(-1)

        twod_mask =   float_mask.unsqueeze(1)* float_mask.unsqueeze(2)
        predicate_score,pre_intermediates = self.predicte_mlp.get_score(arc_tag_relaxed*twod_mask,arc_probs*twod_mask,argument_representation,predict_representation,intermediates=intermediates)
    #    argument_score,arg_intermediates = self.argument_mlp.get_score(arc_tag_relaxed*arc_probs,arc_probs,argument_representation,intermediates=arg_intermediates)
        return arc_tag_logits* arc_tag_relaxed , \
               predicate_score*float_mask#,\argument_score*float_mask



    @overrides
    def forward(self,predict_representation:torch.Tensor,
                argument_representation:torch.Tensor,
                input_arc_tag_logits:torch.Tensor,
                arc_tag_probs:torch.Tensor,
                input_arc_logits:torch.Tensor,
                arc_probs:torch.Tensor,
               verb_indicator :torch.LongTensor,
                mask: torch.Tensor,
                arc_tags:torch.Tensor = None): # pylint: disable=arguments-differ
        '''

        :param predict_representation:
        :param argument_representation:
        :param input_arc_tag_logits:
        :param arc_tag_probs:
        :param mask:
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.
        :param arc_tags:
        :param pre_dropout_mask:
        :param arg_dropout_mask:
        :return:
        '''
        float_mask = mask.float().unsqueeze(-1)
        batch_size, sequence_length = mask.size()
        def get_tag_gradient(arc_tag_probs):
            if arc_tags is not None and self.training and False:
                return - soft_tags
            else:
                return 0

        if arc_tags is not None:
            soft_tags = torch.zeros(batch_size, sequence_length, sequence_length, arc_tag_probs.size(-1),
                                    device=arc_tags.device)
            soft_tags.scatter_(3, arc_tags.unsqueeze(3) + 1, 1)
            soft_tags = soft_tags* float_mask.unsqueeze(1) * float_mask.unsqueeze(2)

        if self.initial_loss:
            if self.denoise and self.training:

                gumbel_noise = _sample_gumbel(input_arc_tag_logits.size(), out=input_arc_tag_logits.new())
                arc_tag_logits =   input_arc_tag_logits + gumbel_noise
            else:

                arc_tag_logits =   input_arc_tag_logits + get_tag_gradient(arc_tag_probs)
            arc_tag_probs = torch.nn.functional.softmax(arc_tag_logits,dim=-1)

            # shape (batch_size, sequence_length, sequence_length,1)
            arc_probs = 1 - arc_tag_probs[:, :, :, 0].unsqueeze(-1)

            arc_probs_list = [arc_probs]
            arc_tag_probs_list = [ arc_tag_probs ]

            intermediates_list = [self.predicte_mlp.get_intermediates(arc_tag_probs,arc_probs,argument_representation,predict_representation)]
    #        arg_intermediates_list = [self.argument_mlp.get_intermediates(arc_tag_probs,arc_probs,argument_representation)]
        else:
            arc_tag_probs_list = []
            arc_probs_list = []

            intermediates_list = []
      #      arg_intermediates_list = []
        if self.iterations > 0:
            for i in range( self.iterations ):

                masked_arc_probs = arc_probs * float_mask.unsqueeze(1) * float_mask.unsqueeze(2)
                masked_arc_tag_probs = arc_tag_probs * masked_arc_probs

                arc_tag_logits_grad,predicative_gradient_arc,intermediates = self.predicte_mlp(masked_arc_tag_probs,masked_arc_probs,argument_representation,predict_representation)

                all_grads = torch.zeros(batch_size, sequence_length, sequence_length, arc_tag_probs.size(-1)-1,
                                    device=arc_tags.device)
                all_grads = torch.cat([-predicative_gradient_arc,all_grads],dim=-1)
                arc_tag_logits_grad = arc_tag_logits_grad + all_grads
          #      argument_gradient,argument_gradient_arc,arg_intermediates = self.argument_mlp(masked_arc_tag_probs,masked_arc_probs,argument_representation)
                if self.detach:
                    arc_tag_logits_grad = arc_tag_logits_grad.detach()
            #        argument_gradient = argument_gradient.detach()

                if self.denoise and self.training:

                    gumbel_noise = _sample_gumbel(arc_tag_logits_grad.size(), out=arc_tag_logits_grad.new())
                    arc_tag_logits =   input_arc_tag_logits + arc_tag_logits_grad + gumbel_noise
                else:

                    arc_tag_logits =   input_arc_tag_logits + arc_tag_logits_grad + get_tag_gradient(arc_tag_probs)
                arc_tag_probs = torch.nn.functional.softmax(arc_tag_logits,dim=-1)

           #     arc_logits = input_arc_logits + predicative_gradient_arc  + argument_gradient_arc + get_arc_gradient(arc_probs)


                # shape (batch_size, sequence_length, sequence_length,1)
                arc_probs = 1 - arc_tag_probs[:, :, :, 0]  # arc_scores.sigmoid()
                arc_probs = arc_probs.unsqueeze(-1)



                if self.stright_through and False:
                    arc_tag_probs = hard(arc_tag_probs,float_mask)

                arc_probs_list.append(arc_probs)
                arc_tag_probs_list.append(arc_tag_probs)
                intermediates_list.append(intermediates)
     #       arg_intermediates_list.append(arg_intermediates)


        return arc_tag_probs_list,arc_probs_list,intermediates_list#,arg_intermediates_list
