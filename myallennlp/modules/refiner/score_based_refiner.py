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
from typing import List ,Tuple
from myallennlp.modules.relaxed_gcn import RelaxedGCN
from myallennlp.modules.reparametrization.gumbel_softmax import hard, masked_gumbel_softmax,inplace_masked_gumbel_softmax

@Seq2SeqEncoder.register("score_refiner")
class ScoreBasedRefiner(Seq2SeqEncoder):
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
                iterations: int = 1,
                 testing_iterations:int = None,
                 iterative_lr: float = 0.01,
                 learn_lr:bool=True,
                 dropout:float=0.3,
                 stright_through:bool=False,
                 detach:bool=False,
                 mean_filed: bool = False,
                 cooldown:float = 0.95,
                 diagonal_mask:bool = True,
                 initial_loss:bool = True,
                 gumbel_t:float = 1.0) -> None:
        super(ScoreBasedRefiner, self).__init__()
        self.gumbel_t = gumbel_t
        self.mean_filed = mean_filed
        self.detach = detach
        self.stright_through = stright_through
        self.cooldown = cooldown
        self.diagonal_mask = diagonal_mask
        self.iterative_lr = torch.nn.Parameter(torch.FloatTensor([iterative_lr]).log(),requires_grad=learn_lr)
        self.iterations = iterations
        self.testing_iterations = testing_iterations if testing_iterations is not None else iterations
        self._dropout = Dropout(dropout)
        self.initial_loss = initial_loss
        self.high_order_feedforward_head = FeedForward(input_dim, 1,
                                                  hidden_dim,
                                                    Activation.by_name("elu")(),dropout=dropout)
        self.high_order_feedforward_dep = FeedForward(input_dim, 1,
                                                  hidden_dim,
                                                    Activation.by_name("elu")(),dropout=dropout)
        high_order_dim = self.high_order_feedforward_head.get_output_dim()
        self.high_order_relations = BilinearMatrixAttention(high_order_dim,high_order_dim, use_input_biases=True, label_dim=2)
    def get_lr(self):
        return self.iterative_lr.exp().item()
    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return False

    def get_high_order_weights(self,inputs,mask):
        head = self._dropout(self.high_order_feedforward_head(inputs))
        dep =self._dropout( self.high_order_feedforward_dep(inputs))
        high_order_weights = self.high_order_relations(head,dep)
      #  high_order_weights = self._dropout(high_order_weights)
        if self.diagonal_mask:
            diagonal_mask = (1-torch.eye(high_order_weights.size(2),device=high_order_weights.device)).unsqueeze(0).unsqueeze(0)
            high_order_weights = high_order_weights * diagonal_mask
        float_mask = mask.float().unsqueeze(1)
        return high_order_weights * float_mask.unsqueeze(2) * float_mask.unsqueeze(3)

    def _score_per_instance(self,
                        attended_arcs: torch.Tensor,
                        relaxed_head: torch.Tensor,
                        relaxed_head_tags: torch.Tensor,
                        mask: torch.Tensor,
                        high_order_weights:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.

        Parameters
        ----------
        head_tag_logits : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_num),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachements of a given word to all other words.
        relaxed_head : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length).
            The indices of the heads for every word.
        relaxed_head_tags : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, num_head_tags).
            The dependency labels of the heads for every word.
        mask : ``torch.Tensor``, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        Returns
        -------
        arc_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc loss.
        tag_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc tag loss.
        """
        float_mask = mask.float()
        batch_size, sequence_length, _ = attended_arcs.size()

        sibling = torch.matmul(relaxed_head,relaxed_head.transpose(1,2))
        sibling_weights = high_order_weights[:,0]
        sibling_score = sibling*sibling_weights * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)


        grand_pa = torch.matmul(relaxed_head,relaxed_head)
        grand_pa_weights = high_order_weights[:,1]
        grand_pa_score = grand_pa*grand_pa_weights * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)

        # We don't care about predictions for the symbolic ROOT token's head,
        # so we remove it from the loss.
        # batch, length
        sibling_score =  sibling_score[:,1:]
        grand_pa_score = grand_pa_score[:,1:]
        return sibling_score , grand_pa_score

    def _high_order_output_distance(self,
                        relaxed_head: torch.Tensor,
                        relaxed_head_tag:torch.Tensor,
                        soft_head: torch.Tensor,
                        soft_tags: torch.Tensor,
                        mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
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

        Returns
        -------
        arc_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc loss.
        tag_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc tag loss.
        """
        float_mask = mask.float()
        batch_size, sequence_length, _ = relaxed_head.size()


        sibling = torch.matmul(relaxed_head,relaxed_head.transpose(1,2))
        gold_sibling = torch.matmul(soft_head,soft_head.transpose(1,2))
        sibling_delta = torch.abs(sibling-gold_sibling)* float_mask.unsqueeze(2) * float_mask.unsqueeze(1)
        sibling_delta = sibling_delta[:,1:]

        grand_pa = torch.matmul(relaxed_head,relaxed_head)
        gold_grand_pa = torch.matmul(soft_head,soft_head)
        grand_pa_delta = torch.abs(grand_pa-gold_grand_pa)* float_mask.unsqueeze(2) * float_mask.unsqueeze(1)
        grand_pa_delta = grand_pa_delta[:,1:]

        return sibling_delta*sibling_delta , grand_pa_delta*grand_pa_delta

    def stational_score(self,
                        relaxed_head: torch.Tensor,
                        soft_tags: torch.Tensor,
                        input_attended_arcs:torch.Tensor,
                        high_order_weights:torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
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

        Returns
        -------
        arc_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc loss.
        tag_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc tag loss.
        """
        float_mask = mask.float()
        minus_mask = (1 - mask).byte().unsqueeze(2)
        batch_size, sequence_length, _ = relaxed_head.size()
        input_head = relaxed_head
        def mask_attended_arcs(attended_arcs):
            attended_arcs = attended_arcs + torch.diag(attended_arcs.new(mask.size(1)).fill_(-numpy.inf))
            attended_arcs.masked_fill_(minus_mask, -numpy.inf)
            return attended_arcs
        sibling_weights = high_order_weights[:, 0]
        grand_pa_weights = high_order_weights[:, 1]

        lr = self.iterative_lr.exp()
        if self.mean_filed:

            attended_arcs = input_attended_arcs + torch.matmul(grand_pa_weights, relaxed_head.transpose(1, 2)) \
                            + torch.matmul(sibling_weights, relaxed_head) + torch.matmul(relaxed_head.transpose(1, 2),
                                                                                         grand_pa_weights) \
                            + torch.matmul(sibling_weights.transpose(1, 2),
                                           relaxed_head)


        else:
            grad = input_attended_arcs + torch.matmul(grand_pa_weights, relaxed_head.transpose(1, 2)) \
                   + torch.matmul(sibling_weights, relaxed_head) + torch.matmul(relaxed_head.transpose(1, 2),
                                                                                grand_pa_weights) \
                   + torch.matmul(sibling_weights.transpose(1, 2),
                                  relaxed_head) - (torch.log(relaxed_head + 1e-10) + 1) * float_mask.unsqueeze(
                1) * float_mask.unsqueeze(2)

            grad = grad - (grad * relaxed_head).sum(2, keepdim=True)
            attended_arcs = input_attended_arcs + lr * grad * relaxed_head
        #    attended_arcs = masked_log_softmax(attended_arcs,mask) * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)
        relaxed_head = masked_softmax(mask_attended_arcs(attended_arcs), mask=mask.unsqueeze(-1))

        if self.stright_through:
            relaxed_head = hard(relaxed_head, mask.unsqueeze(-1))
        relaxed_head = relaxed_head * float_mask.unsqueeze(1) * float_mask.unsqueeze(2)
        relaxed_head.masked_fill_(minus_mask, 0)

        distortion = relaxed_head - input_head
        distortion = distortion[:,1:]
        return torch.abs(distortion)
    def set_tagger(self,tag_bilinear:torch.nn.Module):
        self.tag_bilinear = tag_bilinear
        self.num_labels = tag_bilinear.out_features

    def _get_head_tags_with_relaxed(self,
                       head_tag_representation: torch.Tensor,
                       child_tag_representation: torch.Tensor,
                        relaxed_head: torch.Tensor) -> torch.Tensor:
        """
        Decodes the head tags given the head and child tag representations
        and a tensor of head indices to compute tags for. Note that these are
        either gold or predicted heads, depending on whether this function is
        being called to compute the loss, or if it's being called during inference.

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        relaxed_head : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length,sequence_length). The distribution of the heads
            for every word. It is assumed to be masked already

        Returns
        -------
        head_tag_logits : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length, num_head_tags),
            representing logits for predicting a distribution over tags
            for each arc.
        """


        # shape (batch_size, sequence_length, tag_representation_dim)
        selected_head_tag_representations = torch.matmul(relaxed_head,head_tag_representation)
      #  selected_head_tag_representations = selected_head_tag_representations.contiguous()
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self.tag_bilinear(selected_head_tag_representations,
                                            child_tag_representation)
        return head_tag_logits

    @overrides
    def forward(self,head_tag_representation:torch.Tensor,
                child_tag_representation:torch.Tensor,
                input_attended_arcs:torch.Tensor,
                relaxed_head:torch.Tensor,
                input_head_tag_logits:torch.Tensor,
                relaxed_head_tags:torch.Tensor,
                mask: torch.Tensor,
                head_indices:torch.Tensor = None,
                head_tags:torch.Tensor = None,
                inputs: torch.Tensor = None,
                high_order_weights:torch.Tensor=None): # pylint: disable=arguments-differ
        def mask_attended_arcs(attended_arcs):
            attended_arcs = attended_arcs + torch.diag(attended_arcs.new(mask.size(1)).fill_(-numpy.inf))
            attended_arcs.masked_fill_(minus_mask, -numpy.inf)
            return attended_arcs

        def get_delta_y_gradient(relaxed_head_t):

            if head_indices is not None and self.training :
                return - soft_head/(relaxed_head_t+1e-8)  * float_mask.unsqueeze(1) * float_mask.unsqueeze(2)
            else:
                return 0
        def get_delta_tag_gradient(relaxed_head_tag_t):

            if head_indices is not None and self.training :
                return (relaxed_head_tag_t- soft_tag)  * float_mask.unsqueeze(2)
            else:
                return 0
        float_mask = mask.float()
        minus_mask = (1 - mask).byte().unsqueeze(2)
        batch_size, sequence_length = float_mask.size()
        attended_arcs = input_attended_arcs
        head_tag_logits = input_head_tag_logits
        if high_order_weights is None:
            high_order_weights = self.get_high_order_weights(inputs)

        if head_indices is not None:
            soft_head = torch.zeros(batch_size, sequence_length, sequence_length, device=head_indices.device)
            soft_head.scatter_(2, head_indices.unsqueeze(2), 1)
            soft_head = soft_head * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)


            soft_tag = torch.zeros(batch_size, sequence_length, self.num_labels, device=head_tags.device)
            soft_tag.scatter_(2, head_tags.unsqueeze(2), 1)
            soft_tag = soft_tag * float_mask.unsqueeze(2)

        if self.detach:
            high_order_weights = high_order_weights.detach()

        sibling_weights = high_order_weights[:, 0]
        grand_pa_weights = high_order_weights[:, 1]

        lr = self.iterative_lr.exp()
        if self.initial_loss:
            attended_arcs_list = [attended_arcs + get_delta_y_gradient(relaxed_head)]
            relaxed_head_list = [relaxed_head]
            if self.stright_through:
                relaxed_head = hard(relaxed_head,mask.unsqueeze(-1))
            head_tag_logits_list = [head_tag_logits + get_delta_tag_gradient(relaxed_head_tags)]
            relaxed_head_tags_list = [relaxed_head_tags]

        else:
            attended_arcs_list = []
            relaxed_head_list = []
            if self.stright_through:
                relaxed_head = hard(relaxed_head,mask.unsqueeze(-1))
            head_tag_logits_list = []
            relaxed_head_tags_list = []

        iterations = self.iterations if self.training else self.testing_iterations
        for i in range(iterations):

            if self.mean_filed:

                attended_arcs =  input_attended_arcs  + torch.matmul(grand_pa_weights, relaxed_head.transpose(1, 2)) \
                            + torch.matmul(sibling_weights, relaxed_head) + torch.matmul(relaxed_head.transpose(1, 2),
                                                                                         grand_pa_weights) \
                            + torch.matmul(sibling_weights.transpose(1, 2),
                                           relaxed_head) + get_delta_y_gradient(relaxed_head)


            else:
                grad = input_attended_arcs  + torch.matmul(grand_pa_weights, relaxed_head.transpose(1, 2)) \
                            + torch.matmul(sibling_weights, relaxed_head) + torch.matmul(relaxed_head.transpose(1, 2),
                                                                                         grand_pa_weights) \
                            + torch.matmul(sibling_weights.transpose(1, 2),
                                           relaxed_head) -(torch.log(relaxed_head+1e-10)+1)* float_mask.unsqueeze(1) * float_mask.unsqueeze(2)

                grad = grad - (grad * relaxed_head).sum(2, keepdim=True)
                attended_arcs = attended_arcs + lr * grad * relaxed_head
            #    attended_arcs = masked_log_softmax(attended_arcs,mask) * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)
            if i < self.training :
                relaxed_head = masked_gumbel_softmax(mask_attended_arcs(attended_arcs), tau=self.gumbel_t,
                                                     mask=mask.unsqueeze(-1))
            else:
                relaxed_head = masked_softmax(mask_attended_arcs(attended_arcs), mask=mask.unsqueeze(-1))

            relaxed_head = relaxed_head * float_mask.unsqueeze(1) * float_mask.unsqueeze(2)
            relaxed_head.masked_fill_(minus_mask, 0)
            attended_arcs_list.append(attended_arcs )
            relaxed_head_list.append(relaxed_head)


            if self.stright_through:
                relaxed_head = hard(relaxed_head,mask.unsqueeze(-1))
            head_tag_logits = self._get_head_tags_with_relaxed(head_tag_representation,child_tag_representation,relaxed_head)

            relaxed_head_tags = masked_softmax(head_tag_logits, mask.unsqueeze(-1), dim=2)
            head_tag_logits_list.append(head_tag_logits + get_delta_tag_gradient(relaxed_head_tags))
            relaxed_head_tags_list.append(relaxed_head_tags)

            lr = lr * self.cooldown

        return attended_arcs_list, relaxed_head_list ,head_tag_logits_list, relaxed_head_tags_list
