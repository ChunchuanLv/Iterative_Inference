from overrides import overrides
import torch
from torch.nn.parameter import Parameter

from allennlp.modules.attention.attention import Attention
from allennlp.nn import Activation

from myallennlp.modules.feedforward_as_seq2seq import MyFeedForward
