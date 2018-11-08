from typing import Optional, List

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric
from allennlp.training.metrics.attachment_scores import AttachmentScores


@Metric.register("iterative_attachment_scores")
class IterativeAttachmentScores(Metric):
    """
    Computes labeled and unlabeled attachment scores for a
    dependency parse, as well as sentence level exact match
    for both labeled and unlabeled trees. Note that the input
    to this metric is the sampled predictions, not the distribution
    itself.

    Parameters
    ----------
    ignore_classes : ``List[int]``, optional (default = None)
        A list of label ids to ignore when computing metrics.
    """
    def __init__(self, ignore_classes: List[int] = None) -> None:

        self._ignore_classes: List[int] = ignore_classes or []
        self._attachment_scores = {}
    def __call__(self, # type: ignore
                 predicted_indices: torch.Tensor,
                 predicted_labels: torch.Tensor,
                 gold_indices: torch.Tensor,
                 gold_labels: torch.Tensor,
                 n_iteration:int = 0,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predicted_indices : ``torch.Tensor``, required.
            A tensor of head index predictions of shape (batch_size, timesteps).
        predicted_labels : ``torch.Tensor``, required.
            A tensor of arc label predictions of shape (batch_size, timesteps).
        gold_indices : ``torch.Tensor``, required.
            A tensor of the same shape as ``predicted_indices``.
        gold_labels : ``torch.Tensor``, required.
            A tensor of the same shape as ``predicted_labels``.
        mask: ``torch.Tensor``, optional (default = None).
            A tensor of the same shape as ``predicted_indices``.
        """

        attachment_scores = self._attachment_scores.setdefault(n_iteration,AttachmentScores(self._ignore_classes))

        attachment_scores(predicted_indices,predicted_labels,gold_indices,gold_labels,mask=mask)

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated metrics as a dictionary.
        """

        all_metrics = {}

        for iterations in  sorted(self._attachment_scores)[:-1]:

            metrics =  self._attachment_scores[iterations].get_metric()
            for metric in metrics:
                all_metrics[metric+str(iterations)] = metrics[metric]

        iterations =  len(self._attachment_scores)-1

        metrics =  self._attachment_scores[iterations].get_metric()
        for metric in metrics:
            all_metrics[metric] = metrics[metric]
        if reset:
            self.reset()
        return all_metrics

    @overrides
    def reset(self):
        self._attachment_scores = {}
