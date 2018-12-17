from typing import Optional

import torch

from allennlp.training.metrics.metric import Metric
from allennlp.common.checks import ConfigurationError


@Metric.register("labeled_f1")
class LabeledF1Measure(Metric):
    """
    Computes Precision, Recall and F1 with respect to a given ``positive_label``.
    For example, for a BIO tagging scheme, you would pass the classification index of
    the tag you are interested in, resulting in the Precision, Recall and F1 score being
    calculated for this tag only.
    """
    def __init__(self, negative_label: int) -> None:
        self._negative_label = negative_label
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0


        self._un_true_positives = 0.0
        self._un_true_negatives = 0.0
        self._un_false_positives = 0.0
        self._un_false_negatives = 0.0

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)


        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to F1Measure contains an id >= {}, "
                                     "the number of classes.".format(num_classes))
        if mask is None:
            mask = torch.ones_like(gold_labels)
        mask = mask.float()
        gold_labels = gold_labels.float()
        negative_label_mask = gold_labels.eq(self._negative_label).float()
        positive_label_mask = 1.0 - negative_label_mask

        argmax_predictions = predictions.max(-1)[1].float().squeeze(-1)

        # True Negatives: correct non-positive predictions.
        correct_null_predictions = (argmax_predictions ==
                                    gold_labels).float() * negative_label_mask
        self._true_negatives += (correct_null_predictions.float() * mask).sum()

        # True Positives: correct positively labeled predictions.
        correct_non_null_predictions = (argmax_predictions ==
                                        gold_labels).float() * positive_label_mask
        self._true_positives += (correct_non_null_predictions * mask).sum()

        # False Negatives: incorrect negatively labeled predictions.
        incorrect_null_predictions = (argmax_predictions !=
                                      gold_labels).float() * positive_label_mask
        self._false_negatives += (incorrect_null_predictions * mask).sum()

        # False Positives: incorrect positively labeled predictions
        incorrect_non_null_predictions = (argmax_predictions !=
                                          gold_labels).float() * negative_label_mask
        self._false_positives += (incorrect_non_null_predictions * mask).sum()

        argmax_predictions = argmax_predictions > 0
        gold_labels = gold_labels > 0
        # True Negatives: correct non-positive predictions.
        correct_null_predictions = (argmax_predictions ==
                                    gold_labels).float() * negative_label_mask
        self._un_true_negatives += (correct_null_predictions.float() * mask).sum()

        # True Positives: correct positively labeled predictions.
        correct_non_null_predictions = (argmax_predictions ==
                                        gold_labels).float() * positive_label_mask
        self._un_true_positives += (correct_non_null_predictions * mask).sum()

        # False Negatives: incorrect negatively labeled predictions.
        incorrect_null_predictions = (argmax_predictions !=
                                      gold_labels).float() * positive_label_mask
        self._un_false_negatives += (incorrect_null_predictions * mask).sum()

        # False Positives: incorrect positively labeled predictions
        incorrect_non_null_predictions = (argmax_predictions !=
                                          gold_labels).float() * negative_label_mask
        self._un_false_positives += (incorrect_non_null_predictions * mask).sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        precision = float(self._true_positives) / float(self._true_positives + self._false_positives + 1e-13)
        recall = float(self._true_positives) / float(self._true_positives + self._false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))



        un_precision = float(self._un_true_positives) / float(self._un_true_positives + self._un_false_positives + 1e-13)
        un_recall = float(self._un_true_positives) / float(self._un_true_positives + self._un_false_negatives + 1e-13)
        un_f1_measure = 2. * ((un_precision * un_recall) / (un_precision + un_recall + 1e-13))
        if reset:
            self.reset()

        metrics = {}
        metrics["un_precision"] = un_precision
        metrics["un_recall"] = un_recall
        metrics["un_f1"] = un_f1_measure

        metrics["lablled_precision"] = precision
        metrics["lablled_recall"] = recall
        metrics["lablled_f1"] = f1_measure
        return metrics

    def reset(self):
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0

        self._un_true_positives = 0.0
        self._un_true_negatives = 0.0
        self._un_false_positives = 0.0
        self._un_false_negatives = 0.0
