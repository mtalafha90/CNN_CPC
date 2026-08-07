"""Loss functions for multi-label detection with a text teacher.

The findings are strongly imbalanced — a Baker cyst or a PCL tear appears in a
small fraction of exams — so plain BCE lets the common findings dominate the
gradient. ``MultiLabelLoss`` addresses that with per-label positive weighting
and optional focal down-weighting of easy examples, and it ignores missing
labels so exams annotated for only some findings can still be used.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelLoss(nn.Module):
    """Binary cross entropy with class balancing, focal weighting and smoothing.

    Parameters
    ----------
    pos_weight:
        Per-label weight applied to positive targets. Use
        :func:`compute_pos_weight` to derive it from the training set.
    focal_gamma:
        Zero disables focal weighting. Values around 1-2 help when the negatives
        overwhelm the positives.
    label_smoothing:
        Pulls targets away from hard 0/1, which reduces over-confidence on the
        noisier report-derived labels.
    """

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        focal_gamma: float = 0.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None  # type: ignore[assignment]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # NaN marks "not annotated"; those entries contribute nothing.
        valid = torch.isfinite(targets)
        targets = torch.nan_to_num(targets, nan=0.0)

        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight if self.pos_weight is not None else None,
        )

        if self.focal_gamma > 0:
            probabilities = torch.sigmoid(logits.detach())
            p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
            loss = loss * (1 - p_t).clamp(min=1e-6).pow(self.focal_gamma)

        loss = loss * valid
        denominator = valid.sum().clamp(min=1)
        return loss.sum() / denominator


class DistillationLoss(nn.Module):
    """Match the image model's logits to the text teacher's soft predictions.

    The radiology reports state the findings almost explicitly, so a text model
    reaches a very high score but is useless at test time if reports are not
    provided. Distilling its *soft* probabilities into the image model transfers
    the extra signal: the teacher expresses uncertainty the binary labels throw
    away, and that uncertainty is exactly what a borderline image looks like.
    """

    def __init__(self, temperature: float = 2.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
        valid = torch.isfinite(teacher_probs)
        teacher_probs = torch.nan_to_num(teacher_probs, nan=0.5).clamp(1e-6, 1 - 1e-6)
        temperature = self.temperature

        student = torch.sigmoid(student_logits / temperature).clamp(1e-6, 1 - 1e-6)
        teacher = torch.sigmoid(torch.logit(teacher_probs) / temperature)

        kl = teacher * torch.log(teacher / student) + (1 - teacher) * torch.log(
            (1 - teacher) / (1 - student)
        )
        kl = kl * valid
        # The temperature squared keeps the gradient scale comparable to the
        # hard-label loss as the temperature changes.
        return (kl.sum() / valid.sum().clamp(min=1)) * (temperature**2)


def compute_pos_weight(
    labels, max_weight: float = 20.0, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Derive per-label positive weights as negatives / positives.

    The cap matters: without it a finding present in 0.2% of exams would get a
    weight near 500 and destabilise training.
    """
    import numpy as np

    array = np.nan_to_num(np.asarray(labels, dtype=np.float64))
    positives = array.sum(axis=0)
    negatives = array.shape[0] - positives
    weights = np.where(positives > 0, negatives / np.maximum(positives, 1.0), 1.0)
    weights = np.clip(weights, 1.0, max_weight)
    return torch.tensor(weights, dtype=torch.float32, device=device)
