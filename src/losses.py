"""
Compound segmentation loss (Section III-C).

L = L_Dice + lambda * L_FocalTversky            (Eq. 6)

Note on notation: the Focal Tversky trade-off parameters are named
alpha_FT, beta_FT here (and in the corrected LaTeX) to avoid clashing with
the aggregation hyperparameters alpha, beta used in src/aggregation.py
(Eq. 9) and the Dirichlet concentration kappa used in src/partition.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 4


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """labels: (B, D, H, W) int64 -> (B, C, D, H, W) float one-hot."""
    return F.one_hot(labels.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


class DiceLoss(nn.Module):
    """Eq. 7."""

    def __init__(self, num_classes: int = NUM_CLASSES, eps: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_oh = one_hot(targets, self.num_classes)

        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets_oh, dim=dims)
        denom = torch.sum(probs, dim=dims) + torch.sum(targets_oh, dim=dims)
        dice_per_class = (2.0 * intersection + self.eps) / (denom + self.eps)
        return 1.0 - dice_per_class.mean()


class FocalTverskyLoss(nn.Module):
    """
    Eq. 8:
        L_FT = sum_c ( 1 - (TP_c + eps) / (TP_c + alpha_FT*FP_c + beta_FT*FN_c + eps) ) ^ gamma

    Defaults alpha_FT=0.7, beta_FT=0.3, gamma=1 as reported in Section III-C /
    III-G, favoring recall (penalizing false negatives more than false
    positives).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        alpha_ft: float = 0.7,
        beta_ft: float = 0.3,
        gamma: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha_ft = alpha_ft
        self.beta_ft = beta_ft
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_oh = one_hot(targets, self.num_classes)

        dims = (0, 2, 3, 4)
        tp = torch.sum(probs * targets_oh, dim=dims)
        fp = torch.sum(probs * (1 - targets_oh), dim=dims)
        fn = torch.sum((1 - probs) * targets_oh, dim=dims)

        tversky_index = (tp + self.eps) / (
            tp + self.alpha_ft * fp + self.beta_ft * fn + self.eps
        )
        loss_per_class = (1.0 - tversky_index) ** self.gamma
        return loss_per_class.sum()


class CompoundLoss(nn.Module):
    """Eq. 6: L = L_Dice + lambda * L_FocalTversky."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        lam: float = 0.5,
        alpha_ft: float = 0.7,
        beta_ft: float = 0.3,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.dice = DiceLoss(num_classes=num_classes)
        self.focal_tversky = FocalTverskyLoss(
            num_classes=num_classes, alpha_ft=alpha_ft, beta_ft=beta_ft, gamma=gamma
        )
        self.lam = lam

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice(logits, targets) + self.lam * self.focal_tversky(logits, targets)


if __name__ == "__main__":
    logits = torch.randn(2, 4, 8, 8, 8)
    targets = torch.randint(0, 4, (2, 8, 8, 8))
    loss_fn = CompoundLoss()
    loss = loss_fn(logits, targets)
    print("Compound loss:", loss.item())
