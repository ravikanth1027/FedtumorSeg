"""
Evaluation metrics (Section IV-E):
  1. Dice Similarity Coefficient (DSC)         -- Eq. 17
  2. 95th Percentile Hausdorff Distance (HD95)  -- Eq. 18

Computed per sub-region (WT, TC, ET) as defined in Section III-A:
  class 1 = whole tumor (WT) region   -> label in {1, 2, 3}
  class 2 = tumor core (TC) region    -> label in {2, 3}       (NOTE: see below)
  class 3 = enhancing tumor (ET)      -> label == 3

The paper's class convention (Section III-A) is:
  0 = background, 1 = whole tumor (WT), 2 = tumor core (TC), 3 = enhancing (ET)
which is a *sub-region* labeling scheme (not the raw BraTS NCR/ED/ET labels),
i.e. the segmentation head already outputs WT/TC/ET regions directly. We
therefore treat class indices as cumulative sub-regions for evaluation
purposes: WT = {1,2,3} (whole tumor = any tumor), TC = {2,3} (tumor core,
excluding edema), ET = {3} (enhancing tumor only). If your data pipeline
instead uses raw BraTS labels (NCR/NET=1, ED=2, ET=4), remap them to this
WT/TC/ET convention before calling these functions -- see
`remap_brats_labels_to_subregions` below.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.ndimage import distance_transform_edt


SUBREGIONS = ("WT", "TC", "ET")


def remap_brats_labels_to_subregions(label_volume: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Raw BraTS labels: 0=background, 1=NCR/NET, 2=ED (edema), 4=ET.
    Converts to three binary sub-region masks used for evaluation:
      WT (whole tumor)  = label in {1, 2, 4}
      TC (tumor core)   = label in {1, 4}
      ET (enhancing)    = label == 4
    """
    wt = np.isin(label_volume, [1, 2, 4])
    tc = np.isin(label_volume, [1, 4])
    et = label_volume == 4
    return {"WT": wt, "TC": tc, "ET": et}


def subregion_masks_from_model_output(label_volume: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Given the model's own 4-class output (0=bg, 1=WT, 2=TC, 3=ET, per
    Section III-A), reconstruct the three cumulative binary sub-region
    masks used for Dice/HD95 evaluation.
    """
    wt = label_volume >= 1
    tc = np.isin(label_volume, [2, 3])
    et = label_volume == 3
    return {"WT": wt, "TC": tc, "ET": et}


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-8) -> float:
    """Eq. 17: DSC = 2|P inter G| / (|P| + |G|)."""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    if denom == 0:
        # Both empty -> perfect agreement by convention.
        return 1.0
    return float(2.0 * intersection / (denom + eps))


def hd95(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    """
    Eq. 18: 95th-percentile symmetric Hausdorff distance, in the units of
    `spacing` (mm, if spacing reflects the true voxel size).

    Implemented via Euclidean distance transforms rather than exhaustive
    pairwise distances, which is the standard efficient approach for 3D
    volumes.
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    if pred_mask.sum() == 0 and gt_mask.sum() == 0:
        return 0.0
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        # One mask empty, the other not: undefined / worst-case distance.
        # Convention: return NaN so callers can exclude this case from
        # averages rather than silently biasing the mean.
        return float("nan")

    # Distance transform of the complement of each mask gives, at every
    # voxel, the distance to the nearest foreground voxel of that mask.
    dt_gt = distance_transform_edt(~gt_mask, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_mask, sampling=spacing)

    # Surface distances: distance from each predicted-boundary voxel to GT,
    # and from each GT-boundary voxel to prediction.
    pred_to_gt = dt_gt[pred_mask]
    gt_to_pred = dt_pred[gt_mask]

    all_dists = np.concatenate([pred_to_gt, gt_to_pred])
    return float(np.percentile(all_dists, 95))


def evaluate_subregions(
    pred_label_volume: np.ndarray,
    gt_label_volume: np.ndarray,
    spacing=(1.0, 1.0, 1.0),
    label_convention: str = "model",
) -> Dict[str, Dict[str, float]]:
    """
    Computes Dice and HD95 for WT/TC/ET given full multi-class prediction
    and ground-truth volumes.

    label_convention:
      "model" -> use the paper's 0/1/2/3 = bg/WT/TC/ET convention
                 (subregion_masks_from_model_output)
      "brats" -> use raw BraTS 0/1/2/4 labels (remap_brats_labels_to_subregions)
    """
    if label_convention == "model":
        pred_masks = subregion_masks_from_model_output(pred_label_volume)
        gt_masks = subregion_masks_from_model_output(gt_label_volume)
    elif label_convention == "brats":
        pred_masks = remap_brats_labels_to_subregions(pred_label_volume)
        gt_masks = remap_brats_labels_to_subregions(gt_label_volume)
    else:
        raise ValueError(f"Unknown label_convention: {label_convention!r}")

    results = {}
    for region in SUBREGIONS:
        results[region] = {
            "dice": dice_score(pred_masks[region], gt_masks[region]),
            "hd95": hd95(pred_masks[region], gt_masks[region], spacing=spacing),
        }
    return results


if __name__ == "__main__":
    # Sanity check: identical volumes -> Dice=1.0, HD95=0.0 for every region.
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 4, size=(16, 16, 16))
    perfect_pred = gt.copy()
    results = evaluate_subregions(perfect_pred, gt)
    print("Identical volumes:", results)
    for region in SUBREGIONS:
        assert abs(results[region]["dice"] - 1.0) < 1e-6
        assert results[region]["hd95"] == 0.0
    print("OK: perfect prediction gives Dice=1.0, HD95=0.0 everywhere.")

    # Slightly perturbed prediction -> Dice < 1.0
    noisy_pred = gt.copy()
    noisy_pred[0:2, 0:2, 0:2] = (noisy_pred[0:2, 0:2, 0:2] + 1) % 4
    noisy_results = evaluate_subregions(noisy_pred, gt)
    print("Noisy volumes:", noisy_results)
