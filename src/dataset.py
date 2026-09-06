"""
BraTS 2020 dataset loading and preprocessing (Section IV-A).

Expects the standard BraTS directory layout, one folder per patient:

    BraTS20_Training_XXX/
        BraTS20_Training_XXX_t1.nii.gz
        BraTS20_Training_XXX_t1ce.nii.gz
        BraTS20_Training_XXX_t2.nii.gz
        BraTS20_Training_XXX_flair.nii.gz
        BraTS20_Training_XXX_seg.nii.gz

Preprocessing pipeline (Section IV-A): this module assumes skull-stripping,
z-score normalization, and isotropic resampling have already been applied
upstream (e.g. via the nnU-Net preprocessing pipeline cited in the paper);
here we additionally crop/pad to the fixed 128x128x128 volume size used for
training, and apply per-client scanner and modality-dropout simulation
(Section IV-B/IV-C) at __getitem__ time.

Requires `nibabel` to actually read .nii.gz files:
    pip install nibabel
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - allows module import without torch
    torch = None
    Dataset = object

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - only needed when actually loading files
    nib = None

from .partition import ClientPartition, MODALITIES

TARGET_SHAPE = (128, 128, 128)  # Section III-G


def _load_nifti(path: str) -> np.ndarray:
    if nib is None:
        raise ImportError("nibabel is required to load .nii.gz files: pip install nibabel")
    return np.asarray(nib.load(path).get_fdata(), dtype=np.float32)


def _center_crop_or_pad(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Center-crop if larger than target, zero-pad if smaller, per axis."""
    out = volume
    for axis, target in enumerate(target_shape):
        current = out.shape[axis]
        if current > target:
            start = (current - target) // 2
            out = np.take(out, indices=range(start, start + target), axis=axis)
        elif current < target:
            pad_total = target - current
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            pad_width = [(0, 0)] * out.ndim
            pad_width[axis] = (pad_before, pad_after)
            out = np.pad(out, pad_width, mode="constant", constant_values=0)
    return out


def apply_scanner_shift(
    volume: np.ndarray, scale: float, shift: float
) -> np.ndarray:
    """x_scanned = a * x + b  (Section IV-B, Stage 2, Eq. after Eq. 15)."""
    return (volume * scale + shift).astype(np.float32)


def apply_modality_mask(volume_4ch: np.ndarray, modality_mask: Sequence[int]) -> np.ndarray:
    """
    Input-level modality masking (Eq. 2, Section III-B.2): zero out any
    channel whose modality is unavailable at this client.

    volume_4ch: (4, D, H, W) array ordered as (T1, T1ce, T2, FLAIR).
    modality_mask: length-4 sequence of 0/1.
    """
    out = volume_4ch.copy()
    for ch, present in enumerate(modality_mask):
        if not present:
            out[ch] = 0.0
    return out


class BraTSClientDataset(Dataset):
    """
    A single federated client's local dataset: the patients assigned to it
    by `partition.dirichlet_partition`, with that client's scanner-shift and
    modality-dropout settings applied.
    """

    def __init__(
        self,
        data_root: str,
        partition: ClientPartition,
        target_shape: Tuple[int, int, int] = TARGET_SHAPE,
        label_convention: str = "model",
    ):
        if torch is None:
            raise ImportError("PyTorch is required to use BraTSClientDataset: pip install torch")
        self.data_root = data_root
        self.partition = partition
        self.target_shape = target_shape
        self.label_convention = label_convention

    def __len__(self) -> int:
        return len(self.partition.patient_ids)

    def _patient_dir(self, patient_id: str) -> str:
        return os.path.join(self.data_root, patient_id)

    def _load_patient_volume(self, patient_id: str) -> np.ndarray:
        """Returns (4, D, H, W) array ordered (T1, T1ce, T2, FLAIR)."""
        pdir = self._patient_dir(patient_id)
        suffix_map = {"T1": "t1", "T1ce": "t1ce", "T2": "t2", "FLAIR": "flair"}
        channels = []
        for mod in MODALITIES:
            path = os.path.join(pdir, f"{patient_id}_{suffix_map[mod]}.nii.gz")
            vol = _load_nifti(path)
            vol = _center_crop_or_pad(vol, self.target_shape)
            # Apply this client's simulated scanner shift for this modality.
            scale = self.partition.scanner_scale.get(mod, 1.0)
            shift = self.partition.scanner_shift.get(mod, 0.0)
            vol = apply_scanner_shift(vol, scale, shift)
            channels.append(vol)
        return np.stack(channels, axis=0)

    def _load_label(self, patient_id: str) -> np.ndarray:
        pdir = self._patient_dir(patient_id)
        path = os.path.join(pdir, f"{patient_id}_seg.nii.gz")
        label = _load_nifti(path).astype(np.int64)
        label = _center_crop_or_pad(label, self.target_shape)
        if self.label_convention == "model":
            # Remap raw BraTS labels (0, 1=NCR/NET, 2=ED, 4=ET) to the
            # paper's model-output convention (0=bg, 1=WT, 2=TC, 3=ET),
            # i.e. collapse to sub-region indices rather than raw tissue
            # classes. This mirrors evaluate_subregions in metrics.py.
            remapped = np.zeros_like(label)
            remapped[np.isin(label, [1, 2, 4])] = 1  # WT
            remapped[np.isin(label, [1, 4])] = 2      # TC
            remapped[label == 4] = 3                  # ET
            label = remapped
        return label

    def __getitem__(self, idx: int):
        patient_id = self.partition.patient_ids[idx]
        volume = self._load_patient_volume(patient_id)          # (4, D, H, W)
        label = self._load_label(patient_id)                    # (D, H, W)

        volume = apply_modality_mask(volume, self.partition.modality_mask)  # Eq. 2

        volume_t = torch.from_numpy(volume).float()
        label_t = torch.from_numpy(label).long()
        mask_t = torch.tensor(self.partition.modality_mask, dtype=torch.float32)

        return volume_t, label_t, mask_t


def make_client_dataloader(
    data_root: str,
    partition: ClientPartition,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 2,
):
    """Convenience wrapper matching the paper's batch size = 2 (Section III-G)."""
    from torch.utils.data import DataLoader

    dataset = BraTSClientDataset(data_root, partition)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    # This block only exercises the pure-numpy helpers (crop/pad, scanner
    # shift, modality masking) since it doesn't require real BraTS files,
    # nibabel, or torch to be installed.
    vol = np.random.randn(4, 155, 240, 240).astype(np.float32)
    cropped = _center_crop_or_pad(vol[0], TARGET_SHAPE)
    print("Cropped shape:", cropped.shape)
    assert cropped.shape == TARGET_SHAPE

    shifted = apply_scanner_shift(cropped, scale=1.1, shift=0.05)
    print("Scanner-shifted range:", shifted.min(), shifted.max())

    vol4 = np.random.randn(*((4,) + TARGET_SHAPE)).astype(np.float32)
    masked = apply_modality_mask(vol4, modality_mask=(1, 0, 1, 1))
    assert np.all(masked[1] == 0.0), "T1ce channel should be zeroed out"
    print("OK: modality masking zeroes the dropped channel.")
