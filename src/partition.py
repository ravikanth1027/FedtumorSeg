"""
Non-IID client partitioning and modality dropout simulation
(Section IV-B and IV-C).

Renaming note: the Dirichlet concentration parameter is called `kappa` here
(and in the corrected LaTeX), not `beta`, to avoid the three-way symbol
collision the original manuscript had between the Dirichlet parameter, the
aggregation hyperparameter, and the Focal Tversky trade-off parameter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class ClientPartition:
    client_id: int
    patient_ids: List[str]
    hgg_count: int
    lgg_count: int
    scanner_scale: Dict[str, float] = field(default_factory=dict)   # a_{k,mod}
    scanner_shift: Dict[str, float] = field(default_factory=dict)   # b_{k,mod}
    modality_mask: Sequence[int] = (1, 1, 1, 1)  # m_k, 1=present, 0=dropped

    def modalities_present(self) -> float:
        """M_k = sum_j m_{k,j} / 4  (Section III-D, item 2)."""
        return sum(self.modality_mask) / len(self.modality_mask)


MODALITIES = ("T1", "T1ce", "T2", "FLAIR")


def dirichlet_partition(
    hgg_ids: Sequence[str],
    lgg_ids: Sequence[str],
    num_clients: int,
    kappa: float,
    min_patients: int = 50,
    max_patients: int = 150,
    seed: int = 0,
) -> List[ClientPartition]:
    """
    Stage 1 (Section IV-B): sample p_k ~ Dirichlet(kappa, kappa) over
    (HGG, LGG) grade proportions per client, then assign n_k patients drawn
    ~ Uniform(min_patients, max_patients) with HGG/LGG counts following p_k.

    kappa=1.0  -> IID regime (balanced grade distribution)
    kappa=0.5  -> moderate non-IID regime
    kappa=0.1  -> severe non-IID regime
    """
    rng = np.random.default_rng(seed)
    hgg_ids = list(hgg_ids)
    lgg_ids = list(lgg_ids)
    rng.shuffle(hgg_ids)
    rng.shuffle(lgg_ids)

    hgg_ptr, lgg_ptr = 0, 0
    partitions: List[ClientPartition] = []

    for k in range(num_clients):
        p_hgg, p_lgg = rng.dirichlet([kappa, kappa])
        n_k = int(rng.integers(min_patients, max_patients + 1))

        n_hgg = int(round(n_k * p_hgg))
        n_lgg = n_k - n_hgg

        # Don't run past the available pool; wrap around if needed (only
        # relevant for very large num_clients relative to dataset size).
        client_hgg = [hgg_ids[(hgg_ptr + i) % len(hgg_ids)] for i in range(n_hgg)] if hgg_ids else []
        client_lgg = [lgg_ids[(lgg_ptr + i) % len(lgg_ids)] for i in range(n_lgg)] if lgg_ids else []
        hgg_ptr += n_hgg
        lgg_ptr += n_lgg

        partitions.append(
            ClientPartition(
                client_id=k,
                patient_ids=client_hgg + client_lgg,
                hgg_count=len(client_hgg),
                lgg_count=len(client_lgg),
            )
        )

    return partitions


def apply_scanner_simulation(
    partitions: List[ClientPartition],
    seed: int = 0,
) -> None:
    """
    Stage 2 (Section IV-B): simulate scanner-induced intensity shifts.

        x_scanned = a_{k,mod} * x + b_{k,mod}
        a_{k,mod} ~ N(1.0, 0.2),  b_{k,mod} ~ N(0.0, 0.1)

    Mutates each ClientPartition in place, filling scanner_scale/scanner_shift
    per modality. Applying the actual transform to an image tensor is done in
    dataset.py (`apply_scanner_shift`), since that needs the tensor itself.
    """
    rng = np.random.default_rng(seed)
    for part in partitions:
        for mod in MODALITIES:
            part.scanner_scale[mod] = float(rng.normal(1.0, 0.2))
            part.scanner_shift[mod] = float(rng.normal(0.0, 0.1))


def apply_modality_dropout(
    partitions: List[ClientPartition],
    rho: float,
    dropped_modality: str = "T1ce",
    seed: int = 0,
) -> None:
    """
    Section IV-C: randomly drop `dropped_modality` (T1ce by default, as it
    is the most informative modality for enhancing-tumor delineation) for a
    fraction rho of clients. rho in {0, 0.2, 0.4, 0.6} in the paper.

    Clients selected for dropout get modality_mask = (1, 0, 1, 1) matching
    (T1, T1ce, T2, FLAIR) ordering.
    """
    if dropped_modality not in MODALITIES:
        raise ValueError(f"Unknown modality {dropped_modality!r}, expected one of {MODALITIES}")

    rng = np.random.default_rng(seed)
    num_dropped = int(round(rho * len(partitions)))
    dropped_client_ids = set(
        rng.choice([p.client_id for p in partitions], size=num_dropped, replace=False)
    )

    drop_idx = MODALITIES.index(dropped_modality)
    for part in partitions:
        mask = [1, 1, 1, 1]
        if part.client_id in dropped_client_ids:
            mask[drop_idx] = 0
        part.modality_mask = tuple(mask)


if __name__ == "__main__":
    # Small synthetic sanity check (369 patients: 255 HGG, 114 LGG, as
    # reported in Section IV-A for the real BraTS 2020 training set).
    hgg = [f"HGG_{i:03d}" for i in range(255)]
    lgg = [f"LGG_{i:03d}" for i in range(114)]

    for kappa, label in [(1.0, "IID"), (0.5, "moderate"), (0.1, "severe")]:
        parts = dirichlet_partition(hgg, lgg, num_clients=6, kappa=kappa, seed=42)
        apply_scanner_simulation(parts, seed=42)
        apply_modality_dropout(parts, rho=0.4, seed=42)
        print(f"\n-- kappa={kappa} ({label}) --")
        for p in parts:
            total = p.hgg_count + p.lgg_count
            hgg_frac = p.hgg_count / total if total else 0.0
            print(
                f"  client {p.client_id}: n={total:3d}  HGG={p.hgg_count:3d} "
                f"({hgg_frac:.2f})  LGG={p.lgg_count:3d}  "
                f"modality_mask={p.modality_mask}  M_k={p.modalities_present():.2f}"
            )
