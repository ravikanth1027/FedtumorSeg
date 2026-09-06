"""
Federated training loop -- Algorithm 1 in the paper.

Supports three modes via `--method`:
  fedtumorseg : local FocalTversky+Dice loss, confidence-weighted &
                modality-aware aggregation (Eq. 9), FiLM-conditioned model.
  fedavg      : local Dice-only... no -- same compound loss, but plain
                dataset-size-weighted averaging (the paper's FedAvg
                baseline; only the aggregation rule differs, per
                Section IV-D item 2).
  fedprox     : FedAvg local objective + a proximal term mu/2 * ||theta -
                theta_global||^2 (Eq. 1, Section II-B), dataset-size-weighted
                averaging.

All three share the same backbone (AttentionUNetFiLM) and the same compound
loss for a fair comparison, differing only in the local objective
(FedProx's proximal term) and the server aggregation rule -- matching how
the paper frames these as ablated components of FedTumorSeg (Table III).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .aggregation import ClientUpdate, aggregate_state_dicts, compute_aggregation_weights
from .losses import CompoundLoss
from .model import AttentionUNetFiLM
from .partition import ClientPartition


@dataclass
class FedConfig:
    """
    Rather than hardcoding separate "fedavg"/"fedprox"/"fedtumorseg" code
    paths, we expose the three orthogonal design choices the paper actually
    varies across Tables I and III, so any row of either table can be
    reproduced by setting these three fields:

        proximal            : adds the FedProx proximal term (Eq. 1) to the
                               local objective. False for FedAvg-style
                               baselines, plain FedTumorSeg, and the
                               ablation variants; True only for the FedProx
                               baseline.
        use_film             : enables FiLM bottleneck conditioning (Eq. 3).
        use_confidence_weight: enables confidence-weighted, modality-aware
                               aggregation (Eq. 9) instead of plain
                               dataset-size weighting.

    Table I / III mapping:
        FedAvg              -> proximal=False, use_film=False, use_confidence_weight=False
        FedProx             -> proximal=True,  use_film=False, use_confidence_weight=False
        w/o Modality Cond.  -> proximal=False, use_film=False, use_confidence_weight=True
        w/o Confidence Agg. -> proximal=False, use_film=True,  use_confidence_weight=False
        FedTumorSeg (Full)  -> proximal=False, use_film=True,  use_confidence_weight=True
    """
    proximal: bool = False
    use_film: bool = True
    use_confidence_weight: bool = True

    num_rounds: int = 100
    local_epochs: int = 5                # E, Section III-G
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 2
    lam: float = 0.5                     # compound loss lambda (Eq. 6)
    alpha_ft: float = 0.7                # Focal Tversky alpha_FT (Eq. 8)
    beta_ft: float = 0.3                 # Focal Tversky beta_FT (Eq. 8)
    ft_gamma: float = 1.0
    alpha_agg: float = 0.5               # aggregation alpha (Eq. 9)
    beta_agg: float = 0.3                # aggregation beta (Eq. 9)
    fedprox_mu: float = 0.01             # Section IV-D item 3
    convergence_eps: float = 0.01        # Section IV-E item 3
    convergence_patience: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def name(self) -> str:
        if not self.use_film and not self.use_confidence_weight and not self.proximal:
            return "FedAvg"
        if not self.use_film and not self.use_confidence_weight and self.proximal:
            return "FedProx"
        if not self.use_film and self.use_confidence_weight:
            return "FedTumorSeg (w/o Modality Conditioning)"
        if self.use_film and not self.use_confidence_weight:
            return "FedTumorSeg (w/o Confidence-Weighted Agg.)"
        return "FedTumorSeg (Full)"


def _local_train(
    global_state: dict,
    dataloader,
    cfg: FedConfig,
) -> dict:
    """
    Runs E local epochs of SGD (Adam) starting from `global_state`, returns
    the resulting local state_dict. Adds the FedProx proximal term (Eq. 1)
    when cfg.proximal is True.
    """
    model = AttentionUNetFiLM(use_film=cfg.use_film).to(cfg.device)
    model.load_state_dict(global_state)
    model.train()

    # Keep a frozen copy of the global params for the FedProx penalty.
    global_params = {k: v.clone().detach() for k, v in model.state_dict().items()}

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = CompoundLoss(lam=cfg.lam, alpha_ft=cfg.alpha_ft, beta_ft=cfg.beta_ft, gamma=cfg.ft_gamma)

    for _epoch in range(cfg.local_epochs):
        for x, y, m in dataloader:
            x, y, m = x.to(cfg.device), y.to(cfg.device), m.to(cfg.device)

            optimizer.zero_grad()
            logits = model(x, m)
            loss = loss_fn(logits, y)

            if cfg.proximal:
                prox_term = 0.0
                for name, param in model.named_parameters():
                    prox_term = prox_term + torch.sum((param - global_params[name]) ** 2)
                loss = loss + (cfg.fedprox_mu / 2.0) * prox_term

            loss.backward()
            optimizer.step()

    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


@torch.no_grad()
def _validate(state_dict: dict, dataloader, device: str, use_film: bool) -> float:
    """Mean Dice on a held-out validation loader, D_k^val (Section III-D item 1)."""
    from .metrics import subregion_masks_from_model_output, dice_score

    model = AttentionUNetFiLM(use_film=use_film).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    dice_scores = []
    for x, y, m in dataloader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        logits = model(x, m)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        targets = y.cpu().numpy()
        for b in range(preds.shape[0]):
            pred_masks = subregion_masks_from_model_output(preds[b])
            gt_masks = subregion_masks_from_model_output(targets[b])
            region_dices = [dice_score(pred_masks[r], gt_masks[r]) for r in ("WT", "TC", "ET")]
            dice_scores.append(np.mean(region_dices))

    return float(np.mean(dice_scores)) if dice_scores else 0.0


def run_federated_training(
    client_partitions: List[ClientPartition],
    client_train_loaders: List,
    client_val_loaders: List,
    cfg: FedConfig,
) -> dict:
    """
    Full Algorithm 1 loop. Returns the final global state_dict.
    See FedConfig's docstring for how (proximal, use_film,
    use_confidence_weight) map onto the methods in Table I and the ablation
    variants in Table III.

    client_train_loaders[k] / client_val_loaders[k] must yield batches of
    (volume, label, modality_mask) as produced by dataset.BraTSClientDataset.
    """
    global_model = AttentionUNetFiLM(use_film=cfg.use_film).to(cfg.device)
    global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}

    dice_history = []
    rounds_at_convergence = None

    for t in range(1, cfg.num_rounds + 1):
        updates: List[ClientUpdate] = []

        for k, part in enumerate(client_partitions):
            local_state = _local_train(global_state, client_train_loaders[k], cfg)
            d_val = _validate(local_state, client_val_loaders[k], cfg.device, cfg.use_film)
            m_k = part.modalities_present()

            n_k = len(part.patient_ids)
            updates.append(ClientUpdate(client_id=part.client_id, state_dict=local_state, n_k=n_k, d_val=d_val, m_k=m_k))

        # --- Aggregation (server side) ---
        if cfg.use_confidence_weight:
            weights = compute_aggregation_weights(updates, alpha=cfg.alpha_agg, beta=cfg.beta_agg)  # Eq. 9
        else:
            n = np.array([u.n_k for u in updates], dtype=np.float64)
            weights = n / n.sum()  # standard FedAvg-style weighting

        global_state = aggregate_state_dicts(updates, weights)  # Eq. 12

        round_mean_dice = float(np.mean([u.d_val for u in updates]))
        dice_history.append(round_mean_dice)

        print(f"[{cfg.name}] round {t:3d}  mean local-val Dice={round_mean_dice:.4f}  "
              f"weights={np.round(weights, 3).tolist()}")

        # --- Convergence check (Section IV-E item 3) ---
        if rounds_at_convergence is None and len(dice_history) > cfg.convergence_patience:
            recent = dice_history[-(cfg.convergence_patience + 1):]
            deltas = [abs(recent[i + 1] - recent[i]) for i in range(len(recent) - 1)]
            if all(d < cfg.convergence_eps for d in deltas):
                rounds_at_convergence = t
                print(f"[{cfg.name}] converged at round {t} "
                      f"(<{cfg.convergence_eps} change in val Dice for "
                      f"{cfg.convergence_patience} consecutive rounds)")

    return global_state
