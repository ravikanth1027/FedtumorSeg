"""
Confidence-weighted, modality-aware aggregation (Section III-D).

    w_k = n_k * (1 + alpha * D~_k) * (1 + beta * M~_k)
          --------------------------------------------          (Eq. 9)
          sum_j n_j * (1 + alpha * D~_j) * (1 + beta * M~_j)

    D~_k = (D_k - mu_D) / sigma_D                                (Eq. 10)
    M~_k = (M_k - mu_M) / sigma_M                                (Eq. 11)

    theta_{t+1} = sum_k w_k * theta_k^{t+1}                      (Eq. 12)

Safeguard: D~_k and M~_k are z-scores and can be negative, so the raw
factors (1 + alpha*D~_k) / (1 + beta*M~_k) could in principle go negative
for a severely underperforming or modality-poor client. We clip each factor
at zero before normalizing, so w_k >= 0 always holds and the aggregation
remains a valid convex combination. This matches the safeguard added to the
LaTeX manuscript (Section III-D, immediately after Eq. 9).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class ClientUpdate:
    client_id: int
    state_dict: dict            # model.state_dict() from the client
    n_k: int                    # number of local training samples
    d_val: float                # local validation Dice, D_k^val in [0, 1]
    m_k: float                  # modality completeness, M_k in (0, 1]


def _zscore(values: np.ndarray) -> np.ndarray:
    mu = values.mean()
    sigma = values.std()
    if sigma < 1e-8:
        # All clients identical on this metric this round -> no
        # differentiation; z-scores are all zero.
        return np.zeros_like(values)
    return (values - mu) / sigma


def compute_aggregation_weights(
    updates: List[ClientUpdate],
    alpha: float = 0.5,
    beta: float = 0.3,
) -> np.ndarray:
    """
    Returns an array of non-negative weights w_k (summing to 1), one per
    client update, following Eq. 9-11 with the zero-clipping safeguard.
    """
    n = np.array([u.n_k for u in updates], dtype=np.float64)
    d = np.array([u.d_val for u in updates], dtype=np.float64)
    m = np.array([u.m_k for u in updates], dtype=np.float64)

    d_tilde = _zscore(d)
    m_tilde = _zscore(m)

    perf_factor = np.clip(1.0 + alpha * d_tilde, a_min=0.0, a_max=None)
    modality_factor = np.clip(1.0 + beta * m_tilde, a_min=0.0, a_max=None)

    raw_weights = n * perf_factor * modality_factor

    total = raw_weights.sum()
    if total <= 0:
        # Degenerate case (e.g. every clipped factor hit zero): fall back
        # to plain FedAvg-style weighting by dataset size so training can
        # still proceed.
        return n / n.sum()

    return raw_weights / total


def aggregate_state_dicts(updates: List[ClientUpdate], weights: np.ndarray) -> Dict:
    """
    theta_{t+1} = sum_k w_k * theta_k^{t+1}   (Eq. 12)

    Assumes all client state_dicts share the same keys/shapes (standard FL
    assumption -- all clients train the same architecture).
    """
    import torch  # local import so this module is importable without torch

    agg_state = {}
    keys = updates[0].state_dict.keys()
    for key in keys:
        stacked = torch.stack(
            [updates[i].state_dict[key].float() * float(weights[i]) for i in range(len(updates))],
            dim=0,
        )
        agg_state[key] = stacked.sum(dim=0)
        # Preserve original dtype (e.g. buffers that are int64/long)
        agg_state[key] = agg_state[key].to(updates[0].state_dict[key].dtype)
    return agg_state


if __name__ == "__main__":
    # Pure-numpy sanity check of the weighting logic (no torch needed).
    fake_updates = [
        ClientUpdate(client_id=0, state_dict={}, n_k=100, d_val=0.85, m_k=1.0),
        ClientUpdate(client_id=1, state_dict={}, n_k=80, d_val=0.60, m_k=0.75),  # weak client
        ClientUpdate(client_id=2, state_dict={}, n_k=120, d_val=0.90, m_k=1.0),
        ClientUpdate(client_id=3, state_dict={}, n_k=60, d_val=0.40, m_k=0.5),   # very weak/partial
    ]
    w = compute_aggregation_weights(fake_updates, alpha=0.5, beta=0.3)
    print("Weights:", w, "sum:", w.sum())
    assert np.all(w >= 0), "Weights must be non-negative"
    assert abs(w.sum() - 1.0) < 1e-8, "Weights must sum to 1"
    print("OK: weights are non-negative and normalized.")
