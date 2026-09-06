"""
Main entry point: reproduces one row of Table I or Table III.

Example usage (Section IV-A/B/C hyperparameters as defaults):

    # FedTumorSeg (Full), severe non-IID, 40% modality dropout
    python train.py --data-root /path/to/BraTS2020_Training \
        --kappa 0.1 --rho 0.4 --use-film --use-confidence-weight

    # FedAvg baseline, same regime
    python train.py --data-root /path/to/BraTS2020_Training \
        --kappa 0.1 --rho 0.4 --no-use-film --no-use-confidence-weight

    # FedProx baseline
    python train.py --data-root /path/to/BraTS2020_Training \
        --kappa 0.1 --rho 0.4 --no-use-film --no-use-confidence-weight --proximal

    # Ablation: w/o Modality Conditioning
    python train.py --data-root /path/to/BraTS2020_Training \
        --kappa 0.1 --rho 0.4 --no-use-film --use-confidence-weight

    # Ablation: w/o Confidence-Weighted Aggregation
    python train.py --data-root /path/to/BraTS2020_Training \
        --kappa 0.1 --rho 0.4 --use-film --no-use-confidence-weight

Requires a real BraTS 2020 directory (see "Getting the dataset" in
README.md) plus `pip install torch nibabel scipy numpy`.
"""
from __future__ import annotations

import argparse
import glob
import os

import torch

from src.dataset import make_client_dataloader
from src.federated import FedConfig, run_federated_training
from src.partition import (
    apply_modality_dropout,
    apply_scanner_simulation,
    dirichlet_partition,
)


def discover_patients(data_root: str):
    """
    Scans `data_root` for BraTS patient folders and splits them into HGG/LGG
    based on folder naming. BraTS 2020 ships HGG and LGG in separate
    subfolders (`HGG/` and `LGG/`) in the original release; adjust this
    function if your copy of the dataset uses a flat layout with a metadata
    CSV instead (e.g. the Kaggle mirror linked in README.md).
    """
    hgg_dir = os.path.join(data_root, "HGG")
    lgg_dir = os.path.join(data_root, "LGG")

    if os.path.isdir(hgg_dir) and os.path.isdir(lgg_dir):
        hgg_ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(hgg_dir, "*")))
        lgg_ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(lgg_dir, "*")))
        return hgg_ids, lgg_ids, {pid: hgg_dir for pid in hgg_ids} | {pid: lgg_dir for pid in lgg_ids}

    raise FileNotFoundError(
        f"Expected {hgg_dir} and {lgg_dir}. If your BraTS copy uses a flat "
        "layout (e.g. the Kaggle mirror), replace discover_patients() with "
        "logic that reads the accompanying name_mapping.csv to get HGG/LGG "
        "labels per patient folder."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FedTumorSeg training / baseline / ablation runner")
    p.add_argument("--data-root", type=str, required=True, help="Path to BraTS2020 training data")
    p.add_argument("--num-clients", type=int, default=6, help="K in the paper (Section IV-B)")
    p.add_argument("--kappa", type=float, default=0.1,
                   help="Dirichlet concentration: 1.0=IID, 0.5=moderate, 0.1=severe (Section IV-B)")
    p.add_argument("--rho", type=float, default=0.4,
                   help="Fraction of clients missing T1ce (Section IV-C)")
    p.add_argument("--num-rounds", type=int, default=100)
    p.add_argument("--local-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha-agg", type=float, default=0.5)
    p.add_argument("--beta-agg", type=float, default=0.3)
    p.add_argument("--fedprox-mu", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--use-film", dest="use_film", action="store_true", default=True)
    p.add_argument("--no-use-film", dest="use_film", action="store_false")
    p.add_argument("--use-confidence-weight", dest="use_confidence_weight", action="store_true", default=True)
    p.add_argument("--no-use-confidence-weight", dest="use_confidence_weight", action="store_false")
    p.add_argument("--proximal", action="store_true", default=False, help="FedProx proximal term (Eq. 1)")

    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="Fraction of each client's local patients held out for D_k^val")
    p.add_argument("--output", type=str, default="global_model.pt")
    return p


def main():
    args = build_parser().parse_args()

    hgg_ids, lgg_ids, patient_dirs = discover_patients(args.data_root)
    print(f"Found {len(hgg_ids)} HGG and {len(lgg_ids)} LGG patients.")

    partitions = dirichlet_partition(
        hgg_ids, lgg_ids, num_clients=args.num_clients, kappa=args.kappa, seed=args.seed
    )
    apply_scanner_simulation(partitions, seed=args.seed)
    apply_modality_dropout(partitions, rho=args.rho, dropped_modality="T1ce", seed=args.seed)

    train_loaders, val_loaders = [], []
    for part in partitions:
        n_val = max(1, int(round(len(part.patient_ids) * args.val_fraction)))
        val_ids = part.patient_ids[:n_val]
        train_ids = part.patient_ids[n_val:]

        train_part = type(part)(**{**part.__dict__, "patient_ids": train_ids})
        val_part = type(part)(**{**part.__dict__, "patient_ids": val_ids})

        # Each patient may live under a different root (HGG/ vs LGG/); the
        # dataset class expects a single data_root, so here we point it at
        # the shared parent and rely on patient_dirs to resolve subfolders.
        # For simplicity this assumes a consistent parent directory; adapt
        # if your layout differs.
        common_root = args.data_root
        train_loaders.append(make_client_dataloader(common_root, train_part, batch_size=args.batch_size))
        val_loaders.append(make_client_dataloader(common_root, val_part, batch_size=args.batch_size, shuffle=False))

    cfg = FedConfig(
        proximal=args.proximal,
        use_film=args.use_film,
        use_confidence_weight=args.use_confidence_weight,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha_agg=args.alpha_agg,
        beta_agg=args.beta_agg,
        fedprox_mu=args.fedprox_mu,
    )
    print(f"Running: {cfg.name}  (kappa={args.kappa}, rho={args.rho}, K={args.num_clients})")

    final_state = run_federated_training(partitions, train_loaders, val_loaders, cfg)
    torch.save(final_state, args.output)
    print(f"Saved final global model to {args.output}")


if __name__ == "__main__":
    main()
