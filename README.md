# FedTumorSeg

Reference implementation for *"FedTumorSeg: A Non-IID-Aware Federated
Learning Framework for Enhancing and Non-Enhancing Brain Tumor Sub-Region
Segmentation."*

This code implements the method exactly as specified in the (corrected)
manuscript. Where the manuscript's notation was cleaned up (see the LaTeX
revision notes), the code follows the corrected notation, not the original.

## What's here

```
FedTumorSeg/
├── requirements.txt
├── train.py              # entry point: run one method/ablation config
└── src/
    ├── model.py           # Attention U-Net + FiLM conditioning   (Sec III-B)
    ├── losses.py           # Dice + Focal Tversky compound loss   (Sec III-C, Eq. 6-8)
    ├── aggregation.py       # Confidence-weighted, modality-aware  (Sec III-D, Eq. 9-12)
    │                        #  aggregation, with the negative-weight
    │                        #  clipping safeguard from the revised manuscript
    ├── partition.py         # Dirichlet non-IID partitioning, scanner-shift
    │                        #  simulation, modality dropout        (Sec IV-B, IV-C)
    ├── dataset.py           # BraTS-format dataset loader + Eq. 2 masking
    ├── metrics.py           # Dice (Eq. 17) and HD95 (Eq. 18)
    └── federated.py         # Algorithm 1 (full FL training loop)
```


```bash
# FedTumorSeg (Full), severe non-IID, 40% modality dropout, 6 clients
python train.py --data-root /path/to/BraTS2020_Training \
    --kappa 0.1 --rho 0.4 --num-clients 6 \
    --use-film --use-confidence-weight

# FedAvg baseline, same regime
python train.py --data-root /path/to/BraTS2020_Training \
    --kappa 0.1 --rho 0.4 --num-clients 6 \
    --no-use-film --no-use-confidence-weight

# FedProx baseline
python train.py --data-root /path/to/BraTS2020_Training \
    --kappa 0.1 --rho 0.4 --num-clients 6 \
    --no-use-film --no-use-confidence-weight --proximal
```

`--kappa` sets the Dirichlet heterogeneity regime (`1.0`=IID, `0.5`=moderate,
`0.1`=severe, matching Table II), and `--rho` sets the fraction of clients
missing T1ce (`{0, 0.2, 0.4, 0.6}`, matching Figure 3).

## Getting the dataset

You need real BraTS 2020 data — this repo does not ship it. See the
`Dataset` section of the paper (Section IV-A) and the registration link:
https://www.med.upenn.edu/cbica/brats2020/registration.html

`train.py`'s `discover_patients()` assumes the original `HGG/` and `LGG/`
subfolder layout used by the official release. If you're using a flat-layout
mirror (e.g. some Kaggle copies), adjust `discover_patients()` to read the
accompanying `name_mapping.csv` instead — the rest of the pipeline is
layout-agnostic once you hand it two lists of patient IDs.

## Sanity-checking without real data

Every module's `if __name__ == "__main__":` block runs a synthetic
self-test that doesn't need real BraTS files:

```bash
python -m src.aggregation   # verifies weights are non-negative, sum to 1
python -m src.partition     # verifies HGG/LGG skew increases as kappa -> 0
python -m src.metrics       # verifies Dice=1.0, HD95=0.0 for identical volumes
python -m src.dataset       # verifies crop/pad, scanner shift, modality masking
python -m src.model         # verifies forward-pass shapes (needs torch)
python -m src.losses        # verifies the compound loss computes (needs torch)
```

The first three run with just `numpy`/`scipy` installed; the last three need
`torch` as well.
