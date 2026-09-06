"""
Modality-Aware Attention U-Net (Section III-B of the paper).

Implements:
  - A standard 3D Attention U-Net backbone (Oktay et al., 2018), feature
    channels [64, 128, 256, 512, 1024] as specified in Section III-G.
  - Input-level modality masking: x_masked = x * m  (Eq. 2, Section III-B.2)
  - Bottleneck FiLM conditioning: h_mod = h * gamma(m) + delta(m)  (Eq. 3-5)

Note on notation: the paper's Eq. 3 uses beta(m) for the FiLM shift function.
In the LaTeX source this was renamed to delta(m) to avoid clashing with the
aggregation hyperparameter beta (Eq. 9) and the Focal Tversky beta_FT (Eq. 8).
We follow that renaming here (`delta_mlp` below).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_MODALITIES = 4  # T1, T1ce, T2, FLAIR
NUM_CLASSES = 4      # background, WT, TC, ET (as class indices 0-3, see Sec III-A)


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two 3x3x3 conv-IN-ReLU layers."""
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_ch, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_ch, affine=True),
        nn.ReLU(inplace=True),
    )


class AttentionGate(nn.Module):
    """Attention gate from Oktay et al. (2018), applied on skip connections."""

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.theta_g = nn.Conv3d(gate_ch, inter_ch, kernel_size=1)
        self.phi_x = nn.Conv3d(skip_ch, inter_ch, kernel_size=1)
        self.psi = nn.Conv3d(inter_ch, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.theta_g(gate)
        x = self.phi_x(skip)
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="trilinear", align_corners=False)
        psi = self.relu(g + x)
        alpha = self.sigmoid(self.psi(psi))
        return skip * alpha


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (Eq. 3-5).

        h_mod = h * gamma(m) + delta(m)

    gamma(m) and delta(m) are produced by small MLPs that map the 4-dim
    modality presence vector m in {0,1}^4 to per-channel scale/shift vectors.
    """

    def __init__(self, num_channels: int, num_modalities: int = NUM_MODALITIES, hidden: int = 64):
        super().__init__()
        self.gamma_mlp = nn.Sequential(
            nn.Linear(num_modalities, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_channels),
        )
        self.delta_mlp = nn.Sequential(
            nn.Linear(num_modalities, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_channels),
        )
        # Initialize gamma branch to output ~1 and delta branch to output ~0
        # at the start of training, so FiLM starts close to an identity map.
        nn.init.zeros_(self.gamma_mlp[-1].weight)
        nn.init.ones_(self.gamma_mlp[-1].bias)
        nn.init.zeros_(self.delta_mlp[-1].weight)
        nn.init.zeros_(self.delta_mlp[-1].bias)

    def forward(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # h: (B, C, D, H, W); m: (B, num_modalities)
        gamma = self.gamma_mlp(m).view(m.shape[0], -1, 1, 1, 1)
        delta = self.delta_mlp(m).view(m.shape[0], -1, 1, 1, 1)
        return h * gamma + delta


class AttentionUNetFiLM(nn.Module):
    """
    3D Attention U-Net with:
      (i)  input-level modality masking (applied by the caller / dataset, see
           src/dataset.py `apply_modality_mask`), and
      (ii) FiLM conditioning at the bottleneck (this module).

    Feature channels follow Section III-G: [64, 128, 256, 512, 1024].
    """

    def __init__(
        self,
        in_channels: int = NUM_MODALITIES,
        out_channels: int = NUM_CLASSES,
        channels=(64, 128, 256, 512, 1024),
        use_film: bool = True,
    ):
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.use_film = use_film

        self.enc1 = conv_block(in_channels, c1)
        self.enc2 = conv_block(c1, c2)
        self.enc3 = conv_block(c2, c3)
        self.enc4 = conv_block(c3, c4)
        self.bottleneck = conv_block(c4, c5)

        self.pool = nn.MaxPool3d(2)

        # FiLM conditioning applied at the bottleneck feature map (c5 channels).
        # Toggled off for the "w/o Modality Conditioning" ablation (Table III)
        # and for baselines that don't use modality-aware conditioning.
        self.film = FiLMLayer(num_channels=c5) if use_film else None

        self.up4 = nn.ConvTranspose3d(c5, c4, kernel_size=2, stride=2)
        self.att4 = AttentionGate(c4, c4, c4 // 2)
        self.dec4 = conv_block(c4 * 2, c4)

        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.att3 = AttentionGate(c3, c3, c3 // 2)
        self.dec3 = conv_block(c3 * 2, c3)

        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(c2, c2, c2 // 2)
        self.dec2 = conv_block(c2 * 2, c2)

        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.att1 = AttentionGate(c1, c1, c1 // 2)
        self.dec1 = conv_block(c1 * 2, c1)

        self.out_conv = nn.Conv3d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 4, D, H, W) input MRI volume. Callers should already have
           applied input-level modality masking (x * m broadcast over
           channels) as in Eq. 2 -- see dataset.apply_modality_mask.
        m: (B, 4) modality presence vector, values in {0, 1}.
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        # --- Modality conditioning (Eq. 3), if enabled ---
        if self.use_film:
            b = self.film(b, m)

        d4 = self.up4(b)
        e4_att = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))

        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        return self.out_conv(d1)  # (B, num_classes, D, H, W) logits


if __name__ == "__main__":
    # Quick shape sanity check on a tiny synthetic volume (not the paper's
    # 128^3 resolution, just enough to verify the forward pass wiring).
    model = AttentionUNetFiLM()
    x = torch.randn(2, 4, 32, 32, 32)
    m = torch.tensor([[1, 1, 1, 1], [1, 0, 1, 1]], dtype=torch.float32)
    y = model(x, m)
    print("Output shape:", y.shape)  # expect (2, 4, 32, 32, 32)
