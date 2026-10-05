"""
topo_srnet.py
=============
Topo-SRNet: Topography-Conditioned Meteorological Super-Resolution Network
--------------------------------------------------------------------------

Architecture overview
---------------------

    Coarse weather (B, C_w, H, W)         Static topo (B, C_s, H, W)
           |                                       |
    [WeatherEncoder]                       [TopoEncoder]
           |                                       |
    (B, F, H, W)                           (B, F, H, W)
           |_________________  __________________|
                             |  |
                      [FeatureFusion]   (concat + CBAM attention gate)
                             |
                     (B, 2F, H, W)
                             |
                   [ResidualBlocks] x N_BLOCKS
                             |
                     (B, 2F, H, W)
                             |
                   [OutputHead]  (1x1 conv -> C_w channels)
                             |
                   (B, C_w, H, W)  <-- refined downscaled output

Key design decisions
--------------------
* **Dual-branch encoding**: Weather and topo features are encoded
  independently before fusion. This lets the network learn separate
  inductive biases for transient vs. static fields.
* **CBAM (Channel + Spatial Attention)**: Applied post-fusion to let the
  network focus on topographically significant pixels (ridge crests,
  valley bottoms) for temperature and precipitation adjustment.
* **Residual-in-Residual Dense Blocks (RRDB-lite)**: Each block contains
  two dense layers + a residual skip, giving deep feature reuse without
  vanishing gradients.
* **No sub-pixel upsampling**: The ETL bilinear upsampling already maps
  coarse grids to the fine DEM resolution. Topo-SRNet therefore performs
  *refinement* super-resolution (analogous to SRGAN with bicubic init),
  focusing on physically consistent detail injection rather than raw
  magnification.

Input / Output Contract
-----------------------
    weather_input : torch.Tensor  shape (B, C_w, H, W)  float32
        Bilinearly upsampled, normalised coarse weather variables.
        Default channel order: [Tmax, Tmin, Rainfall, RH, Wind]

    static_input  : torch.Tensor  shape (B, C_s, H, W)  float32
        Normalised static geophysical priors.
        Default channels: [elevation, slope, twi, sin_aspect, cos_aspect,
                           lulc_0 … lulc_6]

    output        : torch.Tensor  shape (B, C_w, H, W)  float32
        Topographically-refined, high-resolution weather predictions.

Usage
-----
    from src.models.topo_srnet import TopoSRNet

    model = TopoSRNet(n_weather_ch=5, n_static_ch=12)
    out   = model(weather_tensor, static_tensor)   # (B, 5, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm → LeakyReLU block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
        slope: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list = [
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=not use_bn)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(slope, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# CBAM: Convolutional Block Attention Module (Woo et al., 2018)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention gate."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        avg = self.mlp(self.avg_pool(x)).view(B, C, 1, 1)
        mx  = self.mlp(self.max_pool(x)).view(B, C, 1, 1)
        return self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    """Spatial attention gate using channel-wise statistics."""

    def __init__(self, kernel: int = 7) -> None:
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv2d(2, 1, kernel, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        feat = torch.cat([avg, mx], dim=1)
        return self.sigmoid(self.conv(feat))


class CBAM(nn.Module):
    """Full CBAM: channel attention followed by spatial attention."""

    def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# ---------------------------------------------------------------------------
# Dense Residual Block (RRDB-lite)
# ---------------------------------------------------------------------------

class DenseLayer(nn.Module):
    """Single dense layer with growth-rate feature concatenation."""

    def __init__(self, in_ch: int, growth: int = 32) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, growth, 3, padding=1, bias=True)
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.act(self.conv(x))], dim=1)


class ResidualDenseBlock(nn.Module):
    """
    RRDB-lite: 3 dense layers + a 1x1 bottleneck projection + residual skip.
    Growth rate default = 32 to keep memory reasonable at 64×64 patches.
    """

    def __init__(self, in_ch: int, growth: int = 32, residual_scale: float = 0.2) -> None:
        super().__init__()
        self.residual_scale = residual_scale

        self.d1 = DenseLayer(in_ch,          growth)
        self.d2 = DenseLayer(in_ch + growth, growth)
        self.d3 = DenseLayer(in_ch + 2 * growth, growth)

        # Bottleneck back to in_ch
        self.bottleneck = nn.Conv2d(in_ch + 3 * growth, in_ch, 1, bias=True)
        nn.init.kaiming_normal_(self.bottleneck.weight, a=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.d3(self.d2(self.d1(x)))
        out = self.bottleneck(out)
        return x + self.residual_scale * out


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block: 3 × ResidualDenseBlock + outer skip."""

    def __init__(self, channels: int, growth: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth)
        self.rdb2 = ResidualDenseBlock(channels, growth)
        self.rdb3 = ResidualDenseBlock(channels, growth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return x + 0.2 * out


# ---------------------------------------------------------------------------
# Encoder branches
# ---------------------------------------------------------------------------

class WeatherEncoder(nn.Module):
    """
    Encodes the bilinearly-upsampled coarse weather tensor into a feature map.

    3-layer progressive convolution to expand receptive field while
    increasing channel depth.
    """

    def __init__(self, in_ch: int, feat_ch: int) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            ConvBnRelu(in_ch,        feat_ch // 2, kernel=3, padding=1),
            ConvBnRelu(feat_ch // 2, feat_ch,      kernel=3, padding=1),
            ConvBnRelu(feat_ch,      feat_ch,      kernel=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)


class TopoEncoder(nn.Module):
    """
    Encodes the static geophysical conditioning tensor (DEM + LULC).

    Uses dilated convolutions (rates 1, 2, 4) to capture multi-scale
    topographic context without reducing spatial resolution.
    """

    def __init__(self, in_ch: int, feat_ch: int) -> None:
        super().__init__()
        mid = feat_ch // 2
        self.conv1 = ConvBnRelu(in_ch, mid, padding=1)                         # dilation=1
        self.conv2 = nn.Sequential(                                              # dilation=2
            nn.Conv2d(mid, mid, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(                                              # dilation=4
            nn.Conv2d(mid, feat_ch, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


# ---------------------------------------------------------------------------
# Fusion module
# ---------------------------------------------------------------------------

class FeatureFusion(nn.Module):
    """
    Fuse weather and topo feature maps via concatenation + CBAM.

    The CBAM gate allows the network to learn which channels/spatial
    locations matter most for topographic correction.
    """

    def __init__(self, feat_ch: int) -> None:
        super().__init__()
        fused_ch = feat_ch * 2
        self.proj = ConvBnRelu(fused_ch, fused_ch, kernel=1, padding=0)
        self.cbam = CBAM(fused_ch, reduction=8)

    def forward(self, w_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([w_feat, t_feat], dim=1)   # (B, 2F, H, W)
        fused = self.proj(fused)
        fused = fused * self.cbam(fused) + fused     # residual attention
        return fused


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TopoSRNet(nn.Module):
    """
    Topo-SRNet: Topography-Conditioned Meteorological Super-Resolution Network.

    Parameters
    ----------
    n_weather_ch : int
        Number of input weather channels (default 5: Tmax, Tmin, Rainfall, RH, Wind).
    n_static_ch : int
        Number of static conditioning channels (default 12: elevation, slope,
        twi, sin_aspect, cos_aspect, lulc × 7).
    feat_ch : int
        Base feature channel count (default 64).
    n_rrdb : int
        Number of RRDB residual blocks in the trunk (default 8).
    growth : int
        Dense layer growth rate inside each RRDB (default 32).
    dropout_p : float
        Dropout probability applied before the output head (default 0.1).
        Provides mild regularisation; set to 0 for deterministic inference.
    """

    def __init__(
        self,
        n_weather_ch: int = 5,
        n_static_ch: int = 12,
        feat_ch: int = 64,
        n_rrdb: int = 8,
        growth: int = 32,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_weather_ch = n_weather_ch
        self.n_static_ch  = n_static_ch

        # ── Encoders ─────────────────────────────────────────────────────────
        self.weather_enc = WeatherEncoder(n_weather_ch, feat_ch)
        self.topo_enc    = TopoEncoder(n_static_ch,  feat_ch)

        # ── Fusion ───────────────────────────────────────────────────────────
        self.fusion = FeatureFusion(feat_ch)

        fused_ch = feat_ch * 2

        # ── RRDB trunk ───────────────────────────────────────────────────────
        self.trunk = nn.Sequential(*[RRDB(fused_ch, growth) for _ in range(n_rrdb)])

        # ── Post-trunk conv (stabilises gradients before output head) ─────────
        self.post_trunk = ConvBnRelu(fused_ch, fused_ch, kernel=3, padding=1)

        # ── Output head ──────────────────────────────────────────────────────
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.head = nn.Sequential(
            nn.Conv2d(fused_ch, feat_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_ch, n_weather_ch, 1),  # 1×1 to map channels
        )

        # Weight initialisation
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        weather: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        weather : Tensor, shape (B, C_w, H, W)
            Bilinearly upsampled, normalised coarse weather grid.
        static  : Tensor, shape (B, C_s, H, W)
            Static geophysical conditioning tensor.

        Returns
        -------
        Tensor, shape (B, C_w, H, W)
            Topographically refined high-resolution weather predictions.
        """
        # Encode each modality
        w_feat = self.weather_enc(weather)   # (B, F, H, W)
        t_feat = self.topo_enc(static)       # (B, F, H, W)

        # Fuse
        fused = self.fusion(w_feat, t_feat)  # (B, 2F, H, W)

        # Residual trunk
        trunk_in = fused
        trunk_out = self.trunk(trunk_in)
        trunk_out = self.post_trunk(trunk_out)

        # Global residual connection (encourages learning the *delta* from
        # the bilinear baseline, not the full field from scratch)
        trunk_out = trunk_out + trunk_in

        # Output
        out = self.dropout(trunk_out)
        out = self.head(out)                 # (B, C_w, H, W)

        # Residual: output = bilinear_baseline + learned_correction
        out = out + weather
        return out

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick model summary helper
# ---------------------------------------------------------------------------

def model_summary(model: TopoSRNet, patch_size: int = 64) -> None:
    """Print a compact parameter and shape summary."""
    device = next(model.parameters()).device
    B = 2
    w = torch.randn(B, model.n_weather_ch, patch_size, patch_size, device=device)
    s = torch.randn(B, model.n_static_ch,  patch_size, patch_size, device=device)

    model.eval()
    with torch.no_grad():
        out = model(w, s)

    total = model.count_parameters()
    print("=" * 56)
    print(f"  Model        : Topo-SRNet")
    print(f"  Weather ch   : {model.n_weather_ch}")
    print(f"  Static  ch   : {model.n_static_ch}")
    print(f"  Input shape  : {tuple(w.shape)}")
    print(f"  Output shape : {tuple(out.shape)}")
    print(f"  Parameters   : {total:,}  ({total/1e6:.2f} M)")
    print("=" * 56)


# ---------------------------------------------------------------------------
# Entry point — quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = TopoSRNet(
        n_weather_ch=5,
        n_static_ch=12,
        feat_ch=64,
        n_rrdb=8,
        growth=32,
    ).to(device)

    model_summary(model, patch_size=64)

    # Forward + backward smoke test
    model.train()
    B, H, W = 2, 64, 64
    weather = torch.randn(B, 5,  H, W, device=device)
    static  = torch.randn(B, 12, H, W, device=device)
    target  = torch.randn(B, 5,  H, W, device=device)

    out  = model(weather, static)
    loss = F.mse_loss(out, target)
    loss.backward()

    print(f"[OK] Forward + backward pass | loss={loss.item():.4f}")
    print(f"[OK] Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
