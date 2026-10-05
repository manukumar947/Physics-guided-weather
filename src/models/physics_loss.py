"""
physics_loss.py
===============
Physics-informed composite loss for Topo-SRNet training.

Loss Function
-------------
    L_total = L_MSE + lambda1 * L_lapse + lambda2 * L_mass

Components
----------

1. **L_MSE** — Standard pixel-wise mean squared error between the
   model's prediction and the bilinear baseline / pseudo-target.

2. **L_lapse** — Environmental Lapse Rate Penalty.
   Enforces the physical constraint that temperature decreases with
   increasing elevation (ELR ≈ 6.5°C / 1000 m).

   For each pixel pair (i, j) in the batch:
       ΔT   = T_pred[i] - T_pred[j]
       Δz   = elevation[i] - elevation[j]
   Penalty is activated when ΔT and Δz have the **same sign**
   (i.e., the predicted temperature *increases* with elevation),
   violating the lapse-rate physics:

       L_lapse = mean( ReLU(ΔT * Δz) )

   Implementation: uses shifted-pixel differences (horizontal +
   vertical neighbours) rather than all pairwise combinations to
   keep complexity O(H×W) instead of O((H×W)²).

3. **L_mass** — Sub-Grid Precipitation Mass Conservation.
   The total rainfall over the fine-resolution sub-grid must be
   consistent with the coarse-grid cell's rainfall amount:

       sum(fine_rain_pred) ≈ scale_factor * coarse_rain

   Loss:
       L_mass = mean( (sum_fine - sum_coarse_scaled)² )

   where ``sum_coarse_scaled`` is the coarse rainfall up-scaled to
   the fine grid's total area, computed per-patch per-batch.

Usage
-----
    from src.models.physics_loss import PhysicsLoss

    criterion = PhysicsLoss(lambda1=0.1, lambda2=0.05)
    loss, breakdown = criterion(
        pred      = model_output,          # (B, C_w, H, W)
        target    = bilinear_baseline,     # (B, C_w, H, W)
        coarse    = coarse_weather,        # (B, C_w, H, W)
        elevation = static[:, 0:1, :, :]  # (B, 1, H, W)  normalised
    )

Channel order assumption (matches generate_sample_data.py)
-----------------------------------------------------------
    Index 0 : Tmax
    Index 1 : Tmin
    Index 2 : Rainfall
    Index 3 : RH
    Index 4 : Wind
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Channel index constants
# ---------------------------------------------------------------------------

IDX_TMAX     = 0
IDX_TMIN     = 1
IDX_RAINFALL = 2
IDX_RH       = 3
IDX_WIND     = 4

TEMP_CHANNEL_INDICES = (IDX_TMAX, IDX_TMIN)


# ---------------------------------------------------------------------------
# Individual loss components
# ---------------------------------------------------------------------------

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Pixel-wise Mean Squared Error.

    Parameters
    ----------
    pred   : (B, C, H, W)
    target : (B, C, H, W)

    Returns
    -------
    scalar Tensor
    """
    return F.mse_loss(pred, target)


def lapse_rate_loss(
    pred_temp: torch.Tensor,
    elevation: torch.Tensor,
    lapse_rate_km: float = 6.5,
) -> torch.Tensor:
    """
    Environmental Lapse Rate Penalty.

    Penalises grid-cell pairs where the predicted temperature *increases*
    with elevation, violating the standard environmental lapse rate.

    Method: horizontal and vertical finite differences.
    For adjacent pixels (i, j):
        dT = T[i] - T[j]   (predicted temp gradient)
        dz = z[i] - z[j]   (elevation gradient)
    Violation ↔ dT * dz > 0  (same sign → temp rises with elevation).

    Penalty = ReLU(dT * dz)  averaged over all edges and batch.

    Note: ``elevation`` is assumed to be normalised to [0,1].  The
    lapse_rate_km parameter is informational only in this soft formulation;
    a hard constraint version would scale dz by the physical elevation range.

    Parameters
    ----------
    pred_temp  : Tensor (B, C_temp, H, W)
        Predicted temperature channels (Tmax and/or Tmin), normalised.
    elevation  : Tensor (B, 1, H, W)
        Elevation channel (normalised to [0, 1]).
    lapse_rate_km : float
        Standard environmental lapse rate in °C/km (default 6.5).
        Used for reference; the soft ReLU penalty is unit-agnostic.

    Returns
    -------
    scalar Tensor — mean lapse-rate violation penalty.
    """
    # Horizontal differences (left - right neighbours)
    dT_h  = pred_temp[:, :, :, :-1] - pred_temp[:, :, :, 1:]   # (B, C_t, H, W-1)
    dz_h  = elevation[:, :, :, :-1] - elevation[:, :, :, 1:]   # (B, 1,   H, W-1)

    # Vertical differences (top - bottom neighbours)
    dT_v  = pred_temp[:, :, :-1, :] - pred_temp[:, :, 1:, :]   # (B, C_t, H-1, W)
    dz_v  = elevation[:, :, :-1, :] - elevation[:, :, 1:, :]   # (B, 1,   H-1, W)

    # Violation: positive when temp and elevation gradients share the same sign
    # Sum over temp channels, broadcast elevation gradient
    violation_h = F.relu(dT_h * dz_h)   # shape (B, C_t, H, W-1)
    violation_v = F.relu(dT_v * dz_v)   # shape (B, C_t, H-1, W)

    loss = violation_h.mean() + violation_v.mean()
    return loss


def mass_conservation_loss(
    pred_rain: torch.Tensor,
    coarse_rain: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Sub-Grid Precipitation Mass Conservation Loss.

    The spatially-averaged fine-grid rainfall prediction must be
    consistent with the coarse-grid cell mean (i.e., the total
    precipitation mass is conserved across scales).

    Constraint:
        mean(pred_rain)  ≈  mean(coarse_rain_upsampled)

    Because both tensors are already on the same fine spatial grid
    (coarse has been bilinearly upsampled in the ETL), we simply
    penalise the difference of their spatial means per batch element.

    Parameters
    ----------
    pred_rain   : Tensor (B, 1, H, W)   fine-grid predicted rainfall (norm.)
    coarse_rain : Tensor (B, 1, H, W)   bilinearly upsampled coarse rainfall
    eps : float
        Numerical stability for relative scaling.

    Returns
    -------
    scalar Tensor — relative mass conservation error.
    """
    # Spatial mean over H × W for each batch element
    pred_mean   = pred_rain.mean(dim=(-1, -2))    # (B, 1)
    coarse_mean = coarse_rain.mean(dim=(-1, -2))  # (B, 1)

    # Clamp coarse to avoid negative values from bilinear interpolation
    coarse_mean = coarse_mean.clamp(min=0.0)

    # Relative conservation error (avoids scale sensitivity)
    denom = coarse_mean + eps
    rel_error = (pred_mean - coarse_mean) / denom  # (B, 1)

    return (rel_error ** 2).mean()


# ---------------------------------------------------------------------------
# Composite loss container
# ---------------------------------------------------------------------------

@dataclass
class LossBreakdown:
    """Named container for individual loss components."""
    total:  float
    mse:    float
    lapse:  float
    mass:   float

    def __repr__(self) -> str:
        return (
            f"Loss(total={self.total:.4f} | "
            f"mse={self.mse:.4f} | "
            f"lapse={self.lapse:.4f} | "
            f"mass={self.mass:.4f})"
        )


class PhysicsLoss(nn.Module):
    """
    Composite physics-informed loss function for Topo-SRNet.

    L_total = L_MSE  +  lambda1 * L_lapse  +  lambda2 * L_mass

    Parameters
    ----------
    lambda1 : float
        Weight on the lapse-rate penalty (default 0.1).
    lambda2 : float
        Weight on the mass-conservation penalty (default 0.05).
    lapse_rate_km : float
        Reference environmental lapse rate in °C/km (default 6.5).
    temp_channels : tuple[int, ...]
        Channel indices of temperature variables in the weather tensor.
        Default: (0, 1) for [Tmax, Tmin].
    rain_channel : int
        Channel index of the rainfall variable (default 2).
    """

    def __init__(
        self,
        lambda1: float = 0.1,
        lambda2: float = 0.05,
        lapse_rate_km: float = 6.5,
        temp_channels: Tuple[int, ...] = TEMP_CHANNEL_INDICES,
        rain_channel: int = IDX_RAINFALL,
    ) -> None:
        super().__init__()
        self.lambda1       = lambda1
        self.lambda2       = lambda2
        self.lapse_rate_km = lapse_rate_km
        self.temp_channels = list(temp_channels)
        self.rain_channel  = rain_channel

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        coarse: torch.Tensor,
        elevation: torch.Tensor,
    ) -> Tuple[torch.Tensor, LossBreakdown]:
        """
        Compute the composite physics loss.

        Parameters
        ----------
        pred      : (B, C_w, H, W)   Model output (normalised).
        target    : (B, C_w, H, W)   Pseudo-target (bilinear baseline or
                                     high-res observation if available).
        coarse    : (B, C_w, H, W)   Bilinearly upsampled coarse input.
        elevation : (B, 1, H, W)     Normalised elevation (static[:, 0:1]).

        Returns
        -------
        loss_total : scalar Tensor   (differentiable)
        breakdown  : LossBreakdown   (detached floats for logging)
        """
        # 1. Reconstruction loss
        L_mse = mse_loss(pred, target)

        # 2. Lapse rate loss (temperature channels only)
        pred_temp   = pred[:, self.temp_channels, :, :]   # (B, 2, H, W)
        L_lapse = lapse_rate_loss(pred_temp, elevation, self.lapse_rate_km)

        # 3. Mass conservation loss (rainfall channel only)
        pred_rain   = pred[:, self.rain_channel:self.rain_channel+1, :, :]
        coarse_rain = coarse[:, self.rain_channel:self.rain_channel+1, :, :]
        # Clamp predictions to non-negative (physically rain >= 0)
        pred_rain_clamped = pred_rain.clamp(min=0.0)
        L_mass = mass_conservation_loss(pred_rain_clamped, coarse_rain)

        # 4. Total weighted loss
        loss_total = L_mse + self.lambda1 * L_lapse + self.lambda2 * L_mass

        breakdown = LossBreakdown(
            total=loss_total.item(),
            mse=L_mse.item(),
            lapse=L_lapse.item(),
            mass=L_mass.item(),
        )

        return loss_total, breakdown

    def extra_repr(self) -> str:
        return (
            f"lambda1={self.lambda1}, lambda2={self.lambda2}, "
            f"lapse_rate_km={self.lapse_rate_km}"
        )


# ---------------------------------------------------------------------------
# Curriculum scheduling helper
# ---------------------------------------------------------------------------

class CurriculumLossScheduler:
    """
    Gradually ramp up physics loss weights over training epochs.

    Motivation: in early training the model's predictions are far from
    physical, so large physics penalties can destabilise learning.
    Ramping allows the MSE term to drive initial convergence, with
    physics constraints tightening as quality improves.

    Strategy: linear warm-up over ``warmup_epochs`` epochs.
        lambda_eff(epoch) = lambda_final * min(1, epoch / warmup_epochs)

    Parameters
    ----------
    criterion   : PhysicsLoss — will be mutated in-place.
    lambda1_final : float
    lambda2_final : float
    warmup_epochs : int
    """

    def __init__(
        self,
        criterion: PhysicsLoss,
        lambda1_final: float = 0.1,
        lambda2_final: float = 0.05,
        warmup_epochs: int = 10,
    ) -> None:
        self.criterion     = criterion
        self.l1_final      = lambda1_final
        self.l2_final      = lambda2_final
        self.warmup_epochs = warmup_epochs

    def step(self, epoch: int) -> None:
        """Update criterion weights for the given epoch (0-indexed)."""
        frac = min(1.0, (epoch + 1) / self.warmup_epochs)
        self.criterion.lambda1 = self.l1_final * frac
        self.criterion.lambda2 = self.l2_final * frac

    def current_weights(self) -> dict:
        return {
            "lambda1": self.criterion.lambda1,
            "lambda2": self.criterion.lambda2,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, C_w, C_s, H, W = 2, 5, 12, 64, 64

    # requires_grad=True so autograd can trace through the loss
    pred      = torch.randn(B, C_w, H, W, requires_grad=True)
    target    = torch.randn(B, C_w, H, W)
    # Use abs for coarse rainfall to avoid near-zero denominator in mass loss
    coarse    = torch.randn(B, C_w, H, W).abs()
    elevation = torch.rand(B, 1, H, W)          # normalised [0, 1]

    criterion = PhysicsLoss(lambda1=0.1, lambda2=0.05)
    loss, bd  = criterion(pred, target, coarse, elevation)

    print("PhysicsLoss smoke test")
    print(f"  {bd}")
    loss.backward()
    print("  [OK] Backward pass completed.")

    # Curriculum test
    scheduler = CurriculumLossScheduler(criterion, warmup_epochs=5)
    for ep in range(6):
        scheduler.step(ep)
        print(f"  Epoch {ep}: {scheduler.current_weights()}")
