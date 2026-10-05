"""
src/models — Model package for Topo-Downscale.

Modules
-------
topo_srnet   : Topo-SRNet dual-branch RRDB super-resolution model
physics_loss : Composite physics-informed loss (MSE + lapse + mass conservation)
train        : PyTorch training loop with checkpoint and CSV logging
"""

from .topo_srnet import TopoSRNet, model_summary
from .physics_loss import PhysicsLoss, CurriculumLossScheduler, LossBreakdown

__all__ = [
    "TopoSRNet",
    "model_summary",
    "PhysicsLoss",
    "CurriculumLossScheduler",
    "LossBreakdown",
]
