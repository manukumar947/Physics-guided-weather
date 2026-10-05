"""
train.py
========
Lightweight PyTorch training loop for Topo-SRNet.

Pipeline
--------
1. Load .npz patch files from data/processed/patches/ (Phase 1 output).
2. Split into train / validation sets.
3. Train with the composite PhysicsLoss (MSE + lapse + mass conservation).
4. Checkpoint best model by validation MSE.
5. Log per-epoch metrics to logs/train_log.csv.

Directory layout
----------------
    checkpoints/
        best_model.pth        — best validation checkpoint
        last_model.pth        — most recent epoch checkpoint
    logs/
        train_log.csv         — per-epoch metric table

Usage
-----
    python src/models/train.py [options]

    Key options:
        --patch_dir   data/processed/patches
        --out_dir     checkpoints
        --epochs      50
        --batch_size  8
        --lr          1e-4
        --feat_ch     64
        --n_rrdb      8
        --lambda1     0.1
        --lambda2     0.05
        --val_frac    0.15
        --device      cuda   (or cpu)
        --resume      checkpoints/last_model.pth  (optional)

Dependencies
------------
    torch, numpy, pathlib, argparse, csv, time
"""

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from topo_srnet import TopoSRNet, model_summary
from physics_loss import PhysicsLoss, CurriculumLossScheduler, LossBreakdown


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PatchDataset(Dataset):
    """
    PyTorch Dataset that loads pre-computed .npz patch files.

    Each .npz contains:
        "coarse" : (C_w, H, W) float32  — upsampled weather input
        "static" : (C_s, H, W) float32  — static geophysical features

    The *target* is defined as the coarse input itself (self-supervised
    refinement).  When real high-resolution observations are available
    (Phase 3+), replace ``target`` with the observed HR field.

    Parameters
    ----------
    patch_dir : Path
        Directory containing patch_XXXXX.npz files.
    augment   : bool
        If True, apply random horizontal/vertical flips for data augmentation.
    """

    def __init__(self, patch_dir: Path, augment: bool = False) -> None:
        self.files   = sorted(patch_dir.glob("*.npz"))
        self.augment = augment

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No .npz patch files found in {patch_dir}.\n"
                "Run src/etl/preprocessor.py first."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = np.load(self.files[idx])
        coarse = torch.from_numpy(data["coarse"].astype(np.float32))  # (C_w, H, W)
        static = torch.from_numpy(data["static"].astype(np.float32))  # (C_s, H, W)

        # Self-supervised target: coarse input (learn refinement delta)
        # In production this would be paired with HR observations
        target = coarse.clone()

        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                coarse = torch.flip(coarse, dims=[-1])
                static = torch.flip(static, dims=[-1])
                target = torch.flip(target, dims=[-1])
            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                coarse = torch.flip(coarse, dims=[-2])
                static = torch.flip(static, dims=[-2])
                target = torch.flip(target, dims=[-2])

        return {"coarse": coarse, "static": static, "target": target}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    rain_ch: int = 2,
) -> Dict[str, float]:
    """
    Compute per-variable diagnostics for logging.

    Returns
    -------
    dict with keys: mae, rmse, rain_rmse, tmax_mae, tmin_mae
    """
    with torch.no_grad():
        diff  = pred - target
        mae   = diff.abs().mean().item()
        rmse  = diff.pow(2).mean().sqrt().item()

        rain_diff = diff[:, rain_ch, :, :]
        rain_rmse = rain_diff.pow(2).mean().sqrt().item()

        tmax_mae = diff[:, 0, :, :].abs().mean().item()
        tmin_mae = diff[:, 1, :, :].abs().mean().item()

    return {
        "mae":       mae,
        "rmse":      rmse,
        "rain_rmse": rain_rmse,
        "tmax_mae":  tmax_mae,
        "tmin_mae":  tmin_mae,
    }


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: TopoSRNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: PhysicsLoss,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Run one full training epoch. Returns averaged metrics."""
    model.train()
    agg = {
        "loss": 0.0, "mse": 0.0, "lapse": 0.0, "mass": 0.0,
        "mae": 0.0, "rmse": 0.0,
    }
    n_batches = len(loader)

    for batch in loader:
        coarse    = batch["coarse"].to(device, non_blocking=True)   # (B, C_w, H, W)
        static    = batch["static"].to(device, non_blocking=True)   # (B, C_s, H, W)
        target    = batch["target"].to(device, non_blocking=True)   # (B, C_w, H, W)
        elevation = static[:, 0:1, :, :]                            # (B, 1, H, W)

        optimizer.zero_grad(set_to_none=True)

        pred = model(coarse, static)                                # (B, C_w, H, W)

        loss, bd = criterion(pred, target, coarse, elevation)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        metrics = compute_metrics(pred.detach(), target.detach())
        agg["loss"]  += bd.total
        agg["mse"]   += bd.mse
        agg["lapse"] += bd.lapse
        agg["mass"]  += bd.mass
        agg["mae"]   += metrics["mae"]
        agg["rmse"]  += metrics["rmse"]

    return {k: v / n_batches for k, v in agg.items()}


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: TopoSRNet,
    loader: DataLoader,
    criterion: PhysicsLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Run one validation pass. Returns averaged metrics."""
    model.eval()
    agg = {
        "loss": 0.0, "mse": 0.0, "lapse": 0.0, "mass": 0.0,
        "mae": 0.0, "rmse": 0.0, "rain_rmse": 0.0,
        "tmax_mae": 0.0, "tmin_mae": 0.0,
    }
    n_batches = len(loader)

    for batch in loader:
        coarse    = batch["coarse"].to(device)
        static    = batch["static"].to(device)
        target    = batch["target"].to(device)
        elevation = static[:, 0:1, :, :]

        pred = model(coarse, static)
        _, bd = criterion(pred, target, coarse, elevation)
        metrics = compute_metrics(pred, target)

        agg["loss"]      += bd.total
        agg["mse"]       += bd.mse
        agg["lapse"]     += bd.lapse
        agg["mass"]      += bd.mass
        agg["mae"]       += metrics["mae"]
        agg["rmse"]      += metrics["rmse"]
        agg["rain_rmse"] += metrics["rain_rmse"]
        agg["tmax_mae"]  += metrics["tmax_mae"]
        agg["tmin_mae"]  += metrics["tmin_mae"]

    return {k: v / n_batches for k, v in agg.items()}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: TopoSRNet,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    val_loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":      epoch,
            "val_loss":   val_loss,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict() if scheduler else None,
            "args":       vars(args),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: TopoSRNet,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", 0), ckpt.get("val_loss", float("inf"))


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

class CSVLogger:
    """Append per-epoch metrics to a CSV file."""

    FIELDS = [
        "epoch", "lr",
        "train_loss", "train_mse", "train_lapse", "train_mass",
        "train_mae", "train_rmse",
        "val_loss", "val_mse", "val_lapse", "val_mass",
        "val_mae", "val_rmse", "val_rain_rmse", "val_tmax_mae", "val_tmin_mae",
        "epoch_secs", "lambda1", "lambda2",
    ]

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def log(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS, extrasaction="ignore")
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # ── Device ────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[*] Device: {device}")

    # ── Dataset & DataLoaders ─────────────────────────────────────────────
    patch_dir = Path(args.patch_dir)
    full_ds   = PatchDataset(patch_dir, augment=True)
    n_total   = len(full_ds)
    n_val     = max(1, int(n_total * args.val_frac))
    n_train   = n_total - n_val

    print(f"[*] Dataset: {n_total} patches  "
          f"(train={n_train}, val={n_val})")

    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    # Disable augmentation on val split
    val_ds.dataset.augment = False

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    # Infer channel counts from first patch
    sample = full_ds[0]
    n_weather_ch = sample["coarse"].shape[0]
    n_static_ch  = sample["static"].shape[0]

    model = TopoSRNet(
        n_weather_ch=n_weather_ch,
        n_static_ch=n_static_ch,
        feat_ch=args.feat_ch,
        n_rrdb=args.n_rrdb,
        growth=args.growth,
        dropout_p=args.dropout_p,
    ).to(device)

    model_summary(model, patch_size=sample["coarse"].shape[-1])

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = PhysicsLoss(
        lambda1=args.lambda1,
        lambda2=args.lambda2,
    )
    loss_scheduler = CurriculumLossScheduler(
        criterion,
        lambda1_final=args.lambda1,
        lambda2_final=args.lambda2,
        warmup_epochs=args.warmup_epochs,
    )

    # ── Optimizer & LR Scheduler ──────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    out_dir = Path(args.out_dir)

    if args.resume and Path(args.resume).exists():
        start_epoch, best_val_loss = load_checkpoint(
            Path(args.resume), model, optimizer, lr_scheduler
        )
        print(f"[*] Resumed from {args.resume} (epoch {start_epoch}, "
              f"best_val={best_val_loss:.4f})")

    # ── Logger ────────────────────────────────────────────────────────────
    logger = CSVLogger(Path("logs") / "train_log.csv")

    # ── Training loop ─────────────────────────────────────────────────────
    print("=" * 60)
    print(f"  Starting training for {args.epochs} epochs")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Curriculum: ramp physics weights
        loss_scheduler.step(epoch)

        # Train
        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.grad_clip
        )

        # Validate
        val_m = validate(model, val_loader, criterion, device)

        # LR step
        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]

        epoch_secs = time.time() - t0

        # Print summary
        print(
            f"Epoch [{epoch+1:>3}/{args.epochs}] "
            f"lr={current_lr:.2e} | "
            f"train_loss={train_m['loss']:.4f} "
            f"(mse={train_m['mse']:.4f} lapse={train_m['lapse']:.4f} mass={train_m['mass']:.4f}) | "
            f"val_loss={val_m['loss']:.4f} "
            f"tmax_mae={val_m.get('tmax_mae', 0):.4f} "
            f"rain_rmse={val_m.get('rain_rmse', 0):.4f} | "
            f"{epoch_secs:.1f}s"
        )

        # Log to CSV
        logger.log({
            "epoch":        epoch + 1,
            "lr":           current_lr,
            "train_loss":   train_m["loss"],
            "train_mse":    train_m["mse"],
            "train_lapse":  train_m["lapse"],
            "train_mass":   train_m["mass"],
            "train_mae":    train_m["mae"],
            "train_rmse":   train_m["rmse"],
            "val_loss":     val_m["loss"],
            "val_mse":      val_m["mse"],
            "val_lapse":    val_m["lapse"],
            "val_mass":     val_m["mass"],
            "val_mae":      val_m["mae"],
            "val_rmse":     val_m["rmse"],
            "val_rain_rmse": val_m.get("rain_rmse", 0),
            "val_tmax_mae": val_m.get("tmax_mae", 0),
            "val_tmin_mae": val_m.get("tmin_mae", 0),
            "epoch_secs":   round(epoch_secs, 2),
            "lambda1":      criterion.lambda1,
            "lambda2":      criterion.lambda2,
        })

        # Save last checkpoint
        save_checkpoint(
            out_dir / "last_model.pth", model, optimizer, lr_scheduler,
            epoch + 1, val_m["loss"], args
        )

        # Save best checkpoint
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            save_checkpoint(
                out_dir / "best_model.pth", model, optimizer, lr_scheduler,
                epoch + 1, best_val_loss, args
            )
            print(f"  [*] New best model saved (val_loss={best_val_loss:.4f})")

    print("=" * 60)
    print(f"[OK] Training complete. Best val_loss = {best_val_loss:.4f}")
    print(f"[OK] Best model   -> {out_dir / 'best_model.pth'}")
    print(f"[OK] Training log -> logs/train_log.csv")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Topo-SRNet with PhysicsLoss")

    # Data
    p.add_argument("--patch_dir",   default="data/processed/patches",
                   help="Directory of .npz patch files")
    p.add_argument("--out_dir",     default="checkpoints",
                   help="Directory to save model checkpoints")
    p.add_argument("--val_frac",    type=float, default=0.15,
                   help="Fraction of patches to use for validation")

    # Model
    p.add_argument("--feat_ch",     type=int,   default=64)
    p.add_argument("--n_rrdb",      type=int,   default=8)
    p.add_argument("--growth",      type=int,   default=32)
    p.add_argument("--dropout_p",   type=float, default=0.1)

    # Training
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--num_workers", type=int,   default=0,
                   help="DataLoader workers (0=main process, safe for Windows)")
    p.add_argument("--seed",        type=int,   default=42)

    # Physics loss
    p.add_argument("--lambda1",         type=float, default=0.1,
                   help="Lapse rate penalty weight")
    p.add_argument("--lambda2",         type=float, default=0.05,
                   help="Mass conservation penalty weight")
    p.add_argument("--warmup_epochs",   type=int,   default=10,
                   help="Epochs to ramp physics weights from 0 to final")

    # Misc
    p.add_argument("--device",  default="auto",
                   help="'auto', 'cuda', or 'cpu'")
    p.add_argument("--resume",  default=None,
                   help="Path to checkpoint to resume from")

    return p.parse_args()


if __name__ == "__main__":
    # Needed for Windows multiprocessing compatibility
    import multiprocessing
    multiprocessing.freeze_support()

    # Reproducibility
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)
