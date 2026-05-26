from __future__ import annotations

import argparse
import json
import math
from typing import Dict, Mapping, Tuple

import torch
from torch.utils.data import DataLoader

from .config import load_config, to_dict
from .data import DyadicMotionDataset, collate_dyadic, move_to_device
from .losses import DyAULoss
from .metrics import evaluate_batch
from .model import DyAU
from .utils import count_parameters, ensure_dir, resolve_device, set_seed, tensor_items


def build_loader(
    manifest: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    max_frames: int,
    derive_missing_pseudo_au: bool,
    pseudo_au_dim: int,
    region_slices: Mapping[str, Tuple[int, int]],
) -> DataLoader:
    dataset = DyadicMotionDataset(
        manifest,
        max_frames=max_frames,
        derive_missing_pseudo_au=derive_missing_pseudo_au,
        pseudo_au_dim=pseudo_au_dim,
        region_slices=region_slices,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_dyadic,
        drop_last=False,
    )


def run_validation(
    model: DyAU,
    criterion: DyAULoss,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    losses = []
    metrics = []
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            outputs = model(
                batch["audio_a"],
                batch["audio_b"],
                batch["mask"],
                batch["motion_a"],
                batch["motion_b"],
            )
            loss_dict = criterion(outputs, batch)
            losses.append(tensor_items(loss_dict))
            metrics.append(tensor_items(evaluate_batch(outputs, batch, model.region_slices)))
    result: Dict[str, float] = {}
    for prefix, values in (("val_loss", losses), ("val_metric", metrics)):
        keys = values[0].keys() if values else []
        for key in keys:
            result[f"{prefix}/{key}"] = sum(item[key] for item in values) / len(values)
    return result


def build_scheduler(optimizer: torch.optim.Optimizer, steps_per_epoch: int, epochs: int, warmup_epochs: int):
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(1, steps_per_epoch * warmup_epochs)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DyAU.")
    parser.add_argument("--config", required=True, help="Path to YAML or JSON config.")
    parser.add_argument("--resume", default="", help="Optional checkpoint to resume from.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.runtime.seed)
    device = resolve_device(cfg.runtime.device)
    output_dir = ensure_dir(cfg.runtime.output_dir)

    model = DyAU(cfg.model).to(device)
    criterion = DyAULoss(model.region_slices, cfg.loss, cfg.region_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    train_loader = build_loader(
        cfg.data.train_manifest,
        cfg.data.batch_size,
        cfg.data.num_workers,
        shuffle=True,
        max_frames=cfg.data.max_frames,
        derive_missing_pseudo_au=cfg.data.derive_missing_pseudo_au,
        pseudo_au_dim=cfg.model.au_dim,
        region_slices=model.region_slices,
    )
    scheduler = build_scheduler(
        optimizer,
        steps_per_epoch=len(train_loader),
        epochs=cfg.optim.epochs,
        warmup_epochs=cfg.optim.warmup_epochs,
    )
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    val_loader = None
    if cfg.data.val_manifest:
        val_loader = build_loader(
            cfg.data.val_manifest,
            cfg.data.batch_size,
            cfg.data.num_workers,
            shuffle=False,
            max_frames=cfg.data.max_frames,
            derive_missing_pseudo_au=cfg.data.derive_missing_pseudo_au,
            pseudo_au_dim=cfg.model.au_dim,
            region_slices=model.region_slices,
        )

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(to_dict(cfg), f, indent=2)

    print(f"Device: {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    global_step = 0
    for epoch in range(start_epoch, cfg.optim.epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["audio_a"],
                batch["audio_b"],
                batch["mask"],
                batch["motion_a"],
                batch["motion_b"],
            )
            loss_dict = criterion(outputs, batch)
            loss_dict["total"].backward()
            if cfg.optim.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            if global_step % cfg.optim.log_every == 0:
                items = tensor_items(loss_dict)
                lr = optimizer.param_groups[0]["lr"]
                msg = " ".join(f"{key}={value:.4f}" for key, value in items.items())
                print(f"epoch={epoch} step={global_step} lr={lr:.6g} {msg}")

        summary: Dict[str, float] = {}
        if val_loader is not None:
            summary = run_validation(model, criterion, val_loader, device)
            msg = " ".join(f"{key}={value:.4f}" for key, value in summary.items())
            print(f"epoch={epoch} validation {msg}")

        if (epoch + 1) % cfg.optim.save_every == 0:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": to_dict(cfg),
                "validation": summary,
            }
            torch.save(ckpt, output_dir / f"checkpoint_{epoch:04d}.pt")
            torch.save(ckpt, output_dir / "latest.pt")


if __name__ == "__main__":
    main()
