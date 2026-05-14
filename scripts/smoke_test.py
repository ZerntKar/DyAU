from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from DyAU.losses import DyAULoss, LossWeights
from DyAU.model import DyAU, DyAUConfig


def main() -> None:
    torch.manual_seed(7)
    cfg = DyAUConfig(
        audio_dim=16,
        motion_dim=32,
        hidden_dim=64,
        interaction_dim=64,
        lip_dim=16,
        exp_dim=16,
        state_dim=16,
        affect_dim=16,
        au_dim=12,
        region_control_dim=8,
        n_queries=4,
        n_heads=4,
        audio_layers=1,
        motion_layers=1,
        interaction_layers=1,
        decoder_layers=1,
        dropout=0.0,
    )
    model = DyAU(cfg)
    batch_size, time = 2, 12
    batch = {
        "audio_a": torch.randn(batch_size, time, cfg.audio_dim),
        "audio_b": torch.randn(batch_size, time, cfg.audio_dim),
        "motion_a": torch.randn(batch_size, time, cfg.motion_dim),
        "motion_b": torch.randn(batch_size, time, cfg.motion_dim),
        "pseudo_au_a": torch.randn(batch_size, time, cfg.au_dim),
        "pseudo_au_b": torch.randn(batch_size, time, cfg.au_dim),
        "mask": torch.ones(batch_size, time, dtype=torch.bool),
    }
    outputs = model(
        batch["audio_a"],
        batch["audio_b"],
        batch["mask"],
        batch["motion_a"],
        batch["motion_b"],
    )
    criterion = DyAULoss(model.region_slices, LossWeights())
    losses = criterion(outputs, batch)
    losses["total"].backward()
    print("Smoke test passed.")
    print(f"motion_a: {tuple(outputs['motion']['a'].shape)}")
    print(f"motion_b: {tuple(outputs['motion']['b'].shape)}")
    print(f"loss: {float(losses['total']):.4f}")


if __name__ == "__main__":
    main()
