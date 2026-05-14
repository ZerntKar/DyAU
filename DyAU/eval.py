from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import DyadicMotionDataset, collate_dyadic, move_to_device
from .metrics import evaluate_batch
from .model import DyAU
from .utils import average_dicts, resolve_device, tensor_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DyAU.")
    parser.add_argument("--config", required=True, help="Path to YAML or JSON config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--manifest", default="", help="Override test manifest.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = args.manifest or cfg.data.test_manifest or cfg.data.val_manifest
    if not manifest:
        raise ValueError("Provide --manifest or set data.test_manifest / data.val_manifest.")
    device = resolve_device(cfg.runtime.device)
    model = DyAU(cfg.model).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    loader = DataLoader(
        DyadicMotionDataset(manifest),
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_dyadic,
    )
    results = []
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            outputs = model(batch["audio_a"], batch["audio_b"], batch["mask"])
            results.append(tensor_items(evaluate_batch(outputs, batch, model.region_slices)))
    print(json.dumps(average_dicts(results), indent=2))


if __name__ == "__main__":
    main()
