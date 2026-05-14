from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


REQUIRED_KEYS = ("audio_a", "audio_b", "motion_a", "motion_b")
OPTIONAL_KEYS = ("pseudo_au_a", "pseudo_au_b")


def _load_manifest(manifest_path: str | Path) -> List[Path]:
    manifest = Path(manifest_path)
    items: List[Path] = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                record = json.loads(line)
                path = Path(record["path"])
            else:
                path = Path(line)
            if not path.is_absolute():
                path = manifest.parent / path
            items.append(path)
    return items


class DyadicMotionDataset(Dataset):
    """Loads dyadic audio/motion samples from NPZ files.

    Each NPZ file is expected to contain:
      audio_a: [T, audio_dim]
      audio_b: [T, audio_dim]
      motion_a: [T, motion_dim]
      motion_b: [T, motion_dim]

    Optional pseudo-AU labels:
      pseudo_au_a: [T, au_dim]
      pseudo_au_b: [T, au_dim]
    """

    def __init__(self, manifest_path: str | Path) -> None:
        self.paths = _load_manifest(manifest_path)
        if not self.paths:
            raise ValueError(f"No samples found in manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, Tensor | str]:
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as item:
            missing = [key for key in REQUIRED_KEYS if key not in item]
            if missing:
                raise KeyError(f"{path} is missing keys: {missing}")
            sample: Dict[str, Tensor | str] = {
                key: torch.from_numpy(item[key]).float()
                for key in REQUIRED_KEYS
            }
            for key in OPTIONAL_KEYS:
                if key in item:
                    sample[key] = torch.from_numpy(item[key]).float()
            sample["path"] = str(path)
            return sample


def _pad_sequence(values: List[Tensor], max_len: int) -> Tensor:
    feature_shape = values[0].shape[1:]
    padded = values[0].new_zeros((len(values), max_len, *feature_shape))
    for idx, value in enumerate(values):
        padded[idx, : value.shape[0]] = value
    return padded


def collate_dyadic(batch: List[Mapping[str, Tensor | str]]) -> Dict[str, Tensor | List[str]]:
    max_len = max(int(item["audio_a"].shape[0]) for item in batch)
    output: Dict[str, Tensor | List[str]] = {}
    for key in REQUIRED_KEYS:
        output[key] = _pad_sequence([item[key] for item in batch], max_len)  # type: ignore[list-item]
    for key in OPTIONAL_KEYS:
        if all(key in item for item in batch):
            output[key] = _pad_sequence([item[key] for item in batch], max_len)  # type: ignore[list-item]
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for idx, item in enumerate(batch):
        mask[idx, : item["audio_a"].shape[0]] = True  # type: ignore[index]
    output["mask"] = mask
    output["path"] = [str(item["path"]) for item in batch]
    return output


def move_to_device(batch: Mapping[str, object], device: torch.device | str) -> Dict[str, object]:
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, Tensor) else value
    return moved
