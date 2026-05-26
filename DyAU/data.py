from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


REQUIRED_KEYS = ("audio_a", "audio_b", "motion_a", "motion_b")
OPTIONAL_KEYS = ("pseudo_au_a", "pseudo_au_b")
PSEUDO_AU_GROUPS = ("mouth_jaw", "brow_eye", "cheek", "head_neck")


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


def _default_region_slices(motion_dim: int) -> Dict[str, Tuple[int, int]]:
    base = motion_dim // len(PSEUDO_AU_GROUPS)
    result: Dict[str, Tuple[int, int]] = {}
    start = 0
    for idx, name in enumerate(PSEUDO_AU_GROUPS):
        end = motion_dim if idx == len(PSEUDO_AU_GROUPS) - 1 else start + base
        result[name] = (start, end)
        start = end
    return result


def derive_pseudo_au_from_motion(
    motion: Tensor,
    region_slices: Mapping[str, Tuple[int, int]],
    au_dim: int,
) -> Tensor:
    """Fallback Pseudo-AU weak labels from local motion variation.

    The current paper defines four Pseudo-AU groups. If OpenFace AU estimates
    are not available in an NPZ file, this function approximates weak labels by
    normalized absolute temporal variation in each semantic region, then repeats
    each group score across its assigned channels.
    """

    slices = dict(region_slices) if region_slices else _default_region_slices(motion.shape[-1])
    channels_per_group = max(1, au_dim // len(PSEUDO_AU_GROUPS))
    velocity = torch.zeros_like(motion)
    if motion.dim() == 2:
        velocity[1:] = (motion[1:] - motion[:-1]).abs()
    else:
        velocity[..., 1:, :] = (motion[..., 1:, :] - motion[..., :-1, :]).abs()
    channels = []
    for group in PSEUDO_AU_GROUPS:
        start, end = slices[group]
        score = velocity[..., start:end].mean(dim=-1, keepdim=True)
        score = score / score.max().clamp_min(1e-6)
        channels.append(score.repeat_interleave(channels_per_group, dim=-1))
    pseudo = torch.cat(channels, dim=-1)
    if pseudo.shape[-1] < au_dim:
        pad = pseudo.new_zeros(*pseudo.shape[:-1], au_dim - pseudo.shape[-1])
        pseudo = torch.cat([pseudo, pad], dim=-1)
    return pseudo[..., :au_dim]


class DyadicMotionDataset(Dataset):
    """Loads dyadic audio/motion samples from NPZ files.

    Each NPZ file is expected to contain:
      audio_a: [T, audio_dim]
      audio_b: [T, audio_dim]
      motion_a: [T, motion_dim]
      motion_b: [T, motion_dim]

    Optional weak Pseudo-AU labels:
      pseudo_au_a: [T, au_dim]
      pseudo_au_b: [T, au_dim]
    """

    def __init__(
        self,
        manifest_path: str | Path,
        max_frames: int = 120,
        derive_missing_pseudo_au: bool = True,
        pseudo_au_dim: int = 32,
        region_slices: Optional[Mapping[str, Tuple[int, int]]] = None,
    ) -> None:
        self.paths = _load_manifest(manifest_path)
        if not self.paths:
            raise ValueError(f"No samples found in manifest: {manifest_path}")
        self.max_frames = max_frames
        self.derive_missing_pseudo_au = derive_missing_pseudo_au
        self.pseudo_au_dim = pseudo_au_dim
        self.region_slices = dict(region_slices or {})

    def __len__(self) -> int:
        return len(self.paths)

    def _crop(self, sample: Dict[str, Tensor | str]) -> Dict[str, Tensor | str]:
        if not self.max_frames:
            return sample
        time = int(sample["audio_a"].shape[0])  # type: ignore[index]
        if time <= self.max_frames:
            return sample
        return {
            key: value[: self.max_frames] if isinstance(value, Tensor) else value
            for key, value in sample.items()
        }

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
            sample = self._crop(sample)
            if self.derive_missing_pseudo_au:
                if "pseudo_au_a" not in sample:
                    sample["pseudo_au_a"] = derive_pseudo_au_from_motion(
                        sample["motion_a"],  # type: ignore[arg-type]
                        self.region_slices,
                        self.pseudo_au_dim,
                    )
                if "pseudo_au_b" not in sample:
                    sample["pseudo_au_b"] = derive_pseudo_au_from_motion(
                        sample["motion_b"],  # type: ignore[arg-type]
                        self.region_slices,
                        self.pseudo_au_dim,
                    )
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
