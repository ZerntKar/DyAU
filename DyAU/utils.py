from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tensor_items(values: Mapping[str, Tensor]) -> Dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in values.items()}


def average_dicts(items: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    total: Dict[str, float] = {}
    count = 0
    for item in items:
        count += 1
        for key, value in item.items():
            total[key] = total.get(key, 0.0) + float(value)
    if count == 0:
        return {}
    return {key: value / count for key, value in total.items()}
