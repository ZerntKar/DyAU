from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from .losses import LossWeights
from .model import DyAUConfig


@dataclass
class DataConfig:
    train_manifest: str = ""
    val_manifest: str = ""
    test_manifest: str = ""
    batch_size: int = 4
    num_workers: int = 0


@dataclass
class OptimConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    grad_clip: float = 1.0
    log_every: int = 20
    save_every: int = 1


@dataclass
class RuntimeConfig:
    device: str = "auto"
    seed: int = 42
    output_dir: str = "runs/DyAU"


@dataclass
class ExperimentConfig:
    model: DyAUConfig = field(default_factory=DyAUConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    loss: LossWeights = field(default_factory=LossWeights)
    region_weights: Dict[str, float] = field(default_factory=dict)


def _update_dataclass(instance: Any, values: Mapping[str, Any]) -> Any:
    for key, value in values.items():
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, Mapping):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to load YAML configs.")
            raw = yaml.safe_load(f) or {}
        else:
            raw = json.load(f)
    cfg = ExperimentConfig()
    return _update_dataclass(cfg, raw)


def to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    return asdict(config)
