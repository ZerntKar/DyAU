from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def masked_l1(pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    loss = (pred - target).abs()
    if mask is None:
        return loss.mean()
    weights = mask.to(loss.dtype).unsqueeze(-1)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0) / pred.shape[-1]


def temporal_difference(x: Tensor, order: int = 1) -> Tensor:
    if order == 1:
        return x[:, 1:] - x[:, :-1]
    if order == 2:
        return x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    raise ValueError(f"Unsupported temporal order: {order}")


def temporal_mask(mask: Optional[Tensor], order: int) -> Optional[Tensor]:
    if mask is None:
        return None
    if order == 1:
        return mask[:, 1:] & mask[:, :-1]
    if order == 2:
        return mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
    raise ValueError(f"Unsupported temporal order: {order}")


@dataclass
class LossWeights:
    rec: float = 1.0
    str: float = 0.2
    au: float = 0.1
    reg: float = 0.5
    temp: float = 0.2
    acc: float = 0.05


class HimaTalkLoss(nn.Module):
    """Multi-objective loss from the method section."""

    def __init__(
        self,
        region_slices: Mapping[str, Tuple[int, int]],
        weights: LossWeights | None = None,
        region_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        super().__init__()
        self.region_slices = {name: (int(s), int(e)) for name, (s, e) in region_slices.items()}
        self.weights = weights or LossWeights()
        self.region_weights = dict(region_weights or {})

    def _reconstruction(self, outputs: Mapping[str, object], batch: Mapping[str, Tensor]) -> Tensor:
        mask = batch.get("mask")
        motion = outputs["motion"]
        return masked_l1(motion["a"], batch["motion_a"], mask) + masked_l1(motion["b"], batch["motion_b"], mask)

    def _structured(self, outputs: Mapping[str, object], mask: Optional[Tensor]) -> Tensor:
        if "motion_latents" not in outputs:
            return torch.tensor(0.0, device=next(iter(outputs["motion"].values())).device)
        latents = outputs["motion_latents"]
        total = 0.0
        for subject in ("a", "b"):
            pred = latents["pred"][subject]
            gt = latents["gt"][subject]
            total = total + masked_l1(pred["lip"], gt["lip"].detach(), mask)
            total = total + masked_l1(pred["exp"], gt["exp"].detach(), mask)
        return total

    def _pseudo_au(self, outputs: Mapping[str, object], batch: Mapping[str, Tensor]) -> Tensor:
        mask = batch.get("mask")
        pseudo = outputs["pseudo_au"]["seq"]
        total = None
        for subject, key in (("a", "pseudo_au_a"), ("b", "pseudo_au_b")):
            if key in batch:
                loss = masked_l1(pseudo[subject], batch[key], mask)
                total = loss if total is None else total + loss
        if total is None:
            device = next(iter(outputs["motion"].values())).device
            return torch.tensor(0.0, device=device)
        return total

    def _region(self, outputs: Mapping[str, object], batch: Mapping[str, Tensor]) -> Tensor:
        mask = batch.get("mask")
        motion = outputs["motion"]
        total = 0.0
        for name, (start, end) in self.region_slices.items():
            weight = self.region_weights.get(name, 1.0)
            total = total + weight * masked_l1(
                motion["a"][..., start:end],
                batch["motion_a"][..., start:end],
                mask,
            )
            total = total + weight * masked_l1(
                motion["b"][..., start:end],
                batch["motion_b"][..., start:end],
                mask,
            )
        return total

    def _temporal(self, outputs: Mapping[str, object], batch: Mapping[str, Tensor], order: int) -> Tensor:
        mask = temporal_mask(batch.get("mask"), order)
        motion = outputs["motion"]
        total = 0.0
        for subject, key in (("a", "motion_a"), ("b", "motion_b")):
            pred_d = temporal_difference(motion[subject], order)
            gt_d = temporal_difference(batch[key], order)
            total = total + masked_l1(pred_d, gt_d, mask)
        return total

    def forward(self, outputs: Mapping[str, object], batch: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        mask = batch.get("mask")
        rec = self._reconstruction(outputs, batch)
        structured = self._structured(outputs, mask)
        au = self._pseudo_au(outputs, batch)
        region = self._region(outputs, batch)
        temp = self._temporal(outputs, batch, order=1)
        acc = self._temporal(outputs, batch, order=2)
        total = (
            self.weights.rec * rec
            + self.weights.str * structured
            + self.weights.au * au
            + self.weights.reg * region
            + self.weights.temp * temp
            + self.weights.acc * acc
        )
        return {
            "total": total,
            "rec": rec.detach(),
            "str": structured.detach(),
            "au": au.detach(),
            "reg": region.detach(),
            "temp": temp.detach(),
            "acc": acc.detach(),
        }
