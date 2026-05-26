from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor

from .losses import masked_l1, temporal_difference


def _masked_std(x: Tensor, mask: Optional[Tensor]) -> Tensor:
    if mask is None:
        return x.std(dim=(0, 1)).mean()
    weights = mask.to(x.dtype).unsqueeze(-1)
    denom = weights.sum().clamp_min(1.0)
    mean = (x * weights).sum(dim=(0, 1), keepdim=True) / denom
    var = (((x - mean) ** 2) * weights).sum(dim=(0, 1)) / denom
    return var.sqrt().mean()


def pearson_corr(x: Tensor, y: Tensor, mask: Optional[Tensor] = None, eps: float = 1e-8) -> Tensor:
    if mask is not None:
        x = x[mask]
        y = y[mask]
    x = x.flatten()
    y = y.flatten()
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).mean() / (x.std(unbiased=False) * y.std(unbiased=False) + eps)


@torch.no_grad()
def evaluate_batch(
    outputs: Mapping[str, object],
    batch: Mapping[str, Tensor],
    region_slices: Mapping[str, Tuple[int, int]],
) -> Dict[str, Tensor]:
    mask = batch.get("mask")
    motion = outputs["motion"]
    pred_a, pred_b = motion["a"], motion["b"]
    gt_a, gt_b = batch["motion_a"], batch["motion_b"]
    mve = 0.5 * (masked_l1(pred_a, gt_a, mask) + masked_l1(pred_b, gt_b, mask))

    mouth = region_slices.get("mouth_jaw") or region_slices.get("mouth")
    if mouth is not None:
        s, e = mouth
        lve = 0.5 * (
            masked_l1(pred_a[..., s:e], gt_a[..., s:e], mask)
            + masked_l1(pred_b[..., s:e], gt_b[..., s:e], mask)
        )
    else:
        lve = mve

    vel = 0.5 * (
        masked_l1(temporal_difference(pred_a, 1), temporal_difference(gt_a, 1), None)
        + masked_l1(temporal_difference(pred_b, 1), temporal_difference(gt_b, 1), None)
    )
    acc = 0.5 * (
        masked_l1(temporal_difference(pred_a, 2), temporal_difference(gt_a, 2), None)
        + masked_l1(temporal_difference(pred_b, 2), temporal_difference(gt_b, 2), None)
    )
    fdd = vel + acc

    pred_delta_a = temporal_difference(pred_a, 1)
    pred_delta_b = temporal_difference(pred_b, 1)
    gt_delta_a = temporal_difference(gt_a, 1)
    gt_delta_b = temporal_difference(gt_b, 1)
    corr_pred = pearson_corr(pred_delta_a, pred_delta_b)
    corr_gt = pearson_corr(gt_delta_a, gt_delta_b)
    rpcc = (corr_pred - corr_gt).abs()
    ic = corr_pred
    return {
        "mve": mve,
        "lve": lve,
        "fdd": fdd,
        "vel": vel,
        "acc": acc,
        "rpcc": rpcc,
        "ic": ic,
    }
