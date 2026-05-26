from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor

from .data import derive_pseudo_au_from_motion
from .losses import masked_l1, temporal_difference, temporal_mask


PAPER_REGIONS = ("mouth_jaw", "brow_eye", "cheek", "head_neck")


def _masked_std(x: Tensor, mask: Optional[Tensor]) -> Tensor:
    if mask is None:
        return x.std(dim=(0, 1), unbiased=False).mean()
    weights = mask.to(x.dtype).unsqueeze(-1)
    denom = weights.sum().clamp_min(1.0)
    mean = (x * weights).sum(dim=(0, 1), keepdim=True) / denom
    var = (((x - mean) ** 2) * weights).sum(dim=(0, 1)) / denom
    return var.sqrt().mean()


def _masked_temporal_std(x: Tensor, mask: Optional[Tensor]) -> Tensor:
    if mask is None:
        return x.std(dim=1, unbiased=False).mean()
    weights = mask.to(x.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (x * weights).sum(dim=1, keepdim=True) / denom
    var = (((x - mean) ** 2) * weights).sum(dim=1) / denom.squeeze(1)
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


def _region_error(
    pred_a: Tensor,
    pred_b: Tensor,
    gt_a: Tensor,
    gt_b: Tensor,
    mask: Optional[Tensor],
    bounds: Tuple[int, int],
) -> Tensor:
    start, end = bounds
    return 0.5 * (
        masked_l1(pred_a[..., start:end], gt_a[..., start:end], mask)
        + masked_l1(pred_b[..., start:end], gt_b[..., start:end], mask)
    )


def facial_dynamic_difference(
    pred_a: Tensor,
    pred_b: Tensor,
    gt_a: Tensor,
    gt_b: Tensor,
    region_slices: Mapping[str, Tuple[int, int]],
    mask: Optional[Tensor],
) -> Tensor:
    """Paper-defined FDD: regional temporal-amplitude std discrepancy."""

    values = []
    for name, bounds in region_slices.items():
        start, end = bounds
        pred_std = 0.5 * (
            _masked_temporal_std(pred_a[..., start:end], mask)
            + _masked_temporal_std(pred_b[..., start:end], mask)
        )
        gt_std = 0.5 * (
            _masked_temporal_std(gt_a[..., start:end], mask)
            + _masked_temporal_std(gt_b[..., start:end], mask)
        )
        values.append((pred_std - gt_std).abs())
    if not values:
        pred_std = 0.5 * (_masked_temporal_std(pred_a, mask) + _masked_temporal_std(pred_b, mask))
        gt_std = 0.5 * (_masked_temporal_std(gt_a, mask) + _masked_temporal_std(gt_b, mask))
        return (pred_std - gt_std).abs()
    return torch.stack(values).mean()


def pseudo_au_error(
    pred_a: Tensor,
    pred_b: Tensor,
    batch: Mapping[str, Tensor],
    region_slices: Mapping[str, Tuple[int, int]],
    mask: Optional[Tensor],
    au_dim: int,
) -> Tensor:
    """PAU-E via the same motion-statistics fallback extractor used by the loader."""

    pred_pau_a = derive_pseudo_au_from_motion(pred_a, region_slices, au_dim)
    pred_pau_b = derive_pseudo_au_from_motion(pred_b, region_slices, au_dim)
    target_a = batch.get("pseudo_au_a")
    target_b = batch.get("pseudo_au_b")
    if target_a is None:
        target_a = derive_pseudo_au_from_motion(batch["motion_a"], region_slices, au_dim)
    if target_b is None:
        target_b = derive_pseudo_au_from_motion(batch["motion_b"], region_slices, au_dim)
    return 0.5 * (
        masked_l1(pred_pau_a, target_a, mask)
        + masked_l1(pred_pau_b, target_b, mask)
    )


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

    region_metrics = {
        name: _region_error(pred_a, pred_b, gt_a, gt_b, mask, region_slices[name])
        for name in PAPER_REGIONS
        if name in region_slices
    }
    lve = region_metrics.get("mouth_jaw", mve)

    delta_mask = temporal_mask(mask, 1)
    acc_mask = temporal_mask(mask, 2)
    vel = 0.5 * (
        masked_l1(temporal_difference(pred_a, 1), temporal_difference(gt_a, 1), delta_mask)
        + masked_l1(temporal_difference(pred_b, 1), temporal_difference(gt_b, 1), delta_mask)
    )
    acc = 0.5 * (
        masked_l1(temporal_difference(pred_a, 2), temporal_difference(gt_a, 2), acc_mask)
        + masked_l1(temporal_difference(pred_b, 2), temporal_difference(gt_b, 2), acc_mask)
    )
    fdd = facial_dynamic_difference(pred_a, pred_b, gt_a, gt_b, region_slices, mask)

    if "pseudo_au_a" in batch:
        au_dim = batch["pseudo_au_a"].shape[-1]
    elif "pseudo_au" in outputs:
        au_dim = outputs["pseudo_au"]["seq"]["a"].shape[-1]
    else:
        au_dim = 32
    pau_e = pseudo_au_error(pred_a, pred_b, batch, region_slices, mask, au_dim)

    pred_delta_a = temporal_difference(pred_a, 1)
    pred_delta_b = temporal_difference(pred_b, 1)
    gt_delta_a = temporal_difference(gt_a, 1)
    gt_delta_b = temporal_difference(gt_b, 1)
    corr_pred = pearson_corr(pred_delta_a, pred_delta_b, delta_mask)
    corr_gt = pearson_corr(gt_delta_a, gt_delta_b, delta_mask)
    rpcc = (corr_pred - corr_gt).abs()
    ic = corr_pred
    metrics = {
        "mve": mve,
        "lve": lve,
        "fdd": fdd,
        "pau_e": pau_e,
        "vel": vel,
        "acc": acc,
        "rpcc": rpcc,
        "ic": ic,
    }
    metrics.update(region_metrics)
    return metrics
