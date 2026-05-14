from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn

from .modules import (
    AdaptiveMotionDecoder,
    DualStreamInteractionEncoder,
    InteractionQuerySummary,
    PseudoAUPrior,
    RegionAwareRefinement,
    SharedAudioEncoder,
    StructuredMotionEncoder,
    default_region_slices,
)


@dataclass
class HimaTalkConfig:
    audio_dim: int = 80
    motion_dim: int = 128
    hidden_dim: int = 256
    interaction_dim: int = 256
    lip_dim: int = 64
    exp_dim: int = 64
    state_dim: int = 64
    affect_dim: int = 64
    au_dim: int = 32
    region_control_dim: int = 32
    n_queries: int = 8
    n_heads: int = 4
    audio_layers: int = 3
    motion_layers: int = 2
    interaction_layers: int = 3
    decoder_layers: int = 3
    dropout: float = 0.1
    pseudo_au_regions: Tuple[str, ...] = ("brow", "eye", "cheek", "mouth", "neck")
    region_slices: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def normalized_region_slices(self) -> Dict[str, Tuple[int, int]]:
        if self.region_slices:
            return {name: (int(bounds[0]), int(bounds[1])) for name, bounds in self.region_slices.items()}
        return default_region_slices(self.motion_dim)


class HimaTalk(nn.Module):
    """Pseudo-AU guided dyadic speech-driven 3D facial motion generator."""

    def __init__(self, config: HimaTalkConfig) -> None:
        super().__init__()
        self.config = config
        self.audio_encoder = SharedAudioEncoder(
            config.audio_dim,
            config.hidden_dim,
            n_layers=config.audio_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        self.motion_encoder = StructuredMotionEncoder(
            config.motion_dim,
            config.hidden_dim,
            config.lip_dim,
            config.exp_dim,
            n_layers=config.motion_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        self.interaction_encoder = DualStreamInteractionEncoder(
            config.hidden_dim,
            config.interaction_dim,
            n_layers=config.interaction_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        self.summary = InteractionQuerySummary(
            config.interaction_dim,
            config.n_queries,
            config.n_heads,
            config.state_dim,
            config.affect_dim,
            dropout=config.dropout,
        )
        self.pseudo_au = PseudoAUPrior(
            config.interaction_dim,
            config.state_dim,
            config.affect_dim,
            config.au_dim,
            config.pseudo_au_regions,
            config.region_control_dim,
            dropout=config.dropout,
        )
        n_regions = len(config.pseudo_au_regions)
        self.decoder_a = AdaptiveMotionDecoder(
            config.hidden_dim,
            config.interaction_dim,
            config.state_dim,
            config.affect_dim,
            config.region_control_dim,
            n_regions,
            config.hidden_dim,
            config.motion_dim,
            subject_index=0,
            n_layers=config.decoder_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        self.decoder_b = AdaptiveMotionDecoder(
            config.hidden_dim,
            config.interaction_dim,
            config.state_dim,
            config.affect_dim,
            config.region_control_dim,
            n_regions,
            config.hidden_dim,
            config.motion_dim,
            subject_index=1,
            n_layers=config.decoder_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        self.refiner = RegionAwareRefinement(
            config.motion_dim,
            config.normalized_region_slices(),
            config.hidden_dim,
            config.interaction_dim,
            config.state_dim,
            config.affect_dim,
            config.region_control_dim,
            config.hidden_dim,
            dropout=config.dropout,
        )

    def forward(
        self,
        audio_a: Tensor,
        audio_b: Tensor,
        mask: Optional[Tensor] = None,
        motion_a: Optional[Tensor] = None,
        motion_b: Optional[Tensor] = None,
    ) -> Dict[str, object]:
        feat_a = self.audio_encoder(audio_a, mask)
        feat_b = self.audio_encoder(audio_b, mask)
        interaction = self.interaction_encoder(feat_a, feat_b, mask)
        summary = self.summary(interaction, mask)
        pseudo_au = self.pseudo_au(
            interaction,
            summary["global"],
            summary["state"],
            summary["affect"],
        )
        init_a = self.decoder_a(
            feat_a,
            summary["global"],
            summary["state"],
            summary["affect"],
            pseudo_au["regions"]["a"],
            mask,
        )
        init_b = self.decoder_b(
            feat_b,
            summary["global"],
            summary["state"],
            summary["affect"],
            pseudo_au["regions"]["b"],
            mask,
        )
        pred_a = self.refiner(
            init_a,
            feat_a,
            summary["global"],
            summary["state"],
            summary["affect"],
            pseudo_au["regions"]["a"],
        )
        pred_b = self.refiner(
            init_b,
            feat_b,
            summary["global"],
            summary["state"],
            summary["affect"],
            pseudo_au["regions"]["b"],
        )
        out: Dict[str, object] = {
            "motion": {"a": pred_a, "b": pred_b},
            "initial_motion": {"a": init_a, "b": init_b},
            "audio_features": {"a": feat_a, "b": feat_b},
            "interaction": interaction,
            "summary": summary,
            "pseudo_au": pseudo_au,
        }
        if motion_a is not None and motion_b is not None:
            out["motion_latents"] = {
                "gt": {
                    "a": self.motion_encoder(motion_a, mask),
                    "b": self.motion_encoder(motion_b, mask),
                },
                "pred": {
                    "a": self.motion_encoder(pred_a, mask),
                    "b": self.motion_encoder(pred_b, mask),
                },
            }
        return out

    @property
    def region_slices(self) -> Mapping[str, Tuple[int, int]]:
        return self.config.normalized_region_slices()


def build_model(config_dict: Mapping[str, object] | HimaTalkConfig) -> HimaTalk:
    if isinstance(config_dict, HimaTalkConfig):
        config = config_dict
    else:
        config = HimaTalkConfig(**dict(config_dict))
    return HimaTalk(config)
