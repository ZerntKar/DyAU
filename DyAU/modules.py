from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn


def _padding_mask(valid_mask: Optional[Tensor]) -> Optional[Tensor]:
    if valid_mask is None:
        return None
    return ~valid_mask.bool()


def build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    depth: int = 2,
    dropout: float = 0.0,
    activation: type[nn.Module] = nn.GELU,
) -> nn.Sequential:
    layers = []
    dim = in_dim
    for _ in range(max(depth - 1, 0)):
        layers.extend([nn.Linear(dim, hidden_dim), activation(), nn.Dropout(dropout)])
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class TemporalTransformer(nn.Module):
    """Small batch-first Transformer encoder for temporal motion/audio streams."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.enabled = n_layers > 0
        if not self.enabled:
            self.encoder = nn.Identity()
            self.norm = nn.LayerNorm(dim)
            return
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if not self.enabled:
            return self.norm(x)
        x = self.encoder(x, src_key_padding_mask=_padding_mask(mask))
        return self.norm(x)


class ConvBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv1d(dim, dim * 2, kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim * 2, dim, kernel_size, padding=padding),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.net[0](x)
        y = y.transpose(1, 2)
        y = self.net[1:](y).transpose(1, 2)
        return x + y


class SharedAudioEncoder(nn.Module):
    """Shared encoder E_aud for the two audio streams."""

    def __init__(
        self,
        audio_dim: int,
        hidden_dim: int,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        conv_kernel: int = 5,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(nn.LayerNorm(audio_dim), nn.Linear(audio_dim, hidden_dim))
        conv_layers = 0 if n_layers == 0 else 2
        self.conv = nn.ModuleList([ConvBlock(hidden_dim, conv_kernel, dropout) for _ in range(conv_layers)])
        self.temporal = TemporalTransformer(hidden_dim, n_heads, n_layers, dropout=dropout)

    def forward(self, audio: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.input_proj(audio)
        for block in self.conv:
            x = block(x)
        return self.temporal(x, mask)


class StructuredMotionEncoder(nn.Module):
    """Motion encoder E_mot with lip/expression latent projections."""

    def __init__(
        self,
        motion_dim: int,
        hidden_dim: int,
        lip_dim: int,
        exp_dim: int,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(nn.LayerNorm(motion_dim), nn.Linear(motion_dim, hidden_dim))
        self.temporal = TemporalTransformer(hidden_dim, n_heads, n_layers, dropout=dropout)
        self.lip_proj = nn.Linear(hidden_dim, lip_dim)
        self.exp_proj = nn.Linear(hidden_dim, exp_dim)

    def forward(self, motion: Tensor, mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        latent = self.temporal(self.input_proj(motion), mask)
        return {
            "latent": latent,
            "lip": self.lip_proj(latent),
            "exp": self.exp_proj(latent),
        }


class DualStreamInteractionEncoder(nn.Module):
    """E_int(F_a, F_b) for cross-subject temporal interaction modeling."""

    def __init__(
        self,
        hidden_dim: int,
        interaction_dim: int,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = hidden_dim * 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, interaction_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_dim, interaction_dim),
        )
        self.temporal = TemporalTransformer(interaction_dim, n_heads, n_layers, dropout=dropout)

    def forward(self, stream_a: Tensor, stream_b: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        fused = torch.cat([stream_a, stream_b], dim=-1)
        return self.temporal(self.fusion(fused), mask)


class InteractionQuerySummary(nn.Module):
    """Learnable-query cross-attention summary module."""

    def __init__(
        self,
        interaction_dim: int,
        n_queries: int,
        n_heads: int,
        state_dim: int,
        affect_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, interaction_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=interaction_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(interaction_dim)
        self.state_head = build_mlp(interaction_dim, interaction_dim, state_dim, depth=2, dropout=dropout)
        self.affect_head = build_mlp(interaction_dim, interaction_dim, affect_dim, depth=2, dropout=dropout)

    def forward(self, interaction: Tensor, mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        batch = interaction.shape[0]
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        summary, attn = self.attn(
            queries,
            interaction,
            interaction,
            key_padding_mask=_padding_mask(mask),
            need_weights=True,
            average_attn_weights=False,
        )
        summary = self.norm(summary)
        global_token = summary.mean(dim=1)
        return {
            "summary_tokens": summary,
            "attention": attn,
            "global": global_token,
            "state": self.state_head(global_token),
            "affect": self.affect_head(global_token),
        }


class PseudoAUPrior(nn.Module):
    """Maps dyadic interaction semantics to subject-wise pseudo-AU controls."""

    def __init__(
        self,
        interaction_dim: int,
        state_dim: int,
        affect_dim: int,
        au_dim: int,
        region_names: Iterable[str],
        region_control_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.region_names = tuple(region_names)
        global_in = interaction_dim + state_dim + affect_dim
        frame_in = interaction_dim + interaction_dim + state_dim + affect_dim
        self.global_net = build_mlp(global_in, interaction_dim, au_dim, depth=2, dropout=dropout, activation=nn.ReLU)
        self.frame_net = build_mlp(frame_in, interaction_dim, au_dim, depth=2, dropout=dropout, activation=nn.ReLU)
        self.subject_embed = nn.Embedding(2, au_dim)
        self.region_heads = nn.ModuleDict(
            {
                name: build_mlp(au_dim, au_dim, region_control_dim, depth=2, dropout=dropout, activation=nn.ReLU)
                for name in self.region_names
            }
        )

    def _subject_prior(
        self,
        base_seq: Tensor,
        subject_index: int,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        subject_id = torch.full((base_seq.shape[0],), subject_index, device=base_seq.device, dtype=torch.long)
        subject_bias = self.subject_embed(subject_id).unsqueeze(1)
        seq = base_seq + subject_bias
        controls = {name: head(seq) for name, head in self.region_heads.items()}
        return seq, controls

    def forward(
        self,
        interaction: Tensor,
        global_token: Tensor,
        state: Tensor,
        affect: Tensor,
    ) -> Dict[str, object]:
        batch, time, _ = interaction.shape
        global_in = torch.cat([global_token, state, affect], dim=-1)
        global_prior = self.global_net(global_in)
        expanded = torch.cat(
            [
                interaction,
                global_token.unsqueeze(1).expand(batch, time, -1),
                state.unsqueeze(1).expand(batch, time, -1),
                affect.unsqueeze(1).expand(batch, time, -1),
            ],
            dim=-1,
        )
        base_seq = self.frame_net(expanded)
        seq_a, controls_a = self._subject_prior(base_seq, 0)
        seq_b, controls_b = self._subject_prior(base_seq, 1)
        return {
            "global": global_prior,
            "seq": {"a": seq_a, "b": seq_b},
            "regions": {"a": controls_a, "b": controls_b},
        }


class AdaptiveMotionDecoder(nn.Module):
    """Conditional decoder D_mot for one target subject."""

    def __init__(
        self,
        audio_hidden_dim: int,
        interaction_dim: int,
        state_dim: int,
        affect_dim: int,
        region_control_dim: int,
        n_regions: int,
        hidden_dim: int,
        motion_dim: int,
        subject_index: int,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.subject_index = subject_index
        self.audio_proj = nn.Linear(audio_hidden_dim, hidden_dim)
        self.global_proj = nn.Linear(interaction_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.affect_proj = nn.Linear(affect_dim, hidden_dim)
        self.au_proj = nn.Linear(region_control_dim * n_regions, hidden_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )
        self.subject_embed = nn.Embedding(2, hidden_dim)
        self.temporal = TemporalTransformer(hidden_dim, n_heads, n_layers, dropout=dropout)
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, motion_dim))

    def forward(
        self,
        audio_feat: Tensor,
        global_token: Tensor,
        state: Tensor,
        affect: Tensor,
        region_controls: Mapping[str, Tensor],
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch, time, _ = audio_feat.shape
        audio_h = self.audio_proj(audio_feat)
        global_h = self.global_proj(global_token).unsqueeze(1).expand(batch, time, -1)
        state_h = self.state_proj(state).unsqueeze(1).expand(batch, time, -1)
        affect_h = self.affect_proj(affect).unsqueeze(1).expand(batch, time, -1)
        au = torch.cat([region_controls[name] for name in sorted(region_controls)], dim=-1)
        au_h = self.au_proj(au)
        gates = self.gate(torch.cat([audio_h, global_h, state_h, affect_h, au_h], dim=-1))
        cond = (
            gates[..., 0:1] * global_h
            + gates[..., 1:2] * state_h
            + gates[..., 2:3] * affect_h
            + gates[..., 3:4] * au_h
        )
        subject_id = torch.full((batch,), self.subject_index, device=audio_feat.device, dtype=torch.long)
        subject_h = self.subject_embed(subject_id).unsqueeze(1)
        x = audio_h + cond + subject_h
        return self.out(self.temporal(x, mask))


@dataclass(frozen=True)
class RegionSlice:
    name: str
    start: int
    end: int

    @property
    def width(self) -> int:
        return self.end - self.start


def default_region_slices(motion_dim: int) -> Dict[str, Tuple[int, int]]:
    names = ["mouth_jaw", "brow_eye", "cheek", "head_neck"]
    base = motion_dim // len(names)
    result: Dict[str, Tuple[int, int]] = {}
    start = 0
    for idx, name in enumerate(names):
        end = motion_dim if idx == len(names) - 1 else start + base
        result[name] = (start, end)
        start = end
    return result


class RegionResidualRefiner(nn.Module):
    def __init__(
        self,
        region_dim: int,
        audio_hidden_dim: int,
        interaction_dim: int,
        state_dim: int,
        affect_dim: int,
        control_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        context_dim = audio_hidden_dim + interaction_dim + state_dim + affect_dim + control_dim
        self.local_proj = nn.Linear(region_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, region_dim),
        )

    def forward(self, local_motion: Tensor, context: Tensor) -> Tensor:
        x = self.local_proj(local_motion) + self.context_proj(context)
        return self.net(x)


class RegionAwareRefinement(nn.Module):
    """Applies residual corrections to configured facial regions."""

    def __init__(
        self,
        motion_dim: int,
        region_slices: Mapping[str, Tuple[int, int]],
        audio_hidden_dim: int,
        interaction_dim: int,
        state_dim: int,
        affect_dim: int,
        control_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.motion_dim = motion_dim
        self.region_slices = {
            name: RegionSlice(name, int(bounds[0]), int(bounds[1]))
            for name, bounds in region_slices.items()
        }
        self.refiners = nn.ModuleDict(
            {
                name: RegionResidualRefiner(
                    spec.width,
                    audio_hidden_dim,
                    interaction_dim,
                    state_dim,
                    affect_dim,
                    control_dim,
                    hidden_dim,
                    dropout=dropout,
                )
                for name, spec in self.region_slices.items()
            }
        )

    def _control_for_region(self, region: str, controls: Mapping[str, Tensor]) -> Tensor:
        if region in controls:
            return controls[region]
        aliases = {
            "mouth": "mouth_jaw",
            "jaw": "mouth_jaw",
            "eye": "brow_eye",
            "brow": "brow_eye",
            "neck": "head_neck",
            "head": "head_neck",
        }
        alias = aliases.get(region)
        if alias in controls:
            return controls[alias]
        first = next(iter(controls.values()))
        return torch.zeros_like(first)

    def forward(
        self,
        motion: Tensor,
        audio_feat: Tensor,
        global_token: Tensor,
        state: Tensor,
        affect: Tensor,
        region_controls: Mapping[str, Tensor],
    ) -> Tensor:
        batch, time, _ = motion.shape
        residual = torch.zeros_like(motion)
        global_exp = global_token.unsqueeze(1).expand(batch, time, -1)
        state_exp = state.unsqueeze(1).expand(batch, time, -1)
        affect_exp = affect.unsqueeze(1).expand(batch, time, -1)
        for name, spec in self.region_slices.items():
            local = motion[..., spec.start : spec.end]
            control = self._control_for_region(name, region_controls)
            context = torch.cat([audio_feat, global_exp, state_exp, affect_exp, control], dim=-1)
            residual[..., spec.start : spec.end] = self.refiners[name](local, context)
        return motion + residual
