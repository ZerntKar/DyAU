# DyAU

Reference PyTorch implementation of **DyAU: Interaction-Aware Regional Priors for Dyadic Speech-Driven 3D Facial Motion Generation**.

This repository implements the model backbone and training objective described
in the paper. It is intended as a clean reference implementation for method
inspection and adaptation. Full paper reproduction still requires the same
external preprocessing pipeline, dataset splits, baseline implementations, and
hardware setup used in the experiments.

## What Is Implemented

- Shared dual-stream audio feature projection for two subjects.
- Structured motion encoder with lip and expression latent supervision.
- Interaction query summary with learnable query tokens and cross-attention.
- Four Pseudo-AU groups: Mouth/Jaw, Brow-Eye, Cheek, and Head-Neck.
- Subject-wise Pseudo-AU priors and region-level control heads.
- Conditional temporal motion decoding for both subjects.
- Region-aware residual MLP refinement.
- Reconstruction, auxiliary structured consistency, Pseudo-AU, region, and
  dynamic losses with first- and second-order temporal terms.
- Paper-aligned evaluation outputs: `mve`, `mouth_jaw`, `brow_eye`, `cheek`,
  `head_neck`, paper-defined `fdd`, `pau_e`, and `rpcc`.

## Layout

```text
code/
  DyAU/
    model.py       # DyAU model assembly
    modules.py     # encoders, query summary, Pseudo-AU, decoder, refiner
    losses.py      # multi-objective training loss
    metrics.py     # paper-style MVE/regional/FDD/PAU-E/rPCC metrics
    data.py        # NPZ dataset, padding collate, fallback Pseudo-AU labels
    train.py       # single-device training entry point
    eval.py        # evaluation entry point
  configs/
    DyAU.yaml      # paper-style default config
```

## Data Format

Training data is loaded from a manifest file. Each line is either a relative
path to an `.npz` file or a JSON line with a `path` field.

Each `.npz` sample must contain pre-extracted, frame-aligned features:

```text
audio_a:  [T, audio_dim]
audio_b:  [T, audio_dim]
motion_a: [T, motion_dim]
motion_b: [T, motion_dim]
```

Optional weak Pseudo-AU labels:

```text
pseudo_au_a: [T, au_dim]
pseudo_au_b: [T, au_dim]
```

The paper uses a pre-trained speech encoder to extract frame-level speech
features. This released code expects those speech features to be extracted
offline and stored as `audio_a` / `audio_b`; the in-repo `SharedAudioEncoder`
is the shared projection and temporal modeling layer on top of those features.

The paper uses OpenFace 2.0 AU estimates when reliable RGB face tracking is
available. This repository does not vendor OpenFace or run it internally. If
`pseudo_au_a` / `pseudo_au_b` are present in the NPZ files, they are used
directly. If they are missing, the loader derives fallback weak labels from
normalized local motion variation in the four semantic regions, matching the
paper's motion-statistics fallback path.

The exact 3D motion representation is dataset-dependent. For FLAME, 3DMM, or
blendshape coefficients, update `model.region_slices` in the config so the
region-aware loss/refinement uses the four paper regions: `mouth_jaw`,
`brow_eye`, `cheek`, and `head_neck`.
