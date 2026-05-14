# DyAU

Reference PyTorch implementation of **Pseudo-AU Guided Dyadic Speech-Driven
3D Facial Motion Generation**.

The code mirrors the method section of the paper:

- shared dual-stream audio encoding for two speakers;
- structured motion encoding with lip and expression latent spaces;
- interaction query summary for speaking dominance, listener feedback, role
  switching, and affective cues;
- subject-wise Pseudo-AU priors as region-level control signals;
- conditional motion decoding for both subjects;
- region-aware residual refinement;
- reconstruction, structured consistency, pseudo-AU, region, velocity, and
  acceleration losses.

## Layout

```text
code/
  DyAU/
    model.py       # DyAU model assembly
    modules.py     # encoders, query summary, pseudo-AU, decoder, refiner
    losses.py      # multi-objective training loss
    metrics.py     # MVE/LVE/FDD/RPCC/IC-style metrics
    data.py        # NPZ dataset and padding collate function
    train.py       # training entry point
    eval.py        # evaluation entry point
  configs/
    DyAU.yaml      # full-size default config
    DyAU_toy.yaml  # tiny config for local smoke training
  scripts/
    make_toy_data.py
    smoke_test.py
```

## Data Format

Training data is loaded from a manifest file. Each line is either a relative
path to an `.npz` file or a JSON line with a `path` field.

Each `.npz` sample must contain:

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

The exact 3D motion representation is dataset-dependent. For FLAME, 3DMM, or
blendshape coefficients, update `model.region_slices` in the config so the
region-aware loss/refinement uses the correct mouth, eye, cheek, and neck
indices.


