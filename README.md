# DyAU

Reference PyTorch implementation of **Pseudo-AU Guided Dyadic Speech-Driven
3D Facial Motion Generation**.

The code mirrors the method section of the paper:

- shared dual-stream audio encoding for two speakers;
- structured motion encoding with lip and expression latent spaces;
- interaction query summary for speaking dominance, listener feedback, role
  switching, and affective cues;
- four Pseudo-AU groups: Mouth/Jaw, Brow-Eye, Cheek, and Head-Neck;
- subject-wise Pseudo-AU priors as region-level control signals;
- conditional motion decoding for both subjects;
- region-aware residual MLP refinement;
- reconstruction, structured consistency, pseudo-AU, region, and dynamic losses
  with first- and second-order temporal terms.

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
region-aware loss/refinement uses the four paper regions: `mouth_jaw`,
`brow_eye`, `cheek`, and `head_neck`.

If explicit `pseudo_au_a` / `pseudo_au_b` labels are missing, the dataset loader
can derive fallback weak labels from normalized local motion variation in these
four regions. This mirrors the paper's motion-statistics fallback when reliable
OpenFace AU estimates are unavailable.


<<<<<<< HEAD
=======
From this repository root:

```bash
cd code
python3 scripts/smoke_test.py
python3 scripts/make_toy_data.py --out toy_data
python3 -m DyAU.train --config configs/DyAU_toy.yaml
python3 -m DyAU.eval --config configs/DyAU_toy.yaml --checkpoint runs/DyAU_toy/latest.pt
```

For real datasets, edit `configs/DyAU.yaml`:

```yaml
data:
  train_manifest: /path/to/train.txt
  val_manifest: /path/to/val.txt
  test_manifest: /path/to/test.txt
```

Then run:

```bash
python3 -m DyAU.train --config configs/DyAU.yaml
```

## Notes

The current paper specifies the main default settings used here: 256-dimensional
shared audio projection, 4-layer 4-head interaction encoder and motion decoder,
8 interaction query tokens, 32-dimensional Pseudo-AU space split into four
8-channel groups, 120-frame input clips, AdamW, 5-epoch warmup, cosine learning
rate decay, and the reported loss/region weights. Dataset-specific facial
parameter indices still need to be set in `model.region_slices`.
>>>>>>> 4c6f80e (update code)
