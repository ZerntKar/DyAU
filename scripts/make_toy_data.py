from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_split(
    root: Path,
    split: str,
    count: int,
    audio_dim: int,
    motion_dim: int,
    au_dim: int,
    min_len: int,
    max_len: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest = root / f"{split}.txt"
    with manifest.open("w", encoding="utf-8") as f:
        for idx in range(count):
            time = int(rng.integers(min_len, max_len + 1))
            t = np.linspace(0.0, 1.0, time, dtype=np.float32)
            audio_a = rng.normal(size=(time, audio_dim)).astype(np.float32)
            audio_b = rng.normal(size=(time, audio_dim)).astype(np.float32)
            base = np.sin(2 * np.pi * t)[:, None]
            motion_a = (rng.normal(scale=0.2, size=(time, motion_dim)) + base).astype(np.float32)
            motion_b = (rng.normal(scale=0.2, size=(time, motion_dim)) - base).astype(np.float32)
            pseudo_au_a = rng.normal(size=(time, au_dim)).astype(np.float32)
            pseudo_au_b = rng.normal(size=(time, au_dim)).astype(np.float32)
            name = f"{split}_{idx:04d}.npz"
            np.savez_compressed(
                split_dir / name,
                audio_a=audio_a,
                audio_b=audio_b,
                motion_a=motion_a,
                motion_b=motion_b,
                pseudo_au_a=pseudo_au_a,
                pseudo_au_b=pseudo_au_b,
            )
            f.write(f"{split}/{name}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic DyAU dataset.")
    parser.add_argument("--out", default="toy_data", help="Output directory.")
    parser.add_argument("--audio-dim", type=int, default=16)
    parser.add_argument("--motion-dim", type=int, default=32)
    parser.add_argument("--au-dim", type=int, default=12)
    parser.add_argument("--min-len", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=24)
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    make_split(root, "train", 12, args.audio_dim, args.motion_dim, args.au_dim, args.min_len, args.max_len, 11)
    make_split(root, "val", 4, args.audio_dim, args.motion_dim, args.au_dim, args.min_len, args.max_len, 13)
    make_split(root, "test", 4, args.audio_dim, args.motion_dim, args.au_dim, args.min_len, args.max_len, 17)
    print(f"Wrote toy dataset to {root.resolve()}")


if __name__ == "__main__":
    main()
