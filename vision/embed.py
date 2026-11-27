# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Every picture of a set as its vector, once, so measuring is a matter of seconds after that.

    .venv/bin/python -I vision/embed.py --packs photos xm3600 --out runs/vision/emb/base
    .venv/bin/python -I vision/embed.py --packs photos --adapters runs/vision-v1/vision.pt --out runs/vision/emb/v1
    .venv/bin/python -I vision/embed.py --packs photos --runtime runs/vision-v1/onnx --only-held-out --out runs/vision/emb/v1-onnx

Writes <out>/<pack>.npz: ids (the pack's order) and vectors (float16 [N, 768], unit length before rounding).
With --runtime the shipped image file computes them on the CPU (vision/runtime.py), which is what gets measured
before a version is saved; --only-held-out leaves out the training pictures, which nothing is measured on.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from model import Towers, device  # noqa: E402
from preprocess import SIZE, preprocess  # noqa: E402
from store import Pack  # noqa: E402

CACHE = ROOT / "data" / "cache" / "vision"


def tower_input(pixels: np.ndarray) -> np.ndarray:
    """uint8 pixels to the tower's input. A stored 256 x 256 photo only needs scaling: resizing it would change nothing."""
    if pixels.shape == (SIZE, SIZE, 3):
        return (pixels.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)
    return preprocess(pixels)


@torch.no_grad()
def embed_pack(model: Towers, pack: Pack, dev: str, batch: int = 32, ids: list[str] | None = None) -> tuple[list[str], np.ndarray]:
    ids = ids if ids is not None else list(pack.rows)
    out = np.zeros((len(ids), 768), dtype=np.float16)
    t0 = time.time()
    for i in range(0, len(ids), batch):
        x = torch.from_numpy(np.stack([tower_input(pack.pixels(k)) for k in ids[i : i + batch]])).to(dev)
        out[i : i + batch] = model.image(x).float().cpu().numpy()
        if (i // batch) % 50 == 49:
            print(f"  {i + batch}/{len(ids)} in {time.time() - t0:.0f} s", flush=True)
    return ids, out


def embed_pack_runtime(rt, pack: Pack, ids: list[str], batch: int = 8) -> np.ndarray:
    """The same with the shipped ONNX file on the CPU."""
    out = np.zeros((len(ids), 768), dtype=np.float16)
    t0 = time.time()
    for i in range(0, len(ids), batch):
        x = np.stack([preprocess(pack.pixels(k), rt.size) for k in ids[i : i + batch]])
        out[i : i + batch] = rt.image_session.run(None, {"pixels": x})[0]
        if (i // batch) % 100 == 99:
            print(f"  {i + batch}/{len(ids)} in {time.time() - t0:.0f} s", flush=True)
    return out


def load(folder: Path, name: str) -> tuple[list[str], np.ndarray]:
    blob = np.load(folder / f"{name}.npz")
    return [str(i) for i in blob["ids"]], blob["vectors"].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packs", nargs="+", required=True)
    parser.add_argument("--adapters", type=Path)
    parser.add_argument("--runtime", type=Path, help="an exported folder: its image file on the CPU instead of the PyTorch tower")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--only-held-out", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.runtime:
        from runtime import Runtime

        rt = Runtime(args.runtime, threads=args.threads)
    else:
        dev = device()
        model = Towers.load(adapters=args.adapters).to(dev)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.packs:
        pack = Pack(CACHE / name)
        ids = [k for k, row in pack.rows.items() if not args.only_held_out or row.get("split", "heldout") != "train"]
        if args.runtime:
            vectors = embed_pack_runtime(rt, pack, ids)
        else:
            ids, vectors = embed_pack(model, pack, dev, ids=ids)
        np.savez(args.out / f"{name}.npz", ids=np.array(ids), vectors=vectors)
        print(f"{name}: {len(ids)} pictures to {args.out / (name + '.npz')}", flush=True)


if __name__ == "__main__":
    main()
