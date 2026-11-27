# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""atelier-vision's image input: any picture as the 256 x 256 square the image tower reads.

    pixels (RGB or RGBA, 8 bits, any size)
      -> transparent parts over white (an icon's SVG has no background of its own)
      -> the whole picture resized to 256 x 256, its proportions not kept (as SigLIP 2 was trained)
      -> every value x / 127.5 - 1, channels first: float32 [3, 256, 256]

The resize is a triangle filter whose width grows with the reduction (what Pillow calls BILINEAR and SigLIP's own
processor uses), in double precision and without rounding in between, rows first. It is written out here instead of
calling Pillow because the runtime has no Pillow: runtime/node/src/vision/preprocess.ts is the same arithmetic in the
same order, and `npm run vision:parity` shows the two give the same numbers. A different resize (a browser's canvas,
Electron's nativeImage) gives slightly different pixels and so slightly different answers; vision/robust.py measures
how much.

This module is the contract with the runtime, like vision/format.py.
"""

from __future__ import annotations

import math

import numpy as np

SIZE = 256


def taps(n_in: int, n_out: int) -> list[tuple[int, np.ndarray]]:
    """For each output position: the first input position it reads and its weights (they sum to 1)."""
    scale = n_in / n_out
    support = max(scale, 1.0)
    out = []
    for i in range(n_out):
        center = (i + 0.5) * scale
        lo = max(0, int(center - support + 0.5))
        hi = min(n_in, int(center + support + 0.5))
        w = np.array([max(0.0, 1.0 - abs((j + 0.5 - center) / support)) for j in range(lo, hi)], dtype=np.float64)
        out.append((lo, w / w.sum()))
    return out


def flatten(pixels: np.ndarray) -> np.ndarray:
    """uint8 [H, W, 3 or 4] to float64 [H, W, 3] with transparent parts over white."""
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError(f"expected pixels [H, W, 3 or 4], got {pixels.shape}")
    rgb = pixels[:, :, :3].astype(np.float64)
    if pixels.shape[2] == 4:
        alpha = pixels[:, :, 3:4].astype(np.float64) / 255.0
        rgb = rgb * alpha + 255.0 * (1.0 - alpha)
    return rgb


def resize(rgb: np.ndarray, size: int = SIZE) -> np.ndarray:
    """float64 [H, W, 3] to float64 [size, size, 3]: along each row first, then along each column."""
    h, w, _ = rgb.shape
    rows = np.empty((h, size, 3), dtype=np.float64)
    for o, (lo, weights) in enumerate(taps(w, size)):
        rows[:, o, :] = np.einsum("hwc,w->hc", rgb[:, lo : lo + len(weights), :], weights)
    out = np.empty((size, size, 3), dtype=np.float64)
    for o, (lo, weights) in enumerate(taps(h, size)):
        out[o] = np.einsum("hwc,h->wc", rows[lo : lo + len(weights)], weights)
    return out


def preprocess(pixels: np.ndarray, size: int = SIZE) -> np.ndarray:
    """uint8 [H, W, 3 or 4] to the tower's input, float32 [3, size, size] in -1..1."""
    return (resize(flatten(pixels), size) / 127.5 - 1.0).transpose(2, 0, 1).astype(np.float32)


def checksum(x: np.ndarray) -> float:
    """A number that changes when the input does, for the runtime's parity check: the sum of every value times a
    position-dependent weight, in double precision."""
    flat = x.astype(np.float64).ravel()
    return float(np.dot(flat, (np.arange(flat.size) % 97 + 1) / 97.0))


def load(path) -> np.ndarray:
    """A picture file as uint8 pixels, upright (a phone's photo carries its rotation as a tag)."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return np.asarray(im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB"))


if __name__ == "__main__":
    # A quick look at how close this is to Pillow's own BILINEAR (which rounds to 8 bits after each pass).
    from PIL import Image

    rng = np.random.default_rng(0)
    for h, w in ((480, 640), (1024, 768), (96, 96), (300, 1200)):
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        ours = resize(flatten(img))
        pil = np.asarray(Image.fromarray(img).resize((SIZE, SIZE), Image.BILINEAR)).astype(np.float64)
        print(f"{w}x{h}: largest difference from Pillow {np.abs(ours - pil).max():.2f} of 255, mean {np.abs(ours - pil).mean():.3f}")
    assert math.isclose(sum(w.sum() for _, w in taps(640, SIZE)), SIZE)
