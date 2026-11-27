# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""How much atelier-vision's answers depend on how a picture reached it.

The model was trained and measured on pictures resized one way (vision/preprocess.py). In a browser a picture comes
through other hands first: Chromium decodes it, Electron may have made it smaller before it crosses to the helper, a
site may have saved it as a rough JPEG. This takes held-out photos from their originals and held-out buttons at their
own size, puts each through those other paths, and reports how far the vector moves and how many of the picture's
held-out questions change their reading (Atelier's cutoffs, vision/decide.py).

    .venv/bin/python -I vision/robust.py --runtime runs/vision-v1/onnx --emb runs/vision/emb/v1-onnx --name vision-robust-v0.1

Reference: the original picture through vision/preprocess.py alone. Paths:
    as stored       photos only: the 256 x 256 JPEG the measurements use (what keeping them small cost)
    smaller first   made at most 768 pixels on its longest edge first, as Atelier does before a picture crosses
                    processes (src/main/decisions/pictures.ts), then the model's own resize
    bicubic, lanczos, nearest   resized to 256 x 256 by another filter instead of the model's own
    jpeg 60         saved as a JPEG of quality 60 and read back, then the model's own resize
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

import numpy as np  # noqa: E402

from decide import probabilities, verdict  # noqa: E402
from format import labels, texts  # noqa: E402
from photos import OI_PHOTO, get  # noqa: E402
from runtime import Runtime  # noqa: E402
from store import Pack  # noqa: E402

CACHE = ROOT / "data" / "cache" / "vision"
RESULTS = ROOT / "analysis" / "results"


def paths(original: np.ndarray, stored: np.ndarray | None) -> dict[str, np.ndarray]:
    from PIL import Image

    im = Image.fromarray(original).convert("RGB")
    out = {}
    if stored is not None:
        out["as stored"] = stored
    k = min(1.0, 768 / max(im.size))
    out["smaller first"] = np.asarray(im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.BILINEAR)) if k < 1 else np.asarray(im)
    for name, how in (("bicubic", Image.BICUBIC), ("lanczos", Image.LANCZOS), ("nearest", Image.NEAREST)):
        out[name] = np.asarray(im.resize((256, 256), how))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=60)
    out["jpeg 60"] = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--photos", type=int, default=300)
    parser.add_argument("--buttons", type=int, default=300)
    parser.add_argument("--emb", type=Path, help="a folder of vectors whose texts.npz holds the texts the reports read already (vision/measure.py --runtime)")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    from PIL import Image

    rt = Runtime(args.runtime, threads=4, text_cache=(args.emb / "texts.npz") if args.emb else None)
    rng = random.Random(11)
    rows: dict[str, list[dict]] = {}
    for line in (CACHE / "eval.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows.setdefault(r["image"], []).append(r)
    photos, controls = Pack(CACHE / "photos"), Pack(CACHE / "controls")
    photo_ids = rng.sample(sorted(k for k in photos.rows if f"photos/{k}" in rows), args.photos)
    button_ids = rng.sample(sorted(k for k in controls.rows if f"controls/{k}" in rows), args.buttons)

    def original(key: str) -> np.ndarray | None:
        try:
            with Image.open(io.BytesIO(get(OI_PHOTO.format(id=key)))) as im:
                return np.asarray(im.convert("RGB"))
        except Exception:
            return None

    with ThreadPoolExecutor(16) as pool:
        originals = list(pool.map(original, photo_ids))
    pictures = [("photos", f"photos/{k}", o, photos.pixels(k)) for k, o in zip(photo_ids, originals) if o is not None]
    pictures += [("buttons", f"controls/{k}", controls.pixels(k), None) for k in button_ids]

    def readings(image: str, vector: np.ndarray) -> list[int | None]:
        out = []
        for r in rows[image]:
            if rt.decline(r["q"]) or r["answer"] is None:
                continue
            cos = [None if t is None else float(vector @ rt.text(t)) for t in texts(r["q"])]
            v = verdict(r["kind"], probabilities(r["kind"], cos, rt.scale, rt.bias, rt.cal))
            truth = labels(r["q"]).index(r["answer"])
            out.append(None if v is None else int(v == truth))
        return out

    stats: dict[str, dict[str, dict]] = {}
    for n, (kind, image, pixels, stored) in enumerate(pictures):
        if n % 25 == 0:
            print(f"  {n} of {len(pictures)} pictures", file=sys.stderr, flush=True)
        flat = pixels if pixels.shape[2] == 3 else np.asarray(Image.alpha_composite(Image.new("RGBA", pixels.shape[1::-1], "white"), Image.fromarray(pixels)).convert("RGB"))
        ref = rt.image(pixels)
        ref_read = readings(image, ref)
        for name, other in paths(flat, stored).items():
            v = rt.image(other)
            read = readings(image, v)
            s = stats.setdefault(kind, {}).setdefault(name, {"cos": [], "questions": 0, "changed": 0, "sure_right_before": 0, "sure_right_after": 0, "sure_wrong_before": 0, "sure_wrong_after": 0})
            s["cos"].append(float(ref @ v))
            s["questions"] += len(read)
            s["changed"] += sum(a != b for a, b in zip(ref_read, read))
            s["sure_right_before"] += sum(a == 1 for a in ref_read)
            s["sure_right_after"] += sum(b == 1 for b in read)
            s["sure_wrong_before"] += sum(a == 0 for a in ref_read)
            s["sure_wrong_after"] += sum(b == 0 for b in read)
    result, lines = {"runtime": str(args.runtime)}, [f"how a picture reached the model, against its original through the model's own resize ({args.runtime})"]
    for kind, per in stats.items():
        n = len(next(iter(per.values()))["cos"])
        lines.append(f"{kind} ({n} held-out pictures):")
        result[kind] = {"pictures": n}
        for name, s in per.items():
            q = max(1, s["questions"])
            row = {"cos_mean": round(float(np.mean(s["cos"])), 4), "cos_lowest": round(min(s["cos"]), 4), "questions": s["questions"], "readings_changed": round(s["changed"] / q, 4),
                   "sure_and_right": [round(s["sure_right_before"] / q, 4), round(s["sure_right_after"] / q, 4)], "sure_and_wrong": [round(s["sure_wrong_before"] / q, 4), round(s["sure_wrong_after"] / q, 4)]}
            result[kind][name] = row
            lines.append(f"  {name:14s} vector cos {row['cos_mean']:.4f} (lowest {row['cos_lowest']:.4f})  readings changed {100 * row['readings_changed']:4.1f}%  "
                         f"sure and right {100 * row['sure_and_right'][0]:.1f}% -> {100 * row['sure_and_right'][1]:.1f}%  sure and wrong {100 * row['sure_and_wrong'][0]:.1f}% -> {100 * row['sure_and_wrong'][1]:.1f}%")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (RESULTS / f"{args.name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
