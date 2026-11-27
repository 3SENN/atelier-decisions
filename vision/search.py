# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Search: a caption in, the right photo out. Measured on Crossmodal-3600 in each language.

    .venv/bin/python -I vision/search.py --emb runs/vision/emb/base --name vision-search-base
    .venv/bin/python -I vision/search.py --emb runs/vision/emb/v1 --adapters runs/vision-v1/vision.pt --name vision-search-v1
    .venv/bin/python -I vision/search.py --runtime releases/vision-v0.1/model --name vision-search-v0.1     # the shipped files

For every caption of every photo: is that photo the first, among the first five, among the first ten of the photos
nearest to the caption's vector? 3,600 photos to choose from. A language the shipped vocabulary can't read is
reported as declined, with what share of its captions the runtime would refuse.

Training never sees these photos or these captions, so a drop from the base's numbers is what the new skills cost
search, and the shipped files' drop from the trained model's is what shrinking them cost.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

import numpy as np  # noqa: E402

from embed import load  # noqa: E402

CACHE = ROOT / "data" / "cache" / "vision"
RESULTS = ROOT / "analysis" / "results"
MAIN = ["en", "nl", "de", "fr", "es"]


def captions() -> list[dict]:
    return [json.loads(line) for line in (CACHE / "xm3600-captions.jsonl").read_text(encoding="utf-8").splitlines()]


AMONG = 40  # as many pictures as Atelier's label_pictures takes from one page


def recall(queries: np.ndarray, truth: np.ndarray, photos: np.ndarray) -> dict:
    """Recall at 1, 5 and 10 of each query's photo among all photos, and at 1 among it and 39 others drawn at random
    (a page's worth: what finding one picture on a page by a description is like)."""
    hits = {1: 0, 5: 0, 10: 0}
    few = 0
    rng = np.random.default_rng(40)
    for i in range(0, len(queries), 2048):
        sims = queries[i : i + 2048] @ photos.T
        own = sims[np.arange(len(sims)), truth[i : i + 2048]]
        rank = (sims > own[:, None]).sum(1)
        for k in hits:
            hits[k] += int((rank < k).sum())
        others = rng.integers(0, photos.shape[0], (len(sims), AMONG - 1))
        beaten = np.take_along_axis(sims, others, axis=1) > own[:, None]
        few += int((~(beaten & (others != truth[i : i + 2048, None])).any(1)).sum())
    return {**{f"r{k}": round(v / max(1, len(queries)), 4) for k, v in hits.items()}, "first_among_40": round(few / max(1, len(queries)), 4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", type=Path, help="the photos' vectors from vision/embed.py")
    parser.add_argument("--adapters", type=Path)
    parser.add_argument("--runtime", type=Path, help="an exported model folder: both towers run as shipped (ONNX)")
    parser.add_argument("--langs", nargs="*", help="only these languages")
    parser.add_argument("--first-caption", action="store_true", help="one caption a photo and language instead of all (a quicker look)")
    parser.add_argument("--keep-case", action="store_true", help="don't lowercase the captions (to see what lowercasing does)")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    rows = captions()
    langs = args.langs or list(rows[0]["captions"])
    if args.runtime:
        from runtime import Runtime
        from store import Pack

        rt = Runtime(args.runtime, threads=8, text_cache=(args.emb / "texts.npz") if args.emb else None)
        pack = Pack(CACHE / "xm3600")
        ids = [r["image"] for r in rows]
        if args.emb:  # the photos' vectors as the shipped image file made them (vision/embed.py --runtime)
            all_ids, vectors = load(args.emb, "xm3600")
            at = {k: i for i, k in enumerate(all_ids)}
            photos = vectors[[at[k] for k in ids]]
            photos /= np.linalg.norm(photos, axis=1, keepdims=True)
        else:
            photos = np.stack([rt.image(pack.pixels(k)) for k in ids])
        def embed(texts: list[str]) -> np.ndarray:
            rt.read(texts)  # together, in batches
            rt.save_texts()
            return np.stack([rt.text(t) for t in texts])

        refuses = rt.unreadable
    else:
        from model import Towers, device, embed_texts, tokenizer

        all_ids, vectors = load(args.emb, "xm3600")
        at = {k: i for i, k in enumerate(all_ids)}
        ids = [r["image"] for r in rows]
        photos = vectors[[at[k] for k in ids]]
        photos /= np.linalg.norm(photos, axis=1, keepdims=True)
        dev = device()
        model, tok = Towers.load(adapters=args.adapters).to(dev), tokenizer()
        if args.keep_case:
            import format as fmt

            fmt.clean = lambda text: " ".join(text.split())
        embed = lambda texts: embed_texts(model, tok, texts, dev)
        refuses = lambda text: None
    result = {"meta": {"photos": len(ids), "from": str(args.runtime or args.emb), "adapters": str(args.adapters) if args.adapters else None}}
    for lang in langs:
        texts, truth = [], []
        for i, r in enumerate(rows):
            for caption in r["captions"].get(lang, [])[: 1 if args.first_caption else None]:
                texts.append(caption)
                truth.append(i)
        keep = [i for i, t in enumerate(texts) if not refuses(t)]
        declined = len(texts) - len(keep)
        if declined > len(texts) / 2:
            result[lang] = {"captions": len(texts), "declined": round(declined / len(texts), 4)}
        else:
            # Recall over the captions it reads; the declined ones are counted beside it.
            result[lang] = {"captions": len(texts), "declined": round(declined / len(texts), 4), **recall(embed([texts[i] for i in keep]), np.array([truth[i] for i in keep]), photos)}
        print(f"  {lang}: {result[lang]}", flush=True)
    mean = lambda ls, k: round(float(np.mean([result[x][k] for x in ls if k in result[x]])), 4) if any(k in result[x] for x in ls) else None
    main_langs = [x for x in MAIN if x in result]
    keys = ("r1", "r5", "r10", "first_among_40")
    result["mean"] = {"en nl de fr es": {k: mean(main_langs, k) for k in keys}, "all answered": {k: mean([x for x in langs if "r1" in result[x]], k) for k in keys}}
    lines = [f"search on Crossmodal-3600, {len(ids)} photos: a caption in, its photo among the nearest 1 / 5 / 10 of all of them; and first among it and {AMONG - 1} others"]
    for lang in langs:
        r = result[lang]
        lines.append(f"  {lang:3s} {r['captions']:5d} captions  " + (f"{100 * r['r1']:5.1f}% / {100 * r['r5']:5.1f}% / {100 * r['r10']:5.1f}%   among {AMONG}: {100 * r['first_among_40']:5.1f}%" if "r1" in r else "declined")
                     + (f"  ({100 * r['declined']:.1f}% of captions declined)" if r["declined"] else ""))
    for name, m in result["mean"].items():
        if m["r1"] is not None:
            lines.append(f"  mean, {name}: {100 * m['r1']:.1f}% / {100 * m['r5']:.1f}% / {100 * m['r10']:.1f}%   among {AMONG}: {100 * m['first_among_40']:.1f}%")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (RESULTS / f"{args.name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
