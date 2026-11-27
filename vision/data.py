# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Which questions are asked about which picture, and what is held out.

    .venv/bin/python -I vision/data.py eval --out data/cache/vision/eval.jsonl      # the held-out questions, fixed
    .venv/bin/python -I vision/data.py stats

Every row is one question about one picture with its answer:
    {"image": "<pack>/<id>", "family", "split", "label", "q": {type, instructions, criteria}, "answer", "kind",
     "form", "lang", "unseen_wording", "n", "half"}

Splits (none of the held-out ones is ever trained on):
    train            trained things on training pictures
    heldout-items    trained things on pictures never seen: other photos, other drawings of an icon by the same sets,
                     other documents and pages of the same kinds
    heldout-look     trained things drawn a way never seen: three icon sets, two page layouts, and for every kind of
                     document and page one language its text was never in
    heldout-kinds    things never trained on, as an answer or as a wrong option: 37 photo classes, 8 button jobs,
                     5 kinds of document, 2 page states
    heldout-none     buttons from the three held-out icon sets whose icon is none of the 63 jobs: Atelier's own
                     question about them ("which of all 63 jobs", answer: none) has nothing right to be sure of

"half" is a or b by the picture's id: vision/calibrate.py fits on half a and vision/measure.py reports half b.

Photos (Open Images, vision/photos.py): a class people confirmed is a yes, a class people rejected is a no, and a
choice's wrong options are rejected classes first. Classes nobody checked on a photo are used as wrong options only
when they are unlikely to be in the background (never INCIDENTAL ones such as a chair, the sea or a hat).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from classes import ALBUMS, HELD_OUT, LANGS, bare, classes  # noqa: E402
from questions import Family, language  # noqa: E402
from store import Pack  # noqa: E402

CACHE = ROOT / "data" / "cache" / "vision"

# In the background of photos of anything: wrong options only where people rejected them for that photo.
INCIDENTAL = {
    "Sea", "Snow", "Mountain", "Beach", "River", "Lake", "Flower", "Houseplant", "Chair", "Couch", "Bed", "Lamp", "Stairs", "Tower", "Bridge (structure)", "Hat",
    "Glasses", "Boot", "Tie", "Watch", "Handbag", "Backpack", "Clock", "Mirror", "Mobile phone", "Poster", "Logo", "Flag", "Book", "Bread", "Car", "Bicycle", "Boat",
    "Bird", "Dog", "Television", "Laptop", "Vase", "Candle", "Coffee cup", "Wine glass", "Microphone", "Helmet", "Skyscraper", "Drawing", "Cartoon", "Screenshot",
    "Fire", "Moon", "Traffic sign", "Truck", "Van", "Bus", "Balloon", "Sink", "Fountain", "Football", "Headphones", "Camera", "Umbrella",
}
INCIDENTAL_ALBUMS = {"wear", "home", "pictures", "tech", "plants", "landscape"}


def half_of(key: str) -> str:
    return "a" if zlib.crc32(("half:" + key).encode()) % 2 == 0 else "b"


# ── families ─────────────────────────────────────────────────────────────────────────────────────────────────────────

_CLASSES = classes()


def _photo_words(label: str, lang: str) -> dict:
    x = _CLASSES[label][lang]
    return {"x": x, "name": bare(x, lang)}


def _album_words(label: str, lang: str) -> dict:
    name = ALBUMS[label][LANGS.index(lang)]
    return {"x": name, "name": name}


PHOTO = Family("photo", _photo_words)
ALBUM = Family("album", _album_words)


def families() -> dict[str, Family]:
    out = {"photo": PHOTO, "album": ALBUM}
    try:
        from synth import synth_families  # documents, pages and buttons

        out.update(synth_families())
    except ImportError:
        pass
    return out


# ── photos ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def photo_questions(key: str, row: dict, rng: random.Random, split: str, unseen: bool) -> list[dict]:
    """The questions one photo gives in one split: a choice of thing, a choice of album, a yes and a no."""
    kinds = split == "heldout-kinds"
    allowed = set(_CLASSES) if kinds else set(_CLASSES) - HELD_OUT
    yes = [c for c in row["yes"] if (c in HELD_OUT) == kinds]
    no = [c for c in row["no"] if c in allowed]
    if not yes:
        return []
    out = []
    present = set(row["yes"])
    lang = language(rng)

    def wrong(n: int, answer: str) -> list[str]:
        """n classes that are not in the photo: rejected ones first, then ones unlikely to be there unnoticed."""
        rejected = [c for c in no if c not in present]
        rng.shuffle(rejected)
        pool = [c for c in allowed if c not in present and c not in rejected and c not in INCIDENTAL and (_CLASSES[c]["album"] != _CLASSES[answer]["album"])]
        if kinds:  # at least half of a new thing's wrong options are new things too
            new = [c for c in pool if c in HELD_OUT]
            rng.shuffle(new)
            pool = new[: max(1, n // 2)] + [c for c in pool if c not in HELD_OUT]
        else:
            rng.shuffle(pool)
        picked = rejected[: max(1, n * 2 // 3)]
        picked += [c for c in pool if c not in picked][: n - len(picked)]
        return picked[:n]

    answer = rng.choice(yes)
    others = wrong(rng.choice([1, 2, 3, 3, 4, 5, 7]), answer)
    if others:
        out.append({"family": "photo", "label": answer, **PHOTO.choice(rng, lang, answer, others, unseen)})
    # The album, when everything confirmed in the photo is from one album and some other album clearly isn't there.
    albums = {_CLASSES[c]["album"] for c in row["yes"]}
    if len(albums) == 1 and not kinds:
        album = next(iter(albums))
        rejected_albums = {_CLASSES[c]["album"] for c in row["no"]} - albums
        other_albums = sorted(rejected_albums | (set(ALBUMS) - albums - INCIDENTAL_ALBUMS))
        rng.shuffle(other_albums)
        n = rng.choice([2, 3, 4, 5])
        if len(other_albums) >= 2:
            out.append({"family": "album", "label": album, **ALBUM.choice(rng, language(rng), album, other_albums[:n], unseen)})
    out.append({"family": "photo", "label": answer, **PHOTO.noul(rng, language(rng), answer, True, unseen)})
    absent = [c for c in no if c not in present]
    if absent:
        c = rng.choice(absent)
        out.append({"family": "photo", "label": c, **PHOTO.noul(rng, language(rng), c, False, unseen)})
    elif rng.random() < 0.5:
        easy = [c for c in allowed if c not in present and c not in INCIDENTAL and _CLASSES[c]["album"] not in albums]
        c = rng.choice(easy)
        out.append({"family": "photo", "label": c, **PHOTO.noul(rng, language(rng), c, False, unseen)})
    for r in out:
        r.update(image=f"photos/{key}", split=split, half=half_of(key))
    return out


def photo_rows(split: str, seed: int, unseen: bool, pack: Pack | None = None) -> list[dict]:
    pack = pack or Pack(CACHE / "photos")
    rng = random.Random(seed)
    rows = []
    for key, row in pack.rows.items():
        if (row["split"] == "train") != (split == "train"):
            continue
        rows += photo_questions(key, row, rng, split, unseen)
    return rows


# ── everything ───────────────────────────────────────────────────────────────────────────────────────────────────────

HELD_OUT_SPLITS = ["heldout-items", "heldout-look", "heldout-kinds", "heldout-none"]


def eval_rows(seed: int = 20261126) -> list[dict]:
    """The fixed held-out questions: every held-out picture asked once, with the unseen wordings mixed in."""
    rows = []
    for split in ("heldout-items", "heldout-kinds"):
        rows += photo_rows(split, seed, unseen=True)
    try:
        from synth import synth_rows

        for split in HELD_OUT_SPLITS:
            rows += synth_rows(split, seed, unseen=True)
    except ImportError:
        pass
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["eval", "stats"])
    parser.add_argument("--out", type=Path, default=CACHE / "eval.jsonl")
    args = parser.parse_args()
    rows = eval_rows()
    if args.cmd == "eval":
        args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"{len(rows)} held-out questions to {args.out}")
    counts: dict[tuple, int] = {}
    for r in rows:
        k = (r["family"], r["split"], r["kind"])
        counts[k] = counts.get(k, 0) + 1
    for k in sorted(counts):
        print(f"  {k[0]:10s} {k[1]:14s} {k[2]:7s} {counts[k]}")
    print(f"  train (one pass): {len(photo_rows('train', 1, False))} photo questions")


if __name__ == "__main__":
    main()
