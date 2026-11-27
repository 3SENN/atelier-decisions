# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Real photos with answers people checked, and captions in many languages to search by.

    python3 -I vision/photos.py fetch          # Open Images' validation photos of the classes in vision/classes.py
    python3 -I vision/photos.py xm3600         # Crossmodal-3600: 3,600 photos, each with captions in 36 languages
    python3 -I vision/photos.py coco35         # COCO-35L's dev captions: ordinary captions as text, in 21 languages

Open Images V7 (https://storage.googleapis.com/openimages/web/index.html; labels CC BY 4.0, Google LLC; the photos
were listed as CC BY 2.0 by their authors on Flickr): every validation photo has labels that people confirmed
("there is a dog") and labels they rejected ("there is no cat"), which gives yes/no questions both answers and gives
a choice its wrong options. Photos with at least one confirmed class of vision/classes.py are fetched. 30% of the
photos (by their id) are held out, and so are the classes in classes.HELD_OUT on every photo.

Crossmodal-3600 (https://google.github.io/crossmodal-3600/, CC BY 4.0, Google LLC; photos from Open Images): never
trained on. It measures search: a caption in, the right photo out, in each language.

COCO-35L (the same page; CC BY 4.0, Google LLC): COCO's captions translated by machine into 35 languages. Only its
dev split's text is used, no photos: a quarter of the captions, in the languages measured, are the ordinary text
the text tower has to keep reading as the base does while it trains (vision/train.py), and what the shipped
vocabulary is chosen from (vision/export.py).

The originals are dropped as they download: each photo is kept as the 256 x 256 square the tower reads
(vision/preprocess.py) at JPEG quality 90, with its author, licence and page for the attribution CC BY asks for.
About 15 KB a photo instead of 300.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tarfile
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

import numpy as np  # noqa: E402

from classes import HELD_OUT, classes  # noqa: E402
from preprocess import flatten, resize  # noqa: E402
from store import Pack, jpeg  # noqa: E402

CACHE = ROOT / "data" / "cache" / "vision"
OI = "https://storage.googleapis.com/openimages"
OI_FILES = {
    "oidv7-class-descriptions.csv": f"{OI}/v7/oidv7-class-descriptions.csv",
    "oidv7-val-annotations-human-imagelabels.csv": f"{OI}/v7/oidv7-val-annotations-human-imagelabels.csv",
    "validation-images-with-rotation.csv": f"{OI}/2018_04/validation/validation-images-with-rotation.csv",
}
OI_PHOTO = "https://open-images-dataset.s3.amazonaws.com/validation/{id}.jpg"
COCO35 = "https://storage.googleapis.com/crossmodal-3600/coco_mt_dev.jsonl.bz2"
XM_IMAGES = "https://open-images-dataset.s3.amazonaws.com/crossmodal-3600/images.tgz"
XM_CAPTIONS = "https://google.github.io/crossmodal-3600/web-data/captions.zip"
# The five languages the questions are written in, then languages only measured: Latin-script ones nobody tuned for,
# and other scripts, which the trimmed vocabulary can't read (the runtime declines them).
XM_LANGS = ["en", "nl", "de", "fr", "es", "it", "pt", "sv", "da", "pl", "id", "tr", "ru", "el", "ar", "he", "hi", "zh", "ja", "ko", "th"]


def split_of(image_id: str) -> str:
    """70% of the photos to train on, 30% never seen, by the photo's own id."""
    return "train" if zlib.crc32(image_id.encode()) % 10 < 7 else "heldout"


def square(data: bytes, rotation: float = 0) -> tuple[bytes, int, int]:
    """A downloaded photo as the stored 256 x 256 JPEG, with the original's width and height."""
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if rotation:
            im = im.rotate(rotation, expand=True)
        pixels = np.asarray(im)
    return jpeg(resize(flatten(pixels))), pixels.shape[1], pixels.shape[0]


def get(url: str, tries: int = 4) -> bytes:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "atelier-decisions"}), timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch(args: argparse.Namespace) -> None:
    oi = CACHE / "oi"
    oi.mkdir(parents=True, exist_ok=True)
    for name, url in OI_FILES.items():
        if not (oi / name).exists():
            (oi / name).write_bytes(get(url))
    table = classes()
    mids = {mid: name for mid, name in csv.reader((oi / "oidv7-class-descriptions.csv").open(encoding="utf-8")) if name in table}
    unknown = set(table) - set(mids.values())
    if unknown:
        raise SystemExit(f"not Open Images classes: {sorted(unknown)}")
    yes: dict[str, list[str]] = {}
    no: dict[str, list[str]] = {}
    for r in csv.DictReader((oi / "oidv7-val-annotations-human-imagelabels.csv").open(encoding="utf-8")):
        name = mids.get(r["LabelName"])
        if name:
            (yes if r["Confidence"] == "1" else no).setdefault(r["ImageID"], []).append(name)
    info = {r["ImageID"]: r for r in csv.DictReader((oi / "validation-images-with-rotation.csv").open(encoding="utf-8")) if r["ImageID"] in yes}
    wanted = sorted(yes)
    if args.limit:
        wanted = wanted[: args.limit]
    pack = Pack(CACHE / "photos")
    todo = [i for i in wanted if i not in pack]
    print(f"{len(wanted)} photos with a confirmed class of the {len(table)} ({len(HELD_OUT)} classes held out); {len(todo)} to fetch", flush=True)

    def one(image_id: str):
        try:
            rotation = float(info[image_id]["Rotation"] or 0)
            return image_id, square(get(OI_PHOTO.format(id=image_id)), rotation)
        except Exception as err:  # a photo that is gone or broken is skipped and listed
            return image_id, err

    failed, t0 = [], time.time()
    with ThreadPoolExecutor(args.threads) as pool:
        for n, (image_id, result) in enumerate(pool.map(one, todo), 1):
            if isinstance(result, Exception):
                failed.append(image_id)
            else:
                data, w, h = result
                r = info[image_id]
                pack.add(image_id, data, split=split_of(image_id), yes=sorted(set(yes[image_id])), no=sorted(set(no.get(image_id, []))), w=w, h=h,
                         author=r["Author"], licence=r["License"], page=r["OriginalLandingURL"])
            if n % 500 == 0:
                print(f"  {n}/{len(todo)} in {time.time() - t0:.0f} s, {len(failed)} failed", flush=True)
    print(f"{len(pack)} photos in {pack.bin} ({pack.bin.stat().st_size / 1e6:.0f} MB); {len(failed)} could not be fetched")


def xm3600(args: argparse.Namespace) -> None:
    out = CACHE / "xm3600-captions.jsonl"
    if not out.exists():
        with zipfile.ZipFile(io.BytesIO(get(XM_CAPTIONS))) as z:
            name = next(n for n in z.namelist() if n.endswith(".jsonl"))
            rows = []
            for line in z.read(name).decode("utf-8").splitlines():
                r = json.loads(line)
                rows.append({"image": r["image/key"], "captions": {lang: r[lang]["caption"] for lang in XM_LANGS if lang in r}})
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    captions = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    print(f"{len(captions)} photos with captions in {len(XM_LANGS)} languages", flush=True)
    pack = Pack(CACHE / "xm3600")
    if len(pack) >= len(captions):
        print(f"{len(pack)} photos already in {pack.bin}")
        return
    # The archive is read as it streams and never written to disk.
    with urllib.request.urlopen(urllib.request.Request(XM_IMAGES, headers={"User-Agent": "atelier-decisions"}), timeout=120) as r:
        with tarfile.open(fileobj=r, mode="r|gz") as tar:
            for n, member in enumerate(tar, 1):
                key = Path(member.name).stem
                if not member.isfile() or key in pack or not member.name.lower().endswith((".jpg", ".jpeg")):
                    continue
                data, w, h = square(tar.extractfile(member).read())
                pack.add(key, data, w=w, h=h)
                if len(pack) % 500 == 0:
                    print(f"  {len(pack)} photos", flush=True)
    print(f"{len(pack)} photos in {pack.bin} ({pack.bin.stat().st_size / 1e6:.0f} MB)")


def untokenize(text: str) -> str:
    """COCO-35L's text has its punctuation spaced out ("a kite , flying ."): put it back."""
    import re

    text = re.sub(r" ([.,!?;:%)\]])", r"\1", text)
    text = re.sub(r"([(\[]) ", r"\1", text)
    return re.sub(r" (['’]) ?(s|t|re|ve|ll|d|m)\b", r"\1\2", text).strip()


def coco35(args: argparse.Namespace) -> None:
    import bz2

    out = CACHE / "coco35-dev.jsonl"
    if out.exists():
        print(f"{sum(1 for _ in out.open(encoding='utf-8'))} captions already in {out}")
        return
    rows, seen = [], set()
    unzip = bz2.BZ2Decompressor()
    left = b""
    with urllib.request.urlopen(urllib.request.Request(COCO35, headers={"User-Agent": "atelier-decisions"}), timeout=120) as r:
        while chunk := r.read(1 << 20):
            lines = (left + unzip.decompress(chunk)).split(b"\n")
            left = lines.pop()
            for line in lines:
                if not line.strip():
                    continue
                row = json.loads(line)
                if zlib.crc32(row["image_id"].encode()) % 4:
                    continue  # a quarter of the captions is plenty
                if row["image_id"] not in seen:
                    seen.add(row["image_id"])
                    rows.append({"lang": "en", "text": untokenize(row["caption_tokenized"])})
                if row["trg_lang"] in XM_LANGS:
                    rows.append({"lang": row["trg_lang"], "text": untokenize(row["translation_tokenized"])})
    CACHE.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["lang"]] = counts.get(r["lang"], 0) + 1
    print(f"{len(rows)} captions to {out} ({out.stat().st_size / 1e6:.0f} MB): {counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--limit", type=int, default=0, help="only the first N photos (a quick try)")
    f.add_argument("--threads", type=int, default=24)
    sub.add_parser("xm3600")
    sub.add_parser("coco35")
    args = parser.parse_args()
    {"fetch": fetch, "xm3600": xm3600, "coco35": coco35}[args.cmd](args)


if __name__ == "__main__":
    main()
