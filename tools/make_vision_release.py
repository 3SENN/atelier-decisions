# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Saves an atelier-vision version under releases/<version>, as tools/make_step_release.py does for atelier-step.

    .venv/bin/python -I tools/make_vision_release.py --version vision-v0.1 --run runs/vision-v1 \\
        --results vision-v0.1.json vision-v0.1.txt vision-search-v0.1.json ...

model/  what the runtime loads (vision/export.py's folder, with vision/calibrate.py's numbers in vision.json): both
        ONNX files, the trimmed tokenizer, the parity fixtures; and what training goes on from (--init): vision.pt,
        the adapters on google/siglip2-base-patch16-256 at the revision vision.json names
eval/   the results its numbers come from (analysis/results), and the run's own log and metrics
data/   pictures.zip   the drawn pictures (buttons, documents, pages) and the photos, as the towers read them, with
                       each photo's author, licence and page
        questions.zip  the held-out questions the numbers are measured on, the captions used as text, and the
                       tables everything was worded from are in the code (vision/)

Then write MODEL_CARD.md next to it and run python tools/hf_release.py publish <version>.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "vision"
PACKS = ["photos", "controls", "documents", "pages"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--run", type=Path, required=True, help="the training run (vision/train.py --out), with its exported onnx/")
    parser.add_argument("--results", nargs="*", default=[], help="files in analysis/results to keep with the version")
    parser.add_argument("--logs", type=Path, nargs="*", default=[], help="training logs to keep with the version")
    args = parser.parse_args()
    out = ROOT / "releases" / args.version
    onnx = args.run / "onnx"
    info = json.loads((onnx / "vision.json").read_text(encoding="utf-8"))
    if "calibration" not in info:
        raise SystemExit(f"{onnx / 'vision.json'} has no calibration: run vision/calibrate.py --runtime {onnx} --out {onnx / 'vision.json'} first")
    (out / "model").mkdir(parents=True, exist_ok=True)
    def place(src: Path, dst: Path) -> None:
        """A hard link where the disk allows it (the model's files are large and the run keeps its own), else a copy."""
        dst.unlink(missing_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    for path in sorted(onnx.iterdir()):
        if path.is_file():
            place(path, out / "model" / path.name)
    for name in ("vision.pt", "metrics.json"):
        if (args.run / name).exists():
            place(args.run / name, out / "model" / name)
    (out / "eval").mkdir(exist_ok=True)
    for name in args.results:
        shutil.copy2(ROOT / "analysis" / "results" / name, out / "eval" / name)
    for path in args.logs:
        shutil.copy2(path, out / "eval" / path.name.replace(".log", ".txt"))
    (out / "data").mkdir(exist_ok=True)
    # Pictures are JPEG and PNG already: stored, not compressed again.
    with zipfile.ZipFile(out / "data" / "pictures.zip", "w", zipfile.ZIP_STORED) as z:
        for name in PACKS:
            z.write(CACHE / f"{name}.bin", f"{name}.bin")
            z.write(CACHE / f"{name}.jsonl", f"{name}.jsonl", zipfile.ZIP_DEFLATED, compresslevel=9)
        # The buttons are drawn from nine icon sets, each under its own licence (NOTICE).
        for licence in sorted((CACHE / "icons").glob("*/LICENSE")):
            z.write(licence, f"licences/icons-{licence.parent.name}.txt", zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(out / "data" / "questions.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in ("eval.jsonl", "coco35-dev.jsonl"):
            z.write(CACHE / name, name)
    files = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name not in ("MANIFEST.json", "MODEL_CARD.md"):
            files[path.relative_to(out).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (out / "MANIFEST.json").write_text(json.dumps({"version": args.version, "date": time.strftime("%Y-%m-%d"), "model": args.run.as_posix(), "base": info["base"],
                                                   "files": files}, indent=1))
    total = sum(f["bytes"] for f in files.values())
    print(f"{out}: {len(files)} files, {total / 1e6:.0f} MB")
    print(f"next: write {out.relative_to(ROOT).as_posix()}/MODEL_CARD.md, then python tools/hf_release.py publish {args.version}")


if __name__ == "__main__":
    main()
