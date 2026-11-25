# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Saves an atelier-step version under releases/<version>, as tools/make_release.py does for the decision model.

    .venv/bin/python -I tools/make_step_release.py --version step-v0.1 --run runs/step-v4 \\
        --data data/cache/m2w-steps.jsonl data/cache/synth-steps-v4.jsonl \\
        --results step-v4-heldout.json step-v4-export.json flows-step-v4-hybrid.json flows-step-v4-hybrid.txt ...

model/  what the runtime loads (step/export.py's folder, with step/calibrate.py's temperatures in step.json): both
        ONNX files, the tokenizer, the parity fixtures; and what training goes on from (--init): step.pt and config.json
eval/   the results its numbers come from (analysis/results)
data/   the steps the run was trained on, zipped

Then write MODEL_CARD.md next to it and run python tools/hf_release.py publish <version>.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--run", type=Path, required=True, help="the training run (step/train.py --out), with its exported onnx/")
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--results", nargs="*", default=[], help="files in analysis/results to keep with the version")
    args = parser.parse_args()
    out = ROOT / "releases" / args.version
    onnx = args.run / "onnx"
    info = json.loads((onnx / "step.json").read_text(encoding="utf-8"))
    if "temperatures" not in info:
        raise SystemExit(f"{onnx / 'step.json'} has no temperatures: run step/calibrate.py --export {onnx} first")
    (out / "model").mkdir(parents=True, exist_ok=True)
    for path in sorted(onnx.iterdir()):
        if path.is_file():
            shutil.copy2(path, out / "model" / path.name)
    for name in ("step.pt", "metrics.json"):
        shutil.copy2(args.run / name, out / "model" / name)
    if args.results:
        (out / "eval").mkdir(exist_ok=True)
        for name in args.results:
            shutil.copy2(ROOT / "analysis" / "results" / name, out / "eval" / name)
    (out / "data").mkdir(exist_ok=True)
    with zipfile.ZipFile(out / "data" / "steps.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in args.data:
            z.write(path, path.name)
    files = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name not in ("MANIFEST.json", "MODEL_CARD.md"):
            files[path.relative_to(out).as_posix()] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / "MANIFEST.json").write_text(json.dumps({"version": args.version, "date": time.strftime("%Y-%m-%d"), "model": args.run.as_posix(),
                                                   "files": files}, indent=1))
    total = sum(f["bytes"] for f in files.values())
    print(f"{out}: {len(files)} files, {total / 1e6:.0f} MB")
    print(f"next: write {out.relative_to(ROOT).as_posix()}/MODEL_CARD.md, then python tools/hf_release.py publish {args.version}")


if __name__ == "__main__":
    main()
