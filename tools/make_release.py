"""Saves a model version under releases/<version>: everything needed to ship it, measure it and keep training it.

    python tools/make_release.py --version v0.2 --model runs/nano-v02-l22 --report v0.2-l22 \
        --data data/teacher/pool-v1.flash.jsonl ... --weights runs/DATA_WEIGHTS.json

Copies the shipped ONNX files, the trimmed tokenizer, calibration and the PyTorch checkpoint; the version's
capability report (analysis/capabilities.py) and the cross-version comparison; zips the teacher-labelled data it
was trained on and the LLM-written inputs; writes MANIFEST.json (SHA-256 of every file). The model card is
written by hand next to it (MODEL_CARD.md), since what a version can't do needs words, not numbers.
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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", required=True, help="name of its capability report")
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "runs" / "tokenizer-32k")
    parser.add_argument("--mini", type=Path, help="atelier-mini run folder (adapters, calibration, logs)")
    parser.add_argument("--mini-runtime", type=Path, help="its runtime folder from student/export_mini.py (GGUF, readout, tokenizer)")
    parser.add_argument("--mini-report", help="name of the mini's capability report")
    parser.add_argument("--extra-results", nargs="*", default=[], help="more files from analysis/results to keep with the version")
    parser.add_argument("--integration", action="store_true", help="also keep integration/atelier-on-device-decisions.patch (Atelier's side)")
    parser.add_argument("--dist", type=Path, help="the mini's download folder (tools/mini_dist.py): keeps its manifest.json and miniManifest.ts")
    args = parser.parse_args()

    out = ROOT / "releases" / args.version
    (out / "model").mkdir(parents=True, exist_ok=True)
    (out / "eval").mkdir(exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    onnx = args.model / "onnx"
    for name in ("model.emb4.int8.onnx", "model.int8.onnx", "tokenizer.json", "tokenizer_config.json", "vocab.json", "calibration.json", "report.json"):
        shutil.copy2(onnx / name, out / "model" / name)
    for name in ("student.pt", "config.json", "calibration-report.json", "train.log"):
        if (args.model / name).exists():
            shutil.copy2(args.model / name, out / "model" / name)
    shutil.copy2(args.tokenizer / "keep.json", out / "model" / "tokenizer-keep.json")
    if args.weights and args.weights.exists():
        shutil.copy2(args.weights, out / "model" / "data-weights.json")
    results = ROOT / "analysis" / "results"
    for src in (results / f"capabilities-{args.report}.json", results / f"atelier-eval-{args.report}.txt", results / f"selective-{args.report}.json",
                ROOT / "docs" / "capabilities" / f"{args.report}.md", ROOT / "docs" / "capabilities" / "README.md"):
        if src.exists():
            shutil.copy2(src, out / "eval" / (src.name if src.name != "README.md" else "all-versions.md"))
    if args.mini:
        (out / "mini").mkdir(exist_ok=True)
        for name in ("mini.pt", "report.json", "calibration.json", "calibration-report.json"):
            if (args.mini / name).exists():
                shutil.copy2(args.mini / name, out / "mini" / name)
        log = ROOT / "runs" / f"train-{args.mini.name}.log"
        if log.exists():
            shutil.copy2(log, out / "mini" / "train.log")
        for path in sorted((args.mini_runtime or args.mini / "runtime").iterdir()):
            shutil.copy2(path, out / "mini" / path.name)
    for name in ([f"capabilities-{args.mini_report}.json"] if args.mini_report else []) + list(args.extra_results):
        if (results / name).exists():
            shutil.copy2(results / name, out / "eval" / name)
    if args.mini_report and (ROOT / "docs" / "capabilities" / f"{args.mini_report}.md").exists():
        shutil.copy2(ROOT / "docs" / "capabilities" / f"{args.mini_report}.md", out / "eval" / f"{args.mini_report}.md")
    with zipfile.ZipFile(out / "data" / "teacher-labels.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in args.data:
            z.write(path, path.resolve().relative_to(ROOT).as_posix())
    with zipfile.ZipFile(out / "data" / "llm-written-inputs.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for folder in ("gen", "gen2", "gen3", "gen4", "security_src"):
            for path in sorted((ROOT / "data" / folder).glob("*.jsonl")):
                z.write(path, path.relative_to(ROOT).as_posix())
        if (ROOT / "data" / "existing_families.txt").exists():
            z.write(ROOT / "data" / "existing_families.txt", "data/existing_families.txt")
    if args.integration:
        (out / "integration").mkdir(exist_ok=True)
        shutil.copy2(ROOT / "integration" / "atelier-on-device-decisions.patch", out / "integration" / "atelier-on-device-decisions.patch")
    if args.dist:
        (out / "download").mkdir(exist_ok=True)
        for name in ("manifest.json", "miniManifest.ts"):
            shutil.copy2(args.dist / name, out / "download" / name)
    files = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            files[path.relative_to(out).as_posix()] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / "MANIFEST.json").write_text(json.dumps({"version": args.version, "date": time.strftime("%Y-%m-%d"), "model": str(args.model),
                                                   "files": files}, indent=1))
    total = sum(f["bytes"] for f in files.values())
    print(f"{out}: {len(files)} files, {total / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
