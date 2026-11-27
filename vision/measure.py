# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""How often atelier-vision decides, how often it is right when it does, and how often it is sure and wrong.

    .venv/bin/python -I vision/measure.py --emb runs/vision/emb/base --name vision-base                  # SigLIP 2 as it comes
    .venv/bin/python -I vision/measure.py --emb runs/vision/emb/v1 --adapters runs/vision-v1/vision.pt \\
        --cal runs/vision-v1/vision.json --name vision-v1
    .venv/bin/python -I vision/measure.py --emb runs/vision/emb/v1-onnx --runtime runs/vision-v1/onnx --name vision-v0.1   # as shipped

Reads the held-out questions (vision/data.py eval) and the pictures' vectors (vision/embed.py), runs every answer's
text through the text tower, and reads the probabilities the way Atelier does (vision/decide.py: yes at 0.7, no at
0.3, a choice at 0.75). Only half b of the pictures is reported: half a is what vision/calibrate.py fitted on.
Writes analysis/results/<name>.json and .txt.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

import numpy as np  # noqa: E402

from decide import RAW, probabilities, verdict  # noqa: E402
from embed import load  # noqa: E402
from format import labels, texts  # noqa: E402

CACHE = ROOT / "data" / "cache" / "vision"
RESULTS = ROOT / "analysis" / "results"


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def picture_vectors(folder: Path, rows: list[dict]) -> dict[str, np.ndarray]:
    """"<pack>/<id>" to its vector, for the packs the rows use."""
    out = {}
    for pack in sorted({r["image"].split("/", 1)[0] for r in rows}):
        ids, vectors = load(folder, pack)
        out.update({f"{pack}/{k}": v for k, v in zip(ids, vectors)})
    return out


def text_vectors(rows: list[dict], adapters: Path | None, runtime=None) -> dict[str, np.ndarray]:
    """Every text any question matches against, as its vector."""
    wanted = sorted({t for r in rows for t in texts(r["q"]) if t})
    if runtime is not None:
        runtime.read(wanted)  # together, in batches
        return {t: runtime.text(t) for t in wanted}
    from model import Towers, device, embed_texts, tokenizer

    dev = device()
    model = Towers.load(adapters=adapters).to(dev)
    vectors = embed_texts(model, tokenizer(), wanted, dev)
    return dict(zip(wanted, vectors))


def cosines(row: dict, pictures: dict[str, np.ndarray], words: dict[str, np.ndarray]) -> list[float | None]:
    v = pictures[row["image"]]
    v = v / np.linalg.norm(v)
    return [None if t is None else float(v @ words[t]) for t in texts(row["q"])]


def judge(rows: list[dict], pictures, words, scale: float, bias: float, cal: dict) -> list[dict]:
    """Each row with p (per label), right (the likeliest answer is the true one) and sure (Atelier's verdict, or None)."""
    out = []
    for r in rows:
        names = labels(r["q"])
        p = probabilities(r["kind"], cosines(r, pictures, words), scale, bias, cal)
        truth = -1 if r["answer"] is None else names.index(r["answer"])  # None: no option is right (a button that is none of the jobs)
        v = verdict(r["kind"], p)
        out.append({**r, "p": p, "right": int(max(range(len(p)), key=lambda i: p[i]) == truth), "sure": v, "sure_right": None if v is None else int(v == truth)})
    return out


def summary(judged: list[dict]) -> dict:
    n = len(judged)
    if not n:
        return {"n": 0}
    sure = [j for j in judged if j["sure"] is not None]
    return {"n": n, "right": round(sum(j["right"] for j in judged) / n, 4), "decides": round(len(sure) / n, 4),
            "right_when_deciding": round(sum(j["sure_right"] for j in sure) / len(sure), 4) if sure else None,
            "sure_and_wrong": round(sum(1 - j["sure_right"] for j in sure) / n, 4)}


def tables(judged: list[dict]) -> dict:
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for j in judged:
        if j["split"] == "heldout-none":
            # No right answer to be sure of: its own table, and not in the totals.
            groups["a button that is none of the 63 jobs"]["named anyway (which of all 63?)" if j["form"] == "every job" else "is it this job? (the answer is no)"].append(j)
            continue
        groups["split"][j["split"]].append(j)
        if j["form"] != "every job":  # Atelier's own question about buttons has its own table
            groups["family, split, type"][f"{j['family']} / {j['split']} / {j['kind']}"].append(j)
        groups["language"][j["lang"]].append(j)
        groups["form"][f"{j['kind']} / {j['form']}"].append(j)
        groups["wording"]["never trained" if j["unseen_wording"] else "trained"].append(j)
        if j["kind"] == "choice":
            groups["options"]["2" if j["n"] == 2 else "3-4" if j["n"] <= 4 else "5-8" if j["n"] <= 8 else "all 63 jobs of a button"].append(j)
        if j["form"] == "every job":
            groups["a button among all 63 jobs"][j["split"]].append(j)
    return {"all": summary([j for j in judged if j["split"] != "heldout-none"]), **{name: {k: summary(v) for k, v in sorted(g.items())} for name, g in groups.items()}}


def report(result: dict) -> str:
    def line(name: str, s: dict) -> str:
        if not s["n"]:
            return f"  {name:44s} none"
        rwd = "-" if s["right_when_deciding"] is None else f"{100 * s['right_when_deciding']:.1f}%"
        return f"  {name:44s} {s['n']:6d}  right {100 * s['right']:5.1f}%  decides {100 * s['decides']:5.1f}%  right when deciding {rwd:>6s}  sure and wrong {100 * s['sure_and_wrong']:4.1f}%"

    out = [line("all held-out questions (half b)", result["all"])]
    for name, group in result.items():
        if name in ("all", "meta"):
            continue
        out.append(f"by {name}:")
        out += [line(k, s) for k, s in group.items()]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=CACHE / "eval.jsonl")
    parser.add_argument("--adapters", type=Path)
    parser.add_argument("--runtime", type=Path, help="an exported folder: its text file and its calibration (the pictures' vectors from embed.py --runtime)")
    parser.add_argument("--cal", type=Path, help="a vision.json with calibration; without it, SigLIP's own reading")
    parser.add_argument("--half", default="b", choices=["a", "b", "all"])
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    rows = [r for r in read_rows(args.questions) if args.half == "all" or r["half"] == args.half]
    cal_file = args.cal or (args.runtime / "vision.json" if args.runtime else None)
    info = json.loads(cal_file.read_text(encoding="utf-8")) if cal_file else {}
    cal = info.get("calibration", RAW)
    if args.runtime:
        from runtime import Runtime

        rt = Runtime(args.runtime, threads=8, text_cache=(args.emb / "texts.npz") if args.emb else None)
        declined = [r for r in rows if rt.decline(r["q"])]
        rows = [r for r in rows if not rt.decline(r["q"])]
        judged = judge(rows, picture_vectors(args.emb, rows), text_vectors(rows, None, rt), rt.scale, rt.bias, cal)
        rt.save_texts()
    else:
        from model import Towers

        base, declined = Towers.load(), []
        judged = judge(rows, picture_vectors(args.emb, rows), text_vectors(rows, args.adapters), base.scale, base.bias, cal)
    result = {"meta": {"questions": args.questions.name, "half": args.half, "pictures": str(args.emb), "adapters": str(args.adapters) if args.adapters else None,
                       "runtime": str(args.runtime) if args.runtime else None, "declined_by_the_runtime": len(declined), "calibration": cal}, **tables(judged)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    text = report(result)
    (RESULTS / f"{args.name}.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
