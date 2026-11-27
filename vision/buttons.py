# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Atelier's rule for a button the page gave no name, measured as Atelier applies it.

In a browser task, a link or button with no name in the page (only an icon) is handed to atelier-vision
(Atelier's src/main/agent/tools/browserSteps.ts nameUnnamed):

    1. which of all 63 jobs is it, by name, in the page's language?   taken when one job has `choice` of the weight
    2. is it really a button for that job? (a yes/no with what a yes looks like)   taken when the yes has `yes`

Named only when both hold; otherwise the button stays out of the step, as every unnamed button did before. The
second question is there for buttons that are none of the jobs (a logo, an anchor): among 63 options something
always looks likeliest.

For every held-out button, in each of the five languages: named right, named wrong or left alone, and for the
buttons that are none of the jobs: named anyway or left alone. Several cutoffs, so the ones Atelier ships with are
a measured choice.

    .venv/bin/python -I vision/buttons.py --emb runs/vision/emb/v1 --adapters runs/vision-v1/vision.pt --cal runs/vision-v1/vision.json --name vision-buttons-v1
    .venv/bin/python -I vision/buttons.py --emb runs/vision/emb/v1-onnx --runtime runs/vision-v1/onnx --name vision-buttons-v0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

import numpy as np  # noqa: E402

import icons  # noqa: E402
from decide import RAW, probabilities  # noqa: E402
from embed import load  # noqa: E402
from questions import ASK, LANGS, SAY  # noqa: E402
from store import Pack  # noqa: E402

CACHE = ROOT / "data" / "cache" / "vision"
RESULTS = ROOT / "analysis" / "results"
RULES = [(0.75, None), (0.75, 0.5), (0.75, 0.7), (0.9, None), (0.9, 0.7), (0.9, 0.9), (0.95, 0.9)]
SPLITS = ["heldout-items", "heldout-look", "heldout-kinds", "heldout-none"]


def gate(job: str, lang: str, words: dict) -> dict:
    """The second question, as Atelier words it: one of the trained ways to ask and to say what a yes looks like."""
    name = words[job][lang][0]
    return {"type": "noul", "instructions": ASK["control"][lang][1].format(name=name, x=name), "criteria": {"true": SAY["control"][lang][0].format(name=name, x=name)}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", type=Path, required=True)
    parser.add_argument("--adapters", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--cal", type=Path)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    words = icons.words()
    jobs = sorted(words)
    pack = Pack(CACHE / "controls")
    ids, vectors = load(args.emb, "controls")
    at = {k: i for i, k in enumerate(ids)}
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    names = {lang: [words[j][lang][0] for j in jobs] for lang in LANGS}
    gates = {lang: [gate(j, lang, words)["criteria"]["true"] for j in jobs] for lang in LANGS}
    wanted = sorted({t for lang in LANGS for t in names[lang] + gates[lang]})
    if args.runtime:
        from runtime import Runtime

        rt = Runtime(args.runtime, threads=8, text_cache=(args.emb / "texts.npz") if args.emb else None)
        rt.read(wanted)
        rt.save_texts()
        text = {t: rt.text(t) for t in wanted}
        scale, bias, cal = rt.scale, rt.bias, rt.cal
    else:
        from model import Towers, device, embed_texts, tokenizer

        dev = device()
        model = Towers.load(adapters=args.adapters).to(dev)
        text = dict(zip(wanted, embed_texts(model, tokenizer(), wanted, dev)))
        scale, bias = model.scale, model.bias
        cal = json.loads(args.cal.read_text(encoding="utf-8")).get("calibration", RAW) if args.cal else RAW
    name_m = {lang: np.stack([text[t] for t in names[lang]]) for lang in LANGS}
    gate_m = {lang: np.stack([text[t] for t in gates[lang]]) for lang in LANGS}
    counts: dict[tuple, dict[str, int]] = {}
    by_lang: dict[tuple, dict[str, int]] = {}
    for key, row in pack.rows.items():
        if row["split"] not in SPLITS or key not in at:
            continue
        v = vectors[at[key]]
        for lang in LANGS:
            p = probabilities("choice", list(map(float, name_m[lang] @ v)), scale, bias, cal)
            top = int(np.argmax(p))
            yes = probabilities("noul", [None, float(gate_m[lang][top] @ v)], scale, bias, cal)[1]
            for rule in RULES:
                named = p[top] >= rule[0] and (rule[1] is None or yes >= rule[1])
                outcome = "left alone" if not named else "named anyway" if row["label"] == "none" else "named right" if jobs[top] == row["label"] else "named wrong"
                for table, k in ((counts, (row["split"], rule)), (by_lang, (row["split"], rule, lang))):
                    table.setdefault(k, {}).setdefault(outcome, 0)
                    table[k][outcome] += 1
    result: dict = {"meta": {"pictures": str(args.emb), "runtime": str(args.runtime) if args.runtime else None, "adapters": str(args.adapters) if args.adapters else None, "languages": LANGS}}
    lines = ["a button the page gave no name, asked which of all 63 jobs it is and then whether it really is that one, in each of five languages",
             "cutoffs (job, yes): named right / named wrong / left alone, of the buttons that have one of the jobs; named anyway, of the ones that have none"]
    share = lambda c, k: c.get(k, 0) / max(1, sum(c.values()))
    for rule in RULES:
        label = f"{rule[0]:.2f}, {'no second question' if rule[1] is None else f'{rule[1]:.2f}'}"
        result[label] = {}
        parts = []
        for split in SPLITS:
            c = counts.get((split, rule), {})
            if split == "heldout-none":
                result[label][split] = {"n": sum(c.values()), "named_anyway": round(share(c, "named anyway"), 4)}
                parts.append(f"none of the jobs: named anyway {100 * share(c, 'named anyway'):.1f}%")
            else:
                named = c.get("named right", 0) + c.get("named wrong", 0)
                result[label][split] = {"n": sum(c.values()), "named_right": round(share(c, "named right"), 4), "named_wrong": round(share(c, "named wrong"), 4),
                                        "left_alone": round(share(c, "left alone"), 4), "right_when_named": round(c.get("named right", 0) / named, 4) if named else None}
                parts.append(f"{split.replace('heldout-', '')}: {100 * share(c, 'named right'):.1f} / {100 * share(c, 'named wrong'):.1f} / {100 * share(c, 'left alone'):.1f}")
        result[label]["by language (icon sets never seen)"] = {lang: {"named_right": round(share(by_lang.get(("heldout-look", rule, lang), {}), "named right"), 4),
                                                                      "named_wrong": round(share(by_lang.get(("heldout-look", rule, lang), {}), "named wrong"), 4)} for lang in LANGS}
        lines.append(f"  ({label:26s}) " + " · ".join(parts))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (RESULTS / f"{args.name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
