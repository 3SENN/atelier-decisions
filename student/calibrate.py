# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Fits one temperature per decision kind on held-out teacher labels. Writes <model>/calibration.json.

Two objectives, both reported:
- "kl": the student's distribution closest to the teacher's (KL), the textbook choice;
- "verdict" (default): Atelier's costs. A verdict the student is sure of and the teacher contradicts (conflict)
  is a wrong decision; sure where the teacher was unsure (overreach) risks one; unsure where the teacher was sure
  (underreach) only hands the decision back to the rule. Cost = 4 x conflict + 1.5 x overreach + 1 x underreach.

    python student/calibrate.py --model runs/nano-l8 --data data/teacher/pool-v1.flash.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student"))
from model import NanoDecider  # noqa: E402
from readers import verdict  # noqa: E402
from train import batches, collate, examples_of, read_jsonl, split_logits, without_eval  # noqa: E402

COST = {"conflict": 4.0, "overreach": 1.5, "underreach": 1.0}
# Per decision, what each kind of miss costs in Atelier (src/main/decisions/uses.ts):
# - lasting: the model only ever skips a chat; skipping one the teacher would read loses a memory, and Atelier
#   says a missed memory costs more than a longer read. Reading anyway is cheap.
# - dictation / target: the model only ever routes away from the safe default (type it in / into the field);
#   a wrong route garbles what was said, the default costs a little time.
# - kind (the click guard): the model only ever adds a stop; missing one before a purchase is the costly side.
COSTS = {
    "lasting": {"conflict": 4.0, "overreach": 3.0, "underreach": 0.5},
    "dictation": {"conflict": 4.0, "overreach": 2.0, "underreach": 0.5},
    "target": {"conflict": 4.0, "overreach": 2.0, "underreach": 0.5},
    "kind": {"conflict": 1.0, "overreach": 1.0, "underreach": 2.0},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--objective", choices=["verdict", "kl"], default="verdict")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, meta = NanoDecider.load(args.model)
    model.eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(meta["base"])
    rows = without_eval([r for p in args.data for r in read_jsonl(p)])
    held = [r for r in rows if int(hashlib.md5(json.dumps(r["request"], sort_keys=True).encode()).hexdigest(), 16) % 20 == 0]
    examples = examples_of(held, tokenizer, meta["max_length"], random.Random(0), augment=False)
    by_kind: dict[str, list[tuple[torch.Tensor, torch.Tensor, list[str]]]] = defaultdict(list)
    with torch.inference_mode():
        for batch in batches(examples, 16384, None):
            b = collate(batch, tokenizer.pad_token_id, args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                flat = model(b["input_ids"], b["attention_mask"], b["rows"], b["cols"])
            for s, e in zip(split_logits(flat.float().cpu(), b["sizes"]), batch, strict=True):
                by_kind[e["kind"]].append((s, torch.tensor(e["teacher"]).softmax(-1), e["labels"]))

    grid = [round(0.5 + 0.05 * i, 2) for i in range(51)]  # 0.5 .. 3.0
    temperatures, report = {}, {}
    for kind, items in by_kind.items():
        teacher_verdicts = [verdict(kind, labels, p.tolist()) for _, p, labels in items]

        def kl(t: float) -> float:
            return sum(float((p * (p.clamp_min(1e-9).log() - (s / t).log_softmax(-1))).sum()) for s, p, _ in items) / len(items)

        def verdicts(t: float) -> dict[str, float]:
            out = {"agree": 0.0, "conflict": 0.0, "overreach": 0.0, "underreach": 0.0}
            for (s, _, labels), v_t in zip(items, teacher_verdicts, strict=True):
                v = verdict(kind, labels, (s / t).softmax(-1).tolist())
                out["agree"] += v == v_t
                out["conflict"] += v is not None and v_t is not None and v != v_t
                out["overreach"] += v is not None and v_t is None
                out["underreach"] += v is None and v_t is not None
            return {k: round(x / len(items), 4) for k, x in out.items()}

        weights = COSTS.get(kind, COST)

        def cost(t: float) -> float:
            v = verdicts(t)
            return sum(weights[k] * v[k] for k in weights)

        t_kl = min(grid, key=kl)
        t_verdict = min(grid, key=lambda t: (round(cost(t), 4), abs(t - 1.0)))  # ties: stay near 1
        chosen = t_verdict if args.objective == "verdict" else t_kl
        temperatures[kind] = chosen
        report[kind] = {"n": len(items), "chosen": chosen, "T_kl": t_kl, "T_verdict": t_verdict,
                        "at_1": verdicts(1.0), "at_T_kl": verdicts(t_kl), "at_T_verdict": verdicts(t_verdict)}
    (args.model / "calibration.json").write_text(json.dumps(temperatures, indent=1))
    (args.model / "calibration-report.json").write_text(json.dumps(report, indent=1))
    print(f"{'kind':<11} {'n':>5} {'T_kl':>5} {'T_v':>5}   verdict agree / conflict / overreach / underreach at T=1 -> T_kl -> T_verdict")
    for kind, r in sorted(report.items()):
        fmt = lambda v: f"{v['agree']:.3f}/{v['conflict']:.3f}/{v['overreach']:.3f}/{v['underreach']:.3f}"  # noqa: E731
        print(f"{kind:<11} {r['n']:>5} {r['T_kl']:>5} {r['T_verdict']:>5}   {fmt(r['at_1'])} -> {fmt(r['at_T_kl'])} -> {fmt(r['at_T_verdict'])}")


if __name__ == "__main__":
    main()
