# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Where the mini is sure and contradicts the teacher on browser question families it never saw, from the scores a
capability or cascade run cached (runs/<mini>/scores-<gguf stem>.json), without running the model again.

    python analysis/family_errors.py --mini runs/mini-v05 --gguf-stem mini-v05-Q5_K_M-e4.decider

Per held-out set and per source (the Fable agent that wrote it: shopping, work, world, agent), per question type and
per kind of answer the teacher gave: decides, right when deciding, sure-and-contradicts. Then the families with the
most sure contradictions, with one example each. Writes analysis/results/family-errors-<mini>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student"))
from format import fixed_kind, option_labels  # noqa: E402
from readers import verdict  # noqa: E402

SETS = {"browser_families": "open-heldout", "browser_families_2": "gen3-heldout", "browser_families_3": "gen4-heldout"}


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", type=Path, required=True)
    parser.add_argument("--gguf-stem", required=True)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # family examples are in 20 languages; python -I ignores PYTHONIOENCODING
    scores =json.loads((args.mini / f"scores-{args.gguf_stem}.json").read_text())
    temps = json.loads((args.mini / "calibration.json").read_text())
    temps = temps.get("temperatures", temps)
    out: dict = {}
    for name, stem in SETS.items():
        groups: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # n, decided, right, sure-and-contradicts
        families: dict[str, list] = defaultdict(lambda: [0, 0, None])
        missing = 0
        with (ROOT / "data" / "teacher" / f"{stem}.flash.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                state = r["request"]["state"]
                for key, q in r["request"]["questions"].items():
                    h = hashlib.md5(json.dumps([state, key, q], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    if h not in scores:
                        missing += 1
                        continue
                    kind, labels = fixed_kind(key, q) or "open", option_labels(q)
                    teacher = verdict(kind, labels, softmax(np.array(r["logits"][key]) / r["temperature"]).tolist())
                    mini = verdict(kind, labels, softmax(np.array(scores[h]) / float(temps.get(kind, 1.0))).tolist())
                    meta = r["meta"]
                    answer = teacher if teacher is not None else "unsure"
                    if q.get("type") == "noul" or len(labels) == 2:
                        answer_group = f"teacher {answer}"
                    else:
                        answer_group = "teacher picks one" if teacher is not None else "teacher unsure"
                    contradicts = mini is not None and teacher is not None and mini != teacher
                    for g in ("ALL", f"source {meta.get('source', '?')}", f"type {q.get('type', '?')}", answer_group,
                              f"lang {'en' if meta.get('lang', 'en') == 'en' else 'other'}"):
                        b = groups[g]
                        b[0] += 1
                        b[1] += mini is not None
                        b[2] += mini is not None and mini == teacher
                        b[3] += contradicts
                    fam = families[meta.get("family", "?")]
                    fam[0] += 1
                    fam[1] += contradicts
                    if contradicts and fam[2] is None:
                        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
                        fam[2] = {"state": text[:300], "question": q.get("instructions", "")[:200], "teacher": teacher, "mini": mini}
        table = {g: {"n": n, "decides": d / n, "right": (ok / d if d else None), "sure_contradicts": c / n} for g, (n, d, ok, c) in groups.items()}
        worst = sorted(((v[1] / v[0], v[1], v[0], k, v[2]) for k, v in families.items() if v[1]), reverse=True)[: args.top]
        out[name] = {"groups": table, "worst_families": [{"family": k, "n": n, "contradicts": c, "example": ex} for _, c, n, k, ex in worst],
                     "missing": missing}
        print(f"== {name} ({missing} questions without a cached score)")
        for g, t in sorted(table.items(), key=lambda kv: (kv[0] != "ALL", kv[0])):
            right = f"{t['right']:.1%}" if t["right"] is not None else "-"
            print(f"  {g:<28} n={t['n']:>4}  decides {t['decides']:>5.1%}  right {right:>6}  sure and contradicts {t['sure_contradicts']:.1%}")
        for w in out[name]["worst_families"][:8]:
            print(f"    {w['family']:<44} {w['contradicts']}/{w['n']}  teacher {w['example']['teacher']} mini {w['example']['mini']}: {w['example']['question'][:90]}")
    dst = ROOT / "analysis" / "results" / f"family-errors-{args.mini.name}.json"
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
