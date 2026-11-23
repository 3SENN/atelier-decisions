# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""How strict must the model be on open questions (not Atelier's own) to be right most of the time when it decides?

A softer temperature for open questions makes Atelier's readers (yes at 0.7, a choice at 0.75, a level at 0.6) see
the model as unsure more often: it decides less, but is right more often when it does. This sweeps that temperature
on each held-out set of question KINDS the model never trained on, and picks it honestly: fitted on one held-out
set, measured on another.

    python analysis/selective.py --model runs/nano-v02-l22 [--target 0.9]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from capabilities import PROBES, Model, softmax  # noqa: E402
from format import encode, fixed_kind, option_labels  # noqa: E402
from readers import verdict  # noqa: E402
from train import read_jsonl  # noqa: E402

SETS = ["open_unseen", "open_browser_families", "open_browser_families_2"]
GRID = [round(0.5 + 0.1 * i, 2) for i in range(56)]  # 0.5 .. 6.0


def collect(model: Model, rows: list[dict]) -> list[dict]:
    """Raw option scores once per open question (no temperature), with the teacher's verdict."""
    out = []
    for r in rows:
        declined = model.unreadable(r["request"]["state"])
        for key, q in r["request"]["questions"].items():
            if fixed_kind(key, q):
                continue
            labels = option_labels(q)
            item = {"labels": labels, "type": q["type"], "family": r["meta"].get("family", ""), "declined": declined,
                    "teacher": verdict("open", labels, softmax(np.array(r["logits"][key]) / r["temperature"]).tolist())}
            if not declined:
                e = encode(model.tok, r["request"]["state"], key, q, 512)
                ids = np.array([e["input_ids"]], dtype=np.int64)
                item["scores"] = model.session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0][e["markers"]]
            out.append(item)
    return out


def at(items: list[dict], t: float) -> dict:
    decided = right = 0
    for it in items:
        if it["declined"]:
            continue
        v = verdict("open", it["labels"], softmax(it["scores"] / t).tolist())
        decided += v is not None
        right += v is not None and v == it["teacher"]
    n = len(items)
    return {"n": n, "coverage": round(decided / n, 4) if n else 0.0, "precision": round(right / decided, 4) if decided else None,
            "right_of_all": round(right / n, 4) if n else 0.0}


def fit(items: list[dict], target: float) -> float | None:
    """The sharpest temperature whose precision reaches the target (most decisions at that precision)."""
    ok = [t for t in GRID if (p := at(items, t)["precision"]) is not None and p >= target]
    return min(ok) if ok else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--file", default="model.emb4.int8.onnx")
    parser.add_argument("--target", type=float, nargs="+", default=[0.85, 0.9, 0.95])
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    folder = args.model / "onnx" if (args.model / "onnx").exists() else args.model
    model = Model(folder, args.file)
    t_now = float(model.temps.get("open", 1.0))
    data = {s: collect(model, read_jsonl(PROBES[s])) for s in SETS if PROBES[s].exists()}
    report: dict = {"model": str(args.model), "T_open_now": t_now, "sets": {}, "fits": []}
    print(f"open-question temperature now: {t_now}")
    for s, items in data.items():
        by_type = {ty: at([i for i in items if i["type"] == ty], t_now) for ty in ("noul", "choice", "score")}
        curve = {t: at(items, t) for t in GRID}
        report["sets"][s] = {"now": at(items, t_now), "now_by_type": by_type, "curve": {str(t): v for t, v in curve.items()}}
        now = report["sets"][s]["now"]
        print(f"\n{s} ({now['n']} questions): now decides {now['coverage']:.0%}, right when deciding {now['precision']}")
        print("  by type: " + ", ".join(f"{ty} {v['coverage']:.0%}/{v['precision']}" for ty, v in by_type.items() if v["n"]))
        print("  T     decides  right-when-deciding")
        for t in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0):
            c = curve[t]
            print(f"  {t:<5} {c['coverage']:>6.0%}   {c['precision']}")
    names = list(data)
    for target in args.target:
        for a in names:
            t = fit(data[a], target)
            for b in names:
                if b == a:
                    continue
                res = at(data[b], t) if t else None
                report["fits"].append({"target": target, "fit_on": a, "test_on": b, "T": t, "result": res})
                if res:
                    print(f"target {target:.0%}: T={t} fitted on {a} -> {b}: decides {res['coverage']:.0%}, right {res['precision']}")
                else:
                    print(f"target {target:.0%}: {a} never reaches it")
    name = args.name or args.model.name
    out = ROOT / "analysis" / "results" / f"selective-{name}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
