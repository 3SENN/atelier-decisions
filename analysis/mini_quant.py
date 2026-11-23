# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Which quantization of atelier-mini should ship? Each GGUF answers the same questions through llama.cpp; compare
coverage and precision against the teacher, and how often each file's verdict equals the first file's.

    python analysis/mini_quant.py --model runs/mini-l24 --gguf runs/mini-l24/mini-l24-Q8_0.decider.gguf \
        runs/mini-l24/mini-l24-Q4_K_M-e4.decider.gguf

Sets: both held-out browser-family sets, new kinds of questions, and every 4th held-out pool request (Atelier's own
decisions). Writes analysis/results/mini-quant-<model>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from capabilities import POOLS, PROBES, MiniModel, held_out, read_jsonl, score_rows  # noqa: E402
from format import fixed_kind, option_labels  # noqa: E402
from readers import verdict  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    sets = {
        "browser_families": read_jsonl(PROBES["open_browser_families"]),
        "browser_families_2": read_jsonl(PROBES["open_browser_families_2"]),
        "new_kinds": read_jsonl(PROBES["open_unseen"]),
        "held_out_quarter": held_out(POOLS)[::4],
    }
    results, first = {}, None
    for gguf in args.gguf:
        model = MiniModel(args.model, gguf)
        out, verdicts, t0, n = {}, {}, time.perf_counter(), 0
        for name, rows in sets.items():
            out[name] = score_rows(model, rows, lambda r: "x", guard=False)["ALL"]
            for i, r in enumerate(rows):
                for key, q in r["request"]["questions"].items():
                    p, _ = model.probs(r["request"]["state"], key, q)
                    verdicts[f"{name}/{i}/{key}"] = verdict(fixed_kind(key, q) or "open", option_labels(q), p.tolist())
                    n += 1
        ms = (time.perf_counter() - t0) * 1000 / n
        model.close()
        first = first or verdicts
        same = sum(verdicts[k] == first[k] for k in first) / len(first)
        results[gguf.name] = {"mb": round(gguf.stat().st_size / 1e6, 1), "ms_per_question": round(ms, 1),
                              "same_verdict_as_first": round(same, 4), **out}
        print(f"{gguf.name} ({results[gguf.name]['mb']} MB, {ms:.0f} ms/question, same verdict as first {same:.1%})")
        for name, row in out.items():
            print(f"  {name:<20} n={row['n']:>5} agree {row['agree']:.1%} decides {row['coverage']:.1%} "
                  f"right {row['precision']:.1%} sure-and-contradicts {row['conflict']:.1%}", flush=True)
    path = ROOT / "analysis" / "results" / f"mini-quant-{args.model.name}.json"
    path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
