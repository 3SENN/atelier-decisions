"""The two models as one decider: atelier-nano answers first (milliseconds), and whatever it declines or is unsure
about goes to atelier-mini; what the mini is unsure about goes to the rule. How right is that, and how often does a
question need the mini?

    python analysis/cascade.py --nano runs/nano-v03-l22 --mini runs/mini-l24-150k \
        --gguf runs/mini-l24-150k/mini-l24-150k-Q8_0.decider.gguf --name v0.3+mini-150k

The nano behaves as shipped (runtime/node): it declines text under 70% Latin script and every question that is not
one of Atelier's own (open questions). Per set and per decision: teacher agreement, coverage, precision and sure
answers that contradict the teacher, for the nano alone, the mini alone and the cascade, and the share of questions
that reach the mini. On the numeric suite the "teacher" is the true answer. Writes analysis/results/cascade-<name>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from capabilities import POOLS, PROBES, MiniModel, Model, held_out, read_jsonl, softmax  # noqa: E402
from format import fixed_kind, option_labels  # noqa: E402
from readers import verdict  # noqa: E402

# Suites answered by code or by construction: the "teacher" is the true answer.
TRUTH = {name: ROOT / "data" / "teacher" / f"{name}-test.truth.jsonl" for name in ("numeric", "numeric2", "security")}


def stats(pairs: list[tuple]) -> dict:
    n = len(pairs)
    decided = [(s, t) for s, t in pairs if s is not None]
    return {"n": n, "agree": round(sum(s == t for s, t in pairs) / n, 4), "coverage": round(len(decided) / n, 4),
            "precision": round(sum(s == t for s, t in decided) / len(decided), 4) if decided else None,
            "conflict": round(sum(t is not None and s != t for s, t in decided) / n, 4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nano", type=Path, required=True)
    parser.add_argument("--nano-file", default="model.emb4.int8.onnx")
    parser.add_argument("--mini", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--mini-first", nargs="*", default=[], help="decisions the mini answers directly (runtime/node --mini-first)")
    args = parser.parse_args()
    folder = args.nano / "onnx" if (args.nano / "onnx").is_dir() else args.nano
    nano, mini = Model(folder, args.nano_file), MiniModel(args.mini, args.gguf)
    sets = {"held_out": held_out(POOLS), **{k: read_jsonl(p) for k, p in PROBES.items() if p.exists()}}
    sets |= {name: read_jsonl(path) for name, path in TRUTH.items() if path.exists()}
    report: dict = {"name": args.name, "nano": str(args.nano), "mini": str(args.mini), "gguf": args.gguf.name}
    for name, rows in sets.items():
        groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for r in rows:
            state = r["request"]["state"]
            declined = nano.unreadable(state)
            for key, q in r["request"]["questions"].items():
                kind, labels = fixed_kind(key, q) or "open", option_labels(q)
                v_t = verdict(kind, labels, softmax(np.array(r["logits"][key]) / r["temperature"]).tolist())
                v_n = None if declined or kind == "open" or kind in args.mini_first else verdict(kind, labels, nano.probs(state, key, q)[0].tolist())
                v_m = verdict(kind, labels, mini.probs(state, key, q)[0].tolist())
                v_c = v_n if v_n is not None else v_m
                for g in ("ALL", r["meta"]["use"] if name == "held_out" else None):
                    if g:
                        for model, v in (("nano", v_n), ("mini", v_m), ("cascade", v_c)):
                            groups[g][model].append((v, v_t))
                        groups[g]["reaches_mini"].append(v_n is None)
        report[name] = {g: {m: stats(p) for m, p in d.items() if m != "reaches_mini"}
                        | {"reaches_mini": round(sum(d["reaches_mini"]) / len(d["reaches_mini"]), 4)} for g, d in groups.items()}
        a = report[name]["ALL"]
        print(f"{name:<24} n={a['nano']['n']:>5}  reaches mini {a['reaches_mini']:.0%}")
        for m in ("nano", "mini", "cascade"):
            s = a[m]
            prec = f"{s['precision']:.1%}" if s["precision"] is not None else "-"
            print(f"    {m:<8} agree {s['agree']:.1%}  decides {s['coverage']:.1%}  right when deciding {prec:>6}  "
                  f"sure and contradicts {s['conflict']:.1%}", flush=True)
        if name == "held_out":
            for g, d in sorted(report[name].items()):
                if g != "ALL":
                    print(f"      {g:<12} reaches mini {d['reaches_mini']:.0%}  " + "  ".join(
                        f"{m} {d[m]['coverage']:.0%}/{(d[m]['precision'] or 0):.1%}" for m in ("nano", "mini", "cascade")))
    mini.close()
    (ROOT / "analysis" / "results" / f"cascade-{args.name}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
