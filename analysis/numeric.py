"""Numeric decisions against the true answer (data/build_numeric.py): does a model, or a teacher, get budgets,
totals, ratings, dates, opening hours, file sizes, limits, unit prices and counts right?

    python analysis/numeric.py --name v0.3-l22 --model runs/nano-v03-l22             # nano (ONNX, like capabilities.py)
    python analysis/numeric.py --name mini-0.8b-40k --model runs/mini-l24 --gguf runs/mini-l24/mini-l24-Q8_0.decider.gguf
    python analysis/numeric.py --name flash --labels data/teacher/numeric-test.flash.jsonl   # a teacher's labels

Reads data/teacher/numeric-test.truth.jsonl (the true answer as a sure distribution, so "agree" means right and
"conflict" means sure and wrong) and reports, per split (families it trained on / families it never saw), family
and margin (how close the numbers are to the threshold): decides, right when it decides, sure and wrong.
Writes analysis/results/numeric-<name>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from capabilities import MiniModel, Model, read_jsonl, score_rows, softmax  # noqa: E402
from format import fixed_kind  # noqa: E402

TRUTH = ROOT / "data" / "teacher" / "numeric-test.truth.jsonl"


class Labels:
    """A teacher's labelled file read as a model."""

    def __init__(self, path: Path) -> None:
        self.rows = {json.dumps(r["request"], sort_keys=True, ensure_ascii=False): r for r in read_jsonl(path)}

    def unreadable(self, state) -> bool:
        return False

    def probs(self, state, key: str, q: dict) -> tuple[np.ndarray, str]:
        r = self.rows[json.dumps({"state": state, "questions": {key: q}}, sort_keys=True, ensure_ascii=False)]
        return softmax(np.array(r["logits"][key]) / r["temperature"]), fixed_kind(key, q) or "open"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--file", default="model.emb4.int8.onnx")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--suite", default="numeric", help="numeric (data/build_numeric.py) or numeric2 (multi-step, data/build_numeric2.py)")
    args = parser.parse_args()
    if args.labels:
        model = Labels(args.labels)
    elif args.gguf:
        model = MiniModel(args.model, args.gguf)
    else:
        folder = args.model / "onnx" if (args.model / "onnx").is_dir() else args.model
        model = Model(folder, args.file)
    rows = read_jsonl(TRUTH.with_name(f"{args.suite}-test.truth.jsonl"))
    report = {"name": args.name,
              "by_split": score_rows(model, rows, lambda r: r["meta"]["split"], guard=False),
              "by_family": score_rows(model, rows, lambda r: f"{r['meta']['split']}/{r['meta']['family']}", guard=False),
              "by_margin": score_rows(model, rows, lambda r: f"{r['meta']['split']}/{r['meta']['margin']}", guard=False)}
    if args.gguf:
        model.close()
    (ROOT / "analysis" / "results" / f"{args.suite}-{args.name}.json").write_text(json.dumps(report, indent=1))
    for part in ("by_split", "by_margin", "by_family"):
        print(f"{args.name} {part}")
        for group, row in sorted(report[part].items()):
            prec = f"{row['precision']:.1%}" if row["precision"] is not None else "-"
            print(f"  {group:<52} n={row['n']:>4} right {row['agree']:.1%}  decides {row['coverage']:.1%}  "
                  f"right when deciding {prec:>6}  sure and wrong {row['conflict']:.1%}")


if __name__ == "__main__":
    main()
