"""Does the split rule cost the mini accuracy on scripts with combining marks?

The teachers (both pplx-decider checkpoints) ship a tokenizer.json with the Qwen2 split rule (letters only), and
transformers 5.17 builds that rule for Qwen3.5 too, so every teacher label and the mini's training used it. The
mini's base, Qwen3.5-0.8B, was pretrained with the Qwen3.5 rule in its own tokenizer.json (letters with their
combining marks), so Hindi, Thai and similar scripts reach the mini in more, smaller pieces than its pretraining saw.

The test: the language probe (unseen MASSIVE utterances, 20 languages, teacher labels) scored by the mini through
llama.cpp with each rule. Languages without combining marks are the control: their token ids, and answers, must not
change.

    python analysis/tokenizer_effect.py --model runs/mini-l24-num --gguf runs/mini-l24-num/mini-l24-num-Q5_K_M-e4.decider.gguf
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student"))
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "third_party"))
from format import fixed_kind, option_labels  # noqa: E402
from readers import verdict  # noqa: E402
from serve_mini import Mini  # noqa: E402
from train import read_jsonl  # noqa: E402

LANGS = ["hi", "th", "ar", "ja", "ru", "en"]  # hi, th: combining marks; the rest: controls


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--base-tokenizer", type=Path, default=ROOT / "models" / "Qwen3.5-0.8B" / "tokenizer.json")
    args = parser.parse_args()
    calibration = json.loads((args.model / "calibration.json").read_text())
    mini = Mini(args.model / "merged", args.gguf, 6, calibration)
    backend = mini.encoder.processor.tokenizer.backend_tokenizer
    rules = {"Qwen2 rule (as trained)": backend.pre_tokenizer, "Qwen3.5 rule (base's file)": Tokenizer.from_file(str(args.base_tokenizer)).pre_tokenizer}
    rows = [r for r in read_jsonl(ROOT / "data" / "teacher" / "probe-lang.flash.jsonl") if r["meta"].get("lang") in LANGS]
    report: dict = {}
    try:
        for name, pre in rules.items():
            backend.pre_tokenizer = pre
            stats: dict[str, list] = defaultdict(lambda: [0, 0, 0, 0, 0])  # n, agree, decided, decided right, tokens
            for r in rows:
                for (key, q), enc in zip(r["request"]["questions"].items(), mini.encoder.plan(r["request"], None, "separate"), strict=True):
                    kind, labels = fixed_kind(key, q) or "open", option_labels(q)
                    v_t = verdict(kind, labels, softmax(np.array(r["logits"][key]) / r["temperature"]).tolist())
                    z = (mini.readout @ mini.hidden(enc.input_ids.tolist())).numpy()[: len(labels)]
                    v = verdict(kind, labels, softmax(z / float(calibration.get(kind, 1.0))).tolist())
                    s = stats[r["meta"]["lang"]]
                    s[0] += 1
                    s[1] += v == v_t
                    s[2] += v is not None
                    s[3] += v is not None and v == v_t
                    s[4] += enc.length
            report[name] = {lang: {"n": s[0], "agree": round(s[1] / s[0], 4), "coverage": round(s[2] / s[0], 4),
                                   "precision": round(s[3] / s[2], 4) if s[2] else None, "tokens": round(s[4] / s[0], 1)}
                            for lang, s in sorted(stats.items())}
            for lang, x in report[name].items():
                prec = f"{x['precision']:.1%}" if x["precision"] is not None else "-"
                print(f"{name:<28} {lang}: agree {x['agree']:.1%}, decides {x['coverage']:.1%}, right when deciding {prec}, "
                      f"{x['tokens']} tokens", flush=True)
    finally:
        mini.close()
    (ROOT / "analysis" / "results" / f"tokenizer-effect-{args.model.name}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
