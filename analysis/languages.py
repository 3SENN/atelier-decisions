# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""How the student does per language, against the teacher, on utterances it never saw (data/pool/probe-lang.jsonl:
20 MASSIVE locales x search-or-ask, voice target, effort), with and without the readability guard of
runtime/node/src/decider.ts (ported here). The guard's thresholds are chosen from this table: decline where the
student's sure answers conflict with the teacher's.

    python analysis/languages.py --model runs/nano-final --probe data/teacher/probe-lang.flash.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student"))
from format import encode, fixed_kind, option_labels, state_text  # noqa: E402
from readers import verdict  # noqa: E402


def unreadable(text: str, ids: list[int], byte_ids: set[int], min_latin: float, max_bytes: float, min_letters: int = 8) -> str | None:
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if len(letters) >= min_letters:
        latin = sum("LATIN" in unicodedata.name(c, "") for c in letters) / len(letters)
        if latin < min_latin:
            return "script"
    if ids and sum(i in byte_ids for i in ids) / len(ids) > max_bytes:
        return "bytes"
    return None


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=ROOT / "data" / "teacher" / "probe-lang.flash.jsonl")
    parser.add_argument("--file", default="model.emb4.int8.onnx")
    parser.add_argument("--min-latin", type=float, default=0.7)
    parser.add_argument("--max-bytes", type=float, default=0.1)
    args = parser.parse_args()

    folder = args.model / "onnx"
    tokenizer = AutoTokenizer.from_pretrained(str(folder))
    byte_ids = {i for t, i in tokenizer.get_vocab().items() if t.startswith("<0x") and t.endswith(">")}
    temps = json.loads((folder / "calibration.json").read_text()) if (folder / "calibration.json").exists() else {}
    session = ort.InferenceSession(str(folder / args.file), providers=["CPUExecutionProvider"])
    stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with args.probe.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            request, lang = row["request"], row["meta"]["lang"]
            text = state_text(request["state"])
            reason = unreadable(text, tokenizer.encode(text, add_special_tokens=False), byte_ids, args.min_latin, args.max_bytes)
            for key, q in request["questions"].items():
                kind, labels = fixed_kind(key, q) or "open", option_labels(q)
                p_t = softmax(np.array(row["logits"][key]) / row["temperature"])
                e = encode(tokenizer, request["state"], key, q, 512)
                ids = np.array([e["input_ids"]], dtype=np.int64)
                scores = session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0]
                p_s = softmax(scores[e["markers"]] / float(temps.get(e["kind"], 1.0)))
                v_t, v_s = verdict(kind, labels, p_t.tolist()), verdict(kind, labels, p_s.tolist())
                v_g = None if reason else v_s
                s = stats[lang]
                s["n"] += 1
                s["agree"] += v_s == v_t
                s["conflict"] += v_s is not None and v_t is not None and v_s != v_t
                s["overreach"] += v_s is not None and v_t is None
                s["teacher_sure"] += v_t is not None
                s["declined"] += reason is not None
                s["agree_guarded"] += v_g == v_t
                s["conflict_guarded"] += v_g is not None and v_t is not None and v_g != v_t
                s["overreach_guarded"] += v_g is not None and v_t is None
    order = ["en", "nl", "de", "fr", "es", "it", "pt", "sv", "id", "pl", "tr", "ru", "el", "ar", "zh", "ja", "ko", "hi", "he", "th"]
    print(f"guard: min Latin {args.min_latin}, max bytes {args.max_bytes}")
    print(f"{'lang':<5} {'n':>4} {'teacher sure':>12} | {'agree':>6} {'conflict':>8} {'overreach':>9} | {'declined':>8} {'agree':>6} {'conflict':>8} {'overreach':>9}")
    for lang in order + sorted(set(stats) - set(order)):
        if lang not in stats:
            continue
        s = stats[lang]
        n = s["n"]
        print(f"{lang:<5} {int(n):>4} {s['teacher_sure'] / n:>12.2f} | {s['agree'] / n:>6.2f} {s['conflict'] / n:>8.3f} {s['overreach'] / n:>9.3f} | "
              f"{s['declined'] / n:>8.2f} {s['agree_guarded'] / n:>6.2f} {s['conflict_guarded'] / n:>8.3f} {s['overreach_guarded'] / n:>9.3f}")


if __name__ == "__main__":
    main()
