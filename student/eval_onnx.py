# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""What quantization costs, in Atelier's terms: each exported ONNX variant on the held-out split (the same 5% the
trainer holds out), compared with the teacher and with the fp32 export, one question per pass as the runtime does.

    python student/eval_onnx.py --model runs/nano-v13-l8 --data data/teacher/pool-v1.flash.jsonl data/teacher/pool-v3.flash.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student"))
from readers import verdict  # noqa: E402
from train import examples_of, read_jsonl, without_eval  # noqa: E402


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--variants", nargs="+", default=["model.onnx", "model.int8.onnx", "model.emb4.int8.onnx"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    folder = args.model / "onnx"
    meta = json.loads(json.dumps(__import__("torch").load(args.model / "student.pt", map_location="cpu", weights_only=True)["meta"]))
    tokenizer = AutoTokenizer.from_pretrained(str(folder))
    temps = json.loads((folder / "calibration.json").read_text()) if (folder / "calibration.json").exists() else {}
    rows = without_eval([r for p in args.data for r in read_jsonl(p)])
    held = [r for r in rows if int(hashlib.md5(json.dumps(r["request"], sort_keys=True).encode()).hexdigest(), 16) % 20 == 0]
    examples = examples_of(held, tokenizer, meta["max_length"], random.Random(0), augment=False)
    if args.limit:
        examples = examples[: args.limit]

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    probs: dict[str, list[np.ndarray]] = {}
    report = {}
    for variant in args.variants:
        session = ort.InferenceSession(str(folder / variant), options, providers=["CPUExecutionProvider"])
        out, times = [], []
        for e in examples:
            ids = np.array([e["input_ids"]], dtype=np.int64)
            t0 = time.perf_counter()
            scores = session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0]
            times.append((time.perf_counter() - t0) * 1000)
            out.append(softmax(scores[e["markers"]] / float(temps.get(e["kind"], 1.0))))
        probs[variant] = out
        report[variant] = {"mb": round((folder / variant).stat().st_size / 1e6, 1),
                           "median_ms": round(float(np.median(times)), 2), "p90_ms": round(float(np.percentile(times, 90)), 2)}

    reference = args.variants[0]
    stats: dict[str, dict[str, dict[str, float]]] = {v: defaultdict(lambda: defaultdict(float)) for v in args.variants}
    for i, e in enumerate(examples):
        p_t = softmax(np.array(e["teacher"]))
        v_t = verdict(e["kind"], e["labels"], p_t.tolist())
        v_ref = verdict(e["kind"], e["labels"], probs[reference][i].tolist())
        for variant in args.variants:
            p = probs[variant][i]
            v = verdict(e["kind"], e["labels"], p.tolist())
            for name in (e["use"], "ALL"):
                s = stats[variant][name]
                s["n"] += 1
                s["verdict_vs_teacher"] += float(v == v_t)
                s["conflict"] += float(v is not None and v_t is not None and v != v_t)
                s["overreach"] += float(v is not None and v_t is None)
                s["verdict_vs_fp32"] += float(v == v_ref)
                s["dp_vs_fp32"] += float(np.abs(p - probs[reference][i]).max())
    for variant in args.variants:
        print(f"\n{variant}: {report[variant]}")
        print(f"{'use':<12} {'n':>5} {'vs teacher':>10} {'conflict':>8} {'overreach':>9} {'vs fp32':>8} {'|dp| fp32':>9}")
        for name, s in sorted(stats[variant].items(), key=lambda kv: (kv[0] == "ALL", kv[0])):
            n = s["n"]
            print(f"{name:<12} {int(n):>5} {s['verdict_vs_teacher'] / n:>10.3f} {s['conflict'] / n:>8.3f} "
                  f"{s['overreach'] / n:>9.3f} {s['verdict_vs_fp32'] / n:>8.3f} {s['dp_vs_fp32'] / n:>9.4f}")


if __name__ == "__main__":
    main()
