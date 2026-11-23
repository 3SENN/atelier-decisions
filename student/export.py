# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Exports the student for devices: vocabulary trimmed to the tokens Atelier's languages use, ONNX, int8.

    python student/export.py --model runs/nano-l8 --vocab-corpus data/pool/pool-v1.jsonl --out runs/nano-l8/onnx
    python student/export.py --fresh-layers 8 --out runs/fresh-l8/onnx      # untrained, for size/latency only

Output folder:
    model.onnx          fp32, input_ids/attention_mask -> per-token option scores (the head runs on every token;
                        the runtime reads the scores at its option markers)
    model.int8.onnx     dynamic int8 matmuls + int8 embedding table
    vocab.json          {"keep": [original token ids in new order], "unk": new id for anything else}
    report.json         sizes, parity with PyTorch, CPU latency per sequence length
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student"))
from format import OPEN, OPT, TAGS, state_text, question_parts  # noqa: E402
from model import NanoDecider  # noqa: E402


class Scores(nn.Module):
    """input_ids, attention_mask -> head score at every position."""

    def __init__(self, model: NanoDecider) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.model.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.model.head(hidden).squeeze(-1)


def keep_ids(tokenizer, corpus: Path | None, min_count: int) -> list[int]:
    """Special/marker tokens, every single-character and byte-fallback piece, and every token seen in the corpus."""
    vocab = tokenizer.get_vocab()
    keep = set(tokenizer.all_special_ids) | {vocab[t] for t in [OPT, OPEN, *TAGS.values()]}
    keep |= {i for t, i in vocab.items() if t.startswith("<0x") or len(t.replace("▁", "")) <= 1}
    if corpus:
        counts: Counter = Counter()
        texts: list[str] = []
        with corpus.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                texts.append(state_text(row["request"]["state"]))
                for key, q in row["request"]["questions"].items():
                    head, options, _ = question_parts(key, q)
                    texts.append(head)
                    texts.extend(options)
        for i in range(0, len(texts), 2048):
            for ids in tokenizer(texts[i : i + 2048], add_special_tokens=False)["input_ids"]:
                counts.update(ids)
        keep |= {i for i, c in counts.items() if c >= min_count}
    return sorted(keep)


def trim(model: NanoDecider, keep: list[int]) -> None:
    emb = model.backbone.embeddings.tok_embeddings
    new = nn.Embedding(len(keep), emb.embedding_dim)
    new.weight.data = emb.weight.data[torch.tensor(keep)].clone()
    model.backbone.embeddings.tok_embeddings = new
    model.backbone.config.vocab_size = len(keep)


def bench(path: Path, lengths: list[int], threads: int, runs: int = 30) -> dict:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    out = {}
    for n in lengths:
        ids = np.random.randint(200, 2000, size=(1, n)).astype(np.int64)
        mask = np.ones((1, n), dtype=np.int64)
        for _ in range(3):
            session.run(None, {"input_ids": ids, "attention_mask": mask})
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            session.run(None, {"input_ids": ids, "attention_mask": mask})
            times.append((time.perf_counter() - t0) * 1000)
        out[n] = round(statistics.median(times), 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--fresh-layers", type=int)
    parser.add_argument("--base", type=Path, default=ROOT / "models" / "mmBERT-small")
    parser.add_argument("--vocab-corpus", type=Path, nargs="?")
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    args = parser.parse_args()

    if args.model:
        model, meta = NanoDecider.load(args.model)
        base = Path(meta["base"])
    else:
        model, base = NanoDecider.from_mmbert(args.base, args.fresh_layers), args.base
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(base))
    args.out.mkdir(parents=True, exist_ok=True)

    keep = keep_ids(tokenizer, args.vocab_corpus, args.min_count) if args.vocab_corpus else list(range(len(tokenizer)))
    remap = {old: new for new, old in enumerate(keep)}
    unk = remap.get(tokenizer.unk_token_id, 0)
    sample_text = "Ik heb twee hotels gevonden. Welke wil je: het Amstel of het Krasnapolsky?"
    sample = [tokenizer.bos_token_id, *tokenizer.encode(sample_text, add_special_tokens=False), tokenizer.eos_token_id]
    with torch.inference_mode():
        ref = Scores(model)(torch.tensor([sample]), torch.ones(1, len(sample), dtype=torch.long)).numpy()
    if len(keep) < len(tokenizer):
        trim(model, keep)
    (args.out / "vocab.json").write_text(json.dumps({"keep": keep, "unk": unk}))
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (args.out / name).write_bytes((base / name).read_bytes())
    if args.model and (args.model / "calibration.json").exists():
        (args.out / "calibration.json").write_bytes((args.model / "calibration.json").read_bytes())

    wrapper = Scores(model).eval()
    ids = torch.tensor([[remap.get(i, unk) for i in sample]])
    mask = torch.ones_like(ids)
    fp32 = args.out / "model.onnx"
    torch.onnx.export(wrapper, (ids, mask), str(fp32), input_names=["input_ids", "attention_mask"],
                      output_names=["scores"], dynamic_axes={"input_ids": {0: "batch", 1: "length"},
                                                             "attention_mask": {0: "batch", 1: "length"},
                                                             "scores": {0: "batch", 1: "length"}},
                      opset_version=18, dynamo=False)

    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8 = args.out / "model.int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, per_channel=True,
                     op_types_to_quantize=["MatMul", "Gather"])

    import onnxruntime as ort

    report: dict = {"vocab": len(keep), "layers": model.backbone.config.num_hidden_layers,
                    "params_body": sum(p.numel() for n, p in model.named_parameters() if "tok_embeddings" not in n),
                    "params_embedding": model.backbone.embeddings.tok_embeddings.weight.numel(),
                    "mb_fp32": round(fp32.stat().st_size / 1e6, 1), "mb_int8": round(int8.stat().st_size / 1e6, 1)}
    for name, path in (("fp32", fp32), ("int8", int8)):
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = session.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0]
        report[f"max_abs_diff_{name}_vs_torch"] = float(np.abs(got - ref).max())
        for threads in args.threads:
            report[f"ms_{name}_{threads}thr"] = bench(path, [16, 64, 128, 256, 512], threads)
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
