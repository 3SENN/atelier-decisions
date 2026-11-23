# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Layer-streaming decider: runs a checkpoint far larger than GPU memory, exactly (BF16, reference math).

The 26B decider needs ~45 GiB for its decoder alone; this GPU has 12. Instead of offloading per forward pass, all
sequences of a job are embedded once, and the decoder layers are then loaded from disk ONE AT A TIME; every
sequence goes through layer i before layer i+1 is read. One pass over the weights (~49 GB of reads) serves the
whole job, so the 26B labels thousands of sequences per hour on a 12 GB card.

While it runs it also measures, per layer, what the layer does to the decision:

- ``cos_marker`` / ``cos_tokens``: cosine between the layer's input and output at the readout marker / over all
  tokens (1 - cos is "block influence", the ShortGPT importance score);
- ``lens_kl`` / ``lens_top1``: the readout applied to that layer's output (after the final norm, a "readout lens"):
  KL from the final answer distribution, and how often its top option already equals the final one.

    python teacher/stream.py --checkpoint models/pplx-decider-v1.3 --requests data/teacher/eval-flash.jsonl \
        --out data/teacher/eval-26b.jsonl --stats analysis/results/layers-26b.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor
from transformers.masking_utils import create_recurrent_attention_mask
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))

from pplx_decider.protocol import MAX_OPTIONS, Encoded, Encoder, answer, left_pad, load_json  # noqa: E402


class Weights:
    """Tensors of a sharded safetensors checkpoint, read on demand."""

    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint
        self.where = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]

    def group(self, prefix: str, device: str) -> dict[str, torch.Tensor]:
        by_file: dict[str, list[str]] = defaultdict(list)
        for name, file in self.where.items():
            if name.startswith(prefix):
                by_file[file].append(name)
        out = {}
        for file, names in by_file.items():
            with safe_open(str(self.checkpoint / file), framework="pt", device=device) as f:
                for name in names:
                    out[name[len(prefix):]] = f.get_tensor(name)
        return out


class StreamingDecider:
    def __init__(self, checkpoint: Path, device: str = "cuda", batch_tokens: int = 16384) -> None:
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.batch_tokens = batch_tokens
        config = load_json(checkpoint / "decision_config.json")
        self.temperature = float(config["temperature"])
        self.codes = list(config["codes"])
        processor = AutoProcessor.from_pretrained(str(checkpoint))
        processor.tokenizer.padding_side = "left"
        self.encoder = Encoder(processor, self.codes)
        self.config = AutoConfig.from_pretrained(str(checkpoint)).text_config
        self.config._attn_implementation = "sdpa"
        self.weights = Weights(checkpoint)
        torch.backends.cuda.enable_cudnn_sdp(False)  # the reference path's SDPA kernel choice

    def _layer(self, index: int) -> torch.nn.Module:
        with torch.device("meta"):
            layer = Qwen3_5DecoderLayer(self.config, index)
        state = self.weights.group(f"language_model.layers.{index}.", str(self.device))
        layer.load_state_dict(state, strict=True, assign=True)
        return layer.to(self.device, torch.bfloat16).eval()

    @torch.inference_mode()
    def run(self, requests: list[dict], stats: bool = True) -> tuple[list[dict[str, list[float]]], dict]:
        """Raw readout logits (valid options, before temperature) per request and question, plus layer stats."""
        t0 = time.perf_counter()
        plans = [self.encoder.plan(request, None, "separate") for request in requests]
        flat: list[tuple[int, Encoded]] = [(i, chunk) for i, plan in enumerate(plans) for chunk in plan]
        order = sorted(range(len(flat)), key=lambda k: flat[k][1].length)
        batches: list[list[int]] = []
        for k in order:
            if batches and (len(batches[-1]) + 1) * flat[k][1].length <= self.batch_tokens:
                batches[-1].append(k)
            else:
                batches.append([k])
        tokens = sum(chunk.length for _, chunk in flat)
        print(f"{len(requests)} requests, {len(flat)} sequences, {tokens:,} tokens, {len(batches)} batches",
              flush=True)

        # Embed every batch once; hidden states stay on the GPU between layers.
        embed = self.weights.group("language_model.embed_tokens.", str(self.device))["weight"]
        rotary = Qwen3_5TextRotaryEmbedding(config=self.config, device=self.device)
        work = []
        for batch in batches:
            chunks = [flat[k][1] for k in batch]
            inputs = left_pad(chunks, self.encoder.pad_id)
            ids = inputs["input_ids"].to(self.device)
            padding = inputs["attention_mask"].to(self.device)
            hidden = F.embedding(ids, embed)
            width = ids.shape[1]
            positions = torch.arange(width, device=self.device).view(1, 1, -1).expand(4, len(batch), -1)
            markers = torch.stack([chunk.markers[-1] + width - chunk.length for chunk in chunks]).to(self.device)
            work.append({
                "hidden": hidden,
                "positions": positions[0],
                "rope": rotary(hidden, positions[1:]),
                "masks": {
                    "full_attention": padding[:, None, None, :].bool(),
                    "linear_attention": create_recurrent_attention_mask(
                        config=self.config, inputs_embeds=hidden, attention_mask=padding),
                },
                "valid": padding.bool(),
                "markers": markers,
                "counts": [chunk.counts[-1] for chunk in chunks],
            })
        del embed
        torch.cuda.empty_cache()

        norm = Qwen3_5RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps).to(self.device, torch.bfloat16)
        norm.load_state_dict(self.weights.group("language_model.norm.", str(self.device)))
        readout = torch.nn.Linear(self.config.hidden_size, MAX_OPTIONS, bias=False, dtype=torch.bfloat16)
        with safe_open(str(self.checkpoint / "readout.safetensors"), framework="pt", device=str(self.device)) as f:
            readout.load_state_dict({key: f.get_tensor(key) for key in f.keys()})
        readout = readout.to(self.device)

        def at_markers(item: dict, hidden: torch.Tensor) -> torch.Tensor:
            rows = torch.arange(hidden.shape[0], device=self.device)
            return hidden[rows, item["markers"]]

        n_layers = self.config.num_hidden_layers
        lens: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
        cos_marker = torch.zeros(n_layers)
        cos_tokens = torch.zeros(n_layers)
        load_s = 0.0
        for index in range(n_layers):
            t_load = time.perf_counter()
            layer = self._layer(index)
            load_s += time.perf_counter() - t_load
            kind = self.config.layer_types[index]
            for item in work:
                before = item["hidden"]
                after = layer(before, position_embeddings=item["rope"], attention_mask=item["masks"][kind],
                              position_ids=item["positions"], past_key_values=None, use_cache=False)
                if stats:
                    cos_marker[index] += F.cosine_similarity(at_markers(item, before).float(),
                                                             at_markers(item, after).float(), dim=-1).sum().cpu()
                    cos = F.cosine_similarity(before.float(), after.float(), dim=-1)
                    cos_tokens[index] += (cos * item["valid"]).sum().cpu() / item["valid"].sum().cpu() * len(cos)
                    lens[index].append(readout(norm(at_markers(item, after))).float().cpu())
                item["hidden"] = after
            del layer
            torch.cuda.empty_cache()
            done = index + 1
            elapsed = time.perf_counter() - t0
            print(f"layer {done}/{n_layers} ({kind}) {elapsed:.0f}s elapsed, ~{elapsed / done * (n_layers - done):.0f}s "
                  f"left (reading weights {load_s:.0f}s)", flush=True)

        # Final answer logits, back in the original sequence order.
        final_rows: list[torch.Tensor] = []
        counts: list[int] = []
        for item in work:
            final_rows.append(readout(norm(at_markers(item, item["hidden"]))).float().cpu())
            counts.extend(item["counts"])
        final = torch.cat(final_rows)
        position = {k: j for j, k in enumerate(k for batch in batches for k in batch)}
        results: list[dict[str, list[float]]] = [{} for _ in requests]
        for k, (index, chunk) in enumerate(flat):
            j = position[k]
            results[index][chunk.keys[-1]] = final[j, : counts[j]].tolist()

        report: dict = {"sequences": len(flat), "tokens": tokens, "seconds": round(time.perf_counter() - t0, 1),
                        "weight_read_seconds": round(load_s, 1), "layer_types": self.config.layer_types}
        if stats:
            n = len(flat)
            kl, top1 = [], []
            for index in range(n_layers):
                layer_logits = torch.cat(lens[index])
                kls, agree = [], []
                for j in range(n):
                    c = counts[j]
                    p = (final[j, :c] / self.temperature).log_softmax(-1)
                    q = (layer_logits[j, :c] / self.temperature).log_softmax(-1)
                    kls.append(float((p.exp() * (p - q)).sum()))
                    agree.append(int(p.argmax() == q.argmax()))
                kl.append(sum(kls) / n)
                top1.append(sum(agree) / n)
            report.update({"cos_marker": (cos_marker / n).tolist(), "cos_tokens": (cos_tokens / n).tolist(),
                           "lens_kl": kl, "lens_top1": top1})
        return results, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True, help="JSONL with a 'request' (or the request itself)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--batch-tokens", type=int, default=16384)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--chunk", type=int, default=0,
                        help="requests per pass over the weights (0: all at once); hidden states of a chunk stay on the GPU")
    args = parser.parse_args()

    with args.requests.open(encoding="utf-8") as f:  # not splitlines(): JSON strings may hold U+2028
        rows = [json.loads(line) for line in f if line.strip()]
    rows = rows[: args.limit] if args.limit else rows
    decider = StreamingDecider(args.checkpoint, batch_tokens=args.batch_tokens)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    if args.out.exists() and args.chunk:  # resume after the last finished chunk
        with args.out.open(encoding="utf-8") as f:
            done = sum(1 for line in f if line.strip())
    size = args.chunk or len(rows)
    report: dict = {}
    for start in range(done, len(rows), size):
        part = rows[start : start + size]
        requests = [row.get("request", row) for row in part]
        logits, report = decider.run(requests, stats=args.stats is not None and not args.chunk)
        with args.out.open("a" if args.chunk else "w", encoding="utf-8") as f:
            for row, request, values in zip(part, requests, logits, strict=True):
                answers = {key: answer(q, (torch.tensor(values[key]) / decider.temperature).softmax(-1).tolist())
                           for key, q in request["questions"].items()}
                f.write(json.dumps({**({"meta": row["meta"]} if "meta" in row else {}), "model": args.checkpoint.name,
                                    "request": request, "logits": values, "temperature": decider.temperature,
                                    "answers": answers}, ensure_ascii=False) + "\n")
        print(f"chunk done: {start + len(part)}/{len(rows)} requests", flush=True)
    if args.stats and report.get("lens_kl"):
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}))


if __name__ == "__main__":
    main()
