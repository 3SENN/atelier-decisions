"""Does llama.cpp compute atelier-mini as PyTorch does, and how fast, at each quantization?

llama-server runs the merged GGUF with non-causal attention and last-token pooling (the final, normed hidden state
at the <|fim_suffix|> marker, which ends every question sequence); the readout is applied here. The same token ids
go through the trained PyTorch model (CPU, fp32) as the reference.

    python analysis/mini_parity.py --model runs/mini-l24 --gguf runs/mini-l24/mini-l24-Q8_0.gguf runs/mini-l24/mini-l24-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student"))
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "third_party"))
from format import fixed_kind, option_labels  # noqa: E402
from mini_tokenizer import mini_processor  # noqa: E402
from pplx_decider.protocol import Encoder  # noqa: E402
from readers import verdict  # noqa: E402
from train import read_jsonl  # noqa: E402
from train_mini import MiniDecider, reference_kernels  # noqa: E402

SERVER = ROOT / "models" / "llama.cpp-b12620" / "bin" / "llama-server.exe"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def embed(port: int, ids: list[int]) -> list[float]:
    body = json.dumps({"content": ids, "embd_normalize": -1}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/embedding", body, {"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=120).read())
    item = out[0] if isinstance(out, list) else out
    e = item["embedding"]
    return e[0] if isinstance(e[0], list) else e


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, nargs="+", required=True)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    reference_kernels()
    torch.set_num_threads(args.threads)
    saved = torch.load(args.model / "mini.pt", map_location="cpu")
    base = Path(saved["base"])
    flash = json.loads((ROOT / "models" / "pplx-decider-flash-v1.3" / "decision_config.json").read_text())
    processor = mini_processor(base)
    encoder = Encoder(processor, flash["codes"])
    rows = read_jsonl(ROOT / "data" / "teacher" / "open-heldout.flash.jsonl") + read_jsonl(ROOT / "data" / "teacher" / "eval-26b.jsonl")
    random.Random(3).shuffle(rows)
    items = []
    for r in rows:
        context = encoder.context(r["request"])
        for key, q in r["request"]["questions"].items():
            enc = encoder.join(context, [(key, q)], [encoder.question_ids(q)])
            labels = option_labels(q)
            kind = fixed_kind(key, q) or "open"
            teacher = verdict(kind, labels, (torch.tensor(r["logits"][key]) / r["temperature"]).softmax(-1).tolist())
            items.append({"ids": enc.input_ids.tolist(), "labels": labels, "kind": kind, "teacher": teacher})
        if len(items) >= args.n:
            break
    items = items[: args.n]

    model = MiniDecider(base, saved["layers"], saved["rank"], 2.0 * saved["rank"], flash["token_ids"]).float().eval()
    missing = model.load_state_dict(saved["trainable"], strict=False)
    assert not missing.unexpected_keys
    readout = load_file(str(args.model / "merged" / "readout.safetensors"))["weight"]
    ref = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for it in items:
            ids = torch.tensor([it["ids"]])
            z = model(ids, torch.ones_like(ids), torch.tensor([len(it["ids"]) - 1]))[0]
            ref.append(z[: len(it["labels"])].softmax(-1))
    t_ref = (time.perf_counter() - t0) / len(items)
    del model
    tokens = sum(len(it["ids"]) for it in items) / len(items)
    print(f"{len(items)} questions, mean {tokens:.0f} tokens; PyTorch fp32 reference {t_ref * 1000:.0f} ms/question")

    def score(probs: list[torch.Tensor]) -> tuple[float, float]:
        decided = right = 0
        for it, p in zip(items, probs, strict=True):
            v = verdict(it["kind"], it["labels"], p.tolist())
            decided += v is not None
            right += v is not None and v == it["teacher"]
        return decided / len(items), (right / decided if decided else float("nan"))

    cov, prec = score(ref)
    print(f"  reference: decides {cov:.0%}, right {prec:.3f} (against the teacher, this sample)")
    for gguf in args.gguf:
        port = free_port()
        server = subprocess.Popen([str(SERVER), "-m", str(gguf), "--embeddings", "--pooling", "last",
                                   "-c", "4096", "-b", "4096", "-ub", "4096", "-t", str(args.threads), "--port", str(port),
                                   "--host", "127.0.0.1", "-np", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            for _ in range(240):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                    break
                except Exception:
                    if server.poll() is not None:
                        raise RuntimeError(server.stderr.read().decode(errors="replace")[-2000:])
                    time.sleep(0.5)
            embed(port, items[0]["ids"])  # warm-up
            probs, t0 = [], time.perf_counter()
            for it in items:
                h = torch.tensor(embed(port, it["ids"]))
                probs.append((readout @ h)[: len(it["labels"])].softmax(-1))
            t = (time.perf_counter() - t0) / len(items)
        finally:
            server.terminate()
            server.wait(timeout=30)
        top = sum(int(p.argmax() == r.argmax()) for p, r in zip(probs, ref, strict=True)) / len(items)
        diff = max(float((p - r).abs().max()) for p, r in zip(probs, ref, strict=True))
        same_verdict = sum(verdict(it["kind"], it["labels"], p.tolist()) == verdict(it["kind"], it["labels"], r.tolist())
                           for it, p, r in zip(items, probs, ref, strict=True)) / len(items)
        cov, prec = score(probs)
        print(f"  {gguf.name} ({gguf.stat().st_size / 1e6:.0f} MB): {t * 1000:.0f} ms/question ({args.threads} threads), "
              f"top answer = reference {top:.1%}, verdict = reference {same_verdict:.1%}, max prob diff {diff:.3f}; "
              f"decides {cov:.0%}, right {prec:.3f}")


if __name__ == "__main__":
    main()
