"""atelier-mini: the teacher's own architecture at 0.8B, distilled on the same labels as the nano.

The nano (an mmBERT encoder) learns the kinds of questions it was shown but is near chance on new kinds of browser
yes/no questions (analysis/selective.py). This tests whether a small model with a general language prior does
better: Qwen3.5-0.8B (Apache 2.0), the same family as the flash teacher (Qwen3.5-4B), read the decider's way, with
pplx_decider's own prompt (system prompt, "State:", "Question:", lettered options, a <|fim_suffix|> readout per
question), full-attention layers made non-causal, and a linear readout over the answer letters, started from the
LM head's rows for those letters.

Trained with low-rank adapters on every linear layer of the 24 blocks (the base stays frozen in bf16, so it fits a
12 GB card) and a fully trained readout; adapters merge into plain weights for export. Same data, data weights, leak
filters, 5% held-out split and loss as student/train.py, so the numbers compare.

    python student/train_mini.py --data data/teacher/pool-*.flash.jsonl --out runs/mini-l24 --questions 40000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student"))
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "third_party"))
from atelier_uses import use_of  # noqa: E402
from format import fixed_kind, option_labels  # noqa: E402
from pplx_decider.model import enable_noncausal_full_attention  # noqa: E402
from mini_tokenizer import mini_processor  # noqa: E402
from pplx_decider.protocol import Encoder  # noqa: E402
from readers import verdict  # noqa: E402
from train import eval_question_keys, loss_of, question_key, read_jsonl, without_eval  # noqa: E402

PROBES = {
    "open_unseen": ROOT / "data" / "teacher" / "probe-open.flash.jsonl",
    "browser_families": ROOT / "data" / "teacher" / "open-heldout.flash.jsonl",
    "browser_families_2": ROOT / "data" / "teacher" / "gen3-heldout.flash.jsonl",
    "numeric": ROOT / "data" / "teacher" / "numeric-test.truth.jsonl",  # the true answer, not a teacher's
}
FLASH = ROOT / "models" / "pplx-decider-flash-v1.3"


class LoRALinear(torch.nn.Module):
    """y = W x + (B A x) * alpha / r, with W frozen and A, B trained (B starts at zero, so training starts at W)."""

    def __init__(self, base: torch.nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / r
        self.A = torch.nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x.to(self.A.dtype) @ self.A.t() @ self.B.t()).to(x.dtype) * self.scale

    def merged(self) -> torch.Tensor:
        return (self.base.weight.float() + self.scale * self.B @ self.A).to(self.base.weight.dtype)


def add_lora(module: torch.nn.Module, r: int, alpha: float) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, LoRALinear(child, r, alpha))
            count += 1
        else:
            count += add_lora(child, r, alpha)
    return count


class MiniDecider(torch.nn.Module):
    def __init__(self, base: Path, layers: int | None, r: int, alpha: float, token_ids: list[int]) -> None:
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

        full = Qwen3_5Model.from_pretrained(str(base), dtype=torch.bfloat16, attn_implementation="sdpa")
        full.visual = torch.nn.Identity()  # text only
        text = full.language_model
        if layers and layers < len(text.layers):
            keep = keep_layers(text.config.layer_types, layers)
            text.layers = torch.nn.ModuleList([text.layers[i] for i in keep])
            text.config.layer_types = [text.config.layer_types[i] for i in keep]
            text.config.num_hidden_layers = len(keep)
            for i, layer in enumerate(text.layers):  # caches and masks index layers by position
                for m in layer.modules():
                    if hasattr(m, "layer_idx"):
                        m.layer_idx = i
        text.requires_grad_(False)
        self.n_lora = add_lora(text.layers, r, alpha)
        enable_noncausal_full_attention(text)
        self.text = text
        hidden = text.config.hidden_size
        self.readout = torch.nn.Linear(hidden, len(token_ids), bias=False, dtype=torch.float32)
        with torch.no_grad():  # tied embeddings: the LM head's rows for the answer letters
            self.readout.weight.copy_(text.embed_tokens.weight[token_ids].float())

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, markers: torch.Tensor) -> torch.Tensor:
        hidden = self.text(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        return self.readout(hidden[torch.arange(len(markers), device=hidden.device), markers].float())

    def trainable(self) -> list[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


def reference_kernels() -> None:
    """Transformers' own PyTorch linear attention instead of fla's Triton kernels (CPU runs, export)."""
    import inspect

    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update"):
        if hasattr(m, name):
            setattr(m, name, inspect.unwrap(getattr(m, name)))


def keep_layers(types: list[str], n: int) -> list[int]:
    """n of the blocks, evenly spread, always keeping the first and the last."""
    total = len(types)
    picks = sorted({round(i * (total - 1) / (n - 1)) for i in range(n)}) if n > 1 else [total - 1]
    return picks


def encode_rows(rows: list[dict], encoder: Encoder, max_length: int, rng: random.Random | None, skip_eval: bool = True) -> tuple[list[dict], int]:
    held = eval_question_keys() if skip_eval else set()
    out, too_long = [], 0
    for row in rows:
        request = row["request"]
        context = encoder.context(request)
        for key, question in request["questions"].items():
            if held and question_key(question) in held:
                continue
            logits = torch.tensor(row["logits"][key], dtype=torch.float32)
            variants = [(question, logits)]
            if rng is not None and question["type"] == "choice" and len(question["criteria"]) > 1:
                order = list(range(len(question["criteria"])))
                rng.shuffle(order)
                items = list(question["criteria"].items())
                variants.append(({**question, "criteria": {items[i][0]: items[i][1] for i in order}}, logits[order]))
            for q, z in variants:
                enc = encoder.join(context, [(key, q)], [encoder.question_ids(q)])
                if enc.length > max_length:
                    too_long += 1
                    continue
                out.append({"ids": enc.input_ids.tolist(), "marker": int(enc.markers[0]), "teacher": (z / row["temperature"]).tolist(),
                            "labels": option_labels(q), "kind": fixed_kind(key, q) or "open", "type": q["type"],
                            "use": row["meta"].get("use", "?"), "family": row["meta"].get("family", "")})
    return out, too_long


def batches(examples: list[dict], max_tokens: int, rng: random.Random | None) -> list[list[dict]]:
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["ids"]))
    out, cur, longest = [], [], 0
    for i in order:
        n = len(examples[i]["ids"])
        if cur and max(longest, n) * (len(cur) + 1) > max_tokens:
            out.append(cur)
            cur, longest = [], 0
        cur.append(examples[i])
        longest = max(longest, n)
    if cur:
        out.append(cur)
    if rng is not None:
        rng.shuffle(out)
    return out


def collate(batch: list[dict], pad: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left padding, as the decider does; the readout marker moves with it."""
    width = max(len(e["ids"]) for e in batch)
    ids = torch.full((len(batch), width), pad, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    markers = torch.empty(len(batch), dtype=torch.long)
    for i, e in enumerate(batch):
        n = len(e["ids"])
        ids[i, width - n:] = torch.tensor(e["ids"])
        mask[i, width - n:] = 1
        markers[i] = width - n + e["marker"]
    return ids.to(device), mask.to(device), markers.to(device)


@torch.inference_mode()
def scores_of(model: MiniDecider, examples: list[dict], pad: int, device, max_tokens: int) -> list[torch.Tensor]:
    model.eval()
    out: dict[int, torch.Tensor] = {}
    index = {id(e): i for i, e in enumerate(examples)}
    for batch in batches(examples, max_tokens, None):
        ids, mask, markers = collate(batch, pad, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(ids, mask, markers).float().cpu()
        for e, z in zip(batch, logits, strict=True):
            out[index[id(e)]] = z[: len(e["labels"])]
    model.train()
    return [out[i] for i in range(len(examples))]


def verdicts(examples: list[dict], scores: list[torch.Tensor], temperature: float = 1.0) -> dict:
    """Agreement with the teacher's verdict, coverage and precision, per group and over all."""
    groups: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for e, s in zip(examples, scores, strict=True):
        v_t = verdict(e["kind"], e["labels"], torch.tensor(e["teacher"]).softmax(-1).tolist())
        v_s = verdict(e["kind"], e["labels"], (s / temperature).softmax(-1).tolist())
        for g in ("ALL", e["use"] if e["kind"] != "open" else f"open:{e['type']}"):
            c = groups[g]
            c["n"] += 1
            c["agree"] += v_s == v_t
            c["decided"] += v_s is not None
            c["right"] += v_s is not None and v_s == v_t
    return {g: {"n": int(c["n"]), "agree": round(c["agree"] / c["n"], 4), "coverage": round(c["decided"] / c["n"], 4),
                "precision": round(c["right"] / c["decided"], 4) if c["decided"] else None} for g, c in groups.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--extra-eval", type=Path, nargs="*", default=[])
    parser.add_argument("--base", type=Path, default=ROOT / "models" / "Qwen3.5-0.8B")
    parser.add_argument("--layers", type=int, default=None, help="keep this many of the 24 blocks (default all)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=40000, help="training questions sampled (after weights)")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--readout-lr", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--eval-limit", type=int, default=None, help="cap each evaluation set (smoke tests)")
    parser.add_argument("--init", type=Path, help="continue from this mini.pt (same base, blocks and rank)")
    parser.add_argument("--weight", nargs="*", default=[], help="override runs/DATA_WEIGHTS.json: file-stem=copies")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cpu":
        reference_kernels()
    processor = mini_processor(args.base)
    processor.tokenizer.padding_side = "left"
    config = json.loads((FLASH / "decision_config.json").read_text())
    codes, token_ids = list(config["codes"]), list(config["token_ids"])
    tok = processor.tokenizer
    assert all(tok.encode(c, add_special_tokens=False) == [t] for c, t in zip(codes, token_ids, strict=True)), "answer letters differ"
    encoder = Encoder(processor, codes)
    pad = encoder.pad_id

    weights_file = ROOT / "runs" / "DATA_WEIGHTS.json"
    weights = json.loads(weights_file.read_text()) if weights_file.exists() else {}
    weights |= {k: float(v) for k, v in (w.split("=", 1) for w in args.weight)}
    rows = without_eval([row for path in args.data for row in read_jsonl(path) for _ in range(int(weights.get(path.stem, 1)))])
    held = lambda r: int(hashlib.md5(json.dumps(r["request"], sort_keys=True).encode()).hexdigest(), 16) % 20 == 0  # noqa: E731
    train_rows = [r for r in rows if not held(r)]
    test_rows = [r for r in rows if held(r)]
    # Sample requests until about the asked number of questions (choice questions count twice: shuffled copies).
    rng.shuffle(train_rows)
    per_row = sum(len(r["request"]["questions"]) for r in train_rows) / max(1, len(train_rows))
    train_rows = train_rows[: int(args.questions / (per_row * 1.3)) + 1]
    seen = {json.dumps(r["request"], sort_keys=True) for r in train_rows}
    test_rows = [r for r in test_rows if json.dumps(r["request"], sort_keys=True) not in seen]
    rng_test = random.Random(1)
    rng_test.shuffle(test_rows)
    test_rows = test_rows[: args.eval_limit or 3000]

    t0 = time.perf_counter()
    train, dropped = encode_rows(train_rows, encoder, args.max_length, rng)
    test, _ = encode_rows(test_rows, encoder, args.max_length, None)
    evals = {}
    for path in args.extra_eval:
        extra_rows = read_jsonl(path)
        for r in extra_rows:
            r["meta"] = {**r.get("meta", {}), "use": "eval:" + use_of(r["request"])}
        evals[path.stem], _ = encode_rows(extra_rows[: args.eval_limit], encoder, 4096, None, skip_eval=False)
    for name, path in PROBES.items():
        if path.exists():
            evals[name], _ = encode_rows(read_jsonl(path)[: args.eval_limit], encoder, 4096, None)
    print(f"{len(train)} training questions ({dropped} over {args.max_length} tokens dropped), {len(test)} held out; "
          f"encoded in {time.perf_counter() - t0:.0f}s; mean length {sum(len(e['ids']) for e in train) / len(train):.0f} tokens", flush=True)

    model = MiniDecider(args.base, args.layers, args.rank, 2.0 * args.rank, token_ids).to(device)
    if args.init:
        start = torch.load(args.init, map_location="cpu")
        assert (start["layers"], start["rank"]) == (args.layers, args.rank), "--init must match --layers and --rank"
        assert not model.load_state_dict(start["trainable"], strict=False).unexpected_keys
        print(f"continuing from {args.init}", flush=True)
    if args.checkpointing:
        model.text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    n_body = sum(p.numel() for n, p in model.text.named_parameters() if "embed_tokens" not in n and ".A" not in n[-2:] and ".B" not in n[-2:])
    n_train = sum(p.numel() for p in model.trainable())
    print(f"blocks {len(model.text.layers)}, body {n_body / 1e6:.0f}M, {model.n_lora} adapted linears, trainable {n_train / 1e6:.1f}M", flush=True)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("readout")]
    optimizer = torch.optim.AdamW([{"params": lora, "lr": args.lr}, {"params": model.readout.parameters(), "lr": args.readout_lr}],
                                  weight_decay=0.0)
    plan = [b for _ in range(math.ceil(args.epochs)) for b in batches(train, args.max_tokens, rng)]
    plan = plan[: int(len(plan) * args.epochs / math.ceil(args.epochs))]
    warmup = max(1, len(plan) // 30)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / len(plan)))))

    def evaluate(tag: str) -> dict:
        result = {}
        for name, examples in [("held_out", test), *evals.items()]:
            sc = scores_of(model, examples, pad, device, args.max_tokens * 2)
            result[name] = {f"T={t}": verdicts(examples, sc, t)["ALL"] for t in (1.0, 1.5, 2.0, 3.0)}
            result[name]["by_group_T1"] = verdicts(examples, sc, 1.0)
            a = result[name]["T=1.0"]
            print(f"  [{tag}] {name:<22} n={a['n']:>5} agree {a['agree']:.3f}  decides {a['coverage']:.0%}  right when deciding {a['precision']}", flush=True)
        return result

    model.train()
    t0, running = time.perf_counter(), 0.0
    for step, batch in enumerate(plan):
        ids, mask, markers = collate(batch, pad, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(ids, mask, markers)
        scores = [z[: len(e["labels"])] for z, e in zip(logits, batch, strict=True)]
        loss = loss_of(scores, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable(), 1.0)
        optimizer.step()
        schedule.step()
        running = 0.98 * running + 0.02 * loss.item() if step else loss.item()
        if step % 50 == 0 or step == len(plan) - 1:
            rate = sum(len(b) for b in plan[: step + 1]) / (time.perf_counter() - t0)
            left = (len(plan) - step - 1) / max(1e-9, (step + 1) / (time.perf_counter() - t0)) / 60
            mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
            print(f"step {step}/{len(plan)} loss {running:.4f} {rate:.1f} q/s ~{left:.0f} min left, peak {mem:.1f} GiB", flush=True)
        if args.eval_every and step and step % args.eval_every == 0:
            evaluate(f"step {step}")

    print("== final", flush=True)
    result = evaluate("final")
    args.out.mkdir(parents=True, exist_ok=True)
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save({"trainable": state, "base": str(args.base), "layers": args.layers, "rank": args.rank,
                "kept": keep_layers(json.loads((args.base / "config.json").read_text())["text_config"]["layer_types"], args.layers) if args.layers else None},
               args.out / "mini.pt")
    (args.out / "report.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "result": result}, indent=1))
    print(f"saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
