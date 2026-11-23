# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Distils a teacher's decisions into the student.

    python student/train.py --data data/teacher/pool-v1.flash.jsonl --layers 8 --out runs/nano-l8

Loss per question: KL(teacher || student) on the teacher's calibrated distribution (what Atelier's cutoffs read),
plus 0.5 * tau^2 * KL at tau=2 for the shape of the rest of the distribution. Choice options are shuffled in a
second copy of every choice question so the student does not learn the order. Held-out requests (5%, by hash)
measure what matters for Atelier: whether the student's VERDICT (sure answer, or unsure so the rule decides) is the
teacher's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student"))
sys.path.insert(0, str(ROOT / "data"))
from atelier_uses import use_of  # noqa: E402
from format import encode, fixed_kind, option_labels  # noqa: E402
from model import NanoDecider  # noqa: E402
from readers import verdict  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    """Not str.splitlines(): JSON strings may hold U+2028 and similar line separators."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# Atelier's eval cases (tests/fixtures/decisions-eval.json, as Atelier sent them to the teacher). Nothing whose
# state is one of them is trained or calibrated on, so Atelier's eval stays a test of what the student learned.
EVAL_REQUESTS = ROOT / "data" / "teacher" / "eval-flash.jsonl"


def state_key(state) -> str:
    text = state if isinstance(state, str) else json.dumps(state, sort_keys=True, ensure_ascii=False)
    return re.sub(r"\s+", " ", text.lower()).strip(" ?.!")


# Fields that are the task's shared vocabulary (a site, a role, a page or field name, label_items' context), not
# a test case's own text: training on www.youtube.com or a "Checkout" page is not seeing an eval case.
SHARED_FIELDS = {"site", "role", "page_title", "focused_field", "context"}


def state_values(state) -> set[str]:
    """A state's distinctive texts: free-text fields (what was said, typed or titled) of four words or more."""
    if not isinstance(state, dict):
        return set()
    keys = {state_key(v) for k, v in state.items() if k not in SHARED_FIELDS and isinstance(v, str)}
    return {k for k in keys if len(k.split()) >= 4}


def without_eval(rows: list[dict]) -> list[dict]:
    """Leaves out every request that shares a distinctive text with one of Atelier's eval cases: the same reply,
    sentence, title or query, even with other fields around it (a dictated sentence into another field)."""
    if not EVAL_REQUESTS.exists():
        return rows
    held: set[str] = set()
    for r in read_jsonl(EVAL_REQUESTS):
        held |= state_values(r["request"]["state"]) | {state_key(r["request"]["state"])}
    kept = [r for r in rows if not (state_values(r["request"]["state"]) & held or state_key(r["request"]["state"]) in held)]
    if len(kept) < len(rows):
        print(f"left out {len(rows) - len(kept)} requests that share a text with Atelier's eval cases", flush=True)
    return kept


_EVAL_QUESTIONS: set[str] | None = None


def eval_question_keys() -> set[str]:
    """The user-provided questions of Atelier's eval cases (skill descriptions, label_items questions). Atelier's
    own fixed questions are the same for everyone and are not counted; these are, so no training question that
    repeats one is learned from: the eval then measures matching new skills and new questions, as in real use."""
    global _EVAL_QUESTIONS
    if _EVAL_QUESTIONS is None:
        _EVAL_QUESTIONS = set()
        if EVAL_REQUESTS.exists():
            for r in read_jsonl(EVAL_REQUESTS):
                for key, q in r["request"]["questions"].items():
                    if fixed_kind(key, q) in (None, "skill"):
                        _EVAL_QUESTIONS.add(question_key(q))
    return _EVAL_QUESTIONS


def question_key(question: dict) -> str:
    """A question by its wording, whatever the order of its options."""
    criteria = question.get("criteria")
    if isinstance(criteria, dict):
        criteria = sorted(criteria.items(), key=lambda kv: kv[0])
    return json.dumps([question.get("type"), question.get("instructions"), criteria], ensure_ascii=False, default=str)


def examples_of(rows: list[dict], tokenizer, max_length: int, rng: random.Random, augment: bool, skip_eval_questions: bool = True) -> list[dict]:
    out = []
    held = eval_question_keys() if skip_eval_questions else set()
    for row in rows:
        request, temperature = row["request"], row["temperature"]
        for key, question in request["questions"].items():
            if held and question_key(question) in held:
                continue
            logits = torch.tensor(row["logits"][key], dtype=torch.float32)
            variants = [(question, logits)]
            if augment and question["type"] == "choice" and len(question["criteria"]) > 1:
                order = list(range(len(question["criteria"])))
                rng.shuffle(order)
                items = list(question["criteria"].items())
                variants.append(({**question, "criteria": {items[i][0]: items[i][1] for i in order}}, logits[order]))
            for q, z in variants:
                enc = encode(tokenizer, request["state"], key, q, max_length)
                out.append({**enc, "teacher": (z / temperature).tolist(), "labels": option_labels(q),
                            "use": row["meta"]["use"], "lang": row["meta"].get("lang", "?")})
    return out


def batches(examples: list[dict], max_tokens: int, rng: random.Random | None) -> list[list[dict]]:
    order = sorted(examples, key=lambda e: len(e["input_ids"]))
    out, current = [], []
    for e in order:
        if current and (len(current) + 1) * len(e["input_ids"]) > max_tokens:
            out.append(current)
            current = []
        current.append(e)
    if current:
        out.append(current)
    if rng:
        rng.shuffle(out)
    return out


def collate(batch: list[dict], pad: int, device) -> dict:
    width = max(len(e["input_ids"]) for e in batch)
    ids = torch.full((len(batch), width), pad, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    rows, cols, sizes = [], [], []
    for i, e in enumerate(batch):
        ids[i, : len(e["input_ids"])] = torch.tensor(e["input_ids"])
        mask[i, : len(e["input_ids"])] = 1
        rows += [i] * len(e["markers"])
        cols += e["markers"]
        sizes.append(len(e["markers"]))
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device), "rows": torch.tensor(rows, device=device),
            "cols": torch.tensor(cols, device=device), "sizes": sizes}


def split_logits(flat: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
    return list(torch.split(flat, sizes))


def loss_of(student: list[torch.Tensor], batch: list[dict]) -> torch.Tensor:
    total = 0.0
    for s, e in zip(student, batch, strict=True):
        t = torch.tensor(e["teacher"], device=s.device)
        p_t, log_s = t.softmax(-1), s.float().log_softmax(-1)
        kl1 = (p_t * (p_t.clamp_min(1e-9).log() - log_s)).sum()
        p_t2, log_s2 = (t / 2).softmax(-1), (s.float() / 2).log_softmax(-1)
        kl2 = (p_t2 * (p_t2.clamp_min(1e-9).log() - log_s2)).sum()
        total = total + kl1 + 0.5 * 4 * kl2
    return total / len(batch)


@torch.inference_mode()
def evaluate(model, examples: list[dict], pad: int, device, max_tokens: int) -> dict:
    device = torch.device(device)
    model.eval()
    stats: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for batch in batches(examples, max_tokens, None):
        b = collate(batch, pad, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            flat = model(b["input_ids"], b["attention_mask"], b["rows"], b["cols"])
        for s, e in zip(split_logits(flat.float(), b["sizes"]), batch, strict=True):
            p_s = s.softmax(-1).cpu()
            p_t = torch.tensor(e["teacher"]).softmax(-1)
            kind = e.get("kind", "open")
            v_s = verdict(kind, e["labels"], p_s.tolist())
            v_t = verdict(kind, e["labels"], p_t.tolist())
            for name in (e["use"], "ALL"):
                st = stats[name]
                st["n"] += 1
                st["top1"] += float(p_s.argmax() == p_t.argmax())
                st["dp"] += float((p_s - p_t).abs().max())
                st["verdict_agree"] += float(v_s == v_t)
                st["student_sure"] += float(v_s is not None)
                st["teacher_sure"] += float(v_t is not None)
                st["conflict"] += float(v_s is not None and v_t is not None and v_s != v_t)
                st["overreach"] += float(v_s is not None and v_t is None)
    model.train()
    return {name: {k: (v / st["n"] if k != "n" else int(v)) for k, v in st.items()} for name, st in stats.items()}


def report(metrics: dict) -> str:
    lines = [f"{'use':<12} {'n':>6} {'top1':>6} {'|dp|':>6} {'verdict':>8} {'sure s/t':>11} {'conflict':>8} {'overreach':>9}"]
    for name in sorted(metrics, key=lambda k: (k == "ALL", k)):
        m = metrics[name]
        lines.append(f"{name:<12} {m['n']:>6} {m['top1']:>6.3f} {m['dp']:>6.3f} {m['verdict_agree']:>8.3f} "
                     f"{m['student_sure']:>5.2f}/{m['teacher_sure']:<5.2f} {m['conflict']:>8.3f} {m['overreach']:>9.3f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--extra-eval", type=Path, nargs="*", default=[], help="labelled JSONL evaluated, never trained")
    parser.add_argument("--base", type=Path, default=ROOT / "models" / "mmBERT-small")
    parser.add_argument("--tokenizer", type=Path, help="a trimmed tokenizer folder (student/trim_tokenizer.py)")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, help="use only the first N labelled requests")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer_dir = args.tokenizer or args.base
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    vocab = json.loads((args.tokenizer / "keep.json").read_text()) if args.tokenizer else None
    pad = tokenizer.pad_token_id

    # runs/DATA_WEIGHTS.json, if present, repeats some files' rows ({"pool-v6.flash": 2}): small, realistic sets
    # (LLM-written inputs) count for more than their size next to large template pools.
    weights_file = ROOT / "runs" / "DATA_WEIGHTS.json"
    weights = json.loads(weights_file.read_text()) if weights_file.exists() else {}
    rows = without_eval([row for path in args.data for row in read_jsonl(path) for _ in range(int(weights.get(path.stem, 1)))])
    if weights:
        print(f"data weights: {weights}", flush=True)
    rows = rows[: args.limit] if args.limit else rows
    held = lambda r: int(hashlib.md5(json.dumps(r["request"], sort_keys=True).encode()).hexdigest(), 16) % 20 == 0  # noqa: E731
    train_rows = [r for r in rows if not held(r)]
    test_rows = [r for r in rows if held(r)]
    t0 = time.perf_counter()
    train = examples_of(train_rows, tokenizer, args.max_length, rng, augment=True)
    test = examples_of(test_rows, tokenizer, args.max_length, rng, augment=False)
    extra = {}
    for path in args.extra_eval:
        extra_rows = read_jsonl(path)
        for r in extra_rows:
            r["meta"] = {**r.get("meta", {}), "use": "eval:" + use_of(r["request"])}
        extra[path.stem] = examples_of(extra_rows, tokenizer, args.max_length, rng, augment=False, skip_eval_questions=False)
    print(f"{len(train)} training questions, {len(test)} held out; encoded in {time.perf_counter() - t0:.0f}s; "
          f"mean length {sum(len(e['input_ids']) for e in train) / len(train):.0f} tokens", flush=True)

    model = NanoDecider.from_mmbert(args.base, args.layers, vocab).to(device)
    n_body = sum(p.numel() for n, p in model.named_parameters() if "embeddings" not in n)
    n_emb = sum(p.numel() for n, p in model.named_parameters() if "embeddings" in n)
    print(f"student: {args.layers} layers, {n_body / 1e6:.1f}M body + {n_emb / 1e6:.1f}M embedding parameters", flush=True)
    head_params = list(model.head.parameters())
    body_params = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    optimizer = torch.optim.AdamW([{"params": body_params, "lr": args.lr}, {"params": head_params, "lr": args.lr * 10}],
                                  weight_decay=0.01)
    steps_per_epoch = len(batches(train, args.max_tokens, None))
    total = int(steps_per_epoch * args.epochs)
    warmup = max(1, total // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))

    step, best = 0, -1.0
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "train.log").open("a", encoding="utf-8")
    model.train()
    while step < total:
        for batch in batches(train, args.max_tokens, rng):
            if step >= total:
                break
            b = collate(batch, pad, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                flat = model(b["input_ids"], b["attention_mask"], b["rows"], b["cols"])
            loss = loss_of(split_logits(flat.float(), b["sizes"]), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 200 == 0:
                print(f"step {step}/{total} loss {loss.item():.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
            if step % steps_per_epoch == 0 or step == total:
                metrics = evaluate(model, test, pad, device, args.max_tokens)
                text = f"== step {step} (epoch {step / steps_per_epoch:.1f}) held-out:\n{report(metrics)}"
                for name, ex in extra.items():
                    text += f"\n-- {name}:\n{report(evaluate(model, ex, pad, device, args.max_tokens))}"
                print(text, flush=True)
                log.write(text + "\n")
                log.flush()
                if metrics["ALL"]["verdict_agree"] > best:
                    best = metrics["ALL"]["verdict_agree"]
                    model.save(args.out, {"base": str(tokenizer_dir), "layers": args.layers, "max_length": args.max_length,
                                          "step": step, "heldout": metrics})
    print(f"best held-out verdict agreement {best:.4f}; saved to {args.out}")


if __name__ == "__main__":
    main()
