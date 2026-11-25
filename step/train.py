# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Trains atelier-step on steps whose answers are known: Mind2Web's recorded actions (step/mind2web.py) and the
generated flows (step/synth.py). Plain cross-entropy on the true answers; no teacher.

Held out: a quarter of Mind2Web's websites (whole sites), two whole flow families and one language of every family.
Measured on each: the operation, the element, the value, the whole step (all of them right), and how often it is sure
by the runtime's cutoffs (an operation at 0.6; an element or a value at 0.5 and 1.5 times the next).

    .venv/bin/python -I step/train.py --m2w data/cache/m2w-steps.jsonl --synth data/cache/synth-steps.jsonl \\
        --nano releases/v0.5/model --out runs/step-v1 --epochs 2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "step"))
sys.path.insert(0, str(ROOT / "data"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from format import CLICKABLE, EDITABLE, OPS, encode  # noqa: E402
from model import StepModel  # noqa: E402


def m2w_examples(path: Path, rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Mind2Web steps as examples: the recorded element among 15-60 of the page's others, in page order."""
    import build_steps as bs

    train, test = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        held = zlib.crc32(t["site"].encode()) % 4 == 0
        values = [{"name": f"value {k + 1}", "value": v} for k, v in enumerate(t["values"])]
        for k, s in enumerate(t["steps"]):
            n = rng.randint(15, 60)
            others = [i for i in range(len(s["elements"])) if i != s["gold"]]
            pick = sorted(rng.sample(others, min(n - 1, len(others))) + [s["gold"]])
            elements = [s["elements"][i] for i in pick]
            gold = pick.index(s["gold"])
            ge = elements[gold]
            if s["op"] == "click" and ge["role"] in EDITABLE or s["op"] == "type" and ge["role"] not in EDITABLE:
                continue
            fields = {}
            if s["op"] == "type" and s["value"]:
                v = next((i for i, x in enumerate(t["values"]) if x == s["value"]), None)
                if v is not None:
                    fields[gold] = v
            ex = {"goal": t["goal"], "values": values, "before": bs.recent(s["before"]).split("\n") if s["before"] else [], "page": f"{t['site']} ({t['domain']})",
                  "elements": elements, "op": s["op"], "gold": gold, "fields": fields, "source": "mind2web", "split": "site-unseen" if held else "train", "lang": "en"}
            (test if held else train).append(ex)
    return train, test


def synth_examples(path: Path) -> tuple[list[dict], list[dict]]:
    train, test = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        r["fields"] = {int(k): v for k, v in r["fields"].items()}
        r["source"] = "synth"
        (train if r["split"] == "train" else test).append(r)
    return train, test


def prepare(tok: Tokenizer, ex: dict, max_len: int) -> dict | None:
    keep = set(ex["fields"]) | ({ex["gold"]} if ex["gold"] is not None else set())
    enc = encode(tok, ex, max_len, keep)
    pos = {i: k for k, i in enumerate(enc["el_index"])}
    if ex["gold"] is not None and ex["gold"] not in pos:
        return None
    roles = [ex["elements"][i]["role"] for i in enc["el_index"]]
    fields = {pos[i]: (v + 1 if v >= 0 else 0) for i, v in ex["fields"].items() if i in pos and v + 1 < len(enc["vals"])}
    return {**enc, "op": OPS.index(ex["op"]), "gold": pos.get(ex["gold"]) if ex["gold"] is not None else None, "roles": roles, "fields": fields,
            "source": ex["source"], "split": ex["split"], "lang": ex.get("lang", "en"), "family": ex.get("family", "mind2web")}


def batches(items: list[dict], tokens: int, rng: random.Random | None):
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    groups, cur, longest = [], [], 0
    for i in order:
        longest = max(longest, len(items[i]["ids"]))
        if cur and longest * (len(cur) + 1) > tokens:
            groups.append(cur)
            cur, longest = [], len(items[i]["ids"])
        cur.append(i)
    if cur:
        groups.append(cur)
    if rng:
        rng.shuffle(groups)
    for g in groups:
        yield [items[i] for i in g]


def collate(batch: list[dict], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(len(b["ids"]) for b in batch)
    ids = torch.zeros(len(batch), n, dtype=torch.long)
    mask = torch.zeros(len(batch), n, dtype=torch.long)
    for k, b in enumerate(batch):
        ids[k, : len(b["ids"])] = torch.tensor(b["ids"])
        mask[k, : len(b["ids"])] = 1
    return ids.to(device), mask.to(device)


def group_mask(roles: list[str], op: int, device: str) -> torch.Tensor:
    want = CLICKABLE if OPS[op] == "click" else EDITABLE
    return torch.tensor([r in want for r in roles], device=device)


def step_losses(model: StepModel, h: torch.Tensor, b: dict, device: str) -> tuple[torch.Tensor, dict]:
    els = torch.tensor(b["els"], device=device)
    vals = torch.tensor(b["vals"], device=device)
    op, el, val = model.heads(h, b["step"], els, vals)
    loss = F.cross_entropy(op[None], torch.tensor([b["op"]], device=device))
    out = {"op": op.detach(), "el": el.detach(), "val": val.detach()}
    if b["gold"] is not None and len(b["els"]):
        m = group_mask(b["roles"], b["op"], device)
        logits = el.masked_fill(~m, -1e4)
        loss = loss + F.cross_entropy(logits[None], torch.tensor([b["gold"]], device=device))
    if b["fields"]:
        rows = torch.tensor(list(b["fields"]), device=device)
        want = torch.tensor(list(b["fields"].values()), device=device)
        loss = loss + 0.5 * F.cross_entropy(val[rows], want)
    return loss, out


def clear(p: torch.Tensor, cutoff: float = 0.5) -> tuple[int, bool]:
    top = torch.topk(p, min(2, p.numel()))
    first = top.values[0].item()
    second = top.values[1].item() if p.numel() > 1 else 0.0
    return int(top.indices[0].item()), first >= cutoff and first >= 1.5 * second


@torch.no_grad()
def evaluate(model: StepModel, items: list[dict], device: str, tokens: int) -> dict:
    model.eval()
    stats: dict[str, dict[str, float]] = {}
    for batch in batches(items, tokens, None):
        ids, mask = collate(batch, device)
        hs = model.hidden(ids, mask)
        for k, b in enumerate(batch):
            _, o = step_losses(model, hs[k], b, device)
            key = f"{b['source']}/{b['split']}"
            s = stats.setdefault(key, {"n": 0, "op": 0, "op_sure": 0, "op_sure_right": 0, "el_n": 0, "el": 0, "el_sure": 0, "el_sure_right": 0, "val_n": 0, "val": 0, "step": 0, "step_sure": 0})
            s["n"] += 1
            p_op = torch.softmax(o["op"], -1)
            op = int(p_op.argmax())
            op_ok = op == b["op"]
            op_sure = p_op[op].item() >= 0.6
            s["op"] += op_ok
            s["op_sure"] += op_sure
            s["op_sure_right"] += op_sure and op_ok
            el_ok = el_sure = True
            if b["gold"] is not None:
                m = group_mask(b["roles"], b["op"], device)
                p = torch.softmax(o["el"].masked_fill(~m, -1e4), -1)
                pick, sure = clear(p)
                el_ok, el_sure = pick == b["gold"], sure
                s["el_n"] += 1
                s["el"] += el_ok
                s["el_sure"] += sure
                s["el_sure_right"] += sure and el_ok
            val_ok = True
            if b["gold"] is not None and b["gold"] in b["fields"]:
                p = torch.softmax(o["val"][b["gold"]], -1)
                val_ok = int(p.argmax()) == b["fields"][b["gold"]]
                s["val_n"] += 1
                s["val"] += val_ok
            ok = op_ok and el_ok and val_ok
            s["step"] += ok
            s["step_sure"] += ok and op_sure and el_sure
    model.train()
    out = {}
    for key, s in sorted(stats.items()):
        out[key] = {"n": s["n"], "op": s["op"] / s["n"], "op_sure": s["op_sure"] / s["n"], "op_right_when_sure": s["op_sure_right"] / max(1, s["op_sure"]),
                    "element": s["el"] / max(1, s["el_n"]), "element_sure": s["el_sure"] / max(1, s["el_n"]), "element_right_when_sure": s["el_sure_right"] / max(1, s["el_sure"]),
                    "value": s["val"] / max(1, s["val_n"]), "step": s["step"] / s["n"], "step_right_and_sure": s["step_sure"] / s["n"]}
    return out


def show(metrics: dict) -> None:
    for key, m in metrics.items():
        print(f"  {key:<24} n={m['n']:<5} op {m['op']:.0%} (sure {m['op_sure']:.0%}, right when sure {m['op_right_when_sure']:.0%}) · element {m['element']:.0%} "
              f"(sure {m['element_sure']:.0%}, right {m['element_right_when_sure']:.0%}) · value {m['value']:.0%} · step {m['step']:.0%} (sure and right {m['step_right_and_sure']:.0%})", flush=True)


def pressure() -> tuple[float, float]:
    """Free disk (GB) and swap in use (MB): training stops before it pushes the machine into swap or a full disk."""
    import shutil
    import subprocess

    free = shutil.disk_usage(Path.home()).free / 1e9
    swap = 0.0
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5).stdout
        swap = float(out.split("used = ")[1].split("M")[0])
    except Exception:
        pass
    return free, swap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2w", type=Path)
    parser.add_argument("--synth", type=Path)
    parser.add_argument("--nano", type=Path, default=ROOT / "releases/v0.5/model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=0, help="keep this many of the nano's layers (0: all)")
    parser.add_argument("--eval-only", action="store_true", help="score --init on the test sets (up to --eval-limit per set) and write --out/eval.json")
    parser.add_argument("--init", type=Path, help="start from a saved atelier-step (to go on after an interruption, with a lower --lr)")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--head-lr", type=float, default=4e-4)
    parser.add_argument("--tokens", type=int, default=3072, help="tokens per batch (padded)")
    parser.add_argument("--min-free-gb", type=float, default=1.0, help="stop (and save) below this much free disk")
    parser.add_argument("--max-swap-growth-mb", type=float, default=512, help="stop (and save) when swap grows this much")
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="training examples (0: all), for a quick look")
    parser.add_argument("--eval-limit", type=int, default=1500)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    rng = random.Random("2028-step-train")
    torch.manual_seed(2028)
    tok = Tokenizer.from_file(str(args.nano / "tokenizer.json"))
    train, test = [], []
    if args.m2w:
        a, b = m2w_examples(args.m2w, rng)
        train += a
        test += b
    if args.synth:
        a, b = synth_examples(args.synth)
        train += a
        test += b
    rng.shuffle(train)
    if args.limit:
        train = train[: args.limit]
    prep = lambda xs: [p for p in (prepare(tok, x, args.max_len) for x in xs) if p]
    train_items = prep(train)
    rng.shuffle(test)
    by_split: dict[str, list[dict]] = {}
    for x in prep(test):
        by_split.setdefault(f"{x['source']}/{x['split']}", []).append(x)
    test_items = [x for xs in by_split.values() for x in xs[: args.eval_limit]]
    print(f"train {len(train_items)} steps, test {len(test_items)} ({', '.join(f'{k} {len(v)}' for k, v in by_split.items())}) on {args.device}", flush=True)

    if args.init:
        model, init_meta = StepModel.load(args.init)
        model = model.to(args.device)
        print(f"from {args.init} ({init_meta.get('updates')} updates)", flush=True)
    else:
        model = StepModel.from_nano(args.nano, args.layers or None).to(args.device)
    if args.eval_only:
        metrics = evaluate(model, test_items, args.device, args.tokens)
        show(metrics)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "eval.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return
    # Activations are recomputed in the backward pass: a fraction of the memory, about a third more time.
    model.backbone.gradient_checkpointing_enable()
    free0, swap0 = pressure()
    print(f"free disk {free0:.1f} GB, swap in use {swap0:.0f} MB", flush=True)
    heads = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    body = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": heads, "lr": args.head_lr}], weight_decay=0.01)
    steps_per_epoch = sum(1 for _ in batches(train_items, args.tokens, None))
    total = int(steps_per_epoch * args.epochs)
    warm = max(1, total // 30)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.05, 1 - s / total))
    print(f"{total} updates ({steps_per_epoch} per epoch), {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters", flush=True)
    print("before training:")
    show(evaluate(model, test_items[:400], args.device, args.tokens))
    step, t0, run_loss = 0, time.time(), 0.0
    best = -1.0
    while step < total:
        for batch in batches(train_items, args.tokens, rng):
            ids, mask = collate(batch, args.device)
            hs = model.hidden(ids, mask)
            loss = sum(step_losses(model, hs[k], b, args.device)[0] for k, b in enumerate(batch)) / len(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run_loss = 0.98 * run_loss + 0.02 * loss.item() if step > 1 else loss.item()
            if step % 10 == 0:
                free, swap = pressure()
                if free < args.min_free_gb or swap - swap0 > args.max_swap_growth_mb:
                    model.save(args.out, {"updates": step, "stopped": f"free disk {free:.1f} GB, swap +{swap - swap0:.0f} MB", "args": {k: str(v) for k, v in vars(args).items()}})
                    print(f"stopped at update {step}: free disk {free:.1f} GB, swap grew {swap - swap0:.0f} MB (saved)", flush=True)
                    return
            if step % 50 == 0:
                rate = step / (time.time() - t0)
                print(f"update {step}/{total} loss {run_loss:.3f} · {rate:.2f} updates/s · {(total - step) / rate / 60:.0f} min left", flush=True)
            if step % max(1, total // 4) == 0 or step >= total:
                metrics = evaluate(model, test_items, args.device, args.tokens)
                print(f"after {step} updates:")
                show(metrics)
                score = sum(m["step"] * m["n"] for m in metrics.values()) / max(1, sum(m["n"] for m in metrics.values()))
                if score > best:
                    best = score
                    model.save(args.out, {"updates": step, "metrics": metrics, "args": {k: str(v) for k, v in vars(args).items()}})
                    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                    print(f"  saved (step right {score:.1%})", flush=True)
            if step >= total:
                break
    print(f"done in {(time.time() - t0) / 60:.0f} min; best step right {best:.1%}")


if __name__ == "__main__":
    main()
