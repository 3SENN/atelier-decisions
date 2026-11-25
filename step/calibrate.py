# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Temperatures for atelier-step's heads, so "sure" means what the runtime takes it to mean.

The runtime acts on an answer the model is sure of and hands the rest to an escalation or the person. Trained on
answers alone, the model is sure of nearly everything, also where it is wrong (on held-out kinds of flow: sure of 99%
of operations, right on 82%). One temperature per head (operation, element, value) is fitted on half of the
held-out steps (minimum negative log-likelihood) and reported on the other half, then written into the exported
step.json, where runtime/node/src/step/model.ts divides the logits by it.

    .venv/bin/python -I step/calibrate.py --model runs/step-v4 --m2w data/cache/m2w-steps.jsonl \\
        --synth data/cache/synth-steps-v4.jsonl --export runs/step-v4/onnx
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "step"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from model import StepModel  # noqa: E402
from train import batches, clear, collate, group_mask, m2w_examples, prepare, step_losses, synth_examples  # noqa: E402


@torch.no_grad()
def collect(model: StepModel, items: list[dict], device: str, tokens: int) -> list[dict]:
    """For every step: the op logits and truth, the element logits over the right kind and the truth, the value logits."""
    out = []
    model.eval()
    for batch in batches(items, tokens, None):
        ids, mask = collate(batch, device)
        hs = model.hidden(ids, mask)
        for k, b in enumerate(batch):
            _, o = step_losses(model, hs[k], b, device)
            r = {"key": f"{b['source']}/{b['split']}", "op": o["op"].float().cpu(), "op_truth": b["op"], "el": None, "val": None}
            if b["gold"] is not None and b["els"]:
                m = group_mask(b["roles"], b["op"], device).cpu()
                r["el"] = o["el"].float().cpu()[m]
                r["el_truth"] = int(m[: b["gold"]].sum())
            if b["gold"] is not None and b["gold"] in b["fields"]:
                r["val"] = o["val"][b["gold"]].float().cpu()
                r["val_truth"] = b["fields"][b["gold"]]
            out.append(r)
    return out


def fit(logits: list[torch.Tensor], truth: list[int]) -> float:
    """The temperature with the least negative log-likelihood, by a search over a grid then a finer one."""
    def nll(t: float) -> float:
        return sum(F.cross_entropy((l / t)[None], torch.tensor([y])).item() for l, y in zip(logits, truth)) / max(1, len(truth))
    best = min((nll(t), t) for t in [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8])[1]
    return min((nll(t), t) for t in [best * f for f in (0.8, 0.9, 1, 1.1, 1.25)])[1]


def report(rows: list[dict], temps: dict[str, float]) -> dict:
    stats: dict[str, dict[str, float]] = {}
    for r in rows:
        s = stats.setdefault(r["key"], {"n": 0, "op_sure": 0, "op_sure_right": 0, "el_n": 0, "el_sure": 0, "el_sure_right": 0})
        s["n"] += 1
        p = torch.softmax(r["op"] / temps["op"], -1)
        sure = p.max().item() >= 0.6
        s["op_sure"] += sure
        s["op_sure_right"] += sure and int(p.argmax()) == r["op_truth"]
        if r["el"] is not None:
            pick, sure = clear(torch.softmax(r["el"] / temps["el"], -1))
            s["el_n"] += 1
            s["el_sure"] += sure
            s["el_sure_right"] += sure and pick == r["el_truth"]
    return {k: {"n": s["n"], "op_sure": s["op_sure"] / s["n"], "op_right_when_sure": s["op_sure_right"] / max(1, s["op_sure"]),
                "element_sure": s["el_sure"] / max(1, s["el_n"]), "element_right_when_sure": s["el_sure_right"] / max(1, s["el_sure"])} for k, s in sorted(stats.items())}


def show(title: str, metrics: dict) -> None:
    print(title)
    for k, m in metrics.items():
        print(f"  {k:<24} n={m['n']:<5} op sure {m['op_sure']:.0%}, right when sure {m['op_right_when_sure']:.0%} · element sure {m['element_sure']:.0%}, right when sure {m['element_right_when_sure']:.0%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--m2w", type=Path)
    parser.add_argument("--synth", type=Path)
    parser.add_argument("--nano", type=Path, default=ROOT / "releases/v0.5/model")
    parser.add_argument("--export", type=Path, help="an exported folder whose step.json gets the temperatures")
    parser.add_argument("--tokens", type=int, default=3072)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    tok = Tokenizer.from_file(str(args.nano / "tokenizer.json"))
    rng = random.Random("2028-step-train")  # the same draw of Mind2Web's page elements as training
    test = []
    if args.m2w:
        test += m2w_examples(args.m2w, rng)[1]
    if args.synth:
        test += synth_examples(args.synth)[1]
    items = [p for p in (prepare(tok, x, args.max_len) for x in test) if p]
    # Halves by a hash of the step: fitted on one, reported on the other.
    half = lambda i: zlib.crc32(str(i).encode()) % 2
    model, _ = StepModel.load(args.model)
    rows = collect(model.to(args.device), items, args.device, args.tokens)
    fit_rows = [r for i, r in enumerate(rows) if half(i) == 0]
    check = [r for i, r in enumerate(rows) if half(i) == 1]
    temps = {
        "op": fit([r["op"] for r in fit_rows], [r["op_truth"] for r in fit_rows]),
        "el": fit([r["el"] for r in fit_rows if r["el"] is not None], [r["el_truth"] for r in fit_rows if r["el"] is not None]),
        "val": fit([r["val"] for r in fit_rows if r["val"] is not None], [r["val_truth"] for r in fit_rows if r["val"] is not None]),
    }
    print(f"temperatures {temps} (fitted on {len(fit_rows)} held-out steps, reported on the other {len(check)})")
    before = report(check, {"op": 1.0, "el": 1.0, "val": 1.0})
    after = report(check, temps)
    show("before:", before)
    show("after:", after)
    if args.export:
        info_path = args.export / "step.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["temperatures"] = {k: round(v, 3) for k, v in temps.items()}
        info["calibration"] = {"before": before, "after": after}
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"written to {info_path}")


if __name__ == "__main__":
    main()
