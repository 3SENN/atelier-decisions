# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""The numbers that make atelier-vision's "sure" mean what Atelier takes it to mean.

Atelier acts on an answer the model is sure of and keeps its rule otherwise. Read SigLIP's own way, the towers are
sure of nearly every choice (also the wrong ones) and say no to nearly every yes/no question. Six numbers fix
that (vision/decide.py):

    choice, score   one temperature each, and one more for a choice among 16 options or more
    noul            a, b, c for a question with a "false" text; a, c for one without

They are fitted by minimum negative log-likelihood on half a of the held-out pictures (pictures and kinds the
towers never trained on) and written into vision.json; vision/measure.py reports on half b. One set of numbers for
every family on purpose: a question nobody measured gets the same reading.

    .venv/bin/python -I vision/calibrate.py --emb runs/vision/emb/base --out runs/vision/base/vision.json
    .venv/bin/python -I vision/calibrate.py --emb runs/vision/emb/v1 --adapters runs/vision-v1/vision.pt --out runs/vision-v1/vision.json
    .venv/bin/python -I vision/calibrate.py --emb runs/vision/emb/v1-onnx --runtime runs/vision-v1/onnx --out runs/vision-v1/onnx/vision.json   # as shipped
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from measure import CACHE, cosines, picture_vectors, read_rows, text_vectors  # noqa: E402
from decide import MANY  # noqa: E402
from format import labels  # noqa: E402


def fit_temperature(z: list[list[float]], truth: list[int]) -> float:
    """The temperature that makes softmax(z / t) fit the answers best."""
    t = torch.zeros(1, requires_grad=True)  # log t
    opt = torch.optim.LBFGS([t], lr=0.5, max_iter=200)
    zs = [torch.tensor(x, dtype=torch.float64) for x in z]

    def loss():
        opt.zero_grad()
        total = sum(-torch.log_softmax(x / t.exp().double(), dim=0)[i] for x, i in zip(zs, truth)) / len(zs)
        total.backward()
        return total

    opt.step(loss)
    return float(t.exp())


def fit_logistic(features: np.ndarray, truth: np.ndarray) -> list[float]:
    """Weights w and an offset c for sigmoid(features @ w + c), by minimum negative log-likelihood."""
    x = torch.tensor(features, dtype=torch.float64)
    y = torch.tensor(truth, dtype=torch.float64)
    mean, std = x.mean(0), x.std(0) + 1e-9  # fitted on standardised features, returned for the raw ones
    xn = (x - mean) / std
    w = torch.zeros(x.shape[1] + 1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=0.5, max_iter=500)

    def loss():
        opt.zero_grad()
        total = torch.nn.functional.binary_cross_entropy_with_logits(xn @ w[:-1] + w[-1], y)
        total.backward()
        return total

    opt.step(loss)
    weights = (w[:-1] / std).detach()
    offset = float(w[-1].detach() - (weights * mean).sum())
    return [float(v) for v in weights] + [offset]


def fit(rows: list[dict], pictures, words, scale: float, bias: float) -> tuple[dict, dict]:
    choice_z, choice_truth, score_z, score_truth, pair, pair_truth, single, single_truth = [], [], [], [], [], [], [], []
    many_z, many_truth = [], []
    for r in rows:
        if r["answer"] is None:
            continue  # a button that is none of the jobs has no answer to fit to
        z = [None if c is None else scale * c + bias for c in cosines(r, pictures, words)]
        truth = labels(r["q"]).index(r["answer"])
        if r["kind"] == "noul":
            if z[0] is None:
                single.append([z[1]])
                single_truth.append(truth)
            else:
                pair.append([z[1], z[0]])
                pair_truth.append(truth)
        elif r["kind"] == "score":
            score_z.append(z)
            score_truth.append(truth)
        elif len(z) >= MANY:
            many_z.append(z)
            many_truth.append(truth)
        else:
            choice_z.append(z)
            choice_truth.append(truth)
    t = fit_temperature(choice_z, choice_truth)
    many = fit_temperature(many_z, many_truth) if many_z else t
    a, b, c = fit_logistic(np.array(pair), np.array(pair_truth))
    a1, c1 = fit_logistic(np.array(single), np.array(single_truth))
    cal = {"choice": {"t": round(t, 3), "many": round(many, 3)}, "score": {"t": round(fit_temperature(score_z, score_truth), 3) if score_z else round(t, 3)},
           "noul": {"pair": {"a": round(a, 4), "b": round(-b, 4), "c": round(c, 3)}, "single": {"a": round(a1, 4), "c": round(c1, 3)}}}
    return cal, {"choice": len(choice_z), "choice among 16 or more": len(many_z), "score": len(score_z), "noul with a false text": len(pair), "noul without": len(single)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=CACHE / "eval.jsonl")
    parser.add_argument("--adapters", type=Path)
    parser.add_argument("--runtime", type=Path, help="an exported folder: fit on what its files give (the pictures' vectors from embed.py --runtime)")
    parser.add_argument("--out", type=Path, required=True, help="vision.json: its calibration is written, the rest kept")
    args = parser.parse_args()
    rows = [r for r in read_rows(args.questions) if r["half"] == "a"]
    if args.runtime:
        from runtime import Runtime

        rt = Runtime(args.runtime, threads=8, raw=True, text_cache=(args.emb / "texts.npz") if args.emb else None)
        rows = [r for r in rows if not rt.decline(r["q"])]
        cal, counts = fit(rows, picture_vectors(args.emb, rows), text_vectors(rows, None, rt), rt.scale, rt.bias)
        rt.save_texts()
    else:
        from model import Towers

        base = Towers.load()
        cal, counts = fit(rows, picture_vectors(args.emb, rows), text_vectors(rows, args.adapters), base.scale, base.bias)
    info = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    info.update(calibration=cal, calibrated_on=counts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"{args.out}: {json.dumps(cal)}\n  fitted on {counts}")


if __name__ == "__main__":
    main()
