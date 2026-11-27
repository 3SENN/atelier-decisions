# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""From how well a picture matches each answer's text to a System One answer.

The towers give cos(picture, text) for every text of a question (vision/format.py texts). SigLIP's own reading of
that is z = scale * cos + bias, the log-odds that the text describes the picture. This module turns the z of a
question's texts into the probabilities Atelier's readers expect:

    choice, score   softmax(z / t) over the options (the bias cancels); with MANY options or more, its own t
    noul            with a "false" text:  sigmoid(a * z_true - b * z_false + c)
                    without one:          sigmoid(a * z_true + c)

t, a, b and c come from vision/calibrate.py (vision.json), fitted so that an answer the model gives with 0.9 is right
about nine times in ten on pictures and kinds of questions it never saw. Uncalibrated, t = 1 makes every choice look
certain: scale is about 113, so a difference of 0.02 in cos is already ten to one.

A choice among dozens of options gets a temperature of its own ("many"). With one temperature for all, sixty wrong
options that each look a little possible add up to more than the right one, and the model is never sure of an
answer it has plainly found: Atelier's question about a button has 63.

runtime/node/src/vision/decide.ts is the same arithmetic.
"""

from __future__ import annotations

import math

MANY = 16  # options from which a choice is read with the "many" temperature
# Before calibration: SigLIP's own reading.
RAW = {"choice": {"t": 1.0, "many": 1.0}, "score": {"t": 1.0}, "noul": {"pair": {"a": 1.0, "b": 0.0, "c": 0.0}, "single": {"a": 1.0, "c": 0.0}}}


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def probabilities(kind: str, cos: list[float | None], scale: float, bias: float, cal: dict = RAW) -> list[float]:
    """The probability of each of format.labels(q), from the cosines of format.texts(q) (None where there was no text)."""
    z = [None if c is None else scale * c + bias for c in cos]
    if kind == "noul":
        z_false, z_true = z
        if z_false is None:
            p = sigmoid(cal["noul"]["single"]["a"] * z_true + cal["noul"]["single"]["c"])
        else:
            pair = cal["noul"]["pair"]
            p = sigmoid(pair["a"] * z_true - pair["b"] * z_false + pair["c"])
        return [1.0 - p, p]
    t = cal[kind].get("many", cal[kind]["t"]) if len(z) >= MANY else cal[kind]["t"]
    return softmax([x / t for x in z])


def answer(q: dict, labels: list[str], p: list[float]) -> dict:
    """pplx_decider.protocol.answer, as runtime/node/src/decider.ts ports it: the System One answer for one question."""
    if q["type"] == "noul":
        return {"type": "noul", "noul": p[1]}
    best = max(range(len(p)), key=lambda i: p[i])
    dist = dict(zip(labels, p))
    if q["type"] == "choice":
        n = len(p)
        confidence = 1.0 if n == 1 else (p[best] - 1 / n) / (1 - 1 / n)
        return {"type": "choice", "probabilities": dist, "choice": labels[best], "confidence": max(0.0, min(1.0, confidence))}
    distance = sum(pi * abs(i - best) for i, pi in enumerate(p))
    midpoint = (len(p) - 1) / 2
    baseline = sum(abs(i - midpoint) for i in range(len(p))) / len(p)
    return {"type": "score", "probabilities": dist, "legend": dict(zip(labels, q["criteria"])), "score": sum(i * pi for i, pi in enumerate(p)),
            "confidence": max(0.0, 1 - distance / baseline)}


# Atelier's readers (src/shared/decisions.ts): what counts as sure.
YES, NO, CHOICE, LEVEL = 0.7, 0.3, 0.75, 0.6


def verdict(kind: str, p: list[float]) -> int | None:
    """The index of the answer Atelier would act on, or None when it would keep its rule."""
    if kind == "noul":
        return 1 if p[1] >= YES else 0 if p[1] <= NO else None
    best = max(range(len(p)), key=lambda i: p[i])
    return best if p[best] >= (CHOICE if kind == "choice" else LEVEL) else None
