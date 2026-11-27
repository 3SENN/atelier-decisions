# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""atelier-vision's questions: Atelier's own System One questions, asked about a picture instead of a text.

A decision model answers typed questions by choosing among the answers it is given; it writes nothing. For a
picture, every answer is a short text, and the model says how well the picture matches each one:

    choice   {"type": "choice", "criteria": {"beach": "A day at the beach", "food": "Food or a meal", ...}}
             one text per option: its description, or the option's own name when it has none.
    score    {"type": "score", "criteria": ["Sharp", "A little blurred", "Too blurred to use"]}
             one text per level, in order. Declined for now (unanswerable): no version has been measured on levels,
             and an answer nobody measured is not one Atelier should act on. The arithmetic is there for when one is.
    noul     {"type": "noul", "instructions": "Is this a receipt?", "criteria": {"true": "A receipt from a shop or a
             restaurant.", "false": "Anything else."}}
             the "true" text is what a yes looks like. Without criteria the question itself is matched. A "false" text is
             matched too when there is one; the reader weighs both (vision/decide.py).

`instructions` say what is being asked. For a choice or a score the model doesn't read them: the options carry the
meaning. So an option must make sense by itself ("Opens the shopping cart", not "the first one").

A text is lowercased (SigLIP 2 was trained on lowercased text), cut to 63 tokens, closed with <eos> and padded to 64
with <pad>: the text tower always reads 64 positions and takes its answer at the last one.

This module is the contract with the runtime: runtime/node/src/vision/format.ts is the same code, and
`npm run vision:parity` shows it gives the same texts and token ids.
"""

from __future__ import annotations

import json

MAX_TOKENS = 64
PAD, EOS = 0, 1
SCORE_UNMEASURED = "levels on a scale are not measured for pictures yet"


def clean(text: str) -> str:
    """One line, single spaces, lowercase."""
    return " ".join(text.split()).lower()


def encode(tok, text: str) -> tuple[list[int], bool]:
    """A text's 64 token ids, and whether it was too long and lost its end. tok: tokenizers.Tokenizer or VisionTok."""
    ids = tok.encode(clean(text), add_special_tokens=False).ids
    cut = len(ids) > MAX_TOKENS - 1
    ids = ids[: MAX_TOKENS - 1] + [EOS]
    return ids + [PAD] * (MAX_TOKENS - len(ids)), cut


def labels(q: dict) -> list[str]:
    """The answers' names, as student/format.py option_labels: a choice's keys, a score's level numbers, false and true."""
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return ["false", "true"]


def _said(value) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def texts(q: dict) -> list[str | None]:
    """The text to match the picture against for each of labels(q); None where there is nothing to match.

    A noul gives [false text or None, true text or None]: the true text is its criteria's "true", else the question.
    """
    if q["type"] == "choice":
        return [_said(v) or _said(k) for k, v in q["criteria"].items()]
    if q["type"] == "score":
        return [_said(v) for v in q["criteria"]]
    criteria = q.get("criteria") or {}
    return [_said(criteria.get("false")), _said(criteria.get("true")) or _said(q.get("instructions"))]


def unanswerable(q: dict) -> str | None:
    """Why this question can't be asked of a picture, or None when it can."""
    kind = q.get("type")
    if kind not in ("choice", "noul", "score"):
        return f"unknown question type {json.dumps(kind)}"
    criteria = q.get("criteria")
    if kind == "choice" and (not isinstance(criteria, dict) or len(criteria) < 2):
        return "a choice needs two or more options"
    if kind == "score" and (not isinstance(criteria, list) or len(criteria) < 2):
        return "a score needs two or more levels"
    if kind == "score":
        return SCORE_UNMEASURED
    if kind == "noul" and criteria is not None and not isinstance(criteria, dict):
        return "the criteria of a yes/no question are {true, false}"
    found = texts(q)
    if kind == "noul":
        return None if found[1] else "nothing says what a yes looks like"
    return None if all(found) else "an option without a text"
