# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Atelier's answer readers and cutoffs (src/shared/decisions.ts and src/main/decisions/uses.ts), in Python.

What matters for Atelier is not the raw probability but the VERDICT a reader makes of it: a sure answer that takes
over from the rule, or "unsure" (None) so the rule decides. The student is judged on whether its verdicts match
the teacher's, and on how often a verdict it makes is wrong.
"""

from __future__ import annotations


def yes_no(p_true: float, yes: float = 0.7, no: float = 0.3):
    return True if p_true >= yes else False if p_true <= no else None


def sure_choice(probs: dict[str, float], cutoff: float = 0.75):
    best = max(probs, key=probs.get)
    return best if probs[best] >= cutoff else None


def sure_level(probs: list[float], cutoff: float = 0.6):
    best = max(range(len(probs)), key=probs.__getitem__)
    return best if probs[best] >= cutoff else None


def verdict(kind: str, labels: list[str], probs: list[float]):
    """The verdict Atelier's reader makes for one question (kind from student.format.fixed_kind, or 'open')."""
    p = dict(zip(labels, probs))
    if kind == "waits":
        return yes_no(p["true"], 0.75, 0.25)
    if kind == "dictation":
        return True if yes_no(p["true"], 0.85, 0.15) is True else None
    if kind == "kind":
        k = sure_choice(p, 0.8)
        return None if k in (None, "none") else k
    if kind == "intent":
        return sure_choice(p, 0.8)
    if kind == "target":
        return True if sure_choice(p, 0.85) == "companion" else None
    if kind == "effort":
        return sure_level(probs, 0.7)
    if kind == "lasting":
        return False if yes_no(p["true"], 0.7, 0.2) is False else None
    if kind == "place":
        return sure_choice(p, 0.75)
    if kind == "skill":
        return yes_no(p["true"], 0.7, 0.2)
    if kind in ("techLevel", "replyStyle"):
        return sure_choice(p, 0.75)
    if kind == "interest":
        return yes_no(p["true"], 0.75, 0.25)
    # label_items and other open questions: yes at 0.7, no at 0.3, a choice at 0.75, a level at 0.6
    if labels == ["false", "true"]:
        return yes_no(p["true"])
    if all(label.isdigit() for label in labels):
        return sure_level(probs, 0.6)
    return sure_choice(p, 0.75)
