# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""atelier-step's input: one browser step as one sequence, answered in one pass.

    <bos> STEP goal <eos>
    VAL none of these VAL name: value VAL name: value ... <eos>
    what was done so far <eos>
    page title <eos>
    EL link Orders  EL field Postcode: empty  TX Added to your cart  EL button Checkout ... <eos>

- STEP: the operation is read here (click, type, done, blocked).
- EL before every control the runtime lists (links, buttons, fields, selects, checkboxes, radios): the element head
  scores each one, so the target is one of these markers. TX before a line of plain page text ("Added to your cart",
  "Payment"): read, never scored. They come in page order.
- VAL before each of the planner's named values; the first is always "none of these". The value of a field is read by
  matching the field's EL marker against every VAL marker.

This module is the contract with the runtime (runtime/node/src/step/format.ts re-implements it token for token).
"""

from __future__ import annotations

STEP, VAL, EL, TX = "<unused14>", "<unused15>", "<unused16>", "<unused17>"
OPS = ["click", "type", "done", "blocked"]
NONE_VALUE = "none of these"
CLICKABLE = {"link", "button", "checkbox", "radio"}
EDITABLE = {"field", "select"}
MAX_LEN = 1024
LIMITS = {"goal": 96, "value": 24, "values": 16, "done": 48, "page": 24, "item": 28}


def describe(e: dict) -> str:
    """runtime/node/src/process/steps.ts describe(), for the step model: the same line for the same element."""
    role, label = e["role"], e.get("label", "")
    if role == "text":
        return label
    if role in ("link", "button"):
        ctx = e.get("context")
        return f"{role} {label}" + (f" ({ctx[:60]})" if ctx and len(label) < 20 else "")
    if role in ("checkbox", "radio"):
        return f"{role} {label}: {'checked' if e.get('checked') else 'not checked'}"
    if role == "select":
        return f"select {label}: {e.get('value') or 'nothing chosen'} (options: {' | '.join((e.get('options') or [])[:8])})"
    return f"field {label}: {e['value'][:40] if e.get('value') else 'empty'}"


def encode(tok, ex: dict, max_len: int = MAX_LEN, keep: set[int] | None = None) -> dict:
    """Token ids of one step and where its markers are.

    ex: {goal, values: [{name, value}], before: [str], page, elements: [{role, label, ...}]}. Elements with role "text"
    get TX and are not scored. keep: element indices that must stay when the page is too long (training keeps the
    answer); without it the page is cut at the end, as the runtime does.
    Returns ids, step (position of STEP), els (positions of EL markers), el_index (which element each EL is), vals
    (positions of VAL markers; vals[0] is "none of these").
    """
    ids_of = lambda t: tok.encode(t, add_special_tokens=False).ids
    sid = lambda t: tok.token_to_id(t)
    bos, eos = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
    ids = [bos, sid(STEP)] + ids_of(ex["goal"])[: LIMITS["goal"]] + [eos]
    vals = []
    for text in [NONE_VALUE] + [f"{v['name']}: {v['value']}" for v in ex["values"][: LIMITS["values"]]]:
        vals.append(len(ids))
        ids += [sid(VAL)] + ids_of(text)[: LIMITS["value"]]
    ids += [eos] + ids_of("\n".join(ex.get("before") or []) or "nothing yet")[-LIMITS["done"]:] + [eos]
    ids += ids_of(ex.get("page") or "")[: LIMITS["page"]] + [eos]
    head = ids
    items = []
    for i, e in enumerate(ex["elements"]):
        marker = TX if e["role"] == "text" else EL
        items.append((i, [sid(marker)] + ids_of(describe(e))[: LIMITS["item"]]))
    budget = max_len - len(head) - 1
    total = sum(len(t) for _, t in items)
    if total > budget:
        if keep is None:
            kept, used = [], 0
            for i, t in items:
                if used + len(t) > budget:
                    break
                kept.append((i, t))
                used += len(t)
            items = kept
        else:
            # Drop text lines first, then controls the answer doesn't need, from the end, until it fits.
            order = [k for k, (i, t) in enumerate(items) if ex["elements"][i]["role"] == "text" and i not in keep][::-1]
            order += [k for k, (i, t) in enumerate(items) if ex["elements"][i]["role"] != "text" and i not in keep][::-1]
            drop = set()
            for k in order:
                if total <= budget:
                    break
                drop.add(k)
                total -= len(items[k][1])
            items = [it for k, it in enumerate(items) if k not in drop]
    els, el_index = [], []
    ids = list(head)
    for i, t in items:
        if ex["elements"][i]["role"] != "text":
            els.append(len(ids))
            el_index.append(i)
        ids += t
    ids.append(eos)
    return {"ids": ids[:max_len], "step": 1, "els": els, "el_index": el_index, "vals": vals}
