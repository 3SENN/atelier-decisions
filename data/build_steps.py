# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Steps of browser tasks, answered by what people really did: the questions the runtime's advance step asks
(runtime/node/src/process/steps.ts), built from Mind2Web's recorded actions (CC BY 4.0, osunlp/Mind2Web).

A process can say "order the Trailrunner 2 in size 43" and let the decision model take it one operation at a time:
which operation (click or type), which element, which value. That is how browser agents built on hosted decision
models work (docs/processes.md). atelier-mini never saw such questions: on the lab's flows it was sure of almost no
step. Mind2Web has 2,350 tasks on 137 real websites with every action recorded, so every step comes with its
answer: the operation, the element (one positive among the page's candidates) and the value typed or chosen.

For each recorded action, one row per question, in the runtime's exact words and state:

  operation    choice: click, type, done, blocked                   truth: click (CLICK) or type (TYPE, SELECT)
  click_target choice over the numbered clickable candidates         truth: the recorded element's number
  type_target  choice over the numbered fields                       truth: the recorded field's number
  value        choice over the task's values ("value 2": "Boston")   truth: the value typed (TYPE)
  option       choice over a select's options                        truth: the option chosen (SELECT)

The state is the runtime's: the goal (the task), the values (every value the task's actions type or choose), the page
(site), the numbered elements (the recorded one and up to 39 others the page offered, shuffled) and what was done so
far. Mind2Web has no page after the last action, so "done" and "blocked" are not taught here; the lab's flows
(lab/apps) are where those come from.

Websites are split, not tasks: a quarter of the sites are held out whole, so the test asks about sites never seen.

    python -I data/build_steps.py fetch --rows-dir data/cache/mind2web_rows --tasks 60
    python -I data/build_steps.py build --rows-dir data/cache/mind2web_rows --out data/pool/pool-steps.jsonl \\
        --test data/pool/steps-test.jsonl --truth-dir data/teacher
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import random
import time
import urllib.request
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

# The runtime's words (runtime/node/src/process/steps.ts): keep them identical.
OPERATION = {
    "instructions": "Advance the goal from the current page with one operation.",
    "criteria": {
        "click": "press a button, follow a link or tick a box that moves the goal forward",
        "type": "put one of the given values into a field that is still empty or wrong",
        "done": "every part of the goal is visibly done on this page",
        "blocked": "the goal cannot go further from here: a value is missing, an error shows, or a person must decide",
    },
}
CLICK_TARGET = "If the next operation is a click, which element?"
TYPE_TARGET = "If the next operation is typing, which field?"
MAX_ELEMENTS = 40
ROWS_API = "https://datasets-server.huggingface.co/rows?dataset=osunlp/Mind2Web&config=default&split=train"


class Tree(HTMLParser):
    """Mind2Web's cleaned HTML as nodes by backend_node_id: tag, attributes, the text inside, a select's options."""

    VOID = {"input", "img", "br", "hr", "meta", "link", "source", "area", "col", "embed", "wbr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.nodes: dict[str, dict] = {}
        self.stack: list[dict] = []

    def handle_starttag(self, tag, attrs):
        a = {k: v or "" for k, v in attrs}
        node = {"tag": tag, "attrs": a, "text": [], "options": []}
        if "backend_node_id" in a:
            self.nodes[a["backend_node_id"]] = node
        if tag == "option" and self.stack:
            for n in reversed(self.stack):
                if n["tag"] == "select":
                    n["options"].append(node)
                    break
        if tag not in self.VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        a = {k: v or "" for k, v in attrs}
        if "backend_node_id" in a:
            self.nodes[a["backend_node_id"]] = {"tag": tag, "attrs": a, "text": [], "options": []}

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i]["tag"] == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        words = data.strip()
        if words:
            for n in self.stack:
                n["text"].append(words)


def clean(text: str, limit: int = 80) -> str:
    text = " ".join(html_lib.unescape(text).split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def element(node: dict) -> dict | None:
    """A candidate as the runtime describes it: role, label, value, options. None when it has nothing to read."""
    tag, a = node["tag"], node["attrs"]
    text = clean(" ".join(node["text"]))
    label = clean(a.get("aria_label") or a.get("aria-label") or "") or text or clean(a.get("placeholder", "")) or clean(a.get("title", "")) or clean(a.get("alt", "")) or clean(a.get("name", ""))
    kind = a.get("type", "").lower()
    if tag == "select":
        options = [clean(" ".join(o["text"]), 40) for o in node["options"] if o["text"]][:12]
        return {"role": "select", "label": label or "select", "value": "", "options": options}
    if tag == "textarea" or (tag == "input" and kind not in ("submit", "button", "reset", "image", "checkbox", "radio", "hidden")):
        if not label:
            return None
        return {"role": "field", "label": label, "value": clean(a.get("input_value") or a.get("value") or "", 40)}
    if tag == "input" and kind in ("checkbox", "radio"):
        return {"role": kind, "label": label or kind, "checked": a.get("input_checked") == "true"} if label else None
    if not label:
        return None
    return {"role": "link" if tag == "a" else "button", "label": label}


def describe(e: dict) -> str:
    """runtime/node/src/process/steps.ts describe(): the same line for the same element."""
    if e["role"] in ("link", "button"):
        return f"{e['role']} {e['label']}"
    if e["role"] in ("checkbox", "radio"):
        return f"{e['role']} {e['label']}: {'checked' if e.get('checked') else 'not checked'}"
    if e["role"] == "select":
        return f"select {e['label']}: {e.get('value') or 'nothing chosen'} (options: {' | '.join(e['options'][:8])})"
    return f"field {e['label']}: {e['value'] if e.get('value') else 'empty'}"


def editable(e: dict) -> bool:
    return e["role"] in ("field", "select")


def recent(reprs: list[str]) -> str:
    """Mind2Web's action_reprs ("[searchbox]  Find a location -> TYPE: Boston") as the runtime's done_so_far."""
    out = []
    for r in reprs[-4:]:
        target, _, act = r.partition("->")
        name = clean(target.split("]", 1)[-1], 50) or "an element"
        out.append(f"clicked {name}" if act.strip().startswith("CLICK") else f"typed into {name}")
    return "\n".join(out) or "nothing yet"


def rows_for_task(task: dict, rng: random.Random, split: str) -> list[dict]:
    values = []
    for a in task["actions"]:
        v = clean(a["operation"].get("value") or "", 60)
        if a["operation"]["op"] in ("TYPE", "SELECT") and v and v not in values:
            values.append(v)
    named = {f"value {i + 1}": v for i, v in enumerate(values)}
    out = []
    for i, a in enumerate(task["actions"]):
        tree = Tree()
        try:
            tree.feed(a.get("cleaned_html") or "")
        except Exception:
            continue
        pos_ids = [c["backend_node_id"] for c in a["pos_candidates"]]
        pos = next((element(tree.nodes[p]) for p in pos_ids if p in tree.nodes and element(tree.nodes[p])), None)
        if pos is None:
            continue
        negs = []
        for c in rng.sample(a["neg_candidates"], len(a["neg_candidates"])):
            n = tree.nodes.get(c["backend_node_id"])
            e = element(n) if n else None
            if e and describe(e) != describe(pos) and describe(e) not in {describe(x) for x in negs}:
                negs.append(e)
            if len(negs) >= MAX_ELEMENTS - 1:
                break
        els = negs + [pos]
        rng.shuffle(els)
        number = {id(e): str(k + 1) for k, e in enumerate(els)}
        op = a["operation"]["op"]
        state = {
            "goal": task["confirmed_task"],
            "values": "\n".join(f"{k}: {v}" for k, v in named.items()) or "none",
            "page": f"{task['website']} ({task.get('domain', '')}, {task.get('subdomain', '')})",
            "elements": "\n".join(f"[{number[id(e)]}] {describe(e)}" for e in els),
            "done_so_far": recent(task["action_reprs"][:i]),
        }
        meta = {"use": "open", "source": "mind2web-steps", "lang": "en", "split": split, "site": task["website"], "task": task["annotation_id"], "step": i}
        want = "click" if op == "CLICK" else "type"
        if want == "click" and editable(pos) or want == "type" and not editable(pos):
            continue  # a click on a field or typing into a button: not a step the runtime would take this way
        out.append({"meta": {**meta, "family": "step-operation", "truth": want}, "request": {"state": state, "questions": {"q": {"type": "choice", **OPERATION}}}})
        group = [e for e in els if (editable(e) if want == "type" else not editable(e))]
        if len(group) >= 2:
            criteria = {number[id(e)]: describe(e)[:90] for e in group}
            out.append({"meta": {**meta, "family": f"step-{want}-target", "truth": number[id(pos)]},
                        "request": {"state": state, "questions": {"q": {"type": "choice", "instructions": CLICK_TARGET if want == "click" else TYPE_TARGET, "criteria": criteria}}}})
        v = clean(a["operation"].get("value") or "", 60)
        if op == "TYPE" and v and len(named) >= 2:
            field_state = {"goal": state["goal"], "values": state["values"], "page": state["page"], "field": describe(pos)}
            criteria = {**{k: (x if len(x) <= 48 else x[:48] + "…") for k, x in named.items()}, "none": "none of these values belongs in this field"}
            truth = next(k for k, x in named.items() if x == v)
            out.append({"meta": {**meta, "family": "step-value", "truth": truth},
                        "request": {"state": field_state, "questions": {"q": {"type": "choice", "instructions": f"Which of the person's values goes into the field “{pos['label']}”?", "criteria": criteria}}}})
        if op == "SELECT" and v and pos["role"] == "select" and len(pos["options"]) >= 2:
            opts = [o for o in pos["options"] if o]
            hit = next((o for o in opts if o.lower() == v.lower()), None) or next((o for o in opts if v.lower() in o.lower()), None)
            if hit:
                field_state = {"goal": state["goal"], "values": state["values"], "page": state["page"], "field": describe(pos)}
                out.append({"meta": {**meta, "family": "step-option", "truth": hit},
                            "request": {"state": field_state, "questions": {"q": {"type": "choice", "instructions": f"Which option of “{pos['label']}” fits the goal and the person's values?", "criteria": {o: None for o in opts}}}}})
    return out


def fetch(rows_dir: Path, tasks: int, start: int) -> None:
    """Mind2Web's training tasks from the Hugging Face datasets server, without raw_html (the bulk of each row)."""
    rows_dir.mkdir(parents=True, exist_ok=True)
    for offset in range(start, start + tasks, 3):
        path = rows_dir / f"rows-{offset:05d}.json"
        if path.exists():
            continue
        with urllib.request.urlopen(f"{ROWS_API}&offset={offset}&length=3", timeout=180) as res:
            body = json.load(res)
        rows = [r["row"] for r in body.get("rows", [])]
        for r in rows:
            for a in r["actions"]:
                a.pop("raw_html", None)
        path.write_text(json.dumps(rows), encoding="utf-8")
        print(f"{path.name}: {len(rows)} tasks")
        time.sleep(1)


def build(rows_dir: Path, out: Path, test: Path, truth_dir: Path | None) -> None:
    tasks = [t for p in sorted(rows_dir.glob("rows-*.json")) for t in json.loads(p.read_text(encoding="utf-8"))]
    sites = sorted({t["website"] for t in tasks})
    held = set(random.Random("2028-steps-sites").sample(sites, max(1, len(sites) // 4)))
    train, tests = [], []
    for t in tasks:
        split = "site-unseen" if t["website"] in held else "train"
        rows = rows_for_task(t, random.Random(f"2028-steps-{t['annotation_id']}"), split)
        (tests if split == "site-unseen" else train).extend(rows)
    for path, rows in ((out, train), (test, tests)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
        if truth_dir:
            truth_dir.mkdir(parents=True, exist_ok=True)
            (truth_dir / f"{path.stem}.truth.jsonl").write_text("".join(json.dumps(truth_row(r), ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
    print(f"{len(tasks)} tasks on {len(sites)} sites ({len(held)} held out)")
    for name, rows in (("train", train), ("test", tests)):
        print(f"{name}: {len(rows)} rows {dict(Counter(r['meta']['family'] for r in rows))}")


def truth_row(row: dict, sure: float = 4.0) -> dict:
    """A teacher-format row whose distribution is the recorded answer (build_numeric.truth_row's convention)."""
    labels = list(row["request"]["questions"]["q"]["criteria"])
    return {**row, "logits": {"q": [sure if label == row["meta"]["truth"] else 0.0 for label in labels]}, "temperature": 1.0, "teacher": "truth"}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--rows-dir", type=Path, required=True)
    f.add_argument("--tasks", type=int, default=60)
    f.add_argument("--start", type=int, default=0)
    b = sub.add_parser("build")
    b.add_argument("--rows-dir", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--test", type=Path, required=True)
    b.add_argument("--truth-dir", type=Path)
    args = parser.parse_args()
    if args.cmd == "fetch":
        fetch(args.rows_dir, args.tasks, args.start)
    else:
        build(args.rows_dir, args.out, args.test, args.truth_dir)


if __name__ == "__main__":
    main()
