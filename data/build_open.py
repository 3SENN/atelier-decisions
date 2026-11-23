# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""The open pool: decisions of every kind, not only Atelier's ten, so the student learns to decide whatever it is
asked (a pack's question, a routine's condition, label_items) instead of declining. Three sources, all labelled by
the teacher:

- gliclass-v2.0 (knowledgator, Apache-2.0): texts with arbitrary label sets;
- synthetic_zeroshot_mixtral_v0.1 (MoritzLaurer, Apache-2.0): texts written for hundreds of professional tasks;
- data/gen2 (written by Claude Fable 5.2 for this project): browser situations in 240 question families. A fifth
  of the families is held out whole (data/pool/open-heldout.jsonl), so the test is new KINDS of questions.

    python data/build_open.py --out data/pool/pool-open.jsonl --heldout data/pool/open-heldout.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"


def as_list(value) -> list[str]:
    """A label list stored as a list, or as the text of one."""
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return [str(x) for x in ast.literal_eval(value)]


def key_of(text: str, taken: set[str]) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    base = "-".join(w for w in words if w not in {"the", "a", "an", "is", "of", "text", "this", "or", "and", "to"})[:24].strip("-") or "option"
    key, n = base, 2
    while key in taken:
        key, n = f"{base}-{n}", n + 1
    taken.add(key)
    return key


def labels_question(labels: list[str], rng: random.Random) -> dict:
    if len(labels) == 1:
        return {"type": "noul", "instructions": f"Does this hold for the text: {labels[0]}"}
    taken: set[str] = set()
    criteria = {key_of(label, taken): label for label in labels}
    instructions = rng.choice(["Which of these best describes the text?", "Pick the option that fits the text best.",
                               "Which statement is true of this text?", "Classify the text."])
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def from_gliclass(rng: random.Random, n: int) -> list[dict]:
    rows = []
    for path in sorted((CACHE / "gliclass").glob("*.parquet")):
        for r in pq.read_table(path).to_pylist():
            try:
                labels = as_list(r["all_labels"])
            except (ValueError, SyntaxError):
                continue
            text = (r["text"] or "").strip()
            if 20 <= len(text) and 1 <= len(labels) <= 12 and all(len(x) < 200 for x in labels):
                rows.append((text[:1500], labels))
    out = []
    for text, labels in rng.sample(rows, min(n, len(rows))):
        labels = labels[:]
        rng.shuffle(labels)
        out.append({"meta": {"use": "open", "source": "gliclass", "lang": "en"}, "request": {"state": text, "questions": {"q": labels_question(labels, rng)}}})
    return out


def from_zeroshot(rng: random.Random, n: int) -> list[dict]:
    seen, rows = set(), []
    for path in sorted((CACHE / "zeroshot").glob("*.parquet")):
        for r in pq.read_table(path, columns=["text", "task_description", "task_hypotheses"]).to_pylist():
            text = (r["text"] or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            try:
                hyps = as_list(r["task_hypotheses"])
            except (ValueError, SyntaxError):
                continue
            if 2 <= len(hyps) <= 10:
                rows.append((text[:1500], (r["task_description"] or "").split(". ")[0].strip().rstrip("."), hyps))
    out = []
    for text, task, hyps in rng.sample(rows, min(n, len(rows))):
        hyps = hyps[:]
        rng.shuffle(hyps)
        taken: set[str] = set()
        q = {"type": "choice", "instructions": f"{task}. Which applies?" if task else "Which applies?",
             "criteria": {key_of(h, taken): h for h in hyps}}
        out.append({"meta": {"use": "open", "source": "zeroshot", "lang": "en"}, "request": {"state": text, "questions": {"q": q}}})
    return out


def from_gen2(rng: random.Random, holdout_share: float) -> tuple[list[dict], list[dict], dict]:
    items: dict[str, list[dict]] = defaultdict(list)
    skipped = 0
    for path in sorted((ROOT / "gen2").glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    q = r["question"]
                    assert q["type"] in ("choice", "noul", "score") and r["state"] and r["family"]
                    if q["type"] == "choice":
                        assert isinstance(q["criteria"], dict) and 2 <= len(q["criteria"]) <= 64
                    if q["type"] == "score":
                        assert isinstance(q["criteria"], list) and 2 <= len(q["criteria"]) <= 10
                except (json.JSONDecodeError, KeyError, AssertionError, TypeError):
                    skipped += 1
                    continue
                source = path.stem.rsplit("_", 1)[0]
                items[f"{source}/{r['family']}"].append({"meta": {"use": "open", "source": f"gen2-{source}", "family": r["family"],
                                                                  "lang": r.get("lang", "en")},
                                                         "request": {"state": r["state"], "questions": {"q": q}}})
    families = sorted(items)
    rng.shuffle(families)
    held = set(families[: int(len(families) * holdout_share)])
    train = [x for f in families if f not in held for x in items[f]]
    test = [x for f in held for x in items[f]]
    return train, test, {"families": len(families), "held_out_families": len(held), "skipped_lines": skipped}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--gliclass", type=int, default=8000)
    parser.add_argument("--zeroshot", type=int, default=6000)
    parser.add_argument("--holdout-share", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--no-gen2", action="store_true", help="leave the LLM-written families out (a second pool)")
    parser.add_argument("--exclude", type=Path, nargs="*", default=[], help="pools whose states to leave out")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    gen_train, gen_test, info = ([], [], {"families": 0, "held_out_families": 0, "skipped_lines": 0}) if args.no_gen2 else from_gen2(rng, args.holdout_share)
    seen = set()
    for path in args.exclude:
        with path.open(encoding="utf-8") as f:
            seen |= {json.dumps(json.loads(line)["request"]["state"], ensure_ascii=False) for line in f if line.strip()}
    fresh = lambda rows: [r for r in rows if json.dumps(r["request"]["state"], ensure_ascii=False) not in seen]  # noqa: E731
    rows = fresh(from_gliclass(rng, args.gliclass)) + fresh(from_zeroshot(rng, args.zeroshot)) + gen_train
    rng.shuffle(rows)
    for path, data in ((args.out, rows), (args.heldout, gen_test)) if not args.no_gen2 else ((args.out, rows),):
        with path.open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"open pool {len(rows)} ({dict(Counter(r['meta']['source'] for r in rows))}); held out {len(gen_test)} "
          f"questions in {info['held_out_families']} of {info['families']} families; skipped {info['skipped_lines']} lines")


if __name__ == "__main__":
    main()
