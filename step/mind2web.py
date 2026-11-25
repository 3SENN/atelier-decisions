# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Mind2Web's recorded browser steps (CC BY 4.0, osunlp/Mind2Web), kept small: for every action, the task, the values
it types or chooses, what was done before, and the page's candidate elements parsed into the runtime's element
lines (the recorded one and up to 79 others, in page order). The HTML is not kept: the whole training split comes
to tens of megabytes instead of gigabytes.

    python -I step/mind2web.py fetch --out data/cache/m2w-steps.jsonl            # resumable; one task per line
    python -I step/mind2web.py from-rows --rows-dir data/cache/mind2web_rows --out data/cache/m2w-steps.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
import build_steps as bs  # noqa: E402  (Tree, element, describe: one parser for both)

ROWS_API = "https://datasets-server.huggingface.co/rows?dataset=osunlp/Mind2Web&config=default&split=train"
KEEP_NEGATIVES = 79


def compact(task: dict) -> dict:
    """One task as {id, site, domain, goal, values, steps: [{op, value, before, elements, gold}]}."""
    values = []
    for a in task["actions"]:
        v = bs.clean(a["operation"].get("value") or "", 60)
        if a["operation"]["op"] in ("TYPE", "SELECT") and v and v not in values:
            values.append(v)
    steps = []
    rng = random.Random(f"m2w-{task['annotation_id']}")
    for i, a in enumerate(task["actions"]):
        tree = bs.Tree()
        try:
            tree.feed(a.get("cleaned_html") or "")
        except Exception:
            continue
        order = {nid: k for k, nid in enumerate(tree.nodes)}  # document order of every node
        gold_id = next((c["backend_node_id"] for c in a["pos_candidates"] if c["backend_node_id"] in tree.nodes and bs.element(tree.nodes[c["backend_node_id"]])), None)
        if gold_id is None:
            continue
        gold = bs.element(tree.nodes[gold_id])
        picked = []
        seen = {bs.describe(gold)}
        for c in rng.sample(a["neg_candidates"], len(a["neg_candidates"])):
            nid = c["backend_node_id"]
            e = bs.element(tree.nodes[nid]) if nid in tree.nodes else None
            if e and bs.describe(e) not in seen:
                seen.add(bs.describe(e))
                picked.append((order[nid], e))
            if len(picked) >= KEEP_NEGATIVES:
                break
        picked.append((order[gold_id], {**gold, "gold": True}))
        picked.sort(key=lambda x: x[0])
        elements = [e for _, e in picked]
        g = next(k for k, e in enumerate(elements) if e.pop("gold", False))
        op = a["operation"]["op"]
        steps.append({"op": "click" if op == "CLICK" else "type", "select": op == "SELECT", "value": bs.clean(a["operation"].get("value") or "", 60),
                      "before": task["action_reprs"][:i], "elements": elements, "gold": g})
    return {"id": task["annotation_id"], "site": task["website"], "domain": task.get("domain", ""), "subdomain": task.get("subdomain", ""),
            "goal": task["confirmed_task"], "values": values, "steps": steps}


def fetch(out: Path, start: int, total: int) -> None:
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    out.parent.mkdir(parents=True, exist_ok=True)
    offset = start
    with out.open("a", encoding="utf-8") as f:
        while offset < start + total:
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(f"{ROWS_API}&offset={offset}&length=3", timeout=240) as res:
                        body = json.load(res)
                    break
                except Exception as err:  # the datasets server is sometimes slow; try again
                    print(f"offset {offset}: {err}; again in {10 * (attempt + 1)} s", flush=True)
                    time.sleep(10 * (attempt + 1))
            else:
                offset += 3
                continue
            rows = [r["row"] for r in body.get("rows", [])]
            if not rows:
                break
            for r in rows:
                if r["annotation_id"] not in done:
                    f.write(json.dumps(compact(r), ensure_ascii=False) + "\n")
                    done.add(r["annotation_id"])
            f.flush()
            offset += 3
            if offset % 60 == 0:
                print(f"{len(done)} tasks", flush=True)
    print(f"{len(done)} tasks in {out}")


def from_rows(rows_dir: Path, out: Path) -> None:
    """Tasks fetched earlier with data/build_steps.py fetch, converted."""
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    with out.open("a", encoding="utf-8") as f:
        for p in sorted(rows_dir.glob("rows-*.json")):
            for r in json.loads(p.read_text(encoding="utf-8")):
                if r["annotation_id"] not in done:
                    f.write(json.dumps(compact(r), ensure_ascii=False) + "\n")
                    done.add(r["annotation_id"])
    print(f"{len(done)} tasks in {out}")


def system_one_rows(path: Path, out: Path, n: int) -> None:
    """The held-out sites' steps (step/train.py's split) as System One questions, for any decision model:
    which operation, and which element, among the same elements atelier-step sees (data/build_steps.py's words)."""
    import zlib

    rng = random.Random("2028-step-compare")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        if zlib.crc32(t["site"].encode()) % 4 != 0:
            continue
        named = {f"value {k + 1}": v for k, v in enumerate(t["values"])}
        for i, s in enumerate(t["steps"]):
            k = rng.randint(15, 60)
            others = [j for j in range(len(s["elements"])) if j != s["gold"]]
            pick = sorted(rng.sample(others, min(k - 1, len(others))) + [s["gold"]])
            els = [s["elements"][j] for j in pick]
            gold = pick.index(s["gold"])
            want = s["op"]
            if (want == "type") != bs.editable(els[gold]):
                continue
            state = {"goal": t["goal"], "values": "\n".join(f"{a}: {b}" for a, b in named.items()) or "none", "page": f"{t['site']} ({t['domain']}, {t['subdomain']})",
                     "elements": "\n".join(f"[{j + 1}] {bs.describe(e)}" for j, e in enumerate(els)), "done_so_far": bs.recent(s["before"])}
            meta = {"use": "open", "source": "mind2web-steps", "split": "site-unseen", "site": t["site"], "task": t["id"], "step": i}
            rows.append({"meta": {**meta, "family": "step-operation", "truth": want}, "request": {"state": state, "questions": {"q": {"type": "choice", **bs.OPERATION}}}})
            group = [(j, e) for j, e in enumerate(els) if bs.editable(e) == (want == "type")]
            if len(group) >= 2:
                criteria = {str(j + 1): bs.describe(e)[:90] for j, e in group}
                rows.append({"meta": {**meta, "family": f"step-{want}-target", "truth": str(gold + 1)},
                             "request": {"state": state, "questions": {"q": {"type": "choice", "instructions": bs.CLICK_TARGET if want == "click" else bs.TYPE_TARGET, "criteria": criteria}}}})
    rng.shuffle(rows)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[:n]), encoding="utf-8")
    print(f"{min(n, len(rows))} of {len(rows)} questions in {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--start", type=int, default=0)
    f.add_argument("--total", type=int, default=1009)
    r = sub.add_parser("from-rows")
    r.add_argument("--rows-dir", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("system-one")
    q.add_argument("--steps", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--n", type=int, default=400)
    args = parser.parse_args()
    if args.cmd == "system-one":
        return system_one_rows(args.steps, args.out, args.n)
    if args.cmd == "fetch":
        fetch(args.out, args.start, args.total)
    else:
        from_rows(args.rows_dir, args.out)


if __name__ == "__main__":
    main()
