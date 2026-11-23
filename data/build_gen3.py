# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Pool gen3: 480 browser decision families written by Claude Fable 5.2 to add new KINDS of questions (data/gen3).

15% of the families are held out whole (--heldout) as a second test of new kinds, next to the 48 held-out gen2
families; a family whose name is one of those is never trained on.

    python data/build_gen3.py --out data/pool/pool-gen3.jsonl --heldout data/pool/gen3-heldout.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--heldout", type=Path, required=True, help="a share of the new families, held out whole: a second test of new kinds")
parser.add_argument("--holdout-share", type=float, default=0.15)
args = parser.parse_args()
held = {json.loads(l)["meta"]["family"] for l in open(ROOT / "pool" / "open-heldout.jsonl", encoding="utf-8")}
rows, skipped, clashes = [], 0, Counter()
for path in sorted((ROOT / "gen3").glob("*.jsonl")):
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
            if r["family"] in held:  # never train on a held-out family, even by name
                clashes[r["family"]] += 1
                continue
            source = path.stem.rsplit("_", 1)[0]
            rows.append({"meta": {"use": "open", "source": f"gen3-{source}", "family": r["family"], "lang": r.get("lang", "en")},
                         "request": {"state": r["state"], "questions": {"q": q}}})
import random  # noqa: E402

key = lambda r: r["meta"]["source"] + "/" + r["meta"]["family"]  # noqa: E731
fams = sorted({key(r) for r in rows})
random.Random(77).shuffle(fams)
test_fams = set(fams[: int(len(fams) * args.holdout_share)])
for path, part in ((args.out, [r for r in rows if key(r) not in test_fams]), (args.heldout, [r for r in rows if key(r) in test_fams])):
    with path.open("w", encoding="utf-8") as f:
        for row in part:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"{len(rows)} requests in {len(fams)} families; {len(test_fams)} families held out ({sum(key(r) in test_fams for r in rows)} requests); "
      f"skipped {skipped}; dropped for held-out names {dict(clashes)}")
