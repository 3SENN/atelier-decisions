"""Pool gen4: about 480 more browser decision families written by Claude Fable 5.2 (data/gen4): transactions and
forms (shop_*), developer and work tools (work_*), everyday life in 20 languages and scripts (world_*), and the agent's
own judgement while acting on pages (agent_*). Written against the list of the 720 families that already existed
(data/existing_families.txt), with answers that vary within each family.

15% of the families are held out whole (--heldout) as a third test of new kinds of questions. A family whose name
is already a family of gen2 or gen3 is dropped: those are either trained on already, or held out as earlier tests.

    python data/build_gen4.py --out data/pool/pool-gen4.jsonl --heldout data/pool/gen4-heldout.jsonl
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--heldout", type=Path, required=True)
parser.add_argument("--holdout-share", type=float, default=0.15)
args = parser.parse_args()
existing = {line.strip() for line in open(ROOT / "existing_families.txt", encoding="utf-8") if line.strip()}
rows, skipped, clashes = [], Counter(), Counter()
for path in sorted((ROOT / "gen4").glob("*.jsonl")):
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                q = r["question"]
                assert q["type"] in ("choice", "noul", "score") and r["state"] and r["family"] and q.get("instructions")
                if q["type"] == "choice":
                    assert isinstance(q["criteria"], dict) and 2 <= len(q["criteria"]) <= 64
                if q["type"] == "score":
                    assert isinstance(q["criteria"], list) and 2 <= len(q["criteria"]) <= 10
                if q["type"] == "noul" and "criteria" in q:
                    assert isinstance(q["criteria"], dict) and set(q["criteria"]) <= {"true", "false"}
            except (json.JSONDecodeError, KeyError, AssertionError, TypeError):
                skipped[path.stem] += 1
                continue
            if r["family"] in existing:
                clashes[r["family"]] += 1
                continue
            source = path.stem.rsplit("_", 1)[0]
            rows.append({"meta": {"use": "open", "source": f"gen4-{source}", "family": r["family"], "lang": r.get("lang", "en")},
                         "request": {"state": r["state"], "questions": {"q": q}}})

key = lambda r: r["meta"]["source"] + "/" + r["meta"]["family"]  # noqa: E731
fams = sorted({key(r) for r in rows})
random.Random(78).shuffle(fams)
test_fams = set(fams[: int(len(fams) * args.holdout_share)])
for path, part in ((args.out, [r for r in rows if key(r) not in test_fams]), (args.heldout, [r for r in rows if key(r) in test_fams])):
    with path.open("w", encoding="utf-8") as f:
        for row in part:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
by_source = Counter(r["meta"]["source"] for r in rows)
langs = Counter(r["meta"]["lang"] for r in rows)
print(f"{len(rows)} requests in {len(fams)} families ({dict(by_source)}); {len(test_fams)} families held out "
      f"({sum(key(r) in test_fams for r in rows)} requests); skipped {dict(skipped)}; dropped for existing names {dict(clashes)}")
print("languages:", dict(langs.most_common()))
