"""Pool v3: targeted requests for what pool v1 covered badly (skills by meaning, facts in passing, effort).

    python data/build_targeted.py --out data/pool/pool-v3.jsonl --exclude data/pool/pool-v1.jsonl data/pool/pool-v2.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atelier_uses as U  # noqa: E402
import lexicon_targeted as T  # noqa: E402
from build_pool import lang  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--exclude", type=Path, nargs="*", default=[])
args = parser.parse_args()
rng = random.Random(args.seed)

rows = []
for _ in range(1600):
    code = lang(rng)
    skills, message = T.skill_request(rng, code)
    rows.append(({"use": "skills", "source": "targeted", "lang": code}, U.skills(message, skills)))
for _ in range(3500):
    code = lang(rng)
    rows.append(({"use": "memory", "source": "targeted", "lang": code}, U.memory(T.memory_chat(rng, code))))
for _ in range(4000):
    code = lang(rng)
    message, previous = T.effort_message_v2(rng, code)
    rows.append(({"use": "effort", "source": "targeted", "lang": code}, U.effort(message, previous)))

seen = set()
for path in args.exclude:
    with path.open(encoding="utf-8") as f:
        seen |= {json.dumps(json.loads(line)["request"], sort_keys=True, ensure_ascii=False) for line in f if line.strip()}
unique = []
for meta, request in rows:
    key = json.dumps(request, sort_keys=True, ensure_ascii=False)
    if key not in seen:
        seen.add(key)
        unique.append({"meta": meta, "request": request})
rng.shuffle(unique)
args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open("w", encoding="utf-8") as f:
    for row in unique:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
by = {}
for r in unique:
    by[r["meta"]["use"]] = by.get(r["meta"]["use"], 0) + 1
print(f"{len(unique)} requests, {sum(len(r['request']['questions']) for r in unique)} questions: {by}")
