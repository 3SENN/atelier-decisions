"""A probe of open questions the student never saw (new instructions, new option sets), to decide whether the
on-device model should answer packs', routines' and label_items' arbitrary questions or hand them back.

    python data/build_probe_open.py --out data/pool/probe-open.jsonl
"""

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from build_v5 import glaive  # noqa: E402

AREAS = {
    "alarm": "Alarms and wake-up times.", "calendar": "Appointments and the agenda.", "play": "Playing music, radio, podcasts or games.",
    "weather": "The weather.", "qa": "General questions and facts.", "iot": "Smart home devices: lights, plugs, vacuum, coffee.",
    "email": "Email and contacts.", "transport": "Trains, taxis, traffic and tickets.", "cooking": "Cooking and recipes.",
    "news": "News.", "recommendation": "Recommendations for places, events or films.", "lists": "Lists and to-dos.",
}
QUESTIONS = [
    ("purchase", {"type": "noul", "instructions": "Is this a request to buy, order or book something?"}),
    ("money", {"type": "noul", "instructions": "Does this involve money: prices, payments, rates or costs?"}),
    ("urgency", {"type": "score", "instructions": "How urgent does this sound?", "criteria": ["Not urgent", "Somewhat urgent", "Very urgent"]}),
]
FAMILY = {"type": "noul", "instructions": "Does the person mention someone in their family (a partner, child, parent, sibling)?"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows = []
    # Texts the student trained on (with whatever question) are not used: new questions about new text.
    seen = set()
    for pool in sorted((ROOT / "pool").glob("pool-v*.jsonl")):
        with pool.open(encoding="utf-8") as f:
            for line in f:
                state = json.loads(line)["request"]["state"]
                seen.add(state if isinstance(state, str) else json.dumps(state, sort_keys=True))
    massive = []
    for locale in ("en-US", "nl-NL"):
        with (ROOT / "cache" / "massive" / "1.1" / "data" / f"{locale}.jsonl").open(encoding="utf-8") as f:
            massive += [json.loads(line) for line in f]
    massive = [m for m in massive if m["scenario"] in AREAS and m["utt"].strip() not in seen and m["utt"].strip()[:500] not in seen]
    for m in rng.sample(massive, 200):
        options = dict(rng.sample(sorted(AREAS.items()), 6))
        options.setdefault(m["scenario"], AREAS[m["scenario"]])
        keys = list(options)
        rng.shuffle(keys)
        rows.append({"meta": {"use": "open", "source": "probe-areas", "lang": m["locale"][:2]},
                     "request": {"state": m["utt"], "questions": {"area": {"type": "choice", "instructions": "Which area is this voice request about?",
                                                                          "criteria": {k: options[k] for k in keys}}}}})
    examples, _ = glaive()
    examples = [e for e in examples if e[0] not in seen and e[0][:2000] not in seen]
    for message, _ in rng.sample(examples, 150):
        key, question = rng.choice(QUESTIONS)
        rows.append({"meta": {"use": "open", "source": f"probe-{key}", "lang": "en"}, "request": {"state": message, "questions": {key: question}}})
    csv.field_size_limit(10**8)
    with (ROOT / "cache" / "persona" / "train.csv").open(encoding="utf-8") as f:
        convos = [row["Best Generated Conversation"] for row in csv.DictReader(f)]
    for convo in rng.sample(convos, 100):
        lines = [re.sub(r"^User 1:", "You:", re.sub(r"^User 2:", "Companion:", line.strip())) for line in convo.split("\n") if line.strip()][:6]
        rows.append({"meta": {"use": "open", "source": "probe-family", "lang": "en"}, "request": {"state": "\n".join(lines), "questions": {"family": FAMILY}}})
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} probe requests -> {args.out}")


if __name__ == "__main__":
    main()
