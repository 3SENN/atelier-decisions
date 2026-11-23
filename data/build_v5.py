# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Pool v5: natural variety where templates generalised badly (found by probing the student on Atelier's eval).

- skills: glaive-function-calling-v2 (Apache-2.0): 113k real requests, each with the functions it was offered.
  A request is scored against its own functions (often the right one, sometimes not: glaive has "I can't do that"
  answers) plus functions drawn from the whole catalogue (thousands of distinct tools), written as skills.
- memory: Synthetic-Persona-Chat (CC BY 4.0): ~40k first-person facts ("I have a black belt in karate") said in
  passing next to an everyday task, dialogue excerpts where people reveal themselves, and lookups with nothing to
  keep, so the teacher shows what is worth remembering across far more than a few dozen template facts.

    python data/build_v5.py --out data/pool/pool-v5.jsonl
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
import atelier_uses as U  # noqa: E402
import lexicon_targeted as T  # noqa: E402

FUNC_RE = re.compile(r'"name":\s*"([^"]+)",\s*"description":\s*"([^"]+)"')


def glaive() -> tuple[list[tuple[str, list[tuple[str, str]]]], list[tuple[str, str]]]:
    data = json.loads((ROOT / "cache" / "glaive" / "glaive-v2.json").read_text(encoding="utf-8"))
    examples, catalog = [], {}
    for row in data:
        functions = [(name.replace("_", "-"), desc) for name, desc in FUNC_RE.findall(row["system"])]
        m = re.match(r"\s*USER:\s*(.*?)\s*(?:\n\s*\n\s*ASSISTANT:|$)", row["chat"], re.S)
        if not functions or not m:
            continue
        message = m.group(1).strip()
        if 8 <= len(message) <= 400:
            examples.append((message, functions))
            catalog.update(functions)
    return examples, sorted(catalog.items())


def personas() -> tuple[list[str], list[list[tuple[str, str]]]]:
    csv.field_size_limit(10**8)
    facts, dialogues = set(), []
    with (ROOT / "cache" / "persona" / "train.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for column in ("user 1 personas", "user 2 personas"):
                facts.update(x.strip() for x in row[column].split("\n") if len(x.strip()) > 8)
            turns = []
            for line in row["Best Generated Conversation"].split("\n"):
                m = re.match(r"User ([12]):\s*(.*)", line.strip())
                if m and m.group(2):
                    turns.append(("You" if m.group(1) == "1" else "Companion", m.group(2)))
            if len(turns) >= 4:
                dialogues.append(turns)
    return sorted(facts), dialogues


EVERYDAY = ["can you find me a recipe for tonight?", "what should I do this weekend?", "help me write a message to my landlord",
            "find a gift for my sister", "what's a good podcast to start?", "plan a short trip for next month",
            "suggest a book for the train", "can you look up opening hours of the gym?", "draft a reply to my neighbour"]
LOOKUPS = ["what's 18% of 240?", "how far is the moon?", "who directed Inception?", "what's the time in Sydney?",
           "how do I convert cups to grams?", "what's the boiling point of water in Fahrenheit?"]
ACKS = ["Sure, here are a few ideas.", "Here's what I found.", "Of course! Take a look below.", "Done, here it is."]
PREFERENCES = ["from now on, {x}", "please always {x}", "remember that {x}", "keep in mind for next time: {x}", "going forward, {x}"]
PREF_X = ["keep replies under five lines", "write in British English", "use metric units", "call me Sam", "don't use emojis",
          "check with me before booking anything", "send summaries as bullet points", "assume I'm vegetarian when suggesting food",
          "I take the train, not the car", "my working hours are 9 to 5"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skills", type=int, default=3000)
    parser.add_argument("--memory", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows, seen = [], set()

    def add(meta: dict, request: dict) -> None:
        key = json.dumps(request, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            rows.append({"meta": meta, "request": request})

    examples, catalog = glaive()
    print(f"glaive: {len(examples)} requests, {len(catalog)} distinct functions")
    for message, own in rng.sample(examples, args.skills):
        k = rng.choice([3, 5, 8])
        chunk = list(dict.fromkeys(own + rng.sample(catalog, k)))[:k]
        rng.shuffle(chunk)
        add({"use": "skills", "source": "glaive", "lang": "en"},
            U.skills(message, [{"name": n, "description": d} for n, d in chunk]))

    facts, dialogues = personas()
    print(f"personas: {len(facts)} facts, {len(dialogues)} dialogues")
    for _ in range(args.memory):
        r = rng.random()
        if r < 0.4:  # a fact in passing, then a task
            fact = rng.choice(facts).rstrip(".")
            task = rng.choice(EVERYDAY)
            you = rng.choice([f"{fact}. {task[0].upper()}{task[1:]}", f"{task[0].upper()}{task[1:]} ({fact[0].lower()}{fact[1:]})"])
            text = f"You: {you}\nCompanion: {rng.choice(ACKS)}"
        elif r < 0.6:  # a dialogue excerpt
            turns = rng.choice(dialogues)
            start = rng.randrange(0, max(1, len(turns) - 4))
            text = "\n".join(f"{who}: {line}" for who, line in turns[start : start + rng.choice([2, 4, 6])])
        elif r < 0.75:  # an explicit lasting preference
            pref = rng.choice(PREFERENCES).format(x=rng.choice(PREF_X))
            text = f"You: {pref}\nCompanion: {rng.choice(['Got it.', 'Will do.', 'Noted, I will.'])}"
        else:  # a lookup or a one-off task: nothing lasting
            q = rng.choice(LOOKUPS + EVERYDAY + T.UNRELATED["en"])
            text = f"You: {q}\nCompanion: {rng.choice(ACKS)}"
        add({"use": "memory", "source": "persona", "lang": "en"}, U.memory(text))

    rng.shuffle(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} requests, {sum(len(r['request']['questions']) for r in rows)} questions -> {args.out}")


if __name__ == "__main__":
    main()
