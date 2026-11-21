"""Builds the unlabelled request pool for distillation: realistic inputs for each of Atelier's decisions, in
Atelier's exact request format (data/atelier_uses.py). The teacher labels them; nothing here needs to be "right".

Sources (all permissive): MASSIVE (CC BY 4.0, en/nl/de/fr/es voice-assistant utterances), WildChat-1M (ODC-BY,
real chat prompts and assistant replies), Mind2Web (CC BY 4.0, web controls), plus combinatorial multilingual
templates for what those lack (search queries, buttons, tabs, agent replies, edits).

    python data/build_pool.py --out data/pool/pool-v1.jsonl --seed 1
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import atelier_uses as U  # noqa: E402
import lexicon as L  # noqa: E402

CACHE = ROOT / "cache"
LANGS = ["en", "nl", "de", "fr", "es"]
LANG_WEIGHTS = [0.45, 0.25, 0.10, 0.10, 0.10]
MASSIVE_LOCALE = {"en": "en-US", "nl": "nl-NL", "de": "de-DE", "fr": "fr-FR", "es": "es-ES"}
WILDCHAT_LANG = {"English": "en", "Dutch": "nl", "German": "de", "French": "fr", "Spanish": "es"}


def lang(rng: random.Random) -> str:
    return rng.choices(LANGS, LANG_WEIGHTS)[0]


# ---------------------------------------------------------------------------------------------------- sources

def massive() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for code, locale in MASSIVE_LOCALE.items():
        path = CACHE / "massive" / "1.1" / "data" / f"{locale}.jsonl"
        with path.open(encoding="utf-8") as f:  # not splitlines(): JSON strings may hold U+2028
            out[code] = [json.loads(line) for line in f if line.strip()]
    return out


def wildchat(limit_rows: int | None = None) -> list[dict]:
    import pyarrow.parquet as pq

    rows: list[dict] = []
    for path in sorted((CACHE / "wildchat").glob("*.parquet")):
        table = pq.read_table(path, columns=["conversation", "language", "toxic", "redacted", "turn"])
        for row in table.to_pylist():
            code = WILDCHAT_LANG.get(row["language"])
            if not code or row["toxic"] or row["redacted"]:
                continue
            turns = [(m["role"], (m["content"] or "").strip()) for m in row["conversation"]]
            if not turns or any(not text for _, text in turns):
                continue
            rows.append({"lang": code, "turns": turns})
            if limit_rows and len(rows) >= limit_rows:
                return rows
    return rows


def mind2web_controls() -> list[dict]:
    """Element labels from Mind2Web's cleaned HTML: buttons, links and submit inputs, with the task's site."""
    out: list[dict] = []
    for path in (CACHE / "mind2web").glob("*.json"):
        for task in json.loads(path.read_text(encoding="utf-8")):
            site = f"https://www.{task['website']}.com/"
            for action in task["actions"]:
                html = action.get("cleaned_html") or ""
                # Mind2Web's cleaned HTML puts a control's words in a <text> child: <a ...><text ...>Sign in</text>
                for tag, text in re.findall(r"<(button|a|input|label|option)\b[^>]*>\s*<text[^>]*>([^<]{2,60})</text>", html):
                    text = " ".join(html_lib.unescape(text).split())
                    if 2 <= len(text) <= 60 and not text.isdigit():
                        out.append({"name": text, "role": "link" if tag == "a" else "button", "site": site,
                                    "page": task["confirmed_task"][:60]})
    seen, unique = set(), []
    for c in out:
        if c["name"].lower() not in seen:
            seen.add(c["name"].lower())
            unique.append(c)
    return unique


def short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0]


# ---------------------------------------------------------------------------------------------------- per use

def pool_search_or_ask(rng, n, mas, wild):
    out = []
    for _ in range(n):
        r = rng.random()
        code = lang(rng)
        if r < 0.35:
            text = L.search_query(rng, code)
            src = "template-search"
        elif r < 0.55:
            text = L.ask_query(rng, code)
            src = "template-ask"
        elif r < 0.80:
            text = rng.choice(mas[code])["utt"]
            src = "massive"
        else:
            convo = rng.choice(wild)
            text, src, code = short(convo["turns"][0][1], rng.choice([60, 120, 300])), "wildchat", convo["lang"]
        if rng.random() < 0.3:
            text = text.lower().rstrip("?.!")
        out.append((U.search_or_ask(text), src, code))
    return out


def pool_needs_you(rng, n, wild):
    out = []
    replies = [(c["lang"], text) for c in wild for role, text in c["turns"] if role == "assistant"]
    for _ in range(n):
        if rng.random() < 0.55:
            code, text = rng.choice(replies)
            # Atelier sends the end of the reply; vary how much of it there is.
            text = text[-rng.choice([200, 600, 1500]):]
            src = "wildchat"
        else:
            code = lang(rng)
            text, src = L.agent_reply(rng, code), "template-reply"
        out.append((U.needs_you(text), src, code))
    return out


def pool_effort(rng, n, mas, wild):
    out = []
    for _ in range(n):
        r = rng.random()
        if r < 0.45:
            convo = rng.choice(wild)
            code = convo["lang"]
            users = [i for i, (role, _) in enumerate(convo["turns"]) if role == "user"]
            i = rng.choice(users)
            previous = convo["turns"][i - 1][1] if i > 0 else ""
            out.append((U.effort(convo["turns"][i][1], previous), "wildchat", code))
        elif r < 0.65:
            code = lang(rng)
            out.append((U.effort(rng.choice(mas[code])["utt"]), "massive", code))
        else:
            code = lang(rng)
            message, previous = L.chat_message(rng, code)
            out.append((U.effort(message, previous), "template-message", code))
    return out


def transcript(turns: list[tuple[str, str]], rng) -> str:
    lines = []
    for role, text in turns[: rng.choice([2, 2, 4, 6])]:
        who = "You" if role == "user" else "Companion"
        lines.append(f"{who}: {short(text, rng.choice([300, 800, 1500]))}")
    return "\n".join(lines)


def pool_memory(rng, n, wild):
    out = []
    for _ in range(n):
        if rng.random() < 0.6:
            convo = rng.choice(wild)
            out.append((U.memory(transcript(convo["turns"], rng)), "wildchat", convo["lang"]))
        else:
            code = lang(rng)
            out.append((U.memory(L.memory_chat(rng, code)), "template-chat", code))
    return out


def pool_dictation(rng, n, mas, wild):
    out = []
    for _ in range(n):
        code = lang(rng)
        field = L.field_text(rng, code) if rng.random() < 0.7 else ""
        r = rng.random()
        if r < 0.4:
            spoken, src = L.edit_request(rng, code), "template-edit"
        elif r < 0.75:
            spoken, src = L.dictated_text(rng, code), "template-dictation"
        elif r < 0.9:
            convo = rng.choice(wild)
            spoken, src, code = short(convo["turns"][0][1], 200), "wildchat", convo["lang"]
        else:
            spoken, src = rng.choice(mas[code])["utt"], "massive"
        out.append((U.dictation(spoken, field), src, code))
    return out


def pool_voice_target(rng, n, mas):
    out = []
    for _ in range(n):
        code = lang(rng)
        label = rng.choice(L.FIELD_LABELS[code])
        value = L.field_text(rng, code) if rng.random() < 0.4 else ""
        r = rng.random()
        if r < 0.45:
            spoken, src = rng.choice(mas[code])["utt"], "massive"
        elif r < 0.75:
            spoken, src = L.dictated_text(rng, code), "template-dictation"
        else:
            spoken, src = L.edit_request(rng, code), "template-edit"
        out.append((U.voice_target(spoken, label, value), src, code))
    return out


def pool_clicks(rng, n, m2w):
    out = []
    for _ in range(n):
        if m2w and rng.random() < 0.2:
            c = rng.choice(m2w)
            out.append((U.agent_click(c["name"], c["role"], c["page"], c["site"]), "mind2web", "en"))
        else:
            code = lang(rng)
            name, role, page, url = L.control(rng, code)
            out.append((U.agent_click(name, role, page, url), "template-control", code))
    return out


def pool_tabs(rng, n):
    out = []
    for _ in range(n):
        code = lang(rng)
        title, url = L.tab(rng, code)
        groups = rng.sample(L.GROUPS[code], k=min(len(L.GROUPS[code]), rng.choice([0, 0, 1, 2, 3, 4])))
        out.append((U.tabs(title, url, groups), "template-tab", code))
    return out


def pool_skills(rng, n, mas, wild):
    out = []
    for _ in range(n):
        code = lang(rng)
        chunk = rng.sample(L.SKILLS, k=rng.choice([3, 5, 8]))
        r = rng.random()
        if r < 0.4:
            convo = rng.choice(wild)
            message, code = short(convo["turns"][0][1], 400), convo["lang"]
        elif r < 0.6:
            message = rng.choice(mas[code])["utt"]
        else:
            # A message aimed at one of the offered skills, so positives are common enough.
            message = L.skill_message(rng, rng.choice(chunk), code)
        out.append((U.skills(message, chunk), "mixed", code))
    return out


def pool_labeling(rng, n):
    out = []
    for _ in range(n):
        code = lang(rng)
        family = rng.choice(L.LABEL_FAMILIES)
        item = family["item"](rng, code)
        questions = family["questions"]
        out.append((U.label_item(item, questions, family["context"]), f"label-{family['name']}", code))
    return out


SIZES = {"searchOrAsk": 12000, "needsYou": 9000, "effort": 9000, "memory": 5000, "dictation": 7000,
         "voiceTarget": 7000, "agentClicks": 12000, "tabs": 9000, "skills": 2500, "labeling": 6000}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[], help="pools whose requests to leave out")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    mas, wild, m2w = massive(), wildchat(), mind2web_controls()
    print(f"sources: massive {sum(map(len, mas.values()))}, wildchat {len(wild)}, mind2web controls {len(m2w)}")
    n = {k: int(v * args.scale) for k, v in SIZES.items()}
    rows = []
    for use, items in [
        ("searchOrAsk", pool_search_or_ask(rng, n["searchOrAsk"], mas, wild)),
        ("needsYou", pool_needs_you(rng, n["needsYou"], wild)),
        ("effort", pool_effort(rng, n["effort"], mas, wild)),
        ("memory", pool_memory(rng, n["memory"], wild)),
        ("dictation", pool_dictation(rng, n["dictation"], mas, wild)),
        ("voiceTarget", pool_voice_target(rng, n["voiceTarget"], mas)),
        ("agentClicks", pool_clicks(rng, n["agentClicks"], m2w)),
        ("tabs", pool_tabs(rng, n["tabs"])),
        ("skills", pool_skills(rng, n["skills"], mas, wild)),
        ("labeling", pool_labeling(rng, n["labeling"])),
    ]:
        for request, src, code in items:
            rows.append({"meta": {"use": use, "source": src, "lang": code}, "request": request})
    # Exact duplicates add nothing, within this pool or against earlier ones.
    seen, unique = set(), []
    for path in args.exclude:
        with path.open(encoding="utf-8") as f:
            seen |= {json.dumps(json.loads(line)["request"], sort_keys=True, ensure_ascii=False) for line in f if line.strip()}
    for row in rows:
        key = json.dumps(row["request"], sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    rng.shuffle(unique)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in unique:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    by_use: dict[str, int] = {}
    for row in unique:
        by_use[row["meta"]["use"]] = by_use.get(row["meta"]["use"], 0) + 1
    seqs = sum(len(r["request"]["questions"]) for r in unique)
    print(f"{len(unique)} requests ({seqs} questions) -> {args.out}: {by_use}")


if __name__ == "__main__":
    main()
