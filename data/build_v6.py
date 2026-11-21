"""Pool v6: natural inputs written by a strong LLM (data/gen/*.jsonl), in Atelier's exact requests.

The writers saw none of Atelier's eval cases; they wrote inputs only (their "intended" field is ignored: the
teacher labels everything). Lines that are not valid JSON or lack a field are skipped and counted.

    python data/build_v6.py --out data/pool/pool-v6.jsonl --exclude data/pool/pool-v*.jsonl
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import atelier_uses as U  # noqa: E402

GEN = ROOT / "gen"
skipped: Counter = Counter()


def lines(pattern: str, fields: tuple[str, ...]) -> list[dict]:
    out = []
    for path in sorted(GEN.glob(pattern)):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped[path.name] += 1
                    continue
                if not isinstance(row, dict) or any(not isinstance(row.get(k), (str, list)) for k in fields):
                    skipped[path.name] += 1
                    continue
                out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows = []

    def add(use: str, lang: str, request: dict) -> None:
        rows.append({"meta": {"use": use, "source": "llm-written", "lang": lang if lang in ("en", "nl", "de", "fr", "es") else "en"},
                     "request": request})

    catalog = {r["name"]: r["description"] for r in lines("skills_catalog*.jsonl", ("name", "description"))}
    names = sorted(catalog)
    for r in lines("skills_requests*.jsonl", ("skill", "message")):
        if r["skill"] not in catalog or len(names) < 9:
            continue
        for _ in range(2):  # the same request against two different line-ups: once offered, once (often) not
            k = rng.choice([3, 5, 8])
            others = rng.sample([n for n in names if n != r["skill"]], k)
            chunk = others[: k - 1] + [r["skill"]] if rng.random() < 0.6 else others
            rng.shuffle(chunk)
            add("skills", r.get("lang", "en"), U.skills(r["message"], [{"name": n, "description": catalog[n]} for n in chunk]))
    for r in lines("skills_none*.jsonl", ("message",)):
        if len(names) >= 8:
            chunk = rng.sample(names, rng.choice([3, 5, 8]))
            add("skills", r.get("lang", "en"), U.skills(r["message"], [{"name": n, "description": catalog[n]} for n in chunk]))
    for r in lines("effort*.jsonl", ("message",)):
        add("effort", r.get("lang", "en"), U.effort(r["message"], r.get("previous_reply_end") or ""))
    for r in lines("memory*.jsonl", ("transcript",)):
        add("memory", r.get("lang", "en"), U.memory(r["transcript"]))
    for r in lines("needsyou*.jsonl", ("reply",)):
        add("needsYou", r.get("lang", "en"), U.needs_you(r["reply"]))
    for r in lines("voice*.jsonl", ("spoken",)):
        add("dictation", r.get("lang", "en"), U.dictation(r["spoken"], r.get("field_text") or ""))
        add("voiceTarget", r.get("lang", "en"), U.voice_target(r["spoken"], r.get("field_label") or "Search", r.get("field_text") or ""))
    for r in lines("search*.jsonl", ("text",)):
        add("searchOrAsk", r.get("lang", "en"), U.search_or_ask(r["text"]))
    for r in lines("clicks*.jsonl", ("name", "url")):
        add("agentClicks", r.get("lang", "en"), U.agent_click(r["name"], r.get("role") or "button", r.get("page") or "", r["url"]))
    for r in lines("tabs*.jsonl", ("title", "url")):
        groups = [g for g in (r.get("groups") or []) if isinstance(g, str)][:8]
        add("tabs", r.get("lang", "en"), U.tabs(r["title"], r["url"], groups))

    seen = set()
    for path in args.exclude:
        with path.open(encoding="utf-8") as f:
            seen |= {json.dumps(json.loads(line)["request"], sort_keys=True, ensure_ascii=False) for line in f if line.strip()}
    unique = []
    for row in rows:
        key = json.dumps(row["request"], sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    rng.shuffle(unique)
    with args.out.open("w", encoding="utf-8") as f:
        for row in unique:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(unique)} requests ({sum(len(r['request']['questions']) for r in unique)} questions): "
          f"{dict(Counter(r['meta']['use'] for r in unique))}; skipped lines {dict(skipped)}; catalog {len(catalog)} skills")


if __name__ == "__main__":
    main()
