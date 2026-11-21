"""How much of mmBERT's 256k vocabulary Atelier's text actually uses, per language.

Counts tokens over the request pools (states, questions, options as the student sees them) and reports, for a
range of kept-vocabulary sizes, the share of token occurrences that would become <unk> per language, and what
the embedding table would weigh.

    python analysis/vocab_coverage.py data/pool/pool-v1.jsonl data/pool/pool-v2.jsonl
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student"))
from format import question_parts, state_text  # noqa: E402

tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models" / "mmBERT-small"))
texts: dict[str, list[str]] = defaultdict(list)
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            lang = row["meta"].get("lang", "?")
            texts[lang].append(state_text(row["request"]["state"]))
            for key, q in row["request"]["questions"].items():
                head, options, _ = question_parts(key, q)
                texts[lang].extend([head, *options])

per_lang: dict[str, Counter] = {}
total = Counter()
for lang, items in texts.items():
    c = Counter()
    for i in range(0, len(items), 4096):
        for ids in tokenizer(items[i : i + 4096], add_special_tokens=False)["input_ids"]:
            c.update(ids)
    per_lang[lang] = c
    total.update(c)

ranked = [t for t, _ in total.most_common()]
print(f"{len(total):,} distinct tokens used out of {len(tokenizer):,}")
print(f"{'kept':>8} {'emb MB int8':>11} {'emb MB 4bit':>11}   unk rate per language")
for k in (8_000, 16_000, 24_000, 32_000, 48_000, 64_000, 96_000, len(ranked)):
    kept = set(ranked[:k])
    rates = []
    for lang, c in sorted(per_lang.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        miss = sum(v for t, v in c.items() if t not in kept)
        rates.append(f"{lang} {100 * miss / n:.3f}%")
    print(f"{k:>8,} {k * 384 / 1e6:>11.1f} {k * 384 / 2e6:>11.1f}   " + "  ".join(rates))
