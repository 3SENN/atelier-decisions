"""Builds a smaller BPE tokenizer from mmBERT's: only the tokens Atelier's text uses, closed under their merges.

Nothing becomes <unk>: a dropped token's text is spelled with the kept pieces it was merged from, down to single
characters and, with byte fallback, raw bytes. Sequences get a little longer instead (measured per language).
Writes tokenizer.json / tokenizer_config.json and keep.json (original token ids in their new order, which is
also the row order of the trimmed embedding table).

    python student/trim_tokenizer.py --size 32000 --corpus data/pool/pool-v1.jsonl data/pool/pool-v2.jsonl \
        --out runs/tokenizer-32k
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student"))
from format import question_parts, state_text  # noqa: E402


def corpus_texts(paths: list[Path]) -> dict[str, list[str]]:
    texts: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                lang = row["meta"].get("lang", "?")
                texts[lang].append(state_text(row["request"]["state"]))
                for key, q in row["request"]["questions"].items():
                    head, options, _ = question_parts(key, q)
                    texts[lang].extend([head, *options])
    return texts


def trim(base: Path, texts: dict[str, list[str]], size: int, out: Path) -> dict:
    spec = json.loads((base / "tokenizer.json").read_text(encoding="utf-8"))
    full = Tokenizer.from_file(str(base / "tokenizer.json"))
    vocab: dict[str, int] = spec["model"]["vocab"]
    inverse = {i: t for t, i in vocab.items()}
    merges = [m if isinstance(m, str) else " ".join(m) for m in spec["model"]["merges"]]

    # Frequencies, balanced so each language gets an equal say (English would otherwise crowd out the rest).
    counts: Counter = Counter()
    for lang, items in texts.items():
        c: Counter = Counter()
        for enc in full.encode_batch(items, add_special_tokens=False):
            c.update(enc.ids)
        total = sum(c.values())
        for i, n in c.items():
            counts[i] += n / total
    added = {t["id"] for t in spec["added_tokens"]}
    keep = set(added)
    keep |= {i for t, i in vocab.items() if t.startswith("<0x") and t.endswith(">") and len(t) == 6}  # bytes
    for i, _ in counts.most_common():
        if len(keep) >= size:
            break
        keep.add(i)
    # Close under merges: a kept token is only reachable if both pieces it is merged from are.
    parents: dict[str, tuple[str, str]] = {}
    for m in merges:
        a, b = m.split(" ", 1)
        parents.setdefault(a + b, (a, b))
    stack = [inverse[i] for i in keep]
    kept_tokens = set(stack)
    while stack:
        token = stack.pop()
        for piece in parents.get(token, ()):
            if piece not in kept_tokens and piece in vocab:
                kept_tokens.add(piece)
                stack.append(piece)
    keep_ids = sorted(vocab[t] for t in kept_tokens if t in vocab)
    remap = {old: new for new, old in enumerate(keep_ids)}

    new = copy.deepcopy(spec)
    new["model"]["vocab"] = {inverse[old]: new_id for old, new_id in remap.items()}
    new["model"]["merges"] = [m for m in merges
                              if (lambda a, b: a in kept_tokens and b in kept_tokens and a + b in kept_tokens)(*m.split(" ", 1))]
    for t in new["added_tokens"]:
        t["id"] = remap[t["id"]]
    post = new.get("post_processor") or {}
    for special in (post.get("special_tokens") or {}).values():
        special["ids"] = [remap[i] for i in special["ids"]]
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokenizer.json").write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")
    (out / "tokenizer_config.json").write_bytes((base / "tokenizer_config.json").read_bytes())
    for name in ("special_tokens_map.json",):
        if (base / name).exists():
            (out / name).write_bytes((base / name).read_bytes())
    (out / "keep.json").write_text(json.dumps(keep_ids))

    # Check: same text back, and how much longer sequences get, per language.
    small = Tokenizer.from_file(str(out / "tokenizer.json"))
    report = {"size_requested": size, "vocab": len(keep_ids), "merges": len(new["model"]["merges"]),
              "tokenizer_mb": round((out / "tokenizer.json").stat().st_size / 1e6, 2), "languages": {}}
    for lang, items in texts.items():
        sample = items[:5000]
        a = full.encode_batch(sample, add_special_tokens=False)
        b = small.encode_batch(sample, add_special_tokens=False)
        same = sum(full.decode(x.ids) == full.decode([keep_ids[i] for i in y.ids]) for x, y in zip(a, b, strict=True))
        report["languages"][lang] = {
            "length_ratio": round(sum(len(y.ids) for y in b) / max(1, sum(len(x.ids) for x in a)), 3),
            "roundtrip": round(same / len(sample), 4),
        }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "models" / "mmBERT-small")
    parser.add_argument("--corpus", type=Path, nargs="+", required=True)
    parser.add_argument("--size", type=int, nargs="+", default=[32000])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    texts = corpus_texts(args.corpus)
    for size in args.size:
        out = args.out if len(args.size) == 1 else args.out.with_name(f"{args.out.name}-{size // 1000}k")
        print(json.dumps(trim(args.base, texts, size, out)), flush=True)


if __name__ == "__main__":
    main()
