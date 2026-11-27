# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""A smaller vocabulary for atelier-vision's text tower, the way atelier-nano got its own.

SigLIP 2's text tower has a 256,000-token embedding table: 197M of the model's 375M parameters, 98 MB even at 4
bits, and a 34 MB tokenizer file. Its tokenizer is Gemma's, the same one mmBERT uses, so atelier-nano's trimmed
vocabulary (the tokens Atelier's own text uses in English, Dutch, German, French and Spanish) carries over id for
id. This module adds the tokens of:

- everyday sentences in each language (Tatoeba's, https://tatoeba.org, CC BY 2.0 FR: up to 400,000 a language; a
  token counts when two of a language's sentences use it). People name anything in a photo, so a vocabulary chosen
  from captions alone splits "hens" and "brass". Only which tokens the sentences use is taken from them;
- COCO-35L's dev captions in those languages (vision/photos.py coco35);
- everything this project's own questions can say.

It closes the set under the tokenizer's merges (student/trim_tokenizer.py's method: a dropped token's text is spelled
with the kept pieces it was merged from, so nothing becomes <unk>, sequences only get a little longer), and writes
the trimmed tokenizer with the list of kept ids, which is also the row order of the trimmed table.

    .venv/bin/python -I vision/vocab.py build --out runs/vision/vocab-12 --langs en nl de fr es it pt sv da pl id tr
    .venv/bin/python -I vision/vocab.py compare --vocab runs/vision/vocab-5 runs/vision/vocab-12     # search with each

What a trimmed vocabulary costs is measured on text it was not chosen from: Crossmodal-3600's captions.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from format import clean  # noqa: E402

CACHE = ROOT / "data" / "cache" / "vision"
BASE = ROOT / "models" / "siglip2-base-patch16-256"
NANO_KEEP = ROOT / "releases" / "v0.6" / "model" / "tokenizer-keep.json"
MAIN = ["en", "nl", "de", "fr", "es"]


def own_texts() -> list[str]:
    """Everything this project's own questions can say: every thing's names and every template, in five languages."""
    import random

    from data import families
    from questions import ASK, CHOOSE, LANGS, OTHER, SAY
    from synth import SPACES
    from classes import ALBUMS, classes

    out = [t for table in (SAY, OTHER, ASK, CHOOSE) for per_lang in table.values() for ts in per_lang.values() for t in ts]
    fams = families()
    spaces = {"photo": list(classes()), "album": list(ALBUMS), **{name: labels for name, (_, labels, _) in SPACES.items()}}
    rng = random.Random(0)
    for name, labels in spaces.items():
        for label in labels:
            for lang in LANGS:
                w = fams[name].words(label, lang)
                out += [v for v in (w["x"], w["name"], w.get("does")) if v] + w.get("say", []) + w.get("ask", []) + w.get("other", [])
                for unseen in (False, True):
                    out += [text for text in fams[name].noul(rng, lang, label, True, unseen)["q"].get("criteria", {}).values()]
    return out


TATOEBA = "https://downloads.tatoeba.org/exports/per_language/{code}/{code}_sentences.tsv.bz2"
ISO3 = {"en": "eng", "nl": "nld", "de": "deu", "fr": "fra", "es": "spa", "it": "ita", "pt": "por", "sv": "swe", "da": "dan", "pl": "pol", "id": "ind", "tr": "tur"}


def sentences(lang: str, cap: int = 400_000) -> list[str]:
    """A language's Tatoeba sentences (fetched once, kept as plain lines)."""
    import bz2
    import urllib.request

    path = CACHE / "tatoeba" / f"{lang}.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(TATOEBA.format(code=ISO3[lang]), timeout=120) as r:
            lines = bz2.decompress(r.read()).decode("utf-8").splitlines()
        # Evenly through the file, so a language with millions of sentences is not only its oldest ones.
        step = max(1, len(lines) // cap)
        path.write_text("\n".join(line.split("\t", 2)[2] for line in lines[::step][:cap] if line.count("\t") >= 2), encoding="utf-8")
    return path.read_text(encoding="utf-8").splitlines()


def sentence_tokens(full, langs: list[str], at_least: int = 2) -> set[int]:
    """The tokens each language's sentences use, at the start of a text and after a space; ones used only once in a
    language are left out (a typo, a name)."""
    from collections import Counter

    keep: set[int] = set()
    for lang in langs:
        counts: Counter = Counter()
        texts = [clean(t) for t in sentences(lang)]
        for i in range(0, len(texts), 20000):
            for enc in full.encode_batch(texts[i : i + 20000], add_special_tokens=False):
                counts.update(set(enc.ids))
        keep |= {i for i, n in counts.items() if n >= at_least}
    return keep


def caption_texts(langs: list[str]) -> list[str]:
    rows = [json.loads(line) for line in (CACHE / "coco35-dev.jsonl").read_text(encoding="utf-8").splitlines()]
    return [r["text"] for r in rows if r["lang"] in langs]


def latin_or_plain(token: str) -> bool:
    """A token whose letters are all Latin script (or that has none: digits, punctuation, symbols)."""
    import unicodedata

    return all("LATIN" in unicodedata.name(ch, "") for ch in token if ch.isalpha())


def build(langs: list[str], out: Path, base: Path = BASE, with_nano: bool = True, rank: int = 0, captions_only: bool = False) -> dict:
    from tokenizers import Tokenizer

    spec = json.loads((base / "tokenizer.json").read_text(encoding="utf-8"))
    full = Tokenizer.from_file(str(base / "tokenizer.json"))
    full.no_padding()
    vocab: dict[str, int] = spec["model"]["vocab"]
    inverse = {i: t for t, i in vocab.items()}
    merges = [m if isinstance(m, str) else " ".join(m) for m in spec["model"]["merges"]]
    keep = {t["id"] for t in spec["added_tokens"]}
    keep |= {i for t, i in vocab.items() if t.startswith("<0x") and t.endswith(">") and len(t) == 6}  # the bytes everything can be spelled with
    if with_nano:
        keep |= set(json.loads(NANO_KEEP.read_text(encoding="utf-8")))
    if rank:
        # The tokenizer's own order instead of sentences: its merges were learned most frequent first, so a token's
        # id is roughly its rank. Every Latin-script token among the first `rank`, whatever language it is from.
        keep |= {i for t, i in vocab.items() if i < rank and latin_or_plain(t)}
    elif not captions_only:
        keep |= sentence_tokens(full, langs)
    texts = [clean(t) for t in own_texts() + caption_texts(langs)]
    for enc in full.encode_batch(texts, add_special_tokens=False):
        keep.update(enc.ids)
    # A word is kept however it stands: at the start of a text ("número") and after a space ("▁número"). An answer is
    # often one word, and sentences rarely start with it.
    for i in [] if captions_only else list(keep):
        t = inverse[i]
        other = t[1:] if t.startswith("▁") else "▁" + t
        if other in vocab:
            keep.add(vocab[other])
    # And the short tokens without letters (digits, punctuation, currency and other signs), with and without the space.
    if not captions_only:
        keep |= {i for t, i in vocab.items() if len(t.lstrip("▁")) <= 3 and not any(ch.isalpha() for ch in t)}
    # Closed under merges: a kept token is only reachable if both pieces it is merged from are.
    parents: dict[str, tuple[str, str]] = {}
    for m in merges:
        a, b = m.split(" ", 1)
        parents.setdefault(a + b, (a, b))
    stack = [inverse[i] for i in keep]
    kept = set(stack)
    while stack:
        for piece in parents.get(stack.pop(), ()):
            if piece not in kept and piece in vocab:
                kept.add(piece)
                stack.append(piece)
    keep_ids = sorted(vocab[t] for t in kept)
    remap = {old: new for new, old in enumerate(keep_ids)}
    new = copy.deepcopy(spec)
    new["model"]["vocab"] = {inverse[old]: i for old, i in remap.items()}
    new["model"]["merges"] = [m for m in merges if (lambda a, b: a in kept and b in kept and a + b in kept)(*m.split(" ", 1))]
    new["padding"] = None  # vision/format.py pads
    for t in new["added_tokens"]:
        t["id"] = remap[t["id"]]
    for special in ((new.get("post_processor") or {}).get("special_tokens") or {}).values():
        special["ids"] = [remap[i] for i in special["ids"]]
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokenizer.json").write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")
    (out / "tokenizer_config.json").write_bytes((base / "tokenizer_config.json").read_bytes())
    (out / "vocab.json").write_text(json.dumps({"keep": keep_ids, "unk": remap[vocab["<unk>"]], "pad": remap[vocab["<pad>"]], "eos": remap[vocab["<eos>"]]}), encoding="utf-8")
    report = {"languages": langs, "with_nano": with_nano, "first_ids": rank, "captions_only": captions_only, "tokens": len(keep_ids), "merges": len(new["model"]["merges"]), "texts": len(texts),
              "tokenizer_mb": round((out / "tokenizer.json").stat().st_size / 1e6, 2), "table_mb_at_4_bits": round(len(keep_ids) * 768 / 2 / 1e6, 1)}
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


class Trimmed:
    """A trimmed tokenizer that answers in the full table's ids, to try a vocabulary on the untrimmed towers."""

    def __init__(self, folder: Path) -> None:
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.keep = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))["keep"]

    def encode(self, text: str, add_special_tokens: bool = False):
        ids = [self.keep[i] for i in self.tok.encode(text, add_special_tokens=False).ids]
        return type("Encoding", (), {"ids": ids})()


def compare(folders: list[Path], langs: list[str], emb: Path) -> None:
    import numpy as np

    from embed import load
    from model import Towers, device, embed_texts, tokenizer
    from search import captions, recall

    dev = device()
    model = Towers.load().to(dev)
    rows = captions()
    all_ids, vectors = load(emb, "xm3600")
    at = {k: i for i, k in enumerate(all_ids)}
    photos = vectors[[at[r["image"]] for r in rows]]
    photos /= np.linalg.norm(photos, axis=1, keepdims=True)
    toks = {"full (256,000)": tokenizer(), **{f"{f.name} ({len(Trimmed(f).keep):,})": Trimmed(f) for f in folders}}
    result: dict[str, dict] = {}
    for lang in langs:
        texts, truth = [], []
        for i, r in enumerate(rows):
            for c in r["captions"].get(lang, []):
                texts.append(c)
                truth.append(i)
        for name, tok in toks.items():
            count = lambda k: sum(len(k.encode(clean(t), add_special_tokens=False).ids) for t in texts[:2000])
            longer = count(tok) / count(toks["full (256,000)"])
            r = recall(embed_texts(model, tok, texts, dev), np.array(truth), photos)
            result.setdefault(name, {})[lang] = {**r, "tokens_per_text": round(longer, 3)}
            print(f"  {lang} {name:28s} {100 * r['r1']:5.1f}% / {100 * r['r5']:5.1f}% / {100 * r['r10']:5.1f}%  texts {longer:.2f}x as long", flush=True)
    out = ROOT / "analysis" / "results" / "vision-vocab.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    lines = ["search on Crossmodal-3600 (3,600 photos) with SigLIP 2's own towers and each vocabulary: right photo first / in the first 5 / in the first 10; how much longer texts get"]
    for name, per in result.items():
        lines.append(f"{name}")
        lines += [f"  {lang}: {100 * r['r1']:5.1f}% / {100 * r['r5']:5.1f}% / {100 * r['r10']:5.1f}%   {r['tokens_per_text']:.2f}x" for lang, r in per.items()]
    out.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--langs", nargs="+", default=MAIN)
    b.add_argument("--without-nano", action="store_true")
    b.add_argument("--captions-only", action="store_true", help="only the nano's tokens, the captions and this project's own texts (the first try; it splits everyday words)")
    b.add_argument("--rank", type=int, default=0, help="instead of sentences: every Latin-script token among the tokenizer's first N ids (tried; it keeps the wrong ones)")
    c = sub.add_parser("compare")
    c.add_argument("--vocab", type=Path, nargs="+", required=True)
    c.add_argument("--langs", nargs="+", default=MAIN + ["it", "pt", "sv", "da", "pl", "id", "tr", "ru", "ja"])
    c.add_argument("--emb", type=Path, default=ROOT / "runs" / "vision" / "emb" / "base")
    args = parser.parse_args()
    if args.cmd == "build":
        print(json.dumps(build(args.langs, args.out, with_nano=not args.without_nano, rank=args.rank, captions_only=args.captions_only), indent=1))
    else:
        compare(args.vocab, args.langs, args.emb)


if __name__ == "__main__":
    main()
