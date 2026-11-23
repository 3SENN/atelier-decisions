# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""A smaller vocabulary for atelier-mini: keep the tokens its training data uses, spell the rest with kept ones.

Qwen3.5's embedding table has 248,320 rows (143 MB at 4 bits, a quarter of the mini's GGUF), and the mini's training
data uses fewer than half of them. The runners tokenize with the full tokenizer and send token ids to llama-server, so
the table can lose every row they never send: a kept token is renumbered; a dropped one is spelled with kept tokens
(the longest kept prefix of its byte-level string, then the next), which always works because every single byte is
kept. Text the data never had (a rare Chinese character, say) still goes through, just in smaller pieces.

    python student/vocab_trim.py runs/mini-v05/mini-v05-Q5_K_M-e4.decider.gguf runs/mini-v05/mini-v05-Q5_K_M-e4v.decider.gguf \
        --tokenizer models/Qwen3.5-0.8B --data data/pool/pool-*.jsonl --runtime runs/mini-v05/runtime

writes the trimmed GGUF and <out>.vocab.json ({"keep": [old ids, ascending]}); with --runtime, also the runtime
folder's decision.json ids checked and vocab.json copied there. serve_mini.Mini and both Node runners map ids through
it when it is present.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class VocabMap:
    """Old (full-tokenizer) ids to the trimmed table's ids."""

    def __init__(self, keep: list[int], id_to_token: dict[int, str]) -> None:
        self.new = {old: i for i, old in enumerate(keep)}
        self.by_string = {id_to_token[old]: i for i, old in enumerate(keep) if old in id_to_token}  # not the GGUF's padding rows
        self.longest = max(len(s) for s in self.by_string)
        self.id_to_token = id_to_token
        self.cache: dict[int, list[int]] = {}

    def spell(self, old: int) -> list[int]:
        if old in self.new:
            return [self.new[old]]
        if old not in self.cache:
            s, out, i = self.id_to_token[old], [], 0
            while i < len(s):
                for j in range(min(len(s), i + self.longest), i, -1):
                    if s[i:j] in self.by_string:
                        out.append(self.by_string[s[i:j]])
                        i = j
                        break
                else:
                    raise ValueError(f"token {old} ({s!r}) can't be spelled with kept tokens")
            self.cache[old] = out
        return self.cache[old]

    def __call__(self, ids: list[int]) -> list[int]:
        return [n for old in ids for n in self.spell(old)]


def id_to_token(tokenizer_json: Path) -> dict[int, str]:
    spec = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    out = {i: t for t, i in spec["model"]["vocab"].items()}
    out |= {a["id"]: a["content"] for a in spec.get("added_tokens", [])}
    return out


def load_map(vocab_json: Path, tokenizer_json: Path) -> VocabMap:
    return VocabMap(json.loads(vocab_json.read_text())["keep"], id_to_token(tokenizer_json))


def byte_symbols() -> set[str]:
    """The 256 one-character symbols byte-level BPE writes bytes as (GPT-2's bytes_to_unicode)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c) for c in cs}


def strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield str(k)
            yield from strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings(v)
    elif x is not None:
        yield json.dumps(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models" / "Qwen3.5-0.8B")
    parser.add_argument("--data", type=Path, nargs="+", required=True, help="training pools whose tokens are kept (not test sets)")
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--runtime", type=Path, help="a runtime folder (student/export_mini.py) to receive vocab.json")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "student"))
    sys.path.insert(0, str(ROOT / "models" / "llama.cpp-b12620" / "src" / "gguf-py"))
    import gguf
    from gguf.scripts.gguf_new_metadata import get_field_data
    from mini_tokenizer import mini_processor

    tok = mini_processor(args.tokenizer).tokenizer
    tokens = id_to_token(args.tokenizer / "tokenizer.json")
    counts: Counter[int] = Counter()
    for path in args.data:
        with path.open(encoding="utf-8") as f:
            for line in f:
                counts.update(tok.encode("\n".join(strings(json.loads(line)["request"])), add_special_tokens=False))
    reader = gguf.GGUFReader(str(args.input), "r")
    vocab = len(reader.fields["tokenizer.ggml.tokens"].data)
    types = [int(reader.fields["tokenizer.ggml.token_type"].parts[i][0]) for i in reader.fields["tokenizer.ggml.token_type"].data]
    gguf_tokens = reader.fields["tokenizer.ggml.tokens"].contents()  # also has the padding rows past the tokenizer's vocabulary
    singles = byte_symbols()
    keep = sorted({i for i in range(vocab) if types[i] != gguf.TokenType.NORMAL}  # control, user-defined, unused
                  | {i for i, t in enumerate(gguf_tokens) if t in singles}  # every byte, so anything can be spelled
                  | {i for i, c in counts.items() if c >= args.min_count})
    # The protocol's own text (template, labels, answer letters) is in every prompt: keep it whole.
    if args.runtime:
        spec = json.loads((args.runtime / "decision.json").read_text(encoding="utf-8"))
        protocol = [spec["prefix"], spec["state_label"], spec["question_label"], spec["options_label"], spec["default_instructions"],
                    *spec["codes"], *[f"{c}: " for c in spec["codes"]]]
        keep = sorted(set(keep) | {i for text in protocol for i in tok.encode(text, add_special_tokens=False)}
                      | {spec["separator_id"], spec["marker_id"]})
    index = {old: i for i, old in enumerate(keep)}
    print(f"{len(counts)} tokens in the data; keeping {len(keep)} of {vocab} ({len(keep) / vocab:.1%})")

    arch = get_field_data(reader, gguf.Keys.General.ARCHITECTURE)
    writer = gguf.GGUFWriter(str(args.output), arch=arch, endianess=reader.endianess)
    alignment = get_field_data(reader, gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment
    kept_strings = {gguf_tokens[i] for i in keep}
    mismatched = sum(1 for i, t in tokens.items() if i < vocab and gguf_tokens[i] != t)
    assert mismatched == 0, f"{mismatched} tokens differ between the GGUF and tokenizer.json"
    for field in reader.fields.values():
        name = field.name
        if name.startswith("GGUF.") or name in (gguf.Keys.General.ARCHITECTURE, gguf.Keys.General.ALIGNMENT):
            continue
        value = field.contents()
        vtype = field.types[0]
        if name == "tokenizer.ggml.tokens":
            value = [value[i] for i in keep]
        elif name == "tokenizer.ggml.token_type":
            value = [value[i] for i in keep]
        elif name == "tokenizer.ggml.merges":
            value = [m for m in value if all(p in kept_strings for p in m.split(" ")) and m.replace(" ", "") in kept_strings]
        elif name.startswith("tokenizer.ggml.") and name.endswith("_token_id"):
            if value not in index:
                print(f"dropping {name} ({value}: not kept)")
                continue
            value = index[value]
        if vtype == gguf.GGUFValueType.ARRAY:
            writer.add_key_value(name, value, vtype, sub_type=field.types[-1])
        else:
            writer.add_key_value(name, value, vtype)
    for t in reader.tensors:
        data = t.data[keep] if t.name == "token_embd.weight" else t.data
        writer.add_tensor_info(t.name, data.shape, data.dtype, data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        writer.write_tensor_data(t.data[keep] if t.name == "token_embd.weight" else t.data)
    writer.close()
    sidecar = args.output.with_suffix(".vocab.json")
    sidecar.write_text(json.dumps({"keep": keep}))
    if args.runtime:
        (args.runtime / "vocab.json").write_text(json.dumps({"keep": keep}))
        # The parity fixture also gets the ids as sent (mapped), so the Node runners' mapping is checked too.
        vmap = VocabMap(keep, tokens)
        parity = args.runtime / "parity.jsonl"
        if parity.exists():
            with parity.open(encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            for r in rows:
                r["mapped"] = {k: vmap(v) for k, v in r["ids"].items()}
            with parity.open("w", encoding="utf-8", newline="\n") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            spelled = sum(len(vmap(v)) != len(v) for r in rows for v in r["ids"].values())
            print(f"{parity}: mapped ids added ({spelled} of {sum(len(r['ids']) for r in rows)} questions have a dropped token)")
    print(f"{args.output}: {args.output.stat().st_size / 1e6:.0f} MB (was {args.input.stat().st_size / 1e6:.0f} MB); {sidecar}")


if __name__ == "__main__":
    main()
