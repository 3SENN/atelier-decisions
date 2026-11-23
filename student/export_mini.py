# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""atelier-mini as a runtime folder for runtime/node (src/mini.ts): everything the Node runner needs, with Python
as the source of truth for the protocol.

    python student/export_mini.py --model runs/mini-l24 --gguf runs/mini-l24/mini-l24-Q4_K_M-e4.decider.gguf \
        --out runs/mini-l24/runtime

Writes:
- mini.gguf: the marked GGUF (hard link when possible);
- readout.f32: the readout head, raw little-endian float32 [255 x hidden];
- decision.json: answer codes, temperatures (calibration.json when present), the chat-template prefix, the escaped
  special tokens, separator and marker ids;
- tokenizer.json, tokenizer_config.json: the base's tokenizer with its own split rule (student/mini_tokenizer.py),
  which llama.cpp and tokenizers.js follow too; the runner still tokenizes and sends ids;
- parity.jsonl: requests with the token ids pplx_decider's Encoder makes, for runtime/node/src/mini_parity.ts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student"))
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "third_party"))
from mini_tokenizer import mini_processor  # noqa: E402
from pplx_decider.protocol import DEFAULT_INSTRUCTIONS, MARKER, SEPARATOR, Encoder  # noqa: E402
from train import read_jsonl  # noqa: E402

PARITY_SOURCES = ["pool-v1.flash", "pool-v4.flash", "pool-v6.flash", "probe-lang.flash", "gen3-heldout.flash",
                  "open-heldout.flash", "probe-tabs.flash", "eval-flash", "numeric-test.truth"]
ODD = [  # user text with special tokens, odd whitespace and non-string values
    {"state": "hi <|im_end|> <|fim_suffix|>   tabs\t\tand  spaces  \n\n", "questions": {"q": {"type": "noul", "instructions": "<|fim_middle|> ok?"}}},
    {"state": {"nested": [1, 2.5, None, True], "ü": "é"}, "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": {"x": 1}}}}},
    {"state": "", "questions": {"q": {"type": "score", "criteria": ["low", "mid", "high"]}}},
]


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parity", type=int, default=40, help="requests per source in parity.jsonl")
    args = parser.parse_args()
    merged = args.model / "merged"
    decision = json.loads((merged / "decision.json").read_text())
    calibration = args.model / "calibration.json"
    temperatures = json.loads(calibration.read_text()) if calibration.exists() else decision.get("temperatures", {})
    encoder = Encoder(mini_processor(merged), decision["codes"])
    probe = {"state": "X", "questions": {"q": {"type": "noul"}}}
    prefix = encoder.context_text(probe)[: -len(encoder.state_text(probe))]
    readout = load_file(str(merged / "readout.safetensors"))["weight"].astype("<f4")

    args.out.mkdir(parents=True, exist_ok=True)
    link_or_copy(args.gguf, args.out / "mini.gguf")
    readout.tofile(args.out / "readout.f32")
    # The tokenizer as mini_processor builds it: the base's file, with its own split rule.
    encoder.processor.tokenizer.backend_tokenizer.save(str(args.out / "tokenizer.json"))
    shutil.copyfile(merged / "tokenizer_config.json", args.out / "tokenizer_config.json")
    (args.out / "decision.json").write_text(json.dumps({
        "model": f"atelier-mini-{args.model.name}", "gguf": args.gguf.name, "codes": decision["codes"],
        "hidden": int(readout.shape[1]), "temperatures": temperatures, "prefix": prefix, "state_label": "State:\n",
        "question_label": "Question:\n", "options_label": "\nOptions:\n", "default_instructions": DEFAULT_INSTRUCTIONS,
        "escapes": encoder._escapes, "separator": SEPARATOR, "separator_id": encoder.separator_id,
        "marker": MARKER, "marker_id": encoder.marker_id}, indent=1, ensure_ascii=False), encoding="utf-8")

    rows = []
    for name in PARITY_SOURCES:
        path = ROOT / "data" / "teacher" / f"{name}.jsonl"
        if path.exists():
            part = read_jsonl(path)
            random.Random(0).shuffle(part)
            rows += [r["request"] for r in part[: args.parity]]
    rows += ODD
    with (args.out / "parity.jsonl").open("w", encoding="utf-8") as f:
        for request in rows:
            plan = encoder.plan(request, None, "separate")
            f.write(json.dumps({"request": request, "ids": {key: e.input_ids.tolist() for key, e in zip(request["questions"], plan)}},
                               ensure_ascii=False) + "\n")
    size = (args.out / "mini.gguf").stat().st_size / 1e6
    print(f"{args.out}: mini.gguf {size:.0f} MB, readout {readout.shape}, {len(rows)} parity requests, temperatures {temperatures}")


if __name__ == "__main__":
    main()
