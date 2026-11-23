# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Marks an atelier-mini GGUF as a decider: non-causal attention (full-attention layers see the whole sequence; the
delta-net layers stay causal by construction) and last-token pooling (the final, normed hidden state at the
<|fim_suffix|> marker that ends each question). Any llama.cpp runtime then computes pplx_decider's forward pass
without flags; the readout (readout.safetensors) is one small matrix product in the caller.

    python student/gguf_decider.py runs/mini-l24/mini-l24-Q8_0.gguf runs/mini-l24/mini-l24-Q8_0.decider.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models" / "llama.cpp-b12620" / "src" / "gguf-py"))
import gguf  # noqa: E402
from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata, get_field_data  # noqa: E402

POOLING_LAST = 3  # llama.h: LLAMA_POOLING_TYPE_LAST


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    reader = gguf.GGUFReader(str(args.input), "r")
    arch = get_field_data(reader, gguf.Keys.General.ARCHITECTURE)
    new = {
        gguf.Keys.Attention.CAUSAL.format(arch=arch): MetadataDetails(gguf.GGUFValueType.BOOL, False),
        gguf.Keys.LLM.POOLING_TYPE.format(arch=arch): MetadataDetails(gguf.GGUFValueType.UINT32, POOLING_LAST),
    }
    writer = gguf.GGUFWriter(str(args.output), arch=arch, endianess=reader.endianess)
    alignment = get_field_data(reader, gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment
    copy_with_new_metadata(reader, writer, new, [])
    print(f"{args.output}: {arch}, attention.causal = false, pooling = last")


if __name__ == "__main__":
    main()
