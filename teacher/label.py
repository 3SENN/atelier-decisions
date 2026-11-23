# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Labels a request pool with a decider that fits on the GPU (flash): raw readout logits per question.

Writes one JSONL row per request as it goes and resumes where it stopped, so a multi-hour run can be interrupted.

    python teacher/label.py --checkpoint models/pplx-decider-flash-v1.3 --pool data/pool/pool-v1.jsonl \
        --out data/teacher/pool-v1.flash.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "teacher"))
from server import load  # noqa: E402  (text-only reference decider)


def read_jsonl(path: Path) -> list[dict]:
    """One JSON value per line. Iterating the file splits on newlines only; str.splitlines() would also split
    on U+2028 and similar characters that JSON keeps unescaped inside strings."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=256, help="requests per decide() call")
    parser.add_argument("--batch-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = read_jsonl(args.pool)
    rows = rows[: args.limit] if args.limit else rows
    done = 0
    if args.out.exists():
        done = len(read_jsonl(args.out))
    print(f"{len(rows)} requests, {done} already labelled", flush=True)
    decider = load(args.checkpoint, "cuda", text_only=True)
    decider.batch_tokens = args.batch_tokens
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0, seen_tokens = time.perf_counter(), 0
    with args.out.open("a", encoding="utf-8") as f:
        for start in range(done, len(rows), args.chunk):
            part = rows[start : start + args.chunk]
            logits = decider.logits([row["request"] for row in part])
            for row, values in zip(part, logits, strict=True):
                f.write(json.dumps({"meta": row["meta"], "request": row["request"], "logits": values,
                                    "temperature": decider.temperature, "teacher": args.checkpoint.name},
                                   ensure_ascii=False) + "\n")
            f.flush()
            seen_tokens += sum(c.length for row in part for c in decider.encoder.plan(row["request"], None, "separate"))
            finished = start + len(part)
            rate = (finished - done) / (time.perf_counter() - t0)
            print(f"{finished}/{len(rows)} requests, {rate:.1f} req/s, {seen_tokens / (time.perf_counter() - t0):,.0f} tok/s, "
                  f"~{(len(rows) - finished) / max(rate, 1e-9) / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    main()
