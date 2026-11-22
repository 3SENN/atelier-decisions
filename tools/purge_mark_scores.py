"""Drops the cached mini scores of requests whose text has combining marks (Hindi, Thai, ...), so the next report
scores them again with the tokenizer the runtime uses (student/mini_tokenizer.py). Scores of every other request are
the same under both split rules, token for token, and stay cached.

    python tools/purge_mark_scores.py runs/mini-l24-num/scores-mini-l24-num-Q5_K_M-e4.decider.json
"""

from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["probe-lang.flash", "probe-open.flash", "open-heldout.flash", "gen3-heldout.flash", "probe-tabs.flash", "numeric-test.truth",
         *[f"pool-v{i}.flash" for i in range(1, 7)]]


def has_marks(value) -> bool:
    text = json.dumps(value, ensure_ascii=False)
    return any(unicodedata.category(c) in ("Mn", "Mc") for c in text)


def main() -> None:
    path = Path(sys.argv[1])
    scores = json.loads(path.read_text())
    dropped = 0
    for name in FILES:
        source = ROOT / "data" / "teacher" / f"{name}.jsonl"
        if not source.exists():
            continue
        with source.open(encoding="utf-8") as f:
            for row in map(json.loads, f):
                state = row["request"]["state"]
                for key, q in row["request"]["questions"].items():
                    if has_marks([state, q]):
                        h = hashlib.md5(json.dumps([state, key, q], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                        dropped += scores.pop(h, None) is not None
    path.write_text(json.dumps(scores))
    print(f"{path}: dropped {dropped} scores of text with combining marks, kept {len(scores)}")


if __name__ == "__main__":
    main()
