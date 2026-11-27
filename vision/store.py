# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Many small pictures in two files, so a set of 20,000 is one thing to hash, zip and move.

    <name>.bin      every picture's file bytes (JPEG or PNG), one after another
    <name>.jsonl    one line per picture: {"id", "at", "bytes", ...whatever the builder knows about it}

A picture is stored as the builder made it. Photos and whole-page pictures are stored already at the tower's 256 x 256
(vision/preprocess.py's resize, then JPEG at quality 90), because the originals would not fit on the disk; small
pictures such as a button's crop are stored at their own size as PNG and resized when read.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np


class Pack:
    def __init__(self, base: Path) -> None:
        self.bin, self.index = Path(str(base) + ".bin"), Path(str(base) + ".jsonl")
        self.rows: dict[str, dict] = {}
        if self.index.exists():
            size = self.bin.stat().st_size if self.bin.exists() else 0
            for line in self.index.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["at"] + row["bytes"] <= size:  # a run that stopped mid-write leaves a last line without its bytes
                    self.rows[row["id"]] = row
        self._read = None

    def __len__(self) -> int:
        return len(self.rows)

    def __contains__(self, key: str) -> bool:
        return key in self.rows

    def add(self, key: str, data: bytes, **meta) -> None:
        self.bin.parent.mkdir(parents=True, exist_ok=True)
        with self.bin.open("ab") as f:
            at = f.seek(0, 2)
            f.write(data)
        row = {"id": key, "at": at, "bytes": len(data), **meta}
        with self.index.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.rows[key] = row

    def raw(self, key: str) -> bytes:
        if self._read is None:
            self._read = self.bin.open("rb")
        row = self.rows[key]
        self._read.seek(row["at"])
        return self._read.read(row["bytes"])

    def pixels(self, key: str) -> np.ndarray:
        """The stored picture as uint8 [H, W, 3 or 4]."""
        from PIL import Image

        with Image.open(io.BytesIO(self.raw(key))) as im:
            return np.asarray(im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB"))


def jpeg(square: np.ndarray, quality: int = 90) -> bytes:
    """float64 or uint8 [H, W, 3] to JPEG bytes."""
    from PIL import Image

    out = io.BytesIO()
    Image.fromarray(np.clip(np.rint(square), 0, 255).astype(np.uint8)).save(out, "JPEG", quality=quality)
    return out.getvalue()
