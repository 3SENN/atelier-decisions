# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Verify a downloaded decider checkpoint against its release-manifest.json (sha256 + size)."""
import hashlib, json, sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "release-manifest.json").read_text())
bad = 0
for name, meta in manifest["files"].items():
    path = root / name
    if not path.exists():
        print(f"MISSING  {name}"); bad += 1; continue
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    ok = h.hexdigest() == meta["sha256"] and path.stat().st_size == meta["bytes"]
    bad += not ok
    print(f"{'ok' if ok else 'BAD':8} {name}")
print("ALL OK" if not bad else f"{bad} problems")
