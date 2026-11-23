# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Every source file starts with the MPL-2.0 notice, as in Atelier (its tests/licenseHeaders.test.ts).

    python tools/check_headers.py          # lists the files without it; exits 1 if there are any
    python tools/check_headers.py --fix    # adds it to them

Covers the code in git (and new files git doesn't ignore): Python, TypeScript and JavaScript, PowerShell, YAML.
Not third_party/ (its own licence) and not releases/ (saved versions are hashed in their MANIFEST.json).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTICE = [
    "Copyright (c) 2026 The Atelier Authors. All rights reserved.",
    "This Source Code Form is subject to the terms of the Mozilla Public",
    "License, v. 2.0. If a copy of the MPL was not distributed with this file,",
    "You can obtain one at https://mozilla.org/MPL/2.0/.",
]
HASH = {".py", ".ps1", ".yml", ".yaml"}
BLOCK = {".ts", ".mjs", ".js"}
SKIP = ("third_party/", "releases/")


def header(suffix: str) -> str:
    if suffix in HASH:
        return "".join(f"# {line}\n" for line in NOTICE) + "\n"
    return "/* " + "\n * ".join(NOTICE) + " */\n\n"


def has_notice(text: str) -> bool:
    head = "\n".join(text.splitlines()[:8])
    return all(part in head for part in ("Copyright (c)", "Mozilla Public", "https://mozilla.org/MPL/2.0/"))


def files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.splitlines()
    return [ROOT / f for f in out if Path(f).suffix in HASH | BLOCK and not f.startswith(SKIP) and (ROOT / f).exists()]


def main() -> None:
    fix = "--fix" in sys.argv[1:]
    sources = files()
    missing = []
    for path in sources:
        text = path.read_text(encoding="utf-8")
        if has_notice(text):
            continue
        missing.append(path.relative_to(ROOT).as_posix())
        if fix:
            shebang = ""
            if text.startswith("#!"):
                shebang, _, text = text.partition("\n")
                shebang += "\n"
            path.write_text(shebang + header(path.suffix) + text, encoding="utf-8", newline="")
    for name in missing:
        print(("added the notice to " if fix else "no MPL-2.0 notice: ") + name)
    if missing and not fix:
        sys.exit(f"{len(missing)} files without the notice: python tools/check_headers.py --fix adds it")
    print(f"{len(sources)} source files, all with the MPL-2.0 notice")


if __name__ == "__main__":
    main()
