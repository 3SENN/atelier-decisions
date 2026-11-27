# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""What atelier-vision makes of icon buttons on live websites (vision/real.mjs collects them).

Atelier's own two questions and cutoffs (vision/buttons.py, Atelier's browserSteps.ts): which of all 63 jobs, by name,
in the page's language; then whether it really is a button for that job. Named when both have 0.90. Next to each,
the name the page's own code gives the button, where it has one. Whether a name is right is for a person to read
off the list: a page's "Settings" on three dots and the model's "more options" are the same button.

    node vision/real.mjs runs/vision/real https://www.youtube.com https://stackoverflow.com/questions ...
    .venv/bin/python -I vision/real.py runs/vision/real --runtime releases/vision-v0.1/model --name vision-real-sites-v0.1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

import numpy as np  # noqa: E402

import icons  # noqa: E402
from buttons import gate  # noqa: E402
from decide import probabilities  # noqa: E402
from questions import LANGS  # noqa: E402
from runtime import Runtime  # noqa: E402

RESULTS = ROOT / "analysis" / "results"
JOB, YES = 0.9, 0.9  # Atelier's cutoffs (PICTURE_SURE, PICTURE_YES)


def main() -> None:
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="vision/real.mjs's output: the pictures and buttons.jsonl")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    rt = Runtime(args.runtime, threads=4)
    words = icons.words()
    jobs = sorted(words)
    rows = [json.loads(line) for line in (args.folder / "buttons.jsonl").read_text(encoding="utf-8").splitlines() if line]
    lines = [f"icon buttons from live websites, asked Atelier's two questions (named at {JOB:.2f} and {YES:.2f}); {args.runtime}",
             "site, what the page's code calls it | what the model calls it, the job's weight / the yes"]
    named = 0
    for r in rows:
        r["truth"] = re.sub(r"^.*'s user avatar$", "a user's avatar", r["truth"])  # a person's name is not ours to keep
        lang = r["lang"].lower().split("-")[0]
        lang = lang if lang in LANGS else "en"
        v = rt.image(np.asarray(Image.open(args.folder / r["file"]).convert("RGB")))
        names = [words[j][lang][0] for j in jobs]
        p = probabilities("choice", [float(v @ rt.text(t)) for t in names], rt.scale, rt.bias, rt.cal)
        top = int(np.argmax(p))
        yes = probabilities("noul", [None, float(v @ rt.text(gate(jobs[top], lang, words)["criteria"]["true"]))], rt.scale, rt.bias, rt.cal)[1]
        r.update(job=jobs[top], name=names[top], weight=round(float(p[top]), 3), yes=round(float(yes), 3), named=bool(p[top] >= JOB and yes >= YES))
        named += r["named"]
        lines.append(f"  {r['site']:22s} {(r['truth'] or '(no name)')[:36]:36s} | {names[top][:18]:18s} {p[top]:.2f} / {yes:.2f}  {'named' if r['named'] else 'left alone'}")
    lines.append(f"{len(rows)} buttons from {len({r['site'] for r in rows})} sites: {named} named, {len(rows) - named} left alone")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.name}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    (RESULTS / f"{args.name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
