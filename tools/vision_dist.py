# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""atelier-vision as Atelier's optional download: the two files Atelier ships to fetch it and to use it.

Atelier downloads atelier-vision only when someone asks for it (Settings › Quick decisions), file by file, and checks
every file against the size and SHA-256 in a manifest compiled into Atelier, so whoever hosts the files can't change
what runs. The files are the version's own model/ folder at its tag on Hugging Face (tools/hf_release.py publish), so
they never change under it. This writes, into releases/<version>/download/:

    visionManifest.ts    for src/main/decisions/device/visionManifest.ts: what to download, with sizes and SHA-256
    visionFiles.ts       for src/main/decisions/device/visionFiles.ts: the files a usable model folder holds
    visionControls.ts    for src/shared/visionControls.ts: the jobs a button with only an icon can have, named in the
                         five languages, exactly as the model was trained and measured on them (vision/icons.py)

    python tools/vision_dist.py --version vision-v0.1
    python tools/vision_dist.py --version vision-v0.1 --atelier ../atelier     # and copies them, with the runtime, into a checkout

With --atelier it also writes Atelier's copy of the runtime from runtime/node/src/vision (model, format, decide,
preprocess: the same code with Atelier's file names and comments), so the two can't drift apart: change the runtime
here, check it with `npm run vision:parity`, and run this again.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

HEADER = """/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

"""
# What the runtime loads: the training checkpoint, metrics and parity fixtures stay with the model project.
SHIPPED = (".onnx", "tokenizer.json", "tokenizer_config.json", "vision.json")


def quote(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def controls() -> str:
    import icons
    from questions import ASK, SAY

    words = icons.words()
    lines = [HEADER + "// What a button with only an icon on it can be for, named in the five languages atelier-vision's questions were",
             "// trained and measured in (docs/decisions.md › Pictures). Written by the atelier-decisions project's",
             "// tools/vision_dist.py from its vision/icons.py: the names are part of what the model was measured on, so they",
             "// change there, with a new version, not here. Jobs marked `unseen` were never trained on: answers about them",
             "// are measured as a kind of button nobody showed the model.", "",
             f"export const CONTROL_LANGS = [{', '.join(quote(lang) for lang in icons.LANGS)}] as const",
             "export type ControlLang = (typeof CONTROL_LANGS)[number]", "",
             "export interface ControlJob {", "  key: string", "  names: Record<ControlLang, string>", "  unseen?: true", "}", "",
             "export const CONTROL_JOBS: readonly ControlJob[] = ["]
    for job in sorted(words):
        names = ", ".join(f"{lang}: {quote(words[job][lang][0])}" for lang in icons.LANGS)
        lines.append(f"  {{ key: {quote(job)}, names: {{ {names} }}{', unseen: true' if job in icons.HELD_OUT_JOBS else ''} }},")
    lines += ["]", "",
              "// The second question about a button the model named (is it really a button for that job?), in two of the",
              "// wordings it was trained on: how it is asked, and what a yes looks like. {name} is the job's name.",
              "export const CONTROL_ASK: Record<ControlLang, string> = {"]
    lines += [f"  {lang}: {quote(ASK['control'][lang][1])}," for lang in icons.LANGS]
    lines += ["}", "export const CONTROL_IS: Record<ControlLang, string> = {"]
    lines += [f"  {lang}: {quote(SAY['control'][lang][0])}," for lang in icons.LANGS]
    lines += ["}", ""]
    return "\n".join(lines)


def manifest(version: str, base: str) -> str:
    files = json.loads((ROOT / "releases" / version / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
    model = [(name[len("model/"):], f) for name, f in sorted(files.items()) if name.startswith("model/") and name.endswith(SHIPPED)]
    lines = [HEADER + "// atelier-vision's optional download (feature.ts): every file with its size and SHA-256, from the atelier-decisions",
             f"// project's releases/{version}/MANIFEST.json (its tools/vision_dist.py writes this file). The base is the version's",
             "// tag on Hugging Face, so these files never change under it; a new version is a new manifest here.", "",
             "import type { MiniManifest } from '../deviceMini'", "",
             "export const VISION_MANIFEST: MiniManifest = {", f"  version: {quote(version)},", f"  base: {quote(base)},", "  model: ["]
    lines += [f"    {{ path: {quote(name)}, url: {quote(name)}, size: {f['bytes']}, sha256: {quote(f['sha256'])} }}," for name, f in model]
    lines += ["  ],", "  servers: {},", "}", ""]
    return "\n".join(lines)


RUNTIME = ROOT / "runtime" / "node" / "src" / "vision"
RENAMES = (("'./format.ts'", "'./visionFormat'"), ("'./decide.ts'", "'./visionDecide'"), ("'./preprocess.ts'", "'./visionPreprocess'"))
# Atelier's file, and the comment it opens with there (the code under it is this project's file, imports renamed).
PORTS = {
    "preprocess.ts": ("visionPreprocess.ts", """
// atelier-vision's image input (docs/decisions.md › Pictures): a line-for-line port of the atelier-decisions project's
// vision/preprocess.py, the contract between training and this runtime. Any picture becomes the 256 x 256 square the
// image tower reads: transparent parts over white, the whole picture resized without keeping its proportions (a
// triangle filter whose width grows with the reduction, rows first, in double precision), every value x / 127.5 - 1,
// channels first. tests/decisionsVision.test.ts checks the numbers on the version's parity fixtures
// (ATELIER_VISION_PARITY=<model folder>)."""),
    "format.ts": ("visionFormat.ts", """
// atelier-vision's questions (docs/decisions.md › Pictures): a line-for-line port of the atelier-decisions project's
// vision/format.py. Atelier's own System One questions, asked about a picture: every answer is a short text, and the
// model says how well the picture matches each one.
//   choice  one text per option: its description, or the option's own name when it has none
//   score   one text per level (declined for now: no version has been measured on levels)
//   noul    the "true" text is what a yes looks like (without criteria, the question itself); a "false" text is matched
//           too when there is one
// A text is lowercased, cut to 63 tokens, closed with <eos> and padded to 64. tests/decisionsVision.test.ts checks the
// token ids on the version's parity fixtures."""),
    "decide.ts": ("visionDecide.ts", """
// From how well a picture matches each answer's text to a System One answer: a port of the atelier-decisions
// project's vision/decide.py. The towers give cos(picture, text) for every text of a question; z = scale * cos + bias
// is SigLIP's own log-odds that the text describes the picture.
//   choice, score   softmax(z / t) over the options
//   noul            with a "false" text: sigmoid(a * z_true - b * z_false + c); without one: sigmoid(a * z_true + c)
// t, a, b and c are the version's calibration (vision.json), fitted on pictures and kinds of questions it never
// trained on, so "sure" means what the readers' cutoffs take it to mean."""),
    "model.ts": ("vision.ts", """
// atelier-vision on ONNX Runtime (CPU), in the decision helper (docs/decisions.md › Pictures). A picture becomes a vector
// once; every question about it after that is a few short texts through the text tower (kept, since the same answers
// are asked of many pictures) and a dot product. It answers Atelier's System One questions about a picture by how well
// the picture matches each answer's text, and writes nothing. The same vectors find a picture by a description. The
// model folder is an optional download (feature.ts) from the atelier-decisions project; this file is its
// runtime/node/src/vision/model.ts, and tests/decisionsVision.test.ts checks it against the version's parity fixtures."""),
}


def port_runtime(atelier: Path) -> None:
    """Atelier's copies of the runtime, in src/main/decisions/device."""
    device = atelier / "src" / "main" / "decisions" / "device"
    for source, (name, head) in PORTS.items():
        text = (RUNTIME / source).read_text(encoding="utf-8")
        notice, _, rest = text.partition(" */\n\n")
        lines = rest.split("\n")
        while lines and lines[0].startswith("//"):
            lines.pop(0)
        body = "\n".join(lines).lstrip("\n")
        for old, new in RENAMES:
            body = body.replace(old, new)
        if source == "model.ts":
            def swap(old: str, new: str) -> None:
                nonlocal body
                if body.count(old) != 1:
                    raise SystemExit(f"runtime/node/src/vision/model.ts no longer has: {old}")
                body = body.replace(old, new)

            swap("import { preprocess, type Picture } from './visionPreprocess'\n",
                 "import { preprocess, type Picture } from './visionPreprocess'\n\nexport { VISION_FILES } from './visionFiles'\nexport type { Answer as VisionAnswer } from './visionDecide'\n"
                 "export type { Question as VisionQuestion } from './visionFormat'\nexport type { Picture } from './visionPreprocess'\n")
            swap("interface VisionInfo {\n  size: number", "interface VisionInfo {\n  format?: number\n  size: number")
            swap("    const info = json('vision.json') as VisionInfo\n",
                 "    const info = json('vision.json') as VisionInfo\n    if (info.format !== 1) throw new Error(`vision model format ${String(info.format)} is not one this version of Atelier reads`)\n")
            swap("const json = (name: string): object => JSON.parse(readFileSync(join(dir, name), 'utf8'))", "const json = (name: string): object => JSON.parse(readFileSync(join(dir, name), 'utf8')) as object")
            swap("// Latin letters by code point, so Python and JavaScript agree on every character (vision/runtime.py LATIN).",
                 "// Latin letters by code point, so this and the training code agree on every character (the project's vision/runtime.py).")
        (device / name).write_text(notice + " */\n" + head + "\n" + ("" if source == "model.ts" else "\n") + body, encoding="utf-8", newline="\n")
        print(f"wrote {device / name}")


def files_list(version: str) -> str:
    files = json.loads((ROOT / "releases" / version / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
    names = sorted(name[len("model/"):] for name in files if name.startswith("model/") and name.endswith(SHIPPED))
    return (HEADER + "// atelier-vision's files, apart from its runtime (vision.ts), so the main process can check a folder without loading\n"
            "// ONNX Runtime: only the decision helper does that.\n\n/** The files a usable vision model folder holds. */\n"
            f"export const VISION_FILES = [{', '.join(quote(n) for n in names)}] as const\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--atelier", type=Path, help="an Atelier checkout to copy the manifest, the job table and the runtime into")
    parser.add_argument("--base", help="where the files are hosted (default: the version's tag on Hugging Face)")
    parser.add_argument("--controls-only", type=Path, help="write only visionControls.ts, into this folder (before a version is saved)")
    parser.add_argument("--runtime-only", action="store_true", help="with --atelier: only its copy of the runtime (before a version is saved)")
    args = parser.parse_args()
    if args.runtime_only:
        if not args.atelier:
            raise SystemExit("--runtime-only needs --atelier <checkout>")
        port_runtime(args.atelier)
        return
    if args.controls_only:
        (args.controls_only / "visionControls.ts").write_text(controls(), encoding="utf-8", newline="\n")
        print(f"wrote {args.controls_only / 'visionControls.ts'}")
        return
    out = ROOT / "releases" / args.version / "download"
    out.mkdir(parents=True, exist_ok=True)
    base = args.base or f"https://huggingface.co/3SENN/atelier-decisions/resolve/{args.version}/releases/{args.version}/model/"
    (out / "visionManifest.ts").write_text(manifest(args.version, base), encoding="utf-8", newline="\n")
    (out / "visionControls.ts").write_text(controls(), encoding="utf-8", newline="\n")
    (out / "visionFiles.ts").write_text(files_list(args.version), encoding="utf-8", newline="\n")
    print(f"wrote {out / 'visionManifest.ts'}, {out / 'visionFiles.ts'} and {out / 'visionControls.ts'}")
    if args.atelier:
        device = args.atelier / "src" / "main" / "decisions" / "device"
        (device / "visionManifest.ts").write_text(manifest(args.version, base), encoding="utf-8", newline="\n")
        (device / "visionFiles.ts").write_text(files_list(args.version), encoding="utf-8", newline="\n")
        (args.atelier / "src" / "shared" / "visionControls.ts").write_text(controls(), encoding="utf-8", newline="\n")
        port_runtime(args.atelier)


if __name__ == "__main__":
    main()
