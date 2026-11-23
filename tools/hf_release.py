# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Saved versions on Hugging Face: the text of releases/<version> lives in git, its weights and data archives there.

Every file of a version is listed with its size and SHA-256 in releases/<version>/MANIFEST.json, which is in git.
Git keeps what can be read and diffed (manifests, model cards, reports, tokenizers, calibration, logs, the
integration patch). The model files and data archives (.onnx, .pt, .gguf, .f32, .zip) are on Hugging Face at the
same paths, so a download puts them back where MANIFEST.json expects them, and is checked against it.

    python tools/hf_release.py fetch v0.5                   # the weights and archives of v0.5, checked
    python tools/hf_release.py fetch v0.5 --only model      # only the nano (model/); mini/ and data/ likewise
    python tools/hf_release.py verify v0.5                  # every file of v0.5 is here and matches
    python tools/hf_release.py verify --all --present       # what is here matches (CI checks the text in git)

Maintainers, with a write token (hf auth login, or HF_TOKEN):

    python tools/hf_release.py publish v0.5                 # upload releases/v0.5 whole, then tag it v0.5
    python tools/hf_release.py publish-card                 # hf/README.md as the repository's model card
    python tools/hf_release.py publish-dist v0.5 runs/mini-dist-v05   # Atelier's mini download (tools/mini_dist.py)

The repository is 3SENN/atelier-decisions unless --repo or ATELIER_HF_REPO names another. publish creates it
private when it doesn't exist yet; make it public on Hugging Face when the version should be. Uploads need
huggingface_hub (requirements.txt); verify needs nothing beyond Python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASES = ROOT / "releases"
DEFAULT_REPO = "3SENN/atelier-decisions"
# What lives on Hugging Face instead of git (.gitignore keeps them out); everything else in a version is in both.
BINARY = (".onnx", ".pt", ".gguf", ".f32", ".zip")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def manifest(version: str) -> dict[str, dict]:
    path = RELEASES / version / "MANIFEST.json"
    if not path.exists():
        sys.exit(f"no {path.relative_to(ROOT)}: is {version} a saved version? (have: {', '.join(versions())})")
    return json.loads(path.read_text(encoding="utf-8"))["files"]


def versions() -> list[str]:
    return sorted(p.parent.name for p in RELEASES.glob("*/MANIFEST.json"))


def verify(version: str, present: bool = False, only: list[str] | None = None, quiet: bool = False) -> int:
    """Problems found: a file that differs from the manifest, or (unless present) one that is missing."""
    bad = checked = 0
    for name, meta in manifest(version).items():
        if only and not any(name.startswith(prefix.rstrip("/") + "/") for prefix in only):
            continue
        path = RELEASES / version / name
        if not path.exists():
            if not present:
                print(f"MISSING  {version}/{name}")
                bad += 1
            continue
        checked += 1
        if path.stat().st_size != meta["bytes"] or sha256(path) != meta["sha256"]:
            print(f"BAD      {version}/{name}")
            bad += 1
        elif not quiet:
            print(f"ok       {version}/{name}")
    print(f"{version}: {checked} files checked, " + ("all match" if not bad else f"{bad} problems"))
    return bad


def hub():
    try:
        import huggingface_hub
    except ImportError:
        sys.exit("needs huggingface_hub: pip install -r requirements.txt (or pip install huggingface_hub)")
    return huggingface_hub


def fetch(repo: str, version: str, only: list[str] | None, revision: str) -> int:
    files = [name for name in manifest(version) if name.endswith(BINARY)]
    if only:
        files = [name for name in files if any(name.startswith(prefix.rstrip("/") + "/") for prefix in only)]
    if not files:
        print(f"{version}: nothing to fetch" + (f" in {', '.join(only)}" if only else ""))
        return 0
    hub().snapshot_download(repo_id=repo, revision=revision, local_dir=ROOT,
                            allow_patterns=[f"releases/{version}/{name}" for name in files])
    return verify(version, only=only, quiet=True)


def publish(repo: str, version: str) -> None:
    if verify(version, quiet=True):
        sys.exit(f"{version} doesn't match its MANIFEST.json: fetch or rebuild it before publishing")
    hf = hub()
    api = hf.HfApi()
    url = api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    folder = RELEASES / version
    total = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
    print(f"uploading {folder.relative_to(ROOT)} ({total / 1e9:.2f} GB) to {url}")
    api.upload_folder(repo_id=repo, folder_path=folder, path_in_repo=f"releases/{version}",
                      commit_message=f"{version}: the saved version, as in MANIFEST.json")
    api.create_tag(repo, tag=version, tag_message=f"atelier decision model {version}", exist_ok=True)
    print(f"{version} published and tagged: https://huggingface.co/{repo}/tree/{version}/releases/{version}")


def publish_card(repo: str) -> None:
    api = hub().HfApi()
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    api.upload_file(repo_id=repo, path_or_fileobj=ROOT / "hf" / "README.md", path_in_repo="README.md",
                    commit_message="Model card")
    print(f"model card published: https://huggingface.co/{repo}")


def publish_dist(repo: str, version: str, folder: Path) -> None:
    """Atelier's optional mini download, made by tools/mini_dist.py: every file checked against its manifest."""
    dist = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    base = f"https://huggingface.co/{repo}/resolve/main/dist/{version}/"
    entries = list(dist["model"]) + [e for files in dist.get("servers", {}).values() for e in files]
    bad = [e["url"] for e in entries
           if not (folder / e["url"]).exists() or (folder / e["url"]).stat().st_size != e["size"]
           or sha256(folder / e["url"]) != e["sha256"]]
    if bad:
        sys.exit(f"{len(bad)} files don't match {folder / 'manifest.json'}: " + ", ".join(bad[:5]))
    if dist.get("base") != base:
        print(f"note: the manifest's base is {dist.get('base')!r}; hosted here, Atelier finds the files at {base}\n"
              f"      (run tools/mini_dist.py with --base {base} so miniManifest.ts says so)")
    api = hub().HfApi()
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=folder, path_in_repo=f"dist/{version}",
                      commit_message=f"{version}: atelier-mini's download for Atelier")
    print(f"{len(entries)} files published under {base}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=os.environ.get("ATELIER_HF_REPO", DEFAULT_REPO))
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch", help="download a version's weights and archives into releases/<version>")
    p.add_argument("version")
    p.add_argument("--only", nargs="+", metavar="FOLDER", help="model (the nano), mini, data")
    p.add_argument("--revision", default="main")
    p = sub.add_parser("verify", help="check files under releases/ against their MANIFEST.json")
    p.add_argument("version", nargs="?")
    p.add_argument("--all", action="store_true", help="every saved version")
    p.add_argument("--present", action="store_true", help="only the files that are here (missing ones are fine)")
    p.add_argument("--quiet", action="store_true")
    p = sub.add_parser("publish", help="upload releases/<version> whole and tag it")
    p.add_argument("version")
    sub.add_parser("publish-card", help="upload hf/README.md as the model card")
    p = sub.add_parser("publish-dist", help="upload Atelier's mini download folder (tools/mini_dist.py)")
    p.add_argument("version")
    p.add_argument("folder", type=Path)
    args = parser.parse_args()

    if args.command == "verify":
        if bool(args.all) == bool(args.version):
            parser.error("verify takes a version or --all")
        bad = sum(verify(v, present=args.present, quiet=args.quiet) for v in (versions() if args.all else [args.version]))
        sys.exit(1 if bad else 0)
    if args.command == "fetch":
        sys.exit(1 if fetch(args.repo, args.version, args.only, args.revision) else 0)
    if args.command == "publish":
        publish(args.repo, args.version)
    elif args.command == "publish-card":
        publish_card(args.repo)
    elif args.command == "publish-dist":
        publish_dist(args.repo, args.version, args.folder)


if __name__ == "__main__":
    main()
