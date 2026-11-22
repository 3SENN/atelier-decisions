"""Folds atelier-mini's adapters into plain weights: a text-only Qwen3.5 checkpoint any runtime can load, plus the
readout head and answer letters next to it.

    python student/merge_mini.py --model runs/mini-l24 --out runs/mini-l24/merged

Writes <out>/model.safetensors and config.json (Qwen3.5 layout, vision tower dropped, pruned blocks renumbered),
the tokenizer files, readout.safetensors ([255 x hidden], float32) and decision.json (letters, token ids,
temperatures). The runtime runs the backbone with non-causal attention and reads the readout off the final, normed
hidden state at each <|fim_suffix|> marker, as pplx_decider does.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
FLASH = ROOT / "models" / "pplx-decider-flash-v1.3"
PREFIX = "model.language_model."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()

    saved = torch.load(args.model / "mini.pt", map_location="cpu")
    base = Path(saved["base"])
    if not base.is_absolute():
        base = ROOT / base
    trained, rank = saved["trainable"], int(saved["rank"])
    scale = 2.0 * rank / rank  # train_mini.py: alpha = 2 x rank
    kept = saved.get("kept")
    dtype = getattr(torch, args.dtype)

    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(base.glob("*.safetensors")):
        tensors.update(load_file(str(shard)))
    text = {k: v for k, v in tensors.items() if k.startswith(PREFIX)}  # vision tower dropped
    config = json.loads((base / "config.json").read_text())
    tc = config["text_config"]
    n_layers = tc["num_hidden_layers"]
    kept = kept or list(range(n_layers))
    new_index = {old: new for new, old in enumerate(kept)}

    out: dict[str, torch.Tensor] = {}
    merged = 0
    for name, w in text.items():
        m = re.match(rf"{re.escape(PREFIX)}layers\.(\d+)\.(.+)$", name)
        if not m:
            out[name] = w.to(dtype) if w.is_floating_point() else w
            continue
        old, rest = int(m.group(1)), m.group(2)
        if old not in new_index:
            continue
        new = new_index[old]
        stem = rest[: -len(".weight")] if rest.endswith(".weight") else None
        a, b = (trained.get(f"text.layers.{new}.{stem}.{x}") for x in "AB") if stem else (None, None)
        if a is not None and b is not None:
            w = w.float() + scale * (b.float() @ a.float())
            merged += 1
        out[f"{PREFIX}layers.{new}.{rest}"] = w.to(dtype) if w.is_floating_point() else w
    expected = sum(1 for k in trained if k.endswith(".A"))
    assert merged == expected, f"merged {merged} of {expected} adapters"

    tc["num_hidden_layers"] = len(kept)
    tc["layer_types"] = [tc["layer_types"][i] for i in kept]
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(out, str(args.out / "model.safetensors"), metadata={"format": "pt"})
    (args.out / "config.json").write_text(json.dumps(config, indent=1))
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
                 "preprocessor_config.json", "video_preprocessor_config.json", "LICENSE"):
        if (base / name).exists():
            shutil.copy2(base / name, args.out / name)
    save_file({"weight": trained["readout.weight"].float().contiguous()}, str(args.out / "readout.safetensors"))
    flash = json.loads((FLASH / "decision_config.json").read_text())
    calibration = args.model / "calibration.json"
    (args.out / "decision.json").write_text(json.dumps({
        "codes": flash["codes"], "token_ids": flash["token_ids"], "kept_blocks": kept, "base": base.name,
        "temperatures": json.loads(calibration.read_text()) if calibration.exists() else {},
        "readout": "final normed hidden state at each <|fim_suffix|>; non-causal full attention; delta-net layers causal",
    }, indent=1))
    body = sum(v.numel() for k, v in out.items() if "embed_tokens" not in k)
    print(f"{args.out}: {len(kept)} blocks, {merged} adapters merged, body {body / 1e6:.0f}M, "
          f"embeddings {out[PREFIX + 'embed_tokens.weight'].numel() / 1e6:.0f}M")


if __name__ == "__main__":
    main()
