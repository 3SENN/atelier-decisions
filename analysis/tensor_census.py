"""Parameter census of a decider checkpoint, from safetensors headers only (nothing is loaded).

Groups every tensor by what it does and says which groups an Atelier text-only decision needs:

    python analysis/tensor_census.py models/pplx-decider-v1.3 [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from collections import defaultdict
from pathlib import Path

BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1, "F8_E4M3": 1}


def headers(root: Path) -> dict[str, tuple[str, list[int]]]:
    """name -> (dtype, shape) for every tensor in every *.safetensors file under root (not locate-span/)."""
    out: dict[str, tuple[str, list[int]]] = {}
    for path in sorted(root.glob("*.safetensors")):
        with path.open("rb") as f:
            (size,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(size))
        prefix = "readout." if path.name == "readout.safetensors" else ""
        for name, meta in header.items():
            if name != "__metadata__":
                out[prefix + name] = (meta["dtype"], meta["shape"])
    return out


def group(name: str) -> tuple[str, str]:
    """(group, role) of a tensor name."""
    if name.startswith("readout."):
        return "readout head [255 codes x hidden]", "needed"
    if name.startswith("visual."):
        return "vision tower (27 ViT blocks + merger)", "unused for text-only decisions"
    if "embed_tokens" in name:
        return "token embeddings (248k vocab)", "needed only for tokens Atelier text uses"
    if name.endswith("language_model.norm.weight"):
        return "final norm", "needed"
    m = re.search(r"layers\.(\d+)\.(linear_attn|self_attn|mlp|input_layernorm|post_attention_layernorm)", name)
    if m:
        part = {"linear_attn": "Gated DeltaNet mixer", "self_attn": "full-attention mixer (non-causal)",
                "mlp": "SwiGLU MLP", "input_layernorm": "norms", "post_attention_layernorm": "norms"}[m.group(2)]
        return f"decoder: {part}", "needed (compute)"
    return "other", "?"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    tensors = headers(args.checkpoint)
    config = json.loads((args.checkpoint / "config.json").read_text())["text_config"]
    groups: dict[str, dict] = defaultdict(lambda: {"params": 0, "bytes": 0, "tensors": 0, "role": ""})
    per_layer: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for name, (dtype, shape) in tensors.items():
        n = 1
        for d in shape:
            n *= d
        g, role = group(name)
        groups[g]["params"] += n
        groups[g]["bytes"] += n * BYTES.get(dtype, 2)
        groups[g]["tensors"] += 1
        groups[g]["role"] = role
        m = re.search(r"layers\.(\d+)\.(\w+)", name)
        if m:
            per_layer[int(m.group(1))][m.group(2)] += n

    total = sum(g["params"] for g in groups.values())
    print(f"# {args.checkpoint.name}: {total / 1e9:.3f}B parameters in {len(tensors)} tensors\n")
    print("| Group | Params | Share | BF16 size | Atelier text decisions |")
    print("|---|---:|---:|---:|---|")
    for g, v in sorted(groups.items(), key=lambda kv: -kv[1]["params"]):
        print(f"| {g} | {v['params'] / 1e6:,.1f} M | {100 * v['params'] / total:.1f}% | "
              f"{v['bytes'] / 2**30:.2f} GiB | {v['role']} |")

    layer_types = config["layer_types"]
    lin = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
    full = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    one = lambda i: sum(per_layer[i].values())  # noqa: E731
    print(f"\nLayers: {len(layer_types)} ({len(lin)} Gated DeltaNet, {len(full)} full attention, every "
          f"{config['full_attention_interval']}th). hidden={config['hidden_size']}, mlp={config['intermediate_size']}")
    if lin:
        print(f"- one Gated DeltaNet layer: {one(lin[0]) / 1e6:,.1f} M params "
              f"(mixer {per_layer[lin[0]]['linear_attn'] / 1e6:,.1f} M, MLP {per_layer[lin[0]]['mlp'] / 1e6:,.1f} M)")
    if full:
        print(f"- one full-attention layer: {one(full[0]) / 1e6:,.1f} M params "
              f"(mixer {per_layer[full[0]]['self_attn'] / 1e6:,.1f} M, MLP {per_layer[full[0]]['mlp'] / 1e6:,.1f} M)")

    needed = sum(v["params"] for g, v in groups.items() if not g.startswith(("vision", "token")))
    print(f"\nCompute path for a text decision (decoder + norm + readout): {needed / 1e9:.3f}B params "
          f"= {needed * 2 / 2**30:.1f} GiB BF16, {needed / 2 / 2**30:.1f} GiB at 4 bits")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"total": total, "groups": groups,
                                         "per_layer": {k: dict(v) for k, v in per_layer.items()}}, indent=1))


if __name__ == "__main__":
    main()
