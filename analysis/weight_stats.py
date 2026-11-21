"""Per-tensor statistics of a decider checkpoint, on the CPU, one tensor at a time: how hard each part is to
quantize. For every 2-D weight: excess kurtosis, the share of weights beyond 6 sigma (outliers), and the relative
error of plain round-to-nearest quantization at 4 and 8 bits (symmetric, groups of 128 along the input).

    python analysis/weight_stats.py models/pplx-decider-flash-v1.3 analysis/results/weights-flash.json
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

torch.set_num_threads(4)


def rtn_error(w: torch.Tensor, bits: int, group: int = 128) -> float:
    rows, cols = w.shape
    if cols % group:
        group = cols
    g = w.reshape(rows, cols // group, group)
    qmax = 2 ** (bits - 1) - 1
    scale = g.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
    q = (g / scale).round().clamp(-qmax - 1, qmax) * scale
    return float((q - g).norm() / g.norm())


def kind_of(name: str) -> str:
    if name.startswith("visual."):
        return "vision"
    if "embed_tokens" in name:
        return "embed"
    m = re.search(r"\.(linear_attn|self_attn|mlp)\.(\w+)", name)
    return f"{m.group(1)}.{m.group(2)}" if m else "other"


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    where = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    rows = []
    by_file = defaultdict(list)
    for name, file in where.items():
        if not name.startswith("visual."):
            by_file[file].append(name)
    for file, names in sorted(by_file.items()):
        with safe_open(str(root / file), framework="pt", device="cpu") as f:
            for name in names:
                w = f.get_tensor(name).float()
                if w.ndim != 2 or "embed_tokens" in name:
                    continue
                std = w.std()
                z = (w - w.mean()) / std
                layer = re.search(r"layers\.(\d+)\.", name)
                rows.append({
                    "name": name, "kind": kind_of(name), "layer": int(layer.group(1)) if layer else -1,
                    "shape": list(w.shape), "kurtosis": float((z ** 4).mean() - 3),
                    "outliers_6sigma": float((z.abs() > 6).float().mean()), "max_over_std": float(w.abs().max() / std),
                    "rtn4": rtn_error(w, 4), "rtn8": rtn_error(w, 8),
                })
        print(f"{file}: {len(rows)} tensors so far", flush=True)
    summary = defaultdict(lambda: defaultdict(float))
    for r in rows:
        s = summary[r["kind"]]
        s["n"] += 1
        for k in ("kurtosis", "outliers_6sigma", "max_over_std", "rtn4", "rtn8"):
            s[k] += r[k]
    table = {k: {m: (v / s["n"] if m != "n" else int(v)) for m, v in s.items()} for k, s in summary.items()}
    worst = sorted(rows, key=lambda r: -r["rtn4"])[:15]
    out.write_text(json.dumps({"summary": table, "worst_rtn4": worst, "tensors": rows}, indent=1))
    print(f"{'kind':<28} {'n':>4} {'kurtosis':>9} {'>6sigma':>9} {'max/std':>8} {'rtn4 err':>9} {'rtn8 err':>9}")
    for k, s in sorted(table.items(), key=lambda kv: -kv[1]["rtn4"]):
        print(f"{k:<28} {s['n']:>4} {s['kurtosis']:>9.2f} {s['outliers_6sigma']:>9.5f} {s['max_over_std']:>8.1f} "
              f"{s['rtn4']:>9.4f} {s['rtn8']:>9.5f}")
    print("worst tensors at 4 bits:")
    for r in worst[:8]:
        print(f"  {r['name']}: rtn4 {r['rtn4']:.4f}, kurtosis {r['kurtosis']:.1f}, max/std {r['max_over_std']:.0f}")


if __name__ == "__main__":
    main()
