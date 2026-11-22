"""Per-decision temperatures for atelier-mini, with the same Atelier costs as student/calibrate.py, fitted on held-out
teacher labels it never trained on (the 5% hash split). Writes <model>/calibration.json and calibration-report.json;
student/merge_mini.py carries them into decision.json.

    python student/calibrate_mini.py --model runs/mini-l24 --data data/teacher/pool-*.flash.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student"))
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "third_party"))
from calibrate import COST, COSTS  # noqa: E402
from mini_tokenizer import mini_processor  # noqa: E402
from pplx_decider.protocol import Encoder  # noqa: E402
from readers import verdict  # noqa: E402
from train import read_jsonl, without_eval  # noqa: E402
from train_mini import FLASH, MiniDecider, encode_rows, reference_kernels, scores_of  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--per-kind", type=int, default=1500)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cpu":
        reference_kernels()
    saved = torch.load(args.model / "mini.pt", map_location="cpu")
    base = Path(saved["base"])
    flash = json.loads((FLASH / "decision_config.json").read_text())
    encoder = Encoder(mini_processor(base), flash["codes"])

    rows = without_eval([r for p in args.data for r in read_jsonl(p)])
    held = [r for r in rows if int(hashlib.md5(json.dumps(r["request"], sort_keys=True).encode()).hexdigest(), 16) % 20 == 0]
    random.Random(5).shuffle(held)
    examples, _ = encode_rows(held, encoder, 1024, None)
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for e in examples:
        if len(by_kind[e["kind"]]) < args.per_kind:
            by_kind[e["kind"]].append(e)
    chosen = [e for items in by_kind.values() for e in items]

    model = MiniDecider(base, saved["layers"], saved["rank"], 2.0 * saved["rank"], flash["token_ids"]).to(device)
    model.load_state_dict(saved["trainable"], strict=False)
    scores = scores_of(model, chosen, encoder.pad_id, device, 16384)
    items_by_kind: dict[str, list] = defaultdict(list)
    for e, s in zip(chosen, scores, strict=True):
        items_by_kind[e["kind"]].append((s, torch.tensor(e["teacher"]).softmax(-1), e["labels"]))

    grid = [round(0.5 + 0.05 * i, 2) for i in range(51)]
    temperatures, report = {}, {}
    for kind, items in sorted(items_by_kind.items()):
        teacher = [verdict(kind, labels, p.tolist()) for _, p, labels in items]

        def counts(t: float) -> dict[str, float]:
            out = {"agree": 0.0, "conflict": 0.0, "overreach": 0.0, "underreach": 0.0}
            for (s, _, labels), v_t in zip(items, teacher, strict=True):
                v = verdict(kind, labels, (s / t).softmax(-1).tolist())
                out["agree"] += v == v_t
                out["conflict"] += v is not None and v_t is not None and v != v_t
                out["overreach"] += v is not None and v_t is None
                out["underreach"] += v is None and v_t is not None
            return {k: round(x / len(items), 4) for k, x in out.items()}

        weights = COSTS.get(kind, COST)
        t_best = min(grid, key=lambda t: (round(sum(weights[k] * counts(t)[k] for k in weights), 4), abs(t - 1.0)))
        temperatures[kind] = t_best
        report[kind] = {"n": len(items), "T": t_best, "at_1": counts(1.0), "at_T": counts(t_best)}
        a, b = report[kind]["at_1"], report[kind]["at_T"]
        print(f"{kind:<11} n={len(items):>5} T={t_best:<5} agree/conflict/overreach/underreach "
              f"{a['agree']:.3f}/{a['conflict']:.3f}/{a['overreach']:.3f}/{a['underreach']:.3f} -> "
              f"{b['agree']:.3f}/{b['conflict']:.3f}/{b['overreach']:.3f}/{b['underreach']:.3f}", flush=True)
    (args.model / "calibration.json").write_text(json.dumps(temperatures, indent=1))
    (args.model / "calibration-report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
