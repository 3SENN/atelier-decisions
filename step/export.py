# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""atelier-step as the runtime loads it: ONNX (int8 layers and embedding table), the tokenizer, and parity fixtures.

    model.int8.onnx     inputs input_ids [1, T], attention_mask [1, T], els [N], vals [M]
                        outputs op [4], el [N], val [N, M]  (logits; the runtime applies softmax)
    tokenizer.json, tokenizer_config.json, config.json, step.json (format version, ops, limits, measured metrics)
    parity.jsonl        steps, their token ids and marker positions, and PyTorch's logits, for runtime/node to match

    .venv/bin/python -I step/export.py --model runs/step-v1 --nano releases/v0.5/model --out runs/step-v1/onnx \\
        --fixtures data/cache/synth-steps.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "step"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from format import LIMITS, MAX_LEN, OPS, encode  # noqa: E402
from model import StepModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--nano", type=Path, default=ROOT / "releases/v0.5/model", help="where the tokenizer comes from")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, help="generated steps (step/synth.py) for parity.jsonl")
    parser.add_argument("--n-fixtures", type=int, default=24)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model, meta = StepModel.load(args.model)
    model.eval()
    tok = Tokenizer.from_file(str(args.nano / "tokenizer.json"))
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(args.nano / name, args.out / name)
    model.backbone.config.save_pretrained(str(args.out))

    # A real step as the example input, so the traced shapes are ordinary ones.
    rows = [json.loads(line) for line in args.fixtures.read_text(encoding="utf-8").splitlines()[: 4000]] if args.fixtures else []
    sample = next((r for r in rows if r["op"] == "type"), None) or {"goal": "Order it", "values": [{"name": "size", "value": "43"}], "before": [], "page": "Shop",
                                                                      "elements": [{"role": "field", "label": "Size", "value": ""}, {"role": "button", "label": "Add to cart"}]}
    enc = encode(tok, sample)
    ids = torch.tensor([enc["ids"]])
    mask = torch.ones_like(ids)
    els = torch.tensor(enc["els"])
    vals = torch.tensor(enc["vals"])
    fp32 = args.out / "model.onnx"
    torch.onnx.export(model, (ids, mask, els, vals), str(fp32), input_names=["input_ids", "attention_mask", "els", "vals"], output_names=["op", "el", "val"],
                      dynamic_axes={"input_ids": {1: "length"}, "attention_mask": {1: "length"}, "els": {0: "n"}, "vals": {0: "m"}, "el": {0: "n"}, "val": {0: "n", 1: "m"}},
                      opset_version=18, dynamo=False)
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8 = args.out / "model.int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul", "Gather"])
    # The smaller file: the embedding table at 4 bits (as atelier-nano ships), the layers at int8.
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer

    tmp = args.out / "model.tmp.onnx"
    emb4 = args.out / "model.emb4.int8.onnx"
    quantize_dynamic(str(fp32), str(tmp), weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul"])
    config = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=4, op_types_to_quantize=("Gather",), quant_axes=(("Gather", 1),))
    q = MatMulNBitsQuantizer(onnx.load(str(tmp)), bits=4, block_size=32, is_symmetric=True, op_types_to_quantize=("Gather",), quant_axes=(("Gather", 1),), algo_config=config)
    q.process()
    q.model.save_model_to_file(str(emb4), use_external_data_format=False)
    tmp.unlink()
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(str(int8), opts, providers=["CPUExecutionProvider"])
    session4 = ort.InferenceSession(str(emb4), opts, providers=["CPUExecutionProvider"])

    def run(e: dict, sess=None) -> list[np.ndarray]:
        feed = {"input_ids": np.array([e["ids"]], dtype=np.int64), "attention_mask": np.ones((1, len(e["ids"])), dtype=np.int64),
                "els": np.array(e["els"], dtype=np.int64), "vals": np.array(e["vals"], dtype=np.int64)}
        return (sess or session).run(None, feed)

    # Parity: PyTorch fp32 against ONNX int8 on real steps, and the fixtures runtime/node checks itself against.
    fixtures, agree, agree4, total = [], 0, 0, 0
    times = []
    for r in rows[:: max(1, len(rows) // args.n_fixtures)][: args.n_fixtures]:
        e = encode(tok, r)
        if not e["els"]:
            continue
        with torch.no_grad():
            op, el, val = model(torch.tensor([e["ids"]]), torch.ones(1, len(e["ids"]), dtype=torch.long), torch.tensor(e["els"]), torch.tensor(e["vals"]))
        t0 = time.perf_counter()
        o_op, o_el, o_val = run(e)
        times.append((time.perf_counter() - t0) * 1000)
        total += 1
        agree += int(int(op.argmax()) == int(o_op.argmax()) and int(el.argmax()) == int(o_el.argmax()))
        f_op, f_el, _ = run(e, session4)
        agree4 += int(int(op.argmax()) == int(f_op.argmax()) and int(el.argmax()) == int(f_el.argmax()))
        fixtures.append({"step": {k: r[k] for k in ("goal", "values", "before", "page", "elements")}, "ids": e["ids"], "els": e["els"], "vals": e["vals"],
                         "op": o_op.tolist(), "el": o_el.tolist(), "val": o_val.tolist()})
    (args.out / "parity.jsonl").write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in fixtures), encoding="utf-8")
    info = {"format": 1, "ops": OPS, "max_len": MAX_LEN, "limits": LIMITS, "markers": {"step": "<unused14>", "val": "<unused15>", "el": "<unused16>", "tx": "<unused17>"},
            "trained": meta.get("updates"), "metrics": meta.get("metrics"), "int8_agrees_with_fp32": f"{agree}/{total}", "emb4_agrees_with_fp32": f"{agree4}/{total}",
            "mb": {"model.int8.onnx": round(int8.stat().st_size / 1e6, 1), "model.emb4.int8.onnx": round(emb4.stat().st_size / 1e6, 1)},
            "cpu_ms_median_2_threads": round(sorted(times)[len(times) // 2], 1) if times else None}
    (args.out / "step.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    fp32.unlink()
    size = int8.stat().st_size / 1e6
    print(f"{int8} {size:.1f} MB, agrees with fp32 on {agree}/{total} steps (operation and element); {emb4.name} {emb4.stat().st_size / 1e6:.1f} MB, {agree4}/{total} · {info['cpu_ms_median_2_threads']} ms median on 2 CPU threads")


if __name__ == "__main__":
    main()
