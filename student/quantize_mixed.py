# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""4-bit embedding table (GatherBlockQuantized) + dynamic int8 matmuls, from an exported fp32 model.onnx.

    python student/quantize_mixed.py runs/nano-l8/onnx [--block 32]
"""
import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
parser.add_argument("--block", type=int, default=32)
parser.add_argument("--bits", type=int, default=4)
args = parser.parse_args()

src = args.folder / "model.onnx"
tmp = args.folder / f"model.emb{args.bits}.tmp.onnx"
out = args.folder / f"model.emb{args.bits}.int8.onnx"
# int8 matmuls first (shape inference cannot see through the 4-bit contrib op), then the 4-bit table.
quantize_dynamic(str(src), str(tmp), weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul"])
config = DefaultWeightOnlyQuantConfig(block_size=args.block, is_symmetric=True, bits=args.bits,
                                      op_types_to_quantize=("Gather",), quant_axes=(("Gather", 1),))
q = MatMulNBitsQuantizer(onnx.load(str(tmp)), bits=args.bits, block_size=args.block, is_symmetric=True,
                         op_types_to_quantize=("Gather",), quant_axes=(("Gather", 1),), algo_config=config)
q.process()
q.model.save_model_to_file(str(out), use_external_data_format=False)
tmp.unlink()
print(f"{out}: {out.stat().st_size / 1e6:.1f} MB")
