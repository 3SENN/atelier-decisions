"""Reference answers from an exported ONNX folder, computed in Python, for runtime/node/src/parity.ts.

    python student/parity_dump.py runs/nano-l8/onnx data/teacher/eval-flash.jsonl runs/nano-l8/onnx/parity.jsonl
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student"))
sys.path.insert(0, str(ROOT / "third_party"))
from format import encode  # noqa: E402
from pplx_decider.protocol import answer  # noqa: E402

folder, requests, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
model_file = sys.argv[4] if len(sys.argv) > 4 else "model.int8.onnx"
tokenizer = AutoTokenizer.from_pretrained(str(folder))
vocab = json.loads((folder / "vocab.json").read_text())
remap = {old: new for new, old in enumerate(vocab["keep"])}
temps = json.loads((folder / "calibration.json").read_text()) if (folder / "calibration.json").exists() else {}
session = ort.InferenceSession(str(folder / model_file), providers=["CPUExecutionProvider"])
with requests.open(encoding="utf-8") as f, out.open("w", encoding="utf-8") as w:
    for line in f:
        request = json.loads(line)["request"]
        answers, all_ids = {}, {}
        for key, q in request["questions"].items():
            e = encode(tokenizer, request["state"], key, q, 512)
            ids = np.array([[remap.get(i, vocab["unk"]) for i in e["input_ids"]]], dtype=np.int64)
            all_ids[key] = ids[0].tolist()
            scores = session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0]
            z = scores[e["markers"]] / float(temps.get(e["kind"], 1.0))
            p = np.exp(z - z.max())
            answers[key] = answer(q, (p / p.sum()).tolist())
        w.write(json.dumps({"request": request, "expect_answers": answers, "expect_ids": all_ids},
                           ensure_ascii=False) + "\n")
print(f"wrote {out}")
