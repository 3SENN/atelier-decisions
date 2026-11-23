# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""The student as a System One server (PyTorch, CPU or GPU), so Atelier's own eval can run against it.

    python student/serve.py --model runs/nano-l8 --port 8010 [--device cpu]

Per-use temperatures fitted on held-out teacher labels (student/calibrate.py) are applied when the model folder
has calibration.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student"))
sys.path.insert(0, str(ROOT / "third_party"))
from format import encode  # noqa: E402
from model import NanoDecider  # noqa: E402
from pplx_decider.protocol import answer  # noqa: E402


class Student:
    def __init__(self, folder: Path, device: str = "cpu", threads: int = 4) -> None:
        torch.set_num_threads(threads)
        self.model, self.meta = NanoDecider.load(folder)
        self.model.eval().to(device)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.meta["base"])
        calibration = folder / "calibration.json"
        self.temperatures = json.loads(calibration.read_text()) if calibration.exists() else {}

    @torch.inference_mode()
    def decide(self, request: dict) -> dict:
        questions = request.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("A request needs at least one question")
        encoded = [encode(self.tokenizer, request.get("state", ""), key, q, self.meta["max_length"])
                   for key, q in questions.items()]
        width = max(len(e["input_ids"]) for e in encoded)
        ids = torch.full((len(encoded), width), self.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        rows, cols = [], []
        for i, e in enumerate(encoded):
            ids[i, : len(e["input_ids"])] = torch.tensor(e["input_ids"])
            mask[i, : len(e["input_ids"])] = 1
            rows += [i] * len(e["markers"])
            cols += e["markers"]
        flat = self.model(ids.to(self.device), mask.to(self.device), torch.tensor(rows, device=self.device),
                          torch.tensor(cols, device=self.device)).float().cpu()
        answers, cursor = {}, 0
        for (key, q), e in zip(questions.items(), encoded, strict=True):
            n = len(e["markers"])
            t = float(self.temperatures.get(e["kind"], 1.0))
            answers[key] = answer(q, (flat[cursor : cursor + n] / t).softmax(-1).tolist())
            cursor += n
        return {"answers": answers, "usage": {"input_tokens": int(mask.sum()), "output_tokens": 0}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    student = Student(args.model, args.device, args.threads)
    name = f"atelier-nano-{args.model.name}"
    app = FastAPI()

    @app.get("/v1/models")
    def models() -> dict:
        return {"data": [{"id": name, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/systemone")
    @app.post("/systemone")
    async def systemone(http: Request) -> JSONResponse:
        body = await http.json()
        t0 = time.perf_counter()
        try:
            out = student.decide({"state": body.get("state"), "questions": body.get("questions")})
        except (ValueError, KeyError, TypeError) as err:
            raise HTTPException(422, str(err)) from err
        return JSONResponse({"model": name, **out, "ms": round((time.perf_counter() - t0) * 1000, 2)})

    print(f"serving {name} on 127.0.0.1:{args.port} ({args.device})", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
