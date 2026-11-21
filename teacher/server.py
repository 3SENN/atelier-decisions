"""A local System One server around a pplx decider checkpoint, so Atelier can use it as "your own server".

    python teacher/server.py --checkpoint models/pplx-decider-flash-v1.3 --port 8009 --log data/teacher/live.jsonl

Atelier's client POSTs {model, state, questions} to /v1/systemone and reads {model, answers, usage}; /v1/models
lists the one model served. Every answered request is appended to --log with the raw readout logits, which is
the "collect" step of the distillation plan: real Atelier decisions, labelled by the teacher, on this computer only.
The server listens on 127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))

import pplx_decider.model as decider_model  # noqa: E402
from pplx_decider import Decider, LengthError, RequestError  # noqa: E402
from pplx_decider.protocol import answer  # noqa: E402


def load(checkpoint: Path, device: str, text_only: bool) -> Decider:
    """The reference decider; with text_only the vision tower is dropped before it reaches the GPU."""
    original = decider_model._backbone

    def backbone(path: Path, config: dict, dtype: torch.dtype) -> torch.nn.Module:
        model = original(path, config, dtype)
        if text_only:
            model.visual = torch.nn.Identity()  # only called with pixel_values, which text requests never carry
        return model

    decider_model._backbone = backbone
    try:
        return Decider.from_pretrained(checkpoint, device=device, mode="separate", locate=False)
    finally:
        decider_model._backbone = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--name", default=None, help="model id to report (default: the checkpoint folder name)")
    parser.add_argument("--log", type=Path, default=None, help="append every request, logits and answers here")
    parser.add_argument("--with-vision", action="store_true")
    args = parser.parse_args()

    name = args.name or args.checkpoint.name
    started = time.perf_counter()
    decider = load(args.checkpoint, args.device, text_only=not args.with_vision)
    print(f"loaded {name} in {time.perf_counter() - started:.1f}s; "
          f"GPU memory {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    log_lock = threading.Lock()
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)

    app = FastAPI()

    @app.get("/v1/models")
    def models() -> dict:
        return {"data": [{"id": name, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/systemone")
    @app.post("/systemone")
    async def systemone(http: Request) -> JSONResponse:
        try:
            body = await http.json()
        except Exception as err:  # noqa: BLE001
            raise HTTPException(400, f"Body is not JSON: {err}") from err
        if not isinstance(body, dict):
            raise HTTPException(400, "Body must be an object")
        request = {"state": body.get("state"), "questions": body.get("questions")}
        t0 = time.perf_counter()
        try:
            # logits() runs the forward pass once; answers are computed from it exactly as decide() would.
            logits = decider.logits([request])[0]
        except (RequestError, LengthError) as err:
            raise HTTPException(422, str(err)) from err
        answers = {
            key: answer(q, (torch.tensor(logits[key]) / decider.temperature).softmax(-1).tolist())
            for key, q in request["questions"].items()
        }
        ms = (time.perf_counter() - t0) * 1000
        tokens = sum(c.length for c in decider.encoder.plan(request, decider.max_length, decider.mode))
        if args.log:
            row = {"t": time.time(), "model": name, "ms": round(ms, 1), "request": request, "logits": logits,
                   "temperature": decider.temperature, "answers": answers}
            with log_lock, args.log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return JSONResponse({"model": name, "answers": answers,
                             "usage": {"input_tokens": tokens, "output_tokens": 0}, "ms": round(ms, 1)})

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
