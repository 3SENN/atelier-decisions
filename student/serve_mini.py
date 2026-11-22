"""atelier-mini as a System One server: llama.cpp runs the backbone, this applies the decider's readout.

    python student/serve_mini.py --model runs/mini-l24 --gguf runs/mini-l24/mini-l24-Q8_0.decider.gguf --port 8020

Starts llama-server (the GGUF marked by student/gguf_decider.py: non-causal attention, last-token pooling) on a
private port, encodes each question as pplx_decider does (system prompt, "State:", "Question:", lettered options,
<|fim_suffix|>), takes the final hidden state there, applies readout.safetensors and the per-kind temperature
(decision.json), and answers in Atelier's System One format. No language guard, open questions answered: this
serves the model on its own, for Atelier's eval.
"""

from __future__ import annotations

import argparse
import atexit
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student"))
sys.path.insert(0, str(ROOT / "third_party"))
from format import fixed_kind  # noqa: E402
from mini_tokenizer import mini_processor  # noqa: E402
from pplx_decider.protocol import Encoder, answer, options  # noqa: E402

SERVER = ROOT / "models" / "llama.cpp-b12620" / "bin" / "llama-server.exe"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Mini:
    def __init__(self, merged: Path, gguf: Path, threads: int, temperatures: dict | None = None) -> None:
        decision = json.loads((merged / "decision.json").read_text())
        self.temperatures = temperatures if temperatures is not None else decision.get("temperatures", {})
        self.readout = load_file(str(merged / "readout.safetensors"))["weight"]
        self.encoder = Encoder(mini_processor(merged), decision["codes"])
        self.port = free_port()
        self.server = subprocess.Popen([str(SERVER), "-m", str(gguf), "--embeddings", "--pooling", "last", "-c", "8192",
                                        "-b", "8192", "-ub", "8192", "-t", str(threads), "--port", str(self.port),
                                        "--host", "127.0.0.1", "-np", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        atexit.register(self.close)  # never leave a llama-server behind, even when the caller crashes
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1)
                return
            except Exception:
                if self.server.poll() is not None:
                    raise RuntimeError("llama-server exited")
                time.sleep(0.5)
        raise RuntimeError("llama-server did not start")

    def hidden(self, ids: list[int]) -> torch.Tensor:
        body = json.dumps({"content": ids, "embd_normalize": -1}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/embedding", body, {"Content-Type": "application/json"})
        out = json.loads(urllib.request.urlopen(req, timeout=300).read())
        e = (out[0] if isinstance(out, list) else out)["embedding"]
        return torch.tensor(e[0] if isinstance(e[0], list) else e)

    def decide(self, request: dict) -> dict:
        plan = self.encoder.plan(request, None, "separate")
        answers, tokens = {}, 0
        for enc, (key, q) in zip(plan, request["questions"].items(), strict=True):
            tokens += enc.length
            z = self.readout @ self.hidden(enc.input_ids.tolist())
            n = len(options(q)[0])
            t = float(self.temperatures.get(fixed_kind(key, q) or "open", 1.0))
            answers[key] = answer(q, (z[:n] / t).softmax(-1).tolist())
        return {"answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}

    def close(self) -> None:
        if self.server.poll() is None:
            self.server.terminate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    calibration = args.model / "calibration.json"  # newer than the merge's decision.json when recalibrated
    mini = Mini(args.model / "merged", args.gguf, args.threads, json.loads(calibration.read_text()) if calibration.exists() else None)
    name = f"atelier-mini-{args.model.name}"
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
            out = mini.decide({"state": body.get("state"), "questions": body.get("questions")})
        except (ValueError, KeyError, TypeError) as err:
            raise HTTPException(422, str(err)) from err
        return JSONResponse({"model": name, **out, "ms": round((time.perf_counter() - t0) * 1000, 2)})

    print(f"serving {name} on 127.0.0.1:{args.port} (llama.cpp {args.gguf.name})", flush=True)
    try:
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    finally:
        mini.close()


if __name__ == "__main__":
    main()
