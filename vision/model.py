# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""atelier-vision: two towers that put a picture and a text in the same space, so a question about a picture is
answered by how close the picture is to each answer's text.

The base is SigLIP 2 (google/siglip2-base-patch16-256, Apache-2.0): an image tower (93M parameters, 256 patches of
16 x 16 pixels, attention pooling) and a text tower (86M, plus a 256k-token embedding table), trained so that
sigmoid(scale * cos(picture, text) + bias) is the chance that the text describes the picture. That one number per
answer is all a decision needs (vision/decide.py), and the picture's vector is reusable: computed once, it answers
every later question about that picture and finds it in a search.

Training (vision/train.py) adds LoRA adapters to both towers' linear layers and merges them into the weights before
export, so the runtime runs two plain ONNX files (vision/export.py).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

from format import encode  # noqa: E402

BASE = ROOT / "models" / "siglip2-base-patch16-256"
BASE_ID = "google/siglip2-base-patch16-256"
BASE_REVISION = "3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab"
DIM = 768


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trained low-rank change: W x + (alpha / r) B A x. B starts at zero."""

    def __init__(self, base: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(r, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale

    def merged(self) -> nn.Linear:
        self.base.weight.data += (self.b @ self.a) * self.scale
        return self.base


def add_lora(module: nn.Module, r: int, alpha: float, last: int | None = None) -> int:
    """LoRA on every linear layer of a tower's encoder blocks (the last `last` blocks, or all). Returns how many."""
    layers = list(module.encoder.layers)
    # Found first, wrapped after: a wrapped layer holds a linear layer itself.
    found = [(parent, name, child) for block in (layers[-last:] if last else layers) for parent in block.modules()
             for name, child in parent.named_children() if isinstance(child, nn.Linear)]
    for parent, name, child in found:
        setattr(parent, name, LoRALinear(child, r, alpha))
    return len(found)


def merge_lora(module: nn.Module) -> None:
    for parent in module.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.merged())


class Towers(nn.Module):
    def __init__(self, siglip) -> None:
        super().__init__()
        self.vision = siglip.vision_model
        self.words = siglip.text_model
        self.scale = float(siglip.logit_scale.detach().exp())
        self.bias = float(siglip.logit_bias.detach())
        self.requires_grad_(False)

    @classmethod
    def load(cls, base: Path = BASE, adapters: Path | None = None) -> Towers:
        """The base's towers; with `adapters` (a vision.pt from vision/train.py), the trained change merged in."""
        from transformers import SiglipModel
        from transformers.utils import logging

        logging.set_verbosity_error()  # the base's config carries CLIP's token ids, which transformers warns about
        # The base is read at 16 bits and computed with at 32: whoever loads it, from the original file or from a
        # copy saved at 16 bits to save disk, starts from exactly the same weights.
        model = cls(SiglipModel.from_pretrained(str(base), dtype=torch.float16).float())
        if adapters is not None:
            blob = torch.load(adapters, map_location="cpu", weights_only=True)
            model.add_adapters(**blob["meta"]["lora"])
            model.load_state_dict(blob["state"], strict=False)
            model.merge()
        return model.eval()

    def add_adapters(self, r: int, alpha: float, last_vision: int | None = None, last_text: int | None = None) -> int:
        self.lora = {"r": r, "alpha": alpha, "last_vision": last_vision, "last_text": last_text}
        return add_lora(self.vision, r, alpha, last_vision) + add_lora(self.words, r, alpha, last_text)

    def merge(self) -> None:
        merge_lora(self.vision)
        merge_lora(self.words)

    def adapter_state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k.endswith((".a", ".b"))}

    def image(self, pixels: torch.Tensor) -> torch.Tensor:
        """float32 [B, 3, 256, 256] in -1..1 (vision/preprocess.py) to unit vectors [B, 768]."""
        return F.normalize(self.vision(pixel_values=pixels).pooler_output, dim=-1)

    def text(self, ids: torch.Tensor) -> torch.Tensor:
        """int64 [B, 64] (vision/format.py encode) to unit vectors [B, 768]."""
        return F.normalize(self.words(input_ids=ids).pooler_output, dim=-1)


def tokenizer(base: Path = BASE):
    """The base's tokenizer without its own padding: vision/format.py pads and cuts."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(base / "tokenizer.json"))
    tok.no_padding()
    tok.no_truncation()
    return tok


def device() -> str:
    """Apple's GPU, a CUDA one, or the CPU; ATELIER_VISION_DEVICE names one (to measure on the CPU while the GPU trains)."""
    import os

    return os.environ.get("ATELIER_VISION_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def embed_texts(model: Towers, tok, texts: list[str], dev: str, batch: int = 128) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        ids = torch.tensor([encode(tok, t)[0] for t in texts[i : i + batch]], device=dev)
        out.append(model.text(ids).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, DIM), dtype=np.float32)


@torch.no_grad()
def embed_pixels(model: Towers, pixels: list[np.ndarray], dev: str) -> np.ndarray:
    """Already preprocessed pictures (float32 [3, 256, 256]) to vectors."""
    return model.image(torch.from_numpy(np.stack(pixels)).to(dev)).float().cpu().numpy()
