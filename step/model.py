# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""atelier-step: a small encoder that answers a whole browser step in one pass.

The backbone is atelier-nano's (mmBERT-small, trimmed vocabulary), which already reads Atelier's pages and controls
in many languages. Three heads read the step's markers (step/format.py):

- op:    the STEP marker → click, type, done, blocked
- el:    every EL marker → one score; the target is the best control of the operation's kind
- value: every field's EL marker against every VAL marker → which named value goes in (or none of them)
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, ModernBertModel

ROOT = Path(__file__).resolve().parents[1]


def _nano_module():
    # student/model.py, loaded by its path: both folders have a model.py and a format.py.
    import importlib.util

    spec = importlib.util.spec_from_file_location("atelier_nano_model", ROOT / "student" / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OPS = ["click", "type", "done", "blocked"]  # step/format.py OPS


class StepModel(nn.Module):
    def __init__(self, backbone: ModernBertModel) -> None:
        super().__init__()
        self.backbone = backbone
        h = backbone.config.hidden_size
        self.op = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, len(OPS)))
        self.el = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
        self.vq = nn.Linear(h, h)
        self.vk = nn.Linear(h, h)
        self.scale = h ** -0.5

    @classmethod
    def from_nano(cls, nano_dir: Path, layers: int | None = None) -> StepModel:
        """atelier-nano's backbone (all its layers, or `layers` of them spread evenly) and new heads."""
        nano, _ = _nano_module().NanoDecider.load(nano_dir)
        backbone = nano.backbone
        if layers and layers < backbone.config.num_hidden_layers:
            total = backbone.config.num_hidden_layers
            keep = sorted({round(i * (total - 1) / (layers - 1)) for i in range(layers)})
            backbone.layers = nn.ModuleList([backbone.layers[i] for i in keep])
            backbone.config.num_hidden_layers = len(keep)
            backbone.config.layer_types = [backbone.config.layer_types[i] for i in keep]
        return cls(backbone)

    def hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def heads(self, h: torch.Tensor, step: int, els: torch.Tensor, vals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For one sequence's hidden states h [T, H]: op logits [4], element logits [N], value logits [N, M]."""
        op = self.op(h[step])
        e = h[els]
        el = self.el(e).squeeze(-1)
        val = (self.vq(e) @ self.vk(h[vals]).T) * self.scale
        return op, el, val

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, els: torch.Tensor, vals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step (batch of one), as exported to ONNX: the STEP marker is always at position 1."""
        h = self.hidden(input_ids, attention_mask)[0]
        return self.heads(h, 1, els, vals)

    def save(self, out: Path, meta: dict) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.backbone.config.save_pretrained(str(out))
        torch.save({"state": self.state_dict(), "meta": meta}, out / "step.pt")

    @classmethod
    def load(cls, out: Path) -> tuple[StepModel, dict]:
        config = AutoConfig.from_pretrained(str(out))
        model = cls(ModernBertModel(config))
        blob = torch.load(out / "step.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(blob["state"])
        return model, blob["meta"]
