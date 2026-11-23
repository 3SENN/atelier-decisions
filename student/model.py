# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""The student: a few layers of mmBERT-small (ModernBERT) and a small head that scores each option marker."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, ModernBertModel


def spread(total: int, keep: int) -> list[int]:
    """`keep` layer indices spread evenly over `total`, always including the first and the last."""
    if keep >= total:
        return list(range(total))
    if keep == 1:
        return [total - 1]
    return sorted({round(i * (total - 1) / (keep - 1)) for i in range(keep)})


def pruned_backbone(path: Path, layers: list[int]) -> ModernBertModel:
    full = ModernBertModel.from_pretrained(str(path), dtype=torch.float32)
    config = copy.deepcopy(full.config)
    config.num_hidden_layers = len(layers)
    config.layer_types = [full.config.layer_types[i] for i in layers]
    small = ModernBertModel(config)
    state = {}
    for name, value in full.state_dict().items():
        if name.startswith("layers."):
            index, rest = name[len("layers."):].split(".", 1)
            if int(index) in layers:
                state[f"layers.{layers.index(int(index))}.{rest}"] = value
        else:
            state[name] = value
    small.load_state_dict(state, strict=True)
    return small


class NanoDecider(nn.Module):
    def __init__(self, backbone: ModernBertModel) -> None:
        super().__init__()
        self.backbone = backbone
        h = backbone.config.hidden_size
        self.head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    @classmethod
    def from_mmbert(cls, path: Path, keep: int, vocab: list[int] | None = None) -> NanoDecider:
        """`keep` layers of mmBERT; with `vocab` (original token ids, in their new order: a trimmed tokenizer's
        keep.json) only those rows of the embedding table."""
        total = AutoConfig.from_pretrained(str(path)).num_hidden_layers
        backbone = pruned_backbone(path, spread(total, keep))
        if vocab is not None:
            old = backbone.embeddings.tok_embeddings
            new = nn.Embedding(len(vocab), old.embedding_dim, padding_idx=vocab.index(0) if 0 in vocab else None)
            new.weight.data = old.weight.data[torch.tensor(vocab)].clone()
            backbone.embeddings.tok_embeddings = new
            backbone.config.vocab_size = len(vocab)
            backbone.config.pad_token_id = vocab.index(backbone.config.pad_token_id)
        return cls(backbone)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, rows: torch.Tensor,
                cols: torch.Tensor) -> torch.Tensor:
        """Logits of the option markers at (rows, cols), flat; the caller groups them by question."""
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(hidden[rows, cols]).squeeze(-1)

    def save(self, out: Path, meta: dict) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.backbone.config.save_pretrained(str(out))
        torch.save({"state": self.state_dict(), "meta": meta}, out / "student.pt")

    @classmethod
    def load(cls, out: Path) -> tuple[NanoDecider, dict]:
        config = AutoConfig.from_pretrained(str(out))
        model = cls(ModernBertModel(config))
        blob = torch.load(out / "student.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(blob["state"])
        return model, blob["meta"]
