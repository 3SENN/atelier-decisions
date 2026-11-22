"""The mini's processor, with the split rule of the base's own tokenizer.json.

transformers 5.17 builds Qwen3.5's tokenizer with the Qwen2 split rule (letters only), not the rule in the file
(letters with their combining marks). The teachers ship the Qwen2 rule in their own tokenizer.json, so their labels
are right as they are; but the mini's base, Qwen3.5-0.8B, was pretrained with the file's rule, and reads Hindi and
Thai better with it (analysis/tokenizer_effect.py: Thai right when deciding 40% -> 70%, Hindi 69% -> 78%; other
languages unchanged, token for token). Every mini script loads its tokenizer here, and the runtimes (llama.cpp,
tokenizers.js) follow the file anyway.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoProcessor


def mini_processor(folder: Path):
    processor = AutoProcessor.from_pretrained(str(folder))
    processor.tokenizer.backend_tokenizer.pre_tokenizer = Tokenizer.from_file(str(Path(folder) / "tokenizer.json")).pre_tokenizer
    return processor
