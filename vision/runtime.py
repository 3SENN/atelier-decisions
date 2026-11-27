# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""atelier-vision as it ships, in Python: the two ONNX files on the CPU, the trimmed tokenizer, the calibration.

This is what runtime/node/src/vision/model.ts does, written again here so the shipped files can be measured with the
same scripts as the trained model (vision/measure.py --runtime, vision/search.py --runtime) and so vision/export.py
can write down what the runtime must answer (parity.jsonl).

    from runtime import Runtime
    rt = Runtime("releases/vision-v0.1/model")
    rt.look(pixels, {"kind": {"type": "choice", "criteria": {"receipt": "A till receipt", "photo": "An ordinary photo"}}})

A picture is uint8 pixels [H, W, 3 or 4]; its vector can be kept and asked again later (look(vector=...)).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

import numpy as np  # noqa: E402

from decide import RAW, answer, probabilities  # noqa: E402
from format import clean, encode, labels, texts, unanswerable  # noqa: E402
from preprocess import preprocess  # noqa: E402

# When the model can't read a text it declines instead of guessing, as atelier-nano does (runtime/node/src/decider.ts):
# the vocabulary is trimmed to Latin-script languages, so other scripts fall apart into raw bytes.
GUARD = {"min_latin": 0.7, "max_bytes": 0.1, "min_letters": 8}
# Latin letters by code point, so Python and JavaScript agree on every character: basic, Latin-1 to Extended-B,
# Extended Additional, fullwidth.
LATIN = [(0x41, 0x5A), (0x61, 0x7A), (0xAA, 0xAA), (0xBA, 0xBA), (0xC0, 0x24F), (0x1E00, 0x1EFF), (0xFF21, 0xFF3A), (0xFF41, 0xFF5A)]


def is_latin(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in LATIN)


def unreadable(text: str, ids: list[int], byte_ids: set[int], guard: dict = GUARD) -> str | None:
    """Why the model can't read this text, or None when it can. ids: its tokens before <eos> and padding."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= guard["min_letters"]:
        latin = sum(1 for c in letters if is_latin(c)) / len(letters)
        if latin < guard["min_latin"]:
            return f"mostly not Latin script ({int(latin * 100 + 0.5)}% Latin)"
    if ids:
        share = sum(1 for i in ids if i in byte_ids) / len(ids)
        if share > guard["max_bytes"]:
            return f"{int(share * 100 + 0.5)}% of the text is outside the vocabulary"
    return None


class Runtime:
    def __init__(self, folder: Path | str, threads: int = 2, raw: bool = False, text_cache: Path | None = None) -> None:
        """text_cache: a file to keep texts' vectors in between measuring scripts (a text takes the shipped text tower a
        tenth of a second, and the reports read tens of thousands, most of them the same ones)."""
        import onnxruntime as ort
        from tokenizers import Tokenizer

        folder = Path(folder)
        self.folder = folder
        self.info = json.loads((folder / "vision.json").read_text(encoding="utf-8"))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.image_session = ort.InferenceSession(str(folder / self.info["files"]["image"]), opts, providers=["CPUExecutionProvider"])
        self.text_session = ort.InferenceSession(str(folder / self.info["files"]["text"]), opts, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
        vocab = self.tok.get_vocab()
        self.byte_ids = {i for t, i in vocab.items() if t.startswith("<0x") and t.endswith(">") and len(t) == 6}
        self.scale, self.bias = self.info["scale"], self.info["bias"]
        self.cal = RAW if raw else self.info.get("calibration", RAW)
        self.guard = self.info.get("guard", GUARD)
        self.size = self.info["size"]
        self._texts: dict[str, np.ndarray] = {}
        self._cache, self._cached = text_cache, 0
        if text_cache is not None and text_cache.exists():
            blob = np.load(text_cache, allow_pickle=False)
            if str(blob["stamp"]) == self._text_stamp(folder):
                self._texts = dict(zip((str(t) for t in blob["texts"]), blob["vectors"].astype(np.float32)))
                self._cached = len(self._texts)

    def _text_stamp(self, folder: Path) -> str:
        path = folder / self.info["files"]["text"]
        return hashlib.sha256(path.read_bytes()).hexdigest()  # another text file: nothing kept is used

    def save_texts(self) -> None:
        """Writes the texts read so far to the cache file, when there are new ones."""
        if self._cache is not None and len(self._texts) > self._cached:
            names = sorted(self._texts)
            self._cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self._cache, stamp=self._text_stamp(self.folder), texts=np.array(names), vectors=np.stack([self._texts[t] for t in names]).astype(np.float32))
            self._cached = len(self._texts)

    def image(self, pixels: np.ndarray) -> np.ndarray:
        """A picture's vector (unit length)."""
        return self.image_session.run(None, {"pixels": preprocess(pixels, self.size)[None]})[0][0]

    def ids(self, text: str) -> tuple[list[int], bool]:
        return encode(self.tok, text)

    def unreadable(self, text: str) -> str | None:
        raw = self.tok.encode(clean(text), add_special_tokens=False).ids
        return unreadable(clean(text), raw, self.byte_ids, self.guard)

    def text(self, text: str) -> np.ndarray:
        """A text's vector (unit length); kept, since the same answers are asked of many pictures."""
        self.read([text])
        return self._texts[text]

    def read(self, wanted: list[str], batch: int = 32) -> None:
        """Every text not read before, through the text tower together (a question with 60 options is one pass, not 60)."""
        new = [t for t in dict.fromkeys(wanted) if t not in self._texts]
        for i in range(0, len(new), batch):
            chunk = new[i : i + batch]
            out = self.text_session.run(None, {"input_ids": np.array([self.ids(t)[0] for t in chunk], dtype=np.int64)})[0]
            self._texts.update(zip(chunk, out))
            if i % (batch * 64) == batch * 63:
                self.save_texts()  # a long reading survives being stopped

    def decline(self, q: dict) -> str | None:
        """Why a question isn't answered: its shape, a text the model can't read, or a text longer than it reads."""
        why = unanswerable(q)
        if why:
            return why
        for text in texts(q):
            if text is None:
                continue
            bad = self.unreadable(text)
            if bad:
                return bad
            if self.ids(text)[1]:
                return "a text is longer than the model reads (63 tokens)"
        return None

    def look(self, pixels: np.ndarray | None = None, questions: dict | None = None, vector: np.ndarray | None = None) -> dict:
        """System One answers for questions about one picture."""
        v = self.image(pixels) if vector is None else vector / np.linalg.norm(vector)
        out = {}
        reasons = {key: self.decline(q) for key, q in (questions or {}).items()}
        self.read([t for key, q in (questions or {}).items() if not reasons[key] for t in texts(q) if t is not None])
        for key, q in (questions or {}).items():
            why = reasons[key]
            if why:
                out[key] = {"type": "refusal", "reason": why}
                continue
            cos = [None if text is None else float(v @ self.text(text)) for text in texts(q)]
            out[key] = answer(q, labels(q), probabilities(q["type"], cos, self.scale, self.bias, self.cal))
        return out
