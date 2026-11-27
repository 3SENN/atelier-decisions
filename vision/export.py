# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""atelier-vision as the runtime loads it: two ONNX files, the trimmed tokenizer, and parity fixtures.

    image.q8.onnx          pixels float32 [N, 3, 256, 256] (vision/preprocess.py) -> embedding [N, 768], unit length
    text.q8.emb4.onnx      input_ids int64 [N, 64] (vision/format.py, the trimmed vocabulary's ids) -> embedding [N, 768]
    tokenizer.json, tokenizer_config.json, vocab.json     the trimmed tokenizer (vision/vocab.py) and the ids it kept
    vision.json            sizes, SigLIP's scale and bias, the calibration (vision/calibrate.py adds it), the guard,
                           and what was measured here: how close the shrunk files are to full precision, and their speed
    parity.jsonl           pictures, texts and questions with what this export answers, for runtime/node to match

The adapters are merged into the towers first, so the files are plain SigLIP-shaped towers. Every layer's weights
are stored at 8 bits and the arithmetic stays in floating point (ONNX Runtime's MatMulNBits, blocks of 128); the
text tower's embedding table is cut to the kept vocabulary and stored at 4 bits, as atelier-nano ships its own.

Not int8 arithmetic, which is what atelier-nano and atelier-step use: with it (dynamic quantization of every
MatMul) a picture's vector kept only 0.83 of its direction (cosine to full precision; 0.67 for the worst picture),
against 0.9997 this way. Weights at 4 bits were tried too: 0.97 for the image tower whatever the method (plain,
HQQ, only the MLPs), which costs answers, so the image tower stays at 8; the text tower can go to 4 (--text-bits 4:
0.996, 34 MB smaller), and which of the two ships is decided by measuring search and the held-out questions.

    .venv/bin/python -I vision/export.py --adapters runs/vision-v1/vision.pt --vocab runs/vision/vocab-12 --out runs/vision-v1/onnx
    .venv/bin/python -I vision/export.py --vocab runs/vision/vocab-12 --out runs/vision/base/onnx      # the base, shrunk the same way
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from format import MAX_TOKENS, encode  # noqa: E402
from model import BASE_ID, BASE_REVISION, Towers, tokenizer  # noqa: E402
from preprocess import SIZE, checksum, preprocess  # noqa: E402
from runtime import GUARD, Runtime  # noqa: E402
from store import Pack  # noqa: E402

CACHE = ROOT / "data" / "cache" / "vision"


class ImageTower(torch.nn.Module):
    def __init__(self, towers: Towers) -> None:
        super().__init__()
        self.vision = towers.vision

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.vision(pixel_values=pixels).pooler_output, dim=-1)


class TextTower(torch.nn.Module):
    def __init__(self, towers: Towers) -> None:
        super().__init__()
        self.words = towers.words

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.words(input_ids=input_ids).pooler_output, dim=-1)


def weights(fp32: Path, out: Path, bits: int = 8) -> None:
    """Every layer's weights at `bits` bits (8: blocks of 128, 4: blocks of 32), computed with in floating point."""
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer

    block = 128 if bits == 8 else 32
    config = DefaultWeightOnlyQuantConfig(block_size=block, is_symmetric=True, bits=bits, op_types_to_quantize=("MatMul",))
    q = MatMulNBitsQuantizer(onnx.load(str(fp32)), bits=bits, block_size=block, is_symmetric=True, op_types_to_quantize=("MatMul",), algo_config=config)
    q.process()
    q.model.save_model_to_file(str(out), use_external_data_format=False)


def table_at_4_bits(src: Path, out: Path) -> None:
    """The token embedding table at 4 bits, as atelier-nano ships its own. The 64-row position table stays as it is."""
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer

    model = onnx.load(str(src))
    rows = {init.name: init.dims[0] for init in model.graph.initializer if len(init.dims) == 2}  # the table has thousands, the positions 64
    small = [node.name for node in model.graph.node if node.op_type == "Gather" and rows.get(node.input[0], 10 ** 9) < 1000]
    config = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=4, op_types_to_quantize=("Gather",), quant_axes=(("Gather", 1),))
    q = MatMulNBitsQuantizer(model, bits=4, block_size=32, is_symmetric=True, op_types_to_quantize=("Gather",), quant_axes=(("Gather", 1),), nodes_to_exclude=small, algo_config=config)
    q.process()
    q.model.save_model_to_file(str(out), use_external_data_format=False)


def shipped_table(rows: torch.Tensor) -> torch.Tensor:
    """An embedding table's rows as the shipped file holds them: stored at 4 bits and read back.

    vision/train.py trains with these in place of the exact rows (and with the trimmed tokenizer), so what the
    adapters learn already allows for the rounding. Trained on the exact table, the text tower's vectors moved when
    the table was rounded afterwards: 0.96 of their direction was left on average, 0.88 for some of the very wordings
    it was trained on.
    """
    import tempfile

    import onnxruntime as ort

    with tempfile.TemporaryDirectory() as tmp:
        exact, rounded = Path(tmp) / "table.onnx", Path(tmp) / "table.emb4.onnx"
        table = torch.nn.Embedding(rows.shape[0], rows.shape[1])
        table.weight.data = rows.detach().float().cpu().clone()
        torch.onnx.export(table, (torch.zeros(4, dtype=torch.long),), str(exact), input_names=["ids"], output_names=["rows"], dynamic_axes={"ids": {0: "n"}, "rows": {0: "n"}},
                          opset_version=18, dynamo=False)
        table_at_4_bits(exact, rounded)
        exact.unlink()
        session = ort.InferenceSession(str(rounded), providers=["CPUExecutionProvider"])
        out = [session.run(None, {"ids": np.arange(i, min(i + 8192, rows.shape[0]), dtype=np.int64)})[0] for i in range(0, rows.shape[0], 8192)]
    return torch.from_numpy(np.concatenate(out))


def sample_pictures(rng: random.Random, n: int) -> list[tuple[str, np.ndarray]]:
    """Pictures of every kind there is on disk: (name, uint8 pixels)."""
    out = []
    for name in ("controls", "photos", "documents", "pages"):
        if (CACHE / f"{name}.jsonl").exists():
            pack = Pack(CACHE / name)
            for key in rng.sample(sorted(pack.rows), min(n, len(pack))):
                out.append((f"{name}/{key}", pack.pixels(key)))
    return out


PARITY_TEXTS = [
    "a dog", "een hond", "ein Hund", "un chien", "un perro", "The photo shows a cat.", "Op de foto: een fiets.", "Zu sehen: ein Zug.", "La photo montre une plage.", "Se ve una montaña.",
    "Opens the shopping cart.", "Sluit dit venster of paneel.", "Startet eine Suche.", "Ouvre les paramètres.", "Elimina esto.", "A till receipt from a shop or a restaurant.",
    "Een factuur van een bedrijf, met te betalen bedragen.", "Die Seite fordert zur Anmeldung auf.", "Un avis sur les cookies recouvre une partie de la page.", "La página todavía se está cargando.",
    "Is there a dog in this photo?", "Staat er een kat op deze foto?", "¿Es este el botón de buscar?", "Animals and pets", "Eten en maaltijden", "search", "x", "",
    "MIXED Case And  double   spaces", "naïve café crème brûlée — “quoted” text’s apostrophe", "Straße, Größe, Übergrößenträger", "número 42, 3,50 €, 100%", "🙂 emoji and symbols → ✓",
    "собака на пляже", "海滩上的狗", "ビーチの犬", "كلب على الشاطئ", "σκύλος στην παραλία",
    "a very long description that goes on and on " * 8,
]
PARITY_QUESTIONS = [
    {"what": {"type": "choice", "instructions": "What does this button do?", "criteria": {"cart": "Opens the shopping cart.", "search": "Starts a search.", "close": "Closes this window or panel.", "menu": "Opens the menu."}}},
    {"kind": {"type": "choice", "criteria": {"kassabon": None, "factuur": None, "brief": None, "foto": None}},
     "receipt": {"type": "noul", "instructions": "Is this a receipt?", "criteria": {"true": "A till receipt from a shop or a restaurant.", "false": "Anything else."}}},
    {"dog": {"type": "noul", "instructions": "Is there a dog in this photo?"}, "cat": {"type": "noul", "instructions": "Is there a cat?", "criteria": {"true": "The photo shows a cat."}},
     "album": {"type": "choice", "instructions": "Which album?", "criteria": {"a": "Animals and pets", "b": "Food and meals", "c": "Landscapes and nature"}}},
    {"sharp": {"type": "score", "instructions": "How much text is in the picture?", "criteria": ["No text at all.", "A few words.", "A page full of text."]},
     "state": {"type": "choice", "criteria": {"ready": "An ordinary page with nothing in the way.", "cookies": "A cookie notice covers part of the page.", "signin": "The page asks you to sign in."}}},
    {"bad": {"type": "choice", "criteria": {"only": "one option"}}, "foreign": {"type": "noul", "instructions": "На фото есть собака?"}, "long": {"type": "noul", "instructions": "word " * 80},
     "empty": {"type": "noul"}},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapters", type=Path, help="a trained run's vision.pt; without it, the base as it comes")
    parser.add_argument("--vocab", type=Path, required=True, help="a trimmed tokenizer's folder (vision/vocab.py build)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, help="a vision.json whose calibration to carry over")
    parser.add_argument("--text-bits", type=int, default=8, choices=[4, 8], help="the text tower's layer weights (its table is always 4 bits)")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    import logging

    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.ERROR)  # it reports every layer
    args.out.mkdir(parents=True, exist_ok=True)
    towers = Towers.load(adapters=args.adapters)
    rng = random.Random(7)

    # The image tower.
    image = ImageTower(towers).eval()
    fp32 = args.out / "image.onnx"
    torch.onnx.export(image, (torch.zeros(1, 3, SIZE, SIZE),), str(fp32), input_names=["pixels"], output_names=["embedding"], dynamic_axes={"pixels": {0: "n"}, "embedding": {0: "n"}},
                      opset_version=18, dynamo=False)
    weights(fp32, args.out / "image.q8.onnx")
    fp32.unlink()

    # The text tower, its table cut to the kept vocabulary.
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json"):
        shutil.copy(args.vocab / name, args.out / name)
    keep = json.loads((args.vocab / "vocab.json").read_text(encoding="utf-8"))["keep"]
    full_tok = tokenizer()
    table = towers.words.embeddings.token_embedding
    full_weight = table.weight.data
    trimmed = torch.nn.Embedding(len(keep), full_weight.shape[1])
    trimmed.weight.data = full_weight[keep].clone()
    towers.words.embeddings.token_embedding = trimmed
    text = TextTower(towers).eval()
    fp32, tmp = args.out / "text.onnx", args.out / "text.tmp.onnx"
    torch.onnx.export(text, (torch.zeros(1, MAX_TOKENS, dtype=torch.long),), str(fp32), input_names=["input_ids"], output_names=["embedding"],
                      dynamic_axes={"input_ids": {0: "n"}, "embedding": {0: "n"}}, opset_version=18, dynamo=False)
    text_file = f"text.q{args.text_bits}.emb4.onnx"
    weights(fp32, tmp, args.text_bits)
    table_at_4_bits(tmp, args.out / text_file)
    fp32.unlink()
    tmp.unlink()

    info = {"format": 1, "base": {"id": BASE_ID, "revision": BASE_REVISION}, "adapters": str(args.adapters) if args.adapters else None, "size": SIZE, "tokens": MAX_TOKENS,
            "scale": towers.scale, "bias": towers.bias, "files": {"image": "image.q8.onnx", "text": text_file}, "guard": GUARD, "vocabulary": len(keep)}
    if args.calibration:
        prior = json.loads(args.calibration.read_text(encoding="utf-8"))
        info.update({k: prior[k] for k in ("calibration", "calibrated_on") if k in prior})
    (args.out / "vision.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # How close the shrunk files are to full precision, on pictures and texts of every kind, and how fast they are.
    rt = Runtime(args.out, threads=args.threads, raw=True)
    pictures = sample_pictures(rng, 24)
    cos_image, ms_image = [], []
    with torch.no_grad():
        for _, pixels in pictures:
            want = image(torch.from_numpy(preprocess(pixels))[None])[0].numpy()
            t0 = time.perf_counter()
            got = rt.image(pixels)
            ms_image.append((time.perf_counter() - t0) * 1000)
            cos_image.append(float(want @ got))
        # The text tower in full precision, read two ways: through the exact table and the base's own tokenizer, and
        # through the table and tokenizer as shipped (what a run trained with --vocab learned to read through: then
        # all that is left between it and the shipped file is the layers' 8 bits).
        towers.words.embeddings.token_embedding = table
        exact_text = TextTower(towers).eval()
        own = [t for t in PARITY_TEXTS if t and not rt.unreadable(t)]
        exact = [exact_text(torch.tensor([encode(full_tok, t)[0]]))[0].numpy() for t in own]
        trimmed.weight.data = shipped_table(full_weight[keep])
        towers.words.embeddings.token_embedding = trimmed
        shipped_text = TextTower(towers).eval()
        cos_text, cos_exact, ms_text = [], [], []
        for t, want_exact in zip(own, exact):
            want = shipped_text(torch.tensor([rt.ids(t)[0]]))[0].numpy()
            rt._texts.clear()
            t0 = time.perf_counter()
            got = rt.text(t)
            ms_text.append((time.perf_counter() - t0) * 1000)
            cos_text.append(float(want @ got))
            cos_exact.append(float(want_exact @ got))
    median = lambda xs: round(sorted(xs)[len(xs) // 2], 1)
    info["measured"] = {
        "image_cos_to_fp32": {"mean": round(float(np.mean(cos_image)), 4), "lowest": round(min(cos_image), 4), "pictures": len(cos_image)},
        "text_cos_to_fp32": {"mean": round(float(np.mean(cos_text)), 4), "lowest": round(min(cos_text), 4), "texts": len(cos_text)},
        "text_cos_to_fp32_with_the_exact_table": {"mean": round(float(np.mean(cos_exact)), 4), "lowest": round(min(cos_exact), 4)},
        "cpu_ms_median": {"threads": args.threads, "picture": median(ms_image), "text": median(ms_text)},
        "mb": {p.name: round(p.stat().st_size / 1e6, 1) for p in sorted(args.out.iterdir()) if p.suffix in (".onnx", ".json") and p.name != "vision.json"},
    }
    (args.out / "vision.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # Parity: what runtime/node must reproduce from the same bytes.
    rt = Runtime(args.out, threads=args.threads)
    rows = []
    fixtures: list[tuple[str, np.ndarray]] = []
    by_pack: dict[str, list] = {}
    for name, pixels in pictures:
        by_pack.setdefault(name.split("/")[0], []).append((name, pixels))
    for pack, n in (("controls", 8), ("photos", 1), ("documents", 1), ("pages", 1)):
        fixtures += by_pack.get(pack, [])[:n]
    noise = np.random.default_rng(3).integers(0, 256, (285, 400, 3), dtype=np.uint8)  # a picture that has to be made smaller, both ways unevenly
    noise[70:200, 90:310] = [200, 40, 40]
    fixtures.append(("made/noise-400x285", noise))
    vectors = []
    for name, pixels in fixtures:
        v = rt.image(pixels)
        vectors.append(v)
        rows.append({"picture": {"name": name, "w": int(pixels.shape[1]), "h": int(pixels.shape[0]), "channels": int(pixels.shape[2]), "pixels": base64.b64encode(pixels.tobytes()).decode()},
                     "checksum": checksum(preprocess(pixels)), "embedding": [round(float(x), 6) for x in v]})
    for t in PARITY_TEXTS:
        ids, cut = rt.ids(t)
        rows.append({"text": t, "ids": ids, "cut": cut, "unreadable": rt.unreadable(t), "embedding": [round(float(x), 6) for x in rt.text(t)]})
    for i, questions in enumerate(PARITY_QUESTIONS):
        k = i % len(vectors)
        rows.append({"look": {"picture": k, "questions": questions}, "answers": rt.look(vector=vectors[k], questions=questions)})
    (args.out / "parity.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    m = info["measured"]
    print(f"{args.out}: image {m['mb']['image.q8.onnx']} MB, text {m['mb'][text_file]} MB, tokenizer {m['mb']['tokenizer.json']} MB; "
          f"cos to full precision: pictures {m['image_cos_to_fp32']['mean']} (lowest {m['image_cos_to_fp32']['lowest']}), texts {m['text_cos_to_fp32']['mean']} (lowest {m['text_cos_to_fp32']['lowest']}) "
          f"reading through the shipped table, {m['text_cos_to_fp32_with_the_exact_table']['mean']} (lowest {m['text_cos_to_fp32_with_the_exact_table']['lowest']}) against the exact table; "
          f"{m['cpu_ms_median']['picture']} ms a picture, {m['cpu_ms_median']['text']} ms a text on {args.threads} CPU threads; {len(rows)} parity rows")


if __name__ == "__main__":
    main()
