# Copyright (c) 2026 The Atelier Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Trains atelier-vision's towers on questions whose answers are known, without losing what the base already knows.

What is learned (LoRA on the linear layers of both towers; the base's weights stay as they are until the merge):

- to tell apart what SigLIP 2 was never shown much of: buttons that are only an icon, kinds of document, the state a
  page is in (vision/synth.py), with the true answer by construction;
- to read an answer however a caller words it: a bare name, a sentence, a question, in five languages
  (vision/questions.py), on those pictures and on photos with human-checked labels (vision/photos.py).

What must not be lost: the base finds a photo by a caption in dozens of languages. Two terms hold it in place:

- every photo's vector stays where the base put it (1 - cos to the base's vector, which vision/embed.py cached);
- the text tower keeps reading ordinary captions as the base did: COCO-35L's dev captions in 21 languages, as text
  only (vision/photos.py coco35). Search is measured on Crossmodal-3600, which training never touches.

Loss per picture: a choice is cross-entropy over its options; a yes/no question is binary cross-entropy on the "true"
text's match, plus, when a "false" text is given, which of the two matches better.

    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.35 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 PYTORCH_ENABLE_MPS_FALLBACK=1
    .venv/bin/python -I vision/embed.py --packs photos xm3600 controls documents pages --out runs/vision/emb/base
    .venv/bin/python -I vision/train.py --out runs/vision-v1 --epochs 2
    .venv/bin/python -I vision/train.py --init runs/vision-v1 --out runs/vision-v2 --lr 5e-5      # to go on from one

--vocab trains with the text tower's embedding table and tokenizer as they ship (the trimmed vocabulary, its rows at
4 bits: vision/export.py shipped_table), so the adapters learn with the rounding in place.

Like step/train.py it watches the machine: it saves and stops when the disk drops below 1 GB free or swap grows by
more than 512 MB.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from data import CACHE, photo_rows  # noqa: E402
from decide import probabilities, verdict  # noqa: E402
from embed import load, tower_input  # noqa: E402
from format import encode, labels, texts  # noqa: E402
from model import Towers, device, embed_texts, tokenizer  # noqa: E402
from store import Pack  # noqa: E402
from synth import PACKS, synth_rows  # noqa: E402


def pressure() -> tuple[float, float]:
    """Free disk (GB) and swap in use (MB), as step/train.py."""
    import shutil
    import subprocess

    free = shutil.disk_usage(Path.home()).free / 1e9
    swap = 0.0
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5).stdout
        swap = float(out.split("used = ")[1].split("M")[0])
    except Exception:
        pass
    return free, swap


class Pictures:
    """Every pack's pictures as tower input, read when asked for."""

    def __init__(self) -> None:
        self.packs = {name: Pack(CACHE / name) for name in ["photos", *PACKS.values()] if (CACHE / f"{name}.jsonl").exists()}

    def __call__(self, image: str) -> np.ndarray:
        pack, key = image.split("/", 1)
        return tower_input(self.packs[pack].pixels(key))


def epoch_examples(seed: int, families: set[str], buttons: bool = False) -> list[dict]:
    """One pass: every training picture with one choice and one yes/no question about it, newly worded each pass.

    buttons: what the second run added after the first was measured: buttons that are none of the jobs (always a
    "no"), and a button's job asked among every trained job at once, as Atelier asks it.
    """
    rows = (photo_rows("train", seed, unseen=False) if families & {"photo", "album"} else []) + synth_rows("train", seed, unseen=False, every_job=buttons, none=buttons)
    by_image: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        if r["family"] in families:
            by_image.setdefault(r["image"], {}).setdefault(r["kind"], []).append(r)
    rng = random.Random(seed)
    out = [{"image": image, "rows": [rng.choice(v) for v in kinds.values()]} for image, kinds in by_image.items()]
    rng.shuffle(out)
    return out


# Texts one update reads at most, apart from the anchors: what the GPU's memory takes with 14 to 16 pictures.
MAX_TEXTS = 112


class Readers(torch.nn.Module):
    """The few numbers that turn a match into a loss; vision/calibrate.py fits its own after training."""

    def __init__(self) -> None:
        super().__init__()
        self.log_t = torch.nn.Parameter(torch.tensor(math.log(1.3)))
        self.k = torch.nn.Parameter(torch.tensor(0.5))
        self.c = torch.nn.Parameter(torch.tensor(4.0))


def losses(model: Towers, readers: Readers, batch: list[dict], pictures: Pictures, tok, base_photo: dict[str, np.ndarray], anchors: list[tuple[str, np.ndarray]],
           dev: str, keep_photo: float, keep_text: float, half: bool) -> tuple[torch.Tensor, dict[str, float]]:
    px = torch.from_numpy(np.stack([pictures(b["image"]) for b in batch])).to(dev)
    wanted: dict[str, int] = {}
    asked: list[list[dict]] = []
    for b in batch:
        asked.append([])
        for r in b["rows"]:
            new = {text for text in texts(r["q"]) if text is not None and text not in wanted}
            if len(wanted) + len(new) > MAX_TEXTS and len(new) > 8:
                continue  # a question with dozens of options, in a batch that already has its share: left for another pass
            for text in texts(r["q"]):
                if text is not None:
                    wanted.setdefault(text, len(wanted))
            asked[-1].append(r)
    n_q = len(wanted)
    for text, _ in anchors:
        wanted.setdefault(text, len(wanted))
    ids = torch.tensor([encode(tok, text)[0] for text in wanted], device=dev)
    # The towers at 16 bits (half the memory, and faster on Apple's GPU); the matching and the losses at 32.
    with torch.autocast(device_type=dev, dtype=torch.float16, enabled=half):
        v = model.image(px)
        t = model.text(ids)
    v, t = v.float(), t.float()
    z = model.scale * (v @ t.T) + model.bias
    choice, noul = [], []
    for i, rows in enumerate(asked):
        for r in rows:
            at = [None if text is None else wanted[text] for text in texts(r["q"])]
            truth = labels(r["q"]).index(r["answer"])
            if r["kind"] == "noul":
                y = torch.tensor(float(truth), device=dev)
                loss = F.binary_cross_entropy_with_logits(readers.k * z[i, at[1]] + readers.c, y)
                if at[0] is not None:  # with a "false" text: also, which of the two fits better
                    loss = 0.5 * loss + 0.5 * F.cross_entropy((readers.k * z[i, [at[0], at[1]]])[None], torch.tensor([truth], device=dev))
                noul.append(loss)
            else:
                choice.append(F.cross_entropy((z[i, at] / readers.log_t.exp())[None], torch.tensor([truth], device=dev)))
    zero = torch.zeros((), device=dev)
    l_choice = torch.stack(choice).mean() if choice else zero
    l_noul = torch.stack(noul).mean() if noul else zero
    photo_at = [i for i, b in enumerate(batch) if b["image"] in base_photo]
    if photo_at:
        target = torch.from_numpy(np.stack([base_photo[batch[i]["image"]] for i in photo_at])).to(dev)
        l_photo = (1 - (v[photo_at] * F.normalize(target, dim=-1)).sum(-1)).mean()
    else:
        l_photo = zero
    if anchors:
        target = torch.from_numpy(np.stack([a for _, a in anchors])).to(dev)
        l_text = (1 - (t[[wanted[text] for text, _ in anchors]] * F.normalize(target, dim=-1)).sum(-1)).mean()
    else:
        l_text = zero
    total = l_choice + l_noul + keep_photo * l_photo + keep_text * l_text
    return total, {"choice": l_choice.item(), "noul": l_noul.item(), "photo": l_photo.item(), "text": l_text.item(), "texts": n_q}


@torch.no_grad()
def evaluate(model: Towers, readers: Readers, rows: list[dict], pictures: Pictures, tok, dev: str) -> dict:
    """A quick look on held-out questions (half a, the half calibration also uses): right, by family and split."""
    model.eval()
    images = sorted({r["image"] for r in rows})
    vec = {}
    for i in range(0, len(images), 48):
        chunk = images[i : i + 48]
        out = model.image(torch.from_numpy(np.stack([pictures(k) for k in chunk])).to(dev)).float().cpu().numpy()
        vec.update(zip(chunk, out))
    wanted = sorted({text for r in rows for text in texts(r["q"]) if text})
    words = dict(zip(wanted, embed_texts(model, tok, wanted, dev)))
    cal = {"choice": {"t": float(readers.log_t.exp())}, "score": {"t": float(readers.log_t.exp())}, "noul": {"pair": {"a": float(readers.k), "b": 0.0, "c": float(readers.c)}, "single": {"a": float(readers.k), "c": float(readers.c)}}}
    groups: dict[str, list[int]] = {}
    sure: dict[str, list[int]] = {}
    for r in rows:
        if r["answer"] is None:
            continue
        cos = [None if text is None else float(vec[r["image"]] @ words[text]) for text in texts(r["q"])]
        p = probabilities(r["kind"], cos, model.scale, model.bias, cal)
        truth = labels(r["q"]).index(r["answer"])
        key = f"{r['family']} / {r['split']} / {r['kind']}"
        groups.setdefault(key, []).append(int(max(range(len(p)), key=lambda i: p[i]) == truth))
        v = verdict(r["kind"], p)
        if v is not None:
            sure.setdefault(key, []).append(int(v == truth))
    model.train()
    return {k: {"n": len(v), "right": round(sum(v) / len(v), 4), "decides": round(len(sure.get(k, [])) / len(v), 4),
                "right_when_deciding": round(sum(sure[k]) / len(sure[k]), 4) if sure.get(k) else None} for k, v in sorted(groups.items())}


def show(metrics: dict) -> None:
    for key, m in metrics.items():
        rwd = "-" if m["right_when_deciding"] is None else f"{m['right_when_deciding']:.0%}"
        print(f"  {key:<40} n={m['n']:<5} right {m['right']:.1%} · decides {m['decides']:.0%} · right when deciding {rwd}", flush=True)


def save(out: Path, model: Towers, readers: Readers, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.adapter_state(), "readers": {k: float(v) for k, v in readers.state_dict().items()}, "meta": {**meta, "lora": model.lora}}, out / "vision.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--init", type=Path, help="a run to go on from (its vision.pt)")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--last-vision", type=int, default=6, help="LoRA on this many of the image tower's last blocks (0: all 12)")
    parser.add_argument("--last-text", type=int, default=6, help="LoRA on this many of the text tower's last blocks (0: all 12)")
    parser.add_argument("--batch", type=int, default=16, help="pictures per update")
    parser.add_argument("--anchors", type=int, default=32, help="captions per update that the text tower must keep reading as the base does")
    parser.add_argument("--full-precision", action="store_true", help="run the towers at 32 bits while training (more memory)")
    parser.add_argument("--anchor-pool", type=int, default=40000, help="how many of the captions to draw them from")
    parser.add_argument("--keep-photo", type=float, default=4.0)
    parser.add_argument("--keep-text", type=float, default=4.0)
    parser.add_argument("--families", nargs="+", default=["photo", "album", "control", "document", "page"])
    parser.add_argument("--buttons", action="store_true", help="also train on buttons that are none of the jobs, and on a button's job among every job at once")
    parser.add_argument("--first-epoch", type=int, default=0, help="the pass to start counting from, so a run that goes on words its questions anew")
    parser.add_argument("--vocab", type=Path, help="a trimmed vocabulary (vision/vocab.py build): train with its tokenizer and with its rows of the embedding table as they ship, at 4 bits")
    parser.add_argument("--base-emb", type=Path, default=ROOT / "runs" / "vision" / "emb" / "base")
    parser.add_argument("--eval-rows", type=int, default=2400)
    parser.add_argument("--limit", type=int, default=0, help="updates (0: all), for a quick look")
    parser.add_argument("--min-free-gb", type=float, default=1.0)
    parser.add_argument("--max-swap-growth-mb", type=float, default=512)
    args = parser.parse_args()
    torch.manual_seed(2026)
    dev = device()
    tok = tokenizer()
    pictures = Pictures()
    families = set(args.families)

    model = Towers.load().to(dev)
    # What the base says, before anything changes: every training photo's vector, and the anchor captions' vectors.
    ids, vectors = load(args.base_emb, "photos")
    base_photo = {f"photos/{k}": v for k, v in zip(ids, vectors)}
    anchor_texts = sorted({json.loads(line)["text"] for line in (CACHE / "coco35-dev.jsonl").read_text(encoding="utf-8").splitlines()})
    anchor_texts = sorted(random.Random(0).sample(anchor_texts, min(len(anchor_texts), args.anchor_pool)))
    cache = args.base_emb / "anchors.npz"
    if cache.exists() and len(np.load(cache)["texts"]) == len(anchor_texts):
        anchor_vectors = np.load(cache)["vectors"].astype(np.float32)
    else:
        anchor_vectors = embed_texts(model, tok, anchor_texts, dev)
        np.savez(cache, texts=np.array(anchor_texts), vectors=anchor_vectors.astype(np.float16))
    print(f"{len(base_photo)} photo vectors and {len(anchor_texts)} anchor captions from the base", flush=True)
    if args.vocab:
        # From here on the text tower reads as it will when shipped: the trimmed tokenizer, and the kept rows of the
        # embedding table as the 4-bit file gives them. What the base said (above) stays the exact base.
        from export import shipped_table
        from vocab import Trimmed

        keep = json.loads((args.vocab / "vocab.json").read_text(encoding="utf-8"))["keep"]
        table = model.words.embeddings.token_embedding.weight
        rounded = shipped_table(table.data[keep])
        moved = 1 - F.cosine_similarity(table.data[keep].cpu().float(), rounded, dim=-1).mean().item()
        table.data[keep] = rounded.to(table.device, table.dtype)
        full = tok
        tok = Trimmed(args.vocab)
        # Captions the shipped tokenizer can't read (other scripts fall apart into bytes, and the runtime declines them)
        # are no longer text to hold in place: the tower could only be bent towards them, not keep them.
        from format import clean
        from runtime import unreadable

        byte_ids = {i for t, i in full.get_vocab().items() if t.startswith("<0x") and t.endswith(">") and len(t) == 6}
        readable = [i for i, t in enumerate(anchor_texts) if not unreadable(clean(t), tok.encode(clean(t)).ids, byte_ids)]
        dropped = len(anchor_texts) - len(readable)
        anchor_texts = [anchor_texts[i] for i in readable]
        anchor_vectors = anchor_vectors[readable]
        print(f"the text tower reads through {args.vocab.name}: {len(keep)} tokens, their rows at 4 bits (1 - cos to the exact rows {moved:.4f}); "
              f"{len(anchor_texts)} anchor captions it can read, {dropped} in other scripts left out", flush=True)

    readers = Readers().to(dev)
    if args.init:
        blob = torch.load(args.init / "vision.pt", map_location="cpu", weights_only=True)
        n = model.add_adapters(**blob["meta"]["lora"])
        model.load_state_dict(blob["state"], strict=False)
        readers.load_state_dict({k: torch.tensor(v) for k, v in blob["readers"].items()})
        print(f"from {args.init} ({blob['meta'].get('updates')} updates)", flush=True)
    else:
        n = model.add_adapters(args.rank, 2.0 * args.rank, args.last_vision or None, args.last_text or None)
    model.to(dev).train()
    trained = [p for name, p in model.named_parameters() if name.endswith((".a", ".b"))]
    for p in trained:
        p.requires_grad_(True)
    print(f"LoRA on {n} linear layers, {sum(p.numel() for p in trained) / 1e6:.2f}M trained parameters, on {dev}", flush=True)

    rng = random.Random(2026)
    eval_file = CACHE / "eval.jsonl"
    held = [json.loads(line) for line in eval_file.read_text(encoding="utf-8").splitlines()]
    held = [r for r in held if r["half"] == "a" and r["family"] in families]
    rng.shuffle(held)
    held = held[: args.eval_rows]

    first = epoch_examples(args.first_epoch, families, args.buttons)
    per_epoch = len(first) // args.batch
    total = args.limit or int(per_epoch * args.epochs)
    opt = torch.optim.AdamW([{"params": trained, "lr": args.lr, "weight_decay": 0.01}, {"params": readers.parameters(), "lr": args.lr * 10, "weight_decay": 0.0}])
    warm = max(1, total // 30)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.05, 1 - s / total))
    free0, swap0 = pressure()
    print(f"{len(first)} training pictures, {total} updates of {args.batch} ({per_epoch} per epoch); free disk {free0:.1f} GB, swap in use {swap0:.0f} MB", flush=True)
    print("before training:")
    show(evaluate(model, readers, held, pictures, tok, dev))

    step, epoch, t0, run = 0, args.first_epoch, time.time(), {}
    best = -1.0
    while step < total:
        examples = first if epoch == args.first_epoch else epoch_examples(epoch, families, args.buttons)
        epoch += 1
        for i in range(0, len(examples) - args.batch + 1, args.batch):
            batch = examples[i : i + args.batch]
            picked = rng.sample(range(len(anchor_texts)), args.anchors) if args.anchors else []
            loss, parts = losses(model, readers, batch, pictures, tok, base_photo, [(anchor_texts[j], anchor_vectors[j]) for j in picked], dev, args.keep_photo, args.keep_text,
                                 half=not args.full_precision and dev != "cpu")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trained, 1.0)
            opt.step()
            sched.step()
            step += 1
            del loss
            if dev == "mps":
                # Every batch has another number of texts, and Apple's allocator keeps a block for each size it has seen.
                torch.mps.empty_cache()
            for k, val in parts.items():
                run[k] = 0.98 * run[k] + 0.02 * val if k in run else val
            if step % 10 == 0:
                free, swap = pressure()
                if free < args.min_free_gb or swap - swap0 > args.max_swap_growth_mb:
                    save(args.out, model, readers, {"updates": step, "stopped": f"free disk {free:.1f} GB, swap +{swap - swap0:.0f} MB", "args": {k: str(v) for k, v in vars(args).items()}})
                    print(f"stopped at update {step}: free disk {free:.1f} GB, swap grew {swap - swap0:.0f} MB (saved)", flush=True)
                    return
            if step % 25 == 0:
                rate = step / (time.time() - t0)
                held_gb = torch.mps.driver_allocated_memory() / 1e9 if dev == "mps" else 0.0
                print(f"update {step}/{total} choice {run['choice']:.3f} noul {run['noul']:.3f} · photo drift {run['photo']:.4f} text drift {run['text']:.4f} · {run['texts']:.0f} texts · "
                      f"{rate * args.batch:.1f} pictures/s · {held_gb:.1f} GB · {(total - step) / rate / 60:.0f} min left", flush=True)
            if step % max(1, total // 4) == 0 or step >= total:
                metrics = evaluate(model, readers, held, pictures, tok, dev)
                print(f"after {step} updates:")
                show(metrics)
                score = sum(m["right"] * m["n"] for m in metrics.values()) / max(1, sum(m["n"] for m in metrics.values()))
                if score > best:
                    best = score
                    save(args.out, model, readers, {"updates": step, "metrics": metrics, "args": {k: str(v) for k, v in vars(args).items()}})
                    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                    print(f"  saved (right {score:.1%} of {len(held)} held-out questions, half a)", flush=True)
            if step >= total:
                break
    print(f"done in {(time.time() - t0) / 60:.0f} min; best {best:.1%}")


if __name__ == "__main__":
    main()
