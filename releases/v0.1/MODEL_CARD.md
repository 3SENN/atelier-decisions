# atelier-nano v0.1

Atelier's own decision model: answers Atelier's quick decisions (System One requests) on the device, without a key,
the network or a GPU. Saved 2026-11-21.

| | |
|---|---|
| Architecture | ModernBERT encoder (mmBERT-small, MIT), 12 of its 22 layers (0, 2, 4, 6, 8, 10, 11, 13, 15, 17, 19, 21: evenly spread), hidden 384, plus a 384-384-1 head scoring each option at a marker token |
| Vocabulary | 43,723 tokens, trimmed from 256k for en/nl/de/fr/es with merge closure and byte fallback (nothing becomes `<unk>`) |
| Parameters | 40M (23.2M body, 16.8M embeddings) |
| Shipped file | `model/model.emb4.int8.onnx`, 34.4 MB: int8 matmuls, 4-bit embedding table (`model.int8.onnx`, 40.7 MB, int8 throughout) |
| Runtime | ONNX Runtime on the CPU (onnxruntime-node 1.31 in Atelier; same file on ORT Mobile); 5.3 ms median per question on 2 threads, 6.9 ms on 1 (Ryzen 5 7600X); loads in about 0.5 s |
| Teacher | perplexity-ai/pplx-decider-flash-v1.3 (4.5B, Apache-2.0), checked against pplx-decider-v1.3 (26B) |
| Training | KL on the teacher's calibrated distributions (+ tau 2), 6 epochs, 182k questions from 6 pools (MASSIVE, WildChat, Mind2Web, glaive function calling, Synthetic-Persona-Chat, templates, and inputs written by Claude Fable 5.2); option order shuffled; nothing that shares a state, a 4+-word text field or a user-provided question with Atelier's eval |
| Calibration | one temperature per decision, chosen with Atelier's costs (`model/calibration.json`) |
| Checkpoint | `model/student.pt` (PyTorch, `student/model.py` `NanoDecider.load`) to continue training from |

## What it does well

Atelier's own decisions, in its own eval (`eval/atelier-eval-final-shipped.txt`; rules alone / with the model):

| Decision | Rules alone | atelier-nano v0.1 | Right when sure |
|---|---|---|---|
| Is a reply waiting on you | 5/10 | 10/10 | 10/10 |
| Voice into a field | 4/8 | 7/8 | 3/3 |
| The companion's clicks | 5/12 | 12/12 | 8/8 |
| Search or ask | 5/10 | 10/10 | 10/10 |
| Who voice is for | 3/6 | 5/6 | 2/2 |
| Effort per message | 3/12 | 9/12 | 7/7 |
| Worth remembering | 5/10 | 10/10 | 5/5 |
| Tab groups | 0/12 | 9/12 | 9/10 |
| Which skill fits | 2/10 | 7/10 | 6/8 (the one test that fails Atelier's 80% bar) |

Held out (6,905 teacher-labelled questions): 88.1% the teacher's verdict, 0.5% sure answers contradicting it.

## What it does not do (by design: it declines, so the rule or Claude decides)

- **Questions that are not Atelier's own** (label_items, routine conditions, packs): declined. On new questions about
  new text its sure answers contradicted the teacher 7.8% of the time (`eval/open-questions-final.txt`).
- **Text it can't read**: under 70% Latin letters or over 10% raw-byte tokens. Answered: en, nl, de, fr, es and, never
  trained, it, pt, sv, id (at most 1.3% contradictions). Declined: ru, el, ar, zh, ja, ko, hi, he, th and most pl/tr
  (`eval/languages-final.txt`).

## Known gaps (v0.1)

- Skills: matching a request to a user's skill by meaning is the weakest decision (7/10).
- Tabs: a tab that belongs to one of the person's own groups can go to the topic instead ("Hotels in Lisbon" →
  travel, not "Lisbon trip").
- Effort: right when sure, but unsure on 5 of 12; the two teachers themselves contradict each other on effort.
- Bolder than the 26B on dictation, memory and search-or-ask (it learned from flash, which is surer than the 26B).
- Many tab groups (18-28 options): no wrong sure answers, but mostly unsure.
- Inputs are cut at 512 tokens (1,536 for "worth remembering", 2,048 for setup).

## Files

`MANIFEST.json` lists every file with its SHA-256. `data/` holds the pools, the teachers' labels and the
LLM-written inputs (zipped); `eval/` every result file the numbers above come from; `integration/` the Atelier patch
(built-in "On this computer" provider). Rebuild or continue: README.md › Reproduce, with `student/train.py
--init releases/v0.1/model` once it exists, or from mmBERT-small as v0.1 was.
