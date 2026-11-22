# atelier v0.4: nano, then mini

Atelier's own decision model, fourth version, saved 2026-11-22. It is two models that act as one decider:

1. **atelier-nano v0.3** (54 MB, unchanged) answers first, in milliseconds.
2. **atelier-mini** (new, 515 MB) gets every question the nano declines (open questions, text it can't read) or is
   unsure about by Atelier's own cutoffs.
3. **The rule** decides what the mini is unsure about too.

On Atelier's own decisions, 72% of questions never leave the nano.

| | atelier-nano v0.3 (`model/`) | **atelier-mini** (`mini/`) |
|---|---|---|
| Architecture | mmBERT-small, 22 layers, option-marker head | Qwen3.5-0.8B (24 blocks: 18 Gated DeltaNet, 6 attention) read the decider's way: non-causal attention, a 255-code readout at the `<|fim_suffix|>` marker |
| Training | flash labels (v0.3 data) | LoRA rank 64 on all 186 linear layers. Distilled from flash on 150k questions, then 40k more mixed with numeric questions answered by code |
| Shipped file | `model.emb4.int8.onnx`, 54.2 MB | `mini.gguf`, 515 MB (Q5_K_M layers, 4-bit embeddings), plus `readout.f32` (1 MB) |
| Runtime | ONNX Runtime (`runtime/node`) | llama.cpp's llama-server (`runtime/node/src/mini.ts`); the runner tokenizes, sends ids and applies the readout |
| Per question, Ryzen 5 7600X | about 10 ms | about 0.2 s (6 threads) |
| Memory, measured | 370 MB for the Node runner (nano and the mini's tokenizer) | 0.8 GB for llama-server (4096-token batch) |

`mini/` also holds the LoRA adapters and readout (`mini.pt`), calibration, the training logs (150k run, then the
numeric run), `decision.json` (protocol, temperatures), the tokenizer and `parity.jsonl`. The token ids that
pplx_decider makes match the Node runner on 608 of 608 questions.

## Atelier's own eval (`eval/atelier-eval-cascade-v0.3+mini-num.txt`; trained without any eval case)

| Decision | Rules alone | nano v0.3 | mini alone | **nano, then mini** |
|---|---|---|---|---|
| Is a reply waiting on you | 5/10 | 10/10 (10/10 sure) | 10/10 (9/9 sure) | 10/10 (10/10 sure) |
| Voice into a field | 4/8 | 6/8 (2/2 sure) | 6/8 (2/2 sure) | 6/8 (2/2 sure) |
| The companion's clicks | 5/12 | 12/12 (8/8 sure) | 12/12 (8/8 sure) | 12/12 (8/8 sure) |
| Search or ask | 5/10 | 10/10 (10/10 sure) | 10/10 (10/10 sure) | 10/10 (10/10 sure) |
| Who voice is for | 3/6 | 6/6 (3/3 sure) | 6/6 (3/3 sure) | 6/6 (3/3 sure) |
| Effort per message | 3/12 | 12/12 (12/12 sure) | 9/12 (8/8 sure) | 12/12 (12/12 sure) |
| Worth remembering | 5/10 | 10/10 (5/5 sure) | 10/10 (5/5 sure) | 10/10 (5/5 sure) |
| Tab groups | 0/12 | 10/12 (10/10 sure) | 10/12 (10/10 sure) | 10/12 (10/10 sure) |
| Which skill fits | 2/10 | 8/10 (8/10 sure) | 9/10 (9/10 sure) | 8/10 (8/10 sure) |
| Labeling 12 emails (label_items) | | declined | topic 9/9, today 11/11 sure | topic 9/9, today 11/11 sure |
| Tests passed | | 10 of 10 | 10 of 10 | 10 of 10 |

Every sure answer is right except in skills. There the nano is sure "no skill fits" on two messages where one
does; the mini alone misses a different one. `--mini-first skill` sends skills straight to the mini: 9/10, at
1.26 s per message instead of 83 ms (`eval/atelier-eval-cascade-v0.3+mini-num-skills.txt`).

## What it can do now (`eval/cascade-v0.3+mini-num.json`, `eval/mini-0.8b-num.md`, `eval/all-versions.md`)

| Test | nano v0.3 alone (as shipped) | **nano, then mini** |
|---|---|---|
| Atelier's decisions, held out: decides / right when deciding | 72% / 95.0% | **76% / 93.9%** (28% reach the mini) |
| New kinds of questions about new text | declined | **76% / 98.2%** |
| Held-out browser question families | declined (37% / 66% if answered) | **44% / 82.6%** |
| A second set of held-out browser families | declined (29% / 62% if answered) | **39% / 83.9%** |
| Numbers, against the true answer (`eval/numeric-*.json`) | declined (49% right if answered) | **50% / 89.5%** |
| 20 unseen languages | 14% / 72% (non-Latin declined) | **26% / 77%** |
| 18-28 tab groups | 47% / 76% | 53% / 74% |
| Label emails, routine conditions, packs | declined | answered |

What the numeric run added: comparing numbers. Before it, the mini answered by the kind of question ("is the price
at or below the budget" got "no" whatever the price). It was right 54% of the time when it decided, while the
teacher was right 87%. Now:

- **Families it trained on:** right 93% when it decides.
- **Five numeric families it never saw:** right 80%.
- **Single comparisons:** 98-100%, including the unseen "delivery radius" family.
- **Everything else unchanged:** no other test got worse.

## What it can't do

- **Arithmetic in several steps:** adding up a cart is right 63% of the time; an age from a birth date 37%; a
  discount percentage 59%. It can't pick the cheapest pack per unit, so it declines.
- **Currency conversion:** right 78% but sure, so 22% sure-wrong, like the teacher. Never let it decide a budget
  across currencies on its own.
- **New browser families:** about one decision in six is wrong (82-84% right), and it is sure and contradicts the
  teacher on 4% of questions.
  - Weakest, and security-relevant: it is sure and wrong on 4 of 10 "does this message contain credentials" cases,
    3 of 10 "is this summary faithful" and 2 of 10 "is there a prompt injection in this tool output". It must never
    be the only guard against leaked credentials or injection.
  - The two teachers agree with each other on 89% of verdicts here, so part of this is the question, not the model.
- **Many tab groups:** with 18-28 groups it decides about half the time.
- **Hindi and Thai:** it decides 12-13% (right 78% and 70%).
- **Size and speed:** the mini is about 20 times slower than the nano and 10 times larger.
  - On phones, the nano stays the first answer. The mini runs through llama.cpp's Android and iOS builds, but that
    path is ported and untested, and its speed there (about a second per question) is an estimate, not a
    measurement.
  - Packaging the mini in Atelier is not done: `integration/` still ships the nano only (`docs/integration.md`).

## Found along the way

- **More of the same data made the mini surer, not more general.** 150k questions instead of 40k improved Atelier's
  held-out decisions (agreement 81.9% to 85.2%). On browser families it never saw, it was no better at equal
  coverage. Variety carries over; volume doesn't.
- **The tokenizer.** transformers 5.17 builds Qwen3.5's tokenizer with the older Qwen2 split rule.
  - The teachers ship that rule in their own files, so their labels are right.
  - The mini's base was pretrained with the newer rule, and reads Thai and Hindi better with it (Thai right when
    deciding 40% to 70%).
  - All mini scripts and runtimes now use the base's rule (`student/mini_tokenizer.py`). This mini was trained with
    the older one; the next training run should use the new rule throughout.
- **Quantization** (`eval/mini-quant-mini-l24.json`):
  - Q5_K_M layers with 4-bit embeddings give the 8-bit file's verdict on 97.9% of 3,099 questions, deciding and
    being right as often.
  - 4 bits everywhere (469 MB) decides 1-2 points less often.
  - The tied 248k-token embedding table is 136 MB at 4 bits; trimming the vocabulary is the next size lever.

## Next

- **Numbers:** more numeric families, with the multi-step ones labelled by code (sums, percentages, dates).
- **Variety:** more Fable-written browser families, the lever that carries over.
- **Retraining:** a from-scratch run with the base's tokenizer rule.
- **Integration:** an optional download of the mini in Atelier's patch (llama-server per platform, started on first
  use like the nano).
