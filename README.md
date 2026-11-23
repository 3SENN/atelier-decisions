# Atelier decision model

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](LICENSE)
[![Weights: Apache 2.0](https://img.shields.io/badge/Weights-Apache_2.0-blue.svg)](LICENSES/Apache-2.0.txt)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-3SENN%2Fatelier--decisions-yellow)](https://huggingface.co/3SENN/atelier-decisions)

A decision model built only for [Atelier](https://github.com/3SENN/atelier)'s quick decisions (is a reply waiting
on you, would this click buy or delete something, is this a search or a question, ...): tens of megabytes,
milliseconds on an ordinary CPU, nothing sent anywhere. It answers Atelier's existing System One requests, so it
drops in where a hosted decision model goes today, and the rule-first contract stays: when it is not sure, the
rule decides.

The model is distilled: big decision models are the teachers, and a tiny student learns their answers.

It is part of Atelier, the browser you make your own, and open source like the rest of it:

| Repository | For |
|---|---|
| [atelier](https://github.com/3SENN/atelier) | The browser, which asks these questions and runs these models |
| **atelier-decisions** (this one) | The on-device decision models: their data, training, measurements and every saved version |
| [atelier-apps](https://github.com/3SENN/atelier-apps) | Building apps, games and packs for Atelier |
| [atelier-packs](https://github.com/3SENN/atelier-packs) | The community index of packs |

Want to help? A decision it gets wrong is the most useful thing to report; see [CONTRIBUTING.md](CONTRIBUTING.md).

| | Teacher: pplx-decider-v1.3 | Teacher: pplx-decider-flash-v1.3 | Student: atelier-nano (final) |
|---|---|---|---|
| Base | Qwen3.8-27B (hybrid Gated DeltaNet + attention) | Qwen3.5-4B | mmBERT-small: 12 of 22 layers, 44k-token vocabulary |
| Parameters | 26.1B | 4.5B | 40M (23.2M body, 16.8M embeddings) |
| On disk | 52 GB (BF16) | 9 GB | **34.4 MB** (int8 layers, 4-bit embedding table) + 2.2 MB tokenizer |
| Runs on | 12 GB GPU via layer streaming | 12 GB GPU | any CPU (ONNX Runtime): 5 ms a question on 2 threads, 7 ms on 1 |
| Atelier eval, right when sure | | 8 of 9 decisions at 100%, skills 90% | 7 of 9 decisions at 100%, tabs 90%, skills 75% |

**Final result** (Atelier's own `tests/decisionsEval.test.ts`, through `runtime/node` on the CPU, guard on, trained
without any eval case; `analysis/results/atelier-eval-final-shipped.txt`):

| Decision | Rules alone | atelier-nano, 34 MB, CPU | flash teacher, 4.5B, GPU |
|---|---|---|---|
| Is a reply waiting on you | 5/10 | 10/10 (10/10 sure) | 10/10 |
| Voice into a field | 4/8 | 7/8 (3/3 sure) | 8/8 |
| The companion's clicks | 5/12 | 12/12 (8/8 sure) | 12/12 |
| Search or ask | 5/10 | 10/10 (10/10 sure) | 10/10 |
| Who voice is for | 3/6 | 5/6 (2/2 sure) | 6/6 |
| Effort per message | 3/12 | 9/12 (7/7 sure) | 11/12 |
| Worth remembering | 5/10 | 10/10 (5/5 sure) | 10/10 |
| Tab groups | 0/12 | 9/12 (9/10 sure) | 11/12 |
| Which skill fits | 2/10 | 7/10 (6/8 sure) | 10/10 |
| label_items, routines, packs | | declined (rule or Claude decide) | answered |
| Median answer | | 3-5 ms | 128-234 ms |

On 6,905 held-out teacher-labelled questions: 88.1% the teacher's verdict, 0.5% sure answers that contradict it.

## Versions

Each is saved whole under `releases/<version>` (model, checkpoint, tokenizer, calibration, its reports, the data it
was trained on, `MANIFEST.json` with SHA-256, and a `MODEL_CARD.md` with what it can't do). Git holds what can be
read; the weights and data archives are at the same paths on
[Hugging Face](https://huggingface.co/3SENN/atelier-decisions). `python tools/hf_release.py fetch v0.5` puts them in
place and checks every file against the manifest. The tables above and under "How we got there" describe v0.1
unless they say otherwise.

| Version | Student | Shipped | Atelier tests passed | Atelier's decisions (held out): decides / right | New kinds of browser questions (held-out families): decides / right |
|---|---|---|---|---|---|
| v0.1 | mmBERT-small, 12 layers | 34.4 MB | 9 of 10 | 67% / 96.5% | 28% / 45% |
| v0.2 | 22 layers + 16k open decisions | 54.2 MB | 10 of 10 | 73% / 95.5% | 24% / 62% |
| v0.3 | 22 layers + 480 browser question families | 54.2 MB | 10 of 10 (effort 12/12) | 72% / 95.0% | 37% / 66%; a second held-out set 29% / 62% |
| v0.4 | nano v0.3, then atelier-mini (Qwen3.5-0.8B) for what the nano declines or is unsure of | 54.2 MB + 515 MB | 10 of 10 (also answers label_items) | 76% / 93.9% (28% reach the mini) | 44% / 83%; second set 39% / 84%; numbers 50% / 89.5% |
| v0.5 | nano v0.3, then mini v0.5: retrained from scratch, plus security and multi-step number questions with known answers and 480 more families | 54.2 MB + 515 MB | 10 of 10 | 76% / 93.9% | 44% / 81%; second set 41% / 83%; third set 38% / 82%; numbers 49% / 93%; **security 88% / 93%** |

## atelier-mini: a decider that works out new questions

The nano learns the kinds of questions it is shown. On kinds it never saw, it is often sure and wrong: on held-out
browser question families it is right 66% of the time when it decides, at any cutoff. On numbers it is at chance:
"is €910 at or below €800" gets a coin flip. More data helps only slowly.

atelier-mini (`student/train_mini.py`) is the teacher's own architecture at 0.8B:

- **Base:** Qwen3.5-0.8B (Apache-2.0): 24 blocks (18 Gated DeltaNet, 6 attention), hidden size 1024, a 248k-token vocabulary.
- **Read the decider's way:**
  - attention is non-causal;
  - the readout picks one of 255 answer codes at the `<|fim_suffix|>` marker;
  - the readout starts from the embeddings of the code letters.
- **Training:**
  - LoRA rank 64 on all 186 linear layers (43.6M trainable);
  - KL distillation on the same flash labels as the nano;
  - v0.4: 150k questions, then 40k more mixed with numeric questions answered by code (`data/build_numeric.py`, below);
  - v0.5: from scratch on 196k questions with the base's own tokenizer rule. The mix adds a third set of browser
    families, multi-step numbers answered by code and security questions answered by construction (below);
  - about 12 questions a second on the RTX 3060.

It keeps the base model's language prior, so it reads kinds of questions nobody wrote training data for far better
than the nano does. Numbers needed teaching (below).

Since v0.4 the two run as one decider (`runtime/node`, `analysis/cascade.py`):

1. The nano answers first.
2. The mini takes what the nano declines (open questions, unreadable text) or is unsure about by Atelier's cutoffs.
3. The rule decides the rest.

| | nano v0.3 alone (as shipped) | mini v0.5 alone | **v0.5: nano, then mini** | v0.4: nano, then mini |
|---|---|---|---|---|
| Size | 54 MB | 515 MB (5-bit layers, 4-bit embeddings) | both | both |
| Time per question, CPU | about 10 ms | 0.14-0.33 s (6 threads) | 10 ms for 72% of Atelier's questions, about 0.2 s for the rest | same |
| Atelier tests | 10 of 10 | 10 of 10 | 10 of 10; sure answers right except the nano's two skill misses and one tab group | 10 of 10 |
| Atelier's decisions, held out: decides / right | 72% / 95.0% | 72% / 96.2% | 76% / 93.9% | 76% / 93.9% |
| New kinds of questions | declined | 75% / 97.9% | 75% / 97.9% | 76% / 98.2% |
| Held-out browser families (three sets) | declined (37% / 66% if answered) | 38-44% / 81-83% | 38-44% / 81-83% | 39-44% / 80-84% |
| Numbers, against the true answer | declined (49% right if answered) | 49% / 93.3% | 49% / 93.3% | 50% / 89.5% |
| Numbers in several steps, against the true answer | declined | 14% / 84.1% (2.2% sure-wrong) | same | 19% / 44.1% (10.9%) |
| Security, answers known by construction | declined | 88% / 93.2% (6.0% sure-wrong) | same | 46% / 73.2% (12.4%) |
| 20 unseen languages | 14% / 72% (non-Latin declined) | 22% / 81% | 28% / 75% | 26% / 77% |

Figures come from `analysis/results/cascade-v0.3+mini-v05.json` (v0.4: `cascade-v0.3+mini-num-newsets.json`),
`docs/capabilities/mini-0.8b-v05.md` and `analysis/results/{numeric,numeric2,security}-*.json`; the model card is
`releases/v0.5/MODEL_CARD.md`.

On quantization: the 5-bit file gives the 8-bit file's verdict on 97.9% of 3,099 questions, and decides and is right
as often (`analysis/mini_quant.py`). 4 bits everywhere (469 MB) decides 1-2 points less often.

**Runtime.** `student/gguf_decider.py` marks a llama.cpp GGUF so that llama.cpp computes the decider exactly. It
writes `attention.causal = false` and last-token pooling into the file. llama-server's `/embedding` then returns the
final hidden state at the marker, and `student/serve_mini.py` applies the readout and calibration.

- **Parity:** against PyTorch on 40 questions, bf16 and Q8_0 give the same verdict 100% of the time; the largest
  probability difference is 0.009 (`analysis/mini_parity.py`).
- **Why llama.cpp:** it runs the same file on CPU, Metal, Vulkan, Android and iOS.
- **In Atelier's runner:** `runtime/node/src/mini.ts` runs the mini on llama-server, and `cascade.ts` puts the nano
  in front of it (`docs/integration.md`). Its token ids match Python on 608 of 608 questions, and its answers match
  to 3×10⁻⁷. Memory: 0.8 GB for llama-server, 370 MB for the Node runner.

**What it can't do yet** (the v0.5 mini; `releases/v0.5/MODEL_CARD.md`):

- **Percentages, totals and conversions in several steps:** it leaves them to the rule. It decides 14% of
  multi-step number questions, mostly dates and times.
- **Security, not alone:**
  - It is sure and wrong on 6% of security questions.
  - On whether a sender belongs to the organisation it names, 16.5%.
  - On whether a summary is faithful, 15%.
  - On budgets across currencies, 13%.
  - It must never be the only guard.
- **New browser families:** right 81-83% when it decides; sure and contradicting the teacher on 5-6%.
- **18-28 tab groups:** decided about half the time.
- **Languages:** 20 unseen languages are decided 22% of the time; Hebrew is weakest (10% decided, right 40%).
- **Size and speed:** about 20 times slower than the nano and 10 times larger, hence the cascade.

Pruning to 12 of the 24 blocks broke it: the language prior went with them. It keeps all 24.

**More of the same data makes it surer, not more general.** These figures are before calibration, from each run's
`report.json`:

| | 40k questions | 150k questions |
|---|---|---|
| Atelier's held-out decisions: agree with the teacher | 81.9% | 85.2% |
| Atelier's held-out decisions: right when deciding | 90.8% | 91.8% |
| Held-out browser families: decides / right | 52% / 82% | 59% / 75% |
| Second held-out set: decides / right | 49% / 84% | 55% / 80% |

At equal coverage, the two are about the same on the families. So variety in the kinds of questions, not volume,
is what carries over, the same lesson the nano taught.

**The teacher can compare numbers; the students never learned to.** On the same code-checked numeric questions, flash
decides 90% and is right 87% of the time (`analysis/results/numeric-flash.json`).

- **Single comparisons:** 100% right (a price against a budget, a quantity against a limit, a distance against a
  radius, seats against a party).
- **Several steps:** weaker. It adds up a cart right 70% of the time, gets percentages right 67-71%, an age from a
  birth date 50%, and the cheapest price per unit 28%.

The pools hold almost no numeric questions, so distillation never passed the skill on.

Before numeric training, the mini answered by the kind of question: "is the price at or below the budget" got "no"
whatever the price. It was right 54-58% of the time when it decided; the nano 49%. So the 150k mini continued for
40k questions, a fifth of them numeric (13 families, answered by code), with 5 numeric families held out whole:

| Numbers test (1,280 questions, five languages) | Decides | Right when deciding | Sure and wrong |
|---|---|---|---|
| nano v0.3 | 57% | 49% | 29% |
| mini, 150k questions | 40% | 54% | 18.5% |
| flash (the teacher) | 90% | 87% | 12% |
| **mini after numeric training** | **50%** | **89.5%** | **5.2%** |
| ... on the 5 families it never saw | 36% | 80% | 7% |

- **Single comparisons:** 98-100% right, including the unseen "delivery radius" family.
- **Where it can't compute** (meeting times, opening hours, deadlines): it mostly declines.
- **No other test got worse:** held-out decisions, new kinds of questions, browser families and Atelier's eval are
  the same or better.

**v0.5: teach what the teacher gets wrong with answers that are known.** Two new test sets have answers that don't
come from a teacher. Both have training families and families held out whole:

- **Multi-step numbers** (`data/build_numeric2.py`): 25 families answered by code. The teacher decides 60% of them
  and is right on only 73% of those (16% sure-wrong).
- **Security** (`data/build_security.py`): 8 families answered by construction. Fable wrote the building blocks;
  the secrets, lookalike domains and altered summaries are planted, so the answer follows from how the question
  was built. The teacher is right on 95%.

The mini was retrained from scratch with both, and with gen4's 408 trained browser families:

| | Decides | Right when deciding | Sure and wrong |
|---|---|---|---|
| Security: v0.4 mini | 46% | 73% | 12.4% |
| Security: **v0.5 mini** | **88%** | **93%** | 6.0% |
| ... the 3 families it never saw | 87% | 91% | 7.7% |
| Multi-step numbers: v0.4 mini | 19% | 44% | 10.9% |
| Multi-step numbers: **v0.5 mini** | 14% | **84%** | **2.2%** |
| ... the 5 families it never saw | 4% | 57% | 1.5% |

- **Security:** about level with the teacher, except two families.
  - Lookalike senders: 16.5% sure-wrong; it takes the real name under someone else's domain for the real one.
  - Faithful summaries: 15% sure-wrong.
- **One pass can count days, not multiply.**
  - It learned date and time arithmetic: business days, warranty windows, earliest arrivals, reaching a shop
    before it closes. There it decides 65-88% and is right 80-96%, better than the teacher.
  - On percentages, VAT, tips, scaled recipes and split bills it never decides, even with 1,000 examples per family.
  - It is not sure there, so it doesn't guess: the rule decides. The teacher guesses and is wrong a quarter of the
    time.
- **More families didn't make it more general.** gen4's 72 held-out families are 82% right (v0.4: 80%); the two
  older sets are unchanged.
  - Three sets of Fable families in, unseen families sit at 81-84% right when deciding.
  - The jump from the nano (140M, 66%) to the mini (0.8B, 83%) on the same data points at size. A 2B mini on
    v0.5's data is being trained to test that.

## Layout

| Path | What |
|---|---|
| `teacher/server.py` | The decider as a System One server on 127.0.0.1, logging every decision (the "collect" step) |
| `teacher/stream.py` | Runs a decider far larger than the GPU (the 26B on 12 GB) by streaming layers from disk, and measures what each layer does to the decision |
| `teacher/label.py` | Labels a request pool with a decider that fits on the GPU (resumable) |
| `data/atelier_uses.py` | Atelier's own questions, ported verbatim from `atelier/src/main/decisions/uses.ts` |
| `data/check_uses.py` | Checks that port byte for byte against requests Atelier really sent |
| `data/build_pool.py`, `data/lexicon.py` | The unlabelled request pool: MASSIVE, WildChat, Mind2Web and multilingual templates |
| `data/build_targeted.py`, `data/build_skills_massive.py`, `data/build_v5.py`, `data/build_v6.py` | Pools aimed at what the student got wrong: skills by meaning (MASSIVE intents, glaive functions), facts in passing (personas), and LLM-written natural inputs (`data/gen`, by Fable) |
| `data/build_probe_open.py` | Probes of questions and text the student never saw: open questions, 20 languages, many tab groups |
| `student/format.py` | The student's input format (the contract with the runtime) |
| `student/model.py`, `student/train.py` | The student and its distillation |
| `student/trim_tokenizer.py` | A smaller BPE tokenizer (merge-closed, byte fallback: nothing becomes `<unk>`) |
| `student/calibrate.py` | Per-decision temperatures so Atelier's cutoffs read the student like the teacher |
| `student/export.py` | ONNX, int8, parity and CPU latency |
| `student/serve.py` | The student as a System One server (PyTorch) |
| `student/eval_onnx.py`, `student/quantize_mixed.py` | What int8 and a 4-bit embedding table cost, in Atelier's verdicts |
| `integration/atelier-on-device-decisions.patch`, `docs/integration.md` | The built-in "On this computer" provider for Atelier (utility process, settings, packaging, tests) |
| `docs/research.md` | What the field offers (teachers, compression, distillation, runtimes, data), with sources |
| `runtime/node` | The on-device runner: TypeScript, `onnxruntime-node`, the pure-JS tokenizer; System One in and out |
| `data/build_open.py`, `data/build_gen3.py` | Open decisions (gliclass, zero-shot tasks) and browser question families written by Fable (`data/gen2`, `data/gen3`), with whole families held out as tests of new kinds of questions |
| `analysis/capabilities.py` | Every version's capability report: Atelier's eval, held out per decision, and probes it never trained on (`docs/capabilities/`) |
| `analysis/selective.py` | Whether a stricter cutoff makes it right more often on new kinds of questions (fitted on one held-out set, measured on another) |
| `student/train_mini.py` | atelier-mini: the teacher's own architecture at 0.8B (Qwen3.5-0.8B read the decider's way), LoRA-distilled on the same labels |
| `student/calibrate_mini.py`, `student/merge_mini.py` | The mini's per-decision temperatures; LoRA merged into the base, the readout and `decision.json` beside it |
| `student/gguf_decider.py`, `student/serve_mini.py` | A llama.cpp GGUF marked to run the decider exactly (non-causal, last-token pooling); the mini as a System One server on llama-server |
| `analysis/mini_parity.py`, `analysis/mini_quant.py` | llama.cpp against PyTorch, per quantization; which quantization keeps the verdicts |
| `data/build_numeric.py`, `analysis/numeric.py` | Numeric decisions (budgets, totals, dates, hours, sizes, limits, unit prices, counts) in five languages, answered by arithmetic, not a teacher; every model and teacher scored against the true answer |
| `analysis/cascade.py` | nano first, the mini for what it declines, the rule last: how right, and how often the mini is needed |
| `student/export_mini.py`, `student/mini_tokenizer.py` | The mini as a runtime folder (GGUF, readout, protocol, tokenizer, parity fixtures); its tokenizer with the base's own split rule |
| `runtime/node/src/mini.ts`, `cascade.ts`, `readers.ts`, `mini_parity.ts` | The mini on llama-server from Node, the cascade, Atelier's readers in TypeScript, token-id parity with Python |
| `analysis/tokenizer_effect.py`, `tools/purge_mark_scores.py` | What the tokenizer's split rule does to Hindi and Thai; re-scoring those after a tokenizer change |
| `data/build_numeric2.py` | Numeric decisions in several steps (sums, percentages, VAT, units, currencies both ways, business days, journeys past midnight, connections, recipes, averages, plans): 20 trained families and 5 held out, answered by code; rows that depend on a convention are dropped |
| `data/security_src/`, `data/build_security.py` | Security decisions answered by construction: building blocks written by Fable (carriers, secrets with typed placeholders, decoys, injections, imperatives for people, faithful and altered summaries), assembled with realistic random secrets and lookalike domains; three families held out |
| `data/existing_families.txt`, `data/gen4/`, `data/build_gen4.py` | The 720 family names Fable was told not to repeat; 480 more Fable-written browser families (shopping and forms, work tools, everyday life in 20 languages, the agent's own judgement); 15% held out as a third test |
| `tools/mini_dist.py` | The mini as Atelier's optional download: the folder to host (model, llama-server per platform, its licence) and the manifest with every file's size and SHA-256 that Atelier compiles in |
| `student/vocab_trim.py` | A mini GGUF with only the vocabulary its data uses (dropped tokens spelled with kept ones; `vocab.json` beside it for the runners): 80 MB smaller, measured and not shipped (`analysis/results/vocab-trim-mini-l24-num.txt`) |
| `analysis/` | Tensor census, teacher comparison, vocabulary coverage; results in `analysis/results/` |
| `releases/<version>`, `tools/make_release.py` | Every saved version, and the script that saves one: its text here, its weights and data archives on Hugging Face |
| `tools/hf_release.py`, `hf/README.md` | A version's files from Hugging Face (`fetch`), checked against its manifest (`verify`); publishing a version, the model card and the mini's download (maintainers) |
| `tools/check_headers.py` | Every source file carries the MPL-2.0 notice (`--fix` adds it) |
| `third_party/pplx_decider` | Perplexity's inference code (Apache-2.0), vendored for the teachers |

## How we got there

**What the 26B is made of** (`analysis/tensor_census.py`, headers only):

| Group | Params | Share | Needed for Atelier's text decisions |
|---|---:|---:|---|
| SwiGLU MLPs (64 layers) | 17.1B | 65.6% | yes |
| Gated DeltaNet mixers (48 layers) | 5.6B | 21.3% | yes |
| Full-attention mixers (16 layers, non-causal) | 1.7B | 6.4% | yes |
| Token embeddings (248k vocabulary) | 1.27B | 4.9% | only the rows Atelier's text uses |
| Vision tower | 0.46B | 1.8% | no |
| Readout head `[255 x 5120]` | 1.3M | 0.0% | yes (a few rows) |

Dropping vision and unused vocabulary saves 6.7% of the 26B and 21% of flash, but the compute path stays 24.4B
(11.3 GiB at 4 bits) and 3.6B (1.7 GiB): pruning and quantizing the decider alone can't reach tens of
megabytes. Distillation into a student 100-1000x smaller is the way; the compressed decider is a faster teacher.

**Where the decision forms** (`teacher/stream.py`, readout applied to every layer's output on Atelier's eval
requests): in both models the answer is unreadable for most of the depth, then forms in a few layers around an
attention layer, and the last layers barely change it.

| | Readable from | 98-99% of final top answers | Small-influence tail |
|---|---|---|---|
| 26B (64 layers) | layer 52 (92%) | layers 56-60 | 57-59, 61-62 (1-cos < 0.01) |
| flash (32 layers) | layer 21 (84%) | layer 24 | 25-31 |

So flash's last ~8 layers are a cheap depth cut before any healing, and the 26B's middle (layers 3-27, block
influence 0.003-0.02) is the classic ShortGPT pruning target.

**Toward a general decider (v0.2 work, `docs/capabilities/README.md`)**. Coverage = how often it decides;
precision = how often it is right when it does. On Atelier's own decisions every version decides about 70% and is
right 96% of the time. On kinds of questions it never saw, the picture by student:

| Student | Size | New kinds of questions: decides / right | Held-out browser families: decides / right |
|---|---|---|---|
| v0.1 (12 layers, Atelier data) | 34 MB | 51% / 81% | 28% / 45% |
| 22 layers, same data | 54 MB | 51% / 83% | 22% / 50% |
| 12 layers + 16k open decisions | 34 MB | 16% / 80% | 17% / 59% |
| **v0.2**: 22 layers + 16k open decisions | 54 MB | 40% / **88%** | 24% / **62%** |
| mmBERT-base 12 layers + 57k open decisions | 83 MB | 40% / 87% | 19% / 62% |
| **v0.3**: 22 layers + 480 browser families (x3) | 54 MB | **64% / 90%** | **37% / 66%** (2nd set 29% / 62%) |
| atelier-mini (Qwen3.5-0.8B), 40k questions | 812 MB | 86% / 95% | 46% / 83% (2nd set 43% / 86%) |
| **atelier-mini v0.4**: 150k questions, then numeric | 515 MB | 76% / **98%** | 44% / 83% (2nd set 39% / 84%) |
| **atelier-mini v0.5**: from scratch, + gen4, multi-step numbers, security | 515 MB | 75% / 98% | 44% / 81% (2nd set 41% / 83%, 3rd 38% / 82%) |

Depth helps Atelier's decisions (v0.2 passes all ten tests); width (mmBERT-base) adds nothing at 50% more size.
Within kinds of questions it has seen plenty of, the student is already good (98.6% right when deciding on held-out
synthetic zero-shot tasks); on new kinds of browser questions it is not, and the teachers agree with each other
on those (flash vs 26B: 89% of verdicts, 1.9% conflicts), so that is a real gap, not teacher noise.

Variety helps, slowly: four times the browser question families (v0.3) moved precision on families it never saw from
62% to 66%. And a stricter cutoff does not rescue it (`analysis/selective.py`, `analysis/results/selective-*.json`):
on held-out families its precision stays at 60-69% at every temperature, because where it is wrong it is sure. On
balanced yes/no questions v0.2 contradicted the teacher on 32 of the 91 it decided; the worst families need
arithmetic or a comparison (is the total within budget, does the rating meet the minimum). Part of the 88-90% on
"new kinds of questions" is base rate (the teacher answers "no" to 85% of them). A 59M encoder learns the kinds of
questions it is shown; it does not work out new ones. The next test is a student with a general language prior:
atelier-mini (`student/train_mini.py`), the teacher's own architecture at 0.8B.

**The two teachers on a broad sample** (`analysis/results/flash-vs-26b-subset.txt`: 2,640 pool requests, 3,425
questions, labelled by both): top answer 91.3%, verdict 84.9%, sure verdicts that contradict each other 0.6%.
They are near identical on skills (99.5% verdict), needsYou (93%), clicks and tabs (90%); flash is surer than
the 26B on dictation (71% vs 54% sure), memory (64% vs 51%) and search or ask (80% vs 71%), so a student taught
by flash is bolder there than the strongest teacher would be; and on effort the two contradict each other 5.4% of
the time, a ceiling for any student and a sign the question itself is ambiguous. Next iteration: 26B labels (or
the two teachers averaged) for dictation, memory and search or ask, and sharper wording for effort in uses.ts.

**Which tensors resist quantization** (`analysis/weight_stats.py`: kurtosis, outliers and plain 4-bit rounding
error for every weight): in both models the MLPs (66% of the 26B) are the most benign (kurtosis < 1, 12% error
at 4 bits). The Gated DeltaNet gate projections `in_proj_a/b` are the hardest (15-18% error in the 26B, kurtosis
up to 11, worst in layers 58-61, where the decision settles) but tiny, so they can stay at 8-16 bits for free. The
output projections `o_proj`/`out_proj` carry outlier columns (kurtosis 10-13, max 58-63 sigma in the 26B), the
case for GPTQ/AWQ, rotations, or 8 bits.

**Teachers agree** (`analysis/compare_teachers.py`, Atelier's eval requests): flash vs 26B top answer 98.6%,
Atelier verdict (sure answer or "unsure") 97.7%, and **no conflicts**: where they differ, one is sure and the
other unsure. Flash labels the bulk pool; the 26B checks a subset.

**The local teacher already beats hosted models on Atelier's eval** (`analysis/results/atelier-eval-flash-v1.3.txt`,
Atelier's own `tests/decisionsEval.test.ts` against `teacher/server.py`): rules alone vs rules with flash taking
over when sure, at 128-234 ms median on an RTX 3060:

| Decision | Rules alone | With flash (local) |
|---|---|---|
| Is a reply waiting on you | 5/10 | 10/10 |
| Voice into a field | 4/8 | 8/8 |
| The companion's clicks | 5/12 | 12/12 |
| Search or ask | 5/10 | 10/10 |
| Who voice is for | 3/6 | 6/6 |
| Effort per message | 3/12 | 11/12 |
| Worth remembering | 5/10 | 10/10 |
| Tab groups | 0/12 | 11/12 |
| Which skill fits | 2/10 | 10/10 (9/10 right when sure) |
| Labeling 12 emails | | 12/12 topic, 11/11 today |

**The student on Atelier's eval** (8 layers, 43.7k vocabulary, int8 ONNX, 32.8 MB, through `runtime/node` on the
CPU; `analysis/results/atelier-eval-nano-*.txt`). Rules alone / with the model taking over where sure:

| Decision | Rules alone | Student v1 (pool v1) | Student v13 (+ targeted pool) | Flash (GPU) |
|---|---|---|---|---|
| Is a reply waiting on you | 5/10 | 10/10 | 10/10 | 10/10 |
| Voice into a field | 4/8 | 8/8 | 6/8 | 8/8 |
| The companion's clicks | 5/12 | 11/12 (7/7 sure) | 11/12 (7/7 sure) | 12/12 |
| Search or ask | 5/10 | 10/10 | 9/10 | 10/10 |
| Who voice is for | 3/6 | 5/6 | 5/6 | 6/6 |
| Effort per message | 3/12 | 4/12 | 5/12 (5/5 sure) | 11/12 |
| Worth remembering | 5/10 | 8/10 (5/7 sure: fails) | 9/10 (5/6 sure) | 10/10 |
| Tab groups | 0/12 | 10/12 | 10/12 | 11/12 |
| Which skill fits | 2/10 | 2/10 (fails) | 3/10 (fails) | 10/10 |
| Median answer time | | 3-7 ms | 3-7 ms | 128-234 ms |

Caveat on this table: the first pool's templates reused phrases from Atelier's eval fixture (tab titles,
dictated sentences, a few queries), so 24 of the 103 eval states, and sentences in 3,408 pool requests, were
seen in training (with the teacher's labels, not the fixture's). Tabs, dictation and voice above are therefore
optimistic. Since then `student/train.py` (and calibration and `eval_onnx.py`) leave out every request that
shares a state, or a free-text field of four words or more, with an eval case; the final students are trained
without them.

Every miss is a coverage problem in the training pool, found by probing probabilities: the student is sure and
right wherever the pool had varied examples (held-out agreement with the teacher: 93.5-94.2% top answer, 89-90%
verdict, 0.3% confident conflicts). Skills needs thousands of distinct request/capability pairs, not the few
hundred templates gave (pool v4: MASSIVE's 60 intents written as skills); effort and memory needed facts and
multi-step tasks in passing and in every language (pool v3).

**Student size** (same data, held-out verdict agreement with the teacher; shipped file with int8 layers and a
4-bit embedding table; one CPU thread at 64 tokens): 4 layers 89.4%, 18.6 MB, 3.0 ms; 8 layers 90.6%, 26.5 MB,
5.9 ms; 12 layers 91.4%, 34.4 MB, 9.2 ms. Twelve is the default (most accurate and, on Atelier's eval, the fewest
sure-but-wrong answers); four is the option for low-end phones.

**Where it should not answer, and doesn't.** Two probes the student never trained on decide its limits:

- *Languages* (`analysis/languages.py`, `analysis/results/languages-l12.txt`; 20 MASSIVE locales, unseen
  utterances): its sure answers contradict the teacher in 0-0.7% of the trained languages and 0.7-1.3% of
  Italian, Portuguese, Swedish and Indonesian, which it never saw, so those are answered. In Arabic it would
  contradict 6.7% and overreach 15%, in Russian 2.7%: the runtime declines text that is under 70% Latin script
  or over 10% raw-byte tokens, which takes every non-Latin language and most Polish and Turkish to 0% conflicts.
- *Open questions* (`data/build_probe_open.py`, `analysis/results/open-questions-l12.txt`; new questions about new
  text): 6% conflicts overall and 36% for "does this involve money?", against under 1% on Atelier's own
  decisions. The runtime therefore declines questions that are not Atelier's (`open: 'decline'`, the default):
  label_items, routine conditions and packs go to the rule or a larger model.

A declined question is answered `refusal`, which Atelier already reads as an even spread: the rule decides.

**Vocabulary** (`student/trim_tokenizer.py`): Atelier's text uses ~109k of mmBERT's 256k tokens. A merge-closed
tokenizer of 43.7k tokens round-trips 100% of the text in every language, costs +3% length in Dutch, +4% in
German/French/Spanish, +11% in English (inflated by code in WildChat), and shrinks the embedding table from 98 MB
to 17 MB (int8) and tokenizer.json from 17.5 MB to 2.2 MB.

**Runtime** (`student/export.py`, untrained 8-layer student, Ryzen 5 7600X, ONNX Runtime, 4 threads): 1.6 ms at
16 tokens, 4.6 ms at 64, 8.9 ms at 128 (int8). The student's inputs average 60-65 tokens, because Atelier's
fixed questions become a one-token tag instead of their 100-250-token wording. The Node runner matches Python
exactly (0 token-id mismatches, 0 probability difference over 348 questions) and answers a request in a 3.1 ms
median.

## Reproduce

Python 3.12 (`uv venv --python 3.12 .venv`), torch 2.14 (CUDA 13), transformers 5.17, `triton-windows`,
`flash-linear-attention` + `fla-core` 0.5.2 (without them the Gated DeltaNet layers fall back to a slow, memory-
hungry PyTorch path): `uv pip install -r requirements.txt` after torch for your machine.

```bash
# A saved version's weights and data, from Hugging Face (instead of training it)
python tools/hf_release.py fetch v0.5

# The teachers and the students' bases
hf download perplexity-ai/pplx-decider-flash-v1.3 --local-dir models/pplx-decider-flash-v1.3 --exclude "locate-span/*"
hf download perplexity-ai/pplx-decider-v1.3 --local-dir models/pplx-decider-v1.3 --exclude "locate-span/*"
hf download jhu-clsp/mmBERT-small --local-dir models/mmBERT-small
python tools/verify_manifest.py models/pplx-decider-v1.3

# Measure a teacher with Atelier's own eval (run vitest from an upper-case C:\ path on Windows)
python teacher/server.py --checkpoint models/pplx-decider-flash-v1.3 --port 8009 --log data/teacher/eval-flash.jsonl
ATELIER_DECISIONS_EVAL=custom ATELIER_DECISIONS_URL=http://127.0.0.1:8009 npx vitest run tests/decisionsEval.test.ts --silent=false

# Pools, labels, student (one GPU job at a time)
python data/build_pool.py --out data/pool/pool-v1.jsonl --seed 1          # and v2 (--seed 2 --exclude ...)
python data/build_targeted.py ...; python data/build_skills_massive.py ...; python data/build_v5.py ...; python data/build_v6.py ...
python teacher/label.py --checkpoint models/pplx-decider-flash-v1.3 --pool data/pool/pool-vN.jsonl --out data/teacher/pool-vN.flash.jsonl
python student/trim_tokenizer.py --size 32000 --corpus data/pool/pool-v1.jsonl data/pool/pool-v2.jsonl --out runs/tokenizer-32k
echo '{"pool-v6.flash": 2}' > runs/DATA_WEIGHTS.json
python student/train.py --data data/teacher/pool-v{1..6}.flash.jsonl --extra-eval data/teacher/eval-26b.jsonl     --tokenizer runs/tokenizer-32k --layers 12 --out runs/nano-final --epochs 6   # leaves out every eval case
python student/calibrate.py --model runs/nano-final --data data/teacher/pool-v{1..6}.flash.jsonl
python student/export.py --model runs/nano-final --out runs/nano-final/onnx && python student/quantize_mixed.py runs/nano-final/onnx
python student/eval_onnx.py --model runs/nano-final --data ...; python analysis/languages.py --model runs/nano-final
cd runtime/node && npm install && node --experimental-strip-types src/server.ts ../../runs/nano-final/onnx 8011

# atelier-mini (llama.cpp b12620 binaries in models/llama.cpp-b12620/bin, its source in .../src)
hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
python data/build_numeric.py --out data/pool/pool-numeric.jsonl --test data/pool/numeric-test.jsonl --truth-dir data/teacher
python student/train_mini.py --data data/teacher/pool-*.flash.jsonl --extra-eval data/teacher/eval-26b.jsonl \
    --out runs/mini-l24-150k --questions 150000 --checkpointing   # pools: v1-v6, open-generic, open2, gen2train, gen3
python student/train_mini.py --data data/teacher/pool-*.flash.jsonl data/teacher/pool-numeric.truth.jsonl \
    --weight pool-numeric.truth=6 --init runs/mini-l24-150k/mini.pt --out runs/mini-l24-num --questions 40000 \
    --lr 1e-4 --readout-lr 5e-4 --checkpointing
python student/calibrate_mini.py --model runs/mini-l24-num --data data/teacher/pool-*.flash.jsonl
python student/merge_mini.py --model runs/mini-l24-num --out runs/mini-l24-num/merged
python models/llama.cpp-b12620/src/convert_hf_to_gguf.py runs/mini-l24-num/merged --outtype bf16 --outfile runs/mini-l24-num/mini-bf16.gguf
python student/gguf_decider.py runs/mini-l24-num/mini-bf16.gguf runs/mini-l24-num/mini-bf16.decider.gguf
models/llama.cpp-b12620/bin/llama-quantize --token-embedding-type q4_K runs/mini-l24-num/mini-bf16.decider.gguf \
    runs/mini-l24-num/mini-Q5_K_M-e4.decider.gguf Q5_K_M
python student/export_mini.py --model runs/mini-l24-num --gguf runs/mini-l24-num/mini-Q5_K_M-e4.decider.gguf --out runs/mini-l24-num/runtime
cd runtime/node && node --experimental-strip-types src/server.ts ../../runs/nano-v03-l22/onnx 8011 4 \
    --mini ../../runs/mini-l24-num/runtime --llama-server <path to llama-server>      # v0.4: nano, then mini

# Measure: numbers against the true answer, the cascade (capabilities.py --gguf for the full report)
python analysis/numeric.py --name mini --model runs/mini-l24-num --gguf runs/mini-l24-num/mini-Q5_K_M-e4.decider.gguf
python analysis/cascade.py --nano runs/nano-v03-l22 --mini runs/mini-l24-num --gguf runs/mini-l24-num/mini-Q5_K_M-e4.decider.gguf --name v0.4

# v0.5: more data of three kinds, then the mini from scratch with the base's tokenizer rule
python data/build_numeric2.py --out data/pool/pool-numeric2.jsonl --test data/pool/numeric2-test.jsonl --truth-dir data/teacher --per-family 1000
python data/build_security.py --out data/pool/pool-security.jsonl --test data/pool/security-test.jsonl --truth-dir data/teacher
python data/build_gen4.py --out data/pool/pool-gen4.jsonl --heldout data/pool/gen4-heldout.jsonl
python teacher/label.py --checkpoint models/pplx-decider-flash-v1.3 --pool data/pool/pool-gen4.jsonl --out data/teacher/pool-gen4.flash.jsonl
python teacher/label.py --checkpoint models/pplx-decider-flash-v1.3 --pool data/pool/gen4-heldout.jsonl --out data/teacher/gen4-heldout.flash.jsonl
#   (teacher/label.py resumes from an existing --out by row count: delete it first when its pool changed)
python student/train_mini.py --data data/teacher/pool-*.flash.jsonl data/teacher/pool-numeric.truth.jsonl \
    data/teacher/pool-numeric2.truth.jsonl data/teacher/pool-security.truth.jsonl \
    --weight pool-gen4.flash=3 pool-numeric.truth=2 pool-numeric2.truth=2 pool-security.truth=2 \
    --extra-eval data/teacher/eval-26b.jsonl --out runs/mini-v05 --questions 200000 --checkpointing
python student/calibrate_mini.py --model runs/mini-v05 --data data/teacher/pool-*.flash.jsonl
#   then merge, convert, mark, quantize and export as above (runs/mini-v05/mini-v05-Q5_K_M-e4.decider.gguf)
for s in numeric numeric2 security; do python analysis/numeric.py --suite $s --name mini-v05 --model runs/mini-v05 \
    --gguf runs/mini-v05/mini-v05-Q5_K_M-e4.decider.gguf; done
python analysis/cascade.py --nano runs/nano-v03-l22 --mini runs/mini-v05 --gguf runs/mini-v05/mini-v05-Q5_K_M-e4.decider.gguf --name v0.5

# The mini as Atelier's download (docs/integration.md), and the vocabulary trim that was measured and not shipped
python tools/mini_dist.py --runtime runs/mini-v05/runtime --version v0.5 --out runs/mini-dist-v05 \
    --server win32-x64=models/llama.cpp-b12620/bin --licence models/llama.cpp-b12620/src/LICENSE [--base https://<host>/]
python student/vocab_trim.py <gguf> <trimmed gguf> --data data/pool/pool-*.jsonl --runtime <a copy of the runtime folder>
```

Notes from the machine every version was made on (Windows, an RTX 3060 with 12 GB): one GPU job at a time (two
processes time-slice so badly on Windows that the labeller stalled), and Windows lets CUDA spill into shared system
memory instead of failing, which makes an over-sized batch crawl rather than crash.

## Licences

- **Code:** the [Mozilla Public License 2.0](LICENSE), like Atelier.
- **Model weights** (every version's nano and mini): the [Apache License 2.0](LICENSES/Apache-2.0.txt). The students'
  bases are mmBERT-small (nano), MIT, and Qwen3.5-0.8B (mini), Apache-2.0; the teachers they learned from are
  Perplexity's pplx-decider-v1.3 and pplx-decider-flash-v1.3, Apache-2.0, run locally.
- **Data written for this project** (`data/gen*`, `data/gen4_src`, `data/security_src` and the family lists in
  `data/`): [CC BY 4.0](LICENSES/CC-BY-4.0.txt).
- **Data from other datasets** keeps its own licence in the pools and labels of each version's `data/` archives:
  MASSIVE, Mind2Web and Synthetic-Persona-Chat (CC BY 4.0), WildChat-1M (ODC-BY), glaive-function-calling-v2,
  gliclass-v2.0 and synthetic_zeroshot_mixtral_v0.1 (Apache-2.0).
- **Third-party code:** `third_party/pplx_decider` (Perplexity, Apache-2.0); llama.cpp, MIT, runs the mini.

[NOTICE](NOTICE) lists every source with its licence and what was changed. The Atelier name and logo are trademarks;
see Atelier's [trademark policy](https://github.com/3SENN/atelier/blob/main/TRADEMARKS.md).

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up, how this project measures, and what a good change looks
like. Report security problems privately: [SECURITY.md](SECURITY.md). Everyone taking part follows the
[Code of Conduct](CODE_OF_CONDUCT.md).
