# Atelier decision model

A decision model built only for [Atelier](https://github.com/3SENN/atelier)'s quick decisions (is a reply waiting
on you, would this click buy or delete something, is this a search or a question, ...): tens of megabytes,
milliseconds on an ordinary CPU, nothing sent anywhere. It answers Atelier's existing System One requests, so it
drops in where a hosted decision model goes today, and the rule-first contract stays: when it is not sure, the
rule decides.

The model is distilled: big decision models are the teachers, and a tiny student learns their answers.

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
was trained on, `MANIFEST.json` with SHA-256, and a `MODEL_CARD.md` with what it can't do). The tables above and
under "How we got there" describe v0.1 unless they say otherwise.

| Version | Student | Shipped | Atelier tests passed | Atelier's decisions (held out): decides / right | New kinds of browser questions (held-out families): decides / right |
|---|---|---|---|---|---|
| v0.1 | mmBERT-small, 12 layers | 34.4 MB | 9 of 10 | 67% / 96.5% | 28% / 45% |
| v0.2 | 22 layers + 16k open decisions | 54.2 MB | 10 of 10 | 73% / 95.5% | 24% / 62% |
| v0.3 | 22 layers + 480 browser question families | 54.2 MB | 10 of 10 (effort 12/12) | 72% / 95.0% | 37% / 66%; a second held-out set 29% / 62% |

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
| `analysis/` | Tensor census, teacher comparison, vocabulary coverage; results in `analysis/results/` |
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
hungry PyTorch path).

```bash
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
```

Notes from this machine: one GPU job at a time (two processes time-slice so badly on Windows that the labeller
stalled), and Windows lets CUDA spill into shared system memory instead of failing, which makes an over-sized
batch crawl rather than crash.

## Licences

Teachers: Apache-2.0 (Perplexity; Qwen bases), run locally. Student base: mmBERT-small, MIT. Data: MASSIVE
(CC BY 4.0), WildChat-1M (ODC-BY), Mind2Web (CC BY 4.0).
