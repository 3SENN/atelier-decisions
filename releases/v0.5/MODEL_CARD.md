# atelier v0.5: the mini learns security questions, and leaves sums to the rule

Atelier's own decision model, fifth version, saved 2026-11-22. Same shape as v0.4:

1. **atelier-nano v0.3** (54 MB, unchanged) answers first, in milliseconds.
2. **atelier-mini v0.5** (new, 515 MB) gets what the nano declines or is unsure about.
3. **The rule** decides what the mini is unsure about too.

The mini is retrained from scratch. It is the same size and architecture, with more data of three kinds:

- **Browser question families:** 480 more, written by Fable (shopping and forms, work tools, everyday life in 20
  languages, the agent's own judgement), labelled by the teacher.
- **Multi-step numbers:** 25 families answered by code, because the teacher is wrong on a quarter of them.
- **Security questions:** 8 families whose answers are known by construction.

| | atelier-nano v0.3 (`model/`) | **atelier-mini v0.5** (`mini/`) |
|---|---|---|
| Architecture | mmBERT-small, 22 layers, option-marker head | Qwen3.5-0.8B read the decider's way (non-causal, 255-code readout at the answer marker) |
| Training | flash labels (v0.3 data) | LoRA rank 64, from scratch, on 195,939 questions (4.4 h on an RTX 3060). The tokenizer reads text with its base's own split rule for the first time. Mix: the flash pools, gen4 ×3, numbers ×2, multi-step numbers ×2, security ×2 |
| Shipped file | `model.emb4.int8.onnx`, 54.2 MB | `mini.gguf`, 515 MB (Q5_K_M layers, 4-bit embeddings), `readout.f32` |
| Per question, Ryzen 5 7600X | about 10 ms | 0.14-0.33 s (6 threads; skills 1 s, long prompts) |

**Checks:**
- **Token ids:** the Node runner and Atelier's helper build the same ids as the training code on 608 of 608 questions.
- **Download:** `download/` holds the manifest for Atelier's optional download, with the size and SHA-256 of every
  file. `integration/` has Atelier's side, now pointing at v0.5.

## Atelier's own eval (`eval/atelier-eval-cascade-v0.3+mini-v05.txt`; trained without any eval case)

| Decision | Rules alone | nano v0.3 | mini v0.5 alone | **nano, then mini v0.5** | v0.4 pair |
|---|---|---|---|---|---|
| Is a reply waiting on you | 5/10 | 10/10 (10/10 sure) | 9/10 (8/8 sure) | 10/10 (10/10 sure) | same |
| Voice into a field | 4/8 | 6/8 (2/2 sure) | 6/8 (2/2 sure) | 6/8 (2/2 sure) | same |
| The companion's clicks | 5/12 | 12/12 (8/8 sure) | 12/12 (8/8 sure) | 12/12 (8/8 sure) | same |
| Search or ask | 5/10 | 10/10 (10/10 sure) | 10/10 (10/10 sure) | 10/10 (10/10 sure) | same |
| Who voice is for | 3/6 | 6/6 (3/3 sure) | 6/6 (3/3 sure) | 6/6 (3/3 sure) | same |
| Effort per message | 3/12 | 12/12 (12/12 sure) | 11/12 (11/11 sure) | 12/12 (12/12 sure) | same |
| Worth remembering | 5/10 | 10/10 (5/5 sure) | 10/10 (5/5 sure) | 10/10 (5/5 sure) | same |
| Tab groups | 0/12 | 10/12 (10/10 sure) | 9/12 (9/10 sure) | 10/12 (**10/11 sure**) | 10/12 (10/10 sure) |
| Which skill fits | 2/10 | 8/10 (8/10 sure) | **10/10 (10/10 sure)** | 8/10 (8/10 sure) | same |
| Labeling 12 emails | | declined | topic 10/10, today 6/6 sure | topic 10/10, today 6/6 sure | topic 9/9, today 11/11 |
| Tests passed | | 10 of 10 | 10 of 10 | 10 of 10 | 10 of 10 |

**Different from v0.4:**
- **New sure-wrong answer:** the mini puts "Hotels in Lisbon - Booking.com" in a "travel" group, not the group the
  test expects.
- **Skills:** the mini alone now gets all 10 right. The pair still misses the nano's two, because the nano is sure
  there.
- **Labeling "is it for today":** decided on 6 of 12 emails instead of 11, all of them right.

## What changed (`eval/cascade-v0.3+mini-v05.json`, `eval/cascade-v0.3+mini-num-newsets.json` for v0.4)

Nano, then mini: decides / right when deciding / sure and wrong.

| Test | v0.4 | **v0.5** |
|---|---|---|
| Atelier's decisions, held out | 76% / 93.9% / 0.6% | 76% / 93.9% / 0.5% |
| New kinds of questions about new text | 76% / 98.2% | 75% / 97.9% |
| Held-out browser families (three sets of 72-80 families) | 39-44% / 80-84% / 4-7% | 38-44% / 81-83% / 5-6% |
| 20 languages | 26% / 76.7% | 28% / 74.9% |
| Numbers, single comparisons, against the true answer | 50% / 89.5% / 5.2% | 49% / **93.3%** / 3.3% |
| **Numbers in several steps**, against the true answer (new) | 19% / 44.1% / 10.9% | 14% / **84.1%** / **2.2%** |
| **Security**, answers known by construction (new) | 46% / 73.2% / 12.4% | **88% / 93.2%** / 6.0% |
| Teacher (flash) on the same multi-step and security sets | 60% / 73% / 16.1% and 94% / 95% / 5.1% | |

### Security (`eval/security-*.json`)

Three of the eight families were never trained on. Each is built from sentences Fable wrote; the right answer
follows from how the question was built:

- planted secrets are Luhn-valid cards, keys and tokens of real shapes, JWTs, PEM blocks;
- lookalike domains are letter swaps or the real name under someone else's domain;
- summaries add or drop facts.

| Family | v0.4 | **v0.5** | teacher |
|---|---|---|---|
| A credential in a message | 68% / 80% | 100% / 98% | 95% / 97% |
| A credential in technical text: logs, configs, tool output (unseen) | 66% / 77% | 96% / 96% | 96% / 93% |
| A prompt injection in page content | 66% / 75% | 100% / 100% | 95% / 95% |
| A prompt injection in tool output (unseen) | 34% / 74% | 76% / 97% | 90% / 90% |
| The link goes where its text says | 11% / 89% | 100% / 100% | 98% / 96% |
| The download is a program | 80% / 78% | 100% / 100% | 98% / 95% |
| The sender belongs to the organisation it names (unseen) | 18% / 57% | 88% / 81%, **16.5% sure-wrong** | 92% / 99% |
| The summary is faithful | 55% / 55% | 46% / 68%, **15% sure-wrong** | 95% / 92% |

### Numbers in several steps (`eval/numeric2-*.json`)

There are 25 families (5 never trained on). Each answer is computed by code, and families whose answer depends on
a convention (business days, 1000 or 1024) keep only the cases where every convention agrees. What the mini
learned splits by kind of arithmetic:

- **Dates and times** (deliver by a date in business days, a warranty still valid, an earliest arrival, reaching
  a shop before it closes): decides 65-88%, right 80-96%, better than the teacher.
- **Multiplying and percentages** (VAT, tips, a price increase, scaling a recipe, splitting a bill, storage): it
  never decides. It is not sure there, so it doesn't guess, and the rule decides.
- **Unseen families:** it decides 4%. Of the questions it gets wrong, it is sure on only 1.5%, against the
  teacher's 19%.

## What it can't do

- **Percentages, totals and conversions in several steps:** it leaves them to the rule. Only the hosted model can
  decide those for now.
- **Security, not alone:**
  - It is sure and wrong on 6% of security questions.
  - On whether a sender belongs to the organisation it names, 16.5%: it takes a lookalike for the real one.
  - On whether a summary is faithful, 15%.
  - On budgets across currencies, 13% (was 22%).
  - It must never be the only guard against leaked credentials, injection or phishing.
- **New browser families:** still about one decision in six wrong (81-83% right), and it is sure and contradicts
  the teacher on 5-6% of questions. More families didn't change that (below).
- **Languages:** 20 unseen languages are decided 22% of the time (was 19%), right 81% (was 85%).
  - Hindi: decides 21% (was 12%).
  - Hebrew is weakest: 10% decided, right 40%.
  - With 150 questions per language, a language's figure moves about 8 points by chance.
- **Size and speed:** unchanged from v0.4. The mini runs on phones only through llama.cpp's mobile builds,
  untested.

## Found along the way

- **More families of the same kind didn't make the mini more general.**
  - It trained on 408 of gen4's 480 families, weighted three times.
  - On gen4's 72 held-out families it is right 82% (v0.4: 80%).
  - On the two older held-out sets it is unchanged.
  - Three sets of Fable families in, the mini's right-when-deciding on unseen families sits at 81-84%. That points
    at the model's size, not the data: from the nano (140M) to the mini (0.8B), the same data moved it from 66% to
    83%. A 2B mini on v0.5's data is in training to test this.
- **Data with known answers works where the teacher is weak.**
  - On security the mini moved from far below the teacher to about level.
  - On multi-step numbers it is now careful (2.2% sure-wrong) where the teacher is confidently wrong (16%).
  - Labels by construction or by code are the way to teach what the teacher gets wrong.
- **A single pass can compare and count days, but not multiply.** The mini reads a question once, with no
  steps in between. Date arithmetic fits that; percentages don't, even with 1,000 examples per family.
- **The tokenizer rule, trained with:** this is the first mini trained with its base's own split rule.
  - Hindi is decided almost twice as often.
  - Thai is unchanged (13%, right 74%).
- **A smaller vocabulary, measured and not shipped** (`eval/vocab-trim-mini-l24-num.txt`, `student/vocab_trim.py`):
  - Keeping only the tokens the data uses cuts the mini by 80 MB (515 to 436 MB).
  - The verdict is the same on 99.7% of questions in trained languages.
  - Korean drops from 13% right to 1%, and Arabic and Thai lose much. The tool and both runners support a trimmed
    file, in case a smaller download matters more.

## Next

- **A 2B mini** (Qwen3.5-2B, same data and question plan, in training) to see whether size moves the unseen
  families.
- **The download host for Atelier:** the manifest's `base` is empty until it is chosen. macOS and Linux server
  builds are still to add.
- **Teaching what the teacher gets wrong:** lookalike senders and faithful summaries.
