# atelier v0.6: a 2B mini, and rules applied to what's on the page

Atelier's own decision model, sixth version, saved 2026-11-23. Same shape as v0.4 and v0.5:

1. **atelier-nano v0.3** (54 MB, unchanged) answers first, in milliseconds.
2. **atelier-mini v0.6** (new, 1.3 GB) gets what the nano declines or is unsure about.
3. **The rule** decides what the mini is unsure about too.

The mini is now Qwen3.5-2B instead of Qwen3.5-0.8B: same tokenizer, same 24-block layout, hidden size 2048 instead
of 1024. It was trained in two steps:

- **The 2B on v0.5's data and plan.** Unseen browser families stopped improving with more data at 0.8B; the
  2B was trained to test whether size moves them. It does (below).
- **Then 40k questions of continuation with a new suite: rules applied to the specifics on a page**
  (`data/build_conditions.py`). Booking a gym class, renewing a library loan, entering a lab, flying a drone:
  two to four conditions, a clause that waives one, a clause that blocks regardless, and the answer computed by
  code. The v0.5 mini, the nano and the 2B alone were all near chance there.

| | atelier-nano v0.3 (`model/`) | **atelier-mini v0.6** (`mini/`) |
|---|---|---|
| Architecture | mmBERT-small, 22 layers, option-marker head | Qwen3.5-2B read the decider's way (non-causal, 255-code readout at the answer marker) |
| Training | flash labels (v0.3 data) | LoRA rank 64 (67.8M trainable). From scratch on 195,939 questions, v0.5's mix (7.6 h on an RTX 3060, `base-train.log`), then 37,545 more with the rules suite weighted 6 (`train.log`) |
| Shipped file | `model.emb4.int8.onnx`, 54.2 MB | `mini.gguf`, 1,286 MB (Q5_K_M layers, 4-bit embeddings), `readout.f32` |
| Per question, Ryzen 5 7600X | about 10 ms | 0.33-0.71 s (6 threads; skills 2.1 s, long prompts) |
| Memory | about 0.2 GB | 1.9 GB peak for llama-server at Atelier's settings (2048-token batch) |

**Checks:**
- **Token ids:** the Node runner and Atelier's helper build the same ids as the training code on 608 of 608 questions.
  Atelier's 115 decision tests pass with v0.6's parity set.
- **Download:** `download/` holds the manifest for Atelier's optional download (1,357 MB with the Windows server),
  with the size and SHA-256 of every file. `integration/` has Atelier's side, now pointing at v0.6.

## Atelier's own eval (`eval/atelier-eval-cascade-v0.3+mini-2b-v05-cond.txt`; trained without any eval case)

| Decision | Rules alone | nano v0.3 | mini v0.6 alone | **nano, then mini v0.6** | v0.5 pair |
|---|---|---|---|---|---|
| Is a reply waiting on you | 5/10 | 10/10 (10/10 sure) | **10/10 (10/10 sure)** | 10/10 (10/10 sure) | same |
| Voice into a field | 4/8 | 6/8 (2/2 sure) | 6/8 (2/2 sure) | 6/8 (2/2 sure) | same |
| The companion's clicks | 5/12 | 12/12 (8/8 sure) | 12/12 (8/8 sure) | 12/12 (8/8 sure) | same |
| Search or ask | 5/10 | 10/10 (10/10 sure) | 10/10 (10/10 sure) | 10/10 (10/10 sure) | same |
| Who voice is for | 3/6 | 6/6 (3/3 sure) | 6/6 (3/3 sure) | 6/6 (3/3 sure) | same |
| Effort per message | 3/12 | 12/12 (12/12 sure) | 11/12 (10/10 sure) | 12/12 (12/12 sure) | same |
| Worth remembering | 5/10 | 10/10 (5/5 sure) | 10/10 (5/5 sure) | 10/10 (5/5 sure) | same |
| Tab groups | 0/12 | 10/12 (10/10 sure) | 10/12 (10/10 sure) | 10/12 (**10/10 sure**) | 10/12 (10/11 sure) |
| Which skill fits | 2/10 | 8/10 (8/10 sure) | 10/10 (10/10 sure) | 8/10 (8/10 sure) | same |
| Labeling 12 emails | | declined | topic 12/12, today 11/11 sure | topic 12/12, today 11/11 sure | topic 10/10, today 6/6 |
| Tests passed | | 10 of 10 | 10 of 10 | 10 of 10 | 10 of 10 |

**Different from v0.5:**
- **No sure answer of the mini is wrong** in Atelier's eval. The Lisbon tab that v0.5 put in a "travel" group is
  decided right.
- **Labeling** decides every email's topic and 11 of 12 "is it for today", all right (v0.5: 10 and 6).
- **The pair's only sure-wrong answers** are the nano's two skill misses, as in every version since v0.3.
- **Slower on the mini's path:** where the mini answers, the pair takes 0.40-0.48 s (v0.5: 0.19-0.22 s). Every
  decision stays inside Atelier's own time limits on this machine: voice target 0.42 s of its 0.7 s, tab groups
  0.95 s at the 90th percentile of 2 s. A slower computer leaves more of them to the rule.

## What changed (`eval/cascade-v0.3+mini-2b-v05-cond.json`; v0.5: `eval/cascade-v0.3+mini-v05.json`)

Nano, then mini: decides / right when deciding / sure and wrong. The middle column is the 2B before the rules suite
(`eval/cascade-v0.3+mini-2b-v05.json`).

| Test | v0.5 (0.8B) | 2B, v0.5's data | **v0.6 (2B + rules)** |
|---|---|---|---|
| Atelier's decisions, held out | 76% / 93.9% / 0.5% | 77% / 94.0% / 0.5% | 77% / 93.7% / 0.5% |
| New kinds of questions about new text | 75% / 97.9% / 0.7% | 80% / 98.9% / 0.0% | 78% / 97.7% / 0.0% |
| **Held-out browser families**, three sets of 72-80 families | 38-44% / 81-83% / 5-6% | 52-59% / 85-89% / 4-6% | **50-60% / 87-89% / 3-5%** |
| 20 languages | 28% / 74.9% | 31% / 76.2% | 31% / 76.5% |
| Numbers, single comparisons, against the true answer | 49% / 93.3% / 3.3% | 61% / 91.8% / 5.0% | 60% / 88.6% / 6.8% |
| Numbers in several steps, against the true answer | 14% / 84.1% / 2.2% | 19% / 84.7% / 2.9% | 16% / **90.0% / 1.6%** |
| Security, answers known by construction | 88% / 93.2% / 6.0% | 94% / 94.2% / 5.5% | 93% / 94.2% / 5.4% |
| **Rules applied to a page**, against the true answer (new) | 32% / 53.8% / 14.7% | 67% / 61.9% / 25.6% | **78% / 83.9% / 12.6%** |

The browser-family sets are judged against the teacher; "sure and wrong" there means sure and contradicting it.

### Size moved the unseen families

Three sets of Fable-written browser families had left the 0.8B at 81-83% right when deciding on families it never
saw. The 2B on exactly the same data and plan:

| Held-out set | 0.8B (v0.5): agrees / decides / right | 2B, same data |
|---|---|---|
| First set (80 families) | 44% / 44% / 81% | 58% / 57% / 86% |
| Second set (72) | 41% / 41% / 83% | 54% / 52% / 89% |
| Third set (72) | 38% / 38% / 82% | 55% / 59% / 85% |

It decides more (52-59% of questions instead of 38-44%) and is right more often, so it agrees with the teacher on
13-17 points more of them. The rules suite then left its accuracy where it was (87-89% right; the first set is
decided 51% instead of 57%) and lowered sure contradictions to 3-5%.

### Rules applied to the specifics on a page (`eval/conditions-*.json`, `data/build_conditions.py`)

Sixteen families in five languages (en, nl, de, fr, es), yes/no, answered by code: 12 trained, 4 held out whole
(blood donation, a photo contest entry, a campfire now, a pet in a rented flat). Each question shows the rules and
the specifics (dates, levels, counts, statuses) as a page would, with clutter around them. Margins say how close the
specifics are to a rule's limit; "exception" rows turn on the clause that waives or blocks.

| | Decides | Right when deciding | Sure and wrong | Exception rows, right / sure and wrong |
|---|---|---|---|---|
| nano v0.3 | 58% | 48.7% | 29.7% | 51-52% / 27-28% |
| mini v0.5 (0.8B) | 32% | 53.8% | 14.7% | 28-42% / 22-24% |
| 2B, before the rules suite | 67% | 61.9% | 25.6% | 41-44% / 36-37% |
| 0.8B + the rules suite (not shipped) | 23% | 82.2% | 4.1% | 56-90% / 3% |
| **v0.6: 2B + the rules suite** | **78%** | **83.9%** | 12.6% | 81-96% / 4-16% |
| ... the 12 trained families | 81% | 92.0% | 6.5% | |
| ... the 4 families it never saw | 75% | 74.9% | **18.8%** | |
| flash (the teacher) | 93% | 93.2% | 6.3% | 95-96% / 4-5% |
| ... on the 4 unseen families | 91% | 96.4% | 3.3% | |

- **Size alone doesn't apply rules.** The 2B decided more than the 0.8B but ignored the exception clauses: right
  41-44% on those rows, below a coin flip.
- **Taught with answers by code, it learns the trained kinds well** (92% right, exception rows 96%), and on the
  four unseen kinds it is right 75% of the time (tenant's pet: 92%; a photo contest entry: 53%).
- **The teacher is better on the unseen kinds** (96%). It is worse on some trained ones: on lab access it is sure
  and wrong 30% of the time. In the cases checked by hand it misread a training date's validity and the waiver
  for supervision; the code's answers were right.

**A disclosure:** the rules suite was designed after the family errors (`eval/family-errors-mini-v05.json`) showed
that v0.5's weakest held-out browser questions were yes/no questions about whether something is allowed or valid.
The suite's families were chosen to avoid every held-out browser family's topic. The browser-family gains above come
from size and were measured before the rules suite existed; the rules suite didn't move them.

### Security (`eval/security-*.json`)

| Family | v0.5 | **v0.6** | teacher |
|---|---|---|---|
| A credential in a message | 100% / 98% | 98% / 96% | 95% / 97% |
| A credential in technical text: logs, configs, tool output (unseen) | 96% / 96% | 94% / 96% | 96% / 93% |
| A prompt injection in page content | 100% / 100% | 100% / 100% | 95% / 95% |
| A prompt injection in tool output (unseen) | 76% / 97% | 84% / 96% | 90% / 90% |
| The link goes where its text says | 100% / 100% | 100% / 99% | 98% / 96% |
| The download is a program | 100% / 100% | 100% / 100% | 98% / 95% |
| The sender belongs to the organisation it names (unseen) | 88% / 81%, 16.5% sure-wrong | 94% / 89%, **10% sure-wrong** | 92% / 99% |
| The summary is faithful | 46% / 68%, 15% sure-wrong | 88% / 77%, **20% sure-wrong** | 95% / 92% |

## What it can't do

- **Rules it never trained on, alone:** it is sure and wrong on 19% of those questions (the teacher: 3%). It
  must not be the only judge of whether something is allowed.
- **Security, not alone:**
  - It is sure and wrong on 5.4% of security questions.
  - On whether a summary is faithful, 20%: it now decides most of them (88%, was 46%), and a fifth of those wrong.
  - On whether a sender belongs to the organisation it names, 10%.
  - It must never be the only guard against leaked credentials, injection or phishing.
- **Numbers:**
  - Percentages, totals and conversions in several steps: it leaves them to the rule (decides 16%).
  - **Regression:** on "does the group fit in the free seats", a numeric family it never trained on, it is sure
    and wrong 26% of the time (the 2B before the rules suite: 2%). It reads "can we all book?" as a rule question
    and answers no. Over all single-comparison numbers: 6.8% sure-wrong (v0.5: 3.3%).
- **New browser families:** about one decision in eight wrong (87-89% right); sure and contradicting the teacher
  on 3-5% of questions.
- **Languages:** 20 unseen languages are decided 26% of the time, right 84% (v0.5: 22% and 81%). Hebrew is still
  weakest (17% decided, right 73%; was 10% and 40%). With 150 questions per language, a language's figure moves
  about 8 points by chance.
- **Size and speed:** a model file 2.5 times v0.5's (1.3 GB) and about twice its time per question. On phones it runs only
  through llama.cpp's mobile builds, untested; the 0.8B (v0.5) is the lighter choice there.

## Found along the way

- **Size moved what data no longer could.** Three sets of families in, the 0.8B was flat at 81-83% on unseen
  families; the 2B on the same data is at 85-89% and decides more. From the nano (140M) to the 0.8B to
  the 2B: 66%, 83%, 86-89%.
- **The 0.8B couldn't take the rules suite without losing other skills; the 2B could.** The same continuation on
  the 0.8B (`eval/conditions-mini-v05-cond.json` and the others with that name):
  - It learned the rules partly: decides 23%, right 82%.
  - In its training checks, held-out browser families fell to 72-77% right, and new kinds of questions from 84% to
    73% agreement.
  - It decided fewer single comparisons (39%, was 49%) and was right on fewer security questions (77%, was 82%).
  - The 2B kept all of those.
- **The teacher has the rules skill and distillation didn't pass it on**, the same story as single comparisons in
  v0.4: flash is right on 93% of rules questions and the minis trained on its labels were near chance. Answers by
  code taught it.
- **A new skill can leak into a neighbour.** The rules suite taught "can we...?" questions to check conditions,
  and one unseen numeric family ("can we all book seats?") now gets "no" where the answer is plainly yes.

## Next

- **Rules on kinds never trained on:** more and more varied rule families (the 12 trained ones carry it to 92%,
  the unseen ones sit at 75%), and seat and capacity questions mixed in so the leak above closes.
- **Faithful summaries:** the 2B decides them now; teach them with more altered summaries.
- **The download host for Atelier:** the manifest's `base` is empty until it is chosen. macOS and Linux server
  builds are still to add.
- **A choice for small computers and phones:** the 0.8B (v0.5) as the lighter mini, if download size or speed
  matters more there.
