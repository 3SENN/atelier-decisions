# atelier-nano v0.3

Atelier's own decision model, third version. Saved 2026-11-22. Same architecture and file size as v0.2; trained
on four times as many kinds of browser questions.

| | v0.2 | **v0.3** |
|---|---|---|
| Architecture | mmBERT-small, all 22 layers, hidden 384, option-marker head | same |
| Shipped file (`model.emb4.int8.onnx`) | 54.2 MB | 54.2 MB |
| Per question, Ryzen 5 7600X | about 10 ms | about 10 ms (6-13 ms median per decision in Atelier's eval) |
| Training data | 6 pools + 15.9k open decisions (192 browser question families written by Claude Fable 5.2) | the same + 40.9k more open decisions and **408 new browser question families** (3,264 situations written by Claude Fable 5.2: content, actions, personal data, life admin); browser families weighted x3 |
| Teacher | pplx-decider-flash-v1.3 | same |

Vocabulary, format, guard (declines text under 70% Latin or over 10% byte tokens) and the default to decline
questions that aren't Atelier's own are unchanged; the runtime and the Atelier patch take this file as they are.

## Atelier's own eval (`eval/atelier-eval-v0.3-l22.txt`; trained without any eval case)

| Decision | Rules alone | v0.2 | **v0.3** | flash teacher |
|---|---|---|---|---|
| Is a reply waiting on you | 5/10 | 10/10 (10/10 sure) | 10/10 (10/10 sure) | 10/10 |
| Voice into a field | 4/8 | 6/8 (2/2 sure) | 6/8 (2/2 sure) | 8/8 |
| The companion's clicks | 5/12 | 12/12 (8/8 sure) | 12/12 (8/8 sure) | 12/12 |
| Search or ask | 5/10 | 10/10 (10/10 sure) | 10/10 (10/10 sure) | 10/10 |
| Who voice is for | 3/6 | 6/6 (3/3 sure) | 6/6 (3/3 sure) | 6/6 |
| Effort per message | 3/12 | 11/12 (11/11 sure) | **12/12 (12/12 sure)** | 11/12 |
| Worth remembering | 5/10 | 10/10 (5/5 sure) | 10/10 (5/5 sure) | 10/10 |
| Tab groups | 0/12 | 10/12 (10/10 sure) | 10/12 (10/10 sure) | 11/12 |
| Which skill fits | 2/10 | 8/10 (7/8 sure) | 8/10 (**8/10 sure**: two sure "no skill" misses) | 10/10 |
| Tests passed | | 10 of 10 | 10 of 10 | |

## Measured gaps (`eval/v0.3-l22.md`, `eval/all-versions.md`, `eval/selective-v0.3-l22.json`)

| | v0.2 | v0.3 |
|---|---|---|
| Held out, Atelier's decisions: decides / right when it decides | 73% / 95.5% | 72% / 95.0% |
| New kinds of questions: decides / right | 40% / 88% | **64% / 90%** |
| Held-out browser question families: decides / right | 24% / 62% | **37% / 66%** |
| A second set of 72 held-out browser families (new): decides / right | | 29% / 62% |
| 20 languages, guarded: contradictions | 0.2% | 0.2% |

Four times the kinds of browser questions moved precision on kinds it never saw by about four points. It still
decides those wrongly a third of the time, and (`analysis/selective.py`) a stricter temperature does not help:
where it is wrong it is sure. Its worst questions need arithmetic or comparison (is the total within budget, does
the rating meet the minimum). So v0.3 keeps declining packs', routines' and label_items' questions by default.
On "new kinds of questions" part of its precision is base rate: the teacher answers "no" to most of them.

The next experiment is a different student, not more data for this one: atelier-mini, the teacher's own
architecture at 0.8B (Qwen3.5-0.8B read the decider's way), to test whether a general language prior generalises
to new kinds of browser questions where a 59M encoder does not.

Other known gaps: dictation is often unsure (no wrong answers); skills has two sure misses; tabs with 18-28 groups
stay mostly unsure; non-Latin languages are declined, not understood.
