# atelier-nano v0.2

Atelier's own decision model, second version. Saved 2026-11-22. The first to pass all ten of Atelier's
decision tests (v0.1 passed nine).

| | v0.1 | **v0.2** |
|---|---|---|
| Architecture | mmBERT-small, 12 of 22 layers | mmBERT-small, **all 22 layers**, hidden 384, the same option-marker head |
| Parameters | 40M | 59M (42.3M body, 16.8M embeddings) |
| Shipped file (`model.emb4.int8.onnx`) | 34.4 MB | **54.2 MB** (int8 layers, 4-bit embedding table) |
| Per question, Ryzen 5 7600X | 5.3 ms (2 threads), 6.9 ms (1) | 10.1 ms (2 threads), 12.4 ms (1); loads in about 0.9 s |
| Training data | 6 pools (Atelier's decisions) | the same 6 pools **plus 15.9k open decisions** (gliclass, synthetic zero-shot tasks, and 1,920 browser situations written by Claude Fable 5.2 in 192 question families) |
| Teacher | pplx-decider-flash-v1.3 | same |

Vocabulary, format, guard (declines text under 70% Latin or over 10% byte tokens) and the default to decline
questions that aren't Atelier's own are unchanged; the runtime and the Atelier patch take this file as they are.

## Atelier's own eval (`eval/atelier-eval-v0.2-l22.txt`; trained without any eval case)

| Decision | Rules alone | v0.1 | **v0.2** | flash teacher |
|---|---|---|---|---|
| Is a reply waiting on you | 5/10 | 10/10 | 10/10 (10/10 sure) | 10/10 |
| Voice into a field | 4/8 | 7/8 (3/3 sure) | 6/8 (2/2 sure) | 8/8 |
| The companion's clicks | 5/12 | 12/12 | 12/12 (8/8 sure) | 12/12 |
| Search or ask | 5/10 | 10/10 | 10/10 (10/10 sure) | 10/10 |
| Who voice is for | 3/6 | 5/6 | **6/6** (3/3 sure) | 6/6 |
| Effort per message | 3/12 | 9/12 (7/7 sure) | **11/12 (11/11 sure)** | 11/12 |
| Worth remembering | 5/10 | 10/10 | 10/10 (5/5 sure) | 10/10 |
| Tab groups | 0/12 | 9/12 (9/10 sure) | **10/12 (10/10 sure)** | 11/12 |
| Which skill fits | 2/10 | 7/10 (6/8 sure) | **8/10 (7/8 sure)** | 10/10 |
| Tests passed | | 9 of 10 | **10 of 10** | |

## Measured gaps (`eval/v0.2-l22.md`, `eval/all-versions.md`)

| | v0.1 | v0.2 |
|---|---|---|
| Held out, Atelier's decisions: decides / right when it decides | 67% / 96.5% | 73% / 95.5% |
| New kinds of questions: decides / right | 51% / 81% | 40% / **88%** |
| Held-out browser question families: decides / right | 28% / 45% | 24% / **62%** |
| 20 languages, guarded: contradictions | 0.2% | 0.2% |

So it decides Atelier's own questions like the teacher; on questions of a kind it never saw it is now right 88% of
the time when it decides, but on new kinds of browser questions only 62%: too low to answer packs', routines' and
label_items' questions, which it still declines by default (the rule or Claude decide). That is the gap the next
version works on: a wider student (mmBERT-base) with four times the open-decision data.

Other known gaps: dictation is more often unsure than v0.1 (no wrong answers); skills still misses one sure
case; tabs with 18-28 groups stay mostly unsure; non-Latin languages are declined, not understood.
