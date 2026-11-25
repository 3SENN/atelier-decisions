# atelier-step v0.1: a small model for the steps of a browser task

The first saved version of atelier-step, 2026-11-25. "Order the Trailrunner 2 in size 43", "fill in this form",
"find me this": in each, someone has to decide on every page what to do next. A language model (or a person) writes
the task down once as a process; atelier-step takes it page by page, on the device, in about a tenth of a second.

It answers one browser step in one pass:

| Head | Answers |
|---|---|
| operation | click, type, done (the goal is visibly reached) or blocked (a login wall, a value nobody gave, an error) |
| element | which control: the best of the operation's kind (links, buttons and boxes for a click; fields and selects for typing) |
| value | for every field, which of the planner's named values goes in, or none of them |

It never writes text. What it types is always one of the values the planner wrote down, and code acts on its
answers: a click guard checks every click, and what it isn't sure of goes to an escalation (a language model) or to
the person.

| | atelier-step v0.1 (`model/`) |
|---|---|
| Architecture | atelier-nano v0.3's encoder (mmBERT-small, 22 layers, the trimmed 43,723-token vocabulary) with three new heads; 59.6M parameters |
| Training | four runs in a chain on an M2 (one epoch each, 56-70 minutes; v1 to v4 in `eval/step-v*-train.txt`), on Mind2Web's recorded actions and generated flows in five languages, every answer known by construction |
| Shipped file | `model.emb4.int8.onnx`, 54.7 MB (int8 layers, 4-bit embedding table); `model.int8.onnx`, 61.0 MB |
| Per step, M2, 2 CPU threads | about 0.08-0.12 s (ONNX Runtime) |
| To train on from | `step.pt` (PyTorch, fp32) and `config.json`: `step/train.py --init releases/step-v0.1/model` |

**Checks:**
- **Token ids and outputs:** the Node runtime (`runtime/node/src/step/`) builds the same ids and markers as the
  training code on 200 of 200 steps, with the same logits (`parity.jsonl`, `npm run step:parity`).
- **The two files:** the 4-bit embedding file agrees with full precision on every exported fixture
  (`eval/step-v4-export.json`).

## Results on held-out steps (`eval/step-v4-heldout.json`)

"Sure" uses the runtime's cutoffs: an operation at 0.6; an element or a value at 0.5 and 1.5 times the next one.

| Held out | Steps | Operation | Element | Value | Whole step right |
|---|---|---|---|---|---|
| Mind2Web, 7 websites never seen | 965 | 92% | 59% (among 15-60 controls) | 81% | 53% |
| Generated flows, a language never seen for that kind | 3,955 | 100% | 98% | 98% | 97% |
| Generated flows, two kinds never seen (sign-up forms, email settings) | 4,449 | 82% | 72% | 100% | 65% |

For comparison, atelier-mini v0.6 (2B, never trained on steps) picks the right element among 30-40 on the same
Mind2Web websites 9% of the time (`eval/steps-mind2web-v0.6-zero-shot.txt`).

**Calibration** (`step/calibrate.py`): one temperature per head (operation 5, element 8, value 3), fitted on half of
the held-out steps and checked on the other half. They are in `model/step.json`, and the runtime applies them.

| Held out | Element sure, before → after | Right when sure, before → after |
|---|---|---|
| Mind2Web websites | 93% → 28% | 60% → 89% |
| Generated, unseen kinds | 100% → 92% | 72% → 72% |

## In real pages (`eval/flows-step-v4-hybrid.txt`, `eval/flows-step-v4-alone.txt`)

Three tasks in Atelier, the open-source browser, over its debugging port, on the lab's mock shop and municipal form.
The click guard stops buying and deleting clicks for the person to allow or refuse.

| Task | Claude decides every step | atelier-step, Claude only where it isn't sure |
|---|---|---|
| Order the Trailrunner 2 in size 43, stop on the payment page | ✓ 81 s, 9 Claude calls | ✓ 17.8 s, 4 Claude calls, 10 steps by atelier-step |
| Find a 2-person tent under €200, add it to the cart | ✓ 35 s, 4 calls | ✓ 10.5 s, 2 calls, 3 steps |
| Report a missed bin collection (two-page Dutch form), stop before sending | ✓ 75 s, 9 calls | ✓ 12.0 s, 2 calls, 12 steps |

**On its own** (no escalation), it finishes none of the three:
- It adds the shoes to the cart in four correct steps, then reaches for "Remove" in the cart (the click guard stops it).
- On the form it doesn't stop where the task says.
- In the tent search it goes in circles.

With an escalation for what it isn't sure of, all three end right, 3 to 6 times faster than Claude deciding every
step, and with less than half the Claude calls.

## Use

```bash
hf download 3SENN/atelier-decisions --include "releases/step-v0.1/model/*" --local-dir .
```

From a clone of the GitHub repository ([3SENN/atelier-decisions](https://github.com/3SENN/atelier-decisions)),
`python tools/hf_release.py fetch step-v0.1 --only model` does the same and checks every file against
`MANIFEST.json`. Then, in `runtime/node`:

```bash
npm run step:parity       # the runtime's tokens and answers against PyTorch's
npm run start:step        # the decision server with atelier-step: POST /v1/step on 127.0.0.1:8011
```

The input is a step as the runtime sees it (`step/format.py`, `runtime/node/src/step/format.ts`):
- the goal;
- the planner's named values ("postcode: 1234 AB");
- what was done so far;
- the page title;
- the page's controls and short text lines, in page order ("field Postcode: empty", "Added to your cart").

`runtime/node/src/process/` runs whole tasks with it (docs/processes.md, docs/step-model.md).

## What it can't do

- **Steer through tasks alone.** On a kind of flow it never saw, a fifth of its operations are wrong while it is
  sure. Run it with the click guard, the process's check steps and an escalation, never as the only judge.
- **Real websites, alone.** It picks the right control 59% of the time among 15-60 on a site it never saw.
  Calibrated, it decides 28% of those and is right on 89%.
- **Judgement.** It can't compare prices, sizes or dates: "a tent under €200" is a check step's job, or the
  escalation's.
- **An option meant by a value in another language.** "Our green bin wasn't emptied" for "Afval niet opgehaald" goes
  to the escalation. When the planner writes the value in the form's language, it is filled directly.
- **Fields outside the form.** A fill step fills every empty field on the page, including a header's search box.
- **Phones.** Not measured. Expect a few times the laptop's 0.08 s; the ONNX file runs with ONNX Runtime Mobile, like
  atelier-nano.

## Data (`data/steps.zip`)

The steps the last run (v4) trained on and was tested on:
- `m2w-steps.jsonl`: Mind2Web's recorded actions, kept compact (the HTML is dropped as it downloads), with a quarter
  of the websites held out whole. `step/mind2web.py fetch` builds it.
- `synth-steps-v4.jsonl`: generated flows (`step/synth.py`, seeded, so the same command builds the same file). The
  people in it are generated: names, streets, postcodes and phones as each country writes them, none of them real.

Mind2Web (The Ohio State University) is CC BY 4.0; the generated flows are CC BY 4.0 like the project's other data.
The weights are Apache-2.0, built from atelier-nano v0.3 (itself from mmBERT-small, MIT).

## Next

- **Steps from real runs.** The lab can log every request; real pages' "done" and "blocked" are what it misses most.
- **More kinds of flow**, so fewer are unseen: consent banners, "Continue to payment", account areas of real sites.
- **Fill only the form's own fields.**
- **Into Atelier** as an on-device step model next to atelier-nano, with Claude (or the person) for what it isn't
  sure of.
