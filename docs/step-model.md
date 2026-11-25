# atelier-step: a small model for the steps of a browser task

The first goal of this project was a decision model small enough to run on any device. atelier-step serves that goal
for one job: driving a web page through a task. "Order the Trailrunner 2 in size 43", "fill in this form", "find me
this": in each, someone has to decide at every page what to do next. A language model (or a person) sets the task up
once (docs/processes.md); atelier-step takes it page by page.

## What it answers

One browser step, in one pass:

| Head | Reads | Answers |
|---|---|---|
| operation | the STEP marker | click, type, done (the goal is visibly reached) or blocked (a login wall, a value nobody gave, an error) |
| element | every control's EL marker | which control: the best of the operation's kind (links, buttons and boxes for a click; fields and selects for typing) |
| value | every field's EL marker against every VAL marker | which of the named values goes into each field, or none of them |

Its input is the step as the runtime sees it (`step/format.py`, ported token for token to
`runtime/node/src/step/format.ts`):

- the goal;
- the planner's named values ("postcode: 1234 AB");
- what was done so far;
- the page title;
- the page's controls and short text lines, in page order ("field Postcode: empty", "Added to your cart").

It never writes text. Whatever is typed is one of the values the planner wrote down, and code acts on its answers:

- the click guard checks every click;
- what it isn't sure of goes to an escalation or to the person (`runtime/node/src/process/steps.ts`).

It is atelier-nano's encoder (mmBERT-small, 22 layers, the trimmed 43,723-token vocabulary) with three new heads: 59.6M
parameters. Exported, it is 54.7 MB (int8 layers, a 4-bit embedding table, as atelier-nano ships) or 61 MB (int8
throughout), and a step takes about 0.08-0.12 s on two CPU threads of an M2. The Node runtime's token ids and outputs
match PyTorch exactly on the exported fixtures (`npm run step:parity`).

## Data: answers known by construction

No teacher labels anything. Two sources:

- **Mind2Web** (CC BY 4.0, `step/mind2web.py`): 1,009 tasks on real websites, with every action people took recorded.
  - For each action: the goal, the values the task types, what was done before, and the recorded element among up to
    79 others on the page, in page order.
  - The HTML is dropped as it downloads, so the training split is 66 MB instead of several GB.
  - A quarter of the websites are held out whole for the test.
- **Generated flows** (`step/synth.py`): shop orders, adding to a cart, searching, municipal and contact forms,
  sign-ups, account areas, cookie banners, and three kinds of blocked flow.
  - English, Dutch, German, French and Spanish.
  - Every page of every flow has its right next step, and every empty field its right value.
  - These teach what Mind2Web can't: "done", "blocked", whole forms, and pages in other languages.
  - Two whole kinds of flow (sign-up forms, marketing settings) and one language of every kind are held out.
  - People are generated for every flow (names, streets, postcodes and phones as each country writes them). None of
    them is the lab's person.
  - Values carry the names a planner might write ("zip code", "shoe size", a bare "value 3"), with distracting values
    among them.
  - Stretch goals ("get to the checkout and stop") end with "done" on the page where the stretch is reached. Their
    wording differs from the lab's processes, which test the model on stretches it never saw word for word.

## Training (on this Mac)

```bash
python3.12 -m venv .venv && .venv/bin/pip install torch "transformers==5.17.*" tokenizers onnx onnxruntime
python3 -I step/mind2web.py fetch --out data/cache/m2w-steps.jsonl          # about 40 minutes, 66 MB
python3 -I step/synth.py --out data/cache/synth-steps-v4.jsonl --flows 3000
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25 PYTORCH_ENABLE_MPS_FALLBACK=1
.venv/bin/python -I step/train.py --m2w data/cache/m2w-steps.jsonl --synth data/cache/synth-steps-v4.jsonl \
  --out runs/step-v4 --epochs 1                       # or --init runs/step-v3 --lr 3e-5 --head-lr 3e-4 to go on from one
.venv/bin/python -I step/train.py ... --init runs/step-v4 --out runs/step-v4 --eval-only --eval-limit 100000   # every held-out step
.venv/bin/python -I step/export.py --model runs/step-v4 --out runs/step-v4/onnx --fixtures data/cache/synth-steps-v4.jsonl
.venv/bin/python -I step/calibrate.py --model runs/step-v4 --m2w data/cache/m2w-steps.jsonl \
  --synth data/cache/synth-steps-v4.jsonl --export runs/step-v4/onnx      # the temperatures, into step.json
.venv/bin/python -I tools/make_step_release.py --version step-v0.1 --run runs/step-v4 \
  --data data/cache/m2w-steps.jsonl data/cache/synth-steps-v4.jsonl --results step-v4-heldout.json ...   # releases/step-v0.1
.venv/bin/python -I tools/hf_release.py publish step-v0.1             # its weights and data to Hugging Face, tagged
```

- **The runs:** each version went on from the one before, so the measured model is a chain:
  - v1: 2,232 updates. The run stopped when the battery ran out; this is the checkpoint saved before that.
  - v2, v3 and v4: one epoch each, 2,684 to 3,209 updates, 56-70 minutes each on an M2 (Apple's GPU through PyTorch
    MPS).
- **Memory:** gradient checkpointing and batches of about 3,000 tokens keep it near 2.6 GB.
- **Interlock:** the training loop checks free disk and swap every 10 updates. It saves and stops when the disk
  drops below 1 GB free or swap grows by more than 512 MB.
- **Why the interlock:** a first run without these settings pushed a machine that was already swapping into a full
  disk.

## Results

Every held-out step (`analysis/results/step-v4-heldout.json`). "Sure" uses the runtime's cutoffs: an operation at
0.6; an element or a value at 0.5 and 1.5 times the next one.

| Held out | Steps | Operation | Element | Value | Whole step right |
|---|---|---|---|---|---|
| Mind2Web, 7 websites never seen | 965 | 92% | 59% (among 15-60 controls) | 81% | 53% |
| Generated flows, a language never seen for that kind | 3,955 | 100% | 98% | 98% | 97% |
| Generated flows, two kinds never seen (sign-up forms, email settings) | 4,449 | 82% | 72% | 100% | 65% |

For comparison, atelier-mini (2B, never trained on steps) picked the right element among 30-40 on the same Mind2Web
websites 9% of the time (`analysis/results/steps-mind2web-v0.6-zero-shot.txt`).

**Calibration** (`step/calibrate.py`): one temperature per head, fitted on half of the held-out steps and checked on
the other half (op 5, element 8, value 3; in step.json, applied by the runtime).

| Held out | Element sure, before → after | Right when sure, before → after |
|---|---|---|
| Mind2Web websites | 93% → 28% | 60% → 89% |
| Generated, unseen kinds | 100% → 92% | 72% → 72% |

On real websites it now hands most element choices on (to the escalation or the person) and is right 89% of the
time when it doesn't. On kinds of flow it never saw, calibration does nothing: it is sure and wrong about one
operation in five there.

**The versions** (each went on from the one before; `analysis/results/step-v*-train.txt`):

| Version | What changed | What it fixed, what it showed |
|---|---|---|
| v1 | first training | the value head learned the generator's value names: on the lab's "shoe size" it picked the postcode |
| v2 | value names varied, distracting values, stretch goals | an unseen language 83% → 97% of whole steps; the lab's flows showed the next problems |
| v3 | people generated per flow (six fixed people before, one of them memorised), longer value lists | the memorised postcode in a description field went away; "added to cart" was missed when the person's details were long |
| v4 | the person's details decoupled from the kind of flow; pages repeated with someone else's details | unseen kinds of flow: operation 77% → 82%; the add-to-cart stretch right in the lab |

**In the lab's real pages** (Atelier over CDP, `npm run lab -- flows --step --escalate --approve`; the same three
flows as docs/processes.md):

| Flow | Claude decides every step | atelier-step, Claude only where it isn't sure |
|---|---|---|
| Order the Trailrunner 2 in size 43, stop on the payment page | ✓ 81 s, 9 Claude calls | ✓ 17.8 s, 4 Claude calls, 10 steps by atelier-step |
| Find a 2-person tent under €200, add it to the cart | ✓ 35 s, 4 calls | ✓ 10.5 s, 2 calls, 3 steps |
| Report a missed bin collection (two-page Dutch form), stop before sending | ✓ 75 s, 9 calls | ✓ 12.0 s, 2 calls, 12 steps |

- **Alone** (no escalation; `analysis/results/flows-step-v4-alone.txt`), it adds the shoes to the cart in four
  correct steps. Then:
  - in the cart it reaches for "Remove" (the click guard stops it);
  - on the form it doesn't stop where the stretch says;
  - in the tent search it goes in circles.
- **None of the three ends right on its own.** With Claude for what it isn't sure of, all three do, in a fifth to a
  seventh of the time.
- **A step takes 0.05-0.6 s.** The larger figures include reading the page after a click.
- **Runtime safeguards** (`runtime/node/src/process/steps.ts`), each added after a failure in these runs:
  - It types only into empty fields, as in training. It had typed a product name into "Quantity: 1".
  - The same operation on the same control again within four operations counts as going in circles, and the step
    is handed on.
  - A control's context is sent only where training had it: a repeated label, or a price next to a product. A
    notice's text on "Go to cart" had turned a sure "done" into another "Add to cart".
  - Right after the person refuses a click, Claude decides the next operation.
  - A second "Add to cart" in one run asks the person first.

## Running it

Saved as `releases/step-v0.1` ([MODEL_CARD.md](../releases/step-v0.1/MODEL_CARD.md)): the run step-v4, calibrated. Its
weights and data are on [Hugging Face](https://huggingface.co/3SENN/atelier-decisions/tree/step-v0.1/releases/step-v0.1).

```bash
python tools/hf_release.py fetch step-v0.1 --only model   # the model, checked against MANIFEST.json
cd runtime/node
npm run step:parity       # the runtime's tokens and answers against PyTorch's
npm run start:step        # the decision server with atelier-step: POST /v1/step
npm run lab -- flows --cdp 9555 --step --escalate --approve -v   # the lab's shop and form flows, driven by it
```

`advance` and `fill` steps use it when the server has it:

- **advance:** one call per page gives the operation, the target and the value, used when clearly ahead.
- **fill:** one call per form gives every field's value, after the page's own autofill hints.
- **Escalation:** what it isn't sure of goes to the escalation (Claude) for that one operation. Without an escalation
  it goes to the person.

## What it can't do yet

- **Steer through flows alone.** It is right about every step of the kinds of flow it was trained on, in languages
  it never saw for them. On a kind of flow it never saw, a fifth of its operations are wrong while it is sure. So it
  runs with the click guard, the process's check steps and an escalation, never as the only judge.
- **Real websites, alone.** It picks the right control 59% of the time among 15-60 on a site it never saw. Calibrated,
  it decides 28% of those and is right on 89%.
- **Judgement.** It can't compare prices, sizes or dates: "a 2-person tent under €200" is a check step's job, or the
  escalation's.
- **An option meant by a value in another language.** "Our green bin wasn't emptied" for "Afval niet opgehaald" goes
  to the escalation. When the planner writes the value in the form's language, it is filled directly.
- **Fields outside the form.** A fill step fills every empty field on the page. In one run Claude, asked about the
  checkout form, also put the product name into the header's search box. It was harmless there, but a fill should
  keep to the form's own fields.
- **Training data it hasn't seen yet.** Steps from real runs (the lab can log every request), more kinds of flow, the
  consent-banner and "Continue to payment" kinds of the click question, and the "done" and "blocked" pages of real
  sites.
- **Phones.** It is 55 MB and about 0.08 s a step on two laptop CPU threads. On a phone, expect a few times that; this
  is not measured. The ONNX file runs with ONNX Runtime Mobile, like atelier-nano.
