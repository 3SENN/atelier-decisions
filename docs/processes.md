# Processes: a task written down once, run by the decision model

A decision model answers one bounded question in a fraction of a second. A task in a browser is many such
questions in a row: sort 22 emails, filter 16 laptops, refuse a cookie banner, find the invoice for last month's
order. A **process** writes the task down once as small steps. After that, only the decision model runs it, item by
item and step by step, with no language model in the loop.

A process comes from one of two places:

- **Set up by a language model.** Claude sees the task and the page once and writes the process (one call, 10-28 s).
  It tries the process on what it saw and fixes what doesn't work. The run itself never calls Claude.
- **Ready-made.** A process in `runtime/node/processes/`, written by hand (or saved from a plan) and run as it is.

Whatever the decision model isn't sure of goes to the person, or to a bucket where a wrong answer costs nothing.
That's the same rule-first contract as in Atelier: when it is not sure, it does not decide.

## Try it

```bash
cd runtime/node
npm install
npm start                      # the decision server: nano, then the 2B mini on the GPU (Metal on a Mac), port 8011
open http://127.0.0.1:8011/lab # the process lab (the playground for single questions is at /)
```

`npm start` needs v0.6's files (`python tools/hf_release.py fetch v0.6`, or the nano from v0.5: they are the same
file) and `llama-server` on the PATH (`brew install llama.cpp`). `npm run start:mini-0.8b` uses v0.5's 0.8B mini
instead, which is about half the time per question.

In the lab:

1. Pick a task on a mock page.
2. Choose the ready-made process or "Set up by Claude".
3. Press Run.

Every item shows each step: what was asked, the probabilities, who answered (nano, mini or code), how long it took,
and where the item went. Against the answers a person would give, the tiles show how much was decided and how much
of that was right.

The same on the command line:

```bash
npm run lab -- list                       # the tasks and the ready-made processes
npm run lab -- run inbox:triage -v        # a ready-made process, every step and its probabilities
npm run lab -- plan inbox:triage          # Claude sets it up (saved in lab/plans/)
npm run lab -- run inbox:triage --planned
npm run lab -- bench --planned -v         # every task, both ways, against what a person expects
npm run lab -- check                      # every process and plan can run; prices read as people write them
```

**The planner** is Claude through the Claude Code CLI on this computer (your Claude subscription). It runs with no
tools, no MCP servers and no settings (`--tools "" --strict-mcp-config --setting-sources ""`), so a plan costs
4,500-7,500 tokens in and 900-2,800 out. Without those flags the CLI would send its own context, about 126,000
tokens, with every plan. With `ANTHROPIC_API_KEY` set, the planner uses the Messages API instead (structured outputs,
`claude-opus-5-5`, effort medium, server-side fallback on). `ATELIER_PLANNER=api|claude` and
`ATELIER_PLANNER_MODEL` choose explicitly.

### In a real browser

Any Chromium with a debugging port works, Atelier included. Use a profile of its own, never the one with your
accounts:

```bash
cd ../atelier && npx electron-vite build      # once
env -u ELECTRON_RUN_AS_NODE node_modules/.bin/electron out/main/index.js \
  --user-data-dir=/tmp/atelier-lab-profile --remote-debugging-port=9555 --use-mock-keychain
```

Open a page in it. Then in the lab choose "Browser tab", or on the command line:

```bash
npm run lab -- tab --cdp 9555 --match news.ycombinator.com \
  --task "Which of today's stories are about AI or machine learning? Put them in ai, the rest in other." --buckets ai,other
npm run lab -- tab --cdp 9555 --match lab/sites/inbox --process inbox-triage --expect inbox:triage   # a mock page, scored
```

The tab is read the way the decision model was trained to read pages: its text, with every control marked
(`[button] Reject all`, `[link] View order`) and the row it sits in. Clicking only happens with `--act` (or "Let it
click"). Even then, a click that pays, buys, deletes, sends or accepts stops and asks (below). The mock pages are
served at `/lab/sites/<fixture>`, so they open in any browser.

## The format

A process is JSON (`src/process/schema.ts`). Its shape suits structured outputs, so Claude's output is held to the
schema exactly. Every step has routes: for each answer (and for `unsure`), what happens next.

| Step | What it does | Answered by |
|---|---|---|
| `ask` | One question about an item's fields or the page: `yes_no`, `choice` (2-8 options, each with a one-line meaning) or `level` | the decision model |
| `check` | Compares one field: `<`, `<=`, `>`, `>=`, `=`, `!=` on a number as pages write it ("€ 1.049,00", "$1,299"), or `contains` / `matches` | code |
| `pick` | Finds a control on the page by what it does: each control is sorted into the step's kinds ("reject-all", "accept-all", "settings", …) in one request, and the one most likely to be a wanted kind is picked when it is clearly ahead. Or by code: the one control whose label or row matches ("placed 14 Oct") | the decision model, or code |
| `act` | A fixed action the planner already knows: click "My orders", open a URL | nothing to decide |

The actions are `goto` (another step), `put` (a bucket), `ask_person`, `click`, `open`, `notify` and `stop`.
`over: "items"` runs the steps for every item on a page (selector and fields as CSS). `over: "page"` runs them once
on the page itself.

**The click guard.** Before any click, two judges look at the control: Atelier's own word list (`agent/guard.ts`,
with Dutch and German added here) and Atelier's own click question (which the nano knows). Either one finding a click
that pays, buys, deletes, sends, posts or accepts stops the process and asks the person. A model can only add a
stop, never lift one.

## What we measured

On 2026-11-24, on an M2 MacBook (24 GB) with v0.6: the nano on the CPU, the 2B mini on Metal. Five mock sites (the
lab's fixtures) hold 87 items with the answer a careful person would give. They were written for this test, in
English, Dutch and German, with hard cases on purpose. Results: `analysis/results/processes-v0.6-mac-m2.{txt,json}`.

| Task | Ready-made: decided / right / wrong | Set up by Claude: decided / right / wrong |
|---|---|---|
| Sort an inbox into needs me, later, archive (22 emails) | 91% / 95% / 1 | 77% / 100% / 0 |
| Bills still to pay (22 emails) | 95% / 95% / 1 | 95% / 100% / 0 |
| Laptops under €900, 16 GB, new (16 results) | 81% / 100% / 0 | 81% / 100% / 0 |
| Jobs a mid-level frontend developer could apply for (14 posts) | 86% / 100% / 0 | 64% / 100% / 0 |
| Refuse a cookie banner (9 pages, 3 languages) | 56% / 100% / 0 | 56% / 100% / 0 |
| Find your way through a shop's account pages (4 goals) | none (a route is the planner's) | 3 of 4 reached; the 4th went on past the help page it was asked for |
| **All** | **86% / 97% / 2 of 83** | **79% / 99% / 1 of 87** |

- **The ready-made processes were revised** after their first runs showed what the model can't do (below).
  So their figures are not a blind measure. Claude's plans are: Claude never saw an expected answer, and its
  try-out only reports dead ends (a click that found nothing, a code pick that matched several rows, an error).
- **Claude's plans are more careful.** It sent security alerts and a possible phishing email to the person to
  check, where the fixture says "needs me". That counts as undecided, not wrong.
- **Time:** one decision takes 0.24-0.45 s, and an item 0.3-0.7 s. Every question in a process is an open one, so it
  goes to the mini; the nano declines open questions and answers only Atelier's own kinds, in milliseconds. A
  22-email inbox takes 9-13 s. Claude's one call takes 10-28 s.
- **In a real tab:** in the test Atelier, over CDP, the ready-made inbox process gave the same result as offline
  (91% / 95%). Claude, setting it up from the live page and finding the selectors itself, decided 20 of 22, all
  right. On Hacker News's real front page ("which stories are about AI"), 30 stories took 7.6 s, and 29 were sorted
  as a person would. With real clicks on the mock shop, Claude's route reached the October order's invoice.
- **Atelier's own eval on this Mac** (`tests/decisionsEval.test.ts` against this server): 10 of 10 tests pass, and
  every sure answer is right except the two known skill misses. Atelier's kinds answer in 10-34 ms median (nano);
  those that reach the mini take 0.25-0.29 s (`analysis/results/atelier-eval-v0.6-mac-m2.txt`).

## What a process can and can't ask of the decision model

These came out of the runs above. The planner's instructions (`PLANNER_SYSTEM` in `src/process/planner.ts`) now
say them.

1. **It sorts a short text well**: what kind of email, is someone waiting on the reader, is this about AI. These
   decided 77-95% of items, right 95-100% of the time.
2. **Numbers, dates and one-word labels go to code.**
   - "8 GB RAM (upgradable to 32 GB)" was read as at least 16 GB (0.80 sure).
   - "Is this the order placed in October?" scored the August order 0.87 against October's 0.91.
   - A `check` or a `pick` by code gets these right every time. Asking *which* amount is installed (a choice) works
     better than asking whether it is at least 16 (a comparison): the 8 GB traps then went to the person instead of
     being wrong.
3. **It can't steer toward a goal.** "Which link gets me closer to my invoice?" failed both ways: as one choice
   among the page's links ("Shop the sale", 0.29) and as a yes/no per link (everything near 0.5). The route through a
   site is the planner's (act steps). The decision model checks where each step landed and picks among controls by
   what they do.
4. **Options must not share their key words.** With "new" and "like new (as new, als nieuw)" as options, a laptop
   sold "New" split 0.62 / 0.31. One answer's meaning that mentions "as new" pulled "New" toward no (0.54-0.58).
5. **A 7-way choice spreads its probability.** "What kind of email is this?" left personal emails unsure (0.39-0.64).
   Asking first whether a person wrote it, then what kind of automated email it is, decided 91% instead of 82%.
6. **Ask the question that rules out most first.** In the jobs task, asking for the level before the place skipped
   two senior roles whose place the model wasn't sure of. That took the task from 64% to 86% decided.
7. **Atelier's click question reads "Reject all" on a cookie banner as a delete (sure, at least 0.8).** In the
   cookie task the right control was picked every time, and the guard then asked the person. The same happens to the
   companion in Atelier: harmless, but a question it shouldn't need to ask. A family of consent-banner controls for
   the click question belongs in the next training set.

## Whole flows: order this, find me that, fill in this form

Sorting and checking text is one kind of task. "Order the Trailrunner 2 in size 43", "find me a 2-person tent under
€200" and "report that our bin wasn't emptied" are another: many pages, fields to fill in, buttons to press. Browser
agents built on hosted decision models (next section) run these one operation at a time. Two step kinds do that here
(`src/process/steps.ts`):

- **`advance`** moves toward a goal one operation per request:
  1. The page becomes a numbered list (`[9] field Postcode: empty`, `[13] radio Standard delivery: not checked`).
  2. One request asks which operation (click, type, done, blocked), which element for a click, and which field for
     typing.
  3. A second question asks which of the process's named values, or which of a select's options, goes in.
  4. The model never writes text: the planner wrote every value down ("description": the text for the form, in the
     form's language).
  5. Every click passes the guard. When the guard stops a click, the person is asked; a yes lets that one click go
     ahead.
- **`fill`** fills every empty field of a form:
  1. The page's own autofill hints first (`autocomplete="email"`, `type="tel"`), as a browser fills addresses.
  2. Then the decision model, field by field.
  3. Then one escalation call for whatever is left.

**Escalation (System 2).** When the decision model isn't sure (an operation below 0.6, or an element or value not
clearly ahead), Claude decides that one operation, or that one form, from the same numbered list. When an
escalation is available, a "blocked" from the small model also goes to Claude: a stretch never ends on the small
model alone. Every decision is recorded with who made it.

```bash
npm run lab -- flows --cdp 9555 --escalate --approve -v      # the ready-made flow processes
npm run lab -- flows --cdp 9555 --planned-only --escalate --approve -v   # Claude writes each from the start page
```

- **The sites** (`lab/apps/`) are real pages, served by the decision server and opened in the browser:
  - a shop with search, product pages with sizes, a cart, a checkout form and a payment page;
  - a two-page Dutch municipal report form ending at "Versturen".
- **The check** reads what ended up in the cart or the form from the page's storage afterwards, and compares it with
  what the person asked for (`lab/flows.json`). Paying or sending counts as a failure.
- **`--approve`** stands in for the person when the guard asks:
  - yes to the steps toward the last page (Checkout, Continue to payment, a box the task asked for);
  - never yes to the click that pays, orders, sends or submits.

### Results (2026-11-24, M2, v0.6, Claude Opus 5.5 through the Claude Code CLI as escalation)

| Flow | Ready-made process | Planned by Claude from the start page |
|---|---|---|
| Order Trailrunner 2, size 43, standard delivery | ✓ on the payment page, every checkout field right, not paid · 81 s · decisions: Claude 12, page hints 3, our model 4 | ✓ same · 109 s · Claude 13, our model 4, code 1 |
| Find a 2-person tent under €200, add it to the cart | ✓ Trailhut 2 (€189) in the cart · 35 s · Claude 4, our model 2 | ✓ same · 56 s · Claude 7, our model 2 |
| Bin not emptied: two-page Dutch form | ✓ every field right, stopped before Versturen · 75 s · Claude 14, page hints 3, our model 2 | ✓ same · 91 s · Claude 15, our model 2, code 2 |

- **The person was asked** two or three times per flow: "Checkout" (Atelier's own word list counts it as a
  purchase), "Continue to payment" and the email-updates box (the model read them as payment and message clicks).
  No flow paid or sent anything.
- **Claude made most of the decisions.** Each escalation takes 6-10 s through the CLI, so flows take one to two
  minutes. Our model decided 2-4 operations per flow. The page's autofill hints filled 3 fields in each ready-made checkout or form.
- **Claude's plans were sound.** It used the decision model for judgement ("is this a tent for two people?"), code
  for prices, and clear stopping points. One plan compared "the page" with €200, which reads the page's first number
  (a cart count); the validator now refuses that.
- **Final checks.** Two of Claude's final checks asked what the page couldn't show ("does the cart hold a 2-person
  tent?" on a cart line that names the tent). They ended with "please look" although the result was right.

### Our model on steps: not trained for it yet

(Since then: atelier-step, a model trained for exactly these steps; [step-model.md](step-model.md). The figures below
are atelier-mini's, before it.)

The same questions, built from Mind2Web's recorded actions on 7 websites held out whole (`data/build_steps.py`,
`npm run lab -- steps`), zero-shot:

| Question | Right (top answer) | Sure (the runtime's cutoffs) |
|---|---|---|
| Which operation (98 click, 26 type) | 56% | 2% |
| Which element to click, among 30-40 | 9% | 0% |
| Which field to type into | 42% | 25% |
| Which value goes in | 55% | 5% |

So on the lab's flows the decision model was sure of almost nothing. That is why Claude decided most operations.
Jev runs this loop on its own: Prequel reports 17 Jev decisions for a Google Flights search. The difference is
training, and this repo's recipe fits it:

1. **Steps from Mind2Web**, answers by construction (`data/build_steps.py`): all 1,009 training tasks of the public
   split, websites held out whole.
2. **"Done" and "blocked", and forms in Dutch, German, French and Spanish**: generated from scripted runs of
   lab-style sites, whose right answer is known by construction. Mind2Web has no page after its last action.
3. **The click question's over-cautious kinds**, all reversible or not yet paying: "Add to cart" (purchase), "Continue
   to payment" (payment), "Reject all" on a cookie banner (delete), an "updates by email" box (message).
4. **Speed.** Today one operation takes 3-9 s on the M2. The mini reads the whole element list again for every
   question; its attention is non-causal, so nothing is reused between questions. Two ways to bring a step down
   toward Jev's 0.2 s:
   - asking a step's questions in one pass (a format change, made in training);
   - fewer elements per request.

Claude's escalation answers are not used as training labels; the data above has its answers by construction.

## How decision models run whole flows elsewhere

Researched 2026-11-24. Most sources are write-ups by the builders themselves, so their numbers are theirs.

**The browser loop (Jev Ultrafast and "hunch", [Prequel, 2 Oct 2026](https://prequel.sh/blog/decision-models-for-browser-agents)).**
This is the pattern behind "order this", "fill in this form" and "find me this" with a decision model:

1. Code turns the page into a numbered list, one element per line: `[2] combobox Where from? · San Francisco`.
2. One request asks three questions at once:
   - `operation`: "Advance the goal from the current page with one operation." The options are CLICK, TYPE_TEXT, DONE
     and BLOCKED.
   - `click_target`: which listed element, for a click.
   - `type_text_target`: which field, for typing.
3. The model never writes text and never points outside the list:
   - Jev Ultrafast has a small language model write the one string to type.
   - hunch uses no language model: the caller gives named values.
4. Code takes over from there:
   - It refuses malformed answers and acts only above 0.75.
   - It asks a separate yes/no risk question and checks words like "pay" or "publish".
   - It checks the element is still there before clicking.
   - It verifies "done" itself (the URL, a confirmation text).
   - Below the cutoff, when the page stops changing, or when a click looks irreversible, it hands a report to a
     language-model agent above it.

They report a Google Flights search in 7 s with 17 Jev requests (178 ms median), and a three-field form in 3.4 s. The
model's weak points they list: dates, counting, steps that need reasoning, and irrelevant text.

**The same shape elsewhere:**

- [system-one-browser-agent](https://github.com/adnanahmaddev/system-one-browser-agent) (MIT) asks four questions per
  step: which candidate, is the task complete (0.82), is this destructive (stop at 0.75), and what kind of page it is.
  Below 0.55 it falls back to a language model.
- ["Classify first, authorise in code"](https://dev.to/_firelinks/system-one-models-in-an-agent-loop-classify-first-authorise-in-code-joh)
  has the model classify a proposed action (read only, reversible, public or irreversible, credentials or money; does
  it serve the task; does the page instruct the agent). Code holds the allow, ask and block rules, and it fails
  closed.
- [Kai Waehner](https://www.kai-waehner.de/blog/2026/10/01/jev-by-typesafe-ai-in-workflow-orchestration-and-process-intelligence/)
  describes Jev inside process engines (Camunda gateways, Airflow, Kestra). High confidence runs automatically,
  medium goes to a language model and then a person, low goes to a person. Every branch is logged with its
  probability and threshold, and thresholds are pinned to a model version.
- A talent-matching service moved candidate screening from an LLM to Jev, at about an eighth of the cost and half the
  time, with the same shortlists (a vendor-reported case study).

**Step models with open weights:**

- [OpenJev-8B](https://huggingface.co/alanhuangya/OpenJev-8B) (Apache-2.0, Qwen3-8B) reports Mind2Web step results.
- Cua's System One form scorer (MIT, according to a directory listing) treats filling a form as classification: a
  field and its candidate values in, one probability per value out, at 706K parameters.
- The [awesome-system-one](https://github.com/andyrewlee/awesome-system-one) list has more: browser, desktop and phone
  agents.

**What this runtime took from them:**

- `advance` is that loop: numbered elements, operation and targets in one request, values named by the planner, a
  check before every click, and an escalation when the model isn't sure.
- `fill` is the form-scorer idea, with the page's own autofill hints first.
- The add-to-cart rule in the guard is "authorise in code".

## Files

| Path | What |
|---|---|
| `runtime/node/src/process/schema.ts` | The format, its JSON Schema for structured outputs, the validator, numbers as pages write them |
| `runtime/node/src/process/run.ts` | The runner: steps, routes, pick, the guard before clicks, scoring against a person |
| `runtime/node/src/process/decide.ts` | Steps as System One questions, answers as verdicts (Atelier's cutoffs), the click guard |
| `runtime/node/src/process/planner.ts` | Claude sets a process up (CLI or API), with one fix-up round for problems and one for its try-out |
| `runtime/node/src/process/steps.ts` | advance and fill: the page as numbered elements, operation and targets in one request, values named by the planner, the escalation |
| `runtime/node/src/process/flows.ts`, `runtime/node/lab/apps/`, `runtime/node/lab/flows.json` | Whole flows on real pages (a shop, a municipal form), checked afterwards against what the person asked for |
| `data/build_steps.py` | Step questions from Mind2Web's recorded actions, answers by construction, sites held out whole |
| `runtime/node/src/process/cdp.ts` | A real browser tab as a surface: the page as marked text, items by selector, clicks |
| `runtime/node/src/process/fixtures.ts`, `lab.ts`, `labServer.ts` | Mock pages (list, single pages, a small site) as surfaces and as HTML; the lab's tasks, plans and routes |
| `runtime/node/src/lab.ts`, `runtime/node/lab.html` | The lab on the command line and in a browser (`/lab`) |
| `runtime/node/processes/` | Ready-made processes |
| `runtime/node/lab/fixtures/` | The mock sites with the answers a person would give (each file's `notes` explain the hard cases) |
| `runtime/node/lab/plans/` | Claude's plans for the lab's tasks, as measured |

## Next

- **In Atelier:** the companion sets a process up from a request ("sort my inbox like this every morning"). Atelier
  then runs it on the person's decision model with `decider()`, so private windows, budgets and the rule fallback
  apply, and shows the "to you" items as one list to go through. Routines and packs (`atelier.ai.decide`) could run
  saved processes. Without a decision model connected, the companion decides as it does today.
- **For the model:**
  - the consent-banner controls for the click question;
  - "is this the page where…" checks after a step;
  - steps from real processes (the lab can log every request, like `teacher/server.py --log`) as a new family;
  - the nano answering open questions of a process's simplest kinds, so a step takes 10 ms instead of 0.3 s.
