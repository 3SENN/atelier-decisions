# Contributing to the Atelier decision model

Thanks for helping. This repository makes the small models [Atelier](https://github.com/3SENN/atelier) uses for its
quick decisions: the data they learn from, their training, how they are measured, and every saved version. Atelier,
where the questions are asked and the answers are read, is its own repository.

## Before you start

- **A decision the model gets wrong:** open a **Wrong decision** issue with the request (remove anything private),
  the version, what it answered and what you expected. These are the most useful reports: the targeted pools (v3 to
  v6) were each built to fix misses like these.
- **Bugs in the code or tools:** search the [issues](https://github.com/3SENN/atelier-decisions/issues) first, then
  open one with the bug template.
- **Bigger changes** (a new student, a new data source, a change to what the runtime declines): open an issue first,
  so we agree on how to measure it before anything is trained.
- **Questions and chat:** [Atelier's Discussions](https://github.com/3SENN/atelier/discussions) or
  [Discord](https://discord.gg/sKUTVCYAZY).
- **Security problems:** never in public. See [SECURITY.md](SECURITY.md).

## Setting up

What each kind of work needs:

| Work | Needs |
|---|---|
| Data builders, the analysis scripts, reading the saved results | Python 3.12, any computer |
| The Node runner (`runtime/node`) | Node.js 22.12 or later, and a version's files (below) |
| Labelling with a teacher, training a student | an NVIDIA GPU with 12 GB or more (every version so far was made on an RTX 3060), the teachers (61 GB on disk) |

```bash
git clone https://github.com/3SENN/atelier-decisions.git
cd atelier-decisions
uv venv --python 3.12 .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu130   # or the build for your machine
uv pip install -r requirements.txt
python tools/hf_release.py fetch v0.5     # v0.5's weights and data from Hugging Face, checked against MANIFEST.json
```

`fetch v0.5 --only model` gets just the nano (about 350 MB with its PyTorch checkpoint). The Node runner then answers
Atelier's requests on this computer:

```bash
cd runtime/node
npm ci
node --experimental-strip-types src/server.ts ../../releases/v0.5/model 8011
```

The README's [Reproduce](README.md#reproduce) section has every step from the teachers to a saved version.

## Checks

CI runs these on every pull request. Run them first:

```bash
python tools/check_headers.py                         # every source file carries the MPL-2.0 notice (--fix adds it)
python -m compileall -q analysis data student teacher tools
python tools/hf_release.py verify --all --present     # nothing under releases/ has changed
cd runtime/node && npm run typecheck
```

## How this project measures

These rules keep the numbers honest. A change that breaks one isn't merged, however good its numbers look.

- **Never train on Atelier's eval.** `student/train.py`, calibration and `student/eval_onnx.py` leave out every
  request that shares a state, or a free-text field of four words or more, with a case in Atelier's
  `tests/decisionsEval.test.ts`. Don't work around that, and keep the eval's phrases out of templates and written
  data: the first pool reused some, and its results had to be marked optimistic.
- **Held out stays held out.** Whole question families are held out as tests of new kinds of questions (in gen2,
  gen3 and gen4, five numeric families in each numeric suite, three security families). Don't train on them, and
  don't tune a cutoff on the set you report.
- **Coverage and precision together.** Report how often a model decides, how often it is right when it does, and
  how often it is sure and wrong. One of the three alone isn't a result.
- **Every number names its file.** A figure in the README or a model card points at the file in `analysis/results/`
  or the version's `eval/` it comes from.
- **Say what it can't do.** A new version's `MODEL_CARD.md` says in words where it fails.
- **Saved versions don't change.** `releases/<version>` is hashed in its `MANIFEST.json`. A fix is a new version.

## Data

- **Data you write** for this project is licensed under CC BY 4.0, like the rest of the written data.
- **Data from another dataset** must have a licence that allows sharing it: each saved version publishes the pools
  and labels it was trained on. Name the dataset and its licence in the builder's docstring and in [NOTICE](NOTICE).
- **Data written by a language model:** say which model wrote it and how, as `data/build_gen3.py` does.
- **No personal data and no real secrets.** The security builders plant random secrets that only look real; keep
  it that way.

## Saving a version (maintainers)

1. `python tools/make_release.py --version vX.Y ...` writes `releases/vX.Y` and its `MANIFEST.json`.
2. Write `releases/vX.Y/MODEL_CARD.md` by hand: what changed, how it was measured, what it can't do.
3. `python tools/hf_release.py publish vX.Y` uploads the version whole to Hugging Face and tags it (needs a write
   token: `hf auth login`).
4. Commit the text. Git ignores the weights and archives, which are on Hugging Face now. Update the README's
   versions table in the same commit.

For a new mini download, `tools/mini_dist.py --base https://huggingface.co/3SENN/atelier-decisions/resolve/main/dist/vX.Y/`
makes the folder and Atelier's `miniManifest.ts`, and `python tools/hf_release.py publish-dist vX.Y <folder>`
hosts it.

## Code

- **Comments say why.** Each script starts with a docstring that says what it is for and how to run it; follow
  the ones around it.
- **One GPU job at a time.** Two processes time-slice badly, and on Windows CUDA spills into shared memory and
  crawls instead of failing.
- **The format is a contract.** `student/format.py` and `runtime/node/src/format.ts` (and the readers on both sides)
  change together, and `runtime/node/src/parity.ts` shows they still agree.

## Commits and pull requests

- Write commit messages in plain sentences that say what changed and why.
- One change per pull request. Keep your branch up to date with `main`; we squash or rebase when merging.
- A maintainer reviews every pull request. Expect questions; they are about the change, not about you.

## Licence and copyright

The code is licensed under the [Mozilla Public License 2.0](LICENSE), the model weights under
[Apache-2.0](LICENSES/Apache-2.0.txt), and the data written for this project under
[CC BY 4.0](LICENSES/CC-BY-4.0.txt). By contributing, you agree that your contribution is licensed the same way
(inbound = outbound), and you confirm you have the right to submit it.

New source files start with the standard notice (`#` comments in Python, PowerShell and YAML):

```js
/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */
```

`python tools/check_headers.py --fix` adds it. Add your name to [AUTHORS](AUTHORS) with your first contribution.

The project follows [Atelier's governance](https://github.com/3SENN/atelier/blob/main/GOVERNANCE.md): a maintainer
merges what is reviewed and measured, and the project lead decides what ships in Atelier.

## Code of conduct

Everyone taking part in this project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
