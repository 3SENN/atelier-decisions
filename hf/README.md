---
license: apache-2.0
language:
  - multilingual
  - en
  - nl
  - de
  - fr
  - es
pipeline_tag: text-classification
library_name: onnx
tags:
  - atelier
  - on-device
  - distillation
  - decision-model
  - onnx
  - gguf
  - llama.cpp
base_model:
  - jhu-clsp/mmBERT-small
  - Qwen/Qwen3.5-0.8B
datasets:
  - AmazonScience/massive
  - allenai/WildChat-1M
  - osunlp/Mind2Web
  - glaiveai/glaive-function-calling-v2
  - google/Synthetic-Persona-Chat
  - knowledgator/gliclass-v2.0
  - MoritzLaurer/synthetic_zeroshot_mixtral_v0.1
---

# Atelier decision model

The small models [Atelier](https://github.com/3SENN/atelier), the open-source browser, uses for its quick
decisions: is a reply waiting on you, would this click buy or delete something, is this a search or a question,
which skill fits. They run on your computer and send nothing anywhere. When they aren't sure, Atelier's rule decides.

- **atelier-nano** (`model/`): mmBERT-small, 22 layers, int8 ONNX with a 4-bit embedding table. **54 MB, about 10 ms
  a question** on an ordinary CPU (ONNX Runtime). Built into Atelier.
- **atelier-mini** (`mini/`): Qwen3.5-0.8B read the way a decider reads (non-causal attention, a 255-code readout),
  as a llama.cpp GGUF. **515 MB, 0.14-0.33 s a question** on the CPU. An optional download in Atelier, for what the
  nano declines or is unsure of.

Both are distilled from Perplexity's [pplx-decider-flash-v1.3](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.3)
(and checked against [pplx-decider-v1.3](https://huggingface.co/perplexity-ai/pplx-decider-v1.3)), and from numeric
and security questions whose answers are computed or known by construction.

**The code, the data builders, the training and every measurement are on GitHub:
[3SENN/atelier-decisions](https://github.com/3SENN/atelier-decisions).** This repository holds the files too large
for git.

## Versions

Each version is saved whole under `releases/<version>/` and tagged with its name. Its `MANIFEST.json` lists every
file with its size and SHA-256, and its `MODEL_CARD.md` says what changed, how it was measured and what it can't do.

| Version | What's new | Model card |
|---|---|---|
| v0.5 | nano v0.3, then mini v0.5: retrained with security and multi-step number questions | [MODEL_CARD.md](releases/v0.5/MODEL_CARD.md) |
| v0.4 | nano v0.3, then the first atelier-mini | [MODEL_CARD.md](releases/v0.4/MODEL_CARD.md) |
| v0.3 | nano: 22 layers and 480 browser question families | [MODEL_CARD.md](releases/v0.3/MODEL_CARD.md) |
| v0.2 | nano: 22 layers and open decisions | [MODEL_CARD.md](releases/v0.2/MODEL_CARD.md) |
| v0.1 | nano: 12 layers, 34 MB | [MODEL_CARD.md](releases/v0.1/MODEL_CARD.md) |

In each version:

| Folder | What |
|---|---|
| `model/` | atelier-nano: `model.emb4.int8.onnx` (the shipped file), `model.int8.onnx`, the PyTorch checkpoint `student.pt`, tokenizer, calibration, training log |
| `mini/` | atelier-mini (v0.4 on): `mini.gguf` (Q5_K_M layers, 4-bit embeddings), `readout.f32`, `decision.json`, the adapters `mini.pt`, tokenizer, calibration, parity fixtures |
| `eval/` | the reports its numbers come from |
| `data/` | what it was trained on: the teacher-labelled pools and the written inputs (zipped) |
| `download/`, `integration/` | Atelier's side: the mini's download manifest and the patch that adds the models to Atelier |

## Use

Download one version (all of v0.5 is about 1 GB; add `model/*` or `mini/*` to the path to take less):

```bash
hf download 3SENN/atelier-decisions --include "releases/v0.5/*" --local-dir .
```

From a clone of the GitHub repository, `python tools/hf_release.py fetch v0.5` does the same and checks every file
against the manifest. The runner in `runtime/node` then answers Atelier's System One requests on 127.0.0.1:

```bash
cd runtime/node && npm ci
node --experimental-strip-types src/server.ts ../../releases/v0.5/model 8011 \
  --mini ../../releases/v0.5/mini --llama-server <path to llama-server>
```

The models read Atelier's request format (`student/format.py`, `runtime/node/src/format.ts`), not free text: they
answer Atelier's questions, plus open questions in the same format (the mini).

## What they can't do

From the v0.5 model card, in short:

- **Not a guard on their own.** The mini is sure and wrong on 6% of security questions (16.5% on lookalike senders,
  15% on whether a summary is faithful). Atelier never relies on it alone.
- **Arithmetic in several steps** (percentages, VAT, split bills): the mini declines and leaves it to the rule.
- **New kinds of browser questions:** right 81-83% of the time when the mini decides; the nano declines them.
- **Languages:** trained on English, Dutch, German, French and Spanish. The nano declines text that is mostly not
  Latin script; the mini decides 22% of 20 unseen languages.

## Licence

The weights are licensed under the Apache License 2.0. atelier-nano is built from
[mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) (MIT), atelier-mini from
[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (Apache-2.0, changed as described above), and both learned
from the pplx-decider teachers (Apache-2.0).

In the `data/` archives, the data written for this project is CC BY 4.0. Text from other datasets keeps its own
licence: MASSIVE, Mind2Web and Synthetic-Persona-Chat (CC BY 4.0), WildChat-1M (ODC-BY), glaive-function-calling-v2,
gliclass-v2.0 and synthetic_zeroshot_mixtral_v0.1 (Apache-2.0). The integration patch is code, under the MPL-2.0 like
Atelier. The full list is in [NOTICE](https://github.com/3SENN/atelier-decisions/blob/main/NOTICE).
