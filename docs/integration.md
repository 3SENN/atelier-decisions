# Using the model in Atelier

## Built in: "On this computer" (patch ready)

`integration/atelier-on-device-decisions.patch` adds Atelier's own decision model as a provider that needs no key,
no network and no setting up. It was built on the branch `on-device-decisions` in a separate worktree
(`../atelier-on-device`), so the main checkout is untouched; nothing is committed.

```bash
cd atelier
git apply ../atelierdecisionmodel/integration/atelier-on-device-decisions.patch   # or: git worktree / branch on-device-decisions
npm install
ATELIER_DECISION_MODEL_SRC=../atelierdecisionmodel/runs/<student>/onnx node scripts/fetch-decision-model.mjs
npm run dev    # Settings › Quick decisions › On this computer › Use it
```

| Piece | Where | What it does |
|---|---|---|
| Provider | `src/shared/decisions.ts` | `device` first in `PROVIDER_DEFAULTS`: `atelier-device://nano/v1/systemone`, no key (`onDevice`), suggested `atelier-nano` |
| Decider | `src/main/decisions/service.ts` | connected when the model's files are here (not by key); never hedged (treated like a server on this computer) |
| Transport | `src/main/decisions/device.ts`, `feature.ts` | `atelier-device://` answered by a utility process instead of the network: started on the first question, warmed 10 s after start when it is among your models, stopped after five quiet minutes; a crash is a failed request (the rule decides), three in a row make it unavailable until restart |
| Model | `src/main/decisions/deviceHelper.ts`, `device/nano.ts`, `device/format.ts` | the runner from `runtime/node` (ONNX Runtime CPU, 2 threads, the pure-JS tokenizer), loaded in the utility process; it declines text it can't read and questions that aren't Atelier's own (`refusal`, so the rule decides) |
| Settings | `QuickDecisions.tsx`, `i18n/{en,nl}/decisions.ts` | a row with Ready / Not installed and "Use it" (adds `atelier-nano`, which turns quick decisions on if it was your first model) |
| Packaging | `electron-builder.yml`, `package.json`, `scripts/fetch-decision-model.mjs` | `native/decisions/build` to `Resources/decision-model` (37 MB: the 34 MB model, tokenizer, vocabulary map, calibration); `onnxruntime-node` 1.31.1 as a dependency (N-API, unpacked from the asar), `@huggingface/tokenizers` bundled; `npm run package` ships without the model and says so when there is none |
| Tests | `tests/decisionsDevice.test.ts`, `tests/decisions.test.ts` | the route, not installed, bad requests, crash limit, cancel, warm-up; the provider end to end (no key, its own address, one request, the rule without files) |
| Docs | `docs/decisions.md` › On this computer | how it works, where the files come from, how it is calibrated |

Checked on this Windows machine: typecheck clean, i18n 100% (8,411 keys), the unit tests passing (the 18 that
fail fail the same on `main`: this Node's SQLite has no FTS5), and the built `out/main/decisionsDevice.js` run as
a real Electron utility process with the final model: first answer 0.9 s (process start and model load, which the
warm-up 10 s after start takes care of), then 1-5 ms; Russian text and an open question answered `refusal`.

Next: let the on-device model hand what it is unsure of to a hosted model in your list, not only what it fails
on (a cascade: most decisions stay on this computer, a few go out, the rule decides when both are unsure); this
changes how the decider reads answers, so it is a decision for Atelier, not part of the patch.

Not yet: crash reports from the utility process are not collected; the model is not code-signed separately (it
is data, read only by ONNX Runtime); macOS x64 has no onnxruntime-node 1.31 binary (onnxruntime-web's WASM build
would be the fallback there).

## Without the patch

Atelier already accepts any System One server as "your own server", and treats one on this computer (localhost,
127.x) as local: no hedged second request, no key.

1. `cd runtime/node && npm install && node --experimental-strip-types src/server.ts <model folder> 8011`
2. Atelier › Settings › Quick decisions › Your own server: address `http://127.0.0.1:8011`, model `atelier-nano`.

The same route makes the teacher a provider (`teacher/server.py`, port 8009), which is also how real decisions are
collected for training: with the teacher as the provider, every request Atelier sends is appended, with the
teacher's logits, to the server's `--log` file.

## Mobile

The same ONNX file runs with ONNX Runtime Mobile (iOS pod / Android AAR, or a reduced-ops build for size), with
`runtime/node/src/format.ts` ported (about 150 lines of string handling) or reused in React Native with
`onnxruntime-react-native`; the tokenizer is the trimmed `tokenizer.json` (2.2 MB). Measured on this CPU with one
thread (roughly a phone core), final 12-layer model: 6.9 ms median, 11.4 ms at the 99th percentile per question;
the 4-layer variant (18.6 MB) about 3 ms.
