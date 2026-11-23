# Using the model in Atelier

## Built in: "On this computer" (patch ready)

`integration/atelier-on-device-decisions.patch` adds Atelier's own decision model as a provider that needs no key,
no network and no setting up, and (later commits) atelier-mini as an optional download for new kinds of questions,
now v0.6. It was built on the branch `on-device-decisions` in a separate worktree (`../atelier-on-device`), so the
main checkout is untouched: six local commits there, not pushed. The patch is `git diff` from Atelier's main (`7430717`)
to that branch and applies cleanly to it.

```bash
cd atelier
git apply ../atelier-decisions/integration/atelier-on-device-decisions.patch   # or: git worktree / branch on-device-decisions
npm install
ATELIER_DECISION_MODEL_SRC=../atelier-decisions/runs/nano-v03-l22/onnx node scripts/fetch-decision-model.mjs   # or releases/v0.6/model
npm run dev    # Settings › Quick decisions › On this computer › Use it
```

| Piece | Where | What it does |
|---|---|---|
| Provider | `src/shared/decisions.ts` | `device` first in `PROVIDER_DEFAULTS`: `atelier-device://nano/v1/systemone`, no key (`onDevice`), suggested `atelier-nano` |
| Decider | `src/main/decisions/service.ts` | connected when the model's files are here (not by key); never hedged (treated like a server on this computer) |
| Transport | `src/main/decisions/device.ts`, `feature.ts` | `atelier-device://` answered by a utility process instead of the network: started on the first question, warmed 10 s after start when it is among your models, stopped after five quiet minutes; a crash is a failed request (the rule decides), three in a row make it unavailable until restart |
| Model | `src/main/decisions/deviceHelper.ts`, `device/nano.ts`, `device/format.ts` | the runner from `runtime/node` (ONNX Runtime CPU, 2 threads, the pure-JS tokenizer), loaded in the utility process; it declines text it can't read and questions that aren't Atelier's own (`refusal`, so the rule decides) |
| Settings | `QuickDecisions.tsx`, `i18n/{en,nl}/decisions.ts` | a row with Ready / Not installed and "Use it" (adds `atelier-nano`, which turns quick decisions on if it was your first model) |
| Packaging | `electron-builder.yml`, `package.json`, `scripts/fetch-decision-model.mjs` | `native/decisions/build` to `Resources/decision-model` (57 MB: the 54 MB model of nano v0.3, tokenizer, vocabulary map, calibration); `onnxruntime-node` 1.31.1 as a dependency (N-API, unpacked from the asar), `@huggingface/tokenizers` bundled; `npm run package` ships without the model and says so when there is none |
| Tests | `tests/decisionsDevice.test.ts`, `tests/decisions.test.ts` | the route, not installed, bad requests, crash limit, cancel, warm-up; the provider end to end (no key, its own address, one request, the rule without files) |
| Docs | `docs/decisions.md` › On this computer | how it works, where the files come from, how it is calibrated |

Checked on this Windows machine: typecheck clean, i18n 100% (8,411 keys), the unit tests passing (the 18 that
fail fail the same on `main`: this Node's SQLite has no FTS5), and the built `out/main/decisionsDevice.js` run as
a real Electron utility process with the final model: first answer 0.9 s (process start and model load, which the
warm-up 10 s after start takes care of), then 1-5 ms; Russian text and an open question answered `refusal`.

The checks above were made with nano v0.1 (34 MB); the patch now fetches v0.3, which Atelier's eval passed through
the same runtime (10 of 10 tests). What the nano is unsure of now goes to the mini on this computer (below); handing
it to a hosted model in your list instead would change how the decider reads answers, so that stays a decision for
Atelier.

Not yet: crash reports from the utility process are not collected; the model is not code-signed separately (it
is data, read only by ONNX Runtime); macOS x64 has no onnxruntime-node 1.31 binary (onnxruntime-web's WASM build
would be the fallback there).

## nano, then mini: the cascade (in the patch, an optional download)

atelier-mini (Qwen3.5-2B since v0.6, Qwen3.5-0.8B before; read the decider's way, run by llama.cpp) works out kinds
of questions the nano was never shown, but it is about 30-60 times slower. So the two run as one decider:

1. **The nano answers first**, in milliseconds.
2. **The mini gets what the nano can't settle:** a question the nano declines (an open question, text it can't read)
   or is unsure about by Atelier's own cutoffs (a port of `student/readers.py`).
3. **The rule decides** what the mini is unsure about too.

### In Atelier (the patch's later commits)

The patch now carries the cascade into Atelier's utility process, and the mini as an optional download:

| Piece | Where | What it does |
|---|---|---|
| Cascade | `src/main/decisions/device/cascade.ts`, `readers.ts` | nano first; a refusal or an answer Atelier's reader would not take goes to the mini; a failed or stopped mini leaves the nano's answers; `routes` says who answered |
| Mini | `device/mini.ts`, `miniFiles.ts` | prompts and token ids as the training code makes them (608/608 with `ATELIER_MINI_PARITY`, v0.4, v0.5 and v0.6), llama-server on 127.0.0.1, readout and temperature in the helper, states cut at 2048 tokens, cancelled questions skipped; a trimmed vocabulary (`vocab.json`) is mapped as the training code maps it |
| Helper | `deviceHelper.ts`, `device.ts` | the mini loads on the first question the nano leaves open (never at warm-up); a new helper when the mini is installed or removed; stopped requests are cancelled in the helper; llama-server stopped by its pid with the helper |
| Download | `deviceMini.ts`, `device/miniManifest.ts` | file by file against sizes and SHA-256 compiled into Atelier, staged and swapped in whole, resumed with range requests, through `net.fetch`; offered only when the manifest has a download location and a server build for this platform |
| Settings | `QuickDecisions.tsx`, `i18n/{en,nl}/decisions.ts` | "New kinds of questions" under On this computer: Download (size), progress, Stop, Remove |
| Tests | `tests/decisionsDeviceMini.test.ts` | routing and fallbacks, readers' cutoffs, the download (checksums, resume, ranges ignored, stop, remove, paths leaving the folder, a folder set from outside), server lifetime, respawn, cancel |

Checked on this Windows machine:

- **Builds and tests:** typecheck clean, i18n 100% (8,419 keys), the decision tests passing (115, with v0.6's
  parity).
- **The download:** the real installer against a local server, 585 MB with every file checked (v0.4's files; v0.5's
  are the same set and sizes, with new checksums; v0.6's are the same set at 1,357 MB, the model 1,286 MB).
- **The built helper in Electron:** run as a utility process with the nano and the mini.
  - The nano kept its own decisions.
  - Russian, Thai, an open price question and three email labels went to the mini: 2 s for the first (llama-server
    starting cold), 0.1-0.6 s after that.
  - llama-server stopped with the helper.

**Where the files live:** the model project's Hugging Face repository, under `dist/<version>/`
(`tools/hf_release.py publish-dist`). `tools/mini_dist.py` makes the folder to host (`model/`, `server/<platform>/`)
and the manifest that Atelier compiles in, whose `base` points there, so Settings offers the download.

- **Platforms:** llama.cpp's own CPU builds of release b12620, checked against the release's SHA-256: Windows x64 and
  arm64, macOS arm64 and x64, Linux x64 and arm64 (24-76 MB each; library links in the archives become copies).
- **Tested:** the Windows x64 server, here. The others are llama.cpp's builds as released, not yet run on a Mac or
  Linux machine with the mini; Atelier marks them executable after download.

```bash
python tools/mini_dist.py --runtime runs/mini-2b-v05-cond/runtime --version v0.6 --out runs/mini-dist-v06 \
    --server win32-x64=models/llama.cpp-b12620/bin --server win32-arm64=<unpacked win-cpu-arm64 build> \
    --server darwin-arm64=<unpacked macos-arm64 build> --server darwin-x64=<unpacked macos-x64 build> \
    --server linux-x64=<unpacked ubuntu-x64 build> --server linux-arm64=<unpacked ubuntu-arm64 build> \
    --licence models/llama.cpp-b12620/src/LICENSE --base https://huggingface.co/3SENN/atelier-decisions/resolve/main/dist/v0.6/
python tools/hf_release.py publish-dist v0.6 runs/mini-dist-v06
cp runs/mini-dist-v06/miniManifest.ts ../atelier-on-device/src/main/decisions/device/miniManifest.ts
```

During development, `ATELIER_DECISION_MINI=<runtime folder with a server/ copy of llama-server>` uses a folder as it
is, and `ATELIER_DECISION_MINI_URL=<base>` tries a download location.

### Outside Atelier (runtime/node)

```bash
python student/export_mini.py --model runs/mini-v05 --gguf runs/mini-v05/mini-v05-Q5_K_M-e4.decider.gguf --out runs/mini-v05/runtime
cd runtime/node
node --experimental-strip-types src/mini_parity.ts ../../runs/mini-v05/runtime        # token ids = Python: 608/608
node --experimental-strip-types src/server.ts ../../runs/nano-v03-l22/onnx 8011 4 --mini ../../runs/mini-v05/runtime \
    --llama-server ../../models/llama.cpp-b12620/bin/llama-server.exe                      # model "atelier-nano+mini"
```

Each answer says which model gave it (`routes`). `--mini-only` serves the mini alone.

The runtime folder comes from `student/export_mini.py`. It holds:

- the marked GGUF;
- the readout (`readout.f32`, 1 MB);
- `decision.json`: answer codes, temperatures, the chat-template prefix, escaped special tokens, separator and marker ids;
- the tokenizer.

The runner tokenizes and sends token ids to llama-server's `/embedding`, which returns the final hidden state at the
marker. The runner applies the readout and the temperature.

**Tokenizer:** the runner uses the base's own `tokenizer.json`, so llama.cpp, tokenizers.js and Python agree token for
token (608 of 608).

- **Teachers and the mini differ:** transformers 5.17 builds Qwen3.5's tokenizer with the older Qwen2 split rule. The
  teachers ship that rule in their own files, so it is right for them. The mini's base was pretrained with the newer
  rule, which keeps combining marks with their letters.
- **It matters for Hindi and Thai:** with the base's rule the mini is right 70% of the time when it decides on Thai
  instead of 40%, and 78% instead of 69% on Hindi (`analysis/tokenizer_effect.py`). Other languages don't change.
- **So every mini script loads the tokenizer through `student/mini_tokenizer.py`**, and v0.5's mini is the first trained
  with it from the start.

**Memory:** v0.6's 2B peaks at 1.9 GB for llama-server at Atelier's 2048-token batch (the 1.3 GB model and 0.6 GB of
compute buffer); the 0.8B took 0.8 GB at a 4096-token batch and about 0.5 GB at 2048. The Node runner with the nano and
the mini's tokenizer takes 370 MB.

**On phones:** llama.cpp runs the same GGUF on Android and iOS. A port needs about 100 lines of prompt building, the
tokenizer and the 255 x 2048 readout (255 x 1024 for the 0.8B). At 2B and 4-5 bits, expect a second or two per
question on a phone CPU, and 1.3 GB to download; that is an estimate, not a measurement, and the 0.8B (v0.5) is the
lighter choice there. The nano stays the phone's first answer.

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
