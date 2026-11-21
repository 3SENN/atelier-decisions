# Research notes (November 2026)

What the field offers for a tiny, private decision model, and how it shaped this project. Items marked
*(unverified)* came from secondary sources.

## The teachers

- **pplx-decider-v1.3** (Perplexity, Apache-2.0, 9 Nov 2026): Qwen3.8-27B fine-tune, 26.1B params, a frozen
  `[255 x hidden]` readout over answer-code tokens at a `<|fim_suffix|>` marker per question, non-causal full
  attention, causal Gated DeltaNet. Recipe (blog and model cards): hundreds of public datasets (much via
  tasksource) turned into state/questions/options; full fine-tune, then a domain LoRA merged at 0.7, then a vision
  LoRA; three-seed weight soups; one calibration temperature (1.226).
- **pplx-decider-flash-v1.3**: Qwen3.5-4B, distilled from the 27B with **KL at temperature 2 on full
  distributions, no cross-entropy** (its `decision_config.json`). Perplexity reports KL generalises better to
  held-out tasks than training on gold labels.
- Community **Decision Index** (leaderboard data of 19 Nov 2026), "Full" score = private same-skill + new-domain:
  karpo-31b 73.9, pplx-v1.2 68.0, **pplx-v1.3 67.7**, kev-27b 59.6, Jev 1.13 59.2, **flash-v1.3 55.1**,
  GLiDE-2B 43.2, GLiDE-0.4B 33.6, amber-150m 24.7, amber-68m 19.2, amber-32m 16.0, crivo-17m 11.7.
- What small students lose is knowledge and new domains, not classification: Retrieval & Classification
  27B 70.5 / flash 67.1 / amber-68m 63.4 / amber-32m 59.9, but Knowledge 60.7 / 49.0 / 9.9 / 7.7. **Atelier's
  decisions are classification over short states, which is exactly what survives shrinking.**
- Kev (open, Apache-2.0, 0.8B-27B, System One API), Clef (Cloudflare, Apache-2.0, GGUFs) and Jev (TypeSafe,
  closed) are the other System One models. Crivo and Amber small students are CC BY-NC: their recipes are useful,
  their weights are not for a product.
- llama.cpp upstream runs the `qwen35` architecture and has `/v1/systemone`, but only the decider's older protocol
  (v1); v1.2/v1.3's joint protocols are not implemented, so the decider runs here through its own Python code.

## Compressing the decider itself

- Hybrid models: Nemotron-Nano-2 (Mamba hybrid, 12B to 9B) scored layers by how much removing each changes the
  output and kept Mamba heads (pruning them hurt); SlimQwen (Qwen3-Next) found dropping the last quarter of
  layers then distilling beat cosine-picked middle layers. Nobody has published on a non-causal hybrid like the
  decider; this project measured it (README › Where the decision forms).
- Classification-tuned models are more prunable than generative ones (chip-tuning: probes at half depth match
  the full model; LEC: a middle layer of Qwen2.5-0.5B matches the last on safety classification).
- Quantization: 4-bit weight-only is near lossless at 4B+ with GPTQ/AWQ or llama.cpp k-quants with an
  importance matrix; keep Gated DeltaNet gates (`in_proj_a/b`, llama.cpp `ssm_alpha/beta`), `out_proj` and
  attention at higher precision (Unsloth's Qwen3.5 measurements). This project's own per-tensor statistics agree:
  `in_proj_a/b` have the highest kurtosis (up to 10) and the worst 4-bit rounding error, and the output projections
  have outlier columns at 36-41 sigma (`analysis/results/weights-*.txt`).
- Conclusion: a compressed decider (text-only, trimmed vocabulary, ~20 of flash's 32 layers, 4-bit) would be
  about 1.1-1.5 GB *(estimate)*: a good desktop teacher or a fallback, not a phone model.

## Distilling into a tiny student

- Loss: KL on the teacher's distribution (Perplexity: tau 2; Crivo: alpha(CE + 0.5 Brier) + (1-alpha) tau^2 KL).
  This project matches the calibrated distribution (what Atelier's cutoffs read) plus tau-2 KL.
- Teacher gap: a mid-size intermediate teacher helps very small students (Crivo: 400M to 32M/68M) *(unverified
  for this setting)*.
- Options: scoring options at marker tokens in one pass (GLiClass uni-encoder) is ~100x faster than one NLI pass
  per option at similar quality; reading options jointly can make answers depend on option order, so options are
  shuffled in training and Atelier's eval checks order flips.
- Calibration: small students are overconfident (board ECE 0.08-0.15 vs 0.04 for the 27B); fit temperatures per
  decision, and keep the rule-first contract so "unsure" stays cheap.
- Backbones (permissive, multilingual, small): **mmBERT-small** (MIT, 22 x 384, 42M non-embedding, 256k vocab,
  8k context; best XNLI in its class), granite-embedding-97m-multilingual-r2 (Apache-2.0, already pruned to
  12 layers), multilingual-e5-small (MIT, 512 tokens). Ettin/ModernBERT are English only; EuroBERT and
  LFM2.5-Encoder cost 3-4x the compute (and LFM's licence caps revenue); jina v5 nano is non-commercial.
- Vocabulary trimming after or before fine-tuning is near lossless if done right; for BPE tokenizers the kept set
  must be closed under merges or text gets split into "orphan" pieces (The Vocabulary Tax, Nov 2026). This
  project trims with merge closure and byte fallback: 100% round-trip, +3-11% length.

## Runtimes

| Runtime | Fit | Notes |
|---|---|---|
| onnxruntime-node 1.31 | Electron main/utility process | N-API, no Electron rebuild; win-x64 DLL 32 MB (DirectML parts optional); no Intel-Mac binary in 1.31 (use onnxruntime-web WASM there) |
| onnxruntime-web | Intel Macs, renderer | WASM 15 MB (3.8 MB gzip); slow without threads |
| ONNX Runtime Mobile | iOS, Android | same model file; reduced-ops builds shrink the binary |
| ExecuTorch, LiteRT | mobile | smaller runtimes, no Node binding |
| llama.cpp | everywhere | runs BERT/ModernBERT embeddings; custom option heads would run outside ggml |
| transformers.js v4 | browser/Node | runs ModernBERT/mmBERT in WebGPU or WASM |

Latency anchors: all-MiniLM-L6 (~11M non-embedding) runs in 7 ms on an iPhone 17 Pro and 21 ms on a OnePlus 12
(ExecuTorch, ~80 tokens). This project's 4-layer student: 0.8 ms at 16 tokens and 2.3 ms at 64 on a Ryzen 5
7600X (ONNX Runtime int8, 4 threads).

## Data with usable licences

MASSIVE (CC BY 4.0, 51 locales incl. nl-NL), WildChat-1M/4.8M (ODC-BY), Mind2Web (CC BY 4.0), FineWeb-2 (ODC-BY,
for buttons and titles at scale). Avoid LMSYS-Chat-1M (no redistribution), WebLINX (NC), MS MARCO/ORCAS (research
only).

## Sources

Perplexity Decider blog and model cards (huggingface.co/perplexity-ai); Decision Index (huggingface.co/spaces/
multimodalart/jev-decision-index); Crivo (arXiv 2610.30970); Nemotron-Nano-2 (2508.14444); SlimQwen (2605.08738);
Jet-Nemotron (2508.15884); chip-tuning (2410.06541); LEC (2412.13435); ShortGPT (2403.03853); Gromov et al.
(2403.17887); Unsloth Qwen3.5 GGUF benchmarks; ParetoQ (2502.02631); mmBERT (2509.06888); GLiClass (2508.07662);
vocabtrimmer (2305.15020); The Vocabulary Tax (2611.19913); Rank-DistiLLM (2405.07920).
