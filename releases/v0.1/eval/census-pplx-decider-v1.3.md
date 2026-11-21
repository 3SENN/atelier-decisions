# pplx-decider-v1.3: 26.087B parameters in 1184 tensors

| Group | Params | Share | BF16 size | Atelier text decisions |
|---|---:|---:|---:|---|
| decoder: SwiGLU MLP | 17,112.8 M | 65.6% | 31.88 GiB | needed (compute) |
| decoder: Gated DeltaNet mixer | 5,562.1 M | 21.3% | 10.36 GiB | needed (compute) |
| decoder: full-attention mixer (non-causal) | 1,677.7 M | 6.4% | 3.13 GiB | needed (compute) |
| token embeddings (248k vocab) | 1,271.4 M | 4.9% | 2.37 GiB | needed only for tokens Atelier text uses |
| vision tower (27 ViT blocks + merger) | 460.7 M | 1.8% | 0.86 GiB | unused for text-only decisions |
| readout head [255 codes x hidden] | 1.3 M | 0.0% | 0.00 GiB | needed |
| decoder: norms | 0.7 M | 0.0% | 0.00 GiB | needed (compute) |
| final norm | 0.0 M | 0.0% | 0.00 GiB | needed |

Layers: 64 (48 Gated DeltaNet, 16 full attention, every 4th). hidden=5120, mlp=17408
- one Gated DeltaNet layer: 383.3 M params (mixer 115.9 M, MLP 267.4 M)
- one full-attention layer: 372.3 M params (mixer 104.9 M, MLP 267.4 M)

Compute path for a text decision (decoder + norm + readout): 24.355B params = 45.4 GiB BF16, 11.3 GiB at 4 bits
