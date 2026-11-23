# pplx-decider-flash-v1.3: 4.540B parameters in 724 tensors

| Group | Params | Share | BF16 size | Atelier text decisions |
|---|---:|---:|---:|---|
| decoder: SwiGLU MLP | 2,264.9 M | 49.9% | 4.22 GiB | needed (compute) |
| decoder: Gated DeltaNet mixer | 1,011.4 M | 22.3% | 1.88 GiB | needed (compute) |
| token embeddings (248k vocab) | 635.7 M | 14.0% | 1.18 GiB | needed only for tokens Atelier text uses |
| vision tower (27 ViT blocks + merger) | 333.5 M | 7.3% | 0.62 GiB | unused for text-only decisions |
| decoder: full-attention mixer (non-causal) | 293.6 M | 6.5% | 0.55 GiB | needed (compute) |
| readout head [255 codes x hidden] | 0.7 M | 0.0% | 0.00 GiB | needed |
| decoder: norms | 0.2 M | 0.0% | 0.00 GiB | needed (compute) |
| final norm | 0.0 M | 0.0% | 0.00 GiB | needed |

Layers: 32 (24 Gated DeltaNet, 8 full attention, every 4th). hidden=2560, mlp=9216
- one Gated DeltaNet layer: 112.9 M params (mixer 42.1 M, MLP 70.8 M)
- one full-attention layer: 107.5 M params (mixer 36.7 M, MLP 70.8 M)

Compute path for a text decision (decoder + norm + readout): 3.571B params = 6.7 GiB BF16, 1.7 GiB at 4 bits
