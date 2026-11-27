/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// atelier-vision's optional download (feature.ts): every file with its size and SHA-256, from the atelier-decisions
// project's releases/vision-v0.1/MANIFEST.json (its tools/vision_dist.py writes this file). The base is the version's
// tag on Hugging Face, so these files never change under it; a new version is a new manifest here.

import type { MiniManifest } from '../deviceMini'

export const VISION_MANIFEST: MiniManifest = {
  version: 'vision-v0.1',
  base: 'https://huggingface.co/3SENN/atelier-decisions/resolve/vision-v0.1/releases/vision-v0.1/model/',
  model: [
    { path: 'image.q8.onnx', url: 'image.q8.onnx', size: 105790485, sha256: '22d1daaa9eca6c3a41686ac6dd81d5ce817cf986264d8f47ce37ee129001c0ac' },
    { path: 'text.q8.emb4.onnx', url: 'text.q8.emb4.onnx', size: 147944376, sha256: '63878a737f6b225e9b1a811a9151bc7f400849b728dea23e2f2c7468cef3ace1' },
    { path: 'tokenizer.json', url: 'tokenizer.json', size: 6263926, sha256: 'c3179fbc32d05d7bf48c38c29cf348302566455ac97afa002dafebfb4664440c' },
    { path: 'tokenizer_config.json', url: 'tokenizer_config.json', size: 47164, sha256: '14afe629fe4959b9e0d51e1852b8d9f7ad074f90a1a7125a4fcdd17f06e78fc8' },
    { path: 'vision.json', url: 'vision.json', size: 1487, sha256: 'a549a480e118f8435bc563d1edd8631ca86edf5b12157f1ba5c31c752ac21f7d' },
  ],
  servers: {},
}
