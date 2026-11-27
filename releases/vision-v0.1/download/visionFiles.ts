/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// atelier-vision's files, apart from its runtime (vision.ts), so the main process can check a folder without loading
// ONNX Runtime: only the decision helper does that.

/** The files a usable vision model folder holds. */
export const VISION_FILES = ['image.q8.onnx', 'text.q8.emb4.onnx', 'tokenizer.json', 'tokenizer_config.json', 'vision.json'] as const
