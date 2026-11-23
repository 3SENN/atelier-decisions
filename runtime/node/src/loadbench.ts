/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// Cold start and memory of one configuration (run once per process): load, first answer, warm answers, RSS.
//   node --experimental-strip-types src/loadbench.ts <model dir> <file> <arena on|off>
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import * as ort from 'onnxruntime-node'
const [dir, file, arena] = process.argv.slice(2)
const rss0 = process.memoryUsage().rss
const t0 = performance.now()
const session = await ort.InferenceSession.create(join(dir, file), {
  executionProviders: ['cpu'], intraOpNumThreads: 2, graphOptimizationLevel: file.endsWith('.ort') ? 'disabled' : 'all',
  enableCpuMemArena: arena === 'on', enableMemPattern: arena === 'on',
})
const loadMs = performance.now() - t0
const ids = new BigInt64Array(64).fill(500n)
const feeds = { input_ids: new ort.Tensor('int64', ids, [1, 64]), attention_mask: new ort.Tensor('int64', new BigInt64Array(64).fill(1n), [1, 64]) }
const f0 = performance.now(); await session.run(feeds); const firstMs = performance.now() - f0
const times: number[] = []
for (let i = 0; i < 50; i++) { const s = performance.now(); await session.run(feeds); times.push(performance.now() - s) }
times.sort((a, b) => a - b)
console.log(JSON.stringify({ file, arena, load_ms: Math.round(loadMs), first_ms: +firstMs.toFixed(1), median_ms: +times[25]!.toFixed(2), rss_mb_added: Math.round((process.memoryUsage().rss - rss0) / 1e6) }))
await session.release()
process.exit(0)
