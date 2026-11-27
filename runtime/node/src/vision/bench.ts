/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// What atelier-vision costs on this computer: loading it, a picture's vector, a new text's vector, a question whose
// texts it has seen before, and the memory it holds. On the version's own parity pictures and texts.
//   node --experimental-strip-types src/vision/bench.ts <exported dir> [threads...]

import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import type { Question } from './format.ts'
import { VisionDecider } from './model.ts'
import type { Picture } from './preprocess.ts'

const [dir, ...threadArgs] = process.argv.slice(2)
if (!dir) throw new Error('usage: bench.ts <exported dir> [threads...]')
const rows = readFileSync(join(dir, 'parity.jsonl'), 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l))
const pictures: Picture[] = rows.filter((r) => r.picture).map((r) => ({ data: new Uint8Array(Buffer.from(r.picture.pixels, 'base64')), width: r.picture.w, height: r.picture.h, channels: r.picture.channels }))
const texts: string[] = rows.filter((r) => typeof r.text === 'string' && r.text && !r.unreadable).map((r) => r.text)
const questions = rows.find((r) => r.look)?.look.questions as Record<string, Question>
const median = (xs: number[]): number => +[...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]!.toFixed(1)

for (const threads of (threadArgs.length ? threadArgs : ['2', '4']).map(Number)) {
  const t0 = performance.now()
  const model = await VisionDecider.load(dir, { threads, keepTexts: 0 })
  const loadMs = performance.now() - t0
  const c0 = performance.now()
  const first = await model.image(pictures[0]!)
  const coldMs = performance.now() - c0
  const picture: number[] = []
  const text: number[] = []
  const asked: number[] = []
  for (let round = 0; round < 3; round++) {
    for (const p of pictures) {
      const s = performance.now()
      await model.image(p)
      picture.push(performance.now() - s)
    }
    for (const t of texts) {
      const s = performance.now()
      await model.text(t)
      text.push(performance.now() - s)
    }
  }
  // A question about a picture whose vector is kept, with texts seen before: what sorting many pictures costs each.
  const warm = await VisionDecider.load(dir, { threads })
  await warm.look({ vector: first }, questions)
  for (let i = 0; i < 200; i++) {
    const s = performance.now()
    await warm.look({ vector: first }, questions)
    asked.push(performance.now() - s)
  }
  console.log(JSON.stringify({
    threads, load_ms: Math.round(loadMs), first_picture_ms: Math.round(coldMs), picture_ms: median(picture), new_text_ms: median(text),
    question_again_ms: +median(asked).toFixed(3), rss_mb: Math.round(process.memoryUsage().rss / 1e6),
  }))
  await model.release()
  await warm.release()
}
// onnxruntime-node's thread pools can keep the process alive after the last session; we're done.
process.exit(0)
