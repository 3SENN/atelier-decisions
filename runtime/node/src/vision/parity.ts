/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// Checks this runtime against vision/export.py's parity.jsonl: the same tower input from the same pixels, the same
// token ids from the same text, the same vectors from the same model, the same answers (and refusals) to the same
// questions.
//   node --experimental-strip-types src/vision/parity.ts <exported dir>

import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import type { Question } from './format.ts'
import { dot, VisionDecider } from './model.ts'
import { checksum, preprocess, type Picture } from './preprocess.ts'

const dir = process.argv[2]
if (!dir) throw new Error('usage: parity.ts <exported dir>')
const model = await VisionDecider.load(dir)
const rows = readFileSync(join(dir, 'parity.jsonl'), 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l))

const vectors: Float32Array[] = []
let pictures = 0
let worstInput = 0
let worstPicture = 1
let texts = 0
let sameIds = 0
let sameGuard = 0
let worstText = 1
let looks = 0
let sameAnswers = 0
let worstP = 0
for (const r of rows) {
  if (r.picture) {
    const picture: Picture = { data: new Uint8Array(Buffer.from(r.picture.pixels, 'base64')), width: r.picture.w, height: r.picture.h, channels: r.picture.channels }
    worstInput = Math.max(worstInput, Math.abs(checksum(preprocess(picture, model.size)) - r.checksum) / Math.max(1, Math.abs(r.checksum)))
    const v = await model.image(picture)
    vectors.push(v)
    worstPicture = Math.min(worstPicture, dot(v, Float32Array.from(r.embedding)))
    pictures++
  } else if (typeof r.text === 'string') {
    const { ids, cut } = model.ids(r.text)
    if (JSON.stringify(ids) === JSON.stringify(r.ids) && cut === r.cut) sameIds++
    else console.log(`ids differ: ${JSON.stringify(r.text.slice(0, 60))}`)
    if (model.unreadable(r.text) === r.unreadable) sameGuard++
    else console.log(`guard differs: ${JSON.stringify(r.text.slice(0, 60))}: ${model.unreadable(r.text)} / ${r.unreadable}`)
    worstText = Math.min(worstText, dot(await model.text(r.text), Float32Array.from(r.embedding)))
    texts++
  } else if (r.look) {
    const out = await model.look({ vector: vectors[r.look.picture] }, r.look.questions as Record<string, Question>)
    for (const [key, want] of Object.entries(r.answers as Record<string, Record<string, unknown>>)) {
      const got = out.answers[key] as Record<string, unknown> | undefined
      looks++
      const p = (a: Record<string, unknown> | undefined): number[] => (a?.type === 'noul' ? [a.noul as number] : a?.probabilities ? Object.values(a.probabilities as Record<string, number>) : [])
      const same = got?.type === want.type && (want.type === 'refusal' ? got?.reason === want.reason : want.type === 'noul' || got?.choice === want.choice)
      const diff = Math.max(0, ...p(want).map((x, i) => Math.abs(x - (p(got)[i] ?? 9))))
      worstP = Math.max(worstP, diff)
      if (same && diff < 1e-3) sameAnswers++
      else console.log(`answer differs: ${key}: ${JSON.stringify(got)} / ${JSON.stringify(want)}`)
    }
  }
}
console.log(`${pictures} pictures: tower input within ${worstInput.toExponential(1)} of the export's, lowest cosine to its vector ${worstPicture.toFixed(6)}`)
console.log(`${sameIds}/${texts} texts encode to the same ids, ${sameGuard}/${texts} are read or declined alike; lowest cosine to the export's vector ${worstText.toFixed(6)}`)
console.log(`${sameAnswers}/${looks} answers the same (largest difference in a probability ${worstP.toExponential(1)})`)
await model.release()
if (sameIds !== texts || sameGuard !== texts || sameAnswers !== looks || worstPicture < 0.9999 || worstText < 0.9999 || worstInput > 1e-5) process.exit(1)
