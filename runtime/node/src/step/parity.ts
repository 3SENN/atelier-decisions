/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// Checks this runtime against step/export.py's parity.jsonl: the same token ids and marker positions from the same
// step, and the same logits from the same model.
//   node --experimental-strip-types src/step/parity.ts <exported dir>

import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { encodeStep, type StepInput } from './format.ts'
import { StepDecider } from './model.ts'

const dir = process.argv[2]
if (!dir) throw new Error('usage: parity.ts <exported dir>')
// parity.jsonl holds the int8 file's logits.
const model = await StepDecider.load(dir, { file: 'model.int8.onnx', raw: true })
const rows = readFileSync(join(dir, 'parity.jsonl'), 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l))
let same = 0
let worst = 0
for (const r of rows) {
  const enc = encodeStep(model.tok, r.step as StepInput)
  const ok = JSON.stringify(enc.ids) === JSON.stringify(r.ids) && JSON.stringify(enc.els) === JSON.stringify(r.els) && JSON.stringify(enc.vals) === JSON.stringify(r.vals)
  if (ok) same++
  else console.log(`ids differ: ${r.step.goal.slice(0, 60)}`)
  const out = await model.logits(r.ids, r.els, r.vals)
  const diff = Math.max(...out.op.map((x, i) => Math.abs(x - r.op[i])), ...out.el.map((x, i) => Math.abs(x - r.el[i])), ...out.val.flat().map((x, i) => Math.abs(x - r.val.flat()[i])))
  worst = Math.max(worst, diff)
}
console.log(`${same}/${rows.length} steps encode to the same ids and markers; largest logit difference ${worst.toExponential(2)}`)
