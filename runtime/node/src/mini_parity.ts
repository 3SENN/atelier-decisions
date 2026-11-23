/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// Does the Node runner build atelier-mini's token ids exactly as pplx_decider's Encoder does?
//   node --experimental-strip-types src/mini_parity.ts <runtime dir from student/export_mini.py>

import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import type { DecisionRequest } from './format.ts'
import { MiniDecider } from './mini.ts'

const dir = process.argv[2]
if (!dir) {
  console.error('usage: mini_parity.ts <runtime dir>')
  process.exit(2)
}
const mini = await MiniDecider.load(dir, { serve: false })
let same = 0
let total = 0
for (const line of readFileSync(join(dir, 'parity.jsonl'), 'utf8').split('\n')) {
  if (!line.trim()) continue
  const { request, ids, mapped } = JSON.parse(line) as { request: DecisionRequest; ids: Record<string, number[]>; mapped?: Record<string, number[]> }
  for (const [key, q] of Object.entries(request.questions)) {
    const ours = mini.ids(request.state, q)
    const ref = ids[key] ?? []
    total++
    // With a trimmed vocabulary, the ids as sent must match Python's mapping too.
    const sentSame = !mapped || JSON.stringify(mini.sent(ours)) === JSON.stringify(mapped[key])
    if (ours.length === ref.length && ours.every((t, i) => t === ref[i]) && sentSame) same++
    else if (total - same <= 3) {
      const i = ours.findIndex((t, j) => t !== ref[j])
      console.log(`differs at token ${i} (${ours.length} vs ${ref.length}):`, ours.slice(Math.max(0, i - 3), i + 5), ref.slice(Math.max(0, i - 3), i + 5))
    }
  }
}
console.log(`identical token ids: ${same}/${total}`)
process.exit(same === total ? 0 : 1)
