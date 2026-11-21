// Parity with the Python reference: token ids (format.ts vs student/format.py) and answers (ONNX in Node vs
// PyTorch). Reads a JSONL of {request, expect_ids?, expect_answers?} written by student/parity_dump.py.
//   node --experimental-strip-types src/parity.ts <model dir> <dump.jsonl>

import { readFileSync } from 'node:fs'
import { NanoDecider, type Answer } from './decider.ts'

const [dir, dump, file] = process.argv.slice(2)
const decider = await NanoDecider.load(dir, { ...(file ? { file } : {}), guard: false, open: 'answer' })
const rows = readFileSync(dump, 'utf8').split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l))
let maxDiff = 0
let flips = 0
let n = 0
let idMismatches = 0
const times: number[] = []
for (const row of rows) {
  const ids = decider.ids(row.request)
  for (const [key, expected] of Object.entries(row.expect_ids as Record<string, number[]>)) {
    if (JSON.stringify(ids[key]) !== JSON.stringify(expected)) {
      if (idMismatches++ < 3) console.log('ids differ:', key, JSON.stringify(row.request.state).slice(0, 80), '\n  js', ids[key].slice(0, 30).join(','), '\n  py', expected.slice(0, 30).join(','))
    }
  }
  const t0 = performance.now()
  const out = await decider.decide(row.request)
  times.push(performance.now() - t0)
  for (const [key, expected] of Object.entries(row.expect_answers as Record<string, Answer>)) {
    const got = out.answers[key]
    n++
    if (got.type === 'noul' && expected.type === 'noul') {
      maxDiff = Math.max(maxDiff, Math.abs(got.noul - expected.noul))
      if (got.noul >= 0.5 !== expected.noul >= 0.5) flips++
    } else if (got.type !== 'noul' && expected.type !== 'noul') {
      for (const k of Object.keys(expected.probabilities)) maxDiff = Math.max(maxDiff, Math.abs(got.probabilities[k] - expected.probabilities[k]))
      if ('choice' in got && 'choice' in expected && got.choice !== expected.choice) flips++
    }
  }
}
times.sort((a, b) => a - b)
console.log(
  JSON.stringify({
    questions: n,
    token_id_mismatches: idMismatches,
    max_abs_prob_diff: Number(maxDiff.toFixed(5)),
    top_answer_flips: flips,
    median_ms_per_request: Number(times[Math.floor(times.length / 2)].toFixed(2)),
    p90_ms: Number(times[Math.floor(times.length * 0.9)].toFixed(2)),
  }),
)
