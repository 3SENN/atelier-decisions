// Latency of the on-device decider on Atelier's real requests, per request (all its questions), cold and warm.
//   node --experimental-strip-types src/bench.ts <model dir> <requests.jsonl> [file] [threads...]

import { readFileSync } from 'node:fs'
import { NanoDecider } from './decider.ts'

const [dir, requestsFile, file = 'model.emb4.int8.onnx', ...threadArgs] = process.argv.slice(2)
const requests = readFileSync(requestsFile, 'utf8').split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l).request)
const pct = (xs: number[], p: number): number => xs[Math.min(xs.length - 1, Math.floor(xs.length * p))]

for (const threads of (threadArgs.length ? threadArgs : ['1', '4']).map(Number)) {
  const t0 = performance.now()
  const decider = await NanoDecider.load(dir, { file, threads, open: 'answer' })
  const loadMs = performance.now() - t0
  const c0 = performance.now()
  await decider.decide(requests[0])
  const coldMs = performance.now() - c0
  const single: number[] = []
  const all: number[] = []
  for (let round = 0; round < 3; round++) {
    for (const request of requests) {
      const s = performance.now()
      await decider.decide(request)
      const ms = performance.now() - s
      all.push(ms)
      if (Object.keys(request.questions).length === 1) single.push(ms)
    }
  }
  single.sort((a, b) => a - b)
  all.sort((a, b) => a - b)
  console.log(JSON.stringify({
    file, threads, load_ms: Math.round(loadMs), first_request_ms: Math.round(coldMs),
    one_question: { median: +pct(single, 0.5).toFixed(2), p90: +pct(single, 0.9).toFixed(2), p99: +pct(single, 0.99).toFixed(2) },
    any_request: { median: +pct(all, 0.5).toFixed(2), p90: +pct(all, 0.9).toFixed(2) },
    rss_mb: Math.round(process.memoryUsage().rss / 1e6),
  }))
}
// onnxruntime-node's thread pools can keep the process alive after the last session; we're done.
process.exit(0)
