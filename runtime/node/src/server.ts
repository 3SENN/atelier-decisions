/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// A System One server on 127.0.0.1 around the on-device decider, so Atelier (or its eval) can use it as
// "your own server":
//   node --experimental-strip-types src/server.ts <nano dir> [port] [threads] [model file] [decline|answer]
//        [--mini <mini runtime dir>] [--mini-only] [--mini-first skill,open] [--llama-server <path>] [--mini-threads <n>]
//        [--mini-gpu-layers <n>] [--step <atelier-step dir>] [--vision <atelier-vision dir>]
// With --mini, atelier-nano answers first and whatever it declines or is unsure about goes to atelier-mini
// (llama.cpp); --mini-only serves the mini alone (then <nano dir> may be "-"). --mini-gpu-layers 99 runs the mini on
// the GPU (Metal on a Mac). With --vision or --step and "-" as the nano dir, only those are served. The address itself (http://127.0.0.1:<port>/) is a playground page for trying questions,
// and /lab runs whole processes (docs/processes.md): ready-made or set up by a language model, over mock pages or a
// real browser tab. With --vision, POST /v1/look answers System One questions about a picture, POST /v1/embed gives
// the vectors to search pictures by, and /look is a page for trying both (docs/vision-model.md).

import { readFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { CascadeDecider } from './cascade.ts'
import { NanoDecider } from './decider.ts'
import { fixedKind, optionLabels, type Question } from './format.ts'
import { MiniDecider } from './mini.ts'
import { labRoute } from './process/labServer.ts'
import { StepDecider } from './step/model.ts'
import type { Question as VisionQuestion } from './vision/format.ts'
import { VisionDecider } from './vision/model.ts'
import { decodePng } from './vision/png.ts'
import type { Picture } from './vision/preprocess.ts'
import { probabilities, verdict } from './readers.ts'

const positional: string[] = []
const flags: Record<string, string | true> = {}
const argv = process.argv.slice(2)
for (let i = 0; i < argv.length; i++) {
  const a = argv[i] ?? ''
  if (!a.startsWith('--')) positional.push(a)
  else if (a === '--mini-only') flags[a.slice(2)] = true
  else flags[a.slice(2)] = argv[++i] ?? ''
}
const [dir, port = '8011', threads = '4', file = 'model.emb4.int8.onnx', open = 'decline'] = positional
const miniDir = typeof flags.mini === 'string' ? flags.mini : null
if (!dir || (flags['mini-only'] && !miniDir)) {
  console.error('usage: server.ts <nano dir> [port] [threads] [model file] [decline|answer] [--mini <dir>] [--mini-only] [--llama-server <path>] [--mini-threads <n>]')
  process.exit(2)
}
const mini = miniDir
  ? await MiniDecider.load(miniDir, {
      server: typeof flags['llama-server'] === 'string' ? flags['llama-server'] : undefined,
      threads: Number(flags['mini-threads'] ?? threads),
      gpuLayers: Number(flags['mini-gpu-layers'] ?? 0),
    })
  : null
const nano = flags['mini-only'] || dir === '-' ? null : await NanoDecider.load(dir, { threads: Number(threads), file, open: open === 'answer' ? 'answer' : 'decline' })
const miniFirst = typeof flags['mini-first'] === 'string' ? flags['mini-first'].split(',').filter(Boolean) : []
/** A request's picture: a PNG file, or pixels already decoded ({pixels, width, height, channels}), both base64. */
function pictureOf(p: { png?: string; pixels?: string; width?: number; height?: number; channels?: number } | undefined): Picture | undefined {
  if (!p) return undefined
  if (typeof p.png === 'string') return decodePng(new Uint8Array(Buffer.from(p.png, 'base64')))
  if (typeof p.pixels !== 'string' || !p.width || !p.height || (p.channels !== 3 && p.channels !== 4)) throw new Error('a picture is {png} or {pixels, width, height, channels: 3 or 4}')
  return { data: new Uint8Array(Buffer.from(p.pixels, 'base64')), width: p.width, height: p.height, channels: p.channels }
}
const stepModel = typeof flags.step === 'string' ? await StepDecider.load(flags.step, { threads: Number(threads) }) : null
const visionModel = typeof flags.vision === 'string' ? await VisionDecider.load(flags.vision, { threads: Number(threads) }) : null
const decider = nano && mini ? new CascadeDecider(nano, mini, { miniFirst: miniFirst as never }) : (nano ?? mini)
if (!decider && !stepModel && !visionModel) throw new Error('no model')
const name = nano && mini ? 'atelier-nano+mini' : nano ? 'atelier-nano' : mini ? mini.name : visionModel ? 'atelier-vision' : 'atelier-step'

const server = createServer(async (req, res) => {
  const send = (status: number, body: unknown): void => {
    res.writeHead(status, { 'content-type': 'application/json' })
    res.end(JSON.stringify(body))
  }
  try {
    if (req.method === 'POST' && req.url === '/v1/step') {
      // atelier-step: one browser step (goal, values, what was done, page, elements) in one pass.
      if (!stepModel) return send(404, { error: 'no step model (start with --step <dir>)' })
      const chunks: Buffer[] = []
      for await (const chunk of req) chunks.push(chunk as Buffer)
      return send(200, await stepModel.decide(JSON.parse(Buffer.concat(chunks).toString('utf8'))))
    }
    if (req.method === 'POST' && (req.url === '/v1/look' || req.url === '/v1/embed')) {
      // atelier-vision. look: {picture or vector, questions, keep?: true to get the picture's vector back}.
      // embed: {texts?: [...], picture?} gives unit vectors: a text's nearest pictures are the ones it describes.
      if (!visionModel) return send(404, { error: 'no vision model (start with --vision <dir>)' })
      const chunks: Buffer[] = []
      for await (const chunk of req) chunks.push(chunk as Buffer)
      const body = JSON.parse(Buffer.concat(chunks).toString('utf8'))
      const picture = pictureOf(body.picture)
      if (req.url === '/v1/embed') {
        const texts: string[] = Array.isArray(body.texts) ? body.texts : []
        const vectors = []
        for (const text of texts) vectors.push(visionModel.unreadable(text) ? null : Array.from(await visionModel.text(text)))
        return send(200, { model: 'atelier-vision', texts: vectors, picture: picture ? Array.from(await visionModel.image(picture)) : undefined })
      }
      const vector = Array.isArray(body.vector) ? Float32Array.from(body.vector as number[]) : undefined
      const out = await visionModel.look({ picture, vector }, (body.questions ?? {}) as Record<string, VisionQuestion>)
      return send(200, { model: 'atelier-vision', answers: out.answers, vector: body.keep ? Array.from(out.vector) : undefined, ms: Math.round(out.ms * 100) / 100 })
    }
    if (req.method === 'GET' && req.url === '/look') {
      // A page for trying atelier-vision: pictures in, the answers a question may have, how well each matches.
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
      return res.end(readFileSync(new URL('../look.html', import.meta.url)))
    }
    if (!decider) {
      // Only pictures or steps are served: the address itself goes to the page for trying pictures.
      if (req.method === 'GET' && req.url === '/' && visionModel) {
        res.writeHead(302, { location: '/look' })
        return res.end()
      }
      return send(404, { error: 'no decision model for text (start with a nano dir)' })
    }
    if (await labRoute(req, res, (request) => decider.decide(request), name)) return
    if (req.method === 'GET' && (req.url === '/' || req.url === '/playground')) {
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
      return res.end(readFileSync(new URL('../playground.html', import.meta.url)))
    }
    if (req.method === 'GET' && req.url?.endsWith('/models')) return send(200, { data: [{ id: name, object: 'model', owned_by: 'local' }] })
    const playground = req.url === '/v1/playground'
    if (req.method !== 'POST' || (!playground && !req.url?.endsWith('/systemone'))) return send(404, { error: 'not found' })
    const chunks: Buffer[] = []
    for await (const chunk of req) chunks.push(chunk as Buffer)
    const body = JSON.parse(Buffer.concat(chunks).toString('utf8'))
    const t0 = performance.now()
    const out = await decider.decide({ state: body.state, questions: body.questions })
    const ms = Math.round((performance.now() - t0) * 100) / 100
    if (!playground) return send(200, { model: name, ...out, ms })
    // The playground also shows what Atelier would do with each answer: its reader's verdict, or null (the rule decides).
    const verdicts = Object.fromEntries(
      Object.entries(body.questions as Record<string, Question>).map(([key, q]) => {
        const labels = optionLabels(q)
        const a = out.answers[key]
        const p = a ? probabilities(a, labels) : null
        return [key, p ? verdict(fixedKind(q) ?? 'open', labels, p) : null]
      }),
    )
    send(200, { model: name, ...out, verdicts, ms })
  } catch (err) {
    send(422, { error: err instanceof Error ? err.message : String(err) })
  }
}).listen(Number(port), '127.0.0.1', () => console.log(`${name} on http://127.0.0.1:${port}${decider ? '/v1/systemone' : visionModel ? '/look' : '/v1/step'}`))

const stop = async (): Promise<void> => {
  server.close()
  await Promise.all([nano?.release(), mini?.release(), visionModel?.release()])
  process.exit(0)
}
process.on('SIGINT', stop)
process.on('SIGTERM', stop)
