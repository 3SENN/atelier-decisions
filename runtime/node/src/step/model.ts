/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// atelier-step on ONNX Runtime (CPU): one pass per browser step gives the operation, a score for every control and,
// for every field, which of the named values goes in.

import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { Tokenizer } from '@huggingface/tokenizers'
import * as ort from 'onnxruntime-node'
import { CLICKABLE, EDITABLE, encodeStep, OPS, type StepInput, type StepOp, type StepTokenize } from './format.ts'

export interface StepAnswer {
  /** Probability of each operation. */
  op: Record<StepOp, number>
  /** For every element of the input: its probability as the click target (among clickable ones) or the field to type
   *  into (among editable ones); null for text lines and for elements cut off the end of a long page. */
  click: Array<number | null>
  type: Array<number | null>
  /** For every editable element: probabilities over [none of these, value 1, value 2, …]; null otherwise. */
  values: Array<number[] | null>
  tokens: number
  ms: number
}

const softmax = (xs: number[]): number[] => {
  const m = Math.max(...xs)
  const e = xs.map((x) => Math.exp(x - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((x) => x / s)
}

export class StepDecider {
  private readonly session: ort.InferenceSession
  readonly tok: StepTokenize
  /** step/calibrate.py's temperatures (step.json), so a probability of 0.9 is right about 9 times in 10 on held-out steps. */
  private readonly temps: { op: number; el: number; val: number }

  private constructor(session: ort.InferenceSession, tok: StepTokenize, temps: { op: number; el: number; val: number }) {
    this.session = session
    this.tok = tok
    this.temps = temps
  }

  static async load(dir: string, options: { threads?: number; file?: string; raw?: boolean } = {}): Promise<StepDecider> {
    const json = (name: string): object => JSON.parse(readFileSync(join(dir, name), 'utf8'))
    const tokenizer = new Tokenizer(json('tokenizer.json'), json('tokenizer_config.json'))
    const one = (token: string): number => {
      const ids = tokenizer.encode(token, { add_special_tokens: false }).ids
      if (ids.length !== 1 || ids[0] === undefined) throw new Error(`${token} is not one token`)
      return ids[0]
    }
    const tok: StepTokenize = { encode: (text) => tokenizer.encode(text, { add_special_tokens: false }).ids, id: one, bos: one('<bos>'), eos: one('<eos>') }
    // The smaller file (4-bit embedding table) when it is there; it gave the same answers as full precision (step/export.py).
    const file = options.file ?? (existsSync(join(dir, 'model.emb4.int8.onnx')) ? 'model.emb4.int8.onnx' : 'model.int8.onnx')
    const session = await ort.InferenceSession.create(join(dir, file), {
      executionProviders: ['cpu'],
      intraOpNumThreads: options.threads ?? 2,
      graphOptimizationLevel: 'all',
    })
    const info = existsSync(join(dir, 'step.json')) ? (json('step.json') as { temperatures?: { op: number; el: number; val: number } }) : {}
    return new StepDecider(session, tok, options.raw ? { op: 1, el: 1, val: 1 } : (info.temperatures ?? { op: 1, el: 1, val: 1 }))
  }

  /** The raw logits for already-encoded input (parity checks use this). */
  async logits(ids: number[], els: number[], vals: number[]): Promise<{ op: number[]; el: number[]; val: number[][] }> {
    const t = (xs: number[], dims: number[]) => new ort.Tensor('int64', BigInt64Array.from(xs.map((x) => BigInt(x))), dims)
    const out = await this.session.run({ input_ids: t(ids, [1, ids.length]), attention_mask: t(ids.map(() => 1), [1, ids.length]), els: t(els, [els.length]), vals: t(vals, [vals.length]) })
    const flat = (name: string) => Array.from(out[name]!.data as Float32Array)
    const val = flat('val')
    const m = vals.length
    return { op: flat('op'), el: flat('el'), val: els.map((_, i) => val.slice(i * m, (i + 1) * m)) }
  }

  async decide(input: StepInput): Promise<StepAnswer> {
    const t0 = performance.now()
    const enc = encodeStep(this.tok, input)
    const n = input.elements.length
    const empty = (): Array<null> => Array.from({ length: n }, () => null)
    if (enc.els.length === 0) {
      return { op: { click: 0, type: 0, done: 0.5, blocked: 0.5 }, click: empty(), type: empty(), values: empty(), tokens: enc.ids.length, ms: performance.now() - t0 }
    }
    const raw = await this.logits(enc.ids, enc.els, enc.vals)
    const t = this.temps
    const op = softmax(raw.op.map((x) => x / t.op))
    // The target is chosen among the controls of the operation's kind, as in training.
    const among = (kinds: Set<string>) => {
      const idx = enc.elIndex.map((i, k) => (kinds.has(input.elements[i]!.role) ? k : -1)).filter((k) => k >= 0)
      const p = idx.length ? softmax(idx.map((k) => raw.el[k]! / t.el)) : []
      const out: Array<number | null> = empty()
      idx.forEach((k, j) => (out[enc.elIndex[k]!] = p[j]!))
      return out
    }
    const values: Array<number[] | null> = empty()
    enc.elIndex.forEach((i, k) => {
      if (EDITABLE.has(input.elements[i]!.role)) values[i] = softmax(raw.val[k]!.map((x) => x / t.val))
    })
    return {
      op: Object.fromEntries(OPS.map((o, i) => [o, op[i]!])) as Record<StepOp, number>,
      click: among(CLICKABLE),
      type: among(EDITABLE),
      values,
      tokens: enc.ids.length,
      ms: performance.now() - t0,
    }
  }
}
