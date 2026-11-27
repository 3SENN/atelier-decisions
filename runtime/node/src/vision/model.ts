/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// atelier-vision on ONNX Runtime (CPU): a picture becomes a vector once; every question about it after that is a few
// short texts through the text tower (kept, since the same answers are asked of many pictures) and a dot product.
// The same vectors find a picture by a description: embed the description, take the nearest pictures.
// vision/runtime.py is the same thing in Python; src/vision/parity.ts checks this one against the exported fixtures.

import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { Tokenizer } from '@huggingface/tokenizers'
import * as ort from 'onnxruntime-node'
import { answer, probabilities, RAW, type Answer, type Calibration } from './decide.ts'
import { clean, encodeText, labels, MAX_TOKENS, texts, unanswerable, type Question, type VisionTokenize } from './format.ts'
import { preprocess, type Picture } from './preprocess.ts'

/**
 * When the model can't read a text it declines instead of guessing, as atelier-nano does: the vocabulary is trimmed
 * to Latin-script languages, so other scripts fall apart into raw bytes.
 */
export interface Guard {
  min_latin: number
  max_bytes: number
  min_letters: number
}
export const DEFAULT_GUARD: Guard = { min_latin: 0.7, max_bytes: 0.1, min_letters: 8 }
// Latin letters by code point, so Python and JavaScript agree on every character (vision/runtime.py LATIN).
const LATIN: Array<[number, number]> = [[0x41, 0x5a], [0x61, 0x7a], [0xaa, 0xaa], [0xba, 0xba], [0xc0, 0x24f], [0x1e00, 0x1eff], [0xff21, 0xff3a], [0xff41, 0xff5a]]
const isLatin = (ch: string): boolean => {
  const code = ch.codePointAt(0) ?? 0
  return LATIN.some(([lo, hi]) => lo <= code && code <= hi)
}

/** Why the model can't read this text, or null when it can. ids: its tokens before <eos> and padding. */
export function unreadable(text: string, ids: number[], byteIds: Set<number>, guard: Guard = DEFAULT_GUARD): string | null {
  const letters = text.match(/\p{L}/gu) ?? []
  if (letters.length >= guard.min_letters) {
    const latin = letters.filter(isLatin).length / letters.length
    if (latin < guard.min_latin) return `mostly not Latin script (${Math.floor(latin * 100 + 0.5)}% Latin)`
  }
  if (ids.length > 0) {
    const share = ids.filter((id) => byteIds.has(id)).length / ids.length
    if (share > guard.max_bytes) return `${Math.floor(share * 100 + 0.5)}% of the text is outside the vocabulary`
  }
  return null
}

interface VisionInfo {
  size: number
  scale: number
  bias: number
  files: { image: string; text: string }
  calibration?: Calibration
  guard?: Guard
}

export interface LookResult {
  answers: Record<string, Answer>
  /** The picture's vector (unit length): keep it to ask again later, or to search by. */
  vector: Float32Array
  ms: number
}

export class VisionDecider {
  private readonly images: ort.InferenceSession
  private readonly words: ort.InferenceSession
  readonly tok: VisionTokenize
  private readonly byteIds: Set<number>
  private readonly info: VisionInfo
  private readonly cal: Calibration
  private readonly guard: Guard
  private readonly kept = new Map<string, Float32Array>()
  private readonly keep: number

  private constructor(images: ort.InferenceSession, words: ort.InferenceSession, tok: VisionTokenize, byteIds: Set<number>, info: VisionInfo, raw: boolean, keep: number) {
    this.images = images
    this.words = words
    this.tok = tok
    this.byteIds = byteIds
    this.info = info
    this.cal = raw ? RAW : (info.calibration ?? RAW)
    this.guard = info.guard ?? DEFAULT_GUARD
    this.keep = keep
  }

  /** raw: SigLIP's own reading instead of the calibration (parity checks use the calibrated one). keepTexts: how many texts' vectors to keep. */
  static async load(dir: string, options: { threads?: number; raw?: boolean; keepTexts?: number } = {}): Promise<VisionDecider> {
    const json = (name: string): object => JSON.parse(readFileSync(join(dir, name), 'utf8'))
    const info = json('vision.json') as VisionInfo
    const spec = json('tokenizer.json') as { model: { vocab: Record<string, number> } }
    const tokenizer = new Tokenizer(spec, json('tokenizer_config.json'))
    const tok: VisionTokenize = { encode: (text) => tokenizer.encode(text, { add_special_tokens: false }).ids }
    const byteIds = new Set(Object.entries(spec.model.vocab).filter(([t]) => /^<0x[0-9A-F]{2}>$/.test(t)).map(([, id]) => id))
    const open = (file: string): Promise<ort.InferenceSession> =>
      ort.InferenceSession.create(join(dir, file), { executionProviders: ['cpu'], intraOpNumThreads: options.threads ?? 2, graphOptimizationLevel: 'all' })
    return new VisionDecider(await open(info.files.image), await open(info.files.text), tok, byteIds, info, options.raw ?? false, options.keepTexts ?? 4000)
  }

  /** Frees the ONNX sessions (see NanoDecider.release). */
  async release(): Promise<void> {
    await Promise.all([this.images.release(), this.words.release()])
  }

  get size(): number {
    return this.info.size
  }

  /** A picture's vector (unit length). */
  async image(picture: Picture): Promise<Float32Array> {
    const size = this.info.size
    const out = await this.images.run({ pixels: new ort.Tensor('float32', preprocess(picture, size), [1, 3, size, size]) })
    return Float32Array.from(out.embedding!.data as Float32Array)
  }

  ids(text: string): { ids: number[]; cut: boolean } {
    return encodeText(this.tok, text)
  }

  unreadable(text: string): string | null {
    const cleaned = clean(text)
    return unreadable(cleaned, this.tok.encode(cleaned), this.byteIds, this.guard)
  }

  /** A text's vector (unit length). */
  async text(text: string): Promise<Float32Array> {
    return (await this.read([text])).get(text)!
  }

  /**
   * Texts' vectors, the ones not read before through the text tower together: a question with 60 options is one pass,
   * not 60. Kept, since the same answers are asked of many pictures.
   */
  async read(wanted: string[], batch = 32): Promise<Map<string, Float32Array>> {
    const out = new Map<string, Float32Array>()
    const fresh: string[] = []
    for (const text of new Set(wanted)) {
      const hit = this.kept.get(text)
      if (hit) out.set(text, hit)
      else fresh.push(text)
    }
    for (let i = 0; i < fresh.length; i += batch) {
      const chunk = fresh.slice(i, i + batch)
      const ids = new BigInt64Array(chunk.length * MAX_TOKENS)
      chunk.forEach((text, row) => this.ids(text).ids.forEach((id, col) => (ids[row * MAX_TOKENS + col] = BigInt(id))))
      const run = await this.words.run({ input_ids: new ort.Tensor('int64', ids, [chunk.length, MAX_TOKENS]) })
      const data = run.embedding!.data as Float32Array
      const width = data.length / chunk.length
      chunk.forEach((text, row) => {
        const vector = Float32Array.from(data.subarray(row * width, (row + 1) * width))
        out.set(text, vector)
        if (this.keep > 0) {
          if (this.kept.size >= this.keep) this.kept.delete(this.kept.keys().next().value as string)
          this.kept.set(text, vector)
        }
      })
    }
    return out
  }

  /** Why a question isn't answered: its shape, a text the model can't read, or a text longer than it reads. */
  decline(q: Question): string | null {
    const why = unanswerable(q)
    if (why) return why
    for (const text of texts(q)) {
      if (text === null) continue
      const bad = this.unreadable(text)
      if (bad) return bad
      if (this.ids(text).cut) return 'a text is longer than the model reads (63 tokens)'
    }
    return null
  }

  /** System One answers for questions about one picture, given as pixels or as a vector kept from before. */
  async look(input: { picture?: Picture; vector?: Float32Array }, questions: Record<string, Question>): Promise<LookResult> {
    const t0 = performance.now()
    const vector = input.vector ?? (input.picture ? await this.image(input.picture) : null)
    if (!vector) throw new Error('A look needs a picture or a vector')
    const answers: Record<string, Answer> = {}
    const entries = Object.entries(questions ?? {}).map(([key, q]) => ({ key, q, why: this.decline(q) }))
    const read = await this.read(entries.flatMap((e) => (e.why ? [] : texts(e.q).filter((t): t is string => t !== null))))
    for (const { key, q, why } of entries) {
      if (why) {
        answers[key] = { type: 'refusal', reason: why }
        continue
      }
      const cos = texts(q).map((text) => (text === null ? null : dot(vector, read.get(text)!)))
      answers[key] = answer(q, labels(q), probabilities(q.type, cos, this.info.scale, this.info.bias, this.cal))
    }
    return { answers, vector, ms: performance.now() - t0 }
  }

  /** The pictures nearest to a description: indices into `vectors`, best first, with their cosine. */
  async search(query: string, vectors: Float32Array[], top = 10): Promise<Array<{ index: number; cos: number }>> {
    const q = await this.text(query)
    return vectors
      .map((v, index) => ({ index, cos: dot(v, q) }))
      .sort((a, b) => b.cos - a.cos)
      .slice(0, top)
  }
}

export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!
  return s
}
