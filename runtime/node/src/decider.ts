// The on-device decider: System One requests in, System One answers out, from a tiny ONNX student.
// No network, no GPU: ONNX Runtime on the CPU, the pure-JS Hugging Face tokenizer, and a vocabulary remap.

import { readFileSync, existsSync } from 'node:fs'
import { join } from 'node:path'
import { Tokenizer } from '@huggingface/tokenizers'
import * as ort from 'onnxruntime-node'
import { encode, optionLabels, stateText, type Content, type DecisionRequest, type Question, type Tokenize } from './format.ts'

export type Answer =
  | { type: 'refusal'; reason: string }
  | { type: 'noul'; noul: number }
  | { type: 'choice'; choice: string; probabilities: Record<string, number>; confidence: number }
  | { type: 'score'; score: number; probabilities: Record<string, number>; legend: Record<string, Content>; confidence: number }

/** pplx_decider.protocol.answer: the System One answer for one question from its probabilities. */
export function answer(q: Question, probabilities: number[]): Answer {
  const keys = optionLabels(q)
  const total = probabilities.reduce((a, b) => a + b, 0)
  const values = probabilities.map((v) => v / total)
  const at = (i: number): number => values[i] ?? 0
  if (q.type === 'noul') return { type: 'noul', noul: at(1) }
  let best = 0
  for (let i = 1; i < values.length; i++) if (at(i) > at(best)) best = i
  const distribution = Object.fromEntries(keys.map((k, i) => [k, at(i)]))
  if (q.type === 'choice') {
    const n = values.length
    const confidence = n === 1 ? 1 : (at(best) - 1 / n) / (1 - 1 / n)
    return { type: 'choice', probabilities: distribution, choice: keys[best] ?? '', confidence: Math.max(0, Math.min(1, confidence)) }
  }
  const levels = q.criteria as Content[]
  const distance = values.reduce((s, p, i) => s + p * Math.abs(i - best), 0)
  const midpoint = (values.length - 1) / 2
  const baseline = values.reduce((s, _, i) => s + Math.abs(i - midpoint), 0) / values.length
  return {
    type: 'score',
    probabilities: distribution,
    legend: Object.fromEntries(keys.map((k, i) => [k, levels[i] ?? null])),
    score: values.reduce((s, p, i) => s + i * p, 0),
    confidence: Math.max(0, 1 - distance / baseline),
  }
}

function softmax(xs: number[]): number[] {
  const m = Math.max(...xs)
  const e = xs.map((x) => Math.exp(x - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((x) => x / s)
}

/**
 * When the model can't read a state, it declines instead of guessing. It learned from English, Dutch, German,
 * French and Spanish with a vocabulary trimmed to them: text in another script falls apart into single letters
 * or raw bytes, and its answers there are not to be trusted. A declined question is answered `refusal`, which
 * Atelier reads as an even spread, so the rule decides. Thresholds are set from the per-language probe
 * (atelierdecisionmodel: analysis/results/languages.txt).
 */
export interface Guard {
  /** Least share of the state's letters in Latin script. */
  minLatin: number
  /** Most share of the state's tokens that are raw bytes (characters outside the vocabulary). */
  maxBytes: number
  /** Below this many letters there is too little text to judge the script by. */
  minLetters: number
}
export const DEFAULT_GUARD: Guard = { minLatin: 0.7, maxBytes: 0.1, minLetters: 8 }

/** Why the model can't read this state, or null when it can. */
export function unreadable(text: string, ids: number[], byteIds: Set<number>, guard: Guard = DEFAULT_GUARD): string | null {
  const letters = text.match(/\p{L}/gu) ?? []
  if (letters.length >= guard.minLetters) {
    const latin = letters.filter((c) => /\p{Script=Latin}/u.test(c)).length / letters.length
    if (latin < guard.minLatin) return `mostly not Latin script (${Math.round(latin * 100)}% Latin)`
  }
  if (ids.length > 0) {
    const bytes = ids.filter((id) => byteIds.has(id)).length / ids.length
    if (bytes > guard.maxBytes) return `${Math.round(bytes * 100)}% of the text is outside the vocabulary`
  }
  return null
}

export interface DeciderOptions {
  /** model.emb4.int8.onnx (default: int8 layers, 4-bit embedding table, the shipped file), model.int8.onnx or model.onnx */
  file?: string
  threads?: number
  maxLength?: number
  /** false to answer whatever the text (for measuring the guard itself). */
  guard?: Guard | false
  /**
   * Questions other than Atelier's own (label_items, routines, packs): 'decline' (default) answers them `refusal`.
   * On questions and text it never saw, the model's sure answers contradicted the teacher's 6% of the time
   * (36% for one kind of question), against under 1% on Atelier's own decisions (atelierdecisionmodel:
   * analysis/results/open-questions-*.txt), so those go to the rule or a larger model instead.
   */
  open?: 'decline' | 'answer'
}

export class NanoDecider {
  private readonly session: ort.InferenceSession
  private readonly tok: Tokenize
  private readonly temperatures: Record<string, number>
  private readonly maxLength: number
  private readonly byteIds: Set<number>
  private readonly guard: Guard | false
  private readonly open: 'decline' | 'answer'

  private constructor(session: ort.InferenceSession, tok: Tokenize, temperatures: Record<string, number>, maxLength: number, byteIds: Set<number>, guard: Guard | false, open: 'decline' | 'answer') {
    this.session = session
    this.tok = tok
    this.temperatures = temperatures
    this.maxLength = maxLength
    this.byteIds = byteIds
    this.guard = guard
    this.open = open
  }

  static async load(dir: string, options: DeciderOptions = {}): Promise<NanoDecider> {
    const json = (name: string): unknown => JSON.parse(readFileSync(join(dir, name), 'utf8'))
    const tokenizer = new Tokenizer(json('tokenizer.json') as object, json('tokenizer_config.json') as object)
    const vocab = json('vocab.json') as { keep: number[]; unk: number }
    const remap = new Map<number, number>()
    vocab.keep.forEach((old, i) => remap.set(old, i))
    const original = (token: string): number => {
      const ids = tokenizer.encode(token, { add_special_tokens: false }).ids
      const id = ids[0]
      if (ids.length !== 1 || id === undefined) throw new Error(`${token} is not one token`)
      return id
    }
    const map = (id: number): number => remap.get(id) ?? vocab.unk
    // Ids are remapped once here, so the format code works in the trimmed vocabulary throughout.
    const tok: Tokenize = {
      encode: (text) => tokenizer.encode(text, { add_special_tokens: false }).ids.map(map),
      id: (token) => map(original(token)),
      bos: map(original('<bos>')),
      eos: map(original('<eos>')),
    }
    const session = await ort.InferenceSession.create(join(dir, options.file ?? 'model.emb4.int8.onnx'), {
      executionProviders: ['cpu'],
      intraOpNumThreads: options.threads ?? 4,
      graphOptimizationLevel: 'all',
    })
    const calibration = join(dir, 'calibration.json')
    const temperatures = existsSync(calibration) ? (json('calibration.json') as Record<string, number>) : {}
    const spec = json('tokenizer.json') as { model: { vocab: Record<string, number> } }
    const byteIds = new Set(Object.entries(spec.model.vocab).filter(([t]) => /^<0x[0-9A-F]{2}>$/.test(t)).map(([, id]) => map(id)))
    return new NanoDecider(session, tok, temperatures, options.maxLength ?? 512, byteIds, options.guard ?? DEFAULT_GUARD, options.open ?? 'decline')
  }

  /** Frees the ONNX session. Call it before the process ends: with sessions left open, onnxruntime-node's
   * thread pools can keep a process from exiting (seen on Windows). */
  async release(): Promise<void> {
    await this.session.release()
  }

  /** The token ids each question is run with (for parity checks). */
  ids(request: DecisionRequest): Record<string, number[]> {
    return Object.fromEntries(Object.entries(request.questions).map(([key, q]) => [key, encode(this.tok, request.state, q, this.maxLength).ids]))
  }

  async decide(request: DecisionRequest): Promise<{ answers: Record<string, Answer>; usage: { input_tokens: number; output_tokens: number } }> {
    const entries = Object.entries(request.questions ?? {})
    if (entries.length === 0) throw new Error('A request needs at least one question')
    if (this.guard) {
      const text = stateText(request.state)
      const reason = unreadable(text, this.tok.encode(text), this.byteIds, this.guard)
      if (reason) return { answers: Object.fromEntries(entries.map(([key]) => [key, { type: 'refusal', reason }])), usage: { input_tokens: 0, output_tokens: 0 } }
    }
    // Dynamic int8 quantization scales activations per tensor, padding included, so batching questions of
    // different lengths would change their answers slightly; one pass per question keeps them independent
    // (and identical to how they were measured).
    if (entries.length > 1) {
      const parts = await Promise.all(entries.map(([key, q]) => this.decide({ state: request.state, questions: { [key]: q } })))
      return {
        answers: Object.assign({}, ...parts.map((p) => p.answers)),
        usage: { input_tokens: parts.reduce((s, p) => s + p.usage.input_tokens, 0), output_tokens: 0 },
      }
    }
    const encoded = entries.map(([, q]) => encode(this.tok, request.state, q, this.maxLength))
    if (this.open === 'decline' && encoded.every((e) => e.kind === 'open')) {
      return { answers: Object.fromEntries(entries.map(([key]) => [key, { type: 'refusal', reason: 'an open question this model was not trained for' }])), usage: { input_tokens: 0, output_tokens: 0 } }
    }
    const width = Math.max(...encoded.map((e) => e.ids.length))
    const ids = new BigInt64Array(encoded.length * width)
    const mask = new BigInt64Array(encoded.length * width)
    encoded.forEach((e, row) => {
      e.ids.forEach((id, col) => {
        ids[row * width + col] = BigInt(id)
        mask[row * width + col] = 1n
      })
    })
    const out = await this.session.run({
      input_ids: new ort.Tensor('int64', ids, [encoded.length, width]),
      attention_mask: new ort.Tensor('int64', mask, [encoded.length, width]),
    })
    const scoresTensor = out.scores
    if (!scoresTensor) throw new Error('The model gave no scores')
    const scores = scoresTensor.data as Float32Array
    const answers: Record<string, Answer> = {}
    let tokens = 0
    entries.forEach(([key, q], row) => {
      const e = encoded[row]
      if (!e) return
      tokens += e.ids.length
      const t = this.temperatures[e.kind] ?? 1
      answers[key] = answer(q, softmax(e.markers.map((m) => (scores[row * width + m] ?? 0) / t)))
    })
    return { answers, usage: { input_tokens: tokens, output_tokens: 0 } }
  }
}
