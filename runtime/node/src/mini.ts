// atelier-mini: Qwen3.5-0.8B read the way pplx-decider reads (non-causal attention, a 255-code readout at the
// <|fim_suffix|> marker), run by llama.cpp. llama-server computes the backbone from a GGUF marked by
// student/gguf_decider.py and returns the final hidden state at the marker; this applies the readout and the
// per-decision temperature and answers in System One's format. The folder comes from student/export_mini.py.
//
// Prompts are built and tokenized here exactly as pplx_decider.protocol.Encoder does (the Hugging Face tokenizer,
// not llama.cpp's, which differs on a few non-Latin strings); src/mini_parity.ts checks the token ids.

import { spawn, type ChildProcess } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { createServer } from 'node:net'
import { join } from 'node:path'
import { Tokenizer } from '@huggingface/tokenizers'
import { answer, type Answer } from './decider.ts'
import { fixedKind, optionLabels, pyJson, type Content, type DecisionRequest, type Question } from './format.ts'

interface Spec {
  model: string
  codes: string[]
  hidden: number
  temperatures: Record<string, number>
  prefix: string
  state_label: string
  question_label: string
  options_label: string
  default_instructions: string
  escapes: string[]
  separator_id: number
  marker_id: number
}

export interface MiniOptions {
  /** The llama-server executable (default: $LLAMA_SERVER, else llama-server on the PATH). */
  server?: string
  threads?: number
  /** Longest sequence in tokens (state and one question; default 4096). Non-causal attention needs a whole question in
   * one batch, so this also sets the compute buffer: about 1 GB resident at 8192, less at 4096. */
  context?: number
  /** Layers on the GPU (Metal, Vulkan, CUDA builds); 0 keeps everything on the CPU. */
  gpuLayers?: number
  /** false: tokenizer only, no llama-server (for parity checks). */
  serve?: boolean
}

const describe = (v: Content | null | undefined): string => (typeof v === 'string' ? v : pyJson(v ?? null))

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const s = createServer()
    s.once('error', reject)
    s.listen(0, '127.0.0.1', () => {
      const address = s.address()
      const port = typeof address === 'object' && address ? address.port : 0
      s.close(() => resolve(port))
    })
  })
}

function softmax(xs: number[]): number[] {
  const m = Math.max(...xs)
  const e = xs.map((x) => Math.exp(x - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((x) => x / s)
}

export class MiniDecider {
  readonly name: string
  private readonly spec: Spec
  private readonly tokenizer: Tokenizer
  private readonly readout: Float32Array
  private readonly server: ChildProcess | null
  private readonly port: number
  private queue: Promise<unknown> = Promise.resolve()

  private constructor(spec: Spec, tokenizer: Tokenizer, readout: Float32Array, server: ChildProcess | null, port: number) {
    this.name = spec.model
    this.spec = spec
    this.tokenizer = tokenizer
    this.readout = readout
    this.server = server
    this.port = port
  }

  static async load(dir: string, options: MiniOptions = {}): Promise<MiniDecider> {
    const json = (name: string): unknown => JSON.parse(readFileSync(join(dir, name), 'utf8'))
    const spec = json('decision.json') as Spec
    const tokenizer = new Tokenizer(json('tokenizer.json') as object, json('tokenizer_config.json') as object)
    const bytes = readFileSync(join(dir, 'readout.f32'))
    const readout = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength))
    if (readout.length !== spec.codes.length * spec.hidden) throw new Error('readout.f32 does not match decision.json')
    if (options.serve === false) return new MiniDecider(spec, tokenizer, readout, null, 0)
    const port = await freePort()
    const context = String(options.context ?? 4096)
    const server = spawn(options.server ?? process.env.LLAMA_SERVER ?? 'llama-server', [
      '-m', join(dir, 'mini.gguf'), '--embeddings', '--pooling', 'last', '-c', context, '-b', context, '-ub', context,
      '-t', String(options.threads ?? 4), '-ngl', String(options.gpuLayers ?? 0), '-np', '1',
      '--host', '127.0.0.1', '--port', String(port),
    ], { stdio: 'ignore', windowsHide: true })
    let exited = false
    server.once('exit', () => { exited = true })
    for (let i = 0; i < 240; i++) {
      if (exited) throw new Error('llama-server exited while loading the mini')
      try {
        if ((await fetch(`http://127.0.0.1:${port}/health`)).ok) return new MiniDecider(spec, tokenizer, readout, server, port)
      } catch { /* not up yet */ }
      await new Promise((r) => setTimeout(r, 500))
    }
    server.kill()
    throw new Error('llama-server did not start')
  }

  async release(): Promise<void> {
    this.server?.kill()
  }

  private escape(text: string): string {
    for (const token of this.spec.escapes) text = text.replaceAll(token, token.replaceAll('<', '＜').replaceAll('>', '＞'))
    return text
  }

  private encode(text: string): number[] {
    return this.tokenizer.encode(text, { add_special_tokens: false }).ids
  }

  /** pplx_decider.protocol.options: the descriptions shown to the model, in answer order. */
  private descriptions(q: Question): string[] {
    if (q.type === 'choice') return Object.entries(q.criteria as Record<string, Content | null>).map(([k, v]) => (v === null ? k : `${k}: ${describe(v)}`))
    if (q.type === 'score') return (q.criteria as Content[]).map((v) => describe(v))
    const c = (q.criteria ?? {}) as Record<string, Content | null>
    return [c.false ? describe(c.false) : 'No / false', c.true ? describe(c.true) : 'Yes / true']
  }

  /** The token ids of one question: the context (template, state), the separator, the question, the marker. */
  ids(state: Content, q: Question): number[] {
    const s = this.spec
    const context = this.encode(s.prefix + s.state_label + this.escape(describe(state)))
    const lines = this.descriptions(q).map((d, i) => `${s.codes[i]}: ${this.escape(d)}`).join('\n')
    const instructions = q.instructions ? describe(q.instructions) : s.default_instructions
    const question = this.encode(s.question_label + this.escape(instructions) + s.options_label + lines)
    return [...context, s.separator_id, ...question, s.marker_id]
  }

  private async hidden(ids: number[]): Promise<number[]> {
    const res = await fetch(`http://127.0.0.1:${this.port}/embedding`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ content: ids, embd_normalize: -1 }),
    })
    if (!res.ok) throw new Error(`llama-server: ${res.status} ${await res.text()}`)
    const out = (await res.json()) as unknown
    const item = (Array.isArray(out) ? out[0] : out) as { embedding: number[] | number[][] }
    const e = item.embedding
    return (Array.isArray(e[0]) ? e[0] : e) as number[]
  }

  /** Probabilities over the question's options, calibrated per decision. */
  async probs(state: Content, q: Question): Promise<{ probs: number[]; tokens: number }> {
    const ids = this.ids(state, q)
    const h = await this.hidden(ids)
    const n = optionLabels(q).length
    const t = this.spec.temperatures[fixedKind(q) ?? 'open'] ?? 1
    const z: number[] = []
    for (let row = 0; row < n; row++) {
      let dot = 0
      const base = row * this.spec.hidden
      for (let j = 0; j < this.spec.hidden; j++) dot += (this.readout[base + j] ?? 0) * (h[j] ?? 0)
      z.push(dot / t)
    }
    return { probs: softmax(z), tokens: ids.length }
  }

  async decide(request: DecisionRequest): Promise<{ answers: Record<string, Answer>; usage: { input_tokens: number; output_tokens: number } }> {
    const entries = Object.entries(request.questions ?? {})
    if (entries.length === 0) throw new Error('A request needs at least one question')
    // One sequence per question (separate mode), one at a time: llama-server runs a single slot.
    const run = async (): Promise<{ answers: Record<string, Answer>; usage: { input_tokens: number; output_tokens: number } }> => {
      const answers: Record<string, Answer> = {}
      let tokens = 0
      for (const [key, q] of entries) {
        const { probs, tokens: n } = await this.probs(request.state, q)
        answers[key] = answer(q, probs)
        tokens += n
      }
      return { answers, usage: { input_tokens: tokens, output_tokens: 0 } }
    }
    const next = this.queue.then(run, run)
    this.queue = next.catch(() => undefined)
    return next
  }
}
