/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// Runs a process over the decision model. No language model is called here: each ask step is one System One
// question, each check is code, each pick is a choice among the page's links and buttons. What the model is unsure
// of follows the step's "unsure" route, and without one it goes to the person (the rule-first contract: when the
// model is not sure, it does not decide).

import type { Content } from '../format.ts'
import { askQuestion, guardClick, optionProbabilities, verdictOf, type Decide } from './decide.ts'
import { parseNumber, type Action, type Process, type Step } from './schema.ts'
import { runAdvance, runFill, type Escalation, type StepContext } from './steps.ts'

export interface Element {
  ref: string
  role: 'link' | 'button' | 'field'
  text: string
  href?: string
  /** The text around it (its row or list item), so three "View order" links can be told apart. */
  context?: string
  /** Fields: what kind (text, textarea, select, checkbox, radio), what is in it, and the page's own hints. */
  input?: 'text' | 'textarea' | 'select' | 'checkbox' | 'radio'
  value?: string
  checked?: boolean
  options?: string[]
  /** The field's autocomplete attribute or input type (email, tel, postal-code…), as a browser's autofill reads it. */
  hint?: string
}

export interface Page {
  url: string
  title: string
  /** The page as text, controls marked the way the decision model was trained on: [button] Pay now, [link] Orders. */
  text: string
  elements: Element[]
}

export interface Item {
  id: string
  fields: Record<string, string>
  /** For measuring: the bucket a person would put it in, 'person' when it should go to them, 'click:<label>' for the
   *  control it should use, or a path ('/account/orders') for the page a search should end on. */
  expect?: string
}

/** Where a process runs: a fixture, a mock site, or a real browser tab. */
export interface Surface {
  read(): Promise<Page>
  items(process: Process): Promise<Item[]>
  click(element: Element): Promise<void>
  open(url: string): Promise<void>
  /** Types text into a field (replacing what is there); a real tab only. */
  type?(element: Element, text: string): Promise<void>
  /** Chooses an option of a select by its text; a real tab only. */
  choose?(element: Element, option: string): Promise<void>
}

export interface StepTrace {
  step: string
  kind: Step['kind'] | 'guard'
  question?: string
  /** What the decision model read (shortened). */
  state?: string
  probabilities?: Record<string, number> | null
  verdict: string | null
  /** nano, mini, code, or words (the click guard's list). */
  by: string
  ms: number
  then: Action
  note?: string
}

export type Outcome =
  | { kind: 'bucket'; bucket: string }
  | { kind: 'person'; reason: string }
  | { kind: 'notified'; text: string }
  | { kind: 'stopped' }
  | { kind: 'error'; message: string }

export interface ItemResult {
  item: Item
  label: string
  outcome: Outcome
  steps: StepTrace[]
  ms: number
  /** Controls clicked (or that would have been, in a dry run), in order. */
  clicked: string[]
  /** The page the item ended on. */
  at: string | null
  /** Against item.expect: right, wrong, or person (it went to the person; whether that was wanted is in expect). */
  score?: 'right' | 'wrong' | 'person'
}

export interface RunOptions {
  /** Click and open for real. Off: clicks are recorded, not made (a dry run). Never on for real accounts in the lab. */
  act?: boolean
  /** Steps one item may take (loops through pick and check). */
  maxHops?: number
  signal?: AbortSignal
  onItem?: (result: ItemResult) => void
  onStep?: (itemId: string, trace: StepTrace) => void
  /** Who decides what the decision model isn't sure of in advance and fill steps (a language model); none: the step's unsure route. */
  escalate?: Escalation
  /** Asked when the click guard stops a click in an advance step; true lets it go ahead (the person said yes). */
  approve?: (question: string, kind: string) => Promise<boolean>
  /** atelier-step for advance and fill steps (src/step); without it they ask the decision model's questions. */
  stepModel?: (input: import('../step/format.ts').StepInput) => Promise<import('../step/model.ts').StepAnswer>
}

export interface RunSummary {
  process: string
  items: number
  outcomes: Record<string, number>
  decisions: number
  sure: number
  byWho: Record<string, number>
  msPerDecision: { median: number; p90: number }
  msTotal: number
  scored: number
  right: number
  wrong: number
  person: number
  /** Asked to the person although a bucket was expected. */
  personNotNeeded: number
}

const PERSON: Action = { do: 'ask_person', target: 'not sure', next: null }
const SHORT = 600

function quantile(values: number[], q: number): number {
  if (values.length === 0) return 0
  const s = [...values].sort((a, b) => a - b)
  return s[Math.min(s.length - 1, Math.floor(q * s.length))] ?? 0
}

function label(item: Item): string {
  const f = item.fields
  return (f.title ?? f.subject ?? f.name ?? f.text ?? Object.values(f)[0] ?? item.id).replace(/\s+/g, ' ').slice(0, 90)
}

function stateFor(step: Step, item: Item, page: Page | null): Content {
  const state: Record<string, Content> = {}
  for (const name of step.read) {
    if (name === 'page') {
      if (page) state.page = `${page.title}\n${page.url}\n${page.text}`
    } else if (item.fields[name] !== undefined) state[name] = item.fields[name]
  }
  const keys = Object.keys(state)
  // One field reads as itself, the way the trained questions show a page or a message.
  return keys.length === 1 ? (state[keys[0]!] as Content) : state
}

function short(state: Content): string {
  const s = typeof state === 'string' ? state : JSON.stringify(state)
  return s.length > SHORT ? `${s.slice(0, SHORT)}…` : s
}

function routeFor(step: Step, verdict: string | null): Action {
  const when = verdict ?? 'unsure'
  return step.routes.find((r) => r.when === when)?.then ?? (verdict === null ? PERSON : { ...PERSON, target: `no route for "${when}"` })
}

function check(step: Step, value: string | undefined): string | null {
  if (value === undefined) return null
  const target = step.value ?? ''
  const op = step.op
  if (op === 'contains') return value.toLowerCase().includes(target.toLowerCase()) ? 'yes' : 'no'
  if (op === 'matches') return new RegExp(target, 'i').test(value) ? 'yes' : 'no'
  const a = parseNumber(value)
  const b = parseNumber(target)
  if (op === '=' || op === '!=') {
    const same = Number.isFinite(a) && Number.isFinite(b) ? a === b : value.trim().toLowerCase() === target.trim().toLowerCase()
    return (op === '=') === same ? 'yes' : 'no'
  }
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null
  const ok = op === '<' ? a < b : op === '<=' ? a <= b : op === '>' ? a > b : a >= b
  return ok ? 'yes' : 'no'
}

/**
 * pick: every control on the page (up to 24) is sorted into the step's kinds, all in one request ("What does the
 * button “Reject all” do?": reject-all, accept-all, settings, other), and the control most likely to be of a wanted
 * kind (step.value, comma-separated) is picked when it is clearly ahead: at least the step's cutoff (0.5) and 1.5
 * times the next control. The decision model sorts a short text well; it cannot steer toward a goal, so a pick is
 * never "which link gets me closer to …".
 */
async function pick(decide: Decide, step: Step, question: string, page: Page, visited: Set<string>, signal?: AbortSignal): Promise<{ element: Element | null; probabilities: Record<string, number> | null; by: string; asked: number }> {
  const seen = new Set<string>()
  const candidates = page.elements
    .filter((e) => {
      const key = `${e.role}:${e.text.toLowerCase()}:${e.context ?? ''}`
      if (!e.text.trim() || e.role === 'field' || seen.has(key)) return false
      seen.add(key)
      // A link back to a page this item has already seen leads in circles.
      return !(e.href && visited.has(pathOf(e.href)))
    })
    .slice(0, 24)
  if (candidates.length === 0) return { element: null, probabilities: null, by: 'code', asked: 0 }
  if (step.op === 'contains' || step.op === 'matches') {
    // Picked by code: the one control whose label or row matches ("Oct", "Preparing for shipment").
    const want = step.value ?? ''
    const re = step.op === 'matches' ? new RegExp(want, 'i') : null
    const hits = candidates.filter((e) => {
      const text = `${e.text} ${e.context ?? ''}`
      return re ? re.test(text) : text.toLowerCase().includes(want.toLowerCase())
    })
    return { element: hits.length === 1 ? hits[0]! : null, probabilities: Object.fromEntries(hits.slice(0, 5).map((e) => [e.context ? `${e.text} (${e.context.slice(0, 40)})` : e.text, 1])), by: 'code', asked: 0 }
  }
  const wanted = new Set((step.value ?? '').split(',').map((v) => v.trim()).filter(Boolean))
  const questions: Record<string, ReturnType<typeof askQuestion>> = {}
  candidates.forEach((e, i) => {
    const near = e.context ? ` (next to: “${e.context.slice(0, 160)}”)` : ''
    questions[`c${i}`] = askQuestion({ ...step, answer: step.answer ?? 'choice', question: `About the ${e.role} “${e.text}”${near} on this page: ${question}` })
  })
  const out = await decide({ state: `${page.title}\n${page.text.slice(0, 1500)}`, questions }, signal)
  const scores = candidates.map((e, i) => {
    const probs = optionProbabilities({ ...step, answer: step.answer ?? 'choice' }, out.answers[`c${i}`]) ?? {}
    let p = 0
    for (const [k, v] of Object.entries(probs)) if (wanted.has(k)) p += v
    return { e, p }
  })
  scores.sort((a, b) => b.p - a.p)
  const [best, second] = scores
  const probabilities = Object.fromEntries(scores.slice(0, 5).map((s) => [s.e.context ? `${s.e.text} (${s.e.context.slice(0, 40)})` : s.e.text, s.p]))
  const by = out.routes?.c0 ?? 'model'
  const sure = best && best.p >= (step.sure ?? 0.5) && best.p >= 1.5 * (second?.p ?? 0)
  return { element: sure ? best.e : null, probabilities, by, asked: candidates.length }
}

/** A control by its label: the same text first, then one that starts with it, then one that contains it. */
function findControl(page: Page | null, label: string): Element | null {
  if (!page || !label.trim()) return null
  const want = label.trim().toLowerCase()
  // Fields too: a toggle or a checkbox is clicked to change it.
  const els = page.elements
  return els.find((e) => e.text.toLowerCase() === want) ?? els.find((e) => e.text.toLowerCase().startsWith(want)) ?? els.find((e) => e.text.toLowerCase().includes(want)) ?? null
}

function pathOf(url: string): string {
  try {
    const u = new URL(url)
    return u.pathname + u.search
  } catch {
    return url
  }
}

async function runItem(process: Process, item: Item, surface: Surface, decide: Decide, opts: RunOptions, pageItem: boolean): Promise<ItemResult> {
  const t0 = performance.now()
  const steps: StepTrace[] = []
  const byId = new Map(process.steps.map((s) => [s.id, s]))
  let page: Page | null = pageItem || process.steps.some((s) => s.read.includes('page') || s.kind === 'pick') ? await surface.read() : null
  let lastPick: Element | null = null
  const memory = new Set<string>()
  const clicked: string[] = []
  const visited = new Set<string>()
  const seen = (p: Page | null) => {
    if (p) visited.add(pathOf(p.url))
    return p
  }
  seen(page)
  const fill = (text: string | null) => (text ?? '').replace(/\{task\}/g, process.task)
  let at: string | null = process.start
  let outcome: Outcome = { kind: 'stopped' }
  const maxHops = opts.maxHops ?? 16
  const record = (t: StepTrace) => {
    steps.push(t)
    opts.onStep?.(item.id, t)
  }
  try {
    for (let hop = 0; at !== null; hop++) {
      if (opts.signal?.aborted) throw new Error('stopped')
      if (hop >= maxHops) {
        outcome = { kind: 'person', reason: `still going after ${maxHops} steps` }
        break
      }
      const step = byId.get(at)!
      const s0 = performance.now()
      let verdict: string | null = null
      let probabilities: Record<string, number> | null | undefined
      let by = 'code'
      let state: Content | undefined
      let note: string | undefined
      let selfRecorded = false
      let reason: string | undefined
      if (step.kind === 'ask') {
        if (pageItem && step.read.includes('page')) page = await surface.read()
        state = stateFor(step, pageItem && page ? { ...item, fields: { ...item.fields, title: page.title, url: page.url } } : item, page)
        const out = await decide({ state, questions: { q: askQuestion({ ...step, question: fill(step.question) }) } }, opts.signal)
        probabilities = optionProbabilities(step, out.answers.q)
        verdict = verdictOf(step, probabilities ?? null)
        by = out.routes?.q ?? 'model'
        if (probabilities === null) note = 'the model declined'
      } else if (step.kind === 'check') {
        const field = step.read[0]!
        const value = field === 'page' ? page?.text : item.fields[field]
        verdict = check(step, value)
        if (verdict === null) note = value === undefined ? `no ${field}` : `no number in ${field}`
      } else if (step.kind === 'act') {
        verdict = 'done'
      } else if (step.kind === 'advance' || step.kind === 'fill') {
        // Many decisions in one step; each records itself (steps.ts).
        const ctx: StepContext = { step, goal: fill(step.question) || process.task, values: process.values, surface, decide, act: opts.act ?? false, escalate: opts.escalate, approve: opts.approve, stepModel: opts.stepModel, memory, signal: opts.signal, record, route: (w) => routeFor(step, w === 'unsure' ? null : w) }
        const r = step.kind === 'advance' ? await runAdvance(ctx) : await runFill(ctx)
        page = seen(r.page)
        clicked.push(...r.clicked)
        verdict = r.verdict
        selfRecorded = true
        reason = r.reason
      } else {
        page = seen(await surface.read())!
        const r = await pick(decide, step, fill(step.question), page, visited, opts.signal)
        lastPick = r.element
        probabilities = r.probabilities
        verdict = r.element ? 'picked' : null
        by = r.by
        note = r.element ? `${r.element.role} “${r.element.text}”${r.element.context ? ` (${r.element.context.slice(0, 60)})` : ''}` : 'nothing sure to use'
        state = { page: page.title, address: pathOf(page.url) }
      }
      let then = routeFor(step, verdict)
      if (selfRecorded && verdict === null && then.do === 'ask_person' && reason) then = { ...then, target: `${then.target}: ${reason}` }
      const keep = (t: StepTrace) => (selfRecorded ? undefined : record(t))
      const asked = step.kind === 'act' ? `${step.routes[0]?.then.do} ${step.routes[0]?.then.target}` : step.question ? fill(step.question) : `${step.read[0]} ${step.op} ${step.value}`
      const trace: StepTrace = { step: step.id, kind: step.kind, question: asked, state: state === undefined ? undefined : short(state), probabilities, verdict, by, ms: performance.now() - s0, then, note }
      // Act on the route.
      if (then.do === 'click') {
        if (!page) page = seen(await surface.read())
        const target = then.target === 'picked' ? lastPick : findControl(page, item.fields[then.target] ?? then.target)
        if (!target || !page) {
          keep(trace)
          outcome = { kind: 'person', reason: 'nothing to click' }
          break
        }
        const g = await guardClick(decide, target, page)
        if (!g.stop && !opts.act) trace.note = `${trace.note ?? ''} · dry run, not clicked`
        keep(trace)
        if (g.stop) {
          const reason = `a ${g.stop} click (${g.by === 'words' ? 'by its words' : 'by the model'}): “${target.text}”`
          record({ step: step.id, kind: 'guard', question: `clicking ${target.role} “${target.text}”`, probabilities: g.probabilities, verdict: g.stop, by: g.by === 'words' ? 'words' : (g.route ?? 'model'), ms: g.ms, then: { do: 'ask_person', target: reason, next: null } })
          outcome = { kind: 'person', reason }
          break
        }
        clicked.push(target.text)
        if (!opts.act) {
          outcome = { kind: 'person', reason: `would click “${target.text}” (dry run)` }
          break
        }
        await surface.click(target)
        page = seen(await surface.read())
        outcome = { kind: 'stopped' }
        at = then.next
        continue
      }
      keep(trace)
      if (then.do === 'goto') at = then.target
      else if (then.do === 'put') {
        outcome = { kind: 'bucket', bucket: then.target }
        at = null
      } else if (then.do === 'ask_person') {
        outcome = { kind: 'person', reason: then.target }
        at = null
      } else if (then.do === 'stop') {
        outcome = { kind: 'stopped' }
        at = null
      } else if (then.do === 'notify') {
        const text = then.target.replace(/\{(\w+)\}/g, (_, k: string) => item.fields[k] ?? '')
        outcome = { kind: 'notified', text }
        at = then.next
      } else if (then.do === 'open') {
        const url = item.fields[then.target] ?? then.target
        if (opts.act) {
          await surface.open(url)
          page = seen(await surface.read())
        }
        at = then.next
      }
    }
  } catch (err) {
    outcome = { kind: 'error', message: err instanceof Error ? err.message : String(err) }
  }
  const result: ItemResult = { item, label: label(item), outcome, steps, ms: performance.now() - t0, clicked, at: page?.url ?? null }
  const expect = item.expect
  if (expect?.startsWith('click:')) {
    const want = expect.slice(6).trim().toLowerCase()
    result.score = clicked.some((c) => c.trim().toLowerCase() === want) ? 'right' : clicked.length === 0 && outcome.kind === 'person' ? 'person' : 'wrong'
  } else if (expect?.startsWith('/')) {
    // The mock site may be served under a prefix (/lab/sites/<fixture>/...): the goal is where the path ends.
    const path = (page ? pathOf(page.url) : '').replace(/\/$/, '') || '/'
    const want = expect.replace(/\/$/, '') || '/'
    const there = path === want || (want !== '/' && path.endsWith(want))
    result.score = outcome.kind === 'bucket' && there ? 'right' : outcome.kind === 'person' ? 'person' : 'wrong'
  } else if (expect === 'person') {
    result.score = outcome.kind === 'person' && clicked.length === 0 ? 'right' : 'wrong'
  } else if (expect !== undefined) {
    if (outcome.kind === 'person') result.score = 'person'
    else if (outcome.kind === 'bucket') result.score = outcome.bucket === expect ? 'right' : 'wrong'
    else if (outcome.kind === 'notified') result.score = expect === 'notify' ? 'right' : 'wrong'
    else result.score = 'wrong'
  }
  return result
}

export async function runProcess(process: Process, surface: Surface, decide: Decide, opts: RunOptions = {}): Promise<{ results: ItemResult[]; summary: RunSummary }> {
  const t0 = performance.now()
  const results: ItemResult[] = []
  if (process.over === 'page') {
    const page = await surface.read()
    const items = await surface.items(process)
    const item = items[0] ?? { id: 'page', fields: { title: page.title, url: page.url } }
    const r = await runItem(process, item, surface, decide, opts, true)
    results.push(r)
    opts.onItem?.(r)
  } else {
    for (const item of await surface.items(process)) {
      if (opts.signal?.aborted) break
      const r = await runItem(process, item, surface, decide, opts, false)
      results.push(r)
      opts.onItem?.(r)
    }
  }
  return { results, summary: summarize(process.id, results, performance.now() - t0) }
}

export function summarize(id: string, results: ItemResult[], msTotal: number): RunSummary {
  const outcomes: Record<string, number> = {}
  const byWho: Record<string, number> = {}
  const times: number[] = []
  let decisions = 0
  let sure = 0
  for (const r of results) {
    const key = r.outcome.kind === 'bucket' ? r.outcome.bucket : r.outcome.kind
    outcomes[key] = (outcomes[key] ?? 0) + 1
    for (const s of r.steps) {
      byWho[s.by] = (byWho[s.by] ?? 0) + 1
      if (s.by === 'code' || s.by === 'words' || s.by === 'claude' || (s.kind === 'fill' && s.verdict !== null && s.by !== 'step')) continue
      decisions++
      times.push(s.ms)
      if (s.verdict !== null) sure++
    }
  }
  const scored = results.filter((r) => r.score)
  return {
    process: id,
    items: results.length,
    outcomes,
    decisions,
    sure,
    byWho,
    msPerDecision: { median: quantile(times, 0.5), p90: quantile(times, 0.9) },
    msTotal,
    scored: scored.length,
    right: scored.filter((r) => r.score === 'right').length,
    wrong: scored.filter((r) => r.score === 'wrong').length,
    person: scored.filter((r) => r.score === 'person').length,
    personNotNeeded: scored.filter((r) => r.score === 'person' && r.item.expect !== 'person').length,
  }
}
