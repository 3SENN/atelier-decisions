/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// The steps that make many decisions: advance and fill. They follow how browser agents built on decision models work
// (docs/processes.md › How decision models run flows elsewhere):
//
//   advance: code turns the page into a numbered list of its elements ("[5] field E-mail: empty"); one request asks
//            which operation (click, type, done, blocked), which element if it is a click and which field if it is
//            typing; a second asks which of the process's named values (or which option) goes in. Code checks the
//            answers, guards clicks, and acts. The model never writes text and never points outside the list.
//   fill:    every empty field of a form gets one of the named values: first from the page's own autofill hints
//            (autocomplete="email", type="tel"), then by asking the decision model about each remaining field.
//
// Whatever the decision model isn't sure of goes to the escalation (a language model, for that one operation or
// that one form) when there is one, and otherwise to the step's "unsure" route. Every operation is recorded with
// who decided it, so a run shows how much of the flow the decision model carried.

import type { Content } from '../format.ts'
import type { StepElement, StepInput as StepModelInput } from '../step/format.ts'
import type { StepAnswer } from '../step/model.ts'
import { guardClick, type Decide } from './decide.ts'
import type { Element, Page, StepTrace, Surface } from './run.ts'
import type { Action, Step, Value } from './schema.ts'

export type Operation = 'click' | 'type' | 'done' | 'blocked'

export interface StepInput {
  goal: string
  values: Value[]
  page: { title: string; url: string }
  /** The numbered elements, as the decision model saw them. */
  elements: string[]
  recent: string[]
}

export interface Escalation {
  /** One operation for an advance step the decision model wasn't sure of; element is a number from the list. */
  step(input: StepInput): Promise<{ operation: Operation; element: number | null; value: string | null } | null>
  /** For the fields the decision model wasn't sure of (by number): the name of the value that goes in, or null. */
  fill(input: { purpose: string; values: Value[]; page: { title: string; url: string }; fields: string[] }): Promise<Record<string, string | null>>
}

export interface StepContext {
  step: Step
  goal: string
  values: Value[]
  surface: Surface
  decide: Decide
  act: boolean
  escalate?: Escalation
  /** The person, asked when the click guard stops a click: true lets that one click go ahead. */
  approve?: (question: string, kind: string) => Promise<boolean>
  /** What happened earlier in this item's run, across steps ("added-to-cart"). */
  memory?: Set<string>
  /** atelier-step: one pass per page for the operation, the target and every field's value (src/step). */
  stepModel?: (input: StepModelInput) => Promise<StepAnswer>
  signal?: AbortSignal
  record: (t: StepTrace) => void
  route: (when: string) => Action
}

export interface StepOutcome {
  /** The answer the step's routes are read with: done, blocked, or null (unsure). */
  verdict: string | null
  page: Page
  clicked: string[]
  /** Why it stopped, when it wasn't done. */
  reason?: string
}

const MAX_ELEMENTS = 60
const ADD_TO_CART = /^(add to (cart|basket|bag)|in (de |het )?(winkelwagen|winkelmand|winkelmandje)|toevoegen aan winkelmand|(zum|in den) warenkorb( hinzufügen| legen)?|ajouter au panier|añadir a la (cesta|carrito)|añadir al carrito)$/i
const OP_CUTOFF = 0.6

/** The page's own hint (autocomplete or input type) → words a value's name would contain. */
const HINTS: Array<[RegExp, RegExp]> = [
  [/^email$/, /e-?mail/i],
  [/^tel(-national)?$/, /phone|telephone|mobile|telefoon/i],
  [/^postal-code$/, /post ?code|zip|postal/i],
  [/^given-name$/, /first name|given name|voornaam/i],
  [/^family-name$/, /last name|family name|surname|achternaam/i],
  [/^name$/, /^(full )?name$|^naam$/i],
  [/^(street-address|address-line1)$/, /street|address|straat|adres/i],
  [/^address-level2$/, /city|town|plaats|stad/i],
  [/^country(-name)?$/, /country|land/i],
]
const NEVER = /^(password|new-password|current-password|one-time-code|cc-|credit-card)/

export function isEditable(e: Element): boolean {
  return e.role === 'field' && (e.input === 'text' || e.input === 'textarea' || e.input === 'select' || e.input === undefined)
}

function isClickable(e: Element): boolean {
  return e.role === 'link' || e.role === 'button' || (e.role === 'field' && (e.input === 'checkbox' || e.input === 'radio'))
}

function describe(e: Element): string {
  if (e.role !== 'field') return `${e.role} ${e.text}${e.context && e.text.length < 20 ? ` (${e.context.slice(0, 60)})` : ''}`
  if (e.input === 'checkbox' || e.input === 'radio') return `${e.input} ${e.text}: ${e.checked ? 'checked' : 'not checked'}`
  if (e.input === 'select') return `select ${e.text}: ${e.value || 'nothing chosen'} (options: ${(e.options ?? []).slice(0, 8).join(' | ')})`
  return `field ${e.text}: ${e.value ? e.value.slice(0, 40) : 'empty'}`
}

function pathOf(url: string): string {
  try {
    const u = new URL(url)
    return u.pathname + u.search
  } catch {
    return url
  }
}

function valueByHint(e: Element, values: Value[]): Value | null {
  if (!e.hint) return null
  for (const [hint, name] of HINTS) if (hint.test(e.hint)) return values.find((v) => name.test(v.name)) ?? null
  return null
}

function best(probs: Record<string, number> | undefined): [string | null, number, number] {
  const sorted = Object.entries(probs ?? {}).sort((a, b) => b[1] - a[1])
  return [sorted[0]?.[0] ?? null, sorted[0]?.[1] ?? 0, sorted[1]?.[1] ?? 0]
}

/** A choice is taken when it is likely enough and clearly ahead of the next one. */
const clear = (p: number, second: number, cutoff = 0.5) => p >= cutoff && p >= 1.5 * second

function optionFor(e: Element, value: string): string | null {
  const v = value.trim().toLowerCase()
  const opts = e.options ?? []
  return opts.find((o) => o.toLowerCase() === v) ?? opts.find((o) => o.toLowerCase().includes(v) || (o.length > 1 && v.includes(o.toLowerCase()))) ?? null
}

async function typeInto(ctx: StepContext, e: Element, value: string): Promise<void> {
  if (!ctx.act) return
  if (e.input === 'select') {
    if (!ctx.surface.choose) throw new Error('this surface cannot choose options')
    await ctx.surface.choose(e, value)
  } else {
    if (!ctx.surface.type) throw new Error('this surface cannot type')
    await ctx.surface.type(e, value)
  }
}

/** Which named value (or option) goes into one field: the page's hint, the decision model, then the escalation. */
/** undefined: not sure (the decision model, then the escalation); null: nothing goes in. */
async function valueFor(ctx: StepContext, e: Element, page: Page, list: string[], recent: string[]): Promise<{ value: string | null | undefined; by: string; probabilities?: Record<string, number> }> {
  if (e.hint && NEVER.test(e.hint)) return { value: null, by: 'code' }
  const hinted = valueByHint(e, ctx.values)
  if (hinted) return { value: e.input === 'select' ? (optionFor(e, hinted.value) ?? hinted.value) : hinted.value, by: 'code' }
  let q
  if (e.input === 'select') {
    const opts = (e.options ?? []).filter((o) => !/^(choose|select|kies|--|wählen)/i.test(o)).slice(0, 20)
    // A value that names an option outright needs no question.
    for (const v of ctx.values) {
      const o = optionFor({ ...e, options: opts }, v.value)
      if (o && v.value.length > 1) return { value: o, by: 'code' }
    }
    q = { type: 'choice' as const, instructions: `Which option of “${e.text}” fits the goal and the person's values?`, criteria: Object.fromEntries(opts.map((o) => [o, null])) }
  } else {
    q = { type: 'choice' as const, instructions: `Which of the person's values goes into the field “${e.text}”?`, criteria: { ...Object.fromEntries(ctx.values.map((v) => [v.name, v.value.length > 48 ? `${v.value.slice(0, 48)}…` : v.value])), none: 'none of these values belongs in this field' } }
  }
  const state: Content = { goal: ctx.goal, values: ctx.values.map((v) => `${v.name}: ${v.value.slice(0, 80)}`).join('\n'), page: page.title, field: describe(e) }
  const out = await ctx.decide({ state, questions: { value: q } }, ctx.signal)
  const a = out.answers.value
  const probs = a && a.type === 'choice' ? a.probabilities : undefined
  const [name, p, second] = best(probs)
  if (name && clear(p, second, 0.6)) {
    if (e.input === 'select') return { value: name, by: out.routes?.value ?? 'model', probabilities: probs }
    return { value: name === 'none' ? null : (ctx.values.find((v) => v.name === name)?.value ?? null), by: out.routes?.value ?? 'model', probabilities: probs }
  }
  if (ctx.escalate) {
    const r = await ctx.escalate.step({ goal: `${ctx.goal} (now: what goes into ${describe(e)})`, values: ctx.values, page: { title: page.title, url: page.url }, elements: list, recent })
    if (r?.value) {
      const named = ctx.values.find((v) => v.name === r.value)?.value ?? r.value
      return { value: e.input === 'select' ? (optionFor(e, named) ?? named) : named, by: 'claude', probabilities: probs }
    }
  }
  return { value: undefined, by: out.routes?.value ?? 'model', probabilities: probs }
}

/** The page as atelier-step reads it: its controls and its short text lines, in page order (step/format.py). */
export function stepItems(page: Page): Array<{ step: StepElement; element: Element | null }> {
  const out: Array<{ step: StepElement; element: Element | null }> = []
  // A control's context is the row it sits in ("#1043 · placed 14 Oct"); in a header or a menu that row is only the
  // other controls' labels, which says nothing, so it is dropped.
  const labels = [...new Set(page.elements.map((e) => e.text).filter((t) => t.length > 1))].sort((a, b) => b.length - a.length)
  // As in training (step/synth.py): context tells apart controls with the same label ("View order" for each order) and
  // goes with a product's price in a list of results; anywhere else it is left out.
  const counts = new Map<string, number>()
  for (const e of page.elements) counts.set(e.text, (counts.get(e.text) ?? 0) + 1)
  const PRICE = /^\s*(?:€|\$|£|EUR)\s?\d|^\s*\d+(?:[.,]\d{2})?\s?(?:€|EUR)/
  const useful = (e: Element): Element => {
    if (!e.context) return e
    let rest = e.context
    for (const l of labels) rest = rest.split(l).join(' ')
    const keep = rest.replace(/[\s·|,()-]+/g, '').length >= 3 && ((counts.get(e.text) ?? 0) > 1 || PRICE.test(e.context))
    return keep ? e : { ...e, context: undefined }
  }
  const text = (t: string) => {
    const s = t.replace(/\s+/g, ' ').trim()
    if (s.length > 1 && /[\p{L}\p{N}]/u.test(s)) out.push({ step: { role: 'text', label: s.slice(0, 160) }, element: null })
  }
  let k = 0
  const marker = /\[(link|button|field)\] /g
  for (const line of page.text.split('\n')) {
    let last = 0
    marker.lastIndex = 0
    for (let m = marker.exec(line); m; m = marker.exec(line)) {
      if (m.index < last) continue
      text(line.slice(last, m.index))
      const e = page.elements[k++]
      let end = m.index + m[0].length
      if (e) {
        out.push({ step: toStepElement(useful(e)), element: e })
        if (line.startsWith(e.text, end)) end += e.text.length
        if (e.role === 'field') {
          const shown = /^: [^[]*?(?= \[|$)/.exec(line.slice(end))
          if (shown) end += shown[0].length
        }
      }
      last = end
      marker.lastIndex = end
    }
    text(line.slice(last))
  }
  for (; k < page.elements.length; k++) out.push({ step: toStepElement(useful(page.elements[k]!)), element: page.elements[k]! })
  return out
}

function toStepElement(e: Element): StepElement {
  if (e.role !== 'field') return { role: e.role, label: e.text, ...(e.context ? { context: e.context } : {}) }
  if (e.input === 'select') return { role: 'select', label: e.text, value: e.value ?? '', options: e.options ?? [] }
  if (e.input === 'checkbox' || e.input === 'radio') return { role: e.input, label: e.text, checked: Boolean(e.checked) }
  return { role: 'field', label: e.text, value: e.value ?? '' }
}

/** One operation from atelier-step; null fields where it isn't sure. */
async function stepAdvice(ctx: StepContext, page: Page, recent: string[]): Promise<{ operation: Operation | null; target: Element | null; value: string | null | undefined; probs: Record<string, number>; why: string }> {
  const items = stepItems(page)
  const a = await ctx.stepModel!({ goal: ctx.goal, values: ctx.values, before: recent, page: `${page.title} (${pathOf(page.url)})`, elements: items.map((x) => x.step) })
  const [op, pOp] = best(a.op)
  const operation = op && pOp >= (ctx.step.sure ?? OP_CUTOFF) ? (op as Operation) : null
  // Among the candidates only (renormalised): typing goes into an empty field or a select with nothing chosen, as in
  // training, where a filled field was never the one to type into.
  const pick = (ps: Array<number | null>, ok: (i: number) => boolean = () => true) => {
    const ranked = ps.map((p, i) => [i, p] as const).filter((x): x is readonly [number, number] => x[1] !== null && ok(x[0]))
    const total = ranked.reduce((t, x) => t + x[1], 0) || 1
    const sorted = ranked.map(([i, p]) => [i, p / total] as const).sort((x, y) => y[1] - x[1])
    const [first, second] = sorted
    return first && clear(first[1], second?.[1] ?? 0) ? first[0] : null
  }
  const empty = (i: number) => !(items[i]!.step.value ?? '').trim()
  let target: Element | null = null
  let value: string | null | undefined
  if (operation === 'click') {
    const i = pick(a.click)
    target = i === null ? null : items[i]!.element
  } else if (operation === 'type') {
    const i = pick(a.type, empty)
    target = i === null ? null : items[i]!.element
    const vs = i === null ? null : a.values[i]
    if (vs) {
      const ranked = vs.map((p, j) => [j, p] as const).sort((x, y) => y[1] - x[1])
      const [first, second] = ranked
      if (first && clear(first[1], second?.[1] ?? 0, 0.5)) value = first[0] === 0 ? null : ctx.values[first[0] - 1]!.value
    }
    if (target && value && target.input === 'select') value = optionFor(target, value) ?? undefined
    // A select whose option no value names outright ("our bin wasn't emptied" → "Afval niet opgehaald"): the decision
    // model's own question over the options decides.
    if (target && target.input === 'select' && value === undefined) value = (await valueFor({ ...ctx, escalate: undefined }, target, page, [], [])).value
  }
  return { operation, target, value, probs: a.op, why: operation === null ? `not sure what to do (${op} ${pOp.toFixed(2)})` : `not sure which element to ${operation}` }
}

/** advance: toward the step's goal one operation at a time, until done, blocked, unsure or out of budget. */
export async function runAdvance(ctx: StepContext): Promise<StepOutcome> {
  const budget = ctx.step.budget ?? 12
  const recent: string[] = []
  // The same operation on the same control with the same value, again within a few operations: going in circles.
  const done: string[] = []
  const refused = new Set<string>()
  let justRefused = false
  const circling = (sig: string) => {
    const again = done.slice(-4).includes(sig)
    done.push(sig)
    return again
  }
  const clicked: string[] = []
  let page = await ctx.surface.read()
  let lastKey = ''
  let same = 0
  for (let n = 1; n <= budget; n++) {
    if (ctx.signal?.aborted) throw new Error('stopped')
    page = await ctx.surface.read()
    const key = `${page.url}\n${page.text}`
    same = key === lastKey ? same + 1 : 0
    lastKey = key
    if (same >= 2) return { verdict: 'blocked', page, clicked, reason: 'the page stopped changing' }
    const els = page.elements.filter((e) => e.text.trim()).slice(0, MAX_ELEMENTS)
    const list = els.map((e, i) => `[${i + 1}] ${describe(e)}`)
    const clickable = Object.fromEntries(els.flatMap((e, i) => (isClickable(e) ? [[String(i + 1), describe(e).slice(0, 90)]] : [])))
    const editable = Object.fromEntries(els.flatMap((e, i) => (isEditable(e) ? [[String(i + 1), describe(e).slice(0, 90)]] : [])))
    const state: Content = {
      goal: ctx.goal,
      values: ctx.values.map((v) => `${v.name}: ${v.value.slice(0, 80)}`).join('\n') || 'none',
      page: `${page.title} (${pathOf(page.url)})`,
      elements: list.join('\n'),
      done_so_far: recent.join('\n') || 'nothing yet',
    }
    const questions: Record<string, { type: 'choice'; instructions: string; criteria: Record<string, string> }> = {
      operation: {
        type: 'choice',
        instructions: 'Advance the goal from the current page with one operation.',
        criteria: {
          click: 'press a button, follow a link or tick a box that moves the goal forward',
          type: 'put one of the given values into a field that is still empty or wrong',
          done: 'every part of the goal is visibly done on this page',
          blocked: 'the goal cannot go further from here: a value is missing, an error shows, or a person must decide',
        },
      },
    }
    if (Object.keys(clickable).length) questions.click_target = { type: 'choice', instructions: 'If the next operation is a click, which element?', criteria: clickable }
    if (Object.keys(editable).length) questions.type_target = { type: 'choice', instructions: 'If the next operation is typing, which field?', criteria: editable }
    const t0 = performance.now()
    if (ctx.stepModel) {
      const r = await stepAdvice(ctx, page, recent)
      let operation: Operation | null = r.operation === 'blocked' && ctx.escalate ? null : r.operation
      let target = r.target
      let value = r.value
      let by = 'step'
      // A control the person refused is never the step model's answer again, and right after a refusal the escalation
      // decides: a refusal shows the small model's way forward was wrong.
      if (operation === 'click' && target && refused.has(target.text)) target = null
      if (justRefused && ctx.escalate) operation = null
      justRefused = false
      let settled = operation === 'done' || operation === 'blocked' || (operation === 'click' && target) || (operation === 'type' && target && value)
      if (settled && target && circling(`${operation}|${target.text}|${target.context ?? ''}|${value ?? ''}|${pathOf(page.url)}`)) {
        settled = false
        r.why = `going in circles (${operation} “${target.text}” again)`
      }
      if (!settled) {
        if (!ctx.escalate) {
          ctx.record({ step: `${ctx.step.id}·${n}`, kind: 'advance', question: r.why, state: list.join('\n').slice(0, 600), probabilities: r.probs, verdict: null, by, ms: performance.now() - t0, then: ctx.route('unsure') })
          return { verdict: null, page, clicked, reason: r.why }
        }
        const e = await ctx.escalate.step({ goal: ctx.goal, values: ctx.values, page: { title: page.title, url: page.url }, elements: list, recent })
        if (!e) return { verdict: null, page, clicked, reason: `${r.why}, and the escalation had no answer` }
        operation = e.operation
        target = e.element ? (els[e.element - 1] ?? null) : null
        value = e.value === null ? null : (ctx.values.find((x) => x.name === e.value)?.value ?? e.value)
        if (target?.input === 'select' && value) value = optionFor(target, value) ?? value
        by = 'claude'
        if ((operation === 'click' || operation === 'type') && !target) return { verdict: null, page, clicked, reason: 'the escalation named no element' }
      }
      const label = target ? `[${els.indexOf(target) + 1}] ${target.text}` : ''
      if (operation === 'done' || operation === 'blocked') {
        ctx.record({ step: `${ctx.step.id}·${n}`, kind: 'advance', question: operation, probabilities: r.probs, verdict: operation, by, ms: performance.now() - t0, then: ctx.route(operation) })
        return { verdict: operation, page, clicked, ...(operation === 'blocked' ? { reason: 'it could not go further' } : {}) }
      }
      if (operation === 'click' && target) {
        const g = await guardClick(ctx.decide, target, page)
        // Adding to a cart passes the guard once; a second time in the same run, the person is asked.
        const adds = ADD_TO_CART.test(target.text.trim())
        if (adds && ctx.memory?.has('added-to-cart') && !g.stop) Object.assign(g, { stop: 'purchase', by: 'words', again: true })
        if (g.stop) {
          const reason = (g as { again?: boolean }).again ? `it would add a second one to the cart: “${target.text}”` : `a ${g.stop} click (${g.by === 'words' ? 'by its words' : 'by the model'}): “${target.text}”`
          const yes = ctx.approve ? await ctx.approve(`May I click “${target.text}”? (${reason})`, g.stop) : false
          ctx.record({ step: `${ctx.step.id}·${n}`, kind: 'guard', question: `click ${label}`, probabilities: g.probabilities, verdict: g.stop, by: g.by === 'words' ? 'words' : (g.route ?? 'model'), ms: performance.now() - t0, then: yes ? { do: 'goto', target: ctx.step.id, next: null } : { do: 'ask_person', target: reason, next: null }, note: yes ? 'the person said yes' : 'asked the person' })
          if (!yes) {
            // "No" means not that control: with an escalation, it finds another way on from this page (once).
            if (!ctx.escalate || refused.has(target.text)) return { verdict: null, page, clicked, reason }
            refused.add(target.text)
            justRefused = true
            recent.push(`was not allowed to click ${target.text}`)
            continue
          }
        }
        ctx.record({ step: `${ctx.step.id}·${n}`, kind: 'advance', question: `click ${label}`, probabilities: r.probs, verdict: 'click', by, ms: performance.now() - t0, then: { do: 'goto', target: ctx.step.id, next: null } })
        if (adds) ctx.memory?.add('added-to-cart')
        clicked.push(target.text)
        recent.push(`clicked ${target.text}`)
        if (!ctx.act) return { verdict: null, page, clicked, reason: `would click “${target.text}” (dry run)` }
        await ctx.surface.click(target)
        continue
      }
      if (operation === 'type' && target && value) {
        ctx.record({ step: `${ctx.step.id}·${n}`, kind: 'advance', question: `type into ${label}: “${value.slice(0, 40)}”`, probabilities: r.probs, verdict: 'type', by, ms: performance.now() - t0, then: { do: 'goto', target: ctx.step.id, next: null } })
        recent.push(`typed into ${target.text}`)
        await typeInto(ctx, target, value)
        continue
      }
      return { verdict: null, page, clicked, reason: 'no value to type' }
    }
    const out = await ctx.decide({ state, questions }, ctx.signal)
    const probs = (k: string) => {
      const a = out.answers[k]
      return a && a.type === 'choice' ? a.probabilities : undefined
    }
    const [op, pOp] = best(probs('operation'))
    const [clickIdx, pClick, pClick2] = best(probs('click_target'))
    const [typeIdx, pType, pType2] = best(probs('type_target'))
    let operation: Operation | null = op && pOp >= (ctx.step.sure ?? OP_CUTOFF) ? (op as Operation) : null
    // "blocked" ends the stretch: with an escalation at hand, the small model alone doesn't get to say it.
    if (operation === 'blocked' && ctx.escalate) operation = null
    let target: Element | null = null
    let by = out.routes?.operation ?? 'model'
    if (operation === 'click') target = clickIdx && clear(pClick, pClick2) ? (els[Number(clickIdx) - 1] ?? null) : null
    if (operation === 'type') target = typeIdx && clear(pType, pType2) ? (els[Number(typeIdx) - 1] ?? null) : null
    let escalatedValue: string | null = null
    if (operation === null || ((operation === 'click' || operation === 'type') && !target)) {
      // Not sure: the escalation decides this one operation, or the step stops here.
      const why = operation === null ? `not sure what to do (${op} ${pOp.toFixed(2)})` : `not sure which element to ${operation}`
      if (!ctx.escalate) {
        ctx.record({ step: `${ctx.step.id}·${n}`, kind: 'advance', question: why, state: list.join('\n').slice(0, 600), probabilities: probs('operation') ?? null, verdict: null, by, ms: performance.now() - t0, then: ctx.route('unsure') })
        return { verdict: null, page, clicked, reason: why }
      }
      const r = await ctx.escalate.step({ goal: ctx.goal, values: ctx.values, page: { title: page.title, url: page.url }, elements: list, recent })
      if (!r) return { verdict: null, page, clicked, reason: `${why}, and the escalation had no answer` }
      operation = r.operation
      target = r.element ? (els[r.element - 1] ?? null) : null
      escalatedValue = r.value
      by = 'claude'
      if ((operation === 'click' || operation === 'type') && !target) return { verdict: null, page, clicked, reason: 'the escalation named no element' }
    }
    const label = target ? `[${els.indexOf(target) + 1}] ${target.text}` : ''
    if (operation === 'done' || operation === 'blocked') {
      ctx.record({ step: `${ctx.step.id}·${n}`, kind: 'advance', question: operation, probabilities: probs('operation') ?? null, verdict: operation, by, ms: performance.now() - t0, then: ctx.route(operation) })
      return { verdict: operation, page, clicked, ...(operation === 'blocked' ? { reason: 'it could not go further' } : {}) }
    }
    if (operation === 'click' && target) {
      const g = await guardClick(ctx.decide, target, page)
      if (g.stop) {
        const reason = `a ${g.stop} click (${g.by === 'words' ? 'by its words' : 'by the model'}): “${target.text}”`
        const yes = ctx.approve ? await ctx.approve(`May I click “${target.text}”? (${reason})`, g.stop) : false
        ctx.record({ step: `${ctx.step.id}·${n}`, kind: 'guard', question: `click ${label}`, probabilities: g.probabilities, verdict: g.stop, by: g.by === 'words' ? 'words' : (g.route ?? 'model'), ms: performance.now() - t0, then: yes ? { do: 'goto', target: ctx.step.id, next: null } : { do: 'ask_person', target: reason, next: null }, note: yes ? 'the person said yes' : 'asked the person' })
        if (!yes) return { verdict: null, page, clicked, reason }
      }
      ctx.record({ step: `${ctx.step.id}·${n}`, kind: 'advance', question: `click ${label}`, probabilities: probs('click_target') ?? null, verdict: 'click', by, ms: performance.now() - t0, then: { do: 'goto', target: ctx.step.id, next: null } })
      clicked.push(target.text)
      recent.push(`clicked ${target.text}`)
      if (!ctx.act) return { verdict: null, page, clicked, reason: `would click “${target.text}” (dry run)` }
      await ctx.surface.click(target)
      continue
    }
    if (operation === 'type' && target) {
      const v = escalatedValue !== null ? { value: ctx.values.find((x) => x.name === escalatedValue)?.value ?? escalatedValue, by: 'claude', probabilities: undefined } : await valueFor(ctx, target, page, list, recent)
      if (v.value === undefined || v.value === null) {
        const reason = `no value for “${target.text}”`
        ctx.record({ step: `${ctx.step.id}·${n}`, kind: 'advance', question: `type into ${label}`, probabilities: v.probabilities ?? null, verdict: null, by: v.by, ms: performance.now() - t0, then: ctx.route('unsure') })
        return { verdict: null, page, clicked, reason }
      }
      ctx.record({ step: `${ctx.step.id}·${n}`, kind: 'advance', question: `type into ${label}: “${v.value.slice(0, 40)}”`, probabilities: probs('type_target') ?? null, verdict: 'type', by: by === 'claude' ? 'claude' : v.by === 'code' ? by : v.by, ms: performance.now() - t0, then: { do: 'goto', target: ctx.step.id, next: null } })
      recent.push(`typed into ${target.text}`)
      await typeInto(ctx, target, v.value)
    }
  }
  return { verdict: null, page, clicked, reason: `still not done after ${budget} operations` }
}

/** fill: every empty field gets one of the named values, or is left for the person. */
export async function runFill(ctx: StepContext): Promise<StepOutcome> {
  const t0 = performance.now()
  const page = await ctx.surface.read()
  const fields = page.elements.filter((e) => isEditable(e) && !(e.value ?? '').trim())
  const list = page.elements.filter((e) => e.text.trim()).slice(0, MAX_ELEMENTS).map((e, i) => `[${i + 1}] ${describe(e)}`)
  const decided = new Map<Element, { value: string | null; by: string }>()
  const unsure: Element[] = []
  // atelier-step reads the whole form once: every field's value from one pass.
  let fromStep: Map<Element, { value: string | null } | null> | null = null
  if (ctx.stepModel) {
    const items = stepItems(page)
    const a = await ctx.stepModel({ goal: ctx.goal, values: ctx.values, before: [], page: `${page.title} (${pathOf(page.url)})`, elements: items.map((x) => x.step) })
    fromStep = new Map()
    items.forEach((it, i) => {
      const vs = a.values[i]
      if (!it.element || !vs) return
      const ranked = vs.map((p, j) => [j, p] as const).sort((x, y) => y[1] - x[1])
      const [first, second] = ranked
      if (!first || !clear(first[1], second?.[1] ?? 0, 0.5)) return fromStep!.set(it.element, null)
      const named = first[0] === 0 ? null : ctx.values[first[0] - 1]!.value
      const v = named && it.element.input === 'select' ? optionFor(it.element, named) : named
      fromStep!.set(it.element, named && it.element.input === 'select' && !v ? null : { value: v })
    })
  }
  for (const e of fields) {
    const hinted = e.hint && !NEVER.test(e.hint) ? valueByHint(e, ctx.values) : null
    if (hinted) {
      decided.set(e, { value: e.input === 'select' ? (optionFor(e, hinted.value) ?? hinted.value) : hinted.value, by: 'code' })
      continue
    }
    if (fromStep) {
      const s = fromStep.get(e)
      if (s) decided.set(e, { value: s.value, by: 'step' })
      else if (e.input === 'select') {
        const v = await valueFor({ ...ctx, escalate: undefined }, e, page, list, [])
        if (v.value === undefined) unsure.push(e)
        else decided.set(e, { value: v.value, by: v.by })
      } else unsure.push(e)
      continue
    }
    const v = await valueFor({ ...ctx, escalate: undefined }, e, page, list, [])
    if (v.value === undefined) unsure.push(e)
    else decided.set(e, { value: v.value, by: v.by })
  }
  if (unsure.length && ctx.escalate) {
    // One call for the whole form, not one per field.
    const numbered = unsure.map((e, i) => `[${i + 1}] ${describe(e)}`)
    const r = await ctx.escalate.fill({ purpose: ctx.goal, values: ctx.values, page: { title: page.title, url: page.url }, fields: numbered })
    for (const [i, e] of unsure.entries()) {
      const name = r[String(i + 1)]
      if (name === undefined) continue
      const named = name === null ? null : (ctx.values.find((v) => v.name === name)?.value ?? name)
      decided.set(e, { value: e.input === 'select' && named ? (optionFor(e, named) ?? named) : named, by: 'claude' })
    }
  }
  const left = fields.filter((e) => !decided.has(e))
  for (const [e, v] of decided) {
    ctx.record({ step: ctx.step.id, kind: 'fill', question: `${describe(e)} ← ${v.value === null ? '(left empty)' : `“${v.value.slice(0, 40)}”`}`, verdict: v.value === null ? 'none' : 'value', by: v.by, ms: 0, then: { do: 'goto', target: ctx.step.id, next: null } })
    if (v.value !== null) await typeInto(ctx, e, v.value)
  }
  const verdict = left.length === 0 ? 'done' : null
  ctx.record({ step: ctx.step.id, kind: 'fill', question: `${fields.length} empty fields: ${decided.size} decided, ${left.length} left`, verdict, by: 'code', ms: performance.now() - t0, then: ctx.route(verdict ?? 'unsure') })
  return { verdict, page: await ctx.surface.read(), clicked: [], ...(left.length ? { reason: `not sure what goes into ${left.map((e) => `“${e.text}”`).join(', ')}` } : {}) }
}
