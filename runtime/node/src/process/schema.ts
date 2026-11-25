/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// A process: a task written down once as small steps, then run over the decision model, step by step, item by
// item. A language model can write one from a task ("sort my inbox into reply, read later and archive"), or it
// comes ready-made from processes/. Running it needs no language model: each step is one bounded question for the
// decision model (ask), a comparison done by code (check), a control chosen by what it does (pick), or a fixed action
// the planner already knows (act: click "My orders"). Two more run many decisions each, the way browser agents built
// on decision models do: advance moves toward a goal one operation at a time (which operation, which element, which
// value: one request), and fill fills a form's fields from the process's named values. Neither writes text: what
// is typed is always one of the values the planner wrote down.
//
// The shape suits structured outputs: options and routes are lists, never maps with free keys, and every field is
// present (null where it doesn't apply), so a planner's output can be held to PROCESS_JSON_SCHEMA exactly.

export type StepKind = 'ask' | 'check' | 'pick' | 'act' | 'advance' | 'fill'
export type AnswerKind = 'yes_no' | 'choice' | 'level'
export type CheckOp = '<' | '<=' | '>' | '>=' | '=' | '!=' | 'contains' | 'matches'
export type ActionKind = 'goto' | 'put' | 'ask_person' | 'click' | 'open' | 'notify' | 'stop'

export interface Option {
  name: string
  meaning: string
}

export interface Action {
  do: ActionKind
  /** goto: a step id; put: a bucket; ask_person / notify: the reason shown; click: 'picked' (a pick's control) or a
   *  control's label as the page shows it; open: a field holding a URL, or a URL. */
  target: string
  /** click, open and notify: the step to continue at afterwards (null: the item ends there). */
  next: string | null
}

export interface Route {
  /** An option's name (choice, level), 'yes' or 'no' (yes_no, check), 'picked' (pick), or 'unsure'. */
  when: string
  then: Action
}

export interface Step {
  id: string
  kind: StepKind
  /** ask: the question about the item or page. pick: the question about one control ("What does this control do?"). */
  question: string | null
  answer: AnswerKind | null
  /** ask and pick: the answers and what each means (yes_no: exactly yes and no; level: lowest first). */
  options: Option[] | null
  /** ask: the item's fields (or 'page') the decision model reads; check: the one field compared. */
  read: string[]
  op: CheckOp | null
  /** check: the value to compare with. pick: the option(s) a control must be, comma-separated. */
  value: string | null
  /** Optional cutoff for a sure answer (0.5-0.99); null uses Atelier's own (yes 0.7 / no 0.3, a choice 0.75, a level 0.6). */
  sure: number | null
  /** advance: at most this many operations (default 12). */
  budget: number | null
  routes: Route[]
}

export interface ItemSource {
  /** CSS selector for each item on a page. */
  selector: string
  fields: Array<{ name: string; selector: string }>
}

export interface Process {
  id: string
  title: string
  /** The task in the person's words. */
  task: string
  /** 'items': every item runs the steps; 'page': the page itself is the one item. */
  over: 'items' | 'page'
  /** How to read items from a page (over: items); null when items come from elsewhere (a fixture, a list). */
  items: ItemSource | null
  /** Where an item can end up (put). */
  buckets: string[]
  /** What may be typed or chosen (advance, fill): named values from the task and the person ("postcode": "3512 AB"). */
  values: Value[]
  start: string
  steps: Step[]
}

export interface Value {
  /** What it is, in a few words ("email address", "shoe size", "message about the bin"). */
  name: string
  value: string
}

const STEP_KINDS: StepKind[] = ['ask', 'check', 'pick', 'act', 'advance', 'fill']
const ANSWERS: AnswerKind[] = ['yes_no', 'choice', 'level']
const OPS: CheckOp[] = ['<', '<=', '>', '>=', '=', '!=', 'contains', 'matches']
const ACTIONS: ActionKind[] = ['goto', 'put', 'ask_person', 'click', 'open', 'notify', 'stop']
export const MAX_STEPS = 12
export const MAX_OPTIONS = 8

const nullable = (schema: object) => ({ anyOf: [schema, { type: 'null' }] })
const strings = { type: 'array', items: { type: 'string' } }

/** The process as a JSON Schema for structured outputs: every property required, additionalProperties false. */
export const PROCESS_JSON_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['id', 'title', 'task', 'over', 'items', 'buckets', 'values', 'start', 'steps'],
  properties: {
    id: { type: 'string', description: 'kebab-case' },
    title: { type: 'string' },
    task: { type: 'string' },
    over: { type: 'string', enum: ['items', 'page'] },
    items: nullable({
      type: 'object',
      additionalProperties: false,
      required: ['selector', 'fields'],
      properties: {
        selector: { type: 'string' },
        fields: {
          type: 'array',
          items: { type: 'object', additionalProperties: false, required: ['name', 'selector'], properties: { name: { type: 'string' }, selector: { type: 'string' } } },
        },
      },
    }),
    buckets: strings,
    values: {
      type: 'array',
      items: { type: 'object', additionalProperties: false, required: ['name', 'value'], properties: { name: { type: 'string' }, value: { type: 'string' } } },
    },
    start: { type: 'string' },
    steps: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'kind', 'question', 'answer', 'options', 'read', 'op', 'value', 'sure', 'budget', 'routes'],
        properties: {
          id: { type: 'string' },
          kind: { type: 'string', enum: STEP_KINDS },
          question: nullable({ type: 'string' }),
          answer: nullable({ type: 'string', enum: ANSWERS }),
          options: nullable({
            type: 'array',
            items: { type: 'object', additionalProperties: false, required: ['name', 'meaning'], properties: { name: { type: 'string' }, meaning: { type: 'string' } } },
          }),
          read: strings,
          op: nullable({ type: 'string', enum: OPS }),
          value: nullable({ type: 'string' }),
          sure: nullable({ type: 'number' }),
          budget: nullable({ type: 'integer' }),
          routes: {
            type: 'array',
            items: {
              type: 'object',
              additionalProperties: false,
              required: ['when', 'then'],
              properties: {
                when: { type: 'string' },
                then: {
                  type: 'object',
                  additionalProperties: false,
                  required: ['do', 'target', 'next'],
                  properties: { do: { type: 'string', enum: ACTIONS }, target: { type: 'string' }, next: nullable({ type: 'string' }) },
                },
              },
            },
          },
        },
      },
    },
  },
} as const

/** Fills what a hand-written process may leave out (null fields, read) so the runner sees one shape. */
export function normalize(raw: unknown): Process {
  const p = raw as Process
  return {
    ...p,
    items: p.items ?? null,
    buckets: p.buckets ?? [],
    values: p.values ?? [],
    steps: (p.steps ?? []).map((s) => ({
      ...s,
      question: s.question ?? null,
      answer: s.answer ?? null,
      options: s.options ?? null,
      read: s.read ?? [],
      op: s.op ?? null,
      // Matching is case-insensitive already; a Python-style (?i) in front would not compile in JavaScript.
      value: s.op === 'matches' && typeof s.value === 'string' ? s.value.replace(/^\(\?i\)/, '') : (s.value ?? null),
      sure: s.sure ?? null,
      budget: s.budget ?? null,
      routes: (s.routes ?? []).map((r) => ({ when: r.when, then: { do: r.then?.do, target: r.then?.target ?? '', next: r.then?.next ?? null } })),
    })),
  }
}

/** Everything wrong with a process, in words a planner can fix; empty when it can run. */
export function problems(p: Process): string[] {
  const out: string[] = []
  if (!p || typeof p !== 'object') return ['not an object']
  if (!/^[a-z0-9]+(-[a-z0-9]+)*$/.test(p.id ?? '')) out.push(`id "${p.id}" must be kebab-case`)
  if (p.over !== 'items' && p.over !== 'page') out.push('over must be "items" or "page"')
  if (!Array.isArray(p.steps) || p.steps.length === 0) return [...out, 'steps is empty']
  if (p.steps.length > MAX_STEPS) out.push(`at most ${MAX_STEPS} steps (has ${p.steps.length})`)
  const ids = new Set<string>()
  for (const s of p.steps) {
    if (ids.has(s.id)) out.push(`step id "${s.id}" is used twice`)
    ids.add(s.id)
  }
  if (!ids.has(p.start)) out.push(`start "${p.start}" is not a step id`)
  const buckets = new Set(p.buckets)
  const valueNames = (p.values ?? []).map((v) => v.name)
  if (new Set(valueNames).size !== valueNames.length) out.push('value names repeat')
  for (const s of p.steps) {
    const at = `step "${s.id}"`
    if (!STEP_KINDS.includes(s.kind)) {
      out.push(`${at}: kind must be ask, check, pick, act, advance or fill`)
      continue
    }
    let answers: string[] = []
    if (s.kind === 'ask') {
      if (!s.question?.trim()) out.push(`${at}: an ask step needs a question`)
      if (!s.answer || !ANSWERS.includes(s.answer)) out.push(`${at}: answer must be yes_no, choice or level`)
      const names = (s.options ?? []).map((o) => o.name)
      if (s.answer === 'yes_no' && (names.length !== 2 || !names.includes('yes') || !names.includes('no'))) out.push(`${at}: a yes_no step's options are exactly "yes" and "no", each with its meaning`)
      if ((s.answer === 'choice' || s.answer === 'level') && (names.length < 2 || names.length > MAX_OPTIONS)) out.push(`${at}: a ${s.answer} step needs 2-${MAX_OPTIONS} options`)
      if (new Set(names).size !== names.length) out.push(`${at}: option names repeat`)
      for (const o of s.options ?? []) if (!o.meaning?.trim()) out.push(`${at}: option "${o.name}" needs a meaning`)
      if (s.read.length === 0) out.push(`${at}: read names no field (use "page" for the whole page)`)
      answers = names
    } else if (s.kind === 'check') {
      if (s.read.length !== 1) out.push(`${at}: a check reads exactly one field`)
      if (s.read[0] === 'page' && s.op && ['<', '<=', '>', '>=', '=', '!='].includes(s.op))
        out.push(`${at}: a comparison on the whole page would read the page's first number (a cart count, a model number); compare an item's field, or ask about the page instead`)
      if (!s.op || !OPS.includes(s.op)) out.push(`${at}: op must be one of ${OPS.join(' ')}`)
      if (s.value === null || s.value === undefined || s.value === '') out.push(`${at}: a check needs a value`)
      if (s.op && ['<', '<=', '>', '>='].includes(s.op) && !Number.isFinite(parseNumber(s.value ?? ''))) out.push(`${at}: "${s.value}" is not a number to compare with`)
      if (s.op === 'matches') {
        try {
          new RegExp(s.value ?? '', 'i')
        } catch {
          out.push(`${at}: "${s.value}" is not a regular expression`)
        }
      }
      answers = ['yes', 'no']
    } else if (s.kind === 'advance') {
      if (!s.question?.trim()) out.push(`${at}: an advance step needs a question: the goal for this stretch`)
      if (s.budget !== null && (s.budget < 1 || s.budget > 40)) out.push(`${at}: budget is 1-40 operations`)
      if (p.over !== 'page') out.push(`${at}: advance works on a page (over: "page")`)
      answers = ['done', 'blocked']
    } else if (s.kind === 'fill') {
      // A ready-made template (task "{goal}") gets its values with the task it runs for.
      if (p.values.length === 0 && !p.task.includes('{goal}')) out.push(`${at}: a fill step needs the process's values`)
      if (p.over !== 'page') out.push(`${at}: fill works on a page (over: "page")`)
      answers = ['done']
    } else if (s.kind === 'act') {
      if (s.routes.length !== 1 || s.routes[0]?.when !== 'done') out.push(`${at}: an act step has exactly one route, "done"`)
      answers = ['done']
    } else {
      // pick by code: op contains/matches, value the text or pattern a control's label or row must have.
      // pick by the model: each control is sorted into the options; value names the option(s) to look for.
      if (s.op === 'contains' || s.op === 'matches') {
        if (!s.value) out.push(`${at}: a pick by code needs a value to match`)
        if (s.op === 'matches') {
          try {
            new RegExp(s.value ?? '', 'i')
          } catch {
            out.push(`${at}: "${s.value}" is not a JavaScript regular expression`)
          }
        }
        answers = ['picked']
      } else {
        if (!s.question?.trim()) out.push(`${at}: a pick step needs a question about one control ("What does this control do?")`)
        if (s.answer !== 'choice' && s.answer !== 'yes_no') out.push(`${at}: a pick's answer is choice or yes_no`)
        const names = (s.options ?? []).map((o) => o.name)
        if (names.length < 2 || names.length > MAX_OPTIONS) out.push(`${at}: a pick needs 2-${MAX_OPTIONS} options (the kinds of control)`)
        if (s.answer === 'yes_no' && (names.length !== 2 || !names.includes('yes') || !names.includes('no'))) out.push(`${at}: a yes_no pick's options are exactly "yes" and "no"`)
        const wanted = (s.value ?? '').split(',').map((v) => v.trim()).filter(Boolean)
        if (wanted.length === 0 || wanted.some((w) => !names.includes(w))) out.push(`${at}: a pick's value names the option(s) to look for, comma-separated (has "${s.value}")`)
      }
      answers = ['picked']
    }
    if (s.sure !== null && (s.sure < 0.5 || s.sure > 0.99)) out.push(`${at}: sure must be between 0.5 and 0.99`)
    const covered = new Set(s.routes.map((r) => r.when))
    for (const r of s.routes) {
      if (r.when !== 'unsure' && !answers.includes(r.when)) out.push(`${at}: route "${r.when}" is not one of its answers (${[...answers, 'unsure'].join(', ')})`)
      if (!ACTIONS.includes(r.then?.do)) out.push(`${at}: route "${r.when}" does "${r.then?.do}", not one of ${ACTIONS.join(' ')}`)
      else if (r.then.do === 'goto' && !ids.has(r.then.target)) out.push(`${at}: goto "${r.then.target}" is not a step id`)
      else if (r.then.do === 'goto' && r.then.target === s.id && s.kind !== 'pick') out.push(`${at}: goes to itself`)
      else if (r.then.do === 'put' && !buckets.has(r.then.target)) out.push(`${at}: bucket "${r.then.target}" is not in buckets`)
      if (r.then?.next && !ids.has(r.then.next)) out.push(`${at}: route "${r.when}" continues at "${r.then.next}", not a step id`)
      if (r.then?.do === 'click' && p.over !== 'page') out.push(`${at}: click works on a page (over: "page")`)
    }
    for (const a of answers) if (!covered.has(a)) out.push(`${at}: no route for "${a}"`)
    if (s.kind === 'act' && covered.has('unsure')) out.push(`${at}: an act step is never unsure`)
  }
  return out
}

/** A number as people write it on pages: "€1.299,00", "$1,299.00", "1 299", "12.5%", "-3". NaN when there is none. */
export function parseNumber(text: string): number {
  const m = /-?\d[\d.,\s  ']*/.exec(text)
  if (!m) return Number.NaN
  let s = m[0].replace(/[\s  ']/g, '').replace(/[.,]$/, '')
  const lastDot = s.lastIndexOf('.')
  const lastComma = s.lastIndexOf(',')
  if (lastDot >= 0 && lastComma >= 0) {
    // Both: the later one is the decimal mark.
    s = lastComma > lastDot ? s.replace(/\./g, '').replace(',', '.') : s.replace(/,/g, '')
  } else if (lastComma >= 0) {
    // "1,299" is a thousand; "12,5" and "0,99" are decimals.
    s = /^-?\d{1,3}(,\d{3})+$/.test(s) ? s.replace(/,/g, '') : s.replace(',', '.')
  } else if (lastDot >= 0 && /^-?\d{1,3}(\.\d{3})+$/.test(s) && (s.split('.').length > 2 || /[€£]|\beur\b/i.test(text))) {
    // "1.299.000", or "€1.299" as Dutch and German shops write it
    s = s.replace(/\./g, '')
  }
  return Number(s)
}
