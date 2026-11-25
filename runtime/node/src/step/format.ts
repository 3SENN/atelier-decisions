/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// atelier-step's input: a line-for-line port of step/format.py (the contract between training and this runtime).
// src/step/parity.ts checks the two give the same token ids, marker positions and answers.

export const STEP = '<unused14>'
export const VAL = '<unused15>'
export const EL = '<unused16>'
export const TX = '<unused17>'
export const OPS = ['click', 'type', 'done', 'blocked'] as const
export type StepOp = (typeof OPS)[number]
export const NONE_VALUE = 'none of these'
export const CLICKABLE = new Set(['link', 'button', 'checkbox', 'radio'])
export const EDITABLE = new Set(['field', 'select'])
export const MAX_LEN = 1024
const LIMITS = { goal: 96, value: 24, values: 16, done: 48, page: 24, item: 28 }

export interface StepElement {
  role: 'link' | 'button' | 'field' | 'select' | 'checkbox' | 'radio' | 'text'
  label: string
  value?: string
  options?: string[]
  checked?: boolean
  context?: string
}

export interface StepInput {
  goal: string
  values: Array<{ name: string; value: string }>
  before: string[]
  page: string
  elements: StepElement[]
}

export interface StepTokenize {
  encode(text: string): number[]
  id(token: string): number
  bos: number
  eos: number
}

export function describe(e: StepElement): string {
  const { role, label } = e
  if (role === 'text') return label
  if (role === 'link' || role === 'button') return `${role} ${label}${e.context && [...label].length < 20 ? ` (${[...e.context].slice(0, 60).join('')})` : ''}`
  if (role === 'checkbox' || role === 'radio') return `${role} ${label}: ${e.checked ? 'checked' : 'not checked'}`
  if (role === 'select') return `select ${label}: ${e.value || 'nothing chosen'} (options: ${(e.options ?? []).slice(0, 8).join(' | ')})`
  return `field ${label}: ${e.value ? [...e.value].slice(0, 40).join('') : 'empty'}`
}

/** Token ids of one step, the EL markers' positions (and which element each is), the VAL markers' positions. */
export function encodeStep(tok: StepTokenize, ex: StepInput, maxLen = MAX_LEN): { ids: number[]; els: number[]; elIndex: number[]; vals: number[] } {
  const ids: number[] = [tok.bos, tok.id(STEP), ...tok.encode(ex.goal).slice(0, LIMITS.goal), tok.eos]
  const vals: number[] = []
  for (const text of [NONE_VALUE, ...ex.values.slice(0, LIMITS.values).map((v) => `${v.name}: ${v.value}`)]) {
    vals.push(ids.length)
    ids.push(tok.id(VAL), ...tok.encode(text).slice(0, LIMITS.value))
  }
  const done = tok.encode(ex.before.join('\n') || 'nothing yet')
  ids.push(tok.eos, ...done.slice(Math.max(0, done.length - LIMITS.done)), tok.eos)
  ids.push(...tok.encode(ex.page).slice(0, LIMITS.page), tok.eos)
  const budget = maxLen - ids.length - 1
  const els: number[] = []
  const elIndex: number[] = []
  let used = 0
  for (const [i, e] of ex.elements.entries()) {
    const item = [tok.id(e.role === 'text' ? TX : EL), ...tok.encode(describe(e)).slice(0, LIMITS.item)]
    if (used + item.length > budget) break // the page is cut at the end, as step/format.py does without keep
    if (e.role !== 'text') {
      els.push(ids.length)
      elIndex.push(i)
    }
    ids.push(...item)
    used += item.length
  }
  ids.push(tok.eos)
  return { ids: ids.slice(0, maxLen), els, elIndex, vals }
}
