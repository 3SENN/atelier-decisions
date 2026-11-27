/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// atelier-vision's questions: a line-for-line port of vision/format.py (the contract between training and this
// runtime). Atelier's own System One questions, asked about a picture: every answer is a short text, and the model says
// how well the picture matches each one.
//   choice  one text per option: its description, or the option's own name when it has none
//   score   one text per level (declined for now: no version has been measured on levels)
//   noul    the "true" text is what a yes looks like (without criteria, the question itself); a "false" text is matched
//           too when there is one
// A text is lowercased, cut to 63 tokens, closed with <eos> and padded to 64. src/vision/parity.ts checks the ids.

export type Content = string | number | boolean | null | Content[] | { [k: string]: Content }
export interface Question {
  type: 'choice' | 'noul' | 'score'
  instructions?: Content | null
  criteria?: Record<string, Content | null> | Content[] | null
}

export const MAX_TOKENS = 64
export const PAD = 0
export const EOS = 1
export const SCORE_UNMEASURED = 'levels on a scale are not measured for pictures yet'

// What Python's str.split() splits on, spelled out so both sides agree on every character.
const SPACE = /[\t\n\v\f\r \x1c-\x1f\x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+/

/** One line, single spaces, lowercase. */
export function clean(text: string): string {
  return text.split(SPACE).filter(Boolean).join(' ').toLowerCase()
}

export interface VisionTokenize {
  /** Token ids without special tokens. */
  encode(text: string): number[]
}

/** A text's 64 token ids, and whether it was too long and lost its end. */
export function encodeText(tok: VisionTokenize, text: string): { ids: number[]; cut: boolean } {
  const raw = tok.encode(clean(text))
  const ids = [...raw.slice(0, MAX_TOKENS - 1), EOS]
  while (ids.length < MAX_TOKENS) ids.push(PAD)
  return { ids, cut: raw.length > MAX_TOKENS - 1 }
}

/** The answers' names: a choice's keys, a score's level numbers, false and true. */
export function labels(q: Question): string[] {
  if (q.type === 'choice') return Object.keys(q.criteria as Record<string, Content>)
  if (q.type === 'score') return (q.criteria as Content[]).map((_, i) => String(i))
  return ['false', 'true']
}

const said = (value: unknown): string | null => (typeof value === 'string' && value.trim() ? value.trim() : null)

/** The text to match the picture against for each of labels(q); null where there is nothing to match. */
export function texts(q: Question): Array<string | null> {
  if (q.type === 'choice') return Object.entries(q.criteria as Record<string, Content | null>).map(([k, v]) => said(v) ?? said(k))
  if (q.type === 'score') return (q.criteria as Content[]).map((v) => said(v))
  const criteria = (q.criteria ?? {}) as Record<string, Content | null>
  return [said(criteria.false), said(criteria.true) ?? said(q.instructions)]
}

/** Why this question can't be asked of a picture, or null when it can. */
export function unanswerable(q: Question): string | null {
  const kind = (q as { type?: unknown }).type
  if (kind !== 'choice' && kind !== 'noul' && kind !== 'score') return `unknown question type ${JSON.stringify(kind ?? null)}`
  const criteria = q.criteria
  const isMap = criteria !== null && criteria !== undefined && typeof criteria === 'object' && !Array.isArray(criteria)
  if (kind === 'choice' && (!isMap || Object.keys(criteria as object).length < 2)) return 'a choice needs two or more options'
  if (kind === 'score' && (!Array.isArray(criteria) || criteria.length < 2)) return 'a score needs two or more levels'
  if (kind === 'score') return SCORE_UNMEASURED
  if (kind === 'noul' && criteria !== null && criteria !== undefined && !isMap) return 'the criteria of a yes/no question are {true, false}'
  const found = texts(q)
  if (kind === 'noul') return found[1] ? null : 'nothing says what a yes looks like'
  return found.every(Boolean) ? null : 'an option without a text'
}
