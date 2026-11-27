/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// From how well a picture matches each answer's text to a System One answer: a port of vision/decide.py.
// The towers give cos(picture, text) for every text of a question; z = scale * cos + bias is SigLIP's own log-odds
// that the text describes the picture.
//   choice, score   softmax(z / t) over the options; with MANY options or more, a temperature of its own
//   noul            with a "false" text: sigmoid(a * z_true - b * z_false + c); without one: sigmoid(a * z_true + c)
// t, a, b and c come from vision/calibrate.py (vision.json).

import type { Content, Question } from './format.ts'

/** Options from which a choice is read with the "many" temperature: sixty wrong options that each look a little possible add up. */
export const MANY = 16

export interface Calibration {
  choice: { t: number; many?: number }
  score: { t: number }
  noul: { pair: { a: number; b: number; c: number }; single: { a: number; c: number } }
}
/** Before calibration: SigLIP's own reading. */
export const RAW: Calibration = { choice: { t: 1, many: 1 }, score: { t: 1 }, noul: { pair: { a: 1, b: 0, c: 0 }, single: { a: 1, c: 0 } } }

export type Answer =
  | { type: 'refusal'; reason: string }
  | { type: 'noul'; noul: number }
  | { type: 'choice'; choice: string; probabilities: Record<string, number>; confidence: number }
  | { type: 'score'; score: number; probabilities: Record<string, number>; legend: Record<string, Content>; confidence: number }

function softmax(xs: number[]): number[] {
  const m = Math.max(...xs)
  const e = xs.map((x) => Math.exp(x - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((x) => x / s)
}

const sigmoid = (x: number): number => (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)))

/** The probability of each of labels(q), from the cosines of texts(q) (null where there was no text). */
export function probabilities(kind: Question['type'], cos: Array<number | null>, scale: number, bias: number, cal: Calibration = RAW): number[] {
  const z = cos.map((c) => (c === null ? null : scale * c + bias))
  if (kind === 'noul') {
    const [zFalse, zTrue] = z as [number | null, number]
    const p = zFalse === null ? sigmoid(cal.noul.single.a * zTrue + cal.noul.single.c) : sigmoid(cal.noul.pair.a * zTrue - cal.noul.pair.b * zFalse + cal.noul.pair.c)
    return [1 - p, p]
  }
  const t = kind === 'choice' && z.length >= MANY ? (cal.choice.many ?? cal.choice.t) : cal[kind].t
  return softmax((z as number[]).map((x) => x / t))
}

/** pplx_decider.protocol.answer, as src/decider.ts has it: the System One answer for one question. */
export function answer(q: Question, names: string[], p: number[]): Answer {
  const at = (i: number): number => p[i] ?? 0
  if (q.type === 'noul') return { type: 'noul', noul: at(1) }
  let best = 0
  for (let i = 1; i < p.length; i++) if (at(i) > at(best)) best = i
  const dist = Object.fromEntries(names.map((k, i) => [k, at(i)]))
  if (q.type === 'choice') {
    const n = p.length
    const confidence = n === 1 ? 1 : (at(best) - 1 / n) / (1 - 1 / n)
    return { type: 'choice', probabilities: dist, choice: names[best] ?? '', confidence: Math.max(0, Math.min(1, confidence)) }
  }
  const levels = q.criteria as Content[]
  const distance = p.reduce((s, pi, i) => s + pi * Math.abs(i - best), 0)
  const midpoint = (p.length - 1) / 2
  const baseline = p.reduce((s, _, i) => s + Math.abs(i - midpoint), 0) / p.length
  return {
    type: 'score',
    probabilities: dist,
    legend: Object.fromEntries(names.map((k, i) => [k, levels[i] ?? null])),
    score: p.reduce((s, pi, i) => s + i * pi, 0),
    confidence: Math.max(0, 1 - distance / baseline),
  }
}
