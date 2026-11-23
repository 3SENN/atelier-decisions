/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// nano first, mini for the rest, the rule last. atelier-nano answers in milliseconds; a question it declines (an
// open question, text it can't read) or is unsure about by Atelier's own cutoffs goes to atelier-mini; what the
// mini is unsure about stays unsure, so Atelier's rule decides. atelier-decisions: analysis/cascade.py measures it.

import type { Answer, NanoDecider } from './decider.ts'
import { fixedKind, optionLabels, type DecisionRequest, type Kind } from './format.ts'
import type { MiniDecider } from './mini.ts'
import { probabilities, verdict } from './readers.ts'

export type Route = 'nano' | 'mini'

export interface CascadeOptions {
  /** Decisions the mini answers directly (the nano is not asked), e.g. ['skill'] where the mini is the better reader. */
  miniFirst?: Array<Kind | 'open'>
}

export class CascadeDecider {
  private readonly nano: NanoDecider
  private readonly mini: MiniDecider
  private readonly miniFirst: Set<string>

  // No parameter properties: node's type stripping does not transform them.
  constructor(nano: NanoDecider, mini: MiniDecider, options: CascadeOptions = {}) {
    this.nano = nano
    this.mini = mini
    this.miniFirst = new Set(options.miniFirst ?? [])
  }

  async decide(request: DecisionRequest): Promise<{ answers: Record<string, Answer>; usage: { input_tokens: number; output_tokens: number }; routes: Record<string, Route> }> {
    const direct = Object.entries(request.questions).filter(([, q]) => this.miniFirst.has(fixedKind(q) ?? 'open'))
    const forNano = Object.fromEntries(Object.entries(request.questions).filter(([, q]) => !this.miniFirst.has(fixedKind(q) ?? 'open')))
    const first = Object.keys(forNano).length
      ? await this.nano.decide({ state: request.state, questions: forNano })
      : { answers: {} as Record<string, Answer>, usage: { input_tokens: 0, output_tokens: 0 } }
    const routes: Record<string, Route> = {}
    const pending: DecisionRequest['questions'] = Object.fromEntries(direct)
    for (const [key] of direct) routes[key] = 'mini'
    for (const [key, q] of Object.entries(forNano)) {
      const a = first.answers[key]
      const labels = optionLabels(q)
      const p = a ? probabilities(a, labels) : null
      const sure = p !== null && verdict(fixedKind(q) ?? 'open', labels, p) !== null
      routes[key] = sure ? 'nano' : 'mini'
      if (!sure) pending[key] = q
    }
    if (Object.keys(pending).length === 0) return { ...first, routes }
    const second = await this.mini.decide({ state: request.state, questions: pending })
    return {
      answers: { ...first.answers, ...second.answers },
      usage: { input_tokens: first.usage.input_tokens + second.usage.input_tokens, output_tokens: 0 },
      routes,
    }
  }

  async release(): Promise<void> {
    await Promise.all([this.nano.release(), this.mini.release()])
  }
}
