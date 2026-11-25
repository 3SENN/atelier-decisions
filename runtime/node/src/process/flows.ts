/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// Whole flows on real pages (lab/apps): order a product, find one, fill in a form. They run in a browser over CDP,
// and afterwards what really ended up in the cart or the form is read from the page and compared with what the
// person asked for. A flow succeeds only when everything is right and nothing was paid or sent.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import type { TabSurface } from './cdp.ts'
import type { Decide } from './decide.ts'
import { loadProcesses } from './fixtures.ts'
import { plan, type Backend, type PlanResult } from './planner.ts'
import { runProcess, type ItemResult, type RunOptions } from './run.ts'
import { normalize, problems, type Process, type Value } from './schema.ts'

export interface FlowTask {
  key: string
  start: string
  storage: string
  process: string | null
  task: string
  values: Value[]
  expect: { path?: string; state: Record<string, unknown> }
}

export interface Flows {
  person: Value[]
  tasks: FlowTask[]
}

export const FLOWS_FILE = new URL('../../lab/flows.json', import.meta.url)
export const FLOW_PLANS_DIR = new URL('../../lab/plans/flows/', import.meta.url)

export function loadFlows(): Flows {
  return JSON.parse(readFileSync(FLOWS_FILE, 'utf8')) as Flows
}

const norm = (v: unknown) => String(v ?? '').trim().replace(/\s+/g, ' ').toLowerCase()

/** What differs between what was expected and what the page holds; empty when it all matches. */
export function differences(expect: unknown, got: unknown, at = ''): string[] {
  if (expect === 'absent-or-false') return got === undefined || got === null || got === false ? [] : [`${at} should not have happened (${JSON.stringify(got)})`]
  if (expect === 'nonempty') return norm(got) ? [] : [`${at} is empty`]
  if (Array.isArray(expect)) {
    if (!Array.isArray(got)) return [`${at} is missing`]
    if (got.length !== expect.length) return [`${at} has ${got.length} entries, not ${expect.length}: ${JSON.stringify(got).slice(0, 160)}`]
    return expect.flatMap((e, i) => differences(e, got[i], `${at}[${i}]`))
  }
  if (expect && typeof expect === 'object') {
    if (!got || typeof got !== 'object') return [`${at || 'state'} is missing`]
    return Object.entries(expect).flatMap(([k, v]) => differences(v, (got as Record<string, unknown>)[k], at ? `${at}.${k}` : k))
  }
  return norm(expect) === norm(got) ? [] : [`${at} is “${got ?? ''}”, not “${expect}”`]
}

export function flowPlanFile(t: FlowTask, backend: Backend, model: string): URL {
  return new URL(`${t.key}.${backend}-${model.replace(/[^\w.-]+/g, '_')}.json`, FLOW_PLANS_DIR)
}

/** A ready-made process for a flow task: the task's words where it says {goal}, the person's and the task's values. */
export function libraryFlowProcess(t: FlowTask, person: Value[]): Process {
  if (!t.process) throw new Error(`${t.key} has no ready-made process`)
  const p = normalize(structuredClone(loadProcesses()[t.process]))
  p.task = t.task
  p.values = [...person, ...t.values]
  return p
}

export async function resetAndOpen(tab: TabSurface, origin: string, t: FlowTask): Promise<void> {
  await tab.open(origin + t.start)
  await tab.cdp.evaluate(`localStorage.removeItem(${JSON.stringify(t.storage)})`)
  await tab.open(origin + t.start)
}

/** Claude sets a flow up from its start page and what Atelier knows about the person; saved under lab/plans/flows/. */
export async function planFlow(tab: TabSurface, t: FlowTask, person: Value[], opts: { backend?: Backend; model?: string } = {}): Promise<PlanResult> {
  const snap = await tab.snapshot()
  const r = await plan(t.task, { page: snap.page, person }, opts)
  mkdirSync(FLOW_PLANS_DIR, { recursive: true })
  writeFileSync(flowPlanFile(t, r.backend, r.model), `${JSON.stringify({ task: t.key, made: new Date().toISOString(), ...r }, null, 2)}\n`)
  return r
}

export function savedFlowPlan(t: FlowTask, backend: Backend = 'claude', model = 'claude-opus-5-5'): Process | null {
  const u = flowPlanFile(t, backend, model)
  if (!existsSync(u)) return null
  const saved = JSON.parse(readFileSync(u, 'utf8'))
  return saved.process ? normalize(saved.process) : null
}

export interface FlowResult {
  key: string
  ok: boolean
  failures: string[]
  result: ItemResult
  /** Decisions in the flow by who made them: the decision model (nano, mini), code, words (the guard's list), claude. */
  byWho: Record<string, number>
  ms: number
}

export async function runFlow(tab: TabSurface, origin: string, t: FlowTask, p: Process, decide: Decide, opts: RunOptions = {}): Promise<FlowResult> {
  const found = problems(p)
  if (found.length) throw new Error(`${p.id}: ${found.join('; ')}`)
  await resetAndOpen(tab, origin, t)
  const t0 = performance.now()
  const { results } = await runProcess(p, tab, decide, { act: true, maxHops: 40, ...opts })
  const ms = performance.now() - t0
  const result = results[0]!
  const got = await tab.cdp.evaluate<{ state: unknown; path: string }>(`({ state: JSON.parse(localStorage.getItem(${JSON.stringify(t.storage)}) || 'null'), path: location.pathname })`)
  const failures = [...differences(t.expect.state, got.state), ...(t.expect.path && t.expect.path !== got.path ? [`ended on ${got.path}, not ${t.expect.path}`] : [])]
  const byWho: Record<string, number> = {}
  for (const s of result.steps) byWho[s.by] = (byWho[s.by] ?? 0) + 1
  return { key: t.key, ok: failures.length === 0, failures, result, byWho, ms }
}
