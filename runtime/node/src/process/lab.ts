/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// The lab: fixture tasks run with a ready-made process (processes/) or one a language model planned (lab/plans/),
// measured against what a person expects. Shared by the command line (src/lab.ts) and the server's /lab page.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import type { Decide } from './decide.ts'
import { libraryProcess, loadProcesses, surfacesFor, type Fixture, type FixtureTask } from './fixtures.ts'
import { plan, type Backend, type PlanResult, type Snapshot, type Trial } from './planner.ts'
import { runProcess, summarize, type ItemResult, type RunOptions, type RunSummary } from './run.ts'
import { normalize, problems, type Process } from './schema.ts'

export const PLANS_DIR = new URL('../../lab/plans/', import.meta.url)

export type Source = 'library' | 'planned'

export function findTask(fixtures: Fixture[], ref: string): { fixture: Fixture; task: FixtureTask } {
  const [fid, key] = ref.split(':')
  const fixture = fixtures.find((f) => f.id === fid)
  if (!fixture) throw new Error(`no fixture "${fid}" (have ${fixtures.map((f) => f.id).join(', ')})`)
  const task = key ? fixture.tasks.find((t) => t.key === key) : fixture.tasks[0]
  if (!task) throw new Error(`no task "${key}" in ${fid} (have ${fixture.tasks.map((t) => t.key).join(', ')})`)
  return { fixture, task }
}

/** What the planner sees for a fixture task: the page, and the items already read (never their expected answers). */
export async function snapshotOf(fixture: Fixture, task: FixtureTask): Promise<Snapshot> {
  const surfaces = surfacesFor(fixture, task)
  // A pages fixture: show a page that has something to decide on (the first one with controls).
  let page = await surfaces[0]!.read()
  for (const s of surfaces) {
    const p = await s.read()
    if (p.elements.length >= 2) {
      page = p
      break
    }
  }
  if (fixture.kind === 'site')
    return { page, explored: Object.entries(fixture.pages ?? {}).map(([path, p]) => ({ path, title: p.title, text: p.text.replace(/\[link:[^\]]+\]/g, '[link]') })) }
  if (fixture.kind !== 'list') return { page }
  const items = await surfaces[0]!.items(normalize({ id: 'x', over: 'items', steps: [] }))
  return { page, items: { count: items.length, fields: fixture.fields ?? Object.keys(items[0]?.fields ?? {}), samples: items.slice(0, 3).map((i) => i.fields) } }
}

export function planPath(fixture: Fixture, task: FixtureTask, backend: Backend, model: string): URL {
  return new URL(`${fixture.id}-${task.key}.${backend}-${model.replace(/[^\w.-]+/g, '_')}.json`, PLANS_DIR)
}

export interface SavedPlan extends PlanResult {
  fixture: string
  task: string
  made: string
}

/**
 * The planner's own try-out: the process runs once on what the planner saw (a site it explored, the first page of
 * a pages fixture, the first three items of a list), never against the expected answers. It reports dead ends a
 * person would spot at a glance: a click that found nothing, a code pick that matched several rows, an error, items
 * that read empty. Being unsure is fine; a decision the model is not sure of is not a mistake in the process.
 */
export function trialOn(fixture: Fixture, task: FixtureTask, decide: Decide): Trial {
  return async (p) => {
    const out: string[] = []
    const surfaces = surfacesFor(fixture, task).slice(0, 1)
    for (const surface of surfaces) {
      const items = (await surface.items(p)).slice(0, 3)
      const { results } = await runProcess(p, { ...surface, items: async () => items }, decide, { act: true })
      for (const r of results) {
        const last = r.steps.at(-1)
        const where = last ? ` at step "${last.step}"` : ''
        if (r.outcome.kind === 'error') out.push(`it stopped with an error${where}: ${r.outcome.message}`)
        else if (r.outcome.kind === 'person' && /nothing to click|still going after/.test(r.outcome.reason)) out.push(`${r.outcome.reason}${where}`)
        else if (r.outcome.kind === 'person' && last?.kind === 'pick' && last.by === 'code') out.push(`the pick by code${where} matched ${Object.keys(last.probabilities ?? {}).length} controls, not exactly one: ${Object.keys(last.probabilities ?? {}).join(' / ') || 'none'}`)
      }
    }
    return [...new Set(out)]
  }
}

export async function planTask(fixture: Fixture, task: FixtureTask, opts: { backend?: Backend; model?: string; decide?: Decide } = {}): Promise<SavedPlan> {
  const r = await plan(task.task, await snapshotOf(fixture, task), { buckets: task.buckets, trial: opts.decide ? trialOn(fixture, task, opts.decide) : undefined, ...opts })
  const saved: SavedPlan = { fixture: fixture.id, task: task.key, made: new Date().toISOString(), ...r }
  mkdirSync(PLANS_DIR, { recursive: true })
  writeFileSync(planPath(fixture, task, r.backend, r.model), `${JSON.stringify(saved, null, 2)}\n`)
  return saved
}

/** The newest saved plan for a fixture task, whichever planner made it (or the one named). */
export function savedPlan(fixture: Fixture, task: FixtureTask, backend?: Backend, model?: string): SavedPlan | null {
  if (backend && model) {
    const u = planPath(fixture, task, backend, model)
    return existsSync(u) ? (JSON.parse(readFileSync(u, 'utf8')) as SavedPlan) : null
  }
  for (const b of ['claude', 'api'] as Backend[]) {
    for (const m of [model ?? 'claude-opus-5-5', 'opus', 'sonnet']) {
      const u = planPath(fixture, task, b, m)
      if (existsSync(u)) return JSON.parse(readFileSync(u, 'utf8')) as SavedPlan
    }
  }
  return null
}

export function processFor(fixture: Fixture, task: FixtureTask, source: Source): Process {
  if (source === 'planned') {
    const p = savedPlan(fixture, task)
    if (!p?.process) throw new Error(`no plan for ${fixture.id}:${task.key} yet (run: lab plan ${fixture.id}:${task.key})`)
    return normalize(p.process)
  }
  if (!task.process) throw new Error('no ready-made process (this task is only set up by a planner)')
  const raw = loadProcesses()[task.process]
  if (!raw) throw new Error(`no process "${task.process}" in processes/`)
  const p = normalize(libraryProcess(raw, task))
  const found = problems(p)
  if (found.length) throw new Error(`${task.process}: ${found.join('; ')}`)
  return p
}

/** One fixture task with one process; a pages fixture runs once per page. Clicks happen only on the fixture. */
export async function runTask(fixture: Fixture, task: FixtureTask, process: Process, decide: Decide, opts: RunOptions = {}): Promise<{ results: ItemResult[]; summary: RunSummary }> {
  const t0 = performance.now()
  const results: ItemResult[] = []
  for (const surface of surfacesFor(fixture, task)) {
    if (opts.signal?.aborted) break
    const r = await runProcess(process, surface, decide, { act: true, ...opts })
    results.push(...r.results)
  }
  return { results, summary: summarize(process.id, results, performance.now() - t0) }
}
