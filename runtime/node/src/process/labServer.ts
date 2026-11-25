/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// The lab's routes on the decision server: the /lab page, the mock sites a browser can open, and runs streamed as
// one JSON object per line (the process, each item as it finishes, the summary).

import { readFileSync } from 'node:fs'
import type { IncomingMessage, ServerResponse } from 'node:http'
import { tabSurface, tabTrial, targets } from './cdp.ts'
import type { Decide } from './decide.ts'
import { loadFixtures, loadProcesses, renderSite } from './fixtures.ts'
import { findTask, planTask, processFor, runTask, savedPlan, type Source } from './lab.ts'
import { plan, type Backend } from './planner.ts'
import { runProcess } from './run.ts'
import { normalize, problems, type Process } from './schema.ts'

async function body(req: IncomingMessage): Promise<any> {
  const chunks: Buffer[] = []
  for await (const chunk of req) chunks.push(chunk as Buffer)
  return chunks.length ? JSON.parse(Buffer.concat(chunks).toString('utf8')) : {}
}

function stream(res: ServerResponse): { write: (o: unknown) => void; end: () => void; signal: AbortSignal } {
  const controller = new AbortController()
  res.on('close', () => controller.abort())
  res.writeHead(200, { 'content-type': 'application/x-ndjson; charset=utf-8', 'cache-control': 'no-store' })
  return { write: (o) => res.write(`${JSON.stringify(o)}\n`), end: () => res.end(), signal: controller.signal }
}

/** Handles a lab request; false when the URL isn't the lab's. */
export async function labRoute(req: IncomingMessage, res: ServerResponse, decide: Decide, model: string): Promise<boolean> {
  const url = new URL(req.url ?? '/', 'http://127.0.0.1')
  const send = (status: number, o: unknown) => {
    res.writeHead(status, { 'content-type': 'application/json' })
    res.end(JSON.stringify(o))
  }
  if (req.method === 'GET' && url.pathname === '/lab') {
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
    res.end(readFileSync(new URL('../../lab.html', import.meta.url)))
    return true
  }
  const app = /^\/lab\/apps\/([\w-]+)(\/.*)?$/.exec(url.pathname)
  if (req.method === 'GET' && app) {
    // A whole small site in one page (lab/apps/<name>.html) that routes by its own path.
    const file = new URL(`../../lab/apps/${app[1]}.html`, import.meta.url)
    let html: Buffer | null = null
    try {
      html = readFileSync(file)
    } catch {}
    res.writeHead(html ? 200 : 404, { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' })
    res.end(html ?? 'Not found')
    return true
  }
  const site = /^\/lab\/sites\/([\w-]+)(\/.*)?$/.exec(url.pathname)
  if (req.method === 'GET' && site) {
    const f = loadFixtures().find((x) => x.id === site[1])
    const html = f ? renderSite(f, site[2] && site[2] !== '/' ? site[2] : '') : null
    res.writeHead(html ? 200 : 404, { 'content-type': 'text/html; charset=utf-8' })
    res.end(html ?? 'Not found')
    return true
  }
  if (!url.pathname.startsWith('/v1/lab')) return false
  try {
    if (req.method === 'GET' && url.pathname === '/v1/lab') {
      const fixtures = loadFixtures().map((f) => ({
        id: f.id,
        kind: f.kind,
        site: f.site,
        title: f.title ?? f.site,
        count: f.kind === 'site' ? Object.keys(f.pages ?? {}).length : (f.items ?? []).length,
        tasks: f.tasks.map((t) => {
          const p = savedPlan(f, t)
          return { ...t, planned: p ? { backend: p.backend, model: p.model, made: p.made, ms: p.ms, tokens: p.tokens, ok: Boolean(p.process) } : null }
        }),
      }))
      send(200, { model, fixtures, processes: loadProcesses() })
      return true
    }
    if (req.method === 'GET' && url.pathname === '/v1/lab/tabs') {
      const list = await targets(Number(url.searchParams.get('cdp') ?? 9555))
      send(200, list.filter((t) => t.type === 'page' && /^https?:/.test(t.url)).map((t) => ({ url: t.url, title: t.title })))
      return true
    }
    if (req.method === 'POST' && url.pathname === '/v1/lab/plan') {
      const b = await body(req)
      const { fixture, task } = findTask(loadFixtures(), b.ref)
      send(200, await planTask(fixture, task, { backend: b.backend as Backend | undefined, model: b.model, decide }))
      return true
    }
    if (req.method === 'POST' && url.pathname === '/v1/lab/run') {
      const b = await body(req)
      const { fixture, task } = findTask(loadFixtures(), b.ref)
      const p: Process = b.source === 'custom' ? normalize(b.process) : processFor(fixture, task, b.source as Source)
      const found = problems(p)
      if (found.length) return (send(422, { error: found.join('; ') }), true)
      const out = stream(res)
      out.write({ type: 'process', process: p, source: b.source })
      const { summary } = await runTask(fixture, task, p, decide, { signal: out.signal, onItem: (r) => out.write({ type: 'item', result: r }) })
      out.write({ type: 'summary', summary })
      out.end()
      return true
    }
    if (req.method === 'POST' && url.pathname === '/v1/lab/tab') {
      const b = await body(req)
      const tab = await tabSurface(Number(b.cdp ?? 9555), b.match ?? '')
      const out = stream(res)
      try {
        let p: Process
        if (b.process) {
          p = normalize(typeof b.process === 'string' ? loadProcesses()[b.process] : b.process)
          if (b.goal) p.task = b.goal
        } else {
          out.write({ type: 'status', text: 'Claude is setting the process up…' })
          const r = await plan(b.task, await tab.snapshot(), { buckets: b.buckets, backend: b.backend, model: b.model, trial: tabTrial(tab) })
          out.write({ type: 'planned', backend: r.backend, model: r.model, ms: r.ms, tokens: r.tokens, rounds: r.rounds, problems: r.problems })
          if (!r.process) throw new Error(`the plan can't run: ${r.problems.join('; ')}`)
          p = r.process
        }
        const found = problems(p)
        if (found.length) throw new Error(found.join('; '))
        out.write({ type: 'process', process: p, source: b.process ? 'library' : 'planned' })
        const { summary } = await runProcess(p, tab, decide, { act: b.act === true, signal: out.signal, onItem: (r) => out.write({ type: 'item', result: r }) })
        out.write({ type: 'summary', summary })
      } catch (err) {
        out.write({ type: 'error', message: err instanceof Error ? err.message : String(err) })
      } finally {
        tab.cdp.close()
        out.end()
      }
      return true
    }
    send(404, { error: 'not found' })
  } catch (err) {
    if (res.headersSent) res.end(`${JSON.stringify({ type: 'error', message: err instanceof Error ? err.message : String(err) })}\n`)
    else send(422, { error: err instanceof Error ? err.message : String(err) })
  }
  return true
}
