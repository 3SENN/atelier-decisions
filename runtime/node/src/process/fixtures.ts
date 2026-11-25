/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// The lab's mock pages (lab/fixtures/*.json): a list page (an inbox, shop results, job posts), a set of single
// pages (cookie banners), or a small site to find your way through. Each one is a surface the runner can work on
// without a browser, and the same fixture renders as HTML (renderSite) so a real browser sees the same page.

import { readdirSync, readFileSync } from 'node:fs'
import type { Element, Item, Page, Surface } from './run.ts'
import type { Process } from './schema.ts'

export interface FixtureTask {
  key: string
  task: string
  buckets: string[]
  process: string
  start?: string
  expect?: string
}

export interface Fixture {
  id: string
  kind: 'list' | 'pages' | 'site'
  site: string
  url?: string
  title?: string
  intro?: string
  chrome?: string[]
  fields?: string[]
  items?: Array<Record<string, unknown> & { id: string; expect?: Record<string, string> }>
  pages?: Record<string, { title: string; text: string }>
  tasks: FixtureTask[]
  notes?: string[]
}

export const FIXTURES_DIR = new URL('../../lab/fixtures/', import.meta.url)
export const PROCESSES_DIR = new URL('../../processes/', import.meta.url)

export function loadFixtures(): Fixture[] {
  return readdirSync(FIXTURES_DIR)
    .filter((f) => f.endsWith('.json'))
    .sort()
    .map((f) => JSON.parse(readFileSync(new URL(f, FIXTURES_DIR), 'utf8')) as Fixture)
}

export function loadProcesses(): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const f of readdirSync(PROCESSES_DIR).filter((f) => f.endsWith('.json')).sort()) out[f.replace(/\.json$/, '')] = JSON.parse(readFileSync(new URL(f, PROCESSES_DIR), 'utf8'))
  return out
}

const MARKER = /\[(link|button|field)(?::([^\]]+))?\]\s*([^[\n]*)/g

/** Page text with [link:/path] markers → the text the model reads ([link] Label) and the controls on it. */
export function parseMarked(text: string, base: string): { text: string; elements: Element[] } {
  const elements: Element[] = []
  const out: string[] = []
  let previous = ''
  for (const line of text.split('\n')) {
    const bare = line.replace(MARKER, '').replace(/[·|\s]+/g, ' ').trim()
    out.push(
      line.replace(MARKER, (_m, role: Element['role'], href: string | undefined, label: string) => {
        const name = label.trim()
        // The row it sits in: the rest of its line, or the line above when it stands alone ("#1043 · 14 Oct …" / [link] View order).
        const context = bare || previous
        elements.push({ ref: `e${elements.length + 1}`, role, text: name, ...(href ? { href: new URL(href, base).toString() } : {}), ...(context ? { context: context.slice(0, 200) } : {}) })
        return `[${role}] ${name}${label.endsWith(' ') ? ' ' : ''}`
      }),
    )
    if (line.trim()) previous = bare
  }
  return { text: out.join('\n'), elements }
}

function listPage(f: Fixture): Page {
  const url = f.url ?? `https://${f.id}.test/`
  const chrome = (f.chrome ?? []).join('\n')
  const lines = (f.items ?? []).map((it) => (f.fields ?? []).map((k) => String(it[k] ?? '')).join(' · '))
  const { text, elements } = parseMarked([chrome, f.intro ?? '', ...lines].filter(Boolean).join('\n'), url)
  return { url, title: f.title ?? f.site, text, elements }
}

function itemOf(f: Fixture, raw: NonNullable<Fixture['items']>[number], task: string | null): Item {
  const fields: Record<string, string> = {}
  for (const [k, v] of Object.entries(raw)) if (k !== 'id' && k !== 'expect' && (typeof v === 'string' || typeof v === 'number')) fields[k] = String(v)
  return { id: raw.id, fields, ...(task && raw.expect?.[task] !== undefined ? { expect: raw.expect[task] } : {}) }
}

/** A list fixture: the page and its items, with what a person expects for the task. */
export function listSurface(f: Fixture, task: string | null): Surface {
  const page = listPage(f)
  return {
    read: async () => page,
    items: async () => (f.items ?? []).map((it) => itemOf(f, it, task)),
    click: async () => {},
    open: async () => {},
  }
}

/** One page of a pages fixture as its own surface: the page is the item. */
export function pageSurface(f: Fixture, raw: NonNullable<Fixture['items']>[number], task: string | null): Surface {
  const url = String(raw.url ?? `https://${f.id}.test/${raw.id}`)
  const { text, elements } = parseMarked(String(raw.text ?? ''), url)
  const page: Page = { url, title: String(raw.title ?? f.site), text, elements }
  const item: Item = { id: raw.id, fields: { title: page.title, url }, ...(task && raw.expect?.[task] !== undefined ? { expect: raw.expect[task] } : {}) }
  return { read: async () => page, items: async () => [item], click: async () => {}, open: async () => {} }
}

/** A site fixture: starts at a path; clicking a link goes to its page. */
export function siteSurface(f: Fixture, t: FixtureTask): Surface {
  const origin = f.url ? new URL(f.url).origin : `https://${f.id}.test`
  let path = t.start ?? '/'
  // Toggles flip when clicked ("[field] Marketing emails: on" → off), so a process can check its own work.
  const edited = new Map<string, string>()
  const read = async (): Promise<Page> => {
    const p = f.pages?.[path]
    const url = origin + path
    if (!p) return { url, title: 'Not found', text: 'This page does not exist.', elements: [] }
    const { text, elements } = parseMarked(edited.get(path) ?? p.text, url)
    return { url, title: p.title, text, elements }
  }
  const FLIP: Record<string, string> = { on: 'off', off: 'on', aan: 'uit', uit: 'aan', an: 'aus', aus: 'an' }
  return {
    read,
    items: async () => [{ id: t.key, fields: { goal: t.task }, ...(t.expect ? { expect: t.expect } : {}) }],
    click: async (e) => {
      if (e.href) path = new URL(e.href).pathname
      else if (e.role === 'button' && /\b(save|opslaan|speichern)\b/i.test(e.text)) {
        // A save button confirms, as sites do.
        const now = edited.get(path) ?? f.pages?.[path]?.text ?? ''
        if (!now.startsWith('Saved: ')) edited.set(path, `Saved: your changes are stored.\n${now}`)
      } else if (e.role === 'field') {
        const m = /^(.*?:\s*)(on|off|aan|uit|an|aus)$/i.exec(e.text)
        if (m) {
          const now = edited.get(path) ?? f.pages?.[path]?.text ?? ''
          edited.set(path, now.replace(`[field] ${e.text}`, `[field] ${m[1]}${FLIP[m[2]!.toLowerCase()]}`))
        }
      }
    },
    open: async (u) => {
      path = new URL(u, origin).pathname
    },
  }
}

/** Every run a fixture task makes: one surface for a list or a site, one per page for a pages fixture. */
export function surfacesFor(f: Fixture, t: FixtureTask): Surface[] {
  if (f.kind === 'list') return [listSurface(f, t.key)]
  if (f.kind === 'site') return [siteSurface(f, t)]
  return (f.items ?? []).map((it) => pageSurface(f, it, t.key))
}

/** A library process for a fixture task, with the task's own words where it says {goal}. */
export function libraryProcess(raw: unknown, t: FixtureTask): Process {
  const p = structuredClone(raw) as Process
  if (p.task.includes('{goal}')) p.task = t.task
  return p
}

// ── HTML, for a real browser ──

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!)

function markedHtml(text: string, linkBase: string): string {
  return text
    .split('\n')
    .map((line) => {
      let html = ''
      let last = 0
      for (const m of line.matchAll(MARKER)) {
        html += esc(line.slice(last, m.index))
        const [, role, href, label] = m as unknown as [string, string, string | undefined, string]
        const name = esc(label.trim())
        if (role === 'button') html += `<button type="button">${name}</button> `
        else if (role === 'field') html += `<label class="field">${name} <input type="checkbox" checked></label> `
        else html += `<a href="${href ? esc(href.startsWith('/') ? linkBase + href : href) : '#'}">${name}</a> `
        last = (m.index ?? 0) + m[0].length
      }
      html += esc(line.slice(last))
      return html.trim() ? `<p>${html}</p>` : ''
    })
    .join('\n')
}

const CSS = `body{font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#fafaf8;color:#1d1d1b}
header{display:flex;gap:10px;flex-wrap:wrap;align-items:center;padding:12px 20px;border-bottom:1px solid #e4e3df;background:#fff}
header .site{font-weight:600;margin-right:12px}main{max-width:860px;margin:0 auto;padding:16px 20px}
article.item{background:#fff;border:1px solid #e4e3df;border-radius:10px;padding:10px 14px;margin:8px 0}
article.item>div{margin:2px 0}.f-subject,.f-name,.f-title{font-weight:600}.f-date,.f-seller,.f-company{color:#6b6b66;font-size:13px}
button{font:inherit;padding:5px 12px;border-radius:8px;border:1px solid #c9c8c2;background:#fff;cursor:pointer;margin:2px}
a{color:#3d5afe}p{margin:6px 0}.field{display:inline-flex;gap:6px;align-items:center}`

function shell(title: string, body: string): string {
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${esc(title)}</title><style>${CSS}</style></head><body>${body}</body></html>`
}

/** The HTML for /lab/sites/<fixture>[/<path>]; null when there is no such page. */
export function renderSite(f: Fixture, rest: string): string | null {
  const base = `/lab/sites/${f.id}`
  if (f.kind === 'list') {
    const chrome = markedHtml((f.chrome ?? []).join(' '), base)
    const items = (f.items ?? [])
      .map((it) => `<article class="item" data-id="${esc(it.id)}">${(f.fields ?? []).map((k) => `<div class="f-${esc(k)}">${esc(String(it[k] ?? ''))}</div>`).join('')}</article>`)
      .join('\n')
    return shell(f.title ?? f.site, `<header><span class="site">${esc(f.site)}</span>${chrome.replace(/<\/?p>/g, '')}</header><main>${f.intro ? `<p>${esc(f.intro)}</p>` : ''}<section id="items">${items}</section></main>`)
  }
  if (f.kind === 'pages') {
    if (!rest) return shell(f.site, `<main><h1>${esc(f.site)}</h1>${(f.items ?? []).map((it) => `<p><a href="${base}/${esc(it.id)}">${esc(String(it.title ?? it.id))}</a></p>`).join('')}</main>`)
    const it = (f.items ?? []).find((i) => i.id === rest.replace(/^\//, ''))
    return it ? shell(String(it.title ?? f.site), `<main>${markedHtml(String(it.text ?? ''), base)}</main>`) : null
  }
  const page = f.pages?.[rest || '/']
  return page ? shell(page.title, `<main>${markedHtml(page.text, base)}</main>`) : null
}
