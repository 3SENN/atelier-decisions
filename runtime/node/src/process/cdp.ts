/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// A real browser tab as a surface, over the Chrome DevTools Protocol: Atelier started with
// --remote-debugging-port=<port> (or any Chromium). It reads the page the way the decision model was trained to
// read pages (text with [link] and [button] markers), reads a process's items with its selectors, and clicks only
// when the run is told to act. Never point it at a profile with real accounts signed in while acting.

import type { Element, Item, Page, Surface } from './run.ts'
import type { Process } from './schema.ts'

interface Target {
  id: string
  type: string
  url: string
  title: string
  webSocketDebuggerUrl?: string
}

export async function targets(port: number): Promise<Target[]> {
  const res = await fetch(`http://127.0.0.1:${port}/json/list`)
  return (await res.json()) as Target[]
}

export class Cdp {
  private ws: WebSocket
  private next = 1
  private pending = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void }>()

  private constructor(ws: WebSocket) {
    this.ws = ws
    ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(String(ev.data))
      const p = msg.id ? this.pending.get(msg.id) : undefined
      if (!p) return
      this.pending.delete(msg.id)
      if (msg.error) p.reject(new Error(msg.error.message))
      else p.resolve(msg.result)
    })
  }

  static async connect(url: string): Promise<Cdp> {
    const ws = new WebSocket(url)
    await new Promise<void>((resolve, reject) => {
      ws.addEventListener('open', () => resolve(), { once: true })
      ws.addEventListener('error', () => reject(new Error(`could not connect to ${url}`)), { once: true })
    })
    return new Cdp(ws)
  }

  send<T = any>(method: string, params: object = {}): Promise<T> {
    const id = this.next++
    this.ws.send(JSON.stringify({ id, method, params }))
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      setTimeout(() => {
        if (this.pending.delete(id)) reject(new Error(`${method} timed out`))
      }, 20000)
    })
  }

  async evaluate<T>(expression: string): Promise<T> {
    const r = await this.send<{ result: { value: T }; exceptionDetails?: { text: string; exception?: { description?: string } } }>('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text)
    return r.result.value
  }

  close(): void {
    this.ws.close()
  }
}

/** Runs in the page: its visible text in reading order, controls marked and given a data-atelier-ref. */
const READ = `(() => {
  const BLOCK = /^(P|DIV|LI|UL|OL|H[1-6]|ARTICLE|SECTION|HEADER|FOOTER|NAV|MAIN|ASIDE|TR|TABLE|FORM|DIALOG|BR|DD|DT|FIGCAPTION|BLOCKQUOTE)$/;
  const out = []; const elements = []; let n = 0; let size = 0;
  const visible = (el) => { const s = getComputedStyle(el); if (s.display === 'none' || s.visibility === 'hidden' || el.getAttribute('aria-hidden') === 'true') return false; const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  const roleOf = (el) => { const t = el.tagName; const role = el.getAttribute('role');
    if (t === 'A' && el.hasAttribute('href')) return 'link';
    if (t === 'BUTTON' || role === 'button' || t === 'SUMMARY' || (t === 'INPUT' && /^(submit|button|reset)$/.test(el.type))) return 'button';
    if (t === 'INPUT' || t === 'SELECT' || t === 'TEXTAREA' || role === 'switch' || role === 'checkbox') return 'field';
    if (role === 'link') return 'link'; return null; };
  const clean = (t) => (t || '').replace(/\\s+/g, ' ').trim();
  // The row a control sits in: the nearest ancestor with more text than the control (when short), or else the
  // block just before the control's own block ("#1043 · placed 14 Oct" / [link] View order).
  const around = (el, own) => { let block = el;
    for (let a = el.parentElement; a && a !== document.body; a = a.parentElement) {
      const t = clean(a.innerText); if (t.length > own.length + 8) { if (t.length <= 240) return t.replace(own, '').trim(); break; } block = a; }
    const prev = block.previousElementSibling; const t = prev ? clean(prev.innerText) : ''; return t && t.length <= 240 ? t : null; };
  // A field: its label (never its value), what is in it, its options, and the hints autofill reads.
  const field = (el) => { const t = el.tagName; const type = (el.getAttribute('type') || '').toLowerCase();
    if (t === 'INPUT' && /^(hidden|submit|button|reset|image|file)$/.test(type)) return null;
    const lab = (el.labels && el.labels[0] && clean(el.labels[0].innerText)) || el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.getAttribute('title') || el.getAttribute('name') || '';
    const text = clean(lab).slice(0, 120); if (!text) return null;
    const input = t === 'SELECT' ? 'select' : t === 'TEXTAREA' ? 'textarea' : type === 'checkbox' || el.getAttribute('role') === 'switch' ? 'checkbox' : type === 'radio' ? 'radio' : 'text';
    const hint = el.getAttribute('autocomplete') && el.getAttribute('autocomplete') !== 'off' ? el.getAttribute('autocomplete') : /^(email|tel|url|number|date|password|search)$/.test(type) ? type : undefined;
    const value = t === 'SELECT' ? (el.selectedIndex > 0 ? clean(el.options[el.selectedIndex].text) : '') : input === 'text' || input === 'textarea' ? String(el.value || '').slice(0, 200) : undefined;
    return { text, input, ...(value !== undefined ? { value } : {}), ...(input === 'checkbox' || input === 'radio' ? { checked: Boolean(el.checked || el.getAttribute('aria-checked') === 'true') } : {}),
      ...(t === 'SELECT' ? { options: [...el.options].map((o) => clean(o.text)).filter(Boolean).slice(0, 40) } : {}), ...(hint ? { hint } : {}) }; };
  const name = (el) => (el.getAttribute('aria-label') || el.innerText || el.value || el.getAttribute('title') || (el.labels && el.labels[0] && el.labels[0].innerText) || el.getAttribute('placeholder') || '').replace(/\\s+/g, ' ').trim().slice(0, 120);
  const walk = (node) => {
    if (size > 9000) return;
    if (node.nodeType === 3) { const t = node.textContent.replace(/\\s+/g, ' '); if (t.trim()) { out.push(t); size += t.length; } return; }
    if (node.nodeType !== 1) return;
    const el = node; if (/^(SCRIPT|STYLE|NOSCRIPT|TEMPLATE|SVG|HEAD)$/.test(el.tagName)) return;
    if (!visible(el) && el.tagName !== 'BR') return;
    const role = roleOf(el);
    if (role === 'field') { const f = field(el); if (f) { const ref = 'e' + (++n); el.setAttribute('data-atelier-ref', ref);
        const context = around(el, f.text);
        elements.push({ ref, role, ...f, ...(context ? { context } : {}) });
        const shown = f.input === 'checkbox' || f.input === 'radio' ? (f.checked ? 'checked' : 'not checked') : (f.value || 'empty');
        out.push(' [field] ' + f.text + ': ' + shown + ' '); size += f.text.length + 20; }
      return; }
    if (role) { const text = name(el); if (text) { const ref = 'e' + (++n); el.setAttribute('data-atelier-ref', ref);
        const context = around(el, text);
        elements.push({ ref, role, text, ...(role === 'link' ? { href: el.href } : {}), ...(context ? { context } : {}) }); out.push(' [' + role + '] ' + text + ' '); size += text.length + 10; }
      return; }
    const block = BLOCK.test(el.tagName); if (block) out.push('\\n');
    for (const c of el.childNodes) walk(c);
    if (block) out.push('\\n');
  };
  walk(document.body);
  const text = out.join('').replace(/[ \\t]+/g, ' ').replace(/ *\\n[ \\n]*/g, '\\n').trim().slice(0, 9000);
  return { url: location.href, title: document.title, text, elements: elements.slice(0, 200) };
})()`

/** Runs in the page: groups of three or more alike siblings with text, as selectors a planner can use for items. */
export const CANDIDATES = `(() => {
  const sel = (el) => { const parts = []; for (let e = el; e && e !== document.body && parts.length < 4; e = e.parentElement) {
      let p = e.tagName.toLowerCase(); if (e.id && /^[a-z][\\w-]*$/i.test(e.id)) { parts.unshift(p + '#' + e.id); break; }
      const cls = [...e.classList].filter((c) => /^[a-z][\\w-]*$/i.test(c) && c.length < 30).slice(0, 2); if (cls.length) p += '.' + cls.join('.');
      parts.unshift(p); } return parts.join(' > '); };
  const groups = new Map();
  for (const el of document.querySelectorAll('body *')) {
    const kids = [...el.children].filter((k) => (k.innerText || '').trim().length > 15);
    if (kids.length < 3) continue;
    const by = new Map(); for (const k of kids) { const key = k.tagName + '.' + [...k.classList].sort().join('.'); by.set(key, [...(by.get(key) || []), k]); }
    for (const [, list] of by) if (list.length >= 3) { const s = sel(list[0]); if (!groups.has(s)) groups.set(s, list); }
  }
  const strip = (el) => { const c = el.cloneNode(true); for (const e of c.querySelectorAll('*')) { for (const a of [...e.attributes]) if (!/^(class|href)$/.test(a.name)) e.removeAttribute(a.name); }
    for (const a of [...c.attributes]) if (a.name !== 'class') c.removeAttribute(a.name); return c.outerHTML.replace(/\\s+/g, ' ').slice(0, 700); };
  return [...groups].map(([selector, list]) => ({ selector, count: document.querySelectorAll(selector).length, samples: list.slice(0, 2).map(strip) }))
    .filter((g) => g.count >= 3).sort((a, b) => b.count - a.count).slice(0, 6);
})()`

function itemsScript(p: Process): string {
  return `(() => [...document.querySelectorAll(${JSON.stringify(p.items?.selector ?? 'body')})].slice(0, 100).map((el, i) => {
    const fields = {}; for (const f of ${JSON.stringify(p.items?.fields ?? [])}) { const x = f.selector === ':self' ? el : el.querySelector(f.selector);
      const v = x ? (x.innerText || x.getAttribute('href') || '').replace(/\\s+/g, ' ').trim() : ''; if (v) fields[f.name] = v.slice(0, 600); }
    if (Object.keys(fields).length === 0) fields.text = (el.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 600);
    return { id: el.getAttribute('data-id') || String(i + 1), fields };
  }))()`
}

export interface TabSurface extends Surface {
  cdp: Cdp
  snapshot(): Promise<{ page: Page; candidates: Array<{ selector: string; count: number; samples: string[] }> }>
}

async function settle(cdp: Cdp): Promise<void> {
  await new Promise((r) => setTimeout(r, 400))
  for (let i = 0; i < 40; i++) {
    const state = await cdp.evaluate<string>('document.readyState').catch(() => 'loading')
    if (state === 'complete') return
    await new Promise((r) => setTimeout(r, 250))
  }
}

/** The first page target whose URL contains `match` (or any web page), as a surface. */
export async function tabSurface(port: number, match = ''): Promise<TabSurface> {
  const list = (await targets(port)).filter((t) => t.type === 'page' && /^https?:/.test(t.url) && t.webSocketDebuggerUrl)
  const t = list.find((x) => x.url.includes(match)) ?? (match ? undefined : list[0])
  if (!t) throw new Error(`no tab ${match ? `with "${match}" in its address ` : ''}on port ${port} (open: ${list.map((x) => x.url).join(', ') || 'none'})`)
  const cdp = await Cdp.connect(t.webSocketDebuggerUrl!)
  await cdp.send('Page.enable').catch(() => {})
  const read = () => cdp.evaluate<Page>(READ)
  return {
    cdp,
    read,
    items: async (p) => cdp.evaluate<Item[]>(itemsScript(p)),
    click: async (e: Element) => {
      const ok = await cdp.evaluate<boolean>(`(() => { const el = document.querySelector('[data-atelier-ref="${e.ref}"]'); if (!el) return false; el.scrollIntoView({ block: 'center' }); el.click(); return true; })()`)
      if (!ok) throw new Error(`“${e.text}” is no longer on the page`)
      await settle(cdp)
    },
    open: async (url) => {
      await cdp.send('Page.navigate', { url })
      await settle(cdp)
    },
    type: async (e: Element, text: string) => {
      // The native setter and input/change events, so pages built on React, Vue and the like see the change.
      const ok = await cdp.evaluate<boolean>(`(() => { const el = document.querySelector('[data-atelier-ref="${e.ref}"]'); if (!el) return false; el.scrollIntoView({ block: 'center' }); el.focus();
        const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
        Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, ${JSON.stringify(text)});
        el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true })); el.blur(); return true; })()`)
      if (!ok) throw new Error(`“${e.text}” is no longer on the page`)
    },
    choose: async (e: Element, option: string) => {
      const ok = await cdp.evaluate<boolean>(`(() => { const el = document.querySelector('[data-atelier-ref="${e.ref}"]'); if (!el) return false; const want = ${JSON.stringify(option.toLowerCase())};
        const o = [...el.options].find((x) => x.text.trim().toLowerCase() === want) || [...el.options].find((x) => x.text.trim().toLowerCase().includes(want)); if (!o) return false;
        el.value = o.value; el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true })); return true; })()`)
      if (!ok) throw new Error(`no option “${option}” in “${e.text}”`)
      await settle(cdp)
    },
    snapshot: async () => ({ page: await read(), candidates: await cdp.evaluate(CANDIDATES) }),
  }
}

/** The planner's try-out on a real tab: its items must be found, and their fields read. */
export function tabTrial(tab: TabSurface): (p: Process) => Promise<string[]> {
  return async (p) => {
    if (p.over !== 'items') return []
    const items = await tab.items(p)
    if (items.length === 0) return [`items.selector "${p.items?.selector}" matches nothing on this page`]
    const names = (p.items?.fields ?? []).map((f) => f.name)
    const empty = names.filter((n) => items.filter((i) => i.fields[n]).length < items.length / 2)
    return empty.length ? [`the field selector(s) for ${empty.join(', ')} find nothing in most items (${items.length} items found)`] : []
  }
}
