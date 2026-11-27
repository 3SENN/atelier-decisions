/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// How often a page needs atelier-vision for its buttons. For each live page: every link and button, split the way
// Atelier's page bridge splits them (its src/preload/page/bridge.ts accessibleName and stepPage): named by the page's
// own code, or with no name anywhere. The unnamed ones that are whole in the viewport and 12-160 px, at most 12 a
// page, are the ones Atelier asks atelier-vision about; each is cropped as Atelier crops it (its box and 3 px, 2x).
//
//   node vision/sites.mjs <out folder> <file with one URL a line>
//   .venv/bin/python -I vision/real.py <out folder> --runtime releases/vision-v0.1/model --name vision-sites-v0.1
//
// Headless, with a throwaway profile; any Chromium works. A site that answers a headless browser with a wall shows
// as a page with next to no text, and vision/real.py leaves it out of the count.
import { spawn } from 'node:child_process'
import { existsSync, mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const out = process.argv[2]
const urls = readFileSync(process.argv[3], 'utf8').split('\n').filter(Boolean)
const browser = ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '/Applications/Chromium.app/Contents/MacOS/Chromium', '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser', '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge', '/usr/bin/google-chrome', '/usr/bin/chromium'].find((p) => existsSync(p))
const port = 9393
const profile = mkdtempSync(join(tmpdir(), 'bench-sites-'))
const proc = spawn(browser, ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--no-first-run', '--window-size=1280,900', '--hide-scrollbars', 'about:blank'], { stdio: 'ignore' })
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
let version
for (let i = 0; i < 50 && !version; i++) { await sleep(200); version = await fetch(`http://127.0.0.1:${port}/json/version`).then((r) => r.json()).catch(() => null) }
const pages = [], buttons = []
const SCAN = `(() => {
  const clean = (s) => (s || '').replace(/\\s+/g, ' ').trim()
  const name = (el) => {
    const lb = el.getAttribute('aria-labelledby')
    if (lb) { const t = clean(lb.split(/\\s+/).map((id) => document.getElementById(id)?.textContent ?? '').join(' ')); if (t) return t }
    const aria = clean(el.getAttribute('aria-label')); if (aria) return aria
    const text = clean(el.innerText ?? el.textContent); if (text) return text
    const img = el.querySelector('img[alt], svg[aria-label], [aria-label]'); if (img) { const t = clean(img.getAttribute('alt') ?? img.getAttribute('aria-label')); if (t) return t }
    return clean(el.getAttribute('title'))
  }
  let named = 0, unnamed = 0, asked = 0; const out = []
  for (const el of document.querySelectorAll('a[href], button, [role=button], [role=link], input[type=button], input[type=submit]')) {
    if (!el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })) continue
    const r = el.getBoundingClientRect(); if (r.width <= 0 || r.height <= 0) continue
    if (el.disabled || el.getAttribute('aria-disabled') === 'true') continue
    if (el.tagName === 'INPUT' ? clean(el.value) || el.type === 'submit' : name(el)) { named++; continue }
    unnamed++
    const whole = r.left >= 0 && r.top >= 0 && r.right <= innerWidth && r.bottom <= innerHeight
    if (whole && r.width >= 12 && r.height >= 12 && r.width <= 160 && r.height <= 160 && asked < 12) { asked++; out.push({ x: r.left, y: r.top, w: r.width, h: r.height, hint: clean((el.getAttribute('class') || '') + ' ' + (el.getAttribute('href') || '') + ' ' + (el.id || '') + ' ' + (el.getAttribute('data-testid') || '')).slice(0, 90) }) }
  }
  return { named, unnamed, asked: out, lang: document.documentElement.lang || '', title: document.title.slice(0, 60), text: (document.body?.innerText || '').length }
})()`
for (const url of urls) {
  const host = new URL(url).hostname.replace(/^www\./, '')
  try {
    const target = await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: 'PUT' }).then((r) => r.json())
    const ws = new WebSocket(target.webSocketDebuggerUrl)
    await new Promise((r) => (ws.onopen = r))
    let id = 0; const waits = new Map()
    ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.id && waits.has(m.id)) { waits.get(m.id)(m.result ?? {}); waits.delete(m.id) } }
    const send = (method, params = {}) => Promise.race([new Promise((r) => { const n = ++id; waits.set(n, r); ws.send(JSON.stringify({ id: n, method, params })) }), sleep(15000).then(() => ({}))])
    await send('Page.enable'); await send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 900, deviceScaleFactor: 2, mobile: false })
    await send('Page.navigate', { url }); await sleep(6000)
    const v = (await send('Runtime.evaluate', { returnByValue: true, expression: SCAN })).result?.value
    if (v) {
      let n = 0
      for (const b of v.asked) {
        const shot = await send('Page.captureScreenshot', { format: 'png', clip: { x: Math.max(0, b.x - 3), y: Math.max(0, b.y - 3), width: b.w + 6, height: b.h + 6, scale: 1 } })
        if (!shot.data) continue
        const file = `${host}-${n++}.png`
        writeFileSync(join(out, file), Buffer.from(shot.data, 'base64'))
        buttons.push({ file, site: host, hint: b.hint, lang: v.lang, w: Math.round(b.w), h: Math.round(b.h) })
      }
      pages.push({ site: host, url, title: v.title, lang: v.lang, text: v.text, named: v.named, unnamed: v.unnamed, asked: n })
      console.error(host, 'named', v.named, 'unnamed', v.unnamed, 'asked', n, 'text', v.text)
    } else console.error(host, 'no answer')
    ws.close(); await fetch(`http://127.0.0.1:${port}/json/close/${target.id}`).catch(() => null)
  } catch (err) { console.error(host, String(err).slice(0, 100)) }
}
writeFileSync(join(out, 'pages.jsonl'), pages.map((r) => JSON.stringify(r)).join('\n') + '\n')
writeFileSync(join(out, 'buttons.jsonl'), buttons.map((r) => JSON.stringify(r)).join('\n') + '\n')
proc.kill(); await sleep(300); rmSync(profile, { recursive: true, force: true })
