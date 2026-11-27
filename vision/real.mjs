/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// Icon-only links and buttons from live websites, to see what atelier-vision makes of buttons nobody drew for it.
// Each is cropped as Atelier crops a control the page gave no name (its box and 3 CSS pixels around it, at 2x), and
// the name the page's own code gives it, where it has one (aria-label, title, alt), is kept as what it really is.
//
//   node vision/real.mjs <out folder> <url> [<url> ...]      then: .venv/bin/python -I vision/real.py <out folder>
//
// Headless, with a throwaway profile; any Chromium works. Sites that answer a headless browser with a wall give
// nothing, which the run says.
import { spawn } from 'node:child_process'
import { existsSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
const [out, ...urls] = process.argv.slice(2)
const browser = ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '/Applications/Chromium.app/Contents/MacOS/Chromium', '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser', '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge', '/usr/bin/google-chrome', '/usr/bin/chromium'].find((p) => existsSync(p))
const port = 9391
const profile = mkdtempSync(join(tmpdir(), 'real-icons-'))
const proc = spawn(browser, ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--no-first-run', '--window-size=1280,900', '--hide-scrollbars', 'about:blank'], { stdio: 'ignore' })
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
let version
for (let i = 0; i < 50 && !version; i++) { await sleep(200); version = await fetch(`http://127.0.0.1:${port}/json/version`).then((r) => r.json()).catch(() => null) }
const rows = []
for (const url of urls) {
  try {
    const target = await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: 'PUT' }).then((r) => r.json())
    const ws = new WebSocket(target.webSocketDebuggerUrl)
    await new Promise((r) => (ws.onopen = r))
    let id = 0; const waits = new Map()
    ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.id && waits.has(m.id)) { waits.get(m.id)(m.result ?? {}); waits.delete(m.id) } }
    const send = (method, params = {}) => new Promise((r) => { const n = ++id; waits.set(n, r); ws.send(JSON.stringify({ id: n, method, params })) })
    await send('Page.enable'); await send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 900, deviceScaleFactor: 2, mobile: false })
    await send('Page.navigate', { url }); await sleep(7000)
    const found = await send('Runtime.evaluate', { returnByValue: true, expression: `(() => {
      const out = []
      for (const el of document.querySelectorAll('a[href], button, [role=button]')) {
        const r = el.getBoundingClientRect()
        if (r.width < 14 || r.height < 14 || r.width > 72 || r.height > 72 || r.top < 0 || r.left < 0 || r.bottom > innerHeight || r.right > innerWidth) continue
        const cs = getComputedStyle(el); if (cs.visibility === 'hidden' || cs.display === 'none' || Number(cs.opacity) < 0.2) continue
        if ((el.innerText || '').trim()) continue
        if (!el.querySelector('svg, img, i, span') && cs.backgroundImage === 'none') continue
        const top = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2); if (!top || !(el === top || el.contains(top) || top.contains(el))) continue
        const name = el.getAttribute('aria-label') || el.getAttribute('title') || el.querySelector('img')?.getAttribute('alt') || el.querySelector('title')?.textContent || ''
        out.push({ x: r.left, y: r.top, w: r.width, h: r.height, name: name.trim().slice(0, 60), named: Boolean(el.getAttribute('aria-label') || el.getAttribute('title')), lang: document.documentElement.lang || '' })
        if (out.length >= 14) break
      }
      return out })()` })
    const host = new URL(url).hostname.replace(/^www\\./, '')
    let n = 0
    for (const b of found.result?.value ?? []) {
      const shot = await send('Page.captureScreenshot', { format: 'png', clip: { x: Math.max(0, b.x - 3), y: Math.max(0, b.y - 3), width: b.w + 6, height: b.h + 6, scale: 1 } })
      if (!shot.data) continue
      const file = `${host}-${n++}.png`
      writeFileSync(join(out, file), Buffer.from(shot.data, 'base64'))
      rows.push({ file, site: host, truth: b.name, named_by_page: b.named, lang: b.lang })
    }
    console.error(host, n)
    ws.close(); await fetch(`http://127.0.0.1:${port}/json/close/${target.id}`).catch(() => null)
  } catch (err) { console.error(url, String(err).slice(0, 100)) }
}
writeFileSync(join(out, 'buttons.jsonl'), rows.map((r) => JSON.stringify(r)).join('\n') + '\n')
proc.kill(); await sleep(300); rmSync(profile, { recursive: true, force: true })
