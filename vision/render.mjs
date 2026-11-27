/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// Draws vision/synth.py's pages with a real browser engine, the one Atelier itself shows pages with (Chromium), so a
// generated button or page looks as it would when Atelier takes its picture.
//
//   node vision/render.mjs --specs <specs.jsonl> --out <folder> [--browser <path>] [--tabs 6] [--port 9377]
//
// specs.jsonl: one page per line, {"id", "html", "width", "height", "scale"?, "clip"?: CSS selector, "margin"?}.
// Writes <folder>/<id>.png: the whole viewport, or the selected element's box with `margin` CSS pixels around it.
// The browser runs headless with a throwaway profile; any Chromium works (Chrome, Brave, Edge, Chromium).
// No dependencies: the DevTools protocol over Node's own WebSocket.

import { spawn } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), []))
const BROWSERS = [
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
  '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
  '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
]
const browser = args.browser ?? BROWSERS.find((p) => existsSync(p))
if (!args.specs || !args.out || !browser) {
  console.error('usage: render.mjs --specs <specs.jsonl> --out <folder> [--browser <path to a Chromium>] [--tabs 6] [--port 9377]')
  process.exit(2)
}
const port = Number(args.port ?? 9377)
const tabs = Number(args.tabs ?? 6)
mkdirSync(args.out, { recursive: true })
const specs = readFileSync(args.specs, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l)).filter((s) => !existsSync(join(args.out, `${s.id}.png`)))

const profile = mkdtempSync(join(tmpdir(), 'atelier-vision-render-'))
const child = spawn(browser, ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--hide-scrollbars', '--no-first-run',
  '--no-default-browser-check', '--disable-extensions', '--force-color-profile=srgb', '--disable-background-networking', '--mute-audio', 'about:blank'], { stdio: 'ignore' })
const stop = () => {
  child.kill()
  try {
    rmSync(profile, { recursive: true, force: true })
  } catch {}
}
process.on('exit', stop)

let ws = null
for (let i = 0; i < 100 && !ws; i++) {
  try {
    const version = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json()
    ws = new WebSocket(version.webSocketDebuggerUrl)
    await new Promise((resolve, reject) => {
      ws.onopen = resolve
      ws.onerror = reject
    })
  } catch {
    ws = null
    await new Promise((r) => setTimeout(r, 200))
  }
}
if (!ws) throw new Error(`no browser on port ${port}`)

let next = 1
const waiting = new Map()
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data)
  const w = waiting.get(msg.id)
  if (!w) return
  waiting.delete(msg.id)
  if (msg.error) w.reject(new Error(msg.error.message))
  else w.resolve(msg.result)
}
const send = (method, params = {}, sessionId) =>
  new Promise((resolve, reject) => {
    const id = next++
    waiting.set(id, { resolve, reject })
    ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }))
  })

let done = 0
let failed = 0
const t0 = Date.now()
async function worker() {
  const { targetId } = await send('Target.createTarget', { url: 'about:blank' })
  const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true })
  const { frameTree } = await send('Page.getFrameTree', {}, sessionId)
  for (;;) {
    const spec = specs.pop()
    if (!spec) break
    try {
      await send('Emulation.setDeviceMetricsOverride', { width: spec.width, height: spec.height, deviceScaleFactor: spec.scale ?? 1, mobile: false }, sessionId)
      await send('Page.setDocumentContent', { frameId: frameTree.frame.id, html: spec.html }, sessionId)
      // Fonts and pictures in, then two frames, so what is captured is what a person would see.
      const settle = `Promise.all([document.fonts.ready, ...[...document.images].map((i) => i.decode().catch(() => {}))]).then(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))))`
      await send('Runtime.evaluate', { expression: settle, awaitPromise: true }, sessionId)
      let clip
      if (spec.clip) {
        const m = spec.margin ?? 0
        const box = await send('Runtime.evaluate', { expression: `JSON.stringify((() => { const r = document.querySelector(${JSON.stringify(spec.clip)}).getBoundingClientRect(); return [r.x, r.y, r.width, r.height] })())`, returnByValue: true }, sessionId)
        const [x, y, w, h] = JSON.parse(box.result.value)
        const left = Math.max(0, Math.floor(x - m))
        const top = Math.max(0, Math.floor(y - m))
        clip = { x: left, y: top, width: Math.min(spec.width, Math.ceil(x + w + m)) - left, height: Math.min(spec.height, Math.ceil(y + h + m)) - top, scale: 1 }
      }
      const shot = await send('Page.captureScreenshot', { format: 'png', ...(clip ? { clip } : {}) }, sessionId)
      writeFileSync(join(args.out, `${spec.id}.png`), Buffer.from(shot.data, 'base64'))
      done++
    } catch (err) {
      failed++
      if (failed <= 5) console.error(`${spec.id}: ${err.message}`)
    }
    if ((done + failed) % 500 === 0) console.log(`  ${done + failed} drawn in ${Math.round((Date.now() - t0) / 1000)} s, ${failed} failed`)
  }
  await send('Target.closeTarget', { targetId })
}
const total = specs.length
await Promise.all(Array.from({ length: Math.min(tabs, Math.max(1, total)) }, worker))
console.log(`${done} of ${total} pages drawn to ${args.out} in ${Math.round((Date.now() - t0) / 1000)} s${failed ? `, ${failed} failed` : ''}`)
ws.close()
stop()
process.exit(failed ? 1 : 0)
