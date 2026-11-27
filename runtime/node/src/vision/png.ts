/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// A PNG's pixels, for the server and the checks: Atelier itself hands over pixels it already decoded (a page's
// capture, a picture through nativeImage), so this runtime has no image library. 8-bit PNGs that are not interlaced:
// grey, grey with alpha, RGB, RGBA and palette (with its transparency).

import { inflateSync } from 'node:zlib'
import type { Picture } from './preprocess.ts'

const SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10]

export function decodePng(file: Uint8Array): Picture {
  if (SIGNATURE.some((b, i) => file[i] !== b)) throw new Error('not a PNG')
  const view = new DataView(file.buffer, file.byteOffset, file.byteLength)
  let width = 0
  let height = 0
  let depth = 0
  let colour = 0
  let interlace = 0
  let palette: Uint8Array | null = null
  let alphas: Uint8Array | null = null
  const data: Uint8Array[] = []
  for (let at = 8; at + 8 <= file.length; ) {
    const length = view.getUint32(at)
    const type = String.fromCharCode(...file.subarray(at + 4, at + 8))
    const body = file.subarray(at + 8, at + 8 + length)
    if (type === 'IHDR') {
      width = view.getUint32(at + 8)
      height = view.getUint32(at + 12)
      depth = body[8]!
      colour = body[9]!
      interlace = body[12]!
    } else if (type === 'PLTE') palette = body
    else if (type === 'tRNS') alphas = body
    else if (type === 'IDAT') data.push(body)
    else if (type === 'IEND') break
    at += 12 + length
  }
  if (depth !== 8 || interlace !== 0) throw new Error('only 8-bit PNGs that are not interlaced')
  const per = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 }[colour]
  if (!per || (colour === 3 && !palette)) throw new Error(`PNG colour type ${colour} is not supported`)
  const raw = inflateSync(Buffer.concat(data))
  const stride = width * per
  const lines = new Uint8Array(height * stride)
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)]!
    const from = y * (stride + 1) + 1
    const to = y * stride
    for (let x = 0; x < stride; x++) {
      const left = x >= per ? lines[to + x - per]! : 0
      const up = y > 0 ? lines[to + x - stride]! : 0
      const corner = x >= per && y > 0 ? lines[to + x - stride - per]! : 0
      let predicted = 0
      if (filter === 1) predicted = left
      else if (filter === 2) predicted = up
      else if (filter === 3) predicted = (left + up) >> 1
      else if (filter === 4) {
        const p = left + up - corner
        const a = Math.abs(p - left)
        const b = Math.abs(p - up)
        const c = Math.abs(p - corner)
        predicted = a <= b && a <= c ? left : b <= c ? up : corner
      }
      lines[to + x] = (raw[from + x]! + predicted) & 255
    }
  }
  if (colour === 2 || colour === 6) return { data: lines, width, height, channels: colour === 2 ? 3 : 4 }
  const out = new Uint8Array(width * height * 4)
  for (let i = 0; i < width * height; i++) {
    if (colour === 3) {
      const k = lines[i]!
      out.set([palette![k * 3]!, palette![k * 3 + 1]!, palette![k * 3 + 2]!, alphas && k < alphas.length ? alphas[k]! : 255], i * 4)
    } else {
      const g = lines[i * per]!
      out.set([g, g, g, colour === 4 ? lines[i * per + 1]! : 255], i * 4)
    }
  }
  return { data: out, width, height, channels: 4 }
}
