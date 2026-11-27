/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// atelier-vision's image input: a line-for-line port of vision/preprocess.py (the contract between training and this
// runtime). Any picture becomes the 256 x 256 square the image tower reads: transparent parts over white, the whole
// picture resized without keeping its proportions (a triangle filter whose width grows with the reduction, rows
// first, in double precision), every value x / 127.5 - 1, channels first. src/vision/parity.ts checks the numbers.

export const SIZE = 256

/** Decoded pixels, row by row from the top left: RGB or RGBA, 8 bits each. */
export interface Picture {
  data: Uint8Array | Uint8ClampedArray
  width: number
  height: number
  channels: 3 | 4
}

/** For each output position: the first input position it reads and its weights (they sum to 1). */
function taps(nIn: number, nOut: number): Array<{ lo: number; w: Float64Array }> {
  const scale = nIn / nOut
  const support = Math.max(scale, 1)
  const out: Array<{ lo: number; w: Float64Array }> = []
  for (let i = 0; i < nOut; i++) {
    const center = (i + 0.5) * scale
    const lo = Math.max(0, Math.trunc(center - support + 0.5))
    const hi = Math.min(nIn, Math.trunc(center + support + 0.5))
    const w = new Float64Array(hi - lo)
    let sum = 0
    for (let j = lo; j < hi; j++) {
      const v = Math.max(0, 1 - Math.abs((j + 0.5 - center) / support))
      w[j - lo] = v
      sum += v
    }
    for (let k = 0; k < w.length; k++) w[k] = w[k]! / sum
    out.push({ lo, w })
  }
  return out
}

/** The tower's input: float32 [3 * size * size], channels first, in -1..1. */
export function preprocess(p: Picture, size = SIZE): Float32Array {
  const { data, width, height, channels } = p
  if (data.length !== width * height * channels) throw new Error(`a ${width} x ${height} picture with ${channels} channels has ${width * height * channels} values, not ${data.length}`)
  // Along each row first: [height, size, 3].
  const rows = new Float64Array(height * size * 3)
  const across = taps(width, size)
  for (let y = 0; y < height; y++) {
    for (let o = 0; o < size; o++) {
      const { lo, w } = across[o]!
      let r = 0
      let g = 0
      let b = 0
      for (let k = 0; k < w.length; k++) {
        const at = (y * width + lo + k) * channels
        let pr = data[at]!
        let pg = data[at + 1]!
        let pb = data[at + 2]!
        if (channels === 4) {
          const alpha = data[at + 3]! / 255
          pr = pr * alpha + 255 * (1 - alpha)
          pg = pg * alpha + 255 * (1 - alpha)
          pb = pb * alpha + 255 * (1 - alpha)
        }
        r += pr * w[k]!
        g += pg * w[k]!
        b += pb * w[k]!
      }
      const to = (y * size + o) * 3
      rows[to] = r
      rows[to + 1] = g
      rows[to + 2] = b
    }
  }
  // Then along each column, straight into channels-first order.
  const out = new Float32Array(3 * size * size)
  const down = taps(height, size)
  for (let o = 0; o < size; o++) {
    const { lo, w } = down[o]!
    for (let x = 0; x < size; x++) {
      for (let c = 0; c < 3; c++) {
        let v = 0
        for (let k = 0; k < w.length; k++) v += rows[((lo + k) * size + x) * 3 + c]! * w[k]!
        out[c * size * size + o * size + x] = v / 127.5 - 1
      }
    }
  }
  return out
}

/** vision/preprocess.py checksum: a number that changes when the input does. */
export function checksum(x: Float32Array): number {
  let sum = 0
  for (let i = 0; i < x.length; i++) sum += x[i]! * (((i % 97) + 1) / 97)
  return sum
}
