// Atelier's answer readers and cutoffs (atelier: src/shared/decisions.ts, src/main/decisions/uses.ts): a line-for-line
// port of student/readers.py. A verdict is a sure answer that takes over from the rule, or null (unsure: the rule
// decides). The cascade uses it to decide whether the nano's answer stands or the question goes to the mini.

import type { Answer } from './decider.ts'
import type { Kind } from './format.ts'

type Verdict = boolean | string | number | null

const yesNo = (pTrue: number, yes = 0.7, no = 0.3): boolean | null => (pTrue >= yes ? true : pTrue <= no ? false : null)

function sureChoice(probs: Record<string, number>, cutoff = 0.75): string | null {
  let best: string | null = null
  for (const [k, p] of Object.entries(probs)) if (best === null || p > (probs[best] ?? -1)) best = k
  return best !== null && (probs[best] ?? 0) >= cutoff ? best : null
}

function sureLevel(probs: number[], cutoff = 0.6): number | null {
  let best = 0
  for (let i = 1; i < probs.length; i++) if ((probs[i] ?? 0) > (probs[best] ?? 0)) best = i
  return (probs[best] ?? 0) >= cutoff ? best : null
}

/** The verdict Atelier's reader makes for one question (kind from fixedKind, or 'open'). */
export function verdict(kind: Kind | 'open', labels: string[], probs: number[]): Verdict {
  const p = Object.fromEntries(labels.map((l, i) => [l, probs[i] ?? 0]))
  const t = p.true ?? 0
  switch (kind) {
    case 'waits':
      return yesNo(t, 0.75, 0.25)
    case 'dictation':
      return yesNo(t, 0.85, 0.15) === true ? true : null
    case 'kind': {
      const k = sureChoice(p, 0.8)
      return k === null || k === 'none' ? null : k
    }
    case 'intent':
      return sureChoice(p, 0.8)
    case 'target':
      return sureChoice(p, 0.85) === 'companion' ? true : null
    case 'effort':
      return sureLevel(probs, 0.7)
    case 'lasting':
      return yesNo(t, 0.7, 0.2) === false ? false : null
    case 'place':
      return sureChoice(p, 0.75)
    case 'skill':
      return yesNo(t, 0.7, 0.2)
    case 'techLevel':
    case 'replyStyle':
      return sureChoice(p, 0.75)
    case 'interest':
      return yesNo(t, 0.75, 0.25)
  }
  // label_items and other open questions: yes at 0.7, no at 0.3, a choice at 0.75, a level at 0.6
  if (labels.length === 2 && labels[0] === 'false' && labels[1] === 'true') return yesNo(t)
  if (labels.every((l) => /^\d+$/.test(l))) return sureLevel(probs, 0.6)
  return sureChoice(p, 0.75)
}

/** An answer's probabilities in option order, or null for a refusal. */
export function probabilities(a: Answer, labels: string[]): number[] | null {
  if (a.type === 'refusal') return null
  if (a.type === 'noul') return [1 - a.noul, a.noul]
  return labels.map((l) => a.probabilities[l] ?? 0)
}
