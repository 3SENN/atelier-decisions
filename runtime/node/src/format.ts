// The student's input format: a line-for-line port of student/format.py (the contract between training and
// this runtime). runtime/node/src/parity.ts checks the two produce the same token ids and answers.

export type Content = string | number | boolean | null | Content[] | { [k: string]: Content }
export interface Question {
  type: 'choice' | 'noul' | 'score'
  instructions?: Content | null
  criteria?: Record<string, Content | null> | Content[] | null
}
export interface DecisionRequest {
  state: Content
  questions: Record<string, Question>
}

export const OPT = '<unused0>'
export const OPEN = '<unused1>'
export const TAGS = {
  waits: '<unused2>',
  dictation: '<unused3>',
  kind: '<unused4>',
  intent: '<unused5>',
  target: '<unused6>',
  effort: '<unused7>',
  lasting: '<unused8>',
  place: '<unused9>',
  skill: '<unused10>',
  techLevel: '<unused11>',
  replyStyle: '<unused12>',
  interest: '<unused13>',
} as const
export type Kind = keyof typeof TAGS
const SIGNATURES: Record<Kind, string> = {
  waits: 'This is the end of an assistant’s reply.',
  dictation: 'Someone spoke into a text field.',
  kind: 'An assistant is about to activate this control',
  intent: 'Someone typed this into their browser’s search box.',
  target: 'Someone spoke while a text field was focused',
  effort: 'Someone sent this message to their AI assistant. How much careful thinking',
  lasting: 'This is a conversation between a person and their AI assistant. Does it tell the assistant anything worth',
  place: 'Someone is sorting their open browser tabs into groups.',
  skill: 'Someone sent this first message to their AI assistant. Would the assistant’s skill',
  techLevel: 'From this setup chat with their new browser assistant',
  replyStyle: 'Did the person say how they like the assistant’s answers?',
  interest: 'Does the person want help with, or do a lot of,',
}
const SKILL_RE = /skill “\/([^”]+)” help with it\? The skill: ([\s\S]*)$/
const GROUP_RE = /^Their own group “([\s\S]*)”: anything that belongs with it\.$/
const HEAD_TAIL = new Set(['waits', 'lasting'])
// Decisions over whole chats run in the background with seconds to spare: they may read more.
const LONG_LIMITS: Record<string, number> = { lasting: 1536, techLevel: 2048, replyStyle: 2048, interest: 2048 }

/** Python's json.dumps(value, ensure_ascii=False): ", " and ": " separators. */
export function pyJson(value: Content | undefined): string {
  if (value === null || value === undefined) return 'null'
  if (Array.isArray(value)) return `[${value.map(pyJson).join(', ')}]`
  if (typeof value === 'object') return `{${Object.entries(value).map(([k, v]) => `${JSON.stringify(k)}: ${pyJson(v)}`).join(', ')}}`
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  return JSON.stringify(value)
}

const text = (v: Content | null | undefined): string => (typeof v === 'string' ? v : pyJson(v ?? null))

export function optionLabels(q: Question): string[] {
  if (q.type === 'choice') return Object.keys(q.criteria as Record<string, Content>)
  if (q.type === 'score') return (q.criteria as Content[]).map((_, i) => String(i))
  return ['false', 'true']
}

export function fixedKind(q: Question): Kind | null {
  const instructions = typeof q.instructions === 'string' ? q.instructions : ''
  for (const [name, signature] of Object.entries(SIGNATURES) as [Kind, string][]) {
    if (instructions.startsWith(signature) || ((name === 'skill' || name === 'interest') && instructions.includes(signature))) return name
  }
  return null
}

export function stateText(state: Content): string {
  if (typeof state === 'string') return state
  if (state && typeof state === 'object' && !Array.isArray(state)) {
    return Object.entries(state).map(([k, v]) => `${k}: ${typeof v === 'string' ? v : pyJson(v)}`).join('\n')
  }
  return pyJson(state)
}

export function questionParts(q: Question): { head: string; options: string[]; kind: Kind | null } {
  const kind = fixedKind(q)
  const labels = optionLabels(q)
  const instructions = typeof q.instructions === 'string' ? q.instructions : ''
  if (kind === 'skill') {
    const m = SKILL_RE.exec(instructions)
    const [name, description] = m ? [m[1] ?? '', m[2] ?? ''] : ['', instructions]
    return { head: `${TAGS.skill} /${name}: ${description}`, options: ['false', 'true'], kind }
  }
  if (kind === 'interest') {
    const topic = (instructions.split(',').pop() ?? '').replace(/^[ ?]+|[ ?]+$/g, '')
    return { head: `${TAGS.interest} ${topic}`, options: ['false', 'true'], kind }
  }
  if (kind === 'place') {
    const criteria = q.criteria as Record<string, Content | null>
    const options = labels.map((label) => {
      const g = GROUP_RE.exec(typeof criteria[label] === 'string' ? (criteria[label] as string) : '')
      return g ? `group: ${g[1]}` : label
    })
    return { head: TAGS.place, options, kind }
  }
  if (kind !== null) {
    if (q.type === 'score') return { head: TAGS[kind], options: labels.map((_, i) => `level ${i}`), kind }
    return { head: TAGS[kind], options: labels, kind }
  }
  const head = `${OPEN} ${q.instructions ? text(q.instructions) : 'Choose the best matching option.'}`
  if (q.type === 'choice') {
    const options = Object.entries(q.criteria as Record<string, Content | null>).map(([k, v]) => (v === null ? k : `${k}: ${text(v)}`))
    return { head, options, kind }
  }
  if (q.type === 'score') return { head, options: (q.criteria as Content[]).map((v) => (typeof v === 'string' ? v : pyJson(v))), kind }
  const c = (q.criteria ?? {}) as Record<string, Content | null>
  return { head, options: [c.false ? `no: ${text(c.false)}` : 'no', c.true ? `yes: ${text(c.true)}` : 'yes'], kind }
}

export interface Tokenize {
  encode(text: string): number[]
  id(token: string): number
  bos: number
  eos: number
}

/** Token ids of one question with its state, and where its option markers are. */
export function encode(tok: Tokenize, state: Content, q: Question, maxLength = 512): { ids: number[]; markers: number[]; kind: string } {
  const { head, options, kind } = questionParts(q)
  const opt = tok.id(OPT)
  const ids = [tok.bos, ...tok.encode(head).slice(0, 160), tok.eos]
  const markers: number[] = []
  for (const option of options) {
    markers.push(ids.length)
    ids.push(opt, ...tok.encode(option).slice(0, 24))
  }
  ids.push(tok.eos)
  const limit = Math.max(maxLength, LONG_LIMITS[kind ?? ''] ?? 0)
  const budget = Math.max(16, limit - ids.length - 1)
  let stateIds = tok.encode(stateText(state))
  if (stateIds.length > budget) stateIds = kind !== null && HEAD_TAIL.has(kind) ? stateIds.slice(-budget) : stateIds.slice(0, budget)
  ids.push(...stateIds, tok.eos)
  return { ids, markers, kind: kind ?? 'open' }
}
