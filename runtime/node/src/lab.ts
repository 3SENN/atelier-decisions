/* Copyright (c) 2026 The Atelier Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

// The process lab on the command line. A decision server must be running (npm start), unless --decider names
// another System One address.
//
//   npm run lab -- list                              fixtures, their tasks and the ready-made processes
//   npm run lab -- check                             every process and saved plan can run; prices read right
//   npm run lab -- run inbox:triage [--planned] [-v] run a task's process over its mock page, item by item
//   npm run lab -- plan inbox:triage [--backend claude|api] [--model claude-opus-5-5]
//                                                    a language model sets the process up (saved in lab/plans/)
//   npm run lab -- bench [--planned] [--out <file>]  every task, measured against what a person expects
//   npm run lab -- steps <rows.jsonl> [--limit n]     step questions with known answers (data/build_steps.py), scored
//   npm run lab -- flows --cdp 9555 [--planned | --planned-only] [--step] [--escalate] [--approve] [--only shop-order] [-v]
//                                                    whole flows on real pages (order, find, a form), checked afterwards
//   npm run lab -- tab --cdp 9555 [--match <url part>] (--process <id or file.json> | --task "<words>" [--buckets a,b]) [--act]
//                     [--expect inbox:triage]       (score a mock page opened in the browser)
//                                                    a real browser tab: Atelier started with --remote-debugging-port

import { readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { tabSurface, tabTrial } from './process/cdp.ts'
import { httpDecide, httpStep, type Decide } from './process/decide.ts'
import { loadFixtures, loadProcesses } from './process/fixtures.ts'
import { findTask, PLANS_DIR, planTask, processFor, runTask, type Source } from './process/lab.ts'
import { claudeEscalation, plan, type Backend, type EscalationLog } from './process/planner.ts'
import { libraryFlowProcess, loadFlows, planFlow, resetAndOpen, runFlow, savedFlowPlan } from './process/flows.ts'
import { runProcess, type ItemResult, type RunSummary } from './process/run.ts'
import { normalize, parseNumber, problems, type Process } from './process/schema.ts'

const argv = process.argv.slice(2)
const flags: Record<string, string | true> = {}
const args: string[] = []
for (let i = 0; i < argv.length; i++) {
  const a = argv[i]!
  if (a === '-v') flags.verbose = true
  else if (a.startsWith('--')) {
    const next = argv[i + 1]
    if (next !== undefined && !next.startsWith('-')) {
      flags[a.slice(2)] = next
      i++
    } else flags[a.slice(2)] = true
  } else args.push(a)
}
const str = (k: string) => (typeof flags[k] === 'string' ? (flags[k] as string) : undefined)
const decide: Decide = httpDecide(str('decider') ?? process.env.ATELIER_DECISIONS_URL ?? 'http://127.0.0.1:8011', { model: str('model-id'), key: process.env.ATELIER_DECISIONS_KEY })

const pct = (a: number, b: number) => (b ? `${Math.round((100 * a) / b)}%` : '-')
const ms = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)} s` : `${Math.round(n)} ms`)
const MARK = { right: '✓', wrong: '✗', person: '?' } as const

function outcomeText(r: ItemResult): string {
  const o = r.outcome
  if (o.kind === 'bucket') return o.bucket
  if (o.kind === 'person') return `person (${o.reason})`
  if (o.kind === 'notified') return `notify: ${o.text}`
  if (o.kind === 'error') return `error: ${o.message}`
  return r.clicked.length ? `clicked “${r.clicked.join('”, “')}”` : 'stopped'
}

function printItem(r: ItemResult, verbose: boolean): void {
  const mark = r.score ? MARK[r.score] : ' '
  const expect = r.score && r.score !== 'right' && r.item.expect ? `  (expected ${r.item.expect})` : ''
  const path = r.steps.map((s) => `${s.step}${s.verdict ? `→${s.verdict}` : '→?'}${s.by === 'code' ? '' : ` ${s.by}`}`).join(' · ')
  console.log(`${mark} ${r.label.padEnd(60).slice(0, 60)} ${outcomeText(r)}${expect}`)
  console.log(`    ${path}  ${ms(r.ms)}`)
  if (verbose)
    for (const s of r.steps) {
      const p = s.probabilities ? Object.entries(s.probabilities).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(', ') : ''
      console.log(`      ${s.step} [${s.by}, ${ms(s.ms)}] ${s.question ?? ''}${p ? `  → ${p}` : ''}${s.note ? `  (${s.note})` : ''}`)
    }
}

function printSummary(s: RunSummary): void {
  const decided = s.scored - s.person
  console.log(
    `\n${s.process}: ${s.items} items in ${ms(s.msTotal)} · ${s.decisions} model decisions, ${pct(s.sure, s.decisions)} sure, ` +
      `median ${ms(s.msPerDecision.median)}, p90 ${ms(s.msPerDecision.p90)} · by ${Object.entries(s.byWho).map(([k, v]) => `${k} ${v}`).join(', ')}`,
  )
  if (s.scored)
    console.log(`  against a person: decided ${decided}/${s.scored} (${pct(decided, s.scored)}), right when decided ${s.right}/${decided} (${pct(s.right, decided)}), wrong ${s.wrong}, to the person ${s.person}`)
}

function printProcess(p: Process): void {
  console.log(`${p.id}: ${p.task}`)
  for (const s of p.steps) {
    const q = s.kind === 'check' ? `${s.read[0]} ${s.op} ${s.value}` : s.question
    const routes = s.routes.map((r) => `${r.when}→${r.then.do === 'goto' ? r.then.target : `${r.then.do}${r.then.target ? ` ${r.then.target}` : ''}`}`).join(', ')
    console.log(`  ${s.id} (${s.kind}${s.answer ? ` ${s.answer}` : ''}): ${q}\n      ${routes}`)
  }
  console.log()
}

async function main(): Promise<void> {
  const cmd = args[0]
  const fixtures = loadFixtures()
  const verbose = flags.verbose === true
  if (cmd === 'list' || !cmd) {
    for (const f of fixtures) for (const t of f.tasks) console.log(`${`${f.id}:${t.key}`.padEnd(22)} ${f.kind.padEnd(6)} ${t.process.padEnd(16)} ${t.task}`)
    console.log(`\nprocesses: ${Object.keys(loadProcesses()).join(', ')}`)
    return
  }
  if (cmd === 'check') {
    // Every ready-made process and saved plan can run, and prices read as people write them.
    let bad = 0
    const numbers: Array<[string, number]> = [['€ 849,00', 849], ['€1.049', 1049], ['EUR 799', 799], ['€899.99', 899.99], ['$1,299.00', 1299], ['1.299.000', 1299000], ['12,5%', 12.5], ['from €729', 729], ['0,99', 0.99], ['1,299', 1299], ['1 299 kr', 1299], ['-3', -3]]
    for (const [text, want] of numbers) if (parseNumber(text) !== want) (bad++, console.log(`✗ parseNumber("${text}") = ${parseNumber(text)}, not ${want}`))
    for (const [id, raw] of Object.entries(loadProcesses())) {
      const found = problems(normalize(raw))
      if (found.length) (bad++, console.log(`✗ processes/${id}: ${found.join('; ')}`))
    }
    for (const dir of [PLANS_DIR, new URL('flows/', PLANS_DIR)])
      for (const f of readdirSync(dir).filter((x) => x.endsWith('.json'))) {
        const saved = JSON.parse(readFileSync(new URL(f, dir), 'utf8'))
        const found = saved.process ? problems(normalize(saved.process)) : ['no process']
        if (found.length) (bad++, console.log(`✗ ${dir.pathname.split('/runtime/node/')[1]}${f}: ${found.join('; ')}`))
      }
    for (const fx of fixtures) for (const t of fx.tasks) if (t.process && !loadProcesses()[t.process]) (bad++, console.log(`✗ ${fx.id}:${t.key} names a missing process ${t.process}`))
    console.log(bad ? `${bad} problem(s)` : 'all processes, plans and number cases are fine')
    if (bad) process.exitCode = 1
    return
  }
  if (cmd === 'plan') {
    const { fixture, task } = findTask(fixtures, args[1] ?? '')
    const r = await planTask(fixture, task, { backend: str('backend') as Backend | undefined, model: str('model'), decide })
    console.log(`${r.backend} ${r.model}: ${ms(r.ms)}, ${r.rounds} round(s), ${r.tokens.input} tokens in, ${r.tokens.output} out`)
    if (r.process) printProcess(r.process)
    else console.log(`can't run: ${r.problems.join('; ')}`)
    return
  }
  if (cmd === 'run') {
    const { fixture, task } = findTask(fixtures, args[1] ?? '')
    const p = processFor(fixture, task, flags.planned ? 'planned' : 'library')
    printProcess(p)
    const { summary } = await runTask(fixture, task, p, decide, { onItem: (r) => printItem(r, verbose) })
    printSummary(summary)
    return
  }
  if (cmd === 'bench') {
    const sources: Source[] = flags.planned === true ? ['library', 'planned'] : ['library']
    const rows: Array<{ ref: string; source: Source; summary: RunSummary; misses: string[] }> = []
    for (const f of fixtures)
      for (const t of f.tasks)
        for (const source of sources) {
          let p: Process
          try {
            p = processFor(f, t, source)
          } catch (err) {
            console.log(`${f.id}:${t.key} ${source}: ${(err as Error).message}`)
            continue
          }
          const { results, summary } = await runTask(f, t, p, decide)
          const misses = results.filter((r) => r.score && r.score !== 'right').map((r) => `${MARK[r.score!]} ${r.item.id} ${r.label.slice(0, 50)}: ${outcomeText(r)} (expected ${r.item.expect})`)
          rows.push({ ref: `${f.id}:${t.key}`, source, summary, misses })
          const s = summary
          const decided = s.scored - s.person
          console.log(`${`${f.id}:${t.key}`.padEnd(18)} ${source.padEnd(8)} decided ${pct(decided, s.scored).padStart(4)}  right ${pct(s.right, decided).padStart(4)}  wrong ${String(s.wrong).padStart(2)}  person ${String(s.person).padStart(2)}  ${s.decisions} decisions, median ${ms(s.msPerDecision.median)}`)
          if (verbose) for (const m of misses) console.log(`    ${m}`)
        }
    const total = (src: Source) => {
      const r = rows.filter((x) => x.source === src).map((x) => x.summary)
      const scored = r.reduce((a, s) => a + s.scored, 0)
      const person = r.reduce((a, s) => a + s.person, 0)
      const right = r.reduce((a, s) => a + s.right, 0)
      const wrong = r.reduce((a, s) => a + s.wrong, 0)
      return { scored, decided: scored - person, right, wrong, person }
    }
    for (const src of sources) {
      const t = total(src)
      console.log(`\n${src}: ${t.scored} items, decided ${t.decided} (${pct(t.decided, t.scored)}), right when decided ${t.right} (${pct(t.right, t.decided)}), wrong ${t.wrong}, to the person ${t.person}`)
    }
    const out = str('out')
    if (out) writeFileSync(out, `${JSON.stringify({ made: new Date().toISOString(), decider: str('decider') ?? 'http://127.0.0.1:8011', rows }, null, 2)}\n`)
    return
  }
  if (cmd === 'steps') {
    // Step questions with known answers (data/build_steps.py): how often the model is right, and sure as the runtime reads it.
    const file = args[1]
    if (!file) throw new Error('steps <rows.jsonl> [--limit n]')
    const rows = readFileSync(file, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l))
    const limit = Number(str('limit') ?? rows.length)
    const fam: Record<string, { n: number; top: number; sure: number; sureRight: number; ms: number[] }> = {}
    for (const r of rows.slice(0, limit)) {
      const t0 = performance.now()
      const out = await decide(r.request)
      const a = out.answers.q
      const probs: Record<string, number> = a && a.type === 'choice' ? a.probabilities : {}
      const sorted = Object.entries(probs).sort((x, y) => y[1] - x[1])
      const [best, p] = sorted[0] ?? [null, 0]
      const second = sorted[1]?.[1] ?? 0
      const family = r.meta.family as string
      // The runtime's cutoffs (steps.ts): an operation at 0.6; an element, a value or an option when clearly ahead.
      const sure = family === 'step-operation' ? p >= 0.6 : p >= (family.endsWith('target') ? 0.5 : 0.6) && p >= 1.5 * second
      const f = (fam[family] ??= { n: 0, top: 0, sure: 0, sureRight: 0, ms: [] })
      f.n++
      f.ms.push(performance.now() - t0)
      if (best === r.meta.truth) f.top++
      if (sure) (f.sure++, best === r.meta.truth && f.sureRight++)
    }
    for (const [k, f] of Object.entries(fam).sort()) {
      const med = [...f.ms].sort((x, y) => x - y)[Math.floor(f.ms.length / 2)] ?? 0
      console.log(`${k.padEnd(20)} ${String(f.n).padStart(4)} questions · right (top answer) ${pct(f.top, f.n).padStart(4)} · sure ${pct(f.sure, f.n).padStart(4)} · right when sure ${pct(f.sureRight, f.sure).padStart(4)} · ${ms(med)}`)
    }
    return
  }
  if (cmd === 'flows') {
    // Whole flows in a real browser: npm run lab -- flows --cdp 9555 [--planned] [--escalate] [--only shop-order]
    const port = Number(str('cdp') ?? 9555)
    const origin = str('origin') ?? 'http://127.0.0.1:8011'
    const flows = loadFlows()
    const tab = await tabSurface(port, str('match') ?? '127.0.0.1:8011')
    const log: EscalationLog = { calls: 0, ms: 0 }
    const escalate = flags.escalate === true ? claudeEscalation({ backend: str('backend') as Backend | undefined }, log) : undefined
    // --approve stands in for the person when the guard asks, the way someone who asked for the task would answer:
    // yes to the steps toward a payment page or a form's last page (Checkout, Continue to payment, a consent box
    // they asked for), never to the click that pays, orders, sends or submits, nor to security, accounts or
    // deleting. Every question is counted.
    let approvals: string[] = []
    const FINAL = /\b(pay now|pay €|pay \$|betalen|bezahlen|place (your )?order|buy now|order now|confirm (and )?pay|send|versturen|verzenden|absenden|submit)\b/i
    const approve = flags.approve === true ? async (q: string, kind: string) => {
      const label = /“([^”]*)”/.exec(q)?.[1] ?? q
      // Nobody who asked for one pair wants a second one in the cart.
      const yes = !['security', 'account', 'delete'].includes(kind) && !FINAL.test(label) && !/a second one/.test(q)
      approvals.push(`${yes ? 'yes' : 'no'}: ${q}`)
      return yes
    } : undefined
    const rows: Array<Record<string, unknown>> = []
    for (const t of flows.tasks.filter((x) => !str('only') || x.key === str('only'))) {
      const sources = flags['planned-only'] === true ? (['planned'] as const) : flags.planned === true ? (['library', 'planned'] as const) : (['library'] as const)
      for (const source of sources) {
        let p: Process | null
        let planMs = 0
        if (source === 'planned') {
          p = flags.replan === true ? null : savedFlowPlan(t)
          if (!p) {
            await resetAndOpen(tab, origin, t)
            const r = await planFlow(tab, t, flows.person, { backend: str('backend') as Backend | undefined })
            console.log(`  planned ${t.key} in ${ms(r.ms)} (${r.tokens.input} tokens in, ${r.tokens.output} out)${r.problems.length ? `: ${r.problems.join('; ')}` : ''}`)
            if (!r.process) continue
            p = r.process
            planMs = r.ms
          }
        } else p = libraryFlowProcess(t, flows.person)
        const before = { ...log }
        approvals = []
        const r = await runFlow(tab, origin, t, p, decide, { escalate, approve, stepModel: flags.step === true ? httpStep(origin) : undefined, onStep: verbose ? (_id, s) => console.log(`    ${s.step} [${s.by}, ${ms(s.ms)}] ${s.question ?? ''}${s.verdict === null ? ' → unsure' : ''}`) : undefined })
        const who = Object.entries(r.byWho).map(([k, v]) => `${k} ${v}`).join(', ')
        const calls = log.calls - before.calls
        console.log(`${r.ok ? '✓' : '✗'} ${t.key.padEnd(16)} ${source.padEnd(8)} ${outcomeText(r.result).slice(0, 70)}  · ${ms(r.ms)} · by ${who}${escalate ? ` · Claude ${calls} call(s), ${ms(log.ms - before.ms)}` : ''}`)
        for (const f of r.failures) console.log(`    ${f}`)
        for (const a of approvals) console.log(`    the person was asked: ${a}`)
        rows.push({ key: t.key, source, ok: r.ok, failures: r.failures, approvals, outcome: r.result.outcome, byWho: r.byWho, ms: r.ms, escalations: calls, escalationMs: log.ms - before.ms, planMs, steps: r.result.steps })
      }
    }
    const out = str('out')
    if (out) writeFileSync(out, `${JSON.stringify({ made: new Date().toISOString(), escalate: Boolean(escalate), rows }, null, 2)}\n`)
    tab.cdp.close()
    return
  }
  if (cmd === 'tab') {
    const port = Number(str('cdp') ?? 9555)
    const tab = await tabSurface(port, str('match') ?? '')
    const page = await tab.read()
    console.log(`tab: ${page.title} (${page.url}), ${page.elements.length} controls`)
    let p: Process
    if (str('process')) {
      const ref = str('process')!
      p = normalize(ref.endsWith('.json') ? JSON.parse(readFileSync(ref, 'utf8')) : loadProcesses()[ref])
      if (str('goal')) p.task = str('goal')!
    } else if (str('task')) {
      const snap = await tab.snapshot()
      const r = await plan(str('task')!, snap, { buckets: str('buckets')?.split(','), backend: str('backend') as Backend | undefined, model: str('model'), trial: tabTrial(tab) })
      console.log(`planned by ${r.backend} ${r.model} in ${ms(r.ms)} (${r.tokens.input} tokens in, ${r.tokens.output} out)`)
      if (!r.process) throw new Error(`the plan can't run: ${r.problems.join('; ')}`)
      p = r.process
      if (str('save')) writeFileSync(str('save')!, `${JSON.stringify(p, null, 2)}\n`)
    } else throw new Error('tab needs --process <id> or --task "<words>"')
    const found = problems(p)
    if (found.length) throw new Error(found.join('; '))
    printProcess(p)
    // --expect inbox:triage scores a mock page opened in the browser: its items carry the fixture's ids.
    let surface = tab as import('./process/run.ts').Surface
    if (str('expect')) {
      const { fixture, task } = findTask(fixtures, str('expect')!)
      const want = new Map((fixture.items ?? []).map((i) => [i.id, i.expect?.[task.key]]))
      surface = { ...tab, items: async (q) => (await tab.items(q)).map((i) => ({ ...i, ...(want.get(i.id) ? { expect: want.get(i.id) } : {}) })) }
    }
    const { summary } = await runProcess(p, surface, decide, { act: flags.act === true, onItem: (r) => printItem(r, verbose) })
    printSummary(summary)
    tab.cdp.close()
    return
  }
  throw new Error(`unknown command "${cmd}"`)
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : err)
  process.exit(1)
})
