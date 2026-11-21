// A System One server on 127.0.0.1 around the on-device decider, so Atelier (or its eval) can use it as
// "your own server":  node --experimental-strip-types src/server.ts <model dir> [port] [threads] [model file] [decline|answer]

import { createServer } from 'node:http'
import { NanoDecider } from './decider.ts'

const [dir, port = '8011', threads = '4', file = 'model.emb4.int8.onnx', open = 'decline'] = process.argv.slice(2)
if (!dir) {
  console.error('usage: server.ts <model dir> [port] [threads] [model file]')
  process.exit(2)
}
const decider = await NanoDecider.load(dir, { threads: Number(threads), file, open: open === 'answer' ? 'answer' : 'decline' })
const name = 'atelier-nano'

const server = createServer(async (req, res) => {
  const send = (status: number, body: unknown): void => {
    res.writeHead(status, { 'content-type': 'application/json' })
    res.end(JSON.stringify(body))
  }
  try {
    if (req.method === 'GET' && req.url?.endsWith('/models')) return send(200, { data: [{ id: name, object: 'model', owned_by: 'local' }] })
    if (req.method !== 'POST' || !req.url?.endsWith('/systemone')) return send(404, { error: 'not found' })
    const chunks: Buffer[] = []
    for await (const chunk of req) chunks.push(chunk as Buffer)
    const body = JSON.parse(Buffer.concat(chunks).toString('utf8'))
    const t0 = performance.now()
    const out = await decider.decide({ state: body.state, questions: body.questions })
    send(200, { model: name, ...out, ms: Math.round((performance.now() - t0) * 100) / 100 })
  } catch (err) {
    send(422, { error: err instanceof Error ? err.message : String(err) })
  }
}).listen(Number(port), '127.0.0.1', () => console.log(`atelier-nano (${file}) on http://127.0.0.1:${port}/v1/systemone`))

const stop = async (): Promise<void> => {
  server.close()
  await decider.release()
  process.exit(0)
}
process.on('SIGINT', stop)
process.on('SIGTERM', stop)
