import { createReadStream, statSync } from 'node:fs'
import { createServer, get as httpGet, request as httpRequest } from 'node:http'
import { extname, join, normalize } from 'node:path'
import { createHash, randomBytes } from 'node:crypto'
import { spawn } from 'node:child_process'

const root = process.env.HERMES_WEB_ROOT ?? '/home/rpw/hermes-desktop-web-artifact'
const backend = new URL(process.env.HERMES_WEB_BACKEND ?? 'http://127.0.0.1:9119')
const types = { '.css': 'text/css', '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.png': 'image/png', '.svg': 'image/svg+xml', '.woff2': 'font/woff2' }
let token = ''
const browserToken = randomBytes(32).toString('base64url')

function websocketFrame(data, opcode = 1) {
  const body = Buffer.from(data)
  const header = body.length < 126 ? Buffer.from([0x80 | opcode, body.length]) : Buffer.from([0x80 | opcode, 126, body.length >> 8, body.length & 255])
  return Buffer.concat([header, body])
}

function browserTerminal(req, socket, head) {
  const accept = createHash('sha1').update(`${req.headers['sec-websocket-key']}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`).digest('base64')
  socket.write(`HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`)
  const shell = spawn('/usr/bin/script', ['-qfec', process.env.SHELL || '/bin/bash', '/dev/null'], { cwd: process.env.HOME, env: { ...process.env, TERM: 'xterm-256color' }, stdio: ['pipe', 'pipe', 'pipe'] })
  const send = data => !socket.destroyed && socket.write(websocketFrame(data))
  shell.stdout.on('data', send)
  shell.stderr.on('data', send)
  shell.on('exit', () => socket.end())
  const heartbeat = setInterval(() => !socket.destroyed && socket.write(websocketFrame('', 9)), 30_000)
  let pending = head
  socket.on('data', chunk => {
    pending = Buffer.concat([pending, chunk])
    while (pending.length >= 2) {
      const masked = Boolean(pending[1] & 0x80)
      let length = pending[1] & 0x7f
      let offset = 2
      if (length === 126) { if (pending.length < 4) return; length = pending.readUInt16BE(2); offset = 4 }
      if (length === 127 || pending.length < offset + (masked ? 4 : 0) + length) return
      const mask = masked ? pending.subarray(offset, offset += 4) : null
      const data = Buffer.from(pending.subarray(offset, offset + length))
      if (mask) for (let i = 0; i < data.length; i++) data[i] ^= mask[i % 4]
      const opcode = pending[0] & 0x0f
      if (opcode === 8) { shell.kill(); return }
      if (opcode === 9) socket.write(websocketFrame(data, 10))
      else if (opcode === 1 || opcode === 2) shell.stdin.write(data)
      pending = pending.subarray(offset + length)
    }
  })
  socket.on('close', () => { clearInterval(heartbeat); shell.kill() })
}

function refreshToken() {
  return new Promise((resolve, reject) => {
    httpGet(backend, res => {
      let html = ''
      res.setEncoding('utf8')
      res.on('data', chunk => { html += chunk })
      res.on('end', () => {
        token = html.match(/__HERMES_SESSION_TOKEN__="([^"]+)"/)?.[1] ?? ''
        token ? resolve() : reject(new Error('dashboard session token unavailable'))
      })
    }).on('error', reject)
  })
}

function proxy(req, res) {
  const headers = { ...req.headers, host: backend.host, 'x-hermes-session-token': token }
  const upstream = httpRequest({ hostname: backend.hostname, port: backend.port, method: req.method, path: req.url, headers }, r => {
    res.writeHead(r.statusCode ?? 502, r.headers)
    r.pipe(res)
  })
  upstream.on('error', error => { res.writeHead(502); res.end(error.message) })
  req.pipe(upstream)
}

const server = createServer((req, res) => {
  if (req.url?.startsWith('/api/')) return proxy(req, res)
  const pathname = decodeURIComponent((req.url ?? '/').split('?')[0])
  let file = normalize(join(root, pathname === '/' ? 'index.html' : pathname))
  if (!file.startsWith(root)) { res.writeHead(403); return res.end() }
  try { if (statSync(file).isDirectory()) file = join(file, 'index.html') } catch { file = join(root, 'index.html') }
  res.setHeader('Content-Type', types[extname(file)] ?? 'application/octet-stream')
  if (file.endsWith('/index.html')) res.setHeader('Set-Cookie', `hermes_ws=${browserToken}; Path=/api; Secure; HttpOnly; SameSite=Strict`)
  createReadStream(file).on('error', () => { res.writeHead(404); res.end() }).pipe(res)
})

server.on('upgrade', (req, socket, head) => {
  // Browsers and Cloudflare routinely reset sockets during navigation. A raw
  // net.Socket emits 'error'; without a listener Node terminates the server.
  socket.on('error', () => socket.destroy())
  if (!req.headers.cookie?.split(';').some(cookie => cookie.trim() === `hermes_ws=${browserToken}`)) {
    socket.end('HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n')
    return
  }
  if (req.url?.startsWith('/api/terminal')) return browserTerminal(req, socket, head)
  const url = new URL(req.url ?? '/api/ws', backend)
  url.searchParams.set('token', token)
  const headers = { ...req.headers, host: backend.host, origin: backend.origin, connection: 'Upgrade', upgrade: 'websocket' }
  const upstream = httpRequest({ hostname: backend.hostname, port: backend.port, method: req.method, path: `${url.pathname}${url.search}`, headers })
  upstream.on('upgrade', (res, peer, peerHead) => {
    peer.on('error', () => socket.destroy())
    socket.write(`HTTP/1.1 101 Switching Protocols\r\n${Object.entries(res.headers).map(([k, v]) => `${k}: ${v}`).join('\r\n')}\r\n\r\n`)
    if (peerHead.length) socket.write(peerHead)
    if (head.length) peer.write(head)
    peer.pipe(socket).pipe(peer)
  })
  upstream.on('response', res => { socket.end(`HTTP/1.1 ${res.statusCode} Unauthorized\r\n\r\n`) })
  upstream.on('error', () => socket.destroy())
  upstream.end()
})

await refreshToken()
server.listen(8788, '127.0.0.1')
