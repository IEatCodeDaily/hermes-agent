#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'

const [entry, adapter, vite] = await Promise.all([
  readFile(new URL('../src/main.tsx', import.meta.url), 'utf8'),
  readFile(new URL('../src/browser-desktop.ts', import.meta.url), 'utf8'),
  readFile(new URL('../vite.config.ts', import.meta.url), 'utf8')
])

assert.match(entry, /import ['"]\.\/browser-desktop['"]/)
assert.match(adapter, /fetch\(apiUrl\(request\.path, request\.profile \?\? undefined\)/)
assert.match(adapter, /buildHermesWebSocketUrl\(\{ path: ['"]\/api\/ws['"] \}\)/)
assert.match(adapter, /if \(!window\.hermesDesktop\) \{\s*window\.hermesDesktop = bridge/)
assert.match(vite, /['"]\/api['"]:\s*\{[\s\S]*?ws: true/)
console.log('browser adapter check: ok')
