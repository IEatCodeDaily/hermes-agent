import { buildHermesWebSocketUrl } from '@hermes/shared'

type DesktopBridge = Window['hermesDesktop']
type ApiRequest = Parameters<DesktopBridge['api']>[0]
type Connection = Awaited<ReturnType<DesktopBridge['getConnection']>>
type NotificationPayload = Parameters<DesktopBridge['notify']>[0]

const noop = () => undefined
const off = () => noop
const ok = async () => ({ ok: true })
const home = '/home/rpw'

function apiUrl(path: string, profile?: string): string {
  const url = new URL(path, window.location.origin)
  if (profile && !url.searchParams.has('profile')) {
    url.searchParams.set('profile', profile)
  }
  return url.toString()
}

async function api<T>(request: ApiRequest): Promise<T> {
  const headers = new Headers()
  let body: BodyInit | undefined

  if (request.upload) {
    const form = new FormData()
    form.append('file', new Blob([request.upload.bytes], { type: request.upload.contentType }), request.upload.filename)
    body = form
  } else if (request.body !== undefined) {
    headers.set('Content-Type', 'application/json')
    body = JSON.stringify(request.body)
  }

  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), request.timeoutMs ?? 30_000)

  try {
    const response = await fetch(apiUrl(request.path, request.profile ?? undefined), {
      body,
      credentials: 'same-origin',
      headers,
      method: request.method ?? (body ? 'POST' : 'GET'),
      signal: controller.signal
    })
    const text = await response.text()
    if (!response.ok) {
      throw new Error(`${response.status}: ${text || response.statusText}`)
    }
    return (text ? JSON.parse(text) : {}) as T
  } finally {
    window.clearTimeout(timeout)
  }
}

const baseUrl = window.location.origin
const connection: Connection = {
  baseUrl,
  isFullscreen: false,
  mode: 'remote',
  authMode: 'oauth',
  nativeOverlayWidth: 0,
  token: '',
  wsUrl: buildHermesWebSocketUrl({ path: '/api/ws' }),
  logs: [],
  windowButtonPosition: null
}
const terminals = new Map<string, { data: Set<(value: string) => void>; exit: Set<(value: { exitCode: number; signal?: number }) => void>; socket: WebSocket }>()
const terminal = {
  cwd: async () => home,
  dispose: async (id: string) => (terminals.get(id)?.socket.close(), terminals.delete(id)),
  onData: (id: string, callback: (value: string) => void) => (terminals.get(id)?.data.add(callback), () => terminals.get(id)?.data.delete(callback)),
  onExit: (id: string, callback: (value: { exitCode: number; signal?: number }) => void) => (terminals.get(id)?.exit.add(callback), () => terminals.get(id)?.exit.delete(callback)),
  resize: async () => true,
  start: async () => {
    const id = crypto.randomUUID()
    const socket = new WebSocket(`${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/terminal`)
    const session = { data: new Set<(value: string) => void>(), exit: new Set<(value: { exitCode: number; signal?: number }) => void>(), socket }
    terminals.set(id, session)
    socket.onmessage = event => session.data.forEach(callback => callback(String(event.data)))
    socket.onclose = () => session.exit.forEach(callback => callback({ exitCode: 0 }))
    await new Promise<void>((resolve, reject) => { socket.onopen = () => resolve(); socket.onerror = () => reject(new Error('Terminal connection failed')) })
    return { id, shell: 'shell' }
  },
  write: async (id: string, data: string) => (terminals.get(id)?.socket.send(data), true)
}

const unsupported = new Proxy(noop, {
  apply: () => Promise.resolve(undefined),
  get: () => unsupported
})

const bridge = new Proxy(
  {
    api,
    getConnection: async () => connection,
    getGatewayWsUrl: async () => ({ ok: true, wsUrl: connection.wsUrl }),
    revalidateConnection: async () => ({ ok: true, rebuilt: false }),
    touchBackend: ok,
    claimAmbientCue: async () => true,
    getOnBattery: async () => false,
    onBackendExit: off,
    onBootProgress: off,
    onBootstrapEvent: off,
    onConnectionApplied: off,
    onPowerResume: off,
    onPreviewFileChanged: off,
    onWindowStateChanged: off,
    openExternal: async (url: string) => void window.open(url, '_blank', 'noopener,noreferrer'),
    openPreviewInBrowser: async (url: string) => void window.open(url, '_blank', 'noopener,noreferrer'),
    writeClipboard: async (text: string) => {
      await navigator.clipboard.writeText(text)
      return true
    },
    readClipboard: () => navigator.clipboard.readText(),
    getPathForFile: () => '',
    notify: async (payload: NotificationPayload) => {
      if (Notification.permission === 'granted') {
        new Notification(payload.title ?? 'Hermes', { body: payload.body })
      }
      return true
    },
    requestMicrophoneAccess: async () => true,
    profile: { get: async () => ({ profile: null }), set: async (profile: string | null) => ({ profile }) },
    settings: { getDefaultProjectDir: async () => ({ dir: home }) },
    terminal,
    sanitizeWorkspaceCwd: async (cwd?: string) => ({ cwd: cwd || home, sanitized: false }),
    setActiveWork: noop,
    setKeepAwake: noop,
    setNativeTheme: noop,
    setPreviewShortcutActive: noop,
    setTitleBarTheme: noop,
    setTranslucency: noop,
    signalDeepLinkReady: ok
  },
  {
    get: (target, key) => {
      const value = Reflect.get(target, key)
      if (value !== undefined) {
        return value
      }
      return typeof key === 'string' && key.startsWith('on') ? off : unsupported
    }
  }
)

if (!window.hermesDesktop) {
  window.hermesDesktop = bridge as unknown as DesktopBridge
}

export { apiUrl }
