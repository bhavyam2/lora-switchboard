'use client'

import { useState, useEffect, useCallback } from 'react'

const API = 'http://localhost:8000'
const CACHE_CAPACITY = 8

interface Result {
  output: string
  adapter_id: string
  latency_ms: number
  batch?: boolean
}

// ── tiny components ────────────────────────────────────────────────────────

function Dot({ active }: { active: boolean }) {
  return (
    <span
      className={`inline-block w-2 h-2 rounded-full flex-shrink-0 mt-1 ${
        active ? 'bg-green-400 shadow-[0_0_6px_#4ade80]' : 'bg-[#30363d]'
      }`}
    />
  )
}

function Badge({ children, color = 'gray' }: { children: React.ReactNode; color?: string }) {
  const colors: Record<string, string> = {
    gray:   'bg-[#21262d] text-[#8b949e] border-[#30363d]',
    green:  'bg-[#0f2d1a] text-green-400  border-[#1a4731]',
    blue:   'bg-[#0c1f3d] text-blue-400   border-[#1a3560]',
    yellow: 'bg-[#2d2000] text-yellow-400 border-[#4a3800]',
  }
  return (
    <span className={`text-xs px-2 py-0.5 rounded border font-mono ${colors[color]}`}>
      {children}
    </span>
  )
}

// ── panels ──────────────────────────────────────────────────────────────────

function RequestPanel({
  prompt, setPrompt,
  adapterId, setAdapterId,
  cachedAdapters,
  onSend, onBatch,
  loading,
  newAdapterId, setNewAdapterId,
  onRegister, registering,
}: {
  prompt: string; setPrompt: (v: string) => void
  adapterId: string; setAdapterId: (v: string) => void
  cachedAdapters: string[]
  onSend: () => void; onBatch: () => void
  loading: boolean
  newAdapterId: string; setNewAdapterId: (v: string) => void
  onRegister: () => void; registering: boolean
}) {
  return (
    <div className="flex flex-col gap-4 p-5 border-r border-[#30363d] overflow-y-auto">
      <p className="text-xs text-[#8b949e] uppercase tracking-widest">Request</p>

      <div className="flex flex-col gap-1">
        <label className="text-xs text-[#8b949e]">Prompt</label>
        <textarea
          className="bg-[#010409] border border-[#30363d] rounded p-3 text-sm text-[#e6edf3]
                     resize-none focus:outline-none focus:border-[#58a6ff] transition-colors h-36"
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          placeholder="Enter your prompt..."
        />
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-xs text-[#8b949e]">Adapter</label>
        {cachedAdapters.length === 0 ? (
          <p className="text-xs text-[#8b949e] italic">No adapters loaded yet — register one below</p>
        ) : (
          <select
            className="bg-[#010409] border border-[#30363d] rounded p-2 text-sm text-[#e6edf3]
                       focus:outline-none focus:border-[#58a6ff] transition-colors"
            value={adapterId}
            onChange={e => setAdapterId(e.target.value)}
          >
            {cachedAdapters.map(id => (
              <option key={id} value={id}>{id}</option>
            ))}
          </select>
        )}
      </div>

      <div className="flex gap-2">
        <button
          onClick={onSend}
          disabled={loading || !adapterId || !prompt}
          className="flex-1 py-2 px-4 rounded text-sm font-medium bg-[#238636] hover:bg-[#2ea043]
                     disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? 'Running…' : 'Send'}
        </button>
        <button
          onClick={onBatch}
          disabled={loading || cachedAdapters.length < 2 || !prompt}
          className="flex-1 py-2 px-4 rounded text-sm font-medium bg-[#1f6feb] hover:bg-[#388bfd]
                     disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          title="Sends prompt to all cached adapters in one batched forward pass"
        >
          Batch
        </button>
      </div>

      <div className="border-t border-[#30363d] pt-4 flex flex-col gap-2">
        <p className="text-xs text-[#8b949e] uppercase tracking-widest">Register Adapter</p>
        <div className="flex gap-2">
          <input
            className="flex-1 bg-[#010409] border border-[#30363d] rounded p-2 text-sm text-[#e6edf3]
                       focus:outline-none focus:border-[#58a6ff] transition-colors"
            placeholder="adapter-name"
            value={newAdapterId}
            onChange={e => setNewAdapterId(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && onRegister()}
          />
          <button
            onClick={onRegister}
            disabled={registering || !newAdapterId.trim()}
            className="px-3 py-2 rounded text-sm bg-[#21262d] hover:bg-[#30363d] border border-[#30363d]
                       disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {registering ? '…' : 'Add'}
          </button>
        </div>
        <p className="text-[10px] text-[#8b949e]">
          Registers a randomly-initialised adapter — simulates loading a real checkpoint.
        </p>
      </div>
    </div>
  )
}

function OutputPanel({ results }: { results: Result[] }) {
  if (results.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-[#8b949e] gap-3 p-8">
        <div className="text-4xl opacity-20">⚡</div>
        <p className="text-sm">Send a request to see output here</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4 p-5 overflow-y-auto">
      <p className="text-xs text-[#8b949e] uppercase tracking-widest">Output</p>
      {results.map((r, i) => (
        <div
          key={i}
          className={`rounded border ${
            i === 0 ? 'border-[#58a6ff] bg-[#0c1a2e]' : 'border-[#30363d] bg-[#161b22]'
          } p-4 flex flex-col gap-3`}
        >
          <div className="flex items-center gap-2 flex-wrap">
            <Badge color={r.batch ? 'blue' : 'green'}>
              {r.batch ? 'batch' : 'single'}
            </Badge>
            <Badge color="gray">{r.adapter_id}</Badge>
            <Badge color="yellow">{r.latency_ms.toFixed(0)}ms</Badge>
          </div>
          <pre className="text-sm text-[#e6edf3] whitespace-pre-wrap leading-relaxed">
            {r.output}
          </pre>
        </div>
      ))}
    </div>
  )
}

function CachePanel({
  cachedAdapters,
  serverOnline,
}: {
  cachedAdapters: string[]
  serverOnline: boolean
}) {
  const filledSlots = cachedAdapters.length
  const pct = Math.round((filledSlots / CACHE_CAPACITY) * 100)

  return (
    <div className="flex flex-col gap-4 p-5 border-l border-[#30363d] overflow-y-auto">
      <div className="flex items-center justify-between">
        <p className="text-xs text-[#8b949e] uppercase tracking-widest">GPU Cache</p>
        <span className={`text-[10px] ${serverOnline ? 'text-green-400' : 'text-red-400'}`}>
          {serverOnline ? '● live' : '○ offline'}
        </span>
      </div>

      <div className="flex flex-col gap-1">
        <div className="flex justify-between text-xs text-[#8b949e]">
          <span>{filledSlots} / {CACHE_CAPACITY} slots</span>
          <span>{pct}%</span>
        </div>
        <div className="h-1.5 rounded-full bg-[#21262d] overflow-hidden">
          <div
            className="h-full rounded-full bg-green-400 transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      <div className="flex flex-col gap-2">
        {cachedAdapters.length === 0 ? (
          <p className="text-xs text-[#8b949e] italic">Empty — register an adapter to get started</p>
        ) : (
          cachedAdapters.map(id => (
            <div key={id} className="flex items-start gap-2 text-sm">
              <Dot active />
              <span className="text-[#e6edf3] break-all">{id}</span>
            </div>
          ))
        )}
      </div>

      {cachedAdapters.length > 0 && (
        <div className="border-t border-[#30363d] pt-3 text-[10px] text-[#8b949e] leading-relaxed">
          Hot adapters are resident in device memory. Cache misses trigger a CPU→GPU transfer and evict the LRU entry.
        </div>
      )}
    </div>
  )
}

// ── root ────────────────────────────────────────────────────────────────────

export default function Home() {
  const [prompt, setPrompt]               = useState('Analyze the following system metrics:')
  const [adapterId, setAdapterId]         = useState('')
  const [newAdapterId, setNewAdapterId]   = useState('')
  const [cachedAdapters, setCachedAdapters] = useState<string[]>([])
  const [results, setResults]             = useState<Result[]>([])
  const [loading, setLoading]             = useState(false)
  const [registering, setRegistering]     = useState(false)
  const [error, setError]                 = useState<string | null>(null)
  const [serverOnline, setServerOnline]   = useState(false)

  const fetchCache = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/v1/adapters/cached`)
      const data = await res.json()
      setCachedAdapters(data.cached ?? [])
      setServerOnline(true)
      if (data.cached?.length > 0 && !adapterId) {
        setAdapterId(data.cached[0])
      }
    } catch {
      setServerOnline(false)
    }
  }, [adapterId])

  useEffect(() => {
    fetchCache()
    const id = setInterval(fetchCache, 2000)
    return () => clearInterval(id)
  }, [fetchCache])

  const registerAdapter = async () => {
    if (!newAdapterId.trim()) return
    setRegistering(true)
    try {
      await fetch(`${API}/api/v1/adapters/register-random?adapter_id=${encodeURIComponent(newAdapterId)}`, {
        method: 'POST',
      })
      if (!adapterId) setAdapterId(newAdapterId)
      setNewAdapterId('')
      await fetchCache()
    } finally {
      setRegistering(false)
    }
  }

  const sendSingle = async () => {
    if (!prompt || !adapterId) return
    setLoading(true)
    setError(null)
    const t0 = performance.now()
    try {
      const res = await fetch(`${API}/api/v1/infer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, adapter_id: adapterId }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail ?? 'Request failed')
      setResults(prev => [
        { output: data.output, adapter_id: adapterId, latency_ms: performance.now() - t0 },
        ...prev.slice(0, 4),
      ])
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unknown error')
    } finally {
      setLoading(false)
      fetchCache()
    }
  }

  const sendBatch = async () => {
    if (!prompt || cachedAdapters.length < 2) return
    setLoading(true)
    setError(null)
    const t0 = performance.now()
    const adapters = cachedAdapters.slice(0, 4)
    try {
      const res = await fetch(`${API}/api/v1/batch-infer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          requests: adapters.map(id => ({ prompt, adapter_id: id })),
          max_new_tokens: 50,
        }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail ?? 'Batch failed')
      const perRequest = (performance.now() - t0) / adapters.length
      setResults(prev => [
        ...data.outputs.map((o: { output: string; adapter_id: string }) => ({
          output: o.output,
          adapter_id: o.adapter_id,
          latency_ms: perRequest,
          batch: true,
        })),
        ...prev.slice(0, 2),
      ])
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unknown error')
    } finally {
      setLoading(false)
      fetchCache()
    }
  }

  return (
    <main className="min-h-screen bg-[#0d1117] text-[#e6edf3] flex flex-col">
      {/* Header */}
      <header className="flex-shrink-0 border-b border-[#30363d] px-6 py-3 flex items-center gap-4">
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${serverOnline ? 'bg-green-400 animate-pulse' : 'bg-red-500'}`} />
          <span className="font-bold tracking-tight">lora-switchboard</span>
        </div>
        <span className="text-[#8b949e] text-sm hidden sm:block">
          multi-tenant LoRA inference engine
        </span>
        {error && (
          <span className="ml-auto text-xs text-red-400 truncate max-w-sm">{error}</span>
        )}
      </header>

      {/* Three-column layout */}
      <div className="flex-1 grid grid-cols-[300px_1fr_240px] min-h-0">
        <RequestPanel
          prompt={prompt} setPrompt={setPrompt}
          adapterId={adapterId} setAdapterId={setAdapterId}
          cachedAdapters={cachedAdapters}
          onSend={sendSingle} onBatch={sendBatch}
          loading={loading}
          newAdapterId={newAdapterId} setNewAdapterId={setNewAdapterId}
          onRegister={registerAdapter} registering={registering}
        />
        <OutputPanel results={results} />
        <CachePanel cachedAdapters={cachedAdapters} serverOnline={serverOnline} />
      </div>
    </main>
  )
}
