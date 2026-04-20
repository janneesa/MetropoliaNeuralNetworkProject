import { useEffect, useMemo, useRef, useState } from 'react'

import './App.css'
import './animations.css'
import ChatHeader from './components/ChatHeader'
import MessageList from './components/MessageList'
import Composer from './components/Composer'
import ModelParamsPanel from './components/ModelParamsPanel'
import './components/ModelParamsPanel.css'

function extractAnswer(data) {
  if (data == null) return ''
  if (typeof data === 'string') return data

  if (typeof data.response === 'string') return data.response
  if (typeof data.answer === 'string') return data.answer
  if (typeof data.message === 'string') return data.message
  if (typeof data.text === 'string') return data.text

  const maybeChoice = data?.choices?.[0]
  const maybeOpenAIMessage = maybeChoice?.message?.content
  if (typeof maybeOpenAIMessage === 'string') return maybeOpenAIMessage
  const maybeOpenAIText = maybeChoice?.text
  if (typeof maybeOpenAIText === 'string') return maybeOpenAIText

  return ''
}

function createId() {
  return (globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`)
}

function normalizeBaseUrl(url) {
  return url.endsWith('/') ? url.slice(0, -1) : url
}

function looksLikeBaseUrl(url) {
  try {
    const u = new URL(url)
    return u.pathname === '' || u.pathname === '/'
  } catch {
    return false
  }
}

function App() {
  const endpoint = useMemo(
    () => import.meta.env.VITE_CHAT_ENDPOINT || '/api/chat',
    [],
  )

  const provider = useMemo(
    () => (import.meta.env.VITE_CHAT_PROVIDER || 'generic').toLowerCase(),
    [],
  )
  const ollamaModel = useMemo(() => import.meta.env.VITE_OLLAMA_MODEL || '', [])

  const [messages, setMessages] = useState([])
  const [prompt, setPrompt] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [error, setError] = useState('')
  const bottomRef = useRef(null)
  const [paramsOpen, setParamsOpen] = useState(false)
  const [modelParams, setModelParams] = useState({ temperature: 1, maxTokens: 1024 })

  async function sendPrompt(userPrompt) {
    const trimmed = userPrompt.trim()
    if (!trimmed || isSending) return

    setError('')
    setIsSending(true)

    const userMessage = { id: createId(), role: 'user', content: trimmed, timestamp: new Date().toISOString() }
    const nextMessages = [...messages, userMessage]
    setMessages(nextMessages)
    setPrompt('')

    try {
      let url = endpoint
      let body = { prompt: trimmed, temperature: modelParams.temperature, max_tokens: modelParams.maxTokens }

      if (provider === 'ollama') {
        if (!ollamaModel) {
          throw new Error(
            'Missing VITE_OLLAMA_MODEL. Set it in .env (e.g. llama3.2, mistral, etc.)',
          )
        }

        if (looksLikeBaseUrl(endpoint)) {
          url = `${normalizeBaseUrl(endpoint)}/api/generate`
        }

        if (url.endsWith('/api/chat')) {
          body = {
            model: ollamaModel,
            messages: nextMessages.map((m) => ({ role: m.role, content: m.content })),
            stream: false,
            temperature: modelParams.temperature,
            max_tokens: modelParams.maxTokens,
          }
        } else {
          body = {
            model: ollamaModel,
            prompt: trimmed,
            stream: false,
            temperature: modelParams.temperature,
            max_tokens: modelParams.maxTokens,
          }
        }
      }

      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      const contentType = res.headers.get('content-type') || ''
      const data = contentType.includes('application/json')
        ? await res.json()
        : await res.text()

      if (!res.ok) {
        const msg =
          (typeof data === 'string' ? data : data?.error || data?.message) ||
          `Request failed (${res.status})`
        throw new Error(msg)
      }

      const answer = extractAnswer(data) || (typeof data === 'string' ? data : '')
      const assistantMessage = {
        id: createId(),
        role: 'Willow',
        content: answer || '(No answer returned)',
        timestamp: new Date().toISOString(),
      }
      setMessages((prev) => [...prev, assistantMessage])
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setMessages((prev) => [
        ...prev,
        {
          id: createId(),
          role: 'Willow',
          content: 'Sorry — something went wrong sending that message.',
          timestamp: new Date().toISOString(),
        },
      ])
    } finally {
      setIsSending(false)
    }
  }

  function onSubmit(e) {
    e.preventDefault()
    void sendPrompt(prompt)
  }

  function onComposerKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void sendPrompt(prompt)
    }
  }

  return (
    <div className="app">
      <main className="chat" aria-label="Chat">
        <div className="chatbox-area">
          <MessageList messages={messages} isSending={isSending} bottomRef={bottomRef} />
        </div>
        <div className="composer-area">
          <Composer
            prompt={prompt}
            setPrompt={setPrompt}
            onSubmit={onSubmit}
            onComposerKeyDown={onComposerKeyDown}
            isSending={isSending}
            error={error}
            onOpenParams={() => setParamsOpen(true)}
          />
        </div>
      </main>
      <ModelParamsPanel
        open={paramsOpen}
        onClose={() => setParamsOpen(false)}
        params={modelParams}
        setParams={setModelParams}
      />
    </div>
  )
}

export default App
