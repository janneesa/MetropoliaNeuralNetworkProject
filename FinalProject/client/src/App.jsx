import { useEffect, useMemo, useRef, useState } from 'react'

import './App.css'
import './animations.css'
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
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`
}

function deriveSettingsEndpoint(chatEndpoint) {
  if (!chatEndpoint || chatEndpoint === '/api/chat') return '/api/settings'

  try {
    const url = new URL(chatEndpoint, window.location.origin)
    url.pathname = url.pathname.replace(/\/api\/chat$/, '/api/settings')
    return chatEndpoint.startsWith('http') ? url.toString() : `${url.pathname}${url.search}`
  } catch {
    return '/api/settings'
  }
}

function normalizeSettings(data) {
  const temperature = Number(data?.temperature ?? 0.5)
  const maxTokens = Number(data?.max_tokens ?? data?.response_length ?? 80)
  const stream = typeof data?.stream === 'boolean' ? data.stream : true

  return {
    temperature: Number.isFinite(temperature) ? temperature : 0.5,
    maxTokens: Number.isFinite(maxTokens) && maxTokens > 0 ? maxTokens : 80,
    stream,
  }
}

function sameSettings(a, b) {
  if (!a || !b) return false
  return (
    a.temperature === b.temperature &&
    a.maxTokens === b.maxTokens &&
    a.stream === b.stream
  )
}

function extractModelLabel(data) {
  const label = data?.model?.label ?? data?.settings?.model?.label
  return typeof label === 'string' && label.trim() ? label : 'Emma'
}

async function readResponseBody(res) {
  const contentType = res.headers.get('content-type') || ''
  return contentType.includes('application/json') ? res.json() : res.text()
}

export default function App() {
  const endpoint = useMemo(
    () => import.meta.env.VITE_CHAT_ENDPOINT || '/api/chat',
    [],
  )

  const settingsEndpoint = useMemo(
    () => import.meta.env.VITE_SETTINGS_ENDPOINT || deriveSettingsEndpoint(endpoint),
    [endpoint],
  )

  const [messages, setMessages] = useState([])
  const [prompt, setPrompt] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [isStreamingResponse, setIsStreamingResponse] = useState(false)
  const [error, setError] = useState('')
  const bottomRef = useRef(null)
  const [paramsOpen, setParamsOpen] = useState(false)
  const [assistantName, setAssistantName] = useState('Emma')
  const [modelParams, setModelParams] = useState({ temperature: 0.5, maxTokens: 80, stream: true })
  const [settingsLoaded, setSettingsLoaded] = useState(false)
  const lastSyncedSettingsRef = useRef({ temperature: 0.5, maxTokens: 80, stream: true })

  function syncServerState(data) {
    if (typeof data !== 'object' || data === null) {
      return
    }

    const hasSettingsPayload = (
      data.settings != null ||
      'temperature' in data ||
      'max_tokens' in data ||
      'response_length' in data ||
      'stream' in data
    )

    if (hasSettingsPayload) {
      const nextSettings = normalizeSettings(data.settings ?? data)
      lastSyncedSettingsRef.current = nextSettings
      setModelParams((prev) => (sameSettings(prev, nextSettings) ? prev : nextSettings))
    }

    setAssistantName(extractModelLabel(data))
  }

  useEffect(() => {
    let isDisposed = false

    async function loadSettings() {
      try {
        const res = await fetch(settingsEndpoint)
        const data = await readResponseBody(res)

        if (!res.ok) {
          throw new Error(data?.error || data?.message || `Request failed (${res.status})`)
        }

        const nextSettings = normalizeSettings(data)
        if (isDisposed) return

        lastSyncedSettingsRef.current = nextSettings
        setModelParams(nextSettings)
        setAssistantName(extractModelLabel(data))
        setError('')
      } catch (e) {
        if (isDisposed) return
        setError(e instanceof Error ? e.message : String(e))
      } finally {
        if (!isDisposed) {
          setSettingsLoaded(true)
        }
      }
    }

    void loadSettings()

    return () => {
      isDisposed = true
    }
  }, [settingsEndpoint])

  useEffect(() => {
    if (!settingsLoaded || sameSettings(modelParams, lastSyncedSettingsRef.current)) {
      return undefined
    }

    const abortController = new AbortController()
    const timeoutId = window.setTimeout(async () => {
      try {
        const res = await fetch(settingsEndpoint, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            temperature: modelParams.temperature,
            max_tokens: modelParams.maxTokens,
            stream: modelParams.stream,
          }),
          signal: abortController.signal,
        })

        const data = await readResponseBody(res)
        if (!res.ok) {
          throw new Error(data?.error || data?.message || `Request failed (${res.status})`)
        }

        syncServerState(data)
        setError('')
      } catch (e) {
        if (e instanceof Error && e.name === 'AbortError') {
          return
        }
        setError(e instanceof Error ? e.message : String(e))
      }
    }, 150)

    return () => {
      abortController.abort()
      window.clearTimeout(timeoutId)
    }
  }, [modelParams, settingsEndpoint, settingsLoaded])

  async function sendStandardPrompt(body) {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })

    const data = await readResponseBody(res)
    if (!res.ok) {
      const msg =
        (typeof data === 'string' ? data : data?.error || data?.message) ||
        `Request failed (${res.status})`
      throw new Error(msg)
    }

    syncServerState(data)

    if (data?.reset) {
      setMessages([])
      return
    }

    const answer = extractAnswer(data) || (typeof data === 'string' ? data : '')
    const assistantMessage = {
      id: createId(),
      role: extractModelLabel(data),
      content: answer || '(No answer returned)',
      timestamp: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, assistantMessage])
  }

  async function sendStreamingPrompt(body) {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })

    if (!res.ok) {
      const data = await readResponseBody(res)
      const msg =
        (typeof data === 'string' ? data : data?.error || data?.message) ||
        `Request failed (${res.status})`
      throw new Error(msg)
    }

    const reader = res.body?.getReader()
    if (!reader) {
      throw new Error('Streaming response is not available in this browser.')
    }

    const decoder = new TextDecoder()
    const assistantMessageId = createId()
    let buffer = ''
    let streamedText = ''
    let hasAssistantMessage = false
    let currentAssistantRole = assistantName

    const setAssistantMessage = (content) => {
      if (!hasAssistantMessage) {
        hasAssistantMessage = true
        setIsStreamingResponse(true)
        setMessages((prev) => [
          ...prev,
          {
            id: assistantMessageId,
            role: currentAssistantRole,
            content,
            timestamp: new Date().toISOString(),
          },
        ])
        return
      }

      setMessages((prev) => prev.map((message) => (
        message.id === assistantMessageId
          ? { ...message, role: currentAssistantRole, content }
          : message
      )))
    }

    const processEvent = (event) => {
      if (typeof event !== 'object' || event === null) {
        return
      }

      syncServerState(event)
      currentAssistantRole = extractModelLabel(event)
      setAssistantName(currentAssistantRole)

      if (hasAssistantMessage) {
        setMessages((prev) => prev.map((message) => (
          message.id === assistantMessageId
            ? { ...message, role: currentAssistantRole }
            : message
        )))
      }

      if (event.type === 'start') {
        return
      }

      if (event.type === 'chunk') {
        const text = typeof event.text === 'string' ? event.text : ''
        if (!text) {
          return
        }
        streamedText += text
        setAssistantMessage(streamedText)
        return
      }

      if (event.type === 'reset') {
        setMessages([])
        return
      }

      if (event.type === 'done') {
        const finalText = typeof event.response === 'string' ? event.response : streamedText
        if (event.reset) {
          setMessages([])
          return
        }
        setAssistantMessage(finalText || '(No answer returned)')
        return
      }

      if (event.type === 'error') {
        throw new Error(event.error || 'Streaming failed.')
      }

      const fallbackText = extractAnswer(event)
      if (fallbackText) {
        streamedText = fallbackText
        setAssistantMessage(fallbackText)
      }
    }

    while (true) {
      const { value, done } = await reader.read()
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done })

      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        const trimmedLine = line.trim()
        if (!trimmedLine) {
          continue
        }
        processEvent(JSON.parse(trimmedLine))
      }

      if (done) {
        break
      }
    }

    const trailingLine = buffer.trim()
    if (trailingLine) {
      processEvent(JSON.parse(trailingLine))
    }
  }

  async function sendPrompt(userPrompt) {
    const trimmed = userPrompt.trim()
    if (!trimmed || isSending) return

    setError('')
    setIsSending(true)
    setIsStreamingResponse(false)

    const userMessage = { id: createId(), role: 'user', content: trimmed, timestamp: new Date().toISOString() }
    const requestSettings = { ...modelParams }

    setMessages((prev) => [...prev, userMessage])
    setPrompt('')

    const requestBody = {
      prompt: trimmed,
      temperature: requestSettings.temperature,
      max_tokens: requestSettings.maxTokens,
      stream: requestSettings.stream,
    }

    try {
      if (requestSettings.stream) {
        await sendStreamingPrompt(requestBody)
      } else {
        await sendStandardPrompt(requestBody)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setMessages((prev) => [
        ...prev,
        {
          id: createId(),
          role: assistantName,
          content: 'Sorry — something went wrong sending that message.',
          timestamp: new Date().toISOString(),
        },
      ])
    } finally {
      setIsStreamingResponse(false)
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
          <MessageList
            messages={messages}
            isSending={isSending}
            isStreamingResponse={isStreamingResponse}
            bottomRef={bottomRef}
            assistantName={assistantName}
          />
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
