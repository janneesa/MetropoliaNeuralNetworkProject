import { useEffect, useLayoutEffect, useRef } from 'react';

function formatTime(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export default function MessageList({ messages, isSending, bottomRef }) {
  const messagesRef = useRef();

  const scrollToBottom = () => {
    const el = messagesRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  };

  useLayoutEffect(() => {
    // Run before paint so the newest message is visible immediately.
    scrollToBottom();
  }, [messages, isSending]);

  useEffect(() => {
    // Run again after layout/animations settle.
    let raf2 = 0;
    const raf1 = requestAnimationFrame(() => {
      scrollToBottom();
      raf2 = requestAnimationFrame(() => {
        scrollToBottom();
      });
    });

    return () => {
      cancelAnimationFrame(raf1);
      if (raf2) cancelAnimationFrame(raf2);
    };
  }, [messages, isSending]);

  return (
    <div className="messages" ref={messagesRef} role="log" aria-live="polite">
      {messages.length === 0 ? (
        <div className="emptyState">
          <p className="emptyTitle">Start a conversation</p>
          <p className="emptySub">Ask me anything!</p>
        </div>
      ) : null}

      {messages.map((m) => (
        <div
          key={m.id}
          className={`messageRow ${m.role === 'user' ? 'isUser' : 'isAssistant'}`}
        >
          <div className="bubble bubble-appear">
            <div className="bubbleRole">{m.role}
              <span className="bubbleTimestamp">{formatTime(m.timestamp)}</span>
            </div>
            <div className="bubbleContent">{m.content}</div>
          </div>
        </div>
      ))}

      {isSending ? (
        <div className="messageRow isAssistant">
          <div className="bubble">
            <div className="bubbleRole">Willow</div>
            <div className="bubbleContent">
              Thinking<span className="thinkingDots" aria-live="polite">
                <span>.</span><span>.</span><span>.</span>
              </span>
            </div>
          </div>
        </div>
      ) : null}

      <div ref={bottomRef} style={{ height: 1 }} />
    </div>
  );
}
