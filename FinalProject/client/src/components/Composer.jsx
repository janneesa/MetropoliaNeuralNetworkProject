import { useState } from 'react';

// Simple modern composer
export default function Composer({ prompt, setPrompt, onSubmit, onComposerKeyDown, isSending, error, onOpenParams }) {
  return (
    <form className="composer" onSubmit={onSubmit}>
      {error ? <div className="error">{error}</div> : null}
      <div className="composerRow">
        <button
          type="button"
          className="paramsButton"
          aria-label="Model parameters"
          onClick={onOpenParams}
          tabIndex={0}
        >
          <span role="img" aria-label="settings">⚙️</span>
        </button>
        <textarea
          className="composerInput"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={onComposerKeyDown}
          placeholder="Send a prompt"
          rows={1}
          disabled={isSending}
          aria-label="Message"
        />
        <button
          className="sendButton"
          type="submit"
          disabled={isSending || !prompt.trim()}
        >
          Send
        </button>
      </div>
    </form>
  );
}
