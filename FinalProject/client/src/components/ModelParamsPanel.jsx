import { useEffect, useState } from 'react';
import './ModelParamsPanel.css';

export default function ModelParamsPanel({ open, onClose, params, setParams }) {
  const [visible, setVisible] = useState(open);
  const [closing, setClosing] = useState(false);

  useEffect(() => {
    if (open) {
      setVisible(true);
      setClosing(false);
    } else if (visible) {
      setClosing(true);
      const timeout = setTimeout(() => {
        setVisible(false);
        setClosing(false);
      }, 480); // match animation duration
      return () => clearTimeout(timeout);
    }
  }, [open]);

  if (!visible) return null;
  return (
    <div className="paramsOverlay" onClick={onClose}>
      <div className={`paramsPanel modern${closing ? ' closing' : ''}`} onClick={e => e.stopPropagation()}>
        <div className="paramsHeader">
          <span className="paramsTitle">Model Parameters</span>
          <button className="closeBtn" onClick={onClose} aria-label="Close">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M6 6L18 18" stroke="#b0b0b8" strokeWidth="2" strokeLinecap="round"/>
              <path d="M18 6L6 18" stroke="#b0b0b8" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
        </div>
        <div className="paramsDivider" />
        <div className="paramsBody">
          <label className="paramToggleRow">
            <span>Stream Response</span>
            <input
              type="checkbox"
              checked={Boolean(params.stream)}
              onChange={e => setParams(p => ({ ...p, stream: e.target.checked }))}
              className="paramCheckbox"
            />
          </label>
          <label className="paramLabel">
            Temperature
            <div className="paramSliderRow">
              <input
                type="range"
                min="0"
                max="2"
                step="0.01"
                value={params.temperature}
                onChange={e => setParams(p => ({ ...p, temperature: Number(e.target.value) }))}
                className="paramSlider"
              />
              <span className="paramValueBubble">{params.temperature}</span>
            </div>
          </label>
          <label className="paramLabel">
            Response Length
            <input
              type="number"
              min="1"
              max="4096"
              value={params.maxTokens}
              onChange={e => setParams(p => ({ ...p, maxTokens: Number(e.target.value) }))}
              className="paramNumber"
            />
          </label>
        </div>
      </div>
    </div>
  );
}
