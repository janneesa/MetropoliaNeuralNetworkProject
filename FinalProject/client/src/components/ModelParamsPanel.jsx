import { useState, useMemo, useEffect } from 'react';
import './ModelParamsPanel.css';

const tempExamples = [
  {
    low: 'The cat sat on the mat.',
    high: 'A turquoise feline pirouetted atop a velvet rug, pondering quantum dreams.'
  },
  {
    low: 'It is raining today.',
    high: 'Raindrops waltz from the sky, painting the city in shimmering silver.'
  },
  {
    low: 'I like pizza.',
    high: 'Pizza is a cosmic wheel of molten joy and infinite possibility!'
  },
  {
    low: 'The dog barked.',
    high: 'A boisterous hound serenaded the moon with operatic barks.'
  },
  {
    low: 'She opened the door.',
    high: 'With a flourish, she unveiled new worlds behind the ancient door.'
  }
];

function getTempExample(temp) {
  const idx = Math.floor(temp * (tempExamples.length - 1) / 2);
  const ex = tempExamples[idx];
  if (temp < 0.7) return ex.low;
  if (temp > 1.3) return ex.high;
  // Blend
  return `${ex.low.slice(0, Math.floor(ex.low.length/2))}...${ex.high.slice(Math.floor(ex.high.length/2))}`;
}

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
            Max Tokens
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
