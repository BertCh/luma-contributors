// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useRef, useState} from 'react';
import {createSandboxShareLink, readSandboxShareLink} from './sandbox-share';

type SandboxSharePanelProps = {
  disabled: boolean;
  saveState(): string;
  loadState(serialized: string): void;
};

/** Loading a link prepares the existing replay controls; it never replaces a running dataset. */
export function SandboxSharePanel({disabled, saveState, loadState}: SandboxSharePanelProps) {
  const [link, setLink] = useState('');
  const [status, setStatus] = useState('');
  const loadStateReference = useRef(loadState);
  loadStateReference.current = loadState;
  const pendingSharedState = useRef<string | null>(null);
  const disabledReference = useRef(disabled);
  disabledReference.current = disabled;
  useEffect(() => {
    const readLocation = () => {
      try {
        pendingSharedState.current = readSandboxShareLink(window.location.href);
        if (pendingSharedState.current && !disabledReference.current) {
          loadStateReference.current(pendingSharedState.current);
          pendingSharedState.current = null;
          setStatus('Shared state loaded into the state editor.');
        }
      } catch (error) {
        pendingSharedState.current = null;
        setStatus(error instanceof Error ? error.message : String(error));
      }
    };
    readLocation();
    window.addEventListener('hashchange', readLocation);
    return () => window.removeEventListener('hashchange', readLocation);
  }, []);
  useEffect(() => {
    if (!disabled && pendingSharedState.current) {
      loadStateReference.current(pendingSharedState.current);
      pendingSharedState.current = null;
      setStatus('Shared state loaded into the state editor.');
    }
  }, [disabled]);

  return (
    <section aria-label="Share sandbox">
      <p className="hint">
        Share links save the current map and analysis settings and regenerate the sample data.
        Recipients need access to this sandbox address. Links do not contain uploaded rows.
      </p>
      <div className="actions">
        <button
          disabled={disabled}
          onClick={() => {
            setLink('');
            try {
              setLink(createSandboxShareLink(saveState(), window.location.href));
              setStatus('Link created. Copy it to share this saved state.');
            } catch (error) {
              setStatus(error instanceof Error ? error.message : String(error));
            }
          }}
        >
          Create share link
        </button>
        {link ? (
          <button
            disabled={disabled}
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(link);
                setStatus('Share link copied.');
              } catch {
                setStatus('Select the link below and copy it manually.');
              }
            }}
          >
            Copy link
          </button>
        ) : null}
      </div>
      {link ? (
        <label>
          Share link
          <textarea
            aria-label="Sandbox share link"
            readOnly
            rows={2}
            value={link}
            onFocus={event => event.currentTarget.select()}
            style={{width: '100%'}}
          />
        </label>
      ) : null}
      {status ? (
        <p role="status" className="hint">
          {status}
        </p>
      ) : null}
    </section>
  );
}
