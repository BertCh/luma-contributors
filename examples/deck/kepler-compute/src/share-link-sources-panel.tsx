// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import {registerLinkedRoleHint} from './share-link-role-hints';
import {
  createSourceShareLink,
  getSourceLinkHost,
  loadSourceLink,
  readSourceShareLink,
  verifyReattachedFile,
  type LinkedSource,
  type LinkedSourceFormat,
  type SourceLinkDescriptor,
  type SourceLinkLoadResult
} from './share-link-sources';

function parseRoles(text: string): Record<string, string> | undefined {
  const roles: Record<string, string> = {};
  for (const part of text.split(',')) {
    const [role, column] = part.split('=').map(value => value.trim());
    if (role && column) roles[role] = column;
  }
  return Object.keys(roles).length ? roles : undefined;
}

/** Create and open links that reference remote data and saved Kepler maps instead of embedding rows. */
export function ShareLinkSourcesPanel({disabled}: {disabled: boolean}) {
  const [url, setUrl] = useState('');
  const [label, setLabel] = useState('linked-data');
  const [format, setFormat] = useState<LinkedSourceFormat>('csv');
  const [sha256, setSha256] = useState('');
  const [rolesText, setRolesText] = useState('');
  const [mapUrl, setMapUrl] = useState('');
  const [link, setLink] = useState('');
  const [status, setStatus] = useState('');
  const [missing, setMissing] = useState<{source: LinkedSource; message: string}[]>([]);
  const [mapMessage, setMapMessage] = useState('');

  const applyResult = (result: SourceLinkLoadResult) => {
    const host = getSourceLinkHost();
    for (const outcome of result.outcomes)
      if (outcome.status === 'loaded')
        registerLinkedRoleHint(outcome.file.name, outcome.source.roles);
    const files = [
      ...result.outcomes.flatMap(outcome => (outcome.status === 'loaded' ? [outcome.file] : [])),
      ...(result.savedMap?.status === 'loaded' ? [result.savedMap.file] : [])
    ];
    if (files.length) {
      if (!host) throw new Error('This host cannot open linked files');
      host.loadFiles(files);
    }
    setMissing(
      result.outcomes.flatMap(outcome =>
        outcome.status === 'needs-reattachment'
          ? [{source: outcome.source, message: outcome.message}]
          : []
      )
    );
    setMapMessage(result.savedMap?.status === 'needs-reattachment' ? result.savedMap.message : '');
    setStatus(
      `${files.length} linked file${files.length === 1 ? '' : 's'} opened in native Kepler; linked column roles are applied by automatic preparation (roles it cannot apply are listed in its report).`
    );
  };
  const openDescriptor = async (descriptor: SourceLinkDescriptor) => {
    setStatus('Loading linked sources...');
    try {
      applyResult(await loadSourceLink(descriptor));
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    }
  };
  useEffect(() => {
    const readLocation = () => {
      try {
        const descriptor = readSourceShareLink(window.location.href);
        if (descriptor) void openDescriptor(descriptor);
      } catch (error) {
        setStatus(error instanceof Error ? error.message : String(error));
      }
    };
    readLocation();
    window.addEventListener('hashchange', readLocation);
    return () => window.removeEventListener('hashchange', readLocation);
  }, []);

  return (
    <section aria-label="Share data links" data-testid="share-link-sources">
      <h3>Share a data link</h3>
      <p className="hint">
        A data link names a public https file (and optionally a saved Kepler map) instead of
        embedding rows. Private or missing data asks you to attach your own copy, which is checked
        against the recorded checksum. This sandbox has no accounts, so access control is the data
        host&apos;s job.
      </p>
      <label>
        Data URL
        <input aria-label="Linked data URL" value={url} onChange={e => setUrl(e.target.value)} />
      </label>
      <label>
        Name
        <input
          aria-label="Linked data name"
          value={label}
          onChange={e => setLabel(e.target.value)}
        />
      </label>
      <label>
        Format
        <select
          aria-label="Linked data format"
          value={format}
          onChange={e => setFormat(e.target.value as LinkedSourceFormat)}
        >
          <option value="csv">CSV</option>
          <option value="arrow">Arrow IPC</option>
          <option value="geojson">GeoJSON</option>
          <option value="json">JSON</option>
        </select>
      </label>
      <label>
        SHA-256 (optional)
        <input
          aria-label="Linked data SHA-256"
          value={sha256}
          onChange={e => setSha256(e.target.value)}
        />
      </label>
      <label>
        Roles (optional, role=column)
        <input
          aria-label="Linked data roles"
          value={rolesText}
          onChange={e => setRolesText(e.target.value)}
        />
      </label>
      <label>
        Saved map JSON URL (optional)
        <input
          aria-label="Linked saved map URL"
          value={mapUrl}
          onChange={e => setMapUrl(e.target.value)}
        />
      </label>
      <div className="actions">
        <button
          disabled={disabled}
          onClick={() => {
            setLink('');
            try {
              const roles = parseRoles(rolesText);
              setLink(
                createSourceShareLink(
                  {
                    sources: url
                      ? [
                          {
                            label,
                            url,
                            format,
                            ...(sha256 ? {sha256} : {}),
                            ...(roles ? {roles} : {})
                          }
                        ]
                      : [],
                    ...(mapUrl ? {savedMapUrl: mapUrl} : {})
                  },
                  window.location.href
                )
              );
              setStatus('Data link created.');
            } catch (error) {
              setStatus(error instanceof Error ? error.message : String(error));
            }
          }}
        >
          Create data link
        </button>
      </div>
      {link ? (
        <textarea
          aria-label="Data share link"
          readOnly
          rows={2}
          value={link}
          style={{width: '100%'}}
        />
      ) : null}
      {missing.map(item => (
        <div key={item.source.url} role="alert" data-testid="reattach-source">
          <p>{item.message}</p>
          <label>
            Attach {item.source.label}
            <input
              type="file"
              aria-label={`Attach ${item.source.label}`}
              onChange={event => {
                const file = event.target.files?.[0];
                event.target.value = '';
                if (!file) return;
                void verifyReattachedFile(item.source, file)
                  .then(() => {
                    registerLinkedRoleHint(file.name, item.source.roles);
                    getSourceLinkHost()?.loadFiles([file]);
                    setMissing(current => current.filter(entry => entry !== item));
                    setStatus(`${file.name} attached.`);
                  })
                  .catch(error =>
                    setStatus(error instanceof Error ? error.message : String(error))
                  );
              }}
            />
          </label>
        </div>
      ))}
      {mapMessage ? <p role="alert">{mapMessage}</p> : null}
      {status ? (
        <p role="status" className="hint">
          {status}
        </p>
      ) : null}
    </section>
  );
}
