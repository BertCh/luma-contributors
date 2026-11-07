// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import {
  findWmsTimeExtentText,
  parseWmsExtentInstants,
  resolveWmsTimeValue
} from './wms-time-extent';

export type WmsTimePickerLayer = Readonly<{
  id: string;
  label: string;
  /** Service URL used to read GetCapabilities when no extent is supplied. */
  serviceUrl?: string;
  layerName?: string;
  version?: string;
}>;

export type WmsTimePickerProps = Readonly<{
  layers: readonly WmsTimePickerLayer[];
  /** Applies a TIME (or clears it with null); false when outside the extent. */
  setTime(layerId: string, time: string | null, extentText?: string): boolean;
  getTime(layerId: string): string | null;
  fetchText?(url: string): Promise<string>;
}>;

function buildCapabilitiesUrl(layer: WmsTimePickerLayer): string | undefined {
  if (!layer.serviceUrl) return undefined;
  const url = new URL(layer.serviceUrl, globalThis.location?.href);
  url.searchParams.set('service', 'WMS');
  url.searchParams.set('request', 'GetCapabilities');
  if (layer.version) url.searchParams.set('version', layer.version);
  return url.toString();
}

/** TIME control for WMS layers: slider over the capabilities extent, or a typed ISO instant. */
export function WmsTimePicker({layers, setTime, getTime, fetchText}: WmsTimePickerProps) {
  const [layerId, setLayerId] = useState(layers[0]?.id ?? '');
  const [extentText, setExtentText] = useState<string>();
  const [defaultValue, setDefaultValue] = useState<string>();
  const [instants, setInstants] = useState<string[]>([]);
  const [manual, setManual] = useState('');
  const [status, setStatus] = useState('');
  const layer = layers.find(candidate => candidate.id === layerId) ?? layers[0];
  useEffect(() => {
    setExtentText(undefined);
    setDefaultValue(undefined);
    setInstants([]);
    setStatus('');
    const url = layer && buildCapabilitiesUrl(layer);
    if (!layer || !url || !layer.layerName) return;
    let isCurrent = true;
    const load = fetchText ?? (async (target: string) => (await fetch(target)).text());
    load(url)
      .then(text => {
        if (!isCurrent) return;
        const found = findWmsTimeExtentText(text, layer.layerName as string);
        if (!found) {
          setStatus('This layer advertises no time dimension; type a TIME value.');
          return;
        }
        setExtentText(found.extentText);
        setDefaultValue(found.defaultValue);
        setInstants(parseWmsExtentInstants(found.extentText));
      })
      .catch(() => isCurrent && setStatus('Could not read the service capabilities.'));
    return () => {
      isCurrent = false;
    };
  }, [layer?.id, layer?.serviceUrl, layer?.layerName]);
  if (!layer) return null;
  const current = getTime(layer.id);
  const currentIndex = current
    ? instants.findIndex(value => Date.parse(value) === Date.parse(current))
    : -1;
  const applyResolved = (requested: string) => {
    // Keywords (current/default) and out-of-extent instants snap to an advertised instant.
    const resolution = extentText
      ? resolveWmsTimeValue(extentText, requested, {defaultValue, allowNearest: true})
      : {kind: 'exact' as const, time: requested};
    if (!resolution.time) return apply(requested);
    apply(resolution.time);
    if (resolution.kind !== 'exact') setStatus(`TIME ${resolution.time} (${resolution.kind})`);
  };
  const apply = (time: string | null) => {
    const accepted = setTime(layer.id, time, extentText);
    setStatus(
      accepted ? (time ? `TIME ${time}` : 'TIME cleared') : 'TIME is outside the advertised extent.'
    );
  };
  return (
    <section aria-label="WMS time" data-testid="wms-time-picker">
      <h2>WMS time</h2>
      <label>
        Layer{' '}
        <select value={layer.id} onChange={event => setLayerId(event.target.value)}>
          {layers.map(candidate => (
            <option key={candidate.id} value={candidate.id}>
              {candidate.label}
            </option>
          ))}
        </select>
      </label>
      {instants.length > 0 ? (
        <div>
          <input
            type="range"
            aria-label="WMS time"
            data-testid="wms-time-slider"
            min={0}
            max={instants.length - 1}
            value={Math.max(0, currentIndex)}
            onChange={event => apply(instants[Number(event.target.value)])}
          />
          <span data-testid="wms-time-value">{current ?? instants[0]}</span>
          <button
            type="button"
            disabled={currentIndex <= 0}
            onClick={() => apply(instants[currentIndex - 1])}
          >
            Previous
          </button>
          <button
            type="button"
            disabled={currentIndex >= instants.length - 1}
            onClick={() => apply(instants[currentIndex + 1])}
          >
            Next
          </button>
        </div>
      ) : null}
      <label>
        TIME{' '}
        <input
          type="text"
          placeholder="2024-05-01T00:00:00Z"
          value={manual}
          data-testid="wms-time-input"
          onChange={event => setManual(event.target.value)}
        />
      </label>
      <button type="button" disabled={!manual.trim()} onClick={() => applyResolved(manual.trim())}>
        Apply
      </button>
      <button type="button" disabled={!extentText} onClick={() => applyResolved('current')}>
        Current
      </button>
      <button type="button" disabled={!extentText} onClick={() => applyResolved('default')}>
        Default
      </button>
      <button type="button" disabled={!current} onClick={() => apply(null)}>
        Clear
      </button>
      <p className="hint" data-testid="wms-time-status" role="status">
        {status}
      </p>
    </section>
  );
}
