// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useState} from 'react';
import {makeBitmapBoundsPatch, type BitmapBounds} from './bitmap-georeference';
import type {BitmapLiveGeoreference} from './raster-live-bitmap';

export type BitmapToolsPanelProps = Readonly<{
  bitmapLayers: readonly {id: string; label: string}[];
  getGeoreference(layerId: string): BitmapLiveGeoreference | undefined;
  /** Decodes the GeoTIFF and returns the bounds patch (nothing is dispatched yet). */
  attachGeoTiff(layerId: string, bytes: Uint8Array): Record<string, unknown>;
  /** Dispatches Kepler's layerVisConfigChange for the layer. */
  applyVisConfig(layerId: string, patch: Record<string, unknown>): void;
}>;

const BOUND_LABELS = ['West', 'South', 'East', 'North'] as const;

/** Bitmap overlay tools: numeric bounds/opacity editor and GeoTIFF attach, dispatching native visConfig. */
export function BitmapToolsPanel({
  bitmapLayers,
  getGeoreference,
  attachGeoTiff,
  applyVisConfig
}: BitmapToolsPanelProps) {
  const [layerId, setLayerId] = useState(bitmapLayers[0]?.id ?? '');
  const [draft, setDraft] = useState<string[] | null>(null);
  const [status, setStatus] = useState('');
  if (bitmapLayers.length === 0) return null;
  const activeId = bitmapLayers.some(layer => layer.id === layerId) ? layerId : bitmapLayers[0].id;
  const live = getGeoreference(activeId);
  const liveBounds = live?.georeference?.bounds;
  const values = draft ?? (liveBounds ? liveBounds.map(String) : ['', '', '', '']);
  return (
    <section aria-label="Bitmap tools" data-testid="bitmap-tools-panel">
      <h2>Bitmap overlay</h2>
      <label>
        Layer{' '}
        <select
          value={activeId}
          onChange={event => {
            setLayerId(event.target.value);
            setDraft(null);
          }}
        >
          {bitmapLayers.map(layer => (
            <option key={layer.id} value={layer.id}>
              {layer.label}
            </option>
          ))}
        </select>
      </label>
      {BOUND_LABELS.map((name, index) => (
        <label key={name}>
          {name}{' '}
          <input
            type="number"
            step="any"
            data-testid={`bitmap-bound-${name.toLowerCase()}`}
            value={values[index]}
            onChange={event => {
              const next = [...values];
              next[index] = event.target.value;
              setDraft(next);
            }}
          />
        </label>
      ))}
      <button
        type="button"
        data-testid="bitmap-apply-bounds"
        onClick={() => {
          try {
            const patch = makeBitmapBoundsPatch(
              values.map(Number) as unknown as BitmapBounds,
              live?.georeference?.opacity ?? 1
            );
            applyVisConfig(activeId, patch);
            setDraft(null);
            setStatus('Bounds applied');
          } catch (error) {
            setStatus((error as Error).message);
          }
        }}
      >
        Apply bounds
      </button>
      <label>
        Attach GeoTIFF{' '}
        <input
          type="file"
          accept=".tif,.tiff,image/tiff"
          data-testid="bitmap-geotiff-input"
          onChange={async event => {
            const file = event.target.files?.[0];
            if (!file) return;
            try {
              const patch = attachGeoTiff(activeId, new Uint8Array(await file.arrayBuffer()));
              applyVisConfig(activeId, patch);
              setDraft(null);
              setStatus(`GeoTIFF attached: ${file.name}`);
            } catch (error) {
              setStatus((error as Error).message);
            }
          }}
        />
      </label>
      <p className="hint" data-testid="bitmap-status" role="status">
        {live?.hasNumericSource
          ? 'Numeric GeoTIFF source attached (identify enabled).'
          : 'Plain image: no numeric values for analysis.'}{' '}
        {status}
      </p>
    </section>
  );
}
