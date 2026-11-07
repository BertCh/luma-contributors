// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useState} from 'react';
import type {RasterTerrainOperation} from './analysis2-raster-terrain';
import type {
  RasterTerrainHost,
  TerrainActionRequest,
  TerrainActionResult
} from './analysis4-raster-terrain-host';

export type RasterTerrainPanelProps = Readonly<{
  terrain: RasterTerrainHost;
  rasterLayers: readonly {id: string; label: string}[];
  /** Whether a current IDW or kriging grid can be used as the elevation input. */
  isDerivedGridAvailable: boolean;
}>;

const OPERATIONS: readonly RasterTerrainOperation[] = ['slope', 'aspect', 'hillshade', 'curvature'];

function downloadBytes(data: Uint8Array<ArrayBuffer>, filename: string): void {
  const url = URL.createObjectURL(new Blob([data], {type: 'image/tiff'}));
  const anchor = document.createElement('a');
  try {
    anchor.href = url;
    anchor.download = filename;
    document.body.append(anchor);
    anchor.click();
  } finally {
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}

/** Terrain derivatives of a TILES numeric raster or a derived grid, published as a native dataset. */
export function RasterTerrainPanel({
  terrain,
  rasterLayers,
  isDerivedGridAvailable
}: RasterTerrainPanelProps) {
  const [inputKind, setInputKind] = useState<'tiles' | 'derived-grid'>('tiles');
  const [rasterLayerId, setRasterLayerId] = useState('');
  const [operation, setOperation] = useState<RasterTerrainOperation>('slope');
  const [zFactor, setZFactor] = useState(1);
  const [azimuthDegrees, setAzimuthDegrees] = useState(315);
  const [altitudeDegrees, setAltitudeDegrees] = useState(45);
  const [engine, setEngine] = useState<'gpu' | 'cpu'>('gpu');
  const [keepOnGpu, setKeepOnGpu] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [message, setMessage] = useState('');
  const [result, setResult] = useState<TerrainActionResult | null>(null);
  const effectiveInput =
    inputKind === 'derived-grid' && isDerivedGridAvailable
      ? 'derived-grid'
      : rasterLayers.length > 0
        ? 'tiles'
        : 'derived-grid';
  const selectedRaster = rasterLayerId || rasterLayers[0]?.id || '';
  const canRun =
    !isRunning && (effectiveInput === 'tiles' ? Boolean(selectedRaster) : isDerivedGridAvailable);
  return (
    <section aria-label="Terrain analysis" data-testid="raster-terrain-panel">
      <h2>Terrain analysis</h2>
      <label>
        Input{' '}
        <select
          value={effectiveInput}
          data-testid="terrain-input-select"
          onChange={event =>
            setInputKind(event.target.value === 'tiles' ? 'tiles' : 'derived-grid')
          }
        >
          <option value="tiles" disabled={rasterLayers.length === 0}>
            Raster tile layer (band 1 elevation)
          </option>
          <option value="derived-grid" disabled={!isDerivedGridAvailable}>
            Derived IDW or kriging grid
          </option>
        </select>
      </label>
      {effectiveInput === 'tiles' ? (
        <label>
          Raster layer{' '}
          <select
            value={selectedRaster}
            data-testid="terrain-raster-select"
            onChange={event => setRasterLayerId(event.target.value)}
          >
            {rasterLayers.map(layer => (
              <option key={layer.id} value={layer.id}>
                {layer.label}
              </option>
            ))}
          </select>
        </label>
      ) : (
        <p className="hint">
          The grid values are read as elevation in meters times the vertical factor. IDW and kriging
          values with other units give slopes in those units.
        </p>
      )}
      <label>
        Operation{' '}
        <select
          value={operation}
          data-testid="terrain-operation-select"
          onChange={event => setOperation(event.target.value as RasterTerrainOperation)}
        >
          {OPERATIONS.map(name => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </label>
      <label>
        Vertical factor{' '}
        <input
          type="number"
          min={0}
          step={0.1}
          value={zFactor}
          onChange={event => setZFactor(Math.max(0, Number(event.target.value)) || 1)}
        />
      </label>
      {operation === 'hillshade' ? (
        <>
          <label>
            Azimuth (degrees clockwise from north){' '}
            <input
              type="number"
              min={0}
              max={360}
              value={azimuthDegrees}
              onChange={event => setAzimuthDegrees(Number(event.target.value) || 0)}
            />
          </label>
          <label>
            Altitude (degrees above horizon){' '}
            <input
              type="number"
              min={0}
              max={90}
              value={altitudeDegrees}
              onChange={event => setAltitudeDegrees(Number(event.target.value) || 0)}
            />
          </label>
        </>
      ) : null}
      <label>
        Engine{' '}
        <select
          value={engine}
          data-testid="terrain-engine-select"
          onChange={event => setEngine(event.target.value === 'cpu' ? 'cpu' : 'gpu')}
        >
          <option value="gpu">GPU (CPU fallback)</option>
          <option value="cpu">CPU reference</option>
        </select>
      </label>
      <label>
        <input
          type="checkbox"
          checked={keepOnGpu}
          disabled={engine === 'cpu' || effectiveInput !== 'tiles'}
          data-testid="terrain-keep-on-gpu"
          onChange={event => setKeepOnGpu(event.target.checked)}
        />{' '}
        Keep cells on the GPU (no readback; point centers with longitude/latitude columns, nodata
        cells are null rows; a table without a map layer)
      </label>
      <button
        type="button"
        disabled={!canRun}
        data-testid="terrain-run"
        onClick={async () => {
          setIsRunning(true);
          setMessage('');
          try {
            const request: TerrainActionRequest = {
              input:
                effectiveInput === 'tiles'
                  ? {kind: 'tiles', rasterLayerId: selectedRaster}
                  : {kind: 'derived-grid'},
              operation,
              zFactor,
              azimuthDegrees,
              altitudeDegrees,
              engine,
              publication: keepOnGpu ? 'resident' : 'table',
              label: `Terrain ${operation} ${
                effectiveInput === 'tiles'
                  ? (rasterLayers.find(layer => layer.id === selectedRaster)?.label ??
                    selectedRaster)
                  : 'derived grid'
              }`
            };
            const next = await terrain.runTerrainAction(request);
            setResult(next);
            setMessage(
              `${next.engineNote ?? ''} ${
                next.published
                  ? next.publication === 'resident'
                    ? `Published ${next.rowCount} cells GPU-resident (${next.publicationReadbackBytes ?? 0} bytes read back).`
                    : `Published ${next.rowCount} cells as a native dataset.`
                  : (next.reason ?? 'Nothing was published.')
              }`.trim()
            );
          } catch (error) {
            setMessage(error instanceof Error ? error.message : 'Terrain analysis failed.');
          } finally {
            setIsRunning(false);
          }
        }}
      >
        {isRunning ? 'Computing...' : 'Compute terrain'}
      </button>
      <button
        type="button"
        disabled={!result?.exportAvailable}
        data-testid="terrain-export"
        onClick={() => {
          terrain
            .exportTerrainGeoTIFF()
            .then(({data, filename}) => downloadBytes(data, filename))
            .catch(error =>
              setMessage(error instanceof Error ? error.message : 'Terrain export failed.')
            );
        }}
      >
        Export GeoTIFF
      </button>
      <p className="hint" role="status" data-testid="terrain-status">
        {message}
      </p>
      {result?.fallbackReason ? (
        <p className="hint" role="note" data-testid="terrain-fallback-reason">
          {`Not run on the GPU: ${result.fallbackReason}`}
        </p>
      ) : null}
    </section>
  );
}
