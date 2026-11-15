// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useState} from 'react';
import {
  RASTER_EMBEDDING_OPERATIONS,
  type RasterEmbeddingOperation,
  type RasterEmbeddingOptions
} from '@luma-contributors/kepler';
import type {createRasterEmbeddingsHost} from './raster-embeddings-host';

type Host = ReturnType<typeof createRasterEmbeddingsHost>;

/** Native Layers → Analysis controls for multiband embeddings and physical flow diagnostics. */
export function RasterEmbeddingsPanel({
  host,
  rasterLayers
}: {
  host: Host;
  rasterLayers: readonly {id: string; label: string}[];
}) {
  const [sourceId, setSourceId] = useState('fixture');
  const [tileId, setTileId] = useState('');
  const [encoding, setEncoding] = useState<'float32' | 'alphaearth-int8'>('float32');
  const [comparisonSourceId, setComparisonSourceId] = useState('');
  const [comparisonTileId, setComparisonTileId] = useState('');
  const [operation, setOperation] = useState<RasterEmbeddingOperation>('similarity');
  const [settings, setSettings] = useState('{}');
  const [demo, setDemo] = useState(true);
  const [url, setUrl] = useState('');
  const [windowText, setWindowText] = useState('');
  const [flowId, setFlowId] = useState('');
  const [endFlowId, setEndFlowId] = useState('');
  const [timeStep, setTimeStep] = useState(1);
  const [stepCount, setStepCount] = useState(16);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const sources = [
    ...rasterLayers,
    ...(host.getLoadedSource() ? [{id: 'loaded-geotiff', label: 'Loaded embedding GeoTIFF'}] : [])
  ];
  const tiles = host.listTiles(sourceId);
  const comparisonTiles = host.listTiles(comparisonSourceId);
  const flows = host.listFlowGrids();
  async function perform(action: () => Promise<unknown>) {
    setBusy(true);
    setMessage('Running…');
    try {
      setMessage(JSON.stringify(await action()));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }
  function readSettings(): RasterEmbeddingOptions {
    const parsed: unknown = JSON.parse(settings);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed))
      throw new Error('Settings must be a JSON object');
    const values = {...parsed} as Record<string, unknown>;
    for (const name of ['queries', 'trainingVectors', 'weights', 'bias']) {
      if (values[name] !== undefined) {
        if (
          !Array.isArray(values[name]) ||
          !(values[name] as unknown[]).every(
            value => typeof value === 'number' && Number.isFinite(value)
          )
        )
          throw new Error(`${name} must be a finite number array`);
        values[name] = Float32Array.from(values[name] as number[]);
      }
    }
    for (const name of ['trainingLabels', 'zoneRaster'])
      if (values[name] !== undefined) {
        if (
          !Array.isArray(values[name]) ||
          !(values[name] as unknown[]).every(
            value =>
              typeof value === 'number' &&
              Number.isInteger(value) &&
              value >= 0 &&
              value <= 0xffffffff
          )
        )
          throw new Error(`${name} must be a uint32 array`);
        values[name] = Uint32Array.from(values[name] as number[]);
      }
    return {...values, operation};
  }
  return (
    <details data-testid="raster-embeddings-panel">
      <summary>Raster embeddings and flow diagnostics</summary>
      <fieldset disabled={busy}>
        <label>
          Embedding GeoTIFF URL
          <input
            aria-label="Embedding GeoTIFF URL"
            value={url}
            onChange={event => setUrl(event.target.value)}
          />
        </label>
        <label>
          GeoTIFF window (optional source-CRS bounds)
          <input
            aria-label="Embedding GeoTIFF window"
            value={windowText}
            placeholder="west, south, east, north"
            onChange={event => setWindowText(event.target.value)}
          />
        </label>
        <button
          disabled={!url}
          onClick={() =>
            void perform(async () => {
              const bounds = windowText.trim() ? windowText.split(',').map(Number) : undefined;
              if (bounds && bounds.length !== 4)
                throw new Error('Window needs four comma-separated source-CRS coordinates');
              const result = await host.loadGeoTiff(
                url,
                bounds as [number, number, number, number] | undefined
              );
              setSourceId('loaded-geotiff');
              setTileId('');
              setDemo(false);
              return result;
            })
          }
        >
          Load embedding GeoTIFF
        </button>
        <label>
          Embedding source
          <select
            aria-label="Embedding source"
            value={sourceId}
            onChange={event => {
              setSourceId(event.target.value);
              setTileId('');
              setDemo(event.target.value === 'fixture');
            }}
          >
            <option value="fixture">Generated four-band fixture (8 × 8)</option>
            {sources.map(source => (
              <option key={source.id} value={source.id}>
                {source.label}
              </option>
            ))}
          </select>
        </label>
        {sourceId !== 'fixture' ? (
          <label>
            Embedding tile
            <select
              aria-label="Embedding tile"
              value={tileId}
              onChange={event => setTileId(event.target.value)}
            >
              <option value="">Choose a loaded tile</option>
              {tiles.map(tile => (
                <option key={tile.id} value={tile.id}>
                  {tile.label}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        <label>
          Embedding encoding
          <select
            aria-label="Embedding encoding"
            value={encoding}
            onChange={event => setEncoding(event.target.value as typeof encoding)}
          >
            <option value="float32">Decoded float vectors</option>
            <option value="alphaearth-int8">AlphaEarth signed int8 (normalize after decode)</option>
          </select>
        </label>
        <label>
          Embedding operation
          <select
            aria-label="Embedding operation"
            value={operation}
            onChange={event => setOperation(event.target.value as RasterEmbeddingOperation)}
          >
            {RASTER_EMBEDDING_OPERATIONS.map(value => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        {operation === 'change' && sourceId !== 'fixture' ? (
          <>
            <label>
              Comparison raster
              <select
                aria-label="Comparison raster"
                value={comparisonSourceId}
                onChange={event => {
                  setComparisonSourceId(event.target.value);
                  setComparisonTileId('');
                }}
              >
                <option value="">Choose comparison raster</option>
                {sources.map(source => (
                  <option key={source.id} value={source.id}>
                    {source.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Comparison tile
              <select
                aria-label="Comparison tile"
                value={comparisonTileId}
                onChange={event => setComparisonTileId(event.target.value)}
              >
                <option value="">Choose aligned tile</option>
                {comparisonTiles.map(tile => (
                  <option key={tile.id} value={tile.id}>
                    {tile.label}
                  </option>
                ))}
              </select>
            </label>
          </>
        ) : null}
        <label>
          <input type="checkbox" checked={demo} onChange={event => setDemo(event.target.checked)} />
          Use illustrative ± first-band prototypes and left/right zones
        </label>
        <p>
          Float bands are decoded vectors. Dequantize reads signed int8 AlphaEarth bands; the
          fixture demonstrates a codec roundtrip. Supervised defaults illustrate inference and do
          not train a model.
        </p>
        <label>
          Embedding settings (JSON)
          <textarea
            aria-label="Embedding settings"
            value={settings}
            onChange={event => setSettings(event.target.value)}
          />
        </label>
        <p>
          Optional: queries, trainingVectors, trainingLabels, classCount, weights, bias,
          neighborCount, clusterCount, resultCount, threshold, scale, zoneRaster, zoneCount. Arrays
          are pixel/sample-major; labels start at zero. All bands share the intersection of their
          validity masks.
        </p>
        <button
          disabled={sourceId !== 'fixture' && !tileId}
          onClick={() =>
            void perform(() =>
              host.run({
                sourceId,
                tileId,
                options: readSettings(),
                demo,
                encoding,
                comparisonSourceId:
                  operation === 'change' && sourceId !== 'fixture' ? comparisonSourceId : undefined,
                comparisonTileId
              })
            )
          }
        >
          Publish embedding result
        </button>
        <hr />
        <label>
          Flow diagnostic source
          <select
            aria-label="Flow diagnostic source"
            value={flowId}
            onChange={event => setFlowId(event.target.value)}
          >
            <option value="">Choose an active Flow Field layer</option>
            {flows.map(flow => (
              <option key={flow.id} value={flow.id}>
                {flow.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          End velocity slice
          <select
            aria-label="End velocity slice"
            value={endFlowId}
            onChange={event => setEndFlowId(event.target.value)}
          >
            <option value="">Frozen source velocity</option>
            {flows.map(flow => (
              <option key={flow.id} value={flow.id}>
                {flow.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Signed time step (seconds)
          <input
            aria-label="Flow diagnostic time step"
            type="number"
            value={timeStep}
            onChange={event => setTimeStep(Number(event.target.value))}
          />
        </label>
        <label>
          Integration steps
          <input
            aria-label="Flow diagnostic steps"
            type="number"
            min={1}
            max={8192}
            value={stepCount}
            onChange={event => setStepCount(Number(event.target.value))}
          />
        </label>
        <p>
          Physical m/s (UV or SPEED_DIR only); elevation gradients are slopes and cannot supply
          velocity. Visual particle speed is removed. Negative steps integrate backward. Fields are
          resampled to regular local Mercator metre grids; FTLE has units s⁻¹. Lost seeds and their
          FTLE stencils are null. Time slices must have identical extents; their interval is the
          absolute integration duration.
        </p>
        <button
          disabled={
            !flowId ||
            !timeStep ||
            !Number.isInteger(stepCount) ||
            stepCount < 1 ||
            stepCount > 8192
          }
          onClick={() =>
            void perform(() =>
              host.runFlow({
                layerId: flowId,
                endLayerId: endFlowId || undefined,
                timeStep,
                stepCount
              })
            )
          }
        >
          Publish flow map and FTLE
        </button>
      </fieldset>
      <p role="status" data-testid="raster-embeddings-status">
        {message}
      </p>
    </details>
  );
}
