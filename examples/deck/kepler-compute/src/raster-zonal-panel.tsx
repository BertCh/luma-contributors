// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useState} from 'react';
import type {RasterZonalActionResult} from './raster-live-host';

export type RasterZonalPanelProps = Readonly<{
  rasterLayers: readonly {id: string; label: string}[];
  polygonDatasets: readonly {id: string; label: string}[];
  run(request: {
    rasterLayerId: string;
    polygonDatasetId: string;
    bandIndex: number;
    label: string;
    engine: 'gpu' | 'cpu';
    terrainOperation?: 'slope' | 'aspect' | 'hillshade' | 'curvature';
  }): Promise<RasterZonalActionResult>;
}>;

/** Zonal statistics: pick a loaded raster tile layer and a polygon dataset, publish zonal_* columns. */
export function RasterZonalPanel({rasterLayers, polygonDatasets, run}: RasterZonalPanelProps) {
  const [rasterLayerId, setRasterLayerId] = useState('');
  const [polygonDatasetId, setPolygonDatasetId] = useState('');
  const [bandIndex, setBandIndex] = useState(1);
  const [engine, setEngine] = useState<'gpu' | 'cpu'>('gpu');
  const [terrainOperation, setTerrainOperation] = useState<
    '' | 'slope' | 'aspect' | 'hillshade' | 'curvature'
  >('');
  const [isRunning, setIsRunning] = useState(false);
  const [message, setMessage] = useState('');
  const selectedRaster = rasterLayerId || rasterLayers[0]?.id || '';
  const selectedPolygons = polygonDatasetId || polygonDatasets[0]?.id || '';
  const canRun = Boolean(selectedRaster && selectedPolygons) && !isRunning;
  const raster = rasterLayers.find(layer => layer.id === selectedRaster);
  const polygons = polygonDatasets.find(dataset => dataset.id === selectedPolygons);
  return (
    <section aria-label="Zonal statistics" data-testid="raster-zonal-panel">
      <h2>Zonal statistics</h2>
      <label>
        Raster layer{' '}
        <select
          value={selectedRaster}
          data-testid="zonal-raster-select"
          onChange={event => setRasterLayerId(event.target.value)}
        >
          {rasterLayers.map(layer => (
            <option key={layer.id} value={layer.id}>
              {layer.label}
            </option>
          ))}
        </select>
      </label>
      <label>
        Polygons{' '}
        <select
          value={selectedPolygons}
          data-testid="zonal-polygon-select"
          onChange={event => setPolygonDatasetId(event.target.value)}
        >
          {polygonDatasets.map(dataset => (
            <option key={dataset.id} value={dataset.id}>
              {dataset.label}
            </option>
          ))}
        </select>
      </label>
      <label>
        Band{' '}
        <input
          type="number"
          min={1}
          step={1}
          value={bandIndex}
          onChange={event => setBandIndex(Math.max(1, Math.floor(Number(event.target.value)) || 1))}
        />
      </label>
      <label>
        Engine{' '}
        <select
          value={engine}
          data-testid="zonal-engine-select"
          onChange={event => setEngine(event.target.value === 'cpu' ? 'cpu' : 'gpu')}
        >
          <option value="gpu">GPU (resident textures)</option>
          <option value="cpu">CPU reference</option>
        </select>
      </label>
      <label>
        Statistic of{' '}
        <select
          value={terrainOperation}
          data-testid="zonal-terrain-select"
          onChange={event => setTerrainOperation(event.target.value as typeof terrainOperation)}
        >
          <option value="">Band values</option>
          <option value="slope">Terrain slope (band as elevation)</option>
          <option value="aspect">Terrain aspect</option>
          <option value="hillshade">Terrain hillshade</option>
          <option value="curvature">Terrain curvature</option>
        </select>
      </label>
      <button
        type="button"
        disabled={!canRun}
        data-testid="zonal-run"
        onClick={async () => {
          setIsRunning(true);
          setMessage('');
          try {
            const result = await run({
              rasterLayerId: selectedRaster,
              polygonDatasetId: selectedPolygons,
              bandIndex,
              label: `Zonal ${raster?.label ?? selectedRaster} by ${polygons?.label ?? selectedPolygons}`,
              engine,
              ...(terrainOperation ? {terrainOperation} : {})
            });
            setMessage(
              result.published
                ? `Published ${result.rowCount} rows (${result.engine ?? engine}${
                    result.fallbackReason ? `, fell back: ${result.fallbackReason}` : ''
                  }); ${result.skippedRowCount} skipped.`
                : (result.reason ?? 'Nothing was published.')
            );
          } catch (error) {
            setMessage(error instanceof Error ? error.message : 'Zonal statistics failed.');
          } finally {
            setIsRunning(false);
          }
        }}
      >
        {isRunning ? 'Computing...' : 'Compute zonal statistics'}
      </button>
      {rasterLayers.length === 0 || polygonDatasets.length === 0 ? (
        <p className="hint">
          Needs a raster tile layer with loaded tiles and a published polygon dataset.
        </p>
      ) : null}
      <p className="hint" role="status" data-testid="zonal-status">
        {message}
      </p>
    </section>
  );
}
