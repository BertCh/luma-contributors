// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useState} from 'react';
import {
  RASTER_TIMESERIES_BINDINGS,
  getRasterTimeSeriesBinding,
  isRasterTimeSeriesGlobalOutput,
  type RasterTimeSeriesOperation,
  type RasterTimeSeriesSettings
} from '@luma-contributors/kepler';
import type {createRasterTimeSeriesHost} from './raster-timeseries-host';
import type {RasterTimeSlice} from './raster-timeseries-stack';

const EXAMPLE_STACK =
  '[{"time":"2020-01-01T00:00:00Z","bands":[1]}, {"time":"2020-02-01T00:00:00Z","bands":[2]}]';
const label = (name: string) =>
  name.replace(/([A-Z])/g, ' $1').replace(/^./, first => first.toUpperCase());

/** Raster-native workflow, separate from the feature-collection catalog. */
export function RasterTimeSeriesPanel({
  host,
  rasterLayers,
  polygonDatasets
}: {
  host: ReturnType<typeof createRasterTimeSeriesHost>;
  rasterLayers: readonly {id: string; label: string}[];
  polygonDatasets: readonly {id: string; label: string}[];
}) {
  const [operation, setOperation] = useState<RasterTimeSeriesOperation>('rolling-mean');
  const [layerId, setLayerId] = useState('');
  const [polygonId, setPolygonId] = useState('');
  const [mapping, setMapping] = useState('');
  const [settingsText, setSettingsText] = useState<Record<string, string>>({});
  const [outputName, setOutputName] = useState('');
  const [outputComponent, setOutputComponent] = useState(0);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState('');
  const [imported, setImported] = useState<string[]>([]);
  const [outputs, setOutputs] = useState<Record<string, {components: number; scope: string}>>({});
  const binding = getRasterTimeSeriesBinding(operation);
  const layers = [...rasterLayers, ...imported.map(id => ({id, label: id}))];
  return (
    // Collapsed like the raster embeddings panel so the side panel keeps Kepler's tabs in view.
    <details aria-label="Raster time series" data-testid="raster-timeseries-panel">
      <summary>Raster time series</summary>
      <p>
        Map numeric bands to acquisition dates on identical grids. Only loaded tiles participate.
        Maximum 256 dates, 65,536 cells and 1,048,576 values. GPU required; selected outputs are
        read back for zonal publication.
      </p>
      <label>
        Raster source{' '}
        <select
          data-testid="timeseries-source"
          value={layerId || layers[0]?.id || ''}
          onChange={event => setLayerId(event.target.value)}
        >
          {layers.map(layer => (
            <option key={layer.id} value={layer.id}>
              {layer.label}
            </option>
          ))}
        </select>
      </label>
      <label>
        Or load a numeric GeoTIFF{' '}
        <input
          type="file"
          accept=".tif,.tiff"
          disabled={running}
          onChange={async event => {
            const file = event.target.files?.[0];
            if (!file) return;
            try {
              const result = host.importGeoTIFF(
                file.name,
                new Uint8Array(await file.arrayBuffer())
              );
              setImported(host.listImportedSources());
              setLayerId(result.id);
              setStatus(`Loaded ${result.id}: ${result.bandCount} bands`);
            } catch (error) {
              setStatus(String(error));
            }
          }}
        />
      </label>
      <button
        type="button"
        disabled={running || imported.length === 0}
        onClick={() => {
          host.clear();
          setImported([]);
          setLayerId('');
          setOutputs({});
          setStatus('Cleared imported time-stack sources and retained results');
        }}
      >
        Clear imported time-stack sources
      </button>
      <label>
        Acquisitions (JSON){' '}
        <textarea
          data-testid="timeseries-mapping"
          value={mapping}
          placeholder={EXAMPLE_STACK}
          onChange={event => setMapping(event.target.value)}
        />
      </label>
      <p>
        Each entry needs an ISO date with timezone and one-based bands. Optionally set layerId to
        another loaded source. Multiple bands per date are supported by composites and COLD.
        Example: <code>{EXAMPLE_STACK}</code>
      </p>
      <label>
        Operation{' '}
        <select
          data-testid="timeseries-operation"
          value={operation}
          onChange={event => {
            setOperation(event.target.value as RasterTimeSeriesOperation);
            setSettingsText({});
            setOutputName('');
            setOutputs({});
            setStatus('');
          }}
        >
          {RASTER_TIMESERIES_BINDINGS.map(binding => (
            <option key={binding.id} value={binding.id}>
              {binding.title}
            </option>
          ))}
        </select>
      </label>
      <p>{binding.note}</p>
      {Object.entries(binding.defaults).map(([name, defaultValue]) => (
        <label key={`${operation}-${name}`}>
          {label(name)}{' '}
          <input
            data-testid={`timeseries-setting-${name}`}
            type={typeof defaultValue === 'number' ? 'number' : 'text'}
            step="any"
            value={
              settingsText[name] ??
              (Array.isArray(defaultValue) ? defaultValue.join(', ') : String(defaultValue))
            }
            onChange={event => setSettingsText({...settingsText, [name]: event.target.value})}
          />
        </label>
      ))}
      {(operation === 'standardized-calibration' || operation === 'standardized-index') && (
        <label>
          Zero atom{' '}
          <select
            data-testid="timeseries-setting-zeroAtom"
            value={settingsText['zeroAtom'] ?? 'default'}
            onChange={event => setSettingsText({...settingsText, zeroAtom: event.target.value})}
          >
            <option value="default">Distribution default</option>
            <option value="on">On</option>
            <option value="off">Off</option>
          </select>
        </label>
      )}
      <label>
        Polygon zones{' '}
        <select
          data-testid="timeseries-zones"
          value={polygonId || polygonDatasets[0]?.id || ''}
          onChange={event => setPolygonId(event.target.value)}
        >
          <option value="">Choose zones</option>
          {polygonDatasets.map(dataset => (
            <option key={dataset.id} value={dataset.id}>
              {dataset.label}
            </option>
          ))}
        </select>
      </label>
      <label>
        Output field (empty = primary output){' '}
        <input
          list="timeseries-outputs"
          data-testid="timeseries-output"
          value={outputName}
          onChange={event => setOutputName(event.target.value)}
        />
        <datalist id="timeseries-outputs">
          {[
            ...new Set([
              ...Object.keys(outputs),
              ...(operation === 'eof'
                ? ['covariance', 'keptCellCount', 'explainedVarianceRatio', 'eofs']
                : [])
            ])
          ].map(name => (
            <option key={name} value={name} />
          ))}
        </datalist>
      </label>
      <label>
        Output component (zero-based date, band, slot, metric or mode){' '}
        <input
          data-testid="timeseries-component"
          type="number"
          min={0}
          value={outputComponent}
          onChange={event => setOutputComponent(Number(event.target.value))}
        />
      </label>
      <p>
        Run once to list all available fields and component counts. Repeat with another
        field/component to publish it. Snapshot covariance produces a global matrix table.
      </p>
      <button
        type="button"
        data-testid="timeseries-run"
        disabled={
          running ||
          !mapping ||
          layers.length === 0 ||
          (!isRasterTimeSeriesGlobalOutput(operation, outputName) && polygonDatasets.length === 0)
        }
        onClick={async () => {
          setRunning(true);
          setStatus('Computing…');
          try {
            const settings: Record<string, number | string | boolean | readonly number[]> = {};
            for (const [name, defaultValue] of Object.entries(binding.defaults)) {
              const text = settingsText[name];
              if (text === undefined) continue;
              settings[name] = Array.isArray(defaultValue)
                ? text.trim()
                  ? text.split(',').map(Number)
                  : []
                : typeof defaultValue === 'number'
                  ? Number(text)
                  : text;
            }
            if (operation === 'standardized-calibration' || operation === 'standardized-index') {
              const choice = settingsText['zeroAtom'] ?? 'default';
              if (choice === 'on' || choice === 'off') settings['zeroAtom'] = choice === 'on';
              else if (choice !== 'default')
                throw new Error('zeroAtom choice must be default, on or off');
            }
            const result = await host.run({
              rasterLayerId: layerId || layers[0]?.id || '',
              slices: JSON.parse(mapping) as RasterTimeSlice[],
              operation,
              settings: settings as RasterTimeSeriesSettings,
              polygonDatasetId: polygonId || polygonDatasets[0]?.id,
              outputName,
              outputComponent
            });
            setOutputs(result.columns);
            setStatus(
              `Published ${result.rowCount} rows (${result.skippedRowCount} skipped; gpu; ${result.contributors.join(', ')}). ${result.outputName}[${result.outputComponent}]; ${result.readbackBytes} bytes read back. Status counts: ${JSON.stringify(result.diagnostics)}. Fields: ${Object.entries(
                result.columns
              )
                .map(([name, column]) => `${name}: ${column.components} ${column.scope} components`)
                .join('; ')}`
            );
          } catch (error) {
            setStatus(error instanceof Error ? error.message : String(error));
          } finally {
            setRunning(false);
          }
        }}
      >
        {running ? 'Computing…' : 'Compute raster time series'}
      </button>
      <p role="status" data-testid="timeseries-status">
        {status}
      </p>
    </details>
  );
}
