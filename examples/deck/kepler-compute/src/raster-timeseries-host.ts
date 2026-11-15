// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  runRasterTimeSeries,
  isRasterTimeSeriesGlobalOutput,
  type RasterTimeSeriesOperation,
  type RasterTimeSeriesSettings,
  type RasterTimeSeriesResult
} from '@luma-contributors/kepler';
import * as arrow from 'apache-arrow';
import {appendZonalColumns, decodeZonalPolygons} from './raster-live-zonal';
import {computeRasterZonalStatistics} from './raster-zonal';
import type {NumericRasterSource} from './raster-tile-ingest';
import {
  bindRasterTimeStack,
  createTimeSeriesOutputSource,
  type RasterTimeSlice
} from './raster-timeseries-stack';
import {readNumericGeoTIFF} from './raster-geotiff-reader';

export type RasterTimeSeriesRequest = {
  rasterLayerId: string;
  slices: readonly RasterTimeSlice[];
  operation: RasterTimeSeriesOperation;
  settings?: RasterTimeSeriesSettings;
  polygonDatasetId?: string;
  /** Defaults to the operation's primary output; component defaults to zero. */
  outputName?: string;
  outputComponent?: number;
};

const PRIMARY_OUTPUTS: Partial<Record<RasterTimeSeriesOperation, string>> = {
  'climate-anomaly': 'anomaly',
  'standardized-index': 'index',
  phenology: 'metrics',
  eof: 'eofs',
  streaming: 'mean'
};

export function createRasterTimeSeriesHost(options: {
  getDevice(): Device | undefined;
  getSource(layerId: string): NumericRasterSource | undefined;
  getPolygonSource(datasetId: string): {table: arrow.Table; geometryColumn: string} | undefined;
  publishTable(options: {
    table: arrow.Table;
    label: string;
    geometryColumn: string;
    isCurrent?(): boolean;
  }): Promise<void>;
}) {
  let revision = 0;
  let latest: {result: RasterTimeSeriesResult; source?: NumericRasterSource} | undefined;
  let running = false;
  let isLatestCurrent: (() => boolean) | undefined;
  const imported = new Map<string, NumericRasterSource>();
  const getSource = (layerId: string) => imported.get(layerId) ?? options.getSource(layerId);
  return {
    /** CPU GeoTIFF decode uses the same reader as the existing raster host; no new format support. */
    importGeoTIFF(id: string, bytes: Uint8Array) {
      const source = readNumericGeoTIFF(bytes, id);
      if (!imported.has(id) && imported.size >= 16)
        throw new Error('At most 16 time-stack GeoTIFF sources may be retained');
      imported.set(id, source);
      revision++;
      latest = undefined;
      isLatestCurrent = undefined;
      return {id, bandCount: source.bandCount};
    },
    listImportedSources: () => [...imported.keys()],
    clear() {
      revision++;
      latest = undefined;
      imported.clear();
      isLatestCurrent = undefined;
    },
    getLatest: () => {
      if (latest && !isLatestCurrent?.()) latest = undefined;
      return latest;
    },
    async run(request: RasterTimeSeriesRequest) {
      if (running) throw new Error('A raster time-series run is already active');
      request = structuredClone(request);
      const device = options.getDevice();
      if (!device) throw new Error('Raster time series requires a GPU device');
      const bound = bindRasterTimeStack(request.rasterLayerId, request.slices, getSource);
      const needsZones = !isRasterTimeSeriesGlobalOutput(request.operation, request.outputName);
      const polygons =
        needsZones && request.polygonDatasetId
          ? options.getPolygonSource(request.polygonDatasetId)
          : undefined;
      if (needsZones && (!polygons || !polygons.table.getChild(polygons.geometryColumn)))
        throw new Error('Select a published polygon dataset for zonal output');
      if (
        polygons &&
        (polygons.table.numRows > 1024 ||
          polygons.table.schema.fields.some(field => field.name.startsWith('zonal_')))
      )
        throw new Error(
          'Choose an original polygon source with at most 1024 rows and no zonal_* columns'
        );
      const geometry = polygons?.table.getChild(polygons.geometryColumn);
      const decodedZones = geometry
        ? Array.from({length: polygons!.table.numRows}, (_, row) =>
            decodeZonalPolygons(geometry.get(row))
          )
        : [];
      if (needsZones && !decodedZones.some(zone => zone?.length))
        throw new Error('No supported polygon rows; time-series output was not published');
      let skippedRowCount = 0;
      const sourceIsCurrent = () => {
        try {
          if (options.getDevice() !== device) return false;
          const refreshed = bindRasterTimeStack(request.rasterLayerId, request.slices, getSource);
          return (
            bound.sources.every((source, index) => source === refreshed.sources[index]) &&
            refreshed.stack.values.length === bound.stack.values.length &&
            refreshed.stack.values.every((value, index) =>
              Object.is(value, bound.stack.values[index])
            )
          );
        } catch {
          return false;
        }
      };
      const polygonSourceIsCurrent = () => {
        if (!polygons) return true;
        const current = options.getPolygonSource(request.polygonDatasetId!);
        return (
          current?.table === polygons.table && current.geometryColumn === polygons.geometryColumn
        );
      };
      const currentRevision = ++revision;
      const publicationIsCurrent = () =>
        currentRevision === revision && sourceIsCurrent() && polygonSourceIsCurrent();
      latest = undefined;
      isLatestCurrent = undefined;
      running = true;
      try {
        const result = await runRasterTimeSeries(
          device,
          bound.stack,
          request.operation,
          request.settings
        );
        if (currentRevision !== revision || options.getDevice() !== device)
          throw new Error('Raster time-series run invalidated before publication');
        if (!sourceIsCurrent()) throw new Error('Raster source changed while computing; run again');
        const outputName =
          request.outputName ||
          PRIMARY_OUTPUTS[request.operation] ||
          Object.keys(result.columns)[0];
        const column = result.columns[outputName];
        if (!column)
          throw new Error(
            `Unknown output ${outputName}; choose ${Object.keys(result.columns).join(', ')}`
          );
        const component = request.outputComponent ?? 0;
        let table: arrow.Table;
        let geometryColumn = '';
        let source: NumericRasterSource | undefined;
        if (column.scope === 'global') {
          const columns: Record<string, arrow.Vector> = {
            row_index: arrow.vectorFromArray(
              Array.from({length: column.values.length / column.components}, (_, index) => index),
              new arrow.Int32()
            )
          };
          for (let entry = 0; entry < column.components; entry++)
            columns[`value_${entry}`] = arrow.vectorFromArray(
              Array.from({length: column.values.length / column.components}, (_, row) => {
                const value = column.values[row * column.components + entry];
                return Number.isFinite(value) ? value : null;
              }),
              new arrow.Float64()
            );
          table = new arrow.Table(columns);
        } else {
          source = createTimeSeriesOutputSource(
            bound.template,
            column.values,
            column.components,
            component,
            outputName
          );
          if (!polygons || !polygonSourceIsCurrent())
            throw new Error('Polygon source changed before publication');
          const statistics = Array.from({length: polygons.table.numRows}, (_, row) => {
            const decoded = decodedZones[row];
            if (!decoded?.length) {
              skippedRowCount++;
              return null;
            }
            const statistics = computeRasterZonalStatistics(source!, decoded, {
              bandIndex: 1,
              maximumPixelCount: 65536
            });
            if (statistics.isTruncated)
              throw new Error('Zonal output exceeded the complete-raster pixel bound');
            if (statistics.scannedCount === 0) {
              skippedRowCount++;
              return null;
            }
            return statistics;
          });
          if (skippedRowCount === polygons.table.numRows)
            throw new Error(
              'No polygon row intersects a loaded raster tile; time-series output was not published'
            );
          table = appendZonalColumns(polygons.table, statistics);
          geometryColumn = polygons.geometryColumn;
        }
        const provenance = JSON.stringify({
          operation: request.operation,
          settings: result.settings,
          slices: request.slices,
          outputName,
          component,
          contributors: result.contributors,
          readbackBytes: result.readbackBytes
        });
        table = new arrow.Table(
          new arrow.Schema(
            table.schema.fields,
            new Map([...table.schema.metadata, ['raster_timeseries', provenance]])
          ),
          table.batches
        );
        await options.publishTable({
          table,
          label: `Time series ${request.operation}: ${outputName}[${component}]`,
          geometryColumn,
          isCurrent: publicationIsCurrent
        });
        if (!publicationIsCurrent())
          throw new Error(
            `${polygonSourceIsCurrent() ? 'Raster' : 'Polygon'} source changed during publication`
          );
        latest = {result, source};
        isLatestCurrent = publicationIsCurrent;
        return {
          published: true,
          rowCount: table.numRows,
          skippedRowCount,
          engine: 'gpu' as const,
          contributors: result.contributors,
          readbackBytes: result.readbackBytes,
          outputName,
          outputComponent: component,
          diagnostics: Object.fromEntries(
            Object.entries(result.columns)
              .filter(
                ([name]) => /status/i.test(name) || name === 'summary' || name === 'cellSummary'
              )
              .map(([name, column]) => {
                const values =
                  name === 'summary' || name === 'cellSummary'
                    ? Array.from(
                        {length: column.values.length / column.components},
                        (_, row) =>
                          column.values[row * column.components + (name === 'summary' ? 0 : 1)]
                      )
                    : Array.from(column.values);
                return [
                  name,
                  values.reduce<Record<string, number>>((counts, value) => {
                    const code = String(value);
                    counts[code] = (counts[code] ?? 0) + 1;
                    return counts;
                  }, {})
                ];
              })
          ),
          columns: Object.fromEntries(
            Object.entries(result.columns).map(([name, column]) => [
              name,
              {components: column.components, scope: column.scope}
            ])
          )
        };
      } finally {
        running = false;
      }
    }
  };
}
