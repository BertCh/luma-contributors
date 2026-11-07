// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {KeplerDatasetRegistry} from '@luma-contributors/kepler';
import type * as arrow from 'apache-arrow';
import {AUTO_PREPARE_FLOAT64_POLICY} from './auto-prepare-plan';
import {BitmapLiveBinding} from './raster-live-bitmap';
import {RasterLiveController, type KeplerRasterLayerLike} from './raster-live-controller';
import {loadNumericGeoTiffFromUrl} from './raster-live-source';
import {installRasterIdentifyTooltip} from './raster-live-tooltip';
import {computeZonalTable} from './raster-live-zonal';
import {computeZonalTableGpu} from './raster-live-zonal-gpu';
import {computeTerrainZonalTable} from './raster-terrain-zonal-action';
import type {RasterTerrainOperation} from './analysis2-raster-terrain';
import {augmentIdentifyWithGpuSample, RasterGpuSampleCache} from './raster-gpu-identify';
import {RasterWgslWrapper} from './raster-wgsl-wrap';
import {
  createRasterTerrainHost,
  type DerivedGridReader,
  type RasterTerrainHostOptions
} from './analysis4-raster-terrain-host';
import {WmsLiveController} from './wms-live-controller';

type LayerLike = {id: string; type: string; config?: unknown};

/** What the host application provides; everything else lives in the raster-live-* files. */
export type RasterLiveHostOptions = {
  getLayers(): readonly LayerLike[];
  getDevice(): Device | undefined;
  getMemoryRegistry?(): KeplerDatasetRegistry | undefined;
  getDeckLayers(): readonly unknown[];
  /** The original Arrow table and geometry column of a published (resident) polygon dataset. */
  getPolygonSource(datasetId: string): {table: arrow.Table; geometryColumn: string} | undefined;
  /** Published polygon datasets offered to the zonal panel. */
  listPolygonDatasets?(): readonly {id: string; label: string}[];
  /** Current IDW or kriging grid, used as a terrain input. */
  getDerivedGrid?(): DerivedGridReader | undefined;
  /** GPU-resident terrain publication (no cell readback); see RasterTerrainHostOptions.publishResident. */
  publishResidentTerrain?: RasterTerrainHostOptions['publishResident'];
  /** Uploads a table through the existing publication path (`publishSource`). */
  publishTable(options: {
    table: arrow.Table;
    label: string;
    geometryColumn: string;
    normalization: typeof NORMALIZATION;
  }): Promise<void>;
};

const NORMALIZATION = {float64: AUTO_PREPARE_FLOAT64_POLICY};

export type RasterZonalActionResult = Readonly<{
  published: boolean;
  label: string;
  rowCount: number;
  skippedRowCount: number;
  reason?: string;
  /** Which engine produced the statistics. */
  engine?: 'gpu' | 'cpu' | 'cpu-fallback';
  fallbackReason?: string;
}>;

/** Dev-only stand-in for a Kepler raster tile layer; lets browser scripts drive the raster panels offline. */
class InjectedRasterTestLayer {
  readonly type = 'rasterTile';
  readonly config: {label: string; isVisible: boolean};
  constructor(
    readonly id: string,
    label: string,
    readonly _loadedTiles: readonly unknown[]
  ) {
    this.config = {label, isVisible: true};
  }
  getHoverData(): null {
    return null;
  }
}

export function createRasterLiveHost(hostOptions: RasterLiveHostOptions) {
  const injectedLayers: InjectedRasterTestLayer[] = [];
  const options: RasterLiveHostOptions = {
    ...hostOptions,
    getLayers: () => [...hostOptions.getLayers(), ...injectedLayers]
  };
  const raster = new RasterLiveController({
    getLayers: () => options.getLayers() as unknown as readonly KeplerRasterLayerLike[],
    getDevice: options.getDevice,
    getMemoryRegistry: options.getMemoryRegistry
  });
  const getResidentResource = (tileId: string, bandIndex: number) =>
    raster.getResidentBandResource(tileId.split('/').slice(0, -3).join('/'), tileId, bandIndex);
  const gpuSampleCache = new RasterGpuSampleCache(getResidentResource);
  const rasterWgsl = new RasterWgslWrapper(raster);
  const wms = new WmsLiveController({
    getLayers: options.getLayers,
    getDeckLayers: options.getDeckLayers
  });
  const bitmap = new BitmapLiveBinding({
    getLayers: () => options.getLayers() as never
  });

  const terrain = createRasterTerrainHost({
    getDevice: options.getDevice,
    getTilesSource: layerId => raster.getSource(layerId),
    getDerivedGrid: options.getDerivedGrid,
    publishTable: ({table, label, geometryColumn}) =>
      options.publishTable({table, label, geometryColumn, normalization: NORMALIZATION}),
    publishResident: options.publishResidentTerrain
  });

  /** Installs the native-tooltip and click patches once a layer of the type exists. */
  function syncLayers(): void {
    const layers = options.getLayers();
    const rasterLayer = layers.find(layer => layer.type === 'rasterTile');
    if (rasterLayer) installRasterIdentifyTooltip(rasterLayer, raster, gpuSampleCache);
    const wmsLayer = layers.find(layer => layer.type === 'wms');
    if (wmsLayer) wms.installIdentify(wmsLayer);
    bitmap.pruneRemovedLayers();
  }

  /** Runnable zonal-statistics action: raster layer x native polygon dataset -> new native dataset. */
  async function runZonalAction(request: {
    rasterLayerId: string;
    polygonDatasetId: string;
    bandIndex?: number;
    label?: string;
    /** Default 'gpu' (resident textures + GPURasterZonalStatistics, CPU fallback). */
    engine?: 'gpu' | 'cpu';
    /** Statistics of this terrain derivative of the raster (band 1 as elevation) instead of its bands. */
    terrainOperation?: RasterTerrainOperation;
  }): Promise<RasterZonalActionResult> {
    const label = request.label ?? `Zonal ${request.rasterLayerId}`;
    const polygons = options.getPolygonSource(request.polygonDatasetId);
    if (!polygons)
      return {
        published: false,
        label,
        rowCount: 0,
        skippedRowCount: 0,
        reason: 'Polygon dataset is not a published geometry source'
      };
    if (request.terrainOperation) {
      const source = raster.getSource(request.rasterLayerId);
      if (!source || source.tiles.length === 0)
        return {
          published: false,
          label,
          rowCount: 0,
          skippedRowCount: 0,
          reason: 'No loaded raster tiles'
        };
      const terrainResult = await computeTerrainZonalTable({
        source,
        device: request.engine === 'cpu' ? undefined : options.getDevice(),
        table: polygons.table,
        geometryColumn: polygons.geometryColumn,
        terrain: {operation: request.terrainOperation, bandIndex: request.bandIndex}
      });
      const terrainEngine = terrainResult.engine === 'gpu' ? 'gpu' : 'cpu';
      if (!terrainResult.table)
        return {
          published: false,
          label,
          rowCount: 0,
          skippedRowCount: terrainResult.skippedRowCount,
          reason: terrainResult.reason,
          engine: terrainEngine,
          fallbackReason: terrainResult.fallbackReason
        };
      await options.publishTable({
        table: terrainResult.table,
        label,
        geometryColumn: polygons.geometryColumn,
        normalization: NORMALIZATION
      });
      return {
        published: true,
        label,
        rowCount: terrainResult.table.numRows,
        skippedRowCount: terrainResult.skippedRowCount,
        engine: terrainEngine,
        fallbackReason: terrainResult.fallbackReason
      };
    }
    const zonalOptions = {
      controller: raster,
      rasterLayerId: request.rasterLayerId,
      table: polygons.table,
      geometryColumn: polygons.geometryColumn,
      bandIndex: request.bandIndex
    };
    const gpuResult =
      request.engine === 'cpu'
        ? undefined
        : await computeZonalTableGpu({...zonalOptions, device: options.getDevice()});
    const result = gpuResult ?? computeZonalTable(zonalOptions);
    const engine = gpuResult?.engine ?? 'cpu';
    const fallbackReason = gpuResult?.fallbackReason;
    if (result.reason)
      return {
        published: false,
        label,
        rowCount: 0,
        skippedRowCount: result.skippedRowCount,
        reason: result.reason,
        engine,
        fallbackReason
      };
    await options.publishTable({
      table: result.table,
      label,
      geometryColumn: polygons.geometryColumn,
      normalization: NORMALIZATION
    });
    return {
      published: true,
      label,
      rowCount: result.table.numRows,
      skippedRowCount: result.skippedRowCount,
      engine,
      fallbackReason
    };
  }

  /** What the raster tools panel lists: loaded raster and WMS layers plus polygon datasets. */
  function getPanelModel() {
    type PanelLayer = {
      id: string;
      type: string;
      config?: {label?: string; visConfig?: {wmsLayer?: {name?: string} | null}};
      meta?: {tilesetDataUrl?: string; wmsVersion?: string};
    };
    const layers = options.getLayers() as readonly PanelLayer[];
    const getLabel = (layer: PanelLayer) => layer.config?.label || layer.id;
    return {
      rasterLayers: layers
        .filter(layer => layer.type === 'rasterTile')
        .map(layer => ({id: layer.id, label: getLabel(layer)})),
      wmsLayers: layers
        .filter(layer => layer.type === 'wms')
        .map(layer => ({
          id: layer.id,
          label: getLabel(layer),
          serviceUrl: layer.meta?.tilesetDataUrl,
          layerName: layer.config?.visConfig?.wmsLayer?.name,
          version: layer.meta?.wmsVersion
        })),
      bitmapLayers: layers
        .filter(layer => layer.type === 'bitmap')
        .map(layer => ({id: layer.id, label: getLabel(layer)})),
      polygonDatasets: options.listPolygonDatasets?.() ?? [],
      isDerivedGridAvailable: terrain.isDerivedGridAvailable()
    };
  }

  const testHooks = {
    getRasterToolsPanelModel: getPanelModel,
    describeRasterPolygonSource: (datasetId: string) => {
      const polygons = options.getPolygonSource(datasetId);
      const column = polygons?.table.getChild(polygons.geometryColumn);
      return polygons && column
        ? {
            geometryColumn: polygons.geometryColumn,
            type: String(column.type),
            rowCount: polygons.table.numRows,
            columnNames: polygons.table.schema.fields.map(field => field.name),
            firstRow: JSON.stringify(column.get(0), (_, value) =>
              typeof value === 'bigint' ? Number(value) : value
            )?.slice(0, 300)
          }
        : null;
    },
    readRasterPolygonColumn: (datasetId: string, columnName: string) => {
      const column = options.getPolygonSource(datasetId)?.table.getChild(columnName);
      return column ? Array.from(column.toArray() as ArrayLike<number>) : null;
    },
    /** Adds a synthetic raster tile layer (loaded Kepler-shaped tiles) for offline browser checks. */
    injectRasterTestLayer: (layerId: string, label: string, loadedTiles: readonly unknown[]) => {
      injectedLayers.push(new InjectedRasterTestLayer(layerId, label, loadedTiles));
      syncLayers();
    },
    syncRasterLiveLayers: syncLayers,
    getRasterLiveSnapshot: (layerId: string) => raster.getSnapshot(layerId) ?? null,
    identifyRasterLive: (layerId: string, longitude: number, latitude: number) =>
      raster.identify(layerId, longitude, latitude) ?? null,
    runRasterZonalAction: runZonalAction,
    runRasterTerrainAction: terrain.runTerrainAction,
    exportRasterTerrainGeoTIFF: async () => {
      const {data, filename} = await terrain.exportTerrainGeoTIFF();
      return {filename, bytes: Array.from(data)};
    },
    identifyRasterLiveGpu: async (layerId: string, longitude: number, latitude: number) => {
      const cpu = raster.identify(layerId, longitude, latitude);
      return cpu ? augmentIdentifyWithGpuSample(cpu, getResidentResource) : null;
    },
    getRasterWgslStatistics: () => ({...rasterWgsl.statistics}),
    loadGeoTiffSource: (url: string) => loadNumericGeoTiffFromUrl(url, 'geotiff-test'),
    setWmsLiveTime: (layerId: string, time: string | null, extentText?: string) =>
      wms.setTime(layerId, time, extentText),
    getWmsLiveState: (layerId: string) => wms.getState(layerId),
    getBitmapLiveGeoreferences: () => bitmap.getGeoreferences(),
    attachBitmapGeoTiff: (layerId: string, bytes: number[]) =>
      bitmap.attachGeoTiff(layerId, Uint8Array.from(bytes)),
    identifyBitmapLive: (layerId: string, longitude: number, latitude: number) =>
      bitmap.identify(layerId, longitude, latitude) ?? null
  };

  return {
    raster,
    wms,
    bitmap,
    syncLayers,
    runZonalAction,
    terrain,
    getPanelModel,
    /** Replaces Kepler's GLSL raster tile sublayers with the WGSL draw; other layers pass through. */
    wrapLayer: <LayerType>(layer: LayerType): LayerType => rasterWgsl.wrapLayer(layer),
    testHooks
  };
}
