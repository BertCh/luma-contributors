// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import * as arrow from 'apache-arrow';
import {
  createKeplerEmbeddingFixture,
  createKeplerEmbeddingDemoOptions,
  runKeplerRasterEmbedding,
  makeRasterEmbeddingCatalogOutput,
  runKeplerLagrangianDiagnostics,
  type KeplerEmbeddingRaster,
  type RasterEmbeddingOptions
} from '@luma-contributors/kepler';
import type {NumericRasterSource, NumericRasterTile} from './raster-tile-ingest';
import type {FlowFieldAnalysisSnapshot} from './flow-field-grid';
import {loadNumericGeoTiffFromUrl} from './raster-live-source';
import {makeArrowTableFromCatalogTable} from './analysis-catalog-table';
import {prepareFlowFieldLagrangian, unprojectFlowPosition} from './flow-field-lagrangian';

/** Interleave GeoTIFF bands and intersect band validity. Int8 values are explicitly packed for decoding. */
export function prepareRasterEmbeddingTile(tile: NumericRasterTile): KeplerEmbeddingRaster {
  const dimensions = tile.bands.length;
  const pixelCount = tile.width * tile.height;
  if (
    dimensions < 1 ||
    dimensions > 128 ||
    pixelCount > 262144 ||
    pixelCount * dimensions > 33554432
  )
    throw new Error(
      'Choose a tile with 1–128 bands, at most 262144 pixels and at most 128 MiB of Float32 vectors'
    );
  const vectors = new Float32Array(pixelCount * dimensions);
  const mask = new Uint32Array(pixelCount).fill(1);
  const bytes = tile.bands.every(band => band.values instanceof Int8Array)
    ? new Int8Array(Math.ceil(vectors.length / 4) * 4)
    : undefined;
  for (let pixel = 0; pixel < pixelCount; pixel++)
    for (let band = 0; band < dimensions; band++) {
      const source = tile.bands[band];
      const offset = pixel * dimensions + band;
      vectors[offset] = source.values[pixel];
      if (!source.validity[pixel] || !Number.isFinite(vectors[offset])) mask[pixel] = 0;
      if (bytes) bytes[offset] = source.values[pixel];
    }
  return {
    width: tile.width,
    height: tile.height,
    dimensions,
    vectors,
    mask,
    quantized: bytes ? new Uint32Array(bytes.buffer) : undefined
  };
}

export function createRasterEmbeddingsHost(options: {
  getDevice(): Device | undefined;
  getSource(layerId: string): NumericRasterSource | undefined;
  listFlowGrids?(): readonly {id: string; label: string}[];
  readFlowGrid?(layerId: string): Promise<FlowFieldAnalysisSnapshot>;
  publishTable(options: {table: arrow.Table; label: string; geometryColumn: string}): Promise<void>;
}) {
  let loadedSource: NumericRasterSource | undefined;
  let generation = 0;
  let destroyed = false;
  const getSource = (id: string) =>
    id === 'loaded-geotiff' ? loadedSource : options.getSource(id);
  const readRaster = (sourceId: string, tileId?: string) => {
    if (sourceId === 'fixture') return createKeplerEmbeddingFixture();
    const source = getSource(sourceId);
    const tile = source?.tiles.find(candidate => candidate.id === tileId);
    if (!tile) throw new Error('Select a loaded multiband raster tile');
    return prepareRasterEmbeddingTile(tile);
  };
  const requireDevice = () => {
    if (destroyed) throw new Error('Analysis host has been destroyed');
    const device = options.getDevice();
    if (!device || device.type !== 'webgpu')
      throw new Error('Embedding and flow diagnostics require the active WebGPU device');
    return device;
  };
  const checkCurrent = (device: Device, runGeneration: number) => {
    if (destroyed || options.getDevice() !== device || runGeneration !== generation)
      throw new Error('Analysis cancelled because its device or source changed');
  };
  return {
    getLoadedSource: () => loadedSource,
    listTiles: (id: string) =>
      getSource(id)?.tiles.map(tile => ({
        id: tile.id,
        label: `${tile.id} (${tile.width} × ${tile.height}, ${tile.bands.length} bands)`
      })) ?? [],
    listFlowGrids: () => options.listFlowGrids?.() ?? [],
    loadGeoTiff: async (url: string, window?: readonly [number, number, number, number]) => {
      const current = ++generation;
      if (
        window &&
        (!window.every(Number.isFinite) || window[2] <= window[0] || window[3] <= window[1])
      )
        throw new Error('GeoTIFF window must be [west, south, east, north] in the source CRS');
      const source = await loadNumericGeoTiffFromUrl(url, 'embedding-geotiff', undefined, {
        overviewLevel: 0,
        window,
        maximumOutputPixelCount: 262144,
        maximumOutputBandCount: 128,
        maximumOutputElementCount: 33554432
      });
      if (destroyed || current !== generation) throw new Error('Raster load cancelled');
      loadedSource = source;
      return {tileCount: source.tiles.length, bandCount: source.bandCount};
    },
    run: async (request: {
      sourceId: string;
      tileId?: string;
      options: RasterEmbeddingOptions;
      comparisonSourceId?: string;
      comparisonTileId?: string;
      demo?: boolean;
      publish?: boolean;
      encoding?: 'float32' | 'alphaearth-int8';
    }) => {
      const device = requireDevice();
      const current = ++generation;
      let raster = readRaster(request.sourceId, request.tileId);
      const prefix: string[] = [];
      const decode = async (input: KeplerEmbeddingRaster) => {
        const decoded = await runKeplerRasterEmbedding(device, input, {operation: 'dequantize'});
        prefix.push(...decoded.contributors);
        const vectors = new Float32Array(input.vectors.length);
        for (let pixel = 0; pixel < input.width * input.height; pixel++)
          for (let band = 0; band < input.dimensions; band++)
            vectors[pixel * input.dimensions + band] = decoded.columns[`band_${band + 1}`][pixel];
        return {...input, vectors, mask: decoded.columns['usable'] as Uint32Array};
      };
      if (
        request.sourceId !== 'fixture' &&
        request.encoding === 'alphaearth-int8' &&
        request.options.operation !== 'dequantize'
      )
        raster = await decode(raster);
      const settings = request.demo
        ? {
            ...createKeplerEmbeddingDemoOptions(raster, request.options.operation),
            ...request.options
          }
        : {...request.options};
      if (request.comparisonSourceId) {
        const source = getSource(request.sourceId);
        const comparison = getSource(request.comparisonSourceId);
        const tile = source?.tiles.find(candidate => candidate.id === request.tileId);
        const afterTile = comparison?.tiles.find(
          candidate => candidate.id === request.comparisonTileId
        );
        if (
          !tile ||
          !afterTile ||
          source?.crs !== comparison?.crs ||
          tile.extent.some((value, index) => value !== afterTile.extent[index])
        )
          throw new Error(
            'Change requires the same CRS and identical tile extents; reprojection is not implicit'
          );
        const after = readRaster(request.comparisonSourceId, request.comparisonTileId);
        settings.after = request.encoding === 'alphaearth-int8' ? await decode(after) : after;
      }
      if (settings.operation === 'dequantize' && request.sourceId === 'fixture') {
        const encoded = await runKeplerRasterEmbedding(device, raster, {operation: 'quantize'});
        raster.quantized = encoded.columns['packed_word'] as Uint32Array;
        prefix.push(...encoded.contributors);
      }
      if (
        settings.operation === 'change' &&
        request.sourceId !== 'fixture' &&
        !request.comparisonSourceId
      )
        throw new Error('Choose an aligned comparison tile for change');
      const result = await runKeplerRasterEmbedding(device, raster, settings);
      checkCurrent(device, current);
      result.contributors.unshift(...prefix);
      const output = makeRasterEmbeddingCatalogOutput(result);
      if (request.publish !== false)
        await options.publishTable({
          table: makeArrowTableFromCatalogTable(output.tables[0]),
          label: `Embedding ${settings.operation} · ${request.sourceId}`,
          geometryColumn: ''
        });
      return {
        rowCount: result.rowCount,
        layout: result.layout,
        contributors: result.contributors,
        columns: Object.keys(result.columns)
      };
    },
    runFlow: async (request: {
      layerId: string;
      endLayerId?: string;
      timeStep: number;
      stepCount: number;
      publish?: boolean;
    }) => {
      const device = requireDevice();
      const current = ++generation;
      if (!options.readFlowGrid) throw new Error('Flow grid snapshots are unavailable');
      const grid = await options.readFlowGrid(request.layerId);
      const prepared = prepareFlowFieldLagrangian(grid);
      if (request.endLayerId) {
        const endGrid = await options.readFlowGrid(request.endLayerId);
        if (
          endGrid.columns !== grid.columns ||
          endGrid.rows !== grid.rows ||
          endGrid.bounds.some((value, index) => value !== grid.bounds[index])
        )
          throw new Error('Time slices must have identical flow grid dimensions and bounds');
        prepared.field.velocitiesEnd = prepareFlowFieldLagrangian(endGrid).field.velocitiesStart;
      }
      const result = await runKeplerLagrangianDiagnostics(device, prepared.field, request);
      checkCurrent(device, current);
      const geometry: (string | null)[] = [];
      const finalLongitude: (number | null)[] = [];
      const finalLatitude: (number | null)[] = [];
      for (let index = 0; index < result.ftle.length; index++) {
        const seed = unprojectFlowPosition(
          prepared.origin[0] + result.seeds[index * 2],
          prepared.origin[1] + result.seeds[index * 2 + 1]
        );
        geometry.push(JSON.stringify({type: 'Point', coordinates: seed}));
        const final = unprojectFlowPosition(
          prepared.origin[0] + result.positions[index * 2],
          prepared.origin[1] + result.positions[index * 2 + 1]
        );
        finalLongitude.push(result.flowValidity[index] ? final[0] : null);
        finalLatitude.push(result.flowValidity[index] ? final[1] : null);
      }
      const table = new arrow.Table({
        seed_index: arrow.vectorFromArray(
          Array.from({length: result.ftle.length}, (_, index) => index),
          new arrow.Uint32()
        ),
        final_longitude: arrow.vectorFromArray(finalLongitude, new arrow.Float64()),
        final_latitude: arrow.vectorFromArray(finalLatitude, new arrow.Float64()),
        ftle: arrow.vectorFromArray(
          Array.from(result.ftle, (value, index) => (result.validity[index] ? value : null)),
          new arrow.Float32()
        ),
        flow_valid: arrow.vectorFromArray(Array.from(result.flowValidity), new arrow.Uint32()),
        ftle_valid: arrow.vectorFromArray(Array.from(result.validity), new arrow.Uint32()),
        geometry: arrow.vectorFromArray(geometry, new arrow.Utf8())
      });
      if (request.publish !== false)
        await options.publishTable({
          table,
          label: `Flow map + FTLE · ${request.layerId}`,
          geometryColumn: 'geometry'
        });
      return {
        rowCount: table.numRows,
        validCount: result.validity.reduce((sum, flag) => sum + flag, 0),
        contributors: result.contributors
      };
    },
    destroy: () => {
      destroyed = true;
      generation++;
      loadedSource = undefined;
    }
  };
}
