// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {computeTerrainGridOnGpu} from '@luma-contributors/kepler';
import {
  RASTER_TERRAIN_NODATA,
  computeRasterTerrainGrid,
  type RasterTerrainGrid,
  type RasterTerrainOptions
} from './analysis2-raster-terrain';
import {
  createNumericRasterSource,
  createNumericRasterTile,
  type NumericRasterSource,
  type NumericRasterTile
} from './raster-tile-ingest';
import {
  computeRasterZonalStatistics,
  type RasterZonalOptions,
  type RasterZonalPolygon,
  type RasterZonalStatistics
} from './raster-zonal';

/**
 * GPU routing of the terrain operators (slope, aspect, hillshade, curvature) over a
 * NumericRasterSource. Each tile is padded with a one-sample halo from its same-zoom neighbors
 * so seams between tiles are computed instead of marked nodata; tile borders without a
 * neighbor stay nodata exactly like the CPU reference. The CPU implementation in
 * analysis2-raster-terrain.ts is the oracle and the fallback.
 */
export type RasterTerrainEngine = 'gpu' | 'cpu-reference';

export type RasterTerrainRunOptions = Readonly<{
  device?: Device;
  /** Forces the CPU reference even when a device exists. */
  engine?: 'cpu';
  /** Fills tile seams from neighboring tiles. Defaults to true. */
  useNeighborHalo?: boolean;
}>;

export type RasterTerrainRunResult = Readonly<{
  source: NumericRasterSource;
  engine: RasterTerrainEngine;
  /** Visible engine note, for example "Terrain slope: GPU (GPUTerrainDerivatives, 2 tiles)". */
  engineNote: string;
  fallbackReason?: string;
}>;

const NEIGHBOR_EPSILON_RATIO = 1e-6;

function getTileNeighbor(
  tiles: readonly NumericRasterTile[],
  tile: NumericRasterTile,
  columnOffset: -1 | 0 | 1,
  rowOffset: -1 | 0 | 1
): NumericRasterTile | undefined {
  const [west, south, east, north] = tile.extent;
  const tileWidth = east - west;
  const tileHeight = north - south;
  const tolerance = Math.max(tileWidth, tileHeight) * NEIGHBOR_EPSILON_RATIO;
  const targetWest = west + columnOffset * tileWidth;
  const targetNorth = north - rowOffset * tileHeight;
  return tiles.find(
    candidate =>
      candidate !== tile &&
      candidate.zoom === tile.zoom &&
      candidate.width === tile.width &&
      candidate.height === tile.height &&
      Math.abs(candidate.extent[0] - targetWest) <= tolerance &&
      Math.abs(candidate.extent[3] - targetNorth) <= tolerance
  );
}

/** Builds the (width + 2) x (height + 2) elevation and validity arrays with neighbor halo samples. */
export function padRasterTileWithHalo(
  tiles: readonly NumericRasterTile[],
  tile: NumericRasterTile,
  bandIndex: number
): {
  elevation: Float32Array;
  validity: Uint8Array;
  width: number;
  height: number;
  extent: readonly [number, number, number, number];
} {
  const {width, height} = tile;
  const paddedWidth = width + 2;
  const paddedHeight = height + 2;
  const elevation = new Float32Array(paddedWidth * paddedHeight);
  const validity = new Uint8Array(paddedWidth * paddedHeight);
  for (let paddedRow = 0; paddedRow < paddedHeight; paddedRow++) {
    const rowOffset = paddedRow === 0 ? -1 : paddedRow === paddedHeight - 1 ? 1 : 0;
    for (let paddedColumn = 0; paddedColumn < paddedWidth; paddedColumn++) {
      const columnOffset = paddedColumn === 0 ? -1 : paddedColumn === paddedWidth - 1 ? 1 : 0;
      const owner =
        rowOffset === 0 && columnOffset === 0
          ? tile
          : getTileNeighbor(tiles, tile, columnOffset, rowOffset);
      const band = owner?.bands.find(candidate => candidate.index === bandIndex);
      if (!owner || !band) continue;
      const column = columnOffset === -1 ? width - 1 : columnOffset === 1 ? 0 : paddedColumn - 1;
      const row = rowOffset === -1 ? height - 1 : rowOffset === 1 ? 0 : paddedRow - 1;
      const sourceIndex = row * width + column;
      const paddedIndex = paddedRow * paddedWidth + paddedColumn;
      if (band.validity[sourceIndex]) {
        elevation[paddedIndex] = band.values[sourceIndex];
        validity[paddedIndex] = 1;
      }
    }
  }
  const [west, south, east, north] = tile.extent;
  const pixelWidth = (east - west) / width;
  const pixelHeight = (north - south) / height;
  return {
    elevation,
    validity,
    width: paddedWidth,
    height: paddedHeight,
    extent: [west - pixelWidth, south - pixelHeight, east + pixelWidth, north + pixelHeight]
  };
}

function cropPaddedGrid(
  grid: {values: Float32Array; validity: Uint8Array},
  width: number,
  height: number
): RasterTerrainGrid {
  const values = new Float32Array(width * height);
  const validity = new Uint8Array(width * height);
  const paddedWidth = width + 2;
  for (let row = 0; row < height; row++)
    for (let column = 0; column < width; column++) {
      const paddedIndex = (row + 1) * paddedWidth + column + 1;
      values[row * width + column] = grid.values[paddedIndex];
      validity[row * width + column] = grid.validity[paddedIndex];
    }
  return {width, height, values, validity};
}

function makeTerrainSource(
  source: NumericRasterSource,
  options: RasterTerrainOptions,
  grids: readonly RasterTerrainGrid[]
): NumericRasterSource {
  return createNumericRasterSource({
    id: `${source.id}:${options.operation}`,
    kind: source.kind,
    crs: source.crs,
    tiles: source.tiles.map((tile, tileIndex) => {
      const grid = grids[tileIndex];
      const stored = new Float32Array(grid.values);
      for (let sampleIndex = 0; sampleIndex < stored.length; sampleIndex++)
        if (!grid.validity[sampleIndex]) stored[sampleIndex] = RASTER_TERRAIN_NODATA;
      return createNumericRasterTile({
        id: `${tile.id}:${options.operation}`,
        width: tile.width,
        height: tile.height,
        extent: tile.extent,
        zoom: tile.zoom,
        tileIndex: tile.tileIndex,
        bands: [
          {index: 1, name: options.operation, values: stored, noDataValue: RASTER_TERRAIN_NODATA}
        ]
      });
    })
  });
}

/** Terrain source on the GPU when a device is given, else (or on any GPU failure) the CPU reference. */
export async function createTerrainRasterSourceAsync(
  source: NumericRasterSource,
  options: RasterTerrainOptions,
  runOptions: RasterTerrainRunOptions = {}
): Promise<RasterTerrainRunResult> {
  const label = `Terrain ${options.operation}`;
  const bandIndex = options.bandIndex ?? 1;
  const useNeighborHalo = runOptions.useNeighborHalo ?? true;
  const makeCpuResult = (reason: string | undefined): RasterTerrainRunResult => ({
    // Use the same neighbor halo AND per-row ground spacing as the GPU. The old fallback
    // invalidated every tile seam, changing zonal populations when GPU execution was unavailable.
    source: makeTerrainSource(
      source,
      options,
      source.tiles.map(tile => {
        const padded = useNeighborHalo
          ? padRasterTileWithHalo(source.tiles, tile, bandIndex)
          : undefined;
        const input = padded
          ? {
              ...tile,
              width: padded.width,
              height: padded.height,
              extent: padded.extent,
              bands: tile.bands
                .filter(band => band.index === bandIndex)
                .map(band => ({...band, values: padded.elevation, validity: padded.validity}))
            }
          : tile;
        const grid = computeRasterTerrainGrid(input, source.crs, {
          ...options,
          groundModel: 'wgs84-per-row'
        });
        return padded ? cropPaddedGrid(grid, tile.width, tile.height) : grid;
      })
    ),
    engine: 'cpu-reference',
    engineNote: reason ? `${label}: CPU reference (${reason})` : `${label}: CPU reference`,
    fallbackReason: reason
  });
  if (runOptions.engine === 'cpu') return makeCpuResult(undefined);
  if (!runOptions.device) return makeCpuResult('no GPU device');
  try {
    const grids: RasterTerrainGrid[] = [];
    for (const tile of source.tiles) {
      const band = tile.bands.find(candidate => candidate.index === bandIndex);
      // Binding error: the requested elevation band is not part of this tile.
      if (!band) throw new Error(`Raster tile ${tile.id} has no band ${bandIndex}`);
      const padded = useNeighborHalo
        ? padRasterTileWithHalo(source.tiles, tile, bandIndex)
        : {
            elevation: Float32Array.from(band.values),
            validity: band.validity,
            width: tile.width,
            height: tile.height,
            extent: tile.extent
          };
      const result = await computeTerrainGridOnGpu(runOptions.device, {
        ...padded,
        crs: source.crs,
        operation: options.operation,
        zFactor: options.zFactor,
        azimuthDegrees: options.azimuthDegrees,
        altitudeDegrees: options.altitudeDegrees
      });
      grids.push(useNeighborHalo ? cropPaddedGrid(result, tile.width, tile.height) : result);
    }
    const contributorName =
      options.operation === 'curvature' ? 'GPUTerrainCurvature' : 'GPUTerrainDerivatives';
    return {
      source: makeTerrainSource(source, options, grids),
      engine: 'gpu',
      engineNote: `${label}: GPU (${contributorName}, ${source.tiles.length} tile${source.tiles.length === 1 ? '' : 's'}${useNeighborHalo ? ', neighbor halo' : ''}; WGS84 ground cell size)`
    };
  } catch (error) {
    return makeCpuResult(error instanceof Error ? error.message : 'GPU terrain failed');
  }
}

/** Zonal statistics of a terrain derivative over polygons, GPU first with CPU fallback. */
export async function computeRasterTerrainZonalStatisticsAsync(
  source: NumericRasterSource,
  polygons: readonly RasterZonalPolygon[],
  terrain: RasterTerrainOptions,
  runOptions: RasterTerrainRunOptions = {},
  zonalOptions: Omit<RasterZonalOptions, 'bandIndex' | 'expression'> = {}
): Promise<RasterTerrainRunResult & {statistics: RasterZonalStatistics}> {
  const result = await createTerrainRasterSourceAsync(source, terrain, runOptions);
  return {
    ...result,
    statistics: computeRasterZonalStatistics(result.source, polygons, {
      ...zonalOptions,
      bandIndex: 1
    })
  };
}
