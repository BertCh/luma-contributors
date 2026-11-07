// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '@luma-contributors/kepler';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPURasterZonalStatistics, makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {
  GPUTerrainCurvature,
  GPUTerrainDerivatives,
  getGPUTerrainCurvatureParameterValues,
  getGPUTerrainDerivativesParameterValues
} from '@luma-contributors/analysis/gpu-terrain';
import {
  RASTER_TERRAIN_NODATA,
  computeRasterTerrainGrid,
  type RasterTerrainOptions
} from './analysis2-raster-terrain';
import {
  createTerrainRasterSourceAsync,
  padRasterTileWithHalo
} from './analysis3-raster-terrain-gpu';
import {getScanTiles} from './raster-gpu-zonal';
import {
  mergeRasterZoneColumns,
  RASTER_ZONE_NONE,
  rasterizeZoneIds,
  type RasterGpuZoneStatistics,
  type RasterZoneColumns
} from './raster-gpu-zonal-zones';
import {
  createNumericRasterSource,
  createNumericRasterTile,
  type NumericRasterSource
} from './raster-tile-ingest';
import {
  computeRasterZonalStatistics,
  type RasterZonalPolygon,
  type RasterZonalStatistics
} from './raster-zonal';

export type RasterTerrainZonalGpuOptions = Readonly<{
  device: Device;
  source: NumericRasterSource;
  /** One entry per zone: the polygons of that zone. */
  zonePolygons: readonly (readonly RasterZonalPolygon[] | null)[];
  terrain: RasterTerrainOptions;
  /** Fills tile seams from same-zoom neighbors. Defaults to true. */
  useNeighborHalo?: boolean;
}>;

export type RasterTerrainZonalGpuResult = Readonly<{
  statistics: (RasterGpuZoneStatistics | null)[];
  tileCount: number;
  /** Bytes read back from the GPU: per-zone columns only, never the derived terrain grid. */
  readbackByteLength: number;
  contributors: readonly string[];
  declinedReason?: 'overlapping-zones' | 'no-tiles';
}>;

/** Kepler hillshade is reported on 0..255; the contributor returns 0..1. */
const HILLSHADE_SCALE = 255;
/** ESRI curvature is -100 times the Zevenbergen-Thorne Laplacian. */
const CURVATURE_SCALE = -100;

export function getTerrainOutputScale(operation: RasterTerrainOptions['operation']): number {
  return operation === 'hillshade'
    ? HILLSHADE_SCALE
    : operation === 'curvature'
      ? CURVATURE_SCALE
      : 1;
}

/**
 * Per-zone statistics of slope, aspect, hillshade or curvature without reading the derived grid
 * back: each tile (with a neighbor halo) runs the terrain contributor into a device buffer that
 * `GPURasterZonalStatistics` reduces in the same submission, so only the zone columns (five
 * words per zone) return to the CPU. Halo cells carry no zone, so they never count. The scale
 * of the output unit (hillshade 0..255, curvature x -100) is linear, so it is applied to the
 * reduced columns (a negative scale swaps minimum and maximum).
 */
export async function computeRasterTerrainZonalStatisticsGpu(
  options: RasterTerrainZonalGpuOptions
): Promise<RasterTerrainZonalGpuResult> {
  const {device, source, zonePolygons, terrain} = options;
  const bandIndex = terrain.bandIndex ?? 1;
  const useNeighborHalo = options.useNeighborHalo ?? true;
  const tiles = getScanTiles(source, zonePolygons);
  const contributorName =
    terrain.operation === 'curvature' ? 'GPUTerrainCurvature' : 'GPUTerrainDerivatives';
  if (tiles.length === 0)
    return {
      statistics: zonePolygons.map(() => null),
      tileCount: 0,
      readbackByteLength: 0,
      contributors: [],
      declinedReason: 'no-tiles'
    };
  const zoneCount = zonePolygons.length;
  const scale = getTerrainOutputScale(terrain.operation);
  const tileColumns: RasterZoneColumns[] = [];
  let readbackByteLength = 0;
  for (const tile of tiles) {
    const zoneRaster = rasterizeZoneIds(tile, source.crs, zonePolygons);
    if (zoneRaster.hasOverlap)
      return {
        statistics: zonePolygons.map(() => null),
        tileCount: tiles.length,
        readbackByteLength,
        contributors: [contributorName],
        declinedReason: 'overlapping-zones'
      };
    if (zoneRaster.assignedCount === 0) continue;
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
    const columns = await reduceTerrainTile(device, {
      id: tile.id,
      source,
      padded,
      zones: padZoneIds(zoneRaster.zones, tile.width, tile.height, padded.width, padded.height),
      zoneCount,
      terrain
    });
    readbackByteLength += zoneCount * 5 * 4;
    tileColumns.push(scaleColumns(columns, scale));
  }
  return {
    statistics: mergeRasterZoneColumns(zoneCount, tileColumns),
    tileCount: tiles.length,
    readbackByteLength,
    contributors: [contributorName]
  };
}

/** Places tile zones inside the halo-padded grid; halo samples belong to no zone. */
export function padZoneIds(
  zones: Uint32Array,
  width: number,
  height: number,
  paddedWidth: number,
  paddedHeight: number
): Uint32Array {
  if (paddedWidth === width && paddedHeight === height) return zones;
  const padded = new Uint32Array(paddedWidth * paddedHeight).fill(RASTER_ZONE_NONE);
  for (let row = 0; row < height; row++)
    padded.set(zones.subarray(row * width, (row + 1) * width), (row + 1) * paddedWidth + 1);
  return padded;
}

function scaleColumns(columns: RasterZoneColumns, scale: number): RasterZoneColumns {
  if (scale === 1) return columns;
  const count = columns.sums.length;
  const sums = new Array<number>(count);
  const minimums = new Array<number>(count);
  const maximums = new Array<number>(count);
  for (let zoneIndex = 0; zoneIndex < count; zoneIndex++) {
    sums[zoneIndex] = columns.sums[zoneIndex] * scale;
    const low = columns.minimums[zoneIndex] * scale;
    const high = columns.maximums[zoneIndex] * scale;
    minimums[zoneIndex] = Math.min(low, high);
    maximums[zoneIndex] = Math.max(low, high);
  }
  return {...columns, sums, minimums, maximums};
}

async function reduceTerrainTile(
  device: Device,
  input: {
    id: string;
    source: NumericRasterSource;
    padded: {
      elevation: ArrayLike<number>;
      validity: Uint8Array;
      width: number;
      height: number;
      extent: readonly [number, number, number, number];
    };
    zones: Uint32Array;
    zoneCount: number;
    terrain: RasterTerrainOptions;
  }
): Promise<RasterZoneColumns> {
  const {padded, zoneCount, terrain} = input;
  const {width, height} = padded;
  const cellCount = width * height;
  // Unsupported tile size: a storage binding cannot hold the padded elevation band.
  if (cellCount * 4 > device.limits.maxStorageBufferBindingSize)
    throw new KeplerAllocationError('capacity', {
      message: 'Terrain tile exceeds the storage binding limit'
    });
  const [west, south, east, north] = padded.extent;
  const grid = makeGPURasterMetadata({
    width,
    height,
    cellSize: [(east - west) / width, (north - south) / height],
    origin: [west, north]
  });
  const cellSizeMode = input.source.crs === 'EPSG:3857' ? 'web-mercator' : 'geographic';
  const elevationValues = new Float32Array(cellCount);
  const validityMask = new Uint32Array(cellCount);
  for (let cellIndex = 0; cellIndex < cellCount; cellIndex++)
    if (padded.validity[cellIndex]) {
      elevationValues[cellIndex] = padded.elevation[cellIndex];
      validityMask[cellIndex] = 1;
    }
  const graph = new GPUCommandGraph(device, {id: `terrain-zonal-${input.id}`});
  const owned: Buffer[] = [];
  const makeInput = (data: Uint32Array | Float32Array): Buffer => {
    const buffer = device.createBuffer({data, usage: Buffer.STORAGE | Buffer.COPY_DST});
    owned.push(buffer);
    return buffer;
  };
  const makeOutput = (length: number): Buffer => {
    const buffer = device.createBuffer({
      byteLength: Math.max(length, 1) * 4,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    owned.push(buffer);
    return buffer;
  };
  const importView = <Format extends 'uint32' | 'float32'>(
    id: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) =>
    graph.createDataView(
      graph.importBuffer({id, byteLength: buffer.byteLength, usage: buffer.usage}, buffer),
      {format, length}
    );
  try {
    const elevation = {
      id: 'elevation',
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: importView('elevation', makeInput(elevationValues), 'float32', cellCount)
      },
      validity: importView('mask', makeInput(validityMask), 'uint32', cellCount)
    } as const;
    const terrainValues = importView('terrain-values', makeOutput(cellCount), 'float32', cellCount);
    const terrainValidity = importView(
      'terrain-validity',
      makeOutput(cellCount),
      'uint32',
      cellCount
    );
    const zFactor = terrain.zFactor ?? 1;
    if (terrain.operation === 'curvature') {
      const parameterValues = getGPUTerrainCurvatureParameterValues({zFactor}, grid);
      graph.add(
        new GPUTerrainCurvature({
          grid,
          elevation,
          parameters: importView(
            'parameters',
            makeInput(parameterValues),
            'float32',
            parameterValues.length
          ),
          method: 'zevenbergen-thorne',
          cellSizeMode,
          borderMode: 'nodata',
          output: {curvatures: {laplacian: terrainValues}, outputValidity: terrainValidity}
        })
      );
    } else {
      const parameterValues = getGPUTerrainDerivativesParameterValues(
        {
          zFactor,
          azimuthDegrees: terrain.azimuthDegrees,
          altitudeDegrees: terrain.altitudeDegrees
        },
        grid
      );
      graph.add(
        new GPUTerrainDerivatives({
          grid,
          elevation,
          parameters: importView(
            'parameters',
            makeInput(parameterValues),
            'float32',
            parameterValues.length
          ),
          cellSizeMode,
          borderMode: 'nodata',
          output: {[terrain.operation]: terrainValues, outputValidity: terrainValidity}
        })
      );
    }
    const columns = {
      cellCounts: makeOutput(zoneCount),
      valueCounts: makeOutput(zoneCount),
      sums: makeOutput(zoneCount),
      minimums: makeOutput(zoneCount),
      maximums: makeOutput(zoneCount)
    };
    graph.add(
      new GPURasterZonalStatistics({
        grid,
        zones: importView('zones', makeInput(input.zones), 'uint32', cellCount),
        values: {
          id: 'terrain',
          format: 'float32',
          storage: {kind: 'buffer', values: terrainValues},
          validity: terrainValidity
        } as const,
        zoneCapacity: zoneCount,
        ignoredZone: RASTER_ZONE_NONE,
        output: {
          cellCounts: importView('cell-counts', columns.cellCounts, 'uint32', zoneCount),
          valueCounts: importView('value-counts', columns.valueCounts, 'uint32', zoneCount),
          sums: importView('sums', columns.sums, 'float32', zoneCount),
          minimums: importView('minimums', columns.minimums, 'float32', zoneCount),
          maximums: importView('maximums', columns.maximums, 'float32', zoneCount)
        }
      })
    );
    const compiled = graph.compile();
    try {
      const commandEncoder = device.createCommandEncoder({id: `terrain-zonal-${input.id}-encoder`});
      compiled.encode(commandEncoder, {parameters: undefined});
      device.submit(commandEncoder.finish());
      const [cellCounts, valueCounts, sums, minimums, maximums] = await Promise.all(
        Object.values(columns).map(buffer => buffer.readAsync())
      );
      const copyU32 = (bytes: Uint8Array) =>
        Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, zoneCount));
      const copyF32 = (bytes: Uint8Array) =>
        Array.from(new Float32Array(bytes.buffer, bytes.byteOffset, zoneCount));
      return {
        cellCounts: copyU32(cellCounts),
        valueCounts: copyU32(valueCounts),
        sums: copyF32(sums),
        minimums: copyF32(minimums),
        maximums: copyF32(maximums)
      };
    } finally {
      compiled.destroy();
    }
  } finally {
    for (const buffer of owned) buffer.destroy();
  }
}

export type RasterTerrainZonalResult = Readonly<{
  /** One entry per zone; null for a zone that touches no loaded tile. */
  statistics: (RasterZonalStatistics | null)[];
  engine: 'gpu' | 'cpu-reference';
  /** Visible engine note, for example "Terrain slope zonal: GPU (GPUTerrainDerivatives + GPURasterZonalStatistics, 2 tiles, no derived-grid readback)". */
  engineNote: string;
  fallbackReason?: string;
  /** Bytes read back from the GPU; zone columns only on the GPU path. */
  readbackByteLength: number;
}>;

/**
 * Zonal statistics of a terrain derivative: fused GPU terrain + zonal reduction first, else the
 * CPU reference (terrain on the CPU, then zonal), with the reason recorded.
 */
export async function computeRasterTerrainZonalStatistics(
  options: Omit<RasterTerrainZonalGpuOptions, 'device'> & {device?: Device}
): Promise<RasterTerrainZonalResult> {
  const label = `Terrain ${options.terrain.operation} zonal`;
  const runCpu = async (reason: string): Promise<RasterTerrainZonalResult> => {
    const terrainSource =
      options.useNeighborHalo === false
        ? (await createTerrainRasterSourceAsync(options.source, options.terrain, {engine: 'cpu'}))
            .source
        : createHaloTerrainSourceCpu(options.source, options.terrain);
    const statistics = options.zonePolygons.map(polygons =>
      polygons?.length
        ? computeRasterZonalStatistics(terrainSource, polygons, {bandIndex: 1})
        : null
    );
    return {
      statistics,
      engine: 'cpu-reference',
      engineNote: `${label}: CPU reference (${reason})`,
      fallbackReason: reason,
      readbackByteLength: 0
    };
  };
  if (!options.device) return runCpu('no GPU device');
  try {
    const result = await computeRasterTerrainZonalStatisticsGpu({
      ...options,
      device: options.device
    });
    if (result.declinedReason) return runCpu(result.declinedReason);
    return {
      statistics: result.statistics as (RasterZonalStatistics | null)[],
      engine: 'gpu',
      engineNote: `${label}: GPU (${result.contributors.join(' + ')} + GPURasterZonalStatistics, ${result.tileCount} tile${result.tileCount === 1 ? '' : 's'}; no derived-grid readback)`,
      readbackByteLength: result.readbackByteLength
    };
  } catch (error) {
    return runCpu(error instanceof Error ? error.message : 'GPU terrain zonal failed');
  }
}

/**
 * CPU reference with the same same-zoom neighbor halo as the GPU path, so tile seams are computed
 * rather than marked nodata and both engines scan the same cells.
 */
function createHaloTerrainSourceCpu(
  source: NumericRasterSource,
  terrain: RasterTerrainOptions
): NumericRasterSource {
  const bandIndex = terrain.bandIndex ?? 1;
  return createNumericRasterSource({
    id: `${source.id}:${terrain.operation}`,
    kind: source.kind,
    crs: source.crs,
    tiles: source.tiles.map(tile => {
      const padded = padRasterTileWithHalo(source.tiles, tile, bandIndex);
      const paddedValues = new Float32Array(padded.elevation);
      for (let index = 0; index < paddedValues.length; index++)
        if (!padded.validity[index]) paddedValues[index] = RASTER_TERRAIN_NODATA;
      const paddedTile = createNumericRasterTile({
        id: `${tile.id}:halo`,
        width: padded.width,
        height: padded.height,
        extent: padded.extent,
        zoom: tile.zoom,
        bands: [{index: 1, values: paddedValues, noDataValue: RASTER_TERRAIN_NODATA}]
      });
      const grid = computeRasterTerrainGrid(paddedTile, source.crs, {
        ...terrain,
        bandIndex: 1,
        groundModel: 'wgs84-per-row'
      });
      const stored = new Float32Array(tile.width * tile.height).fill(RASTER_TERRAIN_NODATA);
      for (let row = 0; row < tile.height; row++)
        for (let column = 0; column < tile.width; column++) {
          const paddedIndex = (row + 1) * padded.width + column + 1;
          if (grid.validity[paddedIndex])
            stored[row * tile.width + column] = grid.values[paddedIndex];
        }
      return createNumericRasterTile({
        id: `${tile.id}:${terrain.operation}`,
        width: tile.width,
        height: tile.height,
        extent: tile.extent,
        zoom: tile.zoom,
        tileIndex: tile.tileIndex,
        bands: [
          {index: 1, name: terrain.operation, values: stored, noDataValue: RASTER_TERRAIN_NODATA}
        ]
      });
    })
  });
}
