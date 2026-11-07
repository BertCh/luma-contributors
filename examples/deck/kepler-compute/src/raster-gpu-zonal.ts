// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device, type Texture} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPURasterZonalStatistics, makeGPURasterBand, makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {type GPURasterBand} from '@luma.gl/graph/gpu-raster';
import type {RasterBandGpuResource} from './raster-band-model';
import {
  mergeRasterZoneColumns,
  RASTER_ZONE_NONE,
  rasterizeZoneIds,
  type RasterGpuZoneStatistics,
  type RasterZoneColumns
} from './raster-gpu-zonal-zones';
import {
  getRasterZonalShift,
  RASTER_GPU_ZONAL_PRECISION_NOTE,
  shiftValuesToFloat32,
  unshiftRasterZoneColumns
} from './raster-gpu-zonal-precision';
import {planRasterReprojection} from './raster-reprojection';
import type {NumericRasterSource, NumericRasterTile} from './raster-tile-ingest';
import type {RasterZonalPolygon} from './raster-zonal';

export type RasterGpuZonalOptions = Readonly<{
  device: Device;
  source: NumericRasterSource;
  /** One entry per zone: the polygons of that zone (a multipolygon row has several). */
  zonePolygons: readonly (readonly RasterZonalPolygon[] | null)[];
  /** One-based band index, default 1. */
  bandIndex?: number;
  /** Resident textures when available, else the CPU band is uploaded to a storage buffer. */
  getResource?: (tileId: string, bandIndex: number) => RasterBandGpuResource | undefined;
  /**
   * Skip the resident texture and upload a per-tile centered copy instead, trading one upload for
   * float64-shifted float32 sums. Default false (the resident texture is used when present).
   */
  preferPrecision?: boolean;
}>;

export type RasterGpuZonalResult = Readonly<{
  statistics: (RasterGpuZoneStatistics | null)[];
  tileCount: number;
  /** Tiles whose values came from the resident r32float texture rather than a fresh upload. */
  residentTileCount: number;
  /** Tiles reduced from a centered upload (float64 mean subtracted, re-added in float64). */
  centeredTileCount?: number;
  /** What the sums guarantee; see RASTER_GPU_ZONAL_PRECISION_NOTE. */
  precisionNote: string;
  /** Set when the GPU path declined (overlapping zones); the caller then runs the CPU path. */
  declinedReason?: 'overlapping-zones' | 'no-tiles';
}>;

export function getScanTiles(
  source: NumericRasterSource,
  zonePolygons: readonly (readonly RasterZonalPolygon[] | null)[]
): NumericRasterTile[] {
  let west = Infinity;
  let south = Infinity;
  let east = -Infinity;
  let north = -Infinity;
  for (const polygons of zonePolygons)
    for (const polygon of polygons ?? [])
      for (const ring of polygon)
        for (const [lng, lat] of ring) {
          west = Math.min(west, lng);
          east = Math.max(east, lng);
          south = Math.min(south, lat);
          north = Math.max(north, lat);
        }
  const plan = planRasterReprojection(source.crs, 'EPSG:4326');
  const [sourceWest, sourceSouth] = plan.transformTargetToSource(west, south);
  const [sourceEast, sourceNorth] = plan.transformTargetToSource(east, north);
  const intersecting = source.tiles.filter(
    tile =>
      !(
        tile.extent[2] < sourceWest ||
        tile.extent[0] > sourceEast ||
        tile.extent[3] < sourceSouth ||
        tile.extent[1] > sourceNorth
      )
  );
  // Highest zoom only, like Kepler and the CPU reference, so parents never double count.
  const maximumZoom = Math.max(...intersecting.map(tile => tile.zoom));
  return intersecting.filter(tile => tile.zoom === maximumZoom);
}

/**
 * Per-zone statistics on the GPU: polygons are rasterized to zone ids on the CPU by pixel center
 * (identical sampling to the CPU reference), then `GPURasterZonalStatistics` reduces the resident
 * r32float band texture (or a storage-buffer upload when the band is not resident) per tile and
 * the per-tile columns are merged here with compensated float64 addition. Sums are float32
 * reductions per tile; non-resident bands are centered first (see RASTER_GPU_ZONAL_PRECISION_NOTE).
 */
export async function computeRasterZonalStatisticsGpu(
  options: RasterGpuZonalOptions
): Promise<RasterGpuZonalResult> {
  const {device, source, zonePolygons} = options;
  const bandIndex = options.bandIndex ?? 1;
  const tiles = getScanTiles(source, zonePolygons);
  if (tiles.length === 0)
    return {
      statistics: zonePolygons.map(() => null),
      tileCount: 0,
      residentTileCount: 0,
      precisionNote: RASTER_GPU_ZONAL_PRECISION_NOTE,
      declinedReason: 'no-tiles'
    };
  const zoneCount = zonePolygons.length;
  const tileColumns: RasterZoneColumns[] = [];
  let residentTileCount = 0;
  let centeredTileCount = 0;
  for (const tile of tiles) {
    const zoneRaster = rasterizeZoneIds(tile, source.crs, zonePolygons);
    if (zoneRaster.hasOverlap)
      return {
        statistics: zonePolygons.map(() => null),
        tileCount: tiles.length,
        residentTileCount,
        precisionNote: RASTER_GPU_ZONAL_PRECISION_NOTE,
        declinedReason: 'overlapping-zones'
      };
    if (zoneRaster.assignedCount === 0) continue;
    const band = tile.bands.find(candidate => candidate.index === bandIndex);
    // Binding error: the requested band is not part of this source.
    if (!band) throw new Error(`Raster tile ${tile.id} has no band ${bandIndex}`);
    const resource = options.getResource?.(tile.id, bandIndex);
    const isResident = Boolean(resource?.valueTexture) && !options.preferPrecision;
    if (isResident) residentTileCount++;
    // Uploaded bands are centered in float64 so the float32 reduction works on small magnitudes.
    const shift = isResident ? 0 : getRasterZonalShift(band.values, band.validity);
    if (!isResident && shift !== 0) centeredTileCount++;
    const columns = await reduceTile(device, tile, zoneRaster.zones, band.validity, zoneCount, {
      texture: isResident ? resource?.valueTexture : undefined,
      values: shift === 0 ? band.values : shiftValuesToFloat32(band.values, shift)
    });
    tileColumns.push(unshiftRasterZoneColumns(columns, shift));
  }
  return {
    statistics: mergeRasterZoneColumns(zoneCount, tileColumns),
    tileCount: tiles.length,
    residentTileCount,
    centeredTileCount,
    precisionNote: RASTER_GPU_ZONAL_PRECISION_NOTE
  };
}

async function reduceTile(
  device: Device,
  tile: NumericRasterTile,
  zones: Uint32Array,
  validity: Uint8Array,
  zoneCount: number,
  input: {texture?: Texture; values: ArrayLike<number>}
): Promise<RasterZoneColumns> {
  const graph = new GPUCommandGraph(device, {id: `raster-zonal-${tile.id}`});
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
  const cellCount = tile.width * tile.height;
  const validityView = importView(
    'validity',
    makeInput(Uint32Array.from(validity)),
    'uint32',
    cellCount
  );
  const grid = makeGPURasterMetadata({width: tile.width, height: tile.height});
  let band: GPURasterBand;
  if (input.texture) {
    const texture = input.texture;
    const textureHandle = graph.importTexture(
      {
        id: 'values-texture',
        format: texture.format as 'r32float',
        width: texture.width,
        height: texture.height,
        usage: texture.props.usage
      },
      texture
    );
    band = {
      id: 'values',
      format: 'float32',
      storage: {kind: 'texture', view: graph.createTextureView(textureHandle, {mipLevelCount: 1})},
      validity: validityView
    } as GPURasterBand;
  } else {
    band = makeGPURasterBand(
      importView('values', makeInput(Float32Array.from(input.values)), 'float32', cellCount),
      grid,
      {id: 'values', validity: validityView}
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
      zones: importView('zones', makeInput(zones), 'uint32', cellCount),
      values: band,
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
    const commandEncoder = device.createCommandEncoder({id: `raster-zonal-${tile.id}-encoder`});
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
    for (const buffer of owned) buffer.destroy();
  }
}
