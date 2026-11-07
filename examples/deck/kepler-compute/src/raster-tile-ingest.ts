// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Numeric raster ingestion for Kepler raster tile and bitmap layers.
 *
 * `NumericRasterSource` is the typed descriptor that analysis code binds to. It carries real
 * numeric samples (never a rendered palette), per-band nodata and a derived validity mask.
 */

export type NumericRasterCrs = 'EPSG:4326' | 'EPSG:3857';

export type NumericRasterValues =
  | Float32Array
  | Float64Array
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array;

export type NumericRasterSampleType =
  | 'float32'
  | 'float64'
  | 'int8'
  | 'uint8'
  | 'int16'
  | 'uint16'
  | 'int32'
  | 'uint32';

export type NumericRasterBand = Readonly<{
  /** One-based band index, matching GeoTIFF and the `b1` names used by band math. */
  index: number;
  name?: string;
  sampleType: NumericRasterSampleType;
  /** Row-major, north-up, `width * height` samples. */
  values: NumericRasterValues;
  /** Source nodata sentinel; `null` means only non-finite floats are invalid. */
  noDataValue: number | null;
  /** 1 where the sample is valid, 0 where it is nodata or non-finite. */
  validity: Uint8Array;
}>;

export type NumericRasterTileIndex = Readonly<{x: number; y: number; z: number}>;

export type NumericRasterTile = Readonly<{
  id: string;
  width: number;
  height: number;
  /** [west, south, east, north] in the source CRS. */
  extent: readonly [number, number, number, number];
  /** XYZ zoom; identify and zonal prefer the highest zoom, like Kepler. Defaults to 0. */
  zoom: number;
  tileIndex?: NumericRasterTileIndex;
  bands: readonly NumericRasterBand[];
}>;

export type NumericRasterSource = Readonly<{
  id: string;
  kind: 'single-raster' | 'xyz-tiles';
  crs: NumericRasterCrs;
  bandCount: number;
  tiles: readonly NumericRasterTile[];
}>;

export type NumericRasterBandInput = Readonly<{
  index?: number;
  name?: string;
  values: NumericRasterValues;
  noDataValue?: number | null;
}>;

const WEB_MERCATOR_HALF_CIRCUMFERENCE = 20037508.342789244;

export function getRasterSampleType(values: NumericRasterValues): NumericRasterSampleType {
  if (values instanceof Float32Array) return 'float32';
  if (values instanceof Float64Array) return 'float64';
  if (values instanceof Int8Array) return 'int8';
  if (values instanceof Uint8Array) return 'uint8';
  if (values instanceof Int16Array) return 'int16';
  if (values instanceof Uint16Array) return 'uint16';
  if (values instanceof Int32Array) return 'int32';
  return 'uint32';
}

/** True when the sample is nodata. The sentinel is rounded through the band type first. */
export function isRasterSampleNoData(value: number, noDataValue: number | null): boolean {
  if (!Number.isFinite(value)) return true;
  return noDataValue !== null && value === noDataValue;
}

/** Converts a nodata sentinel to the exact value a typed array would store. */
export function normalizeNoDataValue(
  values: NumericRasterValues,
  noDataValue: number | null | undefined
): number | null {
  if (noDataValue === null || noDataValue === undefined || Number.isNaN(noDataValue)) return null;
  const converted = new (values.constructor as new (source: number[]) => NumericRasterValues)([
    noDataValue
  ])[0];
  // A sentinel that cannot be represented in the band type can never match a sample.
  return Number.isFinite(converted) ? converted : null;
}

export function createRasterBand(
  input: NumericRasterBandInput,
  fallbackIndex: number,
  sampleCount: number
): NumericRasterBand {
  // Band length must match the tile; a mismatch means a corrupt decode.
  if (input.values.length !== sampleCount) throw new Error('Raster band length mismatch');
  const noDataValue = normalizeNoDataValue(input.values, input.noDataValue);
  const validity = new Uint8Array(sampleCount);
  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
    validity[sampleIndex] = isRasterSampleNoData(input.values[sampleIndex], noDataValue) ? 0 : 1;
  return {
    index: input.index ?? fallbackIndex,
    name: input.name,
    sampleType: getRasterSampleType(input.values),
    values: input.values,
    noDataValue,
    validity
  };
}

export function createNumericRasterTile(options: {
  id: string;
  width: number;
  height: number;
  extent: readonly [number, number, number, number];
  bands: readonly NumericRasterBandInput[];
  zoom?: number;
  tileIndex?: NumericRasterTileIndex;
}): NumericRasterTile {
  const {width, height, extent} = options;
  // Extent must be a positive north-up box; rotated or flipped rasters are refused upstream.
  if (!(width > 0 && height > 0 && extent[0] < extent[2] && extent[1] < extent[3]))
    throw new Error('Raster tile requires positive size and a north-up extent');
  return {
    id: options.id,
    width,
    height,
    extent,
    zoom: options.zoom ?? options.tileIndex?.z ?? 0,
    tileIndex: options.tileIndex,
    bands: options.bands.map((band, bandIndex) =>
      createRasterBand(band, bandIndex + 1, width * height)
    )
  };
}

/** Web Mercator extent in EPSG:3857 meters for an XYZ tile (y grows southward). */
export function getXyzTileExtent(
  index: NumericRasterTileIndex
): readonly [number, number, number, number] {
  const tileCount = 2 ** index.z;
  const tileSize = (2 * WEB_MERCATOR_HALF_CIRCUMFERENCE) / tileCount;
  const west = -WEB_MERCATOR_HALF_CIRCUMFERENCE + index.x * tileSize;
  const north = WEB_MERCATOR_HALF_CIRCUMFERENCE - index.y * tileSize;
  return [west, north - tileSize, west + tileSize, north];
}

export function createNumericRasterSource(options: {
  id: string;
  kind: NumericRasterSource['kind'];
  crs: NumericRasterCrs;
  tiles: readonly NumericRasterTile[];
}): NumericRasterSource {
  const bandCount = options.tiles[0]?.bands.length ?? 0;
  // Every tile in one source must expose the same bands so identify and band math are well defined.
  if (options.tiles.some(tile => tile.bands.length !== bandCount))
    throw new Error('Raster tiles disagree on band count');
  return {...options, bandCount};
}

/** Builds an XYZ tile source; tile edges are shared exactly, so identify applies half-open ownership. */
export function createXyzTileSource(
  id: string,
  tiles: readonly {
    index: NumericRasterTileIndex;
    width: number;
    height: number;
    bands: readonly NumericRasterBandInput[];
  }[]
): NumericRasterSource {
  return createNumericRasterSource({
    id,
    kind: 'xyz-tiles',
    crs: 'EPSG:3857',
    tiles: tiles.map(tile =>
      createNumericRasterTile({
        id: `${id}/${tile.index.z}/${tile.index.x}/${tile.index.y}`,
        width: tile.width,
        height: tile.height,
        extent: getXyzTileExtent(tile.index),
        tileIndex: tile.index,
        bands: tile.bands
      })
    )
  });
}

export function getRasterBandByteLength(tile: NumericRasterTile, gpuBytesPerSample = 4): number {
  // GPU residency stores float32 values plus one validity byte per sample.
  return tile.width * tile.height * (gpuBytesPerSample + 1);
}

export function getRasterSourceByteLength(source: NumericRasterSource): number {
  let total = 0;
  for (const tile of source.tiles)
    for (const band of tile.bands) total += band.values.byteLength + band.validity.byteLength;
  return total;
}
