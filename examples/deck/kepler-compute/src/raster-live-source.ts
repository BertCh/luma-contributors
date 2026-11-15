// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createHttpRangeByteSource,
  readCloudOptimizedGeoTIFF,
  type CogReadOptions
} from './raster-geotiff-cog';
import {
  createNumericRasterSource,
  getXyzTileExtent,
  type NumericRasterBand,
  type NumericRasterSource,
  type NumericRasterTile,
  type NumericRasterValues
} from './raster-tile-ingest';
import {createNumericRasterTile} from './raster-tile-ingest';

/** Structural view of one Kepler raster tile band (`images.imageBands[n]`). */
type KeplerTileBand = {data?: ArrayLike<number>; width?: number; height?: number};

/** Structural view of a deck.gl tile held in Kepler's `RasterTileLayer._loadedTiles`. */
export type KeplerLoadedRasterTile = {
  index?: {x: number; y: number; z: number};
  id?: string;
  data?: {
    images?: {
      imageBands?: KeplerTileBand[];
      imageMask?: KeplerTileBand | null;
      imageRgba?: {data?: ArrayLike<number>; width?: number; height?: number} | null;
    };
    image?: {data?: ArrayLike<number>; width?: number; height?: number};
  } | null;
};

function toNumericValues(data: ArrayLike<number>): NumericRasterValues {
  if (data instanceof Uint8ClampedArray)
    return new Uint8Array(data.buffer, data.byteOffset, data.length);
  if (ArrayBuffer.isView(data) && !(data instanceof DataView)) {
    const constructors = [
      Float32Array,
      Float64Array,
      Int8Array,
      Uint8Array,
      Int16Array,
      Uint16Array,
      Int32Array,
      Uint32Array
    ];
    if (constructors.some(arrayType => data instanceof arrayType))
      return data as NumericRasterValues;
  }
  return Float32Array.from(data);
}

function applyMask(
  band: NumericRasterBand,
  mask: ArrayLike<number> | undefined
): NumericRasterBand {
  if (!mask || mask.length !== band.validity.length) return band;
  const validity = band.validity.slice();
  // Kepler treats mask < 1 as nodata (isNodataMask).
  for (let sampleIndex = 0; sampleIndex < validity.length; sampleIndex++)
    if (!(mask[sampleIndex] >= 1)) validity[sampleIndex] = 0;
  return {...band, validity};
}

/**
 * Copies one loaded Kepler tile into a numeric tile. Returns null for tiles without decoded
 * samples. Numeric bands keep their real values; an RGBA-only (plain XYZ image) tile yields four
 * uint8 bands (R, G, B, A) with alpha 0 treated as nodata, mirroring Kepler's identify.
 */
export function createNumericTileFromKeplerTile(
  tile: KeplerLoadedRasterTile,
  sourceId: string
): NumericRasterTile | null {
  const index = tile.index;
  const images = tile.data?.images;
  if (!index || !images) return null;
  const tileId = `${sourceId}/${index.z}/${index.x}/${index.y}`;
  const extent = getXyzTileExtent(index);
  const bands = images.imageBands;
  if (
    Array.isArray(bands) &&
    bands.length > 0 &&
    bands[0].data &&
    bands[0].width &&
    bands[0].height
  ) {
    const {width, height} = {width: bands[0].width, height: bands[0].height};
    if (bands.some(band => !band.data || band.data.length !== width * height)) return null;
    const created = createNumericRasterTile({
      id: tileId,
      width,
      height,
      extent,
      tileIndex: index,
      bands: bands.map(band => ({values: toNumericValues(band.data as ArrayLike<number>)}))
    });
    const mask = images.imageMask?.data;
    return mask ? {...created, bands: created.bands.map(band => applyMask(band, mask))} : created;
  }
  const rgba = images.imageRgba?.data ? images.imageRgba : tile.data?.image;
  if (
    rgba?.data &&
    rgba.width &&
    rgba.height &&
    rgba.data.length === rgba.width * rgba.height * 4
  ) {
    const pixelCount = rgba.width * rgba.height;
    const channels = [0, 1, 2, 3].map(channel => {
      const values = new Uint8Array(pixelCount);
      for (let pixelIndex = 0; pixelIndex < pixelCount; pixelIndex++)
        values[pixelIndex] = (rgba.data as ArrayLike<number>)[pixelIndex * 4 + channel];
      return values;
    });
    const created = createNumericRasterTile({
      id: tileId,
      width: rgba.width,
      height: rgba.height,
      extent,
      tileIndex: index,
      bands: channels.map(values => ({values}))
    });
    return {...created, bands: created.bands.map(band => applyMask(band, channels[3]))};
  }
  return null;
}

/**
 * Builds a `NumericRasterSource` from the tiles Kepler currently holds for a raster tile layer.
 * Tiles keep their XYZ index, so identify and zonal prefer the highest loaded zoom.
 */
export function createNumericSourceFromKeplerTiles(
  sourceId: string,
  loadedTiles: readonly (KeplerLoadedRasterTile | null | undefined)[]
): NumericRasterSource {
  const tiles: NumericRasterTile[] = [];
  let bandCount = -1;
  for (const loaded of loadedTiles) {
    const tile = loaded ? createNumericTileFromKeplerTile(loaded, sourceId) : null;
    if (!tile) continue;
    // A source holds one band layout; tiles that disagree (for example a placeholder) are skipped.
    if (bandCount < 0) bandCount = tile.bands.length;
    if (tile.bands.length === bandCount) tiles.push(tile);
  }
  return createNumericRasterSource({id: sourceId, kind: 'xyz-tiles', crs: 'EPSG:3857', tiles});
}

/** Fetches and decodes a GeoTIFF (classic, uncompressed; see raster-geotiff-reader) from a URL. */
export async function loadNumericGeoTiffFromUrl(
  url: string,
  sourceId: string,
  signal?: AbortSignal,
  readOptions: Omit<CogReadOptions, 'id' | 'signal'> = {}
): Promise<NumericRasterSource> {
  // Range requests fetch only the header, directories and needed blocks of a COG; a server that
  // ignores Range falls back to one full download inside the byte source.
  const result = await readCloudOptimizedGeoTIFF(createHttpRangeByteSource(url, {signal}), {
    ...readOptions,
    id: sourceId,
    signal
  });
  return result.source;
}

const rasterObjectIdentities = new WeakMap<object, number>();
let nextRasterObjectIdentity = 0;
function getRasterObjectIdentity(value: object | null | undefined): number {
  if (!value) return 0;
  let identity = rasterObjectIdentities.get(value);
  if (identity === undefined) {
    identity = ++nextRasterObjectIdentity;
    rasterObjectIdentities.set(value, identity);
  }
  return identity;
}

/** Replacement generations include decoded payloads, band/mask buffers and dimensions, independent of tile order. */
export function getKeplerTileSignature(
  loadedTiles: readonly (KeplerLoadedRasterTile | null | undefined)[]
): string {
  const bandSignature = (band: KeplerTileBand | null | undefined) =>
    `${getRasterObjectIdentity(band?.data)}:${band?.width}:${band?.height}`;
  return loadedTiles
    .map(tile => {
      if (!tile?.index) return '-';
      const images = tile.data?.images;
      return `${tile.index.z}/${tile.index.x}/${tile.index.y}:${getRasterObjectIdentity(tile.data)}:${images?.imageBands?.map(bandSignature).join(',')}:${bandSignature(images?.imageMask)}:${bandSignature(images?.imageRgba)}:${bandSignature(tile.data?.image)}`;
    })
    .sort()
    .join('|');
}
