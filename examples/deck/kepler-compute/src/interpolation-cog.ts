// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Tiled, optionally deflate-compressed, multiband Float32 GeoTIFF with mean overviews and COG
 * ordering (all IFDs first, then tile data from the smallest overview to full resolution).
 * Classic TIFF, chunky bands. Not validated with GDAL's validate_cloud_optimized_geotiff and no
 * ghost-area header is written. Geographic (EPSG:4326) or declared projected EPSG georeferencing.
 */

type TIFFEntry = {tag: number; type: number; count: number; data: Uint8Array};

export type CloudGeoTIFFBand = {name: string; values: Float32Array};

export type CloudGeoTIFFOptions = {
  width: number;
  height: number;
  /** [west, south, east, north] in CRS units. */
  bounds: readonly [number, number, number, number];
  bands: readonly CloudGeoTIFFBand[];
  crs: {kind: 'geographic' | 'projected'; epsg: number};
  /** Square tile edge, a multiple of 16. Default 256. */
  tileSize?: number;
  compression?: 'none' | 'deflate';
  /** Number of 2x mean overviews; 'auto' halves until the image fits one tile. Default 'auto'. */
  overviews?: number | 'auto';
  /** Provenance JSON stored in ImageDescription of the full-resolution IFD. */
  provenance?: Readonly<Record<string, unknown>>;
};

type Level = {width: number; height: number; bands: Float32Array[]};

export function encodeCloudGeoTIFFLevelCount(
  width: number,
  height: number,
  tileSize: number,
  overviews: number | 'auto'
): number {
  if (overviews !== 'auto') return overviews;
  let count = 0;
  let levelWidth = width;
  let levelHeight = height;
  while (Math.max(levelWidth, levelHeight) > tileSize) {
    levelWidth = Math.ceil(levelWidth / 2);
    levelHeight = Math.ceil(levelHeight / 2);
    count++;
  }
  return count;
}

/** 2x2 mean overview ignoring non-finite cells (NaN nodata); all-nodata blocks stay NaN. */
export function makeMeanOverview(level: Level): Level {
  const width = Math.ceil(level.width / 2);
  const height = Math.ceil(level.height / 2);
  const bands = level.bands.map(source => {
    const target = new Float32Array(width * height);
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        let sum = 0;
        let count = 0;
        for (let dy = 0; dy < 2; dy++)
          for (let dx = 0; dx < 2; dx++) {
            const sourceX = x * 2 + dx;
            const sourceY = y * 2 + dy;
            if (sourceX >= level.width || sourceY >= level.height) continue;
            const value = source[sourceY * level.width + sourceX];
            if (Number.isFinite(value)) {
              sum += value;
              count++;
            }
          }
        target[y * width + x] = count ? sum / count : Number.NaN;
      }
    return target;
  });
  return {width, height, bands};
}

async function deflateBytes(bytes: Uint8Array<ArrayBuffer>): Promise<Uint8Array<ArrayBuffer>> {
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function encodeTiles(
  level: Level,
  tileSize: number,
  compression: 'none' | 'deflate'
): Promise<Uint8Array<ArrayBuffer>[]> {
  const bandCount = level.bands.length;
  const tilesAcross = Math.ceil(level.width / tileSize);
  const tilesDown = Math.ceil(level.height / tileSize);
  const tiles: Uint8Array<ArrayBuffer>[] = [];
  for (let tileY = 0; tileY < tilesDown; tileY++)
    for (let tileX = 0; tileX < tilesAcross; tileX++) {
      // Edge tiles are padded to the full tile size with NaN, as TIFF requires.
      const raw = new Uint8Array(tileSize * tileSize * bandCount * 4);
      const view = new DataView(raw.buffer);
      for (let y = 0; y < tileSize; y++)
        for (let x = 0; x < tileSize; x++) {
          const sourceX = tileX * tileSize + x;
          const sourceY = tileY * tileSize + y;
          const inside = sourceX < level.width && sourceY < level.height;
          for (let band = 0; band < bandCount; band++) {
            const value = inside ? level.bands[band][sourceY * level.width + sourceX] : Number.NaN;
            const offset = ((y * tileSize + x) * bandCount + band) * 4;
            if (Number.isFinite(value)) view.setFloat32(offset, value, true);
            else view.setUint32(offset, 0x7fc00000, true);
          }
        }
      tiles.push(compression === 'deflate' ? await deflateBytes(raw) : raw);
    }
  return tiles;
}

export async function encodeCloudOptimizedGeoTIFF(
  options: CloudGeoTIFFOptions
): Promise<Uint8Array<ArrayBuffer>> {
  const {width, height, bounds, bands, crs} = options;
  const tileSize = options.tileSize ?? 256;
  const compression = options.compression ?? 'deflate';
  const [west, south, east, north] = bounds;
  // Bounded input; grid and band sizes are also checked by the caller's memory reservation.
  if (
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width < 1 ||
    height < 1 ||
    bands.length < 1 ||
    bands.length > 16 ||
    bands.some(band => band.values.length !== width * height) ||
    tileSize % 16 !== 0 ||
    tileSize < 16 ||
    !bounds.every(Number.isFinite) ||
    west >= east ||
    south >= north
  )
    throw new Error('Cloud GeoTIFF requires a finite grid, one value per cell and 1-16 bands');
  const levelCount = encodeCloudGeoTIFFLevelCount(
    width,
    height,
    tileSize,
    options.overviews ?? 'auto'
  );
  const levels: Level[] = [{width, height, bands: bands.map(band => band.values)}];
  for (let index = 0; index < levelCount; index++) levels.push(makeMeanOverview(levels[index]));
  const tileData = await Promise.all(
    levels.map(level => encodeTiles(level, tileSize, compression))
  );

  const description = JSON.stringify({
    format: 'luma.gl-kepler-cog',
    layout: 'ifd-first, smallest overview data first',
    bands: bands.map((band, index) => ({index: index + 1, name: band.name})),
    ...options.provenance
  }).replace(/[\u007f-￿]/g, c => `\\u${c.charCodeAt(0).toString(16).padStart(4, '0')}`);
  const metadata = `<GDALMetadata>${bands
    .map(
      (band, index) =>
        `<Item name="DESCRIPTION" sample="${index}" role="description">${escapeXml(band.name)}</Item>`
    )
    .join('')}</GDALMetadata>`;

  const bandCount = bands.length;
  const geoKeys =
    crs.kind === 'geographic'
      ? [1, 1, 1, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, crs.epsg]
      : [1, 1, 1, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, crs.epsg];

  // Layout pass: IFD sizes are independent of final offsets, so compute them first.
  const ifdEntriesFor = (levelIndex: number, tileOffsets: number[], tileByteCounts: number[]) => {
    const level = levels[levelIndex];
    const entries: TIFFEntry[] = [
      numeric(254, 4, [levelIndex === 0 ? 0 : 1]),
      numeric(256, 4, [level.width]),
      numeric(257, 4, [level.height]),
      numeric(258, 3, new Array(bandCount).fill(32)),
      numeric(259, 3, [compression === 'deflate' ? 8 : 1]),
      numeric(262, 3, [1]),
      numeric(277, 3, [bandCount]),
      numeric(284, 3, [1]),
      numeric(322, 4, [tileSize]),
      numeric(323, 4, [tileSize]),
      numeric(324, 4, tileOffsets),
      numeric(325, 4, tileByteCounts),
      numeric(339, 3, new Array(bandCount).fill(3)),
      numeric(33550, 12, [(east - west) / level.width, (north - south) / level.height, 0]),
      numeric(33922, 12, [0, 0, 0, west, north, 0]),
      numeric(34735, 3, geoKeys),
      ascii(42113, 'nan')
    ];
    if (levelIndex === 0) {
      entries.push(ascii(270, description), ascii(42112, metadata));
    }
    return entries.sort((first, second) => first.tag - second.tag);
  };
  const zeroOffsets = levels.map((_, index) => new Array(tileData[index].length).fill(0));
  const ifdSizes = levels.map((_, index) => {
    const entries = ifdEntriesFor(index, zeroOffsets[index], zeroOffsets[index]);
    return entries;
  });
  let cursor = 8;
  const ifdPositions: number[] = [];
  const extraOffsets: number[][] = [];
  for (const entries of ifdSizes) {
    ifdPositions.push(cursor);
    cursor += 2 + entries.length * 12 + 4;
    const offsets: number[] = [];
    for (const entry of entries) {
      if (entry.data.byteLength > 4) {
        cursor = Math.ceil(cursor / 8) * 8;
        offsets.push(cursor);
        cursor += entry.data.byteLength;
      } else offsets.push(0);
    }
    extraOffsets.push(offsets);
  }
  cursor = Math.ceil(cursor / 8) * 8;
  const dataOffsets: number[][] = levels.map(() => []);
  // COG order: smallest overview first, full resolution last.
  for (let index = levels.length - 1; index >= 0; index--)
    for (const tile of tileData[index]) {
      dataOffsets[index].push(cursor);
      cursor += tile.byteLength;
    }
  const output = new Uint8Array(cursor);
  const view = new DataView(output.buffer);
  view.setUint16(0, 0x4949, true);
  view.setUint16(2, 42, true);
  view.setUint32(4, ifdPositions[0], true);
  for (let index = 0; index < levels.length; index++) {
    const entries = ifdEntriesFor(
      index,
      dataOffsets[index],
      tileData[index].map(tile => tile.byteLength)
    );
    const position = ifdPositions[index];
    view.setUint16(position, entries.length, true);
    for (const [entryIndex, entry] of entries.entries()) {
      const entryPosition = position + 2 + entryIndex * 12;
      view.setUint16(entryPosition, entry.tag, true);
      view.setUint16(entryPosition + 2, entry.type, true);
      view.setUint32(entryPosition + 4, entry.count, true);
      if (entry.data.byteLength <= 4) output.set(entry.data, entryPosition + 8);
      else {
        const offset = extraOffsets[index][entryIndex];
        view.setUint32(entryPosition + 8, offset, true);
        output.set(entry.data, offset);
      }
    }
    view.setUint32(
      position + 2 + entries.length * 12,
      index + 1 < levels.length ? ifdPositions[index + 1] : 0,
      true
    );
    for (const [tileIndex, tile] of tileData[index].entries())
      output.set(tile, dataOffsets[index][tileIndex]);
  }
  return output;
}

function escapeXml(value: string): string {
  return value.replace(/[<>&"]/g, c => `&#${c.charCodeAt(0)};`);
}

function ascii(tag: number, value: string): TIFFEntry {
  const data = new TextEncoder().encode(`${value}\0`);
  return {tag, type: 2, count: data.length, data};
}

function numeric(tag: number, type: 3 | 4 | 12, values: readonly number[]): TIFFEntry {
  const size = type === 3 ? 2 : type === 4 ? 4 : 8;
  const data = new Uint8Array(values.length * size);
  const view = new DataView(data.buffer);
  for (const [index, value] of values.entries()) {
    if (type === 3) view.setUint16(index * size, value, true);
    else if (type === 4) view.setUint32(index * size, value, true);
    else view.setFloat64(index * size, value, true);
  }
  return {tag, type, count: values.length, data};
}
