// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {parseRasterCrs} from './raster-reprojection';
import {
  decodePackBits,
  decodeTiffLzw,
  inflateZlib,
  undoFloatingPointPredictor,
  undoHorizontalPredictor
} from './raster-tiff-codecs';
import {
  createNumericRasterSource,
  createNumericRasterTile,
  type NumericRasterSource,
  type NumericRasterValues
} from './raster-tile-ingest';

import {clamp} from '@math.gl/core';

/** Random-access bytes: a buffer, or an HTTP resource read with Range requests. */
export type ByteSource = {
  read(offset: number, length: number): Promise<Uint8Array>;
};

export type HttpByteSourceStatistics = {
  requestCount: number;
  byteCount: number;
  rangeSupported: boolean | null;
};

/** Whole-buffer byte source for tests and local files. */
export function createBufferByteSource(bytes: Uint8Array): ByteSource {
  return {read: async (offset, length) => bytes.subarray(offset, offset + length)};
}

/**
 * HTTP Range byte source. A server that ignores Range answers 200 with the whole body; that body
 * is cached and sliced, so such servers still work (at the cost of one full download).
 */
export function createHttpRangeByteSource(
  url: string,
  options: {
    fetchImplementation?: typeof fetch;
    headers?: Record<string, string>;
    signal?: AbortSignal;
    statistics?: HttpByteSourceStatistics;
  } = {}
): ByteSource {
  const fetchImplementation = options.fetchImplementation ?? fetch;
  const statistics = options.statistics ?? {requestCount: 0, byteCount: 0, rangeSupported: null};
  let wholeBody: Uint8Array | undefined;
  return {
    async read(offset, length) {
      if (wholeBody) return wholeBody.subarray(offset, offset + length);
      statistics.requestCount++;
      const response = await fetchImplementation(url, {
        signal: options.signal,
        headers: {...options.headers, Range: `bytes=${offset}-${offset + length - 1}`}
      });
      // A failed request is surfaced to the caller, which reports it in the panel.
      if (!response.ok) throw new Error(`GeoTIFF request failed with status ${response.status}`);
      const body = new Uint8Array(await response.arrayBuffer());
      statistics.byteCount += body.byteLength;
      if (response.status === 206) {
        statistics.rangeSupported = true;
        return body;
      }
      statistics.rangeSupported = false;
      wholeBody = body;
      return body.subarray(offset, offset + length);
    }
  };
}

/** Serves reads inside the first `prefetchLength` bytes from one up-front read (IFDs sit there in a COG). */
export function createPrefetchedByteSource(source: ByteSource, prefetchLength = 32768): ByteSource {
  let prefetched: Promise<Uint8Array> | undefined;
  return {
    async read(offset, length) {
      if (offset + length <= prefetchLength) {
        prefetched ??= source.read(0, prefetchLength);
        const bytes = await prefetched;
        // A short file returns fewer bytes than requested; callers read within the file.
        return bytes.subarray(offset, offset + length);
      }
      return source.read(offset, length);
    }
  };
}

export type TiffDirectory = Readonly<{
  width: number;
  height: number;
  tileWidth: number;
  tileHeight: number;
  isTiled: boolean;
  compression: number;
  predictor: number;
  bitsPerSample: number;
  sampleFormat: number;
  samplesPerPixel: number;
  planarConfiguration: number;
  blockOffsets: readonly number[];
  blockByteCounts: readonly number[];
  isReducedResolution: boolean;
  noDataValue: number | null;
  /** Georeferencing tags (only on the full-resolution directory in practice). */
  pixelScale?: readonly number[];
  tiepoint?: readonly number[];
  geoKeys?: readonly number[];
  hasModelTransformation: boolean;
}>;

export type TiffHeader = Readonly<{
  isLittleEndian: boolean;
  isBigTiff: boolean;
  directories: readonly TiffDirectory[];
}>;

const TYPE_BYTES: Record<number, number> = {
  1: 1,
  2: 1,
  3: 2,
  4: 4,
  5: 8,
  6: 1,
  7: 1,
  8: 2,
  9: 4,
  10: 8,
  11: 4,
  12: 8,
  16: 8,
  17: 8,
  18: 8
};

function readValue(view: DataView, type: number, offset: number, little: boolean): number {
  switch (type) {
    case 3:
      return view.getUint16(offset, little);
    case 4:
      return view.getUint32(offset, little);
    case 8:
      return view.getInt16(offset, little);
    case 9:
      return view.getInt32(offset, little);
    case 11:
      return view.getFloat32(offset, little);
    case 12:
      return view.getFloat64(offset, little);
    case 16:
    case 18:
      return Number(view.getBigUint64(offset, little));
    default:
      return view.getUint8(offset);
  }
}

async function readDirectory(
  source: ByteSource,
  offset: number,
  isLittleEndian: boolean,
  isBigTiff: boolean
): Promise<{directory: TiffDirectory; nextOffset: number}> {
  const countBytes = isBigTiff ? 8 : 2;
  const entryBytes = isBigTiff ? 20 : 12;
  const countView = viewOf(await source.read(offset, countBytes));
  const entryCount = isBigTiff
    ? Number(countView.getBigUint64(0, isLittleEndian))
    : countView.getUint16(0, isLittleEndian);
  const tail = isBigTiff ? 8 : 4;
  const body = await source.read(offset + countBytes, entryCount * entryBytes + tail);
  const view = viewOf(body);
  const tags = new Map<number, number[] | string>();
  const inlineLimit = isBigTiff ? 8 : 4;
  for (let entryIndex = 0; entryIndex < entryCount; entryIndex++) {
    const entryOffset = entryIndex * entryBytes;
    const tag = view.getUint16(entryOffset, isLittleEndian);
    const type = view.getUint16(entryOffset + 2, isLittleEndian);
    const count = isBigTiff
      ? Number(view.getBigUint64(entryOffset + 4, isLittleEndian))
      : view.getUint32(entryOffset + 4, isLittleEndian);
    const componentBytes = TYPE_BYTES[type] ?? 1;
    const byteLength = componentBytes * count;
    const valueFieldOffset = entryOffset + (isBigTiff ? 12 : 8);
    let valueView: DataView;
    let valueBase: number;
    if (byteLength <= inlineLimit) {
      valueView = view;
      valueBase = valueFieldOffset;
    } else {
      const dataOffset = isBigTiff
        ? Number(view.getBigUint64(valueFieldOffset, isLittleEndian))
        : view.getUint32(valueFieldOffset, isLittleEndian);
      valueView = viewOf(await source.read(dataOffset, byteLength));
      valueBase = 0;
    }
    if (type === 2) {
      const bytes = new Uint8Array(valueView.buffer, valueView.byteOffset + valueBase, count);
      tags.set(tag, new TextDecoder().decode(bytes).replace(/\0.*$/s, ''));
      continue;
    }
    const values: number[] = new Array(count);
    for (let componentIndex = 0; componentIndex < count; componentIndex++)
      values[componentIndex] = readValue(
        valueView,
        type,
        valueBase + componentIndex * componentBytes,
        isLittleEndian
      );
    tags.set(tag, values);
  }
  const nextOffset = isBigTiff
    ? Number(view.getBigUint64(entryCount * entryBytes, isLittleEndian))
    : view.getUint32(entryCount * entryBytes, isLittleEndian);
  const getNumbers = (tag: number): number[] | undefined => {
    const value = tags.get(tag);
    return Array.isArray(value) ? value : undefined;
  };
  const width = getNumbers(256)?.[0];
  const height = getNumbers(257)?.[0];
  const isTiled = tags.has(322);
  const blockOffsets = getNumbers(isTiled ? 324 : 273);
  const blockByteCounts = getNumbers(isTiled ? 325 : 279);
  // A directory without dimensions or block tables is not a raster image (for example a mask IFD stub).
  if (!width || !height || !blockOffsets || !blockByteCounts)
    throw new Error('Unsupported GeoTIFF layout');
  const noDataText = tags.get(42113);
  const noDataValue =
    typeof noDataText === 'string' && noDataText.trim() ? Number(noDataText.trim()) : null;
  return {
    nextOffset,
    directory: {
      width,
      height,
      tileWidth: isTiled ? (getNumbers(322)?.[0] ?? width) : width,
      tileHeight: isTiled
        ? (getNumbers(323)?.[0] ?? height)
        : Math.min(getNumbers(278)?.[0] ?? height, height),
      isTiled,
      compression: getNumbers(259)?.[0] ?? 1,
      predictor: getNumbers(317)?.[0] ?? 1,
      bitsPerSample: getNumbers(258)?.[0] ?? 1,
      sampleFormat: getNumbers(339)?.[0] ?? 1,
      samplesPerPixel: getNumbers(277)?.[0] ?? 1,
      planarConfiguration: getNumbers(284)?.[0] ?? 1,
      blockOffsets,
      blockByteCounts,
      isReducedResolution: ((getNumbers(254)?.[0] ?? 0) & 1) === 1,
      noDataValue: noDataValue !== null && Number.isNaN(noDataValue) ? null : noDataValue,
      pixelScale: getNumbers(33550),
      tiepoint: getNumbers(33922),
      geoKeys: getNumbers(34735),
      hasModelTransformation: Boolean(getNumbers(34264))
    }
  };
}

function viewOf(bytes: Uint8Array): DataView {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

/** Reads the TIFF header and every image directory (full resolution first, then overviews). */
export async function readTiffHeader(source: ByteSource): Promise<TiffHeader> {
  const header = await source.read(0, 16);
  const view = viewOf(header);
  const isLittleEndian = view.getUint16(0, false) === 0x4949;
  if (!isLittleEndian && view.getUint16(0, false) !== 0x4d4d) throw new Error('Not a TIFF file');
  const magic = view.getUint16(2, isLittleEndian);
  if (magic !== 42 && magic !== 43) throw new Error('Not a TIFF file');
  const isBigTiff = magic === 43;
  let offset = isBigTiff
    ? Number(view.getBigUint64(8, isLittleEndian))
    : view.getUint32(4, isLittleEndian);
  const directories: TiffDirectory[] = [];
  // Cycle guard: a corrupt chain must not loop forever.
  while (offset && directories.length < 64) {
    const {directory, nextOffset} = await readDirectory(source, offset, isLittleEndian, isBigTiff);
    directories.push(directory);
    offset = nextOffset;
  }
  return {isLittleEndian, isBigTiff, directories};
}

export type CogReadOptions = Readonly<{
  id?: string;
  /** Pick the finest level whose pixel count is at most this. Default 16 million. */
  maximumPixelCount?: number;
  /** Pick the coarsest level that is still at least this wide (overrides the pixel cap choice). */
  targetWidth?: number;
  /** Explicit directory index (0 = full resolution). */
  overviewLevel?: number;
  /** Source-CRS window `[west, south, east, north]`; only intersecting blocks are fetched. */
  window?: readonly [number, number, number, number];
  signal?: AbortSignal;
}>;

export type CogReadResult = Readonly<{
  source: NumericRasterSource;
  levelIndex: number;
  levelCount: number;
  blockCount: number;
  fetchedBlockCount: number;
}>;

const SAMPLE_ARRAYS: Record<string, new (buffer: ArrayBuffer) => NumericRasterValues> = {
  '3-32': Float32Array,
  '3-64': Float64Array,
  '1-8': Uint8Array,
  '1-16': Uint16Array,
  '1-32': Uint32Array,
  '2-8': Int8Array,
  '2-16': Int16Array,
  '2-32': Int32Array
};

function readSample(
  view: DataView,
  offset: number,
  sampleFormat: number,
  bitsPerSample: number,
  isLittleEndian: boolean
): number {
  if (sampleFormat === 3)
    return bitsPerSample === 32
      ? view.getFloat32(offset, isLittleEndian)
      : view.getFloat64(offset, isLittleEndian);
  if (sampleFormat === 2)
    return bitsPerSample === 8
      ? view.getInt8(offset)
      : bitsPerSample === 16
        ? view.getInt16(offset, isLittleEndian)
        : view.getInt32(offset, isLittleEndian);
  return bitsPerSample === 8
    ? view.getUint8(offset)
    : bitsPerSample === 16
      ? view.getUint16(offset, isLittleEndian)
      : view.getUint32(offset, isLittleEndian);
}

/** Decompresses and unpredicts one block; returns bytes (big-endian floats for predictor 3). */
export function decodeTiffBlock(
  compressed: Uint8Array,
  directory: TiffDirectory,
  blockWidth: number,
  blockHeight: number,
  isLittleEndian: boolean
): Uint8Array {
  const bytesPerSample = directory.bitsPerSample / 8;
  const planeSamples = directory.planarConfiguration === 2 ? 1 : directory.samplesPerPixel;
  const expectedLength = blockWidth * blockHeight * planeSamples * bytesPerSample;
  let block: Uint8Array;
  switch (directory.compression) {
    case 1:
      block = compressed.slice(0, expectedLength);
      break;
    case 8:
    case 32946:
      block = inflateZlib(compressed, expectedLength).slice();
      break;
    case 5:
      block = decodeTiffLzw(compressed, expectedLength);
      break;
    case 32773:
      block = decodePackBits(compressed, expectedLength);
      break;
    default:
      throw new Error(`GeoTIFF compression ${directory.compression} is not supported`);
  }
  if (block.length < expectedLength) throw new Error('GeoTIFF block is truncated');
  if (directory.predictor === 2)
    undoHorizontalPredictor(
      block,
      blockWidth,
      blockHeight,
      planeSamples,
      directory.bitsPerSample,
      isLittleEndian
    );
  else if (directory.predictor === 3)
    undoFloatingPointPredictor(block, blockWidth, blockHeight, planeSamples, bytesPerSample);
  else if (directory.predictor !== 1) throw new Error('GeoTIFF predictor is not supported');
  return block;
}

/** Chooses the directory index to read. */
export function chooseCogLevel(
  directories: readonly TiffDirectory[],
  options: Pick<CogReadOptions, 'maximumPixelCount' | 'targetWidth' | 'overviewLevel'>
): number {
  const levels = directories
    .map((directory, index) => ({directory, index}))
    .filter(({directory, index}) => index === 0 || directory.isReducedResolution);
  if (options.overviewLevel !== undefined)
    return clamp(options.overviewLevel, 0, levels.length - 1);
  if (options.targetWidth !== undefined) {
    // Coarsest level still at least as wide as requested; the finest when none is.
    let chosen = 0;
    levels.forEach(({directory}, position) => {
      if (directory.width >= (options.targetWidth ?? 0)) chosen = position;
    });
    return chosen;
  }
  const maximum = options.maximumPixelCount ?? 16_000_000;
  const position = levels.findIndex(({directory}) => directory.width * directory.height <= maximum);
  return position === -1 ? levels.length - 1 : position;
}

/**
 * Reads a (Cloud Optimized) GeoTIFF: classic or Big TIFF, strip or tiled, chunky or planar,
 * uncompressed / DEFLATE / LZW / PackBits, predictors 2 and 3, any overview level, and only the
 * blocks that intersect `window`. With an HTTP Range `ByteSource` only the header, directories
 * and needed blocks are transferred. Output is one north-up tile in EPSG:4326 or EPSG:3857.
 */
export async function readCloudOptimizedGeoTIFF(
  source: ByteSource,
  options: CogReadOptions = {}
): Promise<CogReadResult> {
  const header = await readTiffHeader(createPrefetchedByteSource(source));
  const {directories, isLittleEndian} = header;
  const base = directories[0];
  if (base.hasModelTransformation)
    throw new Error('Rotated GeoTIFF (ModelTransformation) is not supported');
  if (!base.pixelScale || !base.tiepoint || base.tiepoint.length !== 6)
    throw new Error('GeoTIFF needs ModelPixelScale and exactly one ModelTiepoint');
  const geoKeys = base.geoKeys ?? [];
  let crsCode: number | undefined;
  let isPixelIsPoint = false;
  for (let keyIndex = 0; keyIndex < (geoKeys[3] ?? 0); keyIndex++) {
    const key = geoKeys[4 + keyIndex * 4];
    const value = geoKeys[7 + keyIndex * 4];
    if (key === 2048 || key === 3072) crsCode = value;
    if (key === 1025) isPixelIsPoint = value === 2;
  }
  if (crsCode === undefined) throw new Error('GeoTIFF has no EPSG code');
  const crs = parseRasterCrs(crsCode);
  const levelIndex = chooseCogLevel(directories, options);
  const levels = directories.filter(
    (directory, index) => index === 0 || directory.isReducedResolution
  );
  const directory = levels[levelIndex];
  const arrayType = SAMPLE_ARRAYS[`${directory.sampleFormat}-${directory.bitsPerSample}`];
  if (!arrayType) throw new Error('Unsupported GeoTIFF sample type');
  // Overviews carry no georeferencing: scale the full-resolution pixel size.
  const scaleX = base.width / directory.width;
  const scaleY = base.height / directory.height;
  const pixelWidth = base.pixelScale[0] * scaleX;
  const pixelHeight = base.pixelScale[1] * scaleY;
  const halfX = isPixelIsPoint ? base.pixelScale[0] / 2 : 0;
  const halfY = isPixelIsPoint ? base.pixelScale[1] / 2 : 0;
  const west = base.tiepoint[3] - base.tiepoint[0] * base.pixelScale[0] - halfX;
  const north = base.tiepoint[4] + base.tiepoint[1] * base.pixelScale[1] + halfY;
  // Pixel window of the request, clamped to the level.
  let columnStart = 0;
  let rowStart = 0;
  let columnEnd = directory.width;
  let rowEnd = directory.height;
  if (options.window) {
    const [windowWest, windowSouth, windowEast, windowNorth] = options.window;
    columnStart = Math.max(0, Math.floor((windowWest - west) / pixelWidth));
    columnEnd = Math.min(directory.width, Math.ceil((windowEast - west) / pixelWidth));
    rowStart = Math.max(0, Math.floor((north - windowNorth) / pixelHeight));
    rowEnd = Math.min(directory.height, Math.ceil((north - windowSouth) / pixelHeight));
    if (columnEnd <= columnStart || rowEnd <= rowStart)
      throw new Error('GeoTIFF window does not intersect the raster');
  }
  const outputWidth = columnEnd - columnStart;
  const outputHeight = rowEnd - rowStart;
  const bytesPerSample = directory.bitsPerSample / 8;
  const planeCount = directory.planarConfiguration === 2 ? directory.samplesPerPixel : 1;
  const bandValues = Array.from(
    {length: directory.samplesPerPixel},
    () => new arrayType(new ArrayBuffer(outputWidth * outputHeight * bytesPerSample))
  );
  const blocksAcross = Math.ceil(directory.width / directory.tileWidth);
  const blocksDown = Math.ceil(directory.height / directory.tileHeight);
  const blocksPerPlane = blocksAcross * blocksDown;
  const firstBlockColumn = Math.floor(columnStart / directory.tileWidth);
  const lastBlockColumn = Math.floor((columnEnd - 1) / directory.tileWidth);
  const firstBlockRow = Math.floor(rowStart / directory.tileHeight);
  const lastBlockRow = Math.floor((rowEnd - 1) / directory.tileHeight);
  let fetchedBlockCount = 0;
  for (let plane = 0; plane < planeCount; plane++)
    for (let blockRow = firstBlockRow; blockRow <= lastBlockRow; blockRow++)
      for (let blockColumn = firstBlockColumn; blockColumn <= lastBlockColumn; blockColumn++) {
        options.signal?.throwIfAborted();
        const blockIndex = plane * blocksPerPlane + blockRow * blocksAcross + blockColumn;
        const byteCount = directory.blockByteCounts[blockIndex];
        // Sparse COG blocks (offset 0, count 0) are nodata fill; leave the zero-initialised samples.
        if (!byteCount || !directory.blockOffsets[blockIndex]) continue;
        const compressed = await source.read(directory.blockOffsets[blockIndex], byteCount);
        fetchedBlockCount++;
        const block = decodeTiffBlock(
          compressed,
          directory,
          directory.tileWidth,
          // The last strip of a stripped image holds only the remaining rows.
          directory.isTiled
            ? directory.tileHeight
            : Math.min(directory.tileHeight, directory.height - blockRow * directory.tileHeight),
          isLittleEndian
        );
        const blockView = viewOf(block);
        const isBigEndianBlock = directory.predictor === 3;
        const originColumn = blockColumn * directory.tileWidth;
        const originRow = blockRow * directory.tileHeight;
        const localColumnStart = Math.max(columnStart, originColumn);
        const localColumnEnd = Math.min(columnEnd, originColumn + directory.tileWidth);
        const localRowStart = Math.max(rowStart, originRow);
        const localRowEnd = Math.min(rowEnd, originRow + directory.tileHeight);
        const planeSamples = directory.planarConfiguration === 2 ? 1 : directory.samplesPerPixel;
        for (let row = localRowStart; row < localRowEnd; row++)
          for (let column = localColumnStart; column < localColumnEnd; column++) {
            const pixel = (row - originRow) * directory.tileWidth + (column - originColumn);
            const outputIndex = (row - rowStart) * outputWidth + (column - columnStart);
            if (directory.planarConfiguration === 2)
              bandValues[plane][outputIndex] = readSample(
                blockView,
                pixel * bytesPerSample,
                directory.sampleFormat,
                directory.bitsPerSample,
                isBigEndianBlock ? false : isLittleEndian
              );
            else
              for (let band = 0; band < planeSamples; band++)
                bandValues[band][outputIndex] = readSample(
                  blockView,
                  (pixel * planeSamples + band) * bytesPerSample,
                  directory.sampleFormat,
                  directory.bitsPerSample,
                  isBigEndianBlock ? false : isLittleEndian
                );
          }
      }
  const tileWest = west + columnStart * pixelWidth;
  const tileNorth = north - rowStart * pixelHeight;
  const id = options.id ?? 'geotiff';
  const tile = createNumericRasterTile({
    id: `${id}/0`,
    width: outputWidth,
    height: outputHeight,
    extent: [
      tileWest,
      tileNorth - outputHeight * pixelHeight,
      tileWest + outputWidth * pixelWidth,
      tileNorth
    ],
    bands: bandValues.map(values => ({values, noDataValue: directory.noDataValue}))
  });
  return {
    source: createNumericRasterSource({id, kind: 'single-raster', crs, tiles: [tile]}),
    levelIndex,
    levelCount: levels.length,
    blockCount: blocksPerPlane * planeCount,
    fetchedBlockCount
  };
}
