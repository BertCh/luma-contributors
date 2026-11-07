// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {decodeTiffBlock, type TiffDirectory} from './raster-geotiff-cog';
import {parseRasterCrs} from './raster-reprojection';
import {
  createNumericRasterSource,
  createNumericRasterTile,
  type NumericRasterSource,
  type NumericRasterValues
} from './raster-tile-ingest';

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

function readTagValue(view: DataView, type: number, offset: number, little: boolean): number {
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

/** Synchronous parse of the first (full-resolution) image directory of an in-memory TIFF. */
export function parseFirstTiffDirectory(
  bytes: Uint8Array,
  view: DataView,
  isLittleEndian: boolean,
  isBigTiff: boolean
): TiffDirectory {
  return parseTiffDirectoryAt(
    bytes,
    view,
    isLittleEndian,
    isBigTiff,
    isBigTiff ? Number(view.getBigUint64(8, isLittleEndian)) : view.getUint32(4, isLittleEndian)
  ).directory;
}

/** Parses the directory chain (full resolution first, then reduced-resolution levels; masks skipped). */
export function parseTiffDirectoryChain(
  bytes: Uint8Array,
  view: DataView,
  isLittleEndian: boolean,
  isBigTiff: boolean,
  maximumDirectoryCount = 32
): TiffDirectory[] {
  const directories: TiffDirectory[] = [];
  let offset = isBigTiff
    ? Number(view.getBigUint64(8, isLittleEndian))
    : view.getUint32(4, isLittleEndian);
  while (offset > 0 && offset < bytes.byteLength && directories.length < maximumDirectoryCount) {
    const parsed = parseTiffDirectoryAt(bytes, view, isLittleEndian, isBigTiff, offset);
    // NewSubfileType bit 2 marks a transparency mask, which is not a raster level.
    if (!parsed.isMask) directories.push(parsed.directory);
    offset = parsed.nextOffset;
  }
  return directories;
}

type ParsedTiffDirectory = {directory: TiffDirectory; nextOffset: number; isMask: boolean};

function parseTiffDirectoryAt(
  bytes: Uint8Array,
  view: DataView,
  isLittleEndian: boolean,
  isBigTiff: boolean,
  directoryOffset: number
): ParsedTiffDirectory {
  const entryCount = isBigTiff
    ? Number(view.getBigUint64(directoryOffset, isLittleEndian))
    : view.getUint16(directoryOffset, isLittleEndian);
  const entryBytes = isBigTiff ? 20 : 12;
  const entriesStart = directoryOffset + (isBigTiff ? 8 : 2);
  const inlineLimit = isBigTiff ? 8 : 4;
  const tags = new Map<number, number[] | string>();
  for (let entryIndex = 0; entryIndex < entryCount; entryIndex++) {
    const entryOffset = entriesStart + entryIndex * entryBytes;
    const tag = view.getUint16(entryOffset, isLittleEndian);
    const type = view.getUint16(entryOffset + 2, isLittleEndian);
    const count = isBigTiff
      ? Number(view.getBigUint64(entryOffset + 4, isLittleEndian))
      : view.getUint32(entryOffset + 4, isLittleEndian);
    const componentBytes = TYPE_BYTES[type] ?? 1;
    const valueFieldOffset = entryOffset + (isBigTiff ? 12 : 8);
    const dataOffset =
      componentBytes * count <= inlineLimit
        ? valueFieldOffset
        : isBigTiff
          ? Number(view.getBigUint64(valueFieldOffset, isLittleEndian))
          : view.getUint32(valueFieldOffset, isLittleEndian);
    if (type === 2) {
      tags.set(
        tag,
        new TextDecoder()
          .decode(bytes.subarray(dataOffset, dataOffset + count))
          .replace(/\0.*$/s, '')
      );
      continue;
    }
    const values: number[] = new Array(count);
    for (let componentIndex = 0; componentIndex < count; componentIndex++)
      values[componentIndex] = readTagValue(
        view,
        type,
        dataOffset + componentIndex * componentBytes,
        isLittleEndian
      );
    tags.set(tag, values);
  }
  const getNumbers = (tag: number): number[] | undefined => {
    const value = tags.get(tag);
    return Array.isArray(value) ? value : undefined;
  };
  const width = getNumbers(256)?.[0];
  const height = getNumbers(257)?.[0];
  const isTiled = tags.has(322);
  const blockOffsets = getNumbers(isTiled ? 324 : 273);
  const blockByteCounts = getNumbers(isTiled ? 325 : 279);
  if (!width || !height || !blockOffsets || !blockByteCounts)
    throw new Error('Unsupported GeoTIFF layout');
  const noDataText = tags.get(42113);
  const noDataValue =
    typeof noDataText === 'string' && noDataText.trim() ? Number(noDataText.trim()) : null;
  const nextOffsetPosition = entriesStart + entryCount * entryBytes;
  const nextOffset =
    nextOffsetPosition + (isBigTiff ? 8 : 4) <= bytes.byteLength
      ? isBigTiff
        ? Number(view.getBigUint64(nextOffsetPosition, isLittleEndian))
        : view.getUint32(nextOffsetPosition, isLittleEndian)
      : 0;
  const directory: TiffDirectory = {
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
  };
  return {directory, nextOffset, isMask: ((getNumbers(254)?.[0] ?? 0) & 4) === 4};
}

export type GeoTiffReadOptions = Readonly<{
  /**
   * Ceiling for the decoded samples of the chosen level (all bands). A finer level that would exceed
   * it is skipped in favour of the next overview. Default 256 MiB.
   */
  maximumDecodedByteLength?: number;
  /**
   * Prefer the coarsest level whose longer side still has at least this many pixels (for example the
   * canvas size) instead of always decoding full resolution. Ignored when absent.
   */
  targetPixelSize?: number;
}>;

export type GeoTiffReadResult = Readonly<{
  source: NumericRasterSource;
  /** Index into the directory chain that was decoded (0 is full resolution). */
  levelIndex: number;
  levelCount: number;
  fullWidth: number;
  fullHeight: number;
  /** True when an overview was decoded because of the size cap or `targetPixelSize`. */
  isReducedResolution: boolean;
  /** Why an overview was chosen, for diagnostics. */
  reason: 'full-resolution' | 'target-pixel-size' | 'decoded-size-cap';
}>;

const DEFAULT_MAXIMUM_DECODED_BYTE_LENGTH = 256 * 1024 * 1024;

function getDecodedByteLength(directory: TiffDirectory): number {
  return (
    directory.width * directory.height * directory.samplesPerPixel * (directory.bitsPerSample / 8)
  );
}

/**
 * Picks the level to decode: the finest one that fits `maximumDecodedByteLength`, moved to a
 * coarser one while the next level still has `targetPixelSize` pixels on its longer side. Pure.
 */
export function selectGeoTiffLevel(
  directories: readonly Pick<
    TiffDirectory,
    'width' | 'height' | 'samplesPerPixel' | 'bitsPerSample'
  >[],
  options: GeoTiffReadOptions = {}
): {levelIndex: number; reason: GeoTiffReadResult['reason']} {
  const maximumBytes = options.maximumDecodedByteLength ?? DEFAULT_MAXIMUM_DECODED_BYTE_LENGTH;
  const sizeOf = (directory: (typeof directories)[number]) =>
    directory.width * directory.height * directory.samplesPerPixel * (directory.bitsPerSample / 8);
  let levelIndex = 0;
  let reason: GeoTiffReadResult['reason'] = 'full-resolution';
  while (levelIndex < directories.length - 1 && sizeOf(directories[levelIndex]) > maximumBytes) {
    levelIndex++;
    reason = 'decoded-size-cap';
  }
  const target = options.targetPixelSize;
  if (target && target > 0)
    while (
      levelIndex < directories.length - 1 &&
      Math.max(directories[levelIndex + 1].width, directories[levelIndex + 1].height) >= target
    ) {
      levelIndex++;
      if (reason === 'full-resolution') reason = 'target-pixel-size';
    }
  return {levelIndex, reason};
}

/**
 * Synchronous GeoTIFF reader for in-memory bytes: classic or Big TIFF, strip or tiled, chunky or
 * planar, uncompressed / DEFLATE / LZW / PackBits with predictors 2 and 3, 8/16/32 bit integer or
 * 32/64 bit float samples, north-up with ModelPixelScale plus one ModelTiepoint, in EPSG:4326 or
 * EPSG:3857. Decodes full resolution unless the size cap or `targetPixelSize` selects an overview
 * (the overview shares the full-resolution extent). Rotation, unsupported codecs and other CRSs are
 * refused with an explicit error. Windowed reads over HTTP use `readCloudOptimizedGeoTIFF`.
 */
export function readNumericGeoTIFF(
  bytes: Uint8Array,
  id = 'geotiff',
  maximumByteLength = 64 * 1024 * 1024,
  options: GeoTiffReadOptions = {}
): NumericRasterSource {
  return readNumericGeoTIFFWithLevel(bytes, id, maximumByteLength, options).source;
}

/** Like `readNumericGeoTIFF`, but also reports which level was decoded and why. */
export function readNumericGeoTIFFWithLevel(
  bytes: Uint8Array,
  id = 'geotiff',
  maximumByteLength = 64 * 1024 * 1024,
  options: GeoTiffReadOptions = {}
): GeoTiffReadResult {
  // Refuse before touching the buffer: bounded memory is part of the contract.
  if (bytes.byteLength > maximumByteLength) throw new Error('GeoTIFF exceeds the ingestion limit');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const byteOrder = view.getUint16(0, false);
  const isLittleEndian = byteOrder === 0x4949;
  if (!isLittleEndian && byteOrder !== 0x4d4d) throw new Error('Not a TIFF file');
  const magic = view.getUint16(2, isLittleEndian);
  if (magic !== 42 && magic !== 43) throw new Error('Not a TIFF file');
  const isBigTiff = magic === 43;
  const directories = parseTiffDirectoryChain(bytes, view, isLittleEndian, isBigTiff);
  const fullDirectory = directories[0];
  if (!fullDirectory) throw new Error('Unsupported GeoTIFF layout');
  // Only levels that are real reductions of the first image are candidates.
  const candidates = directories.filter(
    (directory, index) =>
      index === 0 ||
      (directory.isReducedResolution &&
        directory.width <= fullDirectory.width &&
        directory.height <= fullDirectory.height &&
        directory.samplesPerPixel === fullDirectory.samplesPerPixel &&
        directory.bitsPerSample === fullDirectory.bitsPerSample)
  );
  const {levelIndex, reason} = selectGeoTiffLevel(candidates, options);
  const directory = candidates[levelIndex];
  if (
    getDecodedByteLength(directory) >
    (options.maximumDecodedByteLength ?? DEFAULT_MAXIMUM_DECODED_BYTE_LENGTH)
  )
    throw new Error('GeoTIFF decoded size exceeds the ingestion limit and has no smaller overview');
  const {width: fullWidth, height: fullHeight} = fullDirectory;
  if (fullDirectory.hasModelTransformation)
    throw new Error('Rotated GeoTIFF (ModelTransformation) is not supported');
  const pixelScale = fullDirectory.pixelScale;
  const tiepoint = fullDirectory.tiepoint;
  if (!pixelScale || !tiepoint || tiepoint.length !== 6)
    throw new Error('GeoTIFF needs ModelPixelScale and exactly one ModelTiepoint');
  // GeoKey directory: [version, revision, minor, count, (key, location, count, value)*]
  const geoKeys = fullDirectory.geoKeys ?? [];
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
  const bandValues = decodeDirectoryBands(bytes, directory, isLittleEndian);
  const halfPixelX = isPixelIsPoint ? pixelScale[0] / 2 : 0;
  const halfPixelY = isPixelIsPoint ? pixelScale[1] / 2 : 0;
  const west = tiepoint[3] - tiepoint[0] * pixelScale[0] - halfPixelX;
  const north = tiepoint[4] + tiepoint[1] * pixelScale[1] + halfPixelY;
  const tile = createNumericRasterTile({
    id: `${id}/0`,
    width: directory.width,
    height: directory.height,
    // Every level covers the full-resolution extent.
    extent: [west, north - fullHeight * pixelScale[1], west + fullWidth * pixelScale[0], north],
    bands: bandValues.map(values => ({
      values,
      noDataValue: fullDirectory.noDataValue ?? directory.noDataValue
    }))
  });
  return {
    source: createNumericRasterSource({id, kind: 'single-raster', crs, tiles: [tile]}),
    levelIndex,
    levelCount: candidates.length,
    fullWidth,
    fullHeight,
    isReducedResolution: levelIndex > 0,
    reason
  };
}

function decodeDirectoryBands(
  bytes: Uint8Array,
  directory: TiffDirectory,
  isLittleEndian: boolean
): NumericRasterValues[] {
  const {width, height, samplesPerPixel, bitsPerSample, sampleFormat} = directory;
  const ArrayConstructor = SAMPLE_ARRAYS[`${sampleFormat}-${bitsPerSample}`];
  if (!ArrayConstructor) throw new Error('Unsupported GeoTIFF sample type');
  const bytesPerSample = bitsPerSample / 8;
  const sampleCount = width * height;
  const isPlanar = directory.planarConfiguration === 2;
  const planeCount = isPlanar ? samplesPerPixel : 1;
  const planeSamples = isPlanar ? 1 : samplesPerPixel;
  const bandValues = Array.from(
    {length: samplesPerPixel},
    () => new ArrayConstructor(new ArrayBuffer(sampleCount * bytesPerSample))
  );
  const blocksAcross = Math.ceil(width / directory.tileWidth);
  const blocksDown = Math.ceil(height / directory.tileHeight);
  const blocksPerPlane = blocksAcross * blocksDown;
  for (let plane = 0; plane < planeCount; plane++)
    for (let blockRow = 0; blockRow < blocksDown; blockRow++)
      for (let blockColumn = 0; blockColumn < blocksAcross; blockColumn++) {
        const blockIndex = plane * blocksPerPlane + blockRow * blocksAcross + blockColumn;
        const byteCount = directory.blockByteCounts[blockIndex];
        const blockOffset = directory.blockOffsets[blockIndex];
        // Sparse COG blocks (offset 0, count 0) are nodata fill; leave the zero-initialised samples.
        if (!byteCount || !blockOffset) continue;
        if (blockOffset + byteCount > bytes.byteLength)
          throw new Error('GeoTIFF strip data is truncated');
        const block = decodeTiffBlock(
          bytes.subarray(blockOffset, blockOffset + byteCount),
          directory,
          directory.tileWidth,
          directory.isTiled
            ? directory.tileHeight
            : Math.min(directory.tileHeight, height - blockRow * directory.tileHeight),
          isLittleEndian
        );
        const blockView = new DataView(block.buffer, block.byteOffset, block.byteLength);
        // Predictor 3 leaves big-endian float bytes.
        const blockLittleEndian = directory.predictor === 3 ? false : isLittleEndian;
        const originColumn = blockColumn * directory.tileWidth;
        const originRow = blockRow * directory.tileHeight;
        const rowEnd = Math.min(height, originRow + directory.tileHeight);
        const columnEnd = Math.min(width, originColumn + directory.tileWidth);
        for (let row = originRow; row < rowEnd; row++)
          for (let column = originColumn; column < columnEnd; column++) {
            const pixel = (row - originRow) * directory.tileWidth + (column - originColumn);
            const outputIndex = row * width + column;
            for (let band = 0; band < planeSamples; band++)
              bandValues[isPlanar ? plane : band][outputIndex] = readSample(
                blockView,
                (pixel * planeSamples + band) * bytesPerSample,
                sampleFormat,
                bitsPerSample,
                blockLittleEndian
              );
          }
      }
  return bandValues;
}

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
