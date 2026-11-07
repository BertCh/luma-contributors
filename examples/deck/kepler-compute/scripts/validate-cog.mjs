// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Structural Cloud Optimized GeoTIFF validator for classic little/big-endian TIFF files.
 * Parses the IFD chain and checks tiling, overview flags and ordering, tile offsets/byte counts
 * and (for deflate) that every tile inflates to the exact expected size.
 * Usage: node scripts/validate-cog.mjs <file.tif>   |   import {validateCogBytes}.
 * This is NOT GDAL's validate_cloud_optimized_geotiff (no ghost-area check); run gdalinfo when available.
 */
import {readFileSync} from 'node:fs';
import {inflateSync} from 'node:zlib';
import {fileURLToPath} from 'node:url';

const TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8};

function readValues(view, littleEndian, type, count, position) {
  const values = [];
  const size = TYPE_SIZES[type];
  for (let index = 0; index < count; index++) {
    const at = position + index * size;
    if (type === 3) values.push(view.getUint16(at, littleEndian));
    else if (type === 4) values.push(view.getUint32(at, littleEndian));
    else if (type === 12) values.push(view.getFloat64(at, littleEndian));
    else if (type === 1 || type === 2) values.push(view.getUint8(at));
    else values.push(NaN);
  }
  return values;
}

export function parseTiffIfds(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const littleEndian = view.getUint16(0, false) === 0x4949;
  const errors = [];
  if (view.getUint16(2, littleEndian) !== 42) errors.push('not a classic TIFF (BigTIFF unsupported)');
  const ifds = [];
  let ifdOffset = view.getUint32(4, littleEndian);
  const seen = new Set();
  while (ifdOffset !== 0 && !errors.length) {
    if (seen.has(ifdOffset) || ifdOffset + 2 > bytes.byteLength) {
      errors.push(`bad or cyclic IFD offset ${ifdOffset}`);
      break;
    }
    seen.add(ifdOffset);
    const entryCount = view.getUint16(ifdOffset, littleEndian);
    const tags = new Map();
    let maximumMetadataEnd = ifdOffset + 2 + entryCount * 12 + 4;
    for (let entryIndex = 0; entryIndex < entryCount; entryIndex++) {
      const at = ifdOffset + 2 + entryIndex * 12;
      const tag = view.getUint16(at, littleEndian);
      const type = view.getUint16(at + 2, littleEndian);
      const count = view.getUint32(at + 4, littleEndian);
      const byteLength = (TYPE_SIZES[type] ?? 1) * count;
      const valueAt = byteLength <= 4 ? at + 8 : view.getUint32(at + 8, littleEndian);
      if (byteLength > 4) maximumMetadataEnd = Math.max(maximumMetadataEnd, valueAt + byteLength);
      if (valueAt + byteLength > bytes.byteLength) {
        errors.push(`tag ${tag} value out of bounds`);
        continue;
      }
      tags.set(tag, {type, count, values: readValues(view, littleEndian, type, count, valueAt)});
    }
    ifds.push({offset: ifdOffset, tags, metadataEnd: maximumMetadataEnd});
    ifdOffset = view.getUint32(ifdOffset + 2 + entryCount * 12, littleEndian);
  }
  return {ifds, errors};
}

const first = (ifd, tag) => ifd.tags.get(tag)?.values[0];

export function validateCogBytes(bytes) {
  const {ifds, errors} = parseTiffIfds(bytes);
  const report = {levels: [], errors, warnings: []};
  if (!ifds.length) errors.push('no IFDs');
  let previousArea = Infinity;
  let firstTileOffset = Infinity;
  let metadataEnd = 0;
  const levelStarts = [];
  ifds.forEach((ifd, level) => {
    const width = first(ifd, 256);
    const height = first(ifd, 257);
    const tileWidth = first(ifd, 322);
    const tileHeight = first(ifd, 323);
    const compression = first(ifd, 259) ?? 1;
    const samplesPerPixel = first(ifd, 277) ?? 1;
    const bitsPerSample = first(ifd, 258) ?? 8;
    const planar = first(ifd, 284) ?? 1;
    const subfileType = first(ifd, 254) ?? 0;
    const offsets = ifd.tags.get(324)?.values;
    const counts = ifd.tags.get(325)?.values;
    const prefix = `IFD ${level}`;
    metadataEnd = Math.max(metadataEnd, ifd.metadataEnd);
    if (!tileWidth || !tileHeight || !offsets || !counts) {
      errors.push(`${prefix} is not tiled`);
      return;
    }
    if (tileWidth % 16 || tileHeight % 16) errors.push(`${prefix} tile size not a multiple of 16`);
    if (level === 0 && subfileType & 1) errors.push('IFD 0 must be the full resolution image');
    if (level > 0 && !(subfileType & 1)) errors.push(`${prefix} overview lacks reduced-resolution flag`);
    if (width * height >= previousArea) errors.push(`${prefix} overview is not smaller than the previous level`);
    previousArea = width * height;
    const tilesAcross = Math.ceil(width / tileWidth);
    const tilesDown = Math.ceil(height / tileHeight);
    const expectedCount = tilesAcross * tilesDown * (planar === 2 ? samplesPerPixel : 1);
    if (offsets.length !== expectedCount || counts.length !== expectedCount)
      errors.push(`${prefix} has ${offsets.length}/${counts.length} tiles, expected ${expectedCount}`);
    const rawSize =
      tileWidth * tileHeight * (planar === 2 ? 1 : samplesPerPixel) * (bitsPerSample / 8);
    let levelStart = Infinity;
    let previousEnd = -1;
    let isSequential = true;
    offsets.forEach((offset, tileIndex) => {
      const count = counts[tileIndex];
      levelStart = Math.min(levelStart, offset);
      if (offset + count > bytes.byteLength) errors.push(`${prefix} tile ${tileIndex} out of bounds`);
      if (offset < previousEnd) isSequential = false;
      previousEnd = offset + count;
      if (compression === 1 && count !== rawSize) errors.push(`${prefix} tile ${tileIndex} byte count ${count} != ${rawSize}`);
      if (compression === 8) {
        try {
          const inflated = inflateSync(bytes.subarray(offset, offset + count));
          if (inflated.byteLength !== rawSize)
            errors.push(`${prefix} tile ${tileIndex} inflates to ${inflated.byteLength}, expected ${rawSize}`);
        } catch (error) {
          errors.push(`${prefix} tile ${tileIndex} inflate failed: ${error.message}`);
        }
      }
    });
    if (!isSequential) report.warnings.push(`${prefix} tile data not in ascending order`);
    levelStarts.push(levelStart);
    firstTileOffset = Math.min(firstTileOffset, levelStart);
    report.levels.push({width, height, tileWidth, tileHeight, tiles: offsets.length, compression, samplesPerPixel});
  });
  if (metadataEnd > firstTileOffset) errors.push('IFD metadata is not entirely before the first tile (not COG ordered)');
  // COG order: smallest overview first, full resolution last.
  for (let level = 1; level < levelStarts.length; level++)
    if (levelStarts[level] > levelStarts[level - 1])
      errors.push(`overview ${level} data lies after level ${level - 1} (expected smallest first)`);
  report.isValid = errors.length === 0;
  return report;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const path = process.argv[2];
  if (!path) {
    console.error('usage: node scripts/validate-cog.mjs <file.tif>');
    process.exit(2);
  }
  const report = validateCogBytes(new Uint8Array(readFileSync(path)));
  console.log(JSON.stringify(report, null, 2));
  process.exit(report.isValid ? 0 : 1);
}
