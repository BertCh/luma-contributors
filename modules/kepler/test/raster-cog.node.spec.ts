// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {deflateSync} from 'node:zlib';
import {describe, expect, it} from 'vitest';
import {
  createBufferByteSource,
  readCloudOptimizedGeoTIFF,
  type ByteSource
} from '../../../examples/deck/kepler-compute/src/raster-geotiff-cog';
import {
  parseTiffDirectoryChain,
  readNumericGeoTIFF,
  readNumericGeoTIFFWithLevel,
  selectGeoTiffLevel
} from '../../../examples/deck/kepler-compute/src/raster-geotiff-reader';
import {
  decodePackBits,
  decodeTiffLzw,
  inflateRaw,
  inflateZlib
} from '../../../examples/deck/kepler-compute/src/raster-tiff-codecs';

/** TIFF LZW encoder (MSB-first, early change) used only to produce fixtures. */
function encodeTiffLzw(input: Uint8Array): Uint8Array {
  const output: number[] = [];
  let bitBuffer = 0;
  let bitCount = 0;
  let width = 9;
  const writeCode = (code: number) => {
    bitBuffer = (bitBuffer << width) | code;
    bitCount += width;
    while (bitCount >= 8) {
      bitCount -= 8;
      output.push((bitBuffer >>> bitCount) & 255);
    }
    bitBuffer &= (1 << bitCount) - 1;
  };
  let dictionary = new Map<string, number>();
  let nextCode = 258;
  writeCode(256);
  let current = '';
  for (const byte of input) {
    const candidate = current + String.fromCharCode(byte);
    if (candidate.length === 1 || dictionary.has(candidate)) {
      current = candidate;
      continue;
    }
    writeCode(current.length === 1 ? current.charCodeAt(0) : (dictionary.get(current) as number));
    dictionary.set(candidate, nextCode++);
    // The decoder's table lags the encoder's by one entry, so its threshold is one lower.
    const lagged = nextCode - 1;
    width = lagged >= 2047 ? 12 : lagged >= 1023 ? 11 : lagged >= 511 ? 10 : 9;
    if (nextCode >= 4094) {
      writeCode(256);
      dictionary = new Map();
      nextCode = 258;
      width = 9;
    }
    current = String.fromCharCode(byte);
  }
  if (current)
    writeCode(current.length === 1 ? current.charCodeAt(0) : (dictionary.get(current) as number));
  writeCode(257);
  if (bitCount > 0) output.push((bitBuffer << (8 - bitCount)) & 255);
  return Uint8Array.from(output);
}

type DirectoryFixture = {
  width: number;
  height: number;
  tileSize: number;
  compression: number;
  predictor: number;
  isOverview: boolean;
  values: Float32Array | Uint16Array;
  isFloat: boolean;
  isGeoreferenced: boolean;
};

/** Little-endian classic TIFF writer for tiled single-band fixtures with an IFD chain. */
function buildTiff(directories: DirectoryFixture[]): Uint8Array {
  const chunks: {offset: number; bytes: Uint8Array}[] = [];
  let cursor = 8;
  const place = (bytes: Uint8Array): number => {
    const offset = cursor;
    chunks.push({offset, bytes});
    cursor += bytes.length + (bytes.length % 2);
    return offset;
  };
  const directoryOffsets: number[] = [];
  const encodedDirectories = directories.map(fixture => {
    const bytesPerSample = fixture.isFloat ? 4 : 2;
    const tilesAcross = Math.ceil(fixture.width / fixture.tileSize);
    const tilesDown = Math.ceil(fixture.height / fixture.tileSize);
    const offsets: number[] = [];
    const counts: number[] = [];
    for (let tileRow = 0; tileRow < tilesDown; tileRow++)
      for (let tileColumn = 0; tileColumn < tilesAcross; tileColumn++) {
        const block = Buffer.alloc(fixture.tileSize * fixture.tileSize * bytesPerSample);
        for (let row = 0; row < fixture.tileSize; row++)
          for (let column = 0; column < fixture.tileSize; column++) {
            const sourceRow = tileRow * fixture.tileSize + row;
            const sourceColumn = tileColumn * fixture.tileSize + column;
            const value =
              sourceRow < fixture.height && sourceColumn < fixture.width
                ? fixture.values[sourceRow * fixture.width + sourceColumn]
                : 0;
            const offset = (row * fixture.tileSize + column) * bytesPerSample;
            if (fixture.isFloat) block.writeFloatLE(value, offset);
            else block.writeUInt16LE(value, offset);
          }
        if (fixture.predictor === 2) {
          for (let row = 0; row < fixture.tileSize; row++)
            for (let column = fixture.tileSize - 1; column > 0; column--) {
              const offset = (row * fixture.tileSize + column) * bytesPerSample;
              if (fixture.isFloat) continue;
              block.writeUInt16LE(
                (block.readUInt16LE(offset) - block.readUInt16LE(offset - 2)) & 0xffff,
                offset
              );
            }
        }
        const compressed =
          fixture.compression === 8
            ? deflateSync(block)
            : fixture.compression === 5
              ? encodeTiffLzw(block)
              : block;
        offsets.push(place(Uint8Array.from(compressed)));
        counts.push(compressed.length);
      }
    return {fixture, offsets, counts};
  });
  const geoKeys = [1, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326];
  const scalePlacements = encodedDirectories.map(({fixture}) =>
    fixture.isGeoreferenced
      ? {
          scale: place(new Uint8Array(new Float64Array([0.5, 0.5, 0]).buffer)),
          tiepoint: place(new Uint8Array(new Float64Array([0, 0, 0, 10, 20, 0]).buffer)),
          geoKeys: place(new Uint8Array(new Uint16Array(geoKeys).buffer))
        }
      : undefined
  );
  const arrayOffsets = encodedDirectories.map(({offsets, counts}) => ({
    offsets: place(new Uint8Array(new Uint32Array(offsets).buffer)),
    counts: place(new Uint8Array(new Uint32Array(counts).buffer))
  }));
  const noDataText = new TextEncoder().encode('-9999\0');
  const noDataOffset = place(noDataText);
  const directoryBytes = encodedDirectories.map(({fixture, offsets}, index) => {
    const entries: [number, number, number, number][] = [
      [254, 4, 1, fixture.isOverview ? 1 : 0],
      [256, 4, 1, fixture.width],
      [257, 4, 1, fixture.height],
      [258, 3, 1, fixture.isFloat ? 32 : 16],
      [259, 3, 1, fixture.compression],
      [277, 3, 1, 1],
      [284, 3, 1, 1],
      [317, 3, 1, fixture.predictor],
      [322, 4, 1, fixture.tileSize],
      [323, 4, 1, fixture.tileSize],
      [324, 4, offsets.length, arrayOffsets[index].offsets],
      [325, 4, offsets.length, arrayOffsets[index].counts],
      [339, 3, 1, fixture.isFloat ? 3 : 1]
    ];
    const placement = scalePlacements[index];
    if (placement) {
      entries.push([33550, 12, 3, placement.scale], [33922, 12, 6, placement.tiepoint]);
      entries.push([34735, 3, geoKeys.length, placement.geoKeys]);
      entries.push([42113, 2, noDataText.length, noDataOffset]);
    }
    entries.sort((first, second) => first[0] - second[0]);
    const bytes = Buffer.alloc(2 + entries.length * 12 + 4);
    bytes.writeUInt16LE(entries.length, 0);
    entries.forEach(([tag, type, count, value], entryIndex) => {
      const offset = 2 + entryIndex * 12;
      bytes.writeUInt16LE(tag, offset);
      bytes.writeUInt16LE(type, offset + 2);
      bytes.writeUInt32LE(count, offset + 4);
      if (type === 3 && count === 1) bytes.writeUInt16LE(value, offset + 8);
      else bytes.writeUInt32LE(value, offset + 8);
    });
    return bytes;
  });
  directoryBytes.forEach(bytes => directoryOffsets.push(place(Uint8Array.from(bytes))));
  directoryBytes.forEach((bytes, index) => {
    const next = directoryOffsets[index + 1] ?? 0;
    bytes.writeUInt32LE(next, bytes.length - 4);
    const chunk = chunks.find(candidate => candidate.offset === directoryOffsets[index]);
    if (chunk) chunk.bytes = Uint8Array.from(bytes);
  });
  const file = new Uint8Array(cursor);
  file.set([0x49, 0x49, 42, 0]);
  new DataView(file.buffer).setUint32(4, directoryOffsets[0], true);
  for (const chunk of chunks) file.set(chunk.bytes, chunk.offset);
  return file;
}

function makeRamp(width: number, height: number): Float32Array {
  return Float32Array.from({length: width * height}, (_, index) => index * 0.5);
}

function countingSource(bytes: Uint8Array) {
  const inner = createBufferByteSource(bytes);
  const reads: {offset: number; length: number}[] = [];
  const source: ByteSource = {
    read: (offset, length) => {
      reads.push({offset, length});
      return inner.read(offset, length);
    }
  };
  return {source, reads};
}

describe('TIFF codecs', () => {
  const sample = Uint8Array.from({length: 5000}, (_, index) => (index * 7 + (index >> 3)) & 255);

  it('inflates zlib and raw deflate', () => {
    const compressed = deflateSync(sample);
    expect([...inflateZlib(compressed, sample.length)]).toEqual([...sample]);
    expect([...inflateRaw(compressed.subarray(2), sample.length)]).toEqual([...sample]);
  });

  it('decodes TIFF LZW including table growth past 9 bits', () => {
    const repetitive = Uint8Array.from({length: 20000}, (_, index) => (index % 17) * 3);
    expect([...decodeTiffLzw(encodeTiffLzw(sample), sample.length)]).toEqual([...sample]);
    expect([...decodeTiffLzw(encodeTiffLzw(repetitive), repetitive.length)]).toEqual([
      ...repetitive
    ]);
  });

  it('decodes PackBits runs and literals', () => {
    const encoded = Uint8Array.from([2, 1, 2, 3, 0xfd, 9]);
    expect([...decodePackBits(encoded, 7)]).toEqual([1, 2, 3, 9, 9, 9, 9]);
  });
});

describe('cloud optimized GeoTIFF reader', () => {
  const full = makeRamp(10, 8);
  const overview = makeRamp(5, 4);

  function build(compression: number, predictor = 1) {
    return buildTiff([
      {
        width: 10,
        height: 8,
        tileSize: 4,
        compression,
        predictor,
        isOverview: false,
        values: full,
        isFloat: true,
        isGeoreferenced: true
      },
      {
        width: 5,
        height: 4,
        tileSize: 4,
        compression,
        predictor,
        isOverview: true,
        values: overview,
        isFloat: true,
        isGeoreferenced: false
      }
    ]);
  }

  for (const [name, compression] of [
    ['uncompressed', 1],
    ['deflate', 8],
    ['LZW', 5]
  ] as const)
    it(`reads ${name} tiles at full resolution with georeferencing`, async () => {
      const result = await readCloudOptimizedGeoTIFF(createBufferByteSource(build(compression)));
      const tile = result.source.tiles[0];
      expect(result.levelCount).toBe(2);
      expect(tile.width).toBe(10);
      expect([...tile.extent]).toEqual([10, 16, 15, 20]);
      expect([...tile.bands[0].values]).toEqual([...full]);
      expect(tile.bands[0].noDataValue).toBe(-9999);
    });

  it('selects the overview level and scales its pixel size', async () => {
    const result = await readCloudOptimizedGeoTIFF(createBufferByteSource(build(8)), {
      maximumPixelCount: 30
    });
    const tile = result.source.tiles[0];
    expect(result.levelIndex).toBe(1);
    expect(tile.width).toBe(5);
    expect([...tile.extent]).toEqual([10, 16, 15, 20]);
    expect([...tile.bands[0].values]).toEqual([...overview]);
  });

  it('fetches only blocks that intersect a window', async () => {
    const bytes = build(8);
    const {source, reads} = countingSource(bytes);
    const result = await readCloudOptimizedGeoTIFF(source, {
      overviewLevel: 0,
      window: [10, 18.5, 11.5, 20]
    });
    // 3 x 3 pixels at the north-west corner live in exactly one of the six 4x4 tiles.
    expect(result.fetchedBlockCount).toBe(1);
    expect(result.blockCount).toBe(6);
    const tile = result.source.tiles[0];
    expect(tile.width).toBe(3);
    expect(tile.height).toBe(3);
    expect([...tile.bands[0].values]).toEqual([0, 0.5, 1, 5, 5.5, 6, 10, 10.5, 11]);
    expect(reads.some(read => read.offset === 0 && read.length >= 16)).toBe(true);
  });

  it('undoes the horizontal predictor on integer tiles', async () => {
    const values = Uint16Array.from({length: 40}, (_, index) => 100 + index * 3);
    const bytes = buildTiff([
      {
        width: 8,
        height: 5,
        tileSize: 4,
        compression: 8,
        predictor: 2,
        isOverview: false,
        values,
        isFloat: false,
        isGeoreferenced: true
      }
    ]);
    const result = await readCloudOptimizedGeoTIFF(createBufferByteSource(bytes));
    expect([...result.source.tiles[0].bands[0].values]).toEqual([...values]);
  });
});

describe('synchronous GeoTIFF reader on COGs', () => {
  const full = makeRamp(10, 8);
  for (const [name, compression, predictor] of [
    ['uncompressed', 1, 1],
    ['deflate', 8, 1],
    ['deflate with predictor 2', 8, 2],
    ['LZW', 5, 1]
  ] as const)
    it(`reads ${name} tiled overview COGs at full resolution`, () => {
      const bytes = buildTiff([
        {
          width: 10,
          height: 8,
          tileSize: 4,
          compression,
          predictor,
          isOverview: false,
          values: full,
          isFloat: predictor !== 2,
          isGeoreferenced: true
        },
        {
          width: 5,
          height: 4,
          tileSize: 4,
          compression,
          predictor,
          isOverview: true,
          values: makeRamp(5, 4),
          isFloat: predictor !== 2,
          isGeoreferenced: false
        }
      ]);
      const tile = readNumericGeoTIFF(bytes).tiles[0];
      expect(tile.width).toBe(10);
      expect(tile.height).toBe(8);
      expect([...tile.extent]).toEqual([10, 16, 15, 20]);
      expect([...tile.bands[0].values]).toEqual(
        [...full].map(value => (predictor === 2 ? value | 0 : value))
      );
    });
});

describe('GeoTIFF sync reader overviews and caps', () => {
  const full = makeRamp(10, 8);
  const overview = makeRamp(5, 4);
  const bytes = buildTiff([
    {
      width: 10,
      height: 8,
      tileSize: 4,
      compression: 1,
      predictor: 1,
      isOverview: false,
      values: full,
      isFloat: true,
      isGeoreferenced: true
    },
    {
      width: 5,
      height: 4,
      tileSize: 4,
      compression: 1,
      predictor: 1,
      isOverview: true,
      values: overview,
      isFloat: true,
      isGeoreferenced: false
    }
  ]);

  it('admits bounded embedding windows without silently choosing a generic overview', async () => {
    for (const limits of [
      {maximumOutputPixelCount: 20},
      {maximumOutputElementCount: 20},
      {maximumOutputBandCount: 0}
    ])
      await expect(
        readCloudOptimizedGeoTIFF(createBufferByteSource(bytes), {overviewLevel: 0, ...limits})
      ).rejects.toThrow('decoded window exceeds');
    const result = await readCloudOptimizedGeoTIFF(createBufferByteSource(bytes), {
      overviewLevel: 0,
      maximumOutputPixelCount: 4,
      maximumOutputBandCount: 1,
      maximumOutputElementCount: 4,
      window: [10, 19, 11, 20]
    });
    expect(result.levelIndex).toBe(0);
    expect(result.source.tiles[0].width * result.source.tiles[0].height).toBe(4);
    expect(Array.from(result.source.tiles[0].bands[0].values)).toEqual([
      full[0],
      full[1],
      full[10],
      full[11]
    ]);
  });

  it('parses the directory chain', () => {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const directories = parseTiffDirectoryChain(bytes, view, true, false);
    expect(directories.map(directory => [directory.width, directory.isReducedResolution])).toEqual([
      [10, false],
      [5, true]
    ]);
  });

  it('selects levels by decoded size and target pixel size', () => {
    const levels = [
      {width: 1000, height: 1000, samplesPerPixel: 1, bitsPerSample: 32},
      {width: 500, height: 500, samplesPerPixel: 1, bitsPerSample: 32},
      {width: 250, height: 250, samplesPerPixel: 1, bitsPerSample: 32}
    ];
    expect(selectGeoTiffLevel(levels)).toEqual({levelIndex: 0, reason: 'full-resolution'});
    expect(selectGeoTiffLevel(levels, {maximumDecodedByteLength: 1_500_000})).toEqual({
      levelIndex: 1,
      reason: 'decoded-size-cap'
    });
    expect(selectGeoTiffLevel(levels, {targetPixelSize: 400})).toEqual({
      levelIndex: 1,
      reason: 'target-pixel-size'
    });
    expect(selectGeoTiffLevel(levels, {targetPixelSize: 100}).levelIndex).toBe(2);
    expect(selectGeoTiffLevel(levels, {maximumDecodedByteLength: 1}).levelIndex).toBe(2);
  });

  it('decodes an overview that shares the full-resolution extent', () => {
    const result = readNumericGeoTIFFWithLevel(bytes, 'cog', undefined, {
      maximumDecodedByteLength: 200
    });
    expect(result).toMatchObject({
      levelIndex: 1,
      levelCount: 2,
      fullWidth: 10,
      fullHeight: 8,
      isReducedResolution: true,
      reason: 'decoded-size-cap'
    });
    const tile = result.source.tiles[0];
    expect([tile.width, tile.height]).toEqual([5, 4]);
    expect([...tile.extent]).toEqual([10, 16, 15, 20]);
    expect([...tile.bands[0].values]).toEqual([...overview]);
    expect(tile.bands[0].noDataValue).toBe(-9999);
  });

  it('keeps full resolution by default and refuses when nothing fits', () => {
    expect(readNumericGeoTIFF(bytes).tiles[0].width).toBe(10);
    expect(readNumericGeoTIFFWithLevel(bytes).reason).toBe('full-resolution');
    expect(() =>
      readNumericGeoTIFF(bytes, 'cog', undefined, {maximumDecodedByteLength: 10})
    ).toThrow();
  });
});
