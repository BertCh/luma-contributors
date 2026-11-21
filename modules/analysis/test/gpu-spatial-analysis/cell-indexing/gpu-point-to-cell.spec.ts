// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {cellToLatLng, latLngToCell} from 'h3-js';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUPointToCell,
  type GPUCellIndexFamily
} from '../../../src/gpu-spatial-analysis/cell-indexing/index';
import {h3ToBigInt, quadbinPointToCell} from '../cell-aggregation/cell-aggregation-oracle';
import {createRandom} from '../cell-aggregation/cell-aggregation-points';
import {
  createVectorView,
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  geohashPointToCell,
  getAngularDistance,
  joinKey,
  quadkeyPointToCell,
  s2CellToCenter,
  s2PointToCell
} from './cell-indexing-oracle';
import {requireTestDevice} from '../../utils/require-test-device';
import h3FixtureJson from '../../oracles/fixtures/cell-indexing-h3.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';

const ROWS = 10240;
/** Rows before the hand-picked edge cases (seams, poles) in `createPoints`. */
const UNIFORM_COUNT = ROWS - 64;

/** Random points plus poles, antimeridian, face seams, signed zero, NaN and infinities. */
function createPoints(seed: number): Float32Array {
  const random = createRandom(seed);
  const points: number[] = [];
  for (let index = 0; index < UNIFORM_COUNT; index++) {
    const longitude = random() * 360 - 180;
    // Equal-area latitudes keep the poles populated.
    const latitude = (Math.asin(random() * 2 - 1) * 180) / Math.PI;
    points.push(longitude, latitude);
  }
  const special: [number, number][] = [
    [0, 90],
    [0, -90],
    [180, 90],
    [-180, -90],
    [180, 0],
    [-180, 0],
    [0, 0],
    [-0, -0],
    [45, 0],
    [-45, 0],
    [135, 0],
    [-135, 0],
    [0, 35.264389],
    [90, -35.264389],
    [179.99999, 85.05113],
    [-179.99999, -85.05113],
    [10.40744, 57.64911],
    [-3.7038, 40.4168],
    [-111.709298, 40.256312],
    [1e-30, 1e-30],
    [-1e-40, 1e-40],
    [360, 10],
    [-540, -10],
    [NaN, 10],
    [10, NaN],
    [NaN, NaN],
    [Infinity, 10],
    [10, -Infinity]
  ];
  for (const [longitude, latitude] of special) {
    points.push(longitude, latitude);
  }
  while (points.length < 2 * ROWS) {
    points.push(random() * 360 - 180, random() * 170 - 85);
  }
  return Float32Array.from(points);
}

/** One compiled `GPUPointToCell` graph over a writable position and mask buffer. */
function createIndexer(
  device: Device,
  family: GPUCellIndexFamily,
  resolution: number,
  options: {mask?: boolean; validity?: boolean} = {}
) {
  const positions = createInputBuffer(device, new Float32Array(2 * ROWS));
  const mask = createInputBuffer(device, new Uint32Array(ROWS).fill(1));
  const cells = createOutputBuffer(device, 2 * ROWS);
  const validity = createOutputBuffer(device, ROWS);
  const graph = new GPUCommandGraph(device, {id: `point-to-cell-${family}-${resolution}`});
  graph.add(
    new GPUPointToCell({
      family,
      resolution,
      positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', ROWS),
      mask: options.mask ? importGraphBuffer(graph, 'mask', mask, 'uint32', ROWS) : undefined,
      output: {
        cells: importGraphBuffer(graph, 'cells', cells, 'uint32x2', ROWS),
        validity: options.validity
          ? importGraphBuffer(graph, 'validity', validity, 'uint32', ROWS)
          : undefined
      }
    })
  );
  const compiled = graph.compile();
  const buffers: Buffer[] = [positions, mask, cells, validity];
  return {
    compiled,
    async run(points: Float32Array, rowMask?: Uint32Array) {
      positions.write(points);
      if (rowMask) {
        mask.write(rowMask);
      }
      submitGraph(device, compiled, undefined);
      const words = await readUint32(cells, 2 * ROWS);
      const keys: bigint[] = [];
      for (let row = 0; row < ROWS; row++) {
        keys.push(joinKey(words[2 * row], words[2 * row + 1]));
      }
      const valid = options.validity ? await readUint32(validity, ROWS) : undefined;
      return {keys, valid};
    },
    destroy() {
      compiled.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

const isFinitePoint = (points: Float32Array, row: number) =>
  Number.isFinite(points[2 * row]) && Number.isFinite(points[2 * row + 1]);

/** Runs `family` at `resolution` and checks every row against an exact oracle. */
async function expectExact(
  device: Device,
  family: GPUCellIndexFamily,
  resolution: number,
  oracle: (longitude: number, latitude: number) => bigint
) {
  const points = createPoints(resolution + 11);
  const indexer = createIndexer(device, family, resolution, {validity: true});
  const {keys, valid} = await indexer.run(points);
  let checked = 0;
  for (let row = 0; row < ROWS; row++) {
    if (!isFinitePoint(points, row)) {
      expect(keys[row], `${family} ${resolution} non-finite row ${row}`).toBe(0n);
      expect(valid![row]).toBe(0);
      continue;
    }
    const expected = oracle(points[2 * row], points[2 * row + 1]);
    if (keys[row] !== expected) {
      expect(
        keys[row].toString(16),
        `${family} ${resolution} row ${row} (${points[2 * row]}, ${points[2 * row + 1]})`
      ).toBe(expected.toString(16));
    }
    expect(valid![row]).toBe(1);
    checked++;
  }
  expect(checked).toBeGreaterThan(ROWS - 10);
  indexer.destroy();
}

it('GPUPointToCell quadbin equals the BigInt oracle bit for bit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const resolution of [0, 1, 5, 10, 15, 16, 20, 26]) {
    await expectExact(device, 'quadbin', resolution, (lng, lat) =>
      quadbinPointToCell(lng, lat, resolution)
    );
  }
});

it('GPUPointToCell quadkey equals the BigInt oracle bit for bit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const zoom of [1, 2, 6, 12, 20, 26, 27, 29]) {
    await expectExact(device, 'quadkey', zoom, (lng, lat) => quadkeyPointToCell(lng, lat, zoom));
  }
});

it('GPUPointToCell geohash equals the bisection oracle bit for bit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const length of [1, 2, 3, 5, 8, 11, 12]) {
    await expectExact(device, 'geohash', length, (lng, lat) =>
      geohashPointToCell(lng, lat, length)
    );
  }
});

it('GPUPointToCell S2 mismatch rate against the f64 reference stays bounded per level', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const points = createPoints(5);
  const rates: Record<number, string> = {};
  const widths: Record<number, string> = {};
  // Measured on this suite's 10176 equal-area points (GPU f32 vs f64 reference): no mismatch
  // through level 10, 1e-4 at 11, 5e-4 at 12, 2.6e-3 at 15, 6.6e-2 at 20, 0.26 at 22, 0.78 at 24
  // and about every point beyond 26. The bound is about twice the measurement.
  const maximumRates = [
    ...Array(11).fill(5e-4),
    5e-4,
    1e-3,
    2e-3,
    4e-3,
    5e-3,
    1e-2,
    2e-2,
    4e-2,
    7e-2,
    0.13,
    0.26,
    0.5,
    0.8,
    0.95,
    1,
    1,
    1,
    1,
    1,
    1,
    1
  ];
  // Mismatches are edge neighbors (center distance at most about 1.55 cell widths) through level
  // 22; past it the f32 error exceeds one cell and grows with 2^(level - 22).
  const maximumWidths = (level: number) => (level <= 22 ? 2.5 : 1.6 * 2 ** (level - 22));
  for (let level = 0; level <= 30; level++) {
    const indexer = createIndexer(device, 's2', level, {validity: true});
    const {keys, valid} = await indexer.run(points);
    let mismatches = 0;
    let finiteRows = 0;
    let maximumDistance = 0;
    for (let row = 0; row < ROWS; row++) {
      if (!isFinitePoint(points, row)) {
        expect(keys[row]).toBe(0n);
        expect(valid![row]).toBe(0);
        continue;
      }
      finiteRows += row < UNIFORM_COUNT ? 1 : 0;
      expect(valid![row]).toBe(1);
      const expected = s2PointToCell(points[2 * row], points[2 * row + 1], level);
      if (keys[row] !== expected) {
        // Hand-picked rows sit exactly on face seams or poles; they are checked for being a
        // neighbor but not counted in the random-point rate.
        mismatches += row < UNIFORM_COUNT ? 1 : 0;
        // The GPU cell is the reference cell's neighbor: within ~2.5 cell widths of its center.
        const distance = getAngularDistance(s2CellToCenter(keys[row]), s2CellToCenter(expected));
        maximumDistance = Math.max(maximumDistance, distance / (Math.PI / 2 / 2 ** level));
      }
    }
    // Wrong-level or non-S2 keys would show here: the marker bit and face are always valid.
    widths[level] = maximumDistance.toFixed(2);
    expect(maximumDistance, `level ${level} center distance in cell widths`).toBeLessThan(
      maximumWidths(level)
    );
    const rate = mismatches / finiteRows;
    rates[level] = rate.toExponential(2);
    expect(rate, `level ${level}`).toBeLessThanOrEqual(maximumRates[level]);
    indexer.destroy();
  }
  // eslint-disable-next-line no-console
  console.log(
    'S2 f32 GPU vs f64 oracle, mismatch rate per level',
    rates,
    'max center distance in cell widths',
    widths
  );
});

it('GPUPointToCell H3 agrees with h3-js on f32 inputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const points = createPoints(9);
  // H3_INDEX_WGSL: exact through resolution 4, about 1e-3 at 9, 1.8% at 12; mismatches are grid
  // neighbors. Bounds here are loose; the h3 forward specs measure exhaustively.
  const bounds: Record<number, number> = {0: 0, 3: 0.001, 4: 0.001, 9: 0.01, 12: 0.06};
  const rates: Record<number, number> = {};
  for (const [resolutionKey, bound] of Object.entries(bounds)) {
    const resolution = Number(resolutionKey);
    const indexer = createIndexer(device, 'h3', resolution, {validity: true});
    const {keys, valid} = await indexer.run(points);
    let mismatches = 0;
    let finiteRows = 0;
    for (let row = 0; row < ROWS; row++) {
      if (!isFinitePoint(points, row)) {
        expect(keys[row]).toBe(0n);
        expect(valid![row]).toBe(0);
        continue;
      }
      const longitude = points[2 * row];
      const latitude = points[2 * row + 1];
      // h3-js rejects latitudes outside [-90, 90] and wraps longitude.
      if (Math.abs(latitude) > 90) {
        continue;
      }
      finiteRows++;
      const expected = h3ToBigInt(latLngToCell(latitude, longitude, resolution));
      if (keys[row] !== expected) {
        mismatches++;
        // A mismatch is a neighboring cell: centers within a few cell widths.
        const [expectedLat, expectedLng] = cellToLatLng(expected.toString(16));
        const [actualLat, actualLng] = cellToLatLng(keys[row].toString(16));
        const distance = getAngularDistance([expectedLng, expectedLat], [actualLng, actualLat]);
        expect(distance).toBeLessThan(Math.max(0.5, 5 * 2 ** -resolution));
      } else {
        expect(valid![row]).toBe(1);
      }
    }
    rates[resolution] = mismatches / finiteRows;
    expect(rates[resolution], `h3 resolution ${resolution}`).toBeLessThanOrEqual(bound);
    indexer.destroy();
  }
  // eslint-disable-next-line no-console
  console.log('H3 f32 GPU vs h3-js mismatch rates', rates);
});

it('GPUPointToCell zeroes masked rows and follows per-frame positions without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const oracles: Record<string, (lng: number, lat: number) => bigint> = {
    quadbin: (lng, lat) => quadbinPointToCell(lng, lat, 9),
    quadkey: (lng, lat) => quadkeyPointToCell(lng, lat, 9),
    geohash: (lng, lat) => geohashPointToCell(lng, lat, 5),
    s2: (lng, lat) => s2PointToCell(lng, lat, 6)
  };
  const resolutions: Record<string, number> = {quadbin: 9, quadkey: 9, geohash: 5, s2: 6};
  for (const family of ['quadbin', 'quadkey', 'geohash', 's2'] as const) {
    const indexer = createIndexer(device, family, resolutions[family], {
      mask: true,
      validity: true
    });
    const compiled = indexer.compiled;
    const random = createRandom(31);
    const seenKeys = new Set<bigint>();
    for (const seed of [1, 2, 3, 1]) {
      const points = createPoints(seed);
      const mask = Uint32Array.from({length: ROWS}, () => (random() < 0.3 ? 0 : 1));
      const {keys, valid} = await indexer.run(points, mask);
      for (let row = 0; row < ROWS; row += 7) {
        if (mask[row] === 0 || !isFinitePoint(points, row)) {
          expect(keys[row]).toBe(0n);
          expect(valid![row]).toBe(0);
        } else if (family !== 's2' || row % 3 !== 0) {
          // S2 rows near a seam may differ in f32; the exact comparison is in the S2 test above.
          expect(keys[row], `${family} seed ${seed} row ${row}`).toBe(
            oracles[family](points[2 * row], points[2 * row + 1])
          );
        }
      }
      seenKeys.add(keys[100]);
      expect(indexer.compiled).toBe(compiled);
    }
    expect(seenKeys.size).toBeGreaterThan(1);
    indexer.destroy();
  }
});

/** h3-py point-to-cell references with boundary margins (test/oracles/cells-h3.py). */
const H3_INDEXING_REFERENCE = decodeOracleFixture<{
  versions: Record<string, string>;
  tolerances: {marginBound: number};
  resolutions: number[];
  points: [number, number][];
  cells: Record<string, string[]>;
  margins: Record<string, number[]>;
  neighbors: Record<string, Record<string, string[]>>;
  edgePolicy: {points: [number, number][]};
}>(h3FixtureJson);

it('GPUPointToCell H3 equals h3-py away from cell edges', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Oracle: test/oracles/cells-h3.py (h3-py latlng_to_cell on the float32 inputs). A point farther
  // than marginBound (2e-7 rad, about 1.3 m: twice the largest measured float32 mismatch margin,
  // 9.9e-8 rad) from every edge of its cell must get exactly the h3 cell; a nearer point may get a
  // grid neighbor. Non-finite rows follow the edge policy: cell 0, invalid.
  const {points, cells, margins, neighbors, resolutions, tolerances, edgePolicy} =
    H3_INDEXING_REFERENCE;
  const rows = points.length + edgePolicy.points.length;
  expect(rows).toBeLessThanOrEqual(ROWS);
  const input = new Float32Array(2 * ROWS).fill(NaN);
  [...points, ...edgePolicy.points].forEach(([longitude, latitude], row) => {
    input[2 * row] = longitude;
    input[2 * row + 1] = latitude;
  });
  const counts: Record<number, string> = {};
  let largestMargin = 0;
  for (const resolution of resolutions) {
    const indexer = createIndexer(device, 'h3', resolution, {validity: true});
    const {keys, valid} = await indexer.run(input);
    indexer.destroy();
    let nearEdge = 0;
    for (const [row, cell] of cells[resolution].entries()) {
      expect(valid![row], `h3 ${resolution} row ${row} valid`).toBe(1);
      const actual = keys[row].toString(16);
      if (actual === cell) {
        continue;
      }
      const margin = margins[resolution][row];
      expect(
        margin,
        `h3 ${resolution} row ${row} (${points[row]}) GPU ${actual} != h3 ${cell}, margin ${margin}`
      ).toBeLessThan(tolerances.marginBound);
      expect(neighbors[resolution][row], `h3 ${resolution} row ${row} neighbor`).toContain(actual);
      nearEdge++;
      largestMargin = Math.max(largestMargin, margin);
    }
    for (let row = points.length; row < rows; row++) {
      expect(keys[row], `h3 ${resolution} non-finite row ${row}`).toBe(0n);
      expect(valid![row]).toBe(0);
    }
    counts[resolution] = `${nearEdge}/${Object.keys(neighbors[resolution]).length}`;
  }
  // eslint-disable-next-line no-console
  console.log(
    'H3 GPU vs h3-py: near-edge mismatches / points within the margin bound',
    counts,
    'largest mismatch margin (rad)',
    largestMargin
  );
});

it('GPUPointToCell packed and three uneven batches produce identical row-aligned outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = createPoints(89);
  const mask = Uint32Array.from({length: ROWS}, (_, row) => (row % 13 === 0 ? 0 : 1));
  const lengths = [3, 257, ROWS - 260];
  for (const family of ['quadbin', 'quadkey', 'geohash', 's2', 'h3'] as const) {
    const resolution = family === 'h3' ? 4 : 8;
    const packed = createIndexer(device, family, resolution, {mask: true, validity: true});
    const buffers: Buffer[] = [];
    const graph = new GPUCommandGraph(device);
    const cellBuffers: Buffer[] = [];
    const validityBuffers: Buffer[] = [];
    let rowOffset = 0;
    const positionChunks = [];
    const maskChunks = [];
    const cellChunks = [];
    const validityChunks = [];
    for (const [chunkIndex, length] of lengths.entries()) {
      const positionBuffer = createInputBuffer(
        device,
        positions.slice(rowOffset * 2, (rowOffset + length) * 2)
      );
      const maskBuffer = createInputBuffer(device, mask.slice(rowOffset, rowOffset + length));
      const cellBuffer = createOutputBuffer(device, length * 2);
      const validityBuffer = createOutputBuffer(device, length);
      buffers.push(positionBuffer, maskBuffer, cellBuffer, validityBuffer);
      cellBuffers.push(cellBuffer);
      validityBuffers.push(validityBuffer);
      positionChunks.push(
        importGraphBuffer(graph, `positions-${chunkIndex}`, positionBuffer, 'float32x2', length)
      );
      maskChunks.push(importGraphBuffer(graph, `mask-${chunkIndex}`, maskBuffer, 'uint32', length));
      cellChunks.push(
        importGraphBuffer(graph, `cells-${chunkIndex}`, cellBuffer, 'uint32x2', length)
      );
      validityChunks.push(
        importGraphBuffer(graph, `validity-${chunkIndex}`, validityBuffer, 'uint32', length)
      );
      rowOffset += length;
    }
    graph.add(
      new GPUPointToCell({
        family,
        resolution,
        positions: createVectorView('positions', 'float32x2', positionChunks),
        mask: createVectorView('mask', 'uint32', maskChunks),
        output: {
          cells: createVectorView('cells', 'uint32x2', cellChunks),
          validity: createVectorView('validity', 'uint32', validityChunks)
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const expected = await packed.run(positions, mask);
      const keys: bigint[] = [];
      const validity: number[] = [];
      for (const [chunkIndex, length] of lengths.entries()) {
        const words = await readUint32(cellBuffers[chunkIndex], length * 2);
        for (let row = 0; row < length; row++) {
          keys.push(joinKey(words[row * 2], words[row * 2 + 1]));
        }
        validity.push(...(await readUint32(validityBuffers[chunkIndex], length)));
      }
      expect(keys).toEqual(expected.keys);
      expect(validity).toEqual(expected.valid);
      expectPointCellOracle(family, resolution, positions, mask, keys, validity);
    } finally {
      compiled.destroy();
      packed.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  }
});

it('GPUPointToCell accepts more than 2^24 / 16 rows in three record batches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lengths = [17, 262147, 786429];
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const positionChunks = [];
  const cellChunks = [];
  const outputBuffers: Buffer[] = [];
  for (const [chunkIndex, length] of lengths.entries()) {
    const positions = createInputBuffer(device, new Float32Array(length * 2));
    const cells = createOutputBuffer(device, length * 2);
    buffers.push(positions, cells);
    outputBuffers.push(cells);
    positionChunks.push(
      importGraphBuffer(graph, `positions-${chunkIndex}`, positions, 'float32x2', length)
    );
    cellChunks.push(importGraphBuffer(graph, `cells-${chunkIndex}`, cells, 'uint32x2', length));
  }
  graph.add(
    new GPUPointToCell({
      family: 'quadbin',
      resolution: 8,
      positions: createVectorView('positions', 'float32x2', positionChunks),
      output: {cells: createVectorView('cells', 'uint32x2', cellChunks)}
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const key = quadbinPointToCell(0, 0, 8);
    const low = Number(key & 0xffffffffn);
    const high = Number(key >> 32n);
    expect(lengths.reduce((total, length) => total + length, 0)).toBeGreaterThan(2 ** 24 / 16);
    for (const [chunkIndex, length] of lengths.entries()) {
      const words = await readUint32(outputBuffers[chunkIndex], length * 2);
      expect(words.every((word, index) => word === (index % 2 === 0 ? low : high))).toBe(true);
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
});

it('GPUPointToCell preserves offset boundary batches, signed zeros and repeated encodings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lengths = [0, 1, 255, 256, 257, 0];
  const rows = lengths.reduce((total, length) => total + length, 0);
  const positions = Float32Array.from({length: rows * 2}, (_, component) =>
    component % 2 ? (Math.floor(component / 2) % 150) - 75 : (Math.floor(component / 2) % 350) - 175
  );
  positions[4] = -0;
  positions[5] = 0;
  positions[6] = 0;
  positions[7] = -0;
  const mask = Uint32Array.from({length: rows}, (_, row) => (row % 11 ? 1 : 0));
  for (const family of ['quadbin', 'h3'] as const) {
    const resolution = family === 'quadbin' ? 26 : 12;
    const createRun = (batchLengths: number[], packed: boolean) => {
      const graph = new GPUCommandGraph(device);
      const buffers: Buffer[] = [];
      const positionBuffers: Buffer[] = [];
      const maskBuffers: Buffer[] = [];
      const cellBuffers: Buffer[] = [];
      const validityBuffers: Buffer[] = [];
      const positionChunks = [];
      const maskChunks = [];
      const cellChunks = [];
      const validityChunks = [];
      for (const [chunkIndex, length] of batchLengths.entries()) {
        const positionBuffer = createInputBuffer(
          device,
          new Float32Array((Math.max(length, 1) + 3) * 2)
        );
        const maskBuffer = createInputBuffer(device, new Uint32Array(Math.max(length, 1) + 3));
        const cellBuffer = createOutputBuffer(device, (Math.max(length, 1) + 3) * 2);
        const validityBuffer = createOutputBuffer(device, Math.max(length, 1) + 3);
        cellBuffer.write(new Uint32Array((Math.max(length, 1) + 3) * 2).fill(0xfeedbeef));
        validityBuffer.write(new Uint32Array(Math.max(length, 1) + 3).fill(0xfeedbeef));
        buffers.push(positionBuffer, maskBuffer, cellBuffer, validityBuffer);
        positionBuffers.push(positionBuffer);
        maskBuffers.push(maskBuffer);
        cellBuffers.push(cellBuffer);
        validityBuffers.push(validityBuffer);
        const positionBacking = importGraphBuffer(
          graph,
          `positions-${chunkIndex}`,
          positionBuffer,
          'float32x2'
        );
        const maskBacking = importGraphBuffer(graph, `mask-${chunkIndex}`, maskBuffer, 'uint32');
        const cellBacking = importGraphBuffer(graph, `cells-${chunkIndex}`, cellBuffer, 'uint32x2');
        const validityBacking = importGraphBuffer(
          graph,
          `validity-${chunkIndex}`,
          validityBuffer,
          'uint32'
        );
        positionChunks.push(
          graph.createDataView(positionBacking.buffer, {
            format: 'float32x2',
            length,
            byteOffset: 24
          })
        );
        maskChunks.push(
          graph.createDataView(maskBacking.buffer, {format: 'uint32', length, byteOffset: 12})
        );
        cellChunks.push(
          graph.createDataView(cellBacking.buffer, {format: 'uint32x2', length, byteOffset: 24})
        );
        validityChunks.push(
          graph.createDataView(validityBacking.buffer, {format: 'uint32', length, byteOffset: 12})
        );
      }
      graph.add(
        new GPUPointToCell({
          family,
          resolution,
          precision: family === 'h3' ? 'double-single' : undefined,
          positions: packed
            ? positionChunks[0]
            : createVectorView('positions', 'float32x2', positionChunks),
          mask: packed ? maskChunks[0] : createVectorView('mask', 'uint32', maskChunks),
          output: {
            cells: packed ? cellChunks[0] : createVectorView('cells', 'uint32x2', cellChunks),
            validity: packed
              ? validityChunks[0]
              : createVectorView('validity', 'uint32', validityChunks)
          }
        })
      );
      const compiled = graph.compile();
      return {
        async run(currentPositions: Float32Array, currentMask: Uint32Array) {
          let firstRow = 0;
          for (const [chunkIndex, length] of batchLengths.entries()) {
            if (length) {
              positionBuffers[chunkIndex].write(
                currentPositions.subarray(firstRow * 2, (firstRow + length) * 2),
                24
              );
              maskBuffers[chunkIndex].write(currentMask.subarray(firstRow, firstRow + length), 12);
            }
            firstRow += length;
          }
          submitGraph(device, compiled, undefined);
          const keys: bigint[] = [];
          const validity: number[] = [];
          for (const [chunkIndex, length] of batchLengths.entries()) {
            const words = await readUint32(cellBuffers[chunkIndex], (Math.max(length, 1) + 3) * 2);
            const flags = await readUint32(validityBuffers[chunkIndex], Math.max(length, 1) + 3);
            expect(words.slice(0, 6)).toEqual(new Array<number>(6).fill(0xfeedbeef));
            expect(flags.slice(0, 3)).toEqual(new Array<number>(3).fill(0xfeedbeef));
            for (let row = 0; row < length; row++) {
              keys.push(joinKey(words[6 + row * 2], words[7 + row * 2]));
              validity.push(flags[3 + row]);
            }
            if (!length) {
              expect(words.every(word => word === 0xfeedbeef)).toBe(true);
              expect(flags.every(word => word === 0xfeedbeef)).toBe(true);
            }
          }
          return {keys, validity};
        },
        destroy() {
          compiled.destroy();
          for (const buffer of buffers) buffer.destroy();
        }
      };
    };
    const packed = createRun([rows], true);
    const chunked = createRun(lengths, false);
    try {
      for (const frame of [0, 1, 2]) {
        const currentMask = frame === 1 ? new Uint32Array(rows) : mask;
        const currentPositions =
          frame === 2 ? Float32Array.from(positions, coordinate => -coordinate) : positions;
        const actual = await chunked.run(currentPositions, currentMask);
        expect(actual).toEqual(await packed.run(currentPositions, currentMask));
        expectPointCellOracle(
          family,
          resolution,
          currentPositions,
          currentMask,
          actual.keys,
          actual.validity,
          true
        );
      }
    } finally {
      packed.destroy();
      chunked.destroy();
    }
  }
});

/** Independent references: integer/bisection grids, h3-js, and the documented S2 edge tolerance. */
function expectPointCellOracle(
  family: GPUCellIndexFamily,
  resolution: number,
  positions: Float32Array,
  mask: Uint32Array,
  keys: readonly bigint[],
  validity: readonly number[],
  exactH3 = false
): void {
  let s2Mismatches = 0;
  let h3Mismatches = 0;
  for (let row = 0; row < keys.length; row++) {
    if (!mask[row] || !isFinitePoint(positions, row)) {
      expect(keys[row]).toBe(0n);
      expect(validity[row]).toBe(0);
      continue;
    }
    expect(validity[row]).toBe(1);
    const longitude = positions[row * 2];
    const latitude = positions[row * 2 + 1];
    let expected: bigint;
    switch (family) {
      case 'quadbin':
        expected = quadbinPointToCell(longitude, latitude, resolution);
        break;
      case 'quadkey':
        expected = quadkeyPointToCell(longitude, latitude, resolution);
        break;
      case 'geohash':
        expected = geohashPointToCell(longitude, latitude, resolution);
        break;
      case 'h3':
        expected = h3ToBigInt(latLngToCell(latitude, longitude, resolution));
        break;
      case 's2':
        expected = s2PointToCell(longitude, latitude, resolution);
        break;
    }
    if (family === 'h3' && !exactH3 && keys[row] !== expected) {
      h3Mismatches++;
      const [expectedLatitude, expectedLongitude] = cellToLatLng(expected.toString(16));
      const [actualLatitude, actualLongitude] = cellToLatLng(keys[row].toString(16));
      expect(
        getAngularDistance([expectedLongitude, expectedLatitude], [actualLongitude, actualLatitude])
      ).toBeLessThan(Math.max(0.5, 5 * 2 ** -resolution));
    } else if (family === 's2' && keys[row] !== expected) {
      // A float32 S2 face or edge tie may choose an adjacent reference cell. Count random rows
      // separately from the deliberate seams/poles and retain the existing low-level rate bound.
      if (row < UNIFORM_COUNT) s2Mismatches++;
      expect(getAngularDistance(s2CellToCenter(keys[row]), s2CellToCenter(expected))).toBeLessThan(
        2.5 * (Math.PI / 2 / 2 ** resolution)
      );
    } else {
      expect(keys[row]).toBe(expected);
    }
  }
  expect(s2Mismatches).toBeLessThanOrEqual(Math.ceil(keys.length * 5e-4));
  expect(h3Mismatches).toBeLessThanOrEqual(Math.ceil(keys.length * 1e-3));
}

it('GPUPointToCell accepts adjacent output chunks and different binding prefixes with CPU expected keys', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const positions = Float32Array.from([10, 20, -100, 15, 130, -30]);
  const positionChunks = [0, 8, 248].map((byteOffset, index) => {
    const data = new Float32Array(byteOffset / 4 + 2).fill(77);
    data.set(positions.subarray(index * 2, index * 2 + 2), byteOffset / 4);
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    const backing = importGraphBuffer(graph, `varying-positions-${index}`, buffer, 'float32x2');
    return graph.createDataView(backing.buffer, {format: 'float32x2', length: 1, byteOffset});
  });
  const maskBuffers: Buffer[] = [];
  const maskOffsets = [0, 4, 252];
  const maskChunks = maskOffsets.map((byteOffset, index) => {
    const buffer = createInputBuffer(device, new Uint32Array(byteOffset / 4 + 1).fill(1));
    buffers.push(buffer);
    maskBuffers.push(buffer);
    const backing = importGraphBuffer(graph, `varying-mask-${index}`, buffer, 'uint32');
    return graph.createDataView(backing.buffer, {format: 'uint32', length: 1, byteOffset});
  });
  const cells = createOutputBuffer(device, 6);
  const validity = createOutputBuffer(device, 3);
  buffers.push(cells, validity);
  const cellBacking = importGraphBuffer(graph, 'adjacent-output-cells', cells, 'uint32x2');
  const validityBacking = importGraphBuffer(graph, 'adjacent-output-validity', validity, 'uint32');
  graph.add(
    new GPUPointToCell({
      family: 'quadbin',
      resolution: 8,
      positions: createVectorView('varying-position-chunks', 'float32x2', positionChunks),
      mask: createVectorView('varying-mask-chunks', 'uint32', maskChunks),
      output: {
        cells: createVectorView(
          'adjacent-cell-chunks',
          'uint32x2',
          [0, 8, 16].map(byteOffset =>
            graph.createDataView(cellBacking.buffer, {format: 'uint32x2', length: 1, byteOffset})
          )
        ),
        validity: createVectorView(
          'adjacent-validity-chunks',
          'uint32',
          [0, 4, 8].map(byteOffset =>
            graph.createDataView(validityBacking.buffer, {format: 'uint32', length: 1, byteOffset})
          )
        )
      }
    })
  );
  const compiled = graph.compile();
  try {
    for (const mask of [
      new Uint32Array([1, 1, 1]),
      new Uint32Array([1, 0, 1]),
      new Uint32Array([0, 1, 0])
    ]) {
      maskBuffers.forEach((buffer, index) =>
        buffer.write(mask.subarray(index, index + 1), maskOffsets[index])
      );
      submitGraph(device, compiled, undefined);
      const words = await readUint32(cells, 6);
      expectPointCellOracle(
        'quadbin',
        8,
        positions,
        mask,
        [0, 1, 2].map(row => joinKey(words[row * 2], words[row * 2 + 1])),
        await readUint32(validity, 3)
      );
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
