// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_CELL_GRID_DISTANCE_UNDEFINED,
  GPUCellGridPath
} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-cell-grid-path';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  h3ToBigInt,
  joinCellKey,
  splitCellKey,
  quadbinTileToCell
} from '../cell-aggregation/cell-aggregation-oracle';
import {H3_GRID_PATH_FIXTURES} from './h3-grid-path-fixtures';
import {getS2TopologyCell, getS2TopologyRowOnCPU} from './s2-topology-oracle';
import type {GPUCellFamily} from '../../../src/gpu-spatial-analysis/cell-indexing/cell-index-families';
import {requireTestDevice} from '../../utils/require-test-device';

/** Pairs H3 rejects or the kernel must reject: resolution mismatch, zero cell, masked row. */
const EXTRA_PAIRS: [string, string, 'undefined' | 'masked'][] = [
  ['85283473fffffff', '8528342bfffffff', 'masked'],
  ['85283473fffffff', '8a283082a677fff', 'undefined'],
  ['0', '85283473fffffff', 'undefined'],
  ['85283473fffffff', 'ffffffffffffffff', 'undefined']
];

type Run = {
  distances: number[];
  cells: number[];
  counts: number[];
};

/** Builds and runs one graph over `pairs`, with `maximumPathLength = 0` meaning distance only. */
async function runGridPath(
  device: Device,
  pairs: {origin: bigint; destination: bigint; isMasked?: boolean}[],
  maximumPathLength: number,
  wordOrder: 'little-endian' | 'high-low' = 'little-endian',
  family: GPUCellFamily = 'h3',
  maximumSearchRadius?: number,
  packedSubviews = false
): Promise<Run> {
  const rows = pairs.length;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    if (!buffers.includes(buffer)) buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'cell-grid-path-graph'});
  const words = (select: 'origin' | 'destination') => {
    const array = new Uint32Array(2 * rows);
    pairs.forEach((pair, row) => {
      const [low, high] = splitCellKey(pair[select]);
      array.set(wordOrder === 'high-low' ? [high, low] : [low, high], 2 * row);
    });
    return array;
  };
  const input = (values: Uint32Array) => {
    if (!packedSubviews) return createInputBuffer(device, values);
    const prefixed = new Uint32Array(values.length + 1);
    prefixed[0] = 0xdeadbeef;
    prefixed.set(values, 1);
    return device.createBuffer({data: prefixed, usage: Buffer.STORAGE});
  };
  const originsBuffer = track(input(words('origin')));
  const destinationsBuffer = track(input(words('destination')));
  const maskBuffer = track(input(Uint32Array.from(pairs, pair => (pair.isMasked ? 0 : 1))));
  const cellWordLength = 2 * rows * Math.max(maximumPathLength, 1);
  const distancesByteOffset = packedSubviews ? 4 : 0;
  const cellsByteOffset = packedSubviews ? Math.ceil((4 + rows * 4) / 256) * 256 + 4 : 0;
  const countsByteOffset = packedSubviews
    ? Math.ceil((cellsByteOffset + cellWordLength * 4) / 256) * 256 + 4
    : 0;
  const packed = packedSubviews
    ? track(
        device.createBuffer({
          byteLength: countsByteOffset + rows * 4,
          usage: Buffer.STORAGE | Buffer.COPY_SRC
        })
      )
    : undefined;
  const packedHandle = packed
    ? importGraphBuffer(graph, 'packed-output', packed, 'uint32').buffer
    : undefined;
  const distancesBuffer = track(packed ?? createOutputBuffer(device, rows));
  const cellsBuffer = track(packed ?? createOutputBuffer(device, cellWordLength));
  const countsBuffer = track(packed ?? createOutputBuffer(device, rows));
  const inputView = <Format extends 'uint32' | 'uint32x2'>(
    name: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => {
    const view = importGraphBuffer(graph, name, buffer, format, length);
    return packedSubviews
      ? graph.createDataView(view.buffer, {format, length, byteOffset: 4})
      : view;
  };
  const outputView = <Format extends 'uint32' | 'uint32x2'>(
    name: string,
    buffer: Buffer,
    format: Format,
    length: number,
    byteOffset: number
  ) =>
    packedHandle
      ? graph.createDataView(packedHandle, {format, length, byteOffset})
      : importGraphBuffer(graph, name, buffer, format, length);
  graph.add(
    new GPUCellGridPath({
      family,
      maximumSearchRadius,
      origins: inputView('origins', originsBuffer, 'uint32x2', rows),
      destinations: inputView('destinations', destinationsBuffer, 'uint32x2', rows),
      wordOrder,
      mask: inputView('mask', maskBuffer, 'uint32', rows),
      maximumPathLength: maximumPathLength || undefined,
      output: {
        distances: outputView('distances', distancesBuffer, 'uint32', rows, distancesByteOffset),
        cells: maximumPathLength
          ? outputView('cells', cellsBuffer, 'uint32x2', rows * maximumPathLength, cellsByteOffset)
          : undefined,
        counts: maximumPathLength
          ? outputView('counts', countsBuffer, 'uint32', rows, countsByteOffset)
          : undefined
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const read = async (buffer: Buffer, length: number, byteOffset: number) =>
    packedSubviews
      ? (await readUint32(buffer)).slice(byteOffset / 4, byteOffset / 4 + length)
      : readUint32(buffer, length);
  const result = {
    distances: await read(distancesBuffer, rows, distancesByteOffset),
    cells: await read(cellsBuffer, cellWordLength, cellsByteOffset),
    counts: await read(countsBuffer, rows, countsByteOffset)
  };
  compiled.destroy();
  buffers.forEach(buffer => buffer.destroy());
  return result;
}

function getPathRow(run: Run, row: number, stride: number, wordOrder = 'little-endian'): string[] {
  const path: string[] = [];
  for (let slot = 0; slot < stride; slot++) {
    const index = 2 * (row * stride + slot);
    const low = wordOrder === 'high-low' ? run.cells[index + 1] : run.cells[index];
    const high = wordOrder === 'high-low' ? run.cells[index] : run.cells[index + 1];
    const cell = joinCellKey(low, high);
    path.push(cell === 0n ? '' : cell.toString(16));
  }
  return path;
}

it('GPUCellGridPath distances and paths match python h3 4.5.0 gridDistance and gridPathCells', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const stride = 64;
  const pairs = [
    ...H3_GRID_PATH_FIXTURES.map(([origin, destination]) => ({
      origin: h3ToBigInt(origin),
      destination: h3ToBigInt(destination)
    })),
    ...EXTRA_PAIRS.map(([origin, destination, kind]) => ({
      origin: h3ToBigInt(origin),
      destination: h3ToBigInt(destination),
      isMasked: kind === 'masked'
    }))
  ];
  const startTime = performance.now();
  const run = await runGridPath(device, pairs, stride);
  const elapsed = performance.now() - startTime;

  let sameCount = 0;
  let tieCount = 0;
  let failedCount = 0;
  H3_GRID_PATH_FIXTURES.forEach(([origin, destination, distance, relation, expectedPath], row) => {
    const label = `${origin} -> ${destination}`;
    expect(run.distances[row], `${label} distance`).toBe(
      distance < 0 ? GPU_CELL_GRID_DISTANCE_UNDEFINED : distance
    );
    if (distance < 0) {
      failedCount++;
      expect(run.counts[row], `${label} count`).toBe(0);
      expect(
        getPathRow(run, row, stride).every(cell => cell === ''),
        `${label} empty`
      ).toBe(true);
      return;
    }
    expect(run.counts[row], `${label} count`).toBe(distance + 1);
    if (!expectedPath) {
      // Longer than the pinned paths: the first `stride` cells start at the origin.
      expect(getPathRow(run, row, stride)[0], `${label} origin`).toBe(origin);
      return;
    }
    const actualPath = getPathRow(run, row, stride).slice(0, expectedPath.length);
    if (relation === 'same') {
      sameCount++;
      expect(actualPath, `${label} path`).toEqual(expectedPath);
    } else {
      tieCount++;
      expect(actualPath[0]).toBe(expectedPath[0]);
      expect(actualPath.at(-1)).toBe(expectedPath.at(-1));
      const differing = actualPath.filter((cell, slot) => cell !== expectedPath[slot]).length;
      expect(differing, `${label} cells differing at a cube-rounding tie`).toBeLessThanOrEqual(2);
    }
    expect(
      getPathRow(run, row, stride)
        .slice(expectedPath.length)
        .every(cell => cell === '')
    ).toBe(true);
  });
  EXTRA_PAIRS.forEach(([, , kind], extra) => {
    const row = H3_GRID_PATH_FIXTURES.length + extra;
    expect(run.distances[row], `extra ${kind} ${extra} distance`).toBe(
      GPU_CELL_GRID_DISTANCE_UNDEFINED
    );
    expect(run.counts[row]).toBe(0);
  });
  console.log(
    `GPUCellGridPath ${pairs.length} pairs in ${elapsed.toFixed(0)} ms: ${sameCount} identical paths, ${tieCount} tie deviations, ${failedCount} undefined`
  );
  expect(sameCount).toBeGreaterThan(200);
}, 120_000);

it('GPUCellGridPath distance-only mode, overflow counts and high-low word order', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const pairs = H3_GRID_PATH_FIXTURES.map(([origin, destination]) => ({
    origin: h3ToBigInt(origin),
    destination: h3ToBigInt(destination)
  }));
  const distanceOnly = await runGridPath(device, pairs, 0);
  H3_GRID_PATH_FIXTURES.forEach(([origin, destination, distance], row) => {
    expect(distanceOnly.distances[row], `${origin} -> ${destination}`).toBe(
      distance < 0 ? GPU_CELL_GRID_DISTANCE_UNDEFINED : distance
    );
  });

  // A short stride truncates every pinned path to its first cells and reports the true length.
  const shortStride = 8;
  const truncated = await runGridPath(device, pairs, shortStride, 'high-low');
  let overflowRows = 0;
  H3_GRID_PATH_FIXTURES.forEach(([origin, destination, distance, relation, expectedPath], row) => {
    if (distance < 0) {
      return;
    }
    expect(truncated.counts[row]).toBe(distance + 1);
    if (distance + 1 > shortStride) {
      overflowRows++;
    }
    if (relation === 'same' && expectedPath) {
      expect(
        getPathRow(truncated, row, shortStride, 'high-low'),
        `${origin} -> ${destination}`
      ).toEqual(Array.from({length: shortStride}, (_, slot) => expectedPath[slot] ?? ''));
    }
  });
  expect(overflowRows).toBeGreaterThan(50);
}, 120_000);

it('GPUCellGridPath validates its properties', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'cell-grid-path-validation'});
  const buffer = createInputBuffer(device, new Uint32Array(4));
  const outBuffer = createOutputBuffer(device, 2);
  const origins = importGraphBuffer(graph, 'origins', buffer, 'uint32x2', 2);
  const distances = importGraphBuffer(graph, 'distances', outBuffer, 'uint32', 2);
  const base = {family: 'h3' as const, origins, destinations: origins, output: {distances}};
  expect(() => new GPUCellGridPath({...base, output: {}})).toThrow(
    /output needs distances or cells/
  );
  expect(
    () =>
      new GPUCellGridPath({
        ...base,
        output: {cells: importGraphBuffer(graph, 'cells', buffer, 'uint32x2', 2)}
      })
  ).toThrow(/maximumPathLength/);
  expect(() => new GPUCellGridPath({...base, family: 'bad' as unknown as 'h3'})).toThrow(/family/);
  buffer.destroy();
  outBuffer.destroy();
});

it('Quadbin distances use wrapped Chebyshev steps, including level 26, masks and truncated paths', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const level of [3, 26]) {
    const size = 2 ** level;
    const pairs = [
      {origin: quadbinTileToCell(0, 3, level), destination: quadbinTileToCell(size - 1, 6, level)},
      {origin: quadbinTileToCell(2, 2, level), destination: quadbinTileToCell(2, 2, level)},
      {origin: 0n, destination: quadbinTileToCell(0, 0, level)},
      {
        origin: quadbinTileToCell(0, 3, level),
        destination: quadbinTileToCell(0, 3, level),
        isMasked: true
      }
    ];
    for (const wordOrder of ['little-endian', 'high-low'] as const) {
      const run = await runGridPath(device, pairs, 3, wordOrder, 'quadbin');
      expect(run.distances).toEqual([
        3,
        0,
        GPU_CELL_GRID_DISTANCE_UNDEFINED,
        GPU_CELL_GRID_DISTANCE_UNDEFINED
      ]);
      expect(run.counts).toEqual([4, 1, 0, 0]);
      expect(getPathRow(run, 0, 3, wordOrder)).toEqual(
        [
          quadbinTileToCell(0, 3, level),
          quadbinTileToCell(size - 1, 4, level),
          quadbinTileToCell(size - 1, 5, level)
        ].map(cell => cell.toString(16))
      );
      expect(getPathRow(run, 1, 3, wordOrder)).toEqual([pairs[1].origin.toString(16), '', '']);
    }
  }
  device.destroy();
});

it('S2 distance unfolds level-30 opposite and adjacent faces without uint32 overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const size = 2 ** 30;
  const cell = (face: number, column: number, row: number) =>
    getS2TopologyCell({face, ij: [column, row], level: 30});
  const pairs = [
    // Opposite face centers with matching physical tangent coordinates: one full face plus
    // the two equal distances to its nearest edge and two seam crossings.
    {origin: cell(0, size / 2, size / 2), destination: cell(3, size / 2 - 1, size / 2 - 1)},
    {origin: cell(0, 0, 0), destination: cell(3, size - 1, size - 1)},
    // Adjacent far corners require two complete boundary approaches and a seam-tangent span.
    {origin: cell(0, 0, 0), destination: cell(1, size - 1, size - 1)}
  ];
  const expected = [2 * size - 1, size + 1, 3 * size - 2];
  const result = await runGridPath(device, pairs, 2, 'high-low', 's2');
  expect(result.distances).toEqual(expected);
  expect(result.counts).toEqual(expected.map(distance => distance + 1));
  for (let row = 0; row < pairs.length; row++)
    expect(getPathRow(result, row, 2, 'high-low')[0]).toBe(pairs[row].origin.toString(16));
  const bounded = await runGridPath(device, pairs, 2, 'little-endian', 's2', 8);
  expect(bounded.distances).toEqual(pairs.map(() => GPU_CELL_GRID_DISTANCE_UNDEFINED));
  expect(bounded.counts).toEqual([0, 0, 0]);
  expect(bounded.cells.every(word => word === 0)).toBe(true);
  device.destroy();
});

it('S2 grid distances match independent graph BFS through cube corners and level 30', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const pairs: {origin: bigint; destination: bigint}[] = [];
  const expected: number[] = [];
  for (let face = 0; face < 6; face++) {
    const origin = getS2TopologyCell({face, ij: [0, 0], level: 2});
    const disk = getS2TopologyRowOnCPU({type: 'disk', k: 16}, origin);
    for (let index = 0; index < disk.cells.length; index++) {
      pairs.push({origin, destination: disk.cells[index]});
      expected.push(disk.distances[index]);
    }
  }
  const fine = getS2TopologyCell({face: 3, ij: [10000, 10000], level: 30});
  for (const [column, row] of [
    [10003, 10002],
    [10004, 10004],
    [10005, 10004]
  ]) {
    pairs.push({
      origin: fine,
      destination: getS2TopologyCell({face: 3, ij: [column, row], level: 30})
    });
    const distance = Math.abs(column - 10000) + Math.abs(row - 10000);
    expected.push(distance);
  }
  for (const stride of [3, 10]) {
    const run = await runGridPath(device, pairs, stride, 'little-endian', 's2');
    expect(run.distances).toEqual(expected);
    expect(run.counts).toEqual(
      expected.map(distance => (distance === GPU_CELL_GRID_DISTANCE_UNDEFINED ? 0 : distance + 1))
    );
    for (let row = 0; row < pairs.length; row++) {
      const stored = getPathRow(run, row, stride)
        .filter(Boolean)
        .map(value => BigInt(`0x${value}`));
      if (expected[row] === GPU_CELL_GRID_DISTANCE_UNDEFINED) {
        expect(stored).toEqual([]);
        continue;
      }
      expect(stored[0]).toBe(pairs[row].origin);
      if (stored.length === expected[row] + 1) expect(stored.at(-1)).toBe(pairs[row].destination);
      for (let step = 1; step < stored.length; step++)
        expect(getS2TopologyRowOnCPU({type: 'ring', k: 1}, stored[step - 1]).cells).toContain(
          stored[step]
        );
    }
  }
  device.destroy();
});

it('regular-grid paths preserve every key word, row boundary and zero tail in packed nonaligned layouts', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const family of ['quadbin', 's2'] as const) {
    const same =
      family === 's2'
        ? getS2TopologyCell({face: 5, ij: [0, 0], level: 0})
        : quadbinTileToCell(2, 2, 26);
    const stepCell = (step: number) =>
      family === 's2'
        ? getS2TopologyCell({face: 3, ij: [10000 + step, 10000], level: 30})
        : quadbinTileToCell(10000 + step, 10000, 26);
    const pairs = [
      {origin: same, destination: same},
      {origin: stepCell(0), destination: stepCell(3)},
      {origin: stepCell(0), destination: stepCell(3), isMasked: true},
      {origin: 0n, destination: stepCell(0)}
    ];
    for (const stride of [1, 3, 5])
      for (const wordOrder of ['little-endian', 'high-low'] as const) {
        const result = await runGridPath(device, pairs, stride, wordOrder, family, undefined, true);
        expect(result.distances).toEqual([0, 3, 0xffffffff, 0xffffffff]);
        expect(result.counts).toEqual([1, 4, 0, 0]);
        const expected = Array.from({length: pairs.length * stride}, (_, index) => {
          const row = Math.floor(index / stride);
          const slot = index % stride;
          if (row === 0) return slot === 0 ? same : 0n;
          if (row === 1 && slot <= 3) return stepCell(slot);
          return 0n;
        });
        const words = expected.flatMap(cell => {
          const [low, high] = splitCellKey(cell);
          return wordOrder === 'high-low' ? [high, low] : [low, high];
        });
        expect(result.cells).toEqual(words);
      }
  }
  device.destroy();
});
