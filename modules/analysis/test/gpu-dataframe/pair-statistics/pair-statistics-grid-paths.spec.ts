// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it, expect} from 'vitest';
import type {Buffer, Device} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createNullWebGPUDevice,
  createOutputBuffer,
  getParsedKernelSources,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GPUCrossVariogram} from '../../../src/dataframe-analysis/pair-statistics/gpu-cross-variogram';
import {GPUMarkedPointStatistics} from '../../../src/dataframe-analysis/pair-statistics/gpu-marked-point-statistics';

const ROWS = 2000;

function createRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return state / 2147483648;
  };
}

function createContext(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    data: number[]
  ) => {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data)
    );
    buffers.push(buffer);
    const length = format === 'float32x2' ? data.length / 2 : data.length;
    return importGraphBuffer(graph, `input-${serial++}`, buffer, format, length);
  };
  const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, `output-${serial++}`, buffer, format, length)};
  };
  let compiled: ReturnType<typeof graph.compile> | undefined;
  return {
    graph,
    input,
    output,
    run: () => {
      compiled = graph.compile();
      submitGraph(device, compiled, undefined);
    },
    destroy: () => {
      compiled?.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

/** Same NaN positions, and finite entries within a float32 summation-order tolerance. */
function expectClose(actual: number[], expected: number[]) {
  expect(actual.map(Number.isNaN)).toEqual(expected.map(Number.isNaN));
  actual.forEach((value, index) => {
    if (!Number.isNaN(value)) {
      const reference = expected[index];
      expect(Math.abs(value - reference)).toBeLessThanOrEqual(1e-4 * Math.abs(reference) + 1e-6);
    }
  });
}

function createData(seed: number, rows: number) {
  const random = createRandom(seed);
  const positions = Array.from({length: rows * 2}, () => random() * 100);
  const values = Array.from({length: rows}, (_, row) => Math.sin(row) * 3 + random());
  const secondValues = values.map(value => value * 0.5 + random());
  const times = Array.from({length: rows}, () => Math.floor(random() * 20));
  const mask = Array.from({length: rows}, () => (random() < 0.9 ? 1 : 0));
  // A few non-finite rows exercise complete-case exclusion.
  values[7] = Number.NaN;
  positions[2 * 11] = Number.POSITIVE_INFINITY;
  return {positions, values, secondValues, times, mask};
}

const CROSS_CASES = [
  {name: 'cross with mask', secondValues: true, times: false, mask: true, maximumDistance: 9},
  {name: 'univariate', secondValues: false, times: false, mask: false, maximumDistance: 14},
  {name: 'space-time', secondValues: true, times: true, mask: true, maximumDistance: 9},
  {name: 'invalid distance', secondValues: true, times: false, mask: false, maximumDistance: 0}
];

for (const testCase of CROSS_CASES)
  it(`cross-variogram grid path matches the dense path: ${testCase.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const data = createData(7, ROWS);
    const lagCount = 8;
    const temporalLagCount = testCase.times ? 3 : 1;
    const bins = lagCount * temporalLagCount;
    const results = [];
    for (const gridSize of [undefined, [16, 16] as const]) {
      const context = createContext(device);
      try {
        const semivariances = context.output('float32', bins);
        const pairCounts = context.output('uint32', bins);
        const status = context.output('uint32', bins);
        context.graph.add(
          new GPUCrossVariogram({
            positions: context.input('float32x2', data.positions),
            values: context.input('float32', data.values),
            secondValues: testCase.secondValues
              ? context.input('float32', data.secondValues)
              : undefined,
            times: testCase.times ? context.input('float32', data.times) : undefined,
            mask: testCase.mask ? context.input('uint32', data.mask) : undefined,
            parameters: context.input('float32', [testCase.maximumDistance, 12]),
            lagCount,
            temporalLagCount: testCase.times ? temporalLagCount : undefined,
            gridSize,
            output: {
              semivariances: semivariances.view,
              pairCounts: pairCounts.view,
              status: status.view
            }
          })
        );
        context.run();
        results.push({
          semivariances: await readFloat32(semivariances.buffer, bins),
          pairCounts: await readUint32(pairCounts.buffer, bins),
          status: await readUint32(status.buffer, bins)
        });
      } finally {
        context.destroy();
      }
    }
    const [dense, grid] = results;
    expect(grid.pairCounts).toEqual(dense.pairCounts);
    expect(grid.status).toEqual(dense.status);
    expectClose(grid.semivariances, dense.semivariances);
    if (testCase.maximumDistance > 0) {
      expect(dense.pairCounts.reduce((sum, count) => sum + count, 0)).toBeGreaterThan(1000);
    } else {
      expect(dense.status).toEqual(new Array(bins).fill(1));
    }
  });

it('cross-variogram grid path accepts more than 16,384 rows and counts every pair', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 20000;
  const random = createRandom(11);
  // Integer coordinates keep every pair distance far from the 3.5 cutoff in float32.
  const positions = Array.from({length: rows * 2}, () => Math.floor(random() * 400));
  const values = Array.from({length: rows}, (_, row) => row % 17);
  let expected = 0;
  for (let first = 0; first < rows; first++)
    for (let second = first + 1; second < rows; second++) {
      const dx = positions[2 * first] - positions[2 * second];
      const dy = positions[2 * first + 1] - positions[2 * second + 1];
      if (dx * dx + dy * dy <= 12.25) expected++;
    }
  const context = createContext(device);
  try {
    const pairCounts = context.output('uint32', 4);
    context.graph.add(
      new GPUCrossVariogram({
        positions: context.input('float32x2', positions),
        values: context.input('float32', values),
        parameters: context.input('float32', [3.5]),
        lagCount: 4,
        gridSize: [64, 64],
        output: {
          semivariances: context.output('float32', 4).view,
          pairCounts: pairCounts.view,
          status: context.output('uint32', 4).view
        }
      })
    );
    context.run();
    const counts = await readUint32(pairCounts.buffer, 4);
    expect(counts.reduce((sum, count) => sum + count, 0)).toBe(expected);
    expect(expected).toBeGreaterThan(1000);
  } finally {
    context.destroy();
  }
});

const MARKED_CASES = [
  {name: 'homogeneous translation', extras: false, edgeCorrection: 'translation' as const},
  {name: 'types marks intensities mask, uncorrected', extras: true, edgeCorrection: 'none' as const}
];

for (const testCase of MARKED_CASES)
  it(`marked point statistics grid path matches the dense path: ${testCase.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const data = createData(19, ROWS);
    const random = createRandom(23);
    const types = Array.from({length: ROWS}, () => Math.floor(random() * 3));
    const marks = Array.from({length: ROWS}, () => 1 + random());
    const intensities = Array.from({length: ROWS}, () => 0.15 + random() * 0.1);
    const radiusCount = 10;
    const results = [];
    for (const traversal of [
      {searchStructure: 'brute-force' as const},
      {},
      {gridSize: [12, 12] as const}
    ]) {
      const context = createContext(device);
      try {
        const statistics = context.output('float32', radiusCount * 4);
        const pairCounts = context.output('uint32', radiusCount);
        const status = context.output('uint32', radiusCount);
        context.graph.add(
          new GPUMarkedPointStatistics({
            positions: context.input('float32x2', data.positions),
            types: testCase.extras ? context.input('uint32', types) : undefined,
            typePair: testCase.extras ? [1, 2] : undefined,
            marks: testCase.extras ? context.input('float32', marks) : undefined,
            intensities: testCase.extras ? context.input('float32', intensities) : undefined,
            mask: testCase.extras ? context.input('uint32', data.mask) : undefined,
            parameters: context.input('float32', [5, 5, 95, 90, 8, 0.1]),
            radiusCount,
            edgeCorrection: testCase.edgeCorrection,
            ...traversal,
            output: {
              statistics: statistics.view,
              pairCounts: pairCounts.view,
              status: status.view
            }
          })
        );
        context.run();
        results.push({
          statistics: await readFloat32(statistics.buffer, radiusCount * 4),
          pairCounts: await readUint32(pairCounts.buffer, radiusCount),
          status: await readUint32(status.buffer, radiusCount)
        });
      } finally {
        context.destroy();
      }
    }
    const [dense, ...indexed] = results;
    for (const grid of indexed) {
      expect(grid.pairCounts).toEqual(dense.pairCounts);
      expect(grid.status).toEqual(dense.status);
      expectClose(grid.statistics, dense.statistics);
    }
    expect(dense.pairCounts[radiusCount - 1]).toBeGreaterThan(1000);
  });

/**
 * Integer lattice rows with spacing 1 in a 21 x 21 square: pair distances 1, 2, 3 and 4 fall
 * exactly on the radius-bin edges of a maximum distance of 4 with four bins, and pairs at that
 * maximum straddle the cell boundaries of a lattice whose cells are just over 4 wide.
 */
function createLatticeData() {
  const positions: number[] = [];
  const values: number[] = [];
  for (let y = 0; y <= 20; y++)
    for (let x = 0; x <= 20; x++) {
      positions.push(x, y);
      values.push(Math.sin(x * 0.9) + Math.cos(y * 0.5));
    }
  return {positions, values, rows: values.length};
}

async function runCrossVariogram(
  device: Device,
  data: {positions: number[]; values: number[]},
  maximumDistance: number,
  lagCount: number,
  gridSize?: readonly [number, number]
) {
  const context = createContext(device);
  try {
    const semivariances = context.output('float32', lagCount);
    const pairCounts = context.output('uint32', lagCount);
    const status = context.output('uint32', lagCount);
    context.graph.add(
      new GPUCrossVariogram({
        positions: context.input('float32x2', data.positions),
        values: context.input('float32', data.values),
        parameters: context.input('float32', [maximumDistance]),
        lagCount,
        gridSize,
        output: {
          semivariances: semivariances.view,
          pairCounts: pairCounts.view,
          status: status.view
        }
      })
    );
    context.run();
    return {
      semivariances: await readFloat32(semivariances.buffer, lagCount),
      pairCounts: await readUint32(pairCounts.buffer, lagCount),
      status: await readUint32(status.buffer, lagCount)
    };
  } finally {
    context.destroy();
  }
}

async function runMarkedPointStatistics(
  device: Device,
  data: {positions: number[]; values: number[]},
  window: [number, number, number, number],
  maximumRadius: number,
  radiusCount: number,
  gridSize?: readonly [number, number]
) {
  const context = createContext(device);
  try {
    const statistics = context.output('float32', radiusCount * 4);
    const pairCounts = context.output('uint32', radiusCount);
    const status = context.output('uint32', radiusCount);
    context.graph.add(
      new GPUMarkedPointStatistics({
        positions: context.input('float32x2', data.positions),
        marks: context.input(
          'float32',
          data.values.map(value => value + 3)
        ),
        parameters: context.input('float32', [...window, maximumRadius, 0]),
        radiusCount,
        gridSize,
        searchStructure: gridSize ? 'grid' : 'brute-force',
        output: {statistics: statistics.view, pairCounts: pairCounts.view, status: status.view}
      })
    );
    context.run();
    return {
      statistics: await readFloat32(statistics.buffer, radiusCount * 4),
      pairCounts: await readUint32(pairCounts.buffer, radiusCount),
      status: await readUint32(status.buffer, radiusCount)
    };
  } finally {
    context.destroy();
  }
}

it('cross-variogram grid path matches the dense path for pairs on cell and bin edges', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const data = createLatticeData();
  for (const gridSize of [
    [5, 5],
    [64, 64]
  ] as const) {
    const dense = await runCrossVariogram(device, data, 4, 4);
    const grid = await runCrossVariogram(device, data, 4, 4, gridSize);
    expect(grid.pairCounts).toEqual(dense.pairCounts);
    expect(grid.status).toEqual(dense.status);
    expectClose(grid.semivariances, dense.semivariances);
  }
  // Bins are left-closed: distance 1 lands in bin 1, and the final edge (distance 4) is included.
  const dense = await runCrossVariogram(device, data, 4, 4);
  const rows = data.rows;
  let atMaximum = 0;
  for (let first = 0; first < rows; first++)
    for (let second = first + 1; second < rows; second++) {
      const dx = data.positions[2 * first] - data.positions[2 * second];
      const dy = data.positions[2 * first + 1] - data.positions[2 * second + 1];
      if (dx * dx + dy * dy === 16) atMaximum++;
    }
  expect(atMaximum).toBeGreaterThan(0);
  expect(dense.pairCounts[0]).toBe(0);
});

it('cross-variogram grid path matches the dense path when the radius exceeds the extent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const data = createLatticeData();
  const dense = await runCrossVariogram(device, data, 500, 6);
  const grid = await runCrossVariogram(device, data, 500, 6, [32, 32]);
  expect(grid.pairCounts).toEqual(dense.pairCounts);
  expect(grid.pairCounts.reduce((sum, count) => sum + count, 0)).toBe(
    (data.rows * (data.rows - 1)) / 2
  );
  expect(grid.status).toEqual(dense.status);
  expectClose(grid.semivariances, dense.semivariances);
});

it('marked point statistics grid path matches the dense path for pairs on cell and shell edges', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const data = createLatticeData();
  for (const gridSize of [
    [5, 5],
    [40, 40]
  ] as const) {
    const dense = await runMarkedPointStatistics(device, data, [0, 0, 20, 20], 4, 4);
    const grid = await runMarkedPointStatistics(device, data, [0, 0, 20, 20], 4, 4, gridSize);
    expect(grid.pairCounts).toEqual(dense.pairCounts);
    expect(grid.status).toEqual(dense.status);
    expectClose(grid.statistics, dense.statistics);
    expect(dense.status).toEqual([0, 0, 0, 0]);
  }
});

it('marked point statistics grid path matches the dense path when the radius exceeds the window', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const data = createLatticeData();
  const dense = await runMarkedPointStatistics(device, data, [0, 0, 20, 20], 60, 5);
  const grid = await runMarkedPointStatistics(device, data, [0, 0, 20, 20], 60, 5, [16, 16]);
  expect(grid.pairCounts).toEqual(dense.pairCounts);
  expect(grid.status).toEqual(dense.status);
  expectClose(grid.statistics, dense.statistics);
});

it('pair statistics grid paths loop over at most 256 rows per invocation outside the pair pass', () => {
  // Kernels are recorded on a device with the WebGPU defaults; nothing runs.
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const rows = 100_000;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const cross = new GPUCrossVariogram({
    positions: view('cross-positions', 'float32x2', rows),
    values: view('cross-values', 'float32', rows),
    parameters: view('cross-parameters', 'float32', 1),
    lagCount: 8,
    gridSize: [128, 128],
    output: {
      semivariances: view('cross-semivariances', 'float32', 8),
      pairCounts: view('cross-pair-counts', 'uint32', 8),
      status: view('cross-status', 'uint32', 8)
    }
  });
  const marked = new GPUMarkedPointStatistics({
    positions: view('marked-positions', 'float32x2', rows),
    marks: view('marked-marks', 'float32', rows),
    parameters: view('marked-parameters', 'float32', 6),
    radiusCount: 8,
    gridSize: [128, 128],
    output: {
      statistics: view('marked-statistics', 'float32', 32),
      pairCounts: view('marked-pair-counts', 'uint32', 8),
      status: view('marked-status', 'uint32', 8)
    }
  });
  const sources = getParsedKernelSources([
    ...cross.getCommandNodes(graph),
    ...marked.getCommandNodes(graph)
  ]);
  expect(sources.size).toBeGreaterThan(10);
  for (const [kernelId, source] of sources) {
    // A loop bounded by the row count would run every row in one invocation.
    expect(`${kernelId}: ${/< 100000u; row\+\+/.test(source)}`).toBe(`${kernelId}: false`);
  }
});
