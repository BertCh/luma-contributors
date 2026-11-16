// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUDorlingCartogram} from '../../../src/gpu-spatial-analysis/dorling-cartogram/index';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

async function runLayout(
  device: Device,
  positions: number[] | Float32Array,
  values: number[] | Float32Array,
  maximumIterations: number,
  neighborGrid: boolean | undefined,
  rowTileSize?: number,
  attraction = 0.01,
  options: {
    readAll?: boolean;
    traceGates?: boolean;
    updatedInputs?: {positions: Float32Array; values: Float32Array};
  } = {}
) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const rows = values.length;
  const input = <Format extends 'float32' | 'float32x2'>(
    name: string,
    format: Format,
    data: Float32Array
  ) => {
    const buffer = device.createBuffer({data, usage: Buffer.STORAGE | Buffer.COPY_DST});
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, rows);
  };
  const output = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    length: number,
    words = 1
  ) => {
    const buffer = createOutputBuffer(device, length * words);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const centers = output('centers', 'float32x2', rows, 2);
  const radii = output('radii', 'float32', rows);
  const sourceIds = output('ids', 'uint32', rows);
  const metrics = output('metrics', 'float32x2', 1, 2);
  const status = output('status', 'uint32', 4);
  const positionsView = input('positions', 'float32x2', new Float32Array(positions));
  const positionsBuffer = buffers[buffers.length - 1];
  const valuesView = input('values', 'float32', new Float32Array(values));
  const valuesBuffer = buffers[buffers.length - 1];
  const contributor = new GPUDorlingCartogram({
    positions: positionsView,
    values: valuesView,
    maximumIterations,
    neighborGrid,
    rowTileSize,
    attraction,
    output: {
      positions: centers.view,
      radii: radii.view,
      sourceIds: sourceIds.view,
      metrics: metrics.view,
      status: status.view
    }
  });
  const nodes = contributor.getCommandNodes(graph);
  graph.add(nodes);
  // Read the GPU-written phase predicates after execution. Aligned scalar destinations keep
  // every trace kernel identical while checking all later phases launch zero workgroups.
  const gateBuffers = [
    ...new Set(
      nodes.flatMap(node => (node.condition?.source === 'gpu' ? [node.condition.buffer] : []))
    )
  ];
  const trace = options.traceGates
    ? output('gate-trace', 'uint32', gateBuffers.length * 64)
    : undefined;
  if (trace)
    gateBuffers.forEach((buffer, phase) => {
      graph.add(
        createWGSLKernelNode(graph, {
          id: `trace-${phase}`,
          operation: 'DorlingGateTrace',
          invocationCount: 1,
          bindings: [
            {
              name: 'gateWords',
              view: graph.createDataView(buffer, {format: 'uint32', length: 3}),
              type: 'u32',
              access: 'read'
            },
            {
              name: 'traceWords',
              view: graph.createDataView(trace.view.buffer, {
                format: 'uint32',
                length: 1,
                byteOffset: phase * 256
              }),
              type: 'u32',
              access: 'read_write'
            }
          ],
          body: 'traceWords[traceWordsOffset] = gateWords[gateWordsOffset];'
        })
      );
    });
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    if (options.updatedInputs) {
      await readUint32(status.buffer, 4);
      positionsBuffer.write(options.updatedInputs.positions);
      valuesBuffer.write(options.updatedInputs.values);
      submitGraph(device, compiled, undefined);
    }
    const gateWords = trace ? await readUint32(trace.buffer, gateBuffers.length * 64) : undefined;
    return {
      positions: await readFloat32(centers.buffer, rows * 2),
      radii: options.readAll ? await readFloat32(radii.buffer, rows) : undefined,
      sourceIds: options.readAll ? await readUint32(sourceIds.buffer, rows) : undefined,
      gateGroups: gateWords?.filter((_, index) => index % 64 === 0),
      metrics: await readFloat32(metrics.buffer, 2),
      status: await readUint32(status.buffer, 4)
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

/** Deterministic clustered layout with varied radii and a few excluded rows. */
function makeClusteredLayout(rows: number) {
  let seed = 12345;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const positions: number[] = [];
  const values: number[] = [];
  for (let row = 0; row < rows; row++) {
    const cluster = row % 5;
    positions.push(cluster * 40 + random() * 12, (cluster % 2) * 30 + random() * 12);
    values.push(row % 41 === 7 ? -1 : 0.5 + random() * 6);
  }
  positions[2 * 3] = NaN;
  return {positions, values};
}

for (const maximumIterations of [0, 1, 12, 25]) {
  it(`Dorling neighborGrid matches the all-pairs search over ${maximumIterations} passes`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {positions, values} = makeClusteredLayout(400);
    const reference = await runLayout(device, positions, values, maximumIterations, false);
    const gridded = await runLayout(device, positions, values, maximumIterations, true);
    expect(gridded.status).toEqual(reference.status);
    expect(gridded.metrics[0]).toBeCloseTo(reference.metrics[0], 3);
    expect(gridded.metrics[1]).toBeCloseTo(reference.metrics[1], 2);
    for (let index = 0; index < reference.positions.length; index++) {
      const expected = reference.positions[index];
      if (Number.isNaN(expected)) {
        expect(gridded.positions[index]).toBeNaN();
      } else {
        expect(gridded.positions[index]).toBeCloseTo(expected, 2);
      }
    }
  });
}

it('Dorling neighborGrid resolves a heavily piled layout and honors row tiles', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 120 circles on one point: every cell lookup lands in one cell.
  const rows = 120;
  const positions = Array.from({length: rows * 2}, () => 5);
  const values = Array.from({length: rows}, () => Math.PI);
  const reference = await runLayout(device, positions, values, 30, false);
  const gridded = await runLayout(device, positions, values, 30, true, 32);
  expect(gridded.status).toEqual(reference.status);
  expect(gridded.metrics[0]).toBeCloseTo(reference.metrics[0], 3);
});

it('Dorling neighborGrid handles 16384 circles over 8 passes and reports its timing', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 16384;
  const side = Math.ceil(Math.sqrt(rows));
  const positions = Array.from({length: rows}, (_, row) => [
    row % side,
    Math.floor(row / side)
  ]).flat();
  const values = Array(rows).fill(Math.PI * 0.36);
  const timings: Record<string, number> = {};
  const results: Record<string, number[]> = {};
  for (const neighborGrid of [false, true]) {
    const startTime = performance.now();
    const result = await runLayout(device, positions, values, 8, neighborGrid, 4096);
    timings[neighborGrid ? 'grid' : 'all-pairs'] = performance.now() - startTime;
    results[neighborGrid ? 'grid' : 'all-pairs'] = result.positions;
  }
  console.log(
    `dorling-neighbor-grid rows=${rows} passes=8 all-pairs=${timings['all-pairs'].toFixed(0)}ms grid=${timings['grid'].toFixed(0)}ms`
  );
  expect(Array.from(results['grid']).every(Number.isFinite)).toBe(true);
  expect(results['grid'][0]).toBeCloseTo(results['all-pairs'][0], 2);
});

it('Dorling indexed and all-pairs outputs are identical for independent overlapping pairs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Each row has one overlapping neighbor, so the correction has the same summation order.
  const positions = [-5, -2, -4.5, -2, 10, 4, 10.5, 4, 0, 0, 40, 40];
  const values = [Math.PI * 0.25, Math.PI * 0.25, Math.PI, Math.PI, 0, -1];
  const reference = await runLayout(device, positions, values, 4, false, undefined, 0);
  const indexed = await runLayout(device, positions, values, 4, true, undefined, 0);
  expect(indexed).toEqual(reference);
});

it('Dorling default grid relaxes one million rows and matches the pair solution', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 1000000;
  const positions = new Float32Array(rows * 2);
  const values = new Float32Array(rows).fill(Math.PI * 0.25);
  const columns = 1000;
  for (let row = 0; row < rows; row++) {
    const pair = Math.floor(row / 2);
    positions[row * 2] = (pair % columns) * 4 + (row % 2) * 0.5;
    positions[row * 2 + 1] = Math.floor(pair / columns) * 4;
  }
  const result = await runLayout(device, positions, values, 1, undefined, undefined, 0);
  expect(result.status).toEqual([1, 1, 0, 0]);
  expect(result.metrics[0]).toBeLessThan(0.001);
  expect(result.metrics[1]).toBeCloseTo(0.25, 4);
  // Sampling across the full packed output catches dropped rows, edge cells and indexing errors.
  for (let row = 0; row < rows; row += 997) {
    expect(result.positions[row * 2]).toBeCloseTo(positions[row * 2] + (row % 2 ? 0.25 : -0.25), 3);
    expect(result.positions[row * 2 + 1]).toBe(positions[row * 2 + 1]);
  }
}, 120000);

function makePairs(rows: number, radius = 0.5, translation = 0) {
  const positions = new Float32Array(rows * 2);
  const values = new Float32Array(rows).fill(Math.PI * radius * radius);
  const columns = Math.ceil(Math.sqrt(rows / 2));
  for (let row = 0; row < rows; row++) {
    const pair = Math.floor(row / 2);
    positions[row * 2] = translation + (pair % columns) * 8 + (row % 2) * 0.5;
    positions[row * 2 + 1] = translation + Math.floor(pair / columns) * 8;
  }
  values[7] = -1;
  values[9] = 0;
  positions[6] = NaN;
  return {positions, values};
}

function expectPairSolution(
  result: Awaited<ReturnType<typeof runLayout>>,
  inputs: ReturnType<typeof makePairs>,
  radius = 0.5
) {
  const rows = inputs.values.length;
  let invalidRows = 0;
  const included = (row: number) =>
    row < rows && inputs.values[row] >= 0 && Number.isFinite(inputs.positions[row * 2]);
  for (let row = 0; row < rows; row++) {
    expect(result.sourceIds![row]).toBe(row);
    if (!included(row)) {
      invalidRows++;
      expect(result.positions[row * 2]).toBeNaN();
      expect(result.positions[row * 2 + 1]).toBeNaN();
      expect(result.radii![row]).toBeNaN();
      continue;
    }
    const other = row ^ 1;
    const hasOverlap = inputs.values[row] > 0 && included(other) && inputs.values[other] > 0;
    const shift = hasOverlap ? (radius - 0.25) * (row % 2 ? 1 : -1) : 0;
    expect(result.positions[row * 2]).toBeCloseTo(inputs.positions[row * 2] + shift, 3);
    expect(result.positions[row * 2 + 1]).toBe(inputs.positions[row * 2 + 1]);
    expect(result.radii![row]).toBeCloseTo(inputs.values[row] === 0 ? 0 : radius, 5);
  }
  expect(result.status).toEqual([1, 1, 0, invalidRows]);
  expect(result.metrics[0]).toBeLessThan(0.001);
  expect(result.metrics[1]).toBeCloseTo(radius - 0.25, 4);
}

for (const rows of [255, 256, 257, 65535, 65536]) {
  it(`Dorling indexed and reference routes match independent pair solutions at ${rows} rows`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const inputs = makePairs(rows);
    // Pair 254/255 crosses a tile seam; spatial cells cut many pairs throughout the lattice.
    const reference = await runLayout(device, inputs.positions, inputs.values, 1, false, 255, 0, {
      readAll: true
    });
    const indexed = await runLayout(device, inputs.positions, inputs.values, 1, true, 255, 0, {
      readAll: true
    });
    expectPairSolution(reference, inputs);
    expectPairSolution(indexed, inputs);
  }, 180000);
}

it('Dorling re-encoding resets converged gates, bounds and largest radius', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const initial = makePairs(257);
  const changed = makePairs(257, 1.5, -1000);
  const reused = await runLayout(device, initial.positions, initial.values, 4, true, 64, 0, {
    readAll: true,
    updatedInputs: changed
  });
  const fresh = await runLayout(device, changed.positions, changed.values, 4, true, 64, 0, {
    readAll: true
  });
  expectPairSolution(reused, changed, 1.5);
  expect(reused).toEqual(fresh);
});

it('Dorling separated million-row layouts stop GPU work after convergence for budgets 1 and 100', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 1000000;
  const positions = new Float32Array(rows * 2);
  const values = new Float32Array(rows).fill(Math.PI * 0.25);
  for (let row = 0; row < rows; row++) {
    positions[row * 2] = (row % 1000) * 4;
    positions[row * 2 + 1] = Math.floor(row / 1000) * 4;
  }
  let firstPositions: number[] | undefined;
  for (const budget of [1, 100]) {
    const startTime = performance.now();
    const result = await runLayout(device, positions, values, budget, true, undefined, 0, {
      traceGates: true
    });
    console.log(
      `dorling-separated rows=${rows} budget=${budget} ms=${performance.now() - startTime}`
    );
    expect(result.status).toEqual([0, 1, 0, 0]);
    expect(result.metrics).toEqual([0, 0]);
    expect(result.gateGroups).toHaveLength(budget + 1);
    expect(result.gateGroups![0]).toBeGreaterThan(0);
    expect(result.gateGroups!.slice(1)).toEqual(Array(budget).fill(0));
    expect(result.positions).toEqual(Array.from(positions));
    if (firstPositions) expect(result.positions).toEqual(firstPositions);
    firstPositions = result.positions;
  }
}, 180000);

it('Dorling preserves the independent pair solution across an uneven dispatch tail', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const inputs = makePairs(4097);
  // Full tiles use four workgroups, the five-row tail uses one, and pair 1022/1023 crosses a seam.
  const result = await runLayout(device, inputs.positions, inputs.values, 2, true, 1023, 0, {
    readAll: true
  });
  expectPairSolution(result, inputs);
});
