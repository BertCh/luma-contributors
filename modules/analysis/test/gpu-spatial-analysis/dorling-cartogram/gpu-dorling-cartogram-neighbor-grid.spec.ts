// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUDorlingCartogram} from '../../../src/gpu-spatial-analysis/dorling-cartogram/index';
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
  positions: number[],
  values: number[],
  maximumIterations: number,
  neighborGrid: boolean,
  rowTileSize?: number
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
  graph.add(
    new GPUDorlingCartogram({
      positions: input('positions', 'float32x2', new Float32Array(positions)),
      values: input('values', 'float32', new Float32Array(values)),
      maximumIterations,
      neighborGrid,
      rowTileSize,
      output: {
        positions: centers.view,
        radii: radii.view,
        sourceIds: sourceIds.view,
        metrics: metrics.view,
        status: status.view
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      positions: await readFloat32(centers.buffer, rows * 2),
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
