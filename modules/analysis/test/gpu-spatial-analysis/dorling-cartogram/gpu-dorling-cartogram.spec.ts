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
import {DORLING_PAIR_FIXTURE} from './dorling-invariants-fixture';

async function runLayout(
  device: Device,
  positions: number[],
  values: number[],
  maximumIterations: number,
  areaScale = 1,
  attraction = 0.01,
  updatedValues?: number[],
  rowTileSize?: number
) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const rows = values.length;
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    data: Float32Array | Uint32Array
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
      sourceIds: input(
        'source-ids',
        'uint32',
        new Uint32Array(values.map((_, index) => (index === 0 ? 91 : 7)))
      ),
      maximumIterations,
      areaScale,
      attraction,
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
    if (updatedValues) {
      await readUint32(status.buffer, 4);
      buffers[6].write(new Float32Array(updatedValues));
      submitGraph(device, compiled, undefined);
    }
    return {
      positions: await readFloat32(centers.buffer, rows * 2),
      radii: await readFloat32(radii.buffer, rows),
      ids: await readUint32(sourceIds.buffer, rows),
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

it('Dorling GPU circles preserve areas and IDs and resolve coincident pair', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = DORLING_PAIR_FIXTURE;
  const result = await runLayout(device, fixture.positions, fixture.values, 10);
  expect(result.radii[0]).toBeCloseTo(1, 5);
  expect(result.ids).toEqual(fixture.sourceIds);
  expect(
    Math.hypot(result.positions[0] - result.positions[2], result.positions[1] - result.positions[3])
  ).toBeCloseTo(fixture.separatedDistance, 5);
  expect(result.metrics[0]).toBeLessThanOrEqual(0.001);
  expect(result.metrics[1]).toBeCloseTo(fixture.displacement, 5);
  expect(result.status).toEqual([1, 1, 0, 0]);
});

it('Dorling GPU reports budget exhaustion and excludes invalid rows while retaining zeros', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const budget = await runLayout(device, [0, 0, 0, 0], [Math.PI, Math.PI], 0);
  expect(budget.status).toEqual([0, 0, 1, 0]);
  expect(budget.metrics[0]).toBeCloseTo(2, 5);
  const invalid = await runLayout(
    device,
    [0, 0, 1, 0, 2, 0, 3, 0, NaN, 0],
    [0, -1, NaN, 4, 1],
    2,
    Math.PI
  );
  expect(invalid.status[3]).toBe(3);
  expect(invalid.radii[0]).toBe(0);
  expect(invalid.radii[3]).toBeCloseTo(2, 5);
  expect(invalid.positions[0]).toBe(0);
  for (const row of [1, 2, 4]) {
    expect(invalid.radii[row]).toBeNaN();
    expect(invalid.positions[row * 2]).toBeNaN();
  }
});

it('Dorling GPU resolves heterogeneous clusters without attraction and preserves circle area', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const values = Array.from({length: 16}, (_, row) => Math.PI * (1 + (row % 4)) ** 2);
  const positions = values.flatMap((_, row) => [row % 4, Math.floor(row / 4)]);
  const result = await runLayout(device, positions, values, 256, 1, 0);
  for (let row = 0; row < values.length; row++) {
    expect(result.radii[row] ** 2 * Math.PI).toBeCloseTo(values[row], 3);
    expect(Number.isFinite(result.positions[row * 2])).toBe(true);
  }
  let maximumOverlap = 0;
  let maximumDisplacement = 0;
  for (let row = 0; row < values.length; row++) {
    maximumDisplacement = Math.max(
      maximumDisplacement,
      Math.hypot(
        result.positions[row * 2] - positions[row * 2],
        result.positions[row * 2 + 1] - positions[row * 2 + 1]
      )
    );
    for (let other = row + 1; other < values.length; other++) {
      maximumOverlap = Math.max(
        maximumOverlap,
        result.radii[row] +
          result.radii[other] -
          Math.hypot(
            result.positions[row * 2] - result.positions[other * 2],
            result.positions[row * 2 + 1] - result.positions[other * 2 + 1]
          )
      );
    }
  }
  expect(result.metrics[0]).toBeCloseTo(maximumOverlap, 4);
  expect(result.metrics[1]).toBeCloseTo(maximumDisplacement, 4);
  expect(result.status[1]).toBe(Number(maximumOverlap <= 0.001));
  expect(maximumOverlap).toBeLessThan(0.01);
});

it('Dorling GPU handles supported maximum capacity and reports attraction equilibrium honestly', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 1024;
  const result = await runLayout(
    device,
    Array.from({length: rows}, (_, row) => [row * 3, 0]).flat(),
    Array(rows).fill(Math.PI),
    2
  );
  expect(result.status).toEqual([0, 1, 0, 0]);
  expect(result.metrics).toEqual([0, 0]);
  const equilibrium = await runLayout(
    device,
    [0, 0, 0, 0, 0, 0],
    [Math.PI, Math.PI, Math.PI],
    4,
    1,
    1
  );
  expect(equilibrium.status[1]).toBe(0);
  expect(equilibrium.status[2]).toBe(1);
  expect(equilibrium.metrics[0]).toBeGreaterThan(0.001);
});

it('Dorling GPU row tiling does not change the layout', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 40;
  const positions = Array.from({length: rows}, (_, row) => [
    (row % 8) * 0.9,
    Math.floor(row / 8) * 0.9
  ]).flat();
  const values = Array.from({length: rows}, (_, row) => 1 + (row % 5));
  const whole = await runLayout(device, positions, values, 12);
  const tiled = await runLayout(device, positions, values, 12, 1, 0.01, undefined, 7);
  expect(tiled.positions).toEqual(whole.positions);
  expect(tiled.metrics).toEqual(whole.metrics);
  expect(tiled.status).toEqual(whole.status);
});

it('Dorling re-encoding refreshes radii, status and initialization from changed GPU input', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runLayout(device, [0, 0, 0, 0], [Math.PI, Math.PI], 4, 1, 0, [0, 0]);
  expect(result.radii).toEqual([0, 0]);
  expect(result.positions).toEqual([0, 0, 0, 0]);
  expect(result.status).toEqual([0, 1, 0, 0]);
  expect(result.metrics).toEqual([0, 0]);
});

it('Dorling GPU executes the full 1024-row 256-pass envelope with unresolved overlap', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 1024;
  const result = await runLayout(
    device,
    Array.from({length: rows}, (_, row) => [row % 32, Math.floor(row / 32)]).flat(),
    Array(rows).fill(Math.PI * 4),
    256,
    1,
    1
  );
  expect(result.status).toEqual([256, 0, 1, 0]);
  expect(result.metrics[0]).toBeGreaterThan(0.001);
  expect(result.metrics.every(Number.isFinite)).toBe(true);
  expect(result.positions.every(Number.isFinite)).toBe(true);
}, 60000);
