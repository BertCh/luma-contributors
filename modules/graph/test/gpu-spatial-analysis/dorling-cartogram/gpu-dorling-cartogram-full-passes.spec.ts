// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUDorlingCartogram} from '../../../src/gpu-spatial-analysis/dorling-cartogram';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
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

const MAXIMUM_PAIR_EVALUATIONS = 4e11; // mirrors cartogram-settings.ts MAXIMUM_CARTOGRAM_PAIR_EVALUATIONS

it('Dorling GPU runs 65536 rows at the full allowed pass count without a device loss', async () => {
  const rows = 65536;
  const passes = Math.floor(MAXIMUM_PAIR_EVALUATIONS / (rows * rows));
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const side = Math.ceil(Math.sqrt(rows));
  const positions = Array.from({length: rows}, (_, row) => [
    row % side,
    Math.floor(row / side)
  ]).flat();
  const startTime = performance.now();
  const result = await runLayout(
    device,
    positions,
    Array(rows).fill(Math.PI * 0.36),
    passes,
    1,
    0.01,
    undefined,
    4096
  );
  const milliseconds = performance.now() - startTime;
  console.log(
    `dorling-full-passes rows=${rows} passes=${passes} ms=${milliseconds.toFixed(0)} status=${Array.from(result.status)}`
  );
  expect(Array.from(result.positions).every(Number.isFinite)).toBe(true);
  expect([result.status[0], result.status[3]]).toEqual([passes, 0]);
}, 300000);
