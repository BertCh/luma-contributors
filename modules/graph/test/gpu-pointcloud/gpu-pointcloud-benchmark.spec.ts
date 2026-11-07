// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPUPointCloudNeighborhoods,
  GPUPointCloudFeatures
} from '@luma-contributors/graph/gpu-pointcloud';
import {makeSpatialFixture} from '../spatial-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

// Opt in: VITE_POINTCLOUD_BENCHMARK=true yarn test-headless --no-coverage --maxWorkers=1 <this file>
const environment = (import.meta as unknown as {env?: Record<string, string>}).env;
const enabled = environment?.VITE_POINTCLOUD_BENCHMARK === 'true';
const maximumRows = Number(environment?.VITE_POINTCLOUD_MAX_ROWS ?? 100000);

it.skipIf(!enabled)(
  '3D grid neighborhood + covariance scaled resident benchmark',
  {timeout: 300000},
  async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const queue = (
      device as unknown as {handle: {queue: {onSubmittedWorkDone: () => Promise<void>}}}
    ).handle.queue;
    console.log(
      'POINTCLOUD_DEVICE',
      JSON.stringify({
        info: device.info,
        maximumStorageBindingBytes: device.limits.maxStorageBufferBindingSize,
        maximumBufferBytes: device.limits.maxBufferSize
      })
    );
    for (const rows of [1000, 10000, 100000, 1000000].filter(value => value <= maximumRows)) {
      const capacity = 8;
      // Every individual binding, including each neighbor output, must fit the actual device.
      if (
        rows * capacity * 4 >
        Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize)
      ) {
        console.log(
          'POINTCLOUD_BENCHMARK',
          JSON.stringify({rows, skipped: 'storage binding limit'})
        );
        continue;
      }
      const side = Math.ceil(Math.cbrt(rows));
      const source = new Float32Array(rows * 3);
      for (let row = 0; row < rows; row++) {
        source[row * 3] = (row % side) + 0.25;
        source[row * 3 + 1] = (Math.floor(row / side) % side) + 0.25;
        source[row * 3 + 2] = Math.floor(row / (side * side)) + 0.25;
      }
      const fixture = makeSpatialFixture(device);
      let compiled: ReturnType<typeof fixture.graph.compile> | undefined;
      try {
        const allocationStart = performance.now();
        const positions = fixture.view('float32x3', source);
        const indices = fixture.view('uint32', new Uint32Array(rows * capacity));
        const squaredDistances = fixture.view('float32', new Float32Array(rows * capacity));
        const counts = fixture.view('uint32', new Uint32Array(rows));
        const normals = fixture.view('float32x3', new Float32Array(rows * 3));
        const eigenvalues = fixture.view('float32x3', new Float32Array(rows * 3));
        const shapes = fixture.view('float32x3', new Float32Array(rows * 3));
        const status = fixture.view('uint32', new Uint32Array(rows));
        await queue.onSubmittedWorkDone();
        const allocationUploadMilliseconds = performance.now() - allocationStart;
        const declarationStart = performance.now();
        fixture.graph.add(
          new GPUPointCloudNeighborhoods({
            positions: positions.view,
            radius: 1.5,
            capacity,
            grid: {origin: [0, 0, 0], cellSize: 1, dimensions: [side, side, side]},
            indices: indices.view,
            squaredDistances: squaredDistances.view,
            counts: counts.view
          })
        );
        fixture.graph.add(
          new GPUPointCloudFeatures({
            positions: positions.view,
            indices: indices.view,
            counts: counts.view,
            capacity,
            normals: normals.view,
            eigenvalues: eigenvalues.view,
            shapes: shapes.view,
            status: status.view
          })
        );
        const declarationMilliseconds = performance.now() - declarationStart;
        const compileStart = performance.now();
        compiled = fixture.graph.compile();
        const compileMilliseconds = performance.now() - compileStart;
        const submit = async () => {
          const start = performance.now();
          const encoder = device.createCommandEncoder();
          compiled!.encode(encoder, {parameters: undefined});
          device.submit(encoder.finish());
          await queue.onSubmittedWorkDone();
          return performance.now() - start;
        };
        const coldExecutionMilliseconds = await submit();
        const residentWallMilliseconds = [];
        for (let sample = 0; sample < 3; sample++) residentWallMilliseconds.push(await submit());
        const readStart = performance.now();
        const countBytes = await counts.buffer.readAsync(0, 4);
        const countValues = new Uint32Array(countBytes.buffer, countBytes.byteOffset, 1);
        const statusBytes = await status.buffer.readAsync(0, 4);
        const statusValues = new Uint32Array(statusBytes.buffer, statusBytes.byteOffset, 1);
        const readbackMilliseconds = performance.now() - readStart;
        expect(countValues[0]).toBe(6);
        expect(statusValues[0] & 5).toBe(0);
        console.log(
          'POINTCLOUD_BENCHMARK',
          JSON.stringify({
            rows,
            capacity,
            allocationUploadMilliseconds,
            declarationMilliseconds,
            compileMilliseconds,
            coldExecutionMilliseconds,
            residentWallMilliseconds,
            readbackMilliseconds,
            transientBytes: compiled.stats.physicalTransientResourceBytes,
            logicalTransientBytes: compiled.stats.logicalTransientBytes,
            nodeCount: compiled.stats.nodeOrder.length,
            outputBytes: rows * (capacity * 8 + 44),
            submissionCount: 4,
            firstPointSupport: countValues[0]
          })
        );
      } finally {
        compiled?.destroy();
        fixture.destroy();
      }
    }
  }
);
