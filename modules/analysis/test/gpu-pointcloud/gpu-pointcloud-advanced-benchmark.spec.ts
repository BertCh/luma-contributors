// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPUPointCloudSMRF,
  GPUPointCloudICP,
  GPUPointCloudNeighborhoods
} from '@luma-contributors/analysis/pointcloud-analysis';
import {makeSpatialFixture} from '../spatial-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
const environment = (import.meta as unknown as {env?: Record<string, string>}).env;
const enabled = environment?.VITE_POINTCLOUD_BENCHMARK === 'true';
const maximumRows = Number(environment?.VITE_POINTCLOUD_MAX_ROWS ?? 100000);
for (const mode of ['smrf', 'icp', 'skew'] as const)
  it.skipIf(!enabled)(`${mode} exact indexed scaled benchmark`, {timeout: 300000}, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const queue = (device as unknown as {handle: {queue: {onSubmittedWorkDone(): Promise<void>}}})
      .handle.queue;
    console.log(
      'POINTCLOUD_ADVANCED_DEVICE',
      JSON.stringify({
        info: device.info,
        maximumStorageBindingBytes: device.limits.maxStorageBufferBindingSize,
        maximumBufferBytes: device.limits.maxBufferSize
      })
    );
    for (const rows of [10000, 100000, 1000000].filter(value => value <= maximumRows)) {
      const fixture = makeSpatialFixture(device);
      let compiled: ReturnType<typeof fixture.graph.compile> | undefined;
      try {
        const side =
          mode === 'smrf' ? Math.min(256, Math.floor(Math.sqrt(rows))) : Math.ceil(Math.cbrt(rows));
        const data = new Float32Array(rows * 3);
        for (let row = 0; row < rows; row++) {
          const column = row % side;
          const vertical = Math.floor(row / side) % side;
          data[row * 3] = mode === 'skew' ? 0 : column + 0.25;
          data[row * 3 + 1] = mode === 'skew' ? 0 : vertical + 0.25;
          data[row * 3 + 2] =
            mode === 'smrf'
              ? 0.01 * column +
                (column % 16 >= 7 && column % 16 <= 8 && vertical % 16 >= 7 && vertical % 16 <= 8
                  ? 3
                  : 0)
              : mode === 'skew'
                ? 0
                : Math.floor(row / (side * side)) + 0.25;
        }
        const allocationStart = performance.now();
        const positions = fixture.view('float32x3', data);
        const canary = fixture.view(
          'uint32',
          new Uint32Array(mode === 'icp' ? 1 : mode === 'skew' ? 64 : rows)
        );
        let check: () => Promise<void>;
        if (mode === 'smrf') {
          const classes = fixture.view('uint32', new Uint32Array(rows).fill(6));
          const surface = fixture.view('float32', new Float32Array(side * side));
          const ground = fixture.view('uint32', new Uint32Array(rows));
          const status = fixture.view('uint32', new Uint32Array(rows));
          fixture.graph.add(
            new GPUPointCloudSMRF({
              positions: positions.view,
              classifications: classes.view,
              grid: {origin: [0, 0], cellSize: 1, width: side, height: side},
              window: 4,
              groundSurface: surface.view,
              groundMask: ground.view,
              outputClassifications: canary.view,
              status: status.view
            })
          );
          check = async () => {
            const bytes = await canary.buffer.readAsync(0, Math.min(rows, side * 10) * 4);
            const values = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
            expect(values[0]).toBe(2);
            expect(values[7 * side + 7]).toBe(1);
            expect(values[6 * side + 7]).toBe(2);
          };
        } else if (mode === 'icp') {
          const shifted = Float32Array.from(
            data,
            (value, index) => value + [0.05, -0.03, 0.02][index % 3]
          );
          const target = fixture.view('float32x3', shifted);
          const transform = fixture.view('float32', new Float32Array(16));
          const history = fixture.view('float32x3', new Float32Array(12));
          const mask = fixture.view('uint32', new Uint32Array(rows));
          fixture.graph.add(
            new GPUPointCloudICP({
              source: positions.view,
              target: target.view,
              targetGrid: {
                origin: [-1, -1, -1],
                dimensions: [side + 2, side + 2, side + 2],
                cellSize: 1
              },
              maximumDistance: 0.2,
              iterations: 4,
              robust: {method: 'huber', scale: 0.1},
              transform: transform.view,
              history: history.view,
              inlierMask: mask.view,
              status: canary.view
            })
          );
          check = async () => {
            const bytes = await transform.buffer.readAsync();
            const values = new Float32Array(bytes.buffer, bytes.byteOffset, 16);
            [0.05, -0.03, 0.02].forEach((value, index) =>
              expect(values[12 + index]).toBeCloseTo(value, 3)
            );
            const status = await canary.buffer.readAsync();
            expect(new Uint32Array(status.buffer, status.byteOffset, 1)[0]).toBe(0);
          };
        } else {
          const queryRows = fixture.view(
            'uint32',
            Uint32Array.from({length: 64}, (_, index) => index)
          );
          const indices = fixture.view('uint32', new Uint32Array(256));
          const distances = fixture.view('float32', new Float32Array(256));
          const status = fixture.view('uint32', new Uint32Array(64));
          fixture.graph.add(
            new GPUPointCloudNeighborhoods({
              positions: positions.view,
              queryRows: queryRows.view,
              grid: {origin: [-1, -1, -1], dimensions: [2, 2, 2], cellSize: 1},
              radius: 0.1,
              capacity: 4,
              indices: indices.view,
              squaredDistances: distances.view,
              counts: canary.view,
              status: status.view
            })
          );
          check = async () => {
            const bytes = await canary.buffer.readAsync();
            expect(Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, 64))).toEqual(
              Array(64).fill(rows - 1)
            );
            const ids = await indices.buffer.readAsync(0, 16);
            expect(Array.from(new Uint32Array(ids.buffer, ids.byteOffset, 4))).toEqual([
              1, 2, 3, 4
            ]);
          };
        }
        await queue.onSubmittedWorkDone();
        const allocationDeclarationMilliseconds = performance.now() - allocationStart;
        const startCompile = performance.now();
        compiled = fixture.graph.compile();
        const compileMilliseconds = performance.now() - startCompile;
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
        for (let repeat = 0; repeat < 3; repeat++) residentWallMilliseconds.push(await submit());
        const readStart = performance.now();
        await check();
        const readbackMilliseconds = performance.now() - readStart;
        console.log(
          'POINTCLOUD_ADVANCED_BENCHMARK',
          JSON.stringify({
            mode,
            rows,
            side,
            allocationDeclarationMilliseconds,
            compileMilliseconds,
            coldExecutionMilliseconds,
            residentWallMilliseconds,
            readbackMilliseconds,
            transientBytes: compiled.stats.physicalTransientResourceBytes,
            nodeCount: compiled.stats.nodeOrder.length,
            submissions: 4
          })
        );
      } finally {
        compiled?.destroy();
        fixture.destroy();
      }
    }
  });
