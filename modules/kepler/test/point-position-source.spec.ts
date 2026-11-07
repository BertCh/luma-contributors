// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {prepareArrowTableForGPUAnalytics} from '@luma-contributors/analysis/analysis-core';
import {KeplerDatasetRegistry} from '../src/runtime/dataset-registry';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {
  prepareTabularPointPositions,
  makeGPUVectorsFromTabularPointPositions
} from '../../../examples/deck/kepler-compute/src/point-position-source';
import type {GPUVector} from '@luma.gl/graph/gpu-data';

test('tabular altitude uploads original XYZ chunks and native Arrow null policy without changing XY geometry', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const batches = [[0, -50], [], [null, 150, NaN]].map(heights => {
    const source = new arrow.Table({
      east: arrow.vectorFromArray(
        heights.map(() => -73),
        new arrow.Float32()
      ),
      north: arrow.vectorFromArray(
        heights.map(() => 40),
        new arrow.Float32()
      ),
      height: arrow.vectorFromArray(heights, new arrow.Float32())
    });
    return arrow.tableFromIPC(arrow.tableToIPC(source, 'stream')).batches[0];
  });
  const source = new arrow.Table(batches[0].schema, batches);
  const prepared = prepareTabularPointPositions(prepareArrowTableForGPUAnalytics(source), {
    longitude: 'east',
    latitude: 'north',
    altitude: 'height'
  });
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1024});
  const vectors = makeGPUVectorsFromTabularPointPositions(device, prepared, {
    memoryRegistry: registry
  });
  try {
    expect(vectors.positions.data.map(chunk => chunk.length)).toEqual([2, 0, 3]);
    expect(vectors.positions3D!.data.map(chunk => chunk.length)).toEqual([2, 0, 3]);
    expect(await readFloat32(vectors.positions)).toEqual([
      -73, 40, -73, 40, -73, 40, -73, 40, -73, 40
    ]);
    expect(await readFloat32(vectors.positions3D!)).toEqual([
      -73, 40, 0, -73, 40, -50, -73, 40, 0, -73, 40, 150, -73, 40, 0
    ]);
    const validity: number[] = [];
    for (const chunk of vectors.altitudeValidity!.data) {
      if (!chunk.length) continue;
      const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * 4);
      validity.push(...new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length));
    }
    expect(validity).toEqual([1, 1, 1, 1, 0]);
    expect(source.getChild('height')?.get(2)).toBeNull();
    expect(source.getChild('height')?.get(4)).toBeNaN();
    expect(registry.memoryUsage.reservedBytes.gpu).toBe(144);
  } finally {
    const fence = device.createFence();
    await vectors.destroy(fence.signaled);
    fence.destroy();
    expect(registry.memoryUsage.totalBytes).toBe(0);
    await registry.destroy();
  }
});

async function readFloat32(
  vector: GPUVector<'float32x2'> | GPUVector<'float32x3'>
): Promise<number[]> {
  const width = vector.format === 'float32x3' ? 3 : 2;
  const values: number[] = [];
  for (const chunk of vector.data) {
    if (!chunk.length) continue;
    const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * width * 4);
    values.push(...new Float32Array(bytes.buffer, bytes.byteOffset, chunk.length * width));
  }
  return values;
}
