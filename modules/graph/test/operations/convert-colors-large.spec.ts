// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {convertColorData} from '@luma-contributors/graph/gpu-simple-graph';
import {GPUData} from '@luma-contributors/graph/gpu-data';
import {expect, test} from 'vitest';
import {getTestDevice} from './fixtures';

test('convertColorData converts more rows than fit in one 1D dispatch', async t => {
  const device = await getTestDevice('webgpu');
  if (!device) {
    t.skip('webgpu not available');
    return;
  }
  // 64 rows per workgroup: one workgroup beyond the 1D limit forces a multi-dimensional dispatch.
  const rowCount = (device.limits.maxComputeWorkgroupsPerDimension + 1) * 64 + 5;
  const values = new Uint8Array(rowCount * 4);
  values[(rowCount - 1) * 4] = 77;
  values[(rowCount - 1) * 4 + 3] = 200;
  const buffer = device.createBuffer({
    usage: Buffer.VERTEX | Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC,
    data: values
  });
  const source = new GPUData({buffer, format: 'uint8x4', length: rowCount, ownsBuffer: true});

  const result = convertColorData(source, {});
  await result.evaluate(device);
  const converted = await result.readValue();
  expect(converted.length).toBe(rowCount * 4);
  expect(Array.from(converted.slice(-4))).toEqual([77, 0, 0, 200]);
  result.destroy();
  source.destroy();
});
