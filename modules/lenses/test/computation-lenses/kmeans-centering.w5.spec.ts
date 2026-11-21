// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUSpatialKMeans,
  getGPUSpatialKMeansParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {requireTestDevice} from '../../../analysis/src/testing/require-test-device';
import {readFloat32, submitGraph} from '../../../analysis/src/testing/gpu-contributor-test-utils';
import {expectKMeansMajorityDistances} from './kmeans-centering-assertions';

test('FLT-04: omitted centering keeps the 255-row inertia and minority distance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph<undefined>(device);
  const buffers: Buffer[] = [];
  const upload = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    id: string,
    data: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = device.createBuffer({
      id,
      data,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return {
      buffer,
      view: graph.createDataView(
        graph.importBuffer({id, byteLength: buffer.byteLength, usage: buffer.usage}, buffer),
        {format, length: data.length / (format === 'float32x2' ? 2 : 1)}
      )
    };
  };
  const coordinates = new Float32Array(510);
  for (let row = 0; row < 255; row++) coordinates[row * 2] = row < 254 ? 10_000_001 : 10_000_002;
  const positions = upload('positions', coordinates, 'float32x2');
  const parameters = upload(
    'parameters',
    getGPUSpatialKMeansParameterValues({seed: 3, tolerance: 0}),
    'uint32'
  );
  const centers = upload('centers', new Float32Array(2), 'float32x2');
  const labels = upload('labels', new Uint32Array(255), 'uint32');
  const squaredDistances = upload('distances', new Float32Array(255), 'float32');
  const inertia = upload('inertia', new Float32Array(1), 'float32');
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    graph.add(
      new GPUSpatialKMeans({
        positions: positions.view,
        parameters: parameters.view,
        clusterCount: 1,
        maximumIterations: 2,
        initialization: 'first-valid',
        output: {
          labels: labels.view,
          centers: centers.view,
          squaredDistances: squaredDistances.view,
          inertia: inertia.view
        }
      })
    );
    compiled = graph.compile();
    submitGraph(device, compiled);
    const [objective] = await readFloat32(inertia.buffer, 1);
    const distances = await readFloat32(squaredDistances.buffer, 255);
    // Exact center offset = 1/255. SSE = 254*(1/255)^2 + (254/255)^2 = 254/255.
    // /usr/bin/python3: 254/255, (254/255)**2, (1/255)**2.
    for (const value of [objective, ...distances]) expect(Number.isFinite(value)).toBe(true);
    expect(objective).toBeCloseTo(0.996078431372549, 5);
    expect(distances[254]).toBeCloseTo(0.9921722414455979, 5);
    expectKMeansMajorityDistances(distances);
    // The absolute published center rounds; distances must retain the internal offset.
    expect(await readFloat32(centers.buffer, 2)).toEqual([10_000_001, 0]);
  } finally {
    compiled?.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});
