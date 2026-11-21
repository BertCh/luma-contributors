// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUGeodesicPairs} from '../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../src/testing/gpu-contributor-test-utils';

it('DGN-07 huge longitude matches canonical endpoint and convergence status', async context => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device, context);
  const graph = new GPUCommandGraph(device, {id: 'dgn-07-wrapped-endpoints'});
  const origins = createInputBuffer(device, Float32Array.of(2 ** 62, 0, -176, 0));
  const targets = createInputBuffer(device, Float32Array.of(0, 0, 0, 0));
  const distances = createOutputBuffer(device, 2);
  const converged = createOutputBuffer(device, 2);
  const approximation = createOutputBuffer(device, 2);
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    graph.add(
      new GPUGeodesicPairs({
        // Flattening zero gives a closed-form independent oracle: R * 176 * pi/180.
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'ellipsoidal',
          ellipsoid: {semiMajorAxis: 6371008.8, flattening: 0}
        },
        origins: importGraphBuffer(graph, 'origins', origins, 'float32x2', 2),
        targets: importGraphBuffer(graph, 'targets', targets, 'float32x2', 2),
        output: {
          distances: importGraphBuffer(graph, 'distances', distances, 'float32', 2),
          status: {
            converged: importGraphBuffer(graph, 'converged', converged, 'uint32', 2),
            approximation: importGraphBuffer(graph, 'approximation', approximation, 'uint32', 2)
          }
        }
      })
    );
    compiled = graph.compile();
    submitGraph(device, compiled);
    const actual = await readFloat32(distances);
    expect(actual).toHaveLength(2);
    for (const distance of actual) {
      expect(Number.isFinite(distance)).toBe(true);
      expect(Math.abs(distance - 19570334.12110179)).toBeLessThan(8);
    }
    expect(Math.abs(actual[0] - actual[1])).toBeLessThan(4);
    expect(await readUint32(converged)).toEqual([1, 1]);
    expect(await readUint32(approximation)).toEqual([0, 0]);
  } finally {
    compiled?.destroy();
    for (const buffer of [origins, targets, distances, converged, approximation]) buffer.destroy();
  }
});
