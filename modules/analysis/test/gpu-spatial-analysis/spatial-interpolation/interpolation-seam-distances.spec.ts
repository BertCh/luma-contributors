// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  getSeparationWGSL,
  resolveInterpolationSeparation
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/separation-wgsl';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('PTS-117 spherical distance formulas retain both seam sides without the clipped candidate box', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const positions: number[] = [];
  const queries: number[] = [];
  for (const sign of [-1, 1])
    for (const seam of [false, true]) {
      const longitude = (x: number) => sign * (seam ? (x > 0 ? x - 180 : x + 180) : x);
      for (const offset of [-0.125, -0.25, 0.125, 0.25]) {
        positions.push(longitude(offset), 0);
        queries.push(longitude(-0.0625), 0);
      }
    }
  const sampleBuffer = createInputBuffer(device, Float32Array.from(positions));
  const queryBuffer = createInputBuffer(device, Float32Array.from(queries));
  const resultBuffer = createOutputBuffer(device, 16);
  const samples = importGraphBuffer(graph, 'samples', sampleBuffer, 'float32x2', 16);
  const queryRows = importGraphBuffer(graph, 'queries', queryBuffer, 'float32x2', 16);
  const distances = importGraphBuffer(graph, 'distances', resultBuffer, 'float32', 16);
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'seam-distances',
      operation: 'InterpolationSeamDistanceProbe',
      invocationCount: 16,
      bindings: [
        {name: 'samples', view: samples, type: 'f32', access: 'read'},
        {name: 'queries', view: queryRows, type: 'f32', access: 'read'},
        {name: 'distances', view: distances, type: 'f32', access: 'read_write'}
      ],
      declarations: getSeparationWGSL(
        resolveInterpolationSeparation('seam-distances', {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle'
        })
      ),
      body: `let sample = vec2f(samples[samplesOffset+index*2u], samples[samplesOffset+index*2u+1u]);
let query = vec2f(queries[queriesOffset+index*2u], queries[queriesOffset+index*2u+1u]);
distances[distancesOffset+index] = getSeparationMeters(getSeparationKey(sample,query));`
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const actual = await readFloat32(resultBuffer, 16);
    // Independent equatorial arc lengths R*pi/180*degrees, R=6371008.8 m.
    const expected = [6949.692514595808, 20849.07754378742, 20849.07754378742, 34748.462572979035];
    for (let row = 0; row < 16; row++) {
      expect(Number.isFinite(actual[row])).toBe(true);
      expect(Math.abs(actual[row] - expected[row % 4])).toBeLessThan(5);
    }
  } finally {
    compiled.destroy();
    sampleBuffer.destroy();
    queryBuffer.destroy();
    resultBuffer.destroy();
  }
});
