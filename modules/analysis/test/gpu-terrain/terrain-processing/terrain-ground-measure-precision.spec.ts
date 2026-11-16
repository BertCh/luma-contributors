// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  getTerrainGroundMeasureWGSL,
  terrainBinding
} from '../../../src/gpu-terrain/terrain-processing/terrain-processing-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const tallGrid of [false, true]) {
  it(`preserves neighbouring cell distances above 2^24 ${tallGrid ? 'rows' : 'columns'}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // Only four index pairs are resident: the precision regression needs no giant raster buffer.
    const width = tallGrid ? 2 : 16777218;
    const height = tallGrid ? 16777218 : 2;
    const first = tallGrid ? 16777216 * width : 16777216;
    const forward = first + (tallGrid ? width : 1);
    const diagonal = forward + (tallGrid ? 1 : width);
    const pairs = Uint32Array.from([
      first,
      forward,
      forward,
      first,
      first,
      diagonal,
      diagonal,
      first
    ]);
    const inputBuffer = createInputBuffer(device, pairs);
    const outputBuffer = createOutputBuffer(device, 4);
    const graph = new GPUCommandGraph(device);
    const pairView = importGraphBuffer(graph, 'pairs', inputBuffer, 'uint32', pairs.length);
    const distances = importGraphBuffer(graph, 'distances', outputBuffer, 'float32', 4);
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'sparse-distances',
        operation: 'TerrainGroundMeasurePrecision',
        invocationCount: 4,
        bindings: [
          terrainBinding('pairs', pairView),
          terrainBinding('distances', distances, 'read_write')
        ],
        declarations: `const GRID_WIDTH: u32 = ${width}u;\nconst GRID_HEIGHT: u32 = ${height}u;\n${getTerrainGroundMeasureWGSL({width, height, affine: [3, 4, 1e12, 0, 2, 1e12], pixelInterpretation: 'area'})}`,
        body: 'distances[distancesOffset + index] = getGroundEdgeLength(pairs[pairsOffset + index * 2u], pairs[pairsOffset + index * 2u + 1u]);'
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const actual = await readFloat32(outputBuffer, 4);
      // Exact integer coordinate differences transformed by the affine, independent of GPU code.
      const straight = tallGrid ? Math.hypot(4, 2) : 3;
      const expected = [straight, straight, Math.hypot(3 + 4, 2), Math.hypot(3 + 4, 2)];
      actual.forEach((value, index) => expect(value).toBeCloseTo(expected[index], 5));
    } finally {
      compiled.destroy();
      inputBuffer.destroy();
      outputBuffer.destroy();
    }
  });
}
