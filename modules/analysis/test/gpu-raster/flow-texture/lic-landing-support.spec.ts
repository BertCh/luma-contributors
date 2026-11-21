// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPULineIntegralConvolution} from '../../../src/raster-analysis/flow-texture/gpu-line-integral-convolution';
import {getGPULineIntegralConvolutionParameterValues} from '../../../src/raster-analysis/flow-texture/line-integral-convolution-parameters';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GraphHarness} from '../../gpu-spatial-analysis/trajectory-staypoints/movement-spec-utils';
import {readFloat32} from '../../utils/gpu-contributor-test-utils';
import {expectLICValues} from './flow-output-assertions';

for (const fieldWidth of [1, 2]) {
  for (const cellWidth of [1, -1]) {
    it(`RT-124 LIC rejects unsupported landing noise: field ${fieldWidth}, spacing ${cellWidth}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = new GraphHarness(device, 'lic-landing');
      recordWGSLKernelNodes(harness.graph);
      const values = harness.output('values', 'float32', 2);
      const originX = cellWidth === 1 ? 0 : 2;
      const nodes = new GPULineIntegralConvolution({
        id: 'landing',
        velocities: harness.input(
          'field',
          Float32Array.from({length: fieldWidth * 2}, (_, component) =>
            component % 2 === 0 ? cellWidth : 0
          ),
          'float32x2',
          fieldWidth
        ),
        fieldWidth,
        fieldHeight: 1,
        grid: makeGPURasterMetadata({width: 2, height: 1}),
        stepCount: 1,
        parameters: harness.input(
          'parameters',
          getGPULineIntegralConvolutionParameterValues({
            fieldPlacement: {origin: [originX, 0], cellSize: [cellWidth, 1]},
            outputPlacement: {origin: [originX, 0], cellSize: [cellWidth, 1]},
            stepLength: 0.75
          }),
          'float32',
          12
        ),
        uint32Parameters: harness.input('words', new Uint32Array(4), 'uint32', 4),
        output: {values: values.view}
      }).getCommandNodes(harness.graph);
      // Replace only random noise generation. Exercise the unchanged production convolution
      // with literal noise [0,1], making the former 1/3 result independent of Philox.
      const noiseBinding = getWGSLKernelNodeRecord(harness.graph, 'landing-noise')!.bindings.find(
        binding => binding.name === 'noise'
      )!;
      harness.graph.add(
        createWGSLKernelNode(harness.graph, {
          id: 'fixed-noise',
          operation: 'LICLandingFixture',
          invocationCount: 2,
          bindings: [noiseBinding],
          body: 'noise[noiseOffset + index] = f32(index);'
        })
      );
      for (const node of nodes) if (node.id !== 'landing-noise') harness.graph.add(node);
      await harness.run(async () => {
        const result = await readFloat32(values.buffer, 2);
        // One cell: both landings leave velocity support, so centre noise 0 is all that remains.
        // Two cells: forward landing reaches noise 1; Hann weights 1 and 1/2 give 1/3.
        expectLICValues(result, fieldWidth);
      });
    });
  }
}

it('B2 LIC excludes a supported landing whose speed is below minimumSpeed', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new GraphHarness(device, 'lic-threshold');
  const values = harness.output('values', 'float32', 4);
  harness.graph.add(
    new GPULineIntegralConvolution({
      velocities: harness.input('field', Float32Array.of(1, 0, 1, 0, 0, 0, 0, 0), 'float32x2', 4),
      fieldWidth: 4,
      fieldHeight: 1,
      grid: makeGPURasterMetadata({width: 4, height: 1}),
      stepCount: 1,
      parameters: harness.input(
        'parameters',
        getGPULineIntegralConvolutionParameterValues({
          fieldPlacement: {origin: [0, 0], cellSize: [1, 1]},
          outputPlacement: {origin: [0, 0], cellSize: [1, 1]},
          stepLength: 1.375,
          minimumSpeed: 0.75
        }),
        'float32',
        12
      ),
      uint32Parameters: harness.input('words', Uint32Array.of(17, 0, 0, 0), 'uint32', 4),
      output: {values: values.view}
    })
  );
  await harness.run(async () => {
    const result = await readFloat32(values.buffer, 4);
    // Centre .5, midpoint 1.1875 (speed 1), landing 1.875 (speed .625 < .75).
    // Backward midpoint -.1875 is outside. Only Philox(seed17) centre noise survives.
    expect(Number.isFinite(result[0])).toBe(true);
    expect(result[0]).toBeCloseTo(0.782850980758667, 6);
  });
});
