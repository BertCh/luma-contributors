// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPULineIntegralConvolution} from '../../../src/raster-analysis/flow-texture/gpu-line-integral-convolution';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('RT-124 LIC requires landing velocity support before accumulating noise', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  recordWGSLKernelNodes(graph);
  new GPULineIntegralConvolution({
    id: 'landing',
    velocities: createTransientView(graph, 'velocities', 'float32x2', 1),
    fieldWidth: 1,
    fieldHeight: 1,
    grid: makeGPURasterMetadata({width: 2, height: 1}),
    stepCount: 1,
    parameters: createTransientView(graph, 'parameters', 'float32', 12),
    uint32Parameters: createTransientView(graph, 'words', 'uint32', 4),
    output: {values: createTransientView(graph, 'values', 'float32', 2)}
  }).getCommandNodes(graph);
  const body = getWGSLKernelNodeRecord(graph, 'landing-convolve')!.body;
  // One field cell, noise [0,1]: forward .5 -> .875 -> 1.25; backward midpoint .125
  // is supported but landing -.25 is not. Hann weights are 1 at centre, 1/2 at step 1.
  // Old source therefore predicts (0 + .5*1)/(1+.5) = 1/3; supported-only truth is 0.
  expect(body).toMatch(
    /position = position \+ getStepVector\(middle.xy, cell, stepLength\);\s*if \(getDirection\(position, fieldExtent, directionSign, minimumSpeed\)\.z == 0\.0\) \{\s*break;\s*\}/
  );
  expect(
    body.indexOf('getDirection(position, fieldExtent, directionSign, minimumSpeed).z')
  ).toBeLessThan(body.indexOf('let pixel ='));
});
