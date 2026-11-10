// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPU_SHAPE_DESCRIPTORS_MONOTONE_CHAIN_LARGEST_VERTICES,
  GPUShapeDescriptors
} from '../../../src/gpu-spatial-analysis/shape-descriptors/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('review D-19: orientation guards atan2(0, 0) and keeps the kernel parseable', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'shape-descriptors'});
  const contributor = new GPUShapeDescriptors({
    positions: createTransientView(graph, 'positions', 'float32x2', 8),
    ringOffsets: createTransientView(graph, 'ring-offsets', 'uint32', 3),
    polygonOffsets: createTransientView(graph, 'polygon-offsets', 'uint32', 3),
    featureOffsets: createTransientView(graph, 'feature-offsets', 'uint32', 3),
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    output: {orientation: createTransientView(graph, 'orientation', 'float32', 2)}
  });
  const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
  const source = sources
    .map(entry => entry.source)
    .find(text => text.includes('orientation[orientationOffset'));
  expect(source).toBeDefined();
  expect(() => new WgslReflect(source!)).not.toThrow();
  expect(source).toContain(
    'select(0.0, 0.5 * atan2(2.0 * xy, xx - yy), (xy != 0.0) || (xx != yy))'
  );
});

it("convexityMethod 'auto' decides from the declared largest feature when given", () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'shape-descriptors-auto'});
  let serial = 0;
  const create = (overrides: {maximumVerticesPerFeature?: number; convexityMethod?: never}) =>
    new GPUShapeDescriptors({
      // Ten features of eight vertices: the average is far below the average-size threshold.
      positions: createTransientView(graph, `positions-${serial++}`, 'float32x2', 80),
      ringOffsets: createTransientView(graph, `ring-offsets-${serial++}`, 'uint32', 11),
      polygonOffsets: createTransientView(graph, `polygon-offsets-${serial++}`, 'uint32', 11),
      featureOffsets: createTransientView(graph, `feature-offsets-${serial++}`, 'uint32', 11),
      parameters: createTransientView(graph, `parameters-${serial++}`, 'float32', 4),
      output: {convexity: createTransientView(graph, `convexity-${serial++}`, 'float32', 10)},
      ...overrides
    });
  // Without a declaration the average rule applies, as before.
  expect(create({}).useMonotoneChainConvexity).toBe(false);
  expect(create({maximumVerticesPerFeature: 8}).useMonotoneChainConvexity).toBe(false);
  expect(
    create({maximumVerticesPerFeature: GPU_SHAPE_DESCRIPTORS_MONOTONE_CHAIN_LARGEST_VERTICES - 1})
      .useMonotoneChainConvexity
  ).toBe(false);
  expect(
    create({maximumVerticesPerFeature: GPU_SHAPE_DESCRIPTORS_MONOTONE_CHAIN_LARGEST_VERTICES})
      .useMonotoneChainConvexity
  ).toBe(true);
  // An explicit method ignores the declaration.
  expect(
    create({maximumVerticesPerFeature: 200000, convexityMethod: 'gift-wrapping' as never})
      .useMonotoneChainConvexity
  ).toBe(false);
  expect(() => create({maximumVerticesPerFeature: 0})).toThrow(/maximumVerticesPerFeature/);
  expect(() => create({maximumVerticesPerFeature: 1.5})).toThrow(/maximumVerticesPerFeature/);
});
