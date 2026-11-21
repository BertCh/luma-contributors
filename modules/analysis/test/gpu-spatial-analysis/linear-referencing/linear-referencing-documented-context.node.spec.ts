// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPULinearReferencing} from '../../../src/gpu-spatial-analysis/linear-referencing/gpu-linear-referencing';

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());

function createReferencingProps(graph: GPUCommandGraph) {
  return {
    positions: createTransientView(graph, 'positions', 'float32x2', 2),
    lineOffsets: createTransientView(graph, 'line-offsets', 'uint32', 2),
    radius: createTransientView(graph, 'radius', 'float32', 1),
    queryPositions: createTransientView(graph, 'queries', 'float32x2', 1),
    output: {
      measures: createTransientView(graph, 'measures', 'float32', 1),
      status: {candidateOverflow: createTransientView(graph, 'candidate-overflow', 'uint32', 1)}
    }
  };
}

it('selects great-circle measurement through the documented spatialContext', () => {
  const graph = new GPUCommandGraph(device);
  // The cross-reference option: no planar candidate capacity is needed for a great-circle context.
  const referencing = new GPULinearReferencing({
    ...createReferencingProps(graph),
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
  });
  expect(referencing.spatialContext.coordinateSpace).toBe('longitude-latitude');
  expect(referencing.spatialContext.metric).toBe('great-circle');
});

it('documents no retired coordinateSystem option for linear referencing or geometry measures', () => {
  // `coordinateSystem` is not a prop of GPULinearReferencing, GPULineLocate or GPUGeometryMeasures:
  // a caller following it gets the planar default (and a planar candidateCapacity error).
  const graph = new GPUCommandGraph(device);
  const retired = {...createReferencingProps(graph), coordinateSystem: 'spherical'};
  expect(() => new GPULinearReferencing(retired)).toThrow(/candidateCapacity/);
  const page = readFileSync('docs/api/gpu-spatial-analysis-cross-reference.md', 'utf8');
  expect(page).not.toMatch(/coordinateSystem:/);
});
