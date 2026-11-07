// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUSpatialBalancedSample} from '../../../src/gpu-spatial-analysis/point-process/gpu-spatial-balanced-sample';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('rejects aliasing between sampling status and later-published frame IDs', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const shared = createTransientView(graph, 'shared-output', 'uint32', 2);
  expect(
    () =>
      new GPUSpatialBalancedSample({
        positions: createTransientView(graph, 'positions', 'float32x2', 4),
        frameIds: createTransientView(graph, 'frame-ids', 'uint32', 4),
        allocations: [{stratumId: 0, sampleCount: 2}],
        bounds: [0, 0, 1, 1],
        seed: 1,
        output: {
          frameIds: shared,
          status: shared,
          inclusionProbabilities: createTransientView(graph, 'probabilities', 'float32', 2),
          designWeights: createTransientView(graph, 'weights', 'float32', 2)
        }
      })
  ).toThrow();
});
