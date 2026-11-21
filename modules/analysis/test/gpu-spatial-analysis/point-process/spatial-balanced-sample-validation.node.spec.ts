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

it('DEF-14 balanced sampling accepts resident-only seed and keeps its source explicit', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = {
    positions: createTransientView(graph, 'resident-positions', 'float32x2', 2),
    frameIds: createTransientView(graph, 'resident-frame', 'uint32', 2),
    allocations: [{stratumId: 0, sampleCount: 1}],
    bounds: [0, 0, 1, 1] as const,
    parameters: createTransientView(graph, 'resident-parameters', 'uint32', 2),
    output: {
      frameIds: createTransientView(graph, 'resident-result', 'uint32', 1),
      inclusionProbabilities: createTransientView(graph, 'resident-probability', 'float32', 1),
      designWeights: createTransientView(graph, 'resident-weight', 'float32', 1),
      status: createTransientView(graph, 'resident-status', 'uint32', 2)
    }
  };
  for (const seed of [undefined, 3, 9]) {
    const sample = new GPUSpatialBalancedSample({
      ...props,
      seed,
      id: `resident-seed-${seed ?? 'view'}`
    });
    expect(() => sample.getCommandNodes(graph)).not.toThrow();
    expect(sample.provenance.effectiveSeedSource).toBe('resident-parameters');
    expect([...sample.getReplayParameters(Uint32Array.of(17, 0))]).toEqual([17, 0]);
    expect(() => sample.getReplayParameters()).toThrow(/parameters/);
  }
  expect(() => new GPUSpatialBalancedSample({...props, parameters: undefined})).toThrow(/seed/);
});
