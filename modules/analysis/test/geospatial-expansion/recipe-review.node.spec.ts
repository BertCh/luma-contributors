// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';
import {
  addFacilityScenarioRecipe,
  addNetworkDisruptionRecipe
} from '../../src/gpu-network/recipes/network-scenarios-recipe';
import {addTrackQualityRecipe} from '../../src/gpu-network/recipes/track-quality-recipe';

it('scenario outputs cannot overwrite retained baselines or inputs used by other stages', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `review-${serial++}`, format, length);
  const baseline = view('float32', 3);
  expect(() =>
    addNetworkDisruptionRecipe(graph, {
      search: {
        topology: {
          offsets: view('uint32', 4),
          neighbors: view('uint32', 2),
          weights: view('float32', 2)
        },
        startNodes: view('uint32', 1)
      },
      rowCount: 1,
      baseline: {costs: baseline, status: {converged: view('uint32', 1)}},
      opportunityWeights: view('float32', 3),
      parameters: view('float32', 4),
      outputs: {scenarioCosts: baseline}
    })
  ).toThrow(/^network-disruption-recipe outputs must not share a buffer$/);
  const supply = view('float32', 1);
  expect(() =>
    addFacilityScenarioRecipe(graph, {
      costs: view('float32', 3),
      demand: view('float32', 3),
      supply,
      facilityMask: view('uint32', 1),
      parameters: view('float32', 4),
      outputs: {coveredDemand: supply}
    })
  ).toThrow(/share buffers/);
});

it('track recipe accepts its advertised Frechet epsilon option and publishes the decision', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `track-review-${serial++}`, format, length);
  const result = addTrackQualityRecipe(graph, {
    trajectory: {
      positions: view('float32x2', 3),
      timestamps: view('float32', 3),
      trackOffsets: view('uint32', 2)
    },
    thresholds: view('float32', 2),
    similarity: {
      pairA: view('uint32', 1),
      pairB: view('uint32', 1),
      frechetEpsilon: view('float32', 1)
    }
  });
  expect(result.outputs.similarity?.frechetWithin?.length).toBe(1);
});
