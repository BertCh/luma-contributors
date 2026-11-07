// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUEmpiricalBayesRates} from '../../../src/gpu-spatial-analysis/rate-smoothing/index';
import {
  addExcessRiskRecipe,
  addSpatialRateRecipe
} from '../../../src/gpu-spatial-analysis/recipes/rate-mapping-recipes';
import {
  getGPUCommandNodeKernelSources,
  getStorageBindingCount,
  createNullWebGPUDevice
} from '../../utils/gpu-contributor-test-utils';

it('rate mapping composes existing kernels within core binding limits', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const events = createTransientView(graph, 'events', 'uint32', 4);
  const populations = createTransientView(graph, 'populations', 'uint32', 4);
  const mask = createTransientView(graph, 'mask', 'uint32', 4);
  const weights = {
    offsets: createTransientView(graph, 'offsets', 'uint32', 5),
    neighbors: createTransientView(graph, 'neighbors', 'uint32', 6),
    weights: createTransientView(graph, 'weights', 'float32', 6),
    status: {overflow: createTransientView(graph, 'weights-overflow', 'uint32', 1)}
  };
  const risk = addExcessRiskRecipe(graph, {events, populations, mask});
  const spatial = addSpatialRateRecipe(graph, {events, populations, mask, weights, scale: 100000});
  expect(risk.status.stages).toEqual([]);
  expect(spatial.status.stages).toEqual([{stage: 'weights', status: weights.status}]);
  expect(spatial.status.stages[0].status.overflow).toBe(weights.status.overflow);
  expect(risk.outputs.excessRisk.length).toBe(4);
  expect(spatial.outputs.spatialRates.length).toBe(4);
  const separate = new GPUCommandGraph(createNullWebGPUDevice());
  const values = createTransientView(separate, 'v', 'float32', 4);
  const result = new GPUEmpiricalBayesRates({
    events: values,
    populations: values,
    output: {excessRisk: createTransientView(separate, 'risk', 'float32', 4)}
  });
  const sources = getGPUCommandNodeKernelSources(result.getCommandNodes(separate));
  expect(sources.length).toBe(5);
  for (const {source} of sources) expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
  expect(() =>
    addSpatialRateRecipe(graph, {id: 'bad', events, populations, weights, scale: Infinity})
  ).toThrow(/scale/);
  expect(() =>
    addSpatialRateRecipe(graph, {
      id: 'alias',
      events,
      populations,
      weights,
      output: weights.weights
    })
  ).toThrow();
});

it('excess risk alongside every existing EB output fits eight storage buffers', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const createFloat = (name: string, length = 4) =>
    createTransientView(graph, name, 'float32', length);
  const contributor = new GPUEmpiricalBayesRates({
    events: createFloat('events'),
    populations: createFloat('populations'),
    mask: createTransientView(graph, 'mask', 'uint32', 4),
    output: {
      excessRisk: createFloat('risk'),
      rawRates: createFloat('raw'),
      smoothedRates: createFloat('smoothed'),
      standardizedRates: createFloat('standardized'),
      summary: createFloat('summary', 8)
    }
  });
  for (const {source} of getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))) {
    expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
  }
});
