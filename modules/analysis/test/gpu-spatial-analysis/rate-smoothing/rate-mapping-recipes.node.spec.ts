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
import {computeEmpiricalBayesOracle} from './empirical-bayes-oracle';
import {computeSpatialEmpiricalBayesOracle} from './spatial-empirical-bayes-oracle';

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
  expect(risk.status.stages.map(entry => entry.stage)).toEqual(['excess-risk']);
  expect(risk.status.stages[0].status.invalidCount?.length).toBe(1);
  expect(spatial.status.stages.map(entry => entry.stage)).toEqual(['weights']);
  expect(spatial.status.stages[0].status).toMatchObject(weights.status);
  expect(spatial.status.stages[0].status.overflow).toBe(weights.status.overflow);
  expect(spatial.status.stages[0].status.candidateOverflow?.format).toBe('uint32');
  expect(spatial.status.stages[0].status.candidateOverflow?.length).toBe(1);
  // Review S1: a final-only producer gets a generated candidate word, never its final view.
  expect(spatial.status.stages[0].status.candidateOverflow).not.toBe(weights.status.overflow);
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
  expect(sources.length).toBe(7);
  // Exact uint32 population counting adds a reduction and its scalar publication.
  const counts = sources.filter(kernel => kernel.id.includes('-exact-count-'));
  expect(counts).toHaveLength(2);
  for (const count of counts) {
    expect(count.source).toContain('array<u32>');
    expect(count.source).not.toContain('array<f32>');
  }
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

it('W5-15 supplied final and candidate weights diagnostics preserve singleton identity', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const overflow = createTransientView(graph, 'final-loss', 'uint32', 1);
    const candidateOverflow = createTransientView(graph, 'candidate-loss', 'uint32', 1);
    const status = {overflow, candidateOverflow};
    const result = addSpatialRateRecipe(graph, {
      events: createTransientView(graph, 'events', 'float32', 2),
      populations: createTransientView(graph, 'populations', 'float32', 2),
      weights: {
        offsets: createTransientView(graph, 'offsets', 'uint32', 3),
        neighbors: createTransientView(graph, 'neighbors', 'uint32', 2),
        weights: createTransientView(graph, 'weights', 'float32', 2),
        status
      }
    });
    expect(result.status.stages.map(stage => stage.stage)).toEqual(['weights']);
    expect(result.status.stages[0].status.overflow).toBe(overflow);
    expect(result.status.stages[0].status.candidateOverflow).toBe(candidateOverflow);
    expect(result.outputs.spatialRates.length).toBe(2);
  } finally {
    device.destroy();
  }
});

it('W5-19 negative events are excluded at the focus and donors while zero remains eligible', () => {
  const global = computeEmpiricalBayesOracle({events: [6, -2], populations: [30, 10]});
  // The sole eligible row has 6/30=.2. Pooling the negative sentinel would give 4/40=.1.
  expect(global.count).toBe(1);
  expect(global.pooledRate).toBe(0.2);
  expect(global.raw).toEqual([0.2, NaN]);
  const weights = {offsets: [0, 1, 2], neighbors: [1, 0], weights: [1, 1]};
  const local = computeSpatialEmpiricalBayesOracle({
    events: [6, -2],
    populations: [30, 10],
    ...weights
  });
  expect(local.spatialRate).toEqual([0.2, NaN]);
  expect(
    computeSpatialEmpiricalBayesOracle({events: [0, 2], populations: [30, 10], ...weights})
      .spatialRate
  ).toEqual([0.05, 0.05]);
  expect(
    computeSpatialEmpiricalBayesOracle({events: [0, 0], populations: [30, 10], ...weights})
      .spatialRate
  ).toEqual([0, 0]);
});
