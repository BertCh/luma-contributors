// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, expect, it} from 'vitest';
import {
  GPULocalKriging,
  GPU_LOCAL_KRIGING_MAXIMUM_NEIGHBOR_COUNT,
  GPU_LOCAL_KRIGING_MAXIMUM_SUPPORT_COUNT
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {
  GPULocalRBF,
  GPU_LOCAL_RBF_MAXIMUM_NEIGHBOR_COUNT
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {expectValidQueryShaderLiterals} from '../linear-referencing/gpu-linear-referencing-chunks.test-utils';

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());

function createViews(graph: GPUCommandGraph, supportCount = 1) {
  return {
    positions: createTransientView(graph, 'positions', 'float32x2', 32),
    values: createTransientView(graph, 'values', 'float32', 32),
    observations: createTransientView(graph, 'observations', 'float32x4', 32),
    queries: createTransientView(graph, 'queries', 'float32x2', 1),
    supports: createTransientView(graph, 'supports', 'float32x2', supportCount),
    output: {
      values: createTransientView(graph, 'results', 'float32x4', 1),
      status: createTransientView(graph, 'status', 'uint32', 1)
    }
  };
}

it('names the bounded RBF neighbor limit at construction', () => {
  const graph = new GPUCommandGraph(device);
  const views = createViews(graph);
  for (const neighborCount of [0, 1.5, GPU_LOCAL_RBF_MAXIMUM_NEIGHBOR_COUNT + 1]) {
    expect(() => new GPULocalRBF({...views, id: 'rbf-limit', neighborCount})).toThrow(
      'rbf-limit neighborCount must be an integer in [1, 32]'
    );
  }
  for (const kernel of ['thin-plate-spline', 'cubic', 'quintic', 'gaussian'] as const) {
    const contributor = new GPULocalRBF({
      ...views,
      kernel,
      neighborCount: GPU_LOCAL_RBF_MAXIMUM_NEIGHBOR_COUNT
    });
    for (const {source} of getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))) {
      expectValidQueryShaderLiterals(source);
      const identifiers = source.replace(/\/\/[^\n]*/g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
      expect(identifiers.filter(identifier => WGSL_RESERVED_WORDS.has(identifier))).toEqual([]);
    }
  }
});

it('names the bounded local kriging neighbor and support limits at construction', () => {
  const graph = new GPUCommandGraph(device);
  const props = {
    ...createViews(graph, GPU_LOCAL_KRIGING_MAXIMUM_SUPPORT_COUNT),
    id: 'kriging-limit',
    mode: 'ordinary' as const,
    covariance: {model: 'exponential' as const, sill: 1, range: 2},
    supportCount: GPU_LOCAL_KRIGING_MAXIMUM_SUPPORT_COUNT,
    neighborCount: GPU_LOCAL_KRIGING_MAXIMUM_NEIGHBOR_COUNT
  };
  expect(new GPULocalKriging(props).getCommandNodes(graph).length).toBeGreaterThan(0);
  expect(() => new GPULocalKriging({...props, neighborCount: 33})).toThrow(
    'kriging-limit neighborCount must be an integer in [1, 32]'
  );
  expect(() => new GPULocalKriging({...props, supportCount: 65})).toThrow(
    'kriging-limit supportCount must be an integer in [1, 64]'
  );
});
