// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUConditionalGaussianSimulation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function makeSimulation(observationCount: number, queryCount: number, replicateCount = 1) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let viewIndex = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `simulation-${viewIndex++}`, format, length);
  return {
    device,
    graph,
    props: {
      positions: view('float32x2', observationCount),
      observations: view('float32x4', observationCount),
      queries: view('float32x2', queryCount),
      covariance: {model: 'exponential' as const, range: 1.3, sill: 1},
      seed: 19,
      replicateCount,
      output: {
        values: view('float32', queryCount * replicateCount),
        moments: view('float32x2', queryCount),
        status: view('uint32', replicateCount)
      }
    }
  };
}

it('simulation preserves the cooperative path and declares blocked domains above its old limits', () => {
  for (const [observations, queries, blocked] of [
    [128, 128, false],
    [129, 127, true],
    [127, 129, true],
    [2048, 2048, true]
  ] as const) {
    const fixture = makeSimulation(observations, queries);
    try {
      const nodes = new GPUConditionalGaussianSimulation(fixture.props).getCommandNodes(
        fixture.graph
      );
      expect(nodes.some(node => node.id.endsWith('blocked-realizations'))).toBe(blocked);
      expect(nodes.some(node => node.id.endsWith('cooperative-factor'))).toBe(!blocked);
    } finally {
      fixture.device.destroy();
    }
  }
});

it('simulation rejects unsupported joint domains before allocating retained GPU buffers', () => {
  const fixture = makeSimulation(2048, 2049);
  try {
    expect(() => new GPUConditionalGaussianSimulation(fixture.props)).toThrow(
      'invalid bounded simulation topology'
    );
  } finally {
    fixture.device.destroy();
  }
});

it('simulation checks the dense factor and realization batch against the actual binding limit', () => {
  for (const [observations, queries, replicates, bindingLimit, reason] of [
    [129, 129, 1, 256 * 1024, 'joint covariance factor'],
    [1, 129, 1024, 256 * 1024, 'realization batch']
  ] as const) {
    const fixture = makeSimulation(observations, queries, replicates);
    fixture.device.limits.maxStorageBufferBindingSize = bindingLimit;
    try {
      const contributor = new GPUConditionalGaussianSimulation(fixture.props);
      expect(() => contributor.getCommandNodes(fixture.graph)).toThrow(reason);
    } finally {
      fixture.device.destroy();
    }
  }
});
