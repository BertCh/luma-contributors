// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUMultivariateLocalGearyPermutationTest} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-multivariate-local-geary-permutation-test';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function makeContributor(
  rows: number,
  options: Partial<ConstructorParameters<typeof GPUMultivariateLocalGearyPermutationTest>[0]> = {}
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `tiled-geary-${serial++}`, format, length);
  const contributor = new GPUMultivariateLocalGearyPermutationTest({
    values: view('float32', rows * 3),
    variableCount: 3,
    permutationCount: 500,
    weights: {
      offsets: view('uint32', rows + 1),
      neighbors: view('uint32', rows * 4),
      weights: view('float32', rows * 4)
    },
    output: {
      statistic: view('float32', rows),
      pseudoPValues: view('float32', rows),
      classes: view('uint32', rows)
    },
    ...options
  });
  return {device, graph, contributor, view};
}

it('multivariate Geary selects the tiled strategy above 256 rows or 4096 permutations', () => {
  const small = makeContributor(256);
  const large = makeContributor(257);
  const many = makeContributor(10, {permutationCount: 9999});
  try {
    expect(small.contributor.getStrategy()).toBe('exact');
    expect(large.contributor.getStrategy()).toBe('tiled');
    expect(many.contributor.getStrategy()).toBe('tiled');
  } finally {
    for (const context of [small, large, many]) context.device.destroy();
  }
});

it('multivariate Geary tiled strategy tiles rows and permutations', () => {
  const context = makeContributor(1000, {rowTileSize: 400, permutationTileSize: 200});
  try {
    const nodes = context.contributor.getCommandNodes(context.graph);
    const ids = nodes.map(node => node.id);
    // 3 row tiles of observed, 3 x 3 simulation tiles.
    expect(ids.filter(id => id.includes('-observed-'))).toHaveLength(3);
    expect(ids.filter(id => id.includes('-simulate-'))).toHaveLength(9);
    expect(ids.some(id => id.endsWith('-inference'))).toBe(true);
  } finally {
    context.device.destroy();
  }
});

it('multivariate Geary admits tiled Bonferroni near the command node limit', () => {
  const {device, graph, contributor} = makeContributor(255, {
    strategy: 'tiled',
    correction: 'bonferroni',
    permutationCount: 125,
    rowTileSize: 1,
    permutationTileSize: 1
  });
  try {
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.filter(node => node.id.includes('-simulate-'))).toHaveLength(255 * 125);
    expect(nodes).toHaveLength(32149);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    expect(contributor.estimatedCommandNodeCount).toBeLessThanOrEqual(32768);
  } finally {
    device.destroy();
  }
});

it('multivariate Geary tiled strategy validates capacity and refuses simulations', () => {
  const context = makeContributor(10);
  try {
    expect(() => makeContributor(300, {permutationCount: 10000})).toThrow();
    expect(() => makeContributor(300, {maximumNeighbors: 129})).toThrow();
    expect(() => makeContributor(300, {strategy: 'exact'})).toThrow();
    const {view} = context;
    expect(() =>
      makeContributor(300, {
        output: {
          statistic: view('float32', 300),
          pseudoPValues: view('float32', 300),
          classes: view('uint32', 300),
          simulations: view('float32', 300 * 500)
        }
      })
    ).toThrow();
  } finally {
    context.device.destroy();
  }
});
