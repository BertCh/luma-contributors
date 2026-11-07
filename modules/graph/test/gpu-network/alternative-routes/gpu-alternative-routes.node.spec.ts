// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUAlternativeRoutes} from '../../../src/gpu-network/alternative-routes';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getPrivateMemoryBytes,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import reference from '../../oracles/fixtures/alternative-routes.json';

function setup(maximumFrontierSize = 1024, nodeCount = 256, queryCount = 1) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let next = 0;
  const view = <Format extends 'uint32' | 'float32'>(length: number, format: Format) =>
    createTransientView(graph, `view-${next++}`, format, length);
  const contributor = new GPUAlternativeRoutes({
    maximumFrontierSize,
    topology: {
      offsets: view(nodeCount + 1, 'uint32'),
      neighbors: view(1024, 'uint32'),
      weights: view(1024, 'float32')
    },
    sources: view(queryCount, 'uint32'),
    targets: view(queryCount, 'uint32'),
    routeCount: 32,
    output: {
      vertices: view(queryCount * nodeCount * 32, 'uint32'),
      edges: view(queryCount * nodeCount * 32, 'uint32'),
      lengths: view(queryCount * 32, 'uint32'),
      costs: view(queryCount * 32, 'float32'),
      counts: view(queryCount, 'uint32'),
      complete: view(queryCount, 'uint32')
    }
  });
  return {graph, contributor};
}

it('GPUAlternativeRoutes heap kernels fit core bindings and private memory at maximum nodes/routes', () => {
  const {graph, contributor} = setup(65536, 128);
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.map(node => node.id)).toEqual([
    'alternative-routes-queries',
    'alternative-routes-search',
    'alternative-routes-publish'
  ]);
  for (const [name, source] of getParsedKernelSources(nodes)) {
    expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
    expect(getPrivateMemoryBytes(source), name).toBeLessThanOrEqual(8192);
  }
  const maximal = setup();
  expect(() =>
    getParsedKernelSources(maximal.contributor.getCommandNodes(maximal.graph))
  ).not.toThrow();
});

it('GPUAlternativeRoutes rejects invalid capacities and device binding overflow', () => {
  for (const capacity of [0, -1, 1.5, 65537]) {
    expect(() => setup(capacity)).toThrow();
  }
  const {graph, contributor} = setup(65536);
  expect(() => contributor.getCommandNodes(graph)).toThrow();
});

it('NetworkX edge-route oracle includes distinct multiedges, cycles, deterministic ties and identity', () => {
  expect(reference.version).toBe('3.5');
  const fixture = reference.cases[0];
  expect(fixture.routes.length).toBeGreaterThan(3);
  for (const route of fixture.routes) {
    expect(new Set(route.vertices).size).toBe(route.vertices.length);
    expect(route.edges.length).toBe(route.vertices.length - 1);
  }
  expect(reference.cases[2].routes).toEqual([{edges: [], vertices: [0], cost: 0}]);
  expect(reference.cases[3].routes).toEqual([]);
});
