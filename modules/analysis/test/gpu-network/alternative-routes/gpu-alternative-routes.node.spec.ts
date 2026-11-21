// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUAlternativeRoutes} from '../../../src/gpu-network/alternative-routes/index';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getPrivateMemoryBytes,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import reference from '../../oracles/fixtures/alternative-routes.json';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

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
  const recordedKernelSources1 = getParsedKernelSources(nodes);
  for (const {id: name, source: source, constants} of recordedKernelSources1.configurations) {
    expect(auditWGSLModule(source, constants), name).toEqual([]);
    expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
    expect(getPrivateMemoryBytes(source, constants), name).toBeLessThanOrEqual(8192);
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

it('rejects heap overflow without mutating graph resources and permits a smaller retry', () => {
  const {graph, contributor} = setup(65536);
  const create = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/alternative-routes frontier.*limit/);
  expect(create).not.toHaveBeenCalled();
  expect(() =>
    new GPUAlternativeRoutes({...contributor.props, maximumFrontierSize: 1}).getCommandNodes(graph)
  ).not.toThrow();
});

it('checks uint32 scratch addressing at construction before query products can wrap', () => {
  expect(() => setup(65536, 256, 128)).toThrow(/alternative-routes frontierWords.*4294967295/);
});

it('accepts disjoint aligned output ranges and rejects overlapping writable bindings', () => {
  const {graph, contributor} = setup(1, 2);
  const backing = createTransientView(graph, 'shared-path-storage', 'uint32', 128).buffer;
  const vertices = graph.createDataView(backing, {format: 'uint32', length: 64});
  const edges = graph.createDataView(backing, {format: 'uint32', byteOffset: 256, length: 64});
  const props = {...contributor.props, output: {...contributor.props.output, vertices, edges}};
  expect(() => new GPUAlternativeRoutes(props)).not.toThrow();
  const overlapping = graph.createDataView(backing, {format: 'uint32', byteOffset: 4, length: 64});
  expect(
    () => new GPUAlternativeRoutes({...props, output: {...props.output, edges: overlapping}})
  ).toThrow(/alternative-routes.*overlap/);
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
