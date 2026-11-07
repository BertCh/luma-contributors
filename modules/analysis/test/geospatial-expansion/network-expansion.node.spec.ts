// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {addCellFlowTopologyRecipe} from '../../src/gpu-spatial-analysis/recipes/cell-flow-topology-recipe';
import {GPUNetworkStructure} from '../../src/gpu-network/network-structure/index';
import {GPUAlternativeRoutes} from '../../src/gpu-network/alternative-routes/index';
import {GPUH3CellIdentities} from '../../src/gpu-spatial-analysis/cell-topology/gpu-h3-cell-identities';
import {GPUBoundedFacilityAllocation} from '../../src/gpu-spatial-analysis/location-allocation/gpu-bounded-facility-allocation';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount,
  getPrivateMemoryBytes,
  CORE_PRIVATE_MEMORY_LIMIT_BYTES
} from '../utils/gpu-contributor-test-utils';

it('network expansion kernels parse and stay within eight storage bindings', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let next = 0;
  const view = <Format extends 'uint32' | 'float32' | 'uint32x2'>(format: Format, length: number) =>
    createTransientView(graph, `view-${next++}`, format, length);
  const topology = {
    offsets: view('uint32', 5),
    neighbors: view('uint32', 6),
    weights: view('float32', 6)
  };
  const contributors = [
    new GPUNetworkStructure({
      topology,
      reverseEdgeSlots: view('uint32', 6),
      output: {
        strongComponents: view('uint32', 4),
        bridges: view('uint32', 6),
        articulationPoints: view('uint32', 4),
        biconnectedComponents: view('uint32', 6)
      }
    }),
    new GPUAlternativeRoutes({
      topology,
      sources: view('uint32', 2),
      targets: view('uint32', 2),
      routeCount: 3,
      output: {
        vertices: view('uint32', 24),
        edges: view('uint32', 24),
        lengths: view('uint32', 6),
        costs: view('float32', 6),
        counts: view('uint32', 2),
        complete: view('uint32', 2)
      }
    }),
    new GPUH3CellIdentities({
      cells: view('uint32x2', 3),
      operation: 'vertices',
      output: view('uint32x2', 18)
    }),
    new GPUBoundedFacilityAllocation({
      operation: 'capacitated-p-median',
      facilityCount: 3,
      selectCount: 2,
      costs: view('float32', 9),
      demand: view('float32', 3),
      capacities: view('float32', 3),
      output: {
        selectedFacilities: view('uint32', 3),
        assignments: view('uint32', 3),
        objective: view('float32', 1),
        status: view('uint32', 3)
      }
    })
  ];
  const recipe = addCellFlowTopologyRecipe(graph, {
    cells: view('uint32x2', 3),
    origins: view('uint32', 5),
    destinations: view('uint32', 5),
    values: view('float32', 5)
  });
  for (const contributor of [...contributors, ...recipe.contributors]) {
    const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
    expect(sources.size).toBeGreaterThan(0);
    for (const [name, source] of sources) {
      expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
      expect(getPrivateMemoryBytes(source), name).toBeLessThanOrEqual(
        CORE_PRIVATE_MEMORY_LIMIT_BYTES
      );
    }
  }
});

it('bounded graph traversal scratch fits the core private-memory budget at maximum size', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let next = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `maximum-${next++}`, format, length);
  const topology = {offsets: view('uint32', 1025), neighbors: view('uint32', 2048)};
  const structure = new GPUNetworkStructure({
    topology,
    reverseEdgeSlots: view('uint32', 2048),
    output: {strongComponents: view('uint32', 1024), bridges: view('uint32', 2048)}
  });
  const routes = new GPUAlternativeRoutes({
    topology: {
      offsets: view('uint32', 257),
      neighbors: view('uint32', 1024),
      weights: view('float32', 1024)
    },
    sources: view('uint32', 1),
    targets: view('uint32', 1),
    routeCount: 32,
    output: {
      vertices: view('uint32', 8192),
      edges: view('uint32', 8192),
      lengths: view('uint32', 32),
      costs: view('float32', 32),
      counts: view('uint32', 1),
      complete: view('uint32', 1)
    }
  });
  for (const contributor of [structure, routes])
    for (const [name, source] of getParsedKernelSources(contributor.getCommandNodes(graph))) {
      expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
      expect(getPrivateMemoryBytes(source), name).toBeLessThanOrEqual(
        CORE_PRIVATE_MEMORY_LIMIT_BYTES
      );
    }
});

it('bounded exact network algorithms reject graphs above their documented node caps', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let next = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `cap-view-${next++}`, format, length);
  const createTopology = (nodeCount: number) => ({
    offsets: view('uint32', nodeCount + 1),
    neighbors: view('uint32', 2),
    weights: view('float32', 2)
  });

  expect(
    () =>
      new GPUNetworkStructure({
        topology: createTopology(1025),
        strongComponentsMethod: 'reachability',
        output: {strongComponents: view('uint32', 1025)}
      }),
    'GPUNetworkStructure reachability strong components are bounded to 1024 nodes'
  ).toThrow();
  expect(
    () =>
      new GPUNetworkStructure({
        topology: createTopology(1024),
        output: {strongComponents: view('uint32', 1024)}
      }),
    'GPUNetworkStructure accepts exactly 1024 nodes'
  ).not.toThrow();
  expect(
    () =>
      new GPUAlternativeRoutes({
        topology: createTopology(257),
        sources: view('uint32', 1),
        targets: view('uint32', 1),
        routeCount: 1,
        output: {
          vertices: view('uint32', 257),
          edges: view('uint32', 257),
          lengths: view('uint32', 1),
          costs: view('float32', 1),
          counts: view('uint32', 1),
          complete: view('uint32', 1)
        }
      }),
    'GPUAlternativeRoutes is bounded to 256 nodes'
  ).toThrow();
});
