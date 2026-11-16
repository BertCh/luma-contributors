// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, expect, it, vi} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  createTransientView,
  getViewBindingRange,
  type GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPUNetworkFlow} from '../../../src/gpu-network/network-optimization/gpu-network-flow';
import {GPUNetworkFacilityOptimization} from '../../../src/gpu-network/network-optimization/gpu-network-facility-optimization';
import {GPUNetworkTimeRouting} from '../../../src/gpu-network/network-optimization/gpu-network-time-routing';
import {GPUNetworkRationalAccess} from '../../../src/gpu-network/network-optimization/gpu-network-rational-access';
import {GPUNetworkLouvain} from '../../../src/gpu-network/network-optimization/gpu-network-louvain';
import {GPUNetworkPointRouting} from '../../../src/gpu-network/network-optimization/gpu-network-point-routing';
import * as kernelNodes from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

afterEach(() => vi.restoreAllMocks());

const ROUTES = [
  'flow-reference',
  'flow-parallel',
  'flow-primal-dual',
  'facility',
  'time-affine',
  'time-piecewise',
  'rational-access',
  'louvain-dense',
  'louvain-sparse',
  'point-reference',
  'point-scalable'
] as const;
type Route = (typeof ROUTES)[number];
type Format = 'uint32' | 'uint32x2' | 'float32' | 'float32x2';

function createFixture(route: Route, flowNodeCount = 4) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const views = new Map<string, GraphDataView>();
  const createContributor = (
    replacements: Record<string, GraphDataView> = {}
  ): GPUCommandNodeProducer => {
    const view = <ValueFormat extends Format>(
      name: string,
      format: ValueFormat,
      length: number
    ): GraphDataView<ValueFormat> => {
      if (!views.has(name))
        views.set(
          name,
          createTransientView(graph, `${route}-${name}`, format, length, Buffer.STORAGE)
        );
      return (replacements[name] ?? views.get(name)) as GraphDataView<ValueFormat>;
    };
    const status = () => ({
      invalidCount: view('output.status.invalidCount', 'uint32', 1),
      iterationCount: view('output.status.iterationCount', 'uint32', 1),
      converged: view('output.status.converged', 'uint32', 1),
      iterationLimitReached: view('output.status.iterationLimitReached', 'uint32', 1)
    });
    if (route.startsWith('flow-'))
      return new GPUNetworkFlow({
        id: route,
        nodeCount: flowNodeCount,
        solver: route === 'flow-reference' ? 'reference' : 'parallel',
        mode: route === 'flow-primal-dual' ? 'primal-dual' : undefined,
        edges: view('edges', 'uint32x2', 3),
        capacities: view('capacities', 'uint32', 3),
        costs: view('costs', 'uint32', 3),
        parameters: view('parameters', 'uint32', 3),
        maximumAugmentations: 1,
        maximumRelaxationRounds: route === 'flow-reference' ? undefined : 1,
        output: {
          flow: view('output.flow', 'uint32', 3),
          cut: view('output.cut', 'uint32', flowNodeCount),
          summary: view('output.summary', 'uint32', 8),
          exactCost: view('output.exactCost', 'uint32x2', 1),
          status: status()
        }
      });
    if (route === 'facility')
      return new GPUNetworkFacilityOptimization({
        id: route,
        operation: 'capacitated-p-median',
        facilityCount: 2,
        selectCount: 1,
        maximumCandidates: 1,
        costs: view('costs', 'float32', 4),
        demand: view('demand', 'float32', 2),
        capacities: view('capacities', 'float32', 2),
        threshold: view('threshold', 'float32', 1),
        output: {
          selectedFacilities: view('output.selectedFacilities', 'uint32', 2),
          assignments: view('output.assignments', 'uint32', 2),
          objective: view('output.objective', 'float32', 1),
          summary: view('output.summary', 'uint32', 4),
          status: status()
        }
      });
    if (route.startsWith('time-'))
      return new GPUNetworkTimeRouting({
        id: route,
        stateVertices: view('stateVertices', 'uint32', 2),
        transitions: view('transitions', 'uint32x2', 1),
        costs: view('costs', 'float32x2', route === 'time-piecewise' ? 2 : 1),
        costModel: route === 'time-piecewise' ? 'piecewise-arrival' : 'affine-duration',
        samplesPerTransition: route === 'time-piecewise' ? 2 : undefined,
        endpoints: view('endpoints', 'uint32', 2),
        departure: view('departure', 'float32', 1),
        output: {
          arrivals: view('output.arrivals', 'float32', 2),
          predecessorTransitions: view('output.predecessorTransitions', 'uint32', 2),
          summary: view('output.summary', 'uint32', 4),
          status: status()
        }
      });
    if (route === 'rational-access')
      return new GPUNetworkRationalAccess({
        id: route,
        costs: view('costs', 'float32', 4),
        demand: view('demand', 'float32', 2),
        supply: view('supply', 'float32', 2),
        parameters: view('parameters', 'float32', 6),
        maximumCycles: 1,
        output: {
          assignments: view('output.assignments', 'float32', 4),
          scores: view('output.scores', 'float32', 2),
          unservedDemand: view('output.unservedDemand', 'float32', 2),
          summary: view('output.summary', 'uint32', 4),
          status: status()
        }
      });
    if (route.startsWith('louvain-'))
      return new GPUNetworkLouvain({
        id: route,
        nodeCount: 2,
        maximumLevels: 1,
        maximumIterations: 1,
        ...(route === 'louvain-dense'
          ? {weights: view('weights', 'float32', 4)}
          : {
              topology: {
                offsets: view('topology.offsets', 'uint32', 3),
                neighbors: view('topology.neighbors', 'uint32', 2),
                weights: view('topology.weights', 'float32', 2),
                nodePositions: view('topology.nodePositions', 'float32x2', 2)
              }
            }),
        parameters: view('parameters', 'float32', 2),
        output: {
          communities: view('output.communities', 'uint32', 2),
          objectiveTrace: view('output.objectiveTrace', 'float32', 2),
          summary: view('output.summary', 'uint32', 4),
          status: status()
        }
      });
    return new GPUNetworkPointRouting({
      id: route,
      mode: route === 'point-reference' ? 'reference' : 'scalable',
      maximumRelaxations: 1,
      turns: view('turns', 'uint32x2', 1),
      costs: view('costs', 'float32', 2),
      queries: view('queries', 'uint32x2', 1),
      fractions: view('fractions', 'float32x2', 1),
      output: {
        costs: view('output.costs', 'float32', 1),
        status: view('output.status', 'uint32', 1),
        predecessorEdges: view('output.predecessorEdges', 'uint32', 2)
      }
    });
  };
  createContributor();
  return {device, graph, views, createContributor};
}

function watchDeclarations(graph: GPUCommandGraph) {
  const declarations = [
    vi.spyOn(graph, 'createTransientBuffer'),
    vi.spyOn(graph, 'createDataView'),
    vi.spyOn(graph, 'importBuffer'),
    vi.spyOn(graph, 'getUniqueResourceId'),
    vi.spyOn(graph, 'add'),
    vi.spyOn(kernelNodes, 'createWGSLKernelNode')
  ];
  return () => {
    for (const declaration of declarations) expect(declaration).not.toHaveBeenCalled();
    for (const declaration of declarations) declaration.mockRestore();
  };
}

for (const route of ROUTES) {
  it(`${route} rejects every aligned caller binding overflow without changing the graph`, () => {
    const fixture = createFixture(route);
    const {device, graph, views, createContributor} = fixture;
    try {
      for (const [name, original] of views) {
        const buffer = graph.createTransientBuffer({
          id: `prefixed-${name}`,
          byteLength: 512,
          usage: Buffer.STORAGE
        });
        const prefixed = graph.createDataView(buffer, {
          format: original.format,
          length: original.length,
          byteOffset: 252
        });
        const contributor = createContributor({[name]: prefixed});
        Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {
          value: 255,
          configurable: true
        });
        const unchanged = watchDeclarations(graph);
        expect(() => contributor.getCommandNodes(graph), name).toThrow(
          /needs 2\d\d bytes.*limit 255/
        );
        unchanged();
      }
    } finally {
      device.destroy();
    }
  });

  it(`${route} rejects missing STORAGE on every caller binding without changing the graph`, () => {
    const {device, graph, views, createContributor} = createFixture(route);
    try {
      for (const [name, original] of views) {
        const buffer = graph.createTransientBuffer({
          id: `no-storage-${name}`,
          byteLength: 128,
          usage: Buffer.COPY_SRC | Buffer.COPY_DST
        });
        const replacement = graph.createDataView(buffer, {
          format: original.format,
          length: original.length
        });
        const contributor = createContributor({[name]: replacement});
        const unchanged = watchDeclarations(graph);
        expect(() => contributor.getCommandNodes(graph), name).toThrow(
          /usage must include Buffer.STORAGE/
        );
        unchanged();
      }
    } finally {
      device.destroy();
    }
  });

  it(`${route} checks STORAGE usage of an imported physical buffer before declaring work`, () => {
    const {device, graph, views, createContributor} = createFixture(route);
    const physical = device.createBuffer({
      byteLength: 128,
      usage: Buffer.COPY_SRC | Buffer.COPY_DST
    });
    try {
      const [name, original] = [...views.entries()][0];
      // One physical buffer, one import; the published import validates matching descriptors.
      const handle = graph.importBuffer(
        {id: 'missing-storage', byteLength: 128, usage: physical.usage},
        physical
      );
      const replacement = graph.createDataView(handle, {
        format: original.format,
        length: original.length
      });
      const contributor = createContributor({[name]: replacement});
      const unchanged = watchDeclarations(graph);
      expect(() => contributor.getCommandNodes(graph)).toThrow(/usage must include Buffer.STORAGE/);
      unchanged();
    } finally {
      physical.destroy();
      device.destroy();
    }
  });

  it(`${route} rejects disjoint read-only and writable ranges in one imported physical buffer`, () => {
    const {device, graph, views, createContributor} = createFixture(route);
    const physical = device.createBuffer({byteLength: 512, usage: Buffer.STORAGE});
    try {
      const [inputName, input] = [...views.entries()].find(
        ([name]) => !name.startsWith('output.')
      )!;
      const [outputName, output] = [...views.entries()].find(([name]) =>
        name.startsWith('output.')
      )!;
      const handle = graph.importBuffer(
        {id: 'read-write-alias', byteLength: 512, usage: Buffer.STORAGE},
        physical
      );
      const inputView = graph.createDataView(handle, {format: input.format, length: input.length});
      const outputView = graph.createDataView(handle, {
        format: output.format,
        length: output.length,
        byteOffset: 256
      });
      const unchanged = watchDeclarations(graph);
      expect(() => createContributor({[inputName]: inputView, [outputName]: outputView})).toThrow(
        /outputs must not share buffers with inputs/
      );
      unchanged();
    } finally {
      physical.destroy();
      device.destroy();
    }
  });

  it(`${route} accepts valid STORAGE-only bindings`, () => {
    const {device, graph, createContributor} = createFixture(route);
    try {
      const nodes = createContributor().getCommandNodes(graph);
      expect(nodes.length).toBeGreaterThan(0);
      for (const node of nodes) {
        for (const resource of node.resources ?? []) {
          if (!('buffer' in resource) || !(resource.buffer instanceof GraphDataView)) continue;
          if (!resource.usage.startsWith('storage')) continue;
          const range = getViewBindingRange(resource.buffer);
          expect(resource.buffer.buffer.usage & Buffer.STORAGE).not.toBe(0);
          expect(range.size).toBeGreaterThan(0);
          expect(range.size).toBeLessThanOrEqual(device.limits.maxStorageBufferBindingSize);
          expect(range.offset + range.size).toBeLessThanOrEqual(resource.buffer.buffer.byteLength);
        }
      }
    } finally {
      device.destroy();
    }
  });
}

for (const route of ['time-affine', 'time-piecewise'] as const) {
  it(`${route} checks the private iteration-summary scratch before declaring resources`, () => {
    const {device, graph, createContributor} = createFixture(route);
    try {
      Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 16});
      const contributor = createContributor();
      const unchanged = watchDeclarations(graph);
      expect(() => contributor.getCommandNodes(graph)).toThrow(
        /summary with iteration count needs 20 bytes.*limit 16/
      );
      unchanged();
    } finally {
      device.destroy();
    }
  });
}

for (const route of ['flow-parallel', 'flow-primal-dual'] as const) {
  it(`${route} checks row-sized scratch before declaring resources or kernels`, () => {
    const {device, graph, createContributor} = createFixture(route, 16);
    try {
      Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 100});
      const contributor = createContributor();
      const unchanged = watchDeclarations(graph);
      expect(() => contributor.getCommandNodes(graph)).toThrow(
        route === 'flow-parallel'
          ? /shortest-path labels needs 192 bytes.*limit 100/
          : /push-relabel heights\/excess needs 128 bytes.*limit 100/
      );
      unchanged();
    } finally {
      device.destroy();
    }
  });
}

for (const [route, limit, expected] of [
  ['facility', 16, /facility control state needs 28 bytes.*limit 16/],
  ['louvain-sparse', 1024, /sparse dispatch gate allowance needs 3840 bytes.*limit 1024/],
  ['point-scalable', 32, /frontier control needs 48 bytes.*limit 32/]
] as const) {
  it(`${route} checks derived control/dispatch scratch before changing the graph`, () => {
    const {device, graph, createContributor} = createFixture(route);
    try {
      Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: limit});
      const contributor = createContributor();
      const unchanged = watchDeclarations(graph);
      expect(() => contributor.getCommandNodes(graph)).toThrow(expected);
      unchanged();
    } finally {
      device.destroy();
    }
  });
}
