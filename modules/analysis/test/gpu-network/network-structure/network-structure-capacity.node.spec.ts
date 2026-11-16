// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUNetworkStructure,
  GPU_NETWORK_STRUCTURE_MAXIMUM_NODE_COUNT,
  GPU_NETWORK_STRUCTURE_MAXIMUM_EDGE_COUNT
} from '../../../src/gpu-network/network-structure/gpu-network-structure';
import {createNullWebGPUDevice, createVectorView} from '../../utils/gpu-contributor-test-utils';

function createViews(nodeCount = 4, edgeCount = 6) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let next = 0;
  const view = (length: number) =>
    createTransientView(graph, `capacity-view-${next++}`, 'uint32', length);
  return {
    device,
    graph,
    view,
    props: {
      id: 'structure-capacity',
      topology: {offsets: view(nodeCount + 1), neighbors: view(edgeCount)},
      reverseEdgeSlots: view(edgeCount),
      output: {
        strongComponents: view(nodeCount),
        bridges: view(edgeCount),
        articulationPoints: view(nodeCount),
        biconnectedComponents: view(edgeCount)
      }
    }
  };
}

it('accepts graphs above both old limits and bounds the actual emitted node count', () => {
  for (const [nodeCount, edgeCount] of [
    [1, 0],
    [4, 6],
    [5000, 70000],
    [100000, 400000]
  ]) {
    const {graph, props} = createViews(nodeCount, edgeCount);
    const contributor = new GPUNetworkStructure({...props, strongComponentsMaximumIterations: 1});
    const nodes = contributor.getCommandNodes(graph);
    expect(contributor.mode).toBe('auto');
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    expect(contributor.estimatedCommandNodeCount).toBeLessThanOrEqual(32768);
    for (const node of nodes) {
      const storageResources = (node.resources ?? []).filter(
        resource => 'buffer' in resource && resource.usage !== 'indirect'
      );
      expect(storageResources.length).toBeLessThanOrEqual(8);
    }
  }
});

it('keeps the exact reference route and both of its original named limits', () => {
  for (const [nodeCount, edgeCount] of [
    [1025, 6],
    [4, 65537]
  ]) {
    const {props} = createViews(nodeCount, edgeCount);
    expect(() => new GPUNetworkStructure({...props, mode: 'exact'})).toThrow(
      nodeCount > 1024
        ? /structure-capacity nodeCount.*1024/
        : /structure-capacity edgeCount.*65536/
    );
  }
  const {graph, props} = createViews();
  const contributor = new GPUNetworkStructure({...props, mode: 'exact'});
  expect(contributor.getCommandNodes(graph)).toHaveLength(3);
  expect(contributor.estimatedCommandNodeCount).toBe(3);
});

it('checks storage bindings before emitting nodes, including auxiliary scratch and aligned prefixes', () => {
  const {device, graph, props} = createViews(5000, 70000);
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 300000});
  const contributor = new GPUNetworkStructure(props);
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /structure-capacity endpointsAndTour scratch.*limit 300000/
  );

  const smaller = createViews(5000, 6000);
  Object.defineProperty(smaller.device.limits, 'maxStorageBufferBindingSize', {value: 250000});
  expect(() => new GPUNetworkStructure(smaller.props).getCommandNodes(smaller.graph)).toThrow(
    /structure-capacity auxiliary scratch.*limit 250000/
  );

  const offsetFixture = createViews();
  const whole = offsetFixture.view(100);
  const sliced = offsetFixture.graph.createDataView(whole.buffer, {
    format: 'uint32',
    length: 4,
    byteOffset: 252
  });
  Object.defineProperty(offsetFixture.device.limits, 'maxStorageBufferBindingSize', {value: 256});
  expect(() =>
    new GPUNetworkStructure({
      topology: offsetFixture.props.topology,
      output: {strongComponents: sliced}
    }).getCommandNodes(offsetFixture.graph)
  ).toThrow(/strongComponents.*limit 256/);
});

it('validates indexing bounds, auxiliary indexing and SCC budgets at construction', () => {
  const {props, view} = createViews();
  expect(
    () =>
      new GPUNetworkStructure({
        ...props,
        topology: {...props.topology, offsets: view(GPU_NETWORK_STRUCTURE_MAXIMUM_NODE_COUNT + 2)}
      })
  ).toThrow(/structure-capacity nodeCount/);
  expect(
    () =>
      new GPUNetworkStructure({
        ...props,
        topology: {...props.topology, neighbors: view(GPU_NETWORK_STRUCTURE_MAXIMUM_EDGE_COUNT + 1)}
      })
  ).toThrow(/structure-capacity edgeCount/);
  const auxiliary = createViews(200000000, 0);
  expect(() => new GPUNetworkStructure(auxiliary.props)).toThrow(/auxiliaryCount.*4294967294/);
  for (const strongComponentsMaximumIterations of [0, 4097, 1.5, Number.NaN]) {
    expect(() => new GPUNetworkStructure({...props, strongComponentsMaximumIterations})).toThrow(
      /structure-capacity strongComponentsMaximumIterations.*4096/
    );
  }
});

it('validates optional SCC status shape and disjoint ownership', () => {
  const {props, view} = createViews();
  const strongComponentsStatus = {
    converged: view(1),
    iterationCount: view(1),
    iterationLimitReached: view(1)
  };
  expect(() => new GPUNetworkStructure({...props, strongComponentsStatus})).not.toThrow();
  expect(
    () =>
      new GPUNetworkStructure({
        ...props,
        strongComponentsStatus: {...strongComponentsStatus, iterationCount: view(2)}
      })
  ).toThrow(/strongComponentsStatus.iterationCount.*one/);
  expect(
    () =>
      new GPUNetworkStructure({
        ...props,
        strongComponentsStatus: {
          ...strongComponentsStatus,
          iterationCount: strongComponentsStatus.converged
        }
      })
  ).toThrow(/aligned binding byte ranges must not overlap/);
});

it('counts every node for large edgeless articulation graphs', () => {
  for (const nodeCount of [100000, 1048576]) {
    const {graph, props} = createViews(nodeCount, 0);
    const contributor = new GPUNetworkStructure({
      ...props,
      output: {articulationPoints: props.output.articulationPoints}
    });
    expect(contributor.getCommandNodes(graph).length).toBeLessThanOrEqual(
      contributor.estimatedCommandNodeCount
    );
  }
});

it('omitted mode and SCC controls preserve the exact route throughout the old supported range', () => {
  for (const nodeCount of [513, 1024]) {
    const {graph, props} = createViews(nodeCount, nodeCount - 1);
    const contributor = new GPUNetworkStructure({
      topology: props.topology,
      output: {strongComponents: props.output.strongComponents}
    });
    expect(contributor.mode).toBe('auto');
    expect(contributor.strongComponentsMode).toBe('exact');
    expect(contributor.getCommandNodes(graph)).toHaveLength(2);
    expect(contributor.estimatedCommandNodeCount).toBe(2);
  }
  const {props} = createViews(1025, 1024);
  expect(new GPUNetworkStructure(props).strongComponentsMode).toBe('parallel');
  expect(
    new GPUNetworkStructure({...createViews().props, mode: 'parallel'}).strongComponentsMode
  ).toBe('parallel');
});

it('rejects every overlapping pair across primary outputs and SCC status at construction', () => {
  const names = [
    'strongComponents',
    'bridges',
    'articulationPoints',
    'biconnectedComponents',
    'converged',
    'iterationCount',
    'iterationLimitReached'
  ] as const;
  for (let first = 0; first < names.length; first++) {
    for (let second = first + 1; second < names.length; second++) {
      const {graph, props, view} = createViews(2, 2);
      const fields = {
        ...props.output,
        converged: view(1),
        iterationCount: view(1),
        iterationLimitReached: view(1)
      };
      const shared = view(3);
      for (const name of [names[first], names[second]]) {
        fields[name] = graph.createDataView(shared.buffer, {
          format: 'uint32',
          byteOffset: 4,
          length:
            name === 'converged' || name === 'iterationCount' || name === 'iterationLimitReached'
              ? 1
              : 2
        });
      }
      expect(
        () =>
          new GPUNetworkStructure({
            ...props,
            output: {
              strongComponents: fields.strongComponents,
              bridges: fields.bridges,
              articulationPoints: fields.articulationPoints,
              biconnectedComponents: fields.biconnectedComponents
            },
            strongComponentsStatus: {
              converged: fields.converged,
              iterationCount: fields.iterationCount,
              iterationLimitReached: fields.iterationLimitReached
            }
          })
      ).toThrow(/structure-capacity .*aligned binding byte ranges must not overlap/);
    }
  }
});

it('rejects chunked writable outputs before alias checks or graph declaration', () => {
  const {props, view} = createViews(2, 2);
  const chunked = createVectorView('chunked-labels', 'uint32', [view(1), view(1)]);
  expect(
    () =>
      new GPUNetworkStructure({
        ...props,
        output: {strongComponents: chunked as unknown as GraphDataView<'uint32'>}
      })
  ).toThrow(/must be one packed view/);
});

it('failed aligned-binding and scratch preflight declares no resources or nodes', () => {
  for (const failure of ['input-prefix', 'endpoints', 'auxiliary', 'fixed-dispatch'] as const) {
    const {device, graph, props, view} =
      failure === 'fixed-dispatch'
        ? createViews(1, 0)
        : failure === 'auxiliary'
          ? createViews(5000, 6000)
          : createViews(5000, 70000);
    if (failure === 'input-prefix') {
      const whole = view(75000);
      props.topology.neighbors = graph.createDataView(whole.buffer, {
        format: 'uint32',
        byteOffset: 252,
        length: 70000
      });
    }
    const limit =
      failure === 'fixed-dispatch'
        ? 32
        : failure === 'input-prefix'
          ? 280000
          : failure === 'auxiliary'
            ? 250000
            : 300000;
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: limit});
    const contributor = new GPUNetworkStructure({...props, mode: 'parallel'});
    const declarations = [
      vi.spyOn(graph, 'createTransientBuffer'),
      vi.spyOn(graph, 'createDataView'),
      vi.spyOn(graph, 'importBuffer'),
      vi.spyOn(graph, 'add')
    ];
    try {
      expect(() => contributor.getCommandNodes(graph), failure).toThrow(/limit/);
      for (const declaration of declarations) expect(declaration).not.toHaveBeenCalled();
    } finally {
      for (const declaration of declarations) declaration.mockRestore();
    }
  }
});

it('rejects physical output aliases and aligned-prefix overlap across separately imported handles', () => {
  for (const byteOffset of [0, 4]) {
    const {device, graph, props} = createViews(1, 0);
    const buffer = device.createBuffer({byteLength: 512, usage: Buffer.STORAGE});
    try {
      const descriptor = {byteLength: 512, usage: Buffer.STORAGE};
      const first = graph.importBuffer({...descriptor, id: 'physical-first'}, buffer);
      const second = graph.importBuffer({...descriptor, id: 'physical-second'}, buffer);
      expect(first).not.toBe(second);
      const strongComponents = graph.createDataView(first, {format: 'uint32', length: 1});
      const articulationPoints = graph.createDataView(second, {
        format: 'uint32',
        byteOffset,
        length: 1
      });
      expect(
        () => new GPUNetworkStructure({...props, output: {strongComponents, articulationPoints}})
      ).toThrow(/aligned binding byte ranges must not overlap/);
    } finally {
      buffer.destroy();
    }
  }
});

it('preflights maxBufferSize for every scratch family before declaring resources', () => {
  for (const shape of ['reachable', 'endpoints', 'auxiliary'] as const) {
    const {device, graph, props} =
      shape === 'reachable'
        ? createViews(1024, 2048)
        : shape === 'auxiliary'
          ? createViews(5000, 6000)
          : createViews(5000, 70000);
    const limit = shape === 'reachable' ? 1048576 : shape === 'auxiliary' ? 250000 : 300000;
    Object.defineProperty(device.limits, 'maxBufferSize', {value: limit});
    const contributor = new GPUNetworkStructure(props);
    const declarations = [
      vi.spyOn(graph, 'createTransientBuffer'),
      vi.spyOn(graph, 'createDataView'),
      vi.spyOn(graph, 'add')
    ];
    try {
      expect(() => contributor.getCommandNodes(graph), shape).toThrow(
        /scratch byteLength \(maxBufferSize\)/
      );
      for (const declaration of declarations) expect(declaration).not.toHaveBeenCalled();
    } finally {
      for (const declaration of declarations) declaration.mockRestore();
    }
  }
});

it('preflights the complete logical allocation behind an otherwise small aligned view', () => {
  const {device, graph, props, view} = createViews(2, 2);
  const whole = view(256);
  const strongComponents = graph.createDataView(whole.buffer, {format: 'uint32', length: 2});
  Object.defineProperty(device.limits, 'maxBufferSize', {value: 512});
  const contributor = new GPUNetworkStructure({...props, output: {strongComponents}});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /strongComponents buffer byteLength \(maxBufferSize\)/
    );
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    allocate.mockRestore();
  }
});

it('rejects nonfinite, fractional and out-of-range numeric shader bounds at construction', () => {
  for (const value of [
    Number.NaN,
    Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    -1,
    1.5,
    2 ** 32
  ]) {
    for (const name of ['nodeCount', 'edgeCount'] as const) {
      const {props} = createViews(2, 2);
      Object.defineProperty(
        name === 'nodeCount' ? props.topology.offsets : props.topology.neighbors,
        'length',
        {value: name === 'nodeCount' ? value + 1 : value}
      );
      expect(() => new GPUNetworkStructure(props), `${name}=${value}`).toThrow();
    }
    const {props} = createViews(2, 2);
    expect(
      () => new GPUNetworkStructure({...props, strongComponentsMaximumIterations: value})
    ).toThrow(/strongComponentsMaximumIterations/);
  }
});

it('prefixes every generated resource and node when two structures share one graph', () => {
  const {graph, props, view} = createViews(33, 64);
  const first = new GPUNetworkStructure({
    ...props,
    id: 'first-structure',
    mode: 'parallel',
    strongComponentsMaximumIterations: 1
  });
  const secondOutput = {
    strongComponents: view(33),
    bridges: view(64),
    articulationPoints: view(33),
    biconnectedComponents: view(64)
  };
  const second = new GPUNetworkStructure({
    ...props,
    id: 'second-structure',
    output: secondOutput,
    mode: 'parallel',
    strongComponentsMaximumIterations: 1
  });
  const existing = new Set([
    props.topology.offsets.buffer,
    props.topology.neighbors.buffer,
    props.reverseEdgeSlots.buffer,
    ...Object.values(props.output).map(output => output.buffer),
    ...Object.values(secondOutput).map(output => output.buffer)
  ]);
  for (const contributor of [first, second]) {
    const nodes = contributor.getCommandNodes(graph);
    for (const node of nodes) {
      expect(node.id.startsWith(contributor.id + '-')).toBe(true);
      for (const resource of node.resources ?? []) {
        if (!('buffer' in resource)) continue;
        const buffer =
          'rowByteLength' in resource.buffer ? resource.buffer.buffer : resource.buffer;
        if (!existing.has(buffer)) expect(buffer.id.startsWith(contributor.id + '-')).toBe(true);
      }
      graph.add(node);
    }
  }
});
