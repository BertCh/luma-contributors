// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNetworkStronglyConnectedComponents,
  GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_MAXIMUM_COUNT,
  GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_MAXIMUM_ITERATIONS,
  computeNetworkStructureOnCPU
} from '../../../src/gpu-network/network-structure/index';
import {getGPUAnalysisOperator} from '../../../src/gpu-spatial-analysis/ontology/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createComponentCSR} from './strong-components-fixtures';

function createProps(nodeCount = 5000, edgeCount = 70000) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = (length: number) => createTransientView(graph, `view-${serial++}`, 'uint32', length);
  return {
    graph,
    device,
    view,
    props: {
      id: 'test-components',
      topology: {offsets: view(nodeCount + 1), neighbors: view(edgeCount)},
      output: {
        strongComponents: view(nodeCount),
        status: {
          converged: view(1),
          iterationCount: view(1),
          iterationLimitReached: view(1)
        }
      }
    }
  };
}

it('accepts graphs above both exact-structure limits and budgets every emitted node', () => {
  for (const maximumIterations of [
    1,
    GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_MAXIMUM_ITERATIONS
  ]) {
    const {graph, props} = createProps();
    const contributor = new GPUNetworkStronglyConnectedComponents({...props, maximumIterations});
    expect(contributor.nodeCount).toBe(5000);
    expect(contributor.edgeCount).toBe(70000);
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBe(contributor.estimatedCommandNodeCount);
    expect(nodes.length).toBe(2 + 6 * maximumIterations);
    expect(nodes.length).toBeLessThanOrEqual(32768);
  }
});

it('uses exactly two vertex-sized scratch columns regardless of the step budget', () => {
  for (const maximumIterations of [1, 8]) {
    const {graph, props} = createProps();
    const contributor = new GPUNetworkStronglyConnectedComponents({...props, maximumIterations});
    const nodes = contributor.getCommandNodes(graph);
    const scratch = new Map<string, number>();
    for (const node of nodes) {
      for (const resource of node.resources ?? []) {
        if ('buffer' in resource && resource.buffer) {
          const buffer =
            'rowByteLength' in resource.buffer ? resource.buffer.buffer : resource.buffer;
          if (buffer.id.startsWith('test-components-')) scratch.set(buffer.id, buffer.byteLength);
        }
      }
    }
    expect(Object.fromEntries(scratch)).toEqual({
      'test-components-color': 5000 * 4,
      'test-components-flags': 5000 * 4,
      'test-components-control': 4 * 4,
      'test-components-gate': 4 * 3 * 4
    });
  }
});

it('validates the named step and indexing limits at construction', () => {
  const {props, view} = createProps();
  for (const maximumIterations of [0, -1, 1.5, 4097, Number.NaN, Number.POSITIVE_INFINITY]) {
    expect(() => new GPUNetworkStronglyConnectedComponents({...props, maximumIterations})).toThrow(
      /test-components maximumIterations.*4096/
    );
  }
  expect(
    () =>
      new GPUNetworkStronglyConnectedComponents({
        ...props,
        topology: {
          ...props.topology,
          offsets: view(GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_MAXIMUM_COUNT + 2)
        }
      })
  ).toThrow(/test-components nodeCount.*4294967294/);
  expect(
    () =>
      new GPUNetworkStronglyConnectedComponents({
        ...props,
        topology: {
          ...props.topology,
          neighbors: view(GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_MAXIMUM_COUNT + 1)
        }
      })
  ).toThrow(/test-components edgeCount.*4294967294/);
});

it('validates vertex outputs, scalar status shapes, aliasing and graph membership', () => {
  const {graph, props, view} = createProps();
  expect(
    () =>
      new GPUNetworkStronglyConnectedComponents({
        ...props,
        output: {...props.output, strongComponents: view(4999)}
      })
  ).toThrow(/test-components output.strongComponents.*5000/);
  for (const name of ['converged', 'iterationCount', 'iterationLimitReached'] as const) {
    expect(
      () =>
        new GPUNetworkStronglyConnectedComponents({
          ...props,
          output: {...props.output, status: {...props.output.status, [name]: view(2)}}
        })
    ).toThrow(new RegExp(`test-components output.${name}.*1`));
  }
  expect(
    () =>
      new GPUNetworkStronglyConnectedComponents({
        ...props,
        output: {
          ...props.output,
          status: {...props.output.status, converged: props.output.status.iterationCount}
        }
      })
  ).toThrow(/test-components.*distinct buffers/);
  const small = createProps(1, 1);
  expect(
    () =>
      new GPUNetworkStronglyConnectedComponents({
        ...small.props,
        output: {...small.props.output, strongComponents: small.props.topology.neighbors}
      })
  ).toThrow(/test-components.*share buffers/);
  const other = createProps();
  const contributor = new GPUNetworkStronglyConnectedComponents({
    ...props,
    output: {
      ...props.output,
      status: {...props.output.status, converged: other.props.output.status.converged}
    }
  });
  expect(() => contributor.getCommandNodes(graph)).toThrow(/test-components.*target graph/);
});

it('rejects oversized CSR and vertex scratch bindings before declaring nodes', () => {
  for (const [nodeCount, edgeCount, expectedName] of [
    [33554432, 1, 'topology/output.offsets'],
    [1, 33554433, 'topology/output.neighbors']
  ] as const) {
    const {graph, props} = createProps(nodeCount, edgeCount);
    const contributor = new GPUNetworkStronglyConnectedComponents({...props, maximumIterations: 1});
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      new RegExp(`test-components ${expectedName}.*134217728`)
    );
  }
});

it('registers scale SCC under topological network structure with named limits and reference', () => {
  const descriptor = getGPUAnalysisOperator('GPUNetworkStronglyConnectedComponents');
  expect(descriptor?.tier).toBe('scale');
  expect(descriptor?.primaryCapability).toBe('network-structure');
  expect(descriptor?.coordinateModels).toContain('topological');
  expect(
    descriptor?.limits?.some(limit => limit.name === 'maximumIterations' && limit.maximum === 4096)
  ).toBe(true);
  expect(descriptor?.summary).toContain('Tarjan');
});

it('existing CPU Tarjan oracle labels 5000-vertex components with their lowest IDs', () => {
  const edges: [number, number][] = [];
  for (let vertex = 1; vertex < 4096; vertex++) edges.push([vertex - 1, vertex]);
  edges.push([4095, 0], [4095, 4096]);
  for (let vertex = 4097; vertex < 4999; vertex++) edges.push([vertex - 1, vertex]);
  edges.push([4998, 4096], [4999, 4999]);
  const input = createComponentCSR(5000, edges, 5);
  expect(Array.from(computeNetworkStructureOnCPU(input).strongComponents)).toEqual([
    ...new Array(4096).fill(0),
    ...new Array(903).fill(4096),
    4999
  ]);
});
