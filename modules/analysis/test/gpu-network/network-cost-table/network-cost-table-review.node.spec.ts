// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  computeNetworkCostTableOnCPU,
  CPUNetworkCostTableOperation
} from '../../../src/gpu-network/network-cost-table/cpu-network-cost-table-operation';
import {GPUNetworkCostTable} from '../../../src/gpu-network/network-cost-table/gpu-network-cost-table';
import {
  GPUNetworkCostTableOperation,
  type NetworkCostTableOperationProps
} from '../../../src/gpu-network/network-cost-table/gpu-network-cost-table-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('network:F8 snapshots budgets and validates class-instance status aliases', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `snapshot-${serial++}`, format, length);
  const complete = view('uint32', 1);
  const props = {
    topology: {
      offsets: view('uint32', 13),
      neighbors: view('uint32', 11),
      weights: view('float32', 11)
    },
    costLimit: view('float32', 1),
    laneCount: 3,
    maximumIterations: 1,
    localIterations: 1,
    output: {
      offsets: view('uint32', 13),
      targets: view('uint32', 144),
      costs: view('float32', 144),
      status: {complete}
    }
  };
  class AliasedStatus {
    complete = complete;
    iterationLimitReached = complete;
  }
  expect(
    () =>
      new GPUNetworkCostTable({
        ...props,
        output: {...props.output, status: {complete, iterationLimitReached: complete}}
      })
  ).toThrow();
  expect
    .soft(
      () =>
        new GPUNetworkCostTable({...props, output: {...props.output, status: new AliasedStatus()}})
    )
    .toThrow();
  const table = new GPUNetworkCostTable(props);
  expect(table.estimatedCommandNodeCount).toBe(34);
  props.maximumIterations = 10;
  expect(table.getCommandNodes(graph).length).toBe(34);
});

it('certifies inclusive cutoff membership using float32 path additions', () => {
  const topology = {
    offsets: Uint32Array.of(0, 1, 2, 2),
    neighbors: Uint32Array.of(1, 2),
    weights: Float32Array.of(0.1, 0.4)
  };
  expect(topology.weights[0] + topology.weights[1]).toBeGreaterThan(0.5);
  expect(Math.fround(topology.weights[0] + topology.weights[1])).toBe(0.5);
  for (const maximumIterations of [undefined, 16]) {
    const result = computeNetworkCostTableOnCPU(
      {topology, costLimit: Float32Array.of(0.5)},
      {capacity: 9, maximumIterations}
    );
    expect(result.complete[0]).toBe(1);
    expect(result.targets.slice(result.offsets[0], result.offsets[1])).toEqual(
      Uint32Array.of(0, 1, 2)
    );
    expect(result.costs[2]).toBe(0.5);
  }
});

it('keeps an omitted CPU budget exact while binding an explicit 64-round budget', () => {
  const nodeCount = 1026;
  const topology = {
    offsets: Uint32Array.from({length: nodeCount + 1}, (_, node) => Math.min(node, nodeCount - 1)),
    neighbors: Uint32Array.from({length: nodeCount - 1}, (_, node) => node + 1),
    weights: new Float32Array(nodeCount - 1).fill(1)
  };
  const capacity = (nodeCount * (nodeCount + 1)) / 2;
  const input = {topology, costLimit: Float32Array.of(Infinity)};
  const exact = computeNetworkCostTableOnCPU(input, {capacity});
  expect(exact.requiredCount[0]).toBe(526851);
  expect(exact.targets[exact.offsets[1] - 1]).toBe(1025);
  expect(exact.complete[0]).toBe(1);
  expect(exact.iterationLimitReached[0]).toBe(0);
  const bounded = computeNetworkCostTableOnCPU(input, {capacity, maximumIterations: 64});
  expect(bounded.requiredCount[0]).toBe(526850);
  expect(bounded.targets[bounded.offsets[1] - 1]).toBe(1024);
  expect(bounded.complete[0]).toBe(0);
  expect(bounded.iterationLimitReached[0]).toBe(1);
});

it('validates GPU contributor and both operation iteration limits at construction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `review-${serial++}`, format, length);
  const props = {
    id: 'review-table',
    topology: {
      offsets: view('uint32', 2),
      neighbors: view('uint32', 0),
      weights: view('float32', 0)
    },
    costLimit: view('float32', 1),
    output: {offsets: view('uint32', 2), targets: view('uint32', 1), costs: view('float32', 1)}
  };
  const operationProps: NetworkCostTableOperationProps = {
    id: 'review-table',
    nodeCount: 1,
    capacity: 1,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    costLimit: 'cutoff',
    output: {offsets: 'tableOffsets', targets: 'targets', costs: 'costs'}
  };
  const constructors = [
    (iterations: {maximumIterations?: number; localIterations?: number}) =>
      new GPUNetworkCostTable({...props, ...iterations}),
    ...[GPUNetworkCostTableOperation, CPUNetworkCostTableOperation].map(
      Operation => (iterations: {maximumIterations?: number; localIterations?: number}) =>
        new Operation({...operationProps, ...iterations})
    )
  ];
  try {
    for (const construct of constructors) {
      expect(() => construct({maximumIterations: 0})).toThrow(/^review-table maximumIterations/);
      expect(() => construct({localIterations: 65})).toThrow(/^review-table localIterations/);
      expect(() => construct({maximumIterations: 1, localIterations: 1})).not.toThrow();
      expect(() => construct({maximumIterations: 1024, localIterations: 64})).not.toThrow();
    }
  } finally {
    device.destroy();
  }
});

it('documents capacity-independent convergence and lane-by-node batch sweeps', () => {
  const documentation = readFileSync('docs/api/gpu-network/operators.md', 'utf8');
  const section = documentation.slice(
    documentation.indexOf('### `GPUNetworkCostTable`'),
    documentation.indexOf('### `GPUMapMatching`')
  );
  expect(section).toContain('convergence independently of capacity');
  expect(section).toContain('O(laneCount * nodeCount)');
  expect(section).toContain('converged=1, overflow=1, complete=0');
  const source = readFileSync(
    'modules/analysis/src/gpu-network/network-cost-table/gpu-network-cost-table.ts',
    'utf8'
  );
  expect(source).toContain('O(laneCount * nodeCount)');
});
