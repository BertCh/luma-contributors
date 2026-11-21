// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeNetworkCostTableOnCPU,
  CPUNetworkCostTableOperation
} from '../../../src/gpu-network/network-cost-table/cpu-network-cost-table-operation';
import {GPUNetworkCostTableOperation} from '../../../src/gpu-network/network-cost-table/gpu-network-cost-table-operation';
import {
  computeNetworkCostMatrixOnCPU,
  CPUNetworkCostMatrixOperation
} from '../../../src/gpu-network/network-accessibility/cpu-network-cost-matrix-operation';
import {GPUNetworkCostMatrixOperation} from '../../../src/gpu-network/network-accessibility/gpu-network-cost-matrix-operation';
import {
  computeNetworkReachabilityOnCPU,
  CPUNetworkReachabilityOperation
} from '../../../src/gpu-network/network-reachability/cpu-network-reachability-operation';
import {GPUNetworkReachabilityOperation} from '../../../src/gpu-network/network-reachability/gpu-network-reachability-operation';
import {
  computeNetworkServiceAreasOnCPU,
  CPUNetworkServiceAreasOperation
} from '../../../src/gpu-network/network-analysis/cpu-network-service-areas-operation';
import {GPUNetworkServiceAreasOperation} from '../../../src/gpu-network/network-analysis/gpu-network-service-areas-operation';

it('S2 rejects null precision in every pure and named CPU/GPU form', () => {
  const costPrecision = null as unknown as 'float64';
  const topology = {
    offsets: Uint32Array.of(0, 0),
    neighbors: new Uint32Array(),
    weights: new Float32Array()
  };
  const input = {
    topology,
    costPrecision,
    startNodes: Uint32Array.of(0),
    facilities: Uint32Array.of(0),
    costLimit: Float32Array.of(1)
  };
  const common = {
    costPrecision,
    nodeCount: 1,
    topology: {offsets: 'o', neighbors: 'n', weights: 'w'}
  };
  const invocations = [
    () => computeNetworkCostTableOnCPU(input, {capacity: 1}),
    () => computeNetworkCostMatrixOnCPU(input, {rowCount: 1}),
    () => computeNetworkReachabilityOnCPU(input),
    () => computeNetworkServiceAreasOnCPU(input),
    ...[CPUNetworkCostTableOperation, GPUNetworkCostTableOperation].map(
      Operation => () =>
        new Operation({
          ...common,
          capacity: 1,
          costLimit: 'l',
          output: {offsets: 'a', targets: 'b', costs: 'c'}
        })
    ),
    ...[CPUNetworkCostMatrixOperation, GPUNetworkCostMatrixOperation].map(
      Operation => () =>
        new Operation({...common, rowCount: 1, startNodes: 's', output: {costs: 'c'}})
    ),
    ...[CPUNetworkReachabilityOperation, GPUNetworkReachabilityOperation].map(
      Operation => () => new Operation({...common, startNodes: 's', output: {costs: 'c'}})
    ),
    ...[CPUNetworkServiceAreasOperation, GPUNetworkServiceAreasOperation].map(
      Operation => () =>
        new Operation({...common, facilityCapacity: 1, facilities: 's', output: {assignments: 'a'}})
    )
  ];
  const rejected = invocations.map(invoke => {
    try {
      invoke();
      return false;
    } catch (error) {
      expect(String(error)).toMatch(/costPrecision.*float32-per-edge.*float64.*null/);
      return true;
    }
  });
  expect(rejected).toEqual(Array(12).fill(true));
});

it('S4 omits unrepresentable float64 costs and withholds sparse completeness', async () => {
  const topology = {
    offsets: Uint32Array.of(0, 1, 2, 2),
    neighbors: Uint32Array.of(1, 2),
    weights: Float32Array.of(2e38, 2e38)
  };
  for (const maximumIterations of [undefined, 4]) {
    const result = computeNetworkCostTableOnCPU(
      {topology, costLimit: Float32Array.of(Infinity), costPrecision: 'float64'},
      {capacity: 9, maximumIterations, localIterations: 1}
    );
    // Each edge is finite binary32, but their sum exceeds its maximum finite value.
    // Keep exactly the three diagonal zeros and the two directly representable edges.
    expect(Array.from(result.targets.slice(0, result.offsets[1]))).toEqual([0, 1]);
    expect(result.requiredCount[0]).toBe(5);
    expect(result.costs.slice(0, result.offsets[3]).every(Number.isFinite)).toBe(true);
    expect(Array.from(result.complete)).toEqual([0]);
    expect(Array.from(result.converged)).toEqual([1]);
    expect(Array.from(result.overflow)).toEqual([0]);
  }
  const operation = new CPUNetworkCostTableOperation({
    costPrecision: 'float64',
    nodeCount: 3,
    capacity: 9,
    topology: {offsets: 'o', neighbors: 'n', weights: 'w'},
    costLimit: 'l',
    output: {offsets: 'rows', targets: 'targets', costs: 'costs', status: {complete: 'complete'}}
  });
  const values = {...topology, costLimit: Float32Array.of(Infinity)};
  const inputs = Object.fromEntries(
    Object.entries(values).map(([port, rows]) => [
      port,
      {
        format: rows instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
        length: rows.length,
        chunks: [{values: rows, length: rows.length}],
        pack: () => rows
      }
    ])
  );
  for (const edgeCost of [2e38, 1]) {
    topology.weights.fill(edgeCost);
    const result = await operation.execute({inputs});
    const rowEnd = (result['rows'] as Uint32Array)[1];
    expect(Array.from((result['targets'] as Uint32Array).slice(0, rowEnd))).toEqual(
      edgeCost === 1 ? [0, 1, 2] : [0, 1]
    );
    expect(Array.from(result['complete'] as Uint32Array)).toEqual([edgeCost === 1 ? 1 : 0]);
    expect((result['costs'] as Float32Array).slice(0, rowEnd).every(Number.isFinite)).toBe(true);
  }
});

it.each([
  'float64',
  'float32-per-edge'
] as const)('NOTE hop budgets snapshot source labels (%s)', costPrecision => {
  const topology = {
    offsets: Uint32Array.of(0, 2, 3, 4, 4),
    neighbors: Uint32Array.of(1, 2, 2, 3),
    weights: Float32Array.of(1, 5, 1, 1)
  };
  const input = {
    topology,
    startNodes: Uint32Array.of(0),
    costLimit: Float32Array.of(Infinity),
    costPrecision
  };
  const options = {maximumIterations: 1, localIterations: 2};
  // The only two-edge route to 3 is 0->2->3, cost 5+1=6.
  // 0->1->2->3 costs 3 but needs a third hop, so it cannot be consumed yet.
  const matrix = computeNetworkCostMatrixOnCPU(input, {...options, rowCount: 1});
  const table = computeNetworkCostTableOnCPU(input, {...options, capacity: 16});
  expect(Array.from(matrix.costs)).toEqual([0, 1, 2, 6]);
  const targetIndex = Array.from(table.targets.slice(0, table.offsets[1])).indexOf(3);
  expect(targetIndex).toBeGreaterThanOrEqual(0);
  expect(table.costs[targetIndex]).toBe(6);
  expect(Array.from(matrix.converged)).toEqual([0]);
  expect(Array.from(table.converged)).toEqual([0]);
});
