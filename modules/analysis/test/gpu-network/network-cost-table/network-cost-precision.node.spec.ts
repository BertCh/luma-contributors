// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {
  computeNetworkCostTableOnCPU,
  CPUNetworkCostTableOperation
} from '../../../src/gpu-network/network-cost-table/cpu-network-cost-table-operation';
import {
  computeNetworkCostMatrixOnCPU,
  CPUNetworkCostMatrixOperation
} from '../../../src/gpu-network/network-accessibility/cpu-network-cost-matrix-operation';
import {
  computeNetworkReachabilityOnCPU,
  CPUNetworkReachabilityOperation
} from '../../../src/gpu-network/network-reachability/cpu-network-reachability-operation';
import {
  computeNetworkServiceAreasOnCPU,
  CPUNetworkServiceAreasOperation
} from '../../../src/gpu-network/network-analysis/cpu-network-service-areas-operation';
import {GPUNetworkCostMatrixOperation} from '../../../src/gpu-network/network-accessibility/gpu-network-cost-matrix-operation';
import {GPUNetworkReachabilityOperation} from '../../../src/gpu-network/network-reachability/gpu-network-reachability-operation';
import {GPUNetworkCostTableOperation} from '../../../src/gpu-network/network-cost-table/gpu-network-cost-table-operation';
import {GPUNetworkServiceAreasOperation} from '../../../src/gpu-network/network-analysis/gpu-network-service-areas-operation';

it.each([
  {firstWeight: 0.1, secondWeight: 0.4, cutoff: 0.5, exactSum: 0.5000000074505806},
  {firstWeight: 0.2, secondWeight: 0.8, cutoff: 1, exactSum: 1.0000000149011612}
])('FRM-03 pins the existing inclusive-cutoff defaults for completed and budgeted searches ($cutoff)', ({
  firstWeight,
  secondWeight,
  cutoff,
  exactSum
}) => {
  const topology = {
    offsets: Uint32Array.of(0, 1, 2, 2),
    neighbors: Uint32Array.of(1, 2),
    weights: Float32Array.of(firstWeight, secondWeight)
  };
  // Stored float32 pairs sum in binary64 to the supplied independent literals; float32 rounds to the cutoff.
  expect(topology.weights[0] + topology.weights[1]).toBe(exactSum);
  expect(Math.fround(topology.weights[0] + topology.weights[1])).toBe(cutoff);
  for (const budget of [undefined, 4]) {
    const costLimit = Float32Array.of(cutoff);
    const table = computeNetworkCostTableOnCPU(
      {topology, costLimit},
      {capacity: 9, maximumIterations: budget, localIterations: 1}
    );
    expect(Array.from(table.targets.slice(table.offsets[0], table.offsets[1]))).toEqual([0, 1, 2]);
    expect(table.complete[0]).toBe(1);
    const matrix = computeNetworkCostMatrixOnCPU(
      {topology, costLimit, startNodes: Uint32Array.of(0)},
      {rowCount: 1, maximumIterations: budget, localIterations: 1}
    );
    const reachability = computeNetworkReachabilityOnCPU(
      {topology, costLimit, startNodes: Uint32Array.of(0)},
      {maximumIterations: budget, localIterations: 1}
    );
    for (const result of [matrix, reachability]) {
      expect(result.costs[2]).toBe(Infinity);
      expect(result.converged[0]).toBe(1);
    }
  }
});

for (const [first, second, cutoff] of [
  [0.1, 0.4, 0.5],
  [0.2, 0.8, 1]
]) {
  for (const costPrecision of ['float32-per-edge', 'float64'] as const) {
    it(`FRM-03 forwards ${costPrecision} through pure and named forms at cutoff ${cutoff}`, async () => {
      const topology = {
        offsets: Uint32Array.of(0, 1, 2, 2),
        neighbors: Uint32Array.of(1, 2),
        weights: Float32Array.of(first, second)
      };
      const costLimit = Float32Array.of(cutoff);
      const startNodes = Uint32Array.of(0);
      const reached = costPrecision === 'float32-per-edge';
      for (const maximumIterations of [undefined, 4]) {
        const options = {maximumIterations, localIterations: 1};
        const input = {topology, costLimit, startNodes, costPrecision};
        const matrix = computeNetworkCostMatrixOnCPU(input, {...options, rowCount: 1});
        const reachability = computeNetworkReachabilityOnCPU(input, options);
        const service = computeNetworkServiceAreasOnCPU(
          {...input, facilities: startNodes},
          options
        );
        for (const result of [matrix, reachability, service]) {
          expect(result.costs[2]).toBe(reached ? cutoff : Infinity);
          expect(result.converged[0]).toBe(1);
        }
        expect(Array.from(reachability.predecessors)).toEqual([
          0xffffffff,
          0,
          reached ? 1 : 0xffffffff
        ]);
        expect(Array.from(reachability.predecessorEdges)).toEqual([
          0xffffffff,
          0,
          reached ? 1 : 0xffffffff
        ]);
        expect(Array.from(service.assignments)).toEqual([0, 0, reached ? 0 : 0xffffffff]);
        expect(Array.from(service.predecessors)).toEqual([0xffffffff, 0, reached ? 1 : 0xffffffff]);
        const table = computeNetworkCostTableOnCPU(input, {...options, capacity: 9});
        expect(Array.from(table.targets.slice(0, table.offsets[1]))).toEqual(
          reached ? [0, 1, 2] : [0, 1]
        );
        const common = {
          ...options,
          costPrecision,
          nodeCount: 3,
          topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          costLimit: 'limit'
        };
        const operations = [
          new CPUNetworkCostMatrixOperation({
            ...common,
            rowCount: 1,
            startNodes: 'starts',
            output: {costs: 'costs'}
          }),
          new CPUNetworkReachabilityOperation({
            ...common,
            startNodes: 'starts',
            output: {costs: 'costs', predecessors: 'parents', predecessorEdges: 'edges'}
          }),
          new CPUNetworkServiceAreasOperation({
            ...common,
            facilities: 'starts',
            facilityCapacity: 1,
            output: {assignments: 'assignments', costs: 'costs', predecessors: 'parents'}
          }),
          new CPUNetworkCostTableOperation({
            ...common,
            capacity: 9,
            output: {offsets: 'offsetsResult', targets: 'targets', costs: 'costs'}
          })
        ];
        const values = {...topology, costLimit, startNodes, facilities: startNodes};
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
        for (const [operationIndex, operation] of operations.slice(0, 3).entries()) {
          const result = await operation.execute({inputs});
          expect((result['costs'] as Float32Array)[2]).toBe(reached ? cutoff : Infinity);
          if (operationIndex === 1 || operationIndex === 2)
            expect(Array.from(result['parents'] as Uint32Array)).toEqual([
              0xffffffff,
              0,
              reached ? 1 : 0xffffffff
            ]);
          if (operationIndex === 1)
            expect(Array.from(result['edges'] as Uint32Array)).toEqual([
              0xffffffff,
              0,
              reached ? 1 : 0xffffffff
            ]);
          if (operationIndex === 2)
            expect(Array.from(result['assignments'] as Uint32Array)).toEqual([
              0,
              0,
              reached ? 0 : 0xffffffff
            ]);
        }
        const result = await operations[3].execute({inputs});
        expect(
          Array.from(
            (result['targets'] as Uint32Array).slice(0, (result['offsetsResult'] as Uint32Array)[1])
          )
        ).toEqual(reached ? [0, 1, 2] : [0, 1]);
      }
    });
  }
}

it('FRM-03 rejects unknown precision in all CPU forms before execution', () => {
  const costPrecision = 'float16' as 'float64';
  const topology = {
    offsets: Uint32Array.of(0, 0),
    neighbors: new Uint32Array(),
    weights: new Float32Array()
  };
  const input = {
    topology,
    startNodes: Uint32Array.of(0),
    costLimit: Float32Array.of(1),
    costPrecision
  };
  for (const invoke of [
    () => computeNetworkCostMatrixOnCPU(input, {rowCount: 1}),
    () => computeNetworkReachabilityOnCPU(input),
    () => computeNetworkCostTableOnCPU(input, {capacity: 1}),
    () => computeNetworkServiceAreasOnCPU({...input, facilities: input.startNodes}),
    () =>
      new CPUNetworkCostMatrixOperation({
        costPrecision,
        nodeCount: 1,
        rowCount: 1,
        topology: {offsets: 'o', neighbors: 'n', weights: 'w'},
        startNodes: 's',
        output: {costs: 'c'}
      }),
    () =>
      new CPUNetworkReachabilityOperation({
        costPrecision,
        nodeCount: 1,
        topology: {offsets: 'o', neighbors: 'n', weights: 'w'},
        startNodes: 's',
        output: {costs: 'c'}
      }),
    () =>
      new CPUNetworkCostTableOperation({
        costPrecision,
        nodeCount: 1,
        capacity: 1,
        topology: {offsets: 'o', neighbors: 'n', weights: 'w'},
        costLimit: 'l',
        output: {offsets: 'a', targets: 'b', costs: 'c'}
      }),
    () =>
      new CPUNetworkServiceAreasOperation({
        costPrecision,
        nodeCount: 1,
        facilityCapacity: 1,
        topology: {offsets: 'o', neighbors: 'n', weights: 'w'},
        facilities: 's',
        output: {assignments: 'a'}
      })
  ])
    expect(invoke).toThrow(/costPrecision.*float32-per-edge.*float64.*float16/);
});

it('FRM-03 captures the precision getter once in pure forms, including budgeted searches', () => {
  let reads = 0;
  const input = {
    topology: {
      offsets: Uint32Array.of(0, 1, 2, 2),
      neighbors: Uint32Array.of(1, 2),
      weights: Float32Array.of(0.1, 0.4)
    },
    startNodes: Uint32Array.of(0),
    facilities: Uint32Array.of(0),
    costLimit: Float32Array.of(0.5),
    get costPrecision() {
      reads++;
      return 'float32-per-edge' as const;
    }
  };
  for (const maximumIterations of [undefined, 4]) {
    for (const invoke of [
      () => computeNetworkCostMatrixOnCPU(input, {rowCount: 1, maximumIterations}),
      () => computeNetworkReachabilityOnCPU(input, {maximumIterations}),
      () => computeNetworkCostTableOnCPU(input, {capacity: 9, maximumIterations}),
      () => computeNetworkServiceAreasOnCPU(input, {maximumIterations})
    ]) {
      reads = 0;
      invoke();
      expect(reads).toBe(1);
    }
  }
});

it('FRM-03 GPU named forms reject unsupported float64 instead of ignoring it', () => {
  const common = {
    costPrecision: 'float64' as const,
    nodeCount: 1,
    topology: {offsets: 'o', neighbors: 'n', weights: 'w'}
  };
  for (const invoke of [
    () =>
      new GPUNetworkCostMatrixOperation({
        ...common,
        rowCount: 1,
        startNodes: 's',
        output: {costs: 'c'}
      }),
    () => new GPUNetworkReachabilityOperation({...common, startNodes: 's', output: {costs: 'c'}}),
    () =>
      new GPUNetworkCostTableOperation({
        ...common,
        capacity: 1,
        costLimit: 'l',
        output: {offsets: 'a', targets: 'b', costs: 'c'}
      }),
    () =>
      new GPUNetworkServiceAreasOperation({
        ...common,
        facilityCapacity: 1,
        facilities: 's',
        output: {assignments: 'a'}
      })
  ])
    expect(invoke).toThrow(/costPrecision.*float32-per-edge.*received float64.*CPU/);
});

it('FRM-03 retains ties-to-even through repeated half-ULP edges in pure and named matrix searches', async () => {
  const topology = {
    offsets: Uint32Array.of(0, 1, 2, 3, 3),
    neighbors: Uint32Array.of(1, 2, 3),
    weights: Float32Array.of(1, 2 ** -24, 2 ** -24)
  };
  for (const costPrecision of ['float32-per-edge', 'float64'] as const) {
    for (const maximumIterations of [undefined, 4]) {
      const input = {
        topology,
        startNodes: Uint32Array.of(0),
        costLimit: Float32Array.of(1),
        costPrecision
      };
      const options = {maximumIterations, localIterations: 1};
      // 2^-24 is half a binary32 ULP at 1. Each rounded addition ties to even 1;
      // binary64 instead exceeds the cutoff on the first such edge.
      const expected =
        costPrecision === 'float32-per-edge' ? [0, 1, 1, 1] : [0, 1, Infinity, Infinity];
      const matrix = computeNetworkCostMatrixOnCPU(input, {...options, rowCount: 1});
      expect(Array.from(matrix.costs)).toEqual(expected);
      const reachability = computeNetworkReachabilityOnCPU(input, options);
      expect(Array.from(reachability.costs)).toEqual(expected);
      expect(Array.from(reachability.predecessorEdges)).toEqual(
        costPrecision === 'float32-per-edge'
          ? [0xffffffff, 0, 1, 2]
          : [0xffffffff, 0, 0xffffffff, 0xffffffff]
      );
      const table = computeNetworkCostTableOnCPU(input, {...options, capacity: 16});
      expect(Array.from(table.targets.slice(0, table.offsets[1]))).toEqual(
        costPrecision === 'float32-per-edge' ? [0, 1, 2, 3] : [0, 1]
      );
      const props = {
        ...options,
        costPrecision,
        nodeCount: 4,
        rowCount: 1,
        topology: {offsets: 'o', neighbors: 'n', weights: 'w'},
        startNodes: 's',
        costLimit: 'l',
        output: {costs: 'costs'}
      };
      const operation = new CPUNetworkCostMatrixOperation(props);
      // Mutating the original policy after construction must not change a reused operation.
      props.costPrecision = costPrecision === 'float64' ? 'float32-per-edge' : 'float64';
      const values = {...topology, startNodes: input.startNodes, costLimit: input.costLimit};
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
      for (let submission = 0; submission < 2; submission++) {
        const result = await operation.execute({inputs});
        expect(Array.from(result['costs'] as Float32Array)).toEqual(expected);
      }
    }
  }
});
