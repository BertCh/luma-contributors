// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, expectTypeOf, test} from 'vitest';
import {
  computeNetworkRoutingDistances,
  computeNetworkRoutingPredecessorsWithStatus,
  validateNetworkRoutingArrays,
  type CPUNetworkRoutingInput
} from '../../src/analysis-core/operations/cpu-network-routing';
import * as network from '@luma-contributors/analysis/gpu-network';
import * as allocation from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GPUNetworkIsochronesOutput} from '@luma-contributors/analysis/gpu-network';
import {computeKShortestPathsOnCPU} from '@luma-contributors/analysis/gpu-network';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

const topology = {
  offsets: Uint32Array.of(0, 1, 2, 2),
  neighbors: Uint32Array.of(1, 2),
  weights: Float32Array.of(0.1, 0.4)
};
const input = {topology, startNodes: Uint32Array.of(0), costLimit: Float32Array.of(0.5)};

describe('UNW-01: authoritative shared predecessor edge slots', () => {
  test.each([
    'float64',
    'float32-per-edge'
  ] as const)('retains the least parallel weight at precision %s', costPrecision => {
    const routing: CPUNetworkRoutingInput = {
      topology: {
        offsets: Uint32Array.of(0, 4, 4),
        neighbors: Uint32Array.of(1, 1, 1, 1),
        weights: Float32Array.of(1, 0, 0, Infinity)
      },
      startNodes: Uint32Array.of(0),
      startCosts: Float32Array.of(2 ** 24),
      costPrecision
    };
    const {distances} = computeNetworkRoutingDistances('authoritative', routing);
    // Binary32 ties both 2^24+1 and 2^24+0 at 2^24. The weight, not that rounded sum,
    // selects slot 1; the next zero-weight parallel slot 2 loses by its larger CSR slot.
    expect(Array.from(distances)).toEqual([2 ** 24, 2 ** 24]);
    const result = computeNetworkRoutingPredecessorsWithStatus(routing, distances);
    expect(Array.from(result.predecessors)).toEqual([0xffffffff, 0]);
    expect(result).toHaveProperty('predecessorEdges', Uint32Array.of(0xffffffff, 1));
  });

  test('retains strict-parent slots while preferring the smallest parent over its edge weight', () => {
    const routing: CPUNetworkRoutingInput = {
      topology: {
        offsets: Uint32Array.of(0, 5, 6, 6),
        neighbors: Uint32Array.of(2, 2, 2, 2, 2, 2),
        weights: Float32Array.of(Infinity, -1, 2, 2, NaN, 1)
      },
      startNodes: Uint32Array.of(0, 1),
      startCosts: Float32Array.of(0, 1)
    };
    const {distances} = computeNetworkRoutingDistances('authoritative', routing);
    expect(Array.from(distances)).toEqual([0, 1, 2]);
    const result = computeNetworkRoutingPredecessorsWithStatus(routing, distances);
    expect(Array.from(result.predecessors)).toEqual([0xffffffff, 0xffffffff, 0]);
    expect(result).toHaveProperty('predecessorEdges', Uint32Array.of(0xffffffff, 0xffffffff, 2));
  });
});

describe('FRM-03: shared routing precision', () => {
  for (const maximumHops of [Infinity, 3]) {
    test(`inclusive cutoff and predecessors with maximumHops=${maximumHops}`, () => {
      // IEEE binary32: .1f + .4f = .5000000074505806 in binary64,
      // rounded once to binary32 = .5, so only per-edge rounding admits node 2.
      const roundedInput: CPUNetworkRoutingInput = {
        ...input,
        costPrecision: 'float32-per-edge'
      };
      const rounded = computeNetworkRoutingDistances(
        'precision-test',
        roundedInput,
        1,
        maximumHops
      );
      expect(Number.isFinite(rounded.distances[2])).toBe(true);
      expect(rounded.distances[2]).toBe(0.5);
      expect(computeNetworkRoutingPredecessorsWithStatus(roundedInput, rounded.distances)).toEqual({
        predecessors: Uint32Array.of(0xffffffff, 0, 1),
        predecessorEdges: Uint32Array.of(0xffffffff, 0, 1),
        converged: true
      });
      const precise = computeNetworkRoutingDistances(
        'precision-test',
        {...input, costPrecision: 'float64'},
        1,
        maximumHops
      );
      expect(precise.distances[2]).toBe(Infinity);
      expect(
        computeNetworkRoutingDistances('precision-test', input, 1, maximumHops).distances
      ).toEqual(precise.distances);
    });
  }

  test('rejects an unknown precision even with no searches or predecessor edges', () => {
    const invalid = {...input, costPrecision: 'float16'} as unknown as CPUNetworkRoutingInput;
    expect(() => computeNetworkRoutingDistances('precision-test', invalid)).toThrow(
      /precision-test.*costPrecision.*float32-per-edge.*float64.*float16/
    );
    expect(() => computeNetworkRoutingPredecessorsWithStatus(invalid, new Float64Array(3))).toThrow(
      /costPrecision/
    );
  });
});

function byteWindow(words: Uint32Array): Uint8Array {
  // Nonzero, unaligned offset also exercises the exact byteLength rather than the allocation.
  const storage = new Uint8Array(words.byteLength + 9).fill(0xff);
  storage.set(new Uint8Array(words.buffer, words.byteOffset, words.byteLength), 5);
  return storage.subarray(5, 5 + words.byteLength);
}

test('EGR-01: all public word and explicit byte readers preserve 300', () => {
  const fixtures = [
    {
      words: new Uint32Array(112),
      readWords: network.decodeGPUNetworkStatisticsWords,
      readBytes: network.decodeGPUNetworkStatisticsFromBytes
    },
    {
      words: new Uint32Array(8),
      readWords: network.decodeGPUNetworkCoarseningSummaryWords,
      readBytes: network.decodeGPUNetworkCoarseningSummaryFromBytes
    },
    {
      words: new Uint32Array(4),
      readWords: network.decodeGPUNetworkSubgraphFilterCountsWords,
      readBytes: network.decodeGPUNetworkSubgraphFilterCountsFromBytes
    }
  ];
  for (const fixture of fixtures) {
    fixture.words[0] = 300;
    const bytes = byteWindow(fixture.words);
    expect(fixture.readWords(fixture.words).liveNodeCount).toBe(300);
    expect(fixture.readWords(Array.from(fixture.words)).liveNodeCount).toBe(300);
    expect(fixture.readBytes(bytes).liveNodeCount).toBe(300);
    expect(fixture.readBytes({bytes}).liveNodeCount).toBe(300);
    expect(() => fixture.readWords(Uint32Array.of(300))).toThrow();
    expect(() => fixture.readBytes(bytes.subarray(1))).toThrow();
  }
});

test('UNW-07: public exact histogram reader derives 64 bins from its bounded view', () => {
  // Directed edges 0->1, 0->2: out-degree [2,0,0], in-degree [0,1,1].
  const storage = new Uint32Array(16 + 3 * 64 + 12);
  const words = storage.subarray(0, 16 + 3 * 64);
  words.set([2, 0, 1], 16);
  words.set([1, 2, 0], 16 + 64);
  words.set([0, 2, 1], 16 + 128);
  const exact = network.decodeGPUNetworkStatisticsWords(words);
  expect(exact.outDegreeHistogram.length).toBe(64);
  expect(exact.inDegreeHistogram.length).toBe(64);
  expect(exact.totalDegreeHistogram.length).toBe(64);
  expect(exact.outDegreeHistogram.slice(0, 3)).toEqual([2, 0, 1]);
  expect(exact.inDegreeHistogram.slice(0, 3)).toEqual([1, 2, 0]);
  expect(exact.totalDegreeHistogram.slice(0, 3)).toEqual([0, 2, 1]);
  expect(() => network.decodeGPUNetworkStatisticsWords(storage, {degreeBinCount: 64})).toThrow(
    /length/
  );
  expect(() => network.decodeGPUNetworkStatisticsWords(words, {degreeBinCount: NaN})).toThrow();
});

test('CMP-06: public adapters accompany the directed demand-major cost matrix', () => {
  expect(network).toHaveProperty('addNetworkCatchmentAdapter');
  expect(allocation).toHaveProperty('addNetworkLocationAllocationAdapter');
  // Hand-worked facility totals are 3+1=4 and 10+1=11.
  const matrix = network.computeNetworkCostMatrixOnCPU(
    {
      topology: {
        offsets: Uint32Array.of(0, 2, 4, 4, 4),
        neighbors: Uint32Array.of(2, 3, 2, 3),
        weights: Float32Array.of(3, 10, 1, 1)
      },
      startNodes: Uint32Array.of(0, 1),
      targetNodes: Uint32Array.of(2, 3)
    },
    {rowCount: 2}
  );
  expect(Array.from(matrix.costs)).toEqual([3, 10, 1, 1]);
  const result = allocation.computeLocationAllocationOnCPU(
    {costs: matrix.costs},
    {facilityCount: 2, demandCount: 2, maximumFacilityCount: 1, operation: 'p-median'}
  );
  expect(Array.from(result.selectedFacilities)).toEqual([1, 0]);
  expect(Number.isFinite(result.objective[0])).toBe(true);
  expect(result.objective[0]).toBe(4);
});

test('CMP-06: exported adapters preserve identities, direction, horizon and convergence', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const floats = (name: string, length: number) =>
    createTransientView(graph, name, 'float32', length);
  const integers = (name: string, length: number) =>
    createTransientView(graph, name, 'uint32', length);
  try {
    const matrix: network.GPUNetworkCostMatrixResult = {
      values: floats('matrix-values', 4),
      rowCount: 2,
      columnCount: 2,
      startNodes: integers('matrix-starts', 2),
      targetNodes: integers('matrix-targets', 2),
      startsPerRow: 1,
      direction: 'start-to-target',
      converged: integers('matrix-converged', 1),
      costLimit: floats('matrix-horizon', 1)
    };
    const allocationProps: allocation.GPUNetworkLocationAllocationAdapterProps = {
      matrix,
      facilityAxis: 'columns',
      trips: 'demand-to-facility',
      maximumFacilityCount: 1,
      maximumSwapIterations: 0,
      operation: 'p-median',
      output: {
        selectedFacilities: integers('selected', 2),
        assignments: integers('assignments', 2),
        objective: floats('objective', 1),
        status: integers('status', 7)
      }
    };
    const adapted = allocation.addNetworkLocationAllocationAdapter(graph, allocationProps);
    expect(adapted.facilityNodes).toBe(matrix.targetNodes);
    expect(adapted.contributor.props.costs).toBe(matrix.values);
    expect(adapted.contributor.props.costsComplete).toBe(matrix.converged);
    expect(adapted.contributor.props.costLimit).toBe(matrix.costLimit);
    const catchmentProps: network.GPUNetworkCatchmentAdapterProps = {
      id: 'catchment',
      matrix,
      facilityAxis: 'columns',
      trips: 'demand-to-facility',
      supply: floats('supply', 2),
      demand: floats('demand', 2),
      parameters: floats('parameters', 4),
      output: floats('catchment', 2),
      complete: integers('catchment-complete', 1)
    };
    const catchment = network.addNetworkCatchmentAdapter(graph, catchmentProps);
    expect(catchment.contributor.props.costsComplete).toBe(matrix.converged);
    expect(catchment.contributor.props.costLimit).toBe(matrix.costLimit);
    expect(catchment.contributor.props.status?.complete).toBe(catchmentProps.complete);
    expect(catchment.matrix.values).toBe(matrix.values);
    expect(catchment.matrix.startNodes).toBe(matrix.startNodes);
    expect(catchment.matrix.targetNodes).toBe(matrix.targetNodes);
    expect(catchment.matrix.direction).toBe('start-to-target');
    expect(catchment.matrix.costLimit).toBe(matrix.costLimit);
    expect(catchment.matrix.converged).toBe(matrix.converged);
  } finally {
    device.destroy();
  }
});

test('OUT-10: public path reader separates found, stored and resolved', () => {
  expect(network).toHaveProperty('decodeGPUNetworkPathExtractionOutput');
  const truncated = network.decodeGPUNetworkPathExtractionOutput({
    ids: Uint32Array.of(0),
    pathFound: Uint32Array.of(1, 1),
    lineOffsets: Uint32Array.of(0, 2, 3),
    count: Uint32Array.of(1)
  });
  expect(truncated.map(path => [path.found, path.fullyStored, path.nodes.length])).toEqual([
    [true, false, 0],
    [true, false, 0]
  ]);
  for (const edgeRow of [0, 0xffffffff]) {
    const [path] = network.decodeGPUNetworkPathExtractionOutput({
      ids: Uint32Array.of(0, 1),
      pathFound: Uint32Array.of(1),
      lineOffsets: Uint32Array.of(0, 2),
      count: Uint32Array.of(2),
      edges: {
        ids: Uint32Array.of(0xffffffff),
        edgeRows: Uint32Array.of(edgeRow),
        lineOffsets: Uint32Array.of(0, 1),
        count: Uint32Array.of(1)
      }
    });
    expect(path.edges?.ids[0]).toBe(0xffffffff);
    expect(path.edges?.resolutionValidity[0]).toBe(Number(edgeRow === 0));
  }
});

test('OUT-14: public alternative route layout retains a certified zero-route query', () => {
  const options: network.GPUAlternativeRoutesOutputLayoutOptions = {
    queryCount: 2,
    routeCount: 2,
    nodeCount: 2
  };
  expect(network.getGPUAlternativeRoutesOutputLayout(options).vertices.length).toBe(8);
  const result = network.decodeGPUAlternativeRoutes(
    {
      vertices: Uint32Array.of(
        0,
        1,
        0xffffffff,
        0xffffffff,
        0xffffffff,
        0xffffffff,
        0xffffffff,
        0xffffffff
      ),
      edges: Uint32Array.of(
        0,
        0xffffffff,
        0xffffffff,
        0xffffffff,
        0xffffffff,
        0xffffffff,
        0xffffffff,
        0xffffffff
      ),
      lengths: Uint32Array.of(2, 0, 0, 0),
      costs: Float32Array.of(1, Infinity, Infinity, Infinity),
      counts: Uint32Array.of(1, 0),
      complete: Uint32Array.of(1, 1)
    },
    options
  );
  expect(result.queries[1]).toEqual({count: 0, routes: [], complete: true});
  expect(result.queries[0].routes[0]).toEqual({length: 2, cost: 1, vertices: [0, 1], edges: [0]});
});

test('public result and readback aliases name the producer declarations', () => {
  expectTypeOf<network.NetworkReadbackBytes>().toEqualTypeOf<
    Uint8Array | ArrayBuffer | {bytes: Uint8Array | ArrayBuffer}
  >();
  expectTypeOf<network.GPUAlternativeRoutesReadback>().toEqualTypeOf<
    Parameters<typeof network.decodeGPUAlternativeRoutes>[0]
  >();
  expectTypeOf<network.CPUNetworkTimeRoutingResult>().toEqualTypeOf<
    ReturnType<typeof network.computeNetworkTimeRoutingOnCPU>
  >();
  expectTypeOf<network.CPURadiationFlowModelResult>().toEqualTypeOf<
    ReturnType<typeof network.computeRadiationFlowModelOnCPU>
  >();
  expectTypeOf<network.CPUNetworkTravelTimePercentilesResult>().toEqualTypeOf<
    ReturnType<typeof network.computeNetworkTravelTimePercentilesOnCPU>
  >();
  expectTypeOf<GPUNetworkIsochronesOutput>().toEqualTypeOf<
    NonNullable<
      import('@luma-contributors/analysis/gpu-network').GPUNetworkIsochronesProps['output']
    >
  >();
});

describe('DGN-10: zero queries and zero-node CSR are separate controls', () => {
  test('S3: predecessor validation rejects malformed empty CSR', () => {
    for (const malformed of [
      {offsets: Uint32Array.of(2), neighbors: new Uint32Array(), weights: new Float32Array()},
      {offsets: Uint32Array.of(0), neighbors: Uint32Array.of(0), weights: Float32Array.of(1)}
    ]) {
      expect(() =>
        computeNetworkRoutingPredecessorsWithStatus(
          {topology: malformed, startNodes: new Uint32Array()},
          new Float64Array()
        )
      ).toThrow(/computeNetworkRoutingPredecessorsWithStatus.*zero-node topology/);
    }
  });
  test('both route producers accept zero queries on a valid one-node graph', () => {
    const emptyQueries = {
      topology: {
        offsets: Uint32Array.of(0, 0),
        neighbors: new Uint32Array(),
        weights: new Float32Array()
      },
      sources: new Uint32Array(),
      targets: new Uint32Array()
    };
    expect(network.computeAlternativeRoutesOnCPU(emptyQueries, {routeCount: 2}).counts.length).toBe(
      0
    );
    expect(computeKShortestPathsOnCPU(emptyQueries, {routeCount: 2}).counts.length).toBe(0);
  });
  test('valid zero-node CSR returns empty converged distances and predecessors', () => {
    const emptyInput = {
      topology: {
        offsets: Uint32Array.of(0),
        neighbors: new Uint32Array(),
        weights: new Float32Array()
      },
      startNodes: new Uint32Array()
    };
    expect(validateNetworkRoutingArrays(emptyInput.topology)).toBe(0);
    for (const maximumHops of [Infinity, 0, 2]) {
      const result = computeNetworkRoutingDistances('empty-test', emptyInput, 1, maximumHops);
      expect(result.distances.length).toBe(0);
      expect(result.converged).toBe(true);
      expect(result.scannedEdges).toBe(0);
      expect(computeNetworkRoutingPredecessorsWithStatus(emptyInput, result.distances)).toEqual({
        predecessors: new Uint32Array(),
        predecessorEdges: new Uint32Array(),
        converged: true
      });
    }
    for (const malformed of [
      {...emptyInput.topology, offsets: new Uint32Array()},
      {...emptyInput.topology, offsets: Uint32Array.of(1)},
      {...emptyInput.topology, neighbors: Uint32Array.of(0), weights: Float32Array.of(1)}
    ])
      expect(() => validateNetworkRoutingArrays(malformed)).toThrow(/topology/);
  });
  test('resident ordinary offsets retain their clamped interpretation', () => {
    const result = computeNetworkRoutingDistances('clamp-test', {
      topology: {
        offsets: Uint32Array.of(0, 20, 1),
        neighbors: Uint32Array.of(1),
        weights: Float32Array.of(2)
      },
      startNodes: Uint32Array.of(0)
    });
    expect(Array.from(result.distances)).toEqual([0, 2]);
  });
});

test('UNW-15: public facility preflight and exact SCC limits', () => {
  expect(network.GPU_NETWORK_STRUCTURE_EXACT_MAXIMUM_NODE_COUNT).toBe(1024);
  expect(network.GPU_NETWORK_STRUCTURE_EXACT_MAXIMUM_EDGE_COUNT).toBe(65536);
  expect(
    network.getNetworkFacilitySearchLimits({
      operation: 'capacitated-p-median',
      facilityCount: 1,
      demandCount: 65
    })
  ).toMatchObject({
    candidateCount: 1,
    maximumDemandCount: 64,
    limitingDimension: 'demandCount',
    supported: false
  });
});
