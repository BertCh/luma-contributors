// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  computeNetworkStructureOnCPU,
  CPUNetworkStructureOperation
} from '../../../src/gpu-network/network-structure/cpu-network-structure-operation';
import {
  GPUNetworkStructureOperation,
  type NetworkStructureOperationProps
} from '../../../src/gpu-network/network-structure/gpu-network-structure-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';

const PROPS: NetworkStructureOperationProps = {
  id: 'structure-test',
  nodeCount: 6,
  edgeCount: 13,
  topology: {offsets: 'offsets', neighbors: 'neighbors'},
  reverseEdgeSlots: 'reverse',
  output: {
    strongComponents: 'components',
    bridges: 'bridges',
    articulationPoints: 'articulation',
    biconnectedComponents: 'blocks'
  }
};

it('matches transitive closure on every directed graph with three vertices', () => {
  // Floyd-Warshall is independent of the twin's Tarjan traversal.
  for (let mask = 0; mask < 512; mask++) {
    const offsets = [0];
    const neighbors: number[] = [];
    const reachable = Array.from({length: 3}, (_, source) =>
      Array.from({length: 3}, (_, target) => source === target)
    );
    for (let source = 0; source < 3; source++) {
      for (let target = 0; target < 3; target++)
        if (mask & (1 << (source * 3 + target))) {
          neighbors.push(target);
          reachable[source][target] = true;
        }
      offsets.push(neighbors.length);
    }
    for (let middle = 0; middle < 3; middle++)
      for (let source = 0; source < 3; source++)
        for (let target = 0; target < 3; target++)
          reachable[source][target] ||= reachable[source][middle] && reachable[middle][target];
    const expected = reachable.map((row, vertex) =>
      row.findIndex((canReach, target) => canReach && reachable[target][vertex])
    );
    expect(
      Array.from(
        computeNetworkStructureOnCPU({
          offsets: Uint32Array.from(offsets),
          neighbors: Uint32Array.from(neighbors)
        }).strongComponents
      )
    ).toEqual(expected);
    const bounded = computeNetworkStructureOnCPU(
      {
        offsets: Uint32Array.from(offsets),
        neighbors: Uint32Array.from(neighbors)
      },
      {maximumIterations: 32}
    );
    expect(Array.from(bounded.strongComponents)).toEqual(expected);
    expect(bounded.strongComponentsStatus.converged).toEqual(Uint32Array.of(1));
  }
});

it('labels hand-derived triangle, bridge, parallel-edge block, loop and isolated vertex', () => {
  const result = computeNetworkStructureOnCPU({
    offsets: new Uint32Array([0, 2, 4, 7, 10, 13, 13]),
    neighbors: new Uint32Array([1, 2, 0, 2, 1, 0, 3, 2, 4, 4, 3, 3, 4]),
    reverseEdgeSlots: new Uint32Array([2, 5, 0, 4, 3, 1, 7, 6, 10, 11, 8, 9, 12])
  });
  expect(Array.from(result.strongComponents)).toEqual([0, 0, 0, 0, 0, 5]);
  // Removing 2--3 disconnects the graph. Neither parallel edge disconnects 3--4.
  expect(Array.from(result.bridges)).toEqual([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]);
  expect(Array.from(result.articulationPoints)).toEqual([0, 0, 1, 1, 0, 0]);
  expect(Array.from(result.biconnectedComponents)).toEqual([
    0, 0, 0, 0, 0, 0, 6, 6, 8, 8, 8, 8, 12
  ]);
});

it('handles isolated vertices and ignores invalid neighbor slots', () => {
  const result = computeNetworkStructureOnCPU({
    offsets: new Uint32Array([0, 1, 1]),
    neighbors: new Uint32Array([99]),
    reverseEdgeSlots: new Uint32Array([0xffffffff])
  });
  expect(Array.from(result.strongComponents)).toEqual([0, 1]);
  expect(Array.from(result.biconnectedComponents)).toEqual([0xffffffff]);
  const star = computeNetworkStructureOnCPU({
    offsets: new Uint32Array([0, 2, 3, 4]),
    neighbors: new Uint32Array([1, 2, 0, 0]),
    reverseEdgeSlots: new Uint32Array([2, 3, 0, 1])
  });
  expect(Array.from(star.articulationPoints)).toEqual([1, 0, 0]);
  expect(Array.from(star.bridges)).toEqual([1, 1, 1, 1]);
  expect(Array.from(star.biconnectedComponents)).toEqual([0, 1, 0, 1]);
});

it('ignores an unused empty reverse table when only SCC output is requested', async () => {
  const props: NetworkStructureOperationProps = {
    id: 'structure-unused-reverse',
    nodeCount: 2,
    edgeCount: 1,
    topology: {offsets: 'offsets', neighbors: 'neighbors'},
    reverseEdgeSlots: 'reverse',
    output: {strongComponents: 'components'}
  };
  const offsets = new Uint32Array([0, 1, 1]);
  const neighbors = new Uint32Array([1]);
  const reverseEdgeSlots = new Uint32Array(0);
  // One directed edge 0 -> 1 has no return path. Each vertex is its own SCC.
  const expected = new Uint32Array([0, 1]);
  expect(computeNetworkStructureOnCPU({offsets, neighbors}).strongComponents).toEqual(expected);
  expect(() =>
    constructNetworkOperationContributor(new GPUNetworkStructureOperation(props), {
      offsets: {format: 'uint32', length: offsets.length},
      neighbors: {format: 'uint32', length: neighbors.length},
      reverseEdgeSlots: {format: 'uint32', length: reverseEdgeSlots.length}
    })
  ).not.toThrow();
  const makeInput = (values: Uint32Array): GPUCPUVector => ({
    format: 'uint32',
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const operation = new CPUNetworkStructureOperation(props);
  const result = await operation.execute({
    inputs: {
      offsets: makeInput(offsets),
      neighbors: makeInput(neighbors),
      reverseEdgeSlots: makeInput(reverseEdgeSlots)
    }
  });
  expect(Object.keys(result)).toEqual(['components']);
  expect(result['components']).toEqual(expected);
});

it('returns every declared SCC status graph value from the exact CPU traversal', async () => {
  const props = {
    nodeCount: 4,
    edgeCount: 3,
    topology: {offsets: 'offsets', neighbors: 'neighbors'},
    output: {strongComponents: 'labels'},
    strongComponentsStatus: {
      converged: 'finished',
      iterationCount: 'steps',
      iterationLimitReached: 'limited'
    }
  };
  const vector = (values: Uint32Array): GPUCPUVector => ({
    format: 'uint32',
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const operation = new CPUNetworkStructureOperation(props);
  const rows = await operation.execute({
    inputs: {
      offsets: vector(Uint32Array.of(0, 1, 2, 3, 3)),
      neighbors: vector(Uint32Array.of(1, 2, 0))
    }
  });
  expect(Object.keys(rows).sort()).toEqual(Object.keys(operation.outputs).sort());
  expect(rows['labels']).toEqual(Uint32Array.of(0, 0, 0, 3));
  expect(rows['finished']).toEqual(Uint32Array.of(1));
  expect(rows['steps']).toEqual(Uint32Array.of(1));
  expect(rows['limited']).toEqual(Uint32Array.of(0));
});

it('honors an explicit parallel SCC step budget and invalidates exhausted labels', async () => {
  const vector = (values: Uint32Array): GPUCPUVector => ({
    format: 'uint32',
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  for (const maximumIterations of [1, 32]) {
    const operation = new CPUNetworkStructureOperation({
      mode: 'parallel',
      strongComponentsMaximumIterations: maximumIterations,
      nodeCount: 4,
      edgeCount: 3,
      topology: {offsets: 'offsets', neighbors: 'neighbors'},
      output: {strongComponents: 'labels'},
      strongComponentsStatus: {
        converged: 'finished',
        iterationCount: 'steps',
        iterationLimitReached: 'limited'
      }
    });
    const rows = await operation.execute({
      inputs: {
        offsets: vector(Uint32Array.of(0, 1, 2, 3, 3)),
        neighbors: vector(Uint32Array.of(1, 2, 0))
      }
    });
    const finished = maximumIterations > 1;
    expect(rows['labels']).toEqual(
      finished ? Uint32Array.of(0, 0, 0, 3) : new Uint32Array(4).fill(0xffffffff)
    );
    expect(rows['finished']).toEqual(Uint32Array.of(Number(finished)));
    expect(rows['limited']).toEqual(Uint32Array.of(Number(!finished)));
    expect((rows['steps'] as Uint32Array)[0]).toBeLessThanOrEqual(maximumIterations);
  }
});

it('retains exact resilience outputs when the SCC phase budget is exhausted', () => {
  const result = computeNetworkStructureOnCPU(
    {
      offsets: Uint32Array.of(0, 1, 2, 2),
      neighbors: Uint32Array.of(1, 0),
      reverseEdgeSlots: Uint32Array.of(1, 0)
    },
    {maximumIterations: 1}
  );
  expect(result.strongComponents).toEqual(new Uint32Array(3).fill(0xffffffff));
  expect(result.strongComponentsStatus.iterationLimitReached).toEqual(Uint32Array.of(1));
  expect(result.bridges).toEqual(Uint32Array.of(1, 1));
  expect(result.articulationPoints).toEqual(Uint32Array.of(0, 0, 0));
});

it('validates inconsistent arrays and both operation constructions', () => {
  expect(() =>
    computeNetworkStructureOnCPU({offsets: new Uint32Array([0, 2]), neighbors: new Uint32Array(1)})
  ).toThrow(/offsets/);
  expect(() =>
    computeNetworkStructureOnCPU({
      offsets: new Uint32Array([0, 0]),
      neighbors: new Uint32Array(1),
      reverseEdgeSlots: new Uint32Array(0)
    })
  ).toThrow(/reverseEdgeSlots/);
  expect(() => new CPUNetworkStructureOperation({...PROPS, reverseEdgeSlots: undefined})).toThrow(
    /structure-test.*reverseEdgeSlots/
  );
  expect(() => new CPUNetworkStructureOperation({...PROPS, nodeCount: 0})).toThrow(
    /structure-test.*nodeCount/
  );
  const shapes = {
    offsets: {format: 'uint32' as const, length: 7},
    neighbors: {format: 'uint32' as const, length: 13},
    reverseEdgeSlots: {format: 'uint32' as const, length: 13}
  };
  expect(() =>
    constructNetworkOperationContributor(
      new GPUNetworkStructureOperation({...PROPS, reverseEdgeSlots: undefined}),
      shapes
    )
  ).toThrow(/structure-test.*reverseEdgeSlots/);
  expect(() =>
    constructNetworkOperationContributor(new GPUNetworkStructureOperation(PROPS), shapes)
  ).not.toThrow();
  for (const Operation of [GPUNetworkStructureOperation, CPUNetworkStructureOperation])
    expect(() => new Operation({...PROPS, output: {strongComponents: 'offsets'}})).toThrow(
      /structure-test/
    );
  expect(
    () => new CPUNetworkStructureOperation({...PROPS, nodeCount: 1025, edgeCount: 65537})
  ).not.toThrow();
});
