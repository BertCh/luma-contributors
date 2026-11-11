// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeNetworkLouvainOnCPU,
  CPUNetworkLouvainOperation
} from '../../../src/gpu-network/network-optimization/cpu-network-louvain-operation';
import {
  GPUNetworkLouvainOperation,
  type NetworkLouvainOperationProps
} from '../../../src/gpu-network/network-optimization/gpu-network-louvain-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';

const PROPS: NetworkLouvainOperationProps = {
  id: 'louvain-test',
  nodeCount: 4,
  weights: 'weights',
  parameters: 'parameters',
  output: {communities: 'communities', objectiveTrace: 'trace', summary: 'summary'}
};

function buildWeights(nodeCount: number, directed: boolean): Float32Array {
  const weights = new Float32Array(nodeCount ** 2);
  for (let source = 0; source < nodeCount; source++)
    for (let target = 0; target < nodeCount; target++) {
      if (
        source !== target &&
        Math.floor(source / (nodeCount / 2)) === Math.floor(target / (nodeCount / 2))
      )
        weights[source * nodeCount + target] = directed ? 2 + source : 3;
    }
  weights[(nodeCount / 2 - 1) * nodeCount + nodeCount / 2] = 0.25;
  if (!directed) weights[(nodeCount / 2) * nodeCount + nodeCount / 2 - 1] = 0.25;
  weights[0] = 1; // Includes a self-loop in the objective oracle.
  return weights;
}

function createObjectiveOracle(weights: Float32Array, nodeCount: number, resolution: number) {
  // Independent Newman modularity-matrix formulation, rather than community strength sums.
  const outgoing = Array.from({length: nodeCount}, (_, source) =>
    Array.from({length: nodeCount}, (_, target) => weights[source * nodeCount + target]).reduce(
      (sum, value) => sum + value,
      0
    )
  );
  const incoming = Array.from({length: nodeCount}, (_, target) =>
    Array.from({length: nodeCount}, (_, source) => weights[source * nodeCount + target]).reduce(
      (sum, value) => sum + value,
      0
    )
  );
  const total = outgoing.reduce((sum, value) => sum + value, 0);
  const matrix = Array.from(
    weights,
    (weight, entry) =>
      (weight -
        (resolution * outgoing[Math.floor(entry / nodeCount)] * incoming[entry % nodeCount]) /
          total) /
      total
  );
  return (labels: ArrayLike<number>) =>
    matrix.reduce(
      (sum, value, entry) =>
        sum + (labels[Math.floor(entry / nodeCount)] === labels[entry % nodeCount] ? value : 0),
      0
    );
}

it('attains the independently enumerated optimum on two strongly separated weighted communities', () => {
  for (const directed of [false, true]) {
    const nodeCount = 6;
    const weights = buildWeights(nodeCount, directed);
    const evaluate = createObjectiveOracle(weights, nodeCount, 1);
    let optimum = -Infinity;
    const labels = new Uint32Array(nodeCount);
    const enumeratePartitions = (vertex: number, largest: number) => {
      if (vertex === nodeCount) {
        optimum = Math.max(optimum, evaluate(labels));
        return;
      }
      for (let community = 0; community <= largest + 1; community++) {
        labels[vertex] = community;
        enumeratePartitions(vertex + 1, Math.max(largest, community));
      }
    };
    enumeratePartitions(1, 0);
    for (const seed of [undefined, 0, 0x9e3779b9, 42]) {
      const result = computeNetworkLouvainOnCPU(
        {weights, parameters: Float32Array.of(1, 1e-5)},
        {nodeCount, directed, seed}
      );
      expect(evaluate(result.communities)).toBeCloseTo(optimum, 12);
      for (let first = 0; first < nodeCount; first++)
        for (let second = 0; second < nodeCount; second++)
          expect(result.communities[first] === result.communities[second]).toBe(
            Math.floor(first / 3) === Math.floor(second / 3)
          );
      const sweeps = result.summary[3];
      expect(result.objectiveTrace[sweeps]).toBe(Math.fround(optimum));
      expect(Array.from(result.summary.subarray(0, 2))).toEqual([1, 1]);
      expect(result.objectiveTrace.subarray(sweeps + 1).every(Number.isNaN)).toBe(true);
      for (let row = 1; row <= sweeps; row++)
        expect(result.objectiveTrace[row]).toBeGreaterThanOrEqual(result.objectiveTrace[row - 1]);
    }
  }
});

it('moves an entire coarsened pair after individual vertex moves become stable', () => {
  const result = computeNetworkLouvainOnCPU(
    {
      weights: Float32Array.of(
        0,
        2,
        1.25,
        1.25,
        2,
        0,
        1.25,
        1.25,
        1.25,
        1.25,
        0,
        2,
        1.25,
        1.25,
        2,
        0
      ),
      parameters: Float32Array.of(1, 1e-5)
    },
    {nodeCount: 4}
  );
  // Every strength is 9/2, total directed weight is 18. Singleton Q=-1/4.
  // The first level forms pairs {0,1} and {2,3}: Q=8/18-1/2=-1/18.
  // An individual move out of a pair lowers Q, but merging both pairs gives Q=1-1=0.
  // Two sweeps stabilize each moving level; a final unchanged level takes one sweep.
  expect(Array.from(result.communities)).toEqual([3, 3, 3, 3]);
  expect(Array.from(result.summary)).toEqual([1, 1, 3, 5]);
  const expectedTrace = [-1 / 4, Math.fround(-1 / 18), Math.fround(-1 / 18), 0, 0, 0];
  expect(Array.from(result.objectiveTrace.subarray(0, 6))).toEqual(expectedTrace);
  expect(result.objectiveTrace.subarray(6).every(Number.isNaN)).toBe(true);
});

it('retains the trace budget and reports nonconvergence after a moving sweep', () => {
  const result = computeNetworkLouvainOnCPU(
    {weights: Float32Array.of(0, 2, 2, 0), parameters: Float32Array.of(1, 0)},
    {nodeCount: 2, maximumLevels: 1, maximumIterations: 1}
  );
  expect(Array.from(result.communities)).toEqual([1, 1]);
  expect(Array.from(result.objectiveTrace)).toEqual([-0.5, 0]);
  expect(Array.from(result.summary)).toEqual([1, 0, 1, 1]);
});

it('honors resolution and the minimum accepted modularity improvement', () => {
  // For one undirected edge, singleton Q=-gamma/2 and merged Q=1-gamma.
  const weights = Float32Array.of(0, 2, 2, 0);
  const lowResolution = computeNetworkLouvainOnCPU(
    {weights, parameters: Float32Array.of(0.5, 0)},
    {nodeCount: 2}
  );
  expect(Array.from(lowResolution.communities)).toEqual([1, 1]);
  expect(lowResolution.objectiveTrace[0]).toBe(-0.25);
  expect(lowResolution.objectiveTrace[lowResolution.summary[3]]).toBe(0.5);
  const highTolerance = computeNetworkLouvainOnCPU(
    {weights, parameters: Float32Array.of(1, 1)},
    {nodeCount: 2}
  );
  expect(Array.from(highTolerance.communities)).toEqual([0, 1]);
  expect(Array.from(highTolerance.summary)).toEqual([1, 1, 1, 1]);
  expect(Array.from(highTolerance.objectiveTrace.subarray(0, 2))).toEqual([-0.5, -0.5]);
});

it('handles zero graphs and invalid weights/parameters with every documented output', () => {
  const zero = computeNetworkLouvainOnCPU(
    {weights: new Float32Array(16), parameters: Float32Array.of(1, 0)},
    {nodeCount: 4}
  );
  expect(Array.from(zero.communities)).toEqual([0, 1, 2, 3]);
  expect(Array.from(zero.summary)).toEqual([1, 1, 0, 0]);
  expect(zero.objectiveTrace[0]).toBe(0);
  expect(zero.objectiveTrace.subarray(1).every(Number.isNaN)).toBe(true);
  for (const weight of [-1, NaN, Infinity, 1]) {
    const weights = new Float32Array(16);
    weights[1] = weight;
    const result = computeNetworkLouvainOnCPU(
      {weights, parameters: Float32Array.of(1, 0)},
      {nodeCount: 4}
    );
    expect(Array.from(result.summary)).toEqual([0, 0, 0, 0]);
    expect(Array.from(result.communities)).toEqual([0, 1, 2, 3]);
    expect(result.objectiveTrace.every(Number.isNaN)).toBe(true);
  }
  expect(
    computeNetworkLouvainOnCPU(
      {weights: new Float32Array(16), parameters: Float32Array.of(-1, 0)},
      {nodeCount: 4}
    ).summary[0]
  ).toBe(0);
});

it('rejects inconsistent arrays and validates both operation constructions', () => {
  expect(() =>
    computeNetworkLouvainOnCPU(
      {weights: new Float32Array(15), parameters: Float32Array.of(1, 0)},
      {nodeCount: 4}
    )
  ).toThrow(/weights/);
  expect(() =>
    computeNetworkLouvainOnCPU(
      {weights: new Float32Array(16), parameters: new Float32Array(1)},
      {nodeCount: 4}
    )
  ).toThrow(/parameters/);
  const shapes = {
    weights: {format: 'float32' as const, length: 16},
    parameters: {format: 'float32' as const, length: 2}
  };
  expect(() =>
    constructNetworkOperationContributor(new GPUNetworkLouvainOperation(PROPS), shapes)
  ).not.toThrow();
  for (const seed of [-1, 0x100000000, 0.5]) {
    expect(() => new CPUNetworkLouvainOperation({...PROPS, seed})).toThrow(/louvain-test.*seed/);
    expect(() =>
      constructNetworkOperationContributor(new GPUNetworkLouvainOperation({...PROPS, seed}), shapes)
    ).toThrow(/louvain-test.*seed/);
  }
  expect(() => new CPUNetworkLouvainOperation({...PROPS, maximumIterations: 0})).toThrow(
    /louvain-test.*maximumIterations/
  );
  expect(() =>
    constructNetworkOperationContributor(
      new GPUNetworkLouvainOperation({...PROPS, maximumIterations: 0}),
      shapes
    )
  ).toThrow(/louvain-test.*maximumIterations/);
  expect(
    () =>
      new CPUNetworkLouvainOperation({
        ...PROPS,
        nodeCount: 33,
        maximumLevels: 33,
        maximumIterations: 65
      })
  ).not.toThrow();
  for (const Operation of [GPUNetworkLouvainOperation, CPUNetworkLouvainOperation])
    expect(
      () => new Operation({...PROPS, output: {...PROPS.output, communities: 'weights'}})
    ).toThrow(/louvain-test/);
});
