// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeNetworkRationalAccessOnCPU,
  CPUNetworkRationalAccessOperation
} from '../../../src/gpu-network/network-optimization/cpu-network-rational-access-operation';
import {
  GPUNetworkRationalAccessOperation,
  type NetworkRationalAccessOperationProps
} from '../../../src/gpu-network/network-optimization/gpu-network-rational-access-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';

const PROPS: NetworkRationalAccessOperationProps = {
  id: 'rational-test',
  originCount: 2,
  facilityCount: 2,
  costs: 'costs',
  demand: 'demand',
  supply: 'supply',
  parameters: 'parameters',
  output: {
    assignments: 'assignments',
    scores: 'scores',
    unservedDemand: 'unserved',
    summary: 'summary'
  }
};
const INPUT = {
  costs: Float32Array.of(0, 2, 10, 0),
  demand: Float32Array.of(10, 2),
  supply: Float32Array.of(2, 4),
  parameters: Float32Array.of(1, 1, 1, 1, 50, 0)
};

it('matches a hand-computed simultaneous allocation and reports an integer rounding stall', () => {
  // Initial loads (10,2) give origin 0 costs (5,2.5). Its equalizing shift is 10/3,
  // floored to 3. New loads (7,5) give costs (3.5,3.25), so every further shift floors to 0.
  const result = computeNetworkRationalAccessOnCPU(INPUT, {
    maximumCycles: 3,
    initialLimitCycles: 0
  });
  expect(Array.from(result.assignments)).toEqual([7, 3, 0, 2]);
  expect(Array.from(result.scores)).toEqual([Math.fround(3.425), 1.25]);
  expect(Array.from(result.unservedDemand)).toEqual([0, 0]);
  expect(Array.from(result.summary.subarray(0, 3))).toEqual([1, 0, 3]);
  expect(new Float32Array(result.summary.slice(3).buffer)[0]).toBe(0.25);
});

it('matches the closed-form two-facility fractional equilibrium under an absolute shift cap', () => {
  const demand = 10;
  const firstSupply = 2;
  const secondSupply = 4;
  // x/2 = 2 + (10-x)/4, so x=6, y=4 and the common cost is 3.
  const firstAllocation = (2 + demand / secondSupply) / (1 / firstSupply + 1 / secondSupply);
  const result = computeNetworkRationalAccessOnCPU(
    {
      costs: Float32Array.of(0, 2),
      demand: Float32Array.of(demand),
      supply: Float32Array.of(firstSupply, secondSupply),
      parameters: Float32Array.of(1, 1, 1.5, 1.5, 10, 0)
    },
    {demandMode: 'fractional', stepMode: 'absolute', initialLimitCycles: 0}
  );
  expect(Array.from(result.assignments)).toEqual([firstAllocation, demand - firstAllocation]);
  expect(Array.from(result.scores)).toEqual([3]);
  expect(Array.from(result.summary.subarray(0, 3))).toEqual([1, 1, 3]);
  expect(result.summary[3]).toBe(0);
});

it('applies the simultaneous inflow limit with ties-to-even and recomputes final scores', () => {
  const result = computeNetworkRationalAccessOnCPU(
    {
      costs: new Float32Array(4),
      demand: Float32Array.of(3, 3),
      supply: Float32Array.of(1, 3),
      parameters: Float32Array.of(1, 1, 1, 1, 50, 0)
    },
    {maximumCycles: 1, initialLimitCycles: 1}
  );
  // Each wants to shift 3; combined inflow 6 divided by destination capacity 3
  // limits each shift to 1.5, rounded to the even integer 2.
  expect(Array.from(result.assignments)).toEqual([1, 2, 1, 2]);
  expect(Array.from(result.scores)).toEqual([Math.fround(14 / 9), Math.fround(14 / 9)]);
  expect(Array.from(result.unservedDemand)).toEqual([0, 0]);
  expect(Array.from(result.summary.subarray(0, 3))).toEqual([1, 0, 1]);
  // Final costs 2 and 4/3 give the remaining gap 2/3.
  expect(new Float32Array(result.summary.slice(3).buffer)[0]).toBe(Math.fround(2 / 3));
});

it('honors travel/supply normalization, nonintegral demand and the decaying step floor', () => {
  const normalized = computeNetworkRationalAccessOnCPU(
    {
      costs: Float32Array.of(0, 2),
      demand: Float32Array.of(10),
      supply: Float32Array.of(2, 4),
      parameters: Float32Array.of(2, 3, 1, 1, 1, 1e-6)
    },
    {demandMode: 'fractional', initialLimitCycles: 0}
  );
  // x/6 = 1 + (10-x)/12: x=22/3 and the common cost is 11/9.
  expect(Array.from(normalized.assignments)).toEqual([Math.fround(22 / 3), Math.fround(8 / 3)]);
  expect(normalized.scores[0]).toBe(Math.fround(11 / 9));
  expect(normalized.summary[1]).toBe(1);
  const nonintegral = computeNetworkRationalAccessOnCPU(
    {
      costs: Float32Array.of(0, 0),
      demand: Float32Array.of(2.5),
      supply: Float32Array.of(1, 1),
      parameters: Float32Array.of(1, 1, 1, 1, 1, 0)
    },
    {demandMode: 'fractional', initialLimitCycles: 0}
  );
  expect(Array.from(nonintegral.assignments)).toEqual([1.25, 1.25]);
  expect(nonintegral.summary[1]).toBe(1);
  const decayed = computeNetworkRationalAccessOnCPU(
    {
      costs: Float32Array.of(0, 0),
      demand: Float32Array.of(10),
      supply: Float32Array.of(1, 1),
      parameters: Float32Array.of(1, 1, 0.2, 0.1, 1, 0)
    },
    {maximumCycles: 3, initialLimitCycles: 0}
  );
  // The fractional caps are .2, .1, .1: shifts 2, 1, 1 leave loads 6 and 4.
  expect(Array.from(decayed.assignments)).toEqual([6, 4]);
  expect(decayed.scores[0]).toBe(Math.fround(5.2));
  expect(new Float32Array(decayed.summary.slice(3).buffer)[0]).toBe(2);
  expect(Array.from(decayed.summary.subarray(0, 3))).toEqual([1, 0, 3]);
});

it('preserves demand and missing scores for unreachable and zero-demand origins', () => {
  const result = computeNetworkRationalAccessOnCPU(
    {
      costs: Float32Array.of(Infinity, Infinity, 0, 1, 2, 1),
      demand: Float32Array.of(4, 0, 3),
      supply: Float32Array.of(2, 3),
      parameters: Float32Array.of(2, 1, 1, 1, 1, 0)
    },
    {maximumCycles: 0}
  );
  expect(Array.from(result.assignments)).toEqual([0, 0, 0, 0, 0, 3]);
  expect(Array.from(result.unservedDemand)).toEqual([4, 0, 0]);
  expect(Number.isNaN(result.scores[0])).toBe(true);
  expect(Number.isNaN(result.scores[1])).toBe(true);
  expect(result.scores[2]).toBe(1.5);
  expect(Array.from(result.summary.subarray(0, 3))).toEqual([1, 0, 0]);
});

it('returns documented invalid-data summary and partial initialization; rejects inconsistent arrays', () => {
  const invalid = computeNetworkRationalAccessOnCPU({
    ...INPUT,
    costs: Float32Array.of(0, 2, NaN, 0)
  });
  expect(Array.from(invalid.summary)).toEqual([0, 0, 0, 0]);
  expect(Array.from(invalid.assignments)).toEqual([10, 0, 0, 0]);
  expect(Array.from(invalid.unservedDemand)).toEqual([0, 2]);
  for (const demand of [
    Float32Array.of(0.5, 2),
    Float32Array.of(16777216, 2),
    Float32Array.of(-1, 2)
  ])
    expect(computeNetworkRationalAccessOnCPU({...INPUT, demand}).summary[0]).toBe(0);
  expect(
    computeNetworkRationalAccessOnCPU({...INPUT, supply: Float32Array.of(0, 1)}).summary[0]
  ).toBe(0);
  expect(
    computeNetworkRationalAccessOnCPU({...INPUT, parameters: Float32Array.of(0, 1, 1, 1, 1, 0)})
      .summary[0]
  ).toBe(0);
  expect(() => computeNetworkRationalAccessOnCPU({...INPUT, costs: new Float32Array(3)})).toThrow(
    /costs/
  );
  expect(() =>
    computeNetworkRationalAccessOnCPU({...INPUT, parameters: new Float32Array(5)})
  ).toThrow(/parameters/);
  expect(() =>
    computeNetworkRationalAccessOnCPU({
      ...INPUT,
      demand: new Float32Array(0),
      costs: new Float32Array(0)
    })
  ).toThrow(/demand/);
});

it('validates both operation constructions and removes GPU-only caps on CPU', () => {
  const shapes = {
    costs: {format: 'float32' as const, length: 4},
    demand: {format: 'float32' as const, length: 2},
    supply: {format: 'float32' as const, length: 2},
    parameters: {format: 'float32' as const, length: 6}
  };
  expect(() =>
    constructNetworkOperationContributor(new GPUNetworkRationalAccessOperation(PROPS), shapes)
  ).not.toThrow();
  expect(() =>
    constructNetworkOperationContributor(
      new GPUNetworkRationalAccessOperation({...PROPS, maximumCycles: -1}),
      shapes
    )
  ).toThrow(/rational-test.*maximumCycles/);
  expect(() =>
    constructNetworkOperationContributor(new GPUNetworkRationalAccessOperation(PROPS), {
      ...shapes,
      parameters: {...shapes.parameters, length: 5}
    })
  ).toThrow(/rational-test.*parameters/);
  expect(() => new CPUNetworkRationalAccessOperation({...PROPS, originCount: 0})).toThrow(
    /rational-test.*originCount/
  );
  expect(() => new CPUNetworkRationalAccessOperation({...PROPS, maximumCycles: -1})).toThrow(
    /rational-test.*maximumCycles/
  );
  expect(
    () =>
      new CPUNetworkRationalAccessOperation({
        ...PROPS,
        originCount: 65,
        facilityCount: 33,
        maximumCycles: 4097
      })
  ).not.toThrow();
  for (const Operation of [GPUNetworkRationalAccessOperation, CPUNetworkRationalAccessOperation])
    expect(() => new Operation({...PROPS, output: {...PROPS.output, scores: 'costs'}})).toThrow(
      /rational-test/
    );
});
