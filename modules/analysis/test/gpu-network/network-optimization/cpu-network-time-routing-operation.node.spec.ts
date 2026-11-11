// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeNetworkTimeRoutingOnCPU,
  CPUNetworkTimeRoutingOperation
} from '../../../src/gpu-network/network-optimization/cpu-network-time-routing-operation';
import {
  GPUNetworkTimeRoutingOperation,
  type NetworkTimeRoutingOperationProps
} from '../../../src/gpu-network/network-optimization/gpu-network-time-routing-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';

const PROPS: NetworkTimeRoutingOperationProps = {
  id: 'routing-test',
  stateCount: 5,
  transitionCount: 4,
  stateVertices: 'vertices',
  transitions: 'transitions',
  costs: 'costs',
  endpoints: 'endpoints',
  departure: 'departure',
  output: {arrivals: 'arrivals', predecessorTransitions: 'predecessors', summary: 'summary'}
};

it('matches exhaustive simple-path enumeration on an expanded-state affine network', () => {
  const stateVertices = new Uint32Array([10, 11, 12, 12, 99]);
  const transitions = new Uint32Array([1, 2, 0, 1, 0, 3, 3, 2]);
  const costs = new Float32Array([3, 0.25, 2, 0.5, 20, 0, 1, 0]);
  for (const departure of [0, 1, 4, 8]) {
    const expected = new Float64Array(5).fill(Infinity);
    // Enumerate paths, not label relaxation; FIFO nonnegative costs admit a simple optimum.
    const visitPath = (state: number, time: number, visited: Set<number>) => {
      expected[state] = Math.min(expected[state], time);
      for (let edge = 0; edge < 4; edge++) {
        const target = transitions[edge * 2 + 1];
        if (transitions[edge * 2] === state && !visited.has(target))
          visitPath(
            target,
            time + costs[edge * 2] + costs[edge * 2 + 1] * time,
            new Set([...visited, target])
          );
      }
    };
    visitPath(0, departure, new Set([0]));
    const result = computeNetworkTimeRoutingOnCPU({
      stateVertices,
      transitions,
      costs,
      endpoints: new Uint32Array([0, 12]),
      departure: new Float32Array([departure])
    });
    expect(result.arrivals).toEqual(Float32Array.from(expected));
    expect(result.predecessorTransitions).toEqual(
      new Uint32Array([0xffffffff, 1, 0, 2, 0xffffffff])
    );
    expect(Array.from(result.summary.subarray(0, 3))).toEqual([1, 1, 2]);
    expect(new Float32Array(result.summary.slice(3).buffer)[0]).toBe(result.arrivals[2]);
  }
});

it('matches hand-computed waiting, interpolation and held duration', () => {
  for (const [departure, expected] of [
    [0, 9],
    [7, 10.2],
    [10, 12],
    [30, 35]
  ]) {
    const result = computeNetworkTimeRoutingOnCPU({
      stateVertices: new Uint32Array([0, 1]),
      transitions: new Uint32Array([0, 1]),
      costs: new Float32Array([5, 9, 10, 12, 20, 25]),
      costModel: 'piecewise-arrival',
      samplesPerTransition: 3,
      endpoints: new Uint32Array([0, 1]),
      departure: new Float32Array([departure])
    });
    expect(result.arrivals[1]).toBe(Math.fround(expected));
    expect(Array.from(result.summary.subarray(0, 3))).toEqual([1, 1, 1]);
  }
});

it('retains first strict-improvement ties and lowest target state, and marks invalid input', () => {
  const input = {
    stateVertices: new Uint32Array([0, 1, 1]),
    transitions: new Uint32Array([0, 2, 0, 1, 0, 1]),
    costs: new Float32Array([2, 0, 2, 0, 2, 0]),
    endpoints: new Uint32Array([0, 1]),
    departure: new Float32Array([0])
  };
  const result = computeNetworkTimeRoutingOnCPU(input);
  expect(Array.from(result.predecessorTransitions)).toEqual([0xffffffff, 1, 0]);
  expect(result.summary[2]).toBe(1);
  for (const invalid of [
    {...input, departure: new Float32Array([-1])},
    {...input, costs: new Float32Array([-1, 0, 2, 0, 2, 0])},
    {...input, endpoints: new Uint32Array([3, 1])}
  ]) {
    const rows = computeNetworkTimeRoutingOnCPU(invalid);
    expect(rows.arrivals.every(value => value === Infinity)).toBe(true);
    expect(rows.predecessorTransitions.every(value => value === 0xffffffff)).toBe(true);
    expect(Array.from(rows.summary)).toEqual([0, 0, 0xffffffff, 0x7f800000]);
  }
  const invalidSamples = computeNetworkTimeRoutingOnCPU({
    ...input,
    transitions: new Uint32Array([0, 1]),
    costModel: 'piecewise-arrival',
    samplesPerTransition: 2,
    costs: new Float32Array([0, 2, 0, 3])
  });
  expect(invalidSamples.summary[0]).toBe(0);
});

it('rejects inconsistent arrays and validates both operation constructions', () => {
  const input = {
    stateVertices: new Uint32Array([0, 1]),
    transitions: new Uint32Array([0, 1]),
    costs: new Float32Array([1, 0]),
    endpoints: new Uint32Array([0, 1]),
    departure: new Float32Array([0])
  };
  for (const [argument, values] of Object.entries({
    transitions: new Uint32Array(1),
    costs: new Float32Array(1),
    endpoints: new Uint32Array(1),
    departure: new Float32Array(2)
  }))
    expect(() => computeNetworkTimeRoutingOnCPU({...input, [argument]: values})).toThrow(
      new RegExp(argument)
    );
  expect(
    () =>
      new CPUNetworkTimeRoutingOperation({
        ...PROPS,
        costModel: 'piecewise-arrival',
        samplesPerTransition: 1
      })
  ).toThrow(/routing-test.*samplesPerTransition/);
  const shapes = {
    stateVertices: {format: 'uint32' as const, length: 5},
    transitions: {format: 'uint32x2' as const, length: 4},
    costs: {format: 'float32x2' as const, length: 4},
    endpoints: {format: 'uint32' as const, length: 2},
    departure: {format: 'float32' as const, length: 1}
  };
  expect(() =>
    constructNetworkOperationContributor(
      new GPUNetworkTimeRoutingOperation({...PROPS, maximumRelaxations: -1}),
      shapes
    )
  ).toThrow(/routing-test.*maximumRelaxations/);
  expect(() => new CPUNetworkTimeRoutingOperation({...PROPS, maximumRelaxations: -1})).toThrow(
    /routing-test.*maximumRelaxations/
  );
  expect(() =>
    constructNetworkOperationContributor(new GPUNetworkTimeRoutingOperation(PROPS), shapes)
  ).not.toThrow();
  for (const Operation of [GPUNetworkTimeRoutingOperation, CPUNetworkTimeRoutingOperation])
    expect(() => new Operation({...PROPS, output: {...PROPS.output, arrivals: 'costs'}})).toThrow(
      /routing-test/
    );
  expect(
    () =>
      new CPUNetworkTimeRoutingOperation({
        ...PROPS,
        stateCount: 257,
        transitionCount: 4097,
        maximumRelaxations: 0
      })
  ).not.toThrow();
});
