// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  computeNetworkTimeRoutingOnCPU,
  type CPUNetworkTimeRoutingInput,
  CPUNetworkTimeRoutingOperation
} from '../../../src/gpu-network/network-optimization/cpu-network-time-routing-operation';
import {
  GPUNetworkTimeRoutingOperation,
  type NetworkTimeRoutingOperationProps
} from '../../../src/gpu-network/network-optimization/gpu-network-time-routing-operation';
import {GPUNetworkTimeRouting} from '../../../src/gpu-network/network-optimization/gpu-network-time-routing';
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

it.each([
  0, 1, 2, 3
])('preserves partial routing labels and exact stopping status at budget %s', async maximumRelaxations => {
  const input = {
    stateVertices: Uint32Array.of(0, 1, 2),
    // Reverse edge order requires two sweeps, then one unchanged sweep to certify convergence.
    transitions: Uint32Array.of(1, 2, 0, 1),
    costs: Float32Array.of(1, 0, 1, 0),
    endpoints: Uint32Array.of(0, 2),
    departure: Float32Array.of(0),
    maximumRelaxations
  };
  const result = computeNetworkTimeRoutingOnCPU(input);
  expect(result.arrivals).toEqual(
    Float32Array.of(
      0,
      maximumRelaxations >= 1 ? 1 : Infinity,
      maximumRelaxations >= 2 ? 2 : Infinity
    )
  );
  expect(result.summary[1]).toBe(Number(maximumRelaxations === 3));
  expect(result.status.iterationCount).toEqual(Uint32Array.of(maximumRelaxations));
  expect(result.status.iterationLimitReached).toEqual(
    Uint32Array.of(Number(maximumRelaxations < 3))
  );
  const operation = new CPUNetworkTimeRoutingOperation({
    ...PROPS,
    stateCount: 3,
    transitionCount: 2,
    maximumRelaxations,
    output: {
      ...PROPS.output,
      status: {iterationCount: 'iterations', converged: 'converged', iterationLimitReached: 'limit'}
    }
  });
  const vector = (
    format: GPUCPUVector['format'],
    values: Uint32Array | Float32Array
  ): GPUCPUVector => {
    const length = values.length / (format.endsWith('x2') ? 2 : 1);
    return {format, length, chunks: [{values, length}], pack: () => values};
  };
  const actual = await operation.execute({
    inputs: {
      stateVertices: vector('uint32', input.stateVertices),
      transitions: vector('uint32x2', input.transitions),
      costs: vector('float32x2', input.costs),
      endpoints: vector('uint32', input.endpoints),
      departure: vector('float32', input.departure)
    }
  });
  expect(actual['arrivals']).toEqual(result.arrivals);
  expect(actual['predecessors']).toEqual(result.predecessorTransitions);
  expect(actual['summary']).toEqual(result.summary);
  expect(actual['iterations']).toEqual(result.status.iterationCount);
  expect(actual['converged']).toEqual(result.status.converged);
  expect(actual['limit']).toEqual(result.status.iterationLimitReached);
});

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

it.each([
  [{stateCount: 0}, /stateCount/],
  [{stateCount: 259}, /stateCount/],
  [{transitionCount: -1}, /transitionCount/],
  [{transitionCount: 4097}, /transitionCount/],
  [{maximumRelaxations: 65537}, /maximumRelaxations/],
  [{costModel: 'piecewise-arrival', samplesPerTransition: 1}, /samplesPerTransition/],
  [{costModel: 'piecewise-arrival', samplesPerTransition: 257}, /samplesPerTransition/]
] as const)('validates GPU routing declarations before lowering: %j', (override, diagnostic) => {
  expect(
    () =>
      new GPUNetworkTimeRoutingOperation({
        ...PROPS,
        ...override
      } as NetworkTimeRoutingOperationProps)
  ).toThrow(diagnostic);
});

it.each([
  [{stateCount: 3}, /stateCount/],
  [{transitionCount: 2}, /transitionCount/]
] as const)('checks declared routing dimensions against resolved row counts: %j', (override, diagnostic) => {
  const operation = new GPUNetworkTimeRoutingOperation({
    ...PROPS,
    stateCount: 2,
    transitionCount: 1,
    ...override
  });
  expect(() =>
    constructNetworkOperationContributor(operation, {
      stateVertices: {format: 'uint32', length: 2},
      transitions: {format: 'uint32x2', length: 1},
      costs: {format: 'float32x2', length: 1},
      endpoints: {format: 'uint32', length: 2},
      departure: {format: 'float32', length: 1}
    })
  ).toThrow(diagnostic);
});

it.each([
  [{costModel: 'bogus' as never, costs: Float32Array.of(5, 0)}, /costModel/],
  [
    {
      costModel: 'piecewise-arrival' as const,
      samplesPerTransition: 2,
      waitingPolicy: 'bogus' as never,
      costs: Float32Array.of(0, 5, 1, 6)
    },
    /waitingPolicy/
  ]
])('rejects invalid pure routing options with valid cost rows: %j', (override, diagnostic) => {
  const input: CPUNetworkTimeRoutingInput = {
    stateVertices: Uint32Array.of(0, 1),
    transitions: Uint32Array.of(0, 1),
    endpoints: Uint32Array.of(0, 1),
    departure: Float32Array.of(0),
    ...override
  };
  expect(() => computeNetworkTimeRoutingOnCPU(input)).toThrow(diagnostic);
  for (const Operation of [CPUNetworkTimeRoutingOperation, GPUNetworkTimeRoutingOperation]) {
    expect(
      () =>
        new Operation({
          ...PROPS,
          costModel: override.costModel,
          samplesPerTransition:
            'samplesPerTransition' in override ? override.samplesPerTransition : undefined,
          waitingPolicy: 'waitingPolicy' in override ? override.waitingPolicy : undefined
        })
    ).toThrow(diagnostic);
  }
});

it('accepts both inclusive GPU routing declaration boundaries', () => {
  expect(
    () =>
      new GPUNetworkTimeRoutingOperation({
        ...PROPS,
        stateCount: 1,
        transitionCount: 0,
        maximumRelaxations: 0
      })
  ).not.toThrow();
  expect(
    () =>
      new GPUNetworkTimeRoutingOperation({
        ...PROPS,
        stateCount: 256,
        transitionCount: 4096,
        maximumRelaxations: 65536,
        costModel: 'piecewise-arrival',
        samplesPerTransition: 256,
        waitingPolicy: 'wait-until-first-sample'
      })
  ).not.toThrow();
});

it('reports GPU sample and sweep bounds without imposing a uint32 sample cap on pure CPU routing (S1)', () => {
  expect(
    () =>
      new GPUNetworkTimeRoutingOperation({
        ...PROPS,
        costModel: 'piecewise-arrival',
        samplesPerTransition: 1
      })
  ).toThrow(/samplesPerTransition.*\[2, 256\]/);
  expect(() => new GPUNetworkTimeRoutingOperation({...PROPS, maximumRelaxations: -1})).toThrow(
    /maximumRelaxations.*\[0, 65536\]/
  );
});

it('preserves unbounded pure CPU sample counts when no transitions require storage (S1)', () => {
  // No transitions means no samples are read or allocated, irrespective of the integer sample count.
  const result = computeNetworkTimeRoutingOnCPU({
    stateVertices: Uint32Array.of(0),
    transitions: new Uint32Array(0),
    costs: new Float32Array(0),
    endpoints: Uint32Array.of(0, 0),
    departure: Float32Array.of(1),
    costModel: 'piecewise-arrival',
    samplesPerTransition: 2 ** 32
  });
  expect(result.arrivals).toEqual(Float32Array.of(1));
  expect(result.predecessorTransitions).toEqual(Uint32Array.of(0xffffffff));
  expect(result.summary.slice(0, 3)).toEqual(Uint32Array.of(1, 1, 0));
});

it('captures GPU routing declarations before callers change resolved shape (S2)', () => {
  const props = {...PROPS, stateCount: 2, transitionCount: 1};
  const operation = new GPUNetworkTimeRoutingOperation(props);
  props.transitionCount = 3;
  expect(() =>
    constructNetworkOperationContributor(operation, {
      stateVertices: {format: 'uint32', length: 2},
      transitions: {format: 'uint32x2', length: 3},
      costs: {format: 'float32x2', length: 3},
      endpoints: {format: 'uint32', length: 2},
      departure: {format: 'float32', length: 1}
    })
  ).toThrow(/transitionCount/);
});

it.each([
  false,
  true
])('captures routing model, budget, dimensions and output maps (CPU=%s)', async useCPUOperation => {
  const props: NetworkTimeRoutingOperationProps = {
    ...PROPS,
    stateCount: 2,
    transitionCount: 1,
    maximumRelaxations: 0,
    output: {...PROPS.output, status: {converged: 'converged'}}
  };
  const operation = useCPUOperation
    ? new CPUNetworkTimeRoutingOperation(props)
    : new GPUNetworkTimeRoutingOperation(props);
  props.stateCount = 3;
  props.transitionCount = 3;
  props.costModel = 'piecewise-arrival';
  props.samplesPerTransition = 2;
  props.maximumRelaxations = 3;
  props.output.arrivals = 'mutated';
  props.output.status!.converged = 'mutated-converged';
  if (operation instanceof GPUNetworkTimeRoutingOperation) {
    const contributor = constructNetworkOperationContributor(operation, {
      stateVertices: {format: 'uint32', length: 2},
      transitions: {format: 'uint32x2', length: 1},
      costs: {format: 'float32x2', length: 1},
      endpoints: {format: 'uint32', length: 2},
      departure: {format: 'float32', length: 1}
    }) as GPUNetworkTimeRouting;
    expect(contributor.props.costModel).toBeUndefined();
    expect(contributor.props.maximumRelaxations).toBe(0);
    expect(contributor.props.output.status?.converged).toBeDefined();
  } else {
    const vector = (
      format: GPUCPUVector['format'],
      values: Uint32Array | Float32Array
    ): GPUCPUVector => {
      const length = values.length / (format.endsWith('x2') ? 2 : 1);
      return {format, length, chunks: [{values, length}], pack: () => values};
    };
    const result = await operation.execute({
      inputs: {
        stateVertices: vector('uint32', Uint32Array.of(0, 1)),
        transitions: vector('uint32x2', Uint32Array.of(0, 1)),
        costs: vector('float32x2', Float32Array.of(5, 0)),
        endpoints: vector('uint32', Uint32Array.of(0, 1)),
        departure: vector('float32', Float32Array.of(0))
      }
    });
    expect(result['arrivals']).toEqual(Float32Array.of(0, Infinity));
    expect(result['converged']).toEqual(Uint32Array.of(0));
    expect(result['mutated']).toBeUndefined();
  }
});

it('STA-14 keeps reached arrival overflow separate from convergence', () => {
  for (const cost of [3e38, 7]) {
    const result = computeNetworkTimeRoutingOnCPU({
      stateVertices: Uint32Array.of(0, 1, 2),
      transitions: Uint32Array.of(0, 1, 1, 2),
      costs: Float32Array.of(cost, 0, cost, 0),
      endpoints: Uint32Array.of(0, 2),
      departure: Float32Array.of(0)
    });
    expect(Array.from(result.summary.slice(0, 3))).toEqual([1, 1, 2]);
    expect(Array.from(result.unrepresentable)).toEqual([0, 0, cost === 7 ? 0 : 1]);
    expect(result.arrivals[2]).toBe(cost === 7 ? 14 : Infinity);
  }
});

it('review B1 chooses the direct predecessor from elapsed labels on very large clocks', () => {
  for (const departure of [2 ** 25, 2 ** 54]) {
    const result = computeNetworkTimeRoutingOnCPU({
      stateVertices: Uint32Array.of(0, 1, 2),
      transitions: Uint32Array.of(0, 1, 1, 2, 0, 2),
      costs: Float32Array.of(1.5, 0, 1.5, 0, 2.5, 0),
      endpoints: Uint32Array.of(0, 2),
      departure: Float32Array.of(departure)
    });
    expect(Array.from(result.predecessorTransitions)).toEqual([0xffffffff, 0, 2]);
    expect(Array.from(result.summary.slice(0, 3))).toEqual([1, 1, 2]);
  }
});
