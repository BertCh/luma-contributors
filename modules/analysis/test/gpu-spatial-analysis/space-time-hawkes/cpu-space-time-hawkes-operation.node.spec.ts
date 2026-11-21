// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeSpaceTimeHawkesOnCPU,
  CPUSpaceTimeHawkesOperation,
  CPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS
} from '../../../src/gpu-spatial-analysis/space-time-hawkes/cpu-space-time-hawkes-operation';
import {
  GPUSpaceTimeHawkesOperation,
  type SpaceTimeHawkesOperationProps
} from '../../../src/gpu-spatial-analysis/space-time-hawkes/gpu-space-time-hawkes-operation';
import {GPU_SPACE_TIME_HAWKES_BRANCHING_RATIO_FLOOR} from '../../../src/gpu-spatial-analysis/space-time-hawkes/space-time-hawkes-wgsl';
import {constructNetworkOperationContributor as constructHawkesContributor} from '../../utils/network-operation-test-utils';
import fixture from './hawkes-operation-fixture.json';
import {makeLikelihoodCPUInputs} from '../spatial-interpolation/likelihoods-operation-test-utils';

const PROPS: SpaceTimeHawkesOperationProps = {
  id: 'hawkes-test',
  rowCount: 8,
  positions: 'positions',
  times: 'times',
  parameters: 'parameters',
  output: {
    estimates: 'estimates',
    logLikelihood: 'logLikelihood',
    fittedLogLikelihood: 'fittedLogLikelihood',
    backgroundProbability: 'backgroundProbability',
    status: {
      converged: 'converged',
      iterationCount: 'iterationCount',
      maximumIterations: 'maximumIterations',
      iterationLimitReached: 'iterationLimitReached',
      candidateOverflow: 'candidateOverflow',
      invalidCount: 'invalidCount',
      approximation: 'approximation'
    }
  }
};
const SHAPES = {
  positions: {format: 'float32x2' as const, length: 8},
  times: {format: 'float32' as const, length: 8},
  parameters: {format: 'float32' as const, length: 9}
};
function createInput() {
  return {
    positions: Float32Array.from(fixture.positions),
    times: Float32Array.from(fixture.times),
    parameters: Float32Array.from(fixture.parameters)
  };
}

it('evaluates fitted likelihood at the published one-update estimates', () => {
  const input = {
    positions: Float32Array.of(0, 0, 1, 1),
    times: Float32Array.of(2, 2),
    parameters: Float32Array.of(1, 10, 10, 10, 1e-8, 1, 0.5, 1, 1)
  };
  const result = computeSpaceTimeHawkesOnCPU(input, {maximumIterations: 1});
  // Equal times have no parents: the updated rate is 2 / (1 * 10) and omega hits its floor.
  expect(result.estimates[0]).toBe(Math.fround(0.2));
  expect(result.fittedLogLikelihood[0]).toBe(Math.fround(2 * Math.log(0.2) - 2));
  expect(result.logLikelihood[0]).toBe(Math.fround(-10 - (1 - Math.exp(-8))));
  expect(result.fittedLogLikelihood[0]).not.toBe(result.logLikelihood[0]);
  expect(result.iterationCount[0]).toBe(1);
});

it('leaves endpoint-event probabilities undefined when the fitted temporal mass is zero', () => {
  const result = computeSpaceTimeHawkesOnCPU(
    {
      positions: Float32Array.of(0, 0, 1, 1),
      times: Float32Array.of(10, 10),
      parameters: Float32Array.of(1, 10, 10, 10, 1e-8, 1, 0.5, 1, 1)
    },
    {maximumIterations: 1}
  );
  expect(result.estimates.every(Number.isNaN)).toBe(true);
  expect(Array.from(result.backgroundProbability)).toEqual([NaN, NaN]);
  expect(Number.isNaN(result.fittedLogLikelihood[0])).toBe(true);
});

it('agrees with SciPy simultaneous solution of the finite-window score equations', () => {
  const input = createInput();
  const result = computeSpaceTimeHawkesOnCPU(input, {maximumIterations: 1000});
  fixture.estimates.forEach((expected, index) =>
    expect(Math.abs(result.estimates[index] / expected - 1)).toBeLessThan(2e-6)
  );
  fixture.backgroundProbability.forEach((expected, index) =>
    expect(Math.abs(result.backgroundProbability[index] / expected - 1)).toBeLessThan(2e-6)
  );
  // One float32 output quantum at this magnitude is about 1.9e-6. The independent
  // fixed-point likelihood includes a substantial offspring compensator (about 4.99).
  expect(fixture.offspringCompensator).toBeGreaterThan(4);
  expect(Math.abs(result.logLikelihood[0] - fixture.logLikelihood)).toBeLessThan(2e-6);
  expect(result.converged[0]).toBe(1);
  expect(result.iterationLimitReached[0]).toBe(0);
  expect(result.candidateOverflow[0]).toBe(0);
  expect(result.invalidCount[0]).toBe(0);
  expect(result.approximation[0]).toBe(1);
  // Reordering the catalogue cannot change the fit; probabilities map back to input rows.
  const order = [5, 2, 7, 0, 4, 1, 6, 3];
  const shuffled = computeSpaceTimeHawkesOnCPU(
    {
      ...input,
      positions: Float32Array.from(
        order.flatMap(row => [input.positions[row * 2], input.positions[row * 2 + 1]])
      ),
      times: Float32Array.from(order, row => input.times[row])
    },
    {maximumIterations: 1000}
  );
  expect(shuffled.estimates).toEqual(result.estimates);
  expect(Array.from(shuffled.backgroundProbability)).toEqual(
    order.map(row => result.backgroundProbability[row])
  );
});

it('has the Poisson closed form for simultaneous events and keeps the kernel parameters', () => {
  const input = createInput();
  input.times.fill(2);
  const result = computeSpaceTimeHawkesOnCPU(input);
  const rate = 8 / 1000;
  expect(result.estimates).toEqual(
    Float32Array.of(
      rate,
      GPU_SPACE_TIME_HAWKES_BRANCHING_RATIO_FLOOR,
      input.parameters[7],
      input.parameters[8]
    )
  );
  expect(result.backgroundProbability).toEqual(new Float32Array(8).fill(1));
  expect(result.logLikelihood[0]).toBe(Math.fround(8 * Math.log(rate) - 8));
  expect(result.iterationCount[0]).toBe(2);
});

it('the CPU graph operation honors its iteration budget, ignores the parent cap and publishes every status', async () => {
  const input = createInput();
  const operation = new CPUSpaceTimeHawkesOperation({
    ...PROPS,
    maximumIterations: 1,
    maximumParents: 1
  });
  const output = await operation.execute({
    inputs: makeLikelihoodCPUInputs({
      positions: {format: 'float32x2', values: input.positions},
      times: {format: 'float32', values: input.times},
      parameters: {format: 'float32', values: input.parameters}
    })
  });
  const expected = computeSpaceTimeHawkesOnCPU(input, {maximumIterations: 1});
  for (const [port, values] of Object.entries(expected)) expect(output[port]).toEqual(values);
  expect(expected.iterationCount[0]).toBe(1);
  expect(expected.converged[0]).toBe(0);
  expect(expected.iterationLimitReached[0]).toBe(1);
});

it('publishes the hand-calculated first update and terminates the reported rounding cycle', () => {
  const input = {
    positions: Float32Array.from(fixture.cyclingInput.positions),
    times: Float32Array.from(fixture.cyclingInput.times),
    parameters: Float32Array.from(fixture.cyclingInput.parameters)
  };
  const first = computeSpaceTimeHawkesOnCPU(input, {maximumIterations: 1});
  // Direct enumeration of the six earlier/later pairs gives backgroundSum=1.394800823044734,
  // triggeredSum=2.6051991769552663, lagSum=3.5267182296874493,
  // distanceSum=0.7762238289891579 and windowMass=3.9914701998482.
  // SciPy bounded minimization of -Q(beta), refitting omega=S/M(beta), gives
  // beta=.7012133208939824 and omega=.6592794165712239. Other updates retain the
  // closed forms backgroundSum/100 and sqrt(distanceSum/(2*triggeredSum)).
  expect(first.estimates).toEqual(
    Float32Array.of(
      0.01394800823044734,
      0.6592794165712239,
      0.7012133208939824,
      0.38597397400484873
    )
  );
  expect(first.logLikelihood).toEqual(Float32Array.of(-15.259916788653317));
  expect(first.backgroundProbability).toEqual(
    Float32Array.of(0.14042880777282357, 1, 0.13815156081218397, 0.07406758696668789)
  );
  expect(first.iterationCount[0]).toBe(1);
  expect(first.converged[0]).toBe(0);
  expect(first.iterationLimitReached[0]).toBe(1);
  // The tolerance 1e-20 is below binary64 resolution. The fixed-count updates cycled at rounding
  // level until the budget; the finite-window decay update reaches an exact fixed point (zero
  // relative change) instead. Either way the run terminates within the budget with exactly
  // one of the two flags set.
  const bounded = computeSpaceTimeHawkesOnCPU(input);
  expect(bounded.iterationCount[0]).toBeGreaterThan(1);
  expect(bounded.iterationCount[0]).toBeLessThanOrEqual(
    CPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS
  );
  expect(bounded.converged[0] + bounded.iterationLimitReached[0]).toBe(1);
  expect(bounded.iterationLimitReached[0]).toBe(
    Number(bounded.iterationCount[0] === CPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS)
  );
  expect(bounded.estimates.every(Number.isFinite)).toBe(true);
  expect(bounded.logLikelihood.every(Number.isFinite)).toBe(true);
  expect(bounded.backgroundProbability.every(Number.isFinite)).toBe(true);
  expect(() => computeSpaceTimeHawkesOnCPU(input, {maximumIterations: 0})).toThrow(
    /maximumIterations/
  );
});

it('returns failed-start NaNs for no eligible events and reports inconsistent arrays', () => {
  const input = createInput();
  input.times.fill(-1);
  const result = computeSpaceTimeHawkesOnCPU(input);
  expect(result.estimates.every(Number.isNaN)).toBe(true);
  expect(result.backgroundProbability.every(Number.isNaN)).toBe(true);
  expect(result.iterationCount[0]).toBe(1);
  expect(result.converged[0]).toBe(0);
  expect(result.invalidCount[0]).toBe(8);
  expect(() => computeSpaceTimeHawkesOnCPU({...input, positions: new Float32Array(1)})).toThrow(
    /positions/
  );
  expect(() => computeSpaceTimeHawkesOnCPU({...input, parameters: new Float32Array(8)})).toThrow(
    /parameters/
  );
});

it('validates CPU props and contributor construction; GPU unrolling and parent caps are dropped on CPU', () => {
  expect(() => new CPUSpaceTimeHawkesOperation({...PROPS, rowCount: 1})).toThrow(
    /hawkes-test.*rowCount/
  );
  expect(() => new CPUSpaceTimeHawkesOperation({...PROPS, maximumIterations: 0})).toThrow(
    /hawkes-test.*maximumIterations/
  );
  expect(() => new CPUSpaceTimeHawkesOperation({...PROPS, maximumParents: 0})).toThrow(
    /hawkes-test.*maximumParents/
  );
  expect(
    () => new CPUSpaceTimeHawkesOperation({...PROPS, maximumIterations: 1001, maximumParents: 4097})
  ).not.toThrow();
  for (const Operation of [GPUSpaceTimeHawkesOperation, CPUSpaceTimeHawkesOperation])
    expect(() => new Operation({...PROPS, output: {...PROPS.output, estimates: 'times'}})).toThrow(
      /hawkes-test/
    );
  expect(() =>
    constructHawkesContributor(
      new GPUSpaceTimeHawkesOperation({...PROPS, maximumParents: 4097}),
      SHAPES
    )
  ).toThrow(/hawkes-test.*maximumParents/);
  expect(() =>
    constructHawkesContributor(
      new GPUSpaceTimeHawkesOperation({...PROPS, maximumIterations: 1001}),
      SHAPES
    )
  ).toThrow(/hawkes-test.*maximumIterations/);
  expect(() =>
    constructHawkesContributor(new GPUSpaceTimeHawkesOperation(PROPS), {
      ...SHAPES,
      times: {...SHAPES.times, length: 7}
    })
  ).toThrow(/hawkes-test.*times/);
  expect(() =>
    constructHawkesContributor(new GPUSpaceTimeHawkesOperation(PROPS), SHAPES)
  ).not.toThrow();
});
