// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  computeTransitionDiagnosticsOnCPU,
  CPUTransitionDiagnosticsOperation
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/cpu-transition-diagnostics-operation';
import {
  GPUTransitionDiagnosticsOperation,
  type TransitionDiagnosticsOperationProps
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-diagnostics-operation';
import {GPU_TRANSITION_HOMOGENEITY} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-diagnostics';
import {computeSteadyStateOracle} from './transition-diagnostics-oracle';
import * as giddy from './markov-giddy-fixture';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOperationTestViewGetter} from '../../utils/statistics-dynamics-operation-test-utils';

const PROPS: TransitionDiagnosticsOperationProps = {
  id: 'transition-check',
  counts: 'counts',
  classCount: 2,
  conditionCount: 2,
  output: {
    homogeneity: 'homogeneity',
    steadyStates: 'steadyStates',
    stateSparsity: 'stateSparsity',
    status: {converged: 'converged'}
  }
};

function expectRows(actual: Float32Array, expected: readonly number[], tolerance = 2e-6) {
  expect(actual.length).toBe(expected.length);
  expected.forEach((value, index) => {
    if (Number.isNaN(value)) expect(actual[index]).toBeNaN();
    else
      expect(Math.abs(actual[index] - value), `row ${index}`).toBeLessThanOrEqual(
        tolerance * Math.max(1, Math.abs(value))
      );
  });
}

it('matches the committed external giddy fixture for every output row', () => {
  for (const [
    counts,
    classCount,
    conditionCount,
    expectedHomogeneity,
    expectedSteady,
    expectedSparsity,
    threshold
  ] of [
    [
      giddy.A_COUNTS,
      3,
      3,
      giddy.A_HOMOGENEITY,
      giddy.A_STEADY_STATES,
      giddy.A_SPARSITY,
      giddy.A_SPARSITY_THRESHOLD
    ],
    [
      giddy.B_GOOD_COUNTS,
      4,
      3,
      giddy.B_GOOD_HOMOGENEITY,
      giddy.B_GOOD_STEADY_STATES,
      giddy.B_GOOD_SPARSITY,
      5
    ],
    [
      giddy.B_ALL_COUNTS,
      4,
      5,
      giddy.B_ALL_HOMOGENEITY,
      giddy.B_ALL_STEADY_STATES,
      giddy.B_ALL_SPARSITY,
      5
    ]
  ] as const) {
    const result = computeTransitionDiagnosticsOnCPU(Uint32Array.from(counts), {
      classCount,
      conditionCount,
      minimumTransitions: threshold
    });
    expectRows(result.homogeneity, [
      expectedHomogeneity.Q,
      expectedHomogeneity.Qp,
      expectedHomogeneity.LR,
      expectedHomogeneity.LRp,
      expectedHomogeneity.dof
    ]);
    expectRows(result.steadyStates, expectedSteady);
    expect(Array.from(result.stateSparsity)).toEqual([
      expectedSparsity.sparse,
      expectedSparsity.empty
    ]);
    expect(result.converged[0]).toBe(expectedSteady.some(Number.isNaN) ? 0 : 1);
  }
});

it('matches a different stationary solver and a closed-form df=4 chi-square survival', () => {
  const counts = Uint32Array.of(9, 1, 1, 9, 5, 5, 5, 5, 1, 9, 9, 1);
  const result = computeTransitionDiagnosticsOnCPU(counts, {classCount: 2, conditionCount: 3});
  expectRows(result.steadyStates, new Array(8).fill(0.5));
  const statistic = 25.6;
  expect(result.homogeneity[GPU_TRANSITION_HOMOGENEITY.Q]).toBeCloseTo(statistic, 5);
  expect(result.homogeneity[GPU_TRANSITION_HOMOGENEITY.Q_P_VALUE]).toBeCloseTo(
    Math.exp(-statistic / 2) * (1 + statistic / 2),
    9
  );
  const asymmetric = Uint32Array.of(3, 7, 2, 8);
  expectRows(
    computeTransitionDiagnosticsOnCPU(asymmetric, {classCount: 2}).steadyStates,
    computeSteadyStateOracle(asymmetric, 2)
  );
});

it('keeps biased-chain stationary probabilities finite by matching backward detailed balance', () => {
  const classCount = 36;
  const forwardCount = 1_000_000_000;
  const counts = new Uint32Array(classCount * classCount);
  for (let state = 0; state < classCount; state++) {
    if (state + 1 < classCount) counts[state * classCount + state + 1] = forwardCount;
    if (state > 0) counts[state * classCount + state - 1] = 1;
  }
  counts[classCount * classCount - 1] = forwardCount;
  // Detailed balance: pi[i] P[i,i+1] = pi[i+1] P[i+1,i]. Start at the last
  // state and divide backward, so this oracle never forms growing GTH solution weights.
  // Interior row totals are forwardCount + 1; the first row total is forwardCount.
  const backwardWeights = new Float64Array(classCount);
  backwardWeights[classCount - 1] = 1;
  for (let state = classCount - 2; state > 0; state--) {
    backwardWeights[state] = backwardWeights[state + 1] / forwardCount;
  }
  backwardWeights[0] = backwardWeights[1] / (forwardCount + 1);
  const total = backwardWeights.reduce((sum, weight) => sum + weight, 0);
  const expected = Float32Array.from(backwardWeights, weight => weight / total);
  const result = computeTransitionDiagnosticsOnCPU(counts, {classCount});
  expect(result.steadyStates.every(Number.isFinite)).toBe(true);
  expect(Array.from(result.steadyStates)).toEqual(Array.from(expected));
  expect(result.steadyStates[classCount - 2]).toBe(Math.fround(9.99999999e-10));
  expect(result.steadyStates[classCount - 1]).toBe(1);
  expect(Array.from(result.converged)).toEqual([1]);
});

it('matches hand-derived moderate df=4 homogeneity tails', () => {
  const counts = Uint32Array.of(3, 1, 1, 3, 2, 2, 2, 2, 1, 3, 3, 1);
  const result = computeTransitionDiagnosticsOnCPU(counts, {classCount: 2, conditionCount: 3});
  // Each conditional row totals four; pooled probabilities are [1/2, 1/2].
  // Four extreme rows each add 1 to Q. Both origins contribute (3 - 1)(2 - 1) df.
  // LR = 8 * (3 log(3/2) + log(1/2)); df=4 survival is exp(-x/2)(1+x/2).
  const statistic = 4;
  const likelihoodRatio = 8 * (3 * Math.log(1.5) + Math.log(0.5));
  expect(result.homogeneity[GPU_TRANSITION_HOMOGENEITY.Q]).toBe(statistic);
  expect(result.homogeneity[GPU_TRANSITION_HOMOGENEITY.DEGREES_OF_FREEDOM]).toBe(4);
  expect(result.homogeneity[GPU_TRANSITION_HOMOGENEITY.Q_P_VALUE]).toBeCloseTo(3 * Math.exp(-2), 7);
  expect(result.homogeneity[GPU_TRANSITION_HOMOGENEITY.LR]).toBeCloseTo(likelihoodRatio, 6);
  expect(result.homogeneity[GPU_TRANSITION_HOMOGENEITY.LR_P_VALUE]).toBeCloseTo(
    Math.exp(-likelihoodRatio / 2) * (1 + likelihoodRatio / 2),
    7
  );
  expectRows(result.steadyStates, new Array(8).fill(0.5));
  expect(Array.from(result.stateSparsity)).toEqual([6, 0]);
  expect(Array.from(result.converged)).toEqual([1]);
});

it('matches empty, failed, absent-state and GTH escape semantics', () => {
  const empty = computeTransitionDiagnosticsOnCPU(new Uint32Array(8), {
    classCount: 2,
    conditionCount: 2
  });
  expectRows(empty.homogeneity, [0, NaN, 0, NaN, 0]);
  expect(empty.steadyStates.every(Number.isNaN)).toBe(true);
  expect(Array.from(empty.stateSparsity)).toEqual([0, 4]);
  expect(empty.converged[0]).toBe(1);
  const failed = computeTransitionDiagnosticsOnCPU(Uint32Array.of(0, 1, 0, 0), {classCount: 2});
  expect(failed.steadyStates.every(Number.isNaN)).toBe(true);
  expect(failed.converged[0]).toBe(0);
  const absent = computeTransitionDiagnosticsOnCPU(Uint32Array.of(4, 0, 0, 0), {classCount: 2});
  expectRows(absent.steadyStates, [1, 0]);
  // State 1 escapes to absorbing state 0, but the active support is reducible.
  const reducible = computeTransitionDiagnosticsOnCPU(Uint32Array.of(1, 0, 1, 0), {classCount: 2});
  expectRows(reducible.steadyStates, [NaN, NaN]);
  expect(reducible.converged[0]).toBe(0);
  expect(() => computeTransitionDiagnosticsOnCPU(new Uint32Array(3), {classCount: 2})).toThrow(
    /counts length/
  );
});

it('drops the private-array cap and avoids aggregated uint32 overflow', () => {
  const classCount = 65;
  const cycle = new Uint32Array(classCount * classCount);
  for (let state = 0; state < classCount; state++)
    cycle[state * classCount + ((state + 1) % classCount)] = 1;
  const result = computeTransitionDiagnosticsOnCPU(cycle, {classCount});
  expectRows(result.steadyStates, new Array(classCount).fill(1 / classCount));
  expect(result.converged[0]).toBe(1);
  const large = computeTransitionDiagnosticsOnCPU(Uint32Array.of(0xffffffff, 1, 1, 0xffffffff), {
    classCount: 2
  });
  expectRows(large.steadyStates, [0.5, 0.5]);
  expect(Array.from(large.stateSparsity)).toEqual([0, 0]);
});

it('accepts a safe-integer CPU sparsity threshold above uint32 in the helper and operation', async () => {
  expect(
    () =>
      new GPUTransitionDiagnosticsOperation({
        counts: 'counts',
        classCount: 1,
        minimumTransitions: 2 ** 32,
        output: {stateSparsity: 'sparsity'}
      })
  ).toThrow(/minimumTransitions/);
  const result = computeTransitionDiagnosticsOnCPU(Uint32Array.of(4), {
    classCount: 1,
    minimumTransitions: 2 ** 32
  });
  expect(Array.from(result.stateSparsity)).toEqual([1, 0]);
  const operation = new CPUTransitionDiagnosticsOperation({
    counts: 'counts',
    classCount: 1,
    minimumTransitions: 2 ** 32,
    output: {stateSparsity: 'sparsity'}
  });
  const values = Uint32Array.of(4);
  const actual = await operation.execute({
    inputs: {
      counts: {format: 'uint32', length: 1, chunks: [{values, length: 1}], pack: () => values}
    }
  });
  expect(Array.from(actual['sparsity'] as Uint32Array)).toEqual([1, 0]);
  expect(() =>
    computeTransitionDiagnosticsOnCPU(Uint32Array.of(4), {
      classCount: 1,
      minimumTransitions: Number.MAX_SAFE_INTEGER + 1
    })
  ).toThrow(/minimumTransitions/);
});

it('validates both operation forms at construction and GPU contributor binding', () => {
  for (const Operation of [CPUTransitionDiagnosticsOperation, GPUTransitionDiagnosticsOperation]) {
    expect(
      () => new Operation({...PROPS, output: {...PROPS.output, steadyStates: 'counts'}})
    ).toThrow(/^transition-check outputs.steadyStates/);
  }
  expect(() => new CPUTransitionDiagnosticsOperation({...PROPS, classCount: 0})).toThrow(
    /^transition-check classCount/
  );
  expect(() => new CPUTransitionDiagnosticsOperation({...PROPS, conditionCount: 1})).toThrow(
    /^transition-check output.homogeneity/
  );
  expect(() => new CPUTransitionDiagnosticsOperation({...PROPS, minimumTransitions: 0})).toThrow(
    /^transition-check minimumTransitions/
  );
  expect(
    () =>
      new CPUTransitionDiagnosticsOperation({...PROPS, output: {status: {converged: 'converged'}}})
  ).toThrow(/^transition-check output.status.converged/);
  expect(() => new CPUTransitionDiagnosticsOperation({...PROPS, classCount: 65})).not.toThrow();
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = createOperationTestViewGetter(graph, {
      counts: {format: 'uint32', length: 8},
      homogeneity: {format: 'float32', length: 5},
      steadyStates: {format: 'float32', length: 6},
      stateSparsity: {format: 'uint32', length: 2},
      converged: {format: 'uint32', length: 1}
    });
    expect(() =>
      new GPUTransitionDiagnosticsOperation({...PROPS, classCount: 65}).createContributor(getView)
    ).toThrow(/^transition-check classCount/);
    expect(() =>
      new GPUTransitionDiagnosticsOperation({...PROPS, conditionCount: 1}).createContributor(
        getView
      )
    ).toThrow(/^transition-check output.homogeneity/);
    expect(() =>
      new GPUTransitionDiagnosticsOperation({...PROPS, minimumTransitions: 0}).createContributor(
        getView
      )
    ).toThrow(/^transition-check minimumTransitions/);
    expect(() =>
      new GPUTransitionDiagnosticsOperation({
        ...PROPS,
        output: {status: {converged: 'converged'}}
      }).createContributor(getView)
    ).toThrow(/^transition-check output.status.converged/);
  } finally {
    device.destroy();
  }
});
