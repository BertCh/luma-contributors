// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {computePettittTestOnCPU} from '../../analysis/src/gpu-spatial-analysis/change-points/cpu-pettitt-test-operation';
import {
  computeSpaceTimeHawkesOnCPU,
  CPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS
} from '../../analysis/src/gpu-spatial-analysis/space-time-hawkes/cpu-space-time-hawkes-operation';
import {GPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS} from '../../analysis/src/gpu-spatial-analysis/space-time-hawkes/gpu-space-time-hawkes';
import {getGPUSpaceTimeHawkesParameterValues} from '../../analysis/src/gpu-spatial-analysis/space-time-hawkes/space-time-hawkes-parameters';
import {
  getWitnessLikelihoodScore,
  HAWKES_WITNESS_FINAL
} from './catalog-spacetime-hawkes-reference';

test('DGN-01 producer distinguishes untested Pettitt cells from tested constant cells', () => {
  const result = computePettittTestOnCPU(
    Float32Array.of(1, 2, 3, 4, NaN, 2, NaN, NaN, 7, 7, 7, 7),
    {cellCount: 3, sliceCount: 4, layout: 'cell-major'}
  );
  expect(Number.isFinite(result.pValues[0])).toBe(true);
  expect(result.pValues[0]).toBeCloseTo(0.6023884238244043, 6);
  expect(result.pValues[1]).toBeNaN();
  expect(result.pValues[2]).toBe(1);
});

const positions = Float32Array.of(0, 0, 1, 0, 0, 1, 1, 1);

test('UPT-01 review witness cannot decrease likelihood on its first update', () => {
  const result = computeSpaceTimeHawkesOnCPU(
    {
      positions,
      times: Float32Array.of(0.55, 0.7, 0.85, 1),
      parameters: getGPUSpaceTimeHawkesParameterValues({
        windowArea: 16,
        windowDuration: 1,
        maximumLag: 2,
        maximumDistance: 4,
        initialBackground: 0.1,
        initialBranchingRatio: 0.5,
        initialDecayRate: 0.1,
        initialSpatialStandardDeviation: 0.8
      })
    },
    {maximumIterations: 1}
  );
  expect(Number.isFinite(result.logLikelihood[0])).toBe(true);
  expect(result.logLikelihood[0]).toBeCloseTo(-10.594462900603302, 5);
  expect(Number.isFinite(result.fittedLogLikelihood[0])).toBe(true);
  expect(result.fittedLogLikelihood[0]).toBeGreaterThanOrEqual(result.logLikelihood[0]);
});

test('UPT-01 catalog-normalized witness reaches the independent finite-window optimum', () => {
  const input = {
    positions,
    times: Float32Array.of(0, 0.15, 0.3, 0.45),
    parameters: getGPUSpaceTimeHawkesParameterValues({
      windowArea: 81,
      windowDuration: 1,
      maximumLag: 2,
      maximumDistance: 4,
      initialBackground: 2 / 81,
      initialBranchingRatio: 0.5,
      initialDecayRate: 0.5,
      initialSpatialStandardDeviation: 4
    })
  };
  const first = computeSpaceTimeHawkesOnCPU(input, {maximumIterations: 1});
  expect(Number.isFinite(first.fittedLogLikelihood[0])).toBe(true);
  // A safeguarded optimizer may improve on the profiled EM candidate; it must reach at least it.
  expect(first.fittedLogLikelihood[0]).toBeGreaterThanOrEqual(-14.776229122727967 - 1e-4);
  const final = computeSpaceTimeHawkesOnCPU(input, {maximumIterations: 100});
  expect(Number.isFinite(final.fittedLogLikelihood[0])).toBe(true);
  expect(final.fittedLogLikelihood[0]).toBeCloseTo(-12.526952698424633, 4);
  Array.from(final.estimates).forEach((value, index) => {
    expect(Number.isFinite(value)).toBe(true);
    expect(value).toBeCloseTo(HAWKES_WITNESS_FINAL[index], 3);
  });
  for (const score of getWitnessLikelihoodScore(Array.from(final.estimates))) {
    expect(Number.isFinite(score)).toBe(true);
    expect(Math.abs(score)).toBeLessThan(0.002);
  }
  expect(final.converged[0]).toBe(1);
});

test('DEF-13 ordinary Hawkes forms share the 100-update default', () => {
  expect(GPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS).toBe(100);
  expect(CPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS).toBe(100);
});
