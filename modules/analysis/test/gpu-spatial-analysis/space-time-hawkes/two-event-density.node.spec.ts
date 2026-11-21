// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeSpaceTimeHawkesOnCPU,
  CPUSpaceTimeHawkesOperation
} from '../../../src/gpu-spatial-analysis/space-time-hawkes/cpu-space-time-hawkes-operation';
import {makeLikelihoodCPUInputs} from '../spatial-interpolation/likelihoods-operation-test-utils';
import {expectHawkesDensityOutputs} from './hawkes-density-assertions';
import {TWO_EVENT_DENSITY} from './two-event-density-fixture';

it('PTS-140 evaluates nonzero triggering and both iterate densities in CPU public forms', async () => {
  const input = {
    positions: Float32Array.of(0, 0, 1, 0),
    times: Float32Array.of(0, 1),
    parameters: Float32Array.of(1, 2, 2, 2, 1e-8, 1, 0.5, 1, 1)
  };
  const operation = new CPUSpaceTimeHawkesOperation({
    id: 'density',
    rowCount: 2,
    maximumIterations: 1,
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
        iterationLimitReached: 'iterationLimitReached',
        iterationCount: 'iterationCount',
        candidateOverflow: 'candidateOverflow',
        invalidCount: 'invalidCount',
        approximation: 'approximation'
      }
    }
  });
  const graphResult = await operation.execute({
    inputs: makeLikelihoodCPUInputs({
      positions: {format: 'float32x2', values: input.positions},
      times: {format: 'float32', values: input.times},
      parameters: {format: 'float32', values: input.parameters}
    })
  });
  for (const actual of [computeSpaceTimeHawkesOnCPU(input, {maximumIterations: 1}), graphResult]) {
    expectHawkesDensityOutputs(actual);
  }
});

it('PTS-133 CPU Hawkes captures its execution budget and output names at construction', async () => {
  const props = {
    id: 'captured-density',
    rowCount: 2,
    maximumIterations: 1,
    positions: 'positions',
    times: 'times',
    parameters: 'parameters',
    output: {
      estimates: 'estimates',
      fittedLogLikelihood: 'fittedLogLikelihood',
      status: {iterationCount: 'iterationCount'}
    }
  };
  const operation = new CPUSpaceTimeHawkesOperation(props);
  props.rowCount = 9;
  props.maximumIterations = 50;
  props.output.estimates = 'replacement';
  props.output.status.iterationCount = 'replacementCount';
  const result = await operation.execute({
    inputs: makeLikelihoodCPUInputs({
      positions: {format: 'float32x2', values: Float32Array.of(0, 0, 1, 0)},
      times: {format: 'float32', values: Float32Array.of(0, 1)},
      parameters: {format: 'float32', values: Float32Array.of(1, 2, 2, 2, 1e-8, 1, 0.5, 1, 1)}
    })
  });
  const iterationCount = result['iterationCount'];
  const estimates = result['estimates'];
  if (!(iterationCount instanceof Uint32Array))
    throw new Error('iterationCount must be Uint32Array');
  if (!(estimates instanceof Float32Array)) throw new Error('estimates must be Float32Array');
  expect(iterationCount.length).toBe(1);
  expect(estimates.length).toBe(TWO_EVENT_DENSITY.estimates.length);
  expect(iterationCount[0]).toBe(1);
  expect(result['replacement']).toBeUndefined();
  TWO_EVENT_DENSITY.estimates.forEach((value, index) => {
    expect(Number.isFinite(estimates[index])).toBe(true);
    expect(Math.abs(estimates[index] - value)).toBeLessThan(2e-7 * Math.max(1, Math.abs(value)));
  });
});
