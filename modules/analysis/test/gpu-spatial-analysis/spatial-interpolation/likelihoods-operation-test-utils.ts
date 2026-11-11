// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';
import type {CPUVecchiaLogLikelihoodInput} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-log-likelihood-operation';
import type {VecchiaLogLikelihoodOperationProps} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood-operation';
import type {VecchiaSpaceTimeLogLikelihoodOperationProps} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood-operation';
import type {VecchiaFisherScoringOperationProps} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-fisher-scoring-operation';
import type {OperationGraphInput, OperationGraphRows} from '../../utils/mixed-execution-test-utils';
import fixture from './likelihoods-fixture.json';
import {GPU_VECCHIA_SPACE_TIME_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood';

/** Half the relative spacing of normal float32 values, the usual unit roundoff. */
const FLOAT32_UNIT_ROUNDOFF = 2 ** -24;

export function createLikelihoodInput(
  parameters: readonly number[] = fixture.spatial[0].parameters
): CPUVecchiaLogLikelihoodInput & {times: Float32Array} {
  return {
    positions: Float32Array.from(fixture.positions),
    times: Float32Array.from(fixture.times),
    observations: Float32Array.from(fixture.observations),
    neighborhoodOffsets: Uint32Array.from(fixture.neighborhoodOffsets),
    neighborhoods: Uint32Array.from(fixture.neighborhoods),
    parameters: Float32Array.from(parameters)
  };
}

/** A finite hand-written buffer accepted by the space-time loader; the packer never emits -1. */
export function createStructureCodeLikelihoodInput(
  modelCode = -1
): ReturnType<typeof createLikelihoodInput> {
  const headerLength = 4;
  const structureLength = 4;
  const jointStructureIndex = 2;
  const parameters = new Float32Array(GPU_VECCHIA_SPACE_TIME_PARAMETER_LENGTH);
  parameters.set([1, 2, 1, 0]); // Scale one, metric covariance, anisotropy one.
  parameters.set([modelCode, 0, 1, 1], headerLength + jointStructureIndex * structureLength);
  return {
    positions: new Float32Array([0, 0, 1, 0]),
    times: new Float32Array([0, 0]),
    observations: new Float32Array([2, 0, 0, 1, 1, 0, 0, 1]),
    neighborhoodOffsets: new Uint32Array([0, 0, 1]),
    neighborhoods: new Uint32Array([0]),
    parameters
  };
}

export function getLikelihoodProps(): VecchiaLogLikelihoodOperationProps {
  return {
    id: 'likelihood-test',
    rowCount: 5,
    neighborCount: 4,
    positions: 'positions',
    observations: 'observations',
    neighborhoodOffsets: 'neighborhoodOffsets',
    neighborhoods: 'neighborhoods',
    parameters: 'parameters',
    output: {
      conditionals: 'conditionals',
      conditionalStatus: 'conditionalStatus',
      siteTerms: 'siteTerms',
      siteStatus: 'siteStatus',
      logLikelihood: 'logLikelihood',
      status: 'status'
    }
  };
}

export function getSpaceTimeLikelihoodProps(): VecchiaSpaceTimeLogLikelihoodOperationProps {
  return {
    ...getLikelihoodProps(),
    times: 'times',
    temporalContext: {
      unit: 'second',
      epoch: 0n,
      encoding: 'float32-relative',
      axis: {kind: 'irregular'}
    }
  };
}

export function getFisherProps(): VecchiaFisherScoringOperationProps {
  return {
    ...getLikelihoodProps(),
    output: {siteTerms: 'siteTerms', siteStatus: 'siteStatus', totals: 'totals', status: 'status'}
  };
}

export function getLikelihoodGraphInputs(
  input: ReturnType<typeof createLikelihoodInput>,
  spaceTime = false
): Record<string, OperationGraphInput> {
  return {
    positions: {format: 'float32x2', values: input.positions},
    observations: {format: 'float32x4', values: input.observations},
    neighborhoodOffsets: {format: 'uint32', values: input.neighborhoodOffsets},
    neighborhoods: {format: 'uint32', values: input.neighborhoods},
    parameters: {format: 'float32', values: input.parameters},
    ...(spaceTime ? {times: {format: 'float32' as const, values: input.times}} : {})
  };
}

export function getLikelihoodShapes(spaceTime = false) {
  return Object.fromEntries(
    Object.entries(
      getLikelihoodGraphInputs(
        createLikelihoodInput(spaceTime ? fixture.spaceTime[0].parameters : undefined),
        spaceTime
      )
    ).map(([port, input]) => [
      port,
      {
        format: input.format,
        length: input.values.length / (port === 'positions' ? 2 : port === 'observations' ? 4 : 1)
      }
    ])
  );
}

export function makeLikelihoodCPUInputs(
  inputs: Record<string, OperationGraphInput>
): Record<string, GPUCPUVector> {
  return Object.fromEntries(
    Object.entries(inputs).map(([port, input]) => {
      const length =
        input.values.length /
        (port === 'positions' || port === 'queries' ? 2 : port === 'observations' ? 4 : 1);
      return [
        port,
        {
          format: input.format,
          length,
          chunks: [{values: input.values, length}],
          pack: () => input.values
        }
      ];
    })
  );
}

export function expectLikelihoodFloat(
  actual: number,
  expected: number,
  relativeTolerance = 1e-4
): void {
  if (Number.isNaN(expected)) expect(Number.isNaN(actual)).toBe(true);
  else if (expected === 0) expect(Math.abs(actual)).toBeLessThanOrEqual(1e-7);
  else
    expect(Math.abs(actual - expected)).toBeLessThanOrEqual(relativeTolerance * Math.abs(expected));
}

/**
 * Compare every component, with integer/count/copy outputs exact and split sums decoded as well.
 * Optional independent-oracle sum(abs(weight * observation)) values give conditional means an
 * absolute budget of two float32 unit roundoffs of their terms' magnitude, alongside the existing
 * relative 1e-4 budget. This handles cancellation without weakening variance or other comparisons.
 */
export function expectLikelihoodGraphResults(
  gpu: Record<string, OperationGraphRows>,
  cpu: Record<string, OperationGraphRows>,
  conditionalMeanTermMagnitudes?: readonly number[]
): void {
  expect(Object.keys(gpu).sort()).toEqual(Object.keys(cpu).sort());
  if (conditionalMeanTermMagnitudes !== undefined) {
    expect(conditionalMeanTermMagnitudes.length).toBe(cpu['conditionals'].length / 4);
  }
  for (const port of Object.keys(cpu)) {
    expect(gpu[port].length).toBe(cpu[port].length);
    if (cpu[port] instanceof Uint32Array) {
      expect(Array.from(gpu[port])).toEqual(Array.from(cpu[port]));
      continue;
    }
    for (let index = 0; index < cpu[port].length; index++) {
      if (port === 'conditionals' && index % 4 === 3)
        expect(gpu[port][index]).toBe(cpu[port][index]);
      else if (
        port === 'conditionals' &&
        index % 4 === 0 &&
        conditionalMeanTermMagnitudes !== undefined &&
        Number.isFinite(cpu[port][index])
      ) {
        const relativeBudget = 1e-4 * Math.abs(cpu[port][index]);
        const cancellationBudget =
          2 * FLOAT32_UNIT_ROUNDOFF * conditionalMeanTermMagnitudes[index / 4];
        expect(
          Math.abs(gpu[port][index] - cpu[port][index]),
          `conditionals[${index}]: mean`
        ).toBeLessThanOrEqual(Math.max(relativeBudget, cancellationBudget));
      } else if (
        (port === 'logLikelihood' || port === 'totals') &&
        index % 2 === 1 &&
        Number.isFinite(cpu[port][index])
      ) {
        // The low words encode different remainders because only GPU site cells are rounded.
        // Bound each low-word difference by two float32 quanta of its high word.
        expect(Math.abs(gpu[port][index] - cpu[port][index])).toBeLessThanOrEqual(
          2e-7 * Math.abs(cpu[port][index - 1])
        );
      } else expectLikelihoodFloat(gpu[port][index], cpu[port][index]);
    }
    if (port === 'logLikelihood' || port === 'totals') {
      for (let index = 0; index < cpu[port].length; index += 2) {
        expectLikelihoodFloat(
          gpu[port][index] + gpu[port][index + 1],
          cpu[port][index] + cpu[port][index + 1]
        );
      }
    }
  }
}
