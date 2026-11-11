// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import type {CPUNestedInterpolationValidationInput} from '../../../src/gpu-spatial-analysis/spatial-validation/cpu-nested-interpolation-validation-operation';
import type {NestedInterpolationValidationOperationProps} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation-operation';
import type {GPUNestedInterpolationCandidate} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation';
import type {OperationGraphInput} from '../../utils/mixed-execution-test-utils';
import fixture from './nested-interpolation-validation-operation-fixtures.json';

/** Independent SciPy/LAPACK fixture with thirteen fitting branches. @internal */
export {fixture};

/** Fresh arrays, so mutations in edge cases cannot alter another test. @internal */
export function createNestedInput(): CPUNestedInterpolationValidationInput {
  return {
    positions: new Float32Array(fixture.positions),
    response: new Float32Array(fixture.response),
    innerBlocks: new Uint32Array(fixture.blocks),
    training: new Uint32Array(fixture.folds.flatMap(fold => fold.training)),
    testing: new Uint32Array(fixture.folds.flatMap(fold => fold.testing))
  };
}

/** Shared graph-value props, with a fresh candidate ladder. @internal */
export function createNestedProps(): NestedInterpolationValidationOperationProps {
  return {
    id: 'nested-test',
    rowCount: fixture.response.length,
    foldCount: fixture.folds.length,
    innerFoldCount: fixture.innerFoldCount,
    innerBufferDistance: fixture.innerBufferDistance,
    seed: fixture.seed,
    neighborCount: fixture.neighborCount,
    candidates: fixture.candidates as GPUNestedInterpolationCandidate[],
    positions: 'positions',
    response: 'response',
    training: 'training',
    testing: 'testing',
    innerBlocks: 'innerBlocks',
    output: {
      selected: 'selected',
      scores: 'scores',
      predictions: 'predictions',
      metrics: 'metrics',
      status: 'status'
    }
  };
}

/** Input formats mirror the contributor's scalar-packed output layouts. @internal */
export function createNestedGraphInputs(
  input: CPUNestedInterpolationValidationInput
): Record<string, OperationGraphInput & {values: Float32Array | Uint32Array}> {
  return {
    positions: {format: 'float32x2', values: input.positions},
    response: {format: 'float32', values: input.response},
    training: {format: 'uint32', values: input.training},
    testing: {format: 'uint32', values: input.testing},
    innerBlocks: {format: 'uint32', values: input.innerBlocks}
  };
}

/** Exact copies/counts; relative error for nonzero quantities and an explicit zero floor. @internal */
export function expectNestedRows(
  actual: ArrayLike<number>,
  expected: ArrayLike<number | null>,
  relative = 2e-6,
  absolute = 2e-7
): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < actual.length; index++) {
    const value = expected[index];
    if (value === null || Number.isNaN(value)) expect(Number.isNaN(actual[index])).toBe(true);
    else if (!Number.isFinite(value)) expect(actual[index]).toBe(value);
    else
      expect(
        Math.abs(actual[index] - value),
        `row ${index}: ${actual[index]} versus ${value}`
      ).toBeLessThanOrEqual(absolute + relative * Math.abs(value));
  }
}
