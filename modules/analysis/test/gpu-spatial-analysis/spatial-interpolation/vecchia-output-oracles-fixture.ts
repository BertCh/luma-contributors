// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {getLikelihoodProps} from './likelihoods-operation-test-utils';

// Scalar Gaussian derivation and exact generation command: vecchia-output-oracles.oracle.py.
export const VECCHIA_ORACLE_CONDITIONALS = [
  0, 2, 1, 0, 0.7357588823428847, 1.7293294335267746, 1, 1, -0.2943035529371539, 1.7834635468214197,
  1, 1
];
export const VECCHIA_ORACLE_TERMS = [-2.2655121234846454, -1.9955210472173721, -3.9422696848730756];
export function createScalarLikelihoodInput() {
  return {
    positions: Float32Array.of(0, 0, 1, 0, 2, 0),
    times: Float32Array.of(0, 0, 0),
    observations: Float32Array.of(2, 0, 0, 1, -1, 0.5, 0, 1, 3, 0.25, 0, 1),
    neighborhoodOffsets: Uint32Array.of(0, 0, 1, 2),
    neighborhoods: Uint32Array.of(0, 1),
    parameters: Float32Array.of(2, 0, 0, 0, 1, 0, 2, 3)
  };
}
export function getScalarLikelihoodProps() {
  return {...getLikelihoodProps(), rowCount: 3, neighborCount: 1};
}

export const VECCHIA_TWO_PREDECESSOR_CONDITIONALS = [
  0, 2, 1, 0, 0.7357588823428847, 1.7293294335267746, 1, 1, -0.36787944117144233,
  1.7293294335267746, 0.8646647167633873, 2
];

export function createTwoPredecessorLikelihoodInput() {
  const input = createScalarLikelihoodInput();
  input.observations[5] = 0;
  input.observations[9] = 0;
  input.neighborhoodOffsets = Uint32Array.of(0, 0, 1, 3);
  input.neighborhoods = Uint32Array.of(0, 0, 1);
  return input;
}

/** One-site N(0,2) control; invalid geometry/time cannot define a conditional density. */
export function createMarginalLikelihoodInput(
  spaceTime: boolean,
  invalid: 'none' | 'position' | 'time',
  excluded = false
) {
  const parameters = spaceTime ? new Float32Array(28) : Float32Array.of(2, 0, 0, 0, 1, 0, 2, 3);
  if (spaceTime) {
    parameters.set([2, 2, 1, 0]); // metric exponential covariance C(h,u)=2 exp(-sqrt(h²+u²))
    parameters.set([1, 0, 2, 3], 12);
    parameters[26] = 1;
  }
  return {
    positions: Float32Array.of(invalid === 'position' ? NaN : 0, 0),
    times: Float32Array.of(invalid === 'time' ? NaN : 0),
    observations: Float32Array.of(2, 0, 0, excluded ? 0 : 1),
    neighborhoodOffsets: Uint32Array.of(0, 0),
    neighborhoods: Uint32Array.of(0),
    parameters
  };
}

/** Float32 exp/solve/log on three well-conditioned scalar systems: 2e-5 relative, 1e-6 floor. */
export function expectOracleFloats(actual: ArrayLike<number>, expected: readonly number[]) {
  expect(actual.length).toBe(expected.length);
  expected.forEach((value, index) => {
    expect(Number.isFinite(actual[index]), `finite output[${index}]`).toBe(true);
    expect(Math.abs(actual[index] - value), `output[${index}]`).toBeLessThanOrEqual(
      1e-6 + 2e-5 * Math.abs(value)
    );
  });
}
