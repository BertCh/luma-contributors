// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import type {
  CPUSpatialInteractionInput,
  CPUSpatialInteractionOptions
} from '../../../src/gpu-spatial-analysis/spatial-interaction/cpu-spatial-interaction-operation';

export const INPUT: CPUSpatialInteractionInput = {
  origins: Uint32Array.of(0, 0, 1, 1, 0, 1, 2, 99),
  destinations: Uint32Array.of(0, 1, 0, 1, 0, 1, 2, 99),
  flows: Float32Array.of(10, 20, 30, 40, 5, 15, 0, NaN),
  covariate: Float32Array.of(0, 1, 2, 0, 0, 1, NaN, NaN),
  pairState: Uint32Array.of(1, 1, 1, 1, 1, 1, 0, 2)
};

export const OPTIONS: CPUSpatialInteractionOptions = {
  originCount: 3,
  destinationCount: 3,
  constraint: 'doubly',
  coefficients: [-Math.log(2), 0, -Math.log(2)],
  maximumIterations: 256,
  tolerance: 1e-5
};

// Literals from spatial-interaction-audit.oracle.py, direct NumPy IPF and scipy.special.xlogy.
export const EXPECTED = {
  unconstrained: {
    predictions: [20, 20, 20, 20, 20, 20, 0, NaN],
    summary: [0, 43.42333153533425, 0, 0],
    candidates: [
      92.35957913170877, 0, 0, 0, 43.42333153533425, 0, 0, 0, 92.35957913170877, 0, 0, 0
    ],
    originResiduals: [-25, 25, 0],
    destinationResiduals: [-15, 15, 0]
  },
  origin: {
    predictions: [
      11.666666666666666,
      11.666666666666666,
      28.333333333333332,
      28.333333333333332,
      11.666666666666666,
      28.333333333333332,
      0,
      NaN
    ],
    summary: [0, 21.94094856102566, 0, 1],
    candidates: [
      49.24699876728167, 0, 1, 0, 21.94094856102566, 0, 1, 0, 49.24699876728167, 0, 1, 0
    ],
    originResiduals: [0, 0, 0],
    destinationResiduals: [-6.666666666666664, 6.666666666666671, 0]
  },
  destination: {
    predictions: [15, 25, 15, 25, 15, 25, 0, NaN],
    summary: [0, 35.84318535886307, 0, 1],
    candidates: [
      80.82999692836772, 0, 1, 0, 35.84318535886307, 0, 1, 0, 80.82999692836772, 0, 1, 0
    ],
    originResiduals: [-20, 20, 0],
    destinationResiduals: [0, 0, 0]
  },
  doubly: {
    predictions: [
      10.635044905735578,
      13.729756735771486,
      23.729910188528848,
      30.63512163211426,
      10.635044905735578,
      30.63512163211426,
      0,
      NaN
    ],
    summary: [0, 20.251457017415287, 4.384364495990667e-6, 5],
    candidates: [
      47.81093171513509, 6.147336000391793e-6, 11, 0, 20.251457017415287, 4.384364495990667e-6, 5,
      0, 47.81093171513509, 6.147336000391793e-6, 11, 0
    ],
    originResiduals: [0.00015345275735967334, -0.0001534527573738842, 0],
    destinationResiduals: [0, 0, 0]
  }
};

export const EXHAUSTED = {
  predictions: new Array<number>(8).fill(NaN),
  summary: [NaN, NaN, NaN, NaN],
  candidates: [48.38006309753312, 0.08463257341847107, 1, 1],
  originResiduals: [NaN, NaN, NaN],
  destinationResiduals: [NaN, NaN, NaN]
};

export const OBSERVED_ZERO_INPUT = {...INPUT, flows: Float32Array.of(10, 20, 30, 40, 5, 0, 0, NaN)};
export const OBSERVED_ZERO_OPTIONS: CPUSpatialInteractionOptions = {
  ...OPTIONS,
  constraint: 'unconstrained',
  coefficients: [0]
};
// Six observed cells retain support even when one observed count is zero:
// 105/6=17.5, with the zero-count cell contributing 2*17.5 to deviance.
export const OBSERVED_ZERO_EXPECTED = {
  predictions: [17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 0, NaN],
  summary: [0, 80.09538616003742, 0, 0],
  candidates: [80.09538616003742, 0, 0, 0],
  originResiduals: [-17.5, 17.5, 0],
  destinationResiduals: [-7.5, 7.5, 0]
};

// When every x is zero, all beta values yield the same profiled means. The first
// coefficient (4) must win even though the later candidate (-7) is smaller.
export const TIED_OPTIONS: CPUSpatialInteractionOptions = {
  ...OPTIONS,
  constraint: 'unconstrained',
  coefficients: [4, -7]
};
export const TIED_INPUT = {...INPUT, covariate: new Float32Array(8)};
export const TIED_EXPECTED = {
  ...EXPECTED.unconstrained,
  summary: [4, 43.42333153533425, 0, 0],
  candidates: [43.42333153533425, 0, 0, 0, 43.42333153533425, 0, 0, 0]
};

export const BOUNDED_INPUT: CPUSpatialInteractionInput = {
  origins: Uint32Array.of(0, 0, 1, 1),
  destinations: Uint32Array.of(0, 1, 0, 1),
  flows: Float32Array.of(10, 20, 30, 40),
  covariate: Float32Array.of(0, 1, 2, 0)
};
export const BOUNDED_OPTIONS: CPUSpatialInteractionOptions = {
  originCount: 2,
  destinationCount: 2,
  constraint: 'unconstrained',
  coefficients: [-1, 1],
  calibration: 'bounded',
  maximumCalibrationIterations: 1,
  coefficientTolerance: 1e-5
};
export const BOUNDED_EXPECTED = {
  predictions: [20.535248181217582, 26.00297125685426, 32.926532380710576, 20.535248181217582],
  summary: [0.23606797749978958, 22.86374901049613, 0, 0],
  candidates: [22.86374901049613, 0, 0, 0, 36.054626350046114, 0, 0, 0],
  calibration: [-0.2360679774997898, 1, 1.2360679774997898, 0],
  originResiduals: [NaN, NaN],
  destinationResiduals: [NaN, NaN]
};

export function expectOracleRows(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance: number
): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    if (Number.isNaN(expected[index])) expect(actual[index], `row ${index}`).toBeNaN();
    else {
      expect(Number.isFinite(actual[index]), `finite row ${index}`).toBe(true);
      expect(Math.abs(actual[index] - expected[index]), `row ${index}`).toBeLessThanOrEqual(
        tolerance
      );
    }
  }
}

export function expectOracleResult(
  actual: Record<string, ArrayLike<number> | undefined>,
  expected: Record<string, readonly number[]>,
  status: number,
  gpu = false
): void {
  expect(Array.from(actual['status']!)).toEqual([status]);
  for (const [port, rows] of Object.entries(expected)) {
    // CPU rounds once to float32. GPU exp/log, reductions and repeated scales need
    // 1e-4 absolute at mass <=120; marginal errors need 5e-7. Counts/codes are exact.
    for (let index = 0; index < rows.length; index++) {
      const diagnostic = port === 'summary' || port === 'candidates';
      const slot = port === 'summary' ? index : index % 4;
      const exact = diagnostic && slot >= (port === 'summary' ? 3 : 2);
      const marginalError =
        (port === 'candidates' && slot === 1) || (port === 'summary' && slot === 2);
      const tolerance = exact ? 0 : gpu ? (marginalError ? 5e-7 : 1e-4) : 4e-6;
      expectOracleRows([actual[port]![index]], [rows[index]], tolerance);
    }
  }
}
