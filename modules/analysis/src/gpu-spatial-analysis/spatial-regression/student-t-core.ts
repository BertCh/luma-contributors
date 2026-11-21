// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getBetaContinuedFraction} from './incomplete-beta';

/** Log Gamma(alpha+1/2)/Gamma(alpha), with a fixed recurrence to an accurate asymptotic domain. */
function getLogGammaHalfRatio(argument: number): number {
  let shifted = argument;
  let correction = 0;
  // At most 32 steps for every supported positive integer/effective df.
  for (let step = 0; step < 32 && shifted < 32; step++) {
    correction += Math.log(shifted / (shifted + 0.5));
    shifted += 1;
  }
  const inverse = 1 / shifted;
  return (
    correction + 0.5 * Math.log(shifted) - inverse / 8 + inverse ** 3 / 192 - inverse ** 5 / 640
  );
}

/** Common positive Student t tail core; callers retain their domain and boundary policies. @internal */
export function getStudentTPositiveTail(
  magnitude: number,
  degreesOfFreedom: number,
  maximumIterations = 400
): number {
  const alpha = degreesOfFreedom / 2;
  const beta = 0.5;
  const ratio = magnitude / Math.sqrt(degreesOfFreedom);
  let betaArgument: number;
  let complement: number;
  let logArgument: number;
  let logComplement: number;
  if (ratio <= 1) {
    const squared = ratio * ratio;
    complement = squared / (1 + squared);
    betaArgument = 1 / (1 + squared);
    logArgument = -Math.log1p(squared);
    logComplement = 2 * Math.log(ratio) - Math.log1p(squared);
  } else {
    const inverse = Math.sqrt(degreesOfFreedom) / magnitude;
    const squared = inverse * inverse;
    betaArgument = squared / (1 + squared);
    complement = 1 / (1 + squared);
    logArgument = 2 * Math.log(inverse) - Math.log1p(squared);
    logComplement = -Math.log1p(squared);
  }
  // Stable gamma ratio for large df; subtracting two O(df log df) log-gammas loses it.
  const inverseAlpha = 1 / alpha;
  const logGammaRatio =
    alpha >= 32
      ? 0.5 * Math.log(alpha) - inverseAlpha / 8 + inverseAlpha ** 3 / 192 - inverseAlpha ** 5 / 640
      : getLogGammaHalfRatio(alpha);
  const logPrefactor =
    logGammaRatio - 0.5 * Math.log(Math.PI) + alpha * logArgument + beta * logComplement;
  const prefactor = Math.exp(logPrefactor);
  if (betaArgument < (alpha + 1) / (alpha + beta + 2)) {
    return Math.min(
      1,
      (prefactor * getBetaContinuedFraction(alpha, beta, betaArgument, maximumIterations)) / alpha
    );
  }
  return Math.max(
    0,
    1 - (prefactor * getBetaContinuedFraction(beta, alpha, complement, maximumIterations)) / beta
  );
}
