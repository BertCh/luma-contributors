// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Compile-time settings shared by Gaussian score forms. */
export type GaussianPredictiveScoreOptions = {
  /** Strict exceedance event y > threshold, required when requesting the Brier score. */
  brierThreshold?: number;
  /** Two-tailed noncoverage probability of the central interval; defaults to 0.05. */
  intervalAlpha?: number;
};

/** Validated float32 settings and the central Gaussian interval multiplier. @internal */
export function resolveGaussianPredictiveScoreOptions(
  id: string,
  options: GaussianPredictiveScoreOptions,
  withBrier: boolean,
  withInterval: boolean
): {brierThreshold: number; intervalAlpha: number; intervalMultiplier: number} {
  validateRequirement(
    id,
    !withBrier || options.brierThreshold !== undefined,
    'brierThreshold is required with output.brierScore to define the event; supply a finite threshold'
  );
  const brierThreshold = Math.fround(options.brierThreshold ?? 0);
  validateFiniteNumber(id, 'brierThreshold', brierThreshold);
  const intervalAlpha = Math.fround(options.intervalAlpha ?? 0.05);
  validateRequirement(
    id,
    Number.isFinite(intervalAlpha) && intervalAlpha >= 2 * 2 ** -126 && intervalAlpha < 1,
    'intervalAlpha must have a normal float32 half-tail and be below 1 after float32 rounding; use a value in [2^-125, 1) representable in float32'
  );
  return {
    brierThreshold,
    intervalAlpha,
    intervalMultiplier: withInterval ? getGaussianIntervalMultiplier(intervalAlpha) : 0
  };
}

function evaluatePolynomial(coefficients: readonly number[], argument: number): number {
  return coefficients.reduce((value, coefficient) => value * argument + coefficient, 0);
}

/** Central N(0,1) interval half-width, -Phi^-1(alpha/2), Wichura AS 241. @internal */
export function getGaussianIntervalMultiplier(alpha: number): number {
  const probability = alpha / 2;
  const centered = probability - 0.5;
  if (Math.abs(centered) <= 0.425) {
    const radius = 0.180625 - centered * centered;
    return (
      (-centered *
        evaluatePolynomial(
          [
            2509.0809287301227, 33430.57558358813, 67265.7709270087, 45921.95393154987,
            13731.69376550946, 1971.5909503065514, 133.14166789178438, 3.3871328727963665
          ],
          radius
        )) /
      evaluatePolynomial(
        [
          5226.495278852855, 28729.085735721943, 39307.89580009271, 21213.794301586596,
          5394.196021424751, 687.1870074920579, 42.31333070160091, 1
        ],
        radius
      )
    );
  }
  const radius = Math.sqrt(-Math.log(probability));
  return radius <= 5
    ? evaluatePolynomial(
        [
          0.0007745450142783414, 0.022723844989269185, 0.24178072517745062, 1.2704582524523684,
          3.6478483247632045, 5.769497221460691, 4.630337846156546, 1.4234371107496835
        ],
        radius - 1.6
      ) /
        evaluatePolynomial(
          [
            1.0507500716444169e-9, 0.0005475938084995345, 0.015198666563616458, 0.14810397642748008,
            0.6897673349851, 1.6763848301838038, 2.053191626637759, 1
          ],
          radius - 1.6
        )
    : evaluatePolynomial(
        [
          2.0103343992922881e-7, 0.000027115555687434876, 0.0012426609473880784,
          0.026532189526576123, 0.2965605718285049, 1.7848265399172913, 5.463784911164114,
          6.657904643501103
        ],
        radius - 5
      ) /
        evaluatePolynomial(
          [
            2.0442631033899397e-15, 1.421511758316446e-7, 0.000018463183175100546,
            0.0007868691311456133, 0.014875361290850615, 0.1369298809227358, 0.5998322065558879, 1
          ],
          radius - 5
        );
}

/** Positive series for erf, with a negligible binary64 tail beyond six. @internal */
export function getGaussianScoreErrorFunction(magnitude: number): number {
  if (magnitude > 6) return 1;
  let term = magnitude;
  let sum = term;
  for (let iteration = 1; iteration < 240; iteration++) {
    term *= (2 * magnitude * magnitude) / (2 * iteration + 1);
    sum += term;
    if (term <= 1e-16 * sum) break;
  }
  return Math.min(1, (2 / Math.sqrt(Math.PI)) * Math.exp(-magnitude * magnitude) * sum);
}

/** Direct small normal tail; Numerical Recipes erfc fit (relative error < 1.2e-7). @internal */
export function getGaussianScoreSmallTail(magnitude: number): number {
  const argument = magnitude / Math.SQRT2;
  const fraction = 1 / (1 + 0.5 * argument);
  return (
    0.5 *
    fraction *
    Math.exp(
      -argument * argument -
        1.26551223 +
        fraction *
          (1.00002368 +
            fraction *
              (0.37409196 +
                fraction *
                  (0.09678418 +
                    fraction *
                      (-0.18628806 +
                        fraction *
                          (0.27886807 +
                            fraction *
                              (-1.13520398 +
                                fraction *
                                  (1.48851587 +
                                    fraction * (-0.82215223 + fraction * 0.17087277))))))))
    )
  );
}
