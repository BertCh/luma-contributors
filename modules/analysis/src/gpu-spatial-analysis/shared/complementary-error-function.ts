// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getComplementaryErrorFunction} from '../point-density/complementary-error-function';

/** Compatibility path for the density-owned erfc implementation. */
export {getComplementaryErrorFunction};

/**
 * Standard normal probability in [low, high] (dimensionless z scores).
 * Short intervals use eight-point Gaussian integration, avoiding cancellation even at zero.
 * Other one-sided intervals subtract survival probabilities directly; intervals crossing zero
 * split at zero, using a centered erf series for endpoints within one standard deviation.
 * Reversed endpoints and NaN return NaN; equal endpoints return zero.
 */
export function getNormalIntervalMass(low: number, high: number): number {
  if (Number.isNaN(low) || Number.isNaN(high) || high < low) return NaN;
  if (low === high) return 0;
  const width = high - low;
  if (Number.isFinite(width) && width * Math.max(1, Math.abs(low), Math.abs(high)) <= 0.25) {
    const midpoint = low + width / 2;
    const halfWidth = width / 2;
    // Gauss-Legendre nodes and weights, symmetric pairs on [-1,1].
    const nodes = [0.1834346424956498, 0.525532409916329, 0.7966664774136267, 0.9602898564975363];
    const weights = [0.362683783378362, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763];
    let sum = 0;
    for (let index = 0; index < nodes.length; index++) {
      const offset = halfWidth * nodes[index];
      sum +=
        weights[index] *
        (Math.exp(-0.5 * (midpoint - offset) ** 2) + Math.exp(-0.5 * (midpoint + offset) ** 2));
    }
    return (halfWidth * sum) / Math.sqrt(2 * Math.PI);
  }
  // Match the centered branch at zero instead of inheriting erfc's fitted offset.
  const survival = (value: number) =>
    value <= 1
      ? 0.5 - getCenteredNormalMass(value)
      : 0.5 * getComplementaryErrorFunction(value * Math.SQRT1_2);
  if (low >= 0) return survival(low) - survival(high);
  if (high <= 0) return survival(-high) - survival(-low);
  return getCenteredNormalMass(-low) + getCenteredNormalMass(high);
}

function getCenteredNormalMass(value: number): number {
  if (value > 1) return 0.5 - 0.5 * getComplementaryErrorFunction(value * Math.SQRT1_2);
  let term = value;
  let sum = term;
  for (let order = 1; order <= 20; order++) {
    term *= (-value * value) / (2 * order);
    sum += term / (2 * order + 1);
  }
  return sum / Math.sqrt(2 * Math.PI);
}
