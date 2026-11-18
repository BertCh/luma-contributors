// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Exact 1-based ceil(percentile * departureCount / 100) for validated percentile options.
 * Decompose the supplied binary64 percentile so neither division rounding (56% of 25) nor
 * a fixed epsilon can move the rank across a boundary. Called once per output column on host.
 * @internal
 */
export function getNetworkTravelTimePercentileRank(
  percentile: number,
  departureCount: number
): number {
  const representation = new DataView(new ArrayBuffer(8));
  representation.setFloat64(0, percentile);
  const bits = representation.getBigUint64(0);
  const exponent = Number((bits >> 52n) & 0x7ffn);
  const fraction = bits & 0x000fffffffffffffn;
  const significand = exponent === 0 ? fraction : fraction | (1n << 52n);
  // Valid percentiles are in (0, 100], so the binary exponent always gives a denominator.
  const denominator = 100n << BigInt(exponent === 0 ? 1074 : 1075 - exponent);
  const numerator = significand * BigInt(departureCount);
  return Number((numerator + denominator - 1n) / denominator);
}
