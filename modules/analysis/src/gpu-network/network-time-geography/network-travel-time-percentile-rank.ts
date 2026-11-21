// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Exact 1-based nearest rank `ceil(percentile * departureCount / 100)` for validated percentile
 * options, with the percentile read as the decimal the caller wrote.
 *
 * The percentile is taken as its shortest round-trip decimal (`String(percentile)`, which
 * ECMAScript defines as the shortest decimal that parses back to the same binary64). A literal
 * typed with at most 15 significant digits is therefore read as written: `99.9` is 999/10, not the
 * binary64 just above it, so 99.9% of 1,000 departures is rank 999. The rank is then computed in
 * exact integer arithmetic, so neither division rounding (56% of 25) nor a fixed epsilon can move
 * it across a boundary, and adjacent binary64 values still select their own ranks (the next
 * representable value above 56 prints as `56.00000000000001` and selects rank 15 of 25).
 * Called once per output column on host.
 * @internal
 */
export function getNetworkTravelTimePercentileRank(
  percentile: number,
  departureCount: number
): number {
  const {numerator, denominator} = getShortestDecimalFraction(percentile);
  const scaledNumerator = numerator * BigInt(departureCount);
  const scaledDenominator = 100n * denominator;
  return Number((scaledNumerator + scaledDenominator - 1n) / scaledDenominator);
}

/**
 * Splits the shortest round-trip decimal of a positive finite number into an exact fraction
 * `numerator / denominator` with a power-of-ten denominator.
 */
function getShortestDecimalFraction(value: number): {numerator: bigint; denominator: bigint} {
  const match = /^(\d+)(?:\.(\d+))?(?:e([+-]?\d+))?$/.exec(String(value));
  if (!match) {
    // Validated percentiles are positive and finite, so String() always matches.
    throw new Error(`Percentile ${value} has no decimal representation`);
  }
  const [, integerDigits, fractionDigits = '', exponentText = '0'] = match;
  const digits = BigInt(`${integerDigits}${fractionDigits}`);
  const decimalPlaces = fractionDigits.length - Number(exponentText);
  return decimalPlaces >= 0
    ? {numerator: digits, denominator: 10n ** BigInt(decimalPlaces)}
    : {numerator: digits * 10n ** BigInt(-decimalPlaces), denominator: 1n};
}
