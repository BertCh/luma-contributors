// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

const MAXIMUM_FLOAT32 = 3.4028234663852886e38;

/**
 * Convert host slider bounds into an equivalent inclusive interval of stored scalar values.
 * Directed rounding matters: nearest Float32 rounding can include a source value below a lower
 * bound, while integer columns still accept fractional host limits. Empty intervals use harmless
 * zero parameters plus an explicit empty flag, never invalid shader controls or reversed ranges.
 */
export function normalizeKeplerScalarRange(
  format: 'float32' | 'uint32' | 'sint32',
  minimum: number,
  maximum: number,
  exclusiveMaximum = false
): {minimum: number; maximum: number; empty: boolean} {
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum))
    throw new Error('Scalar range boundaries must be finite');
  if (minimum > maximum) throw new Error('Reversed range; use setEmpty for an empty intersection');
  let lower: number;
  let upper: number;
  if (format === 'float32') {
    lower = Math.fround(minimum);
    if (lower < minimum) lower = adjacentFloat32(lower, 1);
    upper = Math.fround(maximum);
    if (upper > maximum || (exclusiveMaximum && upper === maximum))
      upper = adjacentFloat32(upper, -1);
    lower = Math.max(-MAXIMUM_FLOAT32, lower);
    upper = Math.min(MAXIMUM_FLOAT32, upper);
  } else {
    lower = Math.max(format === 'uint32' ? 0 : -0x80000000, Math.ceil(minimum));
    upper = Math.min(
      format === 'uint32' ? 0xffffffff : 0x7fffffff,
      exclusiveMaximum ? Math.ceil(maximum) - 1 : Math.floor(maximum)
    );
  }
  if (lower > upper) return {minimum: 0, maximum: 0, empty: true};
  return {minimum: lower, maximum: upper, empty: false};
}

/** Move by one representable value, including signed zero and finite/infinite endpoints. */
function adjacentFloat32(value: number, direction: 1 | -1): number {
  if (value === 0) return direction * 2 ** -149;
  if (value === Infinity) return direction < 0 ? MAXIMUM_FLOAT32 : Infinity;
  if (value === -Infinity) return direction > 0 ? -MAXIMUM_FLOAT32 : -Infinity;
  const floats = new Float32Array([value]);
  const words = new Uint32Array(floats.buffer);
  words[0] += value > 0 ? direction : -direction;
  return floats[0];
}
