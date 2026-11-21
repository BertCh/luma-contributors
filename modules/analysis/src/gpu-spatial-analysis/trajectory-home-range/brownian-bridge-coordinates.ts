// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ADDITION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/addition-rounding-error-wgsl';
import {MULTIPLICATION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/multiplication-rounding-error-wgsl';

// These host transforms reproduce the integer WGSL error helpers for finite float32 operands.
function getBridgeAdditionError(first: number, second: number, sum: number): number {
  if (!Number.isFinite(sum) || Math.min(Math.abs(first), Math.abs(second)) < 2 ** -103) return 0;
  if (sum === first) return second;
  if (sum === second) return first;
  return Math.fround(first + second - sum);
}

function getBridgeMultiplicationError(first: number, second: number, product: number): number {
  const exponent = (value: number) => Math.floor(Math.log2(Math.abs(value)));
  if (
    !Number.isFinite(product) ||
    Math.abs(first) < 2 ** -126 ||
    Math.abs(second) < 2 ** -126 ||
    Math.abs(product) < 2 ** -126 ||
    exponent(first) + exponent(second) < -80
  )
    return 0;
  return Math.fround(first * second - product);
}

/** Float32 fix offset from a cell centre, retaining cancellation residuals before the final rounding. @internal */
export function getBrownianBridgeCellFixOffset(
  position: number,
  minimum: number,
  maximum: number,
  cell: number,
  dimension: number
): number {
  const rounded = Math.fround;
  const numerator = rounded(cell + 0.5);
  const fraction = rounded(numerator / dimension);
  const fractionProduct = rounded(fraction * dimension);
  const fractionLow = rounded(
    rounded(
      numerator -
        fractionProduct -
        getBridgeMultiplicationError(fraction, dimension, fractionProduct)
    ) / dimension
  );
  const span = rounded(maximum - minimum);
  const spanLow = getBridgeAdditionError(maximum, -minimum, span);
  const center = rounded(span * fraction);
  const centerLow = rounded(
    rounded(getBridgeMultiplicationError(span, fraction, center) + rounded(spanLow * fraction)) +
      rounded(span * fractionLow)
  );
  const fix = rounded(position - minimum);
  const fixLow = getBridgeAdditionError(position, -minimum, fix);
  const difference = rounded(fix - center);
  const correction = rounded(
    rounded(fixLow - centerLow) + getBridgeAdditionError(fix, -center, difference)
  );
  return rounded(difference + correction);
}

/** Shared cutoff uses squared float32 offsets and variance; no sqrt rounding affects inclusion. @internal */
export function isBrownianBridgeOutsideCutoff(
  startX: number,
  startY: number,
  endX: number,
  endY: number,
  duration: number,
  motion: number,
  errorFrom: number,
  errorTo: number
): boolean {
  const rounded = Math.fround;
  const largestError = Math.max(Math.abs(errorFrom), Math.abs(errorTo));
  const variance = rounded(
    rounded(rounded(0.25 * rounded(duration)) * motion) + rounded(largestError * largestError)
  );
  const limit = rounded(49 * variance);
  const outside = (first: number, second: number) => {
    const nearest =
      first > 0 && second > 0
        ? Math.min(first, second)
        : first < 0 && second < 0
          ? Math.max(first, second)
          : 0;
    return rounded(nearest * nearest) > limit;
  };
  return outside(startX, endX) || outside(startY, endY);
}

/** Integer-recovered residuals survive Metal reassociation of floating-point identities. @internal */
export const BROWNIAN_BRIDGE_COORDINATES_WGSL = `${ADDITION_ROUNDING_ERROR_WGSL}
${MULTIPLICATION_ROUNDING_ERROR_WGSL}
fn bridgeFixOffset(position: f32, minimum: f32, maximum: f32, cell: u32, dimension: u32) -> vec2<f32> {
  let numerator = f32(cell) + 0.5;
  let fraction = numerator / f32(dimension);
  let fractionProduct = fraction * f32(dimension);
  let fractionLow = ((numerator - fractionProduct) - getMultiplicationRoundingError(fraction, f32(dimension), fractionProduct)) / f32(dimension);
  let span = maximum - minimum;
  let spanLow = getAdditionRoundingError(maximum, -minimum, span);
  let center = span * fraction;
  let centerLow = (getMultiplicationRoundingError(span, fraction, center) + spanLow * fraction) + span * fractionLow;
  let fix = position - minimum;
  let fixLow = getAdditionRoundingError(position, -minimum, fix);
  let difference = fix - center;
  let correction = (fixLow - centerLow) + getAdditionRoundingError(fix, -center, difference);
  return vec2<f32>(difference, correction);
}
fn bridgeMeanOffset(start: vec2<f32>, end: vec2<f32>, fraction: f32) -> f32 {
  let difference = end.x - start.x;
  let differenceLow = (end.y - start.y) + getAdditionRoundingError(end.x, -start.x, difference);
  let scaled = difference * fraction;
  let scaledLow = getMultiplicationRoundingError(difference, fraction, scaled) + differenceLow * fraction;
  let mean = start.x + scaled;
  let meanLow = (start.y + scaledLow) + getAdditionRoundingError(start.x, scaled, mean);
  return mean + meanLow;
}
fn bridgeAxisOutside(first: f32, second: f32, limit: f32) -> bool {
  var nearest = 0.0;
  if (first > 0.0 && second > 0.0) { nearest = min(first, second); }
  if (first < 0.0 && second < 0.0) { nearest = max(first, second); }
  return nearest * nearest > limit;
}`;
