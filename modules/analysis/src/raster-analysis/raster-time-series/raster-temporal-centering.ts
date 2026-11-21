// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getFixedPointHeadroomBits} from '../../analysis-core/kernels/fixed-point-accumulator';

/** Shared covariance/projection body. Requires finite(), FIXED_POINT_WGSL, values, limit and kept.
 * First-sample subtraction is exact by Sterbenz only when all samples have its sign and lie within
 * a factor of two. Otherwise zero is an exact anchor, preserving small signed observations.
 * A local 96-bit sum preserves cancellation regardless of acquisition order. @internal */
export function getRasterTemporalCenteringBody(
  sliceCount: number,
  cell: string,
  hasMask: boolean
): string {
  const normalizationExponent = Math.ceil(Math.log2(sliceCount));
  return `let firstValue = values[valuesOffset + ${cell} * ${sliceCount}u];
var lowest = firstValue;
var highest = firstValue;
for (var slice = 0u; slice < ${sliceCount}u; slice++) {
  let source = ${cell} * ${sliceCount}u + slice;
  let value = values[valuesOffset + source];
  if (!finite(value) || abs(value) > limit ${hasMask ? '|| mask[maskOffset + source] == 0u' : ''}) { kept = false; break; }
  lowest = min(lowest, value);
  highest = max(highest, value);
}
var anchor = 0.0;
if ((firstValue > 0.0 && lowest >= firstValue * 0.5 && highest * 0.5 <= firstValue) ||
    (firstValue < 0.0 && lowest * 0.5 >= firstValue && highest <= firstValue * 0.5)) { anchor = firstValue; }
let meanUnitExponent = getFixedPointUnitExponent(getFixedPointExponentBits(limit), ${getFixedPointHeadroomBits(sliceCount)}u);
var total = vec3<u32>(0u);
if (kept) {
  for (var slice = 0u; slice < ${sliceCount}u; slice++) {
    total = addFixedPoint(total, quantizeFixedPoint(values[valuesOffset + ${cell} * ${sliceCount}u + slice] - anchor, meanUnitExponent));
  }
}
let meanOffset = getFixedPointFloat(total, meanUnitExponent - ${normalizationExponent}) * ${getWGSLFloatLiteral(2 ** normalizationExponent / sliceCount)};`;
}

/** Rounded temporal displacement with an integer subtraction boundary. Metal may otherwise turn
 * `(value - anchor) - meanOffset` into `value - (anchor + meanOffset)`, losing small mean offsets.
 * Requires FIXED_POINT_WGSL. The shared body chooses exact first-anchor subtraction or zero.
 * @internal */
export const RASTER_TEMPORAL_CENTERING_WGSL = /* wgsl */ `
fn getRasterTemporalCenteredValue(sampleValue: f32, sampleAnchor: f32,
  sampleMeanOffset: f32, encodingExponent: i32) -> f32 {
  let displacement = quantizeFixedPoint(sampleValue - sampleAnchor, encodingExponent);
  let offsetWords = quantizeFixedPoint(sampleMeanOffset, encodingExponent);
  return getFixedPointFloat(addFixedPoint(displacement, negateFixedPoint(offsetWords)), encodingExponent);
}`;
