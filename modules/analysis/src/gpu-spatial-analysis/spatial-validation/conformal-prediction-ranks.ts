// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {UINT64_WGSL} from '../../analysis-core/kernels/uint64-wgsl';

/** Largest calibration count whose count plus one fits a WGSL uint32. */
export const GPU_CONFORMAL_MAXIMUM_CALIBRATION_COUNT = 0xfffffffe;

/**
 * One-based ranks `floor(alpha (n + 1))` and `ceil((1 - alpha) (n + 1))`, computed
 * exactly for the packed float32 `alpha`. No snapping to a nearby decimal or integer occurs.
 * Thus 0.1 at n = 99 gives ranks 10 / 90, while 0.7 at n = 9 gives 6 / 4: float32
 * 0.7 is slightly below 7/10. Ranks outside [1, n] denote an unbounded side.
 * Accepts counts from zero through `GPU_CONFORMAL_MAXIMUM_CALIBRATION_COUNT`.
 */
export function getConformalPredictionRanks(
  calibrationCount: number,
  miscoverage: number
): {lowerRank: number; upperRank: number} {
  const id = 'getConformalPredictionRanks';
  validateCapacity(
    id,
    'calibrationCount',
    calibrationCount,
    0,
    GPU_CONFORMAL_MAXIMUM_CALIBRATION_COUNT
  );
  const packedMiscoverage = Math.fround(miscoverage);
  validateRequirement(
    id,
    packedMiscoverage > 0 && packedMiscoverage < 1,
    'miscoverage must be in (0, 1) after rounding to float32'
  );
  const bits = new Uint32Array(new Float32Array([packedMiscoverage]).buffer)[0];
  const exponent = bits >>> 23;
  const significand = (bits & 0x7fffff) | (exponent === 0 ? 0 : 0x800000);
  const shift = exponent === 0 ? 149 : 150 - exponent;
  const size = calibrationCount + 1;
  const lowerRank = Number((BigInt(significand) * BigInt(size)) >> BigInt(shift));
  // ceil(size - alpha * size) = size - floor(alpha * size), without rounding 1 - alpha.
  return {lowerRank, upperRank: size - lowerRank};
}

/** Integer implementation of {@link getConformalPredictionRanks}. @internal */
export const CONFORMAL_PREDICTION_RANKS_WGSL = /* wgsl */ `${UINT64_WGSL}
fn getConformalPredictionRanks(calibrationCount: u32, miscoverage: f32) -> vec2<u32> {
  let bits = bitcast<u32>(miscoverage);
  let exponent = bits >> 23u;
  let significand = (bits & 0x7fffffu) | select(0x800000u, 0u, exponent == 0u);
  let shift = select(150u - exponent, 149u, exponent == 0u);
  let size = calibrationCount + 1u;
  let product = multiplyUint32Wide(significand, size);
  let lowerRank = shiftRightUint64(product, shift).y;
  return vec2<u32>(lowerRank, size - lowerRank);
}
`;
