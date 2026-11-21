// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Number of float32 elements in a {@link GPUNetworkLineGraph} parameter view. */
export const GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH = 8;

/** Per-frame turn settings of {@link GPUNetworkLineGraph}. */
export type GPUNetworkLineGraphSettings = {
  /** Cost per radian of absolute turn angle, so a right angle costs `turnCostPerRadian * pi / 2`. Defaults to 0. */
  turnCostPerRadian?: number;
  /** Extra cost of a left turn (counter-clockwise, `angle > straightAngleRadians`). Defaults to 0. */
  leftTurnCost?: number;
  /** Extra cost of a right turn (clockwise, `angle < -straightAngleRadians`). Defaults to 0. */
  rightTurnCost?: number;
  /**
   * Cost of a U-turn (absolute angle at or above `uTurnAngleRadians`), replacing the angle cost. A
   * negative value bans U-turns. Defaults to -1 (banned).
   */
  uTurnCost?: number;
  /** Absolute angle in radians below which a turn counts as straight for left and right extras. Defaults to 0.5. */
  straightAngleRadians?: number;
  /** Absolute angle in radians at or above which a turn is a U-turn. Defaults to 2.9 (about 166 degrees). */
  uTurnAngleRadians?: number;
  /** Number of active banned turns, exactly representable in float32 and below 2^32; at most half the length of `bannedTurns`. Defaults to 0. */
  bannedTurnCount?: number;
};

/**
 * Packs per-frame {@link GPUNetworkLineGraph} settings.
 *
 * Layout (float32): `[turnCostPerRadian, leftTurnCost, rightTurnCost, uTurnCost, straightAngleRadians,
 * bannedTurnCount, uTurnAngleRadians, 0]`.
 * Costs and angles must remain finite in float32. Angles and ordinary costs are nonnegative;
 * any finite negative `uTurnCost` bans U-turns. `bannedTurnCount` must round-trip exactly
 * through float32 and fit uint32. Negative subnormal U-turn costs encode as -1 to
 * retain the ban on devices that flush subnormals to zero.
 *
 * @param settings Turn settings.
 * @param target Optional destination of at least 8 elements.
 * @throws If the target is too short or a value is invalid.
 */
export function getGPUNetworkLineGraphParameterValues(
  settings: GPUNetworkLineGraphSettings = {},
  target: Float32Array = new Float32Array(GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH) {
    throw new Error(
      `GPUNetworkLineGraph parameter target must hold ${GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH} elements`
    );
  }
  for (const [removed, replacement] of [
    ['angleCost', 'turnCostPerRadian'],
    ['straightAngle', 'straightAngleRadians'],
    ['uTurnAngle', 'uTurnAngleRadians']
  ] as const) {
    validateRequirement(
      'GPUNetworkLineGraph',
      !(removed in settings),
      `${removed} was removed; use ${replacement}`
    );
  }
  const turnCostPerRadian = settings.turnCostPerRadian ?? 0;
  const leftTurnCost = settings.leftTurnCost ?? 0;
  const rightTurnCost = settings.rightTurnCost ?? 0;
  const uTurnCost = settings.uTurnCost ?? -1;
  const straightAngleRadians = settings.straightAngleRadians ?? 0.5;
  const uTurnAngleRadians = settings.uTurnAngleRadians ?? 2.9;
  for (const [name, value] of [
    ['turnCostPerRadian', turnCostPerRadian],
    ['leftTurnCost', leftTurnCost],
    ['rightTurnCost', rightTurnCost],
    ['straightAngleRadians', straightAngleRadians],
    ['uTurnAngleRadians', uTurnAngleRadians]
  ] as const) {
    if (!(value >= 0) || !Number.isFinite(Math.fround(value))) {
      throw new Error(`GPUNetworkLineGraph ${name} must be non-negative and finite in float32`);
    }
  }
  const bannedTurnCount = settings.bannedTurnCount ?? 0;
  if (
    !Number.isFinite(Math.fround(uTurnCost)) ||
    (uTurnCost < 0 && !(Math.fround(uTurnCost) < 0))
  ) {
    throw new Error(
      'GPUNetworkLineGraph uTurnCost must be finite in float32 and preserve a negative ban'
    );
  }
  if (
    !Number.isInteger(bannedTurnCount) ||
    bannedTurnCount < 0 ||
    bannedTurnCount >= 2 ** 32 ||
    Math.fround(bannedTurnCount) !== bannedTurnCount
  ) {
    throw new Error(
      'GPUNetworkLineGraph bannedTurnCount must be an exact float32 integer in [0, 4294967295]'
    );
  }
  target[0] = turnCostPerRadian;
  target[1] = leftTurnCost;
  target[2] = rightTurnCost;
  // Preserve a ban on GPUs that flush negative float32 subnormals to signed zero.
  target[3] = uTurnCost < 0 && uTurnCost > -(2 ** -126) ? -1 : uTurnCost;
  target[4] = straightAngleRadians;
  target[5] = bannedTurnCount;
  target[6] = uTurnAngleRadians;
  target[7] = 0;
  return target;
}
