// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The pointer's distance from an anchor picks one of a few values of a compile-time prop: the
// snapped value settles once the pointer rests, so the graph is not recompiled under a moving hand.

import {clamp, type Point} from '../../../interaction';

/** Distance from the anchor, in world units, that maps to the last level. */
const REACH_SPAN = 0.5;

/** The centroid of packed (x, y) positions. */
export function getCentroid(positions: Float32Array): Point {
  let sumX = 0;
  let sumY = 0;
  const count = positions.length / 2;
  for (let row = 0; row < count; row++) {
    sumX += positions[row * 2];
    sumY += positions[row * 2 + 1];
  }
  return [sumX / count, sumY / count];
}

/** Index of the level the pointer's distance from `anchor` selects. */
export function getReachLevel(anchor: Point, pointer: Point, levelCount: number): number {
  const fraction = clamp(
    Math.hypot(pointer[0] - anchor[0], pointer[1] - anchor[1]) / REACH_SPAN,
    0,
    0.9999
  );
  return Math.floor(fraction * levelCount);
}

/** A rest position, one unit to the right of the anchor, that selects level `index`. */
export function getReachRest(anchor: Point, index: number, levelCount: number): Point {
  return [anchor[0] + ((index + 0.5) / levelCount) * REACH_SPAN, anchor[1]];
}

/**
 * The value of a positive parameter that a distance from the input picks, growing logarithmically:
 * the parameter's minimum at the input, its maximum at `span` world units away. A hand near the
 * input then picks a fine value and one far from it a coarse one, so the whole range is reachable
 * inside the picture.
 */
export function getLogarithmicReach(
  parameter: {min: number; max: number; step: number},
  distance: number,
  span: number
): number {
  const amount = clamp(distance / span, 0, 1);
  const value = parameter.min * Math.pow(parameter.max / parameter.min, amount);
  const steps = Math.round((value - parameter.min) / parameter.step);
  return clamp(
    Number((parameter.min + steps * parameter.step).toPrecision(12)),
    parameter.min,
    parameter.max
  );
}

/** The distance from the input that picks `value` under `getLogarithmicReach`. */
export function getLogarithmicReachDistance(
  parameter: {min: number; max: number},
  value: number,
  span: number
): number {
  return (span * Math.log(value / parameter.min)) / Math.log(parameter.max / parameter.min);
}
