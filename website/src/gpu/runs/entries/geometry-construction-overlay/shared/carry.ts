// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The pointer carries the nearest point of a point set: it leans toward the cursor and lets go at
// a distance, so the pick never pops when another point becomes the nearest.

import {clamp, type Point} from '../../../interaction';
import {chooseSticky, createPointIndex, type PointIndex} from '../../../pick';

export type PointCarry = {
  index: PointIndex;
  /** Carried point row, or -1 before the first pick. */
  row: number;
  offset: [number, number];
};

export function createPointCarry(positions: Float32Array): PointCarry {
  return {index: createPointIndex(positions), row: -1, offset: [0, 0]};
}

/** A copy of `positions` with the carried point moved by the carry's offset. */
export function getCarriedPositions(positions: Float32Array, carry: PointCarry): Float32Array {
  const moved = new Float32Array(positions);
  if (carry.row >= 0) {
    moved[carry.row * 2] += carry.offset[0];
    moved[carry.row * 2 + 1] += carry.offset[1];
  }
  return moved;
}

/**
 * Picks the point nearest to the pointer and sets its offset: all of the way to the cursor within
 * `fullReach`, fading to nothing at `lastReach` (world units). Returns whether anything changed.
 */
export function updatePointCarry(
  carry: PointCarry,
  positions: Float32Array,
  pointer: Point,
  fullReach = 0.08,
  lastReach = 0.3
): {changed: boolean; row: number; at: Point} {
  const row = chooseSticky(carry.index, pointer[0], pointer[1], carry.row);
  if (row < 0) return {changed: false, row, at: pointer};
  const deltaX = pointer[0] - positions[row * 2];
  const deltaY = pointer[1] - positions[row * 2 + 1];
  const weight = clamp((lastReach - Math.hypot(deltaX, deltaY)) / (lastReach - fullReach), 0, 1);
  const offset: [number, number] = [deltaX * weight, deltaY * weight];
  const changed =
    row !== carry.row || offset[0] !== carry.offset[0] || offset[1] !== carry.offset[1];
  carry.row = row;
  carry.offset = offset;
  return {
    changed,
    row,
    at: [positions[row * 2] + offset[0], positions[row * 2 + 1] + offset[1]]
  };
}
