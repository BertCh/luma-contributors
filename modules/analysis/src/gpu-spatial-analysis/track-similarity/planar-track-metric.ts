// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {TRAJECTORY_LOCATION_MEMBERSHIP_WGSL} from '../trajectory-staypoints/trajectory-location-membership';

/** Inclusive membership independent of the rounded norm. @internal */
export function isPlanarTrackPointWithin(
  first: readonly number[],
  second: readonly number[],
  radius: number
): boolean {
  if (!Number.isFinite(radius) || radius < 0) return false;
  const horizontal = Math.abs(first[0] - second[0]);
  const vertical = Math.abs(first[1] - second[1]);
  const largest = Math.max(horizontal, vertical);
  if (largest > radius) return false;
  if (largest === 0) return true;
  const scale = 2 ** -Math.floor(Math.log2(Math.max(radius, largest)));
  const difference = (horizontal * scale) ** 2 + (vertical * scale) ** 2 - (radius * scale) ** 2;
  if (Math.abs(difference) > 1e-14) return difference < 0;
  // Ambiguous binary64 comparisons use exact dyadic integers from the input bits.
  const values = [...first, ...second, radius].map(decomposeNumber);
  const exponent = Math.min(...values.map(value => value.exponent));
  const integers = values.map(value => value.significand << BigInt(value.exponent - exponent));
  const deltaHorizontal = integers[0] - integers[2];
  const deltaVertical = integers[1] - integers[3];
  return deltaHorizontal ** 2n + deltaVertical ** 2n <= integers[4] ** 2n;
}
function decomposeNumber(value: number) {
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, value);
  const bits = view.getBigUint64(0);
  const exponent = Number((bits >> 52n) & 2047n);
  const fraction = bits & ((1n << 52n) - 1n);
  return {
    significand: (bits >> 63n ? -1n : 1n) * (fraction | (exponent ? 1n << 52n : 0n)),
    exponent: exponent ? exponent - 1075 : -1074
  };
}

/** Stored-coordinate comparison shared with trajectory locations. @internal */
export const PLANAR_TRACK_MEMBERSHIP_WGSL = `${TRAJECTORY_LOCATION_MEMBERSHIP_WGSL}
fn pointWithin(a: vec2f, b: vec2f, radius: f32) -> bool {
  return withinRadius(a, b, radius);
}`;
