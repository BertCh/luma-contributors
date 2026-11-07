// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// One sequential color ramp shared by the build-time PNG fallback and the WGSL/GLSL shaders, so the
// static picture and the live canvas show the same colors.

export const VALUE_RAMP: readonly (readonly [number, number, number])[] = [
  [0.04, 0.05, 0.16],
  [0.18, 0.12, 0.42],
  [0.43, 0.15, 0.55],
  [0.82, 0.27, 0.45],
  [0.98, 0.55, 0.3],
  [0.99, 0.9, 0.55]
];

/** Piecewise linear lookup, t in 0..1, returns rgb in 0..1. */
export function sampleValueRamp(t: number): [number, number, number] {
  const clamped = Math.min(1, Math.max(0, t));
  const scaled = clamped * (VALUE_RAMP.length - 1);
  const lower = Math.min(VALUE_RAMP.length - 2, Math.floor(scaled));
  const fraction = scaled - lower;
  const a = VALUE_RAMP[lower];
  const b = VALUE_RAMP[lower + 1];
  return [
    a[0] + (b[0] - a[0]) * fraction,
    a[1] + (b[1] - a[1]) * fraction,
    a[2] + (b[2] - a[2]) * fraction
  ];
}
