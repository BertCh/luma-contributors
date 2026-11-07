// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Equal-width bin of `value` within `[lower, upper]`, clamped to `0 .. binCount - 1`. Degenerate ranges map to bin 0. */
export const BIN_INDEX_WGSL = /* wgsl */ `
fn getBinIndex(value: f32, lower: f32, upper: f32, binCount: u32) -> u32 {
  if (!(upper > lower)) { return 0u; }
  let fraction = clamp((value - lower) / (upper - lower), 0.0, 1.0);
  return min(u32(fraction * f32(binCount)), binCount - 1u);
}
`;
