// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL of the seasonal-trend design shared by the RTS-04 monitors. Requires
 * `getGPURasterDateAxisDeclarations` (for `rasterSliceYears`). Declares `sliceYears`, a private
 * per-slice cache of centred fractional years filled by `fillRasterSliceYears()`, and
 * `getHarmonicTrendColumn(slice, column, period)`: column 0 is the intercept, 1 the slope in
 * years, then a cosine and a sine per harmonic. A model without a trend skips column 1 through
 * `includeTrendColumn`.
 */
export function getRasterHarmonicTrendDesignWGSL(
  sliceCount: number,
  includeTrendColumn = true
): string {
  const trendShift = includeTrendColumn ? 0 : 1;
  return `
var<private> sliceYears: array<f32, ${sliceCount}>;
fn fillRasterSliceYears() {
  for (var slice = 0u; slice < ${sliceCount}u; slice++) { sliceYears[slice] = rasterSliceYears(slice); }
}
fn getHarmonicTrendColumn(slice: u32, column: u32, period: f32) -> f32 {
  if (column == 0u) { return 1.0; }
  let designColumn = column + ${trendShift}u;
  if (designColumn == 1u) { return sliceYears[slice]; }
  let harmonic = designColumn / 2u;
  let angle = 6.283185307179586 * f32(harmonic) * sliceYears[slice] / period;
  return select(sin(angle), cos(angle), designColumn % 2u == 0u);
}
`;
}
