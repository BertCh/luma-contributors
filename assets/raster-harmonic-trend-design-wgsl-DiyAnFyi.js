function e(e,t=!0){return`
var<private> sliceYears: array<f32, ${e}>;
fn fillRasterSliceYears() {
  for (var slice = 0u; slice < ${e}u; slice++) { sliceYears[slice] = rasterSliceYears(slice); }
}
fn getHarmonicTrendColumn(slice: u32, column: u32, period: f32) -> f32 {
  if (column == 0u) { return 1.0; }
  let designColumn = column + ${+!t}u;
  if (designColumn == 1u) { return sliceYears[slice]; }
  let harmonic = designColumn / 2u;
  let angle = 6.283185307179586 * f32(harmonic) * sliceYears[slice] / period;
  return select(sin(angle), cos(angle), designColumn % 2u == 0u);
}
`}export{e as t};