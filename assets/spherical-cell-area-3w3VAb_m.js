import{d as e}from"./kernel-C3tfQOn4.js";import{u as t}from"./wgsl-common-DfyXiYB0.js";import{c as n,o as r}from"./wgsl-kernel-nodes-DagJklH2.js";var i=4;function a(t,n){if(!n)throw Error(`${t} longitude-latitude rasters need parameters`);if(e(n,[`float32`],`${t} parameters`),n.length<i)throw Error(`${t} parameters must contain at least ${i} values`)}function o(e){return`
const SPHERE_RADIUS: f32 = ${n(e)};
const DEGREES_TO_RADIANS: f32 = 0.017453292519943295;
fn getCellAreaOfHeight(row: u32, cellHeight: f32) -> f32 {
  let cellWidth = extent[extentOffset + 2u];
  let middle = (extent[extentOffset + 1u] + (f32(row) + 0.5) * cellHeight) * DEGREES_TO_RADIANS;
  // R^2 dLon (sin(top) - sin(bottom)) = 2 R^2 dLon cos(middle) sin(dLat / 2).
  return 2.0 * SPHERE_RADIUS * SPHERE_RADIUS * (cellWidth * DEGREES_TO_RADIANS) * cos(middle) *
    sin(0.5 * cellHeight * DEGREES_TO_RADIANS);
}
fn getCellArea(row: u32) -> f32 {
  return getCellAreaOfHeight(row, extent[extentOffset + 3u]);
}`}function s(e){return`${o(e)}
fn getPlacedCellArea(row: u32) -> f32 {
  let signedCellHeight = extent[extentOffset + 3u];
  let countedRow = select(row, HEIGHT - 1u - row, signedCellHeight < 0.0);
  return getCellAreaOfHeight(countedRow, abs(signedCellHeight));
}`}function c(e,n){return r(e,{id:n.id,operation:n.operation,variant:`spherical-cell-weights${n.cellWeights?`-masked`:``}`,bindings:[{name:`extent`,view:n.extent,type:`f32`,access:`read`},...n.cellWeights?[{name:`cellWeights`,view:n.cellWeights,type:`f32`,access:`read`}]:[],{name:`output`,view:n.output,type:`f32`,access:`read_write`}],invocationCount:n.cellCount,declarations:`const WIDTH: u32 = ${n.width}u;
const HEIGHT: u32 = ${n.height}u;
${s(n.radius)}
${t}`,body:`var weight = ${n.cellWeights?`cellWeights[cellWeightsOffset + index]`:`1.0`};
  weight = weight * getPlacedCellArea(index / WIDTH);
  // Non-finite (NaN or infinity) and non-positive weights carry no area.
  if (!isFiniteValue(weight) || !(weight > 0.0)) {
    weight = 0.0;
  }
  output[outputOffset + index] = weight;`})}export{a as i,s as n,o as r,c as t};