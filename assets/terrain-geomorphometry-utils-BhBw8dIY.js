import{r as e}from"./kernel-C3tfQOn4.js";import{T as t}from"./gpu-contributor-utils-BIwLQON5.js";import{u as n}from"./wgsl-common-DfyXiYB0.js";import{o as r}from"./wgsl-kernel-nodes-DagJklH2.js";import{d as i,f as a}from"./raster-grid-VB0AM_GY.js";function o(e,n,r){i(e,r);let a=n.affine[4],o=a>0?`north`:a<0?`south`:void 0;return t(e,r===void 0||o===void 0||r===o,`rowDirection '${r}' must match the grid: grid.affine[4] is ${a}, so rows advance ${o}; omit rowDirection or correct the grid`),r??o??`south`}function s(e,t,n){n[2]=e.zFactor??1,a(n,t,c)}var c={cellSizeIndex:0,northEdgeIndex:3},l=`
// Value 'overshoot' cells outside the edge sample a, from a and its inward neighbours b and c.
fn extrapolateOutside(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, overshoot: f32, extent: i32) -> vec2<f32> {
  if (a.y == 0.0 || extent < 2) { return a; }
  if (b.y == 0.0) { return vec2<f32>(0.0, 0.0); }
  let firstDifference = a.x - b.x;
  if (extent < 3 || c.y == 0.0) { return vec2<f32>(a.x + overshoot * firstDifference, 1.0); }
  let secondDifference = b.x - c.x;
  return vec2<f32>(
    a.x + 0.5 * overshoot * (3.0 * firstDifference - secondDifference) +
      0.5 * overshoot * overshoot * (firstDifference - secondDifference),
    1.0
  );
}
// 'column' is inside the grid; 'row' may be outside.
fn readRowExtendedSample(column: i32, row: i32) -> vec2<f32> {
  if (row >= 0 && row < HEIGHT_SIGNED) { return readGridSample(column, row); }
  let isBefore = row < 0;
  let edge = select(HEIGHT_SIGNED - 1, 0, isBefore);
  let inward = select(-1, 1, isBefore);
  let overshoot = f32(select(row - (HEIGHT_SIGNED - 1), -row, isBefore));
  return extrapolateOutside(
    readGridSample(column, edge),
    readGridSample(column, clamp(edge + inward, 0, HEIGHT_SIGNED - 1)),
    readGridSample(column, clamp(edge + 2 * inward, 0, HEIGHT_SIGNED - 1)),
    overshoot,
    HEIGHT_SIGNED
  );
}
fn readExtendedSample(column: i32, row: i32) -> vec2<f32> {
  if (column >= 0 && column < WIDTH_SIGNED) { return readRowExtendedSample(column, row); }
  let isBefore = column < 0;
  let edge = select(WIDTH_SIGNED - 1, 0, isBefore);
  let inward = select(-1, 1, isBefore);
  let overshoot = f32(select(column - (WIDTH_SIGNED - 1), -column, isBefore));
  return extrapolateOutside(
    readRowExtendedSample(edge, row),
    readRowExtendedSample(clamp(edge + inward, 0, WIDTH_SIGNED - 1), row),
    readRowExtendedSample(clamp(edge + 2 * inward, 0, WIDTH_SIGNED - 1), row),
    overshoot,
    WIDTH_SIGNED
  );
}`;function u(t,i){let a=i.width+2,o=i.height+2,s=a*o,c=e(t,`${i.id}-padded-values`,`float32`,s),u=e(t,`${i.id}-padded-validity`,`uint32`,s),d=r(t,{id:`${i.id}-one-sided-padding`,operation:i.operation,variant:`one-sided-padding`,bindings:[{name:`sourceValues`,view:i.band.storage.values,type:`f32`,access:`read`},{name:`sourceValidity`,view:i.band.validity,type:`u32`,access:`read`},{name:`paddedValues`,view:c,type:`f32`,access:`read_write`},{name:`paddedValidity`,view:u,type:`u32`,access:`read_write`}],invocationCount:s,declarations:`const WIDTH_SIGNED: i32 = ${i.width};
const HEIGHT_SIGNED: i32 = ${i.height};
const PADDED_WIDTH: u32 = ${a}u;
${n}
fn readGridSample(column: i32, row: i32) -> vec2<f32> {
  let sampleIndex = u32(row * WIDTH_SIGNED + column);
  let value = sourceValues[sourceValuesOffset + sampleIndex];
  let isValid = sourceValidity[sourceValidityOffset + sampleIndex] != 0u && isFiniteValue(value);
  return vec2<f32>(select(0.0, value, isValid), select(0.0, 1.0, isValid));
}
${l}`,body:`let paddedRow = i32(index / PADDED_WIDTH);
  let paddedColumn = i32(index % PADDED_WIDTH);
  let extended = readExtendedSample(paddedColumn - 1, paddedRow - 1);
  let isValid = extended.y != 0.0 && isFiniteValue(extended.x);
  paddedValues[paddedValuesOffset + index] =
    select(bitcast<f32>(0x7fc00000u | (index & 0u)), extended.x, isValid);
  paddedValidity[paddedValidityOffset + index] = select(0u, 1u, isValid);`});return{band:{id:`${i.id}-padded`,format:`float32`,storage:{kind:`buffer`,values:c},validity:u},width:a,height:o,nodes:[d]}}function d(e){return e?`let gradientIndex = (index / WIDTH + 1u) * (WIDTH + 2u) + index % WIDTH + 1u;`:`let gradientIndex = index;`}export{o as a,d as i,l as n,s as o,u as r,c as t};