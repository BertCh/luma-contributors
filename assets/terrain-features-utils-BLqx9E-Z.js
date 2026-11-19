import{a as e}from"./wgsl-common-DfyXiYB0.js";var t=16,n=64;function r(e,r){let i=r??t;if(!Number.isInteger(i)||i<1||i>n)throw Error(`${e} maximumRadiusPixels must be an integer in [1, ${n}]`);return i}function i(e,t){if(!t.every(e=>Number.isFinite(e)&&e>0))throw Error(`${e} cell size must be finite and positive`)}function a(e,t,n,r,i=!1){if(!Number.isFinite(n)||(i?n<=r:n<r))throw Error(`${e} ${t} must be finite and ${i?`greater than`:`at least`} ${r}`)}var o=`
const PI: f32 = 3.141592653589793;
const NAN_BITS: u32 = 0x7fc00000u;
const INFINITY_BITS: u32 = 0x7f800000u;
// WGSL rejects constant-evaluated NaN and infinity; a runtime zero (seed & 0u) defers evaluation.

${e}fn getInfinity(seed: u32) -> f32 { return bitcast<f32>(INFINITY_BITS | (seed & 0u)); }`,s=`
fn isInDisc(pixel: vec2<i32>, centre: vec2<f32>, cell: vec2<f32>, radiusSquared: f32) -> bool {
  let scaledX = (f32(pixel.x) - centre.x) * cell.x;
  let scaledY = (f32(pixel.y) - centre.y) * cell.y;
  return scaledX * scaledX + scaledY * scaledY <= radiusSquared;
}
fn isOnDiscRing(pixel: vec2<i32>, centre: vec2<f32>, cell: vec2<f32>, radiusSquared: f32) -> bool {
  for (var neighborY = -1; neighborY <= 1; neighborY++) {
    for (var neighborX = -1; neighborX <= 1; neighborX++) {
      if ((neighborX != 0 || neighborY != 0) &&
          !isInDisc(pixel + vec2<i32>(neighborX, neighborY), centre, cell, radiusSquared)) {
        return true;
      }
    }
  }
  return false;
}
fn getMaximumRadius(cell: vec2<f32>) -> f32 {
  return min(f32(MAXIMUM_RADIUS_PIXELS) * cell.x, f32(MAXIMUM_RADIUS_PIXELS) * cell.y);
}
fn isRadiusClamped(requested: f32, cell: vec2<f32>) -> bool {
  return requested > getMaximumRadius(cell);
}
fn getEffectiveRadius(requested: f32, cell: vec2<f32>) -> f32 {
  return min(requested, getMaximumRadius(cell));
}
fn getDiscExtent(radius: f32, cellSize: f32) -> i32 {
  return min(i32(floor(radius / cellSize + 0.5)) + 1, MAXIMUM_RADIUS_PIXELS);
}`,c=`
fn isOffsetInDisc(dx: i32, dy: i32, cell: vec2<f32>, radiusSquared: f32) -> bool {
  return isInDisc(vec2<i32>(dx, dy), vec2<f32>(0.0, 0.0), cell, radiusSquared);
}
fn getDiscRowHalfWidth(dy: i32, cell: vec2<f32>, radiusSquared: f32) -> i32 {
  let scaledY = f32(dy) * cell.y;
  let remaining = max(radiusSquared - scaledY * scaledY, 0.0);
  var halfWidth = clamp(i32(floor(sqrt(remaining) / cell.x)), 0, MAXIMUM_RADIUS_PIXELS + 1);
  while (halfWidth > 0 && !isOffsetInDisc(halfWidth, dy, cell, radiusSquared)) { halfWidth--; }
  while (halfWidth <= MAXIMUM_RADIUS_PIXELS && isOffsetInDisc(halfWidth + 1, dy, cell, radiusSquared)) { halfWidth++; }
  if (!isOffsetInDisc(halfWidth, dy, cell, radiusSquared)) { return -1; }
  return halfWidth;
}
fn getDiscColumnHalfHeight(cell: vec2<f32>, radiusSquared: f32) -> i32 {
  var halfHeight = clamp(i32(floor(sqrt(radiusSquared) / cell.y)), 0, MAXIMUM_RADIUS_PIXELS + 1);
  while (halfHeight > 0 && !isOffsetInDisc(0, halfHeight, cell, radiusSquared)) { halfHeight--; }
  while (halfHeight <= MAXIMUM_RADIUS_PIXELS && isOffsetInDisc(0, halfHeight + 1, cell, radiusSquared)) { halfHeight++; }
  return halfHeight;
}`;export{r as a,i,s as n,a as o,o as r,c as t};