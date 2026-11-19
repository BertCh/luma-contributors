import{l as e}from"./gpu-command-graph-BdzdmV6u.js";import{d as t}from"./kernel-C3tfQOn4.js";import{c as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{u as r}from"./wgsl-common-DfyXiYB0.js";import{d as i}from"./raster-algebra-input-5ZxymrFJ.js";function a(n,r,i,a,o){if(i){if(i instanceof e)throw Error(`${n} ${r} must be a single packed view, not a chunked vector`);if(t(i,[a],`${n} ${r}`),i.length<o)throw Error(`${n} ${r} must hold at least ${o} rows`)}}function o(e,t,n){i(e,t,n)}var s=`
fn isSaddleJoined(v: vec4<f32>, level: f32) -> bool {
  let magnitude = max(abs(level), max(max(abs(v.x), abs(v.y)), max(abs(v.z), abs(v.w))));
  var delta = v - vec4<f32>(level);
  if (magnitude > 1.0e18) {
    delta = v / magnitude - vec4<f32>(level / magnitude);
  }
  let determinant = delta.x * delta.z - delta.y * delta.w;
  return select(determinant <= 0.0, determinant >= 0.0, v.x >= level);
}`;function c(e,t){return`
${r}
fn readCell(cell: u32, value: ptr<function, f32>) -> bool {
  let sample = raster[rasterOffset + cell];
  *value = sample;
  // Compare bits: NaN and infinities are nodata, since a crossing cannot interpolate them.
  var isValid = isFiniteValue(sample);
  ${t===void 0?``:`isValid = isValid && sample != ${n(t)};`}
  ${e?`isValid = isValid && validity[validityOffset + cell] != 0u;`:``}
  return isValid;
}

struct Corners {
  valid: bool,
  v: vec4<f32>
}

fn loadCorners(cx: u32, cy: u32) -> Corners {
  var v0 = 0.0;
  var v1 = 0.0;
  var v2 = 0.0;
  var v3 = 0.0;
  let ok0 = readCell(cy * WIDTH + cx, &v0);
  let ok1 = readCell(cy * WIDTH + cx + 1u, &v1);
  let ok2 = readCell((cy + 1u) * WIDTH + cx + 1u, &v2);
  let ok3 = readCell((cy + 1u) * WIDTH + cx, &v3);
  var corners: Corners;
  corners.valid = ok0 && ok1 && ok2 && ok3;
  corners.v = vec4<f32>(v0, v1, v2, v3);
  return corners;
}

// Bit i is set when corner i (v0..v3) is high, i.e. value >= level.
fn getHighMask(v: vec4<f32>, level: f32) -> u32 {
  return select(0u, 1u, v.x >= level) | select(0u, 2u, v.y >= level) |
    select(0u, 4u, v.z >= level) | select(0u, 8u, v.w >= level);
}

// Counter-clockwise boundary walk v0 -> v1 -> v2 -> v3 -> v0: edge k runs from corner k to corner
// (k + 1) % 4, i.e. edges e0 bottom, e1 right, e2 top, e3 left. An exit is high -> low (the
// segment starts there, keeping high on its left); an entry is low -> high (the segment ends).
fn getRotatedMask(mask: u32) -> u32 {
  return ((mask >> 1u) | (mask << 3u)) & 15u;
}
fn getExitMask(mask: u32) -> u32 { return mask & ~getRotatedMask(mask) & 15u; }
fn getEntryMask(mask: u32) -> u32 { return ~mask & getRotatedMask(mask) & 15u; }

fn getEdgeId(k: u32, cx: u32, cy: u32) -> u32 {
  let horizontalCount = HEIGHT * (WIDTH - 1u);
  if (k == 0u) { return cy * (WIDTH - 1u) + cx; }
  if (k == 2u) { return (cy + 1u) * (WIDTH - 1u) + cx; }
  if (k == 3u) { return horizontalCount + cy * WIDTH + cx; }
  return horizontalCount + cy * WIDTH + cx + 1u;
}`}export{a as i,c as n,o as r,s as t};