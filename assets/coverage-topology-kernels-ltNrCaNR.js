import{c as e,o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as n}from"./exact-orientation-wgsl-B_jI-HEc.js";import{l as r}from"./wgsl-common-DfyXiYB0.js";var i=2**24;function a(t,n){return`
const SNAP_TOLERANCE: f32 = ${e(t)};
const SNAP_MAXIMUM_STEPS: f32 = ${e(i)};

fn getSnapOrigin() -> vec2f {
  let firstVertex = ringOffsets[ringOffsetsOffset];
  if (firstVertex >= ringOffsets[ringOffsetsOffset + ${n}u] ||
      firstVertex >= (arrayLength(&positions) - positionsOffset) / 2u) { return vec2f(0.0); }
  let first = vec2f(positions[positionsOffset + 2u * firstVertex], positions[positionsOffset + 2u * firstVertex + 1u]);
  return select(vec2f(0.0), first, isFiniteFloat(first.x) && isFiniteFloat(first.y));
}

fn quantise(value: f32, origin: f32) -> u32 {
  if (SNAP_TOLERANCE <= 0.0) {
    return bitcast<u32>(select(value, 0.0, value == 0.0));
  }
  let steps = (value - origin) / SNAP_TOLERANCE;
  let rounded = select(floor(steps + 0.5), steps, abs(steps) >= 8388608.0);
  return bitcast<u32>(i32(clamp(rounded, -2147483520.0, 2147483520.0))) ^ 0x80000000u;
}

fn isSnapRangeExceeded(value: f32, origin: f32) -> bool {
  return SNAP_TOLERANCE > 0.0 && abs((value - origin) / SNAP_TOLERANCE) >= SNAP_MAXIMUM_STEPS;
}
`}function o(e,n){let i=t(e,{id:`${n.id}-clear`,operation:n.operation,variant:`snap-range-clear`,bindings:[{name:`uncertainCount`,view:n.uncertainCount,type:`u32`,access:`read_write`}],invocationCount:1,body:`uncertainCount[uncertainCountOffset] = 0u;`});return n.snapTolerance>0?[i,t(e,{id:n.id,operation:n.operation,variant:`snap-range`,bindings:[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:n.ringOffsets,type:`u32`,access:`read`},{name:`uncertainCount`,view:n.uncertainCount,type:`atomic<u32>`,access:`read_write`}],invocationCount:n.positions.length,declarations:`${r}
${a(n.snapTolerance,n.ringOffsets.length-1)}`,body:`if (index < ringOffsets[ringOffsetsOffset] || index >= ringOffsets[ringOffsetsOffset + ${n.ringOffsets.length-1}u]) { return; }
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  if (!isFiniteFloat(x) || !isFiniteFloat(y)) { return; }
  let origin = getSnapOrigin();
  if (isSnapRangeExceeded(x, origin.x) || isSnapRangeExceeded(y, origin.y)) {
    atomicAdd(&uncertainCount[uncertainCountOffset], 1u);
  }`})]:[i]}function s(e,r){let{positions:i,ringOffsets:a,polygonOffsets:o,ringFlip:s}=r,c=a.length-1,l=o.length-1;return t(e,{id:r.id,operation:r.operation,variant:`ring-orientation`,bindings:[{name:`positions`,view:i,type:`f32`,access:`read`},{name:`ringOffsets`,view:a,type:`u32`,access:`read`},{name:`polygonOffsets`,view:o,type:`u32`,access:`read`},{name:`ringFlip`,view:s,type:`u32`,access:`read_write`}],invocationCount:c,declarations:`const VERTEX_COUNT: u32 = ${i.length}u;
const POLYGON_COUNT: u32 = ${l}u;
${n}
fn readRingPoint(row: u32) -> vec2f {
  return vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
}`,body:`let begin = ringOffsets[ringOffsetsOffset + index];
  let end = min(ringOffsets[ringOffsetsOffset + index + 1u], VERTEX_COUNT);
  var flip = 0u;
  if (end > begin + 2u && index < polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT]) {
    var polygon = 0u;
    var high = POLYGON_COUNT;
    while (polygon + 1u < high) {
      let middle = (polygon + high) / 2u;
      if (polygonOffsets[polygonOffsetsOffset + middle] <= index) {
        polygon = middle;
      } else {
        high = middle;
      }
    }
    let isShell = polygonOffsets[polygonOffsetsOffset + polygon] == index;
    // Exact turn at the lexicographically smallest vertex, between its nearest distinct neighbours.
    var extreme = begin;
    var center = readRingPoint(begin);
    for (var row = begin + 1u; row < end; row++) {
      let candidate = readRingPoint(row);
      if (candidate.x < center.x || (candidate.x == center.x && candidate.y < center.y)) {
        extreme = row;
        center = candidate;
      }
    }
    let count = end - begin;
    var before = center;
    var after = center;
    var foundBefore = false;
    var foundAfter = false;
    for (var step = 1u; step < count; step++) {
      if (!foundBefore) {
        before = readRingPoint(begin + (extreme - begin + count - step) % count);
        foundBefore = any(before != center);
      }
      if (!foundAfter) {
        after = readRingPoint(begin + (extreme - begin + step) % count);
        foundAfter = any(after != center);
      }
      if (foundBefore && foundAfter) { break; }
    }
    var winding = 0;
    if (foundBefore && foundAfter) { winding = orientSign(before, center, after); }
    if (winding == 0 || winding == 2) {
      // No certified turn: the sign of the float32 shoelace sum about the first vertex.
      let origin = readRingPoint(begin);
      var previous = vec2f(0.0);
      var twiceArea = 0.0;
      for (var vertex = begin + 1u; vertex < end; vertex++) {
        let current = readRingPoint(vertex) - origin;
        twiceArea += previous.x * current.y - previous.y * current.x;
        previous = current;
      }
      winding = i32(sign(twiceArea));
    }
    flip = select(select(0u, 1u, winding > 0), select(0u, 1u, winding < 0), isShell);
  }
  ringFlip[ringFlipOffset + index] = flip;`})}export{s as n,a as r,o as t};