import{a as e}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t}from"./validate-outline-output-passes-DIcqH6Pw.js";import{n}from"./refinement-storage-validation-D9BhQGNW.js";import{r}from"./kernel-C3tfQOn4.js";import{E as i,f as a,y as o}from"./gpu-contributor-utils-BIwLQON5.js";import{o as s}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as c}from"./gpu-geometry-orientation-BW3bBZIA.js";import{i as l}from"./spatial-wgsl-point-helpers-DcOgpqmI.js";var u=class{id;props;constructor(e){this.id=e.id??`normalize-polygons`,this.props=e,i(e.positions,[`float32x2`],this.id);for(let t of[`ringOffsets`,`polygonOffsets`,`featureOffsets`])i(e[t],[`uint32`],this.id),a(this.id,t,e[t].length,2);for(let t of[`positions`,`ringOffsets`,`polygonOffsets`])if(i(e.output[t],[t===`positions`?`float32x2`:`uint32`],this.id),e.output[t].length!==e[t].length)throw Error(`Canonical output length mismatch`);t(this.id,e.output,[Object.keys(e.output)]),o(this.id,Object.values(e.output),[e.positions,e.ringOffsets,e.polygonOffsets,e.featureOffsets])}getCommandNodes(t){let{props:i,id:a}=this;e(t,a,Object.fromEntries([i.positions,i.ringOffsets,i.polygonOffsets,i.featureOffsets,...Object.values(i.output)].map((e,t)=>[`binding.${t}`,e]))),n(t,a,{canonical:(i.ringOffsets.length-1)*16,orientedPositions:i.positions.length*8,orientationFlags:(i.ringOffsets.length-1)*4,parameters:16});let o=r(t,`${a}-canonical-rings`,`uint32x4`,i.ringOffsets.length-1),u=r(t,`${a}-oriented`,`float32x2`,i.positions.length),d=r(t,`${a}-orientation-parameters`,`float32`,4),f=[s(t,{id:`${a}-parameters`,operation:`GPUNormalizePolygons`,invocationCount:4,bindings:[{name:`parameters`,view:d,type:`f32`,access:`read_write`}],body:`parameters[parametersOffset + index] = select(0.0, 1.0, index == 0u);`}),...new c({...i,id:`${a}-orient`,mode:`orient-polygons`,parameters:d,output:{positions:u}}).getCommandNodes(t)],p=`${l(`point`,`positions`)}
fn comparePoints(left: vec2f, right: vec2f) -> i32 {
  if (left.x < right.x || (left.x == right.x && left.y < right.y)) { return -1; }
  if (left.x > right.x || (left.x == right.x && left.y > right.y)) { return 1; }
  return 0;
}`;return[...f,s(t,{id:`${a}-rings`,operation:`GPUNormalizePolygons`,variant:`rings`,invocationCount:i.ringOffsets.length-1,bindings:[{name:`positions`,view:u,type:`f32`,access:`read`},{name:`ringOffsets`,view:i.ringOffsets,type:`u32`,access:`read`},{name:`canonical`,view:o,type:`u32`,access:`read_write`}],declarations:p,body:`let first = ringOffsets[ringOffsetsOffset + index];
let total = ringOffsets[ringOffsetsOffset + index + 1u] - first;
var closed = 0u;
if (total > 1u && all(point(first) == point(first + total - 1u))) { closed = 1u; }
let length = total - closed;
var bestStart = 0u; var bestReverse = 0u;
for (var candidate = 0u; candidate < length; candidate++) {
  for (var reverse = 0u; reverse < 1u; reverse++) {
    for (var element = 0u; element < length; element++) {
      let left = point(first + (candidate + select(element, length - element, reverse == 1u)) % length);
      let right = point(first + (bestStart + select(element, length - element, bestReverse == 1u)) % length);
      let comparison = comparePoints(left, right);
      if (comparison < 0) { bestStart = candidate; bestReverse = reverse; }
      if (comparison != 0) { break; }
    }
  }
}
canonical[canonicalOffset + index * 4u] = bestStart;
canonical[canonicalOffset + index * 4u + 1u] = length;
canonical[canonicalOffset + index * 4u + 2u] = bestReverse;
canonical[canonicalOffset + index * 4u + 3u] = closed;`}),s(t,{id:`${a}-parts`,operation:`GPUNormalizePolygons`,variant:`parts`,invocationCount:i.featureOffsets.length-1,bindings:[{name:`positions`,view:u,type:`f32`,access:`read`},{name:`ringOffsets`,view:i.ringOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:i.polygonOffsets,type:`u32`,access:`read`},{name:`featureOffsets`,view:i.featureOffsets,type:`u32`,access:`read`},{name:`canonical`,view:o,type:`u32`,access:`read`},{name:`outPositions`,view:i.output.positions,type:`f32`,access:`read_write`},{name:`outRingOffsets`,view:i.output.ringOffsets,type:`u32`,access:`read_write`},{name:`outPolygonOffsets`,view:i.output.polygonOffsets,type:`u32`,access:`read_write`}],declarations:`${p}
fn canonicalPoint(ring: u32, element: u32) -> vec2f {
  let base = canonicalOffset + ring * 4u;
  let length = canonical[base + 1u];
  let offset = canonical[base] + select(element, length - element, canonical[base + 2u] == 1u);
  return point(ringOffsets[ringOffsetsOffset + ring] + offset % length);
}
fn compareRings(left: u32, right: u32) -> i32 {
  let leftLength = canonical[canonicalOffset + left * 4u + 1u];
  let rightLength = canonical[canonicalOffset + right * 4u + 1u];
  let leftCount = leftLength + canonical[canonicalOffset + left * 4u + 3u];
  let rightCount = rightLength + canonical[canonicalOffset + right * 4u + 3u];
  if (leftCount != rightCount) { return select(1, -1, leftCount < rightCount); }
  for (var element = 0u; element < min(leftLength, rightLength); element++) {
    let comparison = comparePoints(canonicalPoint(left, element), canonicalPoint(right, element));
    if (comparison != 0) { return comparison; }
  }
  if (leftLength != rightLength) { return select(1, -1, leftLength < rightLength); }
  let leftClosed = canonical[canonicalOffset + left * 4u + 3u];
  let rightClosed = canonical[canonicalOffset + right * 4u + 3u];
  return select(select(1, -1, leftClosed < rightClosed), 0, leftClosed == rightClosed);
}
fn sortedRing(polygon: u32, slot: u32) -> u32 {
  let first = polygonOffsets[polygonOffsetsOffset + polygon];
  let last = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
  if (slot == 0u) { return first; }
  for (var candidate = first + 1u; candidate < last; candidate++) {
    var rank = 1u;
    for (var other = first + 1u; other < last; other++) {
      let comparison = compareRings(other, candidate);
      if (comparison > 0 || (comparison == 0 && other < candidate)) { rank++; }
    }
    if (rank == slot) { return candidate; }
  }
  return first;
}
fn comparePolygons(left: u32, right: u32) -> i32 {
  let leftLength = polygonOffsets[polygonOffsetsOffset + left + 1u] - polygonOffsets[polygonOffsetsOffset + left];
  let rightLength = polygonOffsets[polygonOffsetsOffset + right + 1u] - polygonOffsets[polygonOffsetsOffset + right];
  for (var slot = 0u; slot < min(leftLength, rightLength); slot++) {
    let comparison = compareRings(sortedRing(left, slot), sortedRing(right, slot));
    if (comparison != 0) { return comparison; }
  }
  return select(select(1, -1, leftLength < rightLength), 0, leftLength == rightLength);
}`,body:`let firstPolygon = featureOffsets[featureOffsetsOffset + index];
let lastPolygon = featureOffsets[featureOffsetsOffset + index + 1u];
var destinationRing = polygonOffsets[polygonOffsetsOffset + firstPolygon];
var destinationVertex = ringOffsets[ringOffsetsOffset + destinationRing];
for (var slot = 0u; slot < lastPolygon - firstPolygon; slot++) {
  var polygon = firstPolygon;
  for (var candidate = firstPolygon; candidate < lastPolygon; candidate++) {
    var rank = 0u;
    for (var other = firstPolygon; other < lastPolygon; other++) {
      let comparison = comparePolygons(other, candidate);
      if (comparison > 0 || (comparison == 0 && other < candidate)) { rank++; }
    }
    if (rank == slot) { polygon = candidate; break; }
  }
  outPolygonOffsets[outPolygonOffsetsOffset + firstPolygon + slot] = destinationRing;
  let ringCount = polygonOffsets[polygonOffsetsOffset + polygon + 1u] - polygonOffsets[polygonOffsetsOffset + polygon];
  for (var ringSlot = 0u; ringSlot < ringCount; ringSlot++) {
    let ring = sortedRing(polygon, ringSlot);
    outRingOffsets[outRingOffsetsOffset + destinationRing] = destinationVertex;
    let length = canonical[canonicalOffset + ring * 4u + 1u];
    let closed = canonical[canonicalOffset + ring * 4u + 3u];
    for (var element = 0u; element < length + closed; element++) {
      let current = canonicalPoint(ring, element % length);
      outPositions[outPositionsOffset + destinationVertex * 2u] = current.x + 0.0;
      outPositions[outPositionsOffset + destinationVertex * 2u + 1u] = current.y + 0.0;
      destinationVertex++;
    }
    destinationRing++;
  }
}
// A unique invocation owns the two terminal offsets.
if (index == ${i.featureOffsets.length-2}u) {
  outPolygonOffsets[outPolygonOffsetsOffset + lastPolygon] = destinationRing;
  outRingOffsets[outRingOffsetsOffset + destinationRing] = destinationVertex;
}`})]}};export{u as t};