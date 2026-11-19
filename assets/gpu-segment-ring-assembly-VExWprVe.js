import{r as e,u as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-scan-B4NNTfF9.js";import{t as r}from"./gpu-sort-xRm8l5yn.js";import{c as i,o as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as o,f as s,g as c,v as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{v as d}from"./spatial-context-DwZEd4EZ.js";import{a as f,i as p}from"./graph-chunk-dispatch-zBt8k2-o.js";import{i as ee,m,u as h}from"./wgsl-common-DfyXiYB0.js";import{t as g}from"./gpu-bvh-CBtZpuN_.js";import{t as _}from"./exact-orientation-wgsl-B_jI-HEc.js";import{n as v,t as y}from"./exact-geometry-wgsl-y7TUnqOh.js";import{n as b,t as x}from"./construction-snapshot-DdkX9Pc0.js";var S=8388608,C=1.0625;function w(e,t,n){let r=t?`atomicLoad(&counters[countersOffset + ${n}u])`:`counters[countersOffset + ${n}u]`;return`${h}
const TOLERANCE_CELL: f32 = ${e};
const CELL_SLACK: f32 = ${C};
const CELL_BIAS: f32 = ${S}.0;
fn ringOrigin() -> vec2f { return vec2f(endpoints[endpointsOffset], endpoints[endpointsOffset + 1u]); }
// Largest |x - originX| or |y - originY| of any endpoint, measured on the GPU.
fn ringFrameExtent() -> f32 { return bitcast<f32>(${r}); }
fn ringInverseCell() -> f32 {
  // The 1e-30 floor keeps a zero extent with the derived (zero) tolerance finite.
  return 1.0 / (CELL_SLACK * max(max(TOLERANCE_CELL, ringFrameExtent() * ${2**-22}), 1.0e-30));
}
fn ringQuantize(value: f32, origin: f32) -> u32 {
  let offset = value - origin;
  if (!isFiniteValue(offset)) { return u32(CELL_BIAS); }
  return u32(floor(offset * ringInverseCell()) + CELL_BIAS);
}
fn ringHash(qx: u32, qy: u32, group: u32) -> u32 {
  var h = qx * 0x9E3779B1u;
  h = h ^ (h >> 15u);
  h = (h + qy * 0x85EBCA77u) * 0xC2B2AE3Du;
  h = h ^ (h >> 13u);
  h = (h + group * 0x27D4EB2Fu) * 0x165667B1u;
  h = h ^ (h >> 16u);
  return select(h, 0xfffffffeu, h == 0xffffffffu);
}`}function te(t,n){let{id:i,operation:o}=n,s=n.isHole.length,c=2**Math.ceil(Math.log2(s)),l=2*c-1,u=e(t,`${i}-records`,`uint32`,s*3),d=n=>e(t,`${i}-${n}`,`uint32`,s),f=d(`low-keys`),p=d(`high-keys`),h=d(`rows`),_=d(`sorted-low-keys`),v=d(`low-rows`),y=d(`ordered-high-keys`),b=d(`sorted-high-keys`),x=d(`sorted-rows`),S=e(t,`${i}-minima`,`float32x2`,s),C=e(t,`${i}-maxima`,`float32x2`,s),w=[a(t,{id:`${i}-records`,operation:o,variant:`shell-records`,bindings:[{name:`areas`,view:n.areas,type:`f32`,access:`read`},{name:`isHole`,view:n.isHole,type:`u32`,access:`read`},{name:`count`,view:n.count,type:`u32`,access:`read`},...n.groups?[{name:`groups`,view:n.groups,type:`u32`,access:`read`}]:[],{name:`records`,view:u,type:`u32`,access:`read_write`}],invocationCount:s,body:`let base = recordsOffset + 3u * index;
  records[base] = bitcast<u32>(areas[areasOffset + index]);
  records[base + 1u] = select(2u, isHole[isHoleOffset + index], index < count[countOffset]);
  records[base + 2u] = ${n.groups?`groups[groupsOffset + index]`:`0u`};`}),a(t,{id:`${i}-keys`,operation:o,variant:`shell-keys`,bindings:[{name:`bounds`,view:n.bounds,type:`f32`,access:`read`},{name:`records`,view:u,type:`u32`,access:`read`},{name:`lowKeys`,view:f,type:`u32`,access:`read_write`},{name:`highKeys`,view:p,type:`u32`,access:`read_write`},{name:`rows`,view:h,type:`u32`,access:`read_write`}],invocationCount:s,declarations:`${ee}\n${m}`,body:`let base = recordsOffset + 3u * index;
  var low = 0xffffffffu;
  var high = 0xffffffffu;
  if (records[base + 1u] == 0u) {
    // Halve before adding so finite bounds of either sign cannot overflow at their center.
    let boundsBase = boundsOffset + 4u * index;
    let centerX = 0.5 * bounds[boundsBase] + 0.5 * bounds[boundsBase + 2u];
    let centerY = 0.5 * bounds[boundsBase + 1u] + 0.5 * bounds[boundsBase + 3u];
    let coordinateKeyX = getOrderedKey(centerX);
    let coordinateKeyY = getOrderedKey(centerY);
    low = spreadBits(coordinateKeyX & 0xffffu) | (spreadBits(coordinateKeyY & 0xffffu) << 1u);
    high = spreadBits(coordinateKeyX >> 16u) | (spreadBits(coordinateKeyY >> 16u) << 1u);
  }
  lowKeys[lowKeysOffset + index] = low;
  highKeys[highKeysOffset + index] = high;
  rows[rowsOffset + index] = index;`}),...new r({id:`${i}-sort-low`,keys:f,values:h,outputKeys:_,outputValues:v,algorithm:`radix`}).getCommandNodes(t),a(t,{id:`${i}-order-high-keys`,operation:o,variant:`shell-order-high-keys`,bindings:[{name:`highKeys`,view:p,type:`u32`,access:`read`},{name:`rows`,view:v,type:`u32`,access:`read`},{name:`orderedKeys`,view:y,type:`u32`,access:`read_write`}],invocationCount:s,body:`orderedKeys[orderedKeysOffset + index] = highKeys[highKeysOffset + rows[rowsOffset + index]];`}),...new r({id:`${i}-sort-high`,keys:y,values:v,outputKeys:b,outputValues:x,algorithm:`radix`}).getCommandNodes(t),a(t,{id:`${i}-gather`,operation:o,variant:`shell-gather`,bindings:[{name:`bounds`,view:n.bounds,type:`f32`,access:`read`},{name:`records`,view:u,type:`u32`,access:`read`},{name:`rows`,view:x,type:`u32`,access:`read`},{name:`minima`,view:S,type:`f32`,access:`read_write`},{name:`maxima`,view:C,type:`f32`,access:`read_write`}],invocationCount:s,body:`let ring = rows[rowsOffset + index];
  let unused = records[recordsOffset + 3u * ring + 1u] != 0u;
  let base = boundsOffset + 4u * ring;
  minima[minimaOffset + 2u * index] = select(bounds[base], 3.402823466e+38, unused);
  minima[minimaOffset + 2u * index + 1u] = select(bounds[base + 1u], 3.402823466e+38, unused);
  maxima[maximaOffset + 2u * index] = select(bounds[base + 2u], -3.402823466e+38, unused);
  maxima[maximaOffset + 2u * index + 1u] = select(bounds[base + 3u], -3.402823466e+38, unused);`})],te=new g({id:`${i}-bvh`,minima:S,maxima:C,sourceIds:x,leafCapacity:c,nodeMinima:e(t,`${i}-node-minima`,`float32x2`,l),nodeMaxima:e(t,`${i}-node-maxima`,`float32x2`,l),nodeChildren:e(t,`${i}-node-children`,`uint32x2`,l),leafIds:e(t,`${i}-leaf-ids`,`uint32`,c),count:e(t,`${i}-count`,`uint32`,1),overflow:e(t,`${i}-overflow`,`uint32`,1)});return w.push(...te.getCommandNodes(t)),{records:u,bvh:te,nodes:w}}var T=2**-149,E={coordinateSpaces:[`planar`,`longitude-latitude`],metrics:[`none`,`native`]},D=`GPUSegmentRingAssembly`,O=64,ne=`${_}
${y}
fn ringExactTurnClass(origin: vec2f, vertex: vec2f, end: vec2f, side: i32) -> i32 {
  // Positive toward the interior side.
  let toward = side * SIDE_ORDER;
  if (toward < 0) { return 0; }
  if (toward > 0) { return 2; }
  // Collinear: behind the vertex (dot product negative) is the u-turn.
  var behind = false;
  if (vertex.x != origin.x) {
    behind = (vertex.x > origin.x && end.x < vertex.x) || (vertex.x < origin.x && end.x > vertex.x);
  } else if (vertex.y != origin.y) {
    behind = (vertex.y > origin.y && end.y < vertex.y) || (vertex.y < origin.y && end.y > vertex.y);
  }
  return select(1, 2, behind);
}
// Bit patterns of the two factors of term 'term' of
// cross(b - a, d - c) = bx dy - bx cy - ax dy + ax cy - by dx + by cx + ay dx - ay cx.
// The sign of the term is folded into the first factor.
fn ringCrossFactors(term: u32, a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> vec2u {
  let negate = 0x80000000u;
  switch (term) {
    case 0u: { return vec2u(bitcast<u32>(b.x), bitcast<u32>(d.y)); }
    case 1u: { return vec2u(bitcast<u32>(b.x) ^ negate, bitcast<u32>(c.y)); }
    case 2u: { return vec2u(bitcast<u32>(a.x) ^ negate, bitcast<u32>(d.y)); }
    case 3u: { return vec2u(bitcast<u32>(a.x), bitcast<u32>(c.y)); }
    case 4u: { return vec2u(bitcast<u32>(b.y) ^ negate, bitcast<u32>(d.x)); }
    case 5u: { return vec2u(bitcast<u32>(b.y), bitcast<u32>(c.x)); }
    case 6u: { return vec2u(bitcast<u32>(a.y), bitcast<u32>(d.x)); }
    default: { return vec2u(bitcast<u32>(a.y) ^ negate, bitcast<u32>(c.x)); }
  }
}

// One term as an exact 48-bit product times 2^exponent; 'present' is false for a zero term.
struct RingCrossTerm {
  low: u32,
  high: u32,
  exponent: i32,
  negative: bool,
  present: bool,
}

fn ringCrossTerm(term: u32, a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> RingCrossTerm {
  let factors = ringCrossFactors(term, a, b, c, d);
  var result = RingCrossTerm(0u, 0u, 0, false, false);
  let firstMagnitude = factors.x & 0x7fffffffu;
  let secondMagnitude = factors.y & 0x7fffffffu;
  if (firstMagnitude == 0u || secondMagnitude == 0u) { return result; }
  let firstExponentBits = firstMagnitude >> 23u;
  let secondExponentBits = secondMagnitude >> 23u;
  let firstFraction = firstMagnitude & 0x7fffffu;
  let secondFraction = secondMagnitude & 0x7fffffu;
  let firstSignificand = select(0x800000u | firstFraction, firstFraction, firstExponentBits == 0u);
  let secondSignificand = select(0x800000u | secondFraction, secondFraction, secondExponentBits == 0u);
  let firstExponent = select(i32(firstExponentBits) - 150, -149, firstExponentBits == 0u);
  let secondExponent = select(i32(secondExponentBits) - 150, -149, secondExponentBits == 0u);
  let product = multiply24(firstSignificand, secondSignificand);
  result.low = product.x;
  result.high = product.y;
  result.exponent = firstExponent + secondExponent;
  result.negative = ((factors.x ^ factors.y) >> 31u) == 1u;
  result.present = true;
  return result;
}

fn ringCrossExact(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> i32 {
  var minimumExponent = 100000;
  var maximumExponent = -100000;
  for (var term = 0u; term < 8u; term++) {
    let value = ringCrossTerm(term, a, b, c, d);
    if (!value.present) { continue; }
    minimumExponent = min(minimumExponent, value.exponent);
    maximumExponent = max(maximumExponent, value.exponent);
  }
  if (minimumExponent > maximumExponent) { return 0; }
  if (maximumExponent - minimumExponent > ORIENT_MAXIMUM_SPAN) { return 2; }
  var positive: array<u32, 8>;
  var negative: array<u32, 8>;
  for (var term = 0u; term < 8u; term++) {
    let value = ringCrossTerm(term, a, b, c, d);
    if (!value.present) { continue; }
    let shift = u32(value.exponent - minimumExponent);
    if (value.negative) {
      accumulateShifted(&negative, value.low, value.high, shift);
    } else {
      accumulateShifted(&positive, value.low, value.high, shift);
    }
  }
  for (var slot = 8u; slot > 0u; slot--) {
    let plus = positive[slot - 1u];
    let minus = negative[slot - 1u];
    if (plus != minus) { return select(-1, 1, plus > minus); }
  }
  return 0;
}

// Exact sign of cross(b - a, d - c) for four f32 points: -1, 0, 1, or 2 when it cannot be
// certified (non-finite coordinates, or terms that span more than 200 binary orders).
fn ringCrossSign(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> i32 {
  if (!(orientIsFinite(a.x) && orientIsFinite(a.y) && orientIsFinite(b.x) && orientIsFinite(b.y) &&
        orientIsFinite(c.x) && orientIsFinite(c.y) && orientIsFinite(d.x) && orientIsFinite(d.y))) {
    return 2;
  }
  // Each difference is rounded once and each product once, so the float determinant is within
  // 2.5e-7 of the magnitude sum of the exact one; the bound keeps a factor of two. Products near
  // the smallest f32 values lose more than that and go to the exact sum.
  let first = b - a;
  let second = d - c;
  let detLeft = first.x * second.y;
  let detRight = first.y * second.x;
  let det = detLeft - detRight;
  let magnitude = abs(detLeft) + abs(detRight);
  let bound = 5.0e-7 * magnitude;
  if (magnitude > 1.0e-30) {
    if (det > bound) { return 1; }
    if (-det > bound) { return -1; }
  }
  return ringCrossExact(a, b, c, d);
}

// The same classes for the turn from segment origin -> vertex to segment start -> end (neither
// of zero length), by the exact sign of the cross product of their directions.
fn ringSegmentTurnClass(origin: vec2f, vertex: vec2f, start: vec2f, end: vec2f) -> i32 {
  let side = ringCrossSign(origin, vertex, start, end);
  if (side == 2) { return -1; }
  // Positive toward the interior side.
  let toward = side * SIDE_ORDER;
  if (toward < 0) { return 0; }
  if (toward > 0) { return 2; }
  // Parallel: opposite signs of a non-zero component are the u-turn. A component of one
  // direction is zero exactly when it is zero in the other.
  var behind = false;
  if (vertex.x != origin.x) {
    behind = (vertex.x > origin.x) != (end.x > start.x);
  } else {
    behind = (vertex.y > origin.y) != (end.y > start.y);
  }
  return select(1, 2, behind);
}
`,re=4294967295,ie=class{id;props;spatialContext;constructor(e){e=Object.freeze({...x(e,[`id`,`endpoints`,`count`,`groups`,`pairedOpposites`,`vertexTolerance`,`interiorSide`,`normalizeWinding`,`spatialContext`,`cancelOpposingSegments`,`splitTouchingRings`,`output`]),spatialContext:e.spatialContext&&b(e.spatialContext),output:Object.freeze({...x(e.output,[`ringOffsets`,`positions`,`ringAreas`,`ringIsHole`,`ringShells`,`ringGroups`,`segmentRings`,`segmentVertices`,`segmentFlags`,`polygons`,`status`]),status:x(e.output.status,[`count`,`overflow`,`requiredCount`,`openSegmentCount`,`touchingSegmentCount`]),polygons:e.output.polygons&&x(e.output.polygons,[`kind`,`positions`,`ringOffsets`,`polygonOffsets`,`featureOffsets`,`sourceIds`])})}),this.id=e.id??`segment-ring-assembly`,this.spatialContext=b(d(this.id,e.spatialContext,E)),this.props=e;let n=this.id,{output:r}=e;o(e.endpoints,[`float32x4`],`${n} endpoints`);let i=e.endpoints.length;if(i<1)throw Error(`${n} needs at least one segment row`);if(e.vertexTolerance!==void 0&&c(n,`vertexTolerance`,e.vertexTolerance,T),e.interiorSide!==void 0&&![`left`,`right`].includes(e.interiorSide))throw Error(`${n} interiorSide must be 'left' or 'right'`);for(let[i,a]of[[`count`,e.count],[`output.status.count`,r.status.count],[`output.status.overflow`,r.status.overflow],[`output.status.requiredCount`,r.status.requiredCount],[`output.status.openSegmentCount`,r.status.openSegmentCount],[`output.status.touchingSegmentCount`,r.status.touchingSegmentCount]])if(a&&(t(a,`${n} ${i}`),a.length<1))throw Error(`${n} ${i} must contain one uint32 row`);if(e.groups&&(t(e.groups,`${n} groups`),e.groups.length!==i))throw Error(`${n} groups length must equal the segment row count`);if(r.ringGroups&&!e.groups)throw Error(`${n} output.ringGroups needs the groups input`);if(t(r.ringOffsets,`${n} output.ringOffsets`),r.ringOffsets.length<2)throw Error(`${n} output.ringOffsets needs at least two rows (one ring)`);if(o(r.positions,[`float32x2`],`${n} output.positions`),r.positions.length<1)throw Error(`${n} output.positions must not be empty`);let a=r.ringOffsets.length-1;for(let[e,i]of[[`ringIsHole`,r.ringIsHole],[`ringShells`,r.ringShells],[`ringGroups`,r.ringGroups]])if(i&&(t(i,`${n} output.${e}`),i.length!==a))throw Error(`${n} output.${e} length must equal the ring capacity`);if(r.ringAreas&&(o(r.ringAreas,[`float32`],`${n} output.ringAreas`),r.ringAreas.length!==a))throw Error(`${n} output.ringAreas length must equal the ring capacity`);if(r.polygons){let{polygons:i}=r;if(o(i.positions,[`float32x2`],`${n} output.polygons.positions`),i.positions.length!==r.positions.length)throw Error(`${n} output.polygons.positions must match output.positions length`);if(i.sourceIds){if(!e.groups)throw Error(`${n} output.polygons.sourceIds needs the groups input`);if(t(i.sourceIds,`${n} output.polygons.sourceIds`),i.sourceIds.length!==a)throw Error(`${n} output.polygons.sourceIds must have ${a} rows (the ring capacity)`)}for(let[e,r,o]of[[`ringOffsets`,i.ringOffsets,a+1],[`polygonOffsets`,i.polygonOffsets,a+1],[`featureOffsets`,i.featureOffsets,a+1]])if(t(r,`${n} output.polygons.${e}`),r.length!==o)throw Error(`${n} output.polygons.${e} must have ${o} rows`)}for(let[e,a]of[[`segmentRings`,r.segmentRings],[`segmentVertices`,r.segmentVertices],[`segmentFlags`,r.segmentFlags]])if(a&&(t(a,`${n} output.${e}`),a.length!==i))throw Error(`${n} output.${e} length must equal the segment row count`);k(n,r,!!e.pairedOpposites),u(n,ae(r),[e.endpoints,e.count,e.groups])}getCommandNodes(t){let{id:o,props:c}=this,{output:u}=c;f(t,o,{endpoints:c.endpoints,count:c.count,groups:c.groups,...Object.fromEntries(ae(u).map((e,t)=>[`output ${t}`,e]))});let d=c.endpoints.length,p=u.ringOffsets.length-1,ee=u.positions.length,m=(e,n)=>{l(t.device,o,e,n,`reduce the segment or ring capacity`),s(o,`${e} byteLength`,n,0,t.device.limits.maxBufferSize)};if(m(`ring counters scratch`,6*Uint32Array.BYTES_PER_ELEMENT),m(`segment word scratch`,d*Uint32Array.BYTES_PER_ELEMENT),m(`ring word scratch`,p*Uint32Array.BYTES_PER_ELEMENT),u.ringShells||u.polygons){let e=2**Math.ceil(Math.log2(p));m(`shell records`,p*3*Uint32Array.BYTES_PER_ELEMENT),m(`shell bounds`,p*4*Float32Array.BYTES_PER_ELEMENT),m(`shell BVH nodes`,(2*e-1)*2*Float32Array.BYTES_PER_ELEMENT)}let g=c.vertexTolerance??0,b=c.interiorSide??`left`,x=!!c.normalizeWinding&&b===`right`,S=c.normalizeWinding||b===`left`?1:-1,C=this.spatialContext.coordinateSpace===`longitude-latitude`,T=!!c.groups,E=`${re}u`,ie=(n,r,i)=>e(t,`${o}-${n}`,r,i),k=(e,t=d)=>ie(e,`uint32`,t),A=k(`keys`),oe=k(`ids`),j=k(`sorted-keys`),se=k(`sorted-ids`),M=c.pairedOpposites?k(`duplicate-owner`):void 0,N=k(`claim`),P=!!c.cancelOpposingSegments,F=k(`counters`,6),I=k(`next`),ce=k(`alternatives`),le=k(`swapped`),L=u.segmentFlags??k(`flags`),R=k(`leader`),ue=k(`leader-flags`),de=k(`leader-vertices`),z=k(`ring-scan`),B=k(`vertex-scan`),fe=k(`ring-starts`),pe=k(`ring-leaders`),V=k(`ring-count`,1),me=u.segmentRings??k(`segment-rings`),he=u.segmentVertices??k(`segment-vertices`),ge=k(`close-vertices`),H=[k(`pointer-a`),k(`pointer-b`)],U=[k(`value-a`),k(`value-b`)],_e=i(g),ve=c.vertexTolerance===void 0?`fn ringTolerance() -> f32 { return ringFrameExtent() * ${i(2**-21)}; }`:`fn ringTolerance() -> f32 { return ${_e}; }`,ye=w(_e,!1,4),be=w(_e,!0,4),xe=e=>`min(${e}[${e}Offset], ${d}u)`,W=c.count?[{name:`rowCount`,view:c.count,type:`u32`,access:`read`}]:[],G=c.groups?[{name:`groups`,view:c.groups,type:`u32`,access:`read`}]:[],K=c.count?xe(`rowCount`):`${d}u`,q=e=>T?`groups[groupsOffset + ${e}]`:`0u`,J=(e,n,r,i,s=``)=>a(t,{id:`${o}-${e}`,operation:D,variant:e,bindings:n,invocationCount:r,declarations:s,body:i}),Y=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read`}),X=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),Z=[];Z.push(J(`quantize-init`,[...W,X(`counters`,F)],1,`counters[countersOffset] = 0u;
  counters[countersOffset + 1u] = 0u;
  counters[countersOffset + 2u] = 0u;
  counters[countersOffset + 3u] = 0u;
  counters[countersOffset + 4u] = 0u;
  counters[countersOffset + 5u] = ${K};`),J(`quantize-extent`,[Y(`endpoints`,c.endpoints,`f32`),...W,{name:`counters`,view:F,type:`atomic<u32>`,access:`read_write`}],d,`if (index >= ${K}) { return; }
  let origin = vec2f(endpoints[endpointsOffset], endpoints[endpointsOffset + 1u]);
  let start = vec2f(endpoints[endpointsOffset + 4u * index], endpoints[endpointsOffset + 4u * index + 1u]) - origin;
  let end = vec2f(endpoints[endpointsOffset + 4u * index + 2u], endpoints[endpointsOffset + 4u * index + 3u]) - origin;
  if (!isFiniteValue(start.x) || !isFiniteValue(start.y) || !isFiniteValue(end.x) || !isFiniteValue(end.y)) {
    atomicAdd(&counters[countersOffset + 3u], 1u);
    return;
  }
  let extent = max(max(abs(start.x), abs(start.y)), max(abs(end.x), abs(end.y)));
  atomicMax(&counters[countersOffset + 4u], bitcast<u32>(extent));`,h)),Z.push(J(`hash`,[Y(`endpoints`,c.endpoints,`f32`),...W,...G,X(`keys`,A),X(`ids`,oe),X(`claim`,N),Y(`counters`,F)],d,`ids[idsOffset + index] = index;
  claim[claimOffset + index] = 0xffffffffu;
  var key = 0xffffffffu;
  if (index < ${K}) {
    let start = vec2f(endpoints[endpointsOffset + 4u * index], endpoints[endpointsOffset + 4u * index + 1u]);
    let origin = ringOrigin();
    key = ringHash(ringQuantize(start.x, origin.x), ringQuantize(start.y, origin.y), ${q(`index`)});
  }
  keys[keysOffset + index] = key;`,ye)),Z.push(...new r({id:`${o}-sort`,keys:A,values:oe,outputKeys:j,outputValues:se}).getCommandNodes(t)),M&&Z.push(J(`duplicate-owners`,[Y(`endpoints`,c.endpoints,`f32`),Y(`counters`,F),...G,Y(`sortedKeys`,j),Y(`sortedIds`,se),X(`duplicateOwner`,M)],d,`duplicateOwner[duplicateOwnerOffset + index] = index;
  let valid = min(counters[countersOffset + 5u], ${d}u);
  if (index >= valid) { return; }
  let start = ringStart(index);
  let end = ringEnd(index);
  let group = ${q(`index`)};
  let cellOrigin = ringOrigin();
  let cellX = ringQuantize(start.x, cellOrigin.x);
  let cellY = ringQuantize(start.y, cellOrigin.y);
  var seen = array<u32, 9>();
  var seenCount = 0u;
  var owner = index;
  for (var deltaY = -1; deltaY <= 1; deltaY++) {
    for (var deltaX = -1; deltaX <= 1; deltaX++) {
      let hash = ringHash(cellX + bitcast<u32>(deltaX), cellY + bitcast<u32>(deltaY), group);
      var duplicateHash = false;
      for (var seenIndex = 0u; seenIndex < seenCount; seenIndex++) {
        duplicateHash = duplicateHash || seen[seenIndex] == hash;
      }
      if (duplicateHash) { continue; }
      seen[seenCount] = hash;
      seenCount++;
      var low = 0u;
      var high = ${d}u;
      while (low < high) {
        let middle = low + (high - low) / 2u;
        if (sortedKeys[sortedKeysOffset + middle] < hash) { low = middle + 1u; } else { high = middle; }
      }
      var position = low;
      while (position < ${d}u && sortedKeys[sortedKeysOffset + position] == hash) {
        let candidate = sortedIds[sortedIdsOffset + position];
        position++;
        if (candidate >= valid || candidate >= owner) { continue; }
        if (${T?`groups[groupsOffset + candidate] != group`:`false`}) { continue; }
        let candidateStart = ringStart(candidate);
        let candidateEnd = ringEnd(candidate);
        // Exact stored-coordinate equality makes owner identity transitive, so propagation never
        // depends on another duplicate being written in the same dispatch.
        if (any(candidateStart != start) || any(candidateEnd != end)) { continue; }
        owner = candidate;
      }
    }
  }
  duplicateOwner[duplicateOwnerOffset + index] = owner;`,`${ye}
${ve}
fn ringStart(row: u32) -> vec2f {
  return vec2f(endpoints[endpointsOffset + 4u * row], endpoints[endpointsOffset + 4u * row + 1u]);
}
fn ringEnd(row: u32) -> vec2f {
  return vec2f(endpoints[endpointsOffset + 4u * row + 2u], endpoints[endpointsOffset + 4u * row + 3u]);
}`)),P&&Z.push(J(`cancel`,[Y(`endpoints`,c.endpoints,`f32`),...W,...G,Y(`sortedKeys`,j),Y(`sortedIds`,se),X(`flags`,L),{name:`counters`,view:F,type:`atomic<u32>`,access:`read_write`}],d,`flags[flagsOffset + index] = 0u;
  if (index >= ${K}) { return; }
  let partner = findReverse(index);
  if (partner != ${E} && findReverse(partner) == index) {
    flags[flagsOffset + index] = 8u;
    atomicAdd(&counters[countersOffset + 2u], 1u);
  }`,`${be}
${ve}
fn ringStart(row: u32) -> vec2f {
  return vec2f(endpoints[endpointsOffset + 4u * row], endpoints[endpointsOffset + 4u * row + 1u]);
}
fn ringEnd(row: u32) -> vec2f {
  return vec2f(endpoints[endpointsOffset + 4u * row + 2u], endpoints[endpointsOffset + 4u * row + 3u]);
}
// Lowest segment that starts where 'index' ends and ends where it starts.
fn findReverse(index: u32) -> u32 {
  let valid = ${K};
  let origin = ringStart(index);
  let vertex = ringEnd(index);
  let group = ${q(`index`)};
  let cellOrigin = ringOrigin();
  let qx = ringQuantize(vertex.x, cellOrigin.x);
  let qy = ringQuantize(vertex.y, cellOrigin.y);
  var seen = array<u32, 9>();
  var seenCount = 0u;
  var best = ${E};
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let hash = ringHash(qx + bitcast<u32>(dx), qy + bitcast<u32>(dy), group);
      var duplicate = false;
      for (var j = 0u; j < seenCount; j++) { duplicate = duplicate || seen[j] == hash; }
      if (duplicate) { continue; }
      seen[seenCount] = hash;
      seenCount++;
      var low = 0u;
      var high = ${d}u;
      while (low < high) {
        let middle = low + (high - low) / 2u;
        if (sortedKeys[sortedKeysOffset + middle] < hash) { low = middle + 1u; } else { high = middle; }
      }
      var position = low;
      while (position < ${d}u && sortedKeys[sortedKeysOffset + position] == hash) {
        let candidate = sortedIds[sortedIdsOffset + position];
        position++;
        if (candidate >= valid || candidate == index) { continue; }
        let candidateStart = ringStart(candidate);
        let candidateEnd = ringEnd(candidate);
        if (abs(candidateStart.x - vertex.x) > ringTolerance() || abs(candidateStart.y - vertex.y) > ringTolerance()) { continue; }
        if (abs(candidateEnd.x - origin.x) > ringTolerance() || abs(candidateEnd.y - origin.y) > ringTolerance()) { continue; }
        if (${T?`groups[groupsOffset + candidate] != group`:`false`}) { continue; }
        best = min(best, candidate);
      }
    }
  }
  return best;
}`)),M&&Z.push(J(`mark-duplicate-owners`,[Y(`duplicateOwner`,M),X(`flags`,L)],d,`if (duplicateOwner[duplicateOwnerOffset + index] != index ${P?`&& (flags[flagsOffset + index] & 8u) == 0u`:``}) {
    flags[flagsOffset + index] = flags[flagsOffset + index] | 0x80000000u;
  } ${P?``:`else { flags[flagsOffset + index] = 0u; }`}`));let Se=b===`left`?`1.0`:`-1.0`;Z.push(J(`match`,[Y(`endpoints`,c.endpoints,`f32`),Y(`counters`,F),...G,Y(`sortedKeys`,j),Y(`sortedIds`,se),X(`next`,I),{name:`flags`,view:L,type:`atomic<u32>`,access:`read_write`},X(`alt`,ce)],d,`next[nextOffset + index] = ${E};
  alt[altOffset + index] = ${E};
  ${P?`let carried = atomicLoad(&flags[flagsOffset + index]) & 8u;
  let duplicateMarker = atomicLoad(&flags[flagsOffset + index]) & ${M?`0x80000000u`:`0u`};
  atomicStore(&flags[flagsOffset + index], carried | duplicateMarker);
  if (carried != 0u || duplicateMarker != 0u) { return; }`:`if ((atomicLoad(&flags[flagsOffset + index]) & ${M?`0x80000000u`:`0u`}) != 0u) { return; }
  atomicStore(&flags[flagsOffset + index], 0u);`}
  let valid = min(counters[countersOffset + 5u], ${d}u);
  if (index >= valid) { return; }
  let origin = ringStart(index);
  let vertex = ringEnd(index);
  // A zero-length segment has no direction, so no turn can be ordered against it: it takes no
  // continuation here and is no candidate below. Left in, it reads as straight on, and with a
  // lower index than the real segment into its vertex it takes that segment's continuation.
  if (all(origin == vertex)) {
    atomicStore(&flags[flagsOffset + index], 2u);
    return;
  }
  let group = ${q(`index`)};
  let incoming = ringScaled(vertex - origin, vertex.y);
  let cellOrigin = ringOrigin();
  let qx = ringQuantize(vertex.x, cellOrigin.x);
  let qy = ringQuantize(vertex.y, cellOrigin.y);
  var seen = array<u32, 9>();
  var seenCount = 0u;
  var best = ${E};
  var bestTurn = 0.0;
  var matches = 0u;
  var first = ${E};
  var second = ${E};
  // Exact choice, used when every candidate starts exactly at this segment's end vertex.
  var exactUsable = true;
  var exactBest = ${E};
  var exactBestClass = 0;
  var exactBestEnd = vec2f(0.0);
  // The same choice on the exact directions of the segments, used when a candidate starts
  // within the tolerance of this segment's end vertex but not on it.
  var directionUsable = true;
  var directionBest = ${E};
  var directionBestClass = 0;
  var directionBestStart = vec2f(0.0);
  var directionBestEnd = vec2f(0.0);
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let hash = ringHash(qx + bitcast<u32>(dx), qy + bitcast<u32>(dy), group);
      var duplicate = false;
      for (var j = 0u; j < seenCount; j++) { duplicate = duplicate || seen[j] == hash; }
      if (duplicate) { continue; }
      seen[seenCount] = hash;
      seenCount++;
      var low = 0u;
      var high = ${d}u;
      while (low < high) {
        let middle = low + (high - low) / 2u;
        if (sortedKeys[sortedKeysOffset + middle] < hash) { low = middle + 1u; } else { high = middle; }
      }
      var position = low;
      while (position < ${d}u && sortedKeys[sortedKeysOffset + position] == hash) {
        let candidate = sortedIds[sortedIdsOffset + position];
        position++;
        if (candidate >= valid || candidate == index) { continue; }
        ${M?`if ((atomicLoad(&flags[flagsOffset + candidate]) & 0x80000000u) != 0u) { continue; }`:``}
        ${P?`if ((atomicLoad(&flags[flagsOffset + candidate]) & 8u) != 0u) { continue; }`:``}
        let candidateStart = ringStart(candidate);
        if (abs(candidateStart.x - vertex.x) > ringTolerance() || abs(candidateStart.y - vertex.y) > ringTolerance()) { continue; }
        if (all(ringEnd(candidate) == candidateStart)) { continue; }
        ${c.pairedOpposites?`let candidateEnd = ringEnd(candidate);
        if (abs(candidateEnd.x - origin.x) <= ringTolerance() && abs(candidateEnd.y - origin.y) <= ringTolerance()) { continue; }`:``}
        if (${T?`groups[groupsOffset + candidate] != group`:`false`}) { continue; }
        matches++;
        if (first == ${E}) { first = candidate; } else if (second == ${E}) { second = candidate; }
        let outgoing = ringScaled(ringEnd(candidate) - candidateStart, vertex.y);
        let turn = ringTurnKey(SIDE * (incoming.x * outgoing.y - incoming.y * outgoing.x), dot(incoming, outgoing));
        if (best == ${E} || turn > bestTurn || (turn == bestTurn && candidate < best)) {
          best = candidate;
          bestTurn = turn;
        }
        let turnEnd = ringEnd(candidate);
        if (exactUsable) {
          // Both exact decisions share one call chain. The second sign is evaluated only when
          // the classes tie, preserving the original uncertainty and fallback choices.
          var turnClass = -1;
          var candidateSide = 0;
          var orientationStep = 0u;
          loop {
            var signStart = origin;
            var signEnd = vertex;
            var signPoint = turnEnd;
            if (orientationStep == 1u) {
              signStart = vertex;
              signEnd = turnEnd;
              signPoint = exactBestEnd;
            }
            let exact = exactOrientSigns(
              1u,
              array<vec2f, 4>(signStart, vec2f(0.0), vec2f(0.0), vec2f(0.0)),
              array<vec2f, 4>(signEnd, vec2f(0.0), vec2f(0.0), vec2f(0.0)),
              array<vec2f, 4>(signPoint, vec2f(0.0), vec2f(0.0), vec2f(0.0))
            );
            if (exact.uncertain) { exactUsable = false; break; }
            if (orientationStep == 1u) { candidateSide = exact.signs.x; break; }
            turnClass = ringExactTurnClass(origin, vertex, turnEnd, exact.signs.x);
            if (any(candidateStart != vertex) || exactBest == ${E} || turnClass != exactBestClass || turnClass == 1) { break; }
            orientationStep = 1u;
          }
          if (any(candidateStart != vertex) || !exactUsable) {
            exactUsable = false;
          } else if (exactBest == ${E}) {
            exactBest = candidate;
            exactBestClass = turnClass;
            exactBestEnd = turnEnd;
          } else {
            var order = turnClass - exactBestClass;
            if (order == 0 && turnClass != 1) {
              // Same half-plane: the direction further round toward the interior side turns
              // further (counter-clockwise for the interior on the left).
              order = -candidateSide * SIDE_ORDER;
            }
            if (order > 0 || (order == 0 && candidate < exactBest)) {
              exactBest = candidate;
              exactBestClass = turnClass;
              exactBestEnd = turnEnd;
            }
          }
        }
        if (directionUsable) {
          let turnClass = ringSegmentTurnClass(origin, vertex, candidateStart, turnEnd);
          if (turnClass < 0) {
            directionUsable = false;
          } else if (directionBest == ${E}) {
            directionBest = candidate;
            directionBestClass = turnClass;
            directionBestStart = candidateStart;
            directionBestEnd = turnEnd;
          } else {
            var order = turnClass - directionBestClass;
            if (order == 0 && turnClass != 1) {
              // Same half-plane: the direction further round toward the interior side turns
              // further (counter-clockwise for the interior on the left).
              let side = ringCrossSign(candidateStart, turnEnd, directionBestStart, directionBestEnd);
              if (side == 2) { directionUsable = false; }
              order = -side * SIDE_ORDER;
            }
            if (order > 0 || (order == 0 && candidate < directionBest)) {
              directionBest = candidate;
              directionBestClass = turnClass;
              directionBestStart = candidateStart;
              directionBestEnd = turnEnd;
            }
          }
        }
      }
    }
  }
  // The f32 pseudo-angle decides only where an exact sign cannot be certified (non-finite
  // coordinates). The exact orders hold for longitude/latitude input too: the cos(latitude)
  // scale of the longitude differences is positive and changes no orientation sign.
  if (exactUsable && exactBest != ${E}) {
    best = exactBest;
  } else if (directionUsable && directionBest != ${E}) {
    best = directionBest;
  }
  next[nextOffset + index] = best;
  if (matches == 2u) { alt[altOffset + index] = select(first, second, best == first); }
  var flagBits = 0u;
  if (matches > 1u) { flagBits = 1u; }
  if (matches == 0u) { flagBits = 2u; }
  atomicStore(&flags[flagsOffset + index], flagBits);`,`${ye}
${ve}
const SIDE: f32 = ${Se};
const SIDE_ORDER: i32 = ${b===`left`?1:-1};
// Monotone in the turn angle from the incoming to the outgoing direction, positive toward the
// interior side: 'cross' is the cross product times SIDE, so the key of a scene with the interior
// on the right is the key of its mirror image with the interior on the left, and the u-turn is
// at +2 for both. atan2 is avoided: it misbehaves on signed zeros under fast math.
fn ringTurnKey(cross: f32, dot: f32) -> f32 {
  let denominator = abs(cross) + abs(dot);
  if (denominator == 0.0) { return 0.0; }
  let ratio = cross / denominator;
  if (dot >= 0.0) { return ratio; }
  return select(-2.0, 2.0, cross >= 0.0) - ratio;
}
fn ringStart(row: u32) -> vec2f {
  return vec2f(endpoints[endpointsOffset + 4u * row], endpoints[endpointsOffset + 4u * row + 1u]);
}
fn ringEnd(row: u32) -> vec2f {
  return vec2f(endpoints[endpointsOffset + 4u * row + 2u], endpoints[endpointsOffset + 4u * row + 3u]);
}
fn ringScaled(delta: vec2f, latitude: f32) -> vec2f {
  ${C?`return vec2f(delta.x * max(cos(radians(latitude)), 0.01), delta.y);`:`return delta;`}
}
${ne}`)),Z.push(a(t,{id:`${o}-claim`,operation:D,variant:`claim`,bindings:[Y(`next`,I),{name:`claim`,view:N,type:`atomic<u32>`,access:`read_write`}],invocationCount:d,body:`let claimed = next[nextOffset + index];
  if (claimed != ${E}) { atomicMin(&claim[claimOffset + claimed], index); }`}),a(t,{id:`${o}-resolve`,operation:D,variant:`resolve`,bindings:[X(`next`,I),Y(`claim`,N),X(`flags`,L),{name:`counters`,view:F,type:`atomic<u32>`,access:`read_write`}],invocationCount:d,body:`let claimed = next[nextOffset + index];
  if (claimed != ${E} && claim[claimOffset + claimed] != index) {
    next[nextOffset + index] = ${E};
    flags[flagsOffset + index] = flags[flagsOffset + index] | 4u;
  }
  if ((flags[flagsOffset + index] & 1u) != 0u) {
    atomicAdd(&counters[countersOffset + 1u], 1u);
  }`}));let Ce=Math.max(1,Math.ceil(Math.log2(d))),we=(e,n,r)=>{Z.push(J(`jump-init-${e}`,[Y(`next`,n),X(`pointer`,H[0]),X(`value`,U[0])],d,`pointer[pointerOffset + index] = next[nextOffset + index];
  value[valueOffset + index] = index;`));let i=0;for(let t=0;t<Ce;t++)Z.push(je(`min`,`${e}-${t}`,i)),i=1-i;let s=1-i;Z.push(a(t,{id:`${o}-cut-${e}`,operation:D,variant:`cut`,bindings:[Y(`next`,n),Y(`pointerIn`,H[i]),Y(`valueIn`,U[i]),...W,X(`leader`,R),X(`pointerOut`,H[s]),X(`valueOut`,U[s]),{name:`counters`,view:F,type:`atomic<u32>`,access:`read_write`}],invocationCount:d,body:`let valid = ${K};
  var leaderIndex = ${E};
  var tail = ${E};
  var hopCount = 0u;
  if (index < valid) {
    if (pointerIn[pointerInOffset + index] != ${E}) {
      leaderIndex = valueIn[valueInOffset + index];
      tail = next[nextOffset + index];
      if (tail == leaderIndex) { tail = ${E}; }
      if (tail != ${E}) { hopCount = 1u; }
    } else {
      ${r?`atomicAdd(&counters[countersOffset], 1u);`:``}
    }
  }
  leader[leaderOffset + index] = leaderIndex;
  pointerOut[pointerOutOffset + index] = tail;
  valueOut[valueOutOffset + index] = hopCount;`})),i=s;for(let t=0;t<Ce;t++)Z.push(je(`sum`,`${e}-${t}`,i)),i=1-i;return U[i]},Te=we(1,I,c.splitTouchingRings===!1);c.splitTouchingRings!==!1&&(Z.push(J(`split-touching`,[Y(`next`,I),Y(`alt`,ce),Y(`claim`,N),Y(`leader`,R),X(`swapped`,le)],d,`let chosen = next[nextOffset + index];
  var result = chosen;
  let other = alt[altOffset + index];
  if (chosen != ${E} && other != ${E}) {
    let partner = claim[claimOffset + other];
    if (partner != ${E} && partner != index &&
        alt[altOffset + partner] == chosen && next[nextOffset + partner] == other &&
        leader[leaderOffset + index] != ${E} &&
        leader[leaderOffset + index] == leader[leaderOffset + partner]) {
      result = other;
    }
  }
  swapped[swappedOffset + index] = result;`)),Te=we(2,le,!0)),Z.push(J(`leaders`,[Y(`leader`,R),Y(`hops`,Te),X(`leaderFlags`,ue),X(`leaderVertices`,de)],d,`let isLeader = leader[leaderOffset + index] == index;
  leaderFlags[leaderFlagsOffset + index] = select(0u, 1u, isLeader);
  leaderVertices[leaderVerticesOffset + index] = select(0u, hops[hopsOffset + index] + 2u, isLeader);`),...new n({id:`${o}-ring-scan`,input:ue,output:z,mode:`inclusive`}).getCommandNodes(t),...new n({id:`${o}-vertex-scan`,input:de,output:B,mode:`inclusive`}).getCommandNodes(t),J(`ring-starts`,[Y(`leader`,R),Y(`hops`,Te),Y(`ringScan`,z),Y(`vertexScan`,B),X(`ringStarts`,fe),X(`ringLeaders`,pe)],d,`if (leader[leaderOffset + index] != index) { return; }
  let ring = ringScan[ringScanOffset + index] - 1u;
  ringStarts[ringStartsOffset + ring] = vertexScan[vertexScanOffset + index] - (hops[hopsOffset + index] + 2u);
  ringLeaders[ringLeadersOffset + ring] = index;`),J(`ring-count`,[Y(`ringScan`,z),Y(`vertexScan`,B),Y(`ringStarts`,fe),X(`ringCount`,V)],1,`let total = ringScan[ringScanOffset + ${d-1}u];
  let vertices = vertexScan[vertexScanOffset + ${d-1}u];
  var low = 0u;
  var high = min(total, ${p}u);
  while (low < high) {
    let middle = low + (high - low + 1u) / 2u;
    var end = vertices;
    if (middle < total) { end = ringStarts[ringStartsOffset + middle]; }
    if (end <= ${ee}u) { low = middle; } else { high = middle - 1u; }
  }
  ringCount[ringCountOffset] = low;`)),Z.push(J(`place`,[Y(`leader`,R),Y(`hops`,Te),Y(`ringScan`,z),Y(`vertexScan`,B),Y(`ringCount`,V),X(`segmentRings`,me),X(`segmentVertices`,he),X(`closeVertices`,ge)],d,`segmentRings[segmentRingsOffset + index] = ${E};
  segmentVertices[segmentVerticesOffset + index] = ${E};
  closeVertices[closeVerticesOffset + index] = ${E};
  let head = leader[leaderOffset + index];
  if (head == ${E}) { return; }
  let ring = ringScan[ringScanOffset + head] - 1u;
  if (ring >= ringCount[ringCountOffset]) { return; }
  let ringLength = hops[hopsOffset + head] + 1u;
  let start = vertexScan[vertexScanOffset + head] - (ringLength + 1u);
  let position = hops[hopsOffset + head] - hops[hopsOffset + index];
  ${x?`let slot = (ringLength - position) % ringLength;`:`let slot = position;`}
  segmentRings[segmentRingsOffset + index] = ring;
  segmentVertices[segmentVerticesOffset + index] = start + slot;
  if (position == 0u) { closeVertices[closeVerticesOffset + index] = start + ringLength; }`),...M?[J(`propagate-duplicate-owners`,[Y(`duplicateOwner`,M),Y(`leader`,R),X(`segmentRings`,me),X(`segmentVertices`,he),X(`flags`,L),{name:`counters`,view:F,type:`atomic<u32>`,access:`read_write`}],d,`let owner = duplicateOwner[duplicateOwnerOffset + index];
  let originalFlags = flags[flagsOffset + index];
  // Cancelled rows already contribute to the cancellation counter; leave their flags and open
  // summary untouched so the final counter subtraction accounts for them exactly once.
  if ((originalFlags & 8u) != 0u) { return; }
  if (owner != index) {
    segmentRings[segmentRingsOffset + index] = segmentRings[segmentRingsOffset + owner];
    segmentVertices[segmentVerticesOffset + index] = segmentVertices[segmentVerticesOffset + owner];
    flags[flagsOffset + index] = flags[flagsOffset + owner];
    let ownerFlags = flags[flagsOffset + owner];
    // Trace counted each duplicate with no successor as open; the owner leader still identifies a
    // closed cycle even when ring or vertex capacity prevented writing its output slots.
    if (leader[leaderOffset + owner] != ${E} ||
        (ownerFlags & 8u) != 0u) {
      atomicSub(&counters[countersOffset], 1u);
    }
    if ((flags[flagsOffset + owner] & 1u) != 0u) {
      atomicAdd(&counters[countersOffset + 1u], 1u);
    }
  }`)]:[],J(`write`,[Y(`endpoints`,c.endpoints,`f32`),...M?[Y(`duplicateOwner`,M)]:[],Y(`segmentVertices`,he),Y(`closeVertices`,ge),X(`positions`,u.positions,`f32`)],d,`let slot = segmentVertices[segmentVerticesOffset + index];
  ${M?`if (duplicateOwner[duplicateOwnerOffset + index] != index) { return; }`:``}
  if (slot == ${E}) { return; }
  let x = endpoints[endpointsOffset + 4u * index];
  let y = endpoints[endpointsOffset + 4u * index + 1u];
  positions[positionsOffset + 2u * slot] = x;
  positions[positionsOffset + 2u * slot + 1u] = y;
  let close = closeVertices[closeVerticesOffset + index];
  if (close != ${E}) {
    positions[positionsOffset + 2u * close] = x;
    positions[positionsOffset + 2u * close + 1u] = y;
  }`),J(`offsets`,[Y(`ringStarts`,fe),Y(`ringScan`,z),Y(`vertexScan`,B),Y(`ringCount`,V),X(`ringOffsets`,u.ringOffsets)],p+1,`let total = ringScan[ringScanOffset + ${d-1}u];
  let written = min(index, ringCount[ringCountOffset]);
  var offset = vertexScan[vertexScanOffset + ${d-1}u];
  if (written < total) { offset = ringStarts[ringStartsOffset + written]; }
  ringOffsets[ringOffsetsOffset + index] = offset;`));let Ee=u.ringAreas??ie(`ring-areas`,`float32`,p),De=u.ringIsHole??k(`ring-is-hole`,p),Oe=!!(u.ringShells??u.polygons),ke=k(`ring-lowest-vertices`,p),Q=Oe?e(t,`${o}-ring-bounds`,`float32`,p*4):void 0;Z.push(a(t,{id:`${o}-ring-stats`,operation:D,variant:`ring-stats`,bindings:[Y(`ringOffsets`,u.ringOffsets),Y(`positions`,u.positions,`f32`),Y(`ringCount`,V),X(`areas`,Ee,`f32`),X(`lowestVertices`,ke),...Q?[X(`bounds`,Q,`f32`)]:[]],invocationCount:Math.ceil(p/O)*O,workgroupSize:O,guardIndex:!1,declarations:`const RING_CAPACITY: u32 = ${p}u;
const RING_LANES: u32 = ${O}u;
const BOUNDS_LARGE: f32 = 3.0e38;
var<workgroup> largeBits: array<atomic<u32>, 2>;
var<workgroup> largeBitsCopy: array<u32, 2>;
var<workgroup> largeFirst: array<u32, ${O}>;
var<workgroup> largeLast: array<u32, ${O}>;
var<workgroup> partialSum: array<f32, ${O}>;
var<workgroup> partialMinimum: array<vec2f, ${O}>;
var<workgroup> partialMaximum: array<vec2f, ${O}>;
var<workgroup> partialLowest: array<u32, ${O}>;
fn getVertex(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * vertex], positions[positionsOffset + 2u * vertex + 1u]);
}
// Lexicographic (y, x, row) order of vertex rows: the lowest vertex is an extreme point of the ring.
fn isLowerVertex(vertex: u32, other: u32) -> bool {
  let p = getVertex(vertex);
  let q = getVertex(other);
  return p.y < q.y || (p.y == q.y && (p.x < q.x || (p.x == q.x && vertex < other)));
}
fn publishRing(ring: u32, sum: f32, minimum: vec2f, maximum: vec2f, lowest: u32) {
  let area = 0.5 * sum;
  areas[areasOffset + ring] = area;
  lowestVertices[lowestVerticesOffset + ring] = lowest;
  ${Q?`bounds[boundsOffset + 4u * ring] = minimum.x; bounds[boundsOffset + 4u * ring + 1u] = minimum.y; bounds[boundsOffset + 4u * ring + 2u] = maximum.x; bounds[boundsOffset + 4u * ring + 3u] = maximum.y;`:``}
}`,body:`let lane = localInvocationIndex;
  let ringBase = workgroupIndex * RING_LANES;
  let ownRing = ringBase + lane;
  // Phase 1: this lane's own ring, serially when small; larger rings are queued for phase 2.
  if (ownRing < RING_CAPACITY) {
    if (ownRing >= ringCount[ringCountOffset]) {
      publishRing(ownRing, 0.0, vec2f(0.0), vec2f(0.0), 0u);
    } else {
      let first = ringOffsets[ringOffsetsOffset + ownRing];
      let last = ringOffsets[ringOffsetsOffset + ownRing + 1u];
      if (last - first <= RING_LANES) {
        let origin = getVertex(first);
        var sum = 0.0;
        var minimum = origin;
        var maximum = origin;
        var lowest = first;
        for (var vertex = first; vertex + 1u < last; vertex++) {
          let p = getVertex(vertex);
          let q = getVertex(vertex + 1u);
          minimum = min(minimum, q);
          maximum = max(maximum, q);
          if (isLowerVertex(vertex + 1u, lowest)) { lowest = vertex + 1u; }
          let a = p - origin;
          let b = q - origin;
          sum += a.x * b.y - b.x * a.y;
        }
        publishRing(ownRing, sum, minimum, maximum, lowest);
      } else {
        largeFirst[lane] = first;
        largeLast[lane] = last;
        atomicOr(&largeBits[lane / 32u], 1u << (lane % 32u));
      }
    }
  }
  workgroupBarrier();
  if (lane == 0u) {
    largeBitsCopy[0] = atomicLoad(&largeBits[0]);
    largeBitsCopy[1] = atomicLoad(&largeBits[1]);
  }
  // Phase 2: every barrier below is in workgroup-uniform control flow (the queue is read through
  // workgroupUniformLoad), and the queued rings are visited in ascending order.
  for (var word = 0u; word < 2u; word++) {
    var pending = workgroupUniformLoad(&largeBitsCopy[word]);
    while (pending != 0u) {
      let slot = word * 32u + firstTrailingBit(pending);
      pending = pending & (pending - 1u);
      let first = workgroupUniformLoad(&largeFirst[slot]);
      let last = workgroupUniformLoad(&largeLast[slot]);
      let origin = getVertex(first);
      var sum = 0.0;
      var minimum = vec2f(BOUNDS_LARGE);
      var maximum = vec2f(-BOUNDS_LARGE);
      var lowest = first;
      for (var vertex = first + lane; vertex + 1u < last; vertex += RING_LANES) {
        let p = getVertex(vertex);
        let q = getVertex(vertex + 1u);
        minimum = min(minimum, q);
        maximum = max(maximum, q);
        if (isLowerVertex(vertex + 1u, lowest)) { lowest = vertex + 1u; }
        let a = p - origin;
        let b = q - origin;
        sum += a.x * b.y - b.x * a.y;
      }
      partialSum[lane] = sum;
      partialMinimum[lane] = minimum;
      partialMaximum[lane] = maximum;
      partialLowest[lane] = lowest;
      workgroupBarrier();
      for (var stride = RING_LANES / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
          partialSum[lane] += partialSum[lane + stride];
          partialMinimum[lane] = min(partialMinimum[lane], partialMinimum[lane + stride]);
          partialMaximum[lane] = max(partialMaximum[lane], partialMaximum[lane + stride]);
          if (isLowerVertex(partialLowest[lane + stride], partialLowest[lane])) {
            partialLowest[lane] = partialLowest[lane + stride];
          }
        }
        workgroupBarrier();
      }
      if (lane == 0u) {
        publishRing(ringBase + slot, partialSum[0], min(partialMinimum[0], origin), max(partialMaximum[0], origin), partialLowest[0]);
      }
    }
  }`})),Z.push(J(`ring-orientation`,[Y(`ringOffsets`,u.ringOffsets),Y(`positions`,u.positions,`f32`),Y(`ringCount`,V),X(`areas`,Ee,`f32`),Y(`lowestVertices`,ke),X(`isHole`,De)],p,`isHole[isHoleOffset + index] = 0u;
  if (index >= ringCount[ringCountOffset]) { return; }
  let first = ringOffsets[ringOffsetsOffset + index];
  let last = ringOffsets[ringOffsetsOffset + index + 1u];
  let area = areas[areasOffset + index];
  var orientation = i32(sign(area));
  // Closed ring: rows [first, last - 1) are its distinct vertex slots, row last - 1 repeats first.
  if (last >= first + 4u) {
    let slots = last - first - 1u;
    var lowest = lowestVertices[lowestVerticesOffset + index];
    if (lowest == last - 1u) { lowest = first; }
    let vertex = getRingVertex(lowest);
    var previous = lowest;
    var next = lowest;
    for (var step = 1u; step < slots; step++) {
      previous = select(previous - 1u, last - 2u, previous == first);
      if (any(getRingVertex(previous) != vertex)) { break; }
    }
    for (var step = 1u; step < slots; step++) {
      next = select(next + 1u, first, next == last - 2u);
      if (any(getRingVertex(next) != vertex)) { break; }
    }
    let turn = exactOrientSigns(
      1u,
      array<vec2f, 4>(getRingVertex(previous), vec2f(0.0), vec2f(0.0), vec2f(0.0)),
      array<vec2f, 4>(vertex, vec2f(0.0), vec2f(0.0), vec2f(0.0)),
      array<vec2f, 4>(getRingVertex(next), vec2f(0.0), vec2f(0.0), vec2f(0.0))
    ).signs.x;
    if (turn != 0) { orientation = turn; }
    // A ring that turns at a lowest vertex it passes once encloses area, whatever the f32 sum
    // says: a face thinner than one f32 step can sum to exactly 0, and a ring of area 0 is
    // read downstream as an empty walk and left out of the polygons. Such a ring gets the smallest
    // normal float with the sign of its exact turn. (A walk out and back along one path has no
    // turn at a tip, and passes any other lowest vertex twice: it stays 0.)
    if (area == 0.0 && turn != 0) {
      var passes = 0u;
      var before = getRingVertex(last - 2u);
      for (var row = first; row + 1u < last; row++) {
        let current = getRingVertex(row);
        if (all(current == vertex) && any(before != vertex)) { passes++; }
        before = current;
      }
      if (passes == 1u) { areas[areasOffset + index] = f32(turn) * 1.17549435e-38; }
    }
  }
  isHole[isHoleOffset + index] = select(0u, 1u, orientation * ${S} < 0);`,`${_}
${y}
fn getRingVertex(row: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}`));let $=u.ringGroups??(T?k(`ring-groups`,p):void 0);$&&c.groups&&Z.push(J(`ring-groups`,[Y(`ringLeaders`,pe),Y(`ringCount`,V),...G,X(`ringGroups`,$)],p,`ringGroups[ringGroupsOffset + index] = 0u;
  if (index < ringCount[ringCountOffset]) {
    ringGroups[ringGroupsOffset + index] = groups[groupsOffset + ringLeaders[ringLeadersOffset + index]];
  }`));let Ae=u.ringShells??(u.polygons?k(`ring-shells`,p):void 0);if(Ae){let e=te(t,{id:`${o}-shell-index`,operation:D,bounds:Q,areas:Ee,isHole:De,count:V,groups:$}),{bvh:n}=e;Z.push(...e.nodes,J(`ring-shells`,[Y(`ringOffsets`,u.ringOffsets),Y(`positions`,u.positions,`f32`),Y(`ringBounds`,Q,`f32`),Y(`records`,e.records),Y(`nodeMinima`,n.nodeMinima,`f32`),Y(`nodeMaxima`,n.nodeMaxima,`f32`),Y(`leafIds`,n.leafIds),X(`shells`,Ae)],p,`shells[shellsOffset + index] = ${E};
  // The records pass marks rows beyond the live count, freeing the count binding for bounds.
  let kind = records[recordsOffset + RECORD_WORDS * index + 1u];
  if (kind == 2u) { return; }
  if (kind == 0u) {
    shells[shellsOffset + index] = index;
    return;
  }
  let first = ringOffsets[ringOffsetsOffset + index];
  let a = getRingVertex(first);
  let b = getRingVertex(first + 1u);
  // The probe is the symbolic point infinitesimally beside the midpoint of the hole's first edge,
  // on the side away from the hole's interior (left of a clockwise hole), so it lies in the face
  // the hole bounds even when that edge runs along another ring. Every decision below is exact.
  // Holes wind against the shells (exact orientation), so a hole is clockwise exactly when shells
  // are counter-clockwise.
  let probe = ExactProbe(0.5 * (a + b), a, b, ${S>0?1:-1});
  var best = ${E};
  var bestArea = 0.0;
  var bestBounds = vec4f(0.0);
  let holeBounds = getRingBounds(index);
  var node = 0u;
  loop {
    let hit = findShell(node, holeBounds);
    let shell = hit.x;
    if (shell == ${E}) { break; }
    node = hit.y;
    if (shell >= RING_CAPACITY || records[recordsOffset + RECORD_WORDS * shell + 1u] != 0u) { continue; }
    if (records[recordsOffset + RECORD_WORDS * shell + 2u] != records[recordsOffset + RECORD_WORDS * index + 2u]) { continue; }
    // A shell around the hole is around the hole's bounding box. The boxes are exact (minima and
    // maxima of stored vertices), where the f32 areas of a thin shell and its hole can come out
    // equal or in the wrong order. A bidirected boundary produces a reverse walk of the same box
    // for the unbounded side: the probe, which lies outside that walk, keeps it from owning itself.
    let shellBounds = getRingBounds(shell);
    if (!boundsContain(shellBounds, holeBounds)) { continue; }
    let area = abs(bitcast<f32>(records[recordsOffset + RECORD_WORDS * shell]));
    if (best != ${E}) {
      // Two shells around one point of a planar boundary are nested, and the inner one has its box
      // inside the other's. Only shells whose boxes are equal or do not nest are ordered by their
      // f32 areas (smaller first, lowest index on ties).
      let insideBest = boundsContain(bestBounds, shellBounds);
      let aroundBest = boundsContain(shellBounds, bestBounds);
      if (aroundBest && !insideBest) { continue; }
      if (insideBest == aroundBest && (area > bestArea || (area == bestArea && shell > best))) { continue; }
    }
    // A probe outside the shell's bounding box (or left of it, where a closed ring crosses the ray
    // an even number of times) cannot be inside: skip the vertex walk.
    let point = probe.point;
    if (point.y < shellBounds.y || point.y > shellBounds.w || point.x < shellBounds.x || point.x > shellBounds.z) { continue; }
    let shellFirst = ringOffsets[ringOffsetsOffset + shell];
    let shellLast = ringOffsets[ringOffsetsOffset + shell + 1u];
    let inside = (getShellWinding(probe, shellFirst, shellLast).x & 1) != 0;
    if (inside) {
      best = shell;
      bestArea = area;
      bestBounds = shellBounds;
    }
  }
  shells[shellsOffset + index] = best;`,`const RECORD_WORDS: u32 = 3u;
const RING_CAPACITY: u32 = ${p}u;
const INTERNAL_NODE_COUNT: u32 = ${n.internalNodeCount}u;
${_}
${y}
fn getRingVertex(row: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
// (minimum x, minimum y, maximum x, maximum y) of a ring's vertices.
fn getRingBounds(ring: u32) -> vec4f {
  return vec4f(
    ringBounds[ringBoundsOffset + 4u * ring], ringBounds[ringBoundsOffset + 4u * ring + 1u],
    ringBounds[ringBoundsOffset + 4u * ring + 2u], ringBounds[ringBoundsOffset + 4u * ring + 3u]);
}
// Whether box 'outer' contains box 'inner' (edges may coincide).
fn boundsContain(outer: vec4f, inner: vec4f) -> bool {
  return outer.x <= inner.x && outer.y <= inner.y && outer.z >= inner.z && outer.w >= inner.w;
}
// Stackless traversal: return a leaf and the cursor immediately after it. Rejected candidates
// can continue the caller's loop without revisiting a node, including the final leaf.
fn findShell(startNode: u32, holeBounds: vec4f) -> vec2u {
  var node = startNode;
  while (node != ${E}) {
    let minimum = vec2f(nodeMinima[nodeMinimaOffset + 2u * node], nodeMinima[nodeMinimaOffset + 2u * node + 1u]);
    let maximum = vec2f(nodeMaxima[nodeMaximaOffset + 2u * node], nodeMaxima[nodeMaximaOffset + 2u * node + 1u]);
    let contains = boundsContain(vec4f(minimum, maximum), holeBounds);
    if (contains && node < INTERNAL_NODE_COUNT) {
      node = 2u * node + 1u;
      continue;
    }
    var shell = ${E};
    if (contains) { shell = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT]; }
    loop {
      if (node == 0u || (node & 1u) == 1u) { break; }
      node = (node - 1u) / 2u;
    }
    node = select(node + 1u, ${E}, node == 0u);
    if (shell != ${E}) { return vec2u(shell, node); }
  }
  return vec2u(${E});
}
${v(`getShellWinding`,`getRingVertex`)}`))}if(u.polygons&&Ae){let{polygons:e}=u,i=k(`polygon-keys`,p),a=k(`polygon-ids`,p),s=k(`sorted-polygon-keys`,p),c=k(`sorted-polygon-ids`,p),l=k(`polygon-lengths`,p),d=k(`polygon-shell-flags`,p),f=k(`polygon-length-scan`,p),m=k(`polygon-shell-scan`,p),h=k(`polygon-starts`,p),g=k(`polygon-valid-rings`,1);Z.push(J(`polygon-keys`,[Y(`ringCount`,V),Y(`areas`,Ee,`f32`),Y(`isHole`,De),Y(`shells`,Ae),X(`keys`,i),X(`ids`,a)],p,`ids[idsOffset + index] = index;
  var key = ${E};
  if (index < ringCount[ringCountOffset] && abs(areas[areasOffset + index]) > 0.0 && shells[shellsOffset + index] != ${E}) {
    key = shells[shellsOffset + index] * 2u + isHole[isHoleOffset + index];
  }
  keys[keysOffset + index] = key;`),...new r({id:`${o}-polygon-sort`,keys:i,values:a,outputKeys:s,outputValues:c}).getCommandNodes(t),J(`polygon-lengths`,[Y(`ringOffsets`,u.ringOffsets),Y(`sortedKeys`,s),Y(`sortedIds`,c),X(`lengths`,l),X(`shellFlags`,d)],p,`let key = sortedKeys[sortedKeysOffset + index];
  let ring = sortedIds[sortedIdsOffset + index];
  let isValid = key != ${E};
  lengths[lengthsOffset + index] = select(0u, ringOffsets[ringOffsetsOffset + ring + 1u] - ringOffsets[ringOffsetsOffset + ring] - 1u, isValid);
  shellFlags[shellFlagsOffset + index] = select(0u, 1u, isValid && (key & 1u) == 0u);`),...new n({id:`${o}-polygon-length-scan`,input:l,output:f,mode:`inclusive`}).getCommandNodes(t),...new n({id:`${o}-polygon-shell-scan`,input:d,output:m,mode:`inclusive`}).getCommandNodes(t),J(`polygon-starts`,[Y(`shellFlags`,d),Y(`shellScan`,m),X(`polygonStarts`,h)],p,`if (shellFlags[shellFlagsOffset + index] != 0u) {
    polygonStarts[polygonStartsOffset + shellScan[shellScanOffset + index] - 1u] = index;
  }`),...e.sourceIds&&$?[J(`polygon-groups`,[Y(`shellScan`,m),Y(`polygonStarts`,h),Y(`sortedIds`,c),Y(`ringGroups`,$),X(`sourceIds`,e.sourceIds)],p,`var group = ${E};
  if (index < shellScan[shellScanOffset + ${p-1}u]) {
    group = ringGroups[ringGroupsOffset + sortedIds[sortedIdsOffset + polygonStarts[polygonStartsOffset + index]]];
  }
  sourceIds[sourceIdsOffset + index] = group;`)]:[],J(`polygon-counts`,[Y(`sortedKeys`,s),X(`validRings`,g)],1,`var low = 0u;
  var high = ${p}u;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (sortedKeys[sortedKeysOffset + middle] == ${E}) { high = middle; } else { low = middle + 1u; }
  }
  validRings[validRingsOffset] = low;`),J(`polygon-offsets`,[Y(`lengthScan`,f),Y(`shellScan`,m),Y(`polygonStarts`,h),Y(`validRings`,g),X(`ringOffsetsOut`,e.ringOffsets),X(`polygonOffsetsOut`,e.polygonOffsets),X(`featureOffsets`,e.featureOffsets)],p+1,`let polygonCount = shellScan[shellScanOffset + ${p-1}u];
  // Identity feature table: one feature per polygon, padded flat with the polygon count.
  featureOffsets[featureOffsetsOffset + index] = min(index, polygonCount);
  var ringOffset = 0u;
  if (index > 0u) { ringOffset = lengthScan[lengthScanOffset + index - 1u]; }
  ringOffsetsOut[ringOffsetsOutOffset + index] = ringOffset;
  var polygonOffset = validRings[validRingsOffset];
  if (index < polygonCount) { polygonOffset = polygonStarts[polygonStartsOffset + index]; }
  polygonOffsetsOut[polygonOffsetsOutOffset + index] = polygonOffset;`),J(`polygon-gather`,[Y(`sortedIds`,c),Y(`ringOffsets`,u.ringOffsets),Y(`positions`,u.positions,`f32`),Y(`ringOffsetsOut`,e.ringOffsets),X(`positionsOut`,e.positions,`f32`)],ee,`if (index >= ringOffsetsOut[ringOffsetsOutOffset + ${p}u]) { return; }
  var low = 0u;
  var high = ${p}u;
  while (low + 1u < high) {
    let middle = low + (high - low) / 2u;
    if (ringOffsetsOut[ringOffsetsOutOffset + middle] <= index) { low = middle; } else { high = middle; }
  }
  let ring = sortedIds[sortedIdsOffset + low];
  let source = ringOffsets[ringOffsetsOffset + ring] + (index - ringOffsetsOut[ringOffsetsOutOffset + low]);
  positionsOut[positionsOutOffset + 2u * index] = positions[positionsOffset + 2u * source];
  positionsOut[positionsOutOffset + 2u * index + 1u] = positions[positionsOffset + 2u * source + 1u];`))}return Z.push(J(`publish`,[Y(`ringScan`,z),Y(`ringCount`,V),Y(`counters`,F),X(`outCount`,u.status.count),X(`outOverflow`,u.status.overflow),...u.status.requiredCount?[X(`outTotal`,u.status.requiredCount)]:[]],1,`let total = ringScan[ringScanOffset + ${d-1}u];
  let written = ringCount[ringCountOffset];
  outCount[outCountOffset] = written;
  outOverflow[outOverflowOffset] = select(0u, 1u, written < total || counters[countersOffset + 3u] != 0u);
  ${u.status.requiredCount?`outTotal[outTotalOffset] = total;`:``}`)),(u.status.openSegmentCount||u.status.touchingSegmentCount)&&Z.push(J(`publish-counters`,[Y(`counters`,F),...u.status.openSegmentCount?[X(`outOpen`,u.status.openSegmentCount)]:[],...u.status.touchingSegmentCount?[X(`outTouching`,u.status.touchingSegmentCount)]:[]],1,`${u.status.openSegmentCount?`outOpen[outOpenOffset] = counters[countersOffset] - counters[countersOffset + 2u];`:``}
  ${u.status.touchingSegmentCount?`outTouching[outTouchingOffset] = counters[countersOffset + 1u];`:``}`)),Z;function je(e,n,r){let i=1-r;return a(t,{id:`${o}-${e}-jump-${n}`,operation:D,variant:`${e}-jump`,bindings:[Y(`pointerIn`,H[r]),Y(`valueIn`,U[r]),X(`pointerOut`,H[i]),X(`valueOut`,U[i])],invocationCount:d,body:`let successor = pointerIn[pointerInOffset + index];
  var value = valueIn[valueInOffset + index];
  var pointer = successor;
  if (successor != ${E}) {
    value = ${e===`min`?`min(value, valueIn[valueInOffset + successor])`:`value + valueIn[valueInOffset + successor]`};
    pointer = pointerIn[pointerInOffset + successor];
  }
  pointerOut[pointerOutOffset + index] = pointer;
  valueOut[valueOutOffset + index] = value;`})}}};function k(e,t,n){p(e,Object.fromEntries(A(t)),{separateDispatches:!0});let r=(t,n)=>{p(e,n),u(e,Object.values(n),t)};r([],{segmentRings:t.segmentRings,segmentVertices:t.segmentVertices,segmentFlags:n?t.segmentFlags:void 0}),r([t.segmentVertices],{positions:t.positions}),r([t.positions,t.ringOffsets],{ringAreas:t.ringAreas,ringIsHole:t.ringIsHole}),r([t.positions,t.ringOffsets],{ringShells:t.ringShells}),t.polygons&&(r([t.ringGroups],{sourceIds:t.polygons.sourceIds}),r([],{"polygons.ringOffsets":t.polygons.ringOffsets,"polygons.polygonOffsets":t.polygons.polygonOffsets,"polygons.featureOffsets":t.polygons.featureOffsets}),r([t.positions,t.ringOffsets,t.polygons.ringOffsets],{"polygons.positions":t.polygons.positions})),r([],{"status.count":t.status.count,"status.overflow":t.status.overflow,"status.requiredCount":t.status.requiredCount}),r([],{"status.openSegmentCount":t.status.openSegmentCount,"status.touchingSegmentCount":t.status.touchingSegmentCount})}function A(e){let{kind:t,...n}=e.polygons??{},{status:r,polygons:i,...a}=e;return[...Object.entries(a).map(([e,t])=>[`output.${e}`,t]),...Object.entries(n).map(([e,t])=>[`output.polygons.${e}`,t]),...Object.entries(r).map(([e,t])=>[`output.status.${e}`,t])]}function ae(e){return A(e).map(([,e])=>e)}export{re as n,ie as t};