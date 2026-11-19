import{r as e}from"./kernel-C3tfQOn4.js";import{o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as n,S as r,f as i,y as a}from"./gpu-contributor-utils-BIwLQON5.js";import{i as o,r as s}from"./segment-table-BeVxjLqW.js";import{t as c}from"./gpu-segment-intersection-CGBfhMHr.js";import{v as l}from"./spatial-context-DwZEd4EZ.js";import{i as u}from"./spatial-wgsl-point-helpers-DcOgpqmI.js";import{a as d}from"./graph-chunk-dispatch-zBt8k2-o.js";import{a as f}from"./geometry-storage-validation-T89ZhmCj.js";import{n as p}from"./segment-storage-validation-BUxpxgMF.js";import{t as m}from"./validate-outline-output-passes-DIcqH6Pw.js";import{t as h}from"./ports-xmdKgL07.js";import{A as g}from"./pointer-overlay-Qh8qflcA.js";import{t as _}from"./define-run-B-brkCe4.js";import{r as v}from"./text-DYSgVV-n.js";import{n as y,t as b}from"./pick-Bo3t3hjv.js";import{h as x,i as S,m as C,s as w,v as T}from"./polygon-set-CHja50mr.js";import{o as E}from"./pointer-support-C2rPRVFO.js";import{a as D,s as O}from"./shape-segments-DMcyF4V-.js";import{_o as k}from"./entry-preview-D70TdSbK.js";import{GEOMETRY_VALIDITY_CLASSES_WGSL as A}from"./geometry-validity-classes-BdxQceem.js";var j={nonFinite:1,unclosedRing:2,shortRing:4,repeatedVertex:8,selfIntersection:16,crossingRings:32,holeOutsideShell:64,badOrientation:128,uncertain:256,tooFewPoints:512},M=j.nonFinite|j.unclosedRing|j.shortRing|j.selfIntersection|j.crossingRings|j.holeOutsideShell|j.uncertain|j.tooFewPoints,N=`GPUGeometryValidity`,P=64,F=class{id;props;spatialContext;featureCount;ringCount;polygons;constructor(e){this.id=e.id??`geometry-validity`,this.spatialContext=l(this.id,e.spatialContext),e={...e,points:e.points&&h(this.id,`points`,e.points),lines:e.lines&&h(this.id,`lines`,e.lines)},this.props=e;let{id:t}=this,{polygons:r,lines:o,points:s}=e;if(this.polygons=r,[r,o,s].filter(Boolean).length!==1)throw Error(`${t} requires exactly one of polygons, lines and points`);if(r){n(r.positions,[`float32x2`],`${t} polygons.positions`);for(let[e,i]of[[`featureOffsets`,r.featureOffsets],[`polygonOffsets`,r.polygonOffsets],[`ringOffsets`,r.ringOffsets]])if(n(i,[`uint32`],`${t} polygons.${e}`),i.length<1)throw Error(`${t} polygons.${e} requires a terminal entry`);if(this.featureCount=r.featureOffsets.length-1,this.ringCount=r.ringOffsets.length-1,r.positions.length<1)throw Error(`${t} polygons must contain at least one vertex`);if(!e.output.status?.overflow||e.intersectionCapacity===void 0)throw Error(`${t} polygons require status.overflow and intersectionCapacity`)}else if(o){if(n(o.positions,[`float32x2`],`${t} lines.positions`),n(o.lineOffsets,[`uint32`],`${t} lines.lineOffsets`),o.lineOffsets.length<1)throw Error(`${t} lines.lineOffsets requires a terminal entry`);this.featureCount=o.lineOffsets.length-1,this.ringCount=0}else{let e=s;n(e.positions,[`float32x2`],`${t} points.positions`),this.featureCount=e.positions.length,this.ringCount=0}if(this.featureCount<1)throw Error(`${t} geometry must contain at least one feature`);let{issueCode:c,validity:u,status:d}=e.output;for(let[e,r]of[[`issueCode`,c],[`validity`,u]])if((r||e!==`validity`)&&(n(r,[`uint32`],`${t} output.${e}`),r.length!==this.featureCount))throw Error(`${t} output.${e} length must equal the feature count`);for(let[e,r]of[[`overflow`,d?.overflow],[`intersectionCount`,d?.intersectionCount]])if(r&&(n(r,[`uint32`],`${t} output.status.${e}`),r.length<1))throw Error(`${t} output.status.${e} must contain one uint32 row`);if(a(`${t} output.validity`,[u],[c,d?.overflow,d?.intersectionCount,e.polygons?.positions,e.polygons?.featureOffsets,e.polygons?.polygonOffsets,e.polygons?.ringOffsets,e.lines?.positions,e.lines?.lineOffsets,e.points?.positions]),m(t,{issueCode:c,validity:u,"status.overflow":d?.overflow,"status.intersectionCount":d?.intersectionCount},[]),a(`${t} diagnostic/status outputs`,[c,d?.overflow,d?.intersectionCount],[e.polygons?.positions,e.polygons?.featureOffsets,e.polygons?.polygonOffsets,e.polygons?.ringOffsets,e.lines?.positions,e.lines?.lineOffsets,e.points?.positions]),r&&i(t,`intersectionCapacity`,e.intersectionCapacity??0),e.ringClosure&&![`implicit`,`explicit`].includes(e.ringClosure))throw Error(`${t} ringClosure must be implicit or explicit`);if(e.orientation&&![`counter-clockwise-shell`,`clockwise-shell`,`ignore`].includes(e.orientation))throw Error(`${t} orientation must be counter-clockwise-shell, clockwise-shell or ignore`)}getViews(){let{polygons:e,props:t}=this,n=e?[e.positions,e.featureOffsets,e.polygonOffsets,e.ringOffsets]:t.lines?[t.lines.positions,t.lines.lineOffsets]:[t.points.positions],{issueCode:r,validity:i,status:a}=t.output;return[...n,r,i,a?.overflow,a?.intersectionCount]}getLineOrPointNode(e){let{id:n,props:r,featureCount:i}=this,a=j,{lines:o}=r,c=(o??r.points).positions,l=[{name:`positions`,view:c,type:`f32`,access:`read`}];return o&&l.push({name:`lineOffsets`,view:o.lineOffsets,type:`u32`,access:`read`}),l.push({name:`mask`,view:r.output.issueCode,type:`u32`,access:`read_write`}),t(e,{id:`${n}-simple-kinds`,operation:N,variant:o?`lines`:`points`,bindings:l,invocationCount:i,declarations:`${s}
const VERTEX_COUNT: u32 = ${c.length}u;
const BIT_NON_FINITE: u32 = ${a.nonFinite}u;
const BIT_TOO_FEW: u32 = ${a.tooFewPoints}u;
${u(`vertexAt`,`positions`)}`,body:o?`let start = min(lineOffsets[lineOffsetsOffset + index], VERTEX_COUNT);
  let end = min(lineOffsets[lineOffsetsOffset + index + 1u], VERTEX_COUNT);
  var bits = 0u;
  var distinct = false;
  if (end > start) {
    let first = vertexAt(start);
    for (var k = start; k < end; k++) {
      let vertex = vertexAt(k);
      if (!orientIsFinite(vertex.x) || !orientIsFinite(vertex.y)) { bits = bits | BIT_NON_FINITE; }
      if (vertex.x != first.x || vertex.y != first.y) { distinct = true; }
    }
    if (!distinct) { bits = bits | BIT_TOO_FEW; }
  }
  mask[maskOffset + index] = bits;`:`let vertex = vertexAt(index);
  mask[maskOffset + index] = select(BIT_NON_FINITE, 0u, orientIsFinite(vertex.x) && orientIsFinite(vertex.y));`})}getCommandNodes(n){let{id:i,props:a,polygons:l,featureCount:m,ringCount:h}=this;if(d(n,i,Object.fromEntries(this.getViews().map((e,t)=>[String(t),e]))),a.polygons&&r(i,`polygons.positions rows`,a.polygons.positions.length,o(n.device)),l&&(p(n,i,l.positions.length,l.positions.length,a.intersectionCapacity,a.leafCapacity),f(n,i,{ringBounds:h*16})),!l)return[this.getLineOrPointNode(n),...[`overflow`,`intersectionCount`].flatMap(e=>{let r=a.output.status?.[e];return r?[t(n,{id:`${i}-clear-${e}`,operation:N,variant:`clear-simple-status`,bindings:[{name:`status`,view:r,type:`u32`,access:`read_write`}],invocationCount:1,body:`status[statusOffset] = 0u;`})]:[]}),...this.getValidityNodes(n)];let{issueCode:g,status:_}=a.output,v=a.intersectionCapacity,y=a.orientation??`counter-clockwise-shell`,b=a.ringClosure===`explicit`,x=[];x.push(t(n,{id:`${i}-clear`,operation:N,variant:`clear`,bindings:[{name:`mask`,view:g,type:`u32`,access:`read_write`}],invocationCount:m,body:`mask[maskOffset + index] = 0u;`}));let S=j,C=e(n,`${i}-ring-bounds`,`float32`,Math.max(h,1)*4);x.push(t(n,{id:`${i}-ring-bounds`,operation:N,variant:`ring-bounds`,bindings:[{name:`positions`,view:l.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:l.ringOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:l.polygonOffsets,type:`u32`,access:`read`},{name:`bounds`,view:C,type:`f32`,access:`read_write`}],invocationCount:Math.ceil(h/P)*P,workgroupSize:P,guardIndex:!1,declarations:`${s}
const VERTEX_COUNT: u32 = ${l.positions.length}u;
const POLYGON_COUNT: u32 = ${l.polygonOffsets.length-1}u;
const RING_COUNT: u32 = ${h}u;
const RING_LANES: u32 = ${P}u;
const BOUNDS_LARGE: f32 = 3.4e38;
var<workgroup> queueBits: array<atomic<u32>, 2>;
var<workgroup> queueBitsCopy: array<u32, 2>;
var<workgroup> queuedFirst: array<u32, ${P}>;
var<workgroup> queuedLast: array<u32, ${P}>;
var<workgroup> partialMinimum: array<vec2f, ${P}>;
var<workgroup> partialMaximum: array<vec2f, ${P}>;
var<workgroup> partialNonFinite: array<u32, ${P}>;
${u(`vertexAt`,`positions`)}
// Last polygon whose first ring is at most ring, or SEGMENT_NONE.
fn lastPolygonAtMost(ring: u32) -> u32 {
  if (polygonOffsets[polygonOffsetsOffset] > ring) { return SEGMENT_NONE; }
  var low = 0u;
  var high = POLYGON_COUNT;
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (polygonOffsets[polygonOffsetsOffset + middle] <= ring) { low = middle; } else { high = middle; }
  }
  return low;
}
// Non-finite or empty rings get the whole plane, so the prefilter never rejects through them.
fn writeBounds(ring: u32, minimum: vec2f, maximum: vec2f, nonFinite: u32) {
  let unbounded = nonFinite != 0u || minimum.x > maximum.x;
  let low = select(minimum, vec2f(-BOUNDS_LARGE), unbounded);
  let high = select(maximum, vec2f(BOUNDS_LARGE), unbounded);
  bounds[boundsOffset + 4u * ring] = low.x;
  bounds[boundsOffset + 4u * ring + 1u] = low.y;
  bounds[boundsOffset + 4u * ring + 2u] = high.x;
  bounds[boundsOffset + 4u * ring + 3u] = high.y;
}`,body:`let lane = localInvocationIndex;
  let ringBase = workgroupIndex * RING_LANES;
  let ownRing = ringBase + lane;
  // Phase 1: the shell of a polygon with holes is bounded serially by its own lane when small,
  // and queued for the cooperative phase otherwise.
  if (ownRing < RING_COUNT && POLYGON_COUNT > 0u) {
    let polygon = lastPolygonAtMost(ownRing);
    if (polygon != SEGMENT_NONE && ownRing < polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT]) {
      let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
      let nextRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
      if (ownRing == firstRing && nextRing - firstRing >= 2u) {
        let first = ringOffsets[ringOffsetsOffset + ownRing];
        let last = min(ringOffsets[ringOffsetsOffset + ownRing + 1u], VERTEX_COUNT);
        if (last <= first || last - first <= RING_LANES) {
          var minimum = vec2f(BOUNDS_LARGE);
          var maximum = vec2f(-BOUNDS_LARGE);
          var nonFinite = 0u;
          for (var vertex = first; vertex < last; vertex++) {
            let point = vertexAt(vertex);
            if (!orientIsFinite(point.x) || !orientIsFinite(point.y)) {
              nonFinite = 1u;
            } else {
              minimum = min(minimum, point);
              maximum = max(maximum, point);
            }
          }
          writeBounds(ownRing, minimum, maximum, nonFinite);
        } else {
          queuedFirst[lane] = first;
          queuedLast[lane] = last;
          atomicOr(&queueBits[lane / 32u], 1u << (lane % 32u));
        }
      }
    }
  }
  workgroupBarrier();
  if (lane == 0u) {
    queueBitsCopy[0] = atomicLoad(&queueBits[0]);
    queueBitsCopy[1] = atomicLoad(&queueBits[1]);
  }
  // Phase 2: queued rings in ascending order; all barriers are in workgroup-uniform control flow.
  for (var word = 0u; word < 2u; word++) {
    var pending = workgroupUniformLoad(&queueBitsCopy[word]);
    while (pending != 0u) {
      let slot = word * 32u + firstTrailingBit(pending);
      pending = pending & (pending - 1u);
      let first = workgroupUniformLoad(&queuedFirst[slot]);
      let last = workgroupUniformLoad(&queuedLast[slot]);
      var minimum = vec2f(BOUNDS_LARGE);
      var maximum = vec2f(-BOUNDS_LARGE);
      var nonFinite = 0u;
      for (var vertex = first + lane; vertex < last; vertex += RING_LANES) {
        let point = vertexAt(vertex);
        if (!orientIsFinite(point.x) || !orientIsFinite(point.y)) {
          nonFinite = 1u;
        } else {
          minimum = min(minimum, point);
          maximum = max(maximum, point);
        }
      }
      partialMinimum[lane] = minimum;
      partialMaximum[lane] = maximum;
      partialNonFinite[lane] = nonFinite;
      workgroupBarrier();
      for (var stride = RING_LANES / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
          partialMinimum[lane] = min(partialMinimum[lane], partialMinimum[lane + stride]);
          partialMaximum[lane] = max(partialMaximum[lane], partialMaximum[lane + stride]);
          partialNonFinite[lane] = partialNonFinite[lane] | partialNonFinite[lane + stride];
        }
        workgroupBarrier();
      }
      if (lane == 0u) {
        writeBounds(ringBase + slot, partialMinimum[0], partialMaximum[0], partialNonFinite[0]);
      }
    }
  }`})),x.push(t(n,{id:`${i}-rings`,operation:N,variant:`rings`,bindings:[{name:`positions`,view:l.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:l.ringOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:l.polygonOffsets,type:`u32`,access:`read`},{name:`featureOffsets`,view:l.featureOffsets,type:`u32`,access:`read`},{name:`ringBounds`,view:C,type:`f32`,access:`read`},{name:`mask`,view:g,type:`atomic<u32>`,access:`read_write`}],invocationCount:Math.ceil(h/P)*P,workgroupSize:P,guardIndex:!1,declarations:`${s}
const VERTEX_COUNT: u32 = ${l.positions.length}u;
const POLYGON_COUNT: u32 = ${l.polygonOffsets.length-1}u;
const FEATURE_COUNT: u32 = ${l.featureOffsets.length-1}u;
const RING_COUNT: u32 = ${h}u;
const RING_LANES: u32 = ${P}u;
const EXPLICIT_CLOSURE: bool = ${b};
const CHECK_ORIENTATION: bool = ${y!==`ignore`};
const SHELL_COUNTER_CLOCKWISE: bool = ${y!==`clockwise-shell`};
const BIT_NON_FINITE: u32 = ${S.nonFinite}u;
const BIT_UNCLOSED: u32 = ${S.unclosedRing}u;
const BIT_SHORT: u32 = ${S.shortRing}u;
const BIT_REPEATED: u32 = ${S.repeatedVertex}u;
const BIT_HOLE_OUTSIDE: u32 = ${S.holeOutsideShell}u;
const BIT_ORIENTATION: u32 = ${S.badOrientation}u;
const BIT_UNCERTAIN: u32 = ${S.uncertain}u;
${u(`vertexAt`,`positions`)}
fn samePoint(first: vec2f, second: vec2f) -> bool { return first.x == second.x && first.y == second.y; }
fn readOffset(kind: u32, row: u32) -> u32 {
  if (kind == 0u) { return polygonOffsets[polygonOffsetsOffset + row]; }
  return featureOffsets[featureOffsetsOffset + row];
}
// Last row in [0, count) whose offset is at most value, or SEGMENT_NONE.
fn lastAtMost(kind: u32, count: u32, value: u32) -> u32 {
  if (count == 0u || readOffset(kind, 0u) > value) { return SEGMENT_NONE; }
  var low = 0u;
  var high = count;
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (readOffset(kind, middle) <= value) { low = middle; } else { high = middle; }
  }
  return low;
}
fn ringStartOf(ring: u32) -> u32 { return ringOffsets[ringOffsetsOffset + ring]; }
fn ringEndOf(ring: u32) -> u32 { return min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT); }
// Vertex count without a closing duplicate of the first vertex.
fn effectiveCount(start: u32, end: u32) -> u32 {
  if (end <= start) { return 0u; }
  let count = end - start;
  if (count >= 2u && samePoint(vertexAt(start), vertexAt(end - 1u))) { return count - 1u; }
  return count;
}
const NO_INDEX: u32 = 0xffffffffu;
// Per-ring state of the block, indexed by the ring's lane. Written by the owner lane in phase 1 and
// by lane 0 for cooperatively processed rings; read through workgroupUniformLoad when it steers a barrier.
var<workgroup> ringActive: array<u32, ${P}>;
var<workgroup> ringFeature: array<u32, ${P}>;
var<workgroup> ringIsHole: array<u32, ${P}>;
var<workgroup> ringStart: array<u32, ${P}>;
var<workgroup> ringEnd: array<u32, ${P}>;
var<workgroup> ringFirstRing: array<u32, ${P}>;
var<workgroup> ringLarge: array<u32, ${P}>;
var<workgroup> ringBits: array<u32, ${P}>;
var<workgroup> ringNeeds: array<u32, ${P}>;
var<workgroup> ringEffective: array<u32, ${P}>;
var<workgroup> ringShellStart: array<u32, ${P}>;
var<workgroup> ringShellCount: array<u32, ${P}>;
var<workgroup> ringShellRing: array<u32, ${P}>;
var<workgroup> queueBits: array<atomic<u32>, 2>;
var<workgroup> queueBitsCopy: array<u32, 2>;
var<workgroup> sharedPoint: vec2f;
var<workgroup> sharedResult: array<u32, 2>;
var<workgroup> partialFlags: array<u32, ${P}>;
var<workgroup> partialExtreme: array<u32, ${P}>;
var<workgroup> partialPoint: array<vec2f, ${P}>;
var<workgroup> partialEdge: array<u32, ${P}>;
var<workgroup> partialKind: array<u32, ${P}>;
var<workgroup> partialParity: array<u32, ${P}>;
fn isLexicographicallyLess(candidate: vec2f, best: vec2f) -> bool {
  return candidate.x < best.x || (candidate.x == best.x && candidate.y < best.y);
}
// Winding of the ring at its lexicographically smallest vertex (first one on ties): 1
// counter-clockwise, -1 clockwise, 0 undetermined, 2 uncertain. A known extreme (from the
// cooperative search) skips the serial search when it is not NO_INDEX.
fn ringWinding(start: u32, count: u32, knownExtreme: u32) -> i32 {
  var extreme = 0u;
  if (knownExtreme != NO_INDEX) {
    extreme = knownExtreme;
  } else {
    for (var k = 1u; k < count; k++) {
      if (isLexicographicallyLess(vertexAt(start + k), vertexAt(start + extreme))) { extreme = k; }
    }
  }
  let center = vertexAt(start + extreme);
  var previous = extreme;
  var next = extreme;
  var foundPrevious = false;
  var foundNext = false;
  for (var step = 1u; step < count; step++) {
    if (!foundPrevious) {
      previous = (extreme + count - step) % count;
      foundPrevious = !samePoint(vertexAt(start + previous), center);
    }
    if (!foundNext) {
      next = (extreme + step) % count;
      foundNext = !samePoint(vertexAt(start + next), center);
    }
    // Both neighbours are normally the adjacent vertices: stop instead of walking the whole ring.
    if (foundPrevious && foundNext) { break; }
  }
  if (!foundPrevious || !foundNext) { return 0; }
  return orientSign(vertexAt(start + previous), center, vertexAt(start + next));
}
// Folds the ring-local checks of ring slot into ringBits and decides whether the ring is a hole
// whose containment in its shell must be tested (ringNeeds, with the shell in the other slots arrays).
fn finishRing(slot: u32, start: u32, end: u32, isHole: bool, polygonFirstRing: u32, finiteAll: bool, repeated: bool, extreme: u32) {
  let count = select(0u, end - start, end > start);
  var bits = 0u;
  if (!finiteAll) { bits = bits | BIT_NON_FINITE; }
  if (repeated) { bits = bits | BIT_REPEATED; }
  if (EXPLICIT_CLOSURE && count >= 1u && !samePoint(vertexAt(start), vertexAt(end - 1u))) { bits = bits | BIT_UNCLOSED; }
  let effective = effectiveCount(start, end);
  let isShort = effective < 3u;
  if (isShort) { bits = bits | BIT_SHORT; }
  var needsContainment = 0u;
  if (finiteAll && !isShort) {
    if (CHECK_ORIENTATION) {
      let winding = ringWinding(start, effective, extreme);
      if (winding == 2) { bits = bits | BIT_UNCERTAIN; }
      else if (winding != 0) {
        let wantsCounterClockwise = select(SHELL_COUNTER_CLOCKWISE, !SHELL_COUNTER_CLOCKWISE, isHole);
        if ((winding > 0) != wantsCounterClockwise) { bits = bits | BIT_ORIENTATION; }
      }
    }
    if (isHole) {
      let shellStart = ringStartOf(polygonFirstRing);
      let shellCount = effectiveCount(shellStart, ringEndOf(polygonFirstRing));
      if (shellCount >= 3u) {
        needsContainment = 1u;
        ringEffective[slot] = effective;
        ringShellStart[slot] = shellStart;
        ringShellCount[slot] = shellCount;
        ringShellRing[slot] = polygonFirstRing;
      }
    }
  }
  ringBits[slot] = bits;
  ringNeeds[slot] = needsContainment;
}
// Workgroup-cooperative point-in-ring: 1 inside, 0 outside, 2 on the boundary, 3 uncertain, with
// the serial walk's result: the event (boundary or uncertain) at the lowest edge index wins,
// otherwise the crossing parity decides. A point outside the shell's bounds is outside it.
// Must be called from workgroup-uniform control flow with a uniform argument.
fn locateInShell(point: vec2f, shellStart: u32, count: u32, shellRing: u32, lane: u32) -> u32 {
  if (lane == 0u) {
    let base = ringBoundsOffset + 4u * shellRing;
    sharedResult[0] = select(0u, 1u,
      point.x < ringBounds[base] || point.x > ringBounds[base + 2u] ||
      point.y < ringBounds[base + 1u] || point.y > ringBounds[base + 3u]);
  }
  if (workgroupUniformLoad(&sharedResult[0]) == 1u) { return 0u; }
  var eventEdge = NO_INDEX;
  var eventKind = 0u;
  var parity = 0u;
  for (var k = lane; k < count; k += RING_LANES) {
    let a = vertexAt(shellStart + k);
    let b = vertexAt(shellStart + (k + 1u) % count);
    let inBox = point.x >= min(a.x, b.x) && point.x <= max(a.x, b.x) &&
      point.y >= min(a.y, b.y) && point.y <= max(a.y, b.y);
    let straddles = (a.y > point.y) != (b.y > point.y);
    if (!inBox && !straddles) { continue; }
    let side = orientSign(a, b, point);
    if (side == 2) { eventEdge = k; eventKind = 3u; break; }
    if (inBox && side == 0) { eventEdge = k; eventKind = 2u; break; }
    if (straddles && ((b.y > a.y) == (side > 0))) { parity = parity ^ 1u; }
  }
  partialEdge[lane] = eventEdge;
  partialKind[lane] = eventKind;
  partialParity[lane] = parity;
  workgroupBarrier();
  for (var stride = RING_LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      if (partialEdge[lane + stride] < partialEdge[lane]) {
        partialEdge[lane] = partialEdge[lane + stride];
        partialKind[lane] = partialKind[lane + stride];
      }
      partialParity[lane] = partialParity[lane] ^ partialParity[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    sharedResult[1] = select(partialParity[0], partialKind[0], partialEdge[0] != NO_INDEX);
  }
  return workgroupUniformLoad(&sharedResult[1]);
}`,body:`let lane = localInvocationIndex;
  let ring = workgroupIndex * RING_LANES + lane;
  // Phase 1: this lane's own ring. Small rings run the original serial checks; large rings and
  // holes that need a containment walk are queued (bitmask over the block's lanes).
  if (ring < RING_COUNT) {
    let polygon = lastAtMost(0u, POLYGON_COUNT, ring);
    if (polygon != SEGMENT_NONE && ring < polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT] &&
        polygon < featureOffsets[featureOffsetsOffset + FEATURE_COUNT]) {
      let feature = lastAtMost(1u, FEATURE_COUNT, polygon);
      if (feature != SEGMENT_NONE) {
        let start = ringStartOf(ring);
        let end = ringEndOf(ring);
        let isHole = ring != polygonOffsets[polygonOffsetsOffset + polygon];
        let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
        ringActive[lane] = 1u;
        ringFeature[lane] = feature;
        ringIsHole[lane] = select(0u, 1u, isHole);
        ringStart[lane] = start;
        ringEnd[lane] = end;
        ringFirstRing[lane] = firstRing;
        let count = select(0u, end - start, end > start);
        if (count <= RING_LANES) {
          var finiteAll = true;
          var repeated = false;
          for (var k = 0u; k < count; k++) {
            let vertex = vertexAt(start + k);
            if (!orientIsFinite(vertex.x) || !orientIsFinite(vertex.y)) { finiteAll = false; }
            if (k + 1u < count && samePoint(vertex, vertexAt(start + k + 1u))) { repeated = true; }
          }
          finishRing(lane, start, end, isHole, firstRing, finiteAll, repeated, NO_INDEX);
          if (ringNeeds[lane] == 1u) { atomicOr(&queueBits[lane / 32u], 1u << (lane % 32u)); }
        } else {
          ringLarge[lane] = 1u;
          atomicOr(&queueBits[lane / 32u], 1u << (lane % 32u));
        }
      }
    }
  }
  workgroupBarrier();
  if (lane == 0u) {
    queueBitsCopy[0] = atomicLoad(&queueBits[0]);
    queueBitsCopy[1] = atomicLoad(&queueBits[1]);
  }
  // Phase 2: queued rings in ascending order. Every barrier is in workgroup-uniform control flow:
  // the queue and everything that steers a loop or branch go through workgroupUniformLoad.
  for (var word = 0u; word < 2u; word++) {
    var pending = workgroupUniformLoad(&queueBitsCopy[word]);
    while (pending != 0u) {
      let slot = word * 32u + firstTrailingBit(pending);
      pending = pending & (pending - 1u);
      let start = workgroupUniformLoad(&ringStart[slot]);
      if (workgroupUniformLoad(&ringLarge[slot]) == 1u) {
        // Large ring: lanes scan strided vertices for non-finite coordinates, repeated neighbours
        // and the lexicographic minimum, then a tree combines them.
        let end = workgroupUniformLoad(&ringEnd[slot]);
        let count = select(0u, end - start, end > start);
        let effective = effectiveCount(start, end);
        var flags = 0u;
        var best = NO_INDEX;
        var bestPoint = vec2f(0.0);
        for (var k = lane; k < count; k += RING_LANES) {
          let vertex = vertexAt(start + k);
          if (!orientIsFinite(vertex.x) || !orientIsFinite(vertex.y)) { flags = flags | 1u; }
          if (k + 1u < count && samePoint(vertex, vertexAt(start + k + 1u))) { flags = flags | 2u; }
          if (k < effective && (best == NO_INDEX || isLexicographicallyLess(vertex, bestPoint))) {
            best = k;
            bestPoint = vertex;
          }
        }
        partialFlags[lane] = flags;
        partialExtreme[lane] = best;
        partialPoint[lane] = bestPoint;
        workgroupBarrier();
        for (var stride = RING_LANES / 2u; stride > 0u; stride = stride / 2u) {
          if (lane < stride) {
            partialFlags[lane] = partialFlags[lane] | partialFlags[lane + stride];
            let other = partialExtreme[lane + stride];
            let current = partialExtreme[lane];
            if (other != NO_INDEX && (current == NO_INDEX ||
                isLexicographicallyLess(partialPoint[lane + stride], partialPoint[lane]) ||
                (partialPoint[lane + stride].x == partialPoint[lane].x &&
                 partialPoint[lane + stride].y == partialPoint[lane].y && other < current))) {
              partialExtreme[lane] = other;
              partialPoint[lane] = partialPoint[lane + stride];
            }
          }
          workgroupBarrier();
        }
        if (lane == 0u) {
          finishRing(slot, start, end, ringIsHole[slot] == 1u, ringFirstRing[slot],
            (partialFlags[0] & 1u) == 0u, (partialFlags[0] & 2u) != 0u, partialExtreme[0]);
        }
      }
      // The hole-in-shell test runs on the whole workgroup whatever the hole's size, because the
      // shell may be huge. One hole vertex off the shell boundary decides containment once no ring
      // edges cross.
      if (workgroupUniformLoad(&ringNeeds[slot]) == 1u) {
        let effective = workgroupUniformLoad(&ringEffective[slot]);
        let shellStart = workgroupUniformLoad(&ringShellStart[slot]);
        let shellCount = workgroupUniformLoad(&ringShellCount[slot]);
        let shellRing = workgroupUniformLoad(&ringShellRing[slot]);
        var located = 2u;
        for (var k = 0u; k < effective && located == 2u; k++) {
          if (lane == 0u) { sharedPoint = vertexAt(start + k); }
          let point = workgroupUniformLoad(&sharedPoint);
          located = locateInShell(point, shellStart, shellCount, shellRing, lane);
        }
        if (lane == 0u) {
          if (located == 3u) { ringBits[slot] = ringBits[slot] | BIT_UNCERTAIN; }
          else if (located != 1u) { ringBits[slot] = ringBits[slot] | BIT_HOLE_OUTSIDE; }
        }
      }
    }
  }
  workgroupBarrier();
  if (ringActive[lane] == 1u && ringBits[lane] != 0u) {
    atomicOr(&mask[maskOffset + ringFeature[lane]], ringBits[lane]);
  }`}));let w=e(n,`${i}-pair-left`,`uint32`,v),T=e(n,`${i}-pair-right`,`uint32`,v),E=e(n,`${i}-pair-count`,`uint32`,1),D=e(n,`${i}-kinds`,`uint32`,v),O=e(n,`${i}-left-features`,`uint32`,v),k=e(n,`${i}-left-rings`,`uint32`,v),A=e(n,`${i}-right-rings`,`uint32`,v),M=new c({id:`${i}-intersection`,left:l,sameFeatureOnly:!0,leafCapacity:a.leafCapacity,output:{pairs:{leftRows:w,rightRows:T,status:{count:E,overflow:_?.overflow,requiredCount:_?.intersectionCount}},kinds:D,leftFeatures:O,leftRings:k,rightRings:A}});return x.push(...M.getCommandNodes(n)),x.push(t(n,{id:`${i}-fold`,operation:N,variant:`fold`,bindings:[{name:`pairCount`,view:E,type:`u32`,access:`read`},{name:`kinds`,view:D,type:`u32`,access:`read`},{name:`leftFeatures`,view:O,type:`u32`,access:`read`},{name:`leftRings`,view:k,type:`u32`,access:`read`},{name:`rightRings`,view:A,type:`u32`,access:`read`},{name:`mask`,view:g,type:`atomic<u32>`,access:`read_write`}],invocationCount:v,declarations:`${s}
const BIT_SELF: u32 = ${S.selfIntersection}u;
const BIT_CROSSING: u32 = ${S.crossingRings}u;
const BIT_UNCERTAIN: u32 = ${S.uncertain}u;`,body:`if (index >= pairCount[pairCountOffset]) { return; }
  let kind = kinds[kindsOffset + index];
  var bits = 0u;
  if (kind == KIND_UNCERTAIN) {
    bits = BIT_UNCERTAIN;
  } else if (leftRings[leftRingsOffset + index] == rightRings[rightRingsOffset + index]) {
    bits = BIT_SELF;
  } else if (kind == KIND_PROPER || kind == KIND_OVERLAP) {
    bits = BIT_CROSSING;
  }
  if (bits != 0u) { atomicOr(&mask[maskOffset + leftFeatures[leftFeaturesOffset + index]], bits); }`})),[...x,...this.getValidityNodes(n)]}getValidityNodes(e){let{issueCode:n,validity:r,status:i}=this.props.output;if(!r)return[];let a=this.polygons?i?.overflow:void 0;return[t(e,{id:`${this.id}-validity`,operation:N,variant:`validity`,bindings:[{name:`issueCode`,view:n,type:`u32`,access:`read`},{name:`validity`,view:r,type:`u32`,access:`read_write`},...a?[{name:`overflow`,view:a,type:`u32`,access:`read`}]:[]],invocationCount:this.featureCount,body:`validity[validityOffset + index] = select(0u, 1u,
        (issueCode[issueCodeOffset + index] & ${M}u) == 0u${a?` && overflow[overflowOffset] == 0u`:``});`})]}},I=[[.435,.545,.722],[1,.42,.42],[.96,.835,.28],[.31,.82,.77],[.78,.57,.92],[.55,.55,.55],[.68,.45,.8]],L=[`valid`,`self-intersecting`,`hole outside, rings cross`,`wrong orientation`,`other diagnostic`,`incomplete check`,`uncertain predicate`],R=.8,z=[`counter-clockwise-shell`,`clockwise-shell`,`ignore`],B=[`bow-tie`,`hole-outside`,`repeated-vertex`],V=[[1,`nonFinite`],[2,`unclosedRing`],[4,`shortRing`],[8,`repeatedVertex`],[16,`selfIntersection`],[32,`crossingRings`],[64,`holeOutsideShell`],[128,`badOrientation`],[256,`uncertain`]],H=_({parameters:k,caption:`Polygon edges colored by validity class: valid, self-intersecting, hole outside, orientation.`,createInputs:({seed:e,world:t})=>{let n=w(g(e),{width:t.width,height:t.height*R},{hasHoles:!0,rows:3,columns:5,mixedWinding:!0});return{set:n,index:y(n.centers),focus:{feature:-1}}},build(e,{set:t},n){let r=T(e,t),i=e.allocate(`mask`,`uint32`,t.count),a=e.allocate(`validity`,`uint32`,t.count),o=e.allocate(`overflow`,`uint32`,1);e.graph.add(new F({polygons:{kind:`polygons`,positions:r.positions.view,featureOffsets:r.featureOffsets.view,polygonOffsets:r.polygonOffsets.view,ringOffsets:r.ringOffsets.view},orientation:z[n.orientation],intersectionCapacity:4096,output:{issueCode:i.view,validity:a.view,status:{overflow:o.view}}}));let s=e.allocate(`classes`,`uint32`,t.count);O(e,{id:`validity-classes`,invocationCount:t.count,bindings:[{name:`mask`,buffer:i,access:`read`},{name:`validity`,buffer:a,access:`read`},{name:`classes`,buffer:s,access:`read_write`}],body:A});let c=e.upload(`feature-of-vertex`,`uint32`,S(t));return{outputs:{...r,mask:i,validity:a,overflow:o,vertexMasks:D(e,`class`,s,c)},update:e=>r.positions.write(x(t,e.defects,B))}},rebuildOn:[`orientation`],interaction:{kind:`probe`,hint:`hover a polygon`,rest:({inputs:e,toUnit:t})=>t([e.set.centers[6],e.set.centers[7]]),resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=b(t.index,e.x,e.y,t.focus.feature);if(t.focus.feature=a,a<0)return{};let{set:o}=t,s=x(o,r.defects,B),{first:c,count:l}=C(o,a),u=i(n.mask)?.[a],d=i(n.validity)?.[a],f=u===void 0?[`...`]:V.filter(([e])=>(u&e)!==0).map(([,e])=>e);return{focus:a,glyphs:[{shape:`polyline`,points:s.subarray(c*2,(c+l)*2),closed:!0},...E(s,c,l).map(e=>({shape:`dot`,at:e}))],readout:u===void 0||d===void 0?`...`:d===0&&!(u&-137)?[`incomplete check`,...f].join(` · `):u===0?`valid`:f.join(` · `)}}},draw:({outputs:e})=>[{layer:`segments`,nodes:e.positions,edges:e.edges,values:e.vertexMasks,categories:I,radius:2.4},{layer:`text`,...v({corner:`top-left`,panel:!1,entries:L.map((e,t)=>({label:e,swatch:I[t]}))})}]});export{H as default};