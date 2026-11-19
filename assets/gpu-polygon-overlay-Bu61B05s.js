import{r as e,u as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-scan-B4NNTfF9.js";import{E as r,S as i,T as a,f as o,g as s,l as c,s as l,v as u,x as ee}from"./gpu-contributor-utils-BIwLQON5.js";import{t as d}from"./validate-outline-output-passes-DIcqH6Pw.js";import{o as f}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as te}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as ne}from"./common-CshkX-d_.js";import{t as re}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{n as ie,t as ae}from"./gpu-snap-rounded-noding-BglCAlqG.js";import{i as oe}from"./segment-table-BeVxjLqW.js";import{t as se}from"./segment-bvh-CDgckfAv.js";import{t as ce}from"./gpu-segment-ring-assembly-VExWprVe.js";import{_ as le,v as p}from"./spatial-context-DwZEd4EZ.js";import{t as m}from"./exact-orientation-wgsl-B_jI-HEc.js";import{t as h}from"./exact-geometry-wgsl-y7TUnqOh.js";function ue(e,t,n){let r=n.expandedVertexCount+n.intersections,i=Math.max(n.nodedSegments+n.expandedVertexCount,2*r),a=c(n.outputRingCapacity),s=Math.max(2,c(n.nodedSegments));for(let[c,l]of Object.entries({combinedPositions:n.expandedVertexCount*8,combinedOffsets:(n.inputRingCount+1)*4,hotPixelSegmentTables:r*128,crossingTerms:n.intersections*8*4,candidateListsAndScan:i*2*4,nodedLatticeAndGroups:n.nodedSegments*16,groupHierarchy:(2*s-1)*8,latticeBoundary:n.boundaryCapacity*16,ringPositions:n.outputVertexCapacity*8,ringBounds:n.outputRingCapacity*16,ringShellRecords:n.outputRingCapacity*3*4,ringOffsets:(n.outputRingCapacity+1)*4,ringShellHierarchy:(2*a-1)*8,boundsPartials:Math.min(256,Math.max(1,Math.ceil(Math.max(4*r,n.nodedSegments,n.outputRingCapacity)/4096)))*16,...n.leafCapacity?{requestedHierarchy:(2*n.leafCapacity-1)*8}:{}})){let n=Math.max(4,l);u(e.device,t,c,n,`reduce capacities or split input features`),o(t,`${c} byteLength`,n,4,e.device.limits.maxBufferSize)}}function de(e){return[...l(e.geometry),...l(e.boundary),...l(e.status),...l(e.featureSources??{})]}function fe(t,r){let{id:i,output:a}=r,o=r.ringShells.length,s=e(t,`${i}-source-shell-flags`,`uint32`,o),c=e(t,`${i}-source-shell-ranks`,`uint32`,o),l=e(t,`${i}-source-minimum`,`uint32`,o),u=e(t,`${i}-source-maximum`,`uint32`,o);return[f(t,{id:`${i}-source-shells`,operation:`GPUPolygonOverlay`,variant:`feature-source-shells`,invocationCount:o,bindings:[{name:`shells`,view:r.ringShells,type:`u32`,access:`read`},{name:`areas`,view:r.ringAreas,type:`f32`,access:`read`},{name:`ringCount`,view:r.ringCount,type:`u32`,access:`read`},{name:`flags`,view:s,type:`u32`,access:`read_write`},{name:`minimum`,view:l,type:`u32`,access:`read_write`},{name:`maximum`,view:u,type:`u32`,access:`read_write`}],body:`flags[flagsOffset + index] = select(0u, 1u, index < ringCount[ringCountOffset] && shells[shellsOffset + index] == index && abs(areas[areasOffset + index]) > 0.0);
minimum[minimumOffset + index] = 0xffffffffu;
maximum[maximumOffset + index] = 0u;`}),...new n({id:`${i}-source-shell-scan`,input:s,output:c,mode:`exclusive`}).getCommandNodes(t),f(t,{id:`${i}-reduce-feature-sources`,operation:`GPUPolygonOverlay`,variant:`reduce-feature-sources`,invocationCount:r.segmentRings.length,bindings:[{name:`segmentRings`,view:r.segmentRings,type:`u32`,access:`read`},{name:`shells`,view:r.ringShells,type:`u32`,access:`read`},{name:`ranks`,view:c,type:`u32`,access:`read`},{name:`boundaryCount`,view:r.boundaryCount,type:`u32`,access:`read`},{name:`sourceRows`,view:r.sourceRows,type:`u32`,access:`read`},{name:`operandIds`,view:r.operandIds,type:`u32`,access:`read`},{name:`minimum`,view:l,type:`atomic<u32>`,access:`read_write`},{name:`maximum`,view:u,type:`atomic<u32>`,access:`read_write`}],body:`if (index >= boundaryCount[boundaryCountOffset]) { return; }
let ring = segmentRings[segmentRingsOffset + index];
if (ring == 0xffffffffu) { return; }
let shell = shells[shellsOffset + ring];
if (shell == 0xffffffffu) { return; }
let feature = ranks[ranksOffset + shell];
let owner = (operandIds[operandIdsOffset + index] << 31u) | sourceRows[sourceRowsOffset + index];
atomicMin(&minimum[minimumOffset + feature], owner);
atomicMax(&maximum[maximumOffset + feature], owner);`}),f(t,{id:`${i}-publish-feature-sources`,operation:`GPUPolygonOverlay`,variant:`publish-feature-sources`,invocationCount:o,bindings:[{name:`minimum`,view:l,type:`u32`,access:`read`},{name:`maximum`,view:u,type:`u32`,access:`read`},{name:`sourceRows`,view:a.sourceRows,type:`u32`,access:`read_write`},{name:`operandIds`,view:a.operandIds,type:`u32`,access:`read_write`},{name:`mixed`,view:a.mixed,type:`u32`,access:`read_write`}],body:`let owner = minimum[minimumOffset + index];
sourceRows[sourceRowsOffset + index] = select(owner & 0x7fffffffu, 0xffffffffu, owner == 0xffffffffu);
operandIds[operandIdsOffset + index] = select(owner >> 31u, 0xffffffffu, owner == 0xffffffffu);
mixed[mixedOffset + index] = select(0u, 1u, owner != 0xffffffffu && owner != maximum[maximumOffset + index]);`})]}function g(e,t){let n=t.geometry.ringOffsets.length-1;if(t.featureSources){for(let[i,o]of Object.entries(t.featureSources))r(o,[`uint32`],e),a(e,o.length===n,`output.featureSources.${i}.length must equal ring capacity`);d(e,{positions:t.geometry.positions,featureOffsets:t.geometry.featureOffsets,polygonOffsets:t.geometry.polygonOffsets,ringOffsets:t.geometry.ringOffsets,...Object.fromEntries(Object.entries(t.status).map(([e,t])=>[`status.${e}`,t])),...Object.fromEntries(Object.entries(t.boundary).filter(([e])=>e!==`status`)),...Object.fromEntries(Object.entries(t.boundary.status).map(([e,t])=>[`boundary.status.${e}`,t])),...Object.fromEntries(Object.entries(t.featureSources).map(([e,t])=>[`featureSources.${e}`,t]))},[[`featureSources.sourceRows`,`featureSources.operandIds`,`featureSources.mixed`]])}}var _=`
fn wideSetInteger(destination: ptr<function, array<u32, 6>>, value: i32) {
  let bits = bitcast<u32>(value);
  let extension = select(0u, 0xffffu, value < 0);
  (*destination)[0] = bits & 0xffffu;
  (*destination)[1] = (bits >> 16u) & 0xffffu;
  for (var limb = 2u; limb < 6u; limb++) {
    (*destination)[limb] = extension;
  }
}

// product = left * right modulo 2^96 (two's complement, limbs in [0, 65535]).
fn wideMultiply(
  left: ptr<function, array<u32, 6>>,
  right: ptr<function, array<u32, 6>>,
  product: ptr<function, array<u32, 6>>
) {
  var columns: array<u32, 7>;
  for (var i = 0u; i < 6u; i++) {
    for (var j = 0u; i + j < 6u; j++) {
      let partial = (*left)[i] * (*right)[j];
      columns[i + j] = columns[i + j] + (partial & 0xffffu);
      columns[i + j + 1u] = columns[i + j + 1u] + ((partial >> 16u) & 0xffffu);
    }
  }
  var carry = 0u;
  for (var limb = 0u; limb < 6u; limb++) {
    let total = columns[limb] + carry;
    (*product)[limb] = total & 0xffffu;
    carry = (total >> 16u) & 0xffffu;
  }
}

fn wideAdd(
  left: ptr<function, array<u32, 6>>,
  right: ptr<function, array<u32, 6>>,
  sum: ptr<function, array<u32, 6>>
) {
  var carry = 0u;
  for (var limb = 0u; limb < 6u; limb++) {
    let total = (*left)[limb] + (*right)[limb] + carry;
    (*sum)[limb] = total & 0xffffu;
    carry = (total >> 16u) & 0xffffu;
  }
}

fn wideSubtract(
  left: ptr<function, array<u32, 6>>,
  right: ptr<function, array<u32, 6>>,
  difference: ptr<function, array<u32, 6>>
) {
  var carry = 1u;
  for (var limb = 0u; limb < 6u; limb++) {
    let total = (*left)[limb] + ((*right)[limb] ^ 0xffffu) + carry;
    (*difference)[limb] = total & 0xffffu;
    carry = (total >> 16u) & 0xffffu;
  }
}

fn wideSign(value: ptr<function, array<u32, 6>>) -> i32 {
  if ((*value)[5] >= 0x8000u) { return -1; }
  for (var limb = 0u; limb < 6u; limb++) {
    if ((*value)[limb] != 0u) { return 1; }
  }
  return 0;
}

// N = 2 lo.x dy + (2 Y - 2 lo.y) dx, so that x(Y) = N / (2 dy).
fn crossingNumerator(edge: vec4f, doubledHeight: i32, numerator: ptr<function, array<u32, 6>>) {
  let lowX = i32(edge.x);
  let lowY = i32(edge.y);
  let deltaX = i32(edge.z) - lowX;
  let deltaY = i32(edge.w) - lowY;
  var factorA: array<u32, 6>;
  var factorB: array<u32, 6>;
  var factorC: array<u32, 6>;
  var factorD: array<u32, 6>;
  var termOne: array<u32, 6>;
  var termTwo: array<u32, 6>;
  wideSetInteger(&factorA, 2 * lowX);
  wideSetInteger(&factorB, deltaY);
  wideSetInteger(&factorC, doubledHeight - 2 * lowY);
  wideSetInteger(&factorD, deltaX);
  wideMultiply(&factorA, &factorB, &termOne);
  wideMultiply(&factorC, &factorD, &termTwo);
  wideAdd(&termOne, &termTwo, numerator);
}

fn compareCrossings(first: vec4f, second: vec4f, doubledHeight: i32, offsetSign: i32) -> i32 {
  var numeratorOne: array<u32, 6>;
  var numeratorTwo: array<u32, 6>;
  var heightOne: array<u32, 6>;
  var heightTwo: array<u32, 6>;
  var productOne: array<u32, 6>;
  var productTwo: array<u32, 6>;
  var difference: array<u32, 6>;
  crossingNumerator(first, doubledHeight, &numeratorOne);
  crossingNumerator(second, doubledHeight, &numeratorTwo);
  wideSetInteger(&heightOne, i32(first.w) - i32(first.y));
  wideSetInteger(&heightTwo, i32(second.w) - i32(second.y));
  wideMultiply(&numeratorOne, &heightTwo, &productOne);
  wideMultiply(&numeratorTwo, &heightOne, &productTwo);
  wideSubtract(&productOne, &productTwo, &difference);
  let order = wideSign(&difference);
  if (order != 0) { return order; }
  // Same x at height Y: the order at Y +- e follows the slopes dx/dy.
  var slopeOne: array<u32, 6>;
  var slopeTwo: array<u32, 6>;
  wideSetInteger(&slopeOne, i32(first.z) - i32(first.x));
  wideSetInteger(&slopeTwo, i32(second.z) - i32(second.x));
  wideMultiply(&slopeOne, &heightTwo, &productOne);
  wideMultiply(&slopeTwo, &heightOne, &productTwo);
  wideSubtract(&productOne, &productTwo, &difference);
  return offsetSign * wideSign(&difference);
}
`;function pe(e){return`
const INTERNAL_NODE_COUNT: u32 = ${e.internalNodeCount}u;
const COUNT_ROW: u32 = ${e.countRow}u;
const NO_GROUP: u32 = 0xffffffffu;
const FAR_X: f32 = 3.0e38;
${m}
${h}
${_}
fn readGroup(group: u32) -> vec4f {
  let base = groupLatticeOffset + group * 4u;
  return vec4f(groupLattice[base], groupLattice[base + 1u], groupLattice[base + 2u], groupLattice[base + 3u]);
}
// Whether the box of 'node' can hold an edge that the ray from (rayX, rayY) toward +x meets before
// an edge whose largest x is 'limitX'. All values are exact lattice numbers.
fn nodeMeetsRay(node: u32, rayX: f32, rayY: f32, limitX: f32) -> bool {
  let component = node * 2u;
  let minimum = vec2f(nodeMinima[nodeMinimaOffset + component], nodeMinima[nodeMinimaOffset + component + 1u]);
  let maximum = vec2f(nodeMaxima[nodeMaximaOffset + component], nodeMaxima[nodeMaximaOffset + component + 1u]);
  return minimum.y <= rayY && rayY <= maximum.y && rayX <= maximum.x && minimum.x <= limitX;
}`}function v(e){return`var parent = NO_GROUP;
  let groupCount = counts[countsOffset + COUNT_ROW];
  if (index < groupCount) {
    let edge = readGroup(index);
    let probe = ExactProbe(0.5 * (edge.xy + edge.zw), edge.xy, edge.zw, -1);
    let rayHeight = probe.point.y;
    let doubledHeight = i32(2.0 * rayHeight);
    // The probe sits at Y + o with o infinitesimal: negative exactly when a vertex at height Y
    // counts as above the probe.
    let offsetSign = select(1, -1, exactProbeAbove(probe, probe.point));
    ${e.samePolygon?`let polygon = groupPolygons[groupPolygonsOffset + index];`:``}
    var bestEdge = vec4f(0.0);
    var node = 0u;
    loop {
      let limitX = select(max(bestEdge.x, bestEdge.z), FAR_X, parent == NO_GROUP);
      if (nodeMeetsRay(node, probe.point.x, rayHeight, limitX)) {
        if (node < INTERNAL_NODE_COUNT) {
          node = node * 2u + 1u;
          continue;
        }
        let other = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
        if (other < groupCount${e.samePolygon?` && groupPolygons[groupPolygonsOffset + other] == polygon`:``}) {
          let candidate = readGroup(other);
          if (exactProbeCrossing(probe, candidate.xy, candidate.zw).x == 1) {
            // Both edges are crossed: keep the one meeting the ray at the smaller x.
            if (parent == NO_GROUP || compareCrossings(candidate, bestEdge, doubledHeight, offsetSign) < 0) {
              parent = other;
              bestEdge = candidate;
            }
          }
        }
      }
      // Climb while the node is a right child, then step to the right sibling.
      loop {
        if (node == 0u || (node & 1u) == 1u) { break; }
        node = (node - 1u) / 2u;
      }
      if (node == 0u) { break; }
      node = node + 1u;
    }
  }
  parents[parentsOffset + index] = parent;`}function me(e){return`fn insideWinding(winding: i32) -> bool { return ${e===`even-odd`?`(winding & 1) != 0`:e===`positive`?`winding > 0`:e===`negative`?`winding < 0`:`winding != 0`}; }`}function he(t,n){let{id:r,operation:i,componentCount:a}=n,o=n.parents.length,s={parents:e(t,`${r}-parents-alternate`,`uint32`,o),values:e(t,`${r}-values-alternate`,`uint32`,o*a)},c={parents:n.parents,values:n.values},l=s,u=[];for(let e=0;e<n.rounds;e++)u.push(f(t,{id:`${r}-round-${e}`,operation:i,variant:`pointer-jump`,bindings:[{name:`parentsIn`,view:c.parents,type:`u32`,access:`read`},{name:`valuesIn`,view:c.values,type:`i32`,access:`read`},{name:`parentsOut`,view:l.parents,type:`u32`,access:`read_write`},{name:`valuesOut`,view:l.values,type:`i32`,access:`read_write`}],invocationCount:o,declarations:`const COMPONENT_COUNT: u32 = ${a}u;
const NO_GROUP: u32 = 0xffffffffu;`,body:`let parent = parentsIn[parentsInOffset + index];
  var grandparent = NO_GROUP;
  if (parent != NO_GROUP) { grandparent = parentsIn[parentsInOffset + parent]; }
  for (var component = 0u; component < COMPONENT_COUNT; component++) {
    var value = valuesIn[valuesInOffset + index * COMPONENT_COUNT + component];
    if (parent != NO_GROUP) {
      value = value + valuesIn[valuesInOffset + parent * COMPONENT_COUNT + component];
    }
    valuesOut[valuesOutOffset + index * COMPONENT_COUNT + component] = value;
  }
  parentsOut[parentsOutOffset + index] = grandparent;`})),[c,l]=[l,c];return{nodes:u,values:c.values}}function ge(e){return Math.ceil(Math.log2(Math.max(e,2)))+1}var y=`GPUPolygonOverlay`,b=2**-126,x=2**127,S=23,C=`
fn readLattice(row: u32) -> vec4f {
  let base = sortedLatticeOffset + row * 4u;
  return vec4f(sortedLattice[base], sortedLattice[base + 1u], sortedLattice[base + 2u], sortedLattice[base + 3u]);
}`,_e=`let parent = parents[parentsOffset + index];
  for (var component = 0u; component < COMPONENTS; component++) {
    var value = 0u;
    if (parent != NO_GROUP) { value = delta[deltaOffset + parent * COMPONENTS + component]; }
    values[valuesOffset + index * COMPONENTS + component] = value;
  }`,ve=[`even-odd`,`positive`,`negative`,`nonzero`];function w(e,n,i){if(i.kind!==`polygons`)throw Error(`${e} ${n}.kind must be polygons`);r(i.positions,[`float32x2`],`${e} ${n}.positions`);for(let[r,a]of[[`featureOffsets`,i.featureOffsets],[`polygonOffsets`,i.polygonOffsets],[`ringOffsets`,i.ringOffsets]])if(t(a,`${e} ${n}.${r}`),a.length<2)throw Error(`${e} ${n}.${r} must contain at least two entries`);if(i.sourceIds&&t(i.sourceIds,`${e} ${n}.sourceIds`),i.featureOffsets.length-1>=2147483648)throw Error(`${e} ${n} has too many features for packed overlay provenance`)}function T(e,n,r){for(let[i,a]of Object.entries(r))if(a&&(t(a,`${e} ${n}.${i}`),a.length<1))throw Error(`${e} ${n}.${i} must contain one row`)}var E=class{id;props;spatialContext;fillRule;constructor(e){if(this.id=e.id??`polygon-overlay`,this.spatialContext=p(this.id,e.spatialContext),e.rightSpatialContext!==void 0&&!e.right)throw Error(`${this.id} rightSpatialContext needs right`);le(this.id,`right`,e.rightSpatialContext,this.spatialContext),this.props=e,s(this.id,`vertexTolerance`,e.vertexTolerance,0,x),s(this.id,`vertexTolerance converted to float32`,Math.fround(e.vertexTolerance),b,x),this.fillRule=e.fillRule??`even-odd`;let{id:n}=this,{output:i}=e;if(w(n,`left`,e.left),e.operation===`dissolve`){if(e.right)throw Error(`${n} dissolve accepts only the left operand`)}else if(!e.right)throw Error(`${n} ${e.operation} requires the right operand`);if(e.right&&w(n,`right`,e.right),e.fillRule&&!ve.includes(e.fillRule))throw Error(`${n} fillRule must be even-odd, positive, negative or nonzero`);if(!(e.vertexTolerance>0)||!Number.isFinite(e.vertexTolerance))throw Error(`${n} vertexTolerance must be finite and positive`);for(let[t,r]of Object.entries(e.capacity))o(n,`capacity.${t}`,r);w(n,`output.geometry`,i.geometry),a(n,i.geometry.sourceIds===void 0,`output.geometry.sourceIds is not written: an overlay polygon has no single source row`);let c=i.geometry.ringOffsets.length-1;if(i.geometry.polygonOffsets.length!==c+1||i.geometry.featureOffsets.length!==c+1)throw Error(`${n} output polygon offsets must share one ring capacity`);g(n,i);let l=i.boundary;r(l.endpoints,[`float32x4`],`${n} output.boundary.endpoints`);for(let[e,r]of[[`operandIds`,l.operandIds],[`sourceFeatureIds`,l.sourceFeatureIds]])if(t(r,`${n} output.boundary.${e}`),r.length!==l.endpoints.length)throw Error(`${n} output.boundary.${e} length must equal endpoints length`);T(n,`output.status`,i.status),T(n,`output.boundary.status`,l.status)}getCommandNodes(t){let{id:r,props:a}=this,{left:o,right:s,output:l}=a;te(t,r,Object.fromEntries([o.positions,o.featureOffsets,o.polygonOffsets,o.ringOffsets,s?.positions,s?.featureOffsets,s?.polygonOffsets,s?.ringOffsets,...de(l)].map((e,t)=>[`binding.${t}`,e])));let u=o.ringOffsets.length-1+(s?s.ringOffsets.length-1:0);ue(t,r,{expandedVertexCount:o.positions.length+(s?.positions.length??0)+u,inputRingCount:u,intersections:a.capacity.intersections,nodedSegments:a.capacity.nodedSegments,boundaryCapacity:l.boundary.endpoints.length,outputRingCapacity:l.geometry.ringOffsets.length-1,outputVertexCapacity:l.geometry.positions.length,leafCapacity:a.leafCapacity}),ee(r,t,[o.positions,o.featureOffsets,o.polygonOffsets,o.ringOffsets,s?.positions,s?.featureOffsets,s?.polygonOffsets,s?.ringOffsets,l.geometry.positions,l.geometry.featureOffsets,l.geometry.polygonOffsets,l.geometry.ringOffsets,l.boundary.endpoints,l.boundary.operandIds,l.boundary.sourceFeatureIds,...Object.values(l.boundary.status),...Object.values(l.status)]);let d=a.left.ringOffsets.length-1+(a.right?a.right.ringOffsets.length-1:0);i(r,`left + right positions rows + rings`,a.left.positions.length+(a.right?.positions.length??0)+d,oe(t.device)),i(r,`left + right positions rows + rings + capacity.intersections`,a.left.positions.length+(a.right?.positions.length??0)+d+a.capacity.intersections,ie(t.device,0));let le=s?.positions.length??0,p=o.ringOffsets.length-1,m=s?s.ringOffsets.length-1:0,h=o.positions.length+p,g=h+le+m,_=p+m,pe=e(t,`${r}-combined-positions`,`float32x2`,g),v=e(t,`${r}-combined-offsets`,`uint32`,_+1),b=e(t,`${r}-combined-sources`,`uint32`,_),x=[],ve=s?[{name:`rightPositions`,view:s.positions,type:`f32`,access:`read`}]:[],w=f(t,{id:`${r}-combine-positions`,operation:y,variant:`combine-positions`,bindings:[{name:`leftPositions`,view:o.positions,type:`f32`,access:`read`},...ve,{name:`leftRingOffsets`,view:o.ringOffsets,type:`u32`,access:`read`},...s?[{name:`rightRingOffsets`,view:s.ringOffsets,type:`u32`,access:`read`}]:[],{name:`combinedOffsets`,view:v,type:`u32`,access:`read`},{name:`positions`,view:pe,type:`f32`,access:`read_write`}],invocationCount:g,declarations:`const LEFT_RING_COUNT: u32 = ${p}u;
const RIGHT_RING_COUNT: u32 = ${m}u;
const RING_COUNT: u32 = ${_}u;
fn getLeftVertex(row: u32) -> vec2f {
  return vec2f(leftPositions[leftPositionsOffset + row * 2u], leftPositions[leftPositionsOffset + row * 2u + 1u]);
}
${s?`fn getRightVertex(row: u32) -> vec2f {
  return vec2f(rightPositions[rightPositionsOffset + row * 2u], rightPositions[rightPositionsOffset + row * 2u + 1u]);
}`:``}
// The first stored vertex of the left operand, else of the right operand, else the origin (both
// operands are empty and the noder sees no segment).
fn getPaddingVertex() -> vec2f {
  let leftBegin = leftRingOffsets[leftRingOffsetsOffset];
  if (leftRingOffsets[leftRingOffsetsOffset + LEFT_RING_COUNT] > leftBegin) { return getLeftVertex(leftBegin); }
  ${s?`let rightBegin = rightRingOffsets[rightRingOffsetsOffset];
  if (rightRingOffsets[rightRingOffsetsOffset + RIGHT_RING_COUNT] > rightBegin) { return getRightVertex(rightBegin); }`:``}
  return vec2f(0.0);
}`,body:`var combinedRing = 0u;
  var high = RING_COUNT;
  while (combinedRing + 1u < high) {
    let middle = (combinedRing + high) / 2u;
    if (combinedOffsets[combinedOffsetsOffset + middle] <= index) { combinedRing = middle; } else { high = middle; }
  }
  let fromLeft = combinedRing < LEFT_RING_COUNT;
  let ring = select(combinedRing - LEFT_RING_COUNT, combinedRing, fromLeft);
  let local = index - combinedOffsets[combinedOffsetsOffset + combinedRing];
  var point = vec2f(0.0);
  var sourceBegin = 0u;
  var sourceEnd = 0u;
  if (fromLeft) {
    sourceBegin = leftRingOffsets[leftRingOffsetsOffset + ring];
    sourceEnd = leftRingOffsets[leftRingOffsetsOffset + ring + 1u];
  }${s?` else {
    sourceBegin = rightRingOffsets[rightRingOffsetsOffset + ring];
    sourceEnd = rightRingOffsets[rightRingOffsetsOffset + ring + 1u];
  }`:``}
  if (sourceEnd > sourceBegin) {
    // Ring vertices, then the ring's first vertex for the closing slot and any row after it.
    let sourceRow = select(sourceBegin + local, sourceBegin, local >= sourceEnd - sourceBegin);
    ${s?`if (fromLeft) { point = getLeftVertex(sourceRow); } else { point = getRightVertex(sourceRow); }`:`point = getLeftVertex(sourceRow);`}
  } else {
    point = getPaddingVertex();
  }
  positions[positionsOffset + index * 2u] = point.x;
  positions[positionsOffset + index * 2u + 1u] = point.y;`}),T=s?[{name:`rightFeatureOffsets`,view:s.featureOffsets,type:`u32`,access:`read`},{name:`rightPolygonOffsets`,view:s.polygonOffsets,type:`u32`,access:`read`},{name:`rightRingOffsets`,view:s.ringOffsets,type:`u32`,access:`read`}]:[];x.push(f(t,{id:`${r}-combine-topology`,operation:y,variant:`combine-topology`,bindings:[{name:`leftFeatureOffsets`,view:o.featureOffsets,type:`u32`,access:`read`},{name:`leftPolygonOffsets`,view:o.polygonOffsets,type:`u32`,access:`read`},{name:`leftRingOffsets`,view:o.ringOffsets,type:`u32`,access:`read`},...T,{name:`offsets`,view:v,type:`u32`,access:`read_write`},{name:`sources`,view:b,type:`u32`,access:`read_write`}],invocationCount:_+1,declarations:`const LEFT_RING_COUNT: u32 = ${p}u;
const LEFT_EXPANDED_VERTEX_COUNT: u32 = ${h}u;
const LEFT_POLYGON_COUNT: u32 = ${o.polygonOffsets.length-1}u;
const RIGHT_POLYGON_COUNT: u32 = ${s?s.polygonOffsets.length-1:0}u;`,body:`if (index <= LEFT_RING_COUNT) {
    offsets[offsetsOffset + index] = leftRingOffsets[leftRingOffsetsOffset + index] + index;
  }${s?` else {
    let rightRing = index - LEFT_RING_COUNT;
    offsets[offsetsOffset + index] = LEFT_EXPANDED_VERTEX_COUNT + rightRingOffsets[rightRingOffsetsOffset + rightRing] + rightRing;
  }`:``}
  if (index >= ${_}u) { return; }
  let fromLeft = index < LEFT_RING_COUNT;
  let ring = select(index - LEFT_RING_COUNT, index, fromLeft);
  var polygon = 0u;
  var high = select(RIGHT_POLYGON_COUNT, LEFT_POLYGON_COUNT, fromLeft);
  while (polygon + 1u < high) {
    let middle = (polygon + high) / 2u;
    var start = 0u;
    if (fromLeft) { start = leftPolygonOffsets[leftPolygonOffsetsOffset + middle]; }
    ${s?`else { start = rightPolygonOffsets[rightPolygonOffsetsOffset + middle]; }`:``}
    if (start <= ring) { polygon = middle; } else { high = middle; }
  }
  var feature = 0u;
  high = select(${s?`${s.featureOffsets.length-1}u`:`0u`}, ${o.featureOffsets.length-1}u, fromLeft);
  while (feature + 1u < high) {
    let middle = (feature + high) / 2u;
    var start = 0u;
    if (fromLeft) { start = leftFeatureOffsets[leftFeatureOffsetsOffset + middle]; }
    ${s?`else { start = rightFeatureOffsets[rightFeatureOffsetsOffset + middle]; }`:``}
    if (start <= polygon) { feature = middle; } else { high = middle; }
  }
  sources[sourcesOffset + index] = feature | select(0x80000000u, 0u, fromLeft);`})),x.push(w);let E=e(t,`${r}-ring-polygons`,`uint32`,_);x.push(f(t,{id:`${r}-ring-polygons`,operation:y,variant:`ring-polygons`,bindings:[{name:`leftPolygonOffsets`,view:o.polygonOffsets,type:`u32`,access:`read`},...s?[{name:`rightPolygonOffsets`,view:s.polygonOffsets,type:`u32`,access:`read`}]:[],{name:`ringPolygons`,view:E,type:`u32`,access:`read_write`}],invocationCount:_,declarations:`const LEFT_RING_COUNT: u32 = ${p}u;
const LEFT_POLYGON_COUNT: u32 = ${o.polygonOffsets.length-1}u;
const RIGHT_POLYGON_COUNT: u32 = ${s?s.polygonOffsets.length-1:0}u;`,body:`let fromLeft = index < LEFT_RING_COUNT;
  let ring = select(index - LEFT_RING_COUNT, index, fromLeft);
  var polygon = 0u;
  var high = select(RIGHT_POLYGON_COUNT, LEFT_POLYGON_COUNT, fromLeft);
  while (polygon + 1u < high) {
    let middle = (polygon + high) / 2u;
    var start = 0u;
    if (fromLeft) { start = leftPolygonOffsets[leftPolygonOffsetsOffset + middle]; }
    ${s?`else { start = rightPolygonOffsets[rightPolygonOffsetsOffset + middle]; }`:``}
    if (start <= ring) { polygon = middle; } else { high = middle; }
  }
  ringPolygons[ringPolygonsOffset + index] = polygon + select(LEFT_POLYGON_COUNT, 0u, fromLeft);`}));let D=this._createNodedPort(t);x.push(...new ae({id:`${r}-noding`,spatialContext:this.spatialContext,lines:{kind:`lines`,positions:pe,lineOffsets:v,sourceIds:b},intersectionCapacity:a.capacity.intersections,gridSize:a.vertexTolerance,output:{...D,status:{...D.status,uncertainCount:l.status.uncertainCount}},spatialSort:a.spatialSort,leafCapacity:a.leafCapacity}).getCommandNodes(t));let O=a.capacity.nodedSegments,k=(n,i,a)=>e(t,`${r}-${n}`,i,a),A=(e,n,i,a,o)=>f(t,{id:`${r}-${e}`,operation:y,variant:e,bindings:n,invocationCount:i,declarations:o,body:a}),j=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read`}),M=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),N=k(`piece-lattice`,`float32x4`,O),ye=k(`piece-sigma`,`uint32`,O),be=k(`piece-identity`,`uint32`,O);x.push(A(`piece-canonical`,[j(`segmentCount`,D.status.count),j(`latticeEndpoints`,D.latticeEndpoints,`f32`),M(`pieceLattice`,N,`f32`),M(`pieceSigma`,ye),M(`identity`,be)],O,`identity[identityOffset + index] = index;
  var lattice = vec4f(-1.0);
  var sigma = 0u;
  if (index < min(segmentCount[segmentCountOffset], CAPACITY)) {
    let base = latticeEndpointsOffset + index * 4u;
    let first = vec2f(latticeEndpoints[base], latticeEndpoints[base + 1u]);
    let second = vec2f(latticeEndpoints[base + 2u], latticeEndpoints[base + 3u]);
    // Lattice values are finite integers >= 0; non-finite or collapsed pieces carry no boundary.
    if (first.x >= 0.0 && first.y >= 0.0 && second.x >= 0.0 && second.y >= 0.0 && any(first != second)) {
      let firstIsLow = first.y < second.y || (first.y == second.y && first.x < second.x);
      lattice = select(vec4f(second, first), vec4f(first, second), firstIsLow);
      // +1 when the piece runs lo -> hi, -1 (as a wrapping u32) otherwise.
      sigma = select(0xffffffffu, 1u, firstIsLow);
    }
  }
  let output = pieceLatticeOffset + index * 4u;
  pieceLattice[output] = lattice.x;
  pieceLattice[output + 1u] = lattice.y;
  pieceLattice[output + 2u] = lattice.z;
  pieceLattice[output + 3u] = lattice.w;
  pieceSigma[pieceSigmaOffset + index] = sigma;`,`const CAPACITY: u32 = ${O}u;`));let xe=k(`key-low-y`,`uint32`,O),Se=k(`key-low-x`,`uint32`,O),Ce=k(`key-high-y`,`uint32`,O),we=k(`key-high-x`,`uint32`,O),P=k(`key-polygon`,`uint32`,O);x.push(A(`piece-keys`,[j(`pieceLattice`,N,`f32`),j(`sourceRingIds`,D.sourceRingIds),j(`ringPolygons`,E),M(`keyLowY`,xe),M(`keyLowX`,Se),M(`keyHighY`,Ce),M(`keyHighX`,we),M(`keyPolygon`,P)],O,`let base = pieceLatticeOffset + index * 4u;
  let lattice = vec4f(pieceLattice[base], pieceLattice[base + 1u], pieceLattice[base + 2u], pieceLattice[base + 3u]);
  var lowY = KEY_SENTINEL;
  var lowX = 0u;
  var highY = 0u;
  var highX = 0u;
  var polygon = 0u;
  if (lattice.x >= 0.0) {
    lowY = u32(lattice.y);
    lowX = u32(lattice.x);
    highY = u32(lattice.w);
    highX = u32(lattice.z);
    let ring = sourceRingIds[sourceRingIdsOffset + index];
    if (ring < RING_COUNT) { polygon = ringPolygons[ringPolygonsOffset + ring]; }
  }
  keyLowY[keyLowYOffset + index] = lowY;
  keyLowX[keyLowXOffset + index] = lowX;
  keyHighY[keyHighYOffset + index] = highY;
  keyHighX[keyHighXOffset + index] = highX;
  keyPolygon[keyPolygonOffset + index] = polygon;`,`const RING_COUNT: u32 = ${_}u;
// Above every lattice value (at most 2^22 + 1), within the 23 sorted bits.
const KEY_SENTINEL: u32 = 0x7fffffu;`));let Te=o.polygonOffsets.length-1+(s?s.polygonOffsets.length-1:0),F=re(t,{id:`${r}-piece-sort`,operation:y,indices:be,keys:[{view:xe,keyBits:S},{view:Se,keyBits:S},{view:Ce,keyBits:S},{view:we,keyBits:S},{view:P,keyBits:ne(Math.ceil(Math.log2(Te+1)),1,31)}]});x.push(...F.nodes);let I=k(`sorted-lattice`,`float32x4`,O),Ee=k(`sorted-sigma`,`uint32`,O),L=k(`sorted-polygon`,`uint32`,O);x.push(A(`sort-gather`,[j(`sortedIndices`,F.sortedIndices),j(`pieceLattice`,N,`f32`),j(`pieceSigma`,ye),j(`keyPolygon`,P),M(`sortedLattice`,I,`f32`),M(`sortedSigma`,Ee),M(`sortedPolygon`,L)],O,`let piece = min(sortedIndices[sortedIndicesOffset + index], CAPACITY - 1u);
  for (var component = 0u; component < 4u; component++) {
    sortedLattice[sortedLatticeOffset + index * 4u + component] = pieceLattice[pieceLatticeOffset + piece * 4u + component];
  }
  sortedSigma[sortedSigmaOffset + index] = pieceSigma[pieceSigmaOffset + piece];
  sortedPolygon[sortedPolygonOffset + index] = keyPolygon[keyPolygonOffset + piece];`,`const CAPACITY: u32 = ${O}u;`));let R=k(`flags-one`,`uint32`,O),z=k(`flags-two`,`uint32`,O);x.push(A(`group-flags`,[j(`sortedLattice`,I,`f32`),j(`sortedPolygon`,L),M(`flagsOne`,R),M(`flagsTwo`,z)],O,`let current = readLattice(index);
  var one = 0u;
  var two = 0u;
  if (current.x >= 0.0) {
    if (index == 0u) {
      one = 1u;
      two = 1u;
    } else {
      let differs = any(readLattice(index - 1u) != current);
      two = select(0u, 1u, differs);
      one = select(0u, 1u, differs || sortedPolygon[sortedPolygonOffset + index - 1u] != sortedPolygon[sortedPolygonOffset + index]);
    }
  }
  flagsOne[flagsOneOffset + index] = one;
  flagsTwo[flagsTwoOffset + index] = two;`,C));let B=k(`ordinals-one`,`uint32`,O),De=k(`ordinals-two`,`uint32`,O),Oe=k(`sigma-prefix`,`uint32`,O);for(let[e,i,a]of[[`ordinals-one`,R,B],[`ordinals-two`,z,De],[`sigma-prefix`,Ee,Oe]])x.push(...new n({id:`${r}-${e}`,input:i,output:a,mode:`inclusive`}).getCommandNodes(t));let V=k(`group-counts`,`uint32`,2);x.push(A(`group-counts`,[j(`ordinalsOne`,B),j(`ordinalsTwo`,De),M(`counts`,V)],1,`counts[countsOffset] = ordinalsOne[ordinalsOneOffset + ${O-1}u];
  counts[countsOffset + 1u] = ordinalsTwo[ordinalsTwoOffset + ${O-1}u];`));let H=k(`heads-one`,`uint32`,O),U=k(`heads-two`,`uint32`,O),ke=k(`first-one`,`uint32`,O);x.push(A(`group-heads`,[j(`flagsOne`,R),j(`flagsTwo`,z),j(`ordinalsOne`,B),j(`ordinalsTwo`,De),M(`headsOne`,H),M(`headsTwo`,U),M(`firstOne`,ke)],O,`if (flagsOne[flagsOneOffset + index] != 0u) {
    headsOne[headsOneOffset + ordinalsOne[ordinalsOneOffset + index] - 1u] = index;
  }
  if (flagsTwo[flagsTwoOffset + index] != 0u) {
    let group = ordinalsTwo[ordinalsTwoOffset + index] - 1u;
    headsTwo[headsTwoOffset + group] = index;
    firstOne[firstOneOffset + group] = ordinalsOne[ordinalsOneOffset + index] - 1u;
  }`));let Ae=(e,n)=>{let i=k(`group-lattice-${e}`,`float32x4`,O),a=k(`group-minima-${e}`,`float32x2`,O),o=k(`group-maxima-${e}`,`float32x2`,O);x.push(A(`group-lattice-${e}`,[j(`counts`,V),j(`heads`,n),j(`sortedLattice`,I,`f32`),M(`groupLattice`,i,`f32`),M(`minima`,a,`f32`),M(`maxima`,o,`f32`)],O,`var edge = vec4f(0.0);
  var minimum = vec2f(FAR, FAR);
  var maximum = vec2f(-FAR, -FAR);
  if (index < counts[countsOffset + ${e-1}u]) {
    edge = readLattice(heads[headsOffset + index]);
    minimum = vec2f(min(edge.x, edge.z), edge.y);
    maximum = vec2f(max(edge.x, edge.z), edge.w);
  }
  for (var component = 0u; component < 4u; component++) {
    groupLattice[groupLatticeOffset + index * 4u + component] = edge[component];
  }
  minima[minimaOffset + index * 2u] = minimum.x;
  minima[minimaOffset + index * 2u + 1u] = minimum.y;
  maxima[maximaOffset + index * 2u] = maximum.x;
  maxima[maximaOffset + index * 2u + 1u] = maximum.y;`,`const FAR: f32 = 3.0e38;
${C}`));let{bvh:s,nodes:l}=se(t,{id:`${r}-group-bvh-${e}`,operation:y,minima:a,maxima:o,leafCapacity:Math.max(2,c(O)),spatialSort:!1});return x.push(...l),{groupLattice:i,bvh:s}},je=Ae(1,H),W=k(`group-polygons`,`uint32`,O),G=k(`group-delta`,`uint32`,O);x.push(A(`group-winding-steps`,[j(`counts`,V),j(`heads`,H),j(`sigmaPrefix`,Oe),j(`sortedPolygon`,L),M(`groupPolygons`,W),M(`groupDelta`,G)],O,`var polygon = 0u;
  var delta = 0u;
  let groupCount = counts[countsOffset];
  if (index < groupCount) {
    let start = heads[headsOffset + index];
    var end = ${O}u;
    if (index + 1u < groupCount) { end = heads[headsOffset + index + 1u]; }
    // Sum of the direction signs over the run, from the prefix sums (wrapping, so exact as i32).
    var before = 0u;
    if (start > 0u) { before = sigmaPrefix[sigmaPrefixOffset + start - 1u]; }
    delta = sigmaPrefix[sigmaPrefixOffset + end - 1u] - before;
    polygon = sortedPolygon[sortedPolygonOffset + start];
  }
  groupPolygons[groupPolygonsOffset + index] = polygon;
  groupDelta[groupDeltaOffset + index] = delta;`));let K=k(`parents-one`,`uint32`,O),Me=k(`values-one`,`uint32`,O);x.push(this._createRayShootNode(t,1,je,V,W,K),A(`jump-init-1`,[j(`parents`,K),j(`delta`,G),M(`values`,Me)],O,_e.replaceAll(`COMPONENTS`,`1u`),`const NO_GROUP: u32 = 0xffffffffu;`));let Ne=he(t,{id:`${r}-jump-1`,operation:y,parents:K,values:Me,componentCount:1,rounds:ge(O)});x.push(...Ne.nodes);let Pe=k(`coverage-a`,`uint32`,O),Fe=k(`coverage-b`,`uint32`,O),Ie=o.polygonOffsets.length-1;x.push(A(`group-coverage`,[j(`counts`,V),j(`groupPolygons`,W),j(`groupDelta`,G,`i32`),j(`windings`,Ne.values,`i32`),M(`coverageA`,Pe,`i32`),M(`coverageB`,Fe,`i32`)],O,`var changeA = 0;
  var changeB = 0;
  if (index < counts[countsOffset]) {
    let right = windings[windingsOffset + index];
    let left = right + groupDelta[groupDeltaOffset + index];
    let change = select(0, 1, insideWinding(left)) - select(0, 1, insideWinding(right));
    if (groupPolygons[groupPolygonsOffset + index] < LEFT_POLYGON_COUNT) { changeA = change; } else { changeB = change; }
  }
  coverageA[coverageAOffset + index] = changeA;
  coverageB[coverageBOffset + index] = changeB;`,`const LEFT_POLYGON_COUNT: u32 = ${Ie}u;
${me(this.fillRule)}`));let Le=k(`coverage-prefix-a`,`uint32`,O),Re=k(`coverage-prefix-b`,`uint32`,O);for(let[e,i,a]of[[`coverage-prefix-a`,Pe,Le],[`coverage-prefix-b`,Fe,Re]])x.push(...new n({id:`${r}-${e}`,input:i,output:a,mode:`inclusive`}).getCommandNodes(t));let q=k(`group-coverage-delta`,`uint32`,2*O);x.push(A(`group-coverage-delta`,[j(`counts`,V),j(`firstOne`,ke),j(`prefixA`,Le),j(`prefixB`,Re),M(`coverageDelta`,q)],O,`var deltaA = 0u;
  var deltaB = 0u;
  if (index < counts[countsOffset + 1u]) {
    let first = firstOne[firstOneOffset + index];
    var end = counts[countsOffset];
    if (index + 1u < counts[countsOffset + 1u]) { end = firstOne[firstOneOffset + index + 1u]; }
    var beforeA = 0u;
    var beforeB = 0u;
    if (first > 0u) {
      beforeA = prefixA[prefixAOffset + first - 1u];
      beforeB = prefixB[prefixBOffset + first - 1u];
    }
    deltaA = prefixA[prefixAOffset + end - 1u] - beforeA;
    deltaB = prefixB[prefixBOffset + end - 1u] - beforeB;
  }
  coverageDelta[coverageDeltaOffset + index * 2u] = deltaA;
  coverageDelta[coverageDeltaOffset + index * 2u + 1u] = deltaB;`));let ze=Ae(2,U),J=k(`parents-two`,`uint32`,O),Be=k(`values-two`,`uint32`,2*O);x.push(this._createRayShootNode(t,2,ze,V,void 0,J),A(`jump-init-2`,[j(`parents`,J),j(`delta`,q),M(`values`,Be)],O,_e.replaceAll(`COMPONENTS`,`2u`),`const NO_GROUP: u32 = 0xffffffffu;`));let Ve=he(t,{id:`${r}-jump-2`,operation:y,parents:J,values:Be,componentCount:2,rounds:ge(O)});x.push(...Ve.nodes);let Y=k(`selected`,`uint32`,O),He=k(`orientations`,`uint32`,O),X=k(`selection-scan`,`uint32`,O);x.push(A(`classify-boundary`,[j(`counts`,V),j(`rightCoverage`,Ve.values,`i32`),j(`coverageDelta`,q,`i32`),M(`selected`,Y),M(`orientations`,He)],O,`var keep = false;
  var reverse = false;
  if (index < counts[countsOffset + 1u]) {
    let rightA = rightCoverage[rightCoverageOffset + index * 2u];
    let rightB = rightCoverage[rightCoverageOffset + index * 2u + 1u];
    let leftA = rightA + coverageDelta[coverageDeltaOffset + index * 2u];
    let leftB = rightB + coverageDelta[coverageDeltaOffset + index * 2u + 1u];
    let leftResult = ${this._getOperationExpression(`(leftA > 0)`,`(leftB > 0)`)};
    let rightResult = ${this._getOperationExpression(`(rightA > 0)`,`(rightB > 0)`)};
    keep = leftResult != rightResult;
    // The edge runs lo -> hi with its left side on the left: flip it when the result is on the right.
    reverse = keep && rightResult;
  }
  selected[selectedOffset + index] = select(0u, 1u, keep);
  orientations[orientationsOffset + index] = select(0u, 1u, reverse);`),...new n({id:`${r}-selection-scan`,input:Y,output:X,mode:`inclusive`}).getCommandNodes(t));let Z=l.boundary,Ue=k(`boundary-groups`,`uint32`,Z.endpoints.length),We=k(`lattice-boundary`,`float32x4`,Z.endpoints.length);x.push(A(`emit-boundary`,[j(`selected`,Y),j(`scan`,X),j(`orientations`,He),j(`heads`,U),j(`sortedLattice`,I,`f32`),j(`lattice`,D.lattice,`f32`),M(`endpoints`,Z.endpoints,`f32`),M(`latticeEndpoints`,We,`f32`)],O,`if (selected[selectedOffset + index] == 0u) { return; }
  let outputRow = scan[scanOffset + index] - 1u;
  if (outputRow >= BOUNDARY_CAPACITY) { return; }
  let edge = readLattice(heads[headsOffset + index]);
  let reverse = orientations[orientationsOffset + index] != 0u;
  let start = select(edge.xy, edge.zw, reverse);
  let end = select(edge.zw, edge.xy, reverse);
  let origin = vec2f(lattice[latticeOffset], lattice[latticeOffset + 1u]);
  let cell = lattice[latticeOffset + 2u];
  let outputBase = endpointsOffset + outputRow * 4u;
  let latticeBase = latticeEndpointsOffset + outputRow * 4u;
  latticeEndpoints[latticeBase] = start.x;
  latticeEndpoints[latticeBase + 1u] = start.y;
  latticeEndpoints[latticeBase + 2u] = end.x;
  latticeEndpoints[latticeBase + 3u] = end.y;
  // World position of a lattice point: origin + k * cell (cell is a power of two).
  endpoints[outputBase] = origin.x + start.x * cell;
  endpoints[outputBase + 1u] = origin.y + start.y * cell;
  endpoints[outputBase + 2u] = origin.x + end.x * cell;
  endpoints[outputBase + 3u] = origin.y + end.y * cell;`,`const BOUNDARY_CAPACITY: u32 = ${Z.endpoints.length}u;
${C}`),A(`boundary-groups`,[M(`groups`,Ue)],Z.endpoints.length,`groups[groupsOffset + index] = 0u;`),A(`emit-boundary-provenance`,[j(`selected`,Y),j(`scan`,X),j(`heads`,U),j(`sortedIndices`,F.sortedIndices),j(`packedSources`,D.sourceFeatureIds),M(`operandIds`,Z.operandIds),M(`sourceFeatureIds`,Z.sourceFeatureIds)],O,`if (selected[selectedOffset + index] == 0u) { return; }
  let outputRow = scan[scanOffset + index] - 1u;
  if (outputRow >= BOUNDARY_CAPACITY) { return; }
  // The first piece of the group in sort order owns its provenance.
  let piece = sortedIndices[sortedIndicesOffset + heads[headsOffset + index]];
  let packed = packedSources[packedSourcesOffset + piece];
  operandIds[operandIdsOffset + outputRow] = packed >> 31u;
  sourceFeatureIds[sourceFeatureIdsOffset + outputRow] = packed & 0x7fffffffu;`,`const BOUNDARY_CAPACITY: u32 = ${Z.endpoints.length}u;`),A(`boundary-status`,[j(`scan`,X),j(`nodingOverflow`,D.status.overflow),j(`nodingCandidateOverflow`,D.status.candidateOverflow),M(`count`,Z.status.count),M(`requiredCount`,Z.status.requiredCount),M(`overflow`,Z.status.overflow),M(`candidateOverflow`,Z.status.candidateOverflow)],1,`let required = scan[scanOffset + NODED_CAPACITY - 1u];
  let candidateIncomplete = nodingCandidateOverflow[nodingCandidateOverflowOffset];
  requiredCount[requiredCountOffset] = required;
  count[countOffset] = min(required, BOUNDARY_CAPACITY);
  overflow[overflowOffset] = select(0u, 1u, required > BOUNDARY_CAPACITY || nodingOverflow[nodingOverflowOffset] != 0u);
  candidateOverflow[candidateOverflowOffset] = candidateIncomplete;`,`const BOUNDARY_CAPACITY: u32 = ${Z.endpoints.length}u;
const NODED_CAPACITY: u32 = ${O}u;`));let Q=l.geometry.ringOffsets.length-1,Ge=k(`ring-positions`,`float32x2`,l.geometry.positions.length),Ke=k(`ring-offsets`,`uint32`,Q+1),qe=k(`ring-count`,`uint32`,1),Je=k(`ring-required-count`,`uint32`,1),Ye=k(`ring-overflow`,`uint32`,1),$=[[`ringCount`,`ringCountInput`,l.status.ringCount,qe],[`requiredRingCount`,`ringRequiredCountInput`,l.status.requiredRingCount,Je]],Xe=$.flatMap(([e,t,n,r])=>n?[j(t,r),M(e,n)]:[]),Ze=$.flatMap(([e,t,n])=>n?[`${e}[${e}Offset] = ${t}[${t}Offset];`]:[]).join(`
  `),Qe=$.some(([,,e])=>e)?`-${$.map(([,,e])=>+!!e).join(``)}`:``,$e=l.featureSources?{segmentRings:k(`source-segment-rings`,`uint32`,l.boundary.endpoints.length),ringShells:k(`source-ring-shells`,`uint32`,Q),ringAreas:k(`source-ring-areas`,`float32`,Q)}:void 0;return x.push(...new ce({id:`${r}-assemble`,spatialContext:this.spatialContext,endpoints:We,count:Z.status.count,groups:Ue,vertexTolerance:.25,interiorSide:`left`,normalizeWinding:!0,cancelOpposingSegments:!1,splitTouchingRings:!0,output:{positions:Ge,ringOffsets:Ke,status:{count:qe,requiredCount:Je,overflow:Ye},polygons:l.geometry,...$e}}).getCommandNodes(t)),l.featureSources&&$e&&x.push(...fe(t,{id:r,...$e,ringCount:qe,boundaryCount:Z.status.count,sourceRows:Z.sourceFeatureIds,operandIds:Z.operandIds,output:l.featureSources})),x.push(A(`lattice-to-world`,[j(`lattice`,D.lattice,`f32`),j(`ringOffsets`,l.geometry.ringOffsets),M(`positions`,l.geometry.positions,`f32`)],Math.max(1,l.geometry.positions.length),`if (index >= ringOffsets[ringOffsetsOffset + RING_CAPACITY]) { return; }
  let cell = lattice[latticeOffset + 2u];
  positions[positionsOffset + index * 2u] = lattice[latticeOffset] + positions[positionsOffset + index * 2u] * cell;
  positions[positionsOffset + index * 2u + 1u] = lattice[latticeOffset + 1u] + positions[positionsOffset + index * 2u + 1u] * cell;`,`const RING_CAPACITY: u32 = ${Q}u;`),A(`publish`,[j(`featureOffsets`,l.geometry.featureOffsets),j(`boundaryOverflow`,Z.status.overflow),j(`candidateOverflowInput`,Z.status.candidateOverflow),j(`ringOverflow`,Ye),M(`count`,l.status.count),M(`overflow`,l.status.overflow),M(`candidateOverflow`,l.status.candidateOverflow)],1,`let featureCount = featureOffsets[featureOffsetsOffset + RING_CAPACITY];
  let incomplete = boundaryOverflow[boundaryOverflowOffset] | ringOverflow[ringOverflowOffset];
  count[countOffset] = featureCount;
  overflow[overflowOffset] = incomplete;
  candidateOverflow[candidateOverflowOffset] = candidateOverflowInput[candidateOverflowInputOffset];`,`const RING_CAPACITY: u32 = ${Q}u;`)),Xe.length>0&&x.push(A(`ring-status${Qe}`,Xe,1,Ze)),x}_createRayShootNode(e,t,n,r,i,a){let{bvh:o}=n;return f(e,{id:`${this.id}-ray-shoot-${t}`,operation:y,variant:`ray-shoot-${t}`,bindings:[{name:`counts`,view:r,type:`u32`,access:`read`},{name:`groupLattice`,view:n.groupLattice,type:`f32`,access:`read`},...i?[{name:`groupPolygons`,view:i,type:`u32`,access:`read`}]:[],{name:`nodeMinima`,view:o.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:o.nodeMaxima,type:`f32`,access:`read`},{name:`leafIds`,view:o.leafIds,type:`u32`,access:`read`},{name:`parents`,view:a,type:`u32`,access:`read_write`}],invocationCount:a.length,declarations:pe({internalNodeCount:o.internalNodeCount,countRow:t-1,samePolygon:t===1}),body:v({samePolygon:t===1})})}_createNodedPort(t){let{id:n,props:r}=this,i=r.capacity.nodedSegments,a=r=>e(t,`${n}-${r}`,`uint32`,1);return{endpoints:e(t,`${n}-noded-endpoints`,`float32x4`,i),latticeEndpoints:e(t,`${n}-noded-lattice-endpoints`,`float32x4`,i),lattice:e(t,`${n}-lattice`,`float32`,4),sourceFeatureIds:e(t,`${n}-noded-features`,`uint32`,i),sourceRingIds:e(t,`${n}-noded-rings`,`uint32`,i),sourceEdgeIds:e(t,`${n}-noded-edges`,`uint32`,i),sourceStartParameters:e(t,`${n}-noded-start`,`float32`,i),sourceEndParameters:e(t,`${n}-noded-end`,`float32`,i),status:{count:a(`noded-count`),requiredCount:r.output.status.nodedSegmentRequiredCount??a(`noded-required-count`),overflow:a(`noded-overflow`),candidateOverflow:a(`noded-candidate-overflow`)}}}_getOperationExpression(e,t){switch(this.props.operation){case`intersection`:return`${e} && ${t}`;case`union`:return`${e} || ${t}`;case`difference`:return`${e} && !${t}`;case`symmetric-difference`:return`${e} != ${t}`;case`dissolve`:return e}}};export{ue as i,g as n,de as r,E as t};