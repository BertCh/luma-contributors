import{s as e,v as t}from"./spatial-context-DwZEd4EZ.js";import{r as n}from"./kernel-C3tfQOn4.js";import{d as r}from"./wgsl-common-DfyXiYB0.js";import{o as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as a,S as o,T as s,l as c,w as l,x as u,y as d}from"./gpu-contributor-utils-BIwLQON5.js";import{t as f}from"./segment-bvh-CDgckfAv.js";import{a as p,i as m,s as h,t as g}from"./segment-table-BeVxjLqW.js";import{A as _}from"./pointer-overlay-Qh8qflcA.js";import{t as v}from"./define-run-B-brkCe4.js";import{c as y,r as b}from"./text-DYSgVV-n.js";import{n as x,s as S,v as C,y as w}from"./polygon-set-CHja50mr.js";import{s as T}from"./shape-segments-DMcyF4V-.js";import{l as E,r as D,t as O}from"./pointer-support-C2rPRVFO.js";import{so as k}from"./entry-preview-D70TdSbK.js";var A=Object.freeze({...e,measure:`distance`}),j=`GPUMinimumClearance`,M=1024,N=64,P=class{id;props;spatialContext;featureCount;vertexCount;leafCapacity;spatialSort;constructor(e){this.id=e.id??`minimum-clearance`,this.props=e,this.spatialContext=t(this.id,e.spatialContext,A);let{id:n}=this,{geometry:r}=e;a(r.positions,[`float32x2`],`${n} geometry.positions`);let i=r.kind===`lines`?[[`lineOffsets`,r.lineOffsets]]:[[`featureOffsets`,r.featureOffsets],[`polygonOffsets`,r.polygonOffsets],[`ringOffsets`,r.ringOffsets]];for(let[e,t]of i)if(a(t,[`uint32`],`${n} geometry.${e}`),t.length<1)throw Error(`${n} geometry.${e} requires a terminal entry`);if(this.vertexCount=r.positions.length,this.vertexCount<1)throw Error(`${n} geometry must contain at least one vertex`);if(this.featureCount=(r.kind===`lines`?r.lineOffsets.length:r.featureOffsets.length)-1,this.featureCount<1)throw Error(`${n} geometry must contain at least one feature`);let{clearances:o,lines:u,vertexIds:d}=e.output;if(!o&&!u&&!d)throw Error(`${n} requires at least one of clearances, lines or vertexIds`);for(let[e,t,r]of[[`clearances`,o,`float32`],[`lines`,u,`float32x4`],[`vertexIds`,d,`uint32`]])if(t&&(a(t,[r],`${n} output.${e}`),t.length!==this.featureCount))throw Error(`${n} output.${e} length must equal the feature count`);e.output.status?.candidateOverflow&&(a(e.output.status.candidateOverflow,[`uint32`],`${n} output.status.candidateOverflow`),s(n,e.output.status.candidateOverflow.length>=1,`output.status.candidateOverflow must hold one row`)),this.spatialSort=e.spatialSort??this.vertexCount>=M,this.leafCapacity=e.leafCapacity??c(this.vertexCount),l(n,`leafCapacity`,this.leafCapacity)}getInputViews(){let{geometry:e}=this.props;return e.kind===`lines`?[e.positions,e.lineOffsets]:[e.positions,e.featureOffsets,e.polygonOffsets,e.ringOffsets]}getOutputViews(){let{clearances:e,lines:t,vertexIds:n,status:r}=this.props.output;return[e,t,n,r?.candidateOverflow]}getCommandNodes(e){let{id:t,props:a,featureCount:s,vertexCount:c,leafCapacity:l,spatialSort:_}=this,{geometry:v}=a,{clearances:y,lines:b,vertexIds:x}=a.output;u(t,e,[...this.getInputViews(),...this.getOutputViews()]),o(t,`geometry.positions rows`,c,m(e.device)),d(t,this.getOutputViews(),this.getInputViews());let S=[],C=n(e,`${t}-table`,`uint32`,c*8),w=n(e,`${t}-bounds-minima`,`float32x2`,c),T=n(e,`${t}-bounds-maxima`,`float32x2`,c);S.push(g(e,{id:`${t}-table`,operation:j,geometry:v,table:C,minima:w,maxima:T}));let{bvh:E,nodes:D}=f(e,{id:t,operation:j,minima:w,maxima:T,leafCapacity:l,spatialSort:_});S.push(...D),a.output.status?.candidateOverflow&&S.push(i(e,{id:`${t}-candidate-overflow`,operation:j,variant:`candidate-overflow`,bindings:[{name:`bvhOverflow`,view:E.overflow,type:`u32`,access:`read`},{name:`candidateOverflow`,view:a.output.status.candidateOverflow,type:`u32`,access:`read_write`}],invocationCount:1,body:`candidateOverflow[candidateOverflowOffset] = select(0u, 1u, bvhOverflow[bvhOverflowOffset] != 0u);`}));let O=n(e,`${t}-vertex-features`,`uint32`,c),k=v.kind===`polygons`,A=k?[{name:`ringOffsets`,view:v.ringOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:v.polygonOffsets,type:`u32`,access:`read`},{name:`featureOffsets`,view:v.featureOffsets,type:`u32`,access:`read`}]:[{name:`lineOffsets`,view:v.lineOffsets,type:`u32`,access:`read`}];S.push(i(e,{id:`${t}-vertex-features`,operation:j,variant:`vertex-features`,bindings:[...A,{name:`vertexFeatures`,view:O,type:`u32`,access:`read_write`}],invocationCount:c,declarations:`const SEGMENT_NONE: u32 = ${h}u;
const VERTEX_COUNT: u32 = ${c}u;
${k?`const RING_COUNT: u32 = ${v.ringOffsets.length-1}u;
const POLYGON_COUNT: u32 = ${v.polygonOffsets.length-1}u;
const FEATURE_COUNT: u32 = ${v.featureOffsets.length-1}u;
fn readOffset(kind: u32, row: u32) -> u32 {
  if (kind == 0u) { return ringOffsets[ringOffsetsOffset + row]; }
  if (kind == 1u) { return polygonOffsets[polygonOffsetsOffset + row]; }
  return featureOffsets[featureOffsetsOffset + row];
}`:`const RING_COUNT: u32 = ${v.lineOffsets.length-1}u;
fn readOffset(kind: u32, row: u32) -> u32 { return lineOffsets[lineOffsetsOffset + row]; }`}
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
}`,body:`var feature = SEGMENT_NONE;
  let ring = lastAtMost(0u, RING_COUNT, index);
  if (ring != SEGMENT_NONE) {
    let ringStart = readOffset(0u, ring);
    let ringEnd = min(readOffset(0u, ring + 1u), VERTEX_COUNT);
    if (index < ringEnd) {
      ${k?`if (ringEnd - ringStart >= 3u && ring < readOffset(1u, POLYGON_COUNT)) {
        let polygon = lastAtMost(1u, POLYGON_COUNT, ring);
        if (polygon != SEGMENT_NONE && polygon < readOffset(2u, FEATURE_COUNT)) {
          feature = lastAtMost(2u, FEATURE_COUNT, polygon);
        }
      }`:`feature = ring;`}
    }
  }
  vertexFeatures[vertexFeaturesOffset + index] = feature;`}));let M=n(e,`${t}-vertex-nearest`,`float32x4`,c);S.push(i(e,{id:`${t}-probe`,operation:j,variant:`probe`,bindings:[{name:`positions`,view:v.positions,type:`f32`,access:`read`},{name:`rightTable`,view:C,type:`u32`,access:`read`},{name:`nodeMinima`,view:E.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:E.nodeMaxima,type:`f32`,access:`read`},{name:`leafIds`,view:E.leafIds,type:`u32`,access:`read`},{name:`vertexFeatures`,view:O,type:`u32`,access:`read`},{name:`nearest`,view:M,type:`f32`,access:`read_write`}],invocationCount:c,declarations:`const SEGMENT_NONE: u32 = ${h}u;
${r}
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
const VERTEX_COUNT: u32 = ${c}u;
const INTERNAL_NODE_COUNT: u32 = ${E.internalNodeCount}u;
${p(`right`)}
struct Candidate {
  distanceSquared: f32,
  point: vec2f,
}
// Squared distance from point to the segment (a, b) with a != b, and the closest point.
fn segmentCandidate(point: vec2f, a: vec2f, b: vec2f) -> Candidate {
  let edge = b - a;
  let offset = point - a;
  let lengthSquared = dot(edge, edge);
  let along = dot(offset, edge);
  if (along <= 0.0) { return Candidate(dot(offset, offset), a); }
  if (along >= lengthSquared) {
    let toEnd = point - b;
    return Candidate(dot(toEnd, toEnd), b);
  }
  let side = edge.x * offset.y - edge.y * offset.x;
  return Candidate(side * side / lengthSquared, a + edge * (along / lengthSquared));
}
// Candidate of one segment row for a vertex: the segment itself, or its other endpoint when the
// vertex is one of its endpoints.
fn rowCandidate(point: vec2f, row: u32) -> Candidate {
  let start = rightStart(row);
  let end = rightEnd(row);
  if (point.x == start.x && point.y == start.y) {
    let toEnd = point - end;
    return Candidate(dot(toEnd, toEnd), end);
  }
  if (point.x == end.x && point.y == end.y) {
    let toStart = point - start;
    return Candidate(dot(toStart, toStart), start);
  }
  return segmentCandidate(point, start, end);
}
fn boxDistanceSquared(node: u32, point: vec2f) -> f32 {
  let component = node * 2u;
  let minimum = vec2f(nodeMinima[nodeMinimaOffset + component], nodeMinima[nodeMinimaOffset + component + 1u]);
  let maximum = vec2f(nodeMaxima[nodeMaximaOffset + component], nodeMaxima[nodeMaximaOffset + component + 1u]);
  let gap = max(max(minimum - point, point - maximum), vec2f(0.0));
  return dot(gap, gap);
}`,body:`let feature = vertexFeatures[vertexFeaturesOffset + index];
  var bestSquared = FLOAT32_MAXIMUM;
  var bestPoint = vec2f(0.0);
  var bestRow = SEGMENT_NONE;
  let point = vec2f(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u]);
  if (feature != SEGMENT_NONE && !isNanBits(point.x) && !isNanBits(point.y)) {
    // Seed the bound with this vertex's own segment and the previous slot (usually the other
    // incident segment). Any valid segment of the same feature is a legitimate candidate, so the
    // result is unchanged; the bound just starts at about the shortest incident edge, which
    // prunes almost the whole tree (including other features' overlapping nodes) up front.
    for (var seed = 0u; seed < 2u; seed++) {
      if (seed == 1u && index == 0u) { break; }
      let seedRow = index - seed;
      if (!rightValid(seedRow) || rightFeature(seedRow) != feature) { continue; }
      let seedCandidate = rowCandidate(point, seedRow);
      if (seedCandidate.distanceSquared < bestSquared ||
          (seedCandidate.distanceSquared == bestSquared && seedRow < bestRow)) {
        bestSquared = seedCandidate.distanceSquared;
        bestPoint = seedCandidate.point;
        bestRow = seedRow;
      }
    }
    // Stack entries carry their box distance, tested when pushed and again when popped.
    var stack: array<u32, 64>;
    var stackLower: array<f32, 64>;
    var depth = 1u;
    stack[0] = 0u;
    stackLower[0] = boxDistanceSquared(0u, point);
    loop {
      if (depth == 0u) { break; }
      depth = depth - 1u;
      let node = stack[depth];
      if (stackLower[depth] > bestSquared) { continue; }
      if (node < INTERNAL_NODE_COUNT) {
        let first = node * 2u + 1u;
        let second = first + 1u;
        let firstLower = boxDistanceSquared(first, point);
        let secondLower = boxDistanceSquared(second, point);
        // Push the farther child first so the nearer one is popped next.
        if (firstLower <= secondLower) {
          if (secondLower <= bestSquared) { stack[depth] = second; stackLower[depth] = secondLower; depth = depth + 1u; }
          if (firstLower <= bestSquared) { stack[depth] = first; stackLower[depth] = firstLower; depth = depth + 1u; }
        } else {
          if (firstLower <= bestSquared) { stack[depth] = first; stackLower[depth] = firstLower; depth = depth + 1u; }
          if (secondLower <= bestSquared) { stack[depth] = second; stackLower[depth] = secondLower; depth = depth + 1u; }
        }
        continue;
      }
      let row = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
      if (row >= VERTEX_COUNT || !rightValid(row) || rightFeature(row) != feature) { continue; }
      let candidate = rowCandidate(point, row);
      if (candidate.distanceSquared < bestSquared ||
          (candidate.distanceSquared == bestSquared && row < bestRow)) {
        bestSquared = candidate.distanceSquared;
        bestPoint = candidate.point;
        bestRow = row;
      }
    }
  }
  // A runtime zero keeps the bit pattern of infinity from being rejected as a constant.
  let infinity = bitcast<f32>(0x7f800000u | (index & 0u));
  let found = bestRow != SEGMENT_NONE;
  let base = nearestOffset + index * 4u;
  nearest[base] = select(infinity, sqrt(bestSquared), found);
  nearest[base + 1u] = bestPoint.x;
  nearest[base + 2u] = bestPoint.y;
  nearest[base + 3u] = 0.0;`}));let P=n(e,`${t}-feature-state`,`uint32`,s*2);S.push(i(e,{id:`${t}-feature-minimum`,operation:j,variant:`feature-minimum-${v.kind}`,bindings:[...A,{name:`vertexFeatures`,view:O,type:`u32`,access:`read`},{name:`nearest`,view:M,type:`f32`,access:`read`},{name:`featureState`,view:P,type:`u32`,access:`read_write`}],workgroupSize:N,invocationCount:s*N,guardIndex:!1,declarations:`const SEGMENT_NONE: u32 = ${h}u;
const INFINITY_BITS: u32 = 0x7f800000u;
const VERTEX_COUNT: u32 = ${c}u;
const LANE_COUNT: u32 = ${N}u;
${k?`const RING_COUNT: u32 = ${v.ringOffsets.length-1}u;
const POLYGON_COUNT: u32 = ${v.polygonOffsets.length-1}u;
// First vertex row of a feature row (the feature count gives the end), clamped like the offsets.
fn getFeatureStart(row: u32) -> u32 {
  let polygon = min(featureOffsets[featureOffsetsOffset + row], POLYGON_COUNT);
  let ring = min(polygonOffsets[polygonOffsetsOffset + polygon], RING_COUNT);
  return min(ringOffsets[ringOffsetsOffset + ring], VERTEX_COUNT);
}`:`fn getFeatureStart(row: u32) -> u32 {
  return min(lineOffsets[lineOffsetsOffset + row], VERTEX_COUNT);
}`}
var<workgroup> laneBits: array<u32, ${N}>;
var<workgroup> laneRows: array<u32, ${N}>;
// Lexicographic (distance bits, vertex row) order: non-negative f32 values order like their bits.
fn isBefore(bits: u32, row: u32, otherBits: u32, otherRow: u32) -> bool {
  return bits < otherBits || (bits == otherBits && row < otherRow);
}`,body:`let feature = index / LANE_COUNT;
  let isInRange = index < INVOCATION_COUNT;
  var bestBits = 0xffffffffu;
  var bestRow = 0xffffffffu;
  if (isInRange) {
    let begin = getFeatureStart(feature);
    let end = getFeatureStart(feature + 1u);
    for (var vertex = begin + localInvocationIndex; vertex < end; vertex += LANE_COUNT) {
      let bits = bitcast<u32>(nearest[nearestOffset + vertex * 4u]);
      if (vertexFeatures[vertexFeaturesOffset + vertex] == feature && bits < INFINITY_BITS &&
          isBefore(bits, vertex, bestBits, bestRow)) {
        bestBits = bits;
        bestRow = vertex;
      }
    }
  }
  laneBits[localInvocationIndex] = bestBits;
  laneRows[localInvocationIndex] = bestRow;
  workgroupBarrier();
  for (var stride = LANE_COUNT / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      let otherBits = laneBits[localInvocationIndex + stride];
      let otherRow = laneRows[localInvocationIndex + stride];
      if (isBefore(otherBits, otherRow, laneBits[localInvocationIndex], laneRows[localInvocationIndex])) {
        laneBits[localInvocationIndex] = otherBits;
        laneRows[localInvocationIndex] = otherRow;
      }
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    featureState[featureStateOffset + feature * 2u] = laneBits[0];
    featureState[featureStateOffset + feature * 2u + 1u] = laneRows[0];
  }`}));let F=[{name:`positions`,view:v.positions,type:`f32`,access:`read`},{name:`nearest`,view:M,type:`f32`,access:`read`},{name:`featureState`,view:P,type:`u32`,access:`read`}],I=[];return y&&(F.push({name:`clearances`,view:y,type:`f32`,access:`read_write`}),I.push(`clearances[clearancesOffset + index] = select(infinity, nearest[nearestOffset + row * 4u], found);`)),b&&(F.push({name:`lines`,view:b,type:`f32`,access:`read_write`}),I.push(`let lineBase = linesOffset + index * 4u;
  lines[lineBase] = select(notANumber, positions[positionsOffset + row * 2u], found);
  lines[lineBase + 1u] = select(notANumber, positions[positionsOffset + row * 2u + 1u], found);
  lines[lineBase + 2u] = select(notANumber, nearest[nearestOffset + row * 4u + 1u], found);
  lines[lineBase + 3u] = select(notANumber, nearest[nearestOffset + row * 4u + 2u], found);`)),x&&(F.push({name:`vertexIds`,view:x,type:`u32`,access:`read_write`}),I.push(`vertexIds[vertexIdsOffset + index] = rowWord;`)),S.push(i(e,{id:`${t}-write`,operation:j,variant:`write`,bindings:F,invocationCount:s,declarations:`const SEGMENT_NONE: u32 = 0xffffffffu;`,body:`let rowWord = featureState[featureStateOffset + index * 2u + 1u];
  let found = rowWord != SEGMENT_NONE;
  let row = select(0u, rowWord, found);
  // Runtime zeros keep the bit patterns of infinity and NaN from being rejected as constants.
  let infinity = bitcast<f32>(0x7f800000u | (index & 0u));
  let notANumber = bitcast<f32>(0x7fc00000u | (index & 0u));
  ${I.join(`
  `)}`})),S}},F={rows:2,columns:3},I=v({parameters:k,caption:`The tightest spot of each polygon, joined by its witness line.`,createInputs:({seed:e,world:t})=>{let n=S(_(e),t,{...F,irregularity:.5});return{seed:e,world:t,set:n,...D(n,1)}},build(e,t){let n=C(e,t.set),r=e.allocate(`clearances`,`float32`,t.set.count),i=e.allocate(`lines`,`float32x4`,t.set.count);e.graph.add(new P({geometry:{kind:`polygons`,positions:n.positions.view,featureOffsets:n.featureOffsets.view,polygonOffsets:n.polygonOffsets.view,ringOffsets:n.ringOffsets.view},output:{clearances:r.view,lines:i.view}}));let a=e.allocate(`witness-ends`,`float32x2`,t.set.count*2);return T(e,{id:`witness-ends-kernel`,invocationCount:t.set.count,bindings:[{name:`lines`,buffer:i,access:`read`},{name:`ends`,buffer:a,access:`read_write`}],body:`ends[endsOffset + index * 4u] = lines[linesOffset + index * 4u];
  ends[endsOffset + index * 4u + 1u] = lines[linesOffset + index * 4u + 1u];
  ends[endsOffset + index * 4u + 2u] = lines[linesOffset + index * 4u + 2u];
  ends[endsOffset + index * 4u + 3u] = lines[linesOffset + index * 4u + 3u];`}),{outputs:{...n,clearances:r,lines:i,witnessEnds:a},update:e=>{let r=S(_(t.seed),t.world,{...F,irregularity:e.irregularity});x(r,t.drag),t.set=r,w(n,r)}}},interaction:{kind:`drag`,hint:`squeeze a vertex`,rest:({inputs:e,toUnit:t})=>t(E(e)),resolve(e,{inputs:t,outputs:n,peek:r}){let i=O(t,e,0,.3),a=t.drag?.feature??0,o=r(n.clearances),s=r(n.lines);return{rerun:!0,glyphs:[...i.glyphs,...s?[{shape:`line`,from:[s[a*4],s[a*4+1]],to:[s[a*4+2],s[a*4+3]]}]:[]],readout:`clearance ${o?o[a].toFixed(3):`...`}`}}},async afterRun({outputs:e}){let t=await e.clearances.buffer.readAsync();return new Float32Array(t.slice().buffer)},draw:({inputs:e,outputs:t,theme:n,extra:r})=>{let i=r?Array.from(r,(t,n)=>({text:t.toFixed(3),at:[e.set.centers[n*2],e.set.centers[n*2+1]],role:`value`,anchor:`top`,offset:[0,-6]})):[];return[{layer:`polygons`,positions:t.fillVertices,indices:t.fillIndices,owners:t.fillOwners,values:t.clearances,triangleCount:e.set.fillTriangleCount,range:[0,.08]},{layer:`lines`,context:!0,segments:t.outline,count:e.set.outlineCount},{layer:`lines`,segments:t.lines,color:[...n.ink,1],radius:3.2},{layer:`dots`,positions:t.witnessEnds,color:[...n.ink,1],radius:5.5},{layer:`text`,...y({labels:i},b({corner:`bottom-left`,entries:[{label:`witness line: its length is the clearance`,swatch:[...n.ink]}]}))}]}});export{I as default};