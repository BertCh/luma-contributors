import{t as e}from"./ports-xmdKgL07.js";import{r as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-scan-B4NNTfF9.js";import{o as r}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as i,S as a,f as o,u as s,x as ee}from"./gpu-contributor-utils-BIwLQON5.js";import{t as te}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{i as c}from"./segment-table-BeVxjLqW.js";import{t as ne}from"./gpu-segment-intersection-CGBfhMHr.js";import{t as re}from"./gpu-point-in-polygon-join-B0GGIWim.js";import{t as ie}from"./spatial-join-types-BVxJRclf.js";import{u as ae}from"./wgsl-common-DfyXiYB0.js";import{_ as l,v as u}from"./spatial-context-DwZEd4EZ.js";import{t as d}from"./define-run-B-brkCe4.js";import{o as f}from"./pointer-overlay-Qh8qflcA.js";import{c as p,l as m,o as h,r as g}from"./lines-kx5qHdti.js";import{a as _,c as v,i as y,l as b}from"./polygons-D0hzhPvj.js";import{t as x}from"./triangulation-Nv4HznK-.js";import{Jo as S}from"./entry-preview-D70TdSbK.js";var C=`GPULineClipByPolygon`,w=8,T=8,E=1,oe=2,D=4;function se(e){return s(e,32)}var O=class{id;props;spatialContext;vertexCount;lineCount;pieceCapacity;vertexCapacity;keepInside;constructor(t){this.id=t.id??`line-clip-by-polygon`,this.spatialContext=u(this.id,t.spatialContext),l(this.id,`polygons`,t.polygonsSpatialContext,this.spatialContext),t={...t,lines:e(this.id,`lines`,t.lines)},this.props=t;let{id:n}=this,{lines:r,polygons:a}=t,{pieces:s}=t.output;if(r.kind!==`lines`)throw Error(`${n} lines must be linestring geometry`);if(a.kind!==`polygons`)throw Error(`${n} polygons must be polygon geometry`);i(r.positions,[`float32x2`],`${n} lines.positions`),i(r.lineOffsets,[`uint32`],`${n} lines.lineOffsets`),i(a.positions,[`float32x2`],`${n} polygons.positions`);for(let[e,t]of[[`featureOffsets`,a.featureOffsets],[`polygonOffsets`,a.polygonOffsets],[`ringOffsets`,a.ringOffsets]])if(i(t,[`uint32`],`${n} polygons.${e}`),t.length<1)throw Error(`${n} polygons.${e} requires a terminal entry`);if(r.lineOffsets.length<2)throw Error(`${n} lines.lineOffsets requires at least one line and a terminal entry`);if(r.positions.length<1)throw Error(`${n} lines.positions must hold at least one vertex`);if(a.positions.length<1)throw Error(`${n} polygons.positions must hold at least one vertex`);if(t.mode!==void 0&&t.mode!==`inside`&&t.mode!==`outside`)throw Error(`${n} mode must be 'inside' or 'outside'`);if(o(n,`intersectionCapacity`,t.intersectionCapacity),o(n,`candidateCapacity`,t.candidateCapacity),this.keepInside=t.mode!==`outside`,this.vertexCount=r.positions.length,this.lineCount=r.lineOffsets.length-1,i(s.sourceIds,[`uint32`],`${n} output.pieces.sourceIds`),i(s.geometry.lineOffsets,[`uint32`],`${n} output.pieces.geometry.lineOffsets`),i(s.geometry.positions,[`float32x2`],`${n} output.pieces.geometry.positions`),this.pieceCapacity=s.sourceIds.length,this.vertexCapacity=s.geometry.positions.length,this.pieceCapacity<1||s.geometry.lineOffsets.length!==this.pieceCapacity+1)throw Error(`${n} output.pieces lineOffsets needs sourceIds.length + 1 rows`);if(this.vertexCapacity<1)throw Error(`${n} output.pieces.geometry.positions must be non-empty`);for(let[e,r]of[[`count`,s.status.count],[`vertexCount`,s.status.vertexCount],[`overflow`,s.status.overflow],[`candidateOverflow`,s.status.candidateOverflow],[`requiredCount`,s.status.requiredCount],[`requiredVertexCount`,s.status.requiredVertexCount],[`uncertainCount`,t.output.status?.uncertainCount]])if(r&&(i(r,[`uint32`],`${n} output.${e}`),r.length<1))throw Error(`${n} output.${e} needs one row`)}getCommandNodes(e){let{id:i,props:o,vertexCount:s,lineCount:l,pieceCapacity:u,vertexCapacity:d,keepInside:f}=this,{lines:p,polygons:m}=o,{pieces:h}=o.output;ee(i,e,[p.positions,p.lineOffsets,m.positions,m.featureOffsets,m.polygonOffsets,m.ringOffsets,h.sourceIds,h.geometry.lineOffsets,h.geometry.positions,h.status.count,h.status.vertexCount,h.status.overflow,h.status.candidateOverflow,h.status.requiredCount,h.status.requiredVertexCount,o.output.status?.uncertainCount]);for(let[t,n]of[[`lines.positions rows`,o.lines.positions],[`polygons.positions rows`,o.polygons.positions]])a(i,t,n.length,c(e.device));a(i,`lines.positions rows + 2 x intersectionCapacity`,o.lines.positions.length+2*o.intersectionCapacity,se(e.device));let g=[],_=o.intersectionCapacity,v=_*2,y=s+v,b=(n,r,a)=>t(e,`${i}-${n}`,r,a),x=b(`pair-left`,`uint32`,_),S=b(`pair-right`,`uint32`,_),O=b(`pair-count`,`uint32`,1),k=b(`pair-overflow`,`uint32`,1),A=b(`kinds`,`uint32`,_),j=b(`points`,`float32x2`,_),ce=b(`end-points`,`float32x2`,_),le=b(`segment-uncertain`,`uint32`,1);g.push(...new ne({id:`${i}-intersection`,left:p,right:m,spatialSort:o.spatialSort,leafCapacity:o.leafCapacity,output:{pairs:{leftRows:x,rightRows:S,status:{count:O,overflow:k}},status:{uncertainCount:le},kinds:A,points:j,endPoints:ce}}).getCommandNodes(e));let M=`
const VERTEX_COUNT: u32 = ${s}u;
const LINE_COUNT: u32 = ${l}u;
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
${ae}
// Largest line whose first vertex is at most vertex.
fn lineOfVertex(vertex: u32) -> u32 {
  var low = 0u;
  var high = LINE_COUNT - 1u;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle] <= vertex) { low = middle; } else { high = middle - 1u; }
  }
  return low;
}
// Coordinate of p along the segment direction d, on the dominant axis; ascending along d.
fn keyOf(p: vec2f, d: vec2f) -> f32 {
  if (abs(d.x) >= abs(d.y)) { return p.x * select(-1.0, 1.0, d.x > 0.0); }
  return p.y * select(-1.0, 1.0, d.y > 0.0);
}
fn otherOf(p: vec2f, d: vec2f) -> f32 {
  if (abs(d.x) >= abs(d.y)) { return p.y * select(-1.0, 1.0, d.y >= 0.0); }
  return p.x * select(-1.0, 1.0, d.x >= 0.0);
}`,N=b(`segment-table`,`uint32`,s*4);g.push(r(e,{id:`${i}-segment-ranges`,operation:C,variant:`segment-ranges`,bindings:[{name:`positions`,view:p.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:p.lineOffsets,type:`u32`,access:`read`},{name:`pairLeft`,view:x,type:`u32`,access:`read`},{name:`pairCount`,view:O,type:`u32`,access:`read`},{name:`segmentTable`,view:N,type:`u32`,access:`read_write`}],invocationCount:s,declarations:`${M}
fn lowerBound(value: u32) -> u32 {
  var low = 0u;
  var high = pairCount[pairCountOffset];
  while (low < high) {
    let middle = (low + high) / 2u;
    if (pairLeft[pairLeftOffset + middle] < value) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}`,body:`let line = lineOfVertex(index);
  let lineStart = lineOffsets[lineOffsetsOffset + line];
  let lineEnd = lineOffsets[lineOffsetsOffset + line + 1u];
  let a = vertexAt(index);
  var valid = index >= lineStart && index + 1u < lineEnd && index + 1u < VERTEX_COUNT;
  var b = a;
  if (valid) { b = vertexAt(index + 1u); }
  valid = valid && isFiniteValue(a.x) && isFiniteValue(a.y) && isFiniteValue(b.x) && isFiniteValue(b.y) &&
    (a.x != b.x || a.y != b.y);
  var first = lowerBound(index);
  var last = lowerBound(index + 1u);
  if (!valid) { last = first; }
  let row = segmentTableOffset + index * 4u;
  segmentTable[row] = first;
  segmentTable[row + 1u] = last;
  segmentTable[row + 2u] = select(0u, 1u, valid);
  segmentTable[row + 3u] = line;`}));let P=b(`event-table`,`uint32`,v*w),F=b(`event-segments`,`uint32`,v),I=b(`event-order-keys`,`uint32`,v),L=b(`event-other-keys`,`uint32`,v),R=b(`event-indices`,`uint32`,v);g.push(r(e,{id:`${i}-events`,operation:C,variant:`events`,bindings:[{name:`positions`,view:p.positions,type:`f32`,access:`read`},{name:`pairLeft`,view:x,type:`u32`,access:`read`},{name:`pairCount`,view:O,type:`u32`,access:`read`},{name:`kinds`,view:A,type:`u32`,access:`read`},{name:`points`,view:j,type:`f32`,access:`read`},{name:`endPoints`,view:ce,type:`f32`,access:`read`},{name:`eventTable`,view:P,type:`u32`,access:`read_write`}],invocationCount:v,declarations:`const VERTEX_COUNT: u32 = ${s}u;
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
${M.slice(M.indexOf(`// Coordinate of p`))}`,body:`let pair = index / 2u;
  let which = index % 2u;
  var flags = 0u;
  var point = vec2f(0.0);
  var key = 0.0;
  var other = 0.0;
  var kind = 0u;
  if (pair < pairCount[pairCountOffset]) { kind = kinds[kindsOffset + pair]; }
  if (kind >= 1u && kind <= 4u) {
    let segment = pairLeft[pairLeftOffset + pair];
    let a = vertexAt(segment);
    let b = vertexAt(segment + 1u);
    let direction = b - a;
    if (which == 0u) {
      point = vec2f(points[pointsOffset + pair * 2u], points[pointsOffset + pair * 2u + 1u]);
    } else {
      point = vec2f(endPoints[endPointsOffset + pair * 2u], endPoints[endPointsOffset + pair * 2u + 1u]);
    }
    key = keyOf(point, direction);
    other = otherOf(point, direction);
    let interior = !(point.x == a.x && point.y == a.y) && !(point.x == b.x && point.y == b.y);
    if (interior && (which == 0u || kind == 4u)) { flags = 1u; }
    if (kind == 4u) { flags = flags | 2u; }
  }
  let row = eventTableOffset + index * ${w}u;
  eventTable[row] = flags;
  eventTable[row + 1u] = 0u;
  eventTable[row + 2u] = 0u;
  eventTable[row + 3u] = bitcast<u32>(point.x);
  eventTable[row + 4u] = bitcast<u32>(point.y);
  eventTable[row + 5u] = bitcast<u32>(key);
  eventTable[row + 6u] = bitcast<u32>(other);
  eventTable[row + 7u] = 0u;`}),r(e,{id:`${i}-event-sort-keys`,operation:C,variant:`event-sort-keys`,bindings:[{name:`pairLeft`,view:x,type:`u32`,access:`read`},{name:`eventTable`,view:P,type:`u32`,access:`read`},{name:`eventSegments`,view:F,type:`u32`,access:`read_write`},{name:`eventOrderKeys`,view:I,type:`u32`,access:`read_write`},{name:`eventOtherKeys`,view:L,type:`u32`,access:`read_write`},{name:`eventIndices`,view:R,type:`u32`,access:`read_write`}],invocationCount:v,declarations:`const VERTEX_COUNT: u32 = ${s}u;`,body:`let row = eventTableOffset + index * ${w}u;
  let valid = (eventTable[row] & 1u) != 0u;
  let keyBits = eventTable[row + 5u];
  let otherBits = eventTable[row + 6u];
  eventSegments[eventSegmentsOffset + index] = select(VERTEX_COUNT, pairLeft[pairLeftOffset + index / 2u], valid);
  eventOrderKeys[eventOrderKeysOffset + index] = select(keyBits ^ 0xffffffffu, keyBits ^ 0x80000000u, (keyBits & 0x80000000u) == 0u);
  eventOtherKeys[eventOtherKeysOffset + index] = select(otherBits ^ 0xffffffffu, otherBits ^ 0x80000000u, (otherBits & 0x80000000u) == 0u);
  eventIndices[eventIndicesOffset + index] = index;`}));let z=`
fn eventPointBits(event: u32) -> vec2u {
  let row = eventTableOffset + event * ${w}u;
  return vec2u(eventTable[row + 3u], eventTable[row + 4u]);
}
fn eventPoint(event: u32) -> vec2f { return bitcast<vec2f>(eventPointBits(event)); }
fn eventKey(event: u32) -> f32 { return bitcast<f32>(eventTable[eventTableOffset + event * ${w}u + 5u]); }
`,B=te(e,{id:`${i}-event-order`,operation:C,indices:R,keys:[{view:F,keyBits:Math.max(1,Math.ceil(Math.log2(s+1)))},{view:I},{view:L}]});g.push(...B.nodes);let V=B.sortedPrimaryKeys,H=B.sortedIndices,U=b(`event-unique-flags`,`uint32`,v),W=b(`event-unique-ranks`,`uint32`,v);g.push(r(e,{id:`${i}-event-unique`,operation:C,variant:`event-unique`,bindings:[{name:`sortedSegments`,view:V,type:`u32`,access:`read`},{name:`sortedEvents`,view:H,type:`u32`,access:`read`},{name:`eventTable`,view:P,type:`u32`,access:`read`},{name:`uniqueFlags`,view:U,type:`u32`,access:`read_write`}],invocationCount:v,declarations:`const VERTEX_COUNT: u32 = ${s}u;
${z}`,body:`let segment = sortedSegments[sortedSegmentsOffset + index];
  var unique = 1u;
  if (segment >= VERTEX_COUNT) { unique = 0u; }
  if (unique != 0u && index > 0u && sortedSegments[sortedSegmentsOffset + index - 1u] == segment) {
    let event = sortedEvents[sortedEventsOffset + index];
    let previous = sortedEvents[sortedEventsOffset + index - 1u];
    if (all(eventPointBits(event) == eventPointBits(previous))) { unique = 0u; }
  }
  uniqueFlags[uniqueFlagsOffset + index] = unique;`}),...new n({id:`${i}-event-unique-scan`,input:U,output:W,mode:`exclusive`}).getCommandNodes(e),r(e,{id:`${i}-event-rank`,operation:C,variant:`event-rank`,bindings:[{name:`sortedSegments`,view:V,type:`u32`,access:`read`},{name:`sortedEvents`,view:H,type:`u32`,access:`read`},{name:`uniqueFlags`,view:U,type:`u32`,access:`read`},{name:`uniqueRanks`,view:W,type:`u32`,access:`read`},{name:`eventTable`,view:P,type:`u32`,access:`read_write`}],invocationCount:v,declarations:`const EVENT_COUNT: u32 = ${v}u;
fn lowerBoundSegment(value: u32) -> u32 {
  var low = 0u;
  var high = EVENT_COUNT;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (sortedSegments[sortedSegmentsOffset + middle] < value) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}`,body:`if (uniqueFlags[uniqueFlagsOffset + index] == 0u) { return; }
  let event = sortedEvents[sortedEventsOffset + index];
  let segment = sortedSegments[sortedSegmentsOffset + index];
  let first = lowerBoundSegment(segment);
  let rank = uniqueRanks[uniqueRanksOffset + index] - uniqueRanks[uniqueRanksOffset + first];
  eventTable[eventTableOffset + event * ${w}u + 1u] = 1u;
  eventTable[eventTableOffset + event * ${w}u + 2u] = rank;
  // Word 7 of the globally compacted unique-event row names the original event. Segment bases
  // turn local ranks into these O(1) lookups without reusing pair-table ranges.
  let globalRank = uniqueRanks[uniqueRanksOffset + index];
  eventTable[eventTableOffset + globalRank * ${w}u + 7u] = event;`}));let G=b(`sub-counts`,`uint32`,s),K=b(`segment-unique-bases`,`uint32`,s),ue=b(`sub-starts`,`uint32`,s);g.push(r(e,{id:`${i}-sub-counts`,operation:C,variant:`sub-counts`,bindings:[{name:`segmentTable`,view:N,type:`u32`,access:`read`},{name:`sortedSegments`,view:V,type:`u32`,access:`read`},{name:`uniqueFlags`,view:U,type:`u32`,access:`read`},{name:`uniqueRanks`,view:W,type:`u32`,access:`read`},{name:`segmentUniqueBases`,view:K,type:`u32`,access:`read_write`},{name:`subCounts`,view:G,type:`u32`,access:`read_write`}],invocationCount:s,declarations:`const EVENT_COUNT: u32 = ${v}u;
fn lowerBoundSegment(value: u32) -> u32 {
  var low = 0u;
  var high = EVENT_COUNT;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (sortedSegments[sortedSegmentsOffset + middle] < value) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}
fn uniquePrefix(position: u32) -> u32 {
  if (position < EVENT_COUNT) { return uniqueRanks[uniqueRanksOffset + position]; }
  return uniqueRanks[uniqueRanksOffset + EVENT_COUNT - 1u] + uniqueFlags[uniqueFlagsOffset + EVENT_COUNT - 1u];
}`,body:`let row = segmentTableOffset + index * 4u;
  var count = 0u;
  if (segmentTable[row + 2u] != 0u) {
    let begin = lowerBoundSegment(index);
    let end = lowerBoundSegment(index + 1u);
    segmentUniqueBases[segmentUniqueBasesOffset + index] = uniquePrefix(begin);
    count = 1u + uniquePrefix(end) - uniquePrefix(begin);
  }
  subCounts[subCountsOffset + index] = count;`}),...new n({id:`${i}-sub-scan`,input:G,output:ue,mode:`exclusive`}).getCommandNodes(e));let q=b(`sub-table`,`uint32`,y*T),de=b(`sub-midpoints`,`float32x2`,y);g.push(r(e,{id:`${i}-sub-pieces`,operation:C,variant:`sub-pieces`,bindings:[{name:`positions`,view:p.positions,type:`f32`,access:`read`},{name:`segmentTable`,view:N,type:`u32`,access:`read`},{name:`eventTable`,view:P,type:`u32`,access:`read`},{name:`subStarts`,view:ue,type:`u32`,access:`read`},{name:`subCounts`,view:G,type:`u32`,access:`read`},{name:`segmentUniqueBases`,view:K,type:`u32`,access:`read`},{name:`subTable`,view:q,type:`u32`,access:`read_write`},{name:`subMidpoints`,view:de,type:`f32`,access:`read_write`}],invocationCount:y,declarations:`const VERTEX_COUNT: u32 = ${s}u;
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
${M.slice(M.indexOf(`// Coordinate of p`))}
${z}
fn pointOfRank(base: u32, rank: u32) -> vec2f {
  return eventPoint(eventTable[eventTableOffset + (base + rank) * ${w}u + 7u]);
}`,body:`let total = subStarts[subStartsOffset + VERTEX_COUNT - 1u] + subCounts[subCountsOffset + VERTEX_COUNT - 1u];
  let row = subTableOffset + index * ${T}u;
  if (index >= total) {
    subTable[row + 6u] = 0u;
    subMidpoints[subMidpointsOffset + index * 2u] = 3.0e38;
    subMidpoints[subMidpointsOffset + index * 2u + 1u] = 3.0e38;
    return;
  }
  // Last segment whose first sub-piece is at most index; zero-count segments precede it.
  var low = 0u;
  var high = VERTEX_COUNT - 1u;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (subStarts[subStartsOffset + middle] <= index) { low = middle; } else { high = middle - 1u; }
  }
  let segment = low;
  let local = index - subStarts[subStartsOffset + segment];
  let uniqueCount = subCounts[subCountsOffset + segment] - 1u;
  let first = segmentTable[segmentTableOffset + segment * 4u];
  let last = segmentTable[segmentTableOffset + segment * 4u + 1u];
  let uniqueBase = segmentUniqueBases[segmentUniqueBasesOffset + segment];
  let a = vertexAt(segment);
  let b = vertexAt(segment + 1u);
  let direction = b - a;
  var startPoint = a;
  var endPoint = b;
  if (local > 0u) { startPoint = pointOfRank(uniqueBase, local - 1u); }
  if (local < uniqueCount) { endPoint = pointOfRank(uniqueBase, local); }
  let startKey = keyOf(startPoint, direction);
  let endKey = keyOf(endPoint, direction);
  var flags = ${D}u;
  for (var pair = first; pair < last; pair++) {
    let overlapEvent = pair * 2u;
    if ((eventTable[eventTableOffset + overlapEvent * ${w}u] & 2u) != 0u) {
      let spanLow = min(eventKey(overlapEvent), eventKey(overlapEvent + 1u));
      let spanHigh = max(eventKey(overlapEvent), eventKey(overlapEvent + 1u));
      if (startKey >= spanLow && endKey <= spanHigh) { flags = flags | ${E}u; }
    }
  }
  subTable[row] = bitcast<u32>(startPoint.x);
  subTable[row + 1u] = bitcast<u32>(startPoint.y);
  subTable[row + 2u] = bitcast<u32>(endPoint.x);
  subTable[row + 3u] = bitcast<u32>(endPoint.y);
  subTable[row + 4u] = segmentTable[segmentTableOffset + segment * 4u + 3u];
  subTable[row + 5u] = segment;
  subTable[row + 6u] = flags;
  let midpoint = (startPoint + endPoint) * 0.5;
  subMidpoints[subMidpointsOffset + index * 2u] = midpoint.x;
  subMidpoints[subMidpointsOffset + index * 2u + 1u] = midpoint.y;`}));let fe=b(`sub-features`,`uint32`,y),pe=b(`containment-overflow`,`uint32`,1),me=b(`containment-uncertain`,`uint32`,1);g.push(...new re({id:`${i}-containment`,points:{kind:`points`,positions:de},polygons:{kind:`polygons`,positions:m.positions,featureOffsets:m.featureOffsets,polygonOffsets:m.polygonOffsets,ringOffsets:m.ringOffsets},candidateCapacity:o.candidateCapacity,boundaryRule:`closed`,output:{pointFeatureIds:fe,status:{overflow:pe,uncertainCount:me}}}).getCommandNodes(e));let he=`
const SUB_PIECE_BOUND: u32 = ${y}u;
fn subFlags(piece: u32) -> u32 { return subTable[subTableOffset + piece * ${T}u + 6u]; }
fn isKept(piece: u32) -> bool { return piece < SUB_PIECE_BOUND && (subFlags(piece) & 6u) == 6u; }
// Whether kept sub-piece piece continues the kept sub-piece before it.
fn continuesPrevious(piece: u32) -> bool {
  if (piece == 0u || !isKept(piece) || !isKept(piece - 1u)) { return false; }
  let row = subTableOffset + piece * ${T}u;
  let previous = row - ${T}u;
  return subTable[row + 4u] == subTable[previous + 4u] && subTable[row] == subTable[previous + 2u] &&
    subTable[row + 1u] == subTable[previous + 3u];
}
fn startsRun(piece: u32) -> bool { return isKept(piece) && !continuesPrevious(piece); }
// Whether the end point is a vertex: not when the run goes on along the same segment.
fn emitsEnd(piece: u32) -> bool {
  if (!isKept(piece)) { return false; }
  let sameSegmentNext = continuesPrevious(piece + 1u) &&
    subTable[subTableOffset + (piece + 1u) * ${T}u + 5u] == subTable[subTableOffset + piece * ${T}u + 5u];
  return !sameSegmentNext;
}`,J=b(`run-flags`,`uint32`,y),Y=b(`run-ranks`,`uint32`,y),X=b(`vertex-counts`,`uint32`,y),Z=b(`vertex-starts`,`uint32`,y);g.push(r(e,{id:`${i}-classify`,operation:C,variant:`classify`,bindings:[{name:`subFeatures`,view:fe,type:`u32`,access:`read`},{name:`subTable`,view:q,type:`u32`,access:`read_write`}],invocationCount:y,declarations:`const NO_FEATURE: u32 = ${ie}u;`,body:`let row = subTableOffset + index * ${T}u;
  let flags = subTable[row + 6u];
  if ((flags & ${D}u) == 0u) { return; }
  let inside = (flags & ${E}u) != 0u || subFeatures[subFeaturesOffset + index] != NO_FEATURE;
  if (inside == ${f}) { subTable[row + 6u] = flags | ${oe}u; }`}),r(e,{id:`${i}-run-flags`,operation:C,variant:`run-flags`,bindings:[{name:`subTable`,view:q,type:`u32`,access:`read`},{name:`runFlags`,view:J,type:`u32`,access:`read_write`},{name:`vertexCounts`,view:X,type:`u32`,access:`read_write`}],invocationCount:y,declarations:he,body:`let start = select(0u, 1u, startsRun(index));
  runFlags[runFlagsOffset + index] = start;
  vertexCounts[vertexCountsOffset + index] = select(0u, start + select(0u, 1u, emitsEnd(index)), isKept(index));`}),...new n({id:`${i}-run-scan`,input:J,output:Y,mode:`exclusive`}).getCommandNodes(e),...new n({id:`${i}-vertex-scan`,input:X,output:Z,mode:`exclusive`}).getCommandNodes(e));let Q=b(`run-table`,`uint32`,y*2),ge=b(`flag-state`,`uint32`,3),$=b(`state`,`uint32`,8);g.push(r(e,{id:`${i}-run-table`,operation:C,variant:`run-table`,bindings:[{name:`subTable`,view:q,type:`u32`,access:`read`},{name:`runFlags`,view:J,type:`u32`,access:`read`},{name:`runRanks`,view:Y,type:`u32`,access:`read`},{name:`vertexStarts`,view:Z,type:`u32`,access:`read`},{name:`runTable`,view:Q,type:`u32`,access:`read_write`}],invocationCount:y,body:`if (runFlags[runFlagsOffset + index] == 0u) { return; }
  let rank = runRanks[runRanksOffset + index];
  runTable[runTableOffset + rank * 2u] = vertexStarts[vertexStartsOffset + index];
  runTable[runTableOffset + rank * 2u + 1u] = subTable[subTableOffset + index * ${T}u + 4u];`}),r(e,{id:`${i}-flags`,operation:C,variant:`flags`,bindings:[{name:`pairOverflow`,view:k,type:`u32`,access:`read`},{name:`containmentOverflow`,view:pe,type:`u32`,access:`read`},{name:`segmentUncertain`,view:le,type:`u32`,access:`read`},{name:`containmentUncertain`,view:me,type:`u32`,access:`read`},{name:`flagState`,view:ge,type:`u32`,access:`read_write`}],invocationCount:1,body:`flagState[flagStateOffset] = select(0u, 1u, pairOverflow[pairOverflowOffset] != 0u || containmentOverflow[containmentOverflowOffset] != 0u);
  flagState[flagStateOffset + 1u] = segmentUncertain[segmentUncertainOffset] + containmentUncertain[containmentUncertainOffset];
  flagState[flagStateOffset + 2u] = select(0u, 1u, pairOverflow[pairOverflowOffset] != 0u);`}),r(e,{id:`${i}-finalize`,operation:C,variant:`finalize`,bindings:[{name:`runFlags`,view:J,type:`u32`,access:`read`},{name:`runRanks`,view:Y,type:`u32`,access:`read`},{name:`vertexCounts`,view:X,type:`u32`,access:`read`},{name:`vertexStarts`,view:Z,type:`u32`,access:`read`},{name:`runTable`,view:Q,type:`u32`,access:`read`},{name:`flagState`,view:ge,type:`u32`,access:`read`},{name:`state`,view:$,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const LAST: u32 = ${y-1}u;
const PIECE_CAPACITY: u32 = ${u}u;
const VERTEX_CAPACITY: u32 = ${d}u;`,body:`let totalRuns = runRanks[runRanksOffset + LAST] + runFlags[runFlagsOffset + LAST];
  let totalVertices = vertexStarts[vertexStartsOffset + LAST] + vertexCounts[vertexCountsOffset + LAST];
  var low = 0u;
  var high = min(totalRuns, PIECE_CAPACITY);
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    // End of run middle - 1 is the start of run middle, or the vertex total after the last run.
    var end = totalVertices;
    if (middle < totalRuns) { end = runTable[runTableOffset + middle * 2u]; }
    if (end <= VERTEX_CAPACITY) { low = middle; } else { high = middle - 1u; }
  }
  var written = totalVertices;
  if (low < totalRuns) { written = runTable[runTableOffset + low * 2u]; }
  state[stateOffset] = low;
  state[stateOffset + 1u] = written;
  state[stateOffset + 2u] = totalRuns;
  state[stateOffset + 3u] = totalVertices;
  state[stateOffset + 4u] = select(0u, 1u, low < totalRuns || flagState[flagStateOffset] != 0u);
  state[stateOffset + 5u] = flagState[flagStateOffset + 1u];
  state[stateOffset + 6u] = flagState[flagStateOffset + 2u];`}),r(e,{id:`${i}-write-pieces`,operation:C,variant:`write-pieces`,bindings:[{name:`state`,view:$,type:`u32`,access:`read`},{name:`runTable`,view:Q,type:`u32`,access:`read`},{name:`lineIds`,view:h.sourceIds,type:`u32`,access:`read_write`},{name:`offsets`,view:h.geometry.lineOffsets,type:`u32`,access:`read_write`}],invocationCount:u+1,declarations:`const PIECE_CAPACITY: u32 = ${u}u;`,body:`if (index < state[stateOffset]) {
    lineIds[lineIdsOffset + index] = runTable[runTableOffset + index * 2u + 1u];
    offsets[offsetsOffset + index] = runTable[runTableOffset + index * 2u];
  } else {
    if (index < PIECE_CAPACITY) { lineIds[lineIdsOffset + index] = 0xffffffffu; }
    offsets[offsetsOffset + index] = state[stateOffset + 1u];
  }`}),r(e,{id:`${i}-write-vertices`,operation:C,variant:`write-vertices`,bindings:[{name:`state`,view:$,type:`u32`,access:`read`},{name:`subTable`,view:q,type:`u32`,access:`read`},{name:`runFlags`,view:J,type:`u32`,access:`read`},{name:`runRanks`,view:Y,type:`u32`,access:`read`},{name:`vertexStarts`,view:Z,type:`u32`,access:`read`},{name:`outPositions`,view:h.geometry.positions,type:`f32`,access:`read_write`}],invocationCount:y,declarations:he,body:`if (!isKept(index)) { return; }
  let start = runFlags[runFlagsOffset + index];
  // Rank of the run this sub-piece belongs to: ranks count run starts before the piece.
  let rank = runRanks[runRanksOffset + index] + start - 1u;
  if (rank >= state[stateOffset]) { return; }
  var slot = vertexStarts[vertexStartsOffset + index];
  let row = subTableOffset + index * ${T}u;
  if (start == 1u) {
    outPositions[outPositionsOffset + slot * 2u] = bitcast<f32>(subTable[row]);
    outPositions[outPositionsOffset + slot * 2u + 1u] = bitcast<f32>(subTable[row + 1u]);
    slot = slot + 1u;
  }
  if (emitsEnd(index)) {
    outPositions[outPositionsOffset + slot * 2u] = bitcast<f32>(subTable[row + 2u]);
    outPositions[outPositionsOffset + slot * 2u + 1u] = bitcast<f32>(subTable[row + 3u]);
  }`}));let _e=[{name:`state`,view:$,type:`u32`,access:`read`}],ve=[[`count`,h.status.count,0],[`vertexCount`,h.status.vertexCount,1],[`requiredCount`,h.status.requiredCount,2],[`requiredVertexCount`,h.status.requiredVertexCount,3],[`overflow`,h.status.overflow,4],[`uncertainCount`,o.output.status?.uncertainCount,5],[`candidateOverflow`,h.status.candidateOverflow,6]];for(let[e,t]of ve)t&&_e.push({name:e,view:t,type:`u32`,access:`read_write`});return g.push(r(e,{id:`${i}-scalars`,operation:C,variant:`scalars`,bindings:_e,invocationCount:1,body:ve.filter(([,e])=>e).map(([e,,t])=>`${e}[${e}Offset] = state[stateOffset + ${t}u];`).join(`
  `)})),g}},k=d({parameters:S,caption:`Lines clipped by polygons: the pieces kept inside or outside are colored.`,createInputs:({random:e,world:t})=>({world:t,drag:[0,0],lines:p(e,t,8,14),getBlobs:(e,n)=>y(x(31),t,{rows:2,columns:3,vertexCount:14,irregularity:.3,fill:.95,shiftX:e,shiftY:n})}),build(e,t,n){let r=t.getBlobs(n.shift,n.shift*.5),i=b(e,`polygons`,r),a=e.upload(`positions`,`float32x2`,t.lines.positions),o=e.upload(`line-offsets`,`uint32`,t.lines.lineOffsets),s=h(e,`clipped`,256,2048);e.graph.add(new O({lines:{kind:`lines`,positions:a.view,lineOffsets:o.view},polygons:v(i),mode:n.mode===0?`inside`:`outside`,intersectionCapacity:2048,candidateCapacity:4096,output:{pieces:s.pieces}}));let ee=g(e,`clipped`,s.positions,s.offsets,{lineCount:s.count}),te=e.upload(`original`,`float32x4`,m(t.lines.positions,t.lines.lineOffsets,!1)),c=e.allocate(`outline`,`float32x4`,_(r).length/4);return{outputs:{original:te,outline:c,...ee},update:e=>{let n=t.getBlobs(e.shift+t.drag[0],e.shift*.5+t.drag[1]);i.positions.write(n.positions),c.write(_(n))}}},rebuildOn:[`mode`],interaction:{kind:`drag`,hint:`drag the polygons`,rest:[.5,.5],resolve(e,{inputs:t,values:n}){let r=[t.world.width/2,t.world.height/2];return t.drag=[f((e.x-r[0])/(t.world.width/3),-.7,.7),f((e.y-r[1])/(t.world.height/2),-.7,.7)],{rerun:!0,glyphs:[{shape:`arrow`,from:r,to:[e.x,e.y]},{shape:`crosshair`,at:[e.x,e.y]}],readout:`keep ${n.mode===0?`inside`:`outside`} · shift ${(n.shift+t.drag[0]).toFixed(2)}, ${(n.shift*.5+t.drag[1]).toFixed(2)} cells`}}},draw:({outputs:e})=>[{layer:`lines`,context:!0,segments:e.original,radius:2},{layer:`lines`,context:!0,segments:e.outline,color:[.5,.6,.8,.7],radius:1.4},{layer:`segments`,nodes:e.points,edges:e.edges,values:e.lineIds,radius:6}]});export{k as default};