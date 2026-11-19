import{t as e}from"./ports-xmdKgL07.js";import{r as t,u as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-scan-B4NNTfF9.js";import{o as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{c as a}from"./wgsl-common-DfyXiYB0.js";import{E as o,S as s,f as c,u as l,x as ee}from"./gpu-contributor-utils-BIwLQON5.js";import{t as te}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{i as ne}from"./segment-table-BeVxjLqW.js";import{t as re}from"./gpu-segment-intersection-CGBfhMHr.js";import{v as u}from"./spatial-context-DwZEd4EZ.js";var d=`GPULineSplit`,ie=2147483648,f=8;function p(e){return l(e,32)}var m=4;function ae(e){return Math.max(1,Math.ceil(Math.log2(e+1)))}var h=class{id;props;spatialContext;vertexCount;lineCount;pieceCapacity;vertexCapacity;constructor(t){this.id=t.id??`line-split`,this.spatialContext=u(this.id,t.spatialContext),t={...t,lines:e(this.id,`lines`,t.lines)},this.props=t;let{id:r}=this,{lines:i}=t,{pieces:a}=t.output;if(i.kind!==`lines`)throw Error(`${r} lines must be linestring geometry`);if(o(i.positions,[`float32x2`],`${r} lines.positions`),n(i.lineOffsets,`${r} lines.lineOffsets`),i.lineOffsets.length<2)throw Error(`${r} lines.lineOffsets requires at least one line and a terminal entry`);if(i.positions.length<1)throw Error(`${r} lines.positions must hold at least one vertex`);if(c(r,`intersectionCapacity`,t.intersectionCapacity),a.geometry.kind!==`lines`)throw Error(`${r} pieces.geometry must be linestring geometry`);if(this.vertexCount=i.positions.length,this.lineCount=i.lineOffsets.length-1,n(a.sourceIds,`${r} pieces.sourceIds`),n(a.geometry.lineOffsets,`${r} pieces.geometry.lineOffsets`),o(a.geometry.positions,[`float32x2`],`${r} pieces.geometry.positions`),this.pieceCapacity=a.sourceIds.length,this.vertexCapacity=a.geometry.positions.length,this.pieceCapacity<1||a.geometry.lineOffsets.length!==this.pieceCapacity+1)throw Error(`${r} pieces.geometry.lineOffsets length must be pieces.sourceIds.length + 1`);if(this.vertexCapacity<1)throw Error(`${r} pieces.geometry.positions must be non-empty`);if(a.vertexSourceEdges&&(n(a.vertexSourceEdges,`${r} pieces.vertexSourceEdges`),a.vertexSourceEdges.length!==this.vertexCapacity))throw Error(`${r} pieces.vertexSourceEdges length must equal pieces.geometry.positions length`);for(let[e,i]of[[`count`,a.status.count],[`vertexCount`,a.status.vertexCount],[`overflow`,a.status.overflow],[`candidateOverflow`,a.status.candidateOverflow],[`requiredCount`,a.status.requiredCount],[`requiredVertexCount`,a.status.requiredVertexCount],[`status.uncertainCount`,t.output.status?.uncertainCount]])if(i&&(n(i,`${r} ${e}`),i.length<1))throw Error(`${r} ${e} must contain one uint32 row`)}getCommandNodes(e){let{id:n,props:o,vertexCount:c,lineCount:l,pieceCapacity:u,vertexCapacity:h}=this,{lines:g}=o,{pieces:_}=o.output;ee(n,e,[g.positions,g.lineOffsets,_.sourceIds,_.geometry.lineOffsets,_.geometry.positions,_.status.count,_.status.vertexCount,_.status.overflow,_.status.candidateOverflow,_.status.requiredCount,_.status.requiredVertexCount,_.vertexSourceEdges,o.output.status?.uncertainCount]),s(n,`lines.positions rows`,g.positions.length,ne(e.device)),s(n,`lines + 4 x intersectionCapacity`,l+4*o.intersectionCapacity,p(e.device));let v=[],y=o.intersectionCapacity,b=y*4,x=l+b,S=(r,i,a)=>t(e,`${n}-${r}`,i,a),C=S(`pair-left`,`uint32`,y),w=S(`pair-right`,`uint32`,y),T=S(`pair-count`,`uint32`,1),E=S(`pair-overflow`,`uint32`,1),D=o.output.status?.uncertainCount??S(`uncertain-count`,`uint32`,1),O=S(`non-finite-flag`,`uint32`,1);v.push(i(e,{id:`${n}-clear-non-finite`,operation:d,variant:`clear-non-finite`,bindings:[{name:`flag`,view:O,type:`u32`,access:`read_write`}],invocationCount:1,body:`flag[flagOffset] = 0u;`}),i(e,{id:`${n}-flag-non-finite`,operation:d,variant:`flag-non-finite`,bindings:[{name:`positions`,view:g.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:g.lineOffsets,type:`u32`,access:`read`},{name:`flag`,view:O,type:`atomic<u32>`,access:`read_write`}],invocationCount:c,declarations:`const VERTEX_COUNT: u32 = ${c}u;
const LINE_COUNT: u32 = ${l}u;
${a}`,body:`let firstRow = lineOffsets[lineOffsetsOffset];
  let endRow = min(lineOffsets[lineOffsetsOffset + LINE_COUNT], VERTEX_COUNT);
  if (index < firstRow || index >= endRow) { return; }
  if (!isFiniteBits(positions[positionsOffset + index * 2u]) || !isFiniteBits(positions[positionsOffset + index * 2u + 1u])) {
    atomicOr(&flag[flagOffset], 1u);
  }`}));let k=S(`kinds`,`uint32`,y),A=S(`points`,`float32x2`,y),j=S(`end-points`,`float32x2`,y);v.push(...new re({id:`${n}-intersection`,left:g,spatialSort:o.spatialSort,leafCapacity:o.leafCapacity,output:{pairs:{leftRows:C,rightRows:w,status:{count:T,overflow:E}},status:{uncertainCount:D},kinds:k,points:A,endPoints:j}}).getCommandNodes(e));let M=S(`event-segments`,`uint32`,b),N=S(`event-points`,`float32x2`,b),P=S(`event-indices`,`uint32`,b),oe=S(`event-slots`,`uint32`,b),F=S(`event-phases`,`uint32`,b),I=S(`event-order-keys`,`uint32`,b),L=S(`event-table`,`uint32`,b*2);v.push(i(e,{id:`${n}-event-segments`,operation:d,variant:`event-segments`,bindings:[{name:`pairLeft`,view:C,type:`u32`,access:`read`},{name:`pairRight`,view:w,type:`u32`,access:`read`},{name:`kinds`,view:k,type:`u32`,access:`read`},{name:`pairCount`,view:T,type:`u32`,access:`read`},{name:`eventSegments`,view:M,type:`u32`,access:`read_write`}],invocationCount:b,body:`let pair = index / 4u;
  let which = index % 4u;
  let kind = kinds[kindsOffset + pair];
  // Kinds 1 to 4 intersect at a point or span; 5 (uncertain) and 0 produce no split.
  var valid = pair < pairCount[pairCountOffset] && kind >= 1u && kind <= 4u;
  if (which >= 2u && kind != 4u) { valid = false; }
  var segment = pairLeft[pairLeftOffset + pair];
  if ((which & 1u) == 1u) { segment = pairRight[pairRightOffset + pair]; }
  eventSegments[eventSegmentsOffset + index] = select(0xffffffffu, segment, valid);`}),i(e,{id:`${n}-event-points`,operation:d,variant:`event-points`,bindings:[{name:`eventSegments`,view:M,type:`u32`,access:`read`},{name:`points`,view:A,type:`f32`,access:`read`},{name:`endPoints`,view:j,type:`f32`,access:`read`},{name:`eventPoints`,view:N,type:`f32`,access:`read_write`},{name:`eventIndices`,view:P,type:`u32`,access:`read_write`}],invocationCount:b,body:`let pair = index / 4u;
  var point = vec2f(points[pointsOffset + pair * 2u], points[pointsOffset + pair * 2u + 1u]);
  if (index % 4u >= 2u) {
    point = vec2f(endPoints[endPointsOffset + pair * 2u], endPoints[endPointsOffset + pair * 2u + 1u]);
  }
  eventPoints[eventPointsOffset + index * 2u] = point.x;
  eventPoints[eventPointsOffset + index * 2u + 1u] = point.y;
  eventIndices[eventIndicesOffset + index] = index;`}));let se=`
const VERTEX_COUNT: u32 = ${c}u;
const LINE_COUNT: u32 = ${l}u;
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
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}`;v.push(i(e,{id:`${n}-event-slots`,operation:d,variant:`event-slots`,bindings:[{name:`eventSegments`,view:M,type:`u32`,access:`read`},{name:`eventPoints`,view:N,type:`f32`,access:`read`},{name:`positions`,view:g.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:g.lineOffsets,type:`u32`,access:`read`},{name:`eventSlots`,view:oe,type:`u32`,access:`read_write`},{name:`eventPhases`,view:F,type:`u32`,access:`read_write`},{name:`eventOrderKeys`,view:I,type:`u32`,access:`read_write`},{name:`eventTable`,view:L,type:`u32`,access:`read_write`}],invocationCount:b,declarations:se,body:`let segment = eventSegments[eventSegmentsOffset + index];
  var slot = VERTEX_COUNT;
  var phase = 0u;
  var key = 0.0;
  if (segment != 0xffffffffu) {
    let line = lineOfVertex(segment);
    let lineStart = lineOffsets[lineOffsetsOffset + line];
    let lineEnd = lineOffsets[lineOffsetsOffset + line + 1u];
    let a = vertexAt(segment);
    let b = vertexAt(segment + 1u);
    let p = vec2f(eventPoints[eventPointsOffset + index * 2u], eventPoints[eventPointsOffset + index * 2u + 1u]);
    if (p.x == a.x && p.y == a.y) {
      if (segment > lineStart) { slot = segment; }
    } else if (p.x == b.x && p.y == b.y) {
      if (segment + 1u < lineEnd - 1u) { slot = segment + 1u; }
    } else {
      slot = segment;
      phase = 1u;
      let d = b - a;
      if (abs(d.x) >= abs(d.y)) { key = p.x * select(-1.0, 1.0, d.x > 0.0); }
      else { key = p.y * select(-1.0, 1.0, d.y > 0.0); }
    }
  }
  eventSlots[eventSlotsOffset + index] = slot;
  eventPhases[eventPhasesOffset + index] = phase;
  let keyBits = bitcast<u32>(key);
  eventOrderKeys[eventOrderKeysOffset + index] = select(keyBits ^ 0xffffffffu, keyBits ^ 0x80000000u, (keyBits & 0x80000000u) == 0u);
  eventTable[eventTableOffset + index * 2u] = phase;
  eventTable[eventTableOffset + index * 2u + 1u] = bitcast<u32>(key);`}));let R=te(e,{id:`${n}-event-order`,operation:d,indices:P,keys:[{view:oe,keyBits:ae(c)},{view:F,keyBits:1},{view:I}]});v.push(...R.nodes);let z=R.sortedPrimaryKeys,B=R.sortedIndices,V=`
const VERTEX_COUNT: u32 = ${c}u;
fn eventPhase(event: u32) -> u32 { return eventTable[eventTableOffset + event * 2u]; }
fn eventKey(event: u32) -> f32 { return bitcast<f32>(eventTable[eventTableOffset + event * 2u + 1u]); }
fn eventSame(first: u32, second: u32) -> bool {
  return eventPhase(first) == eventPhase(second) && eventKey(first) == eventKey(second);
}`,H=S(`unique-flags`,`uint32`,b),U=S(`unique-ranks`,`uint32`,b),W=S(`split-table`,`uint32`,b*m);v.push(i(e,{id:`${n}-unique-flags`,operation:d,variant:`unique-flags`,bindings:[{name:`sortedSlots`,view:z,type:`u32`,access:`read`},{name:`sortedEvents`,view:B,type:`u32`,access:`read`},{name:`eventTable`,view:L,type:`u32`,access:`read`},{name:`uniqueFlags`,view:H,type:`u32`,access:`read_write`}],invocationCount:b,declarations:V,body:`let slot = sortedSlots[sortedSlotsOffset + index];
  var flag = 0u;
  if (slot < VERTEX_COUNT) {
    if (index == 0u || sortedSlots[sortedSlotsOffset + index - 1u] != slot) { flag = 1u; }
    else if (!eventSame(sortedEvents[sortedEventsOffset + index], sortedEvents[sortedEventsOffset + index - 1u])) { flag = 1u; }
  }
  uniqueFlags[uniqueFlagsOffset + index] = flag;`}),...new r({id:`${n}-unique-scan`,input:H,output:U,mode:`exclusive`}).getCommandNodes(e),i(e,{id:`${n}-compact-splits`,operation:d,variant:`compact-splits`,bindings:[{name:`sortedSlots`,view:z,type:`u32`,access:`read`},{name:`sortedEvents`,view:B,type:`u32`,access:`read`},{name:`eventTable`,view:L,type:`u32`,access:`read`},{name:`eventPoints`,view:N,type:`u32`,access:`read`},{name:`uniqueFlags`,view:H,type:`u32`,access:`read`},{name:`uniqueRanks`,view:U,type:`u32`,access:`read`},{name:`splitTable`,view:W,type:`u32`,access:`read_write`}],invocationCount:b,declarations:V,body:`if (uniqueFlags[uniqueFlagsOffset + index] == 0u) { return; }
  let rank = uniqueRanks[uniqueRanksOffset + index];
  let event = sortedEvents[sortedEventsOffset + index];
  let row = splitTableOffset + rank * ${m}u;
  splitTable[row] = sortedSlots[sortedSlotsOffset + index];
  splitTable[row + 1u] = eventPhase(event);
  splitTable[row + 2u] = eventPoints[eventPointsOffset + event * 2u];
  splitTable[row + 3u] = eventPoints[eventPointsOffset + event * 2u + 1u];`}));let G=S(`split-bases`,`uint32`,l),K=S(`piece-counts`,`uint32`,l),q=S(`piece-starts`,`uint32`,l);v.push(i(e,{id:`${n}-line-splits`,operation:d,variant:`line-splits`,bindings:[{name:`lineOffsets`,view:g.lineOffsets,type:`u32`,access:`read`},{name:`sortedSlots`,view:z,type:`u32`,access:`read`},{name:`uniqueFlags`,view:H,type:`u32`,access:`read`},{name:`uniqueRanks`,view:U,type:`u32`,access:`read`},{name:`splitBases`,view:G,type:`u32`,access:`read_write`},{name:`pieceCounts`,view:K,type:`u32`,access:`read_write`}],invocationCount:l,declarations:`const EVENT_COUNT: u32 = ${b}u;
fn lowerBound(value: u32) -> u32 {
  var low = 0u;
  var high = EVENT_COUNT;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (sortedSlots[sortedSlotsOffset + middle] < value) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}
fn uniqueBefore(position: u32) -> u32 {
  if (position < EVENT_COUNT) { return uniqueRanks[uniqueRanksOffset + position]; }
  return uniqueRanks[uniqueRanksOffset + EVENT_COUNT - 1u] + uniqueFlags[uniqueFlagsOffset + EVENT_COUNT - 1u];
}`,body:`let lineStart = lineOffsets[lineOffsetsOffset + index];
  let lineEnd = lineOffsets[lineOffsetsOffset + index + 1u];
  let first = uniqueBefore(lowerBound(lineStart));
  let last = uniqueBefore(lowerBound(lineEnd));
  splitBases[splitBasesOffset + index] = first;
  pieceCounts[pieceCountsOffset + index] = select(0u, last - first + 1u, lineEnd >= lineStart + 2u);`}),...new r({id:`${n}-piece-scan`,input:K,output:q,mode:`exclusive`}).getCommandNodes(e));let J=S(`piece-table`,`uint32`,x*f),Y=S(`piece-vertex-counts`,`uint32`,x),X=S(`piece-vertex-starts`,`uint32`,x);v.push(i(e,{id:`${n}-pieces`,operation:d,variant:`pieces`,bindings:[{name:`lineOffsets`,view:g.lineOffsets,type:`u32`,access:`read`},{name:`pieceStarts`,view:q,type:`u32`,access:`read`},{name:`pieceCounts`,view:K,type:`u32`,access:`read`},{name:`splitBases`,view:G,type:`u32`,access:`read`},{name:`splitTable`,view:W,type:`u32`,access:`read`},{name:`pieceTable`,view:J,type:`u32`,access:`read_write`},{name:`pieceVertexCounts`,view:Y,type:`u32`,access:`read_write`}],invocationCount:x,declarations:`const LINE_COUNT: u32 = ${l}u;`,body:`let total = pieceStarts[pieceStartsOffset + LINE_COUNT - 1u] + pieceCounts[pieceCountsOffset + LINE_COUNT - 1u];
  if (index >= total) { pieceVertexCounts[pieceVertexCountsOffset + index] = 0u; return; }
  // Last line whose first piece is at most index.
  var low = 0u;
  var high = LINE_COUNT - 1u;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (pieceStarts[pieceStartsOffset + middle] <= index) { low = middle; } else { high = middle - 1u; }
  }
  let line = low;
  let k = index - pieceStarts[pieceStartsOffset + line];
  let lastPiece = pieceCounts[pieceCountsOffset + line] - 1u;
  let base = splitBases[splitBasesOffset + line];
  var firstVertex = lineOffsets[lineOffsetsOffset + line];
  var lastVertex = lineOffsets[lineOffsetsOffset + line + 1u] - 1u;
  var prefix = 0u;
  var suffix = 0u;
  var prefixPoint = vec2u(0u);
  var suffixPoint = vec2u(0u);
  if (k > 0u) {
    let row = splitTableOffset + (base + k - 1u) * ${m}u;
    let slot = splitTable[row];
    let phase = splitTable[row + 1u];
    firstVertex = slot + phase;
    prefix = phase;
    prefixPoint = vec2u(splitTable[row + 2u], splitTable[row + 3u]);
  }
  if (k < lastPiece) {
    let row = splitTableOffset + (base + k) * ${m}u;
    lastVertex = splitTable[row];
    suffix = splitTable[row + 1u];
    suffixPoint = vec2u(splitTable[row + 2u], splitTable[row + 3u]);
  }
  let row = pieceTableOffset + index * ${f}u;
  pieceTable[row] = line;
  pieceTable[row + 1u] = firstVertex;
  pieceTable[row + 2u] = lastVertex;
  pieceTable[row + 3u] = prefix | (suffix << 1u);
  pieceTable[row + 4u] = prefixPoint.x;
  pieceTable[row + 5u] = prefixPoint.y;
  pieceTable[row + 6u] = suffixPoint.x;
  pieceTable[row + 7u] = suffixPoint.y;
  pieceVertexCounts[pieceVertexCountsOffset + index] = lastVertex - firstVertex + 1u + prefix + suffix;`}),...new r({id:`${n}-vertex-scan`,input:Y,output:X,mode:`exclusive`}).getCommandNodes(e));let{vertexSourceEdges:Z}=_,Q=Z?S(`self-contacts`,`uint32`,c):null;Q&&v.push(i(e,{id:`${n}-self-contacts-clear`,operation:d,variant:`self-contacts-clear`,bindings:[{name:`selfContacts`,view:Q,type:`u32`,access:`read_write`}],invocationCount:c,body:`selfContacts[selfContactsOffset + index] = 0u;`}),i(e,{id:`${n}-self-contacts`,operation:d,variant:`self-contacts`,bindings:[{name:`pairLeft`,view:C,type:`u32`,access:`read`},{name:`pairRight`,view:w,type:`u32`,access:`read`},{name:`pairCount`,view:T,type:`u32`,access:`read`},{name:`lineOffsets`,view:g.lineOffsets,type:`u32`,access:`read`},{name:`selfContacts`,view:Q,type:`atomic<u32>`,access:`read_write`}],invocationCount:y,declarations:`const LINE_COUNT: u32 = ${l}u;
// Largest line whose first vertex is at most vertex.
fn lineOfVertex(vertex: u32) -> u32 {
  var low = 0u;
  var high = LINE_COUNT - 1u;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle] <= vertex) { low = middle; } else { high = middle - 1u; }
  }
  return low;
}`,body:`if (index >= pairCount[pairCountOffset]) { return; }
  let left = pairLeft[pairLeftOffset + index];
  let right = pairRight[pairRightOffset + index];
  if (lineOfVertex(left) != lineOfVertex(right)) { return; }
  atomicStore(&selfContacts[selfContactsOffset + left], 1u);
  atomicStore(&selfContacts[selfContactsOffset + right], 1u);`}));let $=S(`state`,`uint32`,8);v.push(i(e,{id:`${n}-finalize`,operation:d,variant:`finalize`,bindings:[{name:`pieceStarts`,view:q,type:`u32`,access:`read`},{name:`pieceCounts`,view:K,type:`u32`,access:`read`},{name:`vertexStarts`,view:X,type:`u32`,access:`read`},{name:`vertexCounts`,view:Y,type:`u32`,access:`read`},{name:`pairOverflow`,view:E,type:`u32`,access:`read`},{name:`state`,view:$,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const LINE_COUNT: u32 = ${l}u;
const PIECE_CAPACITY: u32 = ${u}u;
const VERTEX_CAPACITY: u32 = ${h}u;
fn pieceEnd(piece: u32) -> u32 { return vertexStarts[vertexStartsOffset + piece] + vertexCounts[vertexCountsOffset + piece]; }`,body:`let totalPieces = pieceStarts[pieceStartsOffset + LINE_COUNT - 1u] + pieceCounts[pieceCountsOffset + LINE_COUNT - 1u];
  var totalVertices = 0u;
  if (totalPieces > 0u) { totalVertices = pieceEnd(totalPieces - 1u); }
  var low = 0u;
  var high = min(totalPieces, PIECE_CAPACITY);
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (pieceEnd(middle - 1u) <= VERTEX_CAPACITY) { low = middle; } else { high = middle - 1u; }
  }
  var written = 0u;
  if (low > 0u) { written = pieceEnd(low - 1u); }
  state[stateOffset] = low;
  state[stateOffset + 1u] = written;
  state[stateOffset + 2u] = totalPieces;
  state[stateOffset + 3u] = totalVertices;
  state[stateOffset + 4u] = select(0u, 1u, low < totalPieces);
  state[stateOffset + 5u] = pairOverflow[pairOverflowOffset];`}),i(e,{id:`${n}-write-pieces`,operation:d,variant:`write-pieces`,bindings:[{name:`state`,view:$,type:`u32`,access:`read`},{name:`pieceTable`,view:J,type:`u32`,access:`read`},{name:`vertexStarts`,view:X,type:`u32`,access:`read`},{name:`lineIds`,view:_.sourceIds,type:`u32`,access:`read_write`},{name:`offsets`,view:_.geometry.lineOffsets,type:`u32`,access:`read_write`}],invocationCount:u+1,declarations:`const PIECE_CAPACITY: u32 = ${u}u;`,body:`let kept = state[stateOffset];
  if (index < kept) {
    lineIds[lineIdsOffset + index] = pieceTable[pieceTableOffset + index * ${f}u];
    offsets[offsetsOffset + index] = vertexStarts[vertexStartsOffset + index];
  } else {
    if (index < PIECE_CAPACITY) { lineIds[lineIdsOffset + index] = 0xffffffffu; }
    offsets[offsetsOffset + index] = state[stateOffset + 1u];
  }`}),i(e,{id:`${n}-write-vertices`,operation:d,variant:Q?`write-vertices-with-edges`:`write-vertices`,bindings:[{name:`state`,view:$,type:`u32`,access:`read`},{name:`pieceTable`,view:J,type:`u32`,access:`read`},{name:`vertexStarts`,view:X,type:`u32`,access:`read`},{name:`positions`,view:g.positions,type:`f32`,access:`read`},{name:`outPositions`,view:_.geometry.positions,type:`f32`,access:`read_write`},...Q&&Z?[{name:`selfContacts`,view:Q,type:`u32`,access:`read`},{name:`sourceEdges`,view:Z,type:`u32`,access:`read_write`}]:[]],invocationCount:h,body:`if (index >= state[stateOffset + 1u]) { return; }
  // Last kept piece whose first vertex is at most index.
  var low = 0u;
  var high = state[stateOffset] - 1u;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (vertexStarts[vertexStartsOffset + middle] <= index) { low = middle; } else { high = middle - 1u; }
  }
  let row = pieceTableOffset + low * ${f}u;
  let firstVertex = pieceTable[row + 1u];
  let runLength = pieceTable[row + 2u] - firstVertex + 1u;
  let flags = pieceTable[row + 3u];
  var local = index - vertexStarts[vertexStartsOffset + low];
  ${Q?`// A prefix point lies inside the edge before firstVertex; input vertex v starts edge v.
  let edge = firstVertex - (flags & 1u) + local;
  var contact = state[stateOffset + 5u] != 0u;
  if (edge < ${c}u && selfContacts[selfContactsOffset + edge] != 0u) { contact = true; }
  sourceEdges[sourceEdgesOffset + index] = edge | select(0u, ${ie}u, contact);`:``}
  var point = vec2f(0.0);
  var resolved = false;
  if ((flags & 1u) != 0u) {
    if (local == 0u) {
      point = vec2f(bitcast<f32>(pieceTable[row + 4u]), bitcast<f32>(pieceTable[row + 5u]));
      resolved = true;
    } else { local = local - 1u; }
  }
  if (!resolved) {
    if (local < runLength) {
      let source = positionsOffset + (firstVertex + local) * 2u;
      point = vec2f(positions[source], positions[source + 1u]);
    } else {
      point = vec2f(bitcast<f32>(pieceTable[row + 6u]), bitcast<f32>(pieceTable[row + 7u]));
    }
  }
  outPositions[outPositionsOffset + index * 2u] = point.x;
  outPositions[outPositionsOffset + index * 2u + 1u] = point.y;`})),v.push(i(e,{id:`${n}-incomplete`,operation:d,variant:`incomplete`,bindings:[{name:`uncertainPairs`,view:D,type:`u32`,access:`read`},{name:`incomplete`,view:O,type:`u32`,access:`read_write`}],invocationCount:1,body:`if (uncertainPairs[uncertainPairsOffset] != 0u) { incomplete[incompleteOffset] = 1u; }`}));let ce=[{name:`state`,view:$,type:`u32`,access:`read`},{name:`incomplete`,view:O,type:`u32`,access:`read`}],le=[[`count`,_.status.count,0],[`vertexCount`,_.status.vertexCount,1],[`requiredCount`,_.status.requiredCount,2],[`requiredVertexCount`,_.status.requiredVertexCount,3],[`overflow`,_.status.overflow,4],[`candidateOverflow`,_.status.candidateOverflow,5]];for(let[e,t]of le)t&&ce.push({name:e,view:t,type:`u32`,access:`read_write`});return v.push(i(e,{id:`${n}-scalars`,operation:d,variant:`scalars`,bindings:ce,invocationCount:1,body:le.filter(([,e])=>e).map(([e,,t])=>e===`overflow`?`overflow[overflowOffset] = select(0u, 1u, state[stateOffset + ${t}u] != 0u || incomplete[incompleteOffset] != 0u);`:`${e}[${e}Offset] = state[stateOffset + ${t}u];`).join(`
  `)})),v}};export{ie as n,p as r,h as t};