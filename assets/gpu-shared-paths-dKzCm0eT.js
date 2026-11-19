import{t as e}from"./ports-xmdKgL07.js";import{r as t}from"./scan-total-C4QddMmi.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-scan-B4NNTfF9.js";import{o as i,t as ee}from"./wgsl-kernel-nodes-DagJklH2.js";import{n as te,r as ne,t as a}from"./list-ranking-SHiVJGmq.js";import{E as o,S as re,f as ie,u as s,x as ae}from"./gpu-contributor-utils-BIwLQON5.js";import{i as oe}from"./segment-table-BeVxjLqW.js";import{t as se}from"./gpu-segment-intersection-CGBfhMHr.js";import{v as c}from"./spatial-context-DwZEd4EZ.js";var l=`GPUSharedPaths`,u=12,d=5;function ce(e){return s(e,48)}var f=class{id;props;spatialContext;leftVertexCount;runCapacity;vertexCapacity;constructor(t){this.id=t.id??`shared-paths`,this.spatialContext=c(this.id,t.spatialContext),t={...t,left:e(this.id,`left`,t.left),right:e(this.id,`right`,t.right)},this.props=t;let{id:n}=this,{runs:r,status:i}=t.output;for(let[e,r]of[[`left`,t.left],[`right`,t.right]]){if(r.kind!==`lines`)throw Error(`${n} ${e} must be linestring geometry`);if(o(r.positions,[`float32x2`],`${n} ${e}.positions`),o(r.lineOffsets,[`uint32`],`${n} ${e}.lineOffsets`),r.lineOffsets.length<2)throw Error(`${n} ${e}.lineOffsets requires at least one line and a terminal entry`);if(r.positions.length<1)throw Error(`${n} ${e}.positions must hold at least one vertex`)}ie(n,`intersectionCapacity`,t.intersectionCapacity),this.leftVertexCount=t.left.positions.length;for(let[e,t]of[[`output.runs.leftSourceIds`,r.leftSourceIds],[`output.runs.rightSourceIds`,r.rightSourceIds],[`output.runs.forward`,r.forward],[`output.runs.offsets`,r.offsets]])o(t,[`uint32`],`${n} ${e}`);if(o(r.positions,[`float32x2`],`${n} output.runs.positions`),this.runCapacity=r.leftSourceIds.length,this.vertexCapacity=r.positions.length,this.runCapacity<1||r.rightSourceIds.length!==this.runCapacity||r.forward.length!==this.runCapacity)throw Error(`${n} output.runs ids need equal nonzero length`);if(r.offsets.length!==this.runCapacity+1)throw Error(`${n} output.runs.offsets needs capacity + 1 rows`);if(this.vertexCapacity<1)throw Error(`${n} output.runs.positions must be non-empty`);for(let[e,t]of[[`count`,i.count],[`vertexCount`,i.vertexCount],[`overflow`,i.overflow],[`requiredCount`,i.requiredCount],[`requiredVertexCount`,i.requiredVertexCount],[`uncertainCount`,i.uncertainCount],[`candidateOverflow`,i.candidateOverflow]])if(t&&(o(t,[`uint32`],`${n} output.status.${e}`),t.length<1))throw Error(`${n} output.status.${e} needs one row`)}getCommandNodes(e){let{id:o,props:s,runCapacity:c,vertexCapacity:f}=this,{left:p,right:m}=s,{runs:h,status:g}=s.output;ae(o,e,[p.positions,p.lineOffsets,m.positions,m.lineOffsets,h.leftSourceIds,h.rightSourceIds,h.forward,h.offsets,h.positions,g.count,g.vertexCount,g.overflow,g.requiredCount,g.requiredVertexCount,g.uncertainCount,g.candidateOverflow]);for(let[t,n]of[[`left`,s.left],[`right`,s.right]])re(o,`${t}.positions rows`,n.positions.length,oe(e.device));ie(o,`intersectionCapacity`,s.intersectionCapacity,1,ce(e.device));let _=[],v=s.intersectionCapacity,y=(t,r,i)=>n(e,`${o}-${t}`,r,i),b=y(`pair-left`,`uint32`,v),x=y(`pair-right`,`uint32`,v),S=y(`pair-count`,`uint32`,1),C=y(`pair-overflow`,`uint32`,1),w=y(`kinds`,`uint32`,v),T=y(`points`,`float32x2`,v),E=y(`end-points`,`float32x2`,v),D=y(`left-rings`,`uint32`,v),O=y(`right-rings`,`uint32`,v);_.push(...new se({id:`${o}-intersection`,left:p,right:m,spatialSort:s.spatialSort,leafCapacity:s.leafCapacity,output:{pairs:{leftRows:b,rightRows:x,status:{count:S,overflow:C}},status:{uncertainCount:g.uncertainCount},kinds:w,points:T,endPoints:E,leftRings:D,rightRings:O}}).getCommandNodes(e));let k=e=>`
fn ${e}Vertex(vertex: u32) -> vec2f {
  return vec2f(${e}Positions[${e}PositionsOffset + vertex * 2u], ${e}Positions[${e}PositionsOffset + vertex * 2u + 1u]);
}`,le=`
// Coordinate of p along direction d on the dominant axis; ascending along d.
fn keyOf(p: vec2f, d: vec2f) -> f32 {
  if (abs(d.x) >= abs(d.y)) { return p.x * select(-1.0, 1.0, d.x > 0.0); }
  return p.y * select(-1.0, 1.0, d.y > 0.0);
}
// Component of d on the dominant axis of reference.
fn axisComponent(d: vec2f, reference: vec2f) -> f32 {
  if (abs(reference.x) >= abs(reference.y)) { return d.x; }
  return d.y;
}`,A=y(`span-table`,`uint32`,v*u);_.push(i(e,{id:`${o}-directions`,operation:l,variant:`directions`,bindings:[{name:`leftPositions`,view:p.positions,type:`f32`,access:`read`},{name:`rightPositions`,view:m.positions,type:`f32`,access:`read`},{name:`pairLeft`,view:b,type:`u32`,access:`read`},{name:`pairRight`,view:x,type:`u32`,access:`read`},{name:`pairCount`,view:S,type:`u32`,access:`read`},{name:`kinds`,view:w,type:`u32`,access:`read`},{name:`spanTable`,view:A,type:`u32`,access:`read_write`}],invocationCount:v,declarations:`${k(`left`)}${k(`right`)}${le}`,body:`let row = spanTableOffset + index * ${u}u;
  var flags = 0u;
  if (index < pairCount[pairCountOffset] && kinds[kindsOffset + index] == 4u) {
    let leftSegment = pairLeft[pairLeftOffset + index];
    let rightSegment = pairRight[pairRightOffset + index];
    let leftDirection = leftVertex(leftSegment + 1u) - leftVertex(leftSegment);
    let rightDirection = rightVertex(rightSegment + 1u) - rightVertex(rightSegment);
    flags = 1u;
    if ((axisComponent(rightDirection, leftDirection) > 0.0) == (axisComponent(leftDirection, leftDirection) > 0.0)) { flags = flags | 2u; }
    spanTable[row + 9u] = leftSegment;
  }
  spanTable[row] = flags;
  spanTable[row + 1u] = 0xffffffffu;
  spanTable[row + 2u] = 0u;`}),i(e,{id:`${o}-spans`,operation:l,variant:`spans`,bindings:[{name:`leftPositions`,view:p.positions,type:`f32`,access:`read`},{name:`pairLeft`,view:b,type:`u32`,access:`read`},{name:`points`,view:T,type:`f32`,access:`read`},{name:`endPoints`,view:E,type:`f32`,access:`read`},{name:`leftRings`,view:D,type:`u32`,access:`read`},{name:`rightRings`,view:O,type:`u32`,access:`read`},{name:`spanTable`,view:A,type:`u32`,access:`read_write`}],invocationCount:v,declarations:`${k(`left`)}${le}`,body:`let row = spanTableOffset + index * ${u}u;
  if ((spanTable[row] & 1u) == 0u) { return; }
  let segment = pairLeft[pairLeftOffset + index];
  let direction = leftVertex(segment + 1u) - leftVertex(segment);
  var first = vec2f(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  var second = vec2f(endPoints[endPointsOffset + index * 2u], endPoints[endPointsOffset + index * 2u + 1u]);
  if (keyOf(second, direction) < keyOf(first, direction)) {
    let swap = first;
    first = second;
    second = swap;
  }
  spanTable[row + 3u] = bitcast<u32>(first.x);
  spanTable[row + 4u] = bitcast<u32>(first.y);
  spanTable[row + 5u] = bitcast<u32>(second.x);
  spanTable[row + 6u] = bitcast<u32>(second.y);
  spanTable[row + 7u] = leftRings[leftRingsOffset + index];
  spanTable[row + 8u] = rightRings[rightRingsOffset + index];`}));let j=`
const PAIR_CAPACITY: u32 = ${v}u;
const SPAN_NONE: u32 = 0xffffffffu;
fn spanWord(span: u32, word: u32) -> u32 { return spanTable[spanTableOffset + span * ${u}u + word]; }
fn isSpan(span: u32) -> bool { return span < PAIR_CAPACITY && (spanWord(span, 0u) & 1u) != 0u; }
fn spanSegment(span: u32) -> u32 { return spanWord(span, 9u); }
// Whether span second starts where span first ends, on the same line pair and direction.
fn follows(first: u32, second: u32) -> bool {
  return isSpan(first) && isSpan(second) && first != second &&
    spanWord(first, 7u) == spanWord(second, 7u) && spanWord(first, 8u) == spanWord(second, 8u) &&
    (spanWord(first, 0u) & 2u) == (spanWord(second, 0u) & 2u) &&
    spanWord(first, 5u) == spanWord(second, 3u) && spanWord(first, 6u) == spanWord(second, 4u);
}`;_.push(i(e,{id:`${o}-links`,operation:l,variant:`links`,bindings:[{name:`pairLeft`,view:b,type:`u32`,access:`read`},{name:`pairCount`,view:S,type:`u32`,access:`read`},{name:`spanTable`,view:A,type:`u32`,access:`read_write`}],invocationCount:v,declarations:`${j}
// In-kernel lower bound on data-dependent keys; GPUGallopingSearch is a batch node and would
// add a pass and a buffer.
fn lowerBound(value: u32) -> u32 {
  var low = 0u;
  var high = pairCount[pairCountOffset];
  while (low < high) {
    let middle = (low + high) / 2u;
    if (pairLeft[pairLeftOffset + middle] < value) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}`,body:`if (!isSpan(index)) { return; }
  let segment = spanSegment(index);
  var first = 0u;
  if (segment > 0u) { first = lowerBound(segment - 1u); }
  let last = lowerBound(segment + 2u);
  var hasPrevious = 0u;
  var next = SPAN_NONE;
  for (var other = first; other < last; other++) {
    if (follows(other, index) && spanSegment(other) <= segment) { hasPrevious = 1u; }
    if (next == SPAN_NONE && follows(index, other) && spanSegment(other) >= segment) { next = other; }
  }
  spanTable[spanTableOffset + index * ${u}u + 1u] = next;
  spanTable[spanTableOffset + index * ${u}u + 2u] = hasPrevious;`}));let M=y(`run-flags`,`uint32`,v),N=y(`run-ranks`,`uint32`,v),P=y(`vertex-counts`,`uint32`,v),F=y(`vertex-starts`,`uint32`,v),I=y(`vertex-total`,`uint32`,1),L=`${j}
fn startsRun(span: u32) -> bool { return isSpan(span) && spanWord(span, 2u) == 0u; }
// Whether the end of span is a vertex of the run: not when the chain goes on in the same segment.
fn emitsEnd(span: u32) -> bool {
  let next = spanWord(span, 1u);
  return next == SPAN_NONE || spanSegment(next) != spanSegment(span);
}`,R=e=>[y(`${e}-a`,`uint32`,v),y(`${e}-b`,`uint32`,v)],z=R(`size-pointer`),B=R(`size-total`),V=R(`emit-pointer`),H=ne(v),U=a(e,{id:`${o}-size-round`,operation:l,variant:`size-round`,length:v,rounds:H,pointers:z,values:[{name:`total`,views:B,rule:`add`}]}),W=B[U.finalIndex],G=a(e,{id:`${o}-emit-round`,operation:l,variant:`emit-round`,length:v,rounds:H,pointers:V,values:[]}),K=V[G.finalIndex];_.push(i(e,{id:`${o}-size-init`,operation:l,variant:`size-init`,bindings:[{name:`spanTable`,view:A,type:`u32`,access:`read`},{name:`sizePointer`,view:z[0],type:`u32`,access:`read_write`},{name:`sizeTotal`,view:B[0],type:`u32`,access:`read_write`},{name:`emitPointer`,view:V[0],type:`u32`,access:`read_write`}],invocationCount:v,declarations:L,body:`let valid = isSpan(index);
  sizePointer[sizePointerOffset + index] = select(SPAN_NONE, spanWord(index, 1u), valid);
  sizeTotal[sizeTotalOffset + index] = select(0u, select(0u, 1u, emitsEnd(index)), valid);
  emitPointer[emitPointerOffset + index] = select(SPAN_NONE, select(spanWord(index, 1u), index, emitsEnd(index)), valid);`}),...U.nodes,...G.nodes,i(e,{id:`${o}-run-sizes`,operation:l,variant:`run-sizes`,bindings:[{name:`spanTable`,view:A,type:`u32`,access:`read`},{name:`sizeTotal`,view:W,type:`u32`,access:`read`},{name:`runFlags`,view:M,type:`u32`,access:`read_write`},{name:`vertexCounts`,view:P,type:`u32`,access:`read_write`}],invocationCount:v,declarations:L,body:`let starts = startsRun(index);
  runFlags[runFlagsOffset + index] = select(0u, 1u, starts);
  // The start point plus one vertex per emitting span of the chain.
  vertexCounts[vertexCountsOffset + index] = select(0u, 1u + sizeTotal[sizeTotalOffset + index], starts);`}),...new r({id:`${o}-run-scan`,input:M,output:N,mode:`exclusive`}).getCommandNodes(e),...new r({id:`${o}-vertex-scan`,input:P,output:F,mode:`exclusive`}).getCommandNodes(e),...t(e,{id:`${o}-vertex`,operation:l,counts:P,offsets:F,rowCount:v,total:I}));let q=y(`run-table`,`uint32`,v*d),J=y(`state`,`uint32`,8);_.push(i(e,{id:`${o}-run-table`,operation:l,variant:`run-table`,bindings:[{name:`spanTable`,view:A,type:`u32`,access:`read`},{name:`runFlags`,view:M,type:`u32`,access:`read`},{name:`runRanks`,view:N,type:`u32`,access:`read`},{name:`vertexStarts`,view:F,type:`u32`,access:`read`},{name:`runTable`,view:q,type:`u32`,access:`read_write`}],invocationCount:v,body:`if (runFlags[runFlagsOffset + index] == 0u) { return; }
  let row = runTableOffset + runRanks[runRanksOffset + index] * ${d}u;
  let source = spanTableOffset + index * ${u}u;
  runTable[row] = vertexStarts[vertexStartsOffset + index];
  runTable[row + 1u] = spanTable[source + 7u];
  runTable[row + 2u] = spanTable[source + 8u];
  runTable[row + 3u] = (spanTable[source] >> 1u) & 1u;
  runTable[row + 4u] = index;`}),i(e,{id:`${o}-finalize`,operation:l,variant:`finalize`,bindings:[{name:`runFlags`,view:M,type:`u32`,access:`read`},{name:`runRanks`,view:N,type:`u32`,access:`read`},{name:`vertexTotal`,view:I,type:`u32`,access:`read`},{name:`runTable`,view:q,type:`u32`,access:`read`},{name:`pairOverflow`,view:C,type:`u32`,access:`read`},{name:`state`,view:J,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const LAST: u32 = ${v-1}u;
const RUN_CAPACITY: u32 = ${c}u;
const VERTEX_CAPACITY: u32 = ${f}u;`,body:`let totalRuns = runRanks[runRanksOffset + LAST] + runFlags[runFlagsOffset + LAST];
  let totalVertices = vertexTotal[vertexTotalOffset];
  var low = 0u;
  var high = min(totalRuns, RUN_CAPACITY);
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    var end = totalVertices;
    if (middle < totalRuns) { end = runTable[runTableOffset + middle * ${d}u]; }
    if (end <= VERTEX_CAPACITY) { low = middle; } else { high = middle - 1u; }
  }
  var written = totalVertices;
  if (low < totalRuns) { written = runTable[runTableOffset + low * ${d}u]; }
  state[stateOffset] = low;
  state[stateOffset + 1u] = written;
  state[stateOffset + 2u] = totalRuns;
  state[stateOffset + 3u] = totalVertices;
  state[stateOffset + 4u] = select(0u, 1u, low < totalRuns || totalVertices > VERTEX_CAPACITY || pairOverflow[pairOverflowOffset] != 0u);
  state[stateOffset + 5u] = select(0u, 1u, pairOverflow[pairOverflowOffset] != 0u);`}),i(e,{id:`${o}-write-runs`,operation:l,variant:`write-runs`,bindings:[{name:`state`,view:J,type:`u32`,access:`read`},{name:`runTable`,view:q,type:`u32`,access:`read`},{name:`leftSourceIds`,view:h.leftSourceIds,type:`u32`,access:`read_write`},{name:`rightSourceIds`,view:h.rightSourceIds,type:`u32`,access:`read_write`},{name:`forward`,view:h.forward,type:`u32`,access:`read_write`},{name:`offsets`,view:h.offsets,type:`u32`,access:`read_write`}],invocationCount:c+1,declarations:`const RUN_CAPACITY: u32 = ${c}u;`,body:`if (index < state[stateOffset]) {
    let row = runTableOffset + index * ${d}u;
    leftSourceIds[leftSourceIdsOffset + index] = runTable[row + 1u];
    rightSourceIds[rightSourceIdsOffset + index] = runTable[row + 2u];
    forward[forwardOffset + index] = runTable[row + 3u];
    offsets[offsetsOffset + index] = runTable[row];
  } else {
    if (index < RUN_CAPACITY) {
      leftSourceIds[leftSourceIdsOffset + index] = 0xffffffffu;
      rightSourceIds[rightSourceIdsOffset + index] = 0xffffffffu;
      forward[forwardOffset + index] = 0u;
    }
    offsets[offsetsOffset + index] = state[stateOffset + 1u];
  }`}));let Y=y(`row-heads`,`uint32`,f),X=y(`row-steps`,`uint32`,f),Z=y(`row-nodes`,`uint32`,f),Q=y(`round-distance`,`uint32`,1);_.push(i(e,{id:`${o}-row-heads`,operation:l,variant:`row-heads`,bindings:[{name:`state`,view:J,type:`u32`,access:`read`},{name:`runTable`,view:q,type:`u32`,access:`read`},{name:`vertexTotal`,view:I,type:`u32`,access:`read`},{name:`emitFinal`,view:K,type:`u32`,access:`read`},{name:`rowHeads`,view:Y,type:`u32`,access:`read_write`},{name:`rowSteps`,view:X,type:`u32`,access:`read_write`},{name:`rowNodes`,view:Z,type:`u32`,access:`read_write`}],invocationCount:f,declarations:`const SPAN_NONE: u32 = 0xffffffffu;`,body:`var head = SPAN_NONE;
  var step = 0u;
  let totalRuns = state[stateOffset];
  if (index < vertexTotal[vertexTotalOffset] && totalRuns > 0u) {
    // Last run whose first vertex row is at most index.
    var low = 0u;
    var high = totalRuns;
    while (high - low > 1u) {
      let middle = (low + high) / 2u;
      if (runTable[runTableOffset + middle * ${d}u] <= index) { low = middle; } else { high = middle; }
    }
    head = runTable[runTableOffset + low * ${d}u + 4u];
    step = index - runTable[runTableOffset + low * ${d}u];
  }
  rowHeads[rowHeadsOffset + index] = head;
  rowSteps[rowStepsOffset + index] = step;
  rowNodes[rowNodesOffset + index] = select(SPAN_NONE, emitFinal[emitFinalOffset + head], head != SPAN_NONE && step == 1u);`}),i(e,{id:`${o}-emit-successor`,operation:l,variant:`emit-successor`,bindings:[{name:`spanTable`,view:A,type:`u32`,access:`read`},{name:`emitFinal`,view:K,type:`u32`,access:`read`},{name:`successor`,view:z[0],type:`u32`,access:`read_write`}],invocationCount:v,declarations:L,body:`let next = spanWord(index, 1u);
  var result = SPAN_NONE;
  if (isSpan(index) && next != SPAN_NONE) { result = emitFinal[emitFinalOffset + next]; }
  successor[successorOffset + index] = result;`}),ee(e,{id:`${o}-round-distance`,operation:l,view:Q,type:`u32`,value:`1u`}));for(let t=0;t<H;t++)_.push(i(e,{id:`${o}-row-round-${t}`,operation:l,variant:`row-round`,bindings:[{name:`sizeTotal`,view:W,type:`u32`,access:`read`},{name:`rowHeads`,view:Y,type:`u32`,access:`read`},{name:`rowSteps`,view:X,type:`u32`,access:`read`},{name:`roundDistance`,view:Q,type:`u32`,access:`read`},{name:`successor`,view:z[t%2],type:`u32`,access:`read`},{name:`rowNodes`,view:Z,type:`u32`,access:`read_write`}],invocationCount:f,declarations:`const SPAN_NONE: u32 = 0xffffffffu;`,body:`let head = rowHeads[rowHeadsOffset + index];
  if (head == SPAN_NONE) { return; }
  let step = rowSteps[rowStepsOffset + index];
  let distance = roundDistance[roundDistanceOffset];
  // Row u = step - 1 of a chain with sizeTotal rows is filled in the round where distance <= u < 2 * distance.
  if (step <= distance || step - 1u >= 2u * distance || step > sizeTotal[sizeTotalOffset + head]) { return; }
  rowNodes[rowNodesOffset + index] = successor[successorOffset + rowNodes[rowNodesOffset + index - distance]];`}),i(e,{id:`${o}-row-distance-${t}`,operation:l,variant:`row-distance`,bindings:[{name:`roundDistance`,view:Q,type:`u32`,access:`read_write`}],invocationCount:1,body:`roundDistance[roundDistanceOffset] = roundDistance[roundDistanceOffset] * 2u;`}),te(e,{id:`${o}-row-pointer`,operation:l,variant:`row-pointer`,length:v,pointers:z,values:[]},t));_.push(i(e,{id:`${o}-write-vertices`,operation:l,variant:`write-vertices`,bindings:[{name:`state`,view:J,type:`u32`,access:`read`},{name:`spanTable`,view:A,type:`u32`,access:`read`},{name:`runRanks`,view:N,type:`u32`,access:`read`},{name:`rowHeads`,view:Y,type:`u32`,access:`read`},{name:`rowSteps`,view:X,type:`u32`,access:`read`},{name:`rowNodes`,view:Z,type:`u32`,access:`read`},{name:`outPositions`,view:h.positions,type:`f32`,access:`read_write`}],invocationCount:f,declarations:L,body:`let head = rowHeads[rowHeadsOffset + index];
  if (head == SPAN_NONE || runRanks[runRanksOffset + head] >= state[stateOffset]) { return; }
  let step = rowSteps[rowStepsOffset + index];
  // Row 0 is the start of the head span; later rows are the ends of the emitting spans in order.
  let span = select(rowNodes[rowNodesOffset + index], head, step == 0u);
  let word = select(5u, 3u, step == 0u);
  outPositions[outPositionsOffset + index * 2u] = bitcast<f32>(spanWord(span, word));
  outPositions[outPositionsOffset + index * 2u + 1u] = bitcast<f32>(spanWord(span, word + 1u));`}));let $=[{name:`state`,view:J,type:`u32`,access:`read`}],ue=[[`count`,g.count,0],[`vertexCount`,g.vertexCount,1],[`requiredCount`,g.requiredCount,2],[`requiredVertexCount`,g.requiredVertexCount,3],[`overflow`,g.overflow,4],[`candidateOverflow`,g.candidateOverflow,5]];for(let[e,t]of ue)t&&$.push({name:e,view:t,type:`u32`,access:`read_write`});return _.push(i(e,{id:`${o}-scalars`,operation:l,variant:`scalars`,bindings:$,invocationCount:1,body:ue.filter(([,e])=>e).map(([e,,t])=>`${e}[${e}Offset] = state[stateOffset + ${t}u];`).join(`
  `)})),_}};export{f as t};