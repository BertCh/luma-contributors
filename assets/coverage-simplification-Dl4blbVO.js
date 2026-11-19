import{n as e,t}from"./gpu-line-simplification-BQp8E9hM.js";import{r as n,u as r}from"./kernel-C3tfQOn4.js";import{t as i}from"./gpu-scan-B4NNTfF9.js";import{o as a,t as o}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as s,S as c,f as l,l as u,x as d,y as f}from"./gpu-contributor-utils-BIwLQON5.js";import{i as p,n as m,t as h}from"./key-pair-grouping-DgpiCtBP.js";import{i as g,t as _}from"./segment-table-BeVxjLqW.js";import{t as v}from"./gpu-segment-intersection-CGBfhMHr.js";import{t as y}from"./segment-bvh-CDgckfAv.js";import{l as b,p as x}from"./wgsl-common-DfyXiYB0.js";import{t as S}from"./exact-orientation-wgsl-B_jI-HEc.js";import{t as C}from"./exact-geometry-wgsl-y7TUnqOh.js";import{r as w,t as T}from"./coverage-topology-kernels-ltNrCaNR.js";import{s as E,v as D}from"./spatial-context-DwZEd4EZ.js";import{d as O,s as k}from"./pointer-overlay-Qh8qflcA.js";import{n as A}from"./tract-coverage-BWGWYaF1.js";import{t as j}from"./define-run-B-brkCe4.js";import{f as M,h as N,r as P}from"./lines-kx5qHdti.js";import{a as F,l as I,s as L}from"./polygons-D0hzhPvj.js";import{is as R}from"./entry-preview-D70TdSbK.js";var ee=Object.freeze({...E,measure:`distance`}),z=`GPUCoverageSimplification`,B=`${h}u`,te=`0x80000000u`,ne=4,V=4,re=V,H=3,U=2,ie=1024,W=class{id;props;spatialContext;constructor(e){let t=e.id??`coverage-simplification`;if(this.id=t,this.spatialContext=D(this.id,e.spatialContext,ee),this.props=e,s(e.positions,[`float32x2`],`${t} positions`),e.positions.length<1)throw Error(`${t} positions must hold at least one vertex`);r(e.ringOffsets,`${t} ringOffsets`),r(e.polygonOffsets,`${t} polygonOffsets`),r(e.featureOffsets,`${t} featureOffsets`);for(let[n,r]of[[`ringOffsets`,e.ringOffsets],[`polygonOffsets`,e.polygonOffsets],[`featureOffsets`,e.featureOffsets]])if(r.length<2)throw Error(`${t} ${n} must hold at least two entries`);if(e.topologyIterationCount!==void 0&&l(t,`topologyIterationCount`,e.topologyIterationCount,0,16),e.topologyPairCapacity!==void 0&&l(t,`topologyPairCapacity`,e.topologyPairCapacity),e.simplifyBoundary!==void 0&&typeof e.simplifyBoundary!=`boolean`)throw Error(`${t} simplifyBoundary must be a boolean`);if(e.snapTolerance!==void 0&&(!Number.isFinite(e.snapTolerance)||e.snapTolerance<0))throw Error(`${t} snapTolerance must be a finite number >= 0`);if(s(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<4)throw Error(`${t} parameters must hold 4 float32 values`);let{output:n}=e;if(s(n.positions,[`float32x2`],`${t} output.positions`),n.positions.length<1)throw Error(`${t} output.positions must hold at least one row`);if(r(n.ringOffsets,`${t} output.ringOffsets`),n.ringOffsets.length!==e.ringOffsets.length)throw Error(`${t} output.ringOffsets length must equal ringOffsets length`);let{overflow:i,requiredCount:a}=n.status;if(r(i,`${t} output.status.overflow`),i.length<1)throw Error(`${t} output.status.overflow must hold one uint32`);if(n.outputMask&&(r(n.outputMask,`${t} output.outputMask`),n.outputMask.length!==e.positions.length))throw Error(`${t} output.outputMask length must equal positions length`);if(n.topologyStatistics&&(r(n.topologyStatistics,`${t} output.topologyStatistics`),n.topologyStatistics.length<V))throw Error(`${t} output.topologyStatistics must hold ${V} uint32 rows`);if(a&&(r(a,`${t} output.status.requiredCount`),a.length<1))throw Error(`${t} output.status.requiredCount must hold one uint32`);if(n.status.converged&&(r(n.status.converged,`${t} output.status.converged`),n.status.converged.length<1))throw Error(`${t} output.status.converged must hold one uint32`);if(n.status.uncertainCount&&(r(n.status.uncertainCount,`${t} output.status.uncertainCount`),n.status.uncertainCount.length<1))throw Error(`${t} output.status.uncertainCount must hold one uint32`);f(t,[n.positions,n.ringOffsets,n.outputMask,i,a,n.topologyStatistics,n.status.converged,n.status.uncertainCount],[e.positions,e.ringOffsets,e.polygonOffsets,e.featureOffsets,e.parameters])}getCommandNodes(e){let{id:r,props:s}=this,{positions:l,ringOffsets:f,polygonOffsets:h,featureOffsets:v,output:x}=s;d(r,e,[l,f,h,v,s.parameters,x.status.converged,x.positions,x.ringOffsets,x.outputMask,x.status.overflow,x.status.requiredCount,x.status.uncertainCount,x.topologyStatistics]),c(r,`positions rows`,s.positions.length,g(e.device));let S=l.length,C=f.length-1,E=v.length-1,D=x.positions.length,O=s.snapTolerance??0,k=2*S,A=(t,i)=>n(e,`${r}-${t}`,`uint32`,i),j=A(`feature-ring-starts`,E+1),M=A(`vertex-high`,S),N=A(`vertex-low`,S),P=A(`vertex-polygon`,S),F=A(`vertex-next`,S),I=A(`vertex-prev`,S),L=[a(e,{id:`${r}-feature-ring-starts`,operation:z,variant:`feature-ring-starts`,bindings:[{name:`featureOffsets`,view:v,type:`u32`,access:`read`},{name:`polygonOffsets`,view:h,type:`u32`,access:`read`},{name:`featureRingStarts`,view:j,type:`u32`,access:`read_write`}],invocationCount:E+1,body:`featureRingStarts[featureRingStartsOffset + index] = polygonOffsets[polygonOffsetsOffset + featureOffsets[featureOffsetsOffset + index]];`}),a(e,{id:`${r}-vertex-topology`,operation:z,variant:`vertex-topology`,bindings:[{name:`positions`,view:l,type:`f32`,access:`read`},{name:`ringOffsets`,view:f,type:`u32`,access:`read`},{name:`featureRingStarts`,view:j,type:`u32`,access:`read`},{name:`vertexHigh`,view:M,type:`u32`,access:`read_write`},{name:`vertexLow`,view:N,type:`u32`,access:`read_write`},{name:`vertexPolygon`,view:P,type:`u32`,access:`read_write`},{name:`vertexNext`,view:F,type:`u32`,access:`read_write`},{name:`vertexPrev`,view:I,type:`u32`,access:`read_write`}],invocationCount:S,declarations:`const RING_COUNT: u32 = ${C}u;
const FEATURE_COUNT: u32 = ${E}u;

${b}
${w(O,C)}`,body:`var ring = 0u;
  var high = RING_COUNT;
  while (ring + 1u < high) {
    let middle = (ring + high) / 2u;
    if (ringOffsets[ringOffsetsOffset + middle] <= index) {
      ring = middle;
    } else {
      high = middle;
    }
  }
  // The coverage identity of a ring is its feature: first ring of feature f is featureRingStarts[f].
  var polygon = 0u;
  high = FEATURE_COUNT;
  while (polygon + 1u < high) {
    let middle = (polygon + high) / 2u;
    if (featureRingStarts[featureRingStartsOffset + middle] <= ring) {
      polygon = middle;
    } else {
      high = middle;
    }
  }
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let ringBegin = ringOffsets[ringOffsetsOffset + ring];
  let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
  let inRing = index >= ringBegin && index < ringEnd &&
    ring >= featureRingStarts[featureRingStartsOffset] && ring < featureRingStarts[featureRingStartsOffset + FEATURE_COUNT] &&
    ring < featureRingStarts[featureRingStartsOffset + polygon + 1u];
  let valid = inRing && isFiniteFloat(x) && isFiniteFloat(y);
  let snapOrigin = getSnapOrigin();
  vertexHigh[vertexHighOffset + index] = select(${B}, quantise(x, snapOrigin.x), valid);
  vertexLow[vertexLowOffset + index] = select(${B}, quantise(y, snapOrigin.y), valid);
  vertexPolygon[vertexPolygonOffset + index] = select(${B}, polygon, valid);
  vertexNext[vertexNextOffset + index] = select(index + 1u, ringBegin, index + 1u == ringEnd);
  let previous = select(index - 1u, ringEnd - 1u, index == ringBegin);
  vertexPrev[vertexPrevOffset + index] = select(previous, previous | ${te}, index == ringBegin);`})],R=p(e,`${r}-vertex`,z,S,M,N),ee=m(e,`${r}-vertex`,z,S,M,N,R.sortedItems),H=A(`point-ids`,S),U=A(`edge-high`,S),W=A(`edge-low`,S);L.push(...R.nodes,...ee.nodes,a(e,{id:`${r}-point-ids`,operation:z,variant:`point-ids`,bindings:[{name:`sortedItems`,view:R.sortedItems,type:`u32`,access:`read`},{name:`groupIndex`,view:ee.groupIndex,type:`u32`,access:`read`},{name:`vertexPolygon`,view:P,type:`u32`,access:`read`},{name:`pointIds`,view:H,type:`u32`,access:`read_write`}],invocationCount:S,body:`let vertex = sortedItems[sortedItemsOffset + index];
  pointIds[pointIdsOffset + vertex] = select(${B},
    groupIndex[groupIndexOffset + index] - 1u, vertexPolygon[vertexPolygonOffset + vertex] != ${B});`}),a(e,{id:`${r}-edge-keys`,operation:z,variant:`edge-keys`,bindings:[{name:`vertexPolygon`,view:P,type:`u32`,access:`read`},{name:`vertexNext`,view:F,type:`u32`,access:`read`},{name:`pointIds`,view:H,type:`u32`,access:`read`},{name:`edgeHigh`,view:U,type:`u32`,access:`read_write`},{name:`edgeLow`,view:W,type:`u32`,access:`read_write`}],invocationCount:S,body:`let first = pointIds[pointIdsOffset + index];
  let second = pointIds[pointIdsOffset + vertexNext[vertexNextOffset + index]];
  let valid = first != ${B} && second != ${B} && first != second &&
    vertexPolygon[vertexPolygonOffset + index] != ${B};
  edgeHigh[edgeHighOffset + index] = select(${B}, min(first, second), valid);
  edgeLow[edgeLowOffset + index] = select(${B}, max(first, second), valid);`}));let G=p(e,`${r}-edge`,z,S,U,W),K=m(e,`${r}-edge`,z,S,U,W,G.sortedItems);L.push(...G.nodes,...K.nodes);let q=A(`edge-partner`,S);L.push(a(e,{id:`${r}-edge-partner`,operation:z,variant:`edge-partner`,bindings:[{name:`sortedItems`,view:G.sortedItems,type:`u32`,access:`read`},{name:`edgeHigh`,view:U,type:`u32`,access:`read`},{name:`vertexPolygon`,view:P,type:`u32`,access:`read`},{name:`groupIndex`,view:K.groupIndex,type:`u32`,access:`read`},{name:`groupStarts`,view:K.groupStarts,type:`u32`,access:`read`},{name:`edgePartner`,view:q,type:`u32`,access:`read_write`}],invocationCount:S,body:`let item = sortedItems[sortedItemsOffset + index];
  let polygon = vertexPolygon[vertexPolygonOffset + item];
  var partner = ${B};
  if (polygon != ${B} && edgeHigh[edgeHighOffset + item] != ${B}) {
    let groupId = groupIndex[groupIndexOffset + index] - 1u;
    let end = groupStarts[groupStartsOffset + groupId + 1u];
    for (var other = groupStarts[groupStartsOffset + groupId]; other < end; other++) {
      let otherPolygon = vertexPolygon[vertexPolygonOffset + sortedItems[sortedItemsOffset + other]];
      if (otherPolygon != ${B} && otherPolygon != polygon) {
        partner = min(partner, otherPolygon);
      }
    }
  }
  edgePartner[edgePartnerOffset + item] = partner;`}));let ue=A(`arc-flags`,S),de=A(`start-flag`,S),fe=A(`start-rank`,S),pe=A(`start-list`,S+1),J=A(`owned-length`,S),me=A(`owned-flag`,S);L.push(a(e,{id:`${r}-arc-flags`,operation:z,variant:`arc-flags`,bindings:[{name:`vertexPrev`,view:I,type:`u32`,access:`read`},{name:`vertexPolygon`,view:P,type:`u32`,access:`read`},{name:`edgePartner`,view:q,type:`u32`,access:`read`},{name:`arcFlags`,view:ue,type:`u32`,access:`read_write`},{name:`startFlag`,view:de,type:`u32`,access:`read_write`}],invocationCount:S,body:`let prevEntry = vertexPrev[vertexPrevOffset + index];
  let previous = prevEntry & 0x7fffffffu;
  let ringBegin = (prevEntry & ${te}) != 0u;
  let valid = vertexPolygon[vertexPolygonOffset + index] != ${B};
  let junction = valid && edgePartner[edgePartnerOffset + previous] != edgePartner[edgePartnerOffset + index];
  arcFlags[arcFlagsOffset + index] = select(0u, 1u, ringBegin || junction) | select(0u, 2u, junction);
  startFlag[startFlagOffset + index] = select(0u, 1u, ringBegin || junction);`}),...new i({id:`${r}-start-rank-scan`,input:de,output:fe,mode:`exclusive`}).getCommandNodes(e),a(e,{id:`${r}-start-list`,operation:z,variant:`start-list`,bindings:[{name:`startFlag`,view:de,type:`u32`,access:`read`},{name:`startRank`,view:fe,type:`u32`,access:`read`},{name:`ringOffsets`,view:f,type:`u32`,access:`read`},{name:`startList`,view:pe,type:`u32`,access:`read_write`}],invocationCount:S,declarations:`const VERTEX_COUNT: u32 = ${S}u;
const RING_COUNT: u32 = ${C}u;`,body:`let flag = startFlag[startFlagOffset + index];
  let rank = startRank[startRankOffset + index];
  if (flag != 0u) {
    startList[startListOffset + rank] = index;
  }
  if (index == VERTEX_COUNT - 1u) {
    startList[startListOffset + rank + flag] = min(ringOffsets[ringOffsetsOffset + RING_COUNT], VERTEX_COUNT);
  }`}),a(e,{id:`${r}-arc-lengths`,operation:z,variant:`arc-lengths`,bindings:[{name:`arcFlags`,view:ue,type:`u32`,access:`read`},{name:`startRank`,view:fe,type:`u32`,access:`read`},{name:`startList`,view:pe,type:`u32`,access:`read`},{name:`vertexPolygon`,view:P,type:`u32`,access:`read`},{name:`edgePartner`,view:q,type:`u32`,access:`read`},{name:`ownedLength`,view:J,type:`u32`,access:`read_write`},{name:`ownedFlag`,view:me,type:`u32`,access:`read_write`}],invocationCount:S,body:`var length = 0u;
  var owned = 0u;
  let polygon = vertexPolygon[vertexPolygonOffset + index];
  if ((arcFlags[arcFlagsOffset + index] & 1u) != 0u && polygon != ${B}) {
    let partner = edgePartner[edgePartnerOffset + index];
    if (partner == ${B} || polygon < partner) {
      owned = 1u;
      // The arc runs to the next arc start in vertex order, or to the end of the ring, whose
      // first vertex is itself an arc start. The closing vertex is repeated.
      let arcEnd = startList[startListOffset + startRank[startRankOffset + index] + 1u];
      length = arcEnd - index + 1u;
    }
  }
    ownedLength[ownedLengthOffset + index] = length;
  ownedFlag[ownedFlagOffset + index] = owned;`}));let he=A(`arc-index`,S),Y=A(`line-start`,S),X=A(`track-offsets`,S+1),ge=n(e,`${r}-line-positions`,`float32x2`,k),_e=A(`line-point`,k);L.push(...new i({id:`${r}-arc-index-scan`,input:me,output:he,mode:`exclusive`}).getCommandNodes(e),...new i({id:`${r}-line-start-scan`,input:J,output:Y,mode:`exclusive`}).getCommandNodes(e),a(e,{id:`${r}-track-offsets-fill`,operation:z,variant:`track-offsets-fill`,bindings:[{name:`lineStart`,view:Y,type:`u32`,access:`read`},{name:`ownedLength`,view:J,type:`u32`,access:`read`},{name:`trackOffsets`,view:X,type:`u32`,access:`read_write`}],invocationCount:S+1,declarations:`const VERTEX_COUNT: u32 = ${S}u;`,body:`trackOffsets[trackOffsetsOffset + index] =
    lineStart[lineStartOffset + VERTEX_COUNT - 1u] + ownedLength[ownedLengthOffset + VERTEX_COUNT - 1u];`}),a(e,{id:`${r}-track-offsets`,operation:z,variant:`track-offsets`,bindings:[{name:`ownedFlag`,view:me,type:`u32`,access:`read`},{name:`arcIndex`,view:he,type:`u32`,access:`read`},{name:`lineStart`,view:Y,type:`u32`,access:`read`},{name:`trackOffsets`,view:X,type:`u32`,access:`read_write`}],invocationCount:S,body:`if (ownedFlag[ownedFlagOffset + index] != 0u) {
    trackOffsets[trackOffsetsOffset + arcIndex[arcIndexOffset + index]] = lineStart[lineStartOffset + index];
  }`}),a(e,{id:`${r}-copy-arcs`,operation:z,variant:`copy-arcs`,bindings:[{name:`positions`,view:l,type:`f32`,access:`read`},{name:`vertexNext`,view:F,type:`u32`,access:`read`},{name:`ownedLength`,view:J,type:`u32`,access:`read`},{name:`lineStart`,view:Y,type:`u32`,access:`read`},{name:`pointIds`,view:H,type:`u32`,access:`read`},{name:`linePositions`,view:ge,type:`f32`,access:`read_write`},{name:`linePoint`,view:_e,type:`u32`,access:`read_write`}],invocationCount:k,declarations:`const VERTEX_COUNT: u32 = ${S}u;`,body:`let total = lineStart[lineStartOffset + VERTEX_COUNT - 1u] + ownedLength[ownedLengthOffset + VERTEX_COUNT - 1u];
  if (index < total) {
    var low = 0u;
    var high = VERTEX_COUNT;
    while (low + 1u < high) {
      let middle = (low + high) / 2u;
      if (lineStart[lineStartOffset + middle] <= index) {
        low = middle;
      } else {
        high = middle;
      }
    }
    let step = index - lineStart[lineStartOffset + low];
    let vertex = select(vertexNext[vertexNextOffset + low + step - 1u], low, step == 0u);
    let pointId = pointIds[pointIdsOffset + vertex];
    let x = positions[positionsOffset + vertex * 2u];
    let y = positions[positionsOffset + vertex * 2u + 1u];
    let finite = pointId != ${B};
    linePositions[linePositionsOffset + index * 2u] = select(0.0, x, finite);
    linePositions[linePositionsOffset + index * 2u + 1u] = select(0.0, y, finite);
    linePoint[linePointOffset + index] = pointId;
  }`}));let ve=n(e,`${r}-line-importance`,`float32`,k),ye=A(`line-keep`,k),be=A(`selection-ids`,k),xe=A(`selection-count`,1),Se=A(`selection-overflow`,1),Ce={id:`${r}-arcs`,spatialContext:this.spatialContext,positions:ge,lineOffsets:X,importance:ve,maximumIterations:s.maximumIterations,parameters:s.parameters,output:{ids:be,outputMask:ye,status:{count:xe,overflow:Se,...x.status.converged?{converged:x.status.converged}:{}}}};L.push(...new t(Ce).getCommandNodes(e));let Z=A(`keep-point`,S);L.push(o(e,{id:`${r}-keep-point-clear`,operation:z,view:Z,type:`u32`,value:`0u`}),a(e,{id:`${r}-mark-junctions`,operation:z,variant:`mark-junctions`,bindings:[{name:`arcFlags`,view:ue,type:`u32`,access:`read`},{name:`pointIds`,view:H,type:`u32`,access:`read`},{name:`vertexPrev`,view:I,type:`u32`,access:`read`},{name:`edgePartner`,view:q,type:`u32`,access:`read`},{name:`keepPoint`,view:Z,type:`atomic<u32>`,access:`read_write`}],invocationCount:S,declarations:`const KEEP_BOUNDARY: bool = ${s.simplifyBoundary===!1};`,body:`let pointId = pointIds[pointIdsOffset + index];
  let previous = vertexPrev[vertexPrevOffset + index] & 0x7fffffffu;
  let isBoundary = KEEP_BOUNDARY && (edgePartner[edgePartnerOffset + index] == ${B} ||
    edgePartner[edgePartnerOffset + previous] == ${B});
  if (((arcFlags[arcFlagsOffset + index] & 2u) != 0u || isBoundary) && pointId != ${B}) {
    atomicMax(&keepPoint[keepPointOffset + pointId], 1u);
  }`}),a(e,{id:`${r}-mark-arcs`,operation:z,variant:`mark-arcs`,bindings:[{name:`lineKeep`,view:ye,type:`u32`,access:`read`},{name:`linePoint`,view:_e,type:`u32`,access:`read`},{name:`trackOffsets`,view:X,type:`u32`,access:`read`},{name:`keepPoint`,view:Z,type:`atomic<u32>`,access:`read_write`}],invocationCount:k,declarations:`const VERTEX_COUNT: u32 = ${S}u;`,body:`let pointId = linePoint[linePointOffset + index];
  if (index < trackOffsets[trackOffsetsOffset + VERTEX_COUNT] && lineKeep[lineKeepOffset + index] != 0u &&
      pointId != ${B}) {
    atomicMax(&keepPoint[keepPointOffset + pointId], 1u);
  }`}));let Q=s.topologyIterationCount??ne;if(L.push(...ce(e,{id:r,positions:l,ringOffsets:f,vertexPolygon:P,pointIds:H,keepPoint:Z,vertexCount:S,ringCount:C})),Q===0&&x.topologyStatistics&&L.push(o(e,{id:`${r}-topology-stats-zero`,operation:z,view:x.topologyStatistics,type:`u32`,value:`0u`})),Q>0){let t=A(`topology-work`,5),i=A(`owned-vertex`,S),c=s.topologyPairCapacity??Math.max(256,4*S);L.push(o(e,{id:`${r}-topology-stats-clear`,operation:z,view:t,type:`u32`,value:`0u`}),a(e,{id:`${r}-owned-vertex`,operation:z,variant:`owned-vertex`,bindings:[{name:`vertexPolygon`,view:P,type:`u32`,access:`read`},{name:`edgePartner`,view:q,type:`u32`,access:`read`},{name:`ownedVertex`,view:i,type:`u32`,access:`read_write`}],invocationCount:S,body:`let polygon = vertexPolygon[vertexPolygonOffset + index];
  let partner = edgePartner[edgePartnerOffset + index];
  ownedVertex[ownedVertexOffset + index] =
    select(0u, select(0u, 1u, partner == ${B} || polygon < partner), polygon != ${B});`}));let d=A(`sweep-segment-table`,S*8),p=n(e,`${r}-sweep-minima`,`float32x2`,S),m=n(e,`${r}-sweep-maxima`,`float32x2`,S);L.push(_(e,{id:`${r}-sweep-segment-table`,operation:z,geometry:{kind:`polygons`,positions:l,ringOffsets:f,polygonOffsets:h,featureOffsets:v},table:d,minima:p,maxima:m}));let{bvh:g,nodes:b}=y(e,{id:`${r}-sweep`,operation:z,minima:p,maxima:m,leafCapacity:u(S),spatialSort:S>=ie});L.push(...b);let w=x.topologyStatistics?Q:Q-1;for(let n=0;n<=w;n++){let a=n===0,o=n===Q;L.push(...le(e,{id:`${r}-round-${n}`,positions:l,ringOffsets:f,polygonOffsets:h,featureOffsets:v,pointIds:H,vertexNext:F,ownedVertex:i,keepPoint:Z,sweepBVH:{nodeMinima:g.nodeMinima,nodeMaxima:g.nodeMaxima,leafIds:g.leafIds,internalNodeCount:g.internalNodeCount},vertexCount:S,ringCount:C,pairCapacity:c,repair:!o,statsView:t,crossingSlot:a?0:o?1:re}))}x.topologyStatistics&&L.push(a(e,{id:`${r}-topology-stats-copy`,operation:z,variant:`topology-stats-copy`,bindings:[{name:`workStats`,view:t,type:`u32`,access:`read`},{name:`topologyStatistics`,view:x.topologyStatistics,type:`u32`,access:`read_write`}],invocationCount:V,body:`topologyStatistics[topologyStatisticsOffset + index] = workStats[workStatsOffset + index];`}))}let $=x.outputMask??A(`keep-mask`,S),we=A(`keep-rank`,S);return L.push(ae(e,`${r}-keep-mask`,H,Z,$),...new i({id:`${r}-keep-rank-scan`,input:$,output:we,mode:`exclusive`}).getCommandNodes(e),oe(e,{id:`${r}-emit-positions`,positions:l,keepMask:$,keepRank:we,outputPositions:x.positions,vertexCount:S}),se(e,{id:`${r}-emit-offsets`,ringOffsets:f,keepMask:$,keepRank:we,outputRingOffsets:x.ringOffsets,overflow:x.status.overflow,requiredCount:x.status.requiredCount,vertexCount:S,ringCount:C,capacity:D})),x.status.uncertainCount&&L.push(...T(e,{id:`${r}-snap-range`,operation:z,positions:l,ringOffsets:f,snapTolerance:O,uncertainCount:x.status.uncertainCount})),L}};function ae(e,t,n,r,i){return a(e,{id:t,operation:z,variant:`keep-mask`,bindings:[{name:`pointIds`,view:n,type:`u32`,access:`read`},{name:`keepPoint`,view:r,type:`u32`,access:`read`},{name:`keepMask`,view:i,type:`u32`,access:`read_write`}],invocationCount:n.length,body:`let pointId = pointIds[pointIdsOffset + index];
  keepMask[keepMaskOffset + index] =
    select(0u, keepPoint[keepPointOffset + pointId], pointId != ${B});`})}function oe(e,t){return a(e,{id:t.id,operation:z,variant:`emit-positions`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`keepMask`,view:t.keepMask,type:`u32`,access:`read`},{name:`keepRank`,view:t.keepRank,type:`u32`,access:`read`},{name:`outputPositions`,view:t.outputPositions,type:`f32`,access:`read_write`}],invocationCount:t.vertexCount,declarations:`const CAPACITY: u32 = ${t.outputPositions.length}u;`,body:`let rank = keepRank[keepRankOffset + index];
  if (keepMask[keepMaskOffset + index] != 0u && rank < CAPACITY) {
    outputPositions[outputPositionsOffset + rank * 2u] = positions[positionsOffset + index * 2u];
    outputPositions[outputPositionsOffset + rank * 2u + 1u] = positions[positionsOffset + index * 2u + 1u];
  }`})}function se(e,t){let{requiredCount:n}=t;return a(e,{id:t.id,operation:z,variant:`emit-offsets`,bindings:[{name:`ringOffsets`,view:t.ringOffsets,type:`u32`,access:`read`},{name:`keepMask`,view:t.keepMask,type:`u32`,access:`read`},{name:`keepRank`,view:t.keepRank,type:`u32`,access:`read`},{name:`outputRingOffsets`,view:t.outputRingOffsets,type:`u32`,access:`read_write`},{name:`overflow`,view:t.overflow,type:`u32`,access:`read_write`},...n?[{name:`requiredCount`,view:n,type:`u32`,access:`read_write`}]:[]],invocationCount:t.ringCount+1,declarations:`const VERTEX_COUNT: u32 = ${t.vertexCount}u;
const RING_COUNT: u32 = ${t.ringCount}u;
const CAPACITY: u32 = ${t.capacity}u;`,body:`let total = keepRank[keepRankOffset + VERTEX_COUNT - 1u] + keepMask[keepMaskOffset + VERTEX_COUNT - 1u];
  var begin = total;
  if (index < RING_COUNT) {
    let ringBegin = ringOffsets[ringOffsetsOffset + index];
    if (ringBegin < VERTEX_COUNT) {
      begin = keepRank[keepRankOffset + ringBegin];
    }
  }
  outputRingOffsets[outputRingOffsetsOffset + index] = min(begin, CAPACITY);
  if (index == RING_COUNT) {
    overflow[overflowOffset] = select(0u, 1u, total > CAPACITY);
    ${n?`requiredCount[requiredCountOffset] = total;`:``}
  }`})}function ce(e,t){let{id:r,vertexCount:i,ringCount:s}=t,c=n(e,`${r}-restore-point`,`uint32`,i);return[o(e,{id:`${r}-restore-point-clear`,operation:z,view:c,type:`u32`,value:`0u`}),a(e,{id:`${r}-ring-minimum`,operation:z,variant:`ring-minimum`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:t.ringOffsets,type:`u32`,access:`read`},{name:`pointIds`,view:t.pointIds,type:`u32`,access:`read`},{name:`keepPoint`,view:t.keepPoint,type:`u32`,access:`read`},{name:`restorePoint`,view:c,type:`atomic<u32>`,access:`read_write`}],invocationCount:s,declarations:`const VERTEX_COUNT: u32 = ${i}u;
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
fn isKept(vertex: u32) -> bool {
  let pointId = pointIds[pointIdsOffset + vertex];
  return pointId != ${B} && keepPoint[keepPointOffset + pointId] != 0u;
}`,body:`let begin = ringOffsets[ringOffsetsOffset + index];
  var end = min(ringOffsets[ringOffsetsOffset + index + 1u], VERTEX_COUNT);
  if (end > begin + 1u && pointIds[pointIdsOffset + end - 1u] == pointIds[pointIdsOffset + begin]) {
    end = end - 1u;
  }
  var keptCount = 0u;
  var firstKept = ${B};
  var secondKept = ${B};
  for (var vertex = begin; vertex < end; vertex++) {
    if (isKept(vertex)) {
      if (keptCount == 0u) { firstKept = vertex; } else if (keptCount == 1u) { secondKept = vertex; }
      keptCount++;
    }
  }
  if (keptCount >= 1u && keptCount < 3u && end > begin + 2u) {
    let anchor = vertexAt(firstKept);
    var other = vertexAt(firstKept);
    var otherVertex = secondKept;
    var haveOther = keptCount >= 2u;
    if (haveOther) {
      other = vertexAt(secondKept);
    } else {
      var best = 0.0;
      for (var vertex = begin; vertex < end; vertex++) {
        if (!isKept(vertex) && pointIds[pointIdsOffset + vertex] != ${B}) {
          let distanceToAnchor = distance(vertexAt(vertex), anchor);
          if (distanceToAnchor > best) { best = distanceToAnchor; otherVertex = vertex; }
        }
      }
      if (otherVertex != ${B}) {
        haveOther = true;
        other = vertexAt(otherVertex);
        atomicMax(&restorePoint[restorePointOffset + pointIds[pointIdsOffset + otherVertex]], 1u);
      }
    }
    if (haveOther) {
      let direction = other - anchor;
      let directionLength = max(length(direction), 1e-30);
      var best = 0.0;
      var thirdVertex = ${B};
      for (var vertex = begin; vertex < end; vertex++) {
        let pointId = pointIds[pointIdsOffset + vertex];
        if (!isKept(vertex) && pointId != ${B} && vertex != otherVertex &&
            pointId != pointIds[pointIdsOffset + otherVertex]) {
          let offset = vertexAt(vertex) - anchor;
          let distanceToLine = abs(direction.x * offset.y - direction.y * offset.x) / directionLength;
          if (distanceToLine > best) { best = distanceToLine; thirdVertex = vertex; }
        }
      }
      if (thirdVertex != ${B}) {
        atomicMax(&restorePoint[restorePointOffset + pointIds[pointIdsOffset + thirdVertex]], 1u);
      }
    }
  }`}),a(e,{id:`${r}-merge-restored`,operation:z,variant:`merge-restored`,bindings:[{name:`restorePoint`,view:c,type:`u32`,access:`read`},{name:`keepPoint`,view:t.keepPoint,type:`u32`,access:`read_write`}],invocationCount:i,body:`keepPoint[keepPointOffset + index] = max(keepPoint[keepPointOffset + index], restorePoint[restorePointOffset + index]);`})]}function G(e,t){let{id:r,vertexCount:i,ringCount:s,keepMask:c,keepRank:l,ringOffsets:u}=t,d=n(e,`${r}-kept-vertex`,`uint32`,i),f=n(e,`${r}-distance-bits`,`uint32`,i),p=n(e,`${r}-span-best`,`uint32`,i),m=`const VERTEX_COUNT: u32 = ${i}u;
const RING_COUNT: u32 = ${s}u;
fn getRing(vertex: u32) -> u32 {
  var ring = 0u;
  var high = RING_COUNT;
  while (ring + 1u < high) {
    let middle = (ring + high) / 2u;
    if (ringOffsets[ringOffsetsOffset + middle] <= vertex) {
      ring = middle;
    } else {
      high = middle;
    }
  }
  return ring;
}
fn getKeptTotal() -> u32 {
  return keepRank[keepRankOffset + VERTEX_COUNT - 1u] + keepMask[keepMaskOffset + VERTEX_COUNT - 1u];
}
// Number of kept vertices before 'position'.
fn getKeptBefore(position: u32) -> u32 {
  return select(keepRank[keepRankOffset + position], getKeptTotal(), position >= VERTEX_COUNT);
}`,h=(e,t)=>({name:e,view:t,type:`u32`,access:`read`});return[a(e,{id:`${r}-kept-vertex`,operation:z,variant:`kept-vertex`,bindings:[h(`keepMask`,c),h(`keepRank`,l),{name:`keptVertex`,view:d,type:`u32`,access:`read_write`}],invocationCount:i,body:`if (keepMask[keepMaskOffset + index] != 0u) {
    keptVertex[keptVertexOffset + keepRank[keepRankOffset + index]] = index;
  }`}),a(e,{id:`${r}-span-ends`,operation:z,variant:`span-ends`,bindings:[h(`ringOffsets`,u),h(`keepMask`,c),h(`keepRank`,l),h(`keptVertex`,d),{name:`spanEnd`,view:t.spanEnd,type:`u32`,access:`read_write`}],invocationCount:i,declarations:m,body:`var end = ${B};
  if (keepMask[keepMaskOffset + index] != 0u) {
    let ring = getRing(index);
    let ringBegin = ringOffsets[ringOffsetsOffset + ring];
    let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
    let rank = keepRank[keepRankOffset + index];
    end = keptVertex[keptVertexOffset + getKeptBefore(ringBegin)];
    if (rank + 1u < getKeptTotal()) {
      let candidate = keptVertex[keptVertexOffset + rank + 1u];
      if (candidate < ringEnd) {
        end = candidate;
      }
    }
  }
  spanEnd[spanEndOffset + index] = end;`}),o(e,{id:`${r}-span-best-clear`,operation:z,view:p,type:`u32`,value:`0u`}),o(e,{id:`${r}-span-key-clear`,operation:z,view:t.spanKey,type:`u32`,value:B}),a(e,{id:`${r}-span-distance`,operation:z,variant:`span-distance`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},h(`ringOffsets`,u),h(`keepMask`,c),h(`keepRank`,l),h(`keptVertex`,d),h(`spanEnd`,t.spanEnd),{name:`distanceBits`,view:f,type:`u32`,access:`read_write`},{name:`spanBest`,view:p,type:`atomic<u32>`,access:`read_write`}],invocationCount:i,declarations:`${m}
${K}
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}`,body:`var bits = 0u;
  if (keepMask[keepMaskOffset + index] == 0u) {
    let owner = getSpanOwner(index);
    if (owner != ${B}) {
      let a = vertexAt(owner);
      let p = vertexAt(index);
      let direction = vertexAt(spanEnd[spanEndOffset + owner]) - a;
      let segmentLength = length(direction);
      let offset = p - a;
      let separation = select(
        distance(p, a),
        abs(direction.x * offset.y - direction.y * offset.x) / max(segmentLength, 1e-30),
        segmentLength > 0.0);
      if (separation > 0.0) {
        bits = bitcast<u32>(separation);
        atomicMax(&spanBest[spanBestOffset + owner], bits);
      }
    }
  }
  distanceBits[distanceBitsOffset + index] = bits;`}),a(e,{id:`${r}-span-split`,operation:z,variant:`span-split`,bindings:[h(`ringOffsets`,u),h(`keepMask`,c),h(`keepRank`,l),h(`keptVertex`,d),h(`distanceBits`,f),h(`spanBest`,p),{name:`spanKey`,view:t.spanKey,type:`atomic<u32>`,access:`read_write`}],invocationCount:i,declarations:`${m}
${K}`,body:`let bits = distanceBits[distanceBitsOffset + index];
  if (bits != 0u && keepMask[keepMaskOffset + index] == 0u) {
    let owner = getSpanOwner(index);
    if (owner != ${B} && bits == spanBest[spanBestOffset + owner]) {
      atomicMin(&spanKey[spanKeyOffset + owner], select(index + VERTEX_COUNT, index, index > owner));
    }
  }`}),a(e,{id:`${r}-span-links`,operation:z,variant:`span-links`,bindings:[h(`ringOffsets`,u),h(`keepMask`,c),h(`keepRank`,l),h(`keptVertex`,d),h(`spanEnd`,t.spanEnd),h(`spanBest`,p),{name:`spanLinks`,view:t.spanLinks,type:`u32`,access:`read_write`}],invocationCount:i,declarations:`${m}
${K}`,body:`var owner = index;
  if (keepMask[keepMaskOffset + index] == 0u) { owner = getSpanOwner(index); }
  var edgeOwner = ${B};
  if (owner != ${B}) {
    let ring = getRing(owner);
    let ringBegin = ringOffsets[ringOffsetsOffset + ring];
    let ringEnd = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
    if (owner >= ringBegin && owner < ringEnd) {
      let after = select(owner + 1u, ringBegin, owner + 1u >= ringEnd);
      if (keepMask[keepMaskOffset + after] == 0u) { edgeOwner = owner; }
    }
  }
  let base = spanLinksOffset + index * ${H}u;
  spanLinks[base] = edgeOwner;
  spanLinks[base + 1u] = spanEnd[spanEndOffset + index];
  spanLinks[base + 2u] = spanBest[spanBestOffset + index];`})]}var K=`
fn getSpanOwner(vertex: u32) -> u32 {
  let ring = getRing(vertex);
  let ringBegin = ringOffsets[ringOffsetsOffset + ring];
  let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
  let firstRank = getKeptBefore(ringBegin);
  let endRank = getKeptBefore(ringEnd);
  if (vertex < ringBegin || vertex >= ringEnd || endRank <= firstRank) {
    return ${B};
  }
  // The kept vertex before it in the ring, else the ring's last one (the span wraps over the end).
  let before = keepRank[keepRankOffset + vertex];
  return keptVertex[keptVertexOffset + select(endRank - 1u, before - 1u, before > firstRank)];
}`;function le(e,t){let{id:r,vertexCount:s,ringCount:c,pairCapacity:l,statsView:u}=t,d=(t,i)=>n(e,`${r}-${t}`,`uint32`,i),f=d(`mask`,s),p=d(`rank`,s),m=n(e,`${r}-compact-positions`,`float32x2`,s),h=d(`compact-ring-offsets`,c+1),g=d(`compact-overflow`,1),_=d(`span-end`,s),y=d(`span-split`,s),b=d(`span-links`,s*H),w=d(`row-info`,s*4),T=d(`bad`,s),E=d(`pair-left`,l),D=d(`pair-right`,l),O=d(`pair-count`,1),k=d(`pair-overflow`,1),A=d(`pair-kinds`,l),{crossingSlot:j}=t,M=[ae(e,`${r}-keep-mask`,t.pointIds,t.keepPoint,f),...new i({id:`${r}-rank-scan`,input:f,output:p,mode:`exclusive`}).getCommandNodes(e),oe(e,{id:`${r}-emit-positions`,positions:t.positions,keepMask:f,keepRank:p,outputPositions:m,vertexCount:s}),se(e,{id:`${r}-emit-offsets`,ringOffsets:t.ringOffsets,keepMask:f,keepRank:p,outputRingOffsets:h,overflow:g,vertexCount:s,ringCount:c,capacity:s}),...G(e,{id:r,positions:t.positions,ringOffsets:t.ringOffsets,keepMask:f,keepRank:p,spanEnd:_,spanKey:y,spanLinks:b,vertexCount:s,ringCount:c}),a(e,{id:`${r}-row-info`,operation:z,variant:`row-info`,bindings:[{name:`keepMask`,view:f,type:`u32`,access:`read`},{name:`keepRank`,view:p,type:`u32`,access:`read`},{name:`spanEnd`,view:_,type:`u32`,access:`read`},{name:`pointIds`,view:t.pointIds,type:`u32`,access:`read`},{name:`ownedVertex`,view:t.ownedVertex,type:`u32`,access:`read`},{name:`rowInfo`,view:w,type:`u32`,access:`read_write`}],invocationCount:s,body:`if (keepMask[keepMaskOffset + index] != 0u) {
    let base = rowInfoOffset + keepRank[keepRankOffset + index] * 4u;
    rowInfo[base] = pointIds[pointIdsOffset + index];
    rowInfo[base + 1u] = pointIds[pointIdsOffset + spanEnd[spanEndOffset + index]];
    rowInfo[base + 2u] = index;
    rowInfo[base + 3u] = ownedVertex[ownedVertexOffset + index];
  }`}),o(e,{id:`${r}-bad-clear`,operation:z,view:T,type:`u32`,value:`0u`}),...new v({id:`${r}-intersect`,left:{kind:`polygons`,positions:m,featureOffsets:t.featureOffsets,polygonOffsets:t.polygonOffsets,ringOffsets:h},output:{pairs:{leftRows:E,rightRows:D,status:{count:O,overflow:k}},kinds:A}}).getCommandNodes(e),a(e,{id:`${r}-classify`,operation:z,variant:`classify-${j}`,bindings:[{name:`pairLeft`,view:E,type:`u32`,access:`read`},{name:`pairRight`,view:D,type:`u32`,access:`read`},{name:`pairKinds`,view:A,type:`u32`,access:`read`},{name:`pairCount`,view:O,type:`u32`,access:`read`},{name:`pairOverflow`,view:k,type:`u32`,access:`read`},{name:`rowInfo`,view:w,type:`u32`,access:`read`},{name:`badFlag`,view:T,type:`atomic<u32>`,access:`read_write`},{name:`stats`,view:u,type:`atomic<u32>`,access:`read_write`}],invocationCount:l,declarations:`const SLOT: u32 = ${j}u;`,body:`if (index == 0u && pairOverflow[pairOverflowOffset] != 0u) {
    atomicMax(&stats[statsOffset + 3u], 1u);
  }
  if (index < pairCount[pairCountOffset]) {
    let kind = pairKinds[pairKindsOffset + index];
    let leftBase = rowInfoOffset + pairLeft[pairLeftOffset + index] * 4u;
    let rightBase = rowInfoOffset + pairRight[pairRightOffset + index] * 4u;
    let a0 = rowInfo[leftBase];
    let a1 = rowInfo[leftBase + 1u];
    let b0 = rowInfo[rightBase];
    let b1 = rowInfo[rightBase + 1u];
    let shares = a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1;
    let identical = (a0 == b0 && a1 == b1) || (a0 == b1 && a1 == b0);
    let offending = select(!shares, !identical, kind == 4u);
    if (offending) {
      atomicMax(&badFlag[badFlagOffset + rowInfo[leftBase + 2u]], 1u);
      atomicMax(&badFlag[badFlagOffset + rowInfo[rightBase + 2u]], 1u);
      if (rowInfo[leftBase + 3u] != 0u && rowInfo[rightBase + 3u] != 0u) {
        atomicAdd(&stats[statsOffset + SLOT], 1u);
      }
    }
  }`}),a(e,{id:`${r}-sweep`,operation:z,variant:`sweep`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`nodeMinima`,view:t.sweepBVH.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:t.sweepBVH.nodeMaxima,type:`f32`,access:`read`},{name:`leafIds`,view:t.sweepBVH.leafIds,type:`u32`,access:`read`},{name:`vertexNext`,view:t.vertexNext,type:`u32`,access:`read`},{name:`spanLinks`,view:b,type:`u32`,access:`read`},{name:`pointIds`,view:t.pointIds,type:`u32`,access:`read`},{name:`badFlag`,view:T,type:`atomic<u32>`,access:`read_write`}],invocationCount:s,declarations:`const VERTEX_COUNT: u32 = ${s}u;
const INTERNAL_NODE_COUNT: u32 = ${t.sweepBVH.internalNodeCount}u;
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
${x}
${S}
${C}
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
// Whether 'point' can lie in the area between the span's chain and its chord: that area is within
// the span's reach (its farthest dropped vertex) of the chord's line. f32, with slack.
fn isWithinSpanReach(point: vec2f, owner: u32, spanEndVertex: u32) -> bool {
  let reach = bitcast<f32>(spanLinks[spanLinksOffset + owner * ${H}u + 2u]);
  if (!(reach > 0.0)) { return false; }
  let start = vertexAt(owner);
  let direction = vertexAt(spanEndVertex) - start;
  let offset = point - start;
  let chordLength = length(direction);
  let separation = select(
    length(offset),
    abs(direction.x * offset.y - direction.y * offset.x) / max(chordLength, 1e-30),
    chordLength > 0.0);
  let slack = 1e-5 * (abs(offset.x) + abs(offset.y) + abs(direction.x) + abs(direction.y));
  return separation <= reach * 1.001 + slack;
}`,body:`let spanEndOfVertex = spanLinks[spanLinksOffset + index * ${H}u + 1u];
  let pointId = pointIds[pointIdsOffset + index];
  // Only kept vertices are tested: a dropped vertex is not part of the result.
  if (spanEndOfVertex == ${B} || pointId == ${B}) { return; }
  let point = vertexAt(index);
  let probe = ExactProbe(point, point, point, EXACT_PROBE_POINT);
  let queryMinimum = vec2f(-FLOAT32_MAXIMUM, point.y);
  let queryMaximum = vec2f(FLOAT32_MAXIMUM, point.y);
  var lastOwner = ${B};
  var node = 0u;
  loop {
    if (nodeOverlaps(node, queryMinimum, queryMaximum)) {
      if (node < INTERNAL_NODE_COUNT) {
        node = node * 2u + 1u;
        continue;
      }
      let edge = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
      var owner = ${B};
      if (edge < VERTEX_COUNT) { owner = spanLinks[spanLinksOffset + edge * ${H}u]; }
      if (owner != ${B} && owner != lastOwner) {
        lastOwner = owner;
        let spanEndVertex = spanLinks[spanLinksOffset + owner * ${H}u + 1u];
        // The ends of the span are on the area's boundary, not inside it.
        if (spanEndVertex != ${B} && pointIds[pointIdsOffset + owner] != pointId &&
            pointIds[pointIdsOffset + spanEndVertex] != pointId &&
            isWithinSpanReach(point, owner, spanEndVertex)) {
          // Winding of the closed ring owner -> chain -> span end -> owner about the point.
          var winding = 0;
          var flags = 0;
          var chainVertex = owner;
          for (var step = 0u; step <= VERTEX_COUNT; step++) {
            var chainNext = vertexNext[vertexNextOffset + chainVertex];
            let isClosing = step > 0u && chainVertex == spanEndVertex;
            if (isClosing) { chainNext = owner; }
            // The single call site of the exact helpers in this kernel.
            let crossing = exactProbeCrossing(probe, vertexAt(chainVertex), vertexAt(chainNext));
            winding += crossing.x;
            flags = flags | crossing.y;
            if (isClosing) { break; }
            chainVertex = chainNext;
          }
          if ((winding & 1) != 0 && flags == 0) {
            atomicOr(&badFlag[badFlagOffset + owner], ${U}u);
          }
        }
      }
    }
    loop {
      if (node == 0u || (node & 1u) == 1u) { break; }
      node = (node - 1u) / 2u;
    }
    if (node == 0u) { break; }
    node = node + 1u;
  }`}),a(e,{id:`${r}-sweep-count`,operation:z,variant:`sweep-count-${j}`,bindings:[{name:`badFlag`,view:T,type:`u32`,access:`read`},{name:`ownedVertex`,view:t.ownedVertex,type:`u32`,access:`read`},{name:`stats`,view:u,type:`atomic<u32>`,access:`read_write`}],invocationCount:s,declarations:`const SLOT: u32 = ${j}u;`,body:`if ((badFlag[badFlagOffset + index] & ${U}u) != 0u && ownedVertex[ownedVertexOffset + index] != 0u) {
    atomicAdd(&stats[statsOffset + SLOT], 1u);
  }`})];return t.repair&&M.push(a(e,{id:`${r}-repair`,operation:z,variant:`repair`,bindings:[{name:`badFlag`,view:T,type:`u32`,access:`read`},{name:`spanSplit`,view:y,type:`u32`,access:`read`},{name:`pointIds`,view:t.pointIds,type:`u32`,access:`read`},{name:`keepPoint`,view:t.keepPoint,type:`atomic<u32>`,access:`read_write`},{name:`stats`,view:u,type:`atomic<u32>`,access:`read_write`}],invocationCount:s,declarations:`const VERTEX_COUNT: u32 = ${s}u;`,body:`let key = spanSplit[spanSplitOffset + index];
  let split = select(key, key - VERTEX_COUNT, key >= VERTEX_COUNT && key != ${B});
  if (badFlag[badFlagOffset + index] != 0u && key != ${B}) {
    let pointId = pointIds[pointIdsOffset + split];
    if (pointId != ${B}) {
      if (atomicMax(&keepPoint[keepPointOffset + pointId], 1u) == 0u) {
        atomicAdd(&stats[statsOffset + 2u], 1u);
      }
    }
  }`})),M}var q=j({parameters:R,caption:`A jagged coverage simplified along shared boundaries; no gaps open between neighbors.`,createInputs:({world:e})=>({...A(e,Math.max(3,Math.round(e.width*2.5)),3,.03),world:e}),build(t,n,r){let i=I(t,`coverage`,n),a=t.allocate(`simplified-positions`,`float32x2`,n.positions.length/2),o=t.allocate(`simplified-ring-offsets`,`uint32`,n.ringOffsets.length),s=t.allocate(`simplified-overflow`,`uint32`,1),c=t.allocate(`tolerance`,`float32`,4);t.graph.add(new W({...L(i),parameters:c.view,simplifyBoundary:r.simplifyBoundary===1,output:{positions:a.view,ringOffsets:o.view,status:{overflow:s.view}}}));let l=P(t,`simplified`,a,o,{closed:!0});return{outputs:{original:t.upload(`original`,`float32x4`,F(n)),...l},update:t=>c.write(e({tolerance:t.tolerance}))}},rebuildOn:[`simplifyBoundary`],interaction:{kind:`reach`,hint:`move off a boundary`,rest:({inputs:e,toUnit:t})=>{let n={positions:e.positions,lineOffsets:e.ringOffsets};return t(N(n,k(R,`tolerance`).default,(e,t)=>M(n,e,t,!0)?.distance??0,e.world,.001))},resolve(e,{inputs:t,values:n}){let r=M({positions:t.positions,lineOffsets:t.ringOffsets},e.x,e.y,!0);if(!r)return{};let i=O(k(R,`tolerance`),r.distance);return{values:{tolerance:i},glyphs:[{shape:`ring`,at:r.point,radius:i,dashed:!0},{shape:`line`,from:r.point,to:[e.x,e.y]},{shape:`dot`,at:r.point}],readout:`tolerance ${i.toFixed(3)} · boundary ${n.simplifyBoundary===1?`simplified`:`kept`}`}}},draw:({outputs:e})=>[{layer:`lines`,context:!0,segments:e.original,radius:2},{layer:`segments`,nodes:e.points,edges:e.edges,values:e.lineIds,radius:2.6}]});export{q as default};