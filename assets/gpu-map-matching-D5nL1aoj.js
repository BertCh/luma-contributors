import{n as e,t}from"./construction-snapshot-DdkX9Pc0.js";import{a as n,m as r,v as i,y as a}from"./spatial-context-DwZEd4EZ.js";import{r as o,s}from"./kernel-C3tfQOn4.js";import{t as c}from"./gpu-scan-B4NNTfF9.js";import{o as l,t as u}from"./wgsl-kernel-nodes-DagJklH2.js";import{A as d,C as f,E as p,S as m,T as h,f as g,g as _,k as v,u as y,v as b,x as ee,y as x}from"./gpu-contributor-utils-BIwLQON5.js";import{a as te,i as S}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as ne}from"./scaled-quotient-wgsl-UDyhQag6.js";import{t as C}from"./fixed-point-accumulator-Ce6W3HVY.js";import{t as re}from"./network-spatial-context-wSR9-WDz.js";import{t as ie}from"./network-workgroup-counters-BG1Cyc9s.js";function w(n){let r=t(n,[`id`,`spatialContext`,`positions`,`trackOffsets`,`parameters`,`topology`,`candidateCount`,`cellSize`,`bounds`,`entryCapacity`,`routeNodeBudget`,`minimumFixSpacingSigmas`,`traversedEdgeCapacity`,`routeTable`,`output`,`pointCount`,`trackCount`,`nodeCount`,`edgeCount`]),i=r.bounds&&t(r.bounds,[`minimum`,`maximum`]),a=r.routeTable&&t(r.routeTable,[`offsets`,`targets`,`costs`,`costLimit`,`status`,`entryCount`]),o=r.output&&t(r.output,[`matchedEdges`,`matchedFractions`,`matchedOffsets`,`snapDistances`,`snappedPositions`,`breaks`,`trackLogLikelihoods`,`matchedCount`,`breakCount`,`status`,`traversedEdges`]),s=o?.traversedEdges&&t(o.traversedEdges,[`edges`,`counts`,`status`]);return Object.freeze({...r,...r.spatialContext?{spatialContext:e(r.spatialContext)}:{},...r.topology?{topology:t(r.topology,[`offsets`,`neighbors`,`weights`,`nodePositions`])}:{},...i?{bounds:Object.freeze({...i,minimum:Object.freeze([...i.minimum]),maximum:Object.freeze([...i.maximum])})}:{},...a?{routeTable:Object.freeze({...a,...a.status?{status:t(a.status,[`complete`])}:{}})}:{},...o?{output:Object.freeze({...o,...o.status?{status:t(o.status,[`overflow`,`candidateOverflow`,`routeOverflow`])}:{},...s?{traversedEdges:Object.freeze({...s,...s.status?{status:t(s.status,[`overflow`,`candidateOverflow`])}:{}})}:{}})}:{}})}var T=Object.freeze({...n,measure:`distance`}),E=`GPUMapMatching`,D=5,ae=3212836864,oe=16,O=Math.floor(4294967295/oe**2),k=2**24;function se(e){return Math.min(O,y(e,12))}function ce(e){return y(e,4)}function le(e,t=8,n=!1){return Math.min(y(e,t*(t+1)*4),y(e,t*12),y(e,(t+1)*4),y(e,8),n?y(e,20):1/0)}function A(e){a(`GPUMapMatching`,e.spatialContext,e.sigma),a(`GPUMapMatching`,e.spatialContext,e.searchRadius);let t=new Float32Array(8);return t[0]=e.sigma,t[1]=e.beta,t[2]=e.searchRadius,t[3]=e.routeFactor??3,t[4]=e.routeSlack??50,t}var j=class{id;props;pointCount;trackCount;nodeCount;edgeCount;candidateCount;gridWidth;gridHeight;spatialContext;constructor(e){e=w(e),this.id=e.id??`map-matching`,this.props=e;let{id:t,props:n}={id:this.id,props:e},{output:a}=e;this.spatialContext=i(t,e.spatialContext,T);let o=r(this.spatialContext);o!==void 0&&f(t,`spatialContext.sphereRadius`,o);let s=d(t,e.topology,{nodePositions:!0});p(e.positions,[`float32x2`],`${t} positions`),p(e.parameters,[`float32`],`${t} parameters`);for(let[n,r]of[[`trackOffsets`,e.trackOffsets],[`output.matchedEdges`,a.matchedEdges],[`output.breaks`,a.breaks],[`output.matchedCount`,a.matchedCount],[`output.breakCount`,a.breakCount],[`output.status.overflow`,a.status?.overflow],[`output.status.candidateOverflow`,a.status?.candidateOverflow],[`output.status.routeOverflow`,a.status?.routeOverflow],[`output.traversedEdges.edges`,a.traversedEdges?.edges],[`output.traversedEdges.counts`,a.traversedEdges?.counts],[`output.traversedEdges.status.overflow`,a.traversedEdges?.status?.overflow],[`output.traversedEdges.status.candidateOverflow`,a.traversedEdges?.status?.candidateOverflow]])r&&p(r,[`uint32`],`${t} ${n}`);for(let[e,n]of[[`output.matchedFractions`,a.matchedFractions],[`output.matchedOffsets`,a.matchedOffsets],[`output.snapDistances`,a.snapDistances],[`output.trackLogLikelihoods`,a.trackLogLikelihoods]])n&&p(n,[`float32`],`${t} ${e}`);if(a.snappedPositions&&p(a.snappedPositions,[`float32x2`],`${t} output.snappedPositions`),this.pointCount=e.positions.length,this.trackCount=e.trackOffsets.length-1,this.nodeCount=s.nodeCount,this.edgeCount=s.edgeCount,m(t,`topology edges`,this.edgeCount,O),this.candidateCount=e.candidateCount??8,this.trackCount<1)throw Error(`${t} trackOffsets needs at least one track`);if(this.pointCount<1||this.edgeCount<1||this.nodeCount<1)throw Error(`${t} needs at least one point, node and edge`);if(e.parameters.length!==8)throw Error(`${t} parameters must contain 8 rows`);g(t,`candidateCount`,this.candidateCount,1,8),g(t,`routeNodeBudget`,e.routeNodeBudget??64,8,128),_(t,`cellSize`,e.cellSize),h(t,Math.fround(e.cellSize)>0,`cellSize must be positive in float32`);let[c,l]=e.bounds.minimum,[u,v]=e.bounds.maximum;for(let[e,n]of[[`minimum.x`,c],[`minimum.y`,l],[`maximum.x`,u],[`maximum.y`,v]])_(t,`bounds.${e}`,n);if(!(u>c)||!(v>l))throw Error(`${t} bounds must have a positive extent`);if(this.gridWidth=Math.max(1,Math.ceil((u-c)/e.cellSize)),this.gridHeight=Math.max(1,Math.ceil((v-l)/e.cellSize)),m(t,`cellSize grid cells`,this.gridWidth*this.gridHeight,k),e.entryCapacity!==void 0&&g(t,`entryCapacity`,e.entryCapacity),e.minimumFixSpacingSigmas!==void 0){if(!(e.minimumFixSpacingSigmas>=0)||!Number.isFinite(e.minimumFixSpacingSigmas))throw Error(`${t} minimumFixSpacingSigmas must be a non-negative number`);_(t,`minimumFixSpacingSigmas`,e.minimumFixSpacingSigmas,0)}if(a.traversedEdges){g(t,`traversedEdgeCapacity`,e.traversedEdgeCapacity??0);let{edges:n,counts:r}=a.traversedEdges,i=a.traversedEdges.status?.overflow;if(n.length!==this.trackCount*e.traversedEdgeCapacity)throw Error(`${t} output.traversedEdges.edges must contain trackCount * traversedEdgeCapacity rows`);if(r.length!==this.trackCount)throw Error(`${t} output.traversedEdges.counts must contain ${this.trackCount} rows`);if(i&&i.length!==1)throw Error(`${t} output.traversedEdges.status.overflow must contain 1 rows`);let o=a.traversedEdges.status?.candidateOverflow;if(o&&o.length!==1)throw Error(`${t} output.traversedEdges.status.candidateOverflow must contain 1 rows`)}for(let[e,n,r]of[[`matchedEdges`,a.matchedEdges,this.pointCount],[`matchedFractions`,a.matchedFractions,this.pointCount],[`matchedOffsets`,a.matchedOffsets,this.pointCount],[`snapDistances`,a.snapDistances,this.pointCount],[`snappedPositions`,a.snappedPositions,this.pointCount],[`breaks`,a.breaks,this.pointCount],[`trackLogLikelihoods`,a.trackLogLikelihoods,this.trackCount],[`matchedCount`,a.matchedCount,1],[`breakCount`,a.breakCount,1],[`status.overflow`,a.status?.overflow,1],[`status.candidateOverflow`,a.status?.candidateOverflow,1],[`status.routeOverflow`,a.status?.routeOverflow,1]])if(n&&n.length!==r)throw Error(`${t} output.${e} must contain ${r} rows`);let{routeTable:y}=e;if(y){for(let[e,n]of[[`routeTable.offsets`,y.offsets],[`routeTable.targets`,y.targets],[`routeTable.status.complete`,y.status?.complete]])n&&p(n,[`uint32`],`${t} ${e}`);if(p(y.costs,[`float32`],`${t} routeTable.costs`),p(y.costLimit,[`float32`],`${t} routeTable.costLimit`),y.offsets.length!==this.nodeCount+1)throw Error(`${t} routeTable.offsets must contain nodeCount + 1 rows`);if(y.costs.length!==y.targets.length)throw Error(`${t} routeTable.costs length must equal routeTable.targets length`);for(let[e,n]of[[`costLimit`,y.costLimit],[`status.complete`,y.status?.complete]])if(n&&n.length!==1)throw Error(`${t} routeTable.${e} must contain exactly one row`)}let b=I(e);x(t,b,F(e)),S(t,Object.fromEntries(b.map((e,t)=>[`output ${t}`,e])))}getCommandNodes(e){let{id:t,props:n,pointCount:i,trackCount:a,nodeCount:d,edgeCount:f,candidateCount:p}=this,{output:h}=n;ee(t,e,[...F(n),...I(n)]),te(e,t,Object.fromEntries([...F(n),...I(n)].map((e,t)=>[`view ${t}`,e]))),m(t,`topology edges`,f,se(e.device)),m(t,`positions rows`,i,le(e.device,p,!!h.traversedEdges)),h.traversedEdges&&m(t,`trackCount * traversedEdgeCapacity`,a*n.traversedEdgeCapacity,y(e.device,4));let _=ce(e.device);n.entryCapacity!==void 0&&g(t,`entryCapacity`,n.entryCapacity,1,_);let v=n.routeNodeBudget??64,x=(n.minimumFixSpacingSigmas??0)>0,{traversedEdges:S}=h,C=this.gridWidth*this.gridHeight,w=n.entryCapacity??Math.min(16*f+256,_);for(let[r,a]of Object.entries({edgeTable:f*12,grid:C*4,entries:w*4,candidates:i*p*12,routes:i*p*(p+1)*4,scores:i*(p+1)*4,backPointers:i*p*4,points:i*4,...S?{hops:i*D*4}:{},...Object.fromEntries([...F(n),...I(n)].filter(e=>e!==void 0).map((e,t)=>[`view ${t}`,s(e).size]))}))b(e.device,t,r,a,`reduce point count, entryCapacity or grid cells`);let[T,O]=n.bounds.minimum,k=n.bounds.maximum[0],A=r(this.spatialContext),j=(n,r,i)=>o(e,`${t}-${n}`,r,Math.max(i,1)),L=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read`}),R=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),Ee=`${ne}
const POINT_COUNT: u32 = ${i}u;
const NODE_COUNT: u32 = ${d}u;
const EDGE_COUNT: u32 = ${f}u;
const K: u32 = ${p}u;
// Route rows: K route lengths, then the number of nodes the route search dropped (an exact f32).
const ROUTE_STRIDE: u32 = ${p+1}u;
const NONE: u32 = 0xffffffffu;
// Back pointer of a restart where a route search into the point dropped nodes: a budget break.
const BUDGET_BREAK: u32 = 0xfffffffeu;
// Predecessor of a fix dropped by the minimum fix spacing (NONE marks the first fix of a track).
const DROPPED_FIX: u32 = 0xfffffffeu;
// Score rows: K scores, then the offset subtracted from them (per-step renormalization).
const SCORE_STRIDE: u32 = ${p+1}u;
const NEG: f32 = -1.0e30;
const GRID_WIDTH: u32 = ${this.gridWidth}u;
const GRID_HEIGHT: u32 = ${this.gridHeight}u;
const CELL_COUNT: u32 = ${C}u;
const ENTRY_CAPACITY: u32 = ${w}u;
const GRID_MINIMUM: vec2<f32> = vec2<f32>(${M(T)}, ${M(O)});
const CELL_SIZE: f32 = ${M(n.cellSize)};
const MAX_EDGE_SPAN: u32 = ${oe}u;
fn getCellX(x: f32) -> u32 {
  return u32(clamp(floor(getScaledQuotient(x - GRID_MINIMUM.x, CELL_SIZE, 1.0, 0)), 0.0, f32(GRID_WIDTH - 1u)));
}
fn getCellY(y: f32) -> u32 {
  return u32(clamp(floor(getScaledQuotient(y - GRID_MINIMUM.y, CELL_SIZE, 1.0, 0)), 0.0, f32(GRID_HEIGHT - 1u)));
}
${re(A)}
${de(A!==void 0,T,k)}`,z=`
fn edgeSource(edge: u32) -> u32 { return edgeTable[edgeTableOffset + edge * 3u]; }
fn edgeTarget(edge: u32) -> u32 { return edgeTable[edgeTableOffset + edge * 3u + 1u]; }
fn edgeLength(edge: u32) -> f32 { return bitcast<f32>(edgeTable[edgeTableOffset + edge * 3u + 2u]); }
fn isEdgeValid(edge: u32) -> bool { return edgeLength(edge) >= 0.0; }`,B=(n,r,i,a,o=``)=>l(e,{id:`${t}-${n}`,operation:E,variant:n,bindings:r,invocationCount:i,declarations:`${Ee}
${r.some(e=>e.name===`nodePositions`)?`fn readNode(node: u32) -> vec2<f32> {
  return vec2<f32>(nodePositions[nodePositionsOffset + node * 2u], nodePositions[nodePositionsOffset + node * 2u + 1u]);
}`:``}
${o}`,body:a}),V=j(`edge-table`,`uint32`,f*3),H=j(`cell-counts`,`uint32`,C),U=j(`cell-ends`,`uint32`,C),De=j(`cursors`,`uint32`,C),Oe=j(`entries`,`uint32`,w),W=j(`candidates`,`uint32`,i*p*3),G=j(`point-starts`,`uint32`,i),K=j(`predecessors`,`uint32`,i),q=j(`routes`,`float32`,i*p*(p+1)),ke=j(`scores`,`float32`,i*(p+1)),J=j(`back-pointers`,`uint32`,i*p),Y=j(`chosen`,`uint32`,i),X=h.breaks??j(`breaks`,`uint32`,i),Z=h.status?.candidateOverflow??j(`candidate-overflow`,`uint32`,1),Ae=()=>{let{routeTable:e}=n;if(!e?.status?.complete)return[];let t=e.status.complete;return[B(`route-bounds`,[x?L(`predecessors`,K):L(`pointStarts`,G),L(`points`,n.positions,`f32`),L(`parameters`,n.parameters,`f32`),L(`candidates`,W),L(`edgeTable`,V),R(`routes`,q,`f32`),L(`tableCostLimit`,e.costLimit,`f32`),L(`tableComplete`,t)],i*p,`let point = index / K;
  let fromSlot = index % K;
  let base = routesOffset + index * ROUTE_STRIDE;
  routes[base + K] = -1.0;
  ${_e(x)}
  if (tableComplete[tableCompleteOffset] == 0u) { return; }
  let fromEdge = getCandidateEdge(previous, fromSlot);
  if (fromEdge == NONE) {
    return;
  }
  let straight = networkDistance(readPoint(point), readPoint(previous));
  let bound = straight * parameters[parametersOffset + 3u] + parameters[parametersOffset + 4u];
  let tail = (1.0 - getCandidateFraction(previous, fromSlot)) * edgeLength(fromEdge);
  let remaining = bound - tail;
  if (!(remaining >= 0.0) || !(remaining <= tableCostLimit[tableCostLimitOffset]) ||
      edgeTarget(fromEdge) >= NODE_COUNT) {
    return;
  }
  routes[base] = bound;
  routes[base + K] = remaining;`,`${z}
${P()}`),B(`route-table`,[L(`candidates`,W),L(`edgeTable`,V),R(`routes`,q,`f32`),L(`tableOffsets`,e.offsets),L(`tableTargets`,e.targets),L(`tableCosts`,e.costs,`f32`),...x?[L(`predecessors`,K)]:[]],i*p,ye(x),`${z}
${ve()}`)]},Q=[];h.status?.overflow&&Q.push(u(e,{id:`${t}-clear-output-overflow`,operation:E,view:h.status.overflow,type:`u32`,value:`0u`})),Q.push(u(e,{id:`${t}-clear-edge-table`,operation:E,view:V,type:`u32`,value:`${ae}u`}),B(`edge-table`,[L(`adjacency`,n.topology.offsets),L(`edgeTargets`,n.topology.neighbors),L(`nodePositions`,n.topology.nodePositions,`f32`),R(`edgeTable`,V)],d,`let end = min(adjacency[adjacencyOffset + index + 1u], EDGE_COUNT);
  let start = readNode(index);
  for (var edge = adjacency[adjacencyOffset + index]; edge < end; edge++) {
    let targetNode = edgeTargets[edgeTargetsOffset + edge];
    var length = -1.0;
    if (targetNode < NODE_COUNT) {
      length = networkDistance(start, readNode(targetNode));
    }
    edgeTable[edgeTableOffset + edge * 3u] = index;
    edgeTable[edgeTableOffset + edge * 3u + 1u] = targetNode;
    edgeTable[edgeTableOffset + edge * 3u + 2u] = bitcast<u32>(length);
  }`),u(e,{id:`${t}-clear-counts`,operation:E,view:H,type:`u32`,value:`0u`}),u(e,{id:`${t}-clear-cursors`,operation:E,view:De,type:`u32`,value:`0u`}),u(e,{id:`${t}-clear-overflow`,operation:E,view:Z,type:`u32`,value:`0u`}));let $=`let a = readNode(edgeSource(index));
  let b = readNode(edgeTarget(index));
  let gridXA = getGridX(a.x);
  let gridXB = getGridXNear(b.x, a.x, gridXA);
  let cellX0 = getCellX(min(gridXA, gridXB));
  let cellX1 = getCellX(max(gridXA, gridXB));
  let cellY0 = getCellY(min(a.y, b.y));
  let cellY1 = getCellY(max(a.y, b.y));`;if(Q.push(B(`cell-count`,[L(`edgeTable`,V),L(`nodePositions`,n.topology.nodePositions,`f32`),R(`cellCounts`,H,`atomic<u32>`)],f,`if (!isEdgeValid(index)) {
    return;
  }
  ${$}
  for (var cellY = cellY0; cellY <= min(cellY1, cellY0 + MAX_EDGE_SPAN - 1u); cellY++) {
    for (var cellX = cellX0; cellX <= min(cellX1, cellX0 + MAX_EDGE_SPAN - 1u); cellX++) {
      atomicAdd(&cellCounts[cellCountsOffset + cellY * GRID_WIDTH + cellX], 1u);
    }
  }`,z),...new c({id:`${t}-cell-scan`,input:H,output:U,mode:`inclusive`}).getCommandNodes(e),B(`cell-fill`,[L(`edgeTable`,V),L(`nodePositions`,n.topology.nodePositions,`f32`),L(`cellEnds`,U),R(`cursors`,De,`atomic<u32>`),R(`entries`,Oe),R(`overflow`,Z,`atomic<u32>`)],f,`if (!isEdgeValid(index)) {
    return;
  }
  ${$}
  if (cellX1 - cellX0 >= MAX_EDGE_SPAN || cellY1 - cellY0 >= MAX_EDGE_SPAN) {
    atomicStore(&overflow[overflowOffset], 1u);
  }
  for (var cellY = cellY0; cellY <= min(cellY1, cellY0 + MAX_EDGE_SPAN - 1u); cellY++) {
    for (var cellX = cellX0; cellX <= min(cellX1, cellX0 + MAX_EDGE_SPAN - 1u); cellX++) {
      let cell = cellY * GRID_WIDTH + cellX;
      let start = select(0u, cellEnds[cellEndsOffset + cell - 1u], cell > 0u);
      let slot = start + atomicAdd(&cursors[cursorsOffset + cell], 1u);
      if (slot < ENTRY_CAPACITY) {
        entries[entriesOffset + slot] = index;
      } else {
        atomicStore(&overflow[overflowOffset], 1u);
      }
    }
  }`,z),B(`candidates`,[L(`points`,n.positions,`f32`),L(`nodePositions`,n.topology.nodePositions,`f32`),L(`edgeTable`,V),L(`cellEnds`,U),L(`entries`,Oe),L(`parameters`,n.parameters,`f32`),R(`candidates`,W)],i,`let point = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  ${ue(A!==void 0)}
  var bestEdge: array<u32, ${p}>;
  var bestFraction: array<f32, ${p}>;
  var bestDistance: array<f32, ${p}>;
  for (var slot = 0u; slot < K; slot++) {
    bestEdge[slot] = NONE;
    bestFraction[slot] = 0.0;
    bestDistance[slot] = 0.0;
  }
  let cellY0 = getCellY(point.y - latitudeExtent);
  let cellY1 = getCellY(point.y + latitudeExtent);
  for (var windowIndex = 0u; windowIndex < WINDOW_COUNT; windowIndex++) {
  let longitudeRange = getLongitudeRange(getGridX(point.x), longitudeExtent, windowIndex);
  if (longitudeRange.x > longitudeRange.y) {
    continue;
  }
  let cellX0 = getCellX(longitudeRange.x);
  let cellX1 = getCellX(longitudeRange.y);
  for (var cellY = cellY0; cellY <= cellY1; cellY++) {
    for (var cellX = cellX0; cellX <= cellX1; cellX++) {
      let cell = cellY * GRID_WIDTH + cellX;
      let start = select(0u, cellEnds[cellEndsOffset + cell - 1u], cell > 0u);
      let end = min(cellEnds[cellEndsOffset + cell], ENTRY_CAPACITY);
      for (var entry = start; entry < end; entry++) {
        let edge = entries[entriesOffset + entry];
        let origin = readNode(edgeSource(edge));
        let segmentEnd = readNode(edgeTarget(edge));
        let fraction = networkSegmentFraction(point, origin, segmentEnd);
        let distance = networkSegmentDistance(point, origin, segmentEnd, fraction);
        if (!(distance <= radius)) {
          continue;
        }
        var isKnown = false;
        for (var slot = 0u; slot < K; slot++) {
          isKnown = isKnown || bestEdge[slot] == edge;
        }
        if (isKnown) {
          continue;
        }
        // Insertion into the sorted list; the last slot is dropped when full.
        var position = K;
        for (var slot = 0u; slot < K; slot++) {
          if (position == K && (bestEdge[slot] == NONE || distance < bestDistance[slot] ||
              (distance == bestDistance[slot] && edge < bestEdge[slot]))) {
            position = slot;
          }
        }
        if (position < K) {
          for (var slot = K - 1u; slot > position; slot--) {
            bestEdge[slot] = bestEdge[slot - 1u];
            bestFraction[slot] = bestFraction[slot - 1u];
            bestDistance[slot] = bestDistance[slot - 1u];
          }
          bestEdge[position] = edge;
          bestFraction[position] = fraction;
          bestDistance[position] = distance;
        }
      }
    }
  }
  }
  for (var slot = 0u; slot < K; slot++) {
    let base = candidatesOffset + (index * K + slot) * 3u;
    candidates[base] = bestEdge[slot];
    candidates[base + 1u] = bitcast<u32>(bestFraction[slot]);
    candidates[base + 2u] = bitcast<u32>(bestDistance[slot]);
  }`,`${z}
${fe(A!==void 0,k)}`),...x?[u(e,{id:`${t}-clear-predecessors`,operation:E,view:K,type:`u32`,value:`0xffffffffu`}),B(`fix-filter`,[L(`trackOffsets`,n.trackOffsets),L(`points`,n.positions,`f32`),L(`parameters`,n.parameters,`f32`),R(`predecessors`,K)],a,`let first = trackOffsets[trackOffsetsOffset + index];
  let last = min(trackOffsets[trackOffsetsOffset + index + 1u], POINT_COUNT);
  let minimumSpacing = max(parameters[parametersOffset], 1.0e-6) * MINIMUM_FIX_SPACING_SIGMAS;
  var kept = NONE;
  for (var point = first; point < last; point++) {
    var predecessor = kept;
    if (kept != NONE && networkDistance(readFixPoint(point), readFixPoint(kept)) < minimumSpacing) {
      predecessor = DROPPED_FIX;
    } else {
      kept = point;
    }
    predecessors[predecessorsOffset + point] = predecessor;
  }`,`const MINIMUM_FIX_SPACING_SIGMAS: f32 = ${M(n.minimumFixSpacingSigmas)};
fn readFixPoint(point: u32) -> vec2<f32> {
  return vec2<f32>(points[pointsOffset + point * 2u], points[pointsOffset + point * 2u + 1u]);
}`)]:[u(e,{id:`${t}-clear-starts`,operation:E,view:G,type:`u32`,value:`0u`}),B(`starts`,[L(`trackOffsets`,n.trackOffsets),R(`pointStarts`,G)],a,`let first = trackOffsets[trackOffsetsOffset + index];
  if (first < POINT_COUNT) {
    pointStarts[pointStartsOffset + first] = 1u;
  }`)],...Ae(),B(`routes`,[x?L(`predecessors`,K):L(`pointStarts`,G),L(`points`,n.positions,`f32`),L(`nodePositions`,n.topology.nodePositions,`f32`),L(`adjacency`,n.topology.offsets),L(`edgeTable`,V),L(`candidates`,W),L(`parameters`,n.parameters,`f32`),R(`routes`,q,`f32`)],i*p,ge(!!n.routeTable?.status?.complete,x),pe(v,N(v),p,z)),B(`forward`,[L(`trackOffsets`,n.trackOffsets),L(`points`,n.positions,`f32`),L(`candidates`,W),L(`parameters`,n.parameters,`f32`),L(`routes`,q,`f32`),R(`scores`,ke,`f32`),R(`backPointers`,J),...x?[L(`predecessors`,K)]:[]],a,be(x),he(p)),B(`backward`,[L(`trackOffsets`,n.trackOffsets),L(`scores`,ke,`f32`),L(`backPointers`,J),R(`chosen`,Y),R(`breaks`,X),...h.trackLogLikelihoods?[R(`logLikelihoods`,h.trackLogLikelihoods,`f32`)]:[],...x?[L(`predecessors`,K)]:[]],a,xe(!!h.trackLogLikelihoods,x),Se())),h.status?.routeOverflow&&Q.push(u(e,{id:`${t}-clear-route-overflow`,operation:E,view:h.status.routeOverflow,type:`u32`,value:`0u`})),Q.push(B(`route-status`,[L(`routes`,q,`f32`),R(`overflow`,Z,`atomic<u32>`),...h.status?.routeOverflow?[R(`routeOverflow`,h.status.routeOverflow,`atomic<u32>`)]:[]],i*p,`let dropped = u32(routes[routesOffset + index * ROUTE_STRIDE + K]);
  if (dropped == 0u) {
    return;
  }
  atomicStore(&overflow[overflowOffset], 1u);
  ${h.status?.routeOverflow?`// Saturating sum: an add that wraps pins the total at the maximum, and every later add
  // that wraps from there pins it again.
  let previous = atomicAdd(&routeOverflow[routeOverflowOffset], dropped);
  if (previous + dropped < previous) {
    atomicMax(&routeOverflow[routeOverflowOffset], 0xffffffffu);
  }`:``}`)),Q.push(B(`matched`,[L(`candidates`,W),L(`chosen`,Y),L(`edgeTable`,V),R(`matchedEdges`,h.matchedEdges),...h.matchedFractions?[R(`fractions`,h.matchedFractions,`f32`)]:[],...h.matchedOffsets?[R(`offsets`,h.matchedOffsets,`f32`)]:[],...h.snapDistances?[R(`distances`,h.snapDistances,`f32`)]:[]],i,`let slot = chosen[chosenOffset + index];
  let isMatched = slot != NONE;
  let base = candidatesOffset + (index * K + min(slot, K - 1u)) * 3u;
  let edge = candidates[base];
  matchedEdges[matchedEdgesOffset + index] = select(NONE, edge, isMatched);
  ${h.matchedFractions?`fractions[fractionsOffset + index] = select(-1.0, bitcast<f32>(candidates[base + 1u]), isMatched);`:``}
  ${h.matchedOffsets?`offsets[offsetsOffset + index] = select(-1.0, bitcast<f32>(candidates[base + 1u]) * edgeLength(select(0u, edge, isMatched)), isMatched);`:``}
  ${h.snapDistances?`distances[distancesOffset + index] = select(-1.0, bitcast<f32>(candidates[base + 2u]), isMatched);`:``}`,z)),S){let r=j(`hops`,`uint32`,i*D),o=n.traversedEdgeCapacity,s=j(`traversed-status`,`uint32`,2);Q.push(u(e,{id:`${t}-clear-traversed-overflow`,operation:E,view:s,type:`u32`,value:`0u`}),B(`hops`,[L(`chosen`,Y),L(`breaks`,X),L(`candidates`,W),L(`points`,n.positions,`f32`),L(`parameters`,n.parameters,`f32`),L(`edgeTable`,V),x?L(`predecessors`,K):L(`pointStarts`,G),R(`hops`,r)],i,Ce(x),`${z}
${P()}`),B(`route-edges`,[L(`trackOffsets`,n.trackOffsets),L(`hops`,r),L(`nodePositions`,n.topology.nodePositions,`f32`),L(`adjacency`,n.topology.offsets),L(`edgeTable`,V),R(`traversedEdges`,S.edges),R(`traversedCounts`,S.counts),R(`traversedStatus`,s,`atomic<u32>`)],a,Te(),`const EDGE_CAPACITY: u32 = ${o}u;
${me(v,N(v),p,z)}
${we(v)}`)),Q.push(B(`traversed-status`,[L(`traversedStatus`,s),R(`candidateOverflow`,Z,`atomic<u32>`),...S.status?.overflow?[R(`outputOverflow`,S.status.overflow)]:[],...S.status?.candidateOverflow?[R(`outputCandidateOverflow`,S.status.candidateOverflow)]:[]],1,`let lost = traversedStatus[traversedStatusOffset + 1u];
  if (lost != 0u) { atomicStore(&candidateOverflow[candidateOverflowOffset], 1u); }
  ${S.status?.overflow?`outputOverflow[outputOverflowOffset] = traversedStatus[traversedStatusOffset];`:``}
  ${S.status?.candidateOverflow?`outputCandidateOverflow[outputCandidateOverflowOffset] = atomicLoad(&candidateOverflow[candidateOverflowOffset]);`:``}`))}if(h.snappedPositions&&Q.push(B(`positions`,[L(`points`,n.positions,`f32`),L(`candidates`,W),L(`chosen`,Y),L(`edgeTable`,V),L(`nodePositions`,n.topology.nodePositions,`f32`),R(`snappedPositions`,h.snappedPositions,`f32`)],i,`var position = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  let slot = chosen[chosenOffset + index];
  if (slot != NONE) {
    let base = candidatesOffset + (index * K + slot) * 3u;
    let edge = candidates[base];
    let origin = readNode(edgeSource(edge));
    position = networkInterpolate(origin, readNode(edgeTarget(edge)), bitcast<f32>(candidates[base + 1u]));
  }
  snappedPositions[snappedPositionsOffset + index * 2u] = position.x;
  snappedPositions[snappedPositionsOffset + index * 2u + 1u] = position.y;`,z)),h.matchedCount||h.breakCount){let n=[h.matchedCount,h.breakCount].filter(e=>e);for(let r of n)Q.push(u(e,{id:`${t}-clear-${r===h.matchedCount?`matched`:`break`}-count`,operation:E,view:r,type:`u32`,value:`0u`}));let r=[],a=[];h.matchedCount&&(a.push(`if (chosen[chosenOffset + index] != NONE) { addCounter(${r.length}u, 1u); }`),r.push({target:`matchedCount`,word:0})),h.breakCount&&(a.push(`if (breaks[breaksOffset + index] != 0u) { addCounter(${r.length}u, 1u); }`),r.push({target:`breakCount`,word:0}));let o=ie({counters:r,rowCount:`POINT_COUNT`,rowSource:a.join(`
  `)});Q.push(l(e,{id:`${t}-summary`,operation:E,variant:`summary`,bindings:[L(`chosen`,Y),L(`breaks`,X),...h.matchedCount?[R(`matchedCount`,h.matchedCount,`atomic<u32>`)]:[],...h.breakCount?[R(`breakCount`,h.breakCount,`atomic<u32>`)]:[]],invocationCount:i,workgroupSize:256,guardIndex:!1,declarations:`${Ee}
${o.declarations}`,body:o.body}))}return Q}};function ue(e){return e?`let radius = min(
    parameters[parametersOffset + 2u],
    2.0 * CELL_SIZE * GEODESIC_DEGREES_TO_RADIANS * NETWORK_EARTH_RADIUS
  );
  let latitudeExtent = getScaledQuotient(radius, NETWORK_EARTH_RADIUS, 1.0, 0) * GEODESIC_RADIANS_TO_DEGREES;
  let bandLatitude = min(abs(point.y) + latitudeExtent, 90.0);
  let longitudeExtent = min(latitudeExtent / max(geodesicCosLatitude(bandLatitude), 1.0e-6), 180.0);`:`let radius = min(parameters[parametersOffset + 2u], 2.0 * CELL_SIZE);
  let latitudeExtent = radius;
  let longitudeExtent = radius;`}function de(e,t,n){return e?`const GRID_CENTER_X: f32 = ${M((t+n)/2)};
fn getGridX(x: f32) -> f32 {
  return GRID_CENTER_X + geodesicWrapLongitudeDelta(x - GRID_CENTER_X);
}
fn getGridXNear(x: f32, anchorX: f32, anchorGridX: f32) -> f32 {
  return anchorGridX + geodesicWrapLongitudeDelta(x - anchorX);
}`:`fn getGridX(x: f32) -> f32 {
  return x;
}
fn getGridXNear(x: f32, anchorX: f32, anchorGridX: f32) -> f32 {
  return x;
}`}function fe(e,t){return e?`const WINDOW_COUNT: u32 = 3u;
const GRID_MAXIMUM_X: f32 = ${M(t)};
fn getLongitudeRange(x: f32, extent: f32, windowIndex: u32) -> vec2<f32> {
  let low = x - extent;
  let high = x + extent;
  let empty = vec2<f32>(1.0, 0.0);
  if (windowIndex == 1u) {
    let isWrapped = low < GRID_MINIMUM.x && low + 360.0 < GRID_MAXIMUM_X;
    return select(empty, vec2<f32>(low + 360.0, GRID_MAXIMUM_X), isWrapped);
  }
  if (windowIndex == 2u) {
    let isWrapped = high > GRID_MAXIMUM_X && high - 360.0 > GRID_MINIMUM.x;
    return select(empty, vec2<f32>(GRID_MINIMUM.x, high - 360.0), isWrapped);
  }
  return vec2<f32>(low, high);
}`:`const WINDOW_COUNT: u32 = 1u;
fn getLongitudeRange(x: f32, extent: f32, windowIndex: u32) -> vec2<f32> {
  return vec2<f32>(x - extent, x + extent);
}`}function M(e){let t=String(Math.fround(e));return/[.eE]/.test(t)?t:`${t}.0`}function N(e){let t=16;for(;t<2*e;)t*=2;return t}function pe(e,t,n,r){return`${r}
${P()}
${me(e,t,n)}`}function me(e,t,n,r=``){return`${r}
const S: u32 = ${e}u;
const HASH_MASK: u32 = ${t-1}u;
var<private> tableNode: array<u32, ${e}>;
var<private> tableDistance: array<f32, ${e}>;
var<private> tableKey: array<f32, ${e}>;
var<private> tableDone: array<u32, ${e}>;
// Edge row that reached each table node with its current distance (NONE for the start).
var<private> tableParent: array<u32, ${e}>;
var<private> tableHash: array<u32, ${t}>;
var<private> pendingTargets: array<u32, ${n}>;
var<private> targetPositions: array<vec2<f32>, ${n}>;
var<private> targetTotal: u32 = 0u;
// Discovered nodes the full table could not keep in the last search.
var<private> droppedCount: u32 = 0u;

// Consistent A* heuristic: the distance to the nearest target node, planar or great-circle. Every
// edge is as long as the straight line (or great circle) between its nodes, so by the triangle
// inequality the heuristic never overestimates; the factor
// below 1 absorbs f32 rounding so it stays consistent in floating point.
fn getHeuristic(node: u32) -> f32 {
  if (targetTotal == 0u || node >= NODE_COUNT) {
    return 0.0;
  }
  let position = readNode(node);
  var nearest = 3.0e38;
  for (var slot = 0u; slot < targetTotal; slot++) {
    nearest = min(nearest, networkDistance(position, targetPositions[slot]));
  }
  return nearest * 0.999999;
}

fn getHashStart(node: u32) -> u32 {
  return ((node * 2654435761u) >> 7u) & HASH_MASK;
}
// Table slot of a node, or NONE. The hash has at least twice as many cells as table slots.
fn findEntry(node: u32) -> u32 {
  var cell = getHashStart(node);
  loop {
    let stored = tableHash[cell];
    if (stored == 0u) {
      return NONE;
    }
    if (tableNode[stored - 1u] == node) {
      return stored - 1u;
    }
    cell = (cell + 1u) & HASH_MASK;
  }
}
// Removes the hash cell of table slot 'slot' with backward-shift deletion: later cells of the
// probe run move into the hole unless their home cell lies cyclically after it, so every lookup
// still reaches its entry without tombstones. The hash always has an empty cell (it has at least
// twice as many cells as table slots), so the loops end.
fn removeHashEntry(slot: u32) {
  var hole = getHashStart(tableNode[slot]);
  while (tableHash[hole] != slot + 1u) {
    hole = (hole + 1u) & HASH_MASK;
  }
  tableHash[hole] = 0u;
  var cell = (hole + 1u) & HASH_MASK;
  loop {
    let stored = tableHash[cell];
    if (stored == 0u) {
      break;
    }
    let home = getHashStart(tableNode[stored - 1u]);
    if (((cell - home) & HASH_MASK) >= ((cell - hole) & HASH_MASK)) {
      tableHash[hole] = stored;
      tableHash[cell] = 0u;
      hole = cell;
    }
    cell = (cell + 1u) & HASH_MASK;
  }
}
// The open (unsettled) slot with the largest key, ties to the highest slot, or NONE.
fn findWorstOpenEntry(count: u32) -> u32 {
  var worst = NONE;
  for (var slot = 0u; slot < count; slot++) {
    if (tableDone[slot] == 0u && (worst == NONE || tableKey[slot] >= tableKey[worst])) {
      worst = slot;
    }
  }
  return worst;
}
fn addEntry(slot: u32, node: u32, distance: f32, parentEdge: u32) {
  tableNode[slot] = node;
  tableParent[slot] = parentEdge;
  tableDistance[slot] = distance;
  tableKey[slot] = distance + getHeuristic(node);
  tableDone[slot] = 0u;
  var cell = getHashStart(node);
  while (tableHash[cell] != 0u) {
    cell = (cell + 1u) & HASH_MASK;
  }
  tableHash[cell] = slot + 1u;
}

// Bounded A* from 'start' over the CSR network; distances within 'limit' land in the private
// table (at most S nodes; a full table counts every node it cannot keep in 'droppedCount' and keeps
// the open nodes with the best keys). Nodes are settled in order of distance plus the heuristic, so a node
// settled (done) has its exact network distance, and the search heads toward the targets instead of
// settling the whole disc of radius 'limit' as plain Dijkstra does. It stops once every node in
// 'pendingTargets' (NONE for unused slots; 'pendingCount' of them are set) is settled, or once the
// best key exceeds 'limit' (no remaining target can be within the bound). Returns the table size.
fn searchNetwork(start: u32, limit: f32, pendingCount: u32) -> u32 {
  var pending = pendingCount;
  droppedCount = 0u;
  targetTotal = 0u;
  for (var slot = 0u; slot < K; slot++) {
    if (pendingTargets[slot] != NONE) {
      targetPositions[targetTotal] = readNode(pendingTargets[slot]);
      targetTotal++;
    }
  }
  // A runtime trip count prevents the Metal back end from losing unrolled integer-table stores.
  let hashCellCount = (HASH_MASK + 1u) * min(arrayLength(&adjacency), 1u);
  for (var slot = 0u; slot < hashCellCount; slot++) {
    tableHash[slot] = 0u;
  }
  var count = 0u;
  addEntry(count, start, 0.0, NONE);
  count = 1u;
  loop {
    var best = NONE;
    // At large finite bounds, limit + 1 rounds back to limit. Select the first finite key
    // explicitly so a key exactly at the inclusive bound is still settled.
    var bestKey = limit;
    for (var slot = 0u; slot < count; slot++) {
      if (tableDone[slot] == 0u && bitcast<u32>(tableKey[slot]) < 0x7f800000u &&
          (best == NONE || tableKey[slot] < bestKey)) {
        best = slot;
        bestKey = tableKey[slot];
      }
    }
    if (best == NONE || bestKey > limit) {
      break;
    }
    let bestDistance = tableDistance[best];
    tableDone[best] = 1u;
    let node = tableNode[best];
    for (var slot = 0u; slot < K; slot++) {
      if (pendingTargets[slot] == node) {
        pendingTargets[slot] = NONE;
        pending--;
      }
    }
    if (pending == 0u) {
      break;
    }
    if (node >= NODE_COUNT) {
      continue;
    }
    let end = min(adjacency[adjacencyOffset + node + 1u], EDGE_COUNT);
    for (var edge = adjacency[adjacencyOffset + node]; edge < end; edge++) {
      if (!isEdgeValid(edge)) {
        continue;
      }
      let candidateDistance = bestDistance + edgeLength(edge);
      if (candidateDistance > limit) {
        continue;
      }
      let neighbor = edgeTarget(edge);
      let found = findEntry(neighbor);
      if (found != NONE) {
        if (tableDone[found] == 0u && candidateDistance < tableDistance[found]) {
          tableDistance[found] = candidateDistance;
          tableParent[found] = edge;
          tableKey[found] = candidateDistance + getHeuristic(neighbor);
        }
      } else if (count < S) {
        addEntry(count, neighbor, candidateDistance, edge);
        count++;
      } else {
        // Full table: the node is dropped, or replaces the open entry with the worst key when its
        // own key is better. Either way the search may now miss or overestimate a route.
        droppedCount++;
        let key = candidateDistance + getHeuristic(neighbor);
        let worst = findWorstOpenEntry(count);
        if (worst != NONE && key < tableKey[worst]) {
          removeHashEntry(worst);
          addEntry(worst, neighbor, candidateDistance, edge);
        }
      }
    }
  }
  return count;
}

fn lookupDistance(node: u32) -> f32 {
  let slot = findEntry(node);
  if (slot != NONE && tableDone[slot] != 0u) {
    return tableDistance[slot];
  }
  return -1.0;
}
`}function P(){return`fn readPoint(point: u32) -> vec2<f32> {
  return vec2<f32>(points[pointsOffset + point * 2u], points[pointsOffset + point * 2u + 1u]);
}
fn getCandidateEdge(point: u32, slot: u32) -> u32 {
  return candidates[candidatesOffset + (point * K + slot) * 3u];
}
fn getCandidateFraction(point: u32, slot: u32) -> f32 {
  return bitcast<f32>(candidates[candidatesOffset + (point * K + slot) * 3u + 1u]);
}
fn getCandidateDistance(point: u32, slot: u32) -> f32 {
  return bitcast<f32>(candidates[candidatesOffset + (point * K + slot) * 3u + 2u]);
}

`}function he(e){return`fn readPoint(point: u32) -> vec2<f32> {
  return vec2<f32>(points[pointsOffset + point * 2u], points[pointsOffset + point * 2u + 1u]);
}
fn getCandidateEdge(point: u32, slot: u32) -> u32 {
  return candidates[candidatesOffset + (point * K + slot) * 3u];
}
fn getCandidateFraction(point: u32, slot: u32) -> f32 {
  return bitcast<f32>(candidates[candidatesOffset + (point * K + slot) * 3u + 1u]);
}
fn getCandidateDistance(point: u32, slot: u32) -> f32 {
  return bitcast<f32>(candidates[candidatesOffset + (point * K + slot) * 3u + 2u]);
}

const LOG_TWO_PI_HALF: f32 = 0.9189385;
const CANDIDATE_COUNT: u32 = ${e}u;`}function ge(e,t){return`let point = index / K;
  let fromSlot = index % K;
  ${e?`// A non-negative trailing word marks a row the route table resolved (route-table kernel).
  if (routes[routesOffset + index * ROUTE_STRIDE + K] >= 0.0) {
    return;
  }
  `:``}for (var to = 0u; to < K; to++) {
    routes[routesOffset + index * ROUTE_STRIDE + to] = -1.0;
  }
  routes[routesOffset + index * ROUTE_STRIDE + K] = 0.0;
  // Point 0 has no predecessor even when it lies before the first track (review B-23).
  ${_e(t)}
  let fromEdge = getCandidateEdge(previous, fromSlot);
  if (fromEdge == NONE) {
    return;
  }
  let straight = networkDistance(readPoint(point), readPoint(previous));
  let bound = straight * parameters[parametersOffset + 3u] + parameters[parametersOffset + 4u];
  let fromLength = edgeLength(fromEdge);
  let fromFraction = getCandidateFraction(previous, fromSlot);
  let tail = (1.0 - fromFraction) * fromLength;
  let remaining = bound - tail;
  if (remaining >= 0.0) {
    var pendingCount = 0u;
    for (var to = 0u; to < K; to++) {
      let toEdge = getCandidateEdge(point, to);
      let isSameEdgeAhead = toEdge == fromEdge && getCandidateFraction(point, to) >= fromFraction;
      let isPending = toEdge != NONE && !isSameEdgeAhead;
      pendingTargets[to] = select(NONE, edgeSource(select(0u, toEdge, isPending)), isPending);
      pendingCount += select(0u, 1u, isPending);
    }
    searchNetwork(edgeTarget(fromEdge), remaining, pendingCount);
    routes[routesOffset + index * ROUTE_STRIDE + K] = f32(min(droppedCount, 16777216u));
  }
  for (var to = 0u; to < K; to++) {
    let toEdge = getCandidateEdge(point, to);
    if (toEdge == NONE) {
      continue;
    }
    let toFraction = getCandidateFraction(point, to);
    var route = -1.0;
    if (toEdge == fromEdge && toFraction >= fromFraction) {
      route = (toFraction - fromFraction) * fromLength;
    } else if (remaining >= 0.0) {
      let network = lookupDistance(edgeSource(toEdge));
      if (network >= 0.0) {
        route = tail + network + toFraction * edgeLength(toEdge);
      }
    }
    // The bound covers the whole route, including the partial edge at either end (review B-24).
    if (route > bound) {
      route = -1.0;
    }
    routes[routesOffset + index * ROUTE_STRIDE + to] = route;
  }`}function _e(e){return e?`let previous = predecessors[predecessorsOffset + point];
  if (previous >= DROPPED_FIX) {
    return;
  }`:`if (point == 0u || pointStarts[pointStartsOffset + point] != 0u) {
    return;
  }
  let previous = point - 1u;`}function ve(){return`fn getCandidateEdge(point: u32, slot: u32) -> u32 {
  return candidates[candidatesOffset + (point * K + slot) * 3u];
}
fn getCandidateFraction(point: u32, slot: u32) -> f32 {
  return bitcast<f32>(candidates[candidatesOffset + (point * K + slot) * 3u + 1u]);
}
// Cost from 'row' to 'targetNode' by binary search of the row's ascending targets, or -1 when the
// table does not list it (the cost then exceeds the table's cost limit).
fn findTableCost(row: u32, targetNode: u32) -> f32 {
  var low = tableOffsets[tableOffsetsOffset + row];
  let end = tableOffsets[tableOffsetsOffset + row + 1u];
  var high = end;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (tableTargets[tableTargetsOffset + middle] < targetNode) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < end && tableTargets[tableTargetsOffset + low] == targetNode) {
    return tableCosts[tableCostsOffset + low];
  }
  return -1.0;
}`}function ye(e){return`let base = routesOffset + index * ROUTE_STRIDE;
  let remaining = routes[base + K];
  if (!(remaining >= 0.0)) {
    return;
  }
  let bound = routes[base];
  let point = index / K;
  let previous = ${e?`predecessors[predecessorsOffset + point]`:`point - 1u`};
  let fromEdge = getCandidateEdge(previous, index % K);
  let fromFraction = getCandidateFraction(previous, index % K);
  let fromLength = edgeLength(fromEdge);
  let tail = (1.0 - fromFraction) * fromLength;
  let startNode = edgeTarget(fromEdge);
  for (var to = 0u; to < K; to++) {
    let toEdge = getCandidateEdge(point, to);
    var route = -1.0;
    if (toEdge != NONE) {
      let toFraction = getCandidateFraction(point, to);
      if (toEdge == fromEdge && toFraction >= fromFraction) {
        route = (toFraction - fromFraction) * fromLength;
      } else {
        let network = findTableCost(startNode, edgeSource(toEdge));
        // The A* search only admits costs within the allowance; the table also lists larger ones.
        if (network >= 0.0 && network <= remaining) {
          route = tail + network + toFraction * edgeLength(toEdge);
        }
      }
      // The bound covers the whole route, including the partial edge at either end (review B-24).
      if (route > bound) {
        route = -1.0;
      }
    }
    routes[base + to] = route;
  }
  routes[base + K] = 0.0;`}function be(e){return`let first = trackOffsets[trackOffsetsOffset + index];
  let last = trackOffsets[trackOffsetsOffset + index + 1u];
  let sigma = max(parameters[parametersOffset], 1.0e-6);
  let beta = max(parameters[parametersOffset + 1u], 1.0e-6);
  let emissionConstant = -log(sigma) - LOG_TWO_PI_HALF;
  let transitionConstant = -log(beta);
  for (var point = first; point < last; point++) {
    var newScore: array<f32, CANDIDATE_COUNT>;
    var newBack: array<u32, CANDIDATE_COUNT>;
    for (var slot = 0u; slot < K; slot++) {
      newScore[slot] = NEG;
      newBack[slot] = NONE;
    }
    var isReachable = false;
    var isBudgetLimited = false;
    ${e?`let previous = predecessors[predecessorsOffset + point];
    if (previous == DROPPED_FIX) {
      // A dropped fix is skipped: neither a source nor a target of a transition.
      for (var slot = 0u; slot < K; slot++) {
        scores[scoresOffset + point * SCORE_STRIDE + slot] = NEG;
        backPointers[backPointersOffset + point * K + slot] = NONE;
      }
      scores[scoresOffset + point * SCORE_STRIDE + K] = 0.0;
      continue;
    }
    if (previous != NONE) {`:`let previous = point - 1u;
    if (point > first) {`}
      let straight = networkDistance(readPoint(point), readPoint(previous));
      for (var fromSlot = 0u; fromSlot < K; fromSlot++) {
        let fromEdge = getCandidateEdge(previous, fromSlot);
        let fromScore = scores[scoresOffset + previous * SCORE_STRIDE + fromSlot];
        if (fromEdge == NONE || fromScore < -1.0e29) {
          continue;
        }
        let routeRow = routesOffset + (point * K + fromSlot) * ROUTE_STRIDE;
        isBudgetLimited = isBudgetLimited || routes[routeRow + K] > 0.0;
        for (var to = 0u; to < K; to++) {
          let route = routes[routeRow + to];
          if (route < 0.0) {
            continue;
          }
          let score = fromScore + transitionConstant - getScaledQuotient(abs(straight - route), beta, 1.0, 0);
          if (score > newScore[to]) {
            newScore[to] = score;
            newBack[to] = fromSlot;
            isReachable = true;
          }
        }
      }
    }
    for (var slot = 0u; slot < K; slot++) {
      if (getCandidateEdge(point, slot) == NONE) {
        newScore[slot] = NEG;
        newBack[slot] = NONE;
        continue;
      }
      let normalized = getScaledQuotient(getCandidateDistance(point, slot), sigma, 1.0, 0);
      let emission = emissionConstant - 0.5 * normalized * normalized;
      if (isReachable) {
        if (newScore[slot] > -1.0e29) {
          newScore[slot] += emission;
        }
      } else {
        // First point of the track or a break: restart from the emission alone. A break where a
        // route search dropped nodes is marked, since the missing route may be the budget's doing.
        newScore[slot] = emission;
        newBack[slot] = select(NONE, BUDGET_BREAK, isBudgetLimited);
      }
    }
    // Renormalize: subtract the best score of the step from every valid score. Every score of the
    // step moves by the same amount, so the best path is unchanged, and the magnitudes stay small
    // over any track length. The subtracted amount is kept for the log-likelihood.
    var maximum = NEG;
    for (var slot = 0u; slot < K; slot++) {
      if (newScore[slot] > -1.0e29) {
        maximum = max(maximum, newScore[slot]);
      }
    }
    let offset = select(0.0, maximum, maximum > -1.0e29);
    for (var slot = 0u; slot < K; slot++) {
      if (newScore[slot] > -1.0e29) {
        newScore[slot] -= offset;
      }
      scores[scoresOffset + point * SCORE_STRIDE + slot] = newScore[slot];
      backPointers[backPointersOffset + point * K + slot] = newBack[slot];
    }
    scores[scoresOffset + point * SCORE_STRIDE + K] = offset;
  }`}function xe(e,t){return`let first = trackOffsets[trackOffsetsOffset + index];
  let last = trackOffsets[trackOffsetsOffset + index + 1u];
  likelihoodSum = vec3<u32>(0u);
  ${e?`// Choose a shared scale from the actual finite terms, excluding sentinel scores.
  var maximumExponent = 0u;
  for (var point = first; point < last; point++) {
    ${t?`if (predecessors[predecessorsOffset + point] == DROPPED_FIX) { continue; }`:``}
    let row = scoresOffset + point * SCORE_STRIDE;
    maximumExponent = max(maximumExponent, getFixedPointExponentBits(scores[row + K]));
    for (var slot = 0u; slot < K; slot++) {
      let score = scores[row + slot];
      if (score > -1.0e29) { maximumExponent = max(maximumExponent, getFixedPointExponentBits(score)); }
    }
  }
  // At most two terms per uint32-indexed point: 33 bits of integer headroom cannot wrap.
  likelihoodUnitExponent = getFixedPointUnitExponent(maximumExponent, 33u);`:``}
  var current = NONE;
  var isPicking = true;
  for (var step = 0u; step < last - first; step++) {
    let point = last - 1u - step;
    ${t?`if (predecessors[predecessorsOffset + point] == DROPPED_FIX) {
      // Skipped: the walk continues with the slot chosen for the next retained fix.
      chosen[chosenOffset + point] = NONE;
      breaks[breaksOffset + point] = 0u;
      continue;
    }`:``}
    ${e?`// The scores of a stretch are renormalized per step: its log-likelihood is its final score
    // plus the amounts subtracted at its points.
    addToSum(scores[scoresOffset + point * SCORE_STRIDE + K]);`:``}
    if (isPicking) {
      var bestScore = -1.0e29;
      current = NONE;
      for (var slot = 0u; slot < K; slot++) {
        let score = scores[scoresOffset + point * SCORE_STRIDE + slot];
        if (score > bestScore) {
          bestScore = score;
          current = slot;
        }
      }
      if (current != NONE) {
        ${e?`addToSum(bestScore);`:``}
      }
    }
    chosen[chosenOffset + point] = current;
    var isBreak = current == NONE;
    var isBudgetBreak = false;
    isPicking = true;
    if (current != NONE) {
      let previous = backPointers[backPointersOffset + point * K + current];
      if (previous < K) {
        current = previous;
        isPicking = false;
      } else {
        isBreak = true;
        isBudgetBreak = previous == BUDGET_BREAK;
      }
    }
    breaks[breaksOffset + point] = select(0u, select(1u, 2u, isBudgetBreak), isBreak && point > first);
  }
  ${e?`logLikelihoods[logLikelihoodsOffset + index] = getSum();`:``}`}function Se(){return`${C}
var<private> likelihoodSum: vec3<u32> = vec3<u32>(0u);
var<private> likelihoodUnitExponent: i32 = 0;
fn addToSum(term: f32) {
  likelihoodSum = addFixedPoint(likelihoodSum, quantizeFixedPoint(term, likelihoodUnitExponent));
}
fn getSum() -> f32 {
  return getFixedPointFloat(likelihoodSum, likelihoodUnitExponent);
}`}function Ce(e){return`let base = hopsOffset + index * ${D}u;
  hops[base] = 0u;
  let slot = chosen[chosenOffset + index];
  if (slot == NONE) {
    return;
  }
  let toEdge = getCandidateEdge(index, slot);
  hops[base + 4u] = toEdge;
  ${e?`let previous = predecessors[predecessorsOffset + index];
  if (previous >= DROPPED_FIX || breaks[breaksOffset + index] != 0u) {`:`let previous = index - 1u;
  if (index == 0u || pointStarts[pointStartsOffset + index] != 0u || breaks[breaksOffset + index] != 0u) {`}
    hops[base] = 1u;
    return;
  }
  let fromSlot = chosen[chosenOffset + previous];
  let fromEdge = getCandidateEdge(previous, fromSlot);
  let fromFraction = getCandidateFraction(previous, fromSlot);
  if (toEdge == fromEdge && getCandidateFraction(index, slot) >= fromFraction) {
    hops[base] = 3u;
    return;
  }
  let straight = networkDistance(readPoint(index), readPoint(previous));
  let bound = straight * parameters[parametersOffset + 3u] + parameters[parametersOffset + 4u];
  let tail = (1.0 - fromFraction) * edgeLength(fromEdge);
  hops[base] = 2u;
  hops[base + 1u] = edgeTarget(fromEdge);
  hops[base + 2u] = edgeSource(toEdge);
  hops[base + 3u] = bitcast<u32>(bound - tail);`}function we(e){return`var<private> emittedCount: u32 = 0u;
var<private> pathEdges: array<u32, ${e}>;
fn emitEdge(track: u32, edge: u32) {
  if (emittedCount < EDGE_CAPACITY) {
    traversedEdges[traversedEdgesOffset + track * EDGE_CAPACITY + emittedCount] = edge;
  } else {
    atomicStore(&traversedStatus[traversedStatusOffset], 1u);
  }
  emittedCount++;
}`}function Te(){return`let first = trackOffsets[trackOffsetsOffset + index];
  let last = trackOffsets[trackOffsetsOffset + index + 1u];
  emittedCount = 0u;
  for (var point = first; point < last; point++) {
    let base = hopsOffset + point * ${D}u;
    let kind = hops[base];
    if (kind == 0u || kind == 3u) {
      continue;
    }
    let toEdge = hops[base + 4u];
    if (kind == 1u) {
      if (emittedCount > 0u) {
        emitEdge(index, NONE);
      }
      emitEdge(index, toEdge);
      continue;
    }
    // Hop over the network: search again with the allowance of the route kernel, then walk the
    // parent edges from the target node back to the start node.
    let startNode = hops[base + 1u];
    let targetNode = hops[base + 2u];
    var pathLength = 0u;
    var isRecovered = true;
    if (startNode != targetNode) {
      for (var slot = 0u; slot < K; slot++) {
        pendingTargets[slot] = NONE;
      }
      pendingTargets[0] = targetNode;
      searchNetwork(startNode, bitcast<f32>(hops[base + 3u]), 1u);
      if (droppedCount > 0u) { atomicStore(&traversedStatus[traversedStatusOffset + 1u], 1u); }
      var node = targetNode;
      loop {
        if (node == startNode) {
          break;
        }
        let entry = findEntry(node);
        if (entry == NONE || tableDone[entry] == 0u || pathLength >= S) {
          isRecovered = false;
          break;
        }
        pathEdges[pathLength] = tableParent[entry];
        pathLength++;
        node = edgeSource(tableParent[entry]);
      }
    }
    if (!isRecovered) {
      atomicStore(&traversedStatus[traversedStatusOffset + 1u], 1u);
      emitEdge(index, NONE);
    } else {
      for (var step = 0u; step < pathLength; step++) {
        emitEdge(index, pathEdges[pathLength - 1u - step]);
      }
    }
    emitEdge(index, toEdge);
  }
  traversedCounts[traversedCountsOffset + index] = emittedCount;`}function F(e){return[e.positions,e.trackOffsets,...v(e.topology),e.parameters,e.routeTable?.offsets,e.routeTable?.targets,e.routeTable?.costs,e.routeTable?.costLimit,e.routeTable?.status?.complete]}function I(e){let{output:t}=e;return[t.matchedEdges,t.matchedFractions,t.matchedOffsets,t.snapDistances,t.snappedPositions,t.breaks,t.trackLogLikelihoods,t.matchedCount,t.breakCount,t.status?.overflow,t.status?.candidateOverflow,t.status?.routeOverflow,t.traversedEdges?.edges,t.traversedEdges?.counts,t.traversedEdges?.status?.overflow,t.traversedEdges?.status?.candidateOverflow]}export{A as n,w as r,j as t};