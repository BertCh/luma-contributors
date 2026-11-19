import{o as e,s as t}from"./network-view-utils-B1U8oNKx.js";import{a as n,v as r}from"./spatial-context-DwZEd4EZ.js";import{n as i}from"./scan-total-C4QddMmi.js";import{r as a}from"./kernel-C3tfQOn4.js";import{t as o}from"./gpu-sort-xRm8l5yn.js";import{o as s,t as c}from"./wgsl-kernel-nodes-DagJklH2.js";import{A as l,E as u,S as d,T as f,k as p,u as m,y as h}from"./gpu-contributor-utils-BIwLQON5.js";import{t as g}from"./geodesic-wgsl-I0CYDKcz.js";import{t as _}from"./gpu-network-reachability-Coypydjh.js";import{t as v}from"./define-run-B-brkCe4.js";import{a as y}from"./pointer-overlay-Qh8qflcA.js";import{_ as b,d as x,h as S,i as C,r as w,y as T}from"./roads-DzovamHj.js";import{s as E}from"./display-Bu20g6o5.js";import{ba as D}from"./entry-preview-D70TdSbK.js";function O(e={},t=new Float32Array(8)){if(t.length<8)throw Error(`GPUNetworkLineGraph parameter target must hold 8 elements`);let n=e.angleCost??0,r=e.leftTurnCost??0,i=e.rightTurnCost??0;for(let[e,t]of[[`angleCost`,n],[`leftTurnCost`,r],[`rightTurnCost`,i]])if(!(t>=0)||!Number.isFinite(t))throw Error(`GPUNetworkLineGraph ${e} must be a non-negative finite number`);let a=e.bannedTurnCount??0;if(!Number.isSafeInteger(a)||a<0)throw Error(`GPUNetworkLineGraph bannedTurnCount must be a non-negative integer`);return t[0]=n,t[1]=r,t[2]=i,t[3]=e.uTurnCost??-1,t[4]=e.straightAngle??.5,t[5]=a,t[6]=e.uTurnAngle??2.9,t[7]=0,t}var k=Object.freeze({...n,measure:`distance`}),A=`GPUNetworkLineGraph`;function j(e){return m(e,8)}function M(e){return m(e,16)}var N=class{id;props;nodeCount;edgeCount;arcCapacity;spatialContext;constructor(e){this.id=e.id??`network-line-graph`,this.props=e,t(this.id,e.output);let{id:n}=this;this.spatialContext=r(n,e.spatialContext,k);let{output:i}=e,{nodeCount:a,edgeCount:o}=l(n,e.topology,{weights:!0,nodePositions:!0});for(let[t,r]of[[`bannedTurns`,e.bannedTurns],[`output.topology.offsets`,i.topology.offsets],[`output.topology.neighbors`,i.topology.neighbors],[`output.arcCount`,i.arcCount],[`output.status.overflow`,i.status?.overflow]])r&&u(r,[`uint32`],`${n} ${t}`);for(let[t,r]of[[`parameters`,e.parameters],[`output.topology.weights`,i.topology.weights]])u(r,[`float32`],`${n} ${t}`);if(i.turnPenalties&&(u(i.turnPenalties,[`float32`],`${n} output.turnPenalties`),f(n,i.turnPenalties.length===i.topology.neighbors.length,`output.turnPenalties length must equal topology.neighbors length`)),this.nodeCount=a,this.edgeCount=o,this.arcCapacity=i.topology.neighbors.length,this.arcCapacity<1)throw Error(`${n} output.topology.neighbors must hold at least one arc`);if(!i.status?.overflow)throw Error(`${n} output.status.overflow is required`);if(e.parameters.length<8)throw Error(`${n} parameters must hold 8 elements`);if(e.bannedTurns&&e.bannedTurns.length%2!=0)throw Error(`${n} bannedTurns length must be even`);if(i.topology.offsets.length!==this.edgeCount+1)throw Error(`${n} output.topology.offsets must contain edgeCount + 1 rows`);if(i.topology.weights.length!==this.arcCapacity)throw Error(`${n} output.topology.weights length must equal neighbors length`);for(let[e,t]of[[`output.arcCount`,i.arcCount],[`output.status.overflow`,i.status?.overflow]])if(t&&t.length!==1)throw Error(`${n} ${e} must contain exactly one row`);let s=L(e);h(n,s,I(e));let c=s.filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(c).size!==c.length)throw Error(`${n} outputs must use separate buffers`)}getCommandNodes(t){let{id:n,props:r,nodeCount:o,edgeCount:l,arcCapacity:u}=this;e(n,t,[...I(r),...L(r)]),this.spatialContext.coordinateSpace===`longitude-latitude`?d(n,`topology edges with a great-circle spatialContext`,l,M(t.device)):d(n,`topology edges`,l,j(t.device)),e(n,t,[],{vectors:l*(this.spatialContext.coordinateSpace===`longitude-latitude`?16:8),counts:(l+1)*4,bannedTurns:(r.bannedTurns?.length??0)*4});let f=[],p=r.bannedTurns?.length?r.bannedTurns:void 0,m=!!p,h=p?p.length/2:0,_=p?this.addBanSortNodes(t,f,p):void 0,v=this.spatialContext.coordinateSpace===`longitude-latitude`,y=v?4:2,b=a(t,`${n}-edge-vectors`,`float32`,y*l),x=a(t,`${n}-counts`,`uint32`,l+1),S=`const NODE_COUNT: u32 = ${o}u;
const EDGE_COUNT: u32 = ${l}u;
const ARC_CAPACITY: u32 = ${u}u;
const EDGE_VECTOR_STRIDE: u32 = ${y}u;
const ARRIVAL_OFFSET: u32 = ${y-2}u;
${F}
${m?`const BAN_CAPACITY: u32 = ${h}u;
// sortedBans holds (fromEdge, toEdge) pairs in ascending fromEdge order; inactive rows have
// fromEdge = EDGE_COUNT and sort last. Find the first pair of fromEdge, then scan its run.
fn isBannedTurn(fromEdge: u32, toEdge: u32) -> bool {
  var low = 0u;
  var high = BAN_CAPACITY;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (sortedBans[sortedBansOffset + 2u * middle] < fromEdge) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  for (var ban = low; ban < BAN_CAPACITY; ban++) {
    if (sortedBans[sortedBansOffset + 2u * ban] != fromEdge) {
      break;
    }
    if (sortedBans[sortedBansOffset + 2u * ban + 1u] == toEdge) {
      return true;
    }
  }
  return false;
}`:`fn isBannedTurn(fromEdge: u32, toEdge: u32) -> bool {
  return false;
}`}
// Only a negative U-turn cost makes getTurnCost negative (every other term is non-negative), so
// with U-turns priced the angle is never needed, and a turn whose edges point forward (positive dot
// product) is below any U-turn threshold clearly above a right angle (1.6 rad leaves a margin for
// backend atan2 error). Both skip the atan2 that the cost would need; the remaining arcs evaluate
// the exact cost as before.
fn isAllowedTurn(fromEdge: u32, toEdge: u32) -> bool {
  if (isBannedTurn(fromEdge, toEdge)) {
    return false;
  }
  if (parameters[parametersOffset + 3u] >= 0.0) {
    return true;
  }
  if (parameters[parametersOffset + 6u] > 1.6) {
    let a = getArrivalDirection(fromEdge);
    let b = getDepartureDirection(toEdge);
    if (dot(a, a) >= 0.5 && dot(b, b) >= 0.5 && dot(a, b) > 0.0) {
      return true;
    }
  }
  return getTurnCost(fromEdge, toEdge) >= 0.0;
}`,C=[{name:`edgeVectors`,view:b,type:`f32`,access:`read`},{name:`parameters`,view:r.parameters,type:`f32`,access:`read`}];_&&C.push({name:`sortedBans`,view:_,type:`u32`,access:`read`});let w=`fn isLiveEdge(edge: u32) -> bool {
  return edge >= offsets[offsetsOffset] && edge < min(offsets[offsetsOffset + NODE_COUNT], EDGE_COUNT);
}`;f.push(s(t,{id:`${n}-edge-vectors`,operation:A,variant:`edge-vectors`,bindings:[{name:`offsets`,view:r.topology.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:r.topology.neighbors,type:`u32`,access:`read`},{name:`nodePositions`,view:r.topology.nodePositions,type:`f32`,access:`read`},{name:`edgeVectors`,view:b,type:`f32`,access:`read_write`}],invocationCount:o,declarations:`const NODE_COUNT: u32 = ${o}u;
const EDGE_COUNT: u32 = ${l}u;
${v?g:``}`,body:v?P:`let start = nodePositions[nodePositionsOffset + 2u * index];
  let startY = nodePositions[nodePositionsOffset + 2u * index + 1u];
  let end = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
  for (var edge = offsets[offsetsOffset + index]; edge < end; edge++) {
    let headNode = neighbors[neighborsOffset + edge];
    var direction = vec2<f32>(0.0);
    if (headNode < NODE_COUNT) {
      let delta = vec2<f32>(
        nodePositions[nodePositionsOffset + 2u * headNode] - start,
        nodePositions[nodePositionsOffset + 2u * headNode + 1u] - startY
      );
      let edgeLength = sqrt(dot(delta, delta));
      if (edgeLength > 0.0) {
        direction = delta / edgeLength;
      }
    }
    edgeVectors[edgeVectorsOffset + 2u * edge] = direction.x;
    edgeVectors[edgeVectorsOffset + 2u * edge + 1u] = direction.y;
  }`})),f.push(s(t,{id:`${n}-count`,operation:A,variant:`count`,bindings:[{name:`offsets`,view:r.topology.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:r.topology.neighbors,type:`u32`,access:`read`},...C,{name:`counts`,view:x,type:`u32`,access:`read_write`}],invocationCount:l+1,declarations:`${S}
${w}`,body:`var count = 0u;
  if (index < EDGE_COUNT && isLiveEdge(index)) {
    let head = neighbors[neighborsOffset + index];
    if (head < NODE_COUNT) {
      let end = min(offsets[offsetsOffset + head + 1u], EDGE_COUNT);
      for (var next = offsets[offsetsOffset + head]; next < end; next++) {
        if (isAllowedTurn(index, next)) {
          count++;
        }
      }
    }
  }
  counts[countsOffset + index] = count;`}));let T=a(t,`${n}-scan-total`,`uint32`,1);if(f.push(...i(t,{id:`${n}-scan`,operation:A,counts:x,offsets:r.output.topology.offsets,rowCount:l+1,total:T})),r.output.arcCount||r.output.status.overflow){let e=[{name:`scanTotal`,view:T,type:`u32`,access:`read`}];r.output.arcCount&&e.push({name:`arcCount`,view:r.output.arcCount,type:`u32`,access:`read_write`}),r.output.status.overflow&&e.push({name:`overflow`,view:r.output.status.overflow,type:`u32`,access:`read_write`}),f.push(s(t,{id:`${n}-publish`,operation:A,variant:`publish`,bindings:e,invocationCount:1,declarations:`const EDGE_COUNT: u32 = ${l}u;
const ARC_CAPACITY: u32 = ${u}u;`,body:`let total = scanTotal[scanTotalOffset];
  ${r.output.arcCount?`arcCount[arcCountOffset] = min(total, ARC_CAPACITY);`:``}
  ${r.output.status.overflow?`overflow[overflowOffset] = select(0u, 1u, total > ARC_CAPACITY);`:``}`}))}let E=8+ +!!_;f.push(c(t,{id:`${n}-clear-line-neighbors`,operation:A,view:r.output.topology.neighbors,type:`u32`,value:`0xffffffffu`}),c(t,{id:`${n}-clear-line-weights`,operation:A,view:r.output.topology.weights,type:`f32`,value:`-1.0`}));let D=E<=t.device.limits.maxStorageBuffersPerShaderStage;return f.push(s(t,{id:`${n}-fill-neighbors`,operation:A,variant:D?`fill`:`fill-neighbors`,bindings:[{name:`offsets`,view:r.topology.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:r.topology.neighbors,type:`u32`,access:`read`},...C,{name:`lineOffsets`,view:r.output.topology.offsets,type:`u32`,access:`read`},{name:`lineNeighbors`,view:r.output.topology.neighbors,type:`u32`,access:`read_write`},...D?[{name:`weights`,view:r.topology.weights,type:`f32`,access:`read`},{name:`lineWeights`,view:r.output.topology.weights,type:`f32`,access:`read_write`}]:[]],invocationCount:l,declarations:`${S}
${w}`,body:`if (!isLiveEdge(index)) {
    return;
  }
  let head = neighbors[neighborsOffset + index];
  if (head >= NODE_COUNT) {
    return;
  }
  var slot = lineOffsets[lineOffsetsOffset + index];
  let end = min(offsets[offsetsOffset + head + 1u], EDGE_COUNT);
  for (var next = offsets[offsetsOffset + head]; next < end && slot < ARC_CAPACITY; next++) {
    ${D?`if (isBannedTurn(index, next)) {
      continue;
    }
    let turnCost = getTurnCost(index, next);
    if (turnCost >= 0.0) {
      if (slot < ARC_CAPACITY) {
        lineNeighbors[lineNeighborsOffset + slot] = next;
        let baseCost = weights[weightsOffset + next];
        lineWeights[lineWeightsOffset + slot] =
          select(-1.0, baseCost + max(turnCost, 0.0), baseCost >= 0.0);
      }
      slot++;
    }`:`if (isAllowedTurn(index, next)) {
      if (slot < ARC_CAPACITY) {
        lineNeighbors[lineNeighborsOffset + slot] = next;
      }
      slot++;
    }`}
  }`})),D||f.push(s(t,{id:`${n}-fill-weights`,operation:A,variant:`fill-weights`,bindings:[{name:`lineOffsets`,view:r.output.topology.offsets,type:`u32`,access:`read`},{name:`lineNeighbors`,view:r.output.topology.neighbors,type:`u32`,access:`read`},{name:`weights`,view:r.topology.weights,type:`f32`,access:`read`},...C,{name:`lineWeights`,view:r.output.topology.weights,type:`f32`,access:`read_write`}],invocationCount:u,declarations:S,body:`if (index >= lineOffsets[lineOffsetsOffset + EDGE_COUNT]) {
    return;
  }
  var low = 0u;
  var high = EDGE_COUNT;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle + 1u] > index) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  let next = lineNeighbors[lineNeighborsOffset + index];
  let baseCost = weights[weightsOffset + next];
  lineWeights[lineWeightsOffset + index] =
    select(-1.0, baseCost + max(getTurnCost(low, next), 0.0), baseCost >= 0.0);`})),f.push(s(t,{id:`${n}-clamp`,operation:A,variant:`clamp`,bindings:[{name:`lineOffsets`,view:r.output.topology.offsets,type:`u32`,access:`read_write`}],invocationCount:l+1,declarations:`const ARC_CAPACITY: u32 = ${u}u;`,body:`lineOffsets[lineOffsetsOffset + index] =
    min(lineOffsets[lineOffsetsOffset + index], ARC_CAPACITY);`})),r.output.turnPenalties&&f.push(c(t,{id:`${n}-clear-turn-penalties`,operation:A,view:r.output.turnPenalties,type:`f32`,value:`-1.0`}),s(t,{id:`${n}-fill-turn-penalties`,operation:A,variant:`fill-turn-penalties`,invocationCount:l,bindings:[{name:`lineOffsets`,view:r.output.topology.offsets,type:`u32`,access:`read`},{name:`lineNeighbors`,view:r.output.topology.neighbors,type:`u32`,access:`read`},{name:`edgeVectors`,view:b,type:`f32`,access:`read`},{name:`parameters`,view:r.parameters,type:`f32`,access:`read`},{name:`linePenalties`,view:r.output.turnPenalties,type:`f32`,access:`read_write`}],declarations:`const EDGE_COUNT: u32 = ${l}u;
const EDGE_VECTOR_STRIDE: u32 = ${y}u;
const ARRIVAL_OFFSET: u32 = ${y-2}u;
${F}`,body:`let end = lineOffsets[lineOffsetsOffset + index + 1u];
for (var slot = lineOffsets[lineOffsetsOffset + index]; slot < end; slot++) {
  let following = lineNeighbors[lineNeighborsOffset + slot];
  if (following < EDGE_COUNT) {
    linePenalties[linePenaltiesOffset + slot] = max(getTurnCost(index, following), 0.0);
  }
}`})),f}addBanSortNodes(e,t,n){let{id:r,edgeCount:i}=this,c=n.length/2,l=(t,n)=>a(e,`${r}-${t}`,`uint32`,n),u=l(`ban-keys`,c),d=l(`ban-values`,c),f=l(`ban-sorted-keys`,c),p=l(`ban-sorted-values`,c),m=l(`sorted-bans`,2*c);return t.push(s(e,{id:`${r}-ban-keys`,operation:A,variant:`ban-keys`,bindings:[{name:`parameters`,view:this.props.parameters,type:`f32`,access:`read`},{name:`bannedTurns`,view:n,type:`u32`,access:`read`},{name:`keys`,view:u,type:`u32`,access:`read_write`},{name:`values`,view:d,type:`u32`,access:`read_write`}],invocationCount:c,declarations:`const EDGE_COUNT: u32 = ${i}u;`,body:`let bannedCount = u32(max(parameters[parametersOffset + 5u], 0.0));
  let fromEdge = bannedTurns[bannedTurnsOffset + 2u * index];
  let isActive = index < bannedCount && fromEdge < EDGE_COUNT;
  keys[keysOffset + index] = select(EDGE_COUNT, fromEdge, isActive);
  values[valuesOffset + index] = bannedTurns[bannedTurnsOffset + 2u * index + 1u];`}),...new o({id:`${r}-ban-sort`,keys:u,values:d,outputKeys:f,outputValues:p,keyBits:Math.max(1,Math.ceil(Math.log2(i+1)))}).getCommandNodes(e),s(e,{id:`${r}-ban-pairs`,operation:A,variant:`ban-pairs`,bindings:[{name:`sortedKeys`,view:f,type:`u32`,access:`read`},{name:`sortedValues`,view:p,type:`u32`,access:`read`},{name:`sortedBans`,view:m,type:`u32`,access:`read_write`}],invocationCount:c,body:`sortedBans[sortedBansOffset + 2u * index] = sortedKeys[sortedKeysOffset + index];
  sortedBans[sortedBansOffset + 2u * index + 1u] = sortedValues[sortedValuesOffset + index];`})),m}},P=`let start = vec2<f32>(
    nodePositions[nodePositionsOffset + 2u * index],
    nodePositions[nodePositionsOffset + 2u * index + 1u]
  );
  let end = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
  for (var edge = offsets[offsetsOffset + index]; edge < end; edge++) {
    let headNode = neighbors[neighborsOffset + edge];
    var departure = vec2<f32>(0.0);
    var arrival = vec2<f32>(0.0);
    if (headNode < NODE_COUNT) {
      let head = vec2<f32>(
        nodePositions[nodePositionsOffset + 2u * headNode],
        nodePositions[nodePositionsOffset + 2u * headNode + 1u]
      );
      if (geodesicCentralAngle(start, head) > 0.0) {
        let departureBearing = geodesicInitialBearingDegrees(start, head) * GEODESIC_DEGREES_TO_RADIANS;
        let returnBearing = geodesicInitialBearingDegrees(head, start) * GEODESIC_DEGREES_TO_RADIANS;
        departure = vec2<f32>(sin(departureBearing), cos(departureBearing));
        arrival = -vec2<f32>(sin(returnBearing), cos(returnBearing));
      }
    }
    edgeVectors[edgeVectorsOffset + 4u * edge] = departure.x;
    edgeVectors[edgeVectorsOffset + 4u * edge + 1u] = departure.y;
    edgeVectors[edgeVectorsOffset + 4u * edge + 2u] = arrival.x;
    edgeVectors[edgeVectorsOffset + 4u * edge + 3u] = arrival.y;
  }`,F=`
fn getDepartureDirection(edge: u32) -> vec2<f32> {
  let base = edgeVectorsOffset + EDGE_VECTOR_STRIDE * edge;
  return vec2<f32>(edgeVectors[base], edgeVectors[base + 1u]);
}
fn getArrivalDirection(edge: u32) -> vec2<f32> {
  let base = edgeVectorsOffset + EDGE_VECTOR_STRIDE * edge + ARRIVAL_OFFSET;
  return vec2<f32>(edgeVectors[base], edgeVectors[base + 1u]);
}
fn getTurnAngle(fromEdge: u32, toEdge: u32) -> f32 {
  let a = getArrivalDirection(fromEdge);
  let b = getDepartureDirection(toEdge);
  if (dot(a, a) < 0.5 || dot(b, b) < 0.5) {
    return 0.0;
  }
  let cross = a.x * b.y - a.y * b.x;
  let projection = dot(a, b);
  // atan2 with a signed-zero x is unreliable on some backends; right angles are common on grids.
  if (projection == 0.0) {
    return select(-1.5707963, 1.5707963, cross > 0.0);
  }
  return atan2(cross, projection);
}
// Turn cost of the turn fromEdge -> toEdge, or a negative value when U-turns are banned.
fn getTurnCost(fromEdge: u32, toEdge: u32) -> f32 {
  let angle = getTurnAngle(fromEdge, toEdge);
  let magnitude = abs(angle);
  if (magnitude >= parameters[parametersOffset + 6u]) {
    return parameters[parametersOffset + 3u];
  }
  var cost = parameters[parametersOffset] * magnitude;
  if (magnitude > parameters[parametersOffset + 4u]) {
    cost += select(
      parameters[parametersOffset + 2u],
      parameters[parametersOffset + 1u],
      angle > 0.0
    );
  }
  return cost;
}`;function I(e){return[...p(e.topology),e.parameters,e.bannedTurns]}function L(e){let{output:t}=e;return[t.topology.offsets,t.topology.neighbors,t.topology.weights,t.turnPenalties,t.arcCount,t.status.overflow]}var R=v({parameters:D,caption:`Cost to arrive along each directed road, with turn penalties from the line graph.`,createInputs:({random:e,world:t})=>{let n=w(C(e,t,10),0),r=n.findNearestNode(t.width*.5,.5),i=Uint32Array.from({length:n.offsets[r+1]-n.offsets[r]},(e,t)=>n.offsets[r]+t),a=0;for(let e=0;e<n.slotCount;e++){let t=n.neighbors[e];a+=n.offsets[t+1]-n.offsets[t]}return{network:n,origin:r,originSlots:i,arcCapacity:a}},build(e,{network:t,origin:n,originSlots:r,arcCapacity:i}){let{topology:a}=b(e,t),o=e.upload(`node-positions`,`float32x2`,t.nodes),s=e.allocate(`parameters`,`float32`,8),c=e.allocate(`line-offsets`,`uint32`,t.slotCount+1),l=e.allocate(`line-neighbors`,`uint32`,i),u=e.allocate(`line-weights`,`float32`,i),d=e.allocate(`overflow`,`uint32`,1);e.graph.add(new N({topology:{...a,nodePositions:o.view},parameters:s.view,output:{topology:{offsets:c.view,neighbors:l.view,weights:u.view},status:{overflow:d.view}}}));let f=e.allocate(`line-costs`,`float32`,t.slotCount);return e.graph.add(new _({topology:{offsets:c.view,neighbors:l.view,weights:u.view},startNodes:e.upload(`start-edges`,`uint32`,r).view,startCosts:e.upload(`start-costs`,`float32`,Float32Array.from(r,e=>t.lengths[e])).view,maximumIterations:64,localIterations:16,output:{costs:f.view}})),{outputs:{origin:e.upload(`origin-position`,`float32x2`,Float32Array.of(t.nodes[n*2],t.nodes[n*2+1])),slots:T(e,t,.012),costs:E(e,{id:`line-costs`,source:f,count:t.slotCount})},update:e=>s.write(O({angleCost:e.angleCost,leftTurnCost:e.leftTurnCost,rightTurnCost:e.rightTurnCost}))}},interaction:y(D,{kind:`pad`,hint:`price the turns`,x:`leftTurnCost`,y:`rightTurnCost`,decorate:e=>({readout:`left ${e.leftTurnCost.toFixed(2)} · right ${e.rightTurnCost.toFixed(2)} · angle ${e.angleCost.toFixed(2)}`})}),draw:({outputs:e,theme:t})=>[S(e.slots,e.costs,{radius:2.4,range:[0,3]}),...x(e.origin,t,{colorIndex:4})]});export{R as default};