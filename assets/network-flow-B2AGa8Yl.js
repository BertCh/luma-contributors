import{n as e,t}from"./network-optimization-bindings-B3Zg-1cS.js";import{r as n,s as r}from"./kernel-C3tfQOn4.js";import{A as i,E as a,T as o,f as s,k as c,p as l,v as u}from"./gpu-contributor-utils-BIwLQON5.js";import{t as d}from"./gpu-scan-B4NNTfF9.js";import{a as f,l as p,o as m}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as h}from"./uint64-wgsl-DCuW6-R6.js";import{n as g,r as _,t as v}from"./network-optimization-status-D0dFmcvb.js";import{t as y}from"./construction-snapshot-DdkX9Pc0.js";import{t as b}from"./define-run-B-brkCe4.js";import{s as x}from"./display-Bu20g6o5.js";import{t as S}from"./pick-Bo3t3hjv.js";import{d as C,h as w,n as T,o as E,p as D,v as O,y as ee}from"./roads-DzovamHj.js";import{Ta as k}from"./entry-preview-D70TdSbK.js";var A=4096,j=4096;function te(e,t,r,i){let{state:a,offsets:o,arcs:s,records:c,maximumRounds:l}=i,u=r.nodeCount,d=r.edges.length,h=!!r.costs,g=h?r.maximumCostPhases??3:r.maximumRelabels??3,_=h?2:1,v=(r,i)=>n(e,`${t}-${r}`,`uint32`,i),y=[v(`height-first`,u*_),v(`height-second`,u*_)],b=v(`excess`,u*_),x=v(`phase-increments`,Math.max(d,1)),S=v(`previous-flow`,Math.max(d,1)),C=(e,t)=>p(e,t),w=(e,t)=>p(e,t,`read_write`),T=(e,t)=>p(e,t,`atomic`),E=f(e,{id:`${t}-push-relabel-dispatch`,operation:`GPUNetworkFlow`,shapes:[[u],[d]]}),D=(n,r,i,a,o)=>m(e,{id:`${t}-${n}`,operation:`GPUNetworkFlow`,variant:h?`primal-dual`:`push-relabel`,invocationCount:r,bindings:i,declarations:`const NONE: u32 = 0xffffffffu;`,body:a,...o===void 0?{}:E.getGate(o)}),O=[];for(let e=0;e<=g;e++){let t=`relabel-${e}`;O.push(D(`${t}-start`,1,[w(`state`,a),E.binding],`
state[stateOffset + 18u] = 0u;
state[stateOffset + 6u] = state[stateOffset];
if (state[stateOffset] != 0u) {
  state[stateOffset + 5u] = 0u; state[stateOffset + 7u] = 0u; state[stateOffset + 17u] = 0u;
  ${E.armWGSL}
} else { ${E.disarmWGSL} }`)),O.push(D(`${t}-initialize-height`,u,[C(`state`,a),w(`height`,y[0])],`
${h?`height[heightOffset + 2u * index] = bitcast<u32>(select(0x7fffffff, 0, index == state[stateOffset + 15u]));
height[heightOffset + 2u * index + 1u] = select(NONE, 0u, index == state[stateOffset + 15u]);`:`height[heightOffset + index] = select(NONE, 0u, index == state[stateOffset + 15u]);`}`,0));for(let e=0;e<l;e++){let n=y[e%2],i=y[1-e%2];O.push(D(`${t}-bfs-${e}`,u,[C(`offsets`,o),C(`arcs`,s),C(`records`,c),C(`flow`,r.output.flow),C(`inputHeight`,n),w(`outputHeight`,i),T(`state`,a)],`
${h?`var distance = bitcast<i32>(inputHeight[inputHeightOffset + 2u * index]);
var hops = inputHeight[inputHeightOffset + 2u * index + 1u];
let originalDistance = distance; let originalHops = hops;`:`var distance = inputHeight[inputHeightOffset + index];
let originalDistance = distance;`}
for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
  let arc = arcs[arcsOffset + slot]; let edge = arc / 2u; let reverse = (arc & 1u) != 0u;
  let amount = flow[flowOffset + edge]; let capacity = records[recordsOffset + 4u * edge + 2u];
  if ((!reverse && amount == capacity) || (reverse && amount == 0u)) { continue; }
  let neighbor = records[recordsOffset + 4u * edge + select(1u, 0u, reverse)];
  ${h?`let neighborDistance = bitcast<i32>(inputHeight[inputHeightOffset + 2u * neighbor]);
  if (neighborDistance == 0x7fffffff) { continue; }
  let unitCost = i32(records[recordsOffset + 4u * edge + 3u]);
  let candidate = neighborDistance + select(unitCost, -unitCost, reverse);
  let candidateHops = inputHeight[inputHeightOffset + 2u * neighbor + 1u] + 1u;
  if (candidate < distance || (candidate == distance && candidateHops < hops)) {
    distance = candidate; hops = candidateHops;
  }`:`let neighborDistance = inputHeight[inputHeightOffset + neighbor];
  if (neighborDistance != NONE) { distance = min(distance, neighborDistance + 1u); }`}
}
${h?`outputHeight[outputHeightOffset + 2u * index] = bitcast<u32>(distance);
outputHeight[outputHeightOffset + 2u * index + 1u] = hops;
if (distance != originalDistance || hops != originalHops) { atomicOr(&state[stateOffset + 5u], 1u); }`:`outputHeight[outputHeightOffset + index] = distance;
if (distance != originalDistance) { atomicOr(&state[stateOffset + 5u], 1u); }`}`,0)),O.push(D(`${t}-bfs-control-${e}`,1,[w(`state`,a),E.binding],`
if (state[stateOffset + 6u] == 0u) { return; }
state[stateOffset + 7u] = 1u - state[stateOffset + 7u];
if (state[stateOffset + 5u] == 0u) {
  state[stateOffset + 6u] = 0u; state[stateOffset + 17u] = 1u; ${E.disarmWGSL}
}
state[stateOffset + 5u] = 0u;`))}O.push(D(`${t}-normalize`,u,[C(`state`,a),C(`first`,y[0]),C(`second`,y[1]),w(`excess`,b)],`
if (state[stateOffset] == 0u) { return; }
${h?`excess[excessOffset + 2u * index] = select(first[firstOffset + 2u * index], second[secondOffset + 2u * index], state[stateOffset + 7u] != 0u);
excess[excessOffset + 2u * index + 1u] = select(first[firstOffset + 2u * index + 1u], second[secondOffset + 2u * index + 1u], state[stateOffset + 7u] != 0u);`:`excess[excessOffset + index] = select(first[firstOffset + index], second[secondOffset + index], state[stateOffset + 7u] != 0u);`}`)),O.push(D(`${t}-height-copy`,u,[C(`state`,a),C(`excess`,b),w(`height`,y[0])],`
if (state[stateOffset] != 0u) {
${h?`height[heightOffset + 2u * index] = excess[excessOffset + 2u * index];
height[heightOffset + 2u * index + 1u] = excess[excessOffset + 2u * index + 1u];`:`height[heightOffset + index] = excess[excessOffset + index];`}
}`)),O.push(D(`${t}-finish-search`,1,[w(`state`,a),C(`height`,y[0]),E.binding],`
${E.disarmWGSL}
if (state[stateOffset] == 0u) { return; }
if (state[stateOffset + 3u] == state[stateOffset + 16u]) {
  state[stateOffset + 2u] = 1u; state[stateOffset] = 0u; return;
}
if (state[stateOffset + 17u] == 0u) { state[stateOffset] = 0u; return; }
if (height[heightOffset + ${h?`2u * state[stateOffset + 14u] + 1u`:`state[stateOffset + 14u]`}] == NONE) {
  state[stateOffset + 2u] = 1u; state[stateOffset] = 0u; return;
}
if (state[stateOffset + 19u] == ${g}u) { state[stateOffset] = 0u; return; }
state[stateOffset + 18u] = 1u; state[stateOffset + 20u] = 0u;
state[stateOffset + 19u] += 1u;
${E.armWGSL}`)),O.push(D(`${t}-initialize-excess`,u,[C(`state`,a),w(`excess`,b)],`
excess[excessOffset + index] = select(0u, state[stateOffset + 16u] - state[stateOffset + 3u], index == state[stateOffset + 14u]);`,0)),O.push(D(`${t}-initialize-increments`,d,[C(`flow`,r.output.flow),w(`previousFlow`,S),w(`increments`,x)],`
previousFlow[previousFlowOffset + index] = flow[flowOffset + index];
increments[incrementsOffset + index] = 0u;`,1));for(let e=0;e<l;e++)O.push(D(`${t}-push-${e}`,u,[C(`offsets`,o),C(`arcs`,s),C(`records`,c),C(`height`,y[0]),T(`flow`,r.output.flow),T(`excess`,b),w(`increments`,x),T(`state`,a)],`
if (index == atomicLoad(&state[stateOffset + 15u])) { return; }
var available = atomicLoad(&excess[excessOffset + index]); let originalAvailable = available;
let vertexHeight = height[heightOffset + ${h?`2u * index + 1u`:`index`}]; if (vertexHeight == NONE || available == 0u) { return; }
for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
  let arc = arcs[arcsOffset + slot]; let edge = arc / 2u; let reverse = (arc & 1u) != 0u;
  let neighbor = records[recordsOffset + 4u * edge + select(1u, 0u, reverse)];
  let neighborHeight = height[heightOffset + ${h?`2u * neighbor + 1u`:`neighbor`}];
  if (neighborHeight == NONE || vertexHeight != neighborHeight + 1u) { continue; }
  ${h?`let unitCost = i32(records[recordsOffset + 4u * edge + 3u]);
  let vertexDistance = bitcast<i32>(height[heightOffset + 2u * index]);
  let neighborDistance = bitcast<i32>(height[heightOffset + 2u * neighbor]);
  if (vertexDistance != neighborDistance + select(unitCost, -unitCost, reverse)) { continue; }`:``}
  let amount = atomicLoad(&flow[flowOffset + edge]);
  let residual = select(records[recordsOffset + 4u * edge + 2u] - amount, amount, reverse);
  let increment = min(available, residual); if (increment == 0u) { continue; }
  if (reverse) { atomicSub(&flow[flowOffset + edge], increment); }
  else { atomicAdd(&flow[flowOffset + edge], increment); }
  increments[incrementsOffset + edge] += increment;
  atomicAdd(&excess[excessOffset + neighbor], increment); available -= increment;
  if (available == 0u) { break; }
}
let sent = originalAvailable - available;
if (sent != 0u) { atomicSub(&excess[excessOffset + index], sent); atomicOr(&state[stateOffset + 5u], 1u); }`,0)),O.push(D(`${t}-push-control-${e}`,1,[w(`state`,a),E.binding],`
if (state[stateOffset + 18u] == 0u) { return; }
if (state[stateOffset + 5u] == 0u) { ${E.disarmWGSL} }
state[stateOffset + 5u] = 0u;`));O.push(D(`${t}-return-start`,1,[C(`state`,a),E.binding],`
if (state[stateOffset + 18u] != 0u) { ${E.armWGSL} } else { ${E.disarmWGSL} }`));for(let e=0;e<l;e++)O.push(D(`${t}-return-${e}`,u,[C(`offsets`,o),C(`arcs`,s),C(`records`,c),C(`height`,y[0]),T(`flow`,r.output.flow),T(`excess`,b),w(`increments`,x),T(`state`,a)],`
if (index == atomicLoad(&state[stateOffset + 15u]) || index == atomicLoad(&state[stateOffset + 14u])) { return; }
var available = atomicLoad(&excess[excessOffset + index]); let originalAvailable = available;
let vertexHeight = height[heightOffset + ${h?`2u * index + 1u`:`index`}]; if (available == 0u || vertexHeight == NONE) { return; }
for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
  let outgoing = arcs[arcsOffset + slot]; let edge = outgoing / 2u; let reverse = (outgoing & 1u) != 0u;
  let neighbor = records[recordsOffset + 4u * edge + select(1u, 0u, reverse)];
  let neighborHeight = height[heightOffset + ${h?`2u * neighbor + 1u`:`neighbor`}];
  if (neighborHeight == NONE || neighborHeight != vertexHeight + 1u) { continue; }
  ${h?`let unitCost = i32(records[recordsOffset + 4u * edge + 3u]);
  let vertexDistance = bitcast<i32>(height[heightOffset + 2u * index]);
  let neighborDistance = bitcast<i32>(height[heightOffset + 2u * neighbor]);
  if (neighborDistance != vertexDistance + select(-unitCost, unitCost, reverse)) { continue; }`:``}
  let decrement = min(available, increments[incrementsOffset + edge]); if (decrement == 0u) { continue; }
  // Undo the incoming push: outgoing forward means the original phase push was reversed.
  if (reverse) { atomicSub(&flow[flowOffset + edge], decrement); }
  else { atomicAdd(&flow[flowOffset + edge], decrement); }
  increments[incrementsOffset + edge] -= decrement;
  atomicAdd(&excess[excessOffset + neighbor], decrement); available -= decrement;
  if (available == 0u) { break; }
}
let returned = originalAvailable - available;
if (returned != 0u) { atomicSub(&excess[excessOffset + index], returned); atomicOr(&state[stateOffset + 5u], 1u); }`,0)),O.push(D(`${t}-return-control-${e}`,1,[w(`state`,a),E.binding],`
if (state[stateOffset + 18u] == 0u) { return; }
if (state[stateOffset + 5u] == 0u) { ${E.disarmWGSL} }
state[stateOffset + 5u] = 0u;`));O.push(D(`${t}-verify`,u,[C(`excess`,b),T(`state`,a)],`
if (atomicLoad(&state[stateOffset + 18u]) != 0u && index != atomicLoad(&state[stateOffset + 14u]) && index != atomicLoad(&state[stateOffset + 15u]) && excess[excessOffset + index] != 0u) {
  atomicOr(&state[stateOffset + 20u], 1u);
}`)),O.push(D(`${t}-restore-if-needed`,d,[C(`state`,a),C(`previousFlow`,S),w(`flow`,r.output.flow)],`
if (state[stateOffset + 18u] != 0u && state[stateOffset + 20u] != 0u) { flow[flowOffset + index] = previousFlow[previousFlowOffset + index]; }`)),O.push(D(`${t}-commit`,1,[w(`state`,a),C(`excess`,b)],`
if (state[stateOffset + 18u] == 0u) { return; }
if (state[stateOffset + 20u] != 0u) { state[stateOffset] = 0u; state[stateOffset + 17u] = 0u; return; }
let increment = excess[excessOffset + state[stateOffset + 15u]];
state[stateOffset + 3u] += increment; state[stateOffset + 4u] += 1u;
if (increment == 0u) { state[stateOffset] = 0u; state[stateOffset + 17u] = 0u; }`))}O.push(D(`cut-start`,1,[w(`state`,a),E.binding],`
state[stateOffset + 6u] = state[stateOffset + 1u]; state[stateOffset + 7u] = 0u;
state[stateOffset + 17u] = 0u; state[stateOffset + 5u] = 0u;
if (state[stateOffset + 1u] != 0u) { ${E.armWGSL} } else { ${E.disarmWGSL} }`)),O.push(D(`cut-initialize`,u,[C(`state`,a),w(`height`,y[0])],`
height[heightOffset + index] = select(NONE, 0u, index == state[stateOffset + 14u]);`,0));for(let e=0;e<l;e++)O.push(D(`cut-bfs-${e}`,u,[C(`offsets`,o),C(`arcs`,s),C(`records`,c),C(`flow`,r.output.flow),C(`inputHeight`,y[e%2]),w(`outputHeight`,y[1-e%2]),T(`state`,a)],`
var distance = inputHeight[inputHeightOffset + index]; let originalDistance = distance;
for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
  let incoming = arcs[arcsOffset + slot] ^ 1u; let edge = incoming / 2u; let reverse = (incoming & 1u) != 0u;
  let amount = flow[flowOffset + edge]; let capacity = records[recordsOffset + 4u * edge + 2u];
  if ((!reverse && amount == capacity) || (reverse && amount == 0u)) { continue; }
  let neighbor = records[recordsOffset + 4u * edge + select(0u, 1u, reverse)];
  let neighborDistance = inputHeight[inputHeightOffset + neighbor];
  if (neighborDistance != NONE) { distance = min(distance, neighborDistance + 1u); }
}
outputHeight[outputHeightOffset + index] = distance;
if (distance != originalDistance) { atomicOr(&state[stateOffset + 5u], 1u); }`,0)),O.push(D(`cut-control-${e}`,1,[w(`state`,a),E.binding],`
if (state[stateOffset + 6u] == 0u) { return; }
state[stateOffset + 7u] = 1u - state[stateOffset + 7u];
if (state[stateOffset + 5u] == 0u) { state[stateOffset + 6u] = 0u; state[stateOffset + 17u] = 1u; ${E.disarmWGSL} }
state[stateOffset + 5u] = 0u;`));return O.push(D(`cut-publish`,u,[C(`state`,a),C(`first`,y[0]),C(`second`,y[1]),w(`cut`,r.output.cut)],`
if (state[stateOffset + 1u] == 0u) { return; }
let distance = select(first[firstOffset + index], second[secondOffset + index], state[stateOffset + 7u] != 0u);
cut[cutOffset + index] = select(0u, 1u, distance != NONE);`)),O.push(D(`cut-finish`,1,[w(`state`,a),C(`cut`,r.output.cut)],`
if (state[stateOffset + 1u] != 0u && state[stateOffset + 17u] != 0u && cut[cutOffset + state[stateOffset + 15u]] == 0u) { state[stateOffset + 2u] = 1u; }`)),O}var M=1024,N=16777215,P=3e7,F=3e7,I=4294967295,L=32767,R=2147483646,z=8192,ne=8,re=1024,ie=3,B=4,V=64,H=21;function U(e){return e.mode===`primal-dual`||e.mode===void 0&&e.maximumCostPhases!==void 0}function W(e){return e.maximumRelaxationRounds??Math.min(e.nodeCount,re)}function G(e){return e.maximumAugmentations??ne}function K(e){return Math.ceil(Math.log2(e.nodeCount))}function q(e){if(!e.costs||U(e)){let t=W(e);return V+(1+(e.costs?e.maximumCostPhases??3:e.maximumRelabels??3))*(6*t+16)+2*t+4}return V+(G(e)+1)*(2*W(e)+3*K(e)+7)}function J(e,t,n){if(s(e,`nodeCount`,t.nodeCount,2,P),s(e,`edges rows`,t.edges.length,0,F),s(e,`maximumRelaxationRounds`,W(t),1,z),s(e,`maximumRelabels`,t.maximumRelabels??3,0,A),s(e,`maximumCostPhases`,t.maximumCostPhases??3,0,j),t.topology){let n=i(e,t.topology);o(e,n.nodeCount===t.nodeCount&&n.edgeCount===t.edges.length,`topology must hold nodeCount vertices and one neighbor per original edge`)}l(e,n)}function ae(e,t,r){let i=r.nodeCount,a=r.edges.length;for(let[n,o]of[[`flow control state`,84],[`dispatch gates`,24],[`edges`,a*8],[`residual CSR arcs`,Math.max(2*a,1)*4],[`edge records`,Math.max(a,1)*16],[`residual CSR offsets`,(i+1)*4],...r.costs&&!U(r)?[[`shortest-path labels`,i*ie*4],[`path doubling scratch`,i*B*4]]:[[`push-relabel heights/excess`,i*(r.costs?8:4)],[`push-relabel edge increments/snapshot`,Math.max(a,1)*4]]])u(e.device,t,n,o,`reduce nodeCount or edges rows, or request a larger device binding limit`);let o=(r,i)=>n(e,`${t}-${r}`,`uint32`,i),s=(e,t)=>p(e,t),c=(e,t)=>p(e,t,`read_write`),l=(e,t)=>p(e,t,`atomic`),g=(n,r,o,s,c,l=``)=>m(e,{id:`${t}-${n}`,operation:`GPUNetworkFlow`,variant:`parallel-residual-csr`,invocationCount:r,bindings:o,declarations:`const NODE_COUNT: u32 = ${i}u;
const EDGE_COUNT: u32 = ${a}u;
const INFINITY: i32 = 0x7fffffff;
const NONE: u32 = 0xffffffffu;
${l}`,body:s,...c?c.getGate(0):{}}),_=o(`state`,H),v=o(`edge-records`,Math.max(a,1)*4),y=o(`incident-counts`,i+1),b=o(`residual-offsets`,i+1),x=o(`residual-arcs`,Math.max(2*a,1)),S=[g(`initialize-state`,H,[c(`state`,_)],`state[stateOffset + index] = 0u;`),g(`initialize-counts`,i+1,[c(`counts`,y),c(`cut`,r.output.cut)],`counts[countsOffset + index] = 0u;
if (index < NODE_COUNT) { cut[cutOffset + index] = 0u; }`),g(`validate-edges`,a,[s(`edges`,r.edges),s(`capacities`,r.capacities),...r.costs?[s(`costs`,r.costs)]:[],c(`flow`,r.output.flow),l(`counts`,y),l(`state`,_),...r.topology?[s(`inputOffsets`,r.topology.offsets),s(`neighbors`,r.topology.neighbors)]:[]],`flow[flowOffset + index] = 0u;
let source = edges[edgesOffset + 2u * index];
let destination = edges[edgesOffset + 2u * index + 1u];
let capacity = capacities[capacitiesOffset + index];
if (source >= NODE_COUNT || destination >= NODE_COUNT || capacity > ${I}u ${r.costs?`|| costs[costsOffset + index] > ${Math.min(L,Math.floor(R/i))}u`:``}) { atomicOr(&state[stateOffset + 8u], 1u); return; }
${r.topology?`if (index < inputOffsets[inputOffsetsOffset + source] || index >= inputOffsets[inputOffsetsOffset + source + 1u] || neighbors[neighborsOffset + index] != destination) { atomicOr(&state[stateOffset + 8u], 1u); }`:``}
let previous = atomicAdd(&state[stateOffset + 9u], capacity);
if (previous > ${I}u - capacity) { atomicOr(&state[stateOffset + 8u], 1u); }
atomicAdd(&counts[countsOffset + source], 1u);
atomicAdd(&counts[countsOffset + destination], 1u);`)];if(r.topology&&S.push(g(`validate-topology`,i+1,[s(`inputOffsets`,r.topology.offsets),l(`state`,_)],`let position = inputOffsets[inputOffsetsOffset + index];
var invalid = position > EDGE_COUNT;
if (index == 0u) { invalid = invalid || position != 0u; }
if (index == NODE_COUNT) { invalid = invalid || position != EDGE_COUNT; }
else { invalid = invalid || position > inputOffsets[inputOffsetsOffset + index + 1u]; }
if (invalid) { atomicOr(&state[stateOffset + 8u], 1u); }`)),S.push(g(`prepare-edge-records`,a,[s(`edges`,r.edges),s(`capacities`,r.capacities),...r.costs?[s(`costs`,r.costs)]:[],c(`records`,v)],`records[recordsOffset + 4u * index] = edges[edgesOffset + 2u * index];
records[recordsOffset + 4u * index + 1u] = edges[edgesOffset + 2u * index + 1u];
records[recordsOffset + 4u * index + 2u] = capacities[capacitiesOffset + index];
records[recordsOffset + 4u * index + 3u] = ${r.costs?`costs[costsOffset + index]`:`0u`};`),...new d({id:`${t}-residual-scan`,input:y,output:b,mode:`exclusive`}).getCommandNodes(e),g(`initialize-cursors`,i,[s(`offsets`,b),c(`counts`,y)],`counts[countsOffset + index] = offsets[offsetsOffset + index];`),g(`scatter-residual-arcs`,a,[s(`edges`,r.edges),s(`state`,_),l(`counts`,y),c(`arcs`,x)],`if (state[stateOffset + 8u] != 0u) { return; }
let source = edges[edgesOffset + 2u * index]; let destination = edges[edgesOffset + 2u * index + 1u];
let forward = atomicAdd(&counts[countsOffset + source], 1u);
let reverse = atomicAdd(&counts[countsOffset + destination], 1u);
arcs[arcsOffset + forward] = 2u * index;
arcs[arcsOffset + reverse] = 2u * index + 1u;`),g(`initialize-problem`,1,[s(`parameters`,r.parameters),c(`state`,_)],`let source = parameters[parametersOffset]; let destination = parameters[parametersOffset + 1u];
let valid = state[stateOffset + 8u] == 0u && source < NODE_COUNT && destination < NODE_COUNT && source != destination;
state[stateOffset] = select(0u, 1u, valid);
state[stateOffset + 1u] = select(0u, 1u, valid);
state[stateOffset + 14u] = source; state[stateOffset + 15u] = destination;
state[stateOffset + 16u] = parameters[parametersOffset + 2u];`)),!r.costs||U(r))S.push(...te(e,t,r,{state:_,offsets:b,arcs:x,records:v,maximumRounds:W(r)}));else{let n=[o(`labels-first`,i*3),o(`labels-second`,i*3)],a=[o(`paths-first`,i*4),o(`paths-second`,i*4)],u=f(e,{id:`${t}-search-dispatch`,operation:`GPUNetworkFlow`,shapes:[[i]]}),d=f(e,{id:`${t}-augment-dispatch`,operation:`GPUNetworkFlow`,shapes:[[i]]}),p=G(r),m=K(r);for(let e=0;e<=p;e++){let t=`search-${e}`;S.push(g(`${t}-start`,1,[c(`state`,_),u.binding],`state[stateOffset + 18u] = 0u; state[stateOffset + 6u] = state[stateOffset];
if (state[stateOffset] != 0u) {
state[stateOffset + 5u] = 0u; state[stateOffset + 7u] = 0u; state[stateOffset + 17u] = 0u;
${u.armWGSL}
} else { ${u.disarmWGSL} }`),g(`${t}-labels`,i,[s(`state`,_),c(`labels`,n[0])],`let source = state[stateOffset + 14u];
labels[labelsOffset + 3u * index] = bitcast<u32>(select(INFINITY, 0, index == source));
labels[labelsOffset + 3u * index + 1u] = select(NONE, 0u, index == source);
labels[labelsOffset + 3u * index + 2u] = NONE;`,u));for(let e=0;e<W(r);e++){let a=n[e%2],o=n[1-e%2];S.push(g(`${t}-relax-${e}`,i,[s(`offsets`,b),s(`arcs`,x),s(`records`,v),s(`flow`,r.output.flow),s(`inputLabels`,a),c(`outputLabels`,o),l(`state`,_)],`let base = 3u * index;
var distance = bitcast<i32>(inputLabels[inputLabelsOffset + base]);
var hops = inputLabels[inputLabelsOffset + base + 1u];
var predecessor = inputLabels[inputLabelsOffset + base + 2u];
let previousDistance = distance; let previousHops = hops;
for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
  // CSR lists outgoing arcs. Its opposite is an incoming arc for this vertex.
  let incoming = arcs[arcsOffset + slot] ^ 1u; let edge = incoming / 2u;
  let reverse = (incoming & 1u) != 0u;
  let amount = flow[flowOffset + edge]; let capacity = records[recordsOffset + 4u * edge + 2u];
  if ((!reverse && amount == capacity) || (reverse && amount == 0u)) { continue; }
  let neighbor = records[recordsOffset + 4u * edge + select(0u, 1u, reverse)];
  let neighborDistance = bitcast<i32>(inputLabels[inputLabelsOffset + 3u * neighbor]);
  if (neighborDistance == INFINITY) { continue; }
  let unitCost = i32(records[recordsOffset + 4u * edge + 3u]);
  let candidate = neighborDistance + select(unitCost, -unitCost, reverse);
  let candidateHops = inputLabels[inputLabelsOffset + 3u * neighbor + 1u] + 1u;
  if (candidate < distance || (candidate == distance && (candidateHops < hops || (candidateHops == hops && incoming < predecessor)))) {
    distance = candidate; hops = candidateHops; predecessor = incoming;
  }
}
outputLabels[outputLabelsOffset + base] = bitcast<u32>(distance);
outputLabels[outputLabelsOffset + base + 1u] = hops;
outputLabels[outputLabelsOffset + base + 2u] = predecessor;
if (distance != previousDistance || hops != previousHops) { atomicOr(&state[stateOffset + 5u], 1u); }`,u),g(`${t}-relax-control-${e}`,1,[c(`state`,_),u.binding],`if (state[stateOffset + 6u] == 0u) { return; }
state[stateOffset + 7u] = 1u - state[stateOffset + 7u];
if (state[stateOffset + 5u] == 0u) {
  state[stateOffset + 6u] = 0u; state[stateOffset + 17u] = 1u; ${u.disarmWGSL}
}
state[stateOffset + 5u] = 0u;`))}S.push(g(`${t}-normalize`,i,[s(`state`,_),s(`first`,n[0]),s(`second`,n[1]),c(`path`,a[0]),c(`cut`,r.output.cut)],`if (state[stateOffset] == 0u) { return; }
let base = 3u * index; let useSecond = state[stateOffset + 7u] != 0u;
let distance = select(first[firstOffset + base], second[secondOffset + base], useSecond);
let predecessor = select(first[firstOffset + base + 2u], second[secondOffset + base + 2u], useSecond);
cut[cutOffset + index] = select(0u, 1u, bitcast<i32>(distance) != INFINITY);
path[pathOffset + 4u * index + 2u] = predecessor;`),g(`${t}-finish`,1,[c(`state`,_),s(`cut`,r.output.cut),d.binding],`${d.disarmWGSL}
if (state[stateOffset] == 0u) { return; }
// The requested amount is already optimal after committed shortest-path augmentations.
// A later cut search need not finish to certify that finite request.
if (state[stateOffset + 3u] == state[stateOffset + 16u]) {
  state[stateOffset + 2u] = 1u; state[stateOffset] = 0u; return;
}
if (state[stateOffset + 17u] == 0u) { state[stateOffset] = 0u; return; }
let reachable = cut[cutOffset + state[stateOffset + 15u]] != 0u;
if (!reachable) {
  state[stateOffset + 2u] = 1u; state[stateOffset] = 0u; return;
}
if (state[stateOffset + 4u] == ${p}u) { state[stateOffset] = 0u; return; }
state[stateOffset + 18u] = 1u;
${d.armWGSL}`),g(`${t}-path-initialize`,i,[s(`edges`,r.edges),s(`capacities`,r.capacities),s(`flow`,r.output.flow),s(`state`,_),c(`path`,a[0])],`let base = 4u * index; let predecessor = path[pathOffset + base + 2u];
var ancestor = index; var residual = ${I}u;
if (predecessor != NONE) {
  let edge = predecessor / 2u; let reverse = (predecessor & 1u) != 0u;
  ancestor = edges[edgesOffset + 2u * edge + select(0u, 1u, reverse)];
  residual = select(capacities[capacitiesOffset + edge] - flow[flowOffset + edge], flow[flowOffset + edge], reverse);
}
path[pathOffset + base] = ancestor; path[pathOffset + base + 1u] = residual;
path[pathOffset + base + 3u] = select(0u, 1u, index == state[stateOffset + 15u]);`,d));for(let e=0;e<m;e++){let n=a[e%2],r=a[1-e%2];S.push(g(`${t}-double-${e}`,i,[s(`inputPath`,n),c(`outputPath`,r)],`let base = 4u * index; let ancestor = inputPath[inputPathOffset + base];
outputPath[outputPathOffset + base] = inputPath[inputPathOffset + 4u * ancestor];
outputPath[outputPathOffset + base + 1u] = min(inputPath[inputPathOffset + base + 1u], inputPath[inputPathOffset + 4u * ancestor + 1u]);
outputPath[outputPathOffset + base + 2u] = inputPath[inputPathOffset + base + 2u];
outputPath[outputPathOffset + base + 3u] = inputPath[inputPathOffset + base + 3u];`,d),g(`${t}-mark-${e}`,i,[s(`inputPath`,n),l(`outputPath`,r)],`let base = 4u * index;
if (inputPath[inputPathOffset + base + 3u] != 0u) {
  let ancestor = inputPath[inputPathOffset + base];
  atomicOr(&outputPath[outputPathOffset + 4u * ancestor + 3u], 1u);
}`,d))}let o=a[m%2];S.push(g(`${t}-increment`,1,[s(`path`,o),c(`state`,_)],`state[stateOffset + 10u] = 0u;
if (state[stateOffset + 18u] == 0u) { return; }
let destination = state[stateOffset + 15u];
let increment = min(path[pathOffset + 4u * destination + 1u], state[stateOffset + 16u] - state[stateOffset + 3u]);
state[stateOffset + 10u] = increment;
state[stateOffset + 3u] += increment; state[stateOffset + 4u] += 1u;`),g(`${t}-augment`,i,[s(`path`,o),s(`state`,_),c(`flow`,r.output.flow)],`let base = 4u * index; let predecessor = path[pathOffset + base + 2u];
if (path[pathOffset + base + 3u] == 0u || predecessor == NONE) { return; }
let edge = predecessor / 2u; let increment = state[stateOffset + 10u];
if ((predecessor & 1u) != 0u) { flow[flowOffset + edge] -= increment; }
else { flow[flowOffset + edge] += increment; }`,d))}}return S.push(g(`collect-objective`,a,[s(`edges`,r.edges),s(`capacities`,r.capacities),s(`flow`,r.output.flow),s(`cut`,r.output.cut),...r.costs?[s(`costs`,r.costs)]:[],l(`state`,_)],`if (atomicLoad(&state[stateOffset + 1u]) == 0u) { return; }
let source = edges[edgesOffset + 2u * index]; let destination = edges[edgesOffset + 2u * index + 1u];
if (cut[cutOffset + source] != 0u && cut[cutOffset + destination] == 0u) {
  atomicAdd(&state[stateOffset + 11u], capacities[capacitiesOffset + index]);
}
let amount = flow[flowOffset + index]; let unitCost = ${r.costs?`costs[costsOffset + index]`:`0u`};
let product = multiplyUint32Wide(amount, unitCost);
let previous = atomicAdd(&state[stateOffset + 12u], product.y);
atomicAdd(&state[stateOffset + 13u], product.x + select(0u, 1u, previous > 0xffffffffu - product.y));`,void 0,h),g(`publish`,1,[s(`state`,_),s(`cut`,r.output.cut),c(`summary`,r.output.summary),...r.output.exactCost?[c(`exactCost`,r.output.exactCost)]:[]],`let valid = state[stateOffset + 1u];
for (var word = 0u; word < 8u; word++) { summary[summaryOffset + word] = 0u; }
${r.output.exactCost?`exactCost[exactCostOffset] = 0u; exactCost[exactCostOffset + 1u] = 0u;`:``}
if (valid == 0u) { return; }
let value = state[stateOffset + 3u]; let required = state[stateOffset + 16u];
summary[summaryOffset] = 1u; summary[summaryOffset + 1u] = state[stateOffset + 2u];
summary[summaryOffset + 2u] = value;
summary[summaryOffset + 3u] = select(required - value, 0u, required == NONE);
summary[summaryOffset + 4u] = state[stateOffset + 4u];
summary[summaryOffset + 5u] = bitcast<u32>(uint64ToFloat(vec2<u32>(state[stateOffset + 13u], state[stateOffset + 12u])));
summary[summaryOffset + 6u] = state[stateOffset + 11u];
summary[summaryOffset + 7u] = select(0u, 1u, state[stateOffset + 17u] != 0u && cut[cutOffset + state[stateOffset + 15u]] == 0u && state[stateOffset + 11u] == value);
${r.output.exactCost?`exactCost[exactCostOffset] = state[stateOffset + 12u]; exactCost[exactCostOffset + 1u] = state[stateOffset + 13u];`:``}`,void 0,h)),S}function oe(e,t,n,r){s(e,`nodeCount with output.paths`,t,2,128),s(e,`edgeCount with output.paths`,n,0,M),s(e,`pathCapacity`,r,1,M)}function Y(e){return e?[e.edgeRows,e.lengths,e.volumes,...Object.values(e.status)]:[]}function se(e,t=`network-flow`){if(!e)return;o(t,typeof e==`object`,`output.paths must be an object`);let n=y(e,[`edgeRows`,`lengths`,`volumes`,`status`]);o(t,typeof n.status==`object`&&n.status!==null,`output.paths.status is required`);let r=Object.freeze({...n,status:y(n.status,[`count`,`requiredCount`,`overflow`,`invalidCount`,`converged`,`iterationLimitReached`])});return X(t,r),r}function X(e,t){o(e,typeof t==`object`&&!!t,`output.paths is required`),o(e,typeof t.status==`object`&&t.status!==null,`output.paths.status is required`);for(let n of[`edgeRows`,`lengths`,`volumes`])o(e,t[n]!==void 0&&t[n]!==null,`output.paths.${n} is required`);for(let n of[`count`,`requiredCount`,`overflow`,`invalidCount`,`converged`,`iterationLimitReached`])o(e,t.status?.[n]!==void 0&&t.status?.[n]!==null,`output.paths.status.${n} is required`)}function ce(e,t,n,r){if(!r)return;X(e,r),oe(e,t,n,r.volumes.length);let i={edgeRows:r.volumes.length*(t-1),lengths:r.volumes.length,volumes:r.volumes.length,count:1,requiredCount:1,overflow:1,invalidCount:1,converged:1,iterationLimitReached:1},s={...r,...r.status};for(let[t,n]of Object.entries(i)){let i=s[t],c=`output.paths.${t in r.status?`status.`:``}${t}`;o(e,!!i,`${c} is required`),a(i,[`uint32`],e),o(e,i.length===n,`${c} must hold ${n} rows`)}}function le(e,t,r,i,a,o,s,c){let l=a.length,u=c.volumes.length,d=n(e,`${t}-path-required-count`,`uint32`,1);return[m(e,{id:`${t}-paths`,operation:`GPUNetworkFlow`,variant:`reference-path-decomposition`,invocationCount:1,workgroupSize:2,bindings:[{name:`edges`,view:i,type:`u32`,access:`read`},{name:`flow`,view:a,type:`u32`,access:`read`},{name:`parameters`,view:o,type:`u32`,access:`read`},{name:`summary`,view:s,type:`u32`,access:`read`},{name:`edgeRows`,view:c.edgeRows,type:`u32`,access:`read_write`},{name:`lengths`,view:c.lengths,type:`u32`,access:`read_write`},{name:`volumes`,view:c.volumes,type:`u32`,access:`read_write`},{name:`requiredCount`,view:d,type:`u32`,access:`read_write`}],body:`
requiredCount[requiredCountOffset] = 0u;
for (var row = 0u; row < ${u}u; row++) {
  lengths[lengthsOffset + row] = 0u;
  volumes[volumesOffset + row] = 0u;
}
for (var slot = 0u; slot < ${c.edgeRows.length}u; slot++) { edgeRows[edgeRowsOffset + slot] = 0xffffffffu; }
if (summary[summaryOffset] == 0u) { return; }
let source = parameters[parametersOffset];
let sink = parameters[parametersOffset + 1u];
var remainingValue = summary[summaryOffset + 2u];
// Runtime bounds prevent constant-loop integer-array initialization miscompilation on Metal.
let vertexLimit = select(0u, ${r}u, summary[summaryOffset] != 0u);
let edgeLimit = min(${l}u, arrayLength(&flow) - flowOffset);
var remaining: array<u32, ${Math.max(1,l)}>;
var parents: array<u32, ${r}>;
var queue: array<u32, ${r}>;
for (var edge = 0u; edge < edgeLimit; edge++) { remaining[edge] = flow[flowOffset + edge]; }
for (var path = 0u; path < edgeLimit && remainingValue > 0u; path++) {
  for (var vertex = 0u; vertex < vertexLimit; vertex++) { parents[vertex] = 0xffffffffu; }
  parents[source] = 0xfffffffeu;
  queue[0] = source;
  var tail = 1u;
  for (var head = 0u; head < tail && parents[sink] == 0xffffffffu; head++) {
    let sourceVertex = queue[head];
    for (var edge = 0u; edge < edgeLimit; edge++) {
      let destinationVertex = edges[edgesOffset + edge * 2u + 1u];
      if (edges[edgesOffset + edge * 2u] == sourceVertex && remaining[edge] > 0u && parents[destinationVertex] == 0xffffffffu) {
        parents[destinationVertex] = edge;
        queue[tail] = destinationVertex;
        tail++;
      }
    }
  }
  var volume = remainingValue;
  var length = 0u;
  var vertex = sink;
  loop {
    if (vertex == source) { break; }
    let edge = parents[vertex];
    volume = min(volume, remaining[edge]);
    vertex = edges[edgesOffset + edge * 2u];
    length++;
  }
  vertex = sink;
  var slot = length;
  loop {
    if (vertex == source) { break; }
    let edge = parents[vertex];
    remaining[edge] -= volume;
    slot--;
    if (path < ${u}u) { edgeRows[edgeRowsOffset + path * ${r-1}u + slot] = edge; }
    vertex = edges[edgesOffset + edge * 2u];
  }
  if (path < ${u}u) {
    lengths[lengthsOffset + path] = length;
    volumes[volumesOffset + path] = volume;
  }
  remainingValue -= volume;
  requiredCount[requiredCountOffset] = path + 1u;
}`}),m(e,{id:`${t}-path-status`,operation:`GPUNetworkFlow`,variant:`path-status`,invocationCount:1,workgroupSize:2,bindings:[{name:`required`,view:d,type:`u32`,access:`read`},{name:`summary`,view:s,type:`u32`,access:`read`},...Object.entries(c.status).map(([e,t])=>({name:e,view:t,type:`u32`,access:`read_write`}))],body:`
count[countOffset] = min(required[requiredOffset], ${u}u);
requiredCount[requiredCountOffset] = required[requiredOffset];
overflow[overflowOffset] = select(0u, 1u, required[requiredOffset] > ${u}u);
invalidCount[invalidCountOffset] = 1u - summary[summaryOffset];
converged[convergedOffset] = summary[summaryOffset] * summary[summaryOffset + 1u];
iterationLimitReached[iterationLimitReachedOffset] = summary[summaryOffset] * (1u - summary[summaryOffset + 1u]);`})]}var ue=65536,de=1024,Z=3,Q=8;function fe(e,t){return e.solver===`parallel`||(e.solver??`auto`)===`auto`&&(e.mode!==void 0||e.maximumCostPhases!==void 0||e.topology!==void 0||e.nodeCount>128||t>1024)?`parallel`:`reference`}function pe(e,t,n,r){o(e,t.solver===void 0||[`auto`,`reference`,`parallel`].includes(t.solver),`solver must be auto, reference or parallel`);let i=fe(t,n);return i===`reference`&&me(e,t),o(e,t.mode===void 0||[`primal-dual`,`successive-shortest-path`].includes(t.mode),`mode must be primal-dual or successive-shortest-path`),o(e,t.mode===void 0||r,`mode requires costs`),i===`parallel`&&(o(e,!r||t.maximumRelabels===void 0,`maximumRelabels requires uncosted parallel flow`),o(e,t.maximumCostPhases===void 0||r&&t.mode!==`successive-shortest-path`,`maximumCostPhases requires costed primal-dual mode`)),i}function me(e,t){o(e,t.maximumRelaxationRounds===void 0&&t.maximumRelabels===void 0&&t.mode===void 0&&t.maximumCostPhases===void 0,`maximumRelaxationRounds, maximumRelabels, mode and maximumCostPhases require solver parallel`)}var he=class{id;props;solver;estimatedCommandNodeCount;constructor(t){t=ge(t),this.props=t,this.id=t.id??`network-flow`;let{id:n}=this;if(this.solver=pe(n,t,t.edges.length,!!t.costs),s(n,`maximumAugmentations`,t.maximumAugmentations??de,0,ue),this.estimatedCommandNodeCount=(this.solver===`parallel`?q(t):1+Number(!!t.topology)+Number(g(t.output.status).length>0))+(t.output.paths?2:0),this.solver===`parallel`)J(n,t,this.estimatedCommandNodeCount);else if(s(n,`nodeCount`,t.nodeCount,2,128),s(n,`edges rows`,t.edges.length,0,M),t.topology){let e=i(n,t.topology);o(n,e.nodeCount===t.nodeCount&&e.edgeCount===t.edges.length,`topology must hold nodeCount vertices and one neighbor per original edge`)}a(t.edges,[`uint32x2`],this.id);for(let e of[t.capacities,t.costs,t.parameters,t.output.flow,t.output.cut,t.output.summary])e&&a(e,[`uint32`],this.id);o(n,t.capacities.length===t.edges.length,`capacities must hold one row per edge`),o(n,!t.costs||t.costs.length===t.edges.length,`costs must hold one row per edge`),o(n,t.parameters.length===Z,`parameters must hold ${Z} rows: source, sink, required flow`),o(n,t.output.flow.length===t.edges.length,`output.flow must hold one row per edge`),o(n,t.output.cut.length===t.nodeCount,`output.cut must hold nodeCount rows`),o(n,t.output.summary.length===Q,`output.summary must hold ${Q} rows`),t.output.exactCost&&(a(t.output.exactCost,[`uint32x2`],this.id),o(n,t.output.exactCost.length===1,`output.exactCost must hold one row`)),ce(n,t.nodeCount,t.edges.length,t.output.paths),_(n,[t.output.flow,t.output.cut,t.output.summary,...t.output.exactCost?[t.output.exactCost]:[],...Y(t.output.paths)],t.output.status),e(this.id,$(t.output),[t.edges,t.capacities,t.costs,t.parameters,...t.topology?c(t.topology):[]])}getCommandNodes(e){let{props:n,id:i}=this;for(let[t,a]of[n.edges,n.capacities,n.costs,n.parameters,...n.topology?c(n.topology):[],...$(n.output)].entries())a&&u(e.device,i,`view ${t} aligned binding range`,r(a).size,`reduce the view length or its alignment prefix`);t(e,i,Object.fromEntries([n.edges,n.capacities,n.costs,n.parameters,...n.topology?c(n.topology):[],...$(n.output)].map((e,t)=>[`view ${t}`,e])));let a=this.solver===`parallel`?ae(e,i,n):[m(e,{id:i,operation:`GPUNetworkFlow`,variant:n.costs?`minimum-cost`:`maximum-flow`,invocationCount:1,workgroupSize:2,bindings:[{name:`edges`,view:n.edges,type:`u32`,access:`read`},{name:`capacities`,view:n.capacities,type:`u32`,access:`read`},...n.costs?[{name:`costs`,view:n.costs,type:`u32`,access:`read`}]:[],{name:`parameters`,view:n.parameters,type:`u32`,access:`read`},{name:`flow`,view:n.output.flow,type:`u32`,access:`read_write`},{name:`cut`,view:n.output.cut,type:`u32`,access:`read_write`},{name:`summary`,view:n.output.summary,type:`u32`,access:`read_write`},...n.output.exactCost?[{name:`exactCost`,view:n.output.exactCost,type:`u32`,access:`read_write`}]:[]],declarations:`const NODE_COUNT: u32 = ${n.nodeCount}u;
const EDGE_COUNT: u32 = ${n.edges.length}u;
const MAXIMUM_AUGMENTATIONS: u32 = ${n.maximumAugmentations??1024}u;
fn edgeCost(edge: u32) -> i32 { return ${n.costs?`i32(costs[costsOffset + edge])`:`0`}; }`,body:`${n.output.exactCost?`exactCost[exactCostOffset] = 0u; exactCost[exactCostOffset + 1u] = 0u;`:``}
for (var word = 0u; word < 8u; word++) { summary[summaryOffset + word] = 0u; }
for (var edge = 0u; edge < EDGE_COUNT; edge++) { flow[flowOffset + edge] = 0u; }
for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { cut[cutOffset + vertex] = 0u; }
let source = parameters[parametersOffset];
let destination = parameters[parametersOffset + 1u];
let required = parameters[parametersOffset + 2u];
if (source >= NODE_COUNT || destination >= NODE_COUNT || source == destination) { return; }
var totalCapacity = 0u;
for (var edge = 0u; edge < EDGE_COUNT; edge++) {
  let endpoints = vec2<u32>(edges[edgesOffset + 2u * edge], edges[edgesOffset + 2u * edge + 1u]);
  let capacity = capacities[capacitiesOffset + edge];
  if (any(endpoints >= vec2<u32>(NODE_COUNT)) || capacity > ${N}u - totalCapacity || edgeCost(edge) < 0 || edgeCost(edge) > ${L}) { return; }
  totalCapacity += capacity;
}
summary[summaryOffset] = 1u;
var value = 0u;
var augmentations = 0u;
var complete = false;
var exhausted = false;
var distances: array<i32, ${n.nodeCount}>;
var predecessor: array<u32, ${n.nodeCount}>;
var reversed: array<bool, ${n.nodeCount}>;
loop {
  for (var vertex = 0u; vertex < NODE_COUNT; vertex++) {
    distances[vertex] = 0x3fffffff;
    predecessor[vertex] = 0xffffffffu;
  }
  distances[source] = 0;
  for (var round = 1u; round < NODE_COUNT; round++) {
    var changed = false;
    for (var edge = 0u; edge < EDGE_COUNT; edge++) {
      let endpoints = vec2<u32>(edges[edgesOffset + 2u * edge], edges[edgesOffset + 2u * edge + 1u]);
      let edgeFlow = flow[flowOffset + edge];
      let cost = edgeCost(edge);
      if (edgeFlow < capacities[capacitiesOffset + edge] && distances[endpoints.x] != 0x3fffffff && distances[endpoints.x] + cost < distances[endpoints.y]) {
        distances[endpoints.y] = distances[endpoints.x] + cost;
        predecessor[endpoints.y] = edge;
        reversed[endpoints.y] = false;
        changed = true;
      }
      if (edgeFlow > 0u && distances[endpoints.y] != 0x3fffffff && distances[endpoints.y] - cost < distances[endpoints.x]) {
        distances[endpoints.x] = distances[endpoints.y] - cost;
        predecessor[endpoints.x] = edge;
        reversed[endpoints.x] = true;
        changed = true;
      }
    }
    if (!changed) { break; }
  }
  exhausted = distances[destination] == 0x3fffffff;
  if (exhausted || value == required) { complete = true; break; }
  if (augmentations == MAXIMUM_AUGMENTATIONS) { break; }
  var increment = min(${N}u, required - value);
  var cursor = destination;
  var hops = 0u;
  loop {
    if (cursor == source || hops == NODE_COUNT) { break; }
    let edge = predecessor[cursor];
    if (edge >= EDGE_COUNT) { summary[summaryOffset] = 0u; return; }
    let endpoints = vec2<u32>(edges[edgesOffset + 2u * edge], edges[edgesOffset + 2u * edge + 1u]);
    increment = min(increment, select(capacities[capacitiesOffset + edge] - flow[flowOffset + edge], flow[flowOffset + edge], reversed[cursor]));
    cursor = select(endpoints.x, endpoints.y, reversed[cursor]);
    hops++;
  }
  if (cursor != source || increment == 0u) { summary[summaryOffset] = 0u; return; }
  cursor = destination;
  loop {
    if (cursor == source) { break; }
    let edge = predecessor[cursor];
    let endpoints = vec2<u32>(edges[edgesOffset + 2u * edge], edges[edgesOffset + 2u * edge + 1u]);
    if (reversed[cursor]) { flow[flowOffset + edge] -= increment; }
    else { flow[flowOffset + edge] += increment; }
    cursor = select(endpoints.x, endpoints.y, reversed[cursor]);
  }
  value += increment;
  augmentations++;
}
var cost = 0.0; var exactLow = 0u; var exactHigh = 0u;
var cutCapacity = 0u;
for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { cut[cutOffset + vertex] = select(0u, 1u, distances[vertex] != 0x3fffffff); }
for (var edge = 0u; edge < EDGE_COUNT; edge++) {
  let endpoints = vec2<u32>(edges[edgesOffset + 2u * edge], edges[edgesOffset + 2u * edge + 1u]);
  let amount = flow[flowOffset + edge]; let unitCost = u32(edgeCost(edge));
  cost += f32(amount) * f32(unitCost);
  let lowProduct = (amount & 0xffffu) * unitCost; let highProduct = (amount >> 16u) * unitCost;
  let productLow = lowProduct + (highProduct << 16u); let productHigh = (highProduct >> 16u) + select(0u, 1u, productLow < lowProduct);
  let previousLow = exactLow; exactLow += productLow; exactHigh += productHigh + select(0u, 1u, exactLow < previousLow);
  if (distances[endpoints.x] != 0x3fffffff && distances[endpoints.y] == 0x3fffffff) { cutCapacity += capacities[capacitiesOffset + edge]; }
}
${n.output.exactCost?`exactCost[exactCostOffset] = exactLow; exactCost[exactCostOffset + 1u] = exactHigh;`:``}
summary[summaryOffset + 1u] = select(0u, 1u, complete);
summary[summaryOffset + 2u] = value;
summary[summaryOffset + 3u] = select(required - value, 0u, required == 0xffffffffu);
summary[summaryOffset + 4u] = augmentations;
summary[summaryOffset + 5u] = bitcast<u32>(cost);
summary[summaryOffset + 6u] = cutCapacity;
summary[summaryOffset + 7u] = select(0u, 1u, exhausted && cutCapacity == value);`})];return this.solver===`reference`&&n.topology&&a.push(m(e,{id:`${i}-validate-reference-topology`,operation:`GPUNetworkFlow`,variant:`validate-reference-topology`,invocationCount:1,workgroupSize:2,bindings:[{name:`edges`,view:n.edges,type:`u32`,access:`read`},{name:`offsets`,view:n.topology.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:n.topology.neighbors,type:`u32`,access:`read`},{name:`flow`,view:n.output.flow,type:`u32`,access:`read_write`},{name:`cut`,view:n.output.cut,type:`u32`,access:`read_write`},{name:`summary`,view:n.output.summary,type:`u32`,access:`read_write`},...n.output.exactCost?[{name:`exactCost`,view:n.output.exactCost,type:`u32`,access:`read_write`}]:[]],declarations:`const NODE_COUNT: u32 = ${n.nodeCount}u; const EDGE_COUNT: u32 = ${n.edges.length}u;`,body:`var valid = offsets[offsetsOffset] == 0u && offsets[offsetsOffset + NODE_COUNT] == EDGE_COUNT;
for (var vertex = 0u; vertex < NODE_COUNT; vertex++) {
  valid = valid && offsets[offsetsOffset + vertex] <= offsets[offsetsOffset + vertex + 1u] && offsets[offsetsOffset + vertex + 1u] <= EDGE_COUNT;
}
for (var edge = 0u; edge < EDGE_COUNT; edge++) {
  let source = edges[edgesOffset + 2u * edge];
  let destination = edges[edgesOffset + 2u * edge + 1u];
  if (source >= NODE_COUNT || destination >= NODE_COUNT) { valid = false; }
  else { valid = valid && edge >= offsets[offsetsOffset + source] && edge < offsets[offsetsOffset + source + 1u] && neighbors[neighborsOffset + edge] == destination; }
}
if (valid) { return; }
for (var edge = 0u; edge < EDGE_COUNT; edge++) { flow[flowOffset + edge] = 0u; }
for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { cut[cutOffset + vertex] = 0u; }
for (var word = 0u; word < 8u; word++) { summary[summaryOffset + word] = 0u; }
${n.output.exactCost?`exactCost[exactCostOffset] = 0u; exactCost[exactCostOffset + 1u] = 0u;`:``}`})),a.push(...v(e,i,`GPUNetworkFlow`,n.output.summary,n.output.status,{invalidCount:`1u - summary[summaryOffset]`,iterationCount:`summary[summaryOffset + 4u]`,converged:`summary[summaryOffset] * summary[summaryOffset + 1u]`,iterationLimitReached:`summary[summaryOffset] * (1u - summary[summaryOffset + 1u])`})),n.output.paths&&a.push(...le(e,i,n.nodeCount,n.edges,n.output.flow,n.parameters,n.output.summary,n.output.paths)),a}};function ge(e){let t=y(e,[`id`,`nodeCount`,`edges`,`capacities`,`costs`,`parameters`,`topology`,`solver`,`mode`,`maximumCostPhases`,`maximumAugmentations`,`maximumRelaxationRounds`,`maximumRelabels`,`output`]),n=y(t.output,[`flow`,`cut`,`summary`,`exactCost`,`paths`,`status`]);return Object.freeze({...t,topology:t.topology&&y(t.topology,[`offsets`,`neighbors`,`weights`,`nodePositions`]),output:Object.freeze({...n,paths:se(n.paths,t.id??`network-flow`),status:n.status&&y(n.status,Array.from(new Set([...Object.keys(n.status),`invalidCount`,`iterationCount`,`converged`,`iterationLimitReached`])))})})}function $(e){return[e.flow,e.cut,e.summary,...e.exactCost?[e.exactCost]:[],...Y(e.paths),...g(e.status)]}var _e=b({parameters:k,caption:`Flow carried by each directed road; dots mark the residual source side of the cut.`,createInputs:({random:e,world:t})=>{let n=E(e,t,54),r=new Uint32Array(n.slotCount*2);for(let e=0;e<n.slotCount;e++)r[e*2]=n.slotSources[e],r[e*2+1]=n.neighbors[e];let i=Uint32Array.from({length:n.slotCount},(e,t)=>1+(Math.imul(Math.min(t,n.reverseSlots[t])+1,2654435761)>>>0)%3),a=Uint32Array.from(n.lengths,e=>Math.round(e*100)),o=n.findNearestNode(t.width,.5);return{network:n,edges:r,base:i,costs:a,source:n.findNearestNode(0,.5),defaultSink:o,sink:o,index:T(n)}},build(e,t){let{network:n}=t,r=e.allocate(`capacities`,`uint32`,n.slotCount),i=e.allocate(`parameters`,`uint32`,3),a=e.allocate(`flow`,`uint32`,n.slotCount),o=e.allocate(`cut`,`uint32`,n.nodeCount),s=e.allocate(`summary`,`uint32`,8),c=e.allocate(`sink-position`,`float32x2`,1);return e.graph.add(new he({nodeCount:n.nodeCount,edges:e.upload(`edges`,`uint32x2`,t.edges).view,capacities:r.view,costs:e.upload(`costs`,`uint32`,t.costs).view,parameters:i.view,output:{flow:a.view,cut:o.view,summary:s.view}})),{outputs:{...O(e,n),slots:ee(e,n,.006),flow:x(e,{id:`flow`,source:a,count:n.slotCount}),cut:o,summary:s,sinkPosition:c,sourcePosition:e.upload(`source-position`,`float32x2`,Float32Array.of(n.nodes[t.source*2],n.nodes[t.source*2+1]))},update:e=>{c.write(Float32Array.of(n.nodes[t.sink*2],n.nodes[t.sink*2+1])),r.write(Uint32Array.from(t.base,t=>t*Math.round(e.capacity))),i.write(Uint32Array.of(t.source,t.sink,Math.round(e.requiredFlow)))}}},interaction:{kind:`place`,hint:`place the sink`,rest:({inputs:e,toUnit:t})=>t([e.network.nodes[e.defaultSink*2],e.network.nodes[e.defaultSink*2+1]]),resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=S(t.index,e.x,e.y,t.sink),o=a!==t.sink;t.sink=a;let s=i(n.summary),c=s?`flow ${s[2]} of ${Math.round(r.requiredFlow)} · cut capacity ${s[6]}`:`required flow ${Math.round(r.requiredFlow)}`;return{rerun:o,glyphs:[{shape:`crosshair`,at:[t.network.nodes[a*2],t.network.nodes[a*2+1]],size:14}],readout:c}}},draw:({outputs:e,theme:t})=>[D(e),w(e.slots,e.flow,{radius:2.2,range:[-2.5,7]}),{layer:`dots`,positions:e.nodes,values:e.cut,ramp:`categorical`,categories:[[...t.muted,0],[...t.categories[3],1]],radius:4.5},...C(e.sourcePosition,t,{colorIndex:4}),...C(e.sinkPosition,t,{colorIndex:2})]});export{_e as default};