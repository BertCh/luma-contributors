import{t as e}from"./network-optimization-bindings-B3Zg-1cS.js";import{r as t,s as n}from"./kernel-C3tfQOn4.js";import{E as r,O as i,T as a,f as o,p as s,v as c,x as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{i as d}from"./graph-chunk-dispatch-zBt8k2-o.js";import{m as f,t as p}from"./gpu-network-reachability-Coypydjh.js";import{o as m}from"./wgsl-kernel-nodes-DagJklH2.js";import{u as h}from"./wgsl-common-DfyXiYB0.js";import{t as g}from"./gpu-scan-B4NNTfF9.js";import{t as _}from"./define-run-B-brkCe4.js";import{a as v,c as y,d as b,h as x,o as S,y as C}from"./roads-DzovamHj.js";import{ga as w}from"./entry-preview-D70TdSbK.js";function T(e,t,n){return[m(e,{id:t,operation:`GPUNetworkPointRouting`,variant:`directed-edge-positions`,invocationCount:n.queries.length,workgroupSize:32,bindings:[{name:`turns`,view:n.turns,type:`u32`,access:`read`},...n.turnPenalties?[{name:`turnPenalties`,view:n.turnPenalties,type:`f32`,access:`read`}]:[],{name:`costs`,view:n.costs,type:`f32`,access:`read`},{name:`queries`,view:n.queries,type:`u32`,access:`read`},{name:`fractions`,view:n.fractions,type:`f32`,access:`read`},{name:`outputCosts`,view:n.output.costs,type:`f32`,access:`read_write`},{name:`status`,view:n.output.status,type:`u32`,access:`read_write`},{name:`predecessors`,view:n.output.predecessorEdges,type:`u32`,access:`read_write`}],declarations:`const EDGE_COUNT: u32 = ${n.costs.length}u;
${h}`,body:`var infinityBits = 0x7f800000u; let infinity = bitcast<f32>(infinityBits);
status[statusOffset + index] = 1u; outputCosts[outputCostsOffset + index] = infinity;
var distances: array<f32, ${n.costs.length}>;
for (var edge = 0u; edge < EDGE_COUNT; edge++) { distances[edge] = infinity; predecessors[predecessorsOffset + index * EDGE_COUNT + edge] = 0xffffffffu; }
let source = queries[queriesOffset + 2u * index]; let destination = queries[queriesOffset + 2u * index + 1u];
let sourceFraction = fractions[fractionsOffset + 2u * index]; let targetFraction = fractions[fractionsOffset + 2u * index + 1u];
if (source >= EDGE_COUNT || destination >= EDGE_COUNT || !isFiniteValue(sourceFraction) || !isFiniteValue(targetFraction) || sourceFraction < 0.0 || sourceFraction > 1.0 || targetFraction < 0.0 || targetFraction > 1.0) { return; }
if (!isFiniteValue(costs[costsOffset + source]) || costs[costsOffset + source] < 0.0 || !isFiniteValue(costs[costsOffset + destination]) || costs[costsOffset + destination] < 0.0) { return; }
for (var turn = 0u; turn < ${n.turns.length}u; turn++) { if (turns[turnsOffset + 2u * turn] >= EDGE_COUNT || turns[turnsOffset + 2u * turn + 1u] >= EDGE_COUNT) { return; }
${n.turnPenalties?`let penalty = turnPenalties[turnPenaltiesOffset + turn]; if (!isFiniteValue(penalty) || penalty < 0.0) { return; }`:``} }
var best = infinity; var targetPredecessor = 0xffffffffu;
if (source == destination && targetFraction >= sourceFraction) { best = (targetFraction - sourceFraction) * costs[costsOffset + source]; }
distances[source] = (1.0 - sourceFraction) * costs[costsOffset + source];
var converged = false;
for (var round = 0u; round < ${n.maximumRelaxations??n.costs.length}u; round++) {
  var changed = false;
  for (var turn = 0u; turn < ${n.turns.length}u; turn++) {
    let incoming = turns[turnsOffset + 2u * turn]; let outgoing = turns[turnsOffset + 2u * turn + 1u];
    let penalty = ${n.turnPenalties?`turnPenalties[turnPenaltiesOffset + turn]`:`0.0`};
    if (!isFiniteValue(distances[incoming]) || !isFiniteValue(costs[costsOffset + outgoing]) || costs[costsOffset + outgoing] < 0.0) { continue; }
    let candidate = distances[incoming] + (penalty + costs[costsOffset + outgoing]);
    if (!isFiniteValue(candidate)) { return; }
    if (candidate < distances[outgoing]) { distances[outgoing] = candidate; predecessors[predecessorsOffset + index * EDGE_COUNT + outgoing] = incoming; changed = true; }
  }
  if (!changed) { converged = true; break; }
}
status[statusOffset + index] = select(2u, 0u, converged);
// Reduce the partial destination separately: its full-edge labels may have absorbed a penalty.
if (${n.maximumRelaxations??n.costs.length}u > 0u) {
  for (var turn = 0u; turn < ${n.turns.length}u; turn++) {
    let incoming = turns[turnsOffset + 2u * turn]; let outgoing = turns[turnsOffset + 2u * turn + 1u];
    if (outgoing != destination || !isFiniteValue(distances[incoming])) { continue; }
    var cursor = incoming;
    for (var hop = 0u; hop < EDGE_COUNT && cursor != source; hop++) {
      if (cursor >= EDGE_COUNT || cursor == destination) { break; }
      cursor = predecessors[predecessorsOffset + index * EDGE_COUNT + cursor];
    }
    if (cursor != source) { continue; }
    let penalty = ${n.turnPenalties?`turnPenalties[turnPenaltiesOffset + turn]`:`0.0`};
    let targetCost = distances[incoming] + (penalty + targetFraction * costs[costsOffset + destination]);
    if (targetCost < best) { best = targetCost; targetPredecessor = incoming; }
  }
}
outputCosts[outputCostsOffset + index] = best;
predecessors[predecessorsOffset + index * EDGE_COUNT + destination] = targetPredecessor;`})]}function E(e,n,r,i,a){let o=r.costs.length,s=r.turns.length,c=(r,i)=>t(e,`${n}-${r}`,`uint32`,i),l=(r,i)=>t(e,`${n}-${r}`,`float32`,i),u=c(`counts`,o+1),d=c(`offsets`,o+1),f=c(`neighbors`,Math.max(s,1)),_=l(`weights`,Math.max(s,1)),v=c(`invalid-network`,1),y=c(`start-nodes`,1),b=l(`start-costs`,1),x=l(`distances`,o),S=c(`predecessors`,o),C=c(`converged`,1),w=c(`query-state`,5),T=t(e,`${n}-paths-first`,`uint32x2`,o),E=t(e,`${n}-paths-second`,`uint32x2`,o),D=e.createDataView(w.buffer,{format:`uint32`,length:1,byteOffset:12}),O=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read`}),k=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),A=`const EDGE_COUNT: u32 = ${o}u;
const TURN_COUNT: u32 = ${s}u;
const NONE: u32 = 0xffffffffu;
const INFINITY_BITS: u32 = 0x7f800000u;
${h}`,j=r.turnPenalties?[O(`turnPenalties`,r.turnPenalties,`f32`)]:[],M=r.turnPenalties?`turnPenalties[turnPenaltiesOffset + index]`:`0.0`,N=(t,r,i,a)=>m(e,{id:`${n}-${t}`,operation:`GPUNetworkPointRouting`,variant:t.replace(/query-\d+-/,``),invocationCount:r,bindings:i,declarations:A,body:a}),P=[N(`reset`,o+1,[k(`counts`,u),k(`invalidNetwork`,v),k(`neighbors`,f),k(`weights`,_,`f32`),k(`state`,w)],`counts[countsOffset + index] = 0u;
if (index == 0u) {
  invalidNetwork[invalidNetworkOffset] = 0u; state[stateOffset + 4u] = 0u;
  // The padded slot is outside offsets when there are no turns.
  neighbors[neighborsOffset] = NONE; weights[weightsOffset] = -1.0;
}`),N(`validate-count`,Math.max(o,s),[O(`turns`,r.turns),O(`costs`,r.costs,`f32`),...j,k(`counts`,u,`atomic<u32>`),k(`invalidNetwork`,v,`atomic<u32>`)],`if (index < TURN_COUNT) {
  let penalty = ${M};
  if (!isFiniteValue(penalty) || penalty < 0.0) { atomicStore(&invalidNetwork[invalidNetworkOffset], 1u); }
  let incoming = turns[turnsOffset + 2u * index]; let outgoing = turns[turnsOffset + 2u * index + 1u];
  if (incoming >= EDGE_COUNT || outgoing >= EDGE_COUNT) { atomicStore(&invalidNetwork[invalidNetworkOffset], 1u); }
  else if (isFiniteValue(costs[costsOffset + incoming]) && costs[costsOffset + incoming] >= 0.0 && isFiniteValue(costs[costsOffset + outgoing]) && costs[costsOffset + outgoing] >= 0.0) { atomicAdd(&counts[countsOffset + incoming], 1u); }
}`),...new g({id:`${n}-scan`,input:u,output:d,mode:`exclusive`}).getCommandNodes(e),N(`reset-cursors`,o+1,[k(`counts`,u)],`counts[countsOffset + index] = 0u;`),N(`scatter`,s,[O(`turns`,r.turns),O(`costs`,r.costs,`f32`),...j,O(`offsets`,d),k(`counts`,u,`atomic<u32>`),k(`neighbors`,f),k(`weights`,_,`f32`)],`let incoming = turns[turnsOffset + 2u * index]; let outgoing = turns[turnsOffset + 2u * index + 1u];
if (incoming >= EDGE_COUNT || outgoing >= EDGE_COUNT) { return; }
if (!isFiniteValue(costs[costsOffset + incoming]) || costs[costsOffset + incoming] < 0.0 || !isFiniteValue(costs[costsOffset + outgoing]) || costs[costsOffset + outgoing] < 0.0) { return; }
let slot = offsets[offsetsOffset + incoming] + atomicAdd(&counts[countsOffset + incoming], 1u);
neighbors[neighborsOffset + slot] = outgoing;
weights[weightsOffset + slot] = ${M} + costs[costsOffset + outgoing];`)];for(let t=0;t<r.queries.length;t++){let c=`query-${t}`;P.push(N(`${c}-seed`,1,[O(`queries`,r.queries),O(`fractions`,r.fractions,`f32`),O(`costs`,r.costs,`f32`),O(`invalidNetwork`,v),k(`state`,w),k(`startNodes`,y),k(`startCosts`,b,`f32`)],`let query = state[stateOffset + 4u];
let source = queries[queriesOffset + 2u * query]; let destination = queries[queriesOffset + 2u * query + 1u];
let sourceFraction = fractions[fractionsOffset + 2u * query]; let targetFraction = fractions[fractionsOffset + 2u * query + 1u];
state[stateOffset] = 1u; state[stateOffset + 1u] = INFINITY_BITS; state[stateOffset + 2u] = NONE;
state[stateOffset + 3u] = ${i}u;
startNodes[startNodesOffset] = NONE; startCosts[startCostsOffset] = 0.0;
if (invalidNetwork[invalidNetworkOffset] != 0u || source >= EDGE_COUNT || destination >= EDGE_COUNT || !isFiniteValue(sourceFraction) || !isFiniteValue(targetFraction) || sourceFraction < 0.0 || sourceFraction > 1.0 || targetFraction < 0.0 || targetFraction > 1.0) { return; }
if (!isFiniteValue(costs[costsOffset + source]) || costs[costsOffset + source] < 0.0 || !isFiniteValue(costs[costsOffset + destination]) || costs[costsOffset + destination] < 0.0) { return; }
state[stateOffset] = 0u;
startNodes[startNodesOffset] = source;
startCosts[startCostsOffset] = (1.0 - sourceFraction) * costs[costsOffset + source];
if (source == destination && targetFraction >= sourceFraction) {
  state[stateOffset + 1u] = bitcast<u32>(max(0.0, (targetFraction - sourceFraction) * costs[costsOffset + source]));
}`)),P.push(...new p({id:`${n}-${c}-search`,topology:{offsets:d,neighbors:f,weights:_},startNodes:y,startCosts:b,maximumIterations:Math.max(i,1),maximumTieIterations:Math.max(i,1),localIterations:64,activeIterations:D,output:{costs:x,predecessors:S,status:{converged:C}}}).getCommandNodes(e)),P.push(N(`${c}-paths-initialize`,o,[O(`queries`,r.queries),O(`state`,w),O(`predecessors`,S),k(`paths`,T)],`let query = state[stateOffset + 4u];
let source = queries[queriesOffset + 2u * query]; let destination = queries[queriesOffset + 2u * query + 1u];
var ancestor = predecessors[predecessorsOffset + index];
var flags = select(0u, 2u, index == destination && source != destination);
if (index == source && state[stateOffset] == 0u) { ancestor = source; flags = 1u; }
paths[pathsOffset + 2u * index] = ancestor;
paths[pathsOffset + 2u * index + 1u] = flags;`));for(let e=0;e<a;e++)P.push(N(`${c}-paths-jump-${e}`,o,[O(`paths`,T),k(`nextPaths`,E)],`let ancestor = paths[pathsOffset + 2u * index];
var nextAncestor = ancestor; var flags = paths[pathsOffset + 2u * index + 1u];
if (ancestor < EDGE_COUNT) {
  nextAncestor = paths[pathsOffset + 2u * ancestor];
  flags |= paths[pathsOffset + 2u * ancestor + 1u];
}
nextPaths[nextPathsOffset + 2u * index] = nextAncestor;
nextPaths[nextPathsOffset + 2u * index + 1u] = flags;`)),[T,E]=[E,T];P.push(N(`${c}-partial-costs`,s,[O(`queries`,r.queries),O(`fractions`,r.fractions,`f32`),O(`turns`,r.turns),...j,O(`costs`,r.costs,`f32`),O(`distances`,x,`f32`),k(`state`,w,`atomic<u32>`),O(`paths`,T)],`let query = atomicLoad(&state[stateOffset + 4u]);
if (${i}u == 0u || atomicLoad(&state[stateOffset]) != 0u) { return; }
let incoming = turns[turnsOffset + 2u * index]; let outgoing = turns[turnsOffset + 2u * index + 1u];
if (incoming >= EDGE_COUNT || outgoing >= EDGE_COUNT) { return; }
let distance = distances[distancesOffset + incoming];
if (!isFiniteValue(distance)) { return; }
if (!isFiniteValue(costs[costsOffset + outgoing]) || costs[costsOffset + outgoing] < 0.0) { return; }
// Retain the reference route's invalid-data status for a reachable full-edge f32 overflow.
if (!isFiniteValue(distance + (${M} + costs[costsOffset + outgoing]))) { atomicStore(&state[stateOffset], 1u); return; }
let flags = paths[pathsOffset + 2u * incoming + 1u];
if (flags != 1u) { return; }
let destination = queries[queriesOffset + 2u * query + 1u];
if (outgoing == destination) {
  let targetFraction = fractions[fractionsOffset + 2u * query + 1u];
  let candidate = distance + (${M} + targetFraction * costs[costsOffset + destination]);
  atomicMin(&state[stateOffset + 1u], bitcast<u32>(max(0.0, candidate)));
}`)),P.push(N(`${c}-target-parent`,s,[O(`queries`,r.queries),O(`fractions`,r.fractions,`f32`),O(`turns`,r.turns),...j,O(`costs`,r.costs,`f32`),O(`distances`,x,`f32`),k(`state`,w,`atomic<u32>`),O(`paths`,T)],`let query = atomicLoad(&state[stateOffset + 4u]);
if (${i}u == 0u || atomicLoad(&state[stateOffset]) != 0u) { return; }
let source = queries[queriesOffset + 2u * query]; let destination = queries[queriesOffset + 2u * query + 1u];
let incoming = turns[turnsOffset + 2u * index]; let outgoing = turns[turnsOffset + 2u * index + 1u];
if (incoming >= EDGE_COUNT || outgoing != destination) { return; }
if (paths[pathsOffset + 2u * incoming + 1u] != 1u) { return; }
let best = bitcast<f32>(atomicLoad(&state[stateOffset + 1u]));
if (!isFiniteValue(best)) { return; }
let sourceFraction = fractions[fractionsOffset + 2u * query]; let targetFraction = fractions[fractionsOffset + 2u * query + 1u];
// Direct forward travel wins ties, just as in the reference kernel.
if (source == destination && targetFraction >= sourceFraction && (targetFraction - sourceFraction) * costs[costsOffset + source] <= best) { return; }
let candidate = distances[distancesOffset + incoming] + (${M} + targetFraction * costs[costsOffset + destination]);
if (candidate == best) { atomicMin(&state[stateOffset + 2u], incoming); }`)),P.push(N(`${c}-copy-parents`,o,[O(`queries`,r.queries),O(`state`,w),O(`predecessors`,S),O(`paths`,T),k(`outputPredecessors`,r.output.predecessorEdges)],`let query = state[stateOffset + 4u];
let destination = queries[queriesOffset + 2u * query + 1u];
var parent = predecessors[predecessorsOffset + index];
if ((paths[pathsOffset + 2u * index + 1u] & 1u) == 0u) { parent = NONE; }
// The terminal edge is partial: its minimizing parent can differ from the full-edge parent
// after f32 absorption, or the full edge can still be unreached when rounds are exhausted.
if (index == destination) { parent = state[stateOffset + 2u]; }
if (state[stateOffset] != 0u || ${i}u == 0u) { parent = NONE; }
outputPredecessors[outputPredecessorsOffset + query * EDGE_COUNT + index] = parent;`)),P.push(N(`${c}-publish`,1,[k(`state`,w),O(`converged`,C),k(`outputCosts`,r.output.costs,`f32`),k(`status`,r.output.status)],`let query = state[stateOffset + 4u];
let invalid = state[stateOffset] != 0u;
let complete = converged[convergedOffset] != 0u && ${i}u > 0u;
status[statusOffset + query] = select(select(2u, 0u, complete), 1u, invalid);
outputCosts[outputCostsOffset + query] = bitcast<f32>(select(state[stateOffset + 1u], INFINITY_BITS, invalid));
state[stateOffset + 4u] = query + 1u;`))}return P}var D=128,O=4096,k=65536,A=4294967294,j=4294967295,M=1024,N=64,P=class{id;props;mode;maximumRelaxations;pathRoundCount;estimatedCommandNodeCount;constructor(e){this.id=e.id??`network-point-routing`,this.props=e,this.mode=e.mode??`scalable`;let{id:t,mode:n}=this;a(t,n===`scalable`||n===`reference`,`mode must be scalable or reference`),this.pathRoundCount=Math.ceil(Math.log2(Math.max(e.costs.length,2)));let c=n===`reference`;this.maximumRelaxations=e.maximumRelaxations??(c?e.costs.length:N),o(t,`costs rows`,e.costs.length,1,c?D:A),o(t,`turns rows`,e.turns.length,0,c?O:j),o(t,`maximumRelaxations`,this.maximumRelaxations,0,c?k:M),o(t,`queries rows`,e.queries.length,0),o(t,`queries rows * costs rows`,e.queries.length*e.costs.length,0),r(e.turns,[`uint32x2`],`${t} turns`),e.turnPenalties&&(r(e.turnPenalties,[`float32`],`${t} turnPenalties`),a(t,e.turnPenalties.length===e.turns.length,`turnPenalties must hold one row per turn`)),r(e.queries,[`uint32x2`],`${t} queries`),r(e.fractions,[`float32x2`],`${t} fractions`);for(let[n,i]of[[`costs`,e.costs],[`output.costs`,e.output.costs]])r(i,[`float32`],`${t} ${n}`);for(let[n,i]of[[`output.status`,e.output.status],[`output.predecessorEdges`,e.output.predecessorEdges]])r(i,[`uint32`],`${t} ${n}`);a(t,e.fractions.length===e.queries.length,`fractions must hold one row per query`),a(t,e.output.costs.length===e.queries.length,`output.costs must hold one row per query`),a(t,e.output.status.length===e.queries.length,`output.status must hold one row per query`),a(t,e.output.predecessorEdges.length===e.queries.length*e.costs.length,`output.predecessorEdges must hold costs rows per query`),u(t,Object.values(e.output),[e.turns,e.turnPenalties,e.costs,e.queries,e.fractions]),d(t,e.output);let l=e=>e.buffer.defaultBuffer?i(e.buffer.defaultBuffer):e.buffer,f=Object.values(e.output).map(l),p=new Set([e.turns,e.turnPenalties,e.costs,e.queries,e.fractions].filter(e=>e!==void 0).map(l));a(t,f.every(e=>!p.has(e)),`output views must use separate physical buffers from inputs`),this.estimatedCommandNodeCount=e.queries.length===0?0:c?1:11+e.queries.length*(2*Math.max(this.maximumRelaxations,1)+14+this.pathRoundCount),s(t,this.estimatedCommandNodeCount)}getCommandNodes(t){let{id:r,props:i}=this;l(r,t,[i.turns,i.turnPenalties,i.costs,i.queries,i.fractions,...Object.values(i.output)]);for(let[e,a]of Object.entries({turns:i.turns,...i.turnPenalties?{turnPenalties:i.turnPenalties}:{},edgeCosts:i.costs,queries:i.queries,fractions:i.fractions,...i.output}))c(t.device,r,e,n(a).size,`reduce costs rows, turns rows or queries rows`);if(e(t,r,{turns:i.turns,turnPenalties:i.turnPenalties,edgeCosts:i.costs,queries:i.queries,fractions:i.fractions,...i.output}),this.mode===`scalable`){let e=f(Math.max(this.maximumRelaxations,1));for(let[n,a]of[[`frontier queues`,i.costs.length*8],[`path certification`,i.costs.length*8],[`CSR offsets`,(i.costs.length+1)*4],[`CSR turn slots`,Math.max(i.turns.length,1)*4],[`tie bits`,Math.max(Math.ceil(Math.max(i.turns.length,1)/32),1)*4],[`query state`,20],[`frontier control`,2*e.controlWordCount*4],[`frontier dispatch`,2*e.dispatchSlotCount*64*4]])c(t.device,r,n,a,`reduce costs rows or turns rows`)}return i.queries.length===0?[]:this.mode===`reference`?T(t,r,i):E(t,r,i,this.maximumRelaxations,this.pathRoundCount)}},F=_({parameters:w,caption:`Cost to reach each directed road from a point on the source road, turns included.`,createInputs:({random:e,world:t})=>{let n=S(e,t,35),r=n.slotCount,i=[];for(let e=0;e<r;e++){let t=n.neighbors[e];for(let r=n.offsets[t];r<n.offsets[t+1];r++)r!==n.reverseSlots[e]&&i.push(e,r)}let a=new Uint32Array(r*4),o=n.offsets[n.findNearestNode(.3,.5)];for(let e=0;e<r*2;e++)a[e*2]=o,a[e*2+1]=e%r;return{network:n,defaultRoad:o,sourceRoad:o,slotSegments:v(n),turns:Uint32Array.from(i),queries:a,edgeCosts:new Float32Array(r)}},build(e,t){let{network:n,queries:r,slotSegments:i}=t,a=n.slotCount*2,o=e.allocate(`edge-costs`,`float32`,n.slotCount),s=e.allocate(`fractions`,`float32x2`,a),c=e.allocate(`query-costs`,`float32`,a),l=e.upload(`queries`,`uint32x2`,r),u=e.allocate(`source-position`,`float32x2`,1),d=e.allocate(`status`,`uint32`,a),f=e.allocate(`predecessors`,`uint32`,a*n.slotCount);return e.graph.add(new P({turns:e.upload(`turns`,`uint32x2`,t.turns).view,costs:o.view,queries:l.view,fractions:s.view,output:{costs:c.view,status:d.view,predecessorEdges:f.view}})),{outputs:{slots:C(e,n,.012),queryCosts:c,sourcePosition:u},update:e=>{for(let e=0;e<a;e++)r[e*2]=t.sourceRoad;l.write(r);let c=t.sourceRoad*4;u.write(Float32Array.of(i[c]+(i[c+2]-i[c])*e.sourceFraction,i[c+1]+(i[c+3]-i[c+1])*e.sourceFraction)),n.fillWeights(t.edgeCosts,e.congestion),o.write(t.edgeCosts),s.write(Float32Array.from({length:a*2},(t,n)=>n%2==0?e.sourceFraction:e.targetFraction))}}},interaction:{kind:`place`,hint:`place the source`,rest:({inputs:e,toUnit:t})=>{let n=e.defaultRoad*4,r=e.slotSegments,i=r[n+2]-r[n],a=r[n+3]-r[n+1],o=Math.hypot(i,a)||1;return t([r[n]+i*w[0].default+a/o*.004,r[n+1]+a*w[0].default-i/o*.004])},resolve(e,{inputs:t,outputs:n,peek:r}){let i=y(t.slotSegments,e.x,e.y),a=i.slot!==t.sourceRoad;t.sourceRoad=i.slot;let o=Math.round(i.fraction*20)/20,s=r(n.queryCosts),c=0;if(s)for(let e of s)Number.isFinite(e)&&e<0x56bc75e2d63100000&&(c=Math.max(c,e));return{values:{sourceFraction:o},rerun:a,glyphs:[{shape:`crosshair`,at:i.point,size:12}],readout:`source road ${i.slot} at ${o.toFixed(2)} · max cost ${c.toFixed(2)}`}}},draw:({outputs:e,theme:t})=>[x(e.slots,e.queryCosts,{radius:3,range:[0,3],bands:12}),...b(e.sourcePosition,t,{colorIndex:4})]});export{F as default};