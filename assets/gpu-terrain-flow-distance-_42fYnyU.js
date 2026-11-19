import{r as e,t}from"./terrain-binding-preflight-laSEObQE.js";import{r as n}from"./kernel-C3tfQOn4.js";import{f as r,p as i}from"./gpu-contributor-utils-BIwLQON5.js";import{u as a}from"./raster-grid-VB0AM_GY.js";import{o}from"./wgsl-kernel-nodes-DagJklH2.js";import{i as s,r as c}from"./terrain-flow-tracing-BfyLX2kv.js";import{u as l}from"./wgsl-common-DfyXiYB0.js";import{d as u}from"./raster-band-utils-BcTxc8_Y.js";import{a as d,i as f,n as p,r as m,t as h}from"./raster-expansion-utils-AZOLseCG.js";import{o as g,s as _}from"./terrain-processing-utils-D-TSF_vl.js";var v=class{id;props;estimatedCommandNodeCount;constructor(e){this.id=e.id??`terrain-flow-distance`,this.props=e;let t=d(this.id,e.grid,[e.flowDirections,e.streams,e.transitionCosts],Object.values(e.output));for(let n of[e.flowDirections,e.streams,e.transitionCosts,...Object.values(e.output)])m(n,t);a(this.id,e.cellSizeMode),s(this.id,e.directionEncoding),u(this.id,e.grid,e.cellSizeMode),r(this.id,`maximumSteps`,e.maximumSteps??t),this.estimatedCommandNodeCount=2+p(_(Math.min(e.maximumSteps??t,t))),i(this.id,this.estimatedCommandNodeCount)}getCommandNodes(r){e(r,this.id,this.props,{"pointer state":16*this.props.grid.width*this.props.grid.height,"gate table":t(_(Math.min(this.props.maximumSteps??this.props.grid.width*this.props.grid.height,this.props.grid.width*this.props.grid.height)))});let{id:i,props:a}=this,{grid:s,output:u}=a,d=s.width*s.height;f(i,r,[a.flowDirections,a.streams,a.transitionCosts,...Object.values(u)]);let p=a.maximumSteps??d,m=_(Math.min(p,d)),v=[0,1].map(e=>n(r,`${i}-state-${e}`,`uint32`,d*4)),y=n(r,`${i}-converged`,`uint32`,1),b=`${c(s.width,s.height,a.directionEncoding)}
${l}
${g(s,a.cellSizeMode)}`,x=`GPUTerrainFlowDistance`;return[o(r,{id:`${i}-seed`,operation:x,invocationCount:d,bindings:[{name:`directions`,view:a.flowDirections,type:`u32`,access:`read`},...a.streams?[{name:`streams`,view:a.streams,type:`u32`,access:`read`}]:[],...a.transitionCosts?[{name:`costs`,view:a.transitionCosts,type:`f32`,access:`read`}]:[],{name:`state`,view:v[0],type:`u32`,access:`read_write`}],declarations:b,body:`let code = getFlowDirectionCode(directions[directionsOffset+index]);
let receiver = getFlowReceiver(index, code);
var kind = 0u; var cost = 0.0;
if (code != 0u && receiver == NO_CELL) { kind = 2u; } else {
 ${a.streams?`let stream = streams[streamsOffset+index]; if (stream == NO_CELL || stream == UNRESOLVED_DIRECTION) { kind = 2u; } else if (stream != 0u) { kind = 1u; }`:``}
 if (kind == 0u && receiver == NO_CELL) { kind = 1u; }
 if (kind == 0u) {
  cost = ${a.transitionCosts?`costs[costsOffset+index]`:`getGroundEdgeLength(index, receiver)`};
  if (!isFiniteValue(cost) || cost < 0.0) { kind = 2u; cost = 0.0; }
 }
}
let offset = stateOffset + index * 4u;
state[offset] = select(NO_CELL, receiver, kind == 0u);
state[offset + 1u] = bitcast<u32>(cost);
state[offset + 2u] = 1u;
state[offset + 3u] = select(UNRESOLVED_DIRECTION, index, kind == 1u);`}),...h(r,{id:`${i}-jump`,operation:x,invocationCount:d,roundCount:m,status:{converged:y},createRound:({nodeId:e,iteration:t,slotBinding:n,markChangedWGSL:i,condition:a,extraResources:s},c)=>o(r,{id:e,operation:x,variant:`jump`,invocationCount:d,bindings:[{name:`previous`,view:v[t%2],type:`u32`,access:`read`},{name:`next`,view:v[(t+1)%2],type:`u32`,access:`read_write`},{name:`status`,view:c,type:`atomic<u32>`,access:`read_write`},n],declarations:`const NO_CELL: u32 = 0xffffffffu;`,condition:a,extraResources:s,body:`let offset = previousOffset + index * 4u; let destination = nextOffset + index * 4u;
for (var component = 0u; component < 4u; component++) { next[destination + component] = previous[offset + component]; }
let parent = previous[offset];
if (parent != NO_CELL) {
 let ancestor = previousOffset + parent * 4u;
 next[destination] = previous[ancestor];
 next[destination + 1u] = bitcast<u32>(bitcast<f32>(previous[offset + 1u]) + bitcast<f32>(previous[ancestor + 1u]));
 next[destination + 2u] = min(${d+1}u, previous[offset + 2u] + previous[ancestor + 2u]);
 next[destination + 3u] = previous[ancestor + 3u];
 ${i}
}`})}),o(r,{id:`${i}-output`,operation:x,invocationCount:d,bindings:[{name:`directions`,view:a.flowDirections,type:`u32`,access:`read`},{name:`state`,view:v[m%2],type:`u32`,access:`read`},{name:`distance`,view:u.distance,type:`f32`,access:`read_write`},{name:`validity`,view:u.validity,type:`u32`,access:`read_write`},{name:`terminalCells`,view:u.terminalCells,type:`u32`,access:`read_write`}],declarations:b,body:`let offset = stateOffset + index * 4u;
let total = bitcast<f32>(state[offset + 1u]);
var terminal = state[offset + 3u];
let startsInvalid = directions[directionsOffset+index] == NO_CELL;
let valid = !startsInvalid && state[offset] == NO_CELL && terminal != UNRESOLVED_DIRECTION && state[offset + 2u] <= ${p}u && isFiniteValue(total);
terminal = select(select(UNRESOLVED_DIRECTION, NO_CELL, startsInvalid), terminal, valid);
distance[distanceOffset+index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), total, valid);
validity[validityOffset+index] = u32(valid); terminalCells[terminalCellsOffset+index] = terminal;`})]}};export{v as t};