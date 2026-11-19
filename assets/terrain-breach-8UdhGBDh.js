import{r as e}from"./kernel-C3tfQOn4.js";import{r as t}from"./raster-kernel-utils-DcIN7G5I.js";import{u as n}from"./raster-band-utils-BcTxc8_Y.js";import{D as r,E as i,S as a,T as o,f as s,g as c,p as l,v as u,x as d}from"./gpu-contributor-utils-BIwLQON5.js";import{a as f,c as p,o as m}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as h}from"./terrain-flow-tracing-BfyLX2kv.js";import{a as g,c as _,n as v,r as y,t as b,u as x}from"./terrain-processing-utils-D-TSF_vl.js";import{t as S}from"./define-run-B-brkCe4.js";import{n as C,t as w}from"./pick-Bo3t3hjv.js";import{st as T}from"./entry-preview-D70TdSbK.js";import{c as E,d as D,i as O,u as k}from"./terrain-Bwc0vRX3.js";function A(t,{id:n,props:r,elevation:i,searchWidth:a,searchHeight:o}){let s=r.pitCells.length;if(s===0)return[];let c=a*o,l=r.maximumSteps+1,u=l*c,d=s*u,g=e(t,`${n}-search`,`uint32`,d*2),v=e(t,`${n}-winners`,`uint32`,s*2),y=e(t,`${n}-search-state`,`uint32`,2),x=f(t,{id:`${n}-search-dispatch`,operation:`GPUTerrainBreach`,shapes:[[s*c]]}),S=p(r.elevationDecrement),C=`${h(r.grid.width,r.grid.height)}
${b}
fn searchOrigin(pit: u32) -> vec2<u32> {
 let column = pit % GRID_WIDTH; let row = pit / GRID_WIDTH;
 return min(vec2<u32>(select(0u, column - ${r.maximumSteps}u, column >= ${r.maximumSteps}u), select(0u, row - ${r.maximumSteps}u, row >= ${r.maximumSteps}u)), vec2<u32>(${r.grid.width-a}u, ${r.grid.height-o}u));
}
fn searchCell(localCell: u32, origin: vec2<u32>) -> u32 {
 return (origin.y + localCell / ${a}u) * GRID_WIDTH + origin.x + localCell % ${a}u;
}
fn searchIndex(cell: u32, origin: vec2<u32>) -> u32 {
 let column = cell % GRID_WIDTH; let row = cell / GRID_WIDTH;
 if (column < origin.x || row < origin.y || column >= origin.x + ${a}u || row >= origin.y + ${o}u) { return NO_CELL; }
 return (row - origin.y) * ${a}u + column - origin.x;
}`,w=[],T=(e,r,i,a,o)=>w.push(m(t,{id:`${n}-${e}`,operation:`GPUTerrainBreach`,variant:e.replace(/-\d+$/,``),invocationCount:r,bindings:i,declarations:C,body:a,...o}));T(`search-initialize`,d,[_(`elevation`,i),_(`pits`,r.pitCells),_(`scratch`,g,`read_write`)],`let pitIndex = index / ${u}u;
let slot = index % ${u}u;
let pit = pits[pitsOffset + pitIndex];
var cost = 0x7f800000u;
if (pit < ${i.length}u && slot < ${c}u) {
 let base = elevation[elevationOffset + pit];
 if (isFiniteValue(base) && base - ${S} < base && searchCell(slot, searchOrigin(pit)) == pit) { cost = 0u; }
}
scratch[scratchOffset + index * 2u] = cost;
scratch[scratchOffset + index * 2u + 1u] = NO_CELL;`),T(`paths-initialize`,s*l,[_(`paths`,r.output.pathCells,`read_write`)],`paths[pathsOffset + index] = NO_CELL;`),T(`search-seed`,s,[_(`winners`,v,`read_write`),_(`state`,y,`read_write`),_(`lengths`,r.output.pathLengths,`read_write`),_(`status`,r.output.pitStatus,`read_write`)],`winners[winnersOffset + index * 2u] = 0x7f800000u;
winners[winnersOffset + index * 2u + 1u] = NO_CELL;
lengths[lengthsOffset + index] = 0u; status[statusOffset + index] = 0u;
if (index == 0u) { state[stateOffset] = 1u; state[stateOffset + 1u] = 0u; }`);for(let e=1;e<=r.maximumSteps;e++)T(`search-gate-${e}`,1,[_(`state`,y,`read_write`),x.binding],`if (state[stateOffset] != 0u) { ${x.armWGSL} } else { ${x.disarmWGSL} }
state[stateOffset] = 0u; state[stateOffset + 1u]++;`),T(`search-relax-${e}`,s*c,[_(`elevation`,i),_(`pits`,r.pitCells),_(`scratch`,g,`read_write`),{name:`state`,view:y,type:`atomic<u32>`,access:`read_write`}],`let pitIndex = index / ${c}u; let localCell = index % ${c}u;
let pit = pits[pitsOffset + pitIndex];
if (pit >= ${i.length}u) { return; }
let base = elevation[elevationOffset + pit];
if (!isFiniteValue(base) || !(base - ${S} < base)) { return; }
let step = atomicLoad(&state[stateOffset + 1u]);
let origin = searchOrigin(pit); let cell = searchCell(localCell, origin);
let original = elevation[elevationOffset + cell];
let cut = max(original - (base - f32(step) * ${S}), 0.0);
if (!isFiniteValue(original) || cut > ${p(r.maximumCutDepth)}) { return; }
let workOffset = scratchOffset + pitIndex * ${u*2}u;
var predecessor = NO_CELL; var cost = infinity();
for (var direction = 0u; direction < 8u; direction++) {
 let neighbor = getFlowReceiver(cell, 1u << direction); if (neighbor == NO_CELL) { continue; }
 let localNeighbor = searchIndex(neighbor, origin); if (localNeighbor == NO_CELL) { continue; }
 let candidate = bitcast<f32>(scratch[workOffset + ((step - 1u) * ${c}u + localNeighbor) * 2u]) + cut;
 if (candidate < cost || (candidate == cost && neighbor < predecessor)) { cost = candidate; predecessor = neighbor; }
}
// Costs are nonnegative; clear the sign bit to canonicalize a possible negative zero.
scratch[workOffset + (step * ${c}u + localCell) * 2u] = bitcast<u32>(cost) & 0x7fffffffu;
scratch[workOffset + (step * ${c}u + localCell) * 2u + 1u] = predecessor;
if (isFiniteValue(cost)) { atomicOr(&state[stateOffset], 1u); }`,x.getGate(0));let E=`let pitIndex = index / ${u}u;
let slot = index % ${u}u; let step = slot / ${c}u;
if (step == 0u) { return; }
let pit = pits[pitsOffset + pitIndex]; if (pit >= ${i.length}u) { return; }
let base = elevation[elevationOffset + pit];
if (!isFiniteValue(base) || !(base - ${S} < base)) { return; }
let original = elevation[elevationOffset + searchCell(slot % ${c}u, searchOrigin(pit))];
let cost = scratch[scratchOffset + index * 2u];
if (!isFiniteValue(original) || original > base - f32(step) * ${S} || cost >= 0x7f800000u) { return; }`,D=[_(`elevation`,i),_(`pits`,r.pitCells),_(`scratch`,g),{name:`winners`,view:v,type:`atomic<u32>`,access:`read_write`}];return T(`search-minimum-cost`,d,D,`${E}\natomicMin(&winners[winnersOffset + pitIndex * 2u], cost);`),T(`search-minimum-slot`,d,D,`${E}\nif (cost == atomicLoad(&winners[winnersOffset + pitIndex * 2u])) { atomicMin(&winners[winnersOffset + pitIndex * 2u + 1u], slot); }`),T(`search-trace`,s,[_(`elevation`,i),_(`pits`,r.pitCells),_(`scratch`,g),_(`winners`,v),_(`paths`,r.output.pathCells,`read_write`),_(`lengths`,r.output.pathLengths,`read_write`),_(`status`,r.output.pitStatus,`read_write`)],`let pit = pits[pitsOffset + index]; if (pit >= ${i.length}u) { return; }
let base = elevation[elevationOffset + pit];
if (!isFiniteValue(base) || !(base - ${S} < base)) { return; }
let slot = winners[winnersOffset + index * 2u + 1u];
if (slot == NO_CELL) { status[statusOffset + index] = 2u; return; }
let origin = searchOrigin(pit); let bestStep = slot / ${c}u;
var current = searchCell(slot % ${c}u, origin);
let pathOffset = pathsOffset + index * ${l}u;
let workOffset = scratchOffset + index * ${u*2}u;
// The only serial walk follows the selected path, bounded by maximumSteps and the node budget.
for (var step = bestStep; step > 0u; step--) {
 paths[pathOffset + step] = current;
 current = scratch[workOffset + (step * ${c}u + searchIndex(current, origin)) * 2u + 1u];
}
paths[pathOffset] = pit; lengths[lengthsOffset + index] = bestStep + 1u; status[statusOffset + index] = 1u;`),w}var j=4294967295,M=34028234663852886e22,N=class{props;id;estimatedCommandNodeCount;constructor(e){this.props=e,this.id=e.id??`terrain-breach`;let t=n(this.id,e.grid);r(this.id,`elevation`,e.elevation,t),e.burnDepth&&r(this.id,`burnDepth`,e.burnDepth,t),x(this.id,e.validity,t),s(this.id,`maximumSteps`,e.maximumSteps),c(this.id,`maximumCutDepth`,e.maximumCutDepth,0),o(this.id,Number.isFinite(e.elevationDecrement)&&Math.fround(e.elevationDecrement)<=M&&e.elevationDecrement>0&&Math.fround(e.elevationDecrement)>0,`elevationDecrement must be a positive finite number that is above zero in float32 and at most TERRAIN_BREACH_MAXIMUM_ELEVATION_DECREMENT (${M}) after float32 rounding`),r(this.id,`output.elevation`,e.output.elevation,t),r(this.id,`output.cutDepth`,e.output.cutDepth,t),r(this.id,`output.modificationMask`,e.output.modificationMask,t),r(this.id,`output.pathCells`,e.output.pathCells,e.pitCells.length*(e.maximumSteps+1)),r(this.id,`output.pathLengths`,e.output.pathLengths,e.pitCells.length),r(this.id,`output.pitStatus`,e.output.pitStatus,e.pitCells.length),r(this.id,`output.unresolvedCount`,e.output.unresolvedCount,1);let u=Math.min(e.grid.width,e.maximumSteps*2+1)*Math.min(e.grid.height,e.maximumSteps*2+1);a(this.id,`search scratch words (pitCells x (maximumSteps + 1) x search window cells x 2)`,e.pitCells.length*(e.maximumSteps+1)*u*2,j),this.estimatedCommandNodeCount=e.pitCells.length>0?15+2*e.maximumSteps:6,l(this.id,this.estimatedCommandNodeCount);let d=e.pitCells.length>0;for(let t of[e.pitCells,e.output.pathCells,e.output.pathLengths,e.output.pitStatus])i(t,[t.format],this.id);y(this.id,[e.elevation,...d?[e.pitCells]:[],...e.burnDepth?[e.burnDepth]:[],...e.validity?[e.validity]:[]],Object.entries(e.output).filter(([e])=>d||![`pathCells`,`pathLengths`,`pitStatus`].includes(e)).map(([,e])=>e),[{reads:[...e.validity?[]:[e.elevation],...e.burnDepth?[e.burnDepth]:[]],writes:[e.output.elevation,e.output.cutDepth,e.output.modificationMask]},{reads:d?[e.output.pitStatus]:[],writes:[e.output.unresolvedCount]},...d?[{reads:[...e.validity?[]:[e.elevation],e.pitCells],writes:[e.output.pathCells,e.output.pathLengths,e.output.pitStatus]},{reads:[e.output.elevation,e.output.modificationMask,e.output.pathCells,e.output.pathLengths],writes:[e.output.pitStatus]}]:[]])}getCommandNodes(n){let{id:r,props:i}=this,a=i.elevation.length,o=i.pitCells.length,s=Math.min(i.grid.width,i.maximumSteps*2+1),c=Math.min(i.grid.height,i.maximumSteps*2+1),l=s*c;d(r,n,[i.elevation,i.pitCells,i.validity,i.burnDepth,...Object.values(i.output)]),v(r,n,[i.elevation,...i.validity?[i.validity]:[],...o>0?[i.pitCells]:[],...i.burnDepth?[i.burnDepth]:[],...Object.entries(i.output).filter(([e])=>o>0||![`pathCells`,`pathLengths`,`pitStatus`].includes(e)).map(([,e])=>e)]);let f=Math.max(1,o*(i.maximumSteps+1)*l*2);u(n.device,r,`search scratch`,f*4,`reduce pitCells.length or maximumSteps`);let y=g(n,{id:r,operation:`GPUTerrainBreach`,values:i.elevation,validity:i.validity}),x=y.values,S=e(n,`${r}-edits`,`uint32`,a),C=p(i.elevationDecrement),w=`${h(i.grid.width,i.grid.height)}\n${b}`,T=[...y.nodes,...A(n,{id:r,props:i,elevation:x,searchWidth:s,searchHeight:c})];return T.push(m(n,{id:`${r}-edit-initialize`,operation:`GPUTerrainBreach`,invocationCount:a,bindings:[_(`source`,x),_(`edits`,S,`read_write`)],body:`let bits = bitcast<u32>(source[sourceOffset + index]); edits[editsOffset + index] = select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);`}),...o>0?[m(n,{id:`${r}-edit-proposals`,operation:`GPUTerrainBreach`,invocationCount:o*(i.maximumSteps+1),bindings:[_(`source`,x),_(`paths`,i.output.pathCells),_(`lengths`,i.output.pathLengths),{name:`edits`,view:S,type:`atomic<u32>`,access:`read_write`}],body:`let pitIndex = index / ${i.maximumSteps+1}u; let step = index % ${i.maximumSteps+1}u;
if (step == 0u || step >= lengths[lengthsOffset + pitIndex]) { return; }
let start = paths[pathsOffset + pitIndex * ${i.maximumSteps+1}u];
let cell = paths[pathsOffset + index];
let value = source[sourceOffset + start] - f32(step) * ${C};
let bits = bitcast<u32>(value);
atomicMin(&edits[editsOffset + cell], select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u));`})]:[],m(n,{id:`${r}-apply`,operation:`GPUTerrainBreach`,invocationCount:a,declarations:w,bindings:[_(`source`,x),_(`edits`,S),_(`elevation`,i.output.elevation,`read_write`),_(`depth`,i.output.cutDepth,`read_write`),_(`mask`,i.output.modificationMask,`read_write`),...i.burnDepth?[_(`burn`,i.burnDepth)]:[]],body:`let original = source[sourceOffset + index]; var edited = original; var flags = 0u;
if (isFiniteValue(original)) {
 let key = edits[editsOffset + index];
 edited = bitcast<f32>(select(~key, key ^ 0x80000000u, (key & 0x80000000u) != 0u));
 if (edited < original) { flags |= 1u; }
 ${i.burnDepth?`let burnDepth = burn[burnOffset + index]; if (!isFiniteValue(burnDepth) || burnDepth < 0.0) { edited = getNaN(); flags |= 4u; } else if (burnDepth > 0.0) { edited -= burnDepth; flags |= 2u; }`:``}
}
elevation[elevationOffset + index] = edited; depth[depthOffset + index] = original - edited; mask[maskOffset + index] = flags;`}),...o>0?[m(n,{id:`${r}-verify-paths`,operation:`GPUTerrainBreach`,invocationCount:o,declarations:b,bindings:[_(`elevation`,i.output.elevation),_(`mask`,i.output.modificationMask),_(`paths`,i.output.pathCells),_(`lengths`,i.output.pathLengths),_(`status`,i.output.pitStatus,`read_write`)],body:`if (status[statusOffset + index] != 1u) { return; }
let pathLength = lengths[lengthsOffset + index];
for (var step = 1u; step < pathLength; step++) {
 let previous = paths[pathsOffset + index * ${i.maximumSteps+1}u + step - 1u];
 let current = paths[pathsOffset + index * ${i.maximumSteps+1}u + step];
 let previousElevation = elevation[elevationOffset + previous];
 let currentElevation = elevation[elevationOffset + current];
 if (((mask[maskOffset + previous] | mask[maskOffset + current]) & 4u) != 0u || !isFiniteValue(previousElevation) || !isFiniteValue(currentElevation) || !(currentElevation < previousElevation)) { status[statusOffset + index] = 2u; }
}`})]:[],...t(n,{id:`${r}-status`,operation:`GPUTerrainBreach`,values:i.output.pitStatus,count:o,condition:`value != 1u`,output:i.output.unresolvedCount})),T}},P=40,F=4294967295,I=[[.6,.6,.65],[.35,.95,.55],[1,.35,.3]],L=S({parameters:T,caption:`Cut depth along the channel from each pit (dots) to its outlet.`,createInputs:e=>{let t=O(e),n=Math.max(1,Math.floor(t.pitCells.length/P)),r=t.pitCells.filter((e,t)=>t%n===0).slice(0,P),i=new Float32Array(r.length*2);return r.forEach((n,r)=>{i[r*2]=(n%t.columns+.5)*t.cellSize,i[r*2+1]=e.world.height-(Math.floor(n/t.columns)+.5)*t.cellSize}),{...t,world:e.world,pitCells:r,pitPositions:i,pitIndex:C(i),focus:{pit:-1}}},build(e,t,n){let r=t.pitCells.length,i=Math.round(n.maximumSteps),a=e.upload(`dem`,`float32`,t.elevation),o=e.upload(`pit-cells`,`uint32`,t.pitCells),s=e.allocate(`breached`,`float32`,t.cellCount),c=e.allocate(`cut-depth`,`float32`,t.cellCount),l=e.allocate(`path-cells`,`uint32`,r*(i+1)),u=e.allocate(`path-lengths`,`uint32`,r),d=e.allocate(`pit-status`,`uint32`,r);return e.graph.add(new N({grid:t.grid,elevation:a.view,pitCells:o.view,maximumSteps:i,maximumCutDepth:n.maximumCutDepth,elevationDecrement:1e-5,output:{elevation:s.view,cutDepth:c.view,modificationMask:e.allocate(`mask`,`uint32`,t.cellCount).view,pathCells:l.view,pathLengths:u.view,pitStatus:d.view,unresolvedCount:e.allocate(`unresolved-count`,`uint32`,1).view}})),{outputs:{elevation:a,cutDepth:c,pits:e.upload(`pits`,`float32x2`,t.pitPositions),pathCells:l,pathLengths:u,pitStatus:d}}},async afterRun({inputs:e,outputs:t,values:n,upload:r}){let[i,a]=await Promise.all([t.pathCells.buffer.readAsync(),t.pathLengths.buffer.readAsync()]),o=new Uint32Array(i.buffer,i.byteOffset,i.byteLength>>2),s=new Uint32Array(a.buffer,a.byteOffset,a.byteLength>>2),c=Math.round(n.maximumSteps)+1,l=[];for(let t=0;t<e.pitCells.length;t++){let n=E(e,e.world,Array.from(o.subarray(t*c,t*c+s[t])));for(let e=0;e+1<s[t];e++)l.push(n[e*2],n[e*2+1],n[e*2+2],n[e*2+3])}return{channels:r(`channels`,`float32x4`,l.length>0?Float32Array.from(l):new Float32Array(4).fill(NaN)),channelCount:l.length/4}},interaction:{kind:`probe`,hint:`hover a pit`,rest:({inputs:e,toUnit:t})=>t([e.pitPositions[0]??.5,e.pitPositions[1]??.5]),resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let o=w(t.pitIndex,e.x,e.y,t.focus.pit);if(t.focus.pit=o,o<0)return{};let s=[{shape:`ring`,at:[t.pitPositions[o*2],t.pitPositions[o*2+1]],radius:t.cellSize*2.5}],c=a(n.pitStatus),l=a(n.pathLengths),u=a(n.pathCells),d=a(n.cutDepth);if(!c||!l||!u||!d)return{glyphs:s,readout:`pit ${o+1}`};if(c[o]!==1)return{glyphs:s,readout:`pit ${o+1} · stays closed · maximumCutDepth ${r.maximumCutDepth}`};let f=Math.round(r.maximumSteps)+1,p=Array.from(u.subarray(o*f,o*f+l[o])).filter(e=>e!==F),m=0;for(let e of p)m=Math.max(m,d[e]);return s.push({shape:`polyline`,points:E(t,i,p)},{shape:`dot`,at:k(t,i,p[p.length-1])}),{glyphs:s,readout:`pit ${o+1} · breached · cut ${m.toFixed(3)} over ${p.length-1} cells`}}},draw:({inputs:e,outputs:t,values:n,extra:r})=>[D(e,t.elevation),{layer:`raster`,values:t.cutDepth,columns:e.columns,rows:e.rows,ramp:`sequential`,range:[0,n.maximumCutDepth],fade:!0},{layer:`lines`,segments:r.channels,count:r.channelCount,radius:2.4,color:[1,.95,.45,1]},{layer:`dots`,positions:t.pits,values:t.pitStatus,categories:I,radius:4}]});export{L as default};