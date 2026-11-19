import{t as e}from"./common-CshkX-d_.js";import{r as t}from"./kernel-C3tfQOn4.js";import{c as n,m as r,r as i,s as a}from"./raster-relaxation-Di9wjtxp.js";import{D as o,E as s,T as c,f as l,p as u,v as d,x as f}from"./gpu-contributor-utils-BIwLQON5.js";import{u as p}from"./raster-grid-VB0AM_GY.js";import{r as m}from"./raster-kernel-utils-DcIN7G5I.js";import{d as h,u as g}from"./raster-band-utils-BcTxc8_Y.js";import{o as _}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as v}from"./terrain-flow-tracing-BfyLX2kv.js";import{a as y,c as b,n as x,o as S,r as C,t as w,u as T}from"./terrain-processing-utils-D-TSF_vl.js";var E=256;function D(e,t=!1){return 6+Number(t)+e+Math.ceil(e/4)}function O(e,t,n,r){for(let[i,a]of[[`state scratch`,n*4],[`iteration status scratch`,16],[`gate slot scratch`,(Math.min(r,4)+1)*E]])d(e.device,t,i,a,`reduce grid dimensions or maximumIterations`)}var k=1024,A=new WeakMap,j=class{props;id;estimatedCommandNodeCount;constructor(t){this.props=t,this.id=t.id??`terrain-branching-flow`;let n=g(this.id,t.grid);p(this.id,t.cellSizeMode),h(this.id,t.grid,t.cellSizeMode);let i=t.mode!==`transport`&&t.mode!==`fractional-catchment`,a=``;if(i&&!t.edgeLengths)try{a=S(t.grid,t.cellSizeMode,!1)}catch{c(this.id,!1,`grid distance parameters must be representable as finite float32 values`)}A.set(this,a),o(this.id,`fractions`,t.fractions,n*8),o(this.id,`values`,t.values,n),t.edgeLengths&&(o(this.id,`edgeLengths`,t.edgeLengths,n*8),s(t.edgeLengths,[`float32`],this.id)),T(this.id,t.validity,n),o(this.id,`output.values`,t.output.values,n),o(this.id,`output.status`,t.output.status,n),o(this.id,`output.unresolvedCount`,t.output.unresolvedCount,1),r(this.id,`maximumIterations`,t.maximumIterations??e(t.grid.width+t.grid.height,128,1024)),l(this.id,`maximumWalkLength`,t.maximumWalkLength??32,1,k),this.estimatedCommandNodeCount=D(t.maximumIterations??e(t.grid.width+t.grid.height,128,1024),!!t.validity),u(this.id,this.estimatedCommandNodeCount),C(this.id,[t.fractions,t.values,...i&&t.edgeLengths?[t.edgeLengths]:[],...t.validity?[t.validity]:[]],Object.values(t.output),[{writes:[t.output.values,t.output.status]},{reads:[t.output.status],writes:[t.output.unresolvedCount]}])}getCommandNodes(e){return N(e,this).nodes}};function M(e,t){return N(e,new j(t))}function N(r,o){let{props:s,id:c}=o,l=s.values.length;f(c,r,[s.edgeLengths]),x(c,r,[s.fractions,s.values,...s.mode!==`transport`&&s.mode!==`fractional-catchment`&&s.edgeLengths?[s.edgeLengths]:[],...s.validity?[s.validity]:[],...Object.values(s.output)]),O(r,c,l,s.maximumIterations??e(s.grid.width+s.grid.height,128,1024));let u=y(r,{id:c,operation:`GPUTerrainBranchingFlow`,values:s.values,validity:s.validity}),d=t(r,`${c}-state`,`uint32`,l),p=n(r,c,`GPUTerrainBranchingFlow`,l),h=`${v(s.grid.width,s.grid.height)}\n${w}`,g=[...u.nodes,_(r,{id:`${c}-initialize`,operation:`GPUTerrainBranchingFlow`,invocationCount:l,bindings:[b(`fractions`,s.fractions),b(`source`,u.values),b(`state`,d,`read_write`)],declarations:h,body:`var valid = isFiniteValue(source[sourceOffset + index]) && source[sourceOffset + index] >= 0.0;
var total = 0.0;
for (var direction = 0u; direction < 8u; direction++) {
 let fraction = fractions[fractionsOffset + index * 8u + direction];
 valid = valid && isFiniteValue(fraction) && fraction >= 0.0;
 if (fraction > 0.0 && getFlowReceiver(index, 1u << direction) == NO_CELL) { valid = false; }
 total += fraction;
}
valid = valid && (total == 0.0 || abs(total - 1.0) <= 0.00001);
${s.mode===`fractional-catchment`?`valid = valid && source[sourceOffset + index] <= 1.0;`:``}
state[stateOffset + index] = select(0xfffffffeu, 0xffffffffu, valid);`})],S=s.mode===`transport`,C=!S&&s.mode!==`fractional-catchment`,T=C&&s.edgeLengths?[b(`lengths`,s.edgeLengths)]:[],E=`${h}
${A.get(o)}
const WAITING: u32 = 0xffffffffu;
const INVALID: u32 = 0xfffffffeu;
fn tryFinalize(cell: u32) -> bool {
 if (atomicLoad(&state[stateOffset + cell]) != WAITING) { return false; }
var value = ${S?`source[sourceOffset + cell]`:`0.0`};
var ready = true;
var hasReceiver = false;
var minimum = infinity();
for (var direction = 0u; direction < 8u; direction++) {
 let neighbor = getFlowReceiver(cell, 1u << direction);
 if (neighbor == NO_CELL) { continue; }
 let fraction = fractions[fractionsOffset + ${S?`neighbor * 8u + ((direction + 4u) % 8u)`:`cell * 8u + direction`}];
 if (!isFiniteValue(fraction) || fraction < 0.0) { ready = false; continue; }
 if (fraction == 0.0) { continue; }
 hasReceiver = true;
 let bits = atomicLoad(&state[stateOffset + neighbor]);
 ready = ready && bits != WAITING && bits != INVALID;
 let downstream = bitcast<f32>(bits);
 ${C?s.edgeLengths?`let edgeLength = lengths[lengthsOffset + cell * 8u + direction];`:`let edgeLength = getGroundEdgeLength(cell, neighbor);`:``}
 ${S||s.mode===`fractional-catchment`?`value += fraction * downstream;`:`ready = ready && isFiniteValue(edgeLength) && edgeLength >= 0.0;
 ${s.mode===`minimum-distance`?`minimum = min(minimum, downstream + edgeLength);`:s.mode===`maximum-distance`?`value = max(value, downstream + edgeLength);`:`value += fraction * (downstream + edgeLength);`}`}
}
${s.mode===`minimum-distance`?`value = select(0.0, minimum, hasReceiver);`:``}
${s.mode===`fractional-catchment`?`if (!hasReceiver) { value = source[sourceOffset + cell]; }`:``}
ready = ready && isFiniteValue(value);
if (!ready) { return false; }
 loop {
  let exchange = atomicCompareExchangeWeak(&state[stateOffset + cell], WAITING, bitcast<u32>(value));
  if (exchange.exchanged) { return true; }
  if (exchange.old_value != WAITING) { return false; }
 }
 return false;
}`;return g.push(a(r,{id:`${c}-reset`,operation:`GPUTerrainBranchingFlow`,state:p})),g.push(...i(r,{id:`${c}-solve`,operation:`GPUTerrainBranchingFlow`,state:p,maxIterations:s.maximumIterations??e(s.grid.width+s.grid.height,128,1024),createRound:({nodeId:e,slotBinding:t,markChangedWGSL:n,condition:i,extraResources:a})=>_(r,{id:e,operation:`GPUTerrainBranchingFlow`,variant:s.mode,invocationCount:l,bindings:[b(`fractions`,s.fractions),b(`source`,u.values),{name:`state`,view:d,type:`atomic<u32>`,access:`read_write`},...T,{name:`status`,view:p.status,type:`atomic<u32>`,access:`read_write`},t],declarations:E,condition:i,extraResources:a,body:`var current = index;
var changed = tryFinalize(current);
var published = changed;
for (var step = 0u; step < ${s.maximumWalkLength??32}u && changed; step++) {
 var successor = NO_CELL;
 for (var direction = 0u; direction < 8u; direction++) {
  let neighbor = getFlowReceiver(current, 1u << direction);
  if (neighbor == NO_CELL) { continue; }
  let fraction = fractions[fractionsOffset + ${S?`current * 8u + direction`:`neighbor * 8u + ((direction + 4u) % 8u)`}];
  if (fraction > 0.0 && tryFinalize(neighbor)) { published = true; if (successor == NO_CELL) { successor = neighbor; } }
 }
 current = successor; changed = successor != NO_CELL;
}
if (published) { ${n} }`})})),g.push(_(r,{id:`${c}-publish`,operation:`GPUTerrainBranchingFlow`,invocationCount:l,bindings:[b(`state`,d),b(`values`,s.output.values,`read_write`),b(`status`,s.output.status,`read_write`)],declarations:w,body:`let bits = state[stateOffset + index];
let statusValue = select(select(1u, 0u, bits == 0xfffffffeu), 2u, bits == 0xffffffffu);
status[statusOffset + index] = statusValue;
values[valuesOffset + index] = select(getNaN(), bitcast<f32>(bits), statusValue == 1u);`}),...m(r,{id:`${c}-status`,operation:`GPUTerrainBranchingFlow`,values:s.output.status,count:l,condition:`value != 1u`,output:s.output.unresolvedCount})),{nodes:g,iterationStatus:p.status}}export{O as i,M as n,D as r,j as t};