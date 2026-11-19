import{n as e}from"./raster-storage-preflight-CIxFc-0g.js";import{S as t,T as n,f as r,s as i,v as a}from"./gpu-contributor-utils-BIwLQON5.js";import{t as o}from"./status-aOwAAXek.js";import{r as s,s as c}from"./kernel-C3tfQOn4.js";import{n as l,r as u}from"./raster-band-utils-BcTxc8_Y.js";import{a as d,o as f}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as p,i as m,r as h}from"./raster-expansion-utils-AZOLseCG.js";import{t as g}from"./define-run-B-brkCe4.js";import{i as _}from"./inputs-CJKKIRKK.js";import{bi as v}from"./entry-preview-D70TdSbK.js";import{a as y,c as b,n as x,r as S,t as C}from"./gestures-C1AoqRjN.js";import{r as w}from"./labels-DBumpLdz.js";import{i as T,n as E,t as D}from"./bands-CVDIl0aE.js";function O(e,t,n,r){let i=n.grid.width*n.grid.height,o=n.maximumIterations??Math.ceil(Math.log2(i))+1,l=s(e,`${t}-forest-components`,`uint32`,i),u=s(e,`${t}-forest-costs`,`uint32`,i),p=s(e,`${t}-forest-edges`,`uint32`,i),m=s(e,`${t}-forest-parent-first`,`uint32`,i),h=s(e,`${t}-forest-parent-second`,`uint32`,i),g=s(e,`${t}-forest-active`,`uint32`,1),_=`GPURasterWatershedSegmentation`,v=d(e,{id:`${t}-forest-dispatch`,operation:_,shapes:[[i]]}),y=(e,t,n=`read`,r=`u32`)=>({name:e,view:t,access:n,type:r});for(let i of[l,u,p,m,h,g,v.view,r.storage.values,r.validity,n.markers,n.output.labels,n.output.status.converged])a(e.device,t,`binding`,c(i).size,`split the raster`);let b=`const WIDTH: u32 = ${n.grid.width}u;
const HEIGHT: u32 = ${n.grid.height}u;
const NONE: u32 = 0xffffffffu;
// Four forward directions: right, down-left, down, down-right.
fn edgeTarget(source: u32, direction: u32) -> u32 {
  let column = source % WIDTH; let row = source / WIDTH;
  if (direction == 0u) { return select(NONE, source + 1u, column + 1u < WIDTH); }
  if (row + 1u >= HEIGHT) { return NONE; }
  if (direction == 1u) { return select(NONE, source + WIDTH - 1u, column > 0u); }
  if (direction == 2u) { return source + WIDTH; }
  return select(NONE, source + WIDTH + 1u, column + 1u < WIDTH);
}
fn costKey(value: f32) -> u32 {
  // Signed zero is one elevation; negative finite floats must sort before nonnegative floats.
  let bits = bitcast<u32>(select(value, 0.0, value == 0.0));
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}`,x=[y(`elevation`,r.storage.values,`read`,`f32`),y(`validity`,r.validity),y(`markers`,n.markers),y(`component`,l)],S=e=>`
if (validity[validityOffset + index] == 0u || markers[markersOffset + index] >= 0xfffffffeu) { return; }
let first = component[componentOffset + index];
for (var direction = 0u; direction < 4u; direction++) {
  ${n.connectivity===8?``:`if (direction == 1u || direction == 3u) { continue; }`}
  let neighbor = edgeTarget(index, direction);
  if (neighbor == NONE) { continue; }
  if (validity[validityOffset + neighbor] == 0u || markers[markersOffset + neighbor] >= 0xfffffffeu) { continue; }
  let second = component[componentOffset + neighbor];
  if (first == second) { continue; }
  let edge = index * 4u + direction;
  let cost = costKey(max(elevation[elevationOffset + index], elevation[elevationOffset + neighbor]));
  ${e}
}`,C=[f(e,{id:`${t}-forest-initialize`,operation:_,variant:`forest-initialize`,invocationCount:i,bindings:[y(`component`,l,`read_write`),y(`bestCost`,u,`read_write`),y(`bestEdge`,p,`read_write`),y(`roundState`,g,`read_write`),y(`converged`,n.output.status.converged,`read_write`)],body:`component[componentOffset + index] = index;
bestCost[bestCostOffset + index] = 0xffffffffu;
bestEdge[bestEdgeOffset + index] = 0xffffffffu;
if (index == 0u) { roundState[roundStateOffset] = 1u; converged[convergedOffset] = 1u; }`})];for(let n=0;n<o;n++){let r=`${t}-forest-round-${n}`;C.push(f(e,{id:`${r}-gate`,operation:_,variant:`forest-gate`,invocationCount:1,bindings:[y(`roundState`,g,`read_write`),v.binding],body:`if (roundState[roundStateOffset] != 0u) { ${v.armWGSL} } else { ${v.disarmWGSL} }
roundState[roundStateOffset] = 0u;`}),f(e,{id:`${r}-cost`,operation:_,variant:`forest-cost`,invocationCount:i,...v.getGate(0),declarations:b,bindings:[...x,y(`bestCost`,u,`read_write`,`atomic<u32>`),y(`roundState`,g,`read_write`,`atomic<u32>`)],body:S(`if (markers[markersOffset + first] == 0u) {
  atomicMin(&bestCost[bestCostOffset + first], cost); atomicStore(&roundState[roundStateOffset], 1u);
}
if (markers[markersOffset + second] == 0u) {
  atomicMin(&bestCost[bestCostOffset + second], cost); atomicStore(&roundState[roundStateOffset], 1u);
}`)}),f(e,{id:`${r}-edge`,operation:_,variant:`forest-edge`,invocationCount:i,...v.getGate(0),declarations:b,bindings:[...x,y(`bestCost`,u),y(`bestEdge`,p,`read_write`,`atomic<u32>`)],body:S(`if (markers[markersOffset + first] == 0u && bestCost[bestCostOffset + first] == cost) {
  atomicMin(&bestEdge[bestEdgeOffset + first], edge);
}
if (markers[markersOffset + second] == 0u && bestCost[bestCostOffset + second] == cost) {
  atomicMin(&bestEdge[bestEdgeOffset + second], edge);
}`)}),f(e,{id:`${r}-hook`,operation:_,variant:`forest-hook`,invocationCount:i,...v.getGate(0),declarations:b,bindings:[y(`component`,l),y(`bestEdge`,p),y(`parent`,m,`read_write`)],body:`var parentIndex = index;
let edge = bestEdge[bestEdgeOffset + index];
if (component[componentOffset + index] == index && edge != NONE) {
  let source = edge / 4u;
  let first = component[componentOffset + source];
  let second = component[componentOffset + edgeTarget(source, edge % 4u)];
  parentIndex = select(first, second, first == index);
  if (bestEdge[bestEdgeOffset + parentIndex] == edge && index < parentIndex) { parentIndex = index; }
}
parent[parentOffset + index] = parentIndex;`}));let a=m,o=h;for(let t=0;t<Math.ceil(Math.log2(i));t++)C.push(f(e,{id:`${r}-jump-${t}`,operation:_,variant:`forest-jump`,invocationCount:i,...v.getGate(0),bindings:[y(`source`,a),y(`destination`,o,`read_write`)],body:`destination[destinationOffset + index] = source[sourceOffset + source[sourceOffset + index]];`})),[a,o]=[o,a];C.push(f(e,{id:`${r}-relabel`,operation:_,variant:`forest-relabel`,invocationCount:i,...v.getGate(0),bindings:[y(`parent`,a),y(`component`,l,`read_write`),y(`bestCost`,u,`read_write`),y(`bestEdge`,p,`read_write`)],body:`component[componentOffset + index] = parent[parentOffset + component[componentOffset + index]];
bestCost[bestCostOffset + index] = 0xffffffffu;
bestEdge[bestEdgeOffset + index] = 0xffffffffu;`}))}return C.push(f(e,{id:`${t}-forest-validate`,operation:_,variant:`forest-validate`,invocationCount:i,declarations:b,bindings:[...x,y(`converged`,n.output.status.converged,`read_write`,`atomic<u32>`)],body:S(`if (markers[markersOffset + first] == 0u || markers[markersOffset + second] == 0u) {
  atomicStore(&converged[convergedOffset], 0u);
}`)}),f(e,{id:`${t}-forest-publish`,operation:_,variant:`forest-publish`,invocationCount:i,bindings:[y(`component`,l),y(`validity`,r.validity),y(`markers`,n.markers),y(`converged`,n.output.status.converged),y(`labels`,n.output.labels,`read_write`)],body:`let marker = markers[markersOffset + index];
var label = 0u;
if (validity[validityOffset + index] != 0u && marker < 0xfffffffeu) {
  label = markers[markersOffset + component[componentOffset + index]];
  if (converged[convergedOffset] == 0u && marker == 0u) { label = 0xfffffffeu; }
}
labels[labelsOffset + index] = label;`})),C}var k=16384,A=1073741823,j=32,M=class{id;props;constructor(e){this.id=e.id??`raster-watershed-segmentation`,this.props=e;let a=p(this.id,e.grid,[...l(e.input),e.markers],i(e.output));if(n(this.id,e.method===void 0||[`fifo`,`minimum-spanning-forest`].includes(e.method),`method must be 'fifo' or 'minimum-spanning-forest'`),n(this.id,e.connectivity===void 0||e.connectivity===4||e.connectivity===8,`connectivity must be 4 or 8`),e.method===`minimum-spanning-forest`){t(this.id,`grid cells`,a,A),n(this.id,e.maximumSteps===void 0,`maximumSteps applies to the fifo method only`);let i=e.maximumIterations??Math.ceil(Math.log2(a))+1;r(this.id,`maximumIterations`,i,1,j)}else t(this.id,`fifo grid cells`,a,k),n(this.id,e.maximumIterations===void 0,`maximumIterations applies to the minimum-spanning-forest method only`);h(e.markers,a,this.id,`markers`),h(e.output.labels,a,this.id,`output.labels`),h(e.output.status.converged,1,this.id,`output.status.converged`),h(e.output.status.iterationLimitReached,1,this.id,`output.status.iterationLimitReached`),o(this.id,e.output.status),n(this.id,Number.isInteger(e.maximumSteps??a)&&(e.maximumSteps??a)>0,`maximumSteps must be a positive integer`)}getCommandNodes(t){let n=this.props,r=n.grid.width*n.grid.height;e(t,this.id,[...l(n.input),n.markers,...i(n.output)],{"canonical band and forest planes":r*4,"forest gate":n.method===`minimum-spanning-forest`?12:0,"fifo queue":n.method!==`minimum-spanning-forest`&&Math.ceil(r/4)*16+Math.ceil(r*3/4)*16>t.device.limits.maxComputeWorkgroupStorageSize?r*12:0});let{id:a,props:o}=this,{grid:c,output:d}=o,p=c.width*c.height;m(a,t,[...l(o.input),o.markers,...i(d)]);let h=u(t,a,o.input,c,!0),g=this.getIterationLimitNodes(t);if(o.method===`minimum-spanning-forest`)return[...h.nodes,...O(t,a,o,h.band),...g];let _=Math.ceil(p/4)*16+Math.ceil(p*3/4)*16<=t.device.limits.maxComputeWorkgroupStorageSize,v=_?[]:[{name:`costs`,view:s(t,`${a}-costs`,`float32`,p),type:`f32`,access:`read_write`},{name:`queueState`,view:s(t,`${a}-queue`,`uint32`,p*3),type:`u32`,access:`read_write`}];return[...h.nodes,f(t,{id:a,operation:`GPURasterWatershedSegmentation`,invocationCount:1,bindings:[{name:`elevation`,view:h.band.storage.values,type:`f32`,access:`read`},{name:`validity`,view:h.band.validity,type:`u32`,access:`read`},{name:`markers`,view:o.markers,type:`u32`,access:`read`},{name:`labels`,view:d.labels,type:`u32`,access:`read_write`},...v,{name:`converged`,view:d.status.converged,type:`u32`,access:`read_write`}],declarations:`
${_?`var<workgroup> costs: array<f32, ${p}>;
var<workgroup> queueState: array<u32, ${p*3}>;
const costsOffset: u32 = 0u;
const queueStateOffset: u32 = 0u;`:``}
const AGE_START: u32 = ${p}u;
const HEAP_START: u32 = ${p*2}u;
var<private> pendingCount: u32;
fn heapCell(slot: u32) -> u32 { return queueState[queueStateOffset + HEAP_START + slot]; }
fn setHeapCell(slot: u32, cell: u32) {
  queueState[queueStateOffset + HEAP_START + slot] = cell;
}
fn precedes(first: u32, second: u32) -> bool {
  let firstCost = costs[costsOffset + first];
  let secondCost = costs[costsOffset + second];
  return firstCost < secondCost || (firstCost == secondCost && queueState[queueStateOffset + AGE_START + first] < queueState[queueStateOffset + AGE_START + second]);
}
fn insertCell(cell: u32) {
  // Extracted costs never decrease. The first discovery has cost max(parent cost, elevation),
  // so no later parent can strictly improve it. Each cell enters the queue only once.
  var slot = pendingCount;
  pendingCount++;
  loop {
    if (slot == 0u) { break; }
    let parent = (slot - 1u) / 2u;
    let parentCell = heapCell(parent);
    if (!precedes(cell, parentCell)) { break; }
    setHeapCell(slot, parentCell);
    slot = parent;
  }
  setHeapCell(slot, cell);
}
fn removeMinimum() -> u32 {
  let result = heapCell(0u);
  pendingCount--;
  if (pendingCount > 0u) {
    let replacement = heapCell(pendingCount);
    var slot = 0u;
    loop {
      var child = slot * 2u + 1u;
      if (child >= pendingCount) { break; }
      if (child + 1u < pendingCount && precedes(heapCell(child + 1u), heapCell(child))) { child++; }
      let childCell = heapCell(child);
      if (!precedes(childCell, replacement)) { break; }
      setHeapCell(slot, childCell);
      slot = child;
    }
    setHeapCell(slot, replacement);
  }
  return result;
}`,body:`
pendingCount = 0u;
var age = 0u;
for (var cell = 0u; cell < ${p}u; cell++) {
  let marker = markers[markersOffset + cell];
  let valid = validity[validityOffset + cell] != 0u;
  let seeded = valid && marker > 0u && marker < 0xfffffffeu;
  labels[labelsOffset + cell] = select(0u, marker, seeded);
  costs[costsOffset + cell] = select(bitcast<f32>(0x7f800000u | (index & 0u)), elevation[elevationOffset + cell], seeded);
  queueState[queueStateOffset + cell] = u32(!valid);
  queueState[queueStateOffset + AGE_START + cell] = select(0xffffffffu, age, seeded);
  if (seeded) { age++; insertCell(cell); }
}
var complete = false;
for (var step = 0u; step <= ${Math.min(o.maximumSteps??p,p)}u; step++) {
  if (pendingCount == 0u) { complete = true; break; }
  if (step == ${Math.min(o.maximumSteps??p,p)}u) { break; }
  let best = removeMinimum();
  let bestCost = costs[costsOffset + best];
  queueState[queueStateOffset + best] = 1u;
  for (var rowOffset = -1; rowOffset <= 1; rowOffset++) {
    for (var columnOffset = -1; columnOffset <= 1; columnOffset++) {
      if ((columnOffset == 0 && rowOffset == 0)${o.connectivity===8?``:` || abs(columnOffset) + abs(rowOffset) > 1`}) { continue; }
      let column = i32(best % ${c.width}u) + columnOffset;
      let row = i32(best / ${c.width}u) + rowOffset;
      if (column < 0 || row < 0 || column >= ${c.width} || row >= ${c.height}) { continue; }
      let neighbor = u32(row) * ${c.width}u + u32(column);
      if (queueState[queueStateOffset + neighbor] != 0u || markers[markersOffset + neighbor] != 0u) { continue; }
      let candidate = max(bestCost, elevation[elevationOffset + neighbor]);
      if (candidate < costs[costsOffset + neighbor]) {
        costs[costsOffset + neighbor] = candidate;
        labels[labelsOffset + neighbor] = labels[labelsOffset + best];
        queueState[queueStateOffset + AGE_START + neighbor] = age;
        age++;
        insertCell(neighbor);
      }
    }
  }
}
converged[convergedOffset] = u32(complete);
if (!complete) {
  for (var cell = 0u; cell < ${p}u; cell++) {
    if (validity[validityOffset + cell] != 0u && markers[markersOffset + cell] == 0u) { labels[labelsOffset + cell] = 0xfffffffeu; }
  }
}`}),...g]}getIterationLimitNodes(e){let{status:t}=this.props.output;return t.iterationLimitReached?[f(e,{id:`${this.id}-iteration-status`,operation:`GPURasterWatershedSegmentation`,variant:`iteration-status`,invocationCount:1,bindings:[{name:`converged`,view:t.converged,type:`u32`,access:`read`},{name:`iterationLimitReached`,view:t.iterationLimitReached,type:`u32`,access:`read_write`}],body:`iterationLimitReached[iterationLimitReachedOffset] = 1u - min(converged[convergedOffset], 1u);`})]:[]}};function N(e){let{columns:t,rows:n,values:r}=e,i=[];for(let e=1;e<n-1;e++)for(let n=1;n<t-1;n++){let a=e*t+n,o=!0;for(let i=-1;i<=1&&o;i++)for(let s=-1;s<=1;s++)if((i!==0||s!==0)&&r[(e+i)*t+n+s]<=r[a]){o=!1;break}o&&i.push(a)}return i.sort((e,t)=>r[e]-r[t])}var P=g({parameters:v,rebuildOn:[`method`,`connectivity`],caption:`A surface and basin markers in; the basin of every cell out.`,createInputs:({world:e})=>{let t=_(e,54);return{...t,minima:N(t),extra:[Math.floor(t.columns*.5),Math.floor(t.rows*.5)]}},build(e,t,n){let r=t.values.length,i=T(e,t,`surface`,t.values),a=e.allocate(`markers`,`uint32`,r),o=e.allocate(`labels`,`uint32`,r),s=e.allocate(`converged`,`uint32`,1);return e.graph.add(new M({grid:E(t),input:i.band,markers:a.view,connectivity:n.connectivity===1?8:4,method:n.method===1?`minimum-spanning-forest`:`fifo`,output:{labels:o.view,status:{converged:s.view}}})),{outputs:{surface:i.buffer,labels:o},update:e=>{let n=new Uint32Array(r);t.minima.slice(0,e.markerCount).forEach((e,t)=>{n[e]=t+1}),n[t.extra[1]*t.columns+t.extra[0]]=e.markerCount+1,a.write(n)}}},async afterRun({inputs:e,outputs:t,upload:n}){let r=b(await t.labels.buffer.readAsync());return{display:n(`display`,`uint32`,C(r,x(r,e.columns,e.rows)))}},interaction:{kind:`place`,hint:`place a marker`,rest:[.5,.5],resolve:(e,{inputs:t,outputs:n,values:r,world:i,peek:a})=>{let{column:o,row:s}=S(t,i,e);t.extra=[o,s];let c=a(n.labels),l=0;if(c)for(let e of c)e===r.markerCount+1&&l++;return{rerun:!0,glyphs:[{shape:`crosshair`,at:y(t,i,o,s),size:10}],readout:`marker at [${o}, ${s}] | basin ${l} cells`}}},draw:({inputs:e,outputs:t,extra:n,theme:r})=>[D(e,t.surface,{context:!0,range:[0,1]}),...n?[D(e,n.display,{ramp:`categorical`,categories:w(r),alpha:.75,hillshade:.4})]:[]]});export{P as default};