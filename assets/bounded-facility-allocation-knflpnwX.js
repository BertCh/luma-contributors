import{r as e}from"./kernel-C3tfQOn4.js";import{o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{n}from"./facility-score-reduction-BRFyW8Cg.js";import{E as r,S as i,T as a,f as o,u as s,v as c,x as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{a as d}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as f}from"./define-run-B-brkCe4.js";import{i as p,l as m,n as h,s as g}from"./access-Co69Wfld.js";import{r as _}from"./display-Bu20g6o5.js";import{p as v,v as y}from"./roads-DzovamHj.js";import{Ua as b}from"./entry-preview-D70TdSbK.js";var x=32,S=8,C=65536,w=4096,T=4294967232,E=class{props;id;subsetCount;constructor(e){this.props=e,this.id=e.id??`bounded-facility-allocation`,a(this.id,[`p-center`,`capacitated-p-median`].includes(e.operation),`operation must be p-center or capacitated-p-median`),o(this.id,`facilityCount`,e.facilityCount,1,x),o(this.id,`selectCount`,e.selectCount,1,Math.min(S,e.facilityCount)),o(this.id,`demandCount`,e.demand.length,1,e.operation===`p-center`?Math.min(T,Math.floor(4294967295/e.facilityCount)):w),this.subsetCount=D(e.facilityCount,e.selectCount),o(this.id,`subsetCount`,this.subsetCount,1,C);for(let[t,n,i]of[[e.costs,`float32`,e.demand.length*e.facilityCount],[e.demand,`float32`,e.demand.length],[e.capacities,`float32`,e.facilityCount],[e.output.selectedFacilities,`uint32`,e.facilityCount],[e.output.assignments,`uint32`,e.demand.length],[e.output.objective,`float32`,1],[e.output.status,`uint32`,3]])if(r(t,[n],this.id),t.length!==i)throw Error(`${this.id} ${n} view length must equal ${i}`);let t=Object.values(e.output).filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(t).size!==t.length)throw Error(`${this.id} outputs must use distinct buffers`);u(this.id,Object.values(e.output),[e.costs,e.demand,e.capacities])}getCommandNodes(r){let{id:a,props:o,subsetCount:u}=this;i(a,`demandCount`,o.demand.length,s(r.device,4*o.facilityCount)),l(a,r,[o.costs,o.demand,o.capacities,...Object.values(o.output)]),d(r,a,{costs:o.costs,demand:o.demand,capacities:o.capacities,...Object.fromEntries(Object.entries(o.output).map(([e,t])=>[`output.${e}`,t]))}),c(r.device,a,`subset scores`,u*16,`reduce facilityCount or selectCount`);let f=e(r,`${a}-scores`,`uint32`,u*4),p=Array.from({length:297},(e,t)=>D(Math.floor(t/9),t%9)),m=`const FACILITY_COUNT: u32 = ${o.facilityCount}u; const SELECT_COUNT: u32 = ${o.selectCount}u; const DEMAND_COUNT: u32 = ${o.demand.length}u;
const CAPACITATED: bool = ${o.operation===`capacitated-p-median`};
const BINOMIALS: array<u32,297> = array<u32,297>(${p.map(e=>`${e}u`).join(`,`)});
var<private> selection: array<u32,8>;
var<private> remaining: array<f32,8>;
fn infiniteCost() -> f32 { var bits: u32 = 0x7f800000u; return bitcast<f32>(bits); }
fn initializeSubset(subset: u32) {
  var rank = subset; var first = 0u;
  for (var slot = 0u; slot < SELECT_COUNT; slot++) {
    for (var candidate = first; candidate < FACILITY_COUNT; candidate++) {
      let suffixCount = BINOMIALS[(FACILITY_COUNT - candidate - 1u) * 9u + SELECT_COUNT - slot - 1u];
      if (rank < suffixCount) { selection[slot] = candidate; remaining[slot] = capacities[capacitiesOffset + candidate]; first = candidate + 1u; break; }
      rank -= suffixCount;
    }
  }
}
fn chooseFacility(row: u32) -> u32 {
  let weight = demand[demandOffset + row]; var best = 0xffffffffu; var bestCost = infiniteCost();
  for (var slot = 0u; slot < SELECT_COUNT; slot++) {
    let cost = costs[costsOffset + row * FACILITY_COUNT + selection[slot]];
    if (cost >= 0.0 && cost < bestCost && (!CAPACITATED || remaining[slot] >= weight)) { best = slot; bestCost = cost; }
  }
  if (best != 0xffffffffu && CAPACITATED) { remaining[best] -= weight; }
  return best;
}
fn isFiniteNonnegative(value: f32) -> bool { return value >= 0.0 && (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }`,h=[{name:`costs`,view:o.costs,type:`f32`,access:`read`},{name:`demand`,view:o.demand,type:`f32`,access:`read`},{name:`capacities`,view:o.capacities,type:`f32`,access:`read`}],g=o.operation===`capacitated-p-median`,_=n(r,{id:`${a}-reduce`,operation:`GPUBoundedFacilityAllocation`,scores:f,candidateCount:u});return[t(r,{id:`${a}-score`,operation:`GPUBoundedFacilityAllocation`,variant:o.operation,invocationCount:u*(g?1:64),workgroupSize:64,guardIndex:g,bindings:[...h,{name:`scores`,view:f,type:`u32`,access:`read_write`}],declarations:g?m:`${m.replace(`var<private> selection`,`var<workgroup> selection`)}
var<workgroup> partialMissing: array<u32, 64>;
var<workgroup> partialMaximum: array<f32, 64>;
var<workgroup> partialInvalid: array<u32, 64>;`,body:g?`initializeSubset(index); var objective = 0.0; var unassigned = 0u; var valid = 1u;
  if (CAPACITATED) { for (var candidate = 0u; candidate < FACILITY_COUNT; candidate++) { if (!isFiniteNonnegative(capacities[capacitiesOffset + candidate])) { valid = 0u; } } }
  for (var row = 0u; row < DEMAND_COUNT; row++) {
    let weight = demand[demandOffset + row];
    if (!isFiniteNonnegative(weight)) { valid = 0u; unassigned++; continue; }
    let slot = chooseFacility(row);
    if (slot == 0xffffffffu) { unassigned++; continue; }
    let cost = costs[costsOffset + row * FACILITY_COUNT + selection[slot]];
    if (CAPACITATED) { objective += weight * cost; } else { objective = max(objective, cost); }
  }
  scores[scoresOffset + index * 4u] = unassigned;
  scores[scoresOffset + index * 4u + 1u] = bitcast<u32>(objective);
  scores[scoresOffset + index * 4u + 2u] = index;
  scores[scoresOffset + index * 4u + 3u] = 1u - valid;`:`if (localInvocationIndex == 0u && workgroupIndex < ${u}u) { initializeSubset(workgroupIndex); }
  workgroupBarrier();
  var missing = 0u; var maximum = 0.0; var invalid = 0u;
  if (workgroupIndex < ${u}u) {
    for (var row = localInvocationIndex; row < DEMAND_COUNT; row += 64u) {
      if (!isFiniteNonnegative(demand[demandOffset + row])) { missing++; invalid = 1u; continue; }
      let slot = chooseFacility(row);
      if (slot == 0xffffffffu) { missing++; continue; }
      maximum = max(maximum, costs[costsOffset + row * FACILITY_COUNT + selection[slot]]);
    }
  }
  partialMissing[localInvocationIndex] = missing;
  partialMaximum[localInvocationIndex] = maximum;
  partialInvalid[localInvocationIndex] = invalid;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (localInvocationIndex < stride) {
      partialMissing[localInvocationIndex] += partialMissing[localInvocationIndex + stride];
      partialMaximum[localInvocationIndex] = max(partialMaximum[localInvocationIndex], partialMaximum[localInvocationIndex + stride]);
      partialInvalid[localInvocationIndex] |= partialInvalid[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u && workgroupIndex < ${u}u) {
    scores[scoresOffset + workgroupIndex * 4u] = partialMissing[0];
    scores[scoresOffset + workgroupIndex * 4u + 1u] = bitcast<u32>(partialMaximum[0]);
    scores[scoresOffset + workgroupIndex * 4u + 2u] = workgroupIndex;
    scores[scoresOffset + workgroupIndex * 4u + 3u] = partialInvalid[0];
  }`}),..._.nodes,t(r,{id:`${a}-choose`,operation:`GPUBoundedFacilityAllocation`,variant:`choose`,invocationCount:1,workgroupSize:2,bindings:[...h,{name:`winner`,view:_.winner,type:`u32`,access:`read`},{name:`selected`,view:o.output.selectedFacilities,type:`u32`,access:`read_write`},...g?[{name:`assignments`,view:o.output.assignments,type:`u32`,access:`read_write`}]:[],{name:`objective`,view:o.output.objective,type:`f32`,access:`read_write`},{name:`status`,view:o.output.status,type:`u32`,access:`read_write`}],declarations:m,body:`initializeSubset(winner[winnerOffset + 2u]);
  for (var candidate = 0u; candidate < FACILITY_COUNT; candidate++) { selected[selectedOffset + candidate] = 0u; }
  for (var slot = 0u; slot < SELECT_COUNT; slot++) { selected[selectedOffset + selection[slot]] = 1u; }
  ${g?`for (var row = 0u; row < DEMAND_COUNT; row++) {
    var assignment = 0xffffffffu;
    if (isFiniteNonnegative(demand[demandOffset + row])) { let slot = chooseFacility(row); if (slot != 0xffffffffu) { assignment = selection[slot]; } }
    assignments[assignmentsOffset + row] = assignment;
  }`:``}
  let missing = winner[winnerOffset]; let valid = 1u - winner[winnerOffset + 3u];
  objective[objectiveOffset] = select(infiniteCost(), bitcast<f32>(winner[winnerOffset + 1u]), missing == 0u && valid == 1u);
  status[statusOffset] = missing; status[statusOffset + 1u] = valid;
  status[statusOffset + 2u] = select(0u, 1u, !CAPACITATED && valid == 1u);`}),...g||o.demand.length===0?[]:[t(r,{id:`${a}-assign`,operation:`GPUBoundedFacilityAllocation`,variant:`p-center-assign`,invocationCount:o.demand.length,bindings:[...h,{name:`winner`,view:_.winner,type:`u32`,access:`read`},{name:`assignments`,view:o.output.assignments,type:`u32`,access:`read_write`}],declarations:m,body:`initializeSubset(winner[winnerOffset + 2u]);
  var assignment = 0xffffffffu;
  if (isFiniteNonnegative(demand[demandOffset + index])) {
    let slot = chooseFacility(index);
    if (slot != 0xffffffffu) { assignment = selection[slot]; }
  }
  assignments[assignmentsOffset + index] = assignment;`})]]}};function D(e,t){if(t>e)return 0;let n=1;for(let r=1;r<=t;r++)n=n*(e-r+1)/r;return Math.round(n)}var O=48,k=8,A=f({parameters:b,caption:`The chosen facilities (bright) and the demand each serves within its capacity.`,createInputs:({random:e,world:t})=>p(e,t,k,O),build(e,t,n){let{network:r}=t,i=m(e,r),a=h(e,i,t.demandNodes,r.nodeCount,t.facilityNodes),o=e.allocate(`capacities`,`float32`,k),s=e.allocate(`selected`,`uint32`,k),c=e.allocate(`assignments`,`uint32`,O),l=e.allocate(`objective`,`float32`,1),u=e.allocate(`status`,`uint32`,3),d=Math.round(n.selectCount);e.graph.add(new E({operation:n.operation===0?`p-center`:`capacitated-p-median`,facilityCount:k,selectCount:d,costs:a.view,demand:e.upload(`demand`,`float32`,Float32Array.from(t.population,e=>Math.round(e))).view,capacities:o.view,output:{selectedFacilities:s.view,assignments:c.view,objective:l.view,status:u.view}}));let f=t.population.reduce((e,t)=>e+Math.round(t),0);return{outputs:{...y(e,r),assignments:c,selected:s,selectedIds:_(e,{id:`selected`,flags:s,count:k}),demands:e.upload(`demands`,`float32x2`,t.demandPositions),candidates:e.upload(`candidates`,`float32x2`,t.facilityPositions)},update:e=>{i.costLimit.write(Float32Array.of(4)),o.write(new Float32Array(k).fill(f/d*e.capacity))}}},rebuildOn:[`operation`,`selectCount`],interaction:{kind:`probe`,hint:`hover a facility`,rest:({inputs:e,toUnit:t})=>t([e.facilityPositions[0],e.facilityPositions[1]]),resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=i(n.selected),o=i(n.assignments),s=0,c=1/0;for(let n=0;n<k;n++){if(a&&a[n]!==1)continue;let r=Math.hypot(t.facilityPositions[n*2]-e.x,t.facilityPositions[n*2+1]-e.y);r<c&&(c=r,s=n)}let l=[t.facilityPositions[s*2],t.facilityPositions[s*2+1]],u=[{shape:`ring`,at:l,radius:.04}],d=0,f=0;if(o)for(let e=0;e<o.length;e++)o[e]===s&&(d++,f+=Math.round(t.population[e]),u.length<200&&u.push({shape:`line`,from:l,to:[t.demandPositions[e*2],t.demandPositions[e*2+1]]}));let p=t.population.reduce((e,t)=>e+Math.round(t),0)/Math.round(r.selectCount)*r.capacity;return{spotlight:null,glyphs:u,readout:`facility ${s+1} · serves ${d} · load ${f} of ${p.toFixed(0)}`}}},draw:({outputs:e,theme:t})=>[v(e),{layer:`dots`,positions:e.demands,values:e.assignments,radius:7},...g(e.candidates,t,{ids:e.selectedIds,radius:11})]});export{A as default};