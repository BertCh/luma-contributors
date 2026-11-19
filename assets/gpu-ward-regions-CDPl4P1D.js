import{t as e}from"./hash-uint32-wL4xFYeV.js";import{t}from"./common-CshkX-d_.js";import{r as n}from"./kernel-C3tfQOn4.js";import{a as r,o as i,s as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as o,S as s,f as c,p as l,u,v as d,x as f,y as p}from"./gpu-contributor-utils-BIwLQON5.js";import{c as m}from"./spatial-weights-CqWysstN.js";import{l as h,r as g}from"./wgsl-common-DfyXiYB0.js";import{a as _,i as v,o as y,r as b,t as x}from"./regionalization-inputs-CyTCPBba.js";import{t as S}from"./gpu-dispatch-utils-CbRzFAgT.js";import{n as C,t as w}from"./construction-snapshot-DdkX9Pc0.js";import{i as T}from"./regionalization-command-budget-BFn_8OgY.js";var E=class{graph;id;rowCount;weights;parents;heads;links;changed;costs;previousEdges;capacity;endpoints;positions;hash;counts;dispatch;hashCapacity;dispatchWGSL;constructor(e,t,i,a){this.graph=e,this.id=t,this.rowCount=i,this.weights=a,this.capacity=Math.max(1,a.neighbors.length),this.hashCapacity=this.capacity*2,d(e.device,t,`contracted edge hash`,this.hashCapacity*4,`reduce the adjacency slot capacity`),d(e.device,t,`contracted edges`,this.capacity*8,`reduce the adjacency slot capacity`);let o=(r,i)=>n(e,`${t}-${r}`,`uint32`,i);this.parents=o(`parents`,i),this.heads=o(`heads`,i),this.endpoints=[0,1].map(r=>n(e,`${t}-edges-${r}`,`uint32x2`,this.capacity)),this.positions=[0,1].map(e=>o(`hash-positions-${e}`,this.capacity)),this.links=n(e,`${t}-edge-links`,`uint32x2`,this.capacity),this.costs=[0,1].map(r=>n(e,`${t}-edge-costs-${r}`,`float32`,this.capacity)),this.changed=o(`changed-regions`,i),this.previousEdges=o(`previous-edges`,this.capacity),this.hash=o(`edge-hash`,this.hashCapacity),this.counts=o(`edge-counts`,2),this.dispatch=r(e,{id:`${t}-edge-dispatch`,operation:`GPUWardRegions`,shapes:[[this.capacity,256]]});let s=S(`GPUWardRegions`,this.capacity,256,e.device.limits.maxComputeWorkgroupsPerDimension);this.dispatchWGSL=`let workgroups = (activeCount + 255u) / 256u;
  dispatch[dispatchOffset] = min(workgroups, ${s.x}u);
  dispatch[dispatchOffset + 1u] = select(1u, min((workgroups + ${s.x-1}u) / ${s.x}u, ${s.y}u), workgroups > ${s.x}u);
  dispatch[dispatchOffset + 2u] = max(1u, (workgroups + ${s.x*s.y-1}u) / ${s.x*s.y}u);`}getInitializeNodes(){let{graph:e,id:t,weights:n,capacity:r,hashCapacity:a}=this;return[i(e,{id:`${t}-initialize-edge-storage`,operation:`GPUWardRegions`,variant:`initialize-edge-storage`,invocationCount:Math.max(r,a),bindings:[{name:`hash`,view:this.hash,type:`u32`,access:`read_write`},{name:`positions`,view:this.positions[0],type:`u32`,access:`read_write`},{name:`edges`,view:this.endpoints[0],type:`u32`,access:`read_write`},{name:`initialCosts`,view:this.costs[0],type:`f32`,access:`read_write`},{name:`counts`,view:this.counts,type:`u32`,access:`read_write`}],declarations:g,body:`if (index < ${a}u) { hash[hashOffset + index] = 0xffffffffu; }
  if (index < ${r}u) { initialCosts[initialCostsOffset + index] = getNaN(); positions[positionsOffset + index] = 0xffffffffu; edges[edgesOffset + 2u * index] = 0xffffffffu; edges[edgesOffset + 2u * index + 1u] = 0xffffffffu; }
  if (index == 0u) { counts[countsOffset] = ${n.neighbors.length}u; counts[countsOffset + 1u] = 0u; }`}),i(e,{id:`${t}-initialize-edges`,operation:`GPUWardRegions`,variant:`initialize-edges`,invocationCount:this.rowCount,bindings:[{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:n.neighbors,type:`u32`,access:`read`},{name:`edges`,view:this.endpoints[0],type:`u32`,access:`read_write`},{name:`changed`,view:this.changed,type:`u32`,access:`read_write`},{name:`parents`,view:this.parents,type:`u32`,access:`read_write`}],body:`parents[parentsOffset + index] = index; changed[changedOffset + index] = 0u;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let other = neighbors[neighborsOffset + slot];
    if (other < ${this.rowCount}u) { edges[edgesOffset + 2u * slot] = min(index, other); edges[edgesOffset + 2u * slot + 1u] = max(index, other); }
  }`})]}getRoundNodes(e,t,n,r){let{graph:a,capacity:o,rowCount:s,heads:c,links:l}=this,u=`${this.id}-contract-${e}`,d=e%2,f=1-d,p=this.endpoints[f],m=this.costs[f],h={operation:`GPUWardRegions`,invocationCount:o,workgroupSize:256,...this.dispatch.getGate(0)},_={name:`counts`,view:this.counts,type:`u32`,access:`read`};return{nodes:[i(a,{id:`${u}-begin`,operation:`GPUWardRegions`,variant:`contract-begin`,invocationCount:1,bindings:[{name:`mergeDispatch`,view:t.view,type:`u32`,access:`read`},{name:`counts`,view:this.counts,type:`u32`,access:`read_write`},this.dispatch.binding],body:`counts[countsOffset + 1u] = 0u;
  let activeCount = select(0u, counts[countsOffset], mergeDispatch[mergeDispatchOffset] != 0u);
  ${this.dispatchWGSL}`}),i(a,{id:`${u}-clear-hash`,variant:`contract-clear-hash`,...h,bindings:[_,{name:`positions`,view:this.positions[d],type:`u32`,access:`read`},{name:`hash`,view:this.hash,type:`u32`,access:`read_write`}],body:`if (index >= counts[countsOffset]) { return; }
  let slot = positions[positionsOffset + index];
  if (slot != 0xffffffffu) { hash[hashOffset + slot] = 0xffffffffu; }`}),i(a,{id:`${u}-map`,variant:`contract-map`,...h,bindings:[_,{name:`edges`,view:this.endpoints[d],type:`u32`,access:`read_write`},{name:`changed`,view:this.changed,type:`u32`,access:`read`},{name:`costs`,view:this.costs[d],type:`f32`,access:`read_write`},{name:`parents`,view:this.parents,type:`u32`,access:`read`}],declarations:g,body:`if (index >= counts[countsOffset]) { return; }
  let first = edges[edgesOffset + 2u * index]; let second = edges[edgesOffset + 2u * index + 1u];
  if (first == 0xffffffffu) { return; }
  if (changed[changedOffset + first] != 0u || changed[changedOffset + second] != 0u) { costs[costsOffset + index] = getNaN(); }
  // Each edge held active roots before the preceding merge, so only one parent hop is needed.
  let firstRoot = parents[parentsOffset + first]; let secondRoot = parents[parentsOffset + second];
  edges[edgesOffset + 2u * index] = min(firstRoot, secondRoot);
  edges[edgesOffset + 2u * index + 1u] = max(firstRoot, secondRoot);`}),i(a,{id:`${u}-clear-heads`,operation:`GPUWardRegions`,variant:`contract-clear-heads`,invocationCount:s,...t.getGate(0),bindings:[{name:`changed`,view:this.changed,type:`u32`,access:`read_write`},{name:`heads`,view:c,type:`u32`,access:`read_write`}],body:`heads[headsOffset + index] = 0xffffffffu; changed[changedOffset + index] = 0u;`}),i(a,{id:`${u}-unique`,variant:`contract-unique`,...h,bindings:[_,{name:`edges`,view:this.endpoints[d],type:`u32`,access:`read`},{name:`positions`,view:this.positions[d],type:`u32`,access:`read_write`},{name:`hash`,view:this.hash,type:`atomic<u32>`,access:`read_write`}],body:`if (index >= counts[countsOffset]) { return; }
  positions[positionsOffset + index] = 0xffffffffu;
  let first = edges[edgesOffset + 2u * index]; let second = edges[edgesOffset + 2u * index + 1u];
  if (first == 0xffffffffu || first == second) { return; }
  var slot = ((first * 0x9e3779b9u) ^ (second * 0x85ebca6bu)) % ${this.hashCapacity}u;
  loop {
    let result = atomicCompareExchangeWeak(&hash[hashOffset + slot], 0xffffffffu, index);
    if (result.exchanged) { positions[positionsOffset + index] = slot; break; }
    // Weak CAS may fail spuriously without a competing insertion.
    if (result.old_value == 0xffffffffu) { continue; }
    let previous = result.old_value;
    if (edges[edgesOffset + 2u * previous] == first && edges[edgesOffset + 2u * previous + 1u] == second) { break; }
    slot = (slot + 1u) % ${this.hashCapacity}u;
  }`}),i(a,{id:`${u}-compact`,variant:`contract-compact`,...h,bindings:[{name:`counts`,view:this.counts,type:`atomic<u32>`,access:`read_write`},{name:`source`,view:this.endpoints[d],type:`u32`,access:`read`},{name:`positions`,view:this.positions[d],type:`u32`,access:`read`},{name:`edges`,view:p,type:`u32`,access:`read_write`},{name:`nextPositions`,view:this.positions[f],type:`u32`,access:`read_write`},{name:`previousEdges`,view:this.previousEdges,type:`u32`,access:`read_write`},{name:`heads`,view:c,type:`atomic<u32>`,access:`read_write`},{name:`links`,view:l,type:`u32`,access:`read_write`}],body:`if (index >= atomicLoad(&counts[countsOffset])) { return; }
  let hashSlot = positions[positionsOffset + index];
  if (hashSlot == 0xffffffffu) { return; }
  let destination = atomicAdd(&counts[countsOffset + 1u], 1u);
  previousEdges[previousEdgesOffset + destination] = index;
  let first = source[sourceOffset + 2u * index]; let second = source[sourceOffset + 2u * index + 1u];
  edges[edgesOffset + 2u * destination] = first; edges[edgesOffset + 2u * destination + 1u] = second;
  nextPositions[nextPositionsOffset + destination] = hashSlot;
  links[linksOffset + 2u * destination] = atomicExchange(&heads[headsOffset + first], destination);
  links[linksOffset + 2u * destination + 1u] = atomicExchange(&heads[headsOffset + second], destination);`}),i(a,{id:`${u}-commit`,operation:`GPUWardRegions`,variant:`contract-commit`,invocationCount:1,bindings:[{name:`mergeDispatch`,view:t.view,type:`u32`,access:`read`},{name:`counts`,view:this.counts,type:`u32`,access:`read_write`},this.dispatch.binding],body:`if (mergeDispatch[mergeDispatchOffset] != 0u) { counts[countsOffset] = counts[countsOffset + 1u]; }
  let activeCount = select(0u, counts[countsOffset], mergeDispatch[mergeDispatchOffset] != 0u);
  ${this.dispatchWGSL}`}),i(a,{id:`${u}-costs`,variant:`contract-costs`,...h,bindings:[_,{name:`edges`,view:p,type:`u32`,access:`read`},{name:`regions`,view:n,type:`f32`,access:`read`},{name:`previousEdges`,view:this.previousEdges,type:`u32`,access:`read`},{name:`previousCosts`,view:this.costs[d],type:`f32`,access:`read`},{name:`costs`,view:m,type:`f32`,access:`read_write`}],body:`if (index >= counts[countsOffset]) { return; }
  let previous = previousCosts[previousCostsOffset + previousEdges[previousEdgesOffset + index]];
  if ((bitcast<u32>(previous) & 0x7fffffffu) <= 0x7f800000u) { costs[costsOffset + index] = previous; return; }
  let first = edges[edgesOffset + 2u * index] * ${r+1}u;
  let second = edges[edgesOffset + 2u * index + 1u] * ${r+1}u;
  let firstCount = regions[regionsOffset + first]; let secondCount = regions[regionsOffset + second];
  var increase = 0.0;
  for (var column = 1u; column <= ${r}u; column++) {
    let difference = regions[regionsOffset + first + column] / firstCount - regions[regionsOffset + second + column] / secondCount;
    increase += firstCount * secondCount / (firstCount + secondCount) * difference * difference;
  }
  costs[costsOffset + index] = increase;`})],adjacency:{heads:c,edges:p,links:l,costs:m}}}getPublishNodes(e){let t=[],n=this.parents,r=e;for(let e=0;e<Math.max(1,Math.ceil(Math.log2(this.rowCount)));e++)t.push(i(this.graph,{id:`${this.id}-publish-${e}`,operation:`GPUWardRegions`,variant:`publish-labels`,invocationCount:this.rowCount,bindings:[{name:`source`,view:n,type:`u32`,access:`read`},{name:`destination`,view:r,type:`u32`,access:`read_write`}],body:`let parent = source[sourceOffset + index]; destination[destinationOffset + index] = source[sourceOffset + parent];`})),[n,r]=[r,n];return n!==e&&t.push(i(this.graph,{id:`${this.id}-publish-copy`,operation:`GPUWardRegions`,variant:`publish-copy`,invocationCount:this.rowCount,bindings:[{name:`source`,view:n,type:`u32`,access:`read`},{name:`destination`,view:e,type:`u32`,access:`read_write`}],body:`destination[destinationOffset + index] = source[sourceOffset + index];`})),t}};function D(e,t){let{id:n,rowCount:a,gate:o,rowWorkgroupSize:s}=t,c=r(e,{id:`${n}-selection-dispatch`,operation:`GPUWardRegions`,shapes:[[a,s],[1,2]]}),l={name:`summary`,view:t.summary,type:`u32`,access:`read`},u=[{name:`regionCost`,view:t.regionCost,type:`u32`,access:`read`},{name:`regionKey`,view:t.regionKey,type:`u32`,access:`read`},{name:`regionOther`,view:t.regionOther,type:`u32`,access:`read`}],d=`const ROW_COUNT: u32 = ${a}u;
fn isMutual(region: u32) -> bool {
  let other = regionOther[regionOtherOffset + region];
  return other > region && other < ROW_COUNT && regionOther[regionOtherOffset + other] == region;
}
fn getWord(region: u32, word: u32) -> u32 {
  if (word == 0u) { return regionCost[regionCostOffset + region]; }
  if (word == 1u) { return regionKey[regionKeyOffset + region]; }
  return region;
}`,f=[i(e,{id:`${n}-selection-reset`,operation:`GPUWardRegions`,variant:`mutual-selection-reset`,invocationCount:256,workgroupSize:256,...o.getGate(1),bindings:[{name:`summary`,view:t.summary,type:`u32`,access:`read_write`},{name:`histogram`,view:t.histogram,type:`u32`,access:`read_write`}],body:`histogram[histogramOffset + index] = 0u; if (index < 7u) { summary[summaryOffset + index] = 0u; }`}),i(e,{id:`${n}-selection-count`,operation:`GPUWardRegions`,variant:`mutual-selection-count`,invocationCount:a,workgroupSize:s,...o.getGate(0),bindings:[...u,{name:`summary`,view:t.summary,type:`atomic<u32>`,access:`read_write`}],declarations:d,body:`if (isMutual(index)) { atomicAdd(&summary[summaryOffset], 1u); }`}),i(e,{id:`${n}-selection-prepare`,operation:`GPUWardRegions`,variant:`mutual-selection-prepare`,invocationCount:1,bindings:[{name:`mergeDispatch`,view:o.view,type:`u32`,access:`read`},{name:`summary`,view:t.summary,type:`u32`,access:`read_write`},{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`status`,view:t.status,type:`u32`,access:`read`},c.binding],body:`var budget = 0u;
  if (mergeDispatch[mergeDispatchOffset] != 0u && status[statusOffset + 3u] == 0u && state[stateOffset] > ${t.targetRegionCount}u) { budget = state[stateOffset] - ${t.targetRegionCount}u; }
  summary[summaryOffset + 1u] = budget; summary[summaryOffset + 2u] = budget;
  ${c.disarmWGSL}
  if (budget > 0u && summary[summaryOffset] > budget) { ${c.armWGSL} }`})];for(let r=0;r<12;r++)f.push(i(e,{id:`${n}-selection-count-${r}`,operation:`GPUWardRegions`,variant:`mutual-selection-digit-count`,invocationCount:a,workgroupSize:s,...c.getGate(0),bindings:[...u,l,{name:`histogram`,view:t.histogram,type:`atomic<u32>`,access:`read_write`}],declarations:d,body:`if (!isMutual(index)) { return; }
  let cursor = summary[summaryOffset + 6u]; let word = cursor / 4u; let shift = 24u - (cursor % 4u) * 8u;
  for (var previous = 0u; previous < word; previous++) { if (getWord(index, previous) != summary[summaryOffset + 3u + previous]) { return; } }
  let value = getWord(index, word);
  if (shift < 24u && (value >> (shift + 8u)) != (summary[summaryOffset + 3u + word] >> (shift + 8u))) { return; }
  atomicAdd(&histogram[histogramOffset + ((value >> shift) & 255u)], 1u);`}),i(e,{id:`${n}-selection-choose-${r}`,operation:`GPUWardRegions`,variant:`mutual-selection-digit-choose`,invocationCount:1,workgroupSize:2,...c.getGate(1),bindings:[{name:`summary`,view:t.summary,type:`u32`,access:`read_write`},{name:`histogram`,view:t.histogram,type:`u32`,access:`read_write`}],body:`let cursor = summary[summaryOffset + 6u]; let word = cursor / 4u; let shift = 24u - (cursor % 4u) * 8u;
  var remaining = summary[summaryOffset + 2u]; var selected = false;
  for (var digit = 0u; digit < 256u; digit++) {
    let count = histogram[histogramOffset + digit]; histogram[histogramOffset + digit] = 0u;
    if (!selected) {
      if (remaining > count) { remaining -= count; }
      else { summary[summaryOffset + 3u + word] |= digit << shift; selected = true; }
    }
  }
  summary[summaryOffset + 2u] = remaining; summary[summaryOffset + 6u] = cursor + 1u;`}));return f.push(i(e,{id:`${n}-select-pairs`,operation:`GPUWardRegions`,variant:`mutual-select-distributed-pairs`,invocationCount:a,workgroupSize:s,...o.getGate(0),bindings:[...u,l,{name:`mergeFlag`,view:t.mergeFlag,type:`u32`,access:`read_write`}],declarations:d,body:`let budget = summary[summaryOffset + 1u];
  var accepted = budget > 0u && isMutual(index);
  if (accepted && summary[summaryOffset] > budget) {
    let cost = getWord(index, 0u); let key = getWord(index, 1u);
    let thresholdCost = summary[summaryOffset + 3u]; let thresholdKey = summary[summaryOffset + 4u];
    accepted = cost < thresholdCost || (cost == thresholdCost && (key < thresholdKey || (key == thresholdKey && index <= summary[summaryOffset + 5u])));
  }
  mergeFlag[mergeFlagOffset + index] = u32(accepted);`})),f}function O(e){return u(e,4)}var k=256,A=1;function j(e={}){return v(`GPUWardRegions`,e.seed)}var M={iterationCount:0,converged:1,iterationLimitReached:2,invalidCount:3,length:4},N=M.length,P=class{estimatedCommandNodeCount;id;props;rowCount;columnCount;constructor(e){let t=w(e,[`id`,`weights`,`values`,`columnCount`,`standardize`,`targetRegionCount`,`maximumIterations`,`mergeSchedule`,`parameters`,`output`]),n=w(t.weights,[`offsets`,`neighbors`,`weights`,`distances`,`status`,`information`,`spatialContext`]),r=n.information?w(n.information,[`rowCount`,`transform`,`symmetric`,`hasSelfWeights`,`provenance`]):void 0;e=Object.freeze({...t,weights:Object.freeze({...n,status:n.status?w(n.status,[`count`,`requiredCount`,`overflow`,`candidateOverflow`,`uncertainCount`,`invalidCount`,`iterationCount`,`converged`,`iterationLimitReached`,`approximation`]):void 0,information:r?Object.freeze({...r,provenance:r.provenance?w(r.provenance,[`source`,`criterion`,`k`,`radius`,`order`]):void 0}):void 0,spatialContext:n.spatialContext?C(n.spatialContext):void 0}),output:w(t.output,[`labels`,`objective`,`status`])}),this.id=e.id??`ward-regions`,this.props=e,this.rowCount=m(this.id,e.weights),this.columnCount=e.columnCount??1;let{id:i,rowCount:a,columnCount:s}=this,{output:u}=e;if(c(i,`columnCount`,s,1,8),c(i,`targetRegionCount`,e.targetRegionCount,1,a),c(i,`maximumIterations`,e.maximumIterations,0,a-e.targetRegionCount),this.estimatedCommandNodeCount=T(a,e.maximumIterations,e.mergeSchedule===`mutual-nearest`),l(i,this.estimatedCommandNodeCount),e.mergeSchedule!==void 0&&e.mergeSchedule!==`sequential`&&e.mergeSchedule!==`mutual-nearest`)throw Error(`${i} mergeSchedule must be 'sequential' or 'mutual-nearest'`);if(_(i,e.parameters,A),o(e.values,[`float32`],`${i} values`),e.values.length!==a*s)throw Error(`${i} values length must equal rows * columnCount`);if(o(u.labels,[`uint32`],`${i} output.labels`),u.labels.length!==a)throw Error(`${i} output.labels length must equal the weights row count`);if(o(u.objective,[`float32`],`${i} output.objective`),o(u.status,[`uint32`],`${i} output.status`),u.objective.length<1)throw Error(`${i} output.objective must hold one value`);if(u.status.length<M.length)throw Error(`${i} output.status must hold ${M.length} words`);p(i,[u.labels,u.objective,u.status],[e.weights.offsets,e.weights.neighbors,e.weights.weights,e.weights.status?.overflow,e.weights.status?.candidateOverflow,e.values,e.parameters])}getCommandNodes(o){let{id:c,props:l,rowCount:u,columnCount:d}=this,{output:p,parameters:m}=l;f(c,o,[l.weights.offsets,l.weights.neighbors,l.weights.weights,l.weights.status?.overflow,l.weights.status?.candidateOverflow,l.values,m,p.labels,p.objective,p.status]),y(c,l.weights,b(o.device)),s(c,`rows x (columnCount + 1)`,u*(d+1),O(o.device));let g=d+1,_=n(o,`${c}-regions`,`float32`,u*g),v=l.mergeSchedule===`mutual-nearest`,S=p.status.length>N,C=n(o,`${c}-state`,`uint32`,v?5:1),w=l.standardize!==!1,T=w?n(o,`${c}-table`,`float32`,u*d):l.values,D=new E(o,c,u,l.weights),A=v?{rowCost:n(o,`${c}-row-cost`,`uint32`,u),rowOther:n(o,`${c}-row-other`,`uint32`,u),regionCost:n(o,`${c}-region-cost`,`uint32`,u),regionKey:n(o,`${c}-region-key`,`uint32`,u),regionOther:n(o,`${c}-region-other`,`uint32`,u),mergeFlag:n(o,`${c}-merge-flag`,`uint32`,u),selectionSummary:n(o,`${c}-selection-summary`,`uint32`,7),selectionHistogram:n(o,`${c}-selection-histogram`,`uint32`,256)}:null,j=n(o,`${c}-candidate-cost-a`,`float32`,u),M=n(o,`${c}-candidate-cost-b`,`float32`,u),P=n(o,`${c}-candidate-first-a`,`uint32`,u),F=n(o,`${c}-candidate-first-b`,`uint32`,u),I=n(o,`${c}-candidate-second-a`,`uint32`,u),L=n(o,`${c}-candidate-second-b`,`uint32`,u),R=n(o,`${c}-candidate-key-a`,`uint32`,u),z=n(o,`${c}-candidate-key-b`,`uint32`,u),B=[];for(let e=v?1:u;e>1;){let n=t(2**Math.ceil(Math.log2(e)),2,k),r=Math.ceil(e/n);B.push({activeCount:e,workgroupSize:n,nextCount:r}),e=r}let V=a(u),H=l.maximumIterations>0?r(o,{id:`${c}-merge-dispatch`,operation:`GPUWardRegions`,shapes:[[u,V],...v?[[k,k]]:[],...B.map(e=>[e.activeCount,e.workgroupSize])]}):null,U=`const ROW_COUNT: u32 = ${u}u;
const COLUMN_COUNT: u32 = ${d}u;
const REGION_STRIDE: u32 = ${g}u;
const TARGET_REGION_COUNT: u32 = ${l.targetRegionCount}u;
const NONE: u32 = 0xffffffffu;
const LARGE_INCREASE: f32 = 3.402823466e+38;
${h}
fn regionCount(region: u32) -> f32 { return regions[regionsOffset + region * REGION_STRIDE]; }
fn regionSum(region: u32, column: u32) -> f32 { return regions[regionsOffset + region * REGION_STRIDE + 1u + column]; }`,W=`${e}
fn pairKey(first: u32, second: u32) -> u32 {
  return hashUint32((first * 0x9e3779b9u) ^ second ^ parameters[parametersOffset]);
}`,G={name:`parameters`,view:m,type:`u32`,access:`read`},K=[...D.getInitializeNodes(),...w?[x(o,{id:`${c}-standardize`,operation:`GPUWardRegions`,values:l.values,scaled:T,rowCount:u,columnCount:d})]:[],i(o,{id:`${c}-initialize-status`,operation:`GPUWardRegions`,variant:`initialize-status`,invocationCount:1,bindings:[{name:`state`,view:C,type:`u32`,access:`read_write`},{name:`objective`,view:p.objective,type:`f32`,access:`read_write`},{name:`status`,view:p.status,type:`u32`,access:`read_write`},...H&&l.weights.status?.overflow?[{name:`inputOverflow`,view:l.weights.status.overflow,type:`u32`,access:`read`}]:[],...H&&l.weights.status?.candidateOverflow?[{name:`inputCandidateOverflow`,view:l.weights.status.candidateOverflow,type:`u32`,access:`read`}]:[],...H?[H.binding]:[]],body:`state[stateOffset] = ${u}u;
  ${v?`state[stateOffset + 1u] = 0u;
  state[stateOffset + 2u] = 0u;
  state[stateOffset + 3u] = 0u;
  state[stateOffset + 4u] = 0u;`:``}
  objective[objectiveOffset] = 0.0;
  status[statusOffset] = 0u;
  status[statusOffset + 1u] = 0u;
  status[statusOffset + 2u] = 0u;
  status[statusOffset + 3u] = 0u;
  ${v&&S?`status[statusOffset + ${N}u] = 0u;`:``}
  ${H?`if (${l.weights.status?.overflow?`inputOverflow[inputOverflowOffset] != 0u`:`false`} ||
      ${l.weights.status?.candidateOverflow?`inputCandidateOverflow[inputCandidateOverflowOffset] != 0u`:`false`}) {
    ${H.disarmWGSL}
  } else {
    ${H.armWGSL}
  }`:``}`}),i(o,{id:`${c}-initialize-regions`,operation:`GPUWardRegions`,variant:`initialize-regions`,invocationCount:u,bindings:[{name:`offsets`,view:l.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:l.weights.neighbors,type:`u32`,access:`read`},{name:`values`,view:l.values,type:`f32`,access:`read`},{name:`table`,view:T,type:`f32`,access:`read`},{name:`labels`,view:p.labels,type:`u32`,access:`read_write`},{name:`regions`,view:_,type:`f32`,access:`read_write`},{name:`status`,view:p.status,type:`atomic<u32>`,access:`read_write`}],declarations:U,body:`labels[labelsOffset + index] = index;
  let regionBase = regionsOffset + index * REGION_STRIDE;
  regions[regionBase] = 1.0;
  var invalidCount = 0u;
  for (var column = 0u; column < COLUMN_COUNT; column++) {
    invalidCount += select(0u, 1u, !isFiniteFloat(values[valuesOffset + index * COLUMN_COUNT + column]));
    regions[regionBase + 1u + column] = table[tableOffset + index * COLUMN_COUNT + column];
  }
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    invalidCount += select(0u, 1u, neighbors[neighborsOffset + slot] >= ROW_COUNT);
  }
  if (invalidCount > 0u) { atomicAdd(&status[statusOffset + 3u], invalidCount); }`})];for(let e=0;e<l.maximumIterations;e++){let t=D.getRoundNodes(e,H,_,d);K.push(...t.nodes);let n=t.adjacency;if(A){K.push(...this.createMutualNearestRoundNodes(o,{iteration:e,buffers:A,adjacency:n,parents:D.parents,changed:D.changed,regions:_,state:C,gate:H,rowWorkgroupSize:V,declarations:U,pairKeyWGSL:W,parametersBinding:G}));continue}K.push(i(o,{id:`${c}-iteration-${e}-candidates`,operation:`GPUWardRegions`,variant:`adjacent-candidates`,invocationCount:u,workgroupSize:V,...H.getGate(0),bindings:[{name:`heads`,view:n.heads,type:`u32`,access:`read`},{name:`edges`,view:n.edges,type:`u32`,access:`read`},{name:`links`,view:n.links,type:`u32`,access:`read`},{name:`costs`,view:n.costs,type:`f32`,access:`read`},{name:`costOut`,view:j,type:`f32`,access:`read_write`},{name:`firstOut`,view:P,type:`u32`,access:`read_write`},{name:`secondOut`,view:I,type:`u32`,access:`read_write`},G],declarations:`const NONE: u32 = 0xffffffffu;
const LARGE_INCREASE: f32 = 3.402823466e+38;
${W}`,body:`var bestFirst = NONE;
  var bestSecond = NONE;
  var bestIncrease = LARGE_INCREASE;
  var bestKey = NONE;
  {
    var edge = heads[headsOffset + index];
    loop {
      if (edge == NONE) { break; }
      let first = edges[edgesOffset + 2u * edge];
      let second = edges[edgesOffset + 2u * edge + 1u];
      let increase = costs[costsOffset + edge];
      edge = links[linksOffset + 2u * edge + select(1u, 0u, first == index)];
      if (first != index) { continue; }
      let key = pairKey(first, second);
      if (increase < bestIncrease || (increase == bestIncrease &&
          (key < bestKey || (key == bestKey && (first < bestFirst || (first == bestFirst && second < bestSecond)))))) {
        bestFirst = first; bestSecond = second; bestIncrease = increase; bestKey = key;
      }
    }
  }
  costOut[costOutOffset + index] = bestIncrease;
  firstOut[firstOutOffset + index] = bestFirst;
  secondOut[secondOutOffset + index] = bestSecond;`}));let r=j,a=P,s=I,f=null,m=M,h=F,v=L,y=z,b=R;for(let[t,{activeCount:n,workgroupSize:l,nextCount:u}]of B.entries()){let d=f?`keyIn[keyInOffset + index]`:`select(pairKey(first, second), NONE, first == NONE)`,p=f?[{name:`keyIn`,view:f,type:`u32`,access:`read`}]:[G];K.push(i(o,{id:`${c}-iteration-${e}-reduce-${t}`,operation:`GPUWardRegions`,variant:`workgroup-reduction`,invocationCount:n,workgroupSize:l,guardIndex:!1,...H.getGate(1+t),bindings:[{name:`costIn`,view:r,type:`f32`,access:`read`},{name:`firstIn`,view:a,type:`u32`,access:`read`},{name:`secondIn`,view:s,type:`u32`,access:`read`},...p,{name:`costOut`,view:m,type:`f32`,access:`read_write`},{name:`firstOut`,view:h,type:`u32`,access:`read_write`},{name:`secondOut`,view:v,type:`u32`,access:`read_write`},{name:`keyOut`,view:y,type:`u32`,access:`read_write`}],declarations:`const INPUT_COUNT: u32 = ${n}u;
const OUTPUT_COUNT: u32 = ${u}u;
const WORKGROUP_SIZE: u32 = ${l}u;
const NONE: u32 = 0xffffffffu;
const LARGE_INCREASE: f32 = 3.402823466e+38;
var<workgroup> sharedCost: array<f32, ${l}>;
var<workgroup> sharedKey: array<u32, ${l}>;
var<workgroup> sharedFirst: array<u32, ${l}>;
var<workgroup> sharedSecond: array<u32, ${l}>;
${f?``:W}
fn isBetter(slot: u32, other: u32) -> bool {
  let cost = sharedCost[slot];
  let otherCost = sharedCost[other];
  if (cost != otherCost) { return cost < otherCost; }
  let key = sharedKey[slot];
  let otherKey = sharedKey[other];
  if (key != otherKey) { return key < otherKey; }
  let first = sharedFirst[slot];
  let otherFirst = sharedFirst[other];
  if (first != otherFirst) { return first < otherFirst; }
  return sharedSecond[slot] < sharedSecond[other];
}`,body:`var cost = LARGE_INCREASE;
  var first = NONE;
  var second = NONE;
  var key = NONE;
  if (index < INPUT_COUNT) {
    cost = costIn[costInOffset + index];
    first = firstIn[firstInOffset + index];
    second = secondIn[secondInOffset + index];
    key = ${d};
  }
  sharedCost[localInvocationIndex] = cost;
  sharedKey[localInvocationIndex] = key;
  sharedFirst[localInvocationIndex] = first;
  sharedSecond[localInvocationIndex] = second;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride && isBetter(localInvocationIndex + stride, localInvocationIndex)) {
      let winner = localInvocationIndex + stride;
      sharedCost[localInvocationIndex] = sharedCost[winner];
      sharedKey[localInvocationIndex] = sharedKey[winner];
      sharedFirst[localInvocationIndex] = sharedFirst[winner];
      sharedSecond[localInvocationIndex] = sharedSecond[winner];
    }
    workgroupBarrier();
  }
  let group = index / WORKGROUP_SIZE;
  if (localInvocationIndex == 0u && group < OUTPUT_COUNT) {
    costOut[costOutOffset + group] = sharedCost[0];
    firstOut[firstOutOffset + group] = sharedFirst[0];
    secondOut[secondOutOffset + group] = sharedSecond[0];
    keyOut[keyOutOffset + group] = sharedKey[0];
  }`}));let g=y;[r,m]=[m,r],[a,h]=[h,a],[s,v]=[v,s],f=g,y=b,b=g}K.push(i(o,{id:`${c}-iteration-${e}-relabel`,operation:`GPUWardRegions`,variant:`merge-parent`,invocationCount:1,bindings:[{name:`first`,view:a,type:`u32`,access:`read`},{name:`second`,view:s,type:`u32`,access:`read`},{name:`parents`,view:D.parents,type:`u32`,access:`read_write`},{name:`changed`,view:D.changed,type:`u32`,access:`read_write`},{name:`status`,view:p.status,type:`u32`,access:`read`},{name:`dispatch`,view:H.view,type:`u32`,access:`read`}],declarations:`const NONE: u32 = 0xffffffffu;`,body:`let firstRegion = first[firstOffset];
  let secondRegion = second[secondOffset];
  if (dispatch[dispatchOffset] != 0u && status[statusOffset + 3u] == 0u && firstRegion != NONE) {
    parents[parentsOffset + secondRegion] = firstRegion;
    changed[changedOffset + firstRegion] = 1u; changed[changedOffset + secondRegion] = 1u;
  }`}),i(o,{id:`${c}-iteration-${e}-apply`,operation:`GPUWardRegions`,variant:`apply-merge`,invocationCount:1,bindings:[{name:`cost`,view:r,type:`f32`,access:`read`},{name:`first`,view:a,type:`u32`,access:`read`},{name:`second`,view:s,type:`u32`,access:`read`},{name:`regions`,view:_,type:`f32`,access:`read_write`},{name:`state`,view:C,type:`u32`,access:`read_write`},{name:`objective`,view:p.objective,type:`f32`,access:`read_write`},{name:`status`,view:p.status,type:`u32`,access:`read_write`},H.binding],declarations:`const COLUMN_COUNT: u32 = ${d}u;
const REGION_STRIDE: u32 = ${g}u;
const TARGET_REGION_COUNT: u32 = ${l.targetRegionCount}u;
const NONE: u32 = 0xffffffffu;`,body:`if (dispatch[dispatchOffset] == 0u) { return; }
  let firstRegion = first[firstOffset];
  let secondRegion = second[secondOffset];
  if (firstRegion != NONE && state[stateOffset] > TARGET_REGION_COUNT && status[statusOffset + 3u] == 0u) {
    let firstBase = regionsOffset + firstRegion * REGION_STRIDE;
    let secondBase = regionsOffset + secondRegion * REGION_STRIDE;
    regions[firstBase] += regions[secondBase];
    regions[secondBase] = 0.0;
    for (var column = 0u; column < COLUMN_COUNT; column++) {
      regions[firstBase + 1u + column] += regions[secondBase + 1u + column];
      regions[secondBase + 1u + column] = 0.0;
    }
    objective[objectiveOffset] += cost[costOffset];
    state[stateOffset] -= 1u;
    status[statusOffset] += 1u;
  }
  // No adjacent pair, the target reached, or invalid input: no later merge can happen.
  if (firstRegion == NONE || state[stateOffset] <= TARGET_REGION_COUNT || status[statusOffset + 3u] != 0u) {
    ${H.disarmWGSL}
  }`}))}return K.push(...D.getPublishNodes(p.labels)),A?(K.push(this.createMutualNearestFinalizeNode(o,C)),K.push(...this.getIncompleteAdjacencySuppressionNodes(o)),K):(K.push(i(o,{id:`${c}-finalize`,operation:`GPUWardRegions`,variant:`finalize`,invocationCount:1,bindings:[{name:`state`,view:C,type:`u32`,access:`read`},{name:`status`,view:p.status,type:`u32`,access:`read_write`}],declarations:`const TARGET_REGION_COUNT: u32 = ${l.targetRegionCount}u;
const MAXIMUM_ITERATIONS: u32 = ${l.maximumIterations}u;`,body:`let remaining = state[stateOffset];
  let completed = status[statusOffset];
  status[statusOffset + 1u] = select(0u, 1u, remaining == TARGET_REGION_COUNT && status[statusOffset + 3u] == 0u);
  status[statusOffset + 2u] = select(0u, 1u, remaining > TARGET_REGION_COUNT && completed == MAXIMUM_ITERATIONS && status[statusOffset + 3u] == 0u);`})),K.push(...this.getIncompleteAdjacencySuppressionNodes(o)),K)}getIncompleteAdjacencySuppressionNodes(e){let{id:t,props:n,rowCount:r}=this,a=n.weights.status?.overflow,o=n.weights.status?.candidateOverflow;return!a&&!o?[]:[i(e,{id:`${t}-suppress-incomplete-adjacency`,operation:`GPUWardRegions`,variant:`suppress-incomplete-adjacency`,invocationCount:r,bindings:[...a?[{name:`inputOverflow`,view:a,type:`u32`,access:`read`}]:[],...o?[{name:`inputCandidateOverflow`,view:o,type:`u32`,access:`read`}]:[],{name:`labels`,view:n.output.labels,type:`u32`,access:`read_write`},{name:`objective`,view:n.output.objective,type:`f32`,access:`read_write`},{name:`status`,view:n.output.status,type:`u32`,access:`read_write`}],body:`let incomplete = ${a?`inputOverflow[inputOverflowOffset] != 0u`:`false`} ||
    ${o?`inputCandidateOverflow[inputCandidateOverflowOffset] != 0u`:`false`};
  if (incomplete) {
    labels[labelsOffset + index] = 0xffffffffu;
    if (index == 0u) {
      var nanBits = 0x7fc00000u;
      objective[objectiveOffset] = bitcast<f32>(nanBits);
      status[statusOffset + ${M.converged}u] = 0u;
      status[statusOffset + ${M.iterationLimitReached}u] = 0u;
    }
  }`})]}createMutualNearestRoundNodes(e,t){let{id:n,props:r,rowCount:a,columnCount:o}=this,{output:s}=r,{iteration:c,buffers:l,adjacency:u,parents:d,changed:f,regions:p,state:m,gate:h,rowWorkgroupSize:g}=t,{pairKeyWGSL:_,parametersBinding:v}=t,y=`${n}-round-${c}`,b={operation:`GPUWardRegions`,invocationCount:a,workgroupSize:g,...h.getGate(0)},x={operation:`GPUWardRegions`,invocationCount:k,workgroupSize:k,guardIndex:!1,...h.getGate(1)},S=`const NONE: u32 = 0xffffffffu;`,C=`${S}
const ROW_COUNT: u32 = ${a}u;`,w={name:`rowCost`,view:l.rowCost,type:`u32`,access:`read`},T={name:`rowOther`,view:l.rowOther,type:`u32`,access:`read`},E={name:`regionCost`,view:l.regionCost,type:`u32`,access:`read`},O={name:`regionKey`,view:l.regionKey,type:`u32`,access:`read`},A=e=>({name:`mergeFlag`,view:l.mergeFlag,type:`u32`,access:e}),j=e=>({name:`regionOther`,view:l.regionOther,type:`u32`,access:e}),M=s.status.length>N;return[i(e,{...b,id:`${y}-reset-nearest`,variant:`mutual-reset-nearest`,bindings:[{name:`regionCost`,view:l.regionCost,type:`u32`,access:`read_write`},{name:`regionKey`,view:l.regionKey,type:`u32`,access:`read_write`},j(`read_write`)],declarations:S,body:`regionCost[regionCostOffset + index] = NONE;
  regionKey[regionKeyOffset + index] = NONE;
  regionOther[regionOtherOffset + index] = NONE;`}),i(e,{...b,id:`${y}-nearest-candidates`,variant:`mutual-nearest-candidates`,bindings:[{name:`heads`,view:u.heads,type:`u32`,access:`read`},{name:`edges`,view:u.edges,type:`u32`,access:`read`},{name:`links`,view:u.links,type:`u32`,access:`read`},{name:`costs`,view:u.costs,type:`f32`,access:`read`},{name:`rowCost`,view:l.rowCost,type:`u32`,access:`read_write`},{name:`rowOther`,view:l.rowOther,type:`u32`,access:`read_write`},v],declarations:`${S}
${_}`,body:`let region = index;
  var bestCost = NONE;
  var bestKey = NONE;
  var bestOther = NONE;
  var edge = heads[headsOffset + index];
  loop {
    if (edge == NONE) { break; }
    let first = edges[edgesOffset + 2u * edge];
    let second = edges[edgesOffset + 2u * edge + 1u];
    let other = select(first, second, first == region);
    let cost = bitcast<u32>(max(costs[costsOffset + edge], 0.0));
    edge = links[linksOffset + 2u * edge + select(1u, 0u, first == region)];
    let key = pairKey(first, second);
    if (cost < bestCost || (cost == bestCost &&
        (key < bestKey || (key == bestKey && other < bestOther)))) {
      bestCost = cost; bestKey = key; bestOther = other;
    }
  }
  rowCost[rowCostOffset + index] = bestCost;
  rowOther[rowOtherOffset + index] = bestOther;`}),i(e,{...b,id:`${y}-nearest-cost`,variant:`mutual-nearest-cost`,bindings:[w,{name:`regionCost`,view:l.regionCost,type:`atomic<u32>`,access:`read_write`}],declarations:S,body:`let cost = rowCost[rowCostOffset + index];
  if (cost != NONE) { atomicMin(&regionCost[regionCostOffset + index], cost); }`}),i(e,{...b,id:`${y}-nearest-key`,variant:`mutual-nearest-key`,bindings:[w,T,E,{name:`regionKey`,view:l.regionKey,type:`atomic<u32>`,access:`read_write`},v],declarations:`${S}
${_}`,body:`let cost = rowCost[rowCostOffset + index];
  let region = index;
  if (cost != NONE && cost == regionCost[regionCostOffset + region]) {
    let other = rowOther[rowOtherOffset + index];
    atomicMin(&regionKey[regionKeyOffset + region], pairKey(min(region, other), max(region, other)));
  }`}),i(e,{...b,id:`${y}-nearest-region`,variant:`mutual-nearest-region`,bindings:[w,T,E,O,{name:`regionOther`,view:l.regionOther,type:`atomic<u32>`,access:`read_write`},v],declarations:`${S}
${_}`,body:`let cost = rowCost[rowCostOffset + index];
  let region = index;
  let other = rowOther[rowOtherOffset + index];
  if (cost != NONE && cost == regionCost[regionCostOffset + region] &&
      pairKey(min(region, other), max(region, other)) == regionKey[regionKeyOffset + region]) {
    atomicMin(&regionOther[regionOtherOffset + region], other);
  }`}),...a>4096?D(e,{id:y,rowCount:a,targetRegionCount:r.targetRegionCount,rowWorkgroupSize:g,gate:h,regionCost:l.regionCost,regionKey:l.regionKey,regionOther:l.regionOther,mergeFlag:l.mergeFlag,summary:l.selectionSummary,histogram:l.selectionHistogram,state:m,status:s.status}):[i(e,{...x,id:`${y}-select-pairs`,variant:`mutual-select-pairs`,bindings:[E,O,j(`read`),A(`read_write`),{name:`state`,view:m,type:`u32`,access:`read`},{name:`status`,view:s.status,type:`u32`,access:`read`}],declarations:`${C}
const TARGET_REGION_COUNT: u32 = ${r.targetRegionCount}u;
const WORKGROUP_SIZE: u32 = ${k}u;
var<workgroup> sharedCount: atomic<u32>;
var<workgroup> uniformValue: u32;
// First region of a reciprocal pair: it and its choice name each other.
fn isMutual(region: u32) -> bool {
  let other = regionOther[regionOtherOffset + region];
  return other != NONE && other > region && other < ROW_COUNT &&
    regionOther[regionOtherOffset + other] == region;
}
fn getWord(region: u32, word: u32) -> u32 {
  if (word == 0u) { return regionCost[regionCostOffset + region]; }
  if (word == 1u) { return regionKey[regionKeyOffset + region]; }
  return region;
}
// Sums the lanes' counts through the shared atomic and returns the total to every lane.
fn reduceCount(lane: u32, laneCount: u32) -> u32 {
  if (lane == 0u) { atomicStore(&sharedCount, 0u); }
  workgroupBarrier();
  atomicAdd(&sharedCount, laneCount);
  workgroupBarrier();
  if (lane == 0u) { uniformValue = atomicLoad(&sharedCount); }
  return workgroupUniformLoad(&uniformValue);
}`,body:`let lane = localInvocationIndex;
  var laneMutual = 0u;
  for (var region = lane; region < ROW_COUNT; region += WORKGROUP_SIZE) {
    if (isMutual(region)) { laneMutual += 1u; }
  }
  let mutualCount = reduceCount(lane, laneMutual);
  // A merge is allowed only down to the target, and never on invalid input.
  let regionsLeft = state[stateOffset];
  var budget = 0u;
  if (regionsLeft > TARGET_REGION_COUNT && status[statusOffset + 3u] == 0u) {
    budget = regionsLeft - TARGET_REGION_COUNT;
  }
  if (lane == 0u) { uniformValue = budget; }
  let uniformBudget = workgroupUniformLoad(&uniformValue);
  var remaining = uniformBudget;
  var threshold0 = 0u;
  var threshold1 = 0u;
  var threshold2 = 0u;
  if (uniformBudget > 0u && mutualCount > uniformBudget) {
    for (var wordIndex = 0u; wordIndex < 3u; wordIndex++) {
      var prefix = 0u;
      for (var bit = 32u; bit > 0u; bit--) {
        let shift = bit - 1u;
        var laneZero = 0u;
        for (var region = lane; region < ROW_COUNT; region += WORKGROUP_SIZE) {
          if (!isMutual(region)) { continue; }
          if (wordIndex > 0u && getWord(region, 0u) != threshold0) { continue; }
          if (wordIndex > 1u && getWord(region, 1u) != threshold1) { continue; }
          let word = getWord(region, wordIndex);
          if ((((word ^ prefix) >> shift) >> 1u) != 0u) { continue; }
          if (((word >> shift) & 1u) == 0u) { laneZero += 1u; }
        }
        let zeroCount = reduceCount(lane, laneZero);
        if (zeroCount < remaining) {
          remaining -= zeroCount;
          prefix |= 1u << shift;
        }
      }
      if (wordIndex == 0u) { threshold0 = prefix; }
      else if (wordIndex == 1u) { threshold1 = prefix; }
      else { threshold2 = prefix; }
    }
  }
  for (var region = lane; region < ROW_COUNT; region += WORKGROUP_SIZE) {
    var accepted = isMutual(region) && uniformBudget > 0u;
    if (accepted && mutualCount > uniformBudget) {
      let word0 = getWord(region, 0u);
      let word1 = getWord(region, 1u);
      accepted = word0 < threshold0 || (word0 == threshold0 &&
        (word1 < threshold1 || (word1 == threshold1 && region <= threshold2)));
    }
    mergeFlag[mergeFlagOffset + region] = select(0u, 1u, accepted);
  }`})],i(e,{...b,id:`${y}-merge-regions`,variant:`mutual-merge-regions`,bindings:[{name:`changed`,view:f,type:`u32`,access:`read_write`},{name:`regions`,view:p,type:`f32`,access:`read_write`},j(`read`),A(`read`)],declarations:`const REGION_STRIDE: u32 = ${o+1}u;`,body:`if (mergeFlag[mergeFlagOffset + index] == 0u) { return; }
  let firstBase = regionsOffset + index * REGION_STRIDE;
  let secondRegion = regionOther[regionOtherOffset + index];
  let secondBase = regionsOffset + secondRegion * REGION_STRIDE;
  changed[changedOffset + index] = 1u; changed[changedOffset + secondRegion] = 1u;
  for (var word = 0u; word < REGION_STRIDE; word++) {
    regions[firstBase + word] += regions[secondBase + word];
  }`}),i(e,{...b,id:`${y}-relabel`,variant:`mutual-relabel`,bindings:[{name:`parents`,view:d,type:`u32`,access:`read_write`},{name:`regions`,view:p,type:`f32`,access:`read_write`},j(`read`),A(`read`)],declarations:`${S}
const REGION_STRIDE: u32 = ${o+1}u;
// First region of the pair whose second region is \`region\`, or NONE.
fn getMergedInto(region: u32) -> u32 {
  let partner = regionOther[regionOtherOffset + region];
  if (partner != NONE && partner < region && mergeFlag[mergeFlagOffset + partner] != 0u &&
      regionOther[regionOtherOffset + partner] == region) {
    return partner;
  }
  return NONE;
}`,body:`let rowTarget = getMergedInto(index);
  if (rowTarget != NONE) { parents[parentsOffset + index] = rowTarget; }
  if (getMergedInto(index) != NONE) {
    for (var word = 0u; word < REGION_STRIDE; word++) {
      regions[regionsOffset + index * REGION_STRIDE + word] = 0.0;
    }
  }`}),i(e,{...x,id:`${y}-sum-merges`,variant:`mutual-sum-merges`,bindings:[E,A(`read`),{name:`state`,view:m,type:`u32`,access:`read_write`}],declarations:`const ROW_COUNT: u32 = ${a}u;
const WORKGROUP_SIZE: u32 = ${k}u;
var<workgroup> sharedCost: array<f32, ${k}>;
var<workgroup> sharedCount: array<u32, ${k}>;`,body:`let lane = localInvocationIndex;
  var cost = 0.0;
  var count = 0u;
  for (var region = lane; region < ROW_COUNT; region += WORKGROUP_SIZE) {
    if (mergeFlag[mergeFlagOffset + region] != 0u) {
      cost += bitcast<f32>(regionCost[regionCostOffset + region]);
      count += 1u;
    }
  }
  sharedCost[lane] = cost;
  sharedCount[lane] = count;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) {
      sharedCost[lane] += sharedCost[lane + stride];
      sharedCount[lane] += sharedCount[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    state[stateOffset + 3u] = sharedCount[0];
    state[stateOffset + 4u] = bitcast<u32>(sharedCost[0]);
  }`}),i(e,{operation:`GPUWardRegions`,id:`${y}-finish`,variant:`mutual-finish`,invocationCount:1,bindings:[{name:`state`,view:m,type:`u32`,access:`read_write`},{name:`objective`,view:s.objective,type:`f32`,access:`read_write`},{name:`status`,view:s.status,type:`u32`,access:`read_write`},h.binding],declarations:`const TARGET_REGION_COUNT: u32 = ${r.targetRegionCount}u;`,body:`if (dispatch[dispatchOffset] == 0u) { return; }
  let merged = state[stateOffset + 3u];
  objective[objectiveOffset] += bitcast<f32>(state[stateOffset + 4u]);
  state[stateOffset] -= merged;
  state[stateOffset + 1u] += 1u;
  status[statusOffset] += merged;
  ${M?`status[statusOffset + ${N}u] = state[stateOffset + 1u];`:``}
  // No reciprocal pair means no adjacent regions remain (the cheapest pair is always reciprocal).
  if (merged == 0u) { state[stateOffset + 2u] = 1u; }
  if (merged == 0u || state[stateOffset] <= TARGET_REGION_COUNT || status[statusOffset + 3u] != 0u) {
    ${h.disarmWGSL}
  }`})]}createMutualNearestFinalizeNode(e,t){let{id:n,props:r}=this;return i(e,{id:`${n}-finalize`,operation:`GPUWardRegions`,variant:`mutual-finalize`,invocationCount:1,bindings:[{name:`state`,view:t,type:`u32`,access:`read`},{name:`status`,view:r.output.status,type:`u32`,access:`read_write`}],declarations:`const TARGET_REGION_COUNT: u32 = ${r.targetRegionCount}u;
override MAXIMUM_ROUNDS: u32;`,constants:{MAXIMUM_ROUNDS:r.maximumIterations},body:`let remaining = state[stateOffset];
  let valid = status[statusOffset + 3u] == 0u;
  status[statusOffset + 1u] = select(0u, 1u, remaining == TARGET_REGION_COUNT && valid);
  status[statusOffset + 2u] = select(0u, 1u, remaining > TARGET_REGION_COUNT && valid &&
    state[stateOffset + 1u] == MAXIMUM_ROUNDS && state[stateOffset + 2u] == 0u);`})}};export{M as n,j as r,P as t};