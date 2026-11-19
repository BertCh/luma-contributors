import{a as e,o as t,s as n}from"./network-view-utils-B1U8oNKx.js";import{r}from"./kernel-C3tfQOn4.js";import{t as i}from"./masked-weak-components-j3xMrDf7.js";import{o as a,t as o}from"./wgsl-kernel-nodes-DagJklH2.js";import{n as s,r as c}from"./load-balanced-for-each-GB0Xy_f0.js";import{A as l,E as u,f as d,k as f}from"./gpu-contributor-utils-BIwLQON5.js";import{t as p}from"./network-workgroup-counters-BG1Cyc9s.js";import{t as m}from"./define-run-B-brkCe4.js";import{n as h,o as g,s as _}from"./pick-Bo3t3hjv.js";import{c as v,n as y,o as b,t as x}from"./text-DYSgVV-n.js";import{o as S}from"./display-Bu20g6o5.js";import{_ as C,i as w,p as T,r as E}from"./roads-DzovamHj.js";import{fa as D}from"./entry-preview-D70TdSbK.js";var O={liveNodeCount:0,liveEdgeCount:1,liveSlotCount:2,componentCount:3,largestComponentSize:4,componentsConverged:5,isolatedNodeCount:6,maximumOutDegree:7,maximumInDegree:8,maximumTotalDegree:9,modularity:10,modularityValid:11,intraCommunitySlotCount:12,selfLoopSlotCount:13};function k(e=32){return 16+3*e}var A=256,j=1024,M=class{id;props;nodeCount;degreeBinCount;constructor(t){this.id=t.id??`network-statistics`,this.props=t;let{id:r}=this;this.nodeCount=l(r,t.topology).nodeCount;let{nodeCount:i}=this;if(this.degreeBinCount=t.degreeBinCount??32,d(r,`degreeBinCount`,this.degreeBinCount,1,4096),t.degreeBinning&&t.degreeBinning!==`linear`&&t.degreeBinning!==`log2`)throw Error(`${r} degreeBinning must be 'linear' or 'log2'`);d(r,`maximumComponentIterations`,t.maximumComponentIterations??32,1,1024);for(let[e,n,a]of[[`nodeMask`,t.nodeMask,i],[`communities`,t.communities,i],[`edgeMask`,t.edgeMask,t.topology.neighbors.length]])if(n&&(u(n,[`uint32`],`${r} ${e}`),n.length!==a))throw Error(`${r} ${e} must contain one row per ${e===`edgeMask`?`neighbor slot`:`node`}`);if(t.parameters&&(u(t.parameters,[`uint32`],`${r} parameters`),t.parameters.length<2))throw Error(`${r} parameters must contain at least two uint32 rows`);if(u(t.output,[`uint32`],`${r} output`),t.output.length!==k(this.degreeBinCount))throw Error(`${r} output must contain exactly ${k(this.degreeBinCount)} rows`);n(r,t.output),e(r,[t.output],[...f(t.topology),t.nodeMask,t.edgeMask,t.communities,t.parameters])}getCommandNodes(e){let{id:n,props:l,nodeCount:u,degreeBinCount:d}=this;t(n,e,[...f(l.topology),l.nodeMask,l.edgeMask,l.communities,l.parameters,l.output],{maskedNeighbors:l.topology.neighbors.length*4,nodeColumns:u*4,componentChanged:4,segmentCarries:c(u,l.topology.neighbors.length)*8,finishPartials:Math.ceil(u/A)*12});let m=!!l.directed,h=!m,g=!!l.communities,_=O,v=`GPUNetworkStatistics`,{offsets:y,neighbors:b}=l.topology,x=r(e,`${n}-masked-neighbors`,`uint32`,b.length),S=r(e,`${n}-component-labels`,`uint32`,u),C=r(e,`${n}-component-converged`,`uint32`,1),w=t=>r(e,`${n}-${t}`,`uint32`,u),T=w(`out-degree`),E=m?w(`in-degree`):void 0,D=w(`component-size`),k=g?w(`community-out-sum`):void 0,M=g&&m?w(`community-in-sum`):void 0,{output:P}=l,F=[];l.nodeMask&&F.push({name:`vertexMask`,view:l.nodeMask,type:`u32`,access:`read`});let I=e=>l.nodeMask?`vertexMask[vertexMaskOffset + ${e}] != 0u`:`true`,L=[],R=(t,r)=>{r&&L.push(o(e,{id:`${n}-${t}`,operation:v,view:r,type:`u32`,value:`0u`}))};R(`zero-output`,P),R(`zero-in-degree`,E),R(`zero-component-size`,D),R(`zero-out-sum`,k),R(`zero-in-sum`,M);let z=r(e,`${n}-self-loop-counts`,`uint32`,u),B=[{name:`neighbors`,view:b,type:`u32`,access:`read`},...F,{name:`maskedOut`,view:x,type:`u32`,access:`read_write`}];l.edgeMask&&B.push({name:`edgeMask`,view:l.edgeMask,type:`u32`,access:`read`}),E&&B.push({name:`inDegree`,view:E,type:`atomic<u32>`,access:`read_write`}),L.push(...s(e,{id:`${n}-slots`,operation:v,variant:`slots`,offsets:y,itemCapacity:b.length,accumulatorType:`u32`,output:T,bindings:B,declarations:`const NODE_COUNT: u32 = ${u}u;
fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> u32 {
  let neighbor = neighbors[neighborsOffset + item];
  var slotLive = ${I(`row`)} && neighbor < NODE_COUNT;
  ${l.nodeMask?`if (slotLive) { slotLive = vertexMask[vertexMaskOffset + neighbor] != 0u; }`:``}
  ${l.edgeMask?`if (slotLive) { slotLive = edgeMask[edgeMaskOffset + item] != 0u; }`:``}
  maskedOut[maskedOutOffset + item] = select(0xffffffffu, neighbor, slotLive);
  if (!slotLive) {
    return 0u;
  }
  ${E?`atomicAdd(&inDegree[inDegreeOffset + neighbor], 1u);`:``}
  return ${h?`select(1u, 2u, neighbor == row)`:`1u`};
}`}),...s(e,{id:`${n}-self-loops`,operation:v,variant:`self-loops`,offsets:y,itemCapacity:b.length,accumulatorType:`u32`,output:z,bindings:[{name:`maskedIn`,view:x,type:`u32`,access:`read`}],declarations:`fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> u32 {
  return select(0u, 1u, maskedIn[maskedInOffset + item] == row);
}`}));let V=p({counters:[{target:`summary`,word:_.liveSlotCount},{target:`summary`,word:_.selfLoopSlotCount}],rowCount:`NODE_COUNT`,rowSource:`let selfCount = selfLoopCounts[selfLoopCountsOffset + index];
  addCounter(0u, outDegree[outDegreeOffset + index]${h?` - selfCount`:``});
  addCounter(1u, selfCount);`});L.push(a(e,{id:`${n}-slots-finalize`,operation:v,variant:`slots-finalize`,bindings:[{name:`selfLoopCounts`,view:z,type:`u32`,access:`read`},{name:`outDegree`,view:T,type:`u32`,access:`read`},{name:`summary`,view:P,type:`atomic<u32>`,access:`read_write`}],invocationCount:u,workgroupSize:256,guardIndex:!1,declarations:`const NODE_COUNT: u32 = ${u}u;
${V.declarations}`,body:V.body})),L.push(...i(e,{id:`${n}-components`,operation:v,nodeCount:u,offsets:y,neighbors:x,labels:S,converged:C,iterations:l.maximumComponentIterations??32})),L.push(a(e,{id:`${n}-component-sizes`,operation:v,variant:`component-sizes`,bindings:[...F,{name:`labels`,view:S,type:`u32`,access:`read`},{name:`componentSize`,view:D,type:`atomic<u32>`,access:`read_write`}],invocationCount:u,body:`if (!(${I(`index`)})) {
    return;
  }
  atomicAdd(&componentSize[componentSizeOffset + labels[labelsOffset + index]], 1u);`}));let H=l.parameters?`max(parameters[parametersOffset + 1u], 1u)`:`1u`,U=l.degreeBinning===`log2`?`fn getBin(degree: u32) -> u32 {
  if (degree == 0u) {
    return 0u;
  }
  return min(1u + firstLeadingBit(degree), BIN_COUNT - 1u);
}`:`fn getBin(degree: u32) -> u32 {
  return min(degree / ${H}, BIN_COUNT - 1u);
}`,W=[...F,{name:`outDegree`,view:T,type:`u32`,access:`read`}];E&&W.push({name:`inDegree`,view:E,type:`u32`,access:`read`}),l.communities&&W.push({name:`communities`,view:l.communities,type:`u32`,access:`read`}),k&&W.push({name:`outSum`,view:k,type:`atomic<u32>`,access:`read_write`}),M&&W.push({name:`inSum`,view:M,type:`atomic<u32>`,access:`read_write`}),l.parameters&&l.degreeBinning!==`log2`&&W.push({name:`parameters`,view:l.parameters,type:`u32`,access:`read`}),W.push({name:`summary`,view:P,type:`atomic<u32>`,access:`read_write`});let G=3*d<=j,K=`const NODE_COUNT: u32 = ${u}u;
const BIN_COUNT: u32 = ${d}u;
const HISTOGRAM_BASE: u32 = 16u;
${U}`,q=g?`let label = communities[communitiesOffset + index];
  if (label >= NODE_COUNT) {
    atomicAdd(&summary[summaryOffset + ${_.modularityValid}u], 1u);
  } else {
    atomicAdd(&outSum[outSumOffset + label], outValue);
    ${M?`atomicAdd(&inSum[inSumOffset + label], inValue);`:``}
  }`:``;if(L.push(a(e,{id:`${n}-degrees`,operation:v,variant:`degrees`,bindings:W,invocationCount:u,...G?{workgroupSize:A,guardIndex:!1,declarations:`${K}
var<workgroup> localHistogram: array<atomic<u32>, ${3*d}>;
var<workgroup> localMaxima: array<atomic<u32>, 3>;
var<workgroup> localCounts: array<atomic<u32>, 2>;`,body:`for (var bin = localInvocationIndex; bin < ${3*d}u; bin += ${A}u) {
    atomicStore(&localHistogram[bin], 0u);
  }
  if (localInvocationIndex < 3u) {
    atomicStore(&localMaxima[localInvocationIndex], 0u);
  }
  if (localInvocationIndex < 2u) {
    atomicStore(&localCounts[localInvocationIndex], 0u);
  }
  workgroupBarrier();
  if (index < NODE_COUNT && ${I(`index`)}) {
    let outValue = outDegree[outDegreeOffset + index];
    let inValue = ${E?`inDegree[inDegreeOffset + index]`:`outValue`};
    let totalValue = ${m?`outValue + inValue`:`outValue`};
    atomicAdd(&localCounts[1], 1u);
    if (totalValue == 0u) {
      atomicAdd(&localCounts[0], 1u);
    }
    atomicMax(&localMaxima[0], outValue);
    atomicMax(&localMaxima[1], inValue);
    atomicMax(&localMaxima[2], totalValue);
    atomicAdd(&localHistogram[getBin(outValue)], 1u);
    atomicAdd(&localHistogram[BIN_COUNT + getBin(inValue)], 1u);
    atomicAdd(&localHistogram[2u * BIN_COUNT + getBin(totalValue)], 1u);
    ${q}
  }
  workgroupBarrier();
  for (var bin = localInvocationIndex; bin < ${3*d}u; bin += ${A}u) {
    let binCount = atomicLoad(&localHistogram[bin]);
    if (binCount > 0u) {
      atomicAdd(&summary[summaryOffset + HISTOGRAM_BASE + bin], binCount);
    }
  }
  if (localInvocationIndex < 3u) {
    let workgroupMaximum = atomicLoad(&localMaxima[localInvocationIndex]);
    if (workgroupMaximum > 0u) {
      atomicMax(&summary[summaryOffset + ${_.maximumOutDegree}u + localInvocationIndex], workgroupMaximum);
    }
  }
  if (localInvocationIndex == 3u) {
    let workgroupIsolated = atomicLoad(&localCounts[0]);
    if (workgroupIsolated > 0u) {
      atomicAdd(&summary[summaryOffset + ${_.isolatedNodeCount}u], workgroupIsolated);
    }
    let workgroupLive = atomicLoad(&localCounts[1]);
    if (workgroupLive > 0u) {
      atomicAdd(&summary[summaryOffset + ${_.liveNodeCount}u], workgroupLive);
    }
  }`}:{declarations:K,body:`if (!(${I(`index`)})) {
    return;
  }
  let outValue = outDegree[outDegreeOffset + index];
  let inValue = ${E?`inDegree[inDegreeOffset + index]`:`outValue`};
  let totalValue = ${m?`outValue + inValue`:`outValue`};
  atomicAdd(&summary[summaryOffset + ${_.liveNodeCount}u], 1u);
  if (totalValue == 0u) {
    atomicAdd(&summary[summaryOffset + ${_.isolatedNodeCount}u], 1u);
  }
  atomicMax(&summary[summaryOffset + ${_.maximumOutDegree}u], outValue);
  atomicMax(&summary[summaryOffset + ${_.maximumInDegree}u], inValue);
  atomicMax(&summary[summaryOffset + ${_.maximumTotalDegree}u], totalValue);
  atomicAdd(&summary[summaryOffset + HISTOGRAM_BASE + getBin(outValue)], 1u);
  atomicAdd(&summary[summaryOffset + HISTOGRAM_BASE + BIN_COUNT + getBin(inValue)], 1u);
  atomicAdd(&summary[summaryOffset + HISTOGRAM_BASE + 2u * BIN_COUNT + getBin(totalValue)], 1u);
  ${q}`}})),l.communities){let t=r(e,`${n}-intra-sums`,`uint32`,u);L.push(...s(e,{id:`${n}-intra-community`,operation:v,variant:`intra-community`,offsets:y,itemCapacity:b.length,accumulatorType:`u32`,output:t,bindings:[{name:`maskedIn`,view:x,type:`u32`,access:`read`},{name:`communities`,view:l.communities,type:`u32`,access:`read`}],declarations:`fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> u32 {
  let neighbor = maskedIn[maskedInOffset + item];
  return select(
    0u,
    1u,
    neighbor != 0xffffffffu &&
      communities[communitiesOffset + neighbor] == communities[communitiesOffset + row]
  );
}`}));let i=p({counters:[{target:`summary`,word:_.intraCommunitySlotCount}],rowCount:`NODE_COUNT`,rowSource:`addCounter(0u, intraSums[intraSumsOffset + index]);`});L.push(a(e,{id:`${n}-intra-community-total`,operation:v,variant:`intra-community-total`,bindings:[{name:`intraSums`,view:t,type:`u32`,access:`read`},{name:`summary`,view:P,type:`atomic<u32>`,access:`read_write`}],invocationCount:u,workgroupSize:256,guardIndex:!1,declarations:`const NODE_COUNT: u32 = ${u}u;
${i.declarations}`,body:i.body}))}let J=Math.ceil(u/A),Y=r(e,`${n}-finish-partials`,`uint32`,3*J),X=[{name:`componentSize`,view:D,type:`u32`,access:`read`}];k&&X.push({name:`outSum`,view:k,type:`u32`,access:`read`}),M&&X.push({name:`inSum`,view:M,type:`u32`,access:`read`}),X.push({name:`summaryIn`,view:P,type:`u32`,access:`read`},{name:`partials`,view:Y,type:`u32`,access:`read_write`});let ee=M?`(f32(outSum[outSumOffset + index]) / totalSlots) * (f32(inSum[inSumOffset + index]) / totalSlots)`:`(f32(outSum[outSumOffset + index]) / totalSlots) * (f32(outSum[outSumOffset + index]) / totalSlots)`,Z=`workgroupBarrier();
  for (var stride = ${A/2}u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      sharedCount[localInvocationIndex] += sharedCount[localInvocationIndex + stride];
      sharedLargest[localInvocationIndex] =
        max(sharedLargest[localInvocationIndex], sharedLargest[localInvocationIndex + stride]);
      sharedSum[localInvocationIndex] += sharedSum[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }`,Q=`var<workgroup> sharedCount: array<u32, ${A}>;
var<workgroup> sharedLargest: array<u32, ${A}>;
var<workgroup> sharedSum: array<f32, ${A}>;`;L.push(a(e,{id:`${n}-finish-partials`,operation:v,variant:`finish-partials`,bindings:X,invocationCount:J*A,workgroupSize:A,guardIndex:!1,declarations:`const NODE_COUNT: u32 = ${u}u;
${Q}`,body:`let liveSlots = summaryIn[summaryInOffset + ${_.liveSlotCount}u];
  let selfLoops = summaryIn[summaryInOffset + ${_.selfLoopSlotCount}u];
  let totalSlots = f32(liveSlots${h?` + selfLoops`:``});
  var componentCount = 0u;
  var largest = 0u;
  var termSum = 0.0;
  if (index < NODE_COUNT) {
    let size = componentSize[componentSizeOffset + index];
    if (size > 0u) {
      componentCount = 1u;
      largest = size;
    }
    ${g?`if (liveSlots > 0u) { termSum = ${ee}; }`:``}
  }
  sharedCount[localInvocationIndex] = componentCount;
  sharedLargest[localInvocationIndex] = largest;
  sharedSum[localInvocationIndex] = termSum;
  ${Z}
  if (localInvocationIndex == 0u) {
    let partial = 3u * (index / ${A}u);
    partials[partialsOffset + partial] = sharedCount[0];
    partials[partialsOffset + partial + 1u] = sharedLargest[0];
    partials[partialsOffset + partial + 2u] = bitcast<u32>(sharedSum[0]);
  }`}));let $=[{name:`partials`,view:Y,type:`u32`,access:`read`},{name:`converged`,view:C,type:`u32`,access:`read`}];return l.parameters&&$.push({name:`parameters`,view:l.parameters,type:`u32`,access:`read`}),$.push({name:`summary`,view:P,type:`u32`,access:`read_write`}),L.push(a(e,{id:`${n}-finish`,operation:v,variant:`finish`,bindings:$,invocationCount:A,workgroupSize:A,guardIndex:!1,declarations:`const PARTIAL_COUNT: u32 = ${J}u;
${Q}`,body:`let liveSlots = summary[summaryOffset + ${_.liveSlotCount}u];
  let selfLoops = summary[summaryOffset + ${_.selfLoopSlotCount}u];
  let totalSlots = f32(liveSlots${h?` + selfLoops`:``});
  var componentCount = 0u;
  var largest = 0u;
  var termSum = 0.0;
  for (var partial = localInvocationIndex; partial < PARTIAL_COUNT; partial += ${A}u) {
    componentCount += partials[partialsOffset + 3u * partial];
    largest = max(largest, partials[partialsOffset + 3u * partial + 1u]);
    termSum += bitcast<f32>(partials[partialsOffset + 3u * partial + 2u]);
  }
  sharedCount[localInvocationIndex] = componentCount;
  sharedLargest[localInvocationIndex] = largest;
  sharedSum[localInvocationIndex] = termSum;
  ${Z}
  if (localInvocationIndex == 0u) {
    let invalidLabels = summary[summaryOffset + ${_.modularityValid}u];
    let intra = summary[summaryOffset + ${_.intraCommunitySlotCount}u]${h?` + selfLoops`:``};
    summary[summaryOffset + ${_.componentCount}u] = sharedCount[0];
    summary[summaryOffset + ${_.largestComponentSize}u] = sharedLargest[0];
    summary[summaryOffset + ${_.componentsConverged}u] = select(0u, 1u, converged[convergedOffset] != 0u);
    summary[summaryOffset + ${_.liveEdgeCount}u] = ${m?`liveSlots`:`(liveSlots - selfLoops) / 2u + selfLoops`};
    ${N(m,g,!!l.parameters)}
    summary[summaryOffset + 14u] = 0u;
    summary[summaryOffset + 15u] = 0u;
  }`})),L}};function N(e,t,n){let r=O;return t?`let modularityValid = invalidLabels == 0u && liveSlots > 0u;
    var modularity = 0.0;
    if (modularityValid) {
      modularity = f32(intra) / totalSlots - ${n?`bitcast<f32>(parameters[parametersOffset])`:`1.0`} * sharedSum[0];
    }
    summary[summaryOffset + ${r.modularity}u] = bitcast<u32>(modularity);
    summary[summaryOffset + ${r.modularityValid}u] = select(0u, 1u, modularityValid);`:`summary[summaryOffset + ${r.modularity}u] = 0u;
    summary[summaryOffset + ${r.modularityValid}u] = 0u;`}var P=12,F=[.06,.05,.94,.3],I=.6,L=.38;function R(e,t){return t?e===0?0:Math.min(1+Math.floor(Math.log2(e)),11):Math.min(e,11)}function z(e,t){let n=t?e===0?0:2**(e-1):e;return e===11?`${n}+`:String(n)}var B=m({parameters:D,caption:`Histogram of node degree from the packed network summary, linked to the intersections.`,createInputs:({random:e,world:t})=>({base:w(e,t,14),world:t,view:null,bin:3}),build(e,t,n){let{base:r,world:i}=t,a=E(r,n.closures),{topology:o}=C(e,a),s=e.allocate(`summary`,`uint32`,k(P));e.graph.add(new M({topology:o,degreeBinCount:P,degreeBinning:n.degreeBinning===0?`linear`:`log2`,output:s.view}));let c=n.degreeBinning===1,l=Float32Array.from(a.nodes,(e,t)=>t%2==0?i.width*.4/2+e*I:L+e*I),u=Uint8Array.from({length:a.nodeCount},(e,t)=>R(a.offsets[t+1]-a.offsets[t],c)),d=Array(P).fill(0);for(let e of u)d[e]++;return t.view={bins:u,counts:d,index:h(l,a.nodeCount)},{outputs:{nodes:e.upload(`display-nodes`,`float32x2`,l),edges:e.upload(`road-edges`,`uint32x2`,a.edgePairs),selection:e.allocate(`selection`,`uint32`,a.nodeCount),hoverBars:e.allocate(`hover-bars`,`float32`,P),totalDegree:S(e,{id:`out-degree`,source:s,offset:16,length:P})}}},rebuildOn:[`degreeBinning`,`closures`],interaction:{kind:`probe`,hint:`hover a bar`,rest:({world:e,toUnit:t})=>t([_(F,e,3.5/P),.17]),resolve(e,{inputs:t,outputs:n,world:r}){let i=t.view;if(!i)return{};let a=t.bin,o=g(F,r,e.x,e.y);if(o.isInside||e.y<L)a=Math.min(11,Math.floor(o.u*P));else{let t=i.index.nearest(e.x,e.y);t>=0&&(a=i.bins[t])}t.bin=a,n.hoverBars.write(Float32Array.from(i.counts,(e,t)=>t===a?e:0)),n.selection.write(Uint32Array.from(i.bins,e=>e===a?0:4294967295));let s=_(F,r,a/P),c=_(F,r,(a+1)/P);return{spotlight:null,glyphs:[{shape:`box`,minimum:[s,F[1]*r.height],maximum:[c,F[3]*r.height]}],readout:`degree class ${a} · ${i.counts[a]} intersections`}}},draw:({inputs:e,outputs:t,theme:n,values:r,world:i})=>{let a=e.view?.counts??[],o=Math.max(1,...a.length>0?a:[1]),s=b(i,F),c=r.degreeBinning===1,l=y({rectangle:s,x:{categories:Array.from({length:P},(e,t)=>z(t,c)),title:c?`node degree (log2 classes)`:`node degree`}}),u=v(l,x({rectangle:s,values:a,range:[0,o],indices:a.flatMap((e,t)=>e>0?[t]:[])}));return[T(t),{layer:`dots`,positions:t.nodes,values:t.selection,ramp:`categorical`,categories:[[...n.categories[1],1]],radius:4.5},{layer:`chart`,kind:`bars`,values:t.totalDegree,ramp:`sequential`,range:[0,o],rectangle:F},{layer:`chart`,kind:`bars`,values:t.hoverBars,color:[...n.ink,1],range:[0,o],rectangle:F},{layer:`text`,...u}]}});export{B as default};