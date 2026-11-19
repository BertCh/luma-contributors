import{t as e}from"./spatial-output-validation-nw5lOmVL.js";import{i as t,r as n}from"./weights-construction-snapshot-CikLt8kl.js";import{i as r,t as i}from"./spatial-weights-loss-u59PcHkg.js";import{r as a}from"./kernel-C3tfQOn4.js";import{t as o}from"./gpu-compaction-B6IzcF0P.js";import{t as s}from"./gpu-uint32-gather-C7ioZBaz.js";import{n as c,o as l}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as u,f as d,m as f,p,v as m,x as h,y as g}from"./gpu-contributor-utils-BIwLQON5.js";import{c as _}from"./spatial-weights-CqWysstN.js";import{l as v}from"./wgsl-common-DfyXiYB0.js";import{a as y}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as b}from"./regionalization-inputs-CyTCPBba.js";var x=`
// Slot of row \`row\` whose neighbor is \`needle\`, or NONE. Rows are strictly ascending.
fn findSlot(row: u32, needle: u32) -> u32 {
  let rowEnd = offsets[offsetsOffset + row + 1u];
  var low = offsets[offsetsOffset + row];
  var high = rowEnd;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (neighbors[neighborsOffset + middle] < needle) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < rowEnd && neighbors[neighborsOffset + low] == needle) {
    return low;
  }
  return NONE;
}

// First row whose slot range ends after \`slot\` (the row that owns the slot).
fn getRowOfSlot(slot: u32) -> u32 {
  var low = 0u;
  var high = ROWS - 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (offsets[offsetsOffset + middle + 1u] > slot) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  return low;
}
`,S=`const NONE: u32 = 0xffffffffu;`,C=`GPUSpatialWeightsMinimumSpanningTree`,w=128,T=7,E={invalidCount:0,candidateOverflow:1,length:2},D=class{id;estimatedCommandNodeCount;props;output;constructor(i){let o=n(i,[`id`,`weights`,`values`,`columnCount`,`standardize`,`output`]),s=n(o.output,[`treeEdgeFlags`,`componentLabels`,`standardizedValues`,`edges`,`status`,`edgeEndpoints`,`edgeDissimilarities`]),c=s.edges?n(s.edges,[`ids`,`status`]):void 0;i=Object.freeze({...o,weights:t(o.weights),output:Object.freeze({...s,...c?{edges:Object.freeze({...c,status:n(c.status,[`count`,`requiredCount`,`overflow`,`candidateOverflow`])})}:{}})});let l=i.id??`minimum-spanning-tree`;this.id=l,this.props=i;let m=_(l,i.weights),h=i.columnCount??1,{output:v}=i;if(d(l,`columnCount`,h,1,32),u(i.values,[`float32`],`${l} values`),i.values.length!==m*h)throw Error(`${l} values length must equal rows * columnCount`);if(u(v.treeEdgeFlags,[`uint32`],`${l} output.treeEdgeFlags`),v.treeEdgeFlags.length!==i.weights.neighbors.length)throw Error(`${l} output.treeEdgeFlags length must equal the weights slot capacity`);if(u(v.componentLabels,[`uint32`],`${l} output.componentLabels`),v.componentLabels.length!==m)throw Error(`${l} output.componentLabels length must equal the weights row count`);if(v.standardizedValues){if(u(v.standardizedValues,[`float32`],`${l} output.standardizedValues`),v.standardizedValues.length!==m*h)throw Error(`${l} output.standardizedValues length must equal rows * columnCount`);if(i.standardize===!1)throw Error(`${l} output.standardizedValues requires standardize`)}if(v.status&&(u(v.status,[`uint32`],`${l} output.status`),v.status.length<E.length))throw Error(`${l} output.status must hold ${E.length} word`);if(v.edges)f(l,v.edges);else if(v.edgeEndpoints||v.edgeDissimilarities)throw Error(`${l} edge details require output.edges`);let y=v.edges?.ids.length??0;if(v.edgeEndpoints&&(u(v.edgeEndpoints,[`uint32`],`${l} output.edgeEndpoints`),v.edgeEndpoints.length<2*y))throw Error(`${l} output.edgeEndpoints must hold two entries per edge slot`);if(v.edgeDissimilarities&&(u(v.edgeDissimilarities,[`float32`],`${l} output.edgeDissimilarities`),v.edgeDissimilarities.length<y))throw Error(`${l} output.edgeDissimilarities must hold one entry per edge slot`);let b=Math.max(1,Math.ceil(Math.log2(m)));this.estimatedCommandNodeCount=w+b*(T+b),p(l,this.estimatedCommandNodeCount),e(l,{treeEdgeFlags:v.treeEdgeFlags,componentLabels:v.componentLabels,standardizedValues:v.standardizedValues,status:v.status,edgeIds:v.edges?.ids,count:v.edges?.status.count,overflow:v.edges?.status.overflow,requiredCount:v.edges?.status.requiredCount,candidateOverflow:v.edges?.status.candidateOverflow,edgeEndpoints:v.edgeEndpoints,edgeDissimilarities:v.edgeDissimilarities},[[v.edges?.ids.length?v.edges.ids:void 0,v.edges?.status.count,v.edges?.status.overflow,v.edges?.status.requiredCount,v.edges?.status.candidateOverflow],...v.edgeEndpoints||v.edgeDissimilarities?[{read:[v.edges?.ids,v.edges?.status.count],write:[v.edgeEndpoints,v.edgeDissimilarities]}]:[]]),g(l,[v.treeEdgeFlags,v.componentLabels,v.standardizedValues,v.status,v.edges?.ids,v.edges?.status.count,v.edges?.status.overflow,v.edges?.status.requiredCount,v.edges?.status.candidateOverflow,v.edgeEndpoints,v.edgeDissimilarities],[i.values,i.weights.offsets,i.weights.neighbors,...r(l,[i.weights])]);let x=v.edges&&(i.weights.status?.overflow||i.weights.status?.candidateOverflow)?v.edges.status.candidateOverflow??a(v.treeEdgeFlags.buffer.graph,`${l}-${v.treeEdgeFlags.buffer.id}-${v.treeEdgeFlags.byteOffset}-candidate-loss`,`uint32`,1):v.edges?.status.candidateOverflow;this.output=Object.freeze({...v,...v.edges?{edges:Object.freeze({...v.edges,status:Object.freeze({...v.edges.status,candidateOverflow:x})})}:{}}),this.props=Object.freeze({...i,output:this.output})}getCommandNodes(e){let{id:t,props:n}=this,{weights:u,output:d}=n,{treeEdgeFlags:f,componentLabels:p}=d;h(t,e,[n.values,f,p,d.standardizedValues,d.status,d.edges?.ids,d.edges?.status.count,d.edges?.status.overflow,d.edges?.status.requiredCount,d.edges?.status.candidateOverflow,d.edgeEndpoints,d.edgeDissimilarities,u.offsets,u.neighbors,...r(t,[u])]);let g=u.offsets.length-1,_=u.neighbors.length,w=n.columnCount??1;y(e,t,{values:n.values,offsets:u.offsets,neighbors:u.neighbors,treeEdgeFlags:d.treeEdgeFlags,componentLabels:d.componentLabels,standardizedValues:d.standardizedValues,status:d.status,edgeIds:d.edges?.ids,count:d.edges?.status.count,overflow:d.edges?.status.overflow,requiredCount:d.edges?.status.requiredCount,edgeEndpoints:d.edgeEndpoints,edgeDissimilarities:d.edgeDissimilarities,inputOverflow:u.status?.overflow,inputCandidateOverflow:u.status?.candidateOverflow,candidateOverflow:d.edges?.status.candidateOverflow});for(let[n,r]of[[`standardized table`,g*w*4],[`row scratch`,Math.max(1,g)*4],[`slot costs and compaction`,Math.max(1,_)*4]])m(e.device,t,n,r,`reduce row or neighbor slot counts`);let T=n.standardize!==!1,D=[{name:`offsets`,view:u.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:u.neighbors,type:`u32`,access:`read`}],O=`const ROWS: u32 = ${g}u;
const COLUMNS: u32 = ${w}u;
${v}
${S}
${x}`,k=i(e,t,[u]),A=[...k.nodes],j=n.values;T&&(j=d.standardizedValues??a(e,`${t}-table`,`float32`,g*w),A.push(b(e,{id:`${t}-standardize`,operation:C,values:n.values,scaled:j,rowCount:g,columnCount:w}))),d.status&&A.push(l(e,{id:`${t}-clear-invalid`,operation:C,variant:`clear-invalid`,bindings:[{name:`status`,view:d.status,type:`u32`,access:`read_write`},...k.view?[{name:`loss`,view:k.view,type:`u32`,access:`read`}]:[]],invocationCount:1,body:`status[statusOffset] = 0u;
  status[statusOffset + ${E.candidateOverflow}u] = ${k.view?`loss[lossOffset]`:`0u`};`}),l(e,{id:`${t}-count-invalid`,operation:C,variant:`count-invalid`,bindings:[{name:`values`,view:n.values,type:`f32`,access:`read`},{name:`status`,view:d.status,type:`atomic<u32>`,access:`read_write`}],invocationCount:g,declarations:`const COLUMNS: u32 = ${w}u;
${v}`,body:`var invalid = 0u;
  for (var column = 0u; column < COLUMNS; column++) {
    invalid += select(1u, 0u, isFiniteFloat(values[valuesOffset + index * COLUMNS + column]));
  }
  if (invalid > 0u) { atomicAdd(&status[statusOffset], invalid); }`}));let M=a(e,`${t}-slot-cost`,`float32`,_),N=a(e,`${t}-slot-edge`,`uint32`,_),P=a(e,`${t}-component`,`uint32`,g),F=a(e,`${t}-best-cost`,`uint32`,g),I=a(e,`${t}-best-edge`,`uint32`,g),L=a(e,`${t}-parent-a`,`uint32`,g),R=a(e,`${t}-parent-b`,`uint32`,g),z=a(e,`${t}-minimum-label`,`uint32`,g);A.push(l(e,{id:`${t}-clear-flags`,operation:C,variant:`clear`,bindings:[{name:`flags`,view:f,type:`u32`,access:`read_write`}],invocationCount:_,body:`flags[flagsOffset + index] = 0u;`}),l(e,{id:`${t}-seed-components`,operation:C,variant:`seed`,bindings:[{name:`component`,view:P,type:`u32`,access:`read_write`},{name:`bestCost`,view:F,type:`u32`,access:`read_write`},{name:`bestEdge`,view:I,type:`u32`,access:`read_write`}],invocationCount:g,body:`component[componentOffset + index] = index;
  bestCost[bestCostOffset + index] = 0xffffffffu;
  bestEdge[bestEdgeOffset + index] = 0xffffffffu;`}),l(e,{id:`${t}-slot-costs`,operation:C,variant:`costs`,bindings:[...D,{name:`table`,view:j,type:`f32`,access:`read`},{name:`slotCost`,view:M,type:`f32`,access:`read_write`},{name:`slotEdge`,view:N,type:`u32`,access:`read_write`}],invocationCount:g,declarations:O,body:`for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    let low = min(index, neighbor);
    let high = max(index, neighbor);
    var cost = 0.0;
    var finite = true;
    for (var column = 0u; column < COLUMNS; column++) {
      let lowValue = table[tableOffset + low * COLUMNS + column];
      let highValue = table[tableOffset + high * COLUMNS + column];
      finite = finite && isFiniteFloat(lowValue) && isFiniteFloat(highValue);
      let delta = lowValue - highValue;
      cost += delta * delta;
    }
    // A link touching a non-finite entry costs +Infinity: chosen last, never a NaN bit pattern. The
    // runtime operand keeps the infinity out of constant evaluation.
    slotCost[slotCostOffset + slot] = select(bitcast<f32>(0x7f800000u | (index & 0u)), cost, finite);
    var edge = slot;
    if (neighbor < index) {
      let twin = findSlot(neighbor, index);
      if (twin != NONE) {
        edge = twin;
      }
    }
    slotEdge[slotEdgeOffset + slot] = edge;
  }`}));let B=Math.max(1,Math.ceil(Math.log2(g)));for(let n=0;n<B;n++){let r=`${t}-round-${n}`;A.push(l(e,{id:`${r}-cheapest-cost`,operation:C,variant:`cheapest-cost`,bindings:[...D,{name:`component`,view:P,type:`u32`,access:`read`},{name:`slotCost`,view:M,type:`f32`,access:`read`},{name:`bestCost`,view:F,type:`atomic<u32>`,access:`read_write`}],invocationCount:g,declarations:`const ROWS: u32 = ${g}u;`,body:`let own = component[componentOffset + index];
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    if (component[componentOffset + neighbors[neighborsOffset + slot]] != own) {
      atomicMin(&bestCost[bestCostOffset + own], bitcast<u32>(slotCost[slotCostOffset + slot]));
    }
  }`}),l(e,{id:`${r}-cheapest-edge`,operation:C,variant:`cheapest-edge`,bindings:[...D,{name:`component`,view:P,type:`u32`,access:`read`},{name:`slotCost`,view:M,type:`f32`,access:`read`},{name:`slotEdge`,view:N,type:`u32`,access:`read`},{name:`bestCost`,view:F,type:`u32`,access:`read`},{name:`bestEdge`,view:I,type:`atomic<u32>`,access:`read_write`}],invocationCount:g,declarations:`const ROWS: u32 = ${g}u;`,body:`let own = component[componentOffset + index];
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    if (component[componentOffset + neighbors[neighborsOffset + slot]] != own &&
        bitcast<u32>(slotCost[slotCostOffset + slot]) == bestCost[bestCostOffset + own]) {
      atomicMin(&bestEdge[bestEdgeOffset + own], slotEdge[slotEdgeOffset + slot]);
    }
  }`}),l(e,{id:`${r}-hook`,operation:C,variant:`hook`,bindings:[...D,{name:`component`,view:P,type:`u32`,access:`read`},{name:`bestEdge`,view:I,type:`u32`,access:`read`},{name:`parentOut`,view:R,type:`u32`,access:`read_write`},{name:`flags`,view:f,type:`atomic<u32>`,access:`read_write`}],invocationCount:g,declarations:O,body:`var parent = index;
  let edge = bestEdge[bestEdgeOffset + index];
  if (component[componentOffset + index] == index && edge != NONE) {
    let first = component[componentOffset + getRowOfSlot(edge)];
    let second = component[componentOffset + neighbors[neighborsOffset + edge]];
    parent = select(first, second, first == index);
    let isMutual = bestEdge[bestEdgeOffset + parent] == edge;
    parent = select(parent, index, isMutual && index < parent);
    atomicStore(&flags[flagsOffset + edge], 1u);
  }
  parentOut[parentOutOffset + index] = parent;`}));let i=Math.max(1,Math.ceil(Math.log2(Math.max(2,Math.ceil(g/2**n)))))+1,a=R,o=L;for(let t=0;t<i-1;t++)A.push(l(e,{id:`${r}-jump-${t}`,operation:C,variant:`jump`,bindings:[{name:`parentIn`,view:a,type:`u32`,access:`read`},{name:`parentOut`,view:o,type:`u32`,access:`read_write`}],invocationCount:g,body:`parentOut[parentOutOffset + index] =
    parentIn[parentInOffset + parentIn[parentInOffset + index]];`})),[a,o]=[o,a];let s=n===B-1;A.push(l(e,{id:`${r}-relabel`,operation:C,variant:`relabel`,bindings:[{name:`root`,view:a,type:`u32`,access:`read`},{name:`component`,view:P,type:`u32`,access:`read_write`},...s?[{name:`minimumLabel`,view:z,type:`u32`,access:`read_write`}]:[{name:`bestCost`,view:F,type:`u32`,access:`read_write`},{name:`bestEdge`,view:I,type:`u32`,access:`read_write`}]],invocationCount:g,body:`let label = component[componentOffset + index];
  component[componentOffset + index] = root[rootOffset + root[rootOffset + label]];
  ${s?`minimumLabel[minimumLabelOffset + index] = 0xffffffffu;`:`bestCost[bestCostOffset + index] = 0xffffffffu;
  bestEdge[bestEdgeOffset + index] = 0xffffffffu;`}`}))}if(A.push(l(e,{id:`${t}-label-minimum`,operation:C,variant:`label-minimum`,bindings:[{name:`component`,view:P,type:`u32`,access:`read`},{name:`minimumLabel`,view:z,type:`atomic<u32>`,access:`read_write`}],invocationCount:g,body:`atomicMin(&minimumLabel[minimumLabelOffset + component[componentOffset + index]], index);`}),...new s({id:`${t}-label-write`,source:z,indices:P,output:p}).getCommandNodes(e)),k.view)for(let[n,r,i]of[[`tree-flags`,f,`0u`],[`component-labels`,p,`0xffffffffu`]])A.push(l(e,{id:`${t}-incomplete-${n}`,operation:C,invocationCount:r.length,bindings:[{name:`loss`,view:k.view,type:`u32`,access:`read`},{name:`result`,view:r,type:`u32`,access:`read_write`}],body:`if (loss[lossOffset] != 0u) { result[resultOffset + index] = ${i}; }`}));if(d.edges){let{edges:n}=d,r=a(e,`${t}-slot-ids`,`uint32`,_),i=a(e,`${t}-compact-slots`,`uint32`,_),s=a(e,`${t}-edge-total`,`uint32`,1);if(A.push(l(e,{id:`${t}-slot-ids`,operation:C,variant:`slot-ids`,bindings:[{name:`slotIds`,view:r,type:`u32`,access:`read_write`}],invocationCount:_,body:`slotIds[slotIdsOffset + index] = index;`}),...new o({id:`${t}-compaction`,input:r,flags:f,output:i,count:s}).getCommandNodes(e),c(e,{id:`${t}-publish`,operation:C,requiredCount:s,compactIds:i,candidateOverflowSources:k.view?[k.view]:[],output:n})),d.edgeEndpoints||d.edgeDissimilarities){let r=[...D,{name:`edgeIds`,view:n.ids,type:`u32`,access:`read`},{name:`edgeCount`,view:n.status.count,type:`u32`,access:`read`},{name:`slotCost`,view:M,type:`f32`,access:`read`},...d.edgeEndpoints?[{name:`endpoints`,view:d.edgeEndpoints,type:`u32`,access:`read_write`}]:[],...d.edgeDissimilarities?[{name:`dissimilarities`,view:d.edgeDissimilarities,type:`f32`,access:`read_write`}]:[]];A.push(l(e,{id:`${t}-edge-details`,operation:C,variant:`edge-details`,bindings:r,invocationCount:n.ids.length,declarations:O,body:`if (index >= edgeCount[edgeCountOffset]) {
    return;
  }
  let slot = edgeIds[edgeIdsOffset + index];
  ${d.edgeEndpoints?`endpoints[endpointsOffset + 2u * index] = getRowOfSlot(slot);
  endpoints[endpointsOffset + 2u * index + 1u] = neighbors[neighborsOffset + slot];`:``}
  ${d.edgeDissimilarities?`dissimilarities[dissimilaritiesOffset + index] = slotCost[slotCostOffset + slot];`:``}`}))}}return A}};export{S as i,E as n,x as r,D as t};