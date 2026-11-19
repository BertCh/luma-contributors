import{i as e,r as t}from"./cell-dispatch-validation-PIkQyAK_.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-gather-BJIYd708.js";import{t as i}from"./gpu-scan-B4NNTfF9.js";import{t as a}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{n as o}from"./sorted-segment-sums-Dvmm6OdR.js";import{o as s,t as c}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as l,T as u,f as d,v as f,x as p,y as m}from"./gpu-contributor-utils-BIwLQON5.js";var h=`GPUHealpixCellAggregation`,g=2147483647,_=class{id;props;constructor(t){let n=t.id??`healpix-cell-aggregation`;this.id=n,this.props=t;let{output:r}=t;d(n,`order`,t.order,0,29),d(n,`cellCapacity`,t.cellCapacity,1,g),u(n,t.scheme===void 0||[`nested`,`uniq`].includes(t.scheme),`scheme must be 'nested' or 'uniq'`),l(t.cells,[`uint32x2`],`${n} cells`);let i=t.cells.length;d(n,`cells.length`,i,1,g);for(let[e,r]of[[`mask`,t.mask],[`values`,t.values]])r&&(l(r,[e===`mask`?`uint32`:`float32`],`${n} ${e}`),u(n,r.length===i,`${e} length must equal the row count`));l(r.cells,[`uint32x2`],`${n} output.cells`),l(r.counts,[`uint32`],`${n} output.counts`),u(n,r.cells.length===t.cellCapacity&&r.counts.length===t.cellCapacity,`output.cells and output.counts must have cellCapacity rows`);for(let e of[`sums`,`minimums`,`maximums`]){let i=r[e];i&&(u(n,!!t.values,`output.${e} needs values`),l(i,[`float32`],`${n} output.${e}`),u(n,i.length===t.cellCapacity,`output.${e} must have cellCapacity rows`))}for(let[e,t]of Object.entries(r.status))t&&(l(t,[`uint32`],`${n} output.status.${e}`),u(n,t.length>=1,`output.status.${e} must hold one row`));e(n,{cells:r.cells,counts:r.counts,sums:r.sums,minimums:r.minimums,maximums:r.maximums,...r.status},[{cells:r.cells,counts:r.counts},{...r.status}]),m(n,[r.cells,r.counts,r.sums,r.minimums,r.maximums,...Object.values(r.status)],[t.cells,t.mask,t.values])}getCommandNodes(e){let{id:l,props:u}=this,{output:d}=u;t(e,l,{cells:u.cells,values:u.values,mask:u.mask,outputCells:d.cells,counts:d.counts,sums:d.sums,minimums:d.minimums,maximums:d.maximums,...d.status},[[`row sort scratch`,2**Math.ceil(Math.log2(Math.max(u.cells.length,1)))*4],[`run offsets scratch`,(u.cellCapacity+1)*4],[`status scratch`,8]]),p(l,e,[u.cells,u.mask,u.values,d.cells,d.counts,d.sums,d.minimums,d.maximums,...Object.values(d.status)]);let m=u.cells.length,g=u.cellCapacity;f(e.device,l,`output.cells`,g*8,`reduce cellCapacity`),f(e.device,l,`cells`,m*8,`aggregate the rows in several calls`);let _=2*u.order+4+ +(u.scheme===`uniq`),v=Math.max(1,Math.min(32,_-32)),y=(t,r,i)=>n(e,`${l}-${t}`,r,i),b=y(`low-keys`,`uint32`,m),x=y(`high-keys`,`uint32`,m),S=y(`left-out`,`uint32`,m),C=y(`row-ids`,`uint32`,m),w=y(`run-starts-flag`,`uint32`,m),T=y(`run-rank`,`uint32`,m),E=y(`run-start`,`uint32`,g+1),D=y(`totals`,`uint32`,2),O=y(`offsets`,`uint32`,g+1),k=a(e,{id:`${l}-sort`,operation:h,indices:C,keys:[{view:S,keyBits:1},..._>32?[{view:x,keyBits:v}]:[],{view:b,keyBits:Math.min(32,_)}]}),A=`const ROWS: u32 = ${m}u;
const CAPACITY: u32 = ${g}u;`,j=[s(e,{id:`${l}-split`,operation:h,variant:`split-keys`,bindings:[{name:`cells`,view:u.cells,type:`u32`,access:`read`},...u.mask?[{name:`rowMask`,view:u.mask,type:`u32`,access:`read`}]:[],...u.values?[{name:`values`,view:u.values,type:`f32`,access:`read`}]:[],{name:`lowKeys`,view:b,type:`u32`,access:`read_write`},{name:`highKeys`,view:x,type:`u32`,access:`read_write`},{name:`leftOut`,view:S,type:`u32`,access:`read_write`},{name:`rowIds`,view:C,type:`u32`,access:`read_write`}],invocationCount:m,declarations:A,body:`var keep = true;
  ${u.mask?`keep = rowMask[rowMaskOffset + index] != 0u;`:``}
  ${u.values?`keep = keep && (bitcast<u32>(values[valuesOffset + index]) & 0x7fffffffu) < 0x7f800000u;`:``}
  lowKeys[lowKeysOffset + index] = cells[cellsOffset + 2u * index];
  highKeys[highKeysOffset + index] = cells[cellsOffset + 2u * index + 1u];
  leftOut[leftOutOffset + index] = select(1u, 0u, keep);
  rowIds[rowIdsOffset + index] = index;`}),...k.nodes,s(e,{id:`${l}-run-starts`,operation:h,variant:`run-starts`,bindings:[{name:`sortedIndices`,view:k.sortedIndices,type:`u32`,access:`read`},{name:`lowKeys`,view:b,type:`u32`,access:`read`},{name:`highKeys`,view:x,type:`u32`,access:`read`},{name:`leftOut`,view:S,type:`u32`,access:`read`},{name:`starts`,view:w,type:`u32`,access:`read_write`}],invocationCount:m,declarations:A,body:`let row = sortedIndices[sortedIndicesOffset + index];
  var isStart = leftOut[leftOutOffset + row] == 0u;
  if (isStart && index > 0u) {
    let previous = sortedIndices[sortedIndicesOffset + index - 1u];
    isStart = leftOut[leftOutOffset + previous] != 0u ||
      lowKeys[lowKeysOffset + previous] != lowKeys[lowKeysOffset + row] ||
      highKeys[highKeysOffset + previous] != highKeys[highKeysOffset + row];
  }
  starts[startsOffset + index] = select(0u, 1u, isStart);`}),...new i({id:`${l}-scan`,input:w,output:T,mode:`inclusive`}).getCommandNodes(e),c(e,{id:`${l}-clear-totals`,operation:h,view:D,type:`u32`,value:`0u`}),s(e,{id:`${l}-run-table`,operation:h,variant:`run-table`,bindings:[{name:`sortedIndices`,view:k.sortedIndices,type:`u32`,access:`read`},{name:`leftOut`,view:S,type:`u32`,access:`read`},{name:`starts`,view:w,type:`u32`,access:`read`},{name:`inclusive`,view:T,type:`u32`,access:`read`},{name:`runStart`,view:E,type:`u32`,access:`read_write`},{name:`totals`,view:D,type:`u32`,access:`read_write`}],invocationCount:m,declarations:A,body:`let rank = inclusive[inclusiveOffset + index];
  if (starts[startsOffset + index] != 0u && rank - 1u <= CAPACITY) {
    runStart[runStartOffset + rank - 1u] = index;
  }
  let isKept = leftOut[leftOutOffset + sortedIndices[sortedIndicesOffset + index]] == 0u;
  var isLastKept = isKept;
  if (isKept && index + 1u < ROWS) {
    isLastKept = leftOut[leftOutOffset + sortedIndices[sortedIndicesOffset + index + 1u]] != 0u;
  }
  if (isLastKept) {
    totals[totalsOffset] = index + 1u;
    totals[totalsOffset + 1u] = rank;
  }`}),s(e,{id:`${l}-table`,operation:h,variant:`cell-table`,bindings:[{name:`sortedIndices`,view:k.sortedIndices,type:`u32`,access:`read`},{name:`lowKeys`,view:b,type:`u32`,access:`read`},{name:`highKeys`,view:x,type:`u32`,access:`read`},{name:`runStart`,view:E,type:`u32`,access:`read`},{name:`totals`,view:D,type:`u32`,access:`read`},{name:`offsets`,view:O,type:`u32`,access:`read_write`},{name:`cellsOut`,view:d.cells,type:`u32`,access:`read_write`},{name:`counts`,view:d.counts,type:`u32`,access:`read_write`}],invocationCount:g+1,declarations:A,body:`let keptRows = totals[totalsOffset];
  let cellTotal = totals[totalsOffset + 1u];
  let isCell = index < cellTotal;
  let begin = select(keptRows, runStart[runStartOffset + index], isCell);
  offsets[offsetsOffset + index] = begin;
  if (index < CAPACITY) {
    var key = vec2u(0u);
    var end = keptRows;
    if (isCell) {
      let row = sortedIndices[sortedIndicesOffset + begin];
      key = vec2u(lowKeys[lowKeysOffset + row], highKeys[highKeysOffset + row]);
      end = select(keptRows, runStart[runStartOffset + index + 1u], index + 1u < cellTotal);
    }
    cellsOut[cellsOutOffset + 2u * index] = key.x;
    cellsOut[cellsOutOffset + 2u * index + 1u] = key.y;
    counts[countsOffset + index] = select(0u, end - begin, isCell);
  }`})];if(u.values){let t=y(`sorted-values`,`float32`,m);j.push(...new r({id:`${l}-gather-values`,source:u.values,indices:k.sortedIndices,output:t}).getCommandNodes(e));for(let[n,r]of[[`sums`,`sum`],[`minimums`,`min`],[`maximums`,`max`]]){let i=d[n];i&&j.push(o(e,{id:`${l}-${n}`,operation:h,segmentCount:g,input:t,segmentOffsets:O,output:i,reduction:r}))}}return j.push(s(e,{id:`${l}-status`,operation:h,variant:`status`,bindings:[{name:`totals`,view:D,type:`u32`,access:`read`},...d.status.count?[{name:`count`,view:d.status.count,type:`u32`,access:`read_write`}]:[],...d.status.requiredCount?[{name:`requiredCount`,view:d.status.requiredCount,type:`u32`,access:`read_write`}]:[],...d.status.overflow?[{name:`overflow`,view:d.status.overflow,type:`u32`,access:`read_write`}]:[],...d.status.invalidCount?[{name:`invalidCount`,view:d.status.invalidCount,type:`u32`,access:`read_write`}]:[]],invocationCount:1,declarations:A,body:`let cellTotal = totals[totalsOffset + 1u];
  ${d.status.count?`count[countOffset] = min(cellTotal, CAPACITY);`:``}
  ${d.status.requiredCount?`requiredCount[requiredCountOffset] = cellTotal;`:``}
  ${d.status.overflow?`overflow[overflowOffset] = select(0u, 1u, cellTotal > CAPACITY);`:``}
  ${d.status.invalidCount?`invalidCount[invalidCountOffset] = ROWS - totals[totalsOffset];`:``}`})),j}};export{_ as t};