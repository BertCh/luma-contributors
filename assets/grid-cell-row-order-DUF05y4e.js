import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-sort-xRm8l5yn.js";import{o as n}from"./wgsl-kernel-nodes-DagJklH2.js";function r(r,i){let{id:a,operation:o,cellOffsets:s,objectIds:c,indexCount:l,rowCount:u}=i,d=s.length-1,f=e(r,`${a}-cell-keys`,`uint32`,u),p=e(r,`${a}-row-ids`,`uint32`,u),m=e(r,`${a}-sorted-keys`,`uint32`,u),h=e(r,`${a}-sorted-rows`,`uint32`,u),g=`const CELL_COUNT: u32 = ${d}u;
const ROW_COUNT: u32 = ${u}u;`;return{nodes:[n(r,{id:`${a}-initialize-cell-keys`,operation:o,variant:`initialize-cell-keys`,bindings:[{name:`cellKeys`,view:f,type:`u32`,access:`read_write`},{name:`rowIds`,view:p,type:`u32`,access:`read_write`}],invocationCount:u,declarations:g,body:`cellKeys[cellKeysOffset + index] = CELL_COUNT;
  rowIds[rowIdsOffset + index] = index;`}),n(r,{id:`${a}-scatter-cell-keys`,operation:o,variant:`scatter-cell-keys`,bindings:[{name:`cellOffsets`,view:s,type:`u32`,access:`read`},{name:`objectIds`,view:c,type:`u32`,access:`read`},{name:`indexCount`,view:l,type:`u32`,access:`read`},{name:`cellKeys`,view:f,type:`u32`,access:`read_write`}],invocationCount:u,declarations:g,body:`if (index >= min(indexCount[indexCountOffset], ROW_COUNT)) {
    return;
  }
  let row = objectIds[objectIdsOffset + index];
  // Largest cell whose first slot is at or before this slot.
  var low = 0u;
  var high = CELL_COUNT - 1u;
  while (low < high) {
    let middle = low + (high - low + 1u) / 2u;
    if (cellOffsets[cellOffsetsOffset + middle] <= index) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  cellKeys[cellKeysOffset + row] = low;`}),...new t({id:`${a}-sort-rows`,keys:f,values:p,outputKeys:m,outputValues:h,keyBits:Math.max(1,Math.ceil(Math.log2(d+1)))}).getCommandNodes(r)],sortedRows:h}}export{r as t};