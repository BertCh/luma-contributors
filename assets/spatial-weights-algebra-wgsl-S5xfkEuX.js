import{r as e}from"./scan-total-C4QddMmi.js";import{r as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-scan-B4NNTfF9.js";import{o as r}from"./wgsl-kernel-nodes-DagJklH2.js";var i=`0xffffffffu`;function a(a,o){let{id:s,operation:c,rows:l,capacity:u}=o,d=t(a,`${s}-counts`,`uint32`,l),f=t(a,`${s}-starts`,`uint32`,l),p=t(a,`${s}-scan-total`,`uint32`,1),{chunk:m}=o,h=m?`CHUNK_ROWS`:`ROWS`,g=`const ROWS: u32 = ${m?m.totalRows:l}u;
${m?`const CHUNK_ROWS: u32 = ${l}u;\nconst WEIGHTS_ROW_START: u32 = ${m.rowStart}u;\n`:``}const CAPACITY: u32 = ${u}u;
const NO_SLOT: u32 = ${i};`,_=m?e=>`{ let index = index + WEIGHTS_ROW_START;\n${o.visit(e)}\n}`:o.visit,v=`${g}\n${o.declarations??``}`,y=[];y.push(r(a,{id:`${s}-counts`,operation:c,variant:`counts`,bindings:[...o.bindings,{name:`counts`,view:d,type:`u32`,access:`read_write`}],invocationCount:l,declarations:v,body:`var count = 0u;
  ${_(`count++;`)}
  counts[countsOffset + index] = count;`})),y.push(...new n({id:`${s}-scan`,input:d,output:f,mode:`exclusive`}).getCommandNodes(a)),y.push(...e(a,{id:s,operation:c,counts:d,offsets:f,rowCount:l,total:p}),r(a,{id:`${s}-offsets`,operation:c,variant:`offsets`,bindings:[{name:`scanTotal`,view:p,type:`u32`,access:`read`},{name:`starts`,view:f,type:`u32`,access:`read`},{name:`outOffsets`,view:o.targetOffsets,type:`u32`,access:`read_write`},{name:`overflow`,view:o.overflow,type:`u32`,access:`read_write`},...o.totalNeighbors?[{name:`totalNeighbors`,view:o.totalNeighbors,type:`u32`,access:`read_write`}]:[]],invocationCount:l+1,declarations:g,body:`if (index < ${h}) {
    outOffsets[outOffsetsOffset + index] = min(starts[startsOffset + index], CAPACITY);
  } else {
    let total = scanTotal[scanTotalOffset];
    outOffsets[outOffsetsOffset + ${h}] = min(total, CAPACITY);
    ${o.initializesOverflow?`overflow[overflowOffset] = select(0u, 1u, total > CAPACITY);`:`if (total > CAPACITY) {
      overflow[overflowOffset] = 1u;
    }`}
    ${o.totalNeighbors?`totalNeighbors[totalNeighborsOffset] = total;`:``}
  }`}));for(let e of o.attributes)y.push(r(a,{id:`${s}-emit-${e.name}`,operation:c,variant:`emit-${e.name}`,bindings:[...o.bindings,...e.bindings??[],{name:`outOffsets`,view:o.targetOffsets,type:`u32`,access:`read`},{name:`outValues`,view:e.output,type:e.type,access:`read_write`}],invocationCount:l,declarations:`${v}\n${e.declarations??``}`,body:`var next = outOffsets[outOffsetsOffset + index];
  let end = outOffsets[outOffsetsOffset + index + 1u];
  ${_(`if (next < end) {
    outValues[outValuesOffset + next] = ${e.value};
    next++;
  }`)}`}));if(o.sortRows){let e=o.attributes[0];y.push(r(a,{id:`${s}-sort-rows`,operation:c,variant:`sort-rows`,bindings:[{name:`outOffsets`,view:o.targetOffsets,type:`u32`,access:`read`},{name:`outValues`,view:e.output,type:`u32`,access:`read_write`}],invocationCount:l,body:`let begin = outOffsets[outOffsetsOffset + index];
  let end = outOffsets[outOffsetsOffset + index + 1u];
  for (var slot = begin + 1u; slot < end; slot++) {
    let key = outValues[outValuesOffset + slot];
    var position = slot;
    while (position > begin && outValues[outValuesOffset + position - 1u] > key) {
      outValues[outValuesOffset + position] = outValues[outValuesOffset + position - 1u];
      position--;
    }
    outValues[outValuesOffset + position] = key;
  }`}))}return y}function o(e,t){return`fn ${e}(begin: u32, end: u32, value: u32) -> bool {
  var low = begin;
  var high = end;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (${t}[${t}Offset + middle] < value) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low < end && ${t}[${t}Offset + low] == value;
}`}function s(e,t,n=`index`){return`var leftNext = leftOffsets[leftOffsetsOffset + ${n}];
  let leftEnd = leftOffsets[leftOffsetsOffset + ${n} + 1u];
  var rightNext = rightOffsets[rightOffsetsOffset + ${n}];
  let rightEnd = rightOffsets[rightOffsetsOffset + ${n} + 1u];
  loop {
    let hasLeft = leftNext < leftEnd;
    let hasRight = rightNext < rightEnd;
    if (!hasLeft && !hasRight) {
      break;
    }
    let leftId = select(NO_SLOT, leftNeighbors[leftNeighborsOffset + leftNext], hasLeft);
    let rightId = select(NO_SLOT, rightNeighbors[rightNeighborsOffset + rightNext], hasRight);
    let id = min(leftId, rightId);
    var leftSlot = NO_SLOT;
    var rightSlot = NO_SLOT;
    if (leftId == id) {
      leftSlot = leftNext;
      leftNext++;
    }
    if (rightId == id) {
      rightSlot = rightNext;
      rightNext++;
    }
    if (${{union:`true`,intersection:`leftSlot != NO_SLOT && rightSlot != NO_SLOT`,difference:`leftSlot != NO_SLOT && rightSlot == NO_SLOT`,symmetricDifference:`(leftSlot == NO_SLOT) != (rightSlot == NO_SLOT)`}[e]}) {
      ${t}
    }
  }`}export{o as n,s as r,a as t};