import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-uint64-scan-_k5PA9Xg.js";import{S as n}from"./gpu-contributor-utils-BIwLQON5.js";import{o as r,t as i}from"./wgsl-kernel-nodes-DagJklH2.js";var a=4294967295;function o(e,t){return n(t.id,`rowCount`,t.rowCount,t.counts.length),n(t.id,`rowCount`,t.rowCount,t.offsets.length),n(t.id,`totalRow`,t.totalRow??0,t.total.length-1),r(e,{id:`${t.id}-carry`,operation:t.operation,variant:`scan-carry`,bindings:[{name:`counts`,view:t.counts,type:`u32`,access:`read`},{name:`offsets`,view:t.offsets,type:`u32`,access:`read`},{name:`total`,view:t.total,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.rowCount,declarations:`override TOTAL_ROW: u32;`,constants:{TOTAL_ROW:t.totalRow??0},body:`let offset = offsets[offsetsOffset + index];
  if (offset + counts[countsOffset + index] < offset) {
    atomicStore(&total[totalOffset + TOTAL_ROW], ${a}u);
  }`})}function s(e,t){return[r(e,{id:`${t.id}-total`,operation:t.operation,variant:`scan-total`,bindings:[{name:`counts`,view:t.counts,type:`u32`,access:`read`},{name:`offsets`,view:t.offsets,type:`u32`,access:`read`},{name:`total`,view:t.total,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`override ROW_COUNT: u32;
override TOTAL_ROW: u32;`,constants:{ROW_COUNT:t.rowCount,TOTAL_ROW:t.totalRow??0},body:`var sum = 0u;
  if (ROW_COUNT > 0u) {
    sum = offsets[offsetsOffset + ROW_COUNT - 1u] + counts[countsOffset + ROW_COUNT - 1u];
  }
  total[totalOffset + TOTAL_ROW] = sum;`}),o(e,t)]}var c=`
fn getSaturatedScanSum(offset: u32, count: u32) -> u32 {
  return offset + min(count, 0xffffffffu - offset);
}
fn getSaturatedScanProduct(count: u32, multiplier: u32) -> u32 {
  if (multiplier == 0u) { return 0u; }
  let maximumCount = 0xffffffffu / multiplier;
  return select(min(count, maximumCount) * multiplier, 0xffffffffu, count > maximumCount);
}
`;function l(o,s){n(s.id,`rowCount`,s.rowCount,a),n(s.id,`rowCount`,s.rowCount,s.counts.length),n(s.id,`rowCount`,s.rowCount,s.offsets.length),n(s.id,`totalRow`,s.totalRow??0,s.total.length-1);let c=o.createDataView(s.counts.buffer,{format:`uint32`,byteOffset:s.counts.byteOffset,length:s.rowCount}),l=e(o,`${s.id}-high-input`,`uint32`,s.rowCount),u=e(o,`${s.id}-prefix-low`,`uint32`,s.rowCount),d=e(o,`${s.id}-prefix-high`,`uint32`,s.rowCount);return[i(o,{id:`${s.id}-clear-high`,operation:s.operation,view:l,type:`u32`,value:`0u`}),...new t({id:s.id,inputLow:c,inputHigh:l,outputLow:u,outputHigh:d}).getCommandNodes(o),r(o,{id:`${s.id}-saturate`,operation:s.operation,variant:`saturated-count-scan`,bindings:[{name:`low`,view:u,type:`u32`,access:`read`},{name:`high`,view:d,type:`u32`,access:`read`},{name:`offsets`,view:s.offsets,type:`u32`,access:`read_write`},{name:`total`,view:s.total,type:`u32`,access:`read_write`}],invocationCount:Math.max(1,s.rowCount),declarations:`override ROW_COUNT: u32;
override TOTAL_ROW: u32;`,constants:{ROW_COUNT:s.rowCount,TOTAL_ROW:s.totalRow??0},body:`if (index == 0u) {
    var sum = 0u;
    if (ROW_COUNT > 0u) {
      sum = select(low[lowOffset + ROW_COUNT - 1u], 0xffffffffu, high[highOffset + ROW_COUNT - 1u] != 0u);
    }
    total[totalOffset + TOTAL_ROW] = sum;
  }
  if (index < ROW_COUNT) {
    var offset = 0u;
    if (index > 0u) {
      offset = select(low[lowOffset + index - 1u], 0xffffffffu, high[highOffset + index - 1u] != 0u);
    }
    offsets[offsetsOffset + index] = offset;
  }`})]}export{o as i,l as n,s as r,c as t};