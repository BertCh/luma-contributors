import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-scan-B4NNTfF9.js";import{t as n}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{o as r}from"./wgsl-kernel-nodes-DagJklH2.js";var i=4294967295;function a(e){return 3*e+1}function o(t,i,a,o,s,c,l){let u=e(t,`${i}-item-ids`,`uint32`,o),d=n(t,{id:`${i}-sort`,operation:a,indices:u,indicesAreIdentity:!0,packKeys:!1,keys:[{view:s,keyBits:l?.high},{view:c,keyBits:l?.low}]});return{nodes:[r(t,{id:`${i}-iota`,operation:a,variant:`iota`,bindings:[{name:`itemIds`,view:u,type:`u32`,access:`read_write`}],invocationCount:o,body:`itemIds[itemIdsOffset + index] = index;`}),...d.nodes],sortedItems:d.sortedIndices}}function s(n,a,o,s,c,l,u){let d=e(n,`${a}-run-flags`,`uint32`,s),f=e(n,`${a}-group-index`,`uint32`,s),p=e(n,`${a}-group-starts`,`uint32`,s+1);return{nodes:[r(n,{id:`${a}-run-flags`,operation:o,variant:`run-flags`,bindings:[{name:`sortedItems`,view:u,type:`u32`,access:`read`},{name:`keyHigh`,view:c,type:`u32`,access:`read`},{name:`keyLow`,view:l,type:`u32`,access:`read`},{name:`flags`,view:d,type:`u32`,access:`read_write`}],invocationCount:s,body:`let item = sortedItems[sortedItemsOffset + index];
  let high = keyHigh[keyHighOffset + item];
  let low = keyLow[keyLowOffset + item];
  var start = index == 0u || (high == ${i}u && low == ${i}u);
  if (!start) {
    let previous = sortedItems[sortedItemsOffset + index - 1u];
    start = high != keyHigh[keyHighOffset + previous] || low != keyLow[keyLowOffset + previous];
  }
  flags[flagsOffset + index] = select(0u, 1u, start);`}),...new t({id:`${a}-group-scan`,input:d,output:f,mode:`inclusive`}).getCommandNodes(n),r(n,{id:`${a}-group-starts`,operation:o,variant:`group-starts`,bindings:[{name:`flags`,view:d,type:`u32`,access:`read`},{name:`groupIndex`,view:f,type:`u32`,access:`read`},{name:`groupStarts`,view:p,type:`u32`,access:`read_write`}],invocationCount:s,body:`if (flags[flagsOffset + index] != 0u) {
    groupStarts[groupStartsOffset + groupIndex[groupIndexOffset + index] - 1u] = index;
  }
  if (index + 1u == INVOCATION_COUNT) {
    groupStarts[groupStartsOffset + groupIndex[groupIndexOffset + index]] = INVOCATION_COUNT;
  }`})],groupIndex:f,groupStarts:p}}export{o as i,s as n,a as r,i as t};