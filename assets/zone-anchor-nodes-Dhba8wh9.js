import{Q as e}from"./dist-BusamGlx.js";import{r as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-sort-xRm8l5yn.js";import{o as r,t as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{a}from"./sorted-segment-sums-Dvmm6OdR.js";import{a as o}from"./graph-chunk-dispatch-zBt8k2-o.js";import{D as s,f as c,v as l}from"./gpu-contributor-utils-BIwLQON5.js";var u=4294967294;function d(d,f){let{id:p,operation:m,zones:h,zoneCount:g,anchorCells:_}=f,v=h.length;c(p,`zoneCount`,g,1,u),s(p,`anchorCells`,_,g),o(d,p,{zones:h,anchorCells:_});for(let[e,t]of Object.entries({effectiveZones:v,counts:g,claims:v,keys:g,rows:g,sortedKeys:g,sortedRows:g}))l(d.device,p,`anchor ${e}`,t*Uint32Array.BYTES_PER_ELEMENT,`reduce the grid size or zoneCount`);let y=(e,n)=>t(d,`${p}-anchor-${e}`,`uint32`,n),b=t(d,`${p}-anchor-zones`,`uint32`,v,e.STORAGE|e.COPY_SRC),x=y(`counts`,g),S=y(`claims`,v),C=y(`keys`,g),w=y(`rows`,g),T=y(`sorted-keys`,g),E=y(`sorted-rows`,g),D=`const ZONE_COUNT: u32 = ${g}u;
const CELL_COUNT: u32 = ${v}u;`;return{zones:b,nodes:[i(d,{id:`${p}-anchor-counts-clear`,operation:m,view:x,type:`u32`,value:`0u`}),i(d,{id:`${p}-anchor-claims-clear`,operation:m,view:S,type:`u32`,value:`0xffffffffu`}),r(d,{id:`${p}-anchor-copy`,operation:m,variant:`anchor-copy`,invocationCount:v,declarations:D,bindings:[{name:`zones`,view:h,type:`u32`,access:`read`},{name:`effectiveZones`,view:b,type:`u32`,access:`read_write`},{name:`counts`,view:x,type:`atomic<u32>`,access:`read_write`}],body:`let zone = zones[zonesOffset + index];
      effectiveZones[effectiveZonesOffset + index] = zone;
      if (zone < ZONE_COUNT) { atomicAdd(&counts[countsOffset + zone], 1u); }`}),r(d,{id:`${p}-anchor-propose`,operation:m,variant:`anchor-propose`,invocationCount:g,declarations:D,bindings:[{name:`zones`,view:h,type:`u32`,access:`read`},{name:`anchorCells`,view:_,type:`u32`,access:`read`},{name:`counts`,view:x,type:`u32`,access:`read`},{name:`claims`,view:S,type:`atomic<u32>`,access:`read_write`}],body:`if (counts[countsOffset + index] != 0u) { return; }
      let anchor = anchorCells[anchorCellsOffset + index];
      if (anchor >= CELL_COUNT) { return; }
      let owner = zones[zonesOffset + anchor];
      if (owner < ZONE_COUNT && counts[countsOffset + owner] < 2u) { return; }
      atomicMin(&claims[claimsOffset + anchor], index);`}),r(d,{id:`${p}-anchor-keys`,operation:m,variant:`anchor-keys`,invocationCount:g,declarations:D,bindings:[{name:`zones`,view:h,type:`u32`,access:`read`},{name:`anchorCells`,view:_,type:`u32`,access:`read`},{name:`claims`,view:S,type:`u32`,access:`read`},{name:`keys`,view:C,type:`u32`,access:`read_write`},{name:`rows`,view:w,type:`u32`,access:`read_write`}],body:`var key = ZONE_COUNT + 1u;
      let anchor = anchorCells[anchorCellsOffset + index];
      if (anchor < CELL_COUNT && claims[claimsOffset + anchor] == index) { key = min(zones[zonesOffset + anchor], ZONE_COUNT); }
      keys[keysOffset + index] = key;
      rows[rowsOffset + index] = index;`}),...new n({id:`${p}-anchor-sort`,keys:C,values:w,outputKeys:T,outputValues:E,keyBits:a(g+1)}).getCommandNodes(d),r(d,{id:`${p}-anchor-claim`,operation:m,variant:`anchor-claim`,invocationCount:g,declarations:D,bindings:[{name:`anchorCells`,view:_,type:`u32`,access:`read`},{name:`counts`,view:x,type:`u32`,access:`read`},{name:`sortedKeys`,view:T,type:`u32`,access:`read`},{name:`sortedRows`,view:E,type:`u32`,access:`read`},{name:`effectiveZones`,view:b,type:`u32`,access:`read_write`}],body:`let owner = sortedKeys[sortedKeysOffset + index];
      if (owner > ZONE_COUNT) { return; }
      var first = 0u; var end = index;
      while (first < end) {
        let middle = first + (end - first) / 2u;
        if (sortedKeys[sortedKeysOffset + middle] < owner) { first = middle + 1u; } else { end = middle; }
      }
      // Stable owner sort retains ascending zone order, exactly the sequential donor quota.
      if (owner < ZONE_COUNT && index - first >= counts[countsOffset + owner] - 1u) { return; }
      let zone = sortedRows[sortedRowsOffset + index];
      effectiveZones[effectiveZonesOffset + anchorCells[anchorCellsOffset + zone]] = zone;`})]}}export{d as t};