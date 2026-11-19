import{r as e}from"./kernel-C3tfQOn4.js";import{o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as n,t as r}from"./raster-kernel-utils-DcIN7G5I.js";function i(i,a){let o=e(i,`${a.id}-usable`,`uint32`,a.pixelCount);return{usable:o,node:t(i,{id:`${a.id}-usable`,operation:a.operation,variant:`usable-mask`,invocationCount:a.pixelCount,workgroupSize:64,bindings:[n(`vectors`,a.vectors),...a.mask?[n(`mask`,a.mask)]:[],n(`usable`,o,!0)],declarations:r,body:`var isUsable = ${a.mask?`mask[maskOffset + index] != 0u`:`true`};
for (var band = 0u; band < ${a.dimensions}u; band++) {
  if (!finite(vectors[vectorsOffset + index * ${a.dimensions}u + band])) { isUsable = false; }
}
usable[usableOffset + index] = select(0u, 1u, isUsable);`})}}export{i as t};