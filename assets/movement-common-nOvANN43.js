import{r as e}from"./kernel-C3tfQOn4.js";import{c as t,o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as r}from"./geodesic-wgsl-I0CYDKcz.js";var i=`0xffffffffu`;function a(e){let n=e===void 0?`length(rowPosition(a) - rowPosition(b))`:`geodesicHaversineDistance(rowPosition(a), rowPosition(b), ${t(e)})`;return`${e===void 0?``:r}
fn rowPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn rowDistance(a: u32, b: u32) -> f32 {
  return ${n};
}
`}function o(e,t){let{id:r,operation:a,trackOffsets:o,rowCount:s,rowStarts:c,rowEnds:l,rowTracks:u}=t;return[n(e,{id:`${r}-reset`,operation:a,variant:`row-bounds`,invocationCount:s,bindings:[{name:`rowStarts`,view:c,type:`u32`,access:`read_write`},{name:`rowEnds`,view:l,type:`u32`,access:`read_write`},{name:`rowTracks`,view:u,type:`u32`,access:`read_write`}],body:`rowStarts[rowStartsOffset + index] = ${i};
  rowEnds[rowEndsOffset + index] = ${i};
  rowTracks[rowTracksOffset + index] = ${i};`}),n(e,{id:`${r}-fill`,operation:a,variant:`row-bounds`,invocationCount:o.length-1,bindings:[{name:`trackOffsets`,view:o,type:`u32`,access:`read`},{name:`rowStarts`,view:c,type:`u32`,access:`read_write`},{name:`rowEnds`,view:l,type:`u32`,access:`read_write`},{name:`rowTracks`,view:u,type:`u32`,access:`read_write`}],body:`let begin = min(trackOffsets[trackOffsetsOffset + index], ${s}u);
  let end = min(trackOffsets[trackOffsetsOffset + index + 1u], ${s}u);
  for (var row = begin; row < end; row++) {
    rowStarts[rowStartsOffset + row] = begin;
    rowEnds[rowEndsOffset + row] = end;
    rowTracks[rowTracksOffset + row] = index;
  }`})]}function s(e,t){return n(e,{id:t.id,operation:t.operation,variant:`row-ids`,invocationCount:t.rowIds.length,bindings:[{name:`rowIds`,view:t.rowIds,type:`u32`,access:`read_write`}],body:`rowIds[rowIdsOffset + index] = index;`})}function c(e){return Math.max(1,32-Math.clz32(Math.max(e-1,1)))}function l(t,r){let{id:i,operation:a,trackOffsets:o,next:s,reach:l,rowCount:u}=r,d=c(u),f=e(t,`${i}-jump-even`,`uint32`,u),p=e(t,`${i}-jump-odd`,`uint32`,u),m=[n(t,{id:`${i}-reset`,operation:a,variant:`chain-reach`,invocationCount:u,bindings:[{name:`reach`,view:l,type:`u32`,access:`read_write`}],body:`reach[reachOffset + index] = 0u;`}),n(t,{id:`${i}-seed`,operation:a,variant:`chain-reach`,invocationCount:o.length-1,bindings:[{name:`trackOffsets`,view:o,type:`u32`,access:`read`},{name:`reach`,view:l,type:`atomic<u32>`,access:`read_write`}],body:`let begin = min(trackOffsets[trackOffsetsOffset + index], ${u}u);
  let end = min(trackOffsets[trackOffsetsOffset + index + 1u], ${u}u);
  if (begin < end) { atomicStore(&reach[reachOffset + begin], 1u); }`})];for(let e=0;e<d;e++){let r=e===0?s:e%2==1?f:p,o=e%2==0?f:p;m.push(n(t,{id:`${i}-pass-${e}`,operation:a,variant:`chain-reach`,invocationCount:u,bindings:[{name:`reach`,view:l,type:`atomic<u32>`,access:`read_write`},{name:`jumpIn`,view:r,type:`u32`,access:`read`},{name:`jumpOut`,view:o,type:`u32`,access:`read_write`}],body:`let jump = jumpIn[jumpInOffset + index];
  if (atomicLoad(&reach[reachOffset + index]) != 0u) { atomicStore(&reach[reachOffset + jump], 1u); }
  jumpOut[jumpOutOffset + index] = jumpIn[jumpInOffset + jump];`}))}return m}function u(e,t){let{id:r,operation:i,tracks:a,count:o,offsets:s,capacity:c,trackCount:l}=t;return n(e,{id:r,operation:i,variant:`track-offsets`,invocationCount:l+1,bindings:[{name:`tracks`,view:a,type:`u32`,access:`read`},{name:`total`,view:o,type:`u32`,access:`read`},{name:`offsets`,view:s,type:`u32`,access:`read_write`}],body:`let totalRows = min(total[totalOffset], ${c}u);
  var low = 0u;
  var high = totalRows;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (tracks[tracksOffset + middle] < index) { low = middle + 1u; }
    else { high = middle; }
  }
  offsets[offsetsOffset + index] = low;`})}export{u as a,s as i,l as n,a as o,o as r,i as t};