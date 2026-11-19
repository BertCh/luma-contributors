import{t as e}from"./sf-bay-ais-CkxLNyrO.js";import{E as t,y as n}from"./gpu-contributor-utils-BIwLQON5.js";import{o as r}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as i}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as a}from"./define-run-B-brkCe4.js";import{o}from"./pick-Bo3t3hjv.js";import{Ro as s}from"./entry-preview-D70TdSbK.js";var c=`GPUOffsetExpansion`,l=4294967295,u=class{id;props;constructor(e){this.id=e.id??`offset-expansion`,this.props=e;let{id:r}=this,{output:i}=e;if(t(e.offsets,[`uint32`],`${r} offsets`),e.offsets.length<2)throw Error(`${r} offsets needs two rows`);if(e.ownerMap&&(t(e.ownerMap,[`uint32`],`${r} ownerMap`),e.ownerMap.length<e.offsets.length-1))throw Error(`${r} ownerMap must hold one row per owner`);if(t(i.owners,[`uint32`],`${r} output.owners`),i.owners.length<1)throw Error(`${r} output.owners must be non-empty`);if(i.localIndex&&(t(i.localIndex,[`uint32`],`${r} output.localIndex`),i.localIndex.length!==i.owners.length))throw Error(`${r} output.localIndex length must equal output.owners length`);for(let[e,n]of[[`count`,i.status.count],[`overflow`,i.status.overflow],[`requiredCount`,i.status.requiredCount]])if(n&&(t(n,[`uint32`],`${r} output.status.${e}`),n.length<1))throw Error(`${r} output.status.${e} must contain one uint32 row`);n(r,[i.owners,i.localIndex,i.status.count,i.status.overflow,i.status.requiredCount],[e.offsets,e.ownerMap])}getCommandNodes(e){let{props:t,id:n}=this,{output:a}=t;i(e,n,Object.fromEntries([t.offsets,t.ownerMap,a.owners,a.localIndex,a.status.count,a.status.overflow,a.status.requiredCount].map((e,t)=>[String(t),e])));let o=t.offsets.length-1,s=[{name:`offsets`,view:t.offsets,type:`u32`,access:`read`},{name:`ownersOut`,view:a.owners,type:`u32`,access:`read_write`},{name:`countOut`,view:a.status.count,type:`u32`,access:`read_write`},{name:`overflowOut`,view:a.status.overflow,type:`u32`,access:`read_write`}];return t.ownerMap&&s.push({name:`ownerMap`,view:t.ownerMap,type:`u32`,access:`read`}),a.localIndex&&s.push({name:`localOut`,view:a.localIndex,type:`u32`,access:`read_write`}),a.status.requiredCount&&s.push({name:`totalOut`,view:a.status.requiredCount,type:`u32`,access:`read_write`}),[r(e,{id:`${n}-expand`,operation:c,bindings:s,invocationCount:a.owners.length,declarations:`const OWNER_COUNT: u32 = ${o}u;
const CAPACITY: u32 = ${a.owners.length}u;
const NO_OWNER: u32 = ${l}u;`,body:`let total = offsets[offsetsOffset + OWNER_COUNT];
  if (index == 0u) {
    countOut[countOutOffset] = min(total, CAPACITY);
    overflowOut[overflowOutOffset] = select(0u, 1u, total > CAPACITY);
    ${a.status.requiredCount?`totalOut[totalOutOffset] = total;`:``}
  }
  // Count the owners whose first row is at or before this row (offsets are non-decreasing).
  var low = 0u;
  var high = OWNER_COUNT;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (offsets[offsetsOffset + middle] <= index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  var owner = NO_OWNER;
  var local = NO_OWNER;
  if (index < total && low > 0u) {
    let ownerRow = low - 1u;
    owner = ${t.ownerMap?`ownerMap[ownerMapOffset + ownerRow]`:`ownerRow`};
    local = index - offsets[offsetsOffset + ownerRow];
  }
  ownersOut[ownersOutOffset + index] = owner;
  ${a.localIndex?`localOut[localOutOffset + index] = local;`:``}`})]}},d=[.07,.56,.95,.92],f=[.07,.08,.95,.44];function p(e,t,n){return(t[0]+n*(t[2]-t[0]))*e.width}var m=a({parameters:s,caption:`Each bar is an AIS fix. The GPU reconstructs its vessel index and its position within the track.`,createInputs:({world:t})=>({fixCounts:e.map(e=>e.length),world:t}),build(e,{fixCounts:t,world:n},r){let i=r.rowCount,a=[0];for(let e=0;e<i;e++)a.push(a[e]+Math.min(t[e],r.maximumFixes));let o=a[i],s=e.upload(`offsets`,`uint32`,Uint32Array.from(a)),c=e.allocate(`owners`,`uint32`,o),l=e.allocate(`local-index`,`uint32`,o),f=e.allocate(`count`,`uint32`,1),m=e.allocate(`overflow`,`uint32`,1);e.graph.add(new u({offsets:s.view,output:{owners:c.view,localIndex:l.view,status:{count:f.view,overflow:m.view}}}));let h=new Float32Array(i*4);for(let e=1;e<=i;e++){let t=p(n,d,a[e]/o);h.set([t,.08*n.height,t,.92*n.height],(e-1)*4)}return{outputs:{owners:c,localIndex:l,rowCount:i,elementCount:o,offsets:a,ticks:e.upload(`ticks`,`float32x4`,h),longestRun:Math.max(...a.slice(1).map((e,t)=>e-a[t]))}}},rebuildOn:[`rowCount`,`maximumFixes`],interaction:{kind:`probe`,hint:`hover a fix`,resolve(e,{inputs:t,outputs:n,peek:r}){let i=o(d,t.world,e.x,.74*t.world.height),a=Math.min(n.elementCount-1,Math.floor(i.u*n.elementCount)),s=r(n.owners)?.[a],c=r(n.localIndex)?.[a];if(s===void 0||c===void 0)return{};let[l,u]=[n.offsets[s],n.offsets[s+1]].map(e=>p(t.world,d,e/n.elementCount)),f=p(t.world,d,(a+.5)/n.elementCount),m=t.world.height;return{focus:a,glyphs:[{shape:`box`,minimum:[l,.56*m],maximum:[u,.92*m]},{shape:`box`,minimum:[l,.08*m],maximum:[u,.44*m]},{shape:`line`,from:[f,.08*m],to:[f,.92*m],dashed:!0}],readout:`fix ${a} -> vessel ${s}, local index ${c}`}}},draw:({outputs:e})=>[{layer:`chart`,kind:`bars`,values:e.owners,range:[-1,e.rowCount],rectangle:d},{layer:`chart`,kind:`bars`,values:e.localIndex,range:[-1,e.longestRun],ramp:`sequential`,rectangle:f},{layer:`lines`,context:!0,segments:e.ticks,radius:1,alpha:.35}]});export{m as default};