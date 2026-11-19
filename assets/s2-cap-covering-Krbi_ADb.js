import{t as e}from"./gpu-scan-B4NNTfF9.js";import{r as t}from"./kernel-C3tfQOn4.js";import{s as n}from"./cell-index-wgsl-Cw7syfdA.js";import{c as r,n as i,s as a}from"./spherical-limits-C7rAYBu5.js";import{v as o}from"./spatial-context-DwZEd4EZ.js";import{c as s,o as c}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as l,S as u,T as d,f,p,x as m,y as h}from"./gpu-contributor-utils-BIwLQON5.js";import{t as g}from"./s2-analysis-wgsl-CSV2LEdb.js";import{n as _,r as v,t as y}from"./s2-covering-support-BNzE_85J.js";import{i as b,r as x}from"./cell-dispatch-validation-PIkQyAK_.js";import{t as S}from"./define-run-B-brkCe4.js";import{d as C,s as w}from"./pointer-overlay-Qh8qflcA.js";import{n as T,t as E}from"./pick-Bo3t3hjv.js";import{r as D,t as O}from"./point-observations-CeqP-hIo.js";import{t as k}from"./cells-Di9l-3nc.js";import{n as A,t as j}from"./polygons--0zAAHu9.js";import{ol as M}from"./entry-preview-D70TdSbK.js";var N=4294967295,P=15,F=1e-6,I=.01,L=1e-6,R=6,z=8,B=64,V=class{props;id;spatialContext;frontierCapacity;estimatedCommandNodeCount;constructor(e){this.props=e,this.id=e.id??`s2-cap-covering`,this.spatialContext=o(this.id,e.spatialContext??i,a),l(e.caps,[`float32x4`],this.id),l(e.output.cells,[`uint32x2`],this.id),l(e.output.counts,[`uint32x2`],this.id),l(e.output.status,[`uint32`],this.id),f(this.id,`output.cells components`,e.output.cells.length*2,0,N),f(this.id,`caps components`,e.caps.length*4,0,N),d(this.id,e.refinement===void 0||e.refinement===`uniform`||e.refinement===`level-synchronous`,`refinement must be 'uniform' or 'level-synchronous'`);let t=e.refinement===`level-synchronous`;f(this.id,`level`,e.level,0,t?P:8),f(this.id,`capacity`,e.capacity,1,Math.min(N,6*4**e.level)),d(this.id,e.output.cells.length===e.caps.length*e.capacity&&e.output.counts.length===e.caps.length&&e.output.status.length===e.caps.length,`output.cells must hold caps times capacity entries, and output.counts and output.status one entry per cap`),r(this.id,e.tolerance??1e-5,F,I),this.frontierCapacity=e.frontierCapacity??z*e.capacity+B,t?(f(this.id,`frontierCapacity`,this.frontierCapacity,Math.max(e.capacity,R),4294967295),u(this.id,`caps times frontierCapacity`,e.caps.length*this.frontierCapacity,2**24)):d(this.id,e.frontierCapacity===void 0,`frontierCapacity needs refinement 'level-synchronous'`),this.estimatedCommandNodeCount=t?4+(e.level+1)*(2+_(e.caps.length*this.frontierCapacity+1)):1,p(this.id,this.estimatedCommandNodeCount),b(this.id,e.output,t?[{counts:e.output.counts,status:e.output.status}]:[e.output]),h(this.id,Object.values(e.output),[e.caps])}getCommandNodes(e){let{props:t,id:r}=this,i=t.refinement===`level-synchronous`,a=t.caps.length*this.frontierCapacity;if(x(e,r,{caps:t.caps,...t.output},i?[[`frontier scratch`,a*8],[`count and offset scratch`,(a+1)*4],[`list count scratch`,t.caps.length*4]]:[]),i&&v(e,r,t.level+1),m(r,e,[t.caps,...Object.values(t.output)]),t.refinement===`level-synchronous`)return this.getLevelSynchronousNodes(e);let o=s(t.tolerance??1e-5);return[c(e,{id:r,operation:`GPUS2CapCovering`,variant:t.mode??`exterior`,invocationCount:t.caps.length,bindings:[{name:`caps`,view:t.caps,type:`f32`,access:`read`},{name:`cells`,view:t.output.cells,type:`u32`,access:`read_write`},{name:`counts`,view:t.output.counts,type:`u32`,access:`read_write`},{name:`status`,view:t.output.status,type:`u32`,access:`read_write`}],declarations:`${n.source}\n${g}`,body:`let offset = capsOffset + index * 4u; let center = vec3f(caps[offset], caps[offset + 1u], caps[offset + 2u]); let radius = caps[offset + 3u];
var failure = 0u; var count = 0u;
if ((bitcast<u32>(dot(center, center)) & 0x7f800000u) == 0x7f800000u || (bitcast<u32>(radius) & 0x7f800000u) == 0x7f800000u || abs(dot(center, center) - 1.0) > 1e-4 || radius < 0.0 || radius > 3.1415928) { failure = 1u; }
if (failure == 0u) {
  for (var candidate = 0u; candidate < ${6*4**t.level}u; candidate++) {
    let face = candidate / ${4**t.level}u; let position = candidate % ${4**t.level}u;
    let cell = dggs_u64_shift_left(vec2u(0u, face), 61u) | dggs_u64_shift_left(vec2u(0u, position), ${61-2*t.level}u) | dggs_u64_shift_left(vec2u(0u, 1u), ${60-2*t.level}u);
    let selected = ${t.mode===`interior`?`3.141592653589793 - s2AnalysisDistance(cell, -center) <= radius - ${o}`:`s2AnalysisDistance(cell, center) <= radius + ${o}`};
    if (selected) {
      if (count < ${t.capacity}u) { let output = cellsOffset + (index * ${t.capacity}u + count) * 2u; cells[output] = cell.y; cells[output + 1u] = cell.x; }
      count += 1u;
    }
  }
}
for (var slot = min(count, ${t.capacity}u); slot < ${t.capacity}u; slot++) { let output = cellsOffset + (index * ${t.capacity}u + slot) * 2u; cells[output] = 0u; cells[output + 1u] = 0u; }
if (count > ${t.capacity}u) { failure |= 2u; }
counts[countsOffset + index * 2u] = min(count, ${t.capacity}u); counts[countsOffset + index * 2u + 1u] = count; status[statusOffset + index] = failure;`})]}getLevelSynchronousNodes(r){let{props:i,id:a,frontierCapacity:o}=this,l=`GPUS2CapCovering`,u=`${i.mode??`exterior`}-level-synchronous`,d=i.caps.length,f=d*o,p=i.level+1,m=s(i.tolerance??1e-5),h=s(L),_=y(r,a,p),v=[t(r,`${a}-frontier-a`,`uint32x2`,f),t(r,`${a}-frontier-b`,`uint32x2`,f)],b=t(r,`${a}-round-counts`,`uint32`,f+1),x=t(r,`${a}-round-offsets`,`uint32`,f+1),S=t(r,`${a}-list-counts`,`uint32`,d),C=[..._.nodes,c(r,{id:`${a}-start`,operation:l,variant:u,invocationCount:d,bindings:[{name:`caps`,view:i.caps,type:`f32`,access:`read`},{name:`list`,view:v[0],type:`u32`,access:`read_write`},{name:`counts`,view:b,type:`u32`,access:`read_write`},{name:`listCounts`,view:S,type:`u32`,access:`read_write`},{name:`status`,view:i.output.status,type:`u32`,access:`read_write`}],body:`let offset = capsOffset + index * 4u; let center = vec3f(caps[offset], caps[offset + 1u], caps[offset + 2u]); let radius = caps[offset + 3u];
var failure = 0u;
if ((bitcast<u32>(dot(center, center)) & 0x7f800000u) == 0x7f800000u || (bitcast<u32>(radius) & 0x7f800000u) == 0x7f800000u || abs(dot(center, center) - 1.0) > 1e-4 || radius < 0.0 || radius > 3.1415928) { failure = 1u; }
status[statusOffset + index] = failure; listCounts[listCountsOffset + index] = select(6u, 0u, failure != 0u);
if (failure == 0u) {
  for (var face = 0u; face < 6u; face++) {
    let row = (index * ${o}u + face) * 2u;
    list[listOffset + row] = 0u; list[listOffset + row + 1u] = (2u * face + 1u) << 28u;
  }
}
if (index == 0u) { counts[countsOffset + ${f}u] = 0u; }`})];for(let t=0;t<p;t++){let s=v[t%2],d=v[(t+1)%2];C.push(c(r,{id:`${a}-classify-${t}`,operation:l,variant:u,invocationCount:f,bindings:[{name:`roundParameters`,view:_.slots[t],type:`u32`,access:`read`},{name:`caps`,view:i.caps,type:`f32`,access:`read`},{name:`list`,view:s,type:`u32`,access:`read`},{name:`counts`,view:b,type:`u32`,access:`read_write`},{name:`listCounts`,view:S,type:`u32`,access:`read`}],declarations:`${n.source}\n${g}`,body:`let capIndex = index / ${o}u; let slot = index % ${o}u;
var count = 0u;
if (slot < min(listCounts[listCountsOffset + capIndex], ${o}u)) {
  let offset = capsOffset + capIndex * 4u; let center = vec3f(caps[offset], caps[offset + 1u], caps[offset + 2u]); let radius = caps[offset + 3u];
  let cell = vec2u(list[listOffset + index * 2u + 1u], list[listOffset + index * 2u]);
  let nearest = s2AnalysisDistance(cell, center);
  if (roundParameters[roundParametersOffset] == ${i.level}u) {
    let selected = ${i.mode===`interior`?`3.141592653589793 - s2AnalysisDistance(cell, -center) <= radius - ${m}`:`nearest <= radius + ${m}`};
    count = select(0u, 1u, selected);
  } else {
    // A child is no closer to the cap than its parent, so the parent bounds every descendant.
    let kept = nearest <= radius ${i.mode===`interior`?`-`:`+`} ${m} + ${h};
    count = select(0u, 4u, kept);
  }
}
counts[countsOffset + index] = count;`}),...new e({id:`${a}-scan-${t}`,input:b,output:x,mode:`exclusive`}).getCommandNodes(r),c(r,{id:`${a}-write-${t}`,operation:l,variant:u,invocationCount:f,bindings:[{name:`source`,view:s,type:`u32`,access:`read`},{name:`destination`,view:d,type:`u32`,access:`read_write`},{name:`counts`,view:b,type:`u32`,access:`read`},{name:`offsets`,view:x,type:`u32`,access:`read`},{name:`listCounts`,view:S,type:`u32`,access:`read_write`},{name:`status`,view:i.output.status,type:`atomic<u32>`,access:`read_write`}],declarations:n.source,body:`let capIndex = index / ${o}u; let slot = index % ${o}u;
let base = offsets[offsetsOffset + capIndex * ${o}u];
let count = counts[countsOffset + index]; let start = offsets[offsetsOffset + index] - base;
if (slot == 0u) {
  // The next list length is unclamped so a frontier overflow stays visible.
  let total = offsets[offsetsOffset + (capIndex + 1u) * ${o}u] - base;
  listCounts[listCountsOffset + capIndex] = total;
  if (total > ${o}u) { atomicOr(&status[statusOffset + capIndex], 4u); }
}
if (count != 0u) {
  let cell = vec2u(source[sourceOffset + index * 2u + 1u], source[sourceOffset + index * 2u]);
  let row = capIndex * ${o}u + start;
  if (count == 1u) {
    if (start < ${o}u) { destination[destinationOffset + row * 2u] = cell.y; destination[destinationOffset + row * 2u + 1u] = cell.x; }
  } else {
    let level = dggs_s2_get_level(cell);
    let marker = dggs_u64_shift_left(vec2u(0u, 1u), 2u * (30u - level)); let childMarker = dggs_u64_shift_right(marker, 2u);
    var child = dggs_u64_add(dggs_u64_subtract(cell, marker), childMarker); let step = dggs_u64_shift_left(childMarker, 1u);
    for (var childSlot = 0u; childSlot < 4u; childSlot++) {
      if (start + childSlot < ${o}u) { destination[destinationOffset + (row + childSlot) * 2u] = child.y; destination[destinationOffset + (row + childSlot) * 2u + 1u] = child.x; }
      child = dggs_u64_add(child, step);
    }
  }
}`}))}let w=v[p%2];return C.push(c(r,{id:`${a}-cells`,operation:l,variant:u,invocationCount:d*i.capacity,bindings:[{name:`list`,view:w,type:`u32`,access:`read`},{name:`listCounts`,view:S,type:`u32`,access:`read`},{name:`status`,view:i.output.status,type:`u32`,access:`read`},{name:`cells`,view:i.output.cells,type:`u32`,access:`read_write`}],body:`let capIndex = index / ${i.capacity}u; let slot = index % ${i.capacity}u;
var low = 0u; var high = 0u;
if ((status[statusOffset + capIndex] & 5u) == 0u && slot < min(listCounts[listCountsOffset + capIndex], ${o}u)) {
  let row = (capIndex * ${o}u + slot) * 2u; low = list[listOffset + row]; high = list[listOffset + row + 1u];
}
cells[cellsOffset + index * 2u] = low; cells[cellsOffset + index * 2u + 1u] = high;`}),c(r,{id:`${a}-publish`,operation:l,variant:u,invocationCount:d,bindings:[{name:`listCounts`,view:S,type:`u32`,access:`read`},{name:`counts`,view:i.output.counts,type:`u32`,access:`read_write`},{name:`status`,view:i.output.status,type:`u32`,access:`read_write`}],body:`var failure = status[statusOffset + index]; var written = 0u; var required = 0u;
if ((failure & 5u) == 0u) {
  required = listCounts[listCountsOffset + index]; written = min(required, ${i.capacity}u);
  if (required > ${i.capacity}u) { failure |= 2u; }
}
counts[countsOffset + index * 2u] = written; counts[countsOffset + index * 2u + 1u] = required; status[statusOffset + index] = failure;`})),C}},H=256,U=w(M,`radius`),W=Math.cos((O[1]+O[3])/2*Math.PI/180),G=Math.PI/180,K=S({parameters:M,geographicWindow:O,caption:`S2 covers for hypothetical search radii around three observed Honshu epicenters.`,createInputs:()=>{let e=D([[141.2,36.6],[142.3,37.5],[141.6,38.1]]);return{centers:e,index:T(e.flat()),held:-1}},build(e,t,n){let r=t.centers.length,i=e.allocate(`caps`,`float32x4`,r),a=e.allocate(`cap-outline`,`float32x4`,r*64),o=e.allocate(`cells`,`uint32x2`,r*H),s=e.allocate(`counts`,`uint32x2`,r),c=e.allocate(`cap-status`,`uint32`,r);return e.graph.add(new V({caps:i.view,level:n.level,mode:n.mode===0?`exterior`:`interior`,capacity:H,output:{cells:o.view,counts:s.view,status:c.view}})),{outputs:{outline:a,counts:s,sources:e.upload(`sources`,`uint32`,Uint32Array.from({length:r*H},(e,t)=>Math.floor(t/H))),fill:k(e,`s2`,o)},update:e=>{i.write(Float32Array.from(t.centers.flatMap(([t,n])=>[...A(t,n),e.radius*G]))),a.write(Float32Array.from(t.centers.flatMap(t=>[...j(t,e.radius)])))}}},rebuildOn:[`level`,`mode`],interaction:{kind:`reach`,hint:`set the cap radius`,rest:({toUnit:e,inputs:t})=>{let[n,r]=t.centers[0];return e([n+U.default/W,r])},resolve(e,{inputs:t,outputs:n,values:r,peek:i}){t.held=E(t.index,e.x,e.y,t.held);let[a,o]=t.centers[t.held],s=Math.hypot((e.x-a)*W,e.y-o),c=C(U,s),l=i(n.counts);return{values:{radius:c},glyphs:[{shape:`line`,from:[a,o],to:[e.x,e.y]},{shape:`dot`,at:[a,o]},{shape:`crosshair`,at:[e.x,e.y],size:6}],readout:`cap radius ${c.toFixed(2)}° · ${l?l[t.held*2]:`…`} cells at level ${r.level}`}}},draw:({outputs:e})=>[{layer:`polygons`,positions:e.fill.boundaries,indices:e.fill.indices,owners:e.fill.owners,values:e.sources,triangleCount:e.fill.triangleCount,alpha:.7},{layer:`triangles`,positions:e.fill.boundaries,triangles:e.fill.wireframe,count:e.fill.wireframeCount,radius:1,color:[1,1,1,.5]},{layer:`lines`,context:!0,segments:e.outline,radius:1.8}]});export{K as default};