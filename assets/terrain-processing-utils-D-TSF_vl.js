import{r as e,s as t}from"./kernel-C3tfQOn4.js";import{E as n,O as r,T as i}from"./gpu-contributor-utils-BIwLQON5.js";import{a,i as o}from"./graph-chunk-dispatch-zBt8k2-o.js";import{l as s}from"./gpu-command-graph-BdzdmV6u.js";import{r as c,u as l}from"./wgsl-common-DfyXiYB0.js";import{r as u,s as d}from"./raster-grid-VB0AM_GY.js";import{a as f}from"./raster-kernel-utils-DcIN7G5I.js";import{c as p,o as m}from"./wgsl-kernel-nodes-DagJklH2.js";function h(e){return e.buffer.defaultBuffer?r(e.buffer.defaultBuffer):e.buffer}function g(e,n,r,a){let o=new Map;for(let[e,i]of[[n,!1],[r,!0]])for(let n of e){if(!a&&n.length===0)continue;let e=t(n),r=a?e.offset:n.byteOffset,s=a?r+e.size:r+(n.length-1)*n.byteStride+n.rowByteLength,c=h(n),l=o.get(c)??{reads:[],writes:[]};(i?l.writes:l.reads).push({start:r,end:s}),o.set(c,l)}for(let t of o.values()){if(a){i(e,t.reads.length===0||t.writes.length===0,`input and output aligned binding usage scopes must not overlap one physical buffer: read-only and writable storage require separate buffers in the same dispatch`);continue}let n=t.reads.sort((e,t)=>e.start-t.start),r=t.writes.sort((e,t)=>e.start-t.start),o=0;for(let t of r){for(;o<n.length&&n[o].end<=t.start;)o++;i(e,o===n.length||t.end<=n[o].start,`input and output logical byte ranges must not overlap, including separately imported physical aliases`)}}}function _(e,t,n){let r={};for(let e of new Set(t.map(e=>e.format))){let n=t.find(t=>t.format===e),i=n.buffer.graph.createDataView(n.buffer,{format:e,length:0}),a=t.map(t=>t.format===e?t:i),o=a.reduce((e,t)=>e+t.length,0);r[e]=new s({id:e,name:e,format:e,length:o,valueLength:o,stride:1,byteStride:n.byteStride,rowByteLength:n.rowByteLength,data:a})}o(e,r);for(let t of n)o(e,Object.fromEntries(t.writes.map((e,t)=>[`output[${t}]`,e])),{strictAlignedBindings:!0}),g(e,t.reads??[],t.writes,!0)}function v(e,t,r,i=[{reads:t,writes:r}]){for(let i of[...t,...r])n(i,[i.format],e);g(e,t,r,!1),_(e,r,i)}function y(e,t,n){a(t,e,Object.fromEntries(n.map((e,t)=>[`view[${t}]`,e])))}function b(e,t,n=`read`){return f(e,t,n===`read_write`)}var x=`${l}${c}
fn infinity() -> f32 { var bits = 0x7f800000u; return bitcast<f32>(bits); }`;function S(e,t=`uniform`,n=!0){let r=`fn getTerrainCoordinateDifference(first: u32, second: u32) -> f32 {
 return select(-f32(second - first), f32(first - second), first >= second);
}`;if(t===`uniform`){let[t,i,a,o]=[e.affine[0],e.affine[1],e.affine[3],e.affine[4]].map(p),s=n?p(Math.abs(e.affine[0]*e.affine[4]-e.affine[1]*e.affine[3])):``;return`${r}
fn getGroundEdgeLength(cell: u32, other: u32) -> f32 {
 let delta = vec2<f32>(getTerrainCoordinateDifference(other % GRID_WIDTH, cell % GRID_WIDTH), getTerrainCoordinateDifference(other / GRID_WIDTH, cell / GRID_WIDTH));
 return length(vec2<f32>(${t} * delta.x + ${i} * delta.y, ${a} * delta.x + ${o} * delta.y));
}
${n?`fn getGroundCellArea(cell: u32) -> f32 { return ${s}; }`:``}`}let i=u(e);return`${r}
fn getGroundCellSizeAtRow(rowCoordinate: f32) -> vec2<f32> {
 let edge = mix(${p(i.northEdge)}, ${p(i.southEdge)}, rowCoordinate / f32(GRID_HEIGHT));
 ${d(t)}
 return vec2<f32>(${p(i.cellSize[0])}, ${p(i.cellSize[1])}) * groundScale;
}
fn getGroundEdgeLength(cell: u32, other: u32) -> f32 {
 let cellRow = f32(cell / GRID_WIDTH); let otherRow = f32(other / GRID_WIDTH);
 let ground = getGroundCellSizeAtRow((cellRow + otherRow + 1.0) * 0.5);
 let columns = abs(getTerrainCoordinateDifference(other % GRID_WIDTH, cell % GRID_WIDTH));
 let rows = abs(getTerrainCoordinateDifference(other / GRID_WIDTH, cell / GRID_WIDTH));
 return length(vec2<f32>(ground.x * columns, ground.y * rows));
}
${n?`fn getGroundCellArea(cell: u32) -> f32 {
 let ground = getGroundCellSizeAtRow(f32(cell / GRID_WIDTH) + 0.5);
 return ground.x * ground.y;
}`:``}`}function C(e,t,n){i(e,!t||t.length===n,`validity must hold one uint32 word per grid cell (nonzero is valid)`)}function w(t,n){let{validity:r}=n;if(!r)return{values:n.values,nodes:[]};let i=e(t,`${n.id}-masked`,`float32`,n.values.length);return{values:i,nodes:[m(t,{id:`${n.id}-mask`,operation:n.operation,variant:`validity-mask`,invocationCount:n.values.length,declarations:x,bindings:[b(`source`,n.values),b(`validity`,r),b(`masked`,i,`read_write`)],body:`masked[maskedOffset + index] = select(getNaN(), source[sourceOffset + index], validity[validityOffset + index] != 0u);`})]}}function T(e){return Math.max(1,Math.ceil(Math.log2(Math.max(e,1))))}function E(e,t){let n=t.ports.filter(e=>e.view);return n.length===0?[]:[m(e,{id:`${t.id}-status-ports`,operation:t.operation,variant:`status-ports`,invocationCount:1,bindings:[b(`source`,t.source),...n.map(e=>b(e.name,e.view,`read_write`))],body:n.map(e=>`${e.name}[${e.name}Offset] = ${e.expression};`).join(`
  `)})]}function D(e,t,n=`output.status`){for(let[r,a]of Object.entries(t))a&&i(e,a.length===1&&a.format===`uint32`,`${n}.${r} must be one uint32 row`)}export{w as a,b as c,E as i,D as l,y as n,S as o,v as r,T as s,x as t,C as u};