import{r as e,s as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-reduction-tNgOVQ_C.js";import{n as r}from"./point-neighbor-grid-index-D0wS1y1I.js";import{u as i}from"./wgsl-common-DfyXiYB0.js";import{n as a}from"./spatial-analysis-packed-kernel-CV3ZCYPQ.js";import{l as o}from"./gpu-command-graph-BdzdmV6u.js";import{O as s,T as c,c as l,f as u,v as d}from"./gpu-contributor-utils-BIwLQON5.js";import{a as f,i as p}from"./graph-chunk-dispatch-zBt8k2-o.js";import{o as m}from"./recipe-utils-CICB7qiS.js";function h(e,n,r,i,u){let d={},m=(e,t)=>{if(e&&typeof e==`object`){if(e instanceof o||`buffer`in e&&`format`in e&&`byteOffset`in e)d[t]=e;else for(let[n,r]of Object.entries(e))m(r,`${t}.${n}`)}};if(m(i,`output`),a(e,n,d),f(e,n,{...r,...d}),u){let e=e=>e.buffer.defaultBuffer?s(e.buffer.defaultBuffer):e.buffer,t=Object.entries(d).flatMap(([e,t])=>l(t).map(t=>({name:e,view:t})));for(let[i,{name:a,view:o}]of t.entries()){if(o.length===0)continue;let s=o.byteOffset+(o.length-1)*o.byteStride+o.rowByteLength,u=[...t.slice(i+1),...Object.entries(r).flatMap(([e,t])=>t?l(t).map(t=>({name:e,view:t})):[])];for(let t of u){if(t.view.length===0||e(o)!==e(t.view))continue;let r=t.view.byteOffset+(t.view.length-1)*t.view.byteStride+t.view.rowByteLength;c(n,o.byteOffset>=r||t.view.byteOffset>=s,`${a} and ${t.name} logical byte ranges must not overlap`)}}for(let e of u){let t=Object.fromEntries(e.filter(e=>d[e]).map(e=>[e,d[e]]));p(n,t);for(let[i,a]of Object.entries(t))for(let t of e)r[t]&&p(n,{[i]:a,[t]:r[t]})}return}p(n,d);let h=e=>e.buffer.defaultBuffer?s(e.buffer.defaultBuffer):e.buffer;for(let e of Object.values(d).flatMap(e=>l(e))){let i=t(e);for(let a of Object.values(r).flatMap(e=>e?l(e):[])){if(h(e)!==h(a))continue;let r=t(a);c(n,i.offset>=r.offset+r.size||r.offset>=i.offset+i.size,`output aligned binding ranges must not overlap inputs`)}}}function g(e,t,n,r,i){d(e,t,n,r,i),u(t,`${n} allocation byteLength (maxBufferSize)`,r,0,e.limits.maxBufferSize)}var _=256,v=4;function y(e){let t=Math.min(_,Math.max(1,Math.ceil(Math.sqrt(e)))),n=v+t*t+1;return{dimension:t,rowOffset:n,byteLength:(n+e)*4}}function b(t,a){let{id:o,positions:s}=a,c=s.length,{dimension:l,rowOffset:u}=y(c),d=l*l;h(t,o,{positions:s,count:a.count},[]),g(t.device,o,`point grid`,(u+c)*4,`reduce positions length`),g(t.device,o,`positions`,c*8,`reduce positions length`);let f=[`minimum-x`,`minimum-y`,`maximum-x`,`maximum-y`].map(n=>e(t,`${o}-${n}-values`,`float32`,c)),p=f.map((n,r)=>e(t,`${o}-extent-${r}`,`float32`,1));t.add(m(t,{id:`${o}-coordinates`,operation:o,invocationCount:c,declarations:i,bindings:[{name:`positions`,view:s,type:`f32`,access:`read`},...a.count?[{name:`count`,view:a.count,type:`u32`,access:`read`}]:[],...f.map((e,t)=>({name:`coordinate${t}`,view:e,type:`f32`,access:`read_write`}))],body:`let horizontal=positions[positionsOffset+2u*index];
let vertical=positions[positionsOffset+2u*index+1u];
let valid=isFiniteValue(horizontal)&&isFiniteValue(vertical)${a.count?`&&index<count[countOffset]`:``};
coordinate0[coordinate0Offset+index]=select(3.402823e38,horizontal,valid);
coordinate1[coordinate1Offset+index]=select(3.402823e38,vertical,valid);
coordinate2[coordinate2Offset+index]=select(-3.402823e38,horizontal,valid);
coordinate3[coordinate3Offset+index]=select(-3.402823e38,vertical,valid);`})),f.forEach((e,r)=>t.add(new n({id:`${o}-extent-reduction-${r}`,input:e,output:p[r],operation:r<2?`min`:`max`}).getCommandNodes(t)));let _=e(t,`${o}-packed`,`uint32`,u+c);t.add(m(t,{id:`${o}-bounds`,operation:o,invocationCount:1,bindings:[...p.map((e,t)=>({name:`extent${t}`,view:e,type:`f32`,access:`read`})),{name:`pointGrid`,view:_,type:`u32`,access:`read_write`}],body:`let lower=vec2<f32>(extent0[extent0Offset],extent1[extent1Offset]);
let upper=vec2<f32>(extent2[extent2Offset],extent3[extent3Offset]);
let populated=all(lower<=upper);
let minimum=select(vec2<f32>(0.0),lower,populated);
let span=select(vec2<f32>(1.0),max(0.5*upper-0.5*lower,vec2<f32>(1.0e-20)),populated);
pointGrid[pointGridOffset]=bitcast<u32>(minimum.x);
pointGrid[pointGridOffset+1u]=bitcast<u32>(minimum.y);
pointGrid[pointGridOffset+2u]=bitcast<u32>(span.x);
pointGrid[pointGridOffset+3u]=bitcast<u32>(span.y);`}));let b={dimension:l,rowOffset:u,data:_},S=e(t,`${o}-cell-keys`,`uint32`,c),C=e(t,`${o}-source-rows`,`uint32`,c);t.add(m(t,{id:`${o}-keys`,operation:o,invocationCount:c,declarations:i+x(b),bindings:[{name:`positions`,view:s,type:`f32`,access:`read`},...a.count?[{name:`count`,view:a.count,type:`u32`,access:`read`}]:[],{name:`pointGrid`,view:_,type:`u32`,access:`read`},{name:`keys`,view:S,type:`u32`,access:`read_write`},{name:`rows`,view:C,type:`u32`,access:`read_write`}],body:`let point=vec2<f32>(positions[positionsOffset+2u*index],positions[positionsOffset+2u*index+1u]);
let valid=isFiniteValue(point.x)&&isFiniteValue(point.y)${a.count?`&&index<count[countOffset]`:``};
let cell=getUrbanPointGridCell(point);
keys[keysOffset+index]=select(${d}u,u32(cell.y)*${l}u+u32(cell.x),valid);
rows[rowsOffset+index]=index;`}));let w=r(t,{id:o,rowCount:c,cellKeys:S,sourceRows:C,cellCount:d});return t.add(w.nodes),t.add(m(t,{id:`${o}-pack`,operation:o,invocationCount:Math.max(c,d+1),bindings:[{name:`offsets`,view:w.cellOffsets,type:`u32`,access:`read`},{name:`rows`,view:w.sortedRows,type:`u32`,access:`read`},{name:`pointGrid`,view:_,type:`u32`,access:`read_write`}],body:`if(index<${d+1}u){pointGrid[pointGridOffset+${v}u+index]=offsets[offsetsOffset+index];}
if(index<${c}u){pointGrid[pointGridOffset+${u}u+index]=rows[rowsOffset+index];}`})),b}function x(e){return`
fn getUrbanPointGridMinimum()->vec2<f32>{return vec2<f32>(bitcast<f32>(pointGrid[pointGridOffset]),bitcast<f32>(pointGrid[pointGridOffset+1u]));}
fn getUrbanPointGridHalfSpan()->vec2<f32>{return vec2<f32>(bitcast<f32>(pointGrid[pointGridOffset+2u]),bitcast<f32>(pointGrid[pointGridOffset+3u]));}
fn getUrbanPointGridCell(point:vec2<f32>)->vec2<i32>{
 let unit=(0.5*point-0.5*getUrbanPointGridMinimum())/getUrbanPointGridHalfSpan();
 return vec2<i32>(clamp(floor(unit*${e.dimension}.0),vec2<f32>(0.0),vec2<f32>(${e.dimension-1}.0)));
}
fn getUrbanPointGridStart(cell:vec2<i32>)->u32{return pointGrid[pointGridOffset+${v}u+u32(cell.y)*${e.dimension}u+u32(cell.x)];}
fn getUrbanPointGridEnd(cell:vec2<i32>)->u32{return pointGrid[pointGridOffset+5u+u32(cell.y)*${e.dimension}u+u32(cell.x)];}
fn getUrbanPointGridRow(slot:u32)->u32{return pointGrid[pointGridOffset+${e.rowOffset}u+slot];}
`}function S(e,t,n,r,i,a=``){return`let gridLower=max(vec2<i32>(0),getUrbanPointGridCell(${t}-vec2<f32>(${n}))-vec2<i32>(1));
let gridUpper=min(vec2<i32>(${e.dimension-1}),getUrbanPointGridCell(${t}+vec2<f32>(${n}))+vec2<i32>(1));
for(var gridRow=gridLower.y;gridRow<=gridUpper.y;gridRow++){
 for(var gridColumn=gridLower.x;gridColumn<=gridUpper.x;gridColumn++){
  let gridCell=vec2<i32>(gridColumn,gridRow);
  for(var gridSlot=getUrbanPointGridStart(gridCell);gridSlot<getUrbanPointGridEnd(gridCell);gridSlot++){
   ${a}
   let ${r}=getUrbanPointGridRow(gridSlot);
   ${i}
  }
 }
}`}function C(e,t){return`let gridCenter=getUrbanPointGridCell(site);
for(var gridRadius=0;gridRadius<${e.dimension};gridRadius++){
 let gridLower=max(vec2<i32>(0),gridCenter-vec2<i32>(gridRadius));
 let gridUpper=min(vec2<i32>(${e.dimension-1}),gridCenter+vec2<i32>(gridRadius));
 for(var gridRow=gridLower.y;gridRow<=gridUpper.y;gridRow++){
  for(var gridColumn=gridLower.x;gridColumn<=gridUpper.x;gridColumn++){
   let gridCell=vec2<i32>(gridColumn,gridRow);
   if(max(abs(gridCell.x-gridCenter.x),abs(gridCell.y-gridCenter.y))!=gridRadius||size==0u){continue;}
   var maximumSquared=0.0;
   for(var vertex=0u;vertex<size;vertex++){let offset=polygon[vertex]-site;maximumSquared=max(maximumSquared,dot(offset,offset));}
   // Work in half-distances to avoid overflowing the extent subtraction.
   let halfSpan=getUrbanPointGridHalfSpan();
   let lower=(0.5*getUrbanPointGridMinimum()-0.5*site)+halfSpan*(vec2<f32>(gridCell)/${e.dimension}.0);
   let upper=lower+halfSpan/${e.dimension}.0;
   // Coordinate-scale rounding slack makes pruning conservative at grid boundaries.
   let slack=1.0e-5*(abs(site)+abs(getUrbanPointGridMinimum())+halfSpan)+vec2<f32>(1.0e-10);
   let separation=max(vec2<f32>(0.0),max(lower,-upper)-slack);
   if(dot(separation,separation)>maximumSquared*(1.0+1.0e-5)){continue;}
   for(var gridSlot=getUrbanPointGridStart(gridCell);gridSlot<getUrbanPointGridEnd(gridCell);gridSlot++){
    let other=getUrbanPointGridRow(gridSlot);
    ${t}
   }
  }
 }
}`}export{C as a,y as i,x as n,h as o,S as r,g as s,b as t};