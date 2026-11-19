import{n as e}from"./raster-storage-preflight-CIxFc-0g.js";import{f as t,g as n,s as r}from"./gpu-contributor-utils-BIwLQON5.js";import{a as i,c as a,g as o,n as s,r as c}from"./raster-band-utils-BcTxc8_Y.js";import{o as l}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as u,i as d,r as f}from"./raster-expansion-utils-AZOLseCG.js";var p=8,m=class{id;props;requiredHalo;constructor(e){e={...e,input:i(e.input)},this.id=e.id??`raster-rank-filter`,this.props=e,o(this.id,e.input,e.grid);let r=u(this.id,e.grid,s(e.input),Object.values(e.output));for(let[t,n]of Object.entries(e.output))f(n,r,this.id,`output.${t}`);t(this.id,`radius`,e.radius,0,p),this.requiredHalo=e.radius,n(this.id,`quantile`,e.quantile??.5,0,1)}getCommandNodes(t){let n=this.props,i=n.grid.width*n.grid.height;e(t,this.id,[...s(n.input),...r(n.output)],{"canonical band planes":i*4});let{id:o,props:u}=this,{grid:f,output:p}=u;a(o,t,u.input,[]),d(o,t,[...s(u.input),...Object.values(p)]);let m=c(t,o,u.input,f,!0);return[...m.nodes,l(t,{id:o,operation:`GPURasterRankFilter`,invocationCount:f.width*f.height,bindings:[{name:`source`,view:m.band.storage.values,type:`f32`,access:`read`},{name:`sourceValidity`,view:m.band.validity,type:`u32`,access:`read`},{name:`values`,view:p.values,type:`f32`,access:`read_write`},{name:`validity`,view:p.validity,type:`u32`,access:`read_write`},...p.changed?[{name:`changed`,view:p.changed,type:`u32`,access:`read_write`}]:[]],body:`var samples: array<f32, ${(2*u.radius+1)**2}>; var sampleCount = 0u;
var valid = sourceValidity[sourceValidityOffset + index] != 0u;
for (var rowOffset = -${u.radius}; rowOffset <= ${u.radius}; rowOffset++) {
 for (var columnOffset = -${u.radius}; columnOffset <= ${u.radius}; columnOffset++) {
  var column = i32(index % ${f.width}u) + columnOffset; var row = i32(index / ${f.width}u) + rowOffset;
  if (column < 0 || row < 0 || column >= ${f.width} || row >= ${f.height}) {
   ${u.borderMode===`nodata`?`${u.noDataPolicy===`ignore`?``:`valid = false;`} continue;`:`column = clamp(column, 0, ${f.width-1}); row = clamp(row, 0, ${f.height-1});`}
  }
  let sampleIndex = u32(row) * ${f.width}u + u32(column);
  if (sourceValidity[sourceValidityOffset + sampleIndex] == 0u) { ${u.noDataPolicy===`ignore`?``:`valid = false;`} continue; }
  let value = source[sourceOffset + sampleIndex]; var insertion = sampleCount;
  loop { if (insertion == 0u || samples[insertion-1u] <= value) { break; } samples[insertion] = samples[insertion-1u]; insertion--; }
  samples[insertion] = value; sampleCount++;
 }
}
valid = valid && sampleCount > 0u;
var result = bitcast<f32>(0x7fc00000u | (index & 0u));
if (valid) { result = samples[min(u32(floor(${(u.quantile??.5).toFixed(9)} * f32(sampleCount))), sampleCount-1u)]; }
values[valuesOffset+index] = result; validity[validityOffset+index] = u32(valid);
${p.changed?`changed[changedOffset+index] = u32(valid && result != source[sourceOffset+index]);`:``}`})]}};export{m as t};