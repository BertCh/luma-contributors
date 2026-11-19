import{t as e}from"./ports-xmdKgL07.js";import{a as t}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as n}from"./validate-outline-output-passes-DIcqH6Pw.js";import{n as r,t as i}from"./refinement-storage-validation-D9BhQGNW.js";import{r as a}from"./kernel-C3tfQOn4.js";import{v as o}from"./spatial-context-DwZEd4EZ.js";import{t as s}from"./gpu-line-split-BIIctsOh.js";import{E as c,f as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{o as d}from"./wgsl-kernel-nodes-DagJklH2.js";import{s as f}from"./hydrography-N8loF-Ds.js";import{t as p}from"./define-run-B-brkCe4.js";import{s as m}from"./pointer-overlay-Qh8qflcA.js";import{r as h}from"./text-DYSgVV-n.js";import{a as g}from"./polygon-set-CHja50mr.js";import{ho as _}from"./entry-preview-D70TdSbK.js";var v=class{id;props;constructor(t){if(this.id=t.id??`line-loop-removal`,o(this.id,t.spatialContext),t={...t,lines:e(this.id,`lines`,t.lines)},this.props=t,c(t.lines.positions,[`float32x2`],this.id),c(t.lines.lineOffsets,[`uint32`],this.id),l(this.id,`lines.lineOffsets.length`,t.lines.lineOffsets.length,2),l(this.id,`intersectionCapacity`,t.intersectionCapacity,1),c(t.output.positions,[`float32x2`],this.id),c(t.output.lineOffsets,[`uint32`],this.id),c(t.output.sourceEdges,[`uint32`],this.id),c(t.output.status,[`uint32x4`],this.id),t.output.lineOffsets.length!==t.lines.lineOffsets.length||t.output.sourceEdges.length!==t.output.positions.length||t.output.status.length!==1)throw Error(`Loop output capacity mismatch`);n(this.id,t.output,[[`positions`,`sourceEdges`],[`lineOffsets`,`status`]]),u(this.id,Object.values(t.output),[t.lines.positions,t.lines.lineOffsets])}getCommandNodes(e){let{props:n,id:o}=this;t(e,o,Object.fromEntries([n.lines.positions,n.lines.lineOffsets,...Object.values(n.output)].map((e,t)=>[`binding.${t}`,e]))),i(e,o,n.lines.positions.length,n.lines.lineOffsets.length-1,n.intersectionCapacity),r(e,o,{stack:(n.lines.positions.length+n.intersectionCapacity*8)*16,lineState:(n.lines.lineOffsets.length-1)*8});let c=n.lines.lineOffsets.length-1,l=c+4*n.intersectionCapacity,u=n.lines.positions.length+8*n.intersectionCapacity,f=a(e,`${o}-split-positions`,`float32x2`,u),p=a(e,`${o}-split-offsets`,`uint32`,l+1),m=a(e,`${o}-split-sources`,`uint32`,l),h=a(e,`${o}-split-edges`,`uint32`,u),g=a(e,`${o}-split-count`,`uint32`,1),_=a(e,`${o}-split-required`,`uint32`,1),v=a(e,`${o}-split-overflow`,`uint32`,1),y=a(e,`${o}-split-candidate-overflow`,`uint32`,1),b=a(e,`${o}-stack`,`uint32x4`,u),x=a(e,`${o}-line-state`,`uint32x2`,c),S=a(e,`${o}-output-start`,`uint32`,c);return[...new s({id:`${o}-split`,lines:n.lines,spatialContext:n.spatialContext,intersectionCapacity:n.intersectionCapacity,output:{pieces:{geometry:{kind:`lines`,positions:f,lineOffsets:p},sourceIds:m,vertexSourceEdges:h,status:{count:g,requiredCount:_,overflow:v,candidateOverflow:y}}}}).getCommandNodes(e),d(e,{id:`${o}-erase`,operation:`GPULineLoopRemoval`,variant:`chronological-erasure`,invocationCount:c,bindings:[{name:`positions`,view:f,type:`f32`,access:`read`},{name:`offsets`,view:p,type:`u32`,access:`read`},{name:`sourceIds`,view:m,type:`u32`,access:`read`},{name:`sourceEdges`,view:h,type:`u32`,access:`read`},{name:`count`,view:g,type:`u32`,access:`read`},{name:`stack`,view:b,type:`u32`,access:`read_write`},{name:`lineState`,view:x,type:`u32`,access:`read_write`}],declarations:`fn findFirstPiece(line: u32) -> u32 {
  var low = 0u; var high = count[countOffset];
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (sourceIds[sourceIdsOffset + middle] < line) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}`,body:`let line = index;
let firstPiece = findFirstPiece(line); let endPiece = findFirstPiece(line + 1u);
let base = offsets[offsetsOffset + firstPiece];
var kept = 0u;
for (var piece = firstPiece; piece < endPiece; piece++) {
  if (sourceIds[sourceIdsOffset + piece] != line) { continue; }
  let first = offsets[offsetsOffset + piece]; let last = offsets[offsetsOffset + piece + 1u];
  for (var vertex = first; vertex < last; vertex++) {
    let current = vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
    var found = kept;
    for (var previous = 0u; previous < kept; previous++) {
      let destination = stackOffset + (base + previous) * 4u;
      let oldPoint = vec2f(bitcast<f32>(stack[destination]), bitcast<f32>(stack[destination + 1u]));
      if (all(current == oldPoint)) { found = previous; break; }
    }
    if (found < kept) { kept = found + 1u; }
    else { kept++; }
    let destination = stackOffset + (base + kept - 1u) * 4u;
    stack[destination] = bitcast<u32>(current.x); stack[destination + 1u] = bitcast<u32>(current.y);
    if (vertex + 1u < last) { stack[destination + 2u] = sourceEdges[sourceEdgesOffset + vertex] & 0x7fffffffu; }
    else if (kept > 1u) { stack[destination + 2u] = stack[destination - 2u]; }
  }
}
if (kept < 2u) { kept = 0u; }
lineState[lineStateOffset + line * 2u] = base; lineState[lineStateOffset + line * 2u + 1u] = kept;`}),d(e,{id:`${o}-publish-offsets`,operation:`GPULineLoopRemoval`,variant:`publish-offsets`,invocationCount:1,bindings:[{name:`lineState`,view:x,type:`u32`,access:`read`},{name:`nodingOverflow`,view:v,type:`u32`,access:`read`},{name:`candidateOverflow`,view:y,type:`u32`,access:`read`},{name:`outputStart`,view:S,type:`u32`,access:`read_write`},{name:`outOffsets`,view:n.output.lineOffsets,type:`u32`,access:`read_write`},{name:`status`,view:n.output.status,type:`u32`,access:`read_write`}],body:`var required = 0u; var written = 0u;
for (var line = 0u; line < ${c}u; line++) {
  let length = lineState[lineStateOffset + line * 2u + 1u];
  outOffsets[outOffsetsOffset + line] = written; required += length;
  // Drop whole lines that would exceed capacity, preserving valid path boundaries.
  if (written + length > ${n.output.positions.length}u) { outputStart[outputStartOffset + line] = 0xffffffffu; continue; }
  outputStart[outputStartOffset + line] = written; written += length;
}
outOffsets[outOffsetsOffset + ${c}u] = written;
status[statusOffset] = written; status[statusOffset + 1u] = required;
status[statusOffset + 2u] = select(0u, 1u, written < required); status[statusOffset + 3u] = nodingOverflow[nodingOverflowOffset] | candidateOverflow[candidateOverflowOffset];`}),d(e,{id:`${o}-publish`,operation:`GPULineLoopRemoval`,variant:`publish`,invocationCount:c,bindings:[{name:`stack`,view:b,type:`u32`,access:`read`},{name:`lineState`,view:x,type:`u32`,access:`read`},{name:`outputStart`,view:S,type:`u32`,access:`read`},{name:`outPositions`,view:n.output.positions,type:`f32`,access:`read_write`},{name:`outEdges`,view:n.output.sourceEdges,type:`u32`,access:`read_write`}],body:`let start = outputStart[outputStartOffset + index];
if (start == 0xffffffffu) { return; }
let base = lineState[lineStateOffset + index * 2u]; let length = lineState[lineStateOffset + index * 2u + 1u];
for (var vertex = 0u; vertex < length; vertex++) {
  let source = stackOffset + (base + vertex) * 4u; let written = start + vertex;
  outPositions[outPositionsOffset + written * 2u] = bitcast<f32>(stack[source]);
  outPositions[outPositionsOffset + written * 2u + 1u] = bitcast<f32>(stack[source + 1u]);
  outEdges[outEdgesOffset + written] = stack[source + 2u];
}`})]}},y=240,b=10,x=.3,S=120,C=m(_,`tangle`);function w(e,t){let n=Float32Array.from(e);if(!t)return n;let r=t.target[0]-e[t.vertex*2],i=t.target[1]-e[t.vertex*2+1];for(let e=-10;e<=b;e++){let a=t.vertex+e;if(a<0||a>=y)continue;let o=.5*(1+Math.cos(Math.PI*e/11));n[a*2]+=r*o,n[a*2+1]+=i*o}return n}function T(e,t,n){return Float32Array.from(e.flatMap(([e,r],i)=>{let a=i/239*8*Math.PI;return[e-t*n*Math.sin(a),r-t*n*Math.cos(a)]}))}var E=p({parameters:_,caption:`Loops erased from a meandering curve in drawing order.`,createInputs:({world:e})=>{let t=f(0,y,e),n=.04*e.width;return{reach:t,radius:n,base:T(t,n,C.default),drag:null}},build(e,t,n){let r=e.upload(`positions`,`float32x2`,T(t.reach,t.radius,n.tangle)),i=e.upload(`line-offsets`,`uint32`,Uint32Array.of(0,y)),a=e.allocate(`cleaned`,`float32x2`,480),o=e.allocate(`cleaned-offsets`,`uint32`,2),s=e.allocate(`source-edges`,`uint32`,480),c=e.allocate(`status`,`uint32x4`,1);return e.graph.add(new v({lines:{kind:`lines`,positions:r.view,lineOffsets:i.view},intersectionCapacity:512,output:{positions:a.view,lineOffsets:o.view,sourceEdges:s.view,status:c.view}})),{outputs:{positions:r,cleaned:a,status:c,ids:e.upload(`ids`,`uint32`,g(480))},update:e=>{t.base=T(t.reach,t.radius,e.tangle),r.write(w(t.base,t.drag))}}},interaction:{kind:`drag`,hint:`pull the curve`,rest:({inputs:e,toUnit:t})=>t([e.base[240],e.base[241]]),resolve(e,{inputs:t,outputs:n,peek:r}){let i=t.drag?.vertex??S,a=1/0;for(let n=0;n<y;n++){let r=Math.hypot(t.base[n*2]-e.x,t.base[n*2+1]-e.y);r<a&&(a=r,i=n)}let o=t.drag?.vertex;o!==void 0&&a>.02&&Math.hypot(t.base[o*2]-e.x,t.base[o*2+1]-e.y)<=x&&(i=o);let s=[e.x,e.y];t.drag={vertex:i,target:s};let c=[t.base[i*2],t.base[i*2+1]],l=r(n.status)?.[0];return{rerun:!0,glyphs:[{shape:`line`,from:c,to:s,dashed:!0},{shape:`crosshair`,at:s}],readout:`${y} -> ${l??`...`} vertices after loop removal`}}},draw:({outputs:e,theme:t})=>[{layer:`paths`,context:!0,nodes:e.positions,ids:e.ids,count:y,radius:2,color:[...t.categories[2],.9]},{layer:`paths`,nodes:e.cleaned,ids:e.ids,count:e.status,radius:2.6},{layer:`text`,...h({corner:`bottom-left`,entries:[{label:`kept by the operation`,swatch:[...t.ink]},{label:`loop removed`,swatch:2}]})}]});export{E as default};