import{r as e}from"./gpu-command-graph-BdzdmV6u.js";import{t}from"./gpu-dispatch-utils-CbRzFAgT.js";import{o as n,t as r}from"./kernel-C3tfQOn4.js";import{a as i,i as a,n as o,o as s,r as c}from"./graph-chunk-dispatch-zBt8k2-o.js";import{E as l,S as u,T as d,c as f,f as p,p as m,y as h}from"./gpu-contributor-utils-BIwLQON5.js";import{o as g,s as _}from"./cell-index-wgsl-Cw7syfdA.js";import{t as v}from"./cell-keys-Dl1tIQ_Z.js";import{t as y}from"./h3-neighbor-wgsl-DdYw-iG5.js";import{i as b,n as x,t as S}from"./cell-topology-wgsl-gkK5LfSP.js";import{t as C}from"./s2-neighbor-wgsl-ByEyc4ST.js";import{t as w}from"./s2-topology-wgsl-ByzVxuin.js";var T=256,E=4294967295,D=16777216,O=`GPUCellTopology`,k=4096;function A(e){return 3*e*(e+1)+1}function j(e,t){if(d(`GPUCellTopology`,[`quadbin`,`h3`,`s2`].includes(e),`family must be 'quadbin', 'h3' or 's2'`),t.type===`disk`||t.type===`ring`){let{k:n}=t;if(!Number.isInteger(n)||n<0||n>8)throw Error(`${t.type} k must be an integer in [0, 8]`);return t.type===`disk`?e===`h3`?A(n):e===`s2`?2*n*(n+1)+1:(2*n+1)**2:n===0?1:(e===`h3`?6:e===`s2`?4:8)*n}if(g(`GPUCellTopology`,e,t.resolution),t.type===`parent`)return 1;g(`GPUCellTopology`,e,t.inputResolution);let n=t.resolution-t.inputResolution;if(n<0)throw Error(`children resolution must not be below inputResolution`);let r=(e===`h3`?7:4)**n;if(r>4096)throw Error(`children stride ${r} exceeds ${k}; reduce the resolution depth`);return r}var M=class{id;props;stride;estimatedCommandNodeCount;chunkPlan;recordChunks;constructor(e){this.id=e.id??`cell-topology`,this.props=e;let t=this.id;d(t,[`quadbin`,`h3`,`s2`].includes(e.family),`family must be 'quadbin', 'h3' or 's2'`),d(t,e.wordOrder===void 0||[`little-endian`,`high-low`].includes(e.wordOrder),`wordOrder must be 'little-endian' or 'high-low'`);try{this.stride=j(e.family,e.operation)}catch(e){throw Error(`${t} ${e.message}`)}let{output:n}=e;d(t,!n.distances||e.operation.type===`disk`,`output.distances is only available for the disk operation`),this.chunkPlan=c(t,{cells:e.cells,mask:e.mask,counts:n.counts}),p(t,`cells row count`,this.chunkPlan.totalLength);for(let e of this.chunkPlan.chunks)e.views.cells&&l(e.views.cells,[`uint32x2`],`${t} cells chunk ${e.chunkIndex}`),u(t,`cells chunk components`,e.length*2,E),e.views.mask&&l(e.views.mask,[`uint32`],`${t} mask chunk ${e.chunkIndex}`),u(t,`mask chunk components`,e.length,E),e.views.counts&&l(e.views.counts,[`uint32`],`${t} counts chunk ${e.chunkIndex}`),u(t,`counts chunk components`,e.length,E);let r={};if(r.outputCells=n.cells?f(n.cells):[],n.cells){d(t,n.cells.length===this.chunkPlan.totalLength*this.stride,`output.cells length must equal input rows * record stride`),d(t,r.outputCells.length===this.chunkPlan.chunks.length,`output.cells must match input chunk topology`);for(let e of this.chunkPlan.chunks){let n=r.outputCells[e.chunkIndex];l(n,[`uint32x2`],`${t} outputCells chunk ${e.chunkIndex}`),d(t,n.length===e.length*this.stride,`output.cells chunk ${e.chunkIndex} length must equal input chunk rows * record stride`),u(t,`outputCells chunk components`,n.length*2,E)}}if(r.distances=n.distances?f(n.distances):[],n.distances){d(t,n.distances.length===this.chunkPlan.totalLength*this.stride,`output.distances length must equal input rows * record stride`),d(t,r.distances.length===this.chunkPlan.chunks.length,`output.distances must match input chunk topology`);for(let e of this.chunkPlan.chunks){let n=r.distances[e.chunkIndex];l(n,[`uint32`],`${t} distances chunk ${e.chunkIndex}`),d(t,n.length===e.length*this.stride,`output.distances chunk ${e.chunkIndex} length must equal input chunk rows * record stride`),u(t,`distances chunk components`,n.length*1,E)}}this.recordChunks=r,a(t,e.output),this.estimatedCommandNodeCount=+(this.chunkPlan.chunks.length>0)+this.chunkPlan.nonemptyChunkCount,m(t,this.estimatedCommandNodeCount),h(t,[n.cells,n.distances,n.counts],[e.cells,e.mask])}getCommandNodes(a){let{id:c,props:l,stride:d}=this,{family:f,operation:p,output:m}=l;u(c,`storage bindings`,2+Number(!!l.mask)+Object.values(l.output).filter(Boolean).length,a.device.limits.maxStorageBuffersPerShaderStage),s(a,c,this.chunkPlan),i(a,c,{"output.cells":m.cells,"output.distances":m.distances});let h=l.operation.type===`children`?this.stride:1,g=this.chunkPlan.chunks.map(e=>t(c,Math.max(e.length*h,1),T,a.device.limits.maxComputeWorkgroupsPerDimension)),{slots:E,nodes:k}=o(a,{id:`${c}-chunks`,plan:this.chunkPlan}),M,N=this.chunkPlan.chunks.filter(e=>e.length>0).map(t=>{let i=f===`h3`,a=f===`s2`,o=[{name:`chunkParameters`,view:E[t.chunkIndex],type:`u32`,access:`read`},{name:`cells`,view:t.views.cells,type:`u32`,access:`read`},...l.mask?[{name:`rowMask`,view:t.views.mask,type:`u32`,access:`read`}]:[],{name:`outCells`,view:this.recordChunks.outputCells[t.chunkIndex],type:`u32`,access:`read_write`},...m.distances?[{name:`outDistances`,view:this.recordChunks.distances[t.chunkIndex],type:`u32`,access:`read_write`}]:[],...m.counts?[{name:`outCounts`,view:t.views.counts,type:`u32`,access:`read_write`}]:[]],s=p.type===`disk`||p.type===`ring`,u=s?p.k:0,k=s?i?A(u):j(f,{type:`disk`,k:u}):1,N=p.type===`parent`||p.type===`children`?p.resolution:0,P=p.type===`children`?p.inputResolution:0,F=p.type===`children`?p.resolution-P:0,I=i?`dggs_h3_get_resolution(key)`:a?`dggs_s2_get_level(key)`:`((key.x >> 20u) & 0x1fu)`,L=i?`dggs_h3_is_valid_cell_id(key)`:a?`dggs_s2_is_valid_cell_id(key)`:`cellIsValidQuadbin(key)`,R=`true`;p.type===`parent`?R=`${I} >= TARGET_RESOLUTION`:p.type===`children`&&(R=`${I} == INPUT_RESOLUTION`);let z=e=>`
  let words = vec2u(cells[cellsOffset + 2u * ${e}], cells[cellsOffset + 2u * ${e} + 1u]);
  let key = ${l.wordOrder===`high-low`?`words`:`words.yx`};
  var isValid = ${L} && ${R};
  ${l.mask?`isValid = isValid && rowMask[rowMaskOffset + ${e}] != 0u;`:``}`,B=(e,t,n)=>`
  outCells[outCellsOffset + 2u * ${e}] = ${t}.y;
  outCells[outCellsOffset + 2u * ${e} + 1u] = ${t}.x;
  ${m.distances?`outDistances[outDistancesOffset + ${e}] = ${n??`0u`};`:``}`,V;if(s)V=`${z(`index`)}
  var written = 0u;
  if (isValid) {
    written = ${i||a?`cellTopologyNeighborhood`:`cellTopologyQuadbinNeighborhood`}(key, ${p.type===`ring`});
  }
  for (var j = 0u; j < STRIDE; j++) {
    let slot = index * STRIDE + j;
    if (j < written) {${B(`slot`,`cellTopologyCells[j]`,`cellTopologyDistances[j]`)}
    } else {${B(`slot`,`vec2u(0u)`)}
    }
  }
  ${m.counts?`outCounts[outCountsOffset + index] = written;`:``}`;else if(p.type===`parent`){let e=i?`dggs_h3_get_parent(key, TARGET_RESOLUTION)`:a?`cellTopologyS2GetParent(key, TARGET_RESOLUTION)`:`cellTopologyQuadbinGetParent(key, TARGET_RESOLUTION)`;V=`${z(`index`)}
  var parent = vec2u(0u);
  if (isValid) {
    parent = ${e};
  }${B(`index`,`parent`)}
  ${m.counts?`outCounts[outCountsOffset + index] = select(0u, 1u, isValid);`:``}`}else{let e=i?`cellTopologyH3GetChild(key, TARGET_RESOLUTION, DEPTH, slotInRow)`:a?`cellTopologyS2GetChild(key, TARGET_RESOLUTION, DEPTH, slotInRow)`:`cellTopologyQuadbinGetChild(key, TARGET_RESOLUTION, DEPTH, slotInRow)`,t=i?`cellTopologyH3GetChildCount(key, DEPTH)`:`STRIDE`;V=`let row = index / STRIDE;
  let slotInRow = index % STRIDE;${z(`row`)}
  var child = vec2u(0u);
  if (isValid) {
    child = ${e};
  }${B(`index`,`child`)}
  ${m.counts?`if (slotInRow == 0u) {\n    outCounts[outCountsOffset + row] = select(0u, ${t}, isValid);\n  }`:``}`}let H={id:`${c}-topology`,operation:O,variant:`${f}-${p.type}`,bindings:o,invocationCount:t.length*h,declarations:`const STRIDE: u32 = ${d}u;
const CELL_TOPOLOGY_K: u32 = ${u}u;
const CELL_TOPOLOGY_CAPACITY: u32 = ${k}u;
const TARGET_RESOLUTION: u32 = ${N}u;
const INPUT_RESOLUTION: u32 = ${P}u;
const DEPTH: u32 = ${F}u;
${i||a?_.source:``}
${v}
${i?y:``}
${b}
${i?x:``}
${a?`${C}\n${w}`:``}
${i||a?`
const CELL_TOPOLOGY_NEIGHBOR_COUNT: u32 = ${i?6:4}u;
fn cellTopologyNeighbor(cell: vec2u, direction: u32) -> vec2u {
  return ${i?`cellTopologyH3Neighbor(cell, direction + 1u)`:`s2Neighbor(cell, direction)`};
}
${S}`:``}`,body:V},U=H.bindings.map((e,t)=>`@group(0) @binding(${t}) var<storage, ${e.access}> ${e.name}: array<${e.type}>;`).join(`
`),W={cells:2,rowMask:1,outCells:d*2,outDistances:d*1,outCounts:1},G=H.bindings.filter(e=>e.name!==`chunkParameters`).map(e=>`let ${e.name}Offset = arrayLength(&${e.name}) - rowCount * ${W[e.name]}u;`).join(`
  `),K=M??`${U}
${H.declarations??``}
@compute @workgroup_size(${T})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
  @builtin(num_workgroups) workgroupCount: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32) {
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${D}u) { return; }
  let index = workgroupIndex * ${T}u + localInvocationIndex;
  let rowCount = chunkParameters[1];
  if (index >= rowCount * ${h}u) { return; }
  ${G}
  ${H.body}
}`;M=K;let q=H.bindings,J=H.operation,Y=g[t.chunkIndex];return e({id:`${H.id}-${t.chunkIndex}`,resources:H.bindings.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`})),workload:{operation:H.operation,...H.variant?{variant:H.variant}:{},commandCount:1,maximumWorkgroupCount:Y.x*Y.y*Y.z,maximumInvocationCount:t.length*h},compile:({device:e})=>{let t=new r(e,{id:J,source:K,shaderLayout:{bindings:q.map((e,t)=>({name:e.name,group:0,location:t,type:e.access===`read`?`read-only-storage`:`storage`}))}});return{encode:({computePass:e,getBuffer:r})=>t.dispatch(e,{bindings:Object.fromEntries(q.map(e=>[e.name,n(e.view,r)])),...Y}),destroy:()=>t.destroy()}}})});return[...k,...N]}};export{j as n,M as t};