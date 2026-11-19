import{c as e,o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-sort-xRm8l5yn.js";import{d as i,u as a}from"./raster-band-utils-BcTxc8_Y.js";import{D as o,S as s,f as c,p as l,v as u}from"./gpu-contributor-utils-BIwLQON5.js";import{u as d}from"./raster-grid-VB0AM_GY.js";import{o as f,t as p}from"./fixed-point-accumulator-Ce6W3HVY.js";import{n as m,r as h}from"./parallel-merge-tree-DHFVp61C.js";import{t as g}from"./gpu-scan-B4NNTfF9.js";import{r as _}from"./terrain-flow-tracing-BfyLX2kv.js";import{a as v,c as y,i as b,l as x,n as S,o as C,r as w,t as T,u as E}from"./terrain-processing-utils-D-TSF_vl.js";import{t as D}from"./define-run-B-brkCe4.js";import{it as O}from"./entry-preview-D70TdSbK.js";import{a as k,d as A,l as j}from"./terrain-Bwc0vRX3.js";function M(e,r){let{id:i,hierarchy:a,elevation:o,sortedCells:s,tree:c,levels:l}=r,u=o.length,d=u*9,f=`GPUTerrainDepressionHierarchy`,p=`${_(a.grid.width,a.grid.height)}
${T}
fn activationRank(node: u32) -> u32 {
 if (node < ${u}u) { return node; }
 return (node - ${u}u) / 8u;
}`,m=n(e,`${i}-ranks`,`uint32`,u),{nodes:v,createWordView:b,appendKernel:x,appendJumpNodes:S,parents:C}=h(e,{id:`${i}-activation`,operation:f,nodeCount:d,boundedSplicing:!0,spliceMaximumWork:r.spliceMaximumWork,edgeCount:u*8,declarations:p,edgeBindings:[y(`order`,s),y(`ranks`,m),y(`elevation`,o)],edgeBody:`let rank = index / 8u; let cell = order[orderOffset + rank];
if (!isFiniteValue(elevation[elevationOffset + cell])) { return; }
let neighbor = getFlowReceiver(cell, 1u << (index % 8u));
if (neighbor == NO_CELL) { return; }
let neighborRank = ranks[ranksOffset + neighbor];
if (neighborRank >= rank) { return; }
let firstEndpoint = rank; let secondEndpoint = neighborRank; let node = ${u}u + index;`});v.unshift(t(e,{id:`${i}-activation-ranks`,operation:f,invocationCount:u,bindings:[y(`order`,s),y(`ranks`,m,`read_write`)],body:`ranks[ranksOffset + order[orderOffset + index]] = index;`}));let w=b(`tops`,d);x(`tops-init`,d,[y(`parents`,C),y(`tops`,w,`read_write`)],`let parent = parents[parentsOffset + index];
var next = index;
if (parent != NONE && activationRank(parent) == activationRank(index)) { next = parent; }
tops[topsOffset + index] = next;`,p),S(`tops`,d,1,Math.ceil(Math.log2(d))+1,w,`let next = table[tableOffset + index]; let jumped = table[tableOffset + next];
scratch[scratchOffset + index] = jumped;
if (jumped != next) { atomicOr(&pending[pendingOffset], 1u); }`);let E=b(`cell-parents`,u),D=b(`child-counts`,u),O=b(`first-children`,u);x(`cells-init`,u,[y(`tops`,w),y(`parents`,C),y(`cellParents`,E,`read_write`),y(`childCounts`,D,`read_write`),y(`firstChildren`,O,`read_write`)],`let parent = parents[parentsOffset + tops[topsOffset + index]];
var parentRank = NONE;
if (parent != NONE) { parentRank = activationRank(parent); }
cellParents[cellParentsOffset + index] = parentRank;
childCounts[childCountsOffset + index] = 0u; firstChildren[firstChildrenOffset + index] = NONE;`,p),x(`cell-children`,u,[y(`cellParents`,E),{name:`childCounts`,view:D,type:`atomic<u32>`,access:`read_write`},{name:`firstChildren`,view:O,type:`atomic<u32>`,access:`read_write`}],`let parent = cellParents[cellParentsOffset + index];
if (parent != NONE) {
 atomicAdd(&childCounts[childCountsOffset + parent], 1u);
 atomicMin(&firstChildren[firstChildrenOffset + parent], index);
}`);let k=b(`flags`,u),A=b(`offsets`,u),j=b(`links`,u*2);x(`links-init`,u,[y(`order`,s),y(`elevation`,o),y(`childCounts`,D),y(`firstChildren`,O),y(`cellParents`,E),y(`flags`,k,`read_write`),y(`links`,j,`read_write`)],`let valid = isFiniteValue(elevation[elevationOffset + order[orderOffset + index]]);
let event = valid && childCounts[childCountsOffset + index] != 1u;
flags[flagsOffset + index] = u32(event);
links[linksOffset + index * 2u] = select(firstChildren[firstChildrenOffset + index], index, event || !valid);
links[linksOffset + index * 2u + 1u] = cellParents[cellParentsOffset + index];`,p),S(`links`,u,2,Math.ceil(Math.log2(u))+1,j,`let owner = table[tableOffset + index * 2u];
let jumpedOwner = table[tableOffset + owner * 2u];
var parent = table[tableOffset + index * 2u + 1u];
if (parent != NONE && flags[flagsOffset + parent] == 0u) { parent = table[tableOffset + parent * 2u + 1u]; }
scratch[scratchOffset + index * 2u] = jumpedOwner;
scratch[scratchOffset + index * 2u + 1u] = parent;
if (jumpedOwner != owner || (parent != NONE && flags[flagsOffset + parent] == 0u)) { atomicOr(&pending[pendingOffset], 1u); }`,``,[y(`flags`,k)]),v.push(...new g({id:`${i}-activation-scan`,input:k,output:A,mode:`exclusive`}).getCommandNodes(e));let M=b(`metadata`,u*2),N=b(`ancestors`,u);return x(`metadata-init`,u,[y(`metadata`,M,`read_write`),y(`ancestors`,N,`read_write`),y(`links`,j),y(`flags`,k)],`metadata[metadataOffset + index * 2u] = NONE; metadata[metadataOffset + index * 2u + 1u] = NONE;
ancestors[ancestorsOffset + index] = select(NONE, links[linksOffset + index * 2u + 1u], flags[flagsOffset + index] != 0u);`),x(`metadata-cells`,u,[y(`order`,s),y(`elevation`,o),y(`outlets`,a.outlets),y(`links`,j),{name:`metadata`,view:M,type:`atomic<u32>`,access:`read_write`}],`let cell = order[orderOffset + index]; if (!isFiniteValue(elevation[elevationOffset + cell])) { return; }
let owner = links[linksOffset + index * 2u];
atomicMin(&metadata[metadataOffset + owner * 2u], index);
if (outlets[outletsOffset + cell] != 0u) { atomicMin(&metadata[metadataOffset + owner * 2u + 1u], cell); }`,p),S(`metadata`,u,1,Math.ceil(Math.log2(u))+1,N,`var parent = table[tableOffset + index];
if (parent != NONE) {
 atomicMin(&metadata[metadataOffset + parent * 2u], atomicLoad(&metadata[metadataOffset + index * 2u]));
 atomicMin(&metadata[metadataOffset + parent * 2u + 1u], atomicLoad(&metadata[metadataOffset + index * 2u + 1u]));
 parent = table[tableOffset + parent];
}
scratch[scratchOffset + index] = parent;
if (parent != NONE) { atomicOr(&pending[pendingOffset], 1u); }`,``,[{name:`metadata`,view:M,type:`atomic<u32>`,access:`read_write`}]),x(`record`,u,[y(`order`,s),y(`elevation`,o),y(`flags`,k),y(`offsets`,A),y(`links`,j),y(`metadata`,M),y(`tree`,c,`read_write`),y(`levels`,l,`read_write`)],`if (flags[flagsOffset + index] == 0u) { return; }
let destination = offsets[offsetsOffset + index];
let parent = links[linksOffset + index * 2u + 1u];
var parentNode = NONE; var spill = infinity();
if (parent != NONE) { parentNode = offsets[offsetsOffset + parent]; spill = elevation[elevationOffset + order[orderOffset + parent]]; }
tree[treeOffset + destination * 4u] = parentNode;
tree[treeOffset + destination * 4u + 1u] = order[orderOffset + metadata[metadataOffset + index * 2u]];
tree[treeOffset + destination * 4u + 2u] = metadata[metadataOffset + index * 2u + 1u];
tree[treeOffset + destination * 4u + 3u] = 1u;
levels[levelsOffset + destination * 2u] = elevation[elevationOffset + order[orderOffset + index]];
levels[levelsOffset + destination * 2u + 1u] = spill;`,p),x(`ownership`,u,[y(`order`,s),y(`elevation`,o),y(`links`,j),y(`offsets`,A),y(`flags`,k),y(`ownership`,a.output.cellNodes,`read_write`),y(`status`,a.output.status,`read_write`)],`let cell = order[orderOffset + index];
var owner = NONE;
if (isFiniteValue(elevation[elevationOffset + cell])) { owner = offsets[offsetsOffset + links[linksOffset + index * 2u]]; }
ownership[ownershipOffset + cell] = owner;
if (index == ${u-1}u) {
 let required = offsets[offsetsOffset + index] + flags[flagsOffset + index];
 status[statusOffset] = min(required, ${a.nodeCapacity}u);
 status[statusOffset + 1u] = required; status[statusOffset + 2u] = u32(required > ${a.nodeCapacity}u);
}`,p),v}var N=238609294;function P(e){let t=[e.nodes,e.levels,e.cellNodes,e.curves,e.status];for(let n of[e.ports?.status.requiredCount,e.ports?.status.overflow])n&&t.push(n);return t}var F=class{props;id;estimatedCommandNodeCount;constructor(e){this.props=e,this.id=e.id??`terrain-depression-hierarchy`;let t=a(this.id,e.grid);s(this.id,`grid cell count`,t,N),this.estimatedCommandNodeCount=286+2*m(t*9)+3*(Math.ceil(Math.log2(t*9))+1)+6*(Math.ceil(Math.log2(t))+1),l(this.id,this.estimatedCommandNodeCount),d(this.id,e.cellSizeMode),i(this.id,e.grid,e.cellSizeMode),o(this.id,`elevation`,e.elevation,t),o(this.id,`outlets`,e.outlets,t),o(this.id,`output.cellNodes`,e.output.cellNodes,t),E(this.id,e.validity,t),c(this.id,`nodeCapacity`,e.nodeCapacity),o(this.id,`output.nodes`,e.output.nodes,e.nodeCapacity*4),o(this.id,`output.levels`,e.output.levels,e.nodeCapacity*2),o(this.id,`output.curves`,e.output.curves,e.nodeCapacity*e.stages.length*2),o(this.id,`output.status`,e.output.status,3),x(this.id,{requiredCount:e.output.ports?.status.requiredCount,overflow:e.output.ports?.status.overflow},`output.ports.status`),w(this.id,[e.elevation,e.outlets,e.stages,...e.validity?[e.validity]:[]],P(e.output),[{reads:e.validity?[]:[e.elevation],writes:[e.output.cellNodes,e.output.status]},{reads:[e.output.status],writes:[e.output.nodes,e.output.levels]},...e.stages.length?[{reads:[...e.validity?[]:[e.elevation],e.output.cellNodes,e.output.nodes,e.stages,e.output.status],writes:[e.output.curves]}]:[],{reads:[e.output.status],writes:[e.output.ports?.status.requiredCount,e.output.ports?.status.overflow].filter(e=>!!e)}])}getCommandNodes(i){let{id:a,props:o}=this,s=o.elevation.length;S(a,i,[o.elevation,...o.validity?[o.validity]:[],o.outlets,o.stages,...P(o.output)]),u(i.device,a,`activation scratch`,s*18*4,`reduce grid dimensions`);let c=v(i,{id:a,operation:`GPUTerrainDepressionHierarchy`,values:o.elevation,validity:o.validity}),l=c.values,d=(o.cellSizeMode??`uniform`)===`uniform`,m=n(i,`${a}-tree`,`uint32`,s*4),h=n(i,`${a}-levels`,`float32`,s*2),g=n(i,`${a}-keys`,`uint32`,s),x=n(i,`${a}-cells`,`uint32`,s),w=n(i,`${a}-sorted-keys`,`uint32`,s),E=n(i,`${a}-sorted-cells`,`uint32`,s),D=Math.max(1,o.nodeCapacity*o.stages.length*2),O=n(i,`${a}-curve-sums-middle`,`uint32`,D),k=n(i,`${a}-curve-sums-high`,`uint32`,D),A=f(s),j=[...c.nodes,t(i,{id:`${a}-order-keys`,operation:`GPUTerrainDepressionHierarchy`,invocationCount:s,bindings:[y(`elevation`,l),y(`keys`,g,`read_write`),y(`cells`,x,`read_write`)],declarations:T,body:`let value = elevation[elevationOffset + index];
let bits = bitcast<u32>(select(value, 0.0, value == 0.0));
keys[keysOffset + index] = select(0xffffffffu, select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u), isFiniteValue(value));
cells[cellsOffset + index] = index;`}),...new r({id:`${a}-order`,keys:g,values:x,outputKeys:w,outputValues:E}).getCommandNodes(i),...M(i,{id:a,hierarchy:o,elevation:l,sortedCells:E,tree:m,levels:h}),t(i,{id:`${a}-publish`,operation:`GPUTerrainDepressionHierarchy`,invocationCount:o.nodeCapacity,bindings:[y(`tree`,m),y(`sourceLevels`,h),y(`status`,o.output.status),y(`nodes`,o.output.nodes,`read_write`),y(`levels`,o.output.levels,`read_write`)],declarations:T,body:`let valid = status[statusOffset + 2u] == 0u && index < status[statusOffset];
for (var component = 0u; component < 4u; component++) { nodes[nodesOffset + index * 4u + component] = 0xffffffffu; }
levels[levelsOffset + index * 2u] = getNaN(); levels[levelsOffset + index * 2u + 1u] = getNaN();
nodes[nodesOffset + index * 4u + 3u] = 0u;
if (valid) {
 for (var component = 0u; component < 4u; component++) { nodes[nodesOffset + index * 4u + component] = tree[treeOffset + index * 4u + component]; }
 levels[levelsOffset + index * 2u] = sourceLevels[sourceLevelsOffset + index * 2u]; levels[levelsOffset + index * 2u + 1u] = sourceLevels[sourceLevelsOffset + index * 2u + 1u];
}`})];return j.push(...b(i,{id:a,operation:`GPUTerrainDepressionHierarchy`,source:o.output.status,ports:[{name:`requiredCount`,view:o.output.ports?.status.requiredCount,expression:`source[sourceOffset + 1u]`},{name:`overflow`,view:o.output.ports?.status.overflow,expression:`source[sourceOffset + 2u]`}]})),o.stages.length>0&&j.push(t(i,{id:`${a}-curves`,operation:`GPUTerrainDepressionHierarchy`,invocationCount:o.stages.length,declarations:`${T}
${d?``:`${_(o.grid.width,o.grid.height)}\n${C(o.grid,o.cellSizeMode)}`}
// Areas and volumes are fixed-point sums (fixed-point-accumulator.ts): integer additions in a
// unit set by the largest term of the stage, so no float32 partial sum rounds per cell and the
// compiler has no float identity to fold. A compensated float sum kept its correction in one
// float32 word, which itself rounds once a node holds a few hundred thousand cells.
// 'curves' is the float32 output bound as u32: it holds the low word of each sum until the value
// is decoded, then the bits of the float32 result.
${p}
const MISSING_BITS: u32 = 0x7fc00000u;
fn loadCurveSum(slot: u32) -> vec3<u32> {
 return vec3<u32>(curves[curvesOffset + slot], sumsMiddle[sumsMiddleOffset + slot], sumsHigh[sumsHighOffset + slot]);
}
fn addCurveSum(slot: u32, value: vec3<u32>) {
 let next = addFixedPoint(loadCurveSum(slot), value);
 curves[curvesOffset + slot] = next.x; sumsMiddle[sumsMiddleOffset + slot] = next.y; sumsHigh[sumsHighOffset + slot] = next.z;
}`,bindings:[y(`elevation`,l),y(`ownership`,o.output.cellNodes),y(`nodes`,o.output.nodes),y(`stages`,o.stages),y(`status`,o.output.status),{name:`curves`,view:o.output.curves,type:`u32`,access:`read_write`},y(`sumsMiddle`,O,`read_write`),y(`sumsHigh`,k,`read_write`)],body:`let stage = stages[stagesOffset + index];
let valid = status[statusOffset + 2u] == 0u && isFiniteValue(stage);
let nodeCount = status[statusOffset];
for (var node = 0u; node < ${o.nodeCapacity}u; node++) {
 let slot = (node * ${o.stages.length}u + index) * 2u;
 // A zero sum for the nodes that exist, NaN for the others and for an invalid hierarchy or stage.
 let initial = select(MISSING_BITS, 0u, valid && node < nodeCount);
 for (var component = 0u; component < 2u; component++) {
  curves[curvesOffset + slot + component] = initial;
  sumsMiddle[sumsMiddleOffset + slot + component] = 0u;
  sumsHigh[sumsHighOffset + slot + component] = 0u;
 }
}
if (!valid) { return; }
${d?`let uniformCellArea = ${e(Math.abs(o.grid.affine[0]*o.grid.affine[4]-o.grid.affine[1]*o.grid.affine[3]))};`:``}
// The largest area and volume term of this stage set the two fixed-point units.
var areaExponent = 0u; var volumeExponent = 0u;
for (var cell = 0u; cell < ${s}u; cell++) {
 ${d?`let cellArea = uniformCellArea;`:`let cellArea = getGroundCellArea(cell);`}
 let depth = stage - elevation[elevationOffset + cell];
 if (ownership[ownershipOffset + cell] < nodeCount && depth >= 0.0) {
  areaExponent = max(areaExponent, getFixedPointExponentBits(cellArea));
  volumeExponent = max(volumeExponent, getFixedPointExponentBits(depth * cellArea));
 }
}
let areaUnit = getFixedPointUnitExponent(areaExponent, ${A}u);
let volumeUnit = getFixedPointUnitExponent(volumeExponent, ${A}u);
for (var cell = 0u; cell < ${s}u; cell++) {
 ${d?`let cellArea = uniformCellArea;`:`let cellArea = getGroundCellArea(cell);`}
 let owner = ownership[ownershipOffset + cell];
 let depth = stage - elevation[elevationOffset + cell];
 if (owner < nodeCount && depth >= 0.0) {
  let slot = (owner * ${o.stages.length}u + index) * 2u;
  addCurveSum(slot, quantizeFixedPoint(cellArea, areaUnit));
  addCurveSum(slot + 1u, quantizeFixedPoint(depth * cellArea, volumeUnit));
 }
}
// Parents are created after their children. Each exclusive cell contribution moves once per edge,
// as an exact integer. A node is decoded once every child has been added to it and it has been
// added to its parent: decoding replaces its low word by the float32 bits.
for (var node = 0u; node < nodeCount; node++) {
 let slot = (node * ${o.stages.length}u + index) * 2u;
 let areaSum = loadCurveSum(slot);
 let volumeSum = loadCurveSum(slot + 1u);
 let parent = nodes[nodesOffset + node * 4u];
 if (parent < nodeCount) {
  let destination = (parent * ${o.stages.length}u + index) * 2u;
  addCurveSum(destination, areaSum);
  addCurveSum(destination + 1u, volumeSum);
 }
 curves[curvesOffset + slot] = bitcast<u32>(getFixedPointFloat(areaSum, areaUnit));
 curves[curvesOffset + slot + 1u] = bitcast<u32>(getFixedPointFloat(volumeSum, volumeUnit));
}`})),j}},I=4294967295,L=[.04,.04,.96,.3];function R(e){let t=k(e,40),n=new Uint32Array(t.cellCount);for(let e=0;e<t.cellCount;e++){let r=e%t.columns,i=Math.floor(e/t.columns),a=r===0||i===0||i===t.rows-1||r===t.columns-1;n[e]=+!!a}return{...t,borderMask:n,focus:{node:-2,source:null}}}function z(e,t){let[n,r]=e.subPitCenters[0];return t([n/e.columns*(e.columns*e.cellSize),(1-r/e.rows)*(e.rows*e.cellSize)])}var B=D({parameters:O,caption:`Water depth when each depression is filled to its spill level; the chart is the hovered depression over the stages.`,createInputs:e=>R(e),build(e,n,r){let i=Math.round(r.stages),a=Math.round(r.nodeCapacity),[o,s]=n.elevationRange,c=Float32Array.from({length:i},(e,t)=>o+(t+1)/i*(s-o)),l=e.upload(`dem`,`float32`,n.elevation),u=e.upload(`outlets`,`uint32`,n.borderMask),d=e.allocate(`nodes`,`uint32`,a*4),f=e.allocate(`levels`,`float32`,a*2),p=e.allocate(`cell-nodes`,`uint32`,n.cellCount),m=e.allocate(`curves`,`float32`,a*i*2);e.graph.add(new F({grid:n.grid,elevation:l.view,outlets:u.view,stages:e.upload(`stages`,`float32`,c).view,nodeCapacity:a,output:{nodes:d.view,levels:f.view,cellNodes:p.view,curves:m.view,status:e.allocate(`status`,`uint32`,3).view}}));let h=e.allocate(`fill-depth`,`float32`,n.cellCount);e.graph.add({getCommandNodes:e=>[t(e,{id:`fill-depth`,operation:`fill-depth`,bindings:[{name:`owners`,view:p.view,type:`u32`,access:`read`},{name:`levels`,view:f.view,type:`f32`,access:`read`},{name:`ground`,view:l.view,type:`f32`,access:`read`},{name:`depth`,view:h.view,type:`f32`,access:`read_write`}],invocationCount:n.cellCount,body:`let node = owners[ownersOffset + index];
  let spill = levels[levelsOffset + min(node, ${a-1}u) * 2u + 1u];
  let missing = bitcast<f32>(0x7fc00000u | (index & 0u));
  depth[depthOffset + index] = select(missing, spill - ground[groundOffset + index], node < ${a}u && spill < 1.0e20);`})]});let g=e.allocate(`highlight`,`uint32`,n.cellCount),_=e.allocate(`curve`,`float32`,i*2),[v,y]=n.subPitCenters[0],b=Math.floor(y)*n.columns+Math.floor(v);return e.graph.add({getCommandNodes:e=>[t(e,{id:`default-curve`,operation:`default-curve`,bindings:[{name:`owners`,view:p.view,type:`u32`,access:`read`},{name:`nodes`,view:d.view,type:`u32`,access:`read`},{name:`curves`,view:m.view,type:`f32`,access:`read`},{name:`chart`,view:_.view,type:`f32`,access:`read_write`}],invocationCount:i*2,body:`let kind = index / ${i}u;
  let stage = index % ${i}u;
  var node = owners[ownersOffset + ${b}u];
  if (node < ${a}u) {
    let parent = nodes[nodesOffset + node * 4u];
    if (parent < ${a}u) { node = parent; }
  }
  var maximum = 1.0e-9;
  for (var entry = 0u; entry < ${a*i}u; entry++) {
    let value = curves[curvesOffset + entry * 2u + kind];
    if (value == value && value < 1.0e20) { maximum = max(maximum, value); }
  }
  var own = 0.0;
  if (node < ${a}u) {
    let value = curves[curvesOffset + (node * ${i}u + stage) * 2u + kind];
    if (value == value && value < 1.0e20) { own = value; }
  }
  chart[chartOffset + index] = own / maximum;`})]}),{outputs:{elevation:l,nodes:d,levels:f,cellNodes:p,curves:m,fillDepth:h,highlight:g,curve:_}}},rebuildOn:[`nodeCapacity`,`stages`],interaction:{kind:`probe`,hint:`hover a depression`,rest:({inputs:e,toUnit:t})=>z(e,t),resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a,toUnit:o}){let s=a(n.cellNodes),c=a(n.nodes),l=a(n.levels),u=a(n.curves);if(!s||!c||!l||!u)return{};let d=Math.round(r.nodeCapacity),f=Math.round(r.stages),p=j(t,i,e.x,e.y),[m,h]=z(t,o),g=Math.hypot(e.unitX-m,e.unitY-h)<.003,_=!g&&s[p.index]<d?s[p.index]:-1,v=s[p.index]<d?s[p.index]:-1;g&&v>=0&&c[v*4]<d&&(v=c[v*4]);let y=_*1e5+v*100+f;if(t.focus.node!==y||t.focus.source!==s){t.focus={node:y,source:s};let e=_<0?I:c[_*4],r=(e,t)=>{let n=e;for(let e=0;e<d&&n<d;e++){if(n===t)return!0;n=c[n*4]}return!1};n.highlight.write(Uint32Array.from(s,t=>_<0||t>=d?0:r(t,_)?2:e<d&&r(t,e)?1:0));let i=1e-9,a=1e-9;for(let e=0;e<d*f;e++)i=Math.max(i,u[e*2]||0),a=Math.max(a,u[e*2+1]||0);let o=new Float32Array(f*2);if(v>=0)for(let e=0;e<f;e++)o[e]=(u[(v*f+e)*2]||0)/i,o[f+e]=(u[(v*f+e)*2+1]||0)/a;n.curve.write(o)}if(_<0)return{focus:-1,spotlight:null};let b=l[_*2+1],x=c[_*4],S=u[(_*f+f-1)*2+1]||0;return{focus:_,spotlight:null,glyphs:[{shape:`crosshair`,at:[e.x,e.y]}],readout:`depression ${_} · spill ${Number.isFinite(b)?b.toFixed(3):`open`} · ${x<d?`parent ${x}`:`root`} · volume ${S.toFixed(2)}`}}},draw:({inputs:e,outputs:t})=>[A(e,t.elevation),{layer:`raster`,values:t.fillDepth,columns:e.columns,rows:e.rows,ramp:`sequential`,range:[0,.06],alpha:.85},{layer:`raster`,values:t.highlight,columns:e.columns,rows:e.rows,categories:[[0,0,0,0],[1,1,1,.22],[1,.92,.35,.5]]},{layer:`chart`,kind:`bars`,values:t.curve,ramp:`sequential`,range:[0,1],rectangle:L}]});export{B as default};