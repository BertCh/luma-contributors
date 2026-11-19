import{o as e,r as t,t as n}from"./kernel-C3tfQOn4.js";import{r}from"./gpu-command-graph-BdzdmV6u.js";import{t as i}from"./gpu-dispatch-utils-CbRzFAgT.js";import{t as a}from"./hash-uint32-wL4xFYeV.js";import{t as o}from"./gpu-sort-xRm8l5yn.js";import{a as s,o as c}from"./wgsl-kernel-nodes-DagJklH2.js";import{n as l}from"./sorted-segment-offsets-ChmOtA9F.js";import{a as ee}from"./sorted-segment-sums-Dvmm6OdR.js";import{E as u,S as d,f,p,u as m,v as h,x as g,y as _}from"./gpu-contributor-utils-BIwLQON5.js";import{a as te}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as v}from"./spatial-output-validation-nw5lOmVL.js";import{c as y}from"./spatial-weights-CqWysstN.js";import{l as b}from"./wgsl-common-DfyXiYB0.js";import{a as x,i as S,n as C,o as ne,r as w,s as re,t as ie}from"./regionalization-inputs-CyTCPBba.js";import{n as T}from"./regionalization-command-budget-BFn_8OgY.js";var E=256;function ae(t,a){let o=Math.ceil(a.inputCount/2),s=t.createDataView(a.input.buffer,{format:`float32`,length:a.inputCount}),c=t.createDataView(a.output.buffer,{format:`float32`,length:o}),l=i(`GPUAZPRegions`,o,E,t.device.limits.maxComputeWorkgroupsPerDimension);return r({id:a.id,resources:[{buffer:s,usage:`storage-read`},{buffer:c,usage:`storage-read-write`}],workload:{operation:`GPUAZPRegions`,variant:`deterministic-objective-reduction`,commandCount:1,maximumInvocationCount:o,maximumWorkgroupCount:l.x*l.y*l.z},compile:({device:t})=>{let r=new n(t,{id:`azp-objective-reduction`,source:D,shaderLayout:{bindings:[{name:`valuesIn`,type:`read-only-storage`,group:0,location:0},{name:`valuesOut`,type:`storage`,group:0,location:1}]}});return{encode:({computePass:t,getBuffer:n})=>r.dispatch(t,{bindings:{valuesIn:e(s,n),valuesOut:e(c,n)},x:l.x,y:l.y,z:l.z}),destroy:()=>r.destroy()}}})}var D=`
@group(0) @binding(0) var<storage, read> valuesIn: array<f32>;
@group(0) @binding(1) var<storage, read_write> valuesOut: array<f32>;

@compute @workgroup_size(${E})
fn main(
  @builtin(workgroup_id) workgroupIndex: vec3<u32>,
  @builtin(num_workgroups) workgroupCount: vec3<u32>,
  @builtin(local_invocation_index) lane: u32
) {
  let index = (workgroupIndex.x + workgroupCount.x * (workgroupIndex.y + workgroupCount.y * workgroupIndex.z)) * ${E}u + lane;
  if (index >= arrayLength(&valuesOut)) { return; }
  let left = 2u * index;
  var total = valuesIn[left];
  if (left + 1u < arrayLength(&valuesIn)) { total += valuesIn[left + 1u]; }
  valuesOut[index] = total;
}`,O=256,k=16;function A(e,t,n,r,i,a){for(let[o,s]of[[`AZP attribute table`,n*r*4],[`AZP region records`,i*(r+1+Number(a))*4],[`AZP state`,(a?n+3:2)*4],[`AZP move records`,n*8],[`AZP claims`,i*16],[`AZP region offsets`,(i+1)*4],[`AZP sort histogram`,n>O?Math.ceil(n/O)*k*4:0],[`AZP iteration dispatch`,24]])h(e.device,t,o,s,`reduce rows, columns or regionCapacity`)}function oe(e){return m(e,4)}var se=4e-6,j=256,M=1;function N(e={}){return S(`GPUAZPRegions`,e.seed)}var P={iterationCount:0,converged:1,iterationLimitReached:2,invalidCount:3,length:4},F=class{estimatedCommandNodeCount;id;props;rowCount;columnCount;constructor(e){e=Object.freeze({...e}),this.id=e.id??`azp-regions`,this.props=e,this.rowCount=y(this.id,e.weights),this.columnCount=e.columnCount??1;let{id:t,rowCount:n,columnCount:r}=this,{output:i}=e;if(f(t,`columnCount`,r,1,8),f(t,`regionCapacity`,e.regionCapacity,2,n),f(t,`maximumIterations`,e.maximumIterations,1,256),this.estimatedCommandNodeCount=T(n,e.maximumIterations),p(t,this.estimatedCommandNodeCount),f(t,`minimumRegionSize`,e.minimumRegionSize??1),x(t,e.parameters,M),u(e.values,[`float32`],`${t} values`),e.values.length!==n*r)throw Error(`${t} values length must equal rows * columnCount`);if(!!e.extensiveValues!=!!e.extensiveThreshold)throw Error(`${t} extensiveValues and extensiveThreshold are required together`);if(e.extensiveValues&&e.extensiveThreshold){if(u(e.extensiveValues,[`float32`],`${t} extensiveValues`),u(e.extensiveThreshold,[`float32`],`${t} extensiveThreshold`),e.extensiveValues.length!==n)throw Error(`${t} extensiveValues length must equal the weights row count`);if(e.extensiveThreshold.length<1)throw Error(`${t} extensiveThreshold must hold one value`)}for(let[n,r]of[[`initialLabels`,e.initialLabels],[`output.labels`,i.labels],[`output.status`,i.status]])u(r,[`uint32`],`${t} ${n}`);if(e.initialLabels.length!==n||i.labels.length!==n)throw Error(`${t} label views must equal the weights row count`);if(u(i.objective,[`float32`],`${t} output.objective`),i.objective.length<1)throw Error(`${t} output.objective must hold one value`);if(i.status.length<P.length)throw Error(`${t} output.status must hold ${P.length} words`);_(t,[i.labels,i.objective,i.status],[e.parameters,e.weights.offsets,e.weights.neighbors,e.weights.weights,e.values,e.initialLabels,e.extensiveValues,e.extensiveThreshold]),v(t,i,[[i.labels,i.status],[i.objective,i.status],{read:[i.labels],write:[i.status]},{read:[i.status],write:[i.objective]}])}getCommandNodes(e){let{id:n,props:r,rowCount:i,columnCount:u}=this,{output:f}=r;g(n,e,[r.parameters,r.weights.offsets,r.weights.neighbors,r.weights.weights,r.values,r.initialLabels,r.extensiveValues,r.extensiveThreshold,f.labels,f.objective,f.status]),te(e,n,{parameters:r.parameters,offsets:r.weights.offsets,neighbors:r.weights.neighbors,weights:r.weights.weights,values:r.values,initialLabels:r.initialLabels,extensiveValues:r.extensiveValues,extensiveThreshold:r.extensiveThreshold,...f}),ne(n,r.weights,w(e.device)),re(e,n,r.weights,i),A(e,n,i,u,r.regionCapacity,!!r.extensiveValues);let p=oe(e.device);d(n,`values rows x columnCount`,i*u,p),d(n,`regionCapacity x region record words`,r.regionCapacity*(u+1+ +!!r.extensiveValues),p);let m=!!r.extensiveValues,h=u+1+ +!!m,_=r.standardize!==!1,v=_?t(e,`${n}-table`,`float32`,i*u):r.values,y=C(e,{id:n,operation:`GPUAZPRegions`,weights:r.weights,rowCount:i}),x=t(e,`${n}-component-a`,`uint32`,i),S=t(e,`${n}-regions`,`float32`,r.regionCapacity*h),T=r.minimumRegionSize??1,E=t(e,`${n}-state`,`uint32`,m?i+3:2),D=t(e,`${n}-candidate-delta-a`,`float32`,i),O=t(e,`${n}-candidate-delta-b`,`float32`,i),k=t(e,`${n}-moves`,`uint32`,2*i),M=t(e,`${n}-claims`,`uint32`,4*r.regionCapacity),N=s(e,{id:`${n}-iteration-dispatch`,operation:`GPUAZPRegions`,shapes:[[i],[r.regionCapacity]]}),P=t(e,`${n}-label-sort-keys`,`uint32`,i),F=t(e,`${n}-label-sort-indices`,`uint32`,i),I=t(e,`${n}-label-sorted-keys`,`uint32`,i),L=t(e,`${n}-label-sorted-rows`,`uint32`,i),R=t(e,`${n}-region-offsets`,`uint32`,r.regionCapacity+1),z=`const ROWS: u32 = ${i}u;
const COLUMNS: u32 = ${u}u;
const REGION_CAPACITY: u32 = ${r.regionCapacity}u;
const REGION_STRIDE: u32 = ${h}u;
const MINIMUM_SIZE: u32 = ${T}u;
const NONE: u32 = 0xffffffffu;
const LARGE_DELTA: f32 = 3.402823466e+38;
${b}
fn regionCount(region: u32) -> u32 { return u32(regions[regionsOffset + region * REGION_STRIDE]); }
fn regionSum(region: u32, column: u32) -> f32 { return regions[regionsOffset + region * REGION_STRIDE + 1u + column]; }`,ce=2*Math.ceil(Math.log2(Math.max(i,2)))+8,le=Math.ceil(Math.log2(Math.max(i,2)))+1,B=t(e,`${n}-contiguity-hooks`,`uint32`,i),V=t(e,`${n}-contiguity-changed`,`uint32`,1),H=s(e,{id:`${n}-contiguity-dispatch`,operation:`GPUAZPRegions`,shapes:[[i]]}),U=`const ROWS: u32 = ${i}u;
const REGION_CAPACITY: u32 = ${r.regionCapacity}u;`,W=[c(e,{id:`${n}-contiguity-initialize`,operation:`GPUAZPRegions`,variant:`contiguity-initialize`,invocationCount:i,bindings:[{name:`parent`,view:x,type:`u32`,access:`read_write`},{name:`hooks`,view:B,type:`u32`,access:`read_write`}],body:`parent[parentOffset + index] = index;
  hooks[hooksOffset + index] = index;`}),c(e,{id:`${n}-contiguity-arm`,operation:`GPUAZPRegions`,variant:`contiguity-arm`,invocationCount:1,bindings:[{name:`changed`,view:V,type:`u32`,access:`read_write`},H.binding],body:`changed[changedOffset] = 0u;
  ${H.armWGSL}`})];for(let t=0;t<ce;t++){W.push(c(e,{id:`${n}-contiguity-${t}-hook`,operation:`GPUAZPRegions`,variant:`contiguity-hook`,invocationCount:i,...H.getGate(0),bindings:[{name:`offsets`,view:y.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:y.weights.neighbors,type:`u32`,access:`read`},{name:`labels`,view:f.labels,type:`u32`,access:`read`},{name:`parent`,view:x,type:`u32`,access:`read`},{name:`hooks`,view:B,type:`atomic<u32>`,access:`read_write`}],declarations:U,body:`let label = labels[labelsOffset + index];
  if (label >= REGION_CAPACITY) { return; }
  let own = parent[parentOffset + index];
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ROWS || labels[labelsOffset + neighbor] != label) { continue; }
    let other = parent[parentOffset + neighbor];
    if (other != own) {
      atomicMin(&hooks[hooksOffset + max(own, other)], min(own, other));
    }
  }`}),c(e,{id:`${n}-contiguity-${t}-apply`,operation:`GPUAZPRegions`,variant:`contiguity-apply`,invocationCount:i,...H.getGate(0),bindings:[{name:`parent`,view:x,type:`u32`,access:`read_write`},{name:`hooks`,view:B,type:`u32`,access:`read_write`},{name:`changed`,view:V,type:`atomic<u32>`,access:`read_write`}],body:`let hookedRoot = hooks[hooksOffset + index];
  if (hookedRoot < index) {
    parent[parentOffset + index] = hookedRoot;
    hooks[hooksOffset + index] = index;
    atomicOr(&changed[changedOffset], 1u);
  }`}),c(e,{id:`${n}-contiguity-${t}-finish`,operation:`GPUAZPRegions`,variant:`contiguity-finish-round`,invocationCount:1,bindings:[{name:`changed`,view:V,type:`u32`,access:`read_write`},H.binding],body:`if (changed[changedOffset] == 0u) {
    ${H.disarmWGSL}
  }
  changed[changedOffset] = 0u;`}));for(let r=0;r<le;r++)W.push(c(e,{id:`${n}-contiguity-${t}-jump-${r}`,operation:`GPUAZPRegions`,variant:`contiguity-jump`,invocationCount:i,...H.getGate(0),bindings:[{name:`parent`,view:x,type:`atomic<u32>`,access:`read_write`}],body:`let own = atomicLoad(&parent[parentOffset + index]);
  atomicStore(&parent[parentOffset + index], atomicLoad(&parent[parentOffset + own]));`}))}W.push(c(e,{id:`${n}-initial-contiguity`,operation:`GPUAZPRegions`,variant:`initial-label-contiguity`,invocationCount:i,bindings:[{name:`offsets`,view:y.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:y.weights.neighbors,type:`u32`,access:`read`},{name:`labels`,view:f.labels,type:`u32`,access:`read`},{name:`parent`,view:x,type:`u32`,access:`read`},{name:`sortedRows`,view:L,type:`u32`,access:`read`},{name:`regionOffsets`,view:R,type:`u32`,access:`read`},{name:`status`,view:f.status,type:`atomic<u32>`,access:`read_write`}],declarations:U,body:`let label = labels[labelsOffset + index];
  if (label >= REGION_CAPACITY) { return; }
  let root = parent[parentOffset + index];
  // A row is bad when the trees are not stars or a same-label edge joins different roots: the
  // round bound ran out before convergence, which is reported as invalid input.
  var invalid = select(0u, 1u, parent[parentOffset + root] != root);
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor < ROWS && labels[labelsOffset + neighbor] == label &&
        parent[parentOffset + neighbor] != root) {
      invalid = 1u;
    }
  }
  // Each region's smallest row is its component root; any other root is an extra component.
  if (root == index) {
    let first = sortedRows[sortedRowsOffset + regionOffsets[regionOffsetsOffset + label]];
    invalid += select(0u, 1u, first != index);
  }
  if (invalid > 0u) { atomicAdd(&status[statusOffset + 3u], invalid); }`}));let G=[...y.nodes,..._?[ie(e,{id:`${n}-standardize`,operation:`GPUAZPRegions`,values:r.values,scaled:v,rowCount:i,columnCount:u})]:[],c(e,{id:`${n}-initialize-status`,operation:`GPUAZPRegions`,variant:`initialize-status`,invocationCount:1,bindings:[{name:`state`,view:E,type:`u32`,access:`read_write`},{name:`objective`,view:f.objective,type:`f32`,access:`read_write`},{name:`status`,view:f.status,type:`u32`,access:`read_write`}],body:`state[stateOffset] = 0u;
  objective[objectiveOffset] = 0.0;
  status[statusOffset] = 0u;
  status[statusOffset + 1u] = 0u;
  status[statusOffset + 2u] = 0u;
  status[statusOffset + 3u] = 0u;`}),...m?[c(e,{id:`${n}-pack-extensive`,operation:`GPUAZPRegions`,variant:`pack-extensive`,invocationCount:i+1,bindings:[{name:`extensive`,view:r.extensiveValues,type:`f32`,access:`read`},{name:`threshold`,view:r.extensiveThreshold,type:`f32`,access:`read`},{name:`state`,view:E,type:`u32`,access:`read_write`},{name:`status`,view:f.status,type:`atomic<u32>`,access:`read_write`}],declarations:`const ROWS: u32 = ${i}u;
${b}`,body:`// Words 2..ROWS + 1 hold row extensive values and word ROWS + 2 the threshold, as float bits.
  var value = threshold[thresholdOffset];
  if (index < ROWS) { value = extensive[extensiveOffset + index]; }
  if (!isFiniteFloat(value) || value < 0.0) { atomicAdd(&status[statusOffset + 3u], 1u); }
  state[stateOffset + 2u + index] = bitcast<u32>(value);`})]:[],c(e,{id:`${n}-validate-and-copy`,operation:`GPUAZPRegions`,variant:`parallel-validation`,invocationCount:i,bindings:[{name:`offsets`,view:r.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:r.weights.neighbors,type:`u32`,access:`read`},{name:`values`,view:r.values,type:`f32`,access:`read`},{name:`initialLabels`,view:r.initialLabels,type:`u32`,access:`read`},{name:`labels`,view:f.labels,type:`u32`,access:`read_write`},{name:`status`,view:f.status,type:`atomic<u32>`,access:`read_write`},{name:`sortKeys`,view:P,type:`u32`,access:`read_write`},{name:`sortIndices`,view:F,type:`u32`,access:`read_write`}],declarations:`const ROWS: u32 = ${i}u;
const COLUMNS: u32 = ${u}u;
const REGION_CAPACITY: u32 = ${r.regionCapacity}u;
${b}`,body:`let region = initialLabels[initialLabelsOffset + index];
  labels[labelsOffset + index] = region;
  sortKeys[sortKeysOffset + index] = min(region, REGION_CAPACITY);
  sortIndices[sortIndicesOffset + index] = index;
  var invalidCount = select(0u, 1u, region >= REGION_CAPACITY);
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    invalidCount += select(0u, 1u, neighbors[neighborsOffset + slot] >= ROWS);
  }
  for (var column = 0u; column < COLUMNS; column++) {
    invalidCount += select(0u, 1u, !isFiniteFloat(values[valuesOffset + index * COLUMNS + column]));
  }
  if (invalidCount > 0u) { atomicAdd(&status[statusOffset + 3u], invalidCount); }`}),...new o({id:`${n}-label-sort`,keys:P,values:F,outputKeys:I,outputValues:L,keyBits:ee(r.regionCapacity)}).getCommandNodes(e),...l(e,{id:`${n}-regions`,operation:`GPUAZPRegions`,groupCount:r.regionCapacity,sortedGroupKeys:I,segmentOffsets:R}),...W,c(e,{id:`${n}-region-statistics`,operation:`GPUAZPRegions`,variant:`segmented-region-statistics`,invocationCount:r.regionCapacity,bindings:[{name:`sortedRows`,view:L,type:`u32`,access:`read`},{name:`regionOffsets`,view:R,type:`u32`,access:`read`},{name:`values`,view:v,type:`f32`,access:`read`},{name:`regions`,view:S,type:`f32`,access:`read_write`},...m?[{name:`state`,view:E,type:`u32`,access:`read`}]:[]],declarations:z,body:`let region = index;
  let base = regionsOffset + region * REGION_STRIDE;
  let begin = regionOffsets[regionOffsetsOffset + region];
  let end = regionOffsets[regionOffsetsOffset + region + 1u];
  for (var column = 0u; column < COLUMNS; column++) { regions[base + 1u + column] = 0.0; }
  ${m?`regions[base + 1u + COLUMNS] = 0.0;`:``}
  for (var position = begin; position < end; position++) {
    let row = sortedRows[sortedRowsOffset + position];
    ${m?`regions[base + 1u + COLUMNS] += bitcast<f32>(state[stateOffset + 2u + row]);`:``}
    for (var column = 0u; column < COLUMNS; column++) {
      regions[base + 1u + column] += values[valuesOffset + row * COLUMNS + column];
    }
  }
  regions[base] = f32(end - begin);`})];G.push(c(e,{id:`${n}-initialize-state`,operation:`GPUAZPRegions`,variant:`initialize-state`,invocationCount:1,bindings:[{name:`state`,view:E,type:`u32`,access:`read_write`},{name:`status`,view:f.status,type:`u32`,access:`read`},N.binding],body:`let invalid = status[statusOffset + 3u] > 0u;
  state[stateOffset] = select(0u, 1u, invalid);
  if (invalid) {
    ${N.disarmWGSL}
  } else {
    ${N.armWGSL}
  }`}));let K=`const ROWS: u32 = ${i}u;
const REGION_CAPACITY: u32 = ${r.regionCapacity}u;
const NONE: u32 = 0xffffffffu;`,q=`${a}
fn tieKey(row: u32) -> u32 {
  return hashUint32(row ^ parameters[parametersOffset]);
}`,J=e=>`let gain = moves[movesOffset + 2u * index];
  if (gain == 0u) { return; }
  let source = labels[labelsOffset + index];
  let destination = moves[movesOffset + 2u * index + 1u];
  ${e===`gain`?`atomicMax(&claims[claimsOffset + source], gain);
  atomicMax(&claims[claimsOffset + destination], gain);`:`if (atomicLoad(&claims[claimsOffset + source]) != gain ||
      atomicLoad(&claims[claimsOffset + destination]) != gain) { return; }
  let key = tieKey(index);
  ${e===`key`?`atomicMax(&claims[claimsOffset + REGION_CAPACITY + source], key);
  atomicMax(&claims[claimsOffset + REGION_CAPACITY + destination], key);`:`if (atomicLoad(&claims[claimsOffset + REGION_CAPACITY + source]) != key ||
      atomicLoad(&claims[claimsOffset + REGION_CAPACITY + destination]) != key) { return; }
  // The move holds the largest (gain, key) of both its regions, and the keys of distinct rows
  // differ, so no other move selected this round touches either region.
  atomicStore(&claims[claimsOffset + 2u * REGION_CAPACITY + source], index);
  atomicStore(&claims[claimsOffset + 3u * REGION_CAPACITY + destination], index);
  atomicOr(&state[stateOffset + 1u], 1u);`}`}`,Y=[{name:`moves`,view:k,type:`u32`,access:`read`},{name:`labels`,view:f.labels,type:`u32`,access:`read`},{name:`claims`,view:M,type:`atomic<u32>`,access:`read_write`}];G.push(c(e,{id:`${n}-initialize-claims`,operation:`GPUAZPRegions`,variant:`initialize-claims`,invocationCount:r.regionCapacity,bindings:[{name:`claims`,view:M,type:`u32`,access:`read_write`}],declarations:K,body:`claims[claimsOffset + index] = 0u;
  claims[claimsOffset + REGION_CAPACITY + index] = 0u;
  claims[claimsOffset + 2u * REGION_CAPACITY + index] = NONE;
  claims[claimsOffset + 3u * REGION_CAPACITY + index] = NONE;`}));for(let t=0;t<r.maximumIterations;t++)G.push(c(e,{id:`${n}-iteration-${t}-candidates`,operation:`GPUAZPRegions`,variant:`local-articulation-candidates`,invocationCount:i,...N.getGate(0),bindings:[{name:`offsets`,view:y.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:y.weights.neighbors,type:`u32`,access:`read`},{name:`values`,view:v,type:`f32`,access:`read`},{name:`labels`,view:f.labels,type:`u32`,access:`read`},{name:`regions`,view:S,type:`f32`,access:`read`},{name:`state`,view:E,type:`u32`,access:`read`},{name:`moves`,view:k,type:`u32`,access:`read_write`}],declarations:`${z}
const NOISE_THRESHOLD: f32 = ${se};
const LOCAL_CAPACITY: u32 = 128u;
const LOCAL_HASH_SIZE: u32 = ${j}u;
const MAXIMUM_SCAN_DEGREE: u32 = 64u;
// Local neighborhood of the candidate row: 2064 bytes of private memory,
// independent of the row count. localRows lists the rows, localSlots is an open-addressing hash
// from row to its position in localRows, localStack and localVisited drive the search.
var<private> localRows: array<u32, 128>;
var<private> localSlots: array<u32, ${j}>;
var<private> localStack: array<u32, 128>;
var<private> localVisited: array<u32, 4>;
var<private> localCount: u32;
fn getLocalBucket(vertex: u32) -> u32 { return (vertex * 2654435761u) >> 24u; }
// Position of \`vertex\` in localRows, or NONE.
fn findLocal(vertex: u32) -> u32 {
  var bucket = getLocalBucket(vertex);
  for (var probe = 0u; probe < LOCAL_HASH_SIZE; probe++) {
    let entry = localSlots[bucket];
    if (entry == NONE) { return NONE; }
    if (localRows[entry] == vertex) { return entry; }
    bucket = (bucket + 1u) & (LOCAL_HASH_SIZE - 1u);
  }
  return NONE;
}
// Adds \`vertex\` once. Returns false when the list is full.
fn addLocal(vertex: u32) -> bool {
  var bucket = getLocalBucket(vertex);
  for (var probe = 0u; probe < LOCAL_HASH_SIZE; probe++) {
    let entry = localSlots[bucket];
    if (entry == NONE) {
      if (localCount >= LOCAL_CAPACITY) { return false; }
      localRows[localCount] = vertex;
      localSlots[bucket] = localCount;
      localCount++;
      return true;
    }
    if (localRows[entry] == vertex) { return true; }
    bucket = (bucket + 1u) & (LOCAL_HASH_SIZE - 1u);
  }
  return false;
}
// Local articulation test. Returns true only when region \`source\` minus \`row\` is certainly
// connected; false means "not proven", never "disconnected".
//
// Proof. The source region A is connected (the initial labels are checked, and every applied move
// keeps it so). Let N be the rows of A adjacent to \`row\`. Take any v in A minus row: a path in A
// leads from v to row; its last step before the first visit to row lies in N, so v is connected
// inside A minus row to some member of N. Hence A minus row is connected when all members of N
// lie in one component of A minus row. The search below builds S, N plus the rows of A adjacent
// to a member of N (all different from \`row\`), then explores the subgraph induced by S, which is a
// subgraph of A minus row, from one member of N. Reaching every member of N through it proves that
// they share a component of A minus row. A search that runs out of capacity or meets a neighbor
// list longer than MAXIMUM_SCAN_DEGREE answers false.
fn isRemovalSafe(row: u32, source: u32) -> bool {
  localCount = 0u;
  for (var bucket = 0u; bucket < LOCAL_HASH_SIZE; bucket++) { localSlots[bucket] = NONE; }
  for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor < ROWS && neighbor != row && labels[labelsOffset + neighbor] == source) {
      if (!addLocal(neighbor)) { return false; }
    }
  }
  let ringCount = localCount;
  // A row with no same-region neighbor cannot leave a connected region of two or more rows
  // connected (and cannot exist in one); a single neighbor makes the row a leaf.
  if (ringCount < 2u) { return ringCount == 1u; }
  for (var position = 0u; position < ringCount; position++) {
    let member = localRows[position];
    let begin = offsets[offsetsOffset + member];
    let end = offsets[offsetsOffset + member + 1u];
    if (end - begin > MAXIMUM_SCAN_DEGREE) { return false; }
    for (var slot = begin; slot < end; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor < ROWS && neighbor != row && labels[labelsOffset + neighbor] == source) {
        if (!addLocal(neighbor)) { return false; }
      }
    }
  }
  for (var word = 0u; word < 4u; word++) { localVisited[word] = 0u; }
  localVisited[0] = 1u;
  localStack[0] = 0u;
  var stackSize = 1u;
  var reachedRing = 1u;
  while (stackSize > 0u) {
    stackSize--;
    let current = localRows[localStack[stackSize]];
    let begin = offsets[offsetsOffset + current];
    let end = offsets[offsetsOffset + current + 1u];
    if (end - begin > MAXIMUM_SCAN_DEGREE) { return false; }
    for (var slot = begin; slot < end; slot++) {
      let found = findLocal(neighbors[neighborsOffset + slot]);
      if (found != NONE && (localVisited[found >> 5u] & (1u << (found & 31u))) == 0u) {
        localVisited[found >> 5u] |= 1u << (found & 31u);
        localStack[stackSize] = found;
        stackSize++;
        if (found < ringCount) { reachedRing++; }
      }
    }
    if (reachedRing == ringCount) { return true; }
  }
  return false;
}`,body:`let row = index;
  let source = labels[labelsOffset + row];
  let sourceCount = regionCount(source);
  var movable = sourceCount > MINIMUM_SIZE && state[stateOffset] == 0u;
  ${m?`if (movable) {
    // The rest of the source region must still reach the extensive threshold.
    let rowExtensive = bitcast<f32>(state[stateOffset + 2u + row]);
    movable = regions[regionsOffset + source * REGION_STRIDE + 1u + COLUMNS] - rowExtensive >=
      bitcast<f32>(state[stateOffset + 2u + ROWS]);
  }`:``}
  var bestDelta = LARGE_DELTA;
  var bestScale = 0.0;
  var bestDestination = NONE;
  // Cheap exact delta first: only an improving move pays for the removal test below.
  if (movable) {
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) { continue; }
      let destination = labels[labelsOffset + neighbor];
      if (destination == source) { continue; }
      let destinationCount = regionCount(destination);
      var delta = 0.0;
      var scale = 0.0;
      for (var column = 0u; column < COLUMNS; column++) {
        let value = values[valuesOffset + row * COLUMNS + column];
        let sourceSum = regionSum(source, column);
        let destinationSum = regionSum(destination, column);
        let sourceBefore = sourceSum * sourceSum / f32(sourceCount);
        let destinationBefore = destinationSum * destinationSum / f32(destinationCount);
        let sourceAfter = (sourceSum - value) * (sourceSum - value) / f32(sourceCount - 1u);
        let destinationAfter = (destinationSum + value) * (destinationSum + value) / f32(destinationCount + 1u);
        delta += sourceBefore + destinationBefore - sourceAfter - destinationAfter;
        scale += sourceBefore + destinationBefore + sourceAfter + destinationAfter;
      }
      if (delta < bestDelta || (delta == bestDelta && destination < bestDestination)) {
        bestDelta = delta; bestScale = scale; bestDestination = destination;
      }
    }
  }
  var gainBits = 0u;
  // delta is the gain's negative: the sum of squares after minus before.
  if (bestDestination != NONE && bestDelta < -NOISE_THRESHOLD * bestScale && isRemovalSafe(row, source)) {
    gainBits = bitcast<u32>(-bestDelta);
  } else {
    bestDestination = NONE;
  }
  // Positive float32 bit patterns order like the values, so the claim stages compare bits.
  moves[movesOffset + 2u * row] = gainBits;
  moves[movesOffset + 2u * row + 1u] = bestDestination;`}),c(e,{id:`${n}-iteration-${t}-claim-gain`,operation:`GPUAZPRegions`,variant:`claim-gain`,invocationCount:i,...N.getGate(0),bindings:Y,declarations:K,body:J(`gain`)}),c(e,{id:`${n}-iteration-${t}-claim-key`,operation:`GPUAZPRegions`,variant:`claim-key`,invocationCount:i,...N.getGate(0),bindings:[...Y,{name:`parameters`,view:r.parameters,type:`u32`,access:`read`}],declarations:`${K}\n${q}`,body:J(`key`)}),c(e,{id:`${n}-iteration-${t}-select`,operation:`GPUAZPRegions`,variant:`select-moves`,invocationCount:i,...N.getGate(0),bindings:[...Y,{name:`parameters`,view:r.parameters,type:`u32`,access:`read`},{name:`state`,view:E,type:`atomic<u32>`,access:`read_write`}],declarations:`${K}\n${q}`,body:J(`winner`)}),c(e,{id:`${n}-iteration-${t}-apply`,operation:`GPUAZPRegions`,variant:`apply-selected-moves`,invocationCount:r.regionCapacity,...N.getGate(1),bindings:[{name:`moves`,view:k,type:`u32`,access:`read`},{name:`claims`,view:M,type:`atomic<u32>`,access:`read_write`},{name:`values`,view:v,type:`f32`,access:`read`},{name:`labels`,view:f.labels,type:`u32`,access:`read_write`},{name:`regions`,view:S,type:`f32`,access:`read_write`},...m?[{name:`state`,view:E,type:`u32`,access:`read`}]:[]],declarations:z,body:`let region = index;
  let leavingRow = atomicLoad(&claims[claimsOffset + 2u * REGION_CAPACITY + region]);
  let joiningRow = atomicLoad(&claims[claimsOffset + 3u * REGION_CAPACITY + region]);
  let base = regionsOffset + region * REGION_STRIDE;
  if (leavingRow != NONE) {
    regions[base] -= 1.0;
    ${m?`regions[base + 1u + COLUMNS] -= bitcast<f32>(state[stateOffset + 2u + leavingRow]);`:``}
    for (var column = 0u; column < COLUMNS; column++) {
      regions[base + 1u + column] -= values[valuesOffset + leavingRow * COLUMNS + column];
    }
    labels[labelsOffset + leavingRow] = moves[movesOffset + 2u * leavingRow + 1u];
  }
  if (joiningRow != NONE) {
    regions[base] += 1.0;
    ${m?`regions[base + 1u + COLUMNS] += bitcast<f32>(state[stateOffset + 2u + joiningRow]);`:``}
    for (var column = 0u; column < COLUMNS; column++) {
      regions[base + 1u + column] += values[valuesOffset + joiningRow * COLUMNS + column];
    }
  }
  atomicStore(&claims[claimsOffset + region], 0u);
  atomicStore(&claims[claimsOffset + REGION_CAPACITY + region], 0u);
  atomicStore(&claims[claimsOffset + 2u * REGION_CAPACITY + region], NONE);
  atomicStore(&claims[claimsOffset + 3u * REGION_CAPACITY + region], NONE);`}),c(e,{id:`${n}-iteration-${t}-finish`,operation:`GPUAZPRegions`,variant:`finish-round`,invocationCount:1,bindings:[{name:`state`,view:E,type:`u32`,access:`read_write`},{name:`status`,view:f.status,type:`u32`,access:`read_write`},N.binding],body:`if (state[stateOffset] != 0u) { return; }
  if (state[stateOffset + 1u] == 0u) {
    state[stateOffset] = 1u;
    ${N.disarmWGSL}
    return;
  }
  state[stateOffset + 1u] = 0u;
  status[statusOffset] += 1u;`}));G.push(c(e,{id:`${n}-objective-rows`,operation:`GPUAZPRegions`,variant:`parallel-objective`,invocationCount:i,bindings:[{name:`labels`,view:f.labels,type:`u32`,access:`read`},{name:`values`,view:v,type:`f32`,access:`read`},{name:`regions`,view:S,type:`f32`,access:`read`},{name:`deltaOut`,view:D,type:`f32`,access:`read_write`},{name:`status`,view:f.status,type:`u32`,access:`read`}],declarations:z,body:`if (status[statusOffset + 3u] > 0u) { deltaOut[deltaOutOffset + index] = 0.0; return; }
    let region = labels[labelsOffset + index];
    let count = f32(regionCount(region));
    var contribution = 0.0;
    for (var column = 0u; column < COLUMNS; column++) {
      let difference = values[valuesOffset + index * COLUMNS + column] - regionSum(region, column) / count;
      contribution += difference * difference;
    }
  deltaOut[deltaOutOffset + index] = contribution;`}));let X=i,Z=D,Q=O,$=0;for(;X>1;){let t=Math.ceil(X/2);G.push(ae(e,{id:`${n}-objective-reduce-${$}`,input:Z,output:Q,inputCount:X})),X=t,[Z,Q]=[Q,Z],$++}return G.push(c(e,{id:`${n}-objective-publish`,operation:`GPUAZPRegions`,variant:`objective-publish`,invocationCount:1,bindings:[{name:`objectiveIn`,view:Z,type:`f32`,access:`read`},{name:`objectiveOut`,view:f.objective,type:`f32`,access:`read_write`},{name:`status`,view:f.status,type:`u32`,access:`read`}],body:`objectiveOut[objectiveOutOffset] = select(objectiveIn[objectiveInOffset], 0.0, status[statusOffset + 3u] > 0u);`}),c(e,{id:`${n}-finalize`,operation:`GPUAZPRegions`,variant:`finalize`,invocationCount:1,bindings:[{name:`state`,view:E,type:`u32`,access:`read`},{name:`status`,view:f.status,type:`u32`,access:`read_write`}],declarations:`const MAXIMUM_ITERATIONS: u32 = ${r.maximumIterations}u;`,body:`let invalid = status[statusOffset + 3u] > 0u;
  status[statusOffset + 1u] = select(0u, 1u, !invalid && state[stateOffset] != 0u);
  status[statusOffset + 2u] = select(0u, 1u, !invalid && state[stateOffset] == 0u && status[statusOffset] == MAXIMUM_ITERATIONS);`})),G}};export{A as i,P as n,N as r,F as t};