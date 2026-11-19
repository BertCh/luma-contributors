import{c as e,d as t,i as n,o as r,r as i,s as a,t as o,u as s}from"./kernel-C3tfQOn4.js";import{l as c}from"./gpu-command-graph-BdzdmV6u.js";import{t as l}from"./gpu-group-aggregation-CEYb6GJw.js";import{n as u,t as d}from"./gpu-dispatch-utils-CbRzFAgT.js";import{f,i as p}from"./gpu-contributor-utils-BIwLQON5.js";import{_ as m,b as h,o as g,r as _,s as v,x as y}from"./embedding-DKQ2dOol.js";import{t as b}from"./define-run-B-brkCe4.js";import{ii as x}from"./entry-preview-D70TdSbK.js";var S=4294967295,C=256;function w(e,t){let n=d(t.id,t.elementCount,64,t.maxComputeWorkgroupsPerDimension??e.device.limits.maxComputeWorkgroupsPerDimension);e.addComputePass({id:t.id,resources:t.resources,compile:({device:e})=>{let i=new o(e,{id:t.id,source:t.source,shaderLayout:{bindings:Object.keys(t.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:a})=>{let o={};for(let[e,n]of Object.entries(t.bindings))o[e]=r(n,a);let s=o;i.dispatch(e,{bindings:s,x:n.x,y:n.y,z:n.z})},destroy:()=>i.destroy()}}})}function T(e,t,n){return d(e,t,64,n)}function E(e){return u(e,64)}function D(e,n){if(!Number.isSafeInteger(e.dimensions)||e.dimensions<1||e.dimensions>S)throw Error(`${n} dimensions must be a positive uint32 integer`);if(!Number.isSafeInteger(e.rowCount)||e.rowCount<0||e.rowCount>S)throw Error(`${n} row count must be a non-negative uint32 integer`);if(!Array.isArray(e.chunks))throw Error(`${n} chunks must preserve an ordered array of embedding allocations`);let r=0;for(let[i,a]of e.chunks.entries()){let o=`${n} chunk ${i}`;if(t(a.values,[`float32`],`${o} flat values`),!Number.isSafeInteger(a.rowCount)||a.rowCount<0)throw Error(`${o} row count must be a non-negative safe integer`);if(!Number.isSafeInteger(a.rowStride)||a.rowStride<e.dimensions)throw Error(`${o} row stride must contain every embedding dimension`);if(!Number.isSafeInteger(a.byteOffset)||a.byteOffset<0||a.byteOffset%Float32Array.BYTES_PER_ELEMENT!==0||a.byteOffset!==a.values.byteOffset)throw Error(`${o} byte offset must match its aligned flat float32 view`);if(!Number.isSafeInteger(a.sourceRowOffset)||a.sourceRowOffset<0||a.sourceRowOffset+a.rowCount>S)throw Error(`${o} source rows must fit below the reserved invalid uint32 ID`);let c=a.rowCount===0?0:(a.rowCount-1)*a.rowStride+e.dimensions;if(!Number.isSafeInteger(c)||c>a.values.length||a.byteOffset+c*Float32Array.BYTES_PER_ELEMENT>a.values.buffer.byteLength)throw Error(`${o} rows exceed their declared packed flat float32 view`);for(let[e,t]of[[`source-row IDs`,a.sourceRowIds],[`validity flags`,a.validity]])if(t&&(s(t,`${o} ${e}`),t.length<a.rowCount))throw Error(`${o} ${e} must contain one value per source row`);if(r+=a.rowCount,!Number.isSafeInteger(r)||r>S)throw Error(`${n} total chunk row count must fit in uint32`)}if(r!==e.rowCount)throw Error(`${n} row count must match the sum of its source chunk rows`)}function O(e,t,n){let r=t instanceof c?t.data:[t];for(let e of r)s(e,n);if(t instanceof c){if(t.data.length!==e.chunks.length||t.data.some((t,n)=>t.length!==e.chunks[n].rowCount))throw Error(`${n} must preserve the embedding matrix chunk topology`)}else if(t.length!==e.rowCount)throw Error(`${n} must contain one value per embedding row`)}function k(e,t,n,r){return e.createDataView(t.buffer,{format:`uint32`,length:r,byteOffset:t.byteOffset+n*Uint32Array.BYTES_PER_ELEMENT})}function A(e,t,n){return k(e,t instanceof c?t.data[n.chunkIndex]:t,t instanceof c?n.chunkRowOffset:n.logicalRowOffset,n.rowCount)}function j(e,t,n=2**53-1){if(D(t,`GPU embedding matrix`),!Number.isSafeInteger(n)||n<1)throw Error(`GPU embedding tile row count must be a positive integer`);let r=e.device.limits.maxStorageBufferBindingSize,i=t.dimensions*Float32Array.BYTES_PER_ELEMENT,o=Math.floor((r-(C-Uint32Array.BYTES_PER_ELEMENT))/Uint32Array.BYTES_PER_ELEMENT);if(o<1)throw Error(`GPU embedding row metadata exceeds maxStorageBufferBindingSize`);let s=[],c=0;for(let[l,u]of t.chunks.entries()){let d=0;for(;d<u.rowCount;){let f=u.values.byteOffset+d*u.rowStride*Float32Array.BYTES_PER_ELEMENT,p=r-f%256;if(p<i)throw Error(`GPU embedding row exceeds maxStorageBufferBindingSize`);let m=Math.floor((p-i)/(u.rowStride*Float32Array.BYTES_PER_ELEMENT))+1,h=Math.min(u.rowCount-d,n,m,o),g=e.createDataView(u.values.buffer,{format:`float32`,length:(h-1)*u.rowStride+t.dimensions,byteOffset:f});if(a(g).size>r)throw Error(`GPU embedding tile exceeds maxStorageBufferBindingSize`);s.push({chunk:u,chunkIndex:l,chunkRowOffset:d,logicalRowOffset:c+d,sourceRowOffset:u.sourceRowOffset+d,rowCount:h,values:g,...u.sourceRowIds?{sourceRowIds:k(e,u.sourceRowIds,d,h)}:{},...u.validity?{validity:k(e,u.validity,d,h)}:{}}),d+=h}c+=u.rowCount}return s}var M=4294967295,N=10,P=class{id;dataset;clusterCount;centroids;labels;counts;status;maxIterations;seed;registered=!1;constructor(e){if(this.id=e.id??`gpu-k-means`,this.dataset=e.dataset,this.clusterCount=e.clusterCount,this.centroids=e.centroids,this.labels=e.labels,this.counts=e.counts,this.status=e.status,this.maxIterations=e.maxIterations??N,this.seed=e.seed??`evenly-spaced`,D(this.dataset,`${this.id} dataset`),!Number.isSafeInteger(this.clusterCount)||this.clusterCount<1||this.clusterCount>M||!Number.isSafeInteger(this.clusterCount*this.dataset.dimensions)||this.clusterCount*this.dataset.dimensions>M)throw Error(`${this.id} cluster count and centroid component count must fit in uint32`);if(!Number.isSafeInteger(this.maxIterations)||this.maxIterations<1)throw Error(`${this.id} maxIterations must be a positive integer`);if(this.seed!==`evenly-spaced`)throw Error(`${this.id} seed strategy must be evenly-spaced`);if(!Number.isSafeInteger(this.dataset.rowCount)||this.dataset.rowCount>M)throw Error(`${this.id} dataset row count must fit in uint32`);if(t(this.centroids,[`float32`],`${this.id} centroids`),O(this.dataset,this.labels,`${this.id} labels`),s(this.counts,`${this.id} counts`),this.centroids.length<this.clusterCount*this.dataset.dimensions)throw Error(`${this.id} centroids must contain clusterCount * dimensions values`);if(this.counts.length!==this.clusterCount)throw Error(`${this.id} counts must contain exactly clusterCount values`);if(this.status&&(s(this.status,`${this.id} status`),this.status.length<3))throw Error(`${this.id} status must contain iteration, change, and convergence values`);F(this)}addToGraph(e){if(this.registered)throw Error(`${this.id} clustering has already been added to a command graph`);let t=this.dataset.chunks.flatMap(e=>[e.values,...e.validity?[e.validity]:[],...e.sourceRowIds?[e.sourceRowIds]:[]]),n=this.labels instanceof c?this.labels.data:[this.labels],r=[this.centroids,this.counts,...n,...this.status?[this.status]:[]];if([...t,...r].some(t=>t.buffer.graph!==e))throw Error(`${this.id} inputs and outputs must belong to the target graph`);let a=j(e,this.dataset),o=this.clusterCount*this.dataset.dimensions,s=this.status??i(e,`${this.id}-status`,`uint32`,3),u=i(e,`${this.id}-seed-distances`,`uint32`,this.clusterCount),d=i(e,`${this.id}-seed-rows`,`uint32`,this.clusterCount),f=i(e,`${this.id}-centroid-sums`,`float32`,o);I(e,this,s,u,d);for(let[t,n]of a.entries()){let r=A(e,this.labels,n);L(e,`${this.id}-labels-tile-${t}`,r),R(e,this,n,t,u,d),z(e,this,n,t,d)}for(let t=0;t<this.maxIterations;t++){let n=`${this.id}-iteration-${t}`;B(e,n,s);for(let[t,r]of a.entries())V(e,this,r,A(e,this.labels,r),s,`${n}-assign-tile-${t}`);e.add(new l({id:`${n}-counts`,keys:this.labels,output:this.counts})),H(e,`${n}-clear-sums`,f,s);for(let[t,r]of a.entries())U(e,this,r,A(e,this.labels,r),f,s,`${n}-sum-tile-${t}`);W(e,this,f,s,`${n}-centroids`),G(e,`${n}-status`,s)}this.registered=!0}};function F(e){let t=e.labels instanceof c?e.labels.data:[e.labels],r=[e.centroids,e.counts,...e.status?[e.status]:[]];for(let t=0;t<r.length;t++)for(let n=t+1;n<r.length;n++)if(r[t].buffer===r[n].buffer)throw Error(`${e.id} writable outputs must use separate graph buffers`);if(t.some(e=>r.some(t=>e.buffer===t.buffer)))throw Error(`${e.id} labels and aggregate outputs must use separate graph buffers`);let i=e.dataset.chunks.flatMap(e=>[e.values,...e.validity?[e.validity]:[],...e.sourceRowIds?[e.sourceRowIds]:[]]);for(let a of[...r,...t])if(i.some(e=>n(e,a)))throw Error(`${e.id} writable outputs must not overlap source embedding data`)}function I(t,n,r,i,a){let o=n.clusterCount*n.dataset.dimensions,s=Math.max(o,n.clusterCount,3),c=`
@group(0) @binding(0) var<storage, read_write> centroidValues: array<f32>;
@group(0) @binding(1) var<storage, read_write> convergenceState: array<u32>;
@group(0) @binding(2) var<storage, read_write> seedDistances: array<u32>;
@group(0) @binding(3) var<storage, read_write> seedRows: array<u32>;
@compute @workgroup_size(64) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${E(T(n.id,s,t.device.limits.maxComputeWorkgroupsPerDimension))}
  if (index < ${o}u) {
    centroidValues[${e(n.centroids)}u + index] = 0.0;
  }
  if (index < ${n.clusterCount}u) {
    seedDistances[${e(i)}u + index] = 0xffffffffu;
    seedRows[${e(a)}u + index] = 0xffffffffu;
  }
  if (index < 3u) {
    convergenceState[${e(r)}u + index] =
      select(0u, ${n.dataset.rowCount===0?`1u`:`0u`}, index == 2u);
  }
}`;w(t,{id:`${n.id}-initialize`,source:c,resources:[{buffer:n.centroids,usage:`storage-write`},{buffer:r,usage:`storage-write`},{buffer:i,usage:`storage-write`},{buffer:a,usage:`storage-write`}],bindings:{centroidValues:n.centroids,convergenceState:r,seedDistances:i,seedRows:a},elementCount:s})}function L(t,n,r){w(t,{id:n,source:`
@group(0) @binding(0) var<storage, read_write> clusterLabels: array<u32>;
@compute @workgroup_size(64) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${E(T(n,r.length,t.device.limits.maxComputeWorkgroupsPerDimension))}
  if (index < ${r.length}u) {
    clusterLabels[${e(r)}u + index] = 0xffffffffu;
  }
}`,resources:[{buffer:r,usage:`storage-write`}],bindings:{clusterLabels:r},elementCount:r.length})}function R(t,n,r,i,a,o){let s=T(n.id,n.clusterCount,t.device.limits.maxComputeWorkgroupsPerDimension),c=3,l=r.validity?`@group(0) @binding(${c++}) var<storage, read> rowValidity: array<u32>;`:``,u=r.sourceRowIds?`@group(0) @binding(${c++}) var<storage, read> sourceRowIds: array<u32>;`:``,d=r.validity?`if (rowValidity[${e(r.validity)}u + row] == 0u) { continue; }`:``,f=r.sourceRowIds?`if (sourceRowIds[${e(r.sourceRowIds)}u + row] == 0xffffffffu) { continue; }`:``,p=`
@group(0) @binding(0) var<storage, read> embeddingValues: array<f32>;
@group(0) @binding(1) var<storage, read_write> seedDistances: array<u32>;
@group(0) @binding(2) var<storage, read_write> seedRows: array<u32>;
${l}
${u}
@compute @workgroup_size(64) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${E(s)}
  if (index >= ${n.clusterCount}u) { return; }
  let targetRow = min(
    u32(floor(f32(index) * f32(${n.dataset.rowCount}u) / f32(${n.clusterCount}u))),
    ${n.dataset.rowCount-1}u
  );
  var bestDistance = seedDistances[${e(a)}u + index];
  var bestRow = seedRows[${e(o)}u + index];
  for (var row = 0u; row < ${r.rowCount}u; row++) {
    ${d}
    ${f}
    var finiteRow = true;
    for (var dimension = 0u; dimension < ${n.dataset.dimensions}u; dimension++) {
      let value = embeddingValues[
        ${e(r.values)}u + row * ${r.chunk.rowStride}u + dimension
      ];
      if (!(value == value && abs(value) <= 3.402823466e+38)) { finiteRow = false; break; }
    }
    if (!finiteRow) { continue; }
    let logicalRow = ${r.logicalRowOffset}u + row;
    let cyclicDistance = select(
      logicalRow - targetRow,
      ${n.dataset.rowCount}u - targetRow + logicalRow,
      logicalRow < targetRow
    );
    if (cyclicDistance < bestDistance) {
      bestDistance = cyclicDistance;
      bestRow = logicalRow;
    }
  }
  seedDistances[${e(a)}u + index] = bestDistance;
  seedRows[${e(o)}u + index] = bestRow;
}`;w(t,{id:`${n.id}-seed-select-tile-${i}`,source:p,resources:[{buffer:r.values,usage:`storage-read`},{buffer:a,usage:`storage-read-write`},{buffer:o,usage:`storage-read-write`},...r.validity?[{buffer:r.validity,usage:`storage-read`}]:[],...r.sourceRowIds?[{buffer:r.sourceRowIds,usage:`storage-read`}]:[]],bindings:{embeddingValues:r.values,seedDistances:a,seedRows:o,...r.validity?{rowValidity:r.validity}:{},...r.sourceRowIds?{sourceRowIds:r.sourceRowIds}:{}},elementCount:n.clusterCount})}function z(t,n,r,i,a){let o=n.clusterCount*n.dataset.dimensions,s=`
@group(0) @binding(0) var<storage, read> embeddingValues: array<f32>;
@group(0) @binding(1) var<storage, read> seedRows: array<u32>;
@group(0) @binding(2) var<storage, read_write> centroidValues: array<f32>;
@compute @workgroup_size(64) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${E(T(n.id,o,t.device.limits.maxComputeWorkgroupsPerDimension))}
  if (index >= ${o}u) { return; }
  let clusterIndex = index / ${n.dataset.dimensions}u;
  let sourceRow = seedRows[${e(a)}u + clusterIndex];
  if (sourceRow < ${r.logicalRowOffset}u ||
      sourceRow >= ${r.logicalRowOffset+r.rowCount}u) { return; }
  let tileRow = sourceRow - ${r.logicalRowOffset}u;
  let dimension = index % ${n.dataset.dimensions}u;
  centroidValues[${e(n.centroids)}u + index] =
    embeddingValues[${e(r.values)}u + tileRow * ${r.chunk.rowStride}u + dimension];
}`;w(t,{id:`${n.id}-seed-copy-tile-${i}`,source:s,resources:[{buffer:r.values,usage:`storage-read`},{buffer:a,usage:`storage-read`},{buffer:n.centroids,usage:`storage-write`}],bindings:{embeddingValues:r.values,seedRows:a,centroidValues:n.centroids},elementCount:o})}function B(t,n,r){let i=`
@group(0) @binding(0) var<storage, read_write> convergenceState: array<u32>;
@compute @workgroup_size(64) fn main(
  @builtin(global_invocation_id) globalInvocationId: vec3<u32>
) {
  if (globalInvocationId.x == 0u &&
      convergenceState[${e(r)}u + 2u] == 0u) {
    convergenceState[${e(r)}u + 1u] = 0u;
  }
}`;w(t,{id:`${n}-reset-changes`,source:i,resources:[{buffer:r,usage:`storage-read-write`}],bindings:{convergenceState:r},elementCount:1})}function V(t,n,r,i,a,o){let s=T(o,r.rowCount,t.device.limits.maxComputeWorkgroupsPerDimension),c=4,l=r.validity?`@group(0) @binding(${c++}) var<storage, read> rowValidity: array<u32>;`:``,u=r.sourceRowIds?`@group(0) @binding(${c++}) var<storage, read> sourceRowIds: array<u32>;`:``,d=r.validity?`if (rowValidity[${e(r.validity)}u + index] == 0u) { return; }`:``,f=r.sourceRowIds?`if (sourceRowIds[${e(r.sourceRowIds)}u + index] == 0xffffffffu) { return; }`:``;w(t,{id:o,source:`
@group(0) @binding(0) var<storage, read> embeddingValues: array<f32>;
@group(0) @binding(1) var<storage, read> centroidValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> clusterLabels: array<u32>;
@group(0) @binding(3) var<storage, read_write> convergenceState: array<atomic<u32>>;
${l}
${u}
@compute @workgroup_size(64) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${E(s)}
  if (index >= ${r.rowCount}u ||
      atomicLoad(&convergenceState[${e(a)}u + 2u]) != 0u) { return; }
  ${d}
  ${f}
  var bestDistance = 3.402823466e+38;
  var bestCluster = 0xffffffffu;
  for (var clusterIndex = 0u; clusterIndex < ${n.clusterCount}u; clusterIndex++) {
    var distance = 0.0;
    var finiteRow = true;
    for (var dimension = 0u; dimension < ${n.dataset.dimensions}u; dimension++) {
      let sourceValue = embeddingValues[
        ${e(r.values)}u + index * ${r.chunk.rowStride}u + dimension
      ];
      let centroidValue = centroidValues[
        ${e(n.centroids)}u +
        clusterIndex * ${n.dataset.dimensions}u + dimension
      ];
      if (!(sourceValue == sourceValue && abs(sourceValue) <= 3.402823466e+38) ||
          !(centroidValue == centroidValue && abs(centroidValue) <= 3.402823466e+38)) {
        finiteRow = false;
        break;
      }
      let difference = sourceValue - centroidValue;
      distance += difference * difference;
    }
    if (finiteRow && (bestCluster == 0xffffffffu || distance < bestDistance)) {
      bestDistance = distance;
      bestCluster = clusterIndex;
    }
  }
  let previous = clusterLabels[${e(i)}u + index];
  clusterLabels[${e(i)}u + index] = bestCluster;
  if (previous != bestCluster) {
    atomicAdd(&convergenceState[${e(a)}u + 1u], 1u);
  }
}`,resources:[{buffer:r.values,usage:`storage-read`},{buffer:n.centroids,usage:`storage-read`},{buffer:i,usage:`storage-read-write`},{buffer:a,usage:`storage-read-write`},...r.validity?[{buffer:r.validity,usage:`storage-read`}]:[],...r.sourceRowIds?[{buffer:r.sourceRowIds,usage:`storage-read`}]:[]],bindings:{embeddingValues:r.values,centroidValues:n.centroids,clusterLabels:i,convergenceState:a,...r.validity?{rowValidity:r.validity}:{},...r.sourceRowIds?{sourceRowIds:r.sourceRowIds}:{}},elementCount:r.rowCount})}function H(t,n,r,i){w(t,{id:n,source:`
@group(0) @binding(0) var<storage, read_write> centroidSums: array<f32>;
@group(0) @binding(1) var<storage, read> convergenceState: array<u32>;
@compute @workgroup_size(64) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${E(T(n,r.length,t.device.limits.maxComputeWorkgroupsPerDimension))}
  if (index < ${r.length}u &&
      convergenceState[${e(i)}u + 2u] == 0u) {
    centroidSums[${e(r)}u + index] = 0.0;
  }
}`,resources:[{buffer:r,usage:`storage-write`},{buffer:i,usage:`storage-read`}],bindings:{centroidSums:r,convergenceState:i},elementCount:r.length})}function U(t,n,r,i,a,o,s){let c=n.clusterCount*n.dataset.dimensions;w(t,{id:s,source:`
@group(0) @binding(0) var<storage, read> embeddingValues: array<f32>;
@group(0) @binding(1) var<storage, read> clusterLabels: array<u32>;
@group(0) @binding(2) var<storage, read_write> centroidSums: array<f32>;
@group(0) @binding(3) var<storage, read> convergenceState: array<u32>;
@group(0) @binding(4) var<storage, read> clusterCounts: array<u32>;
@compute @workgroup_size(64) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${E(T(s,c,t.device.limits.maxComputeWorkgroupsPerDimension))}
  if (index >= ${c}u ||
      convergenceState[${e(o)}u + 2u] != 0u) { return; }
  let clusterIndex = index / ${n.dataset.dimensions}u;
  let dimension = index % ${n.dataset.dimensions}u;
  let clusterCount = clusterCounts[${e(n.counts)}u + clusterIndex];
  var total = centroidSums[${e(a)}u + index];
  for (var row = 0u; row < ${r.rowCount}u; row++) {
    if (clusterLabels[${e(i)}u + row] == clusterIndex) {
      total += embeddingValues[
        ${e(r.values)}u + row * ${r.chunk.rowStride}u + dimension
      ] / f32(clusterCount);
    }
  }
  centroidSums[${e(a)}u + index] = total;
}`,resources:[{buffer:r.values,usage:`storage-read`},{buffer:i,usage:`storage-read`},{buffer:a,usage:`storage-read-write`},{buffer:o,usage:`storage-read`},{buffer:n.counts,usage:`storage-read`}],bindings:{embeddingValues:r.values,clusterLabels:i,centroidSums:a,convergenceState:o,clusterCounts:n.counts},elementCount:c})}function W(t,n,r,i,a){let o=n.clusterCount*n.dataset.dimensions;w(t,{id:a,source:`
@group(0) @binding(0) var<storage, read> centroidSums: array<f32>;
@group(0) @binding(1) var<storage, read> clusterCounts: array<u32>;
@group(0) @binding(2) var<storage, read_write> centroidValues: array<f32>;
@group(0) @binding(3) var<storage, read> convergenceState: array<u32>;
@compute @workgroup_size(64) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${E(T(a,o,t.device.limits.maxComputeWorkgroupsPerDimension))}
  if (index >= ${o}u ||
      convergenceState[${e(i)}u + 2u] != 0u) { return; }
  let clusterIndex = index / ${n.dataset.dimensions}u;
  let count = clusterCounts[${e(n.counts)}u + clusterIndex];
  if (count != 0u) {
    centroidValues[${e(n.centroids)}u + index] =
      centroidSums[${e(r)}u + index];
  }
}`,resources:[{buffer:r,usage:`storage-read`},{buffer:n.counts,usage:`storage-read`},{buffer:n.centroids,usage:`storage-write`},{buffer:i,usage:`storage-read`}],bindings:{centroidSums:r,clusterCounts:n.counts,centroidValues:n.centroids,convergenceState:i},elementCount:o})}function G(t,n,r){w(t,{id:n,source:`
@group(0) @binding(0) var<storage, read_write> convergenceState: array<u32>;
@compute @workgroup_size(64) fn main(
  @builtin(global_invocation_id) globalInvocationId: vec3<u32>
) {
  if (globalInvocationId.x != 0u ||
      convergenceState[${e(r)}u + 2u] != 0u) { return; }
  convergenceState[${e(r)}u] += 1u;
  if (convergenceState[${e(r)}u + 1u] == 0u) {
    convergenceState[${e(r)}u + 2u] = 1u;
  }
}`,resources:[{buffer:r,usage:`storage-read-write`}],bindings:{convergenceState:r},elementCount:1})}var K=class{props;id;clustering;commandNodes=new WeakMap;constructor(e){this.props=e;let t=e.id??`raster-embedding-kmeans`;this.id=t;let{pixelCount:n,dimensions:r,output:i}=e;h(t,n,r),y(t,`vectors`,e.vectors,n,r,e.mask),f(t,`clusterCount`,e.clusterCount),f(t,`maximumIterations`,e.maximumIterations??10,1,64),this.clustering=new P({id:t,dataset:m(e.vectors,n,r,e.mask),clusterCount:e.clusterCount,centroids:i.centroids,labels:i.labels,counts:i.counts,status:i.status,maxIterations:e.maximumIterations})}getCommandNodes(e){let t=this.commandNodes.get(e);if(t)return t;let n=Object.freeze(p(e,()=>this.clustering.addToGraph(e)));return this.commandNodes.set(e,n),n}},q=[1,2,4,16],J=[[0,0,0,0],[1,1,1,.3]],Y=3,X=b({parameters:x,rebuildOn:[`clusterCount`,`maximumIterations`],caption:`Clusters found from the vectors alone, colored by mean NIR; point to lighten one. Six Sentinel-2 bands stand in for a learned embedding.`,createInputs:({aspect:e})=>({..._(e),focus:{cluster:-1,build:-1},orderedBuild:-1}),build(e,t,n){let r=Math.round(n.clusterCount),i=e.upload(`vectors`,`float32`,t.vectors),a=e.allocate(`labels`,`uint32`,t.pixelCount),o=e.allocate(`counts`,`uint32`,r),s=e.allocate(`status`,`uint32`,3),c=e.allocate(`centroids`,`float32`,r*6);return e.graph.add(new K({pixelCount:t.pixelCount,dimensions:6,vectors:i.view,clusterCount:r,maximumIterations:q[Math.round(n.maximumIterations)],output:{labels:a.view,centroids:c.view,counts:o.view,status:s.view}})),{outputs:{labels:a,counts:o,status:s,centroids:c,mask:e.allocate(`focus-mask`,`uint32`,t.pixelCount),ordered:e.allocate(`ordered-labels`,`uint32`,t.pixelCount)}}},interaction:{kind:`probe`,hint:`point at a cluster`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,world:r,peek:i,build:a}){let{index:o}=g(t,r,e.x,e.y),s=i(n.labels),c=i(n.counts),l=i(n.status),u=i(n.centroids);if(s&&u&&a!==t.orderedBuild){t.orderedBuild=a;let e=u.length/6,r=Array.from({length:e},(e,t)=>t).sort((e,t)=>u[e*6+Y]-u[t*6+Y]),i=new Uint32Array(e);r.forEach((e,t)=>{i[e]=t}),n.ordered.write(Uint32Array.from(s,t=>t<e?i[t]:t))}let d=s?s[o]:-1;s&&(d!==t.focus.cluster||a!==t.focus.build)&&(t.focus={cluster:d,build:a},n.mask.write(Uint32Array.from(s,e=>+(e===d))));let f=c&&d>=0?c[d]:0,p=c?c.length:0;return{spotlight:null,focus:d,glyphs:[v(t,r,o)],readout:`${(100*f/t.pixelCount).toFixed(0)}% of pixels · cluster of ${p} · ${l&&l[2]?`converged`:`stopped`} in ${l?l[0]:`-`}`}}},draw:({inputs:e,outputs:t})=>[{layer:`raster`,values:t.ordered,columns:e.columns,rows:e.rows},{layer:`raster`,values:t.mask,columns:e.columns,rows:e.rows,categories:J}]});export{X as default};