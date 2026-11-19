import{d as e,r as t}from"./gpu-command-graph-BdzdmV6u.js";import{c as n,d as r,o as i,t as a,u as o}from"./kernel-C3tfQOn4.js";var s=256,c=128,l=64,u=4294967295,d=class{id;strategy;resolvedStrategy;minima;maxima;sourceIds;leafCapacity;nodeMinima;nodeMaxima;nodeChildren;leafIds;count;overflow;dimension;nodeCount;internalNodeCount;levelCount;rootNode=0;topology=`complete-binary`;updatePolicy=`refit`;stats;constructor(t){if(this.id=t.id??`gpu-bvh`,this.strategy=t.strategy??`auto`,this.minima=t.minima,this.maxima=t.maxima,this.sourceIds=t.sourceIds,this.leafCapacity=t.leafCapacity,this.nodeMinima=t.nodeMinima,this.nodeMaxima=t.nodeMaxima,this.nodeChildren=t.nodeChildren,this.leafIds=t.leafIds,this.count=t.count,this.overflow=t.overflow,this.dimension=this.minima.format===`float32x2`?2:3,this.minima.length>u)throw Error(`${this.id} source row count exceeds uint32 range`);if(!Number.isSafeInteger(this.leafCapacity)||!v(this.leafCapacity))throw Error(`${this.id} leafCapacity must be a positive power of two`);if(![`auto`,`fused`,`level`].includes(this.strategy))throw Error(`${this.id} strategy must be auto, fused, or level`);let n=this.minima.buffer.graph.device.limits,i=this.leafCapacity<=c&&this.leafCapacity<=n.maxComputeInvocationsPerWorkgroup&&this.leafCapacity<=n.maxComputeWorkgroupSizeX&&this.leafCapacity*l<=n.maxComputeWorkgroupStorageSize;if(this.strategy===`fused`&&!i)throw Error(`${this.id} fused strategy exceeds portable single-workgroup limits`);if(this.resolvedStrategy=this.strategy===`level`||!i?`level`:`fused`,this.nodeCount=this.leafCapacity*2-1,this.internalNodeCount=this.leafCapacity-1,this.levelCount=Math.log2(this.leafCapacity)+1,!Number.isSafeInteger(this.nodeCount)||this.nodeCount>u)throw Error(`${this.id} node count exceeds uint32 range`);if(r(this.minima,[`float32x2`,`float32x3`],`${this.id} minima`),r(this.maxima,[`float32x2`,`float32x3`],`${this.id} maxima`),r(this.nodeMinima,[`float32x2`,`float32x3`],`${this.id} nodeMinima`),r(this.nodeMaxima,[`float32x2`,`float32x3`],`${this.id} nodeMaxima`),r(this.nodeChildren,[`uint32x2`],`${this.id} nodeChildren`),o(this.leafIds,`${this.id} leafIds`),o(this.count,`${this.id} count`),o(this.overflow,`${this.id} overflow`),this.sourceIds&&o(this.sourceIds,`${this.id} sourceIds`),this.minima.format!==this.maxima.format||this.minima.length!==this.maxima.length)throw Error(`${this.id} minima and maxima must have matching formats and lengths`);if(this.sourceIds&&this.sourceIds.length!==this.minima.length)throw Error(`${this.id} sourceIds.length must equal bounds length`);if(this.nodeMinima.format!==this.minima.format||this.nodeMaxima.format!==this.minima.format||this.nodeMinima.length!==this.nodeCount||this.nodeMaxima.length!==this.nodeCount)throw Error(`${this.id} node bounds must match source format and node count`);if(this.nodeChildren.length!==this.nodeCount)throw Error(`${this.id} nodeChildren.length must equal node count`);if(this.leafIds.length!==this.leafCapacity)throw Error(`${this.id} leafIds.length must equal leafCapacity`);if(this.count.length<1||this.overflow.length<1)throw Error(`${this.id} count and overflow must each contain one uint32 row`);let a=e(this.minima.format).byteLength;this.stats={dimension:this.dimension,leafCapacity:this.leafCapacity,internalNodeCount:this.internalNodeCount,nodeCount:this.nodeCount,levelCount:this.levelCount,outputByteLength:this.nodeCount*(a*2+Uint32Array.BYTES_PER_ELEMENT*2)+this.leafCapacity*Uint32Array.BYTES_PER_ELEMENT+Uint32Array.BYTES_PER_ELEMENT*2}}getCommandNodes(e){let t=[];if([this.minima,this.maxima,...this.sourceIds?[this.sourceIds]:[],this.nodeMinima,this.nodeMaxima,this.nodeChildren,this.leafIds,this.count,this.overflow].some(t=>t.buffer.graph!==e))throw Error(`${this.id} views must belong to the target graph`);if(this.resolvedStrategy===`fused`)t.push(...f(e,this));else{t.push(...p(e,this,_(this.nodeCount,e.device.limits.maxComputeWorkgroupsPerDimension)));for(let n=this.levelCount-2;n>=0;n--)t.push(...h(e,this,n))}return this.sourceIds&&t.push(...m(e,this,this.sourceIds)),t}};function f(e,t){let r=[],i=`
const SOURCE_COUNT: u32 = ${t.minima.length}u;
const STORED_COUNT: u32 = ${Math.min(t.minima.length,t.leafCapacity)}u;
const LEAF_CAPACITY: u32 = ${t.leafCapacity}u;
const INTERNAL_NODE_COUNT: u32 = ${t.internalNodeCount}u;
const DIMENSION: u32 = ${t.dimension}u;
const MINIMA_OFFSET: u32 = ${n(t.minima)}u;
const MAXIMA_OFFSET: u32 = ${n(t.maxima)}u;
const NODE_MINIMA_OFFSET: u32 = ${n(t.nodeMinima)}u;
const NODE_MAXIMA_OFFSET: u32 = ${n(t.nodeMaxima)}u;
const CHILDREN_OFFSET: u32 = ${n(t.nodeChildren)}u;
const LEAF_IDS_OFFSET: u32 = ${n(t.leafIds)}u;
const COUNT_OFFSET: u32 = ${n(t.count)}u;
const OVERFLOW_OFFSET: u32 = ${n(t.overflow)}u;
@group(0) @binding(0) var<storage, read> sourceMinima: array<f32>;
@group(0) @binding(1) var<storage, read> sourceMaxima: array<f32>;
@group(0) @binding(2) var<storage, read_write> nodeMinima: array<f32>;
@group(0) @binding(3) var<storage, read_write> nodeMaxima: array<f32>;
@group(0) @binding(4) var<storage, read_write> nodeChildren: array<u32>;
@group(0) @binding(5) var<storage, read_write> leafIds: array<u32>;
@group(0) @binding(6) var<storage, read_write> outputCount: array<u32>;
@group(0) @binding(7) var<storage, read_write> outputOverflow: array<u32>;

var<workgroup> sharedMinima: array<vec4<f32>, ${t.leafCapacity*2}>;
var<workgroup> sharedMaxima: array<vec4<f32>, ${t.leafCapacity*2}>;

fn finite(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823466e+38;
}

@compute @workgroup_size(${t.leafCapacity}) fn main(
  @builtin(local_invocation_index) localIndex: u32
) {
  var minimum = vec4<f32>(3.402823466e+38);
  var maximum = vec4<f32>(-3.402823466e+38);
  let leafNode = INTERNAL_NODE_COUNT + localIndex;
  let leafComponent = leafNode * DIMENSION;
  let leafChildComponent = leafNode * 2u;
  nodeChildren[CHILDREN_OFFSET + leafChildComponent] = ${u}u;
  nodeChildren[CHILDREN_OFFSET + leafChildComponent + 1u] = ${u}u;
  leafIds[LEAF_IDS_OFFSET + localIndex] = ${u}u;

  if (localIndex < STORED_COUNT) {
    let sourceComponent = localIndex * DIMENSION;
    var valid = true;
    for (var axis = 0u; axis < DIMENSION; axis++) {
      let sourceMinimum = sourceMinima[MINIMA_OFFSET + sourceComponent + axis];
      let sourceMaximum = sourceMaxima[MAXIMA_OFFSET + sourceComponent + axis];
      valid = valid && finite(sourceMinimum) && finite(sourceMaximum) &&
        sourceMinimum <= sourceMaximum;
    }
    leafIds[LEAF_IDS_OFFSET + localIndex] = localIndex;
    if (valid) {
      for (var axis = 0u; axis < DIMENSION; axis++) {
        minimum[axis] = sourceMinima[MINIMA_OFFSET + sourceComponent + axis];
        maximum[axis] = sourceMaxima[MAXIMA_OFFSET + sourceComponent + axis];
      }
    }
  }

  for (var axis = 0u; axis < DIMENSION; axis++) {
    nodeMinima[NODE_MINIMA_OFFSET + leafComponent + axis] = minimum[axis];
    nodeMaxima[NODE_MAXIMA_OFFSET + leafComponent + axis] = maximum[axis];
  }
  sharedMinima[localIndex] = minimum;
  sharedMaxima[localIndex] = maximum;

  if (localIndex < INTERNAL_NODE_COUNT) {
    let childComponent = localIndex * 2u;
    nodeChildren[CHILDREN_OFFSET + childComponent] = localIndex * 2u + 1u;
    nodeChildren[CHILDREN_OFFSET + childComponent + 1u] = localIndex * 2u + 2u;
  }
  if (localIndex == 0u) {
    outputCount[COUNT_OFFSET] = SOURCE_COUNT;
    outputOverflow[OVERFLOW_OFFSET] = select(0u, 1u, SOURCE_COUNT > LEAF_CAPACITY);
  }
  workgroupBarrier();

  var sourceOffset = 0u;
  var destinationOffset = LEAF_CAPACITY;
  for (var levelNodeCount = LEAF_CAPACITY / 2u;
       levelNodeCount > 0u;
       levelNodeCount = levelNodeCount / 2u) {
    if (localIndex < levelNodeCount) {
      let firstChild = sourceOffset + localIndex * 2u;
      let reducedMinimum = min(sharedMinima[firstChild], sharedMinima[firstChild + 1u]);
      let reducedMaximum = max(sharedMaxima[firstChild], sharedMaxima[firstChild + 1u]);
      sharedMinima[destinationOffset + localIndex] = reducedMinimum;
      sharedMaxima[destinationOffset + localIndex] = reducedMaximum;

      let nodeIndex = levelNodeCount - 1u + localIndex;
      let nodeComponent = nodeIndex * DIMENSION;
      for (var axis = 0u; axis < DIMENSION; axis++) {
        nodeMinima[NODE_MINIMA_OFFSET + nodeComponent + axis] = reducedMinimum[axis];
        nodeMaxima[NODE_MAXIMA_OFFSET + nodeComponent + axis] = reducedMaximum[axis];
      }
    }
    workgroupBarrier();
    let previousSourceOffset = sourceOffset;
    sourceOffset = destinationOffset;
    destinationOffset = previousSourceOffset;
  }
}`,a=[{buffer:t.minima,usage:`storage-read`},{buffer:t.maxima,usage:`storage-read`},{buffer:t.nodeMinima,usage:`storage-write`},{buffer:t.nodeMaxima,usage:`storage-write`},{buffer:t.nodeChildren,usage:`storage-write`},{buffer:t.leafIds,usage:`storage-write`},{buffer:t.count,usage:`storage-write`},{buffer:t.overflow,usage:`storage-write`}];return r.push(...g(e,{id:`${t.id}-fused-refit`,source:i,resources:a,bindings:{sourceMinima:t.minima,sourceMaxima:t.maxima,nodeMinima:t.nodeMinima,nodeMaxima:t.nodeMaxima,nodeChildren:t.nodeChildren,leafIds:t.leafIds,outputCount:t.count,outputOverflow:t.overflow},dispatchCount:1})),r}function p(e,t,r){let i=[],a=`
const SOURCE_COUNT: u32 = ${t.minima.length}u;
const STORED_COUNT: u32 = ${Math.min(t.minima.length,t.leafCapacity)}u;
const LEAF_CAPACITY: u32 = ${t.leafCapacity}u;
const INTERNAL_NODE_COUNT: u32 = ${t.internalNodeCount}u;
const NODE_COUNT: u32 = ${t.nodeCount}u;
const DIMENSION: u32 = ${t.dimension}u;
const MINIMA_OFFSET: u32 = ${n(t.minima)}u;
const MAXIMA_OFFSET: u32 = ${n(t.maxima)}u;
const NODE_MINIMA_OFFSET: u32 = ${n(t.nodeMinima)}u;
const NODE_MAXIMA_OFFSET: u32 = ${n(t.nodeMaxima)}u;
const CHILDREN_OFFSET: u32 = ${n(t.nodeChildren)}u;
const LEAF_IDS_OFFSET: u32 = ${n(t.leafIds)}u;
const COUNT_OFFSET: u32 = ${n(t.count)}u;
const OVERFLOW_OFFSET: u32 = ${n(t.overflow)}u;
@group(0) @binding(0) var<storage, read> sourceMinima: array<f32>;
@group(0) @binding(1) var<storage, read> sourceMaxima: array<f32>;
@group(0) @binding(2) var<storage, read_write> nodeMinima: array<f32>;
@group(0) @binding(3) var<storage, read_write> nodeMaxima: array<f32>;
@group(0) @binding(4) var<storage, read_write> nodeChildren: array<u32>;
@group(0) @binding(5) var<storage, read_write> leafIds: array<u32>;
@group(0) @binding(6) var<storage, read_write> outputCount: array<u32>;
@group(0) @binding(7) var<storage, read_write> outputOverflow: array<u32>;

fn finite(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823466e+38;
}

@compute @workgroup_size(${s}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>
) {
  let workgroupIndex = (workgroupId.z * ${r.y}u + workgroupId.y) * ${r.x}u + workgroupId.x;
  let nodeIndex = workgroupIndex * ${s}u + localId.x;
  if (nodeIndex >= NODE_COUNT) { return; }
  let nodeComponent = nodeIndex * DIMENSION;
  for (var axis = 0u; axis < DIMENSION; axis++) {
    nodeMinima[NODE_MINIMA_OFFSET + nodeComponent + axis] = 3.402823466e+38;
    nodeMaxima[NODE_MAXIMA_OFFSET + nodeComponent + axis] = -3.402823466e+38;
  }
  let childComponent = nodeIndex * 2u;
  if (nodeIndex < INTERNAL_NODE_COUNT) {
    nodeChildren[CHILDREN_OFFSET + childComponent] = nodeIndex * 2u + 1u;
    nodeChildren[CHILDREN_OFFSET + childComponent + 1u] = nodeIndex * 2u + 2u;
  } else {
    nodeChildren[CHILDREN_OFFSET + childComponent] = ${u}u;
    nodeChildren[CHILDREN_OFFSET + childComponent + 1u] = ${u}u;
    let leafIndex = nodeIndex - INTERNAL_NODE_COUNT;
    leafIds[LEAF_IDS_OFFSET + leafIndex] = ${u}u;
    if (leafIndex < STORED_COUNT) {
      var valid = true;
      let sourceComponent = leafIndex * DIMENSION;
      for (var axis = 0u; axis < DIMENSION; axis++) {
        let minimum = sourceMinima[MINIMA_OFFSET + sourceComponent + axis];
        let maximum = sourceMaxima[MAXIMA_OFFSET + sourceComponent + axis];
        valid = valid && finite(minimum) && finite(maximum) && minimum <= maximum;
      }
      leafIds[LEAF_IDS_OFFSET + leafIndex] = leafIndex;
      if (valid) {
        for (var axis = 0u; axis < DIMENSION; axis++) {
          nodeMinima[NODE_MINIMA_OFFSET + nodeComponent + axis] =
            sourceMinima[MINIMA_OFFSET + sourceComponent + axis];
          nodeMaxima[NODE_MAXIMA_OFFSET + nodeComponent + axis] =
            sourceMaxima[MAXIMA_OFFSET + sourceComponent + axis];
        }
      }
    }
  }
  if (nodeIndex == 0u) {
    outputCount[COUNT_OFFSET] = SOURCE_COUNT;
    outputOverflow[OVERFLOW_OFFSET] = select(0u, 1u, SOURCE_COUNT > LEAF_CAPACITY);
  }
}`,o=[{buffer:t.minima,usage:`storage-read`},{buffer:t.maxima,usage:`storage-read`},{buffer:t.nodeMinima,usage:`storage-write`},{buffer:t.nodeMaxima,usage:`storage-write`},{buffer:t.nodeChildren,usage:`storage-write`},{buffer:t.leafIds,usage:`storage-write`},{buffer:t.count,usage:`storage-write`},{buffer:t.overflow,usage:`storage-write`}];return i.push(...g(e,{id:`${t.id}-load-leaves`,source:a,resources:o,bindings:{sourceMinima:t.minima,sourceMaxima:t.maxima,nodeMinima:t.nodeMinima,nodeMaxima:t.nodeMaxima,nodeChildren:t.nodeChildren,leafIds:t.leafIds,outputCount:t.count,outputOverflow:t.overflow},dispatchSize:r})),i}function m(e,t,r){let i=[],a=Math.min(r.length,t.leafCapacity),o=_(a,e.device.limits.maxComputeWorkgroupsPerDimension),c=`
const STORED_COUNT: u32 = ${a}u;
const SOURCE_IDS_OFFSET: u32 = ${n(r)}u;
const LEAF_IDS_OFFSET: u32 = ${n(t.leafIds)}u;
@group(0) @binding(0) var<storage, read> sourceIds: array<u32>;
@group(0) @binding(1) var<storage, read_write> leafIds: array<u32>;

@compute @workgroup_size(${s}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>
) {
  let workgroupIndex = (workgroupId.z * ${o.y}u + workgroupId.y) * ${o.x}u + workgroupId.x;
  let leafIndex = workgroupIndex * ${s}u + localId.x;
  if (leafIndex >= STORED_COUNT) { return; }
  let sourceIndex = leafIds[LEAF_IDS_OFFSET + leafIndex];
  if (sourceIndex == ${u}u) { return; }
  leafIds[LEAF_IDS_OFFSET + leafIndex] = sourceIds[SOURCE_IDS_OFFSET + sourceIndex];
}`;return i.push(...g(e,{id:`${t.id}-remap-source-ids`,source:c,resources:[{buffer:r,usage:`storage-read`},{buffer:t.leafIds,usage:`storage-read-write`}],bindings:{sourceIds:r,leafIds:t.leafIds},dispatchSize:o})),i}function h(e,t,r){let i=[],a=2**r-1,o=2**r,c=`
const FIRST_NODE: u32 = ${a}u;
const LEVEL_NODE_COUNT: u32 = ${o}u;
const DIMENSION: u32 = ${t.dimension}u;
const NODE_MINIMA_OFFSET: u32 = ${n(t.nodeMinima)}u;
const NODE_MAXIMA_OFFSET: u32 = ${n(t.nodeMaxima)}u;
const CHILDREN_OFFSET: u32 = ${n(t.nodeChildren)}u;
@group(0) @binding(0) var<storage, read_write> nodeMinima: array<f32>;
@group(0) @binding(1) var<storage, read_write> nodeMaxima: array<f32>;
@group(0) @binding(2) var<storage, read> nodeChildren: array<u32>;

@compute @workgroup_size(${s}) fn main(
  @builtin(global_invocation_id) globalId: vec3<u32>
) {
  if (globalId.x >= LEVEL_NODE_COUNT) { return; }
  let nodeIndex = FIRST_NODE + globalId.x;
  let childComponent = nodeIndex * 2u;
  let left = nodeChildren[CHILDREN_OFFSET + childComponent];
  let right = nodeChildren[CHILDREN_OFFSET + childComponent + 1u];
  for (var axis = 0u; axis < DIMENSION; axis++) {
    nodeMinima[NODE_MINIMA_OFFSET + nodeIndex * DIMENSION + axis] = min(
      nodeMinima[NODE_MINIMA_OFFSET + left * DIMENSION + axis],
      nodeMinima[NODE_MINIMA_OFFSET + right * DIMENSION + axis]
    );
    nodeMaxima[NODE_MAXIMA_OFFSET + nodeIndex * DIMENSION + axis] = max(
      nodeMaxima[NODE_MAXIMA_OFFSET + left * DIMENSION + axis],
      nodeMaxima[NODE_MAXIMA_OFFSET + right * DIMENSION + axis]
    );
  }
}`;return i.push(...g(e,{id:`${t.id}-refit-depth-${r}`,source:c,resources:[{buffer:t.nodeMinima,usage:`storage-read-write`},{buffer:t.nodeMaxima,usage:`storage-read-write`},{buffer:t.nodeChildren,usage:`storage-read`}],bindings:{nodeMinima:t.nodeMinima,nodeMaxima:t.nodeMaxima,nodeChildren:t.nodeChildren},dispatchCount:Math.ceil(o/s)})),i}function g(e,n){let r=[];return r.push(t({id:n.id,resources:n.resources,compile:({device:e})=>{let t=new a(e,{id:n.id,source:n.source,shaderLayout:{bindings:Object.keys(n.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:r})=>{let a={};for(let[e,t]of Object.entries(n.bindings))a[e]=i(t,r);n.dispatchSize?t.dispatch(e,{bindings:a,x:n.dispatchSize.x,y:n.dispatchSize.y,z:n.dispatchSize.z}):t.dispatch(e,{bindings:a,x:n.dispatchCount})},destroy:()=>t.destroy()}}})),r}function _(e,t){let n=Math.floor(t),r=Math.max(1,Math.ceil(e/s)),i=Math.min(r,n),a=Math.min(Math.ceil(r/i),n),o=Math.ceil(r/i/a);if(o>n)throw Error(`GPUBVH requires ${r} workgroups, exceeding the 3D dispatch limit of ${n} per dimension`);return{x:i,y:a,z:o}}function v(e){return e>0&&Number.isInteger(Math.log2(e))}export{d as t};