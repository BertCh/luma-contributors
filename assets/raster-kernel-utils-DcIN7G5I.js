import{c as e,o as t}from"./wgsl-common-DfyXiYB0.js";import{a as n,o as r,r as i,s as a,t as o}from"./kernel-C3tfQOn4.js";import{E as s,T as c,f as l,v as u,x as d}from"./gpu-contributor-utils-BIwLQON5.js";import{t as f}from"./raster-view-validation-CSMxj-cg.js";import{Q as p}from"./dist-BusamGlx.js";import{l as m,r as h}from"./gpu-command-graph-BdzdmV6u.js";import{r as g,t as _}from"./gpu-command-dispatch-metadata-Bpi5I1Ez.js";import{n as v,r as y,t as b}from"./gpu-chunk-utils-DMYEopiS.js";import{n as x,t as S}from"./gpu-dispatch-utils-CbRzFAgT.js";import{i as C,n as w,r as T,t as E}from"./gpu-fft-utils-C1LfmzeI.js";import{a as D}from"./graph-chunk-dispatch-zBt8k2-o.js";import{o as O}from"./wgsl-kernel-nodes-DagJklH2.js";var k=`\
struct GPUFFT2DParameters {
  width: u32,
  height: u32,
  axis: u32,
  passKind: u32,
  transformSize: u32,
  stage: u32,
  directionSign: f32,
  normalizationScale: f32,
  inputOffset: u32,
  outputOffset: u32,
  padding: vec2u,
};

@group(0) @binding(0) var<storage, read> inputValues: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<vec2f>;
@group(0) @binding(2) var<uniform> parameters: GPUFFT2DParameters;

${E}

fn getLinearIndex(xCoordinate: u32, yCoordinate: u32, batchIndex: u32) -> u32 {
  return batchIndex * parameters.width * parameters.height +
    yCoordinate * parameters.width + xCoordinate;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) globalIdentifier: vec3u) {
  if (globalIdentifier.x >= parameters.width || globalIdentifier.y >= parameters.height) {
    return;
  }

  let horizontal = parameters.axis == 0u;
  let coordinate = select(globalIdentifier.y, globalIdentifier.x, horizontal);
  var sourceCoordinate = coordinate;

  if (parameters.passKind == 0u) {
    sourceCoordinate = reverseLowBits(coordinate, parameters.stage);
  } else {
    let butterflySpan = 1u << parameters.stage;
    let butterflyHalfSpan = butterflySpan >> 1u;
    let butterflyOffset = coordinate & (butterflySpan - 1u);
    let twiddleIndex = butterflyOffset & (butterflyHalfSpan - 1u);
    let butterflyStart = coordinate - butterflyOffset;
    let firstCoordinate = butterflyStart + twiddleIndex;
    let secondCoordinate = firstCoordinate + butterflyHalfSpan;

    let firstX = select(globalIdentifier.x, firstCoordinate, horizontal);
    let firstY = select(firstCoordinate, globalIdentifier.y, horizontal);
    let secondX = select(globalIdentifier.x, secondCoordinate, horizontal);
    let secondY = select(secondCoordinate, globalIdentifier.y, horizontal);
    let firstValue = inputValues[parameters.inputOffset + getLinearIndex(firstX, firstY, globalIdentifier.z)];
    let secondValue = inputValues[parameters.inputOffset + getLinearIndex(secondX, secondY, globalIdentifier.z)];
    let angle = parameters.directionSign * 2.0 * GPU_FFT_PI *
      f32(twiddleIndex) / f32(butterflySpan);
    let twiddle = vec2f(cos(angle), sin(angle));
    let rotatedSecondValue = multiplyComplex(secondValue, twiddle);
    let butterflyValue = select(
      firstValue + rotatedSecondValue,
      firstValue - rotatedSecondValue,
      butterflyOffset >= butterflyHalfSpan
    );
    outputValues[parameters.outputOffset + getLinearIndex(globalIdentifier.x, globalIdentifier.y, globalIdentifier.z)] =
      butterflyValue * parameters.normalizationScale;
    return;
  }

  let sourceX = select(globalIdentifier.x, sourceCoordinate, horizontal);
  let sourceY = select(sourceCoordinate, globalIdentifier.y, horizontal);
  outputValues[parameters.outputOffset + getLinearIndex(globalIdentifier.x, globalIdentifier.y, globalIdentifier.z)] =
    inputValues[parameters.inputOffset + getLinearIndex(sourceX, sourceY, globalIdentifier.z)] *
      parameters.normalizationScale;
}
`,A=2,j=w,M=class{id;stats;props;constructor(e){this.props=e,this.id=e.id??`gpu-fft2d`,this.stats=I(e.width,e.height,e.batchCount??1);let t=e.direction??`forward`;if(t!==`forward`&&t!==`inverse`)throw Error(`GPUFFT2D direction must be forward or inverse`);for(let t of[e.input,e.output]){if(t.length<this.stats.elementCount*(e.batchCount??1))throw Error(`GPUFFT2D views must contain width * height * batchCount complex values`);for(let e of g(t))if(e.format!==`float32x2`||e.byteStride!==8||e.rowByteLength!==8||e.byteOffset%8!=0)throw Error(`GPUFFT2D requires packed, vec2-aligned float32x2 views`)}}getCommandNodes(e){y(e,[this.props.input],[this.props.output]);let t=F(e.device,this.props);if(!t.supported)throw Error(t.reason);let r=this.stats.elementCount*(this.props.batchCount??1),o=n(e,this.props.input,r),s=n(e,this.props.output,r),c=[],l=!(o instanceof m)&&!(s instanceof m)&&[o,s].every(t=>a(t).size<=e.device.limits.maxStorageBufferBindingSize)&&this.stats.complexBufferByteLength<=e.device.limits.maxBufferSize&&(this.props.batchCount??1)<=e.device.limits.maxComputeWorkgroupsPerDimension,u=l?r:this.stats.elementCount,d=i(e,`${this.id}-scratch`,`float32x2`,u);if(l)return N(e,this,o,s,d,this.props.batchCount??1,this.id);let f=i(e,`${this.id}-field`,`float32x2`,u),p=this.stats.passCount%2==0?f:d;for(let t=0;t<(this.props.batchCount??1);t++){let n=t*u,r=`${this.id}-batch-${t}`;c.push(...P(e,`${r}-gather`,v(e,o,n,u),p,!0)),c.push(...N(e,this,p,f,d,1,r)),c.push(...P(e,`${r}-scatter`,v(e,s,n,u),f,!1))}return c}};function N(e,t,n,i,a,s,c){let l=[],u=L(t.props.width,t.props.height),d=n;for(let[e,n]of u.entries()){let f=(u.length-e)%2==0?a:i,m=d,g=`${c}-${e}`,v=t.props.direction??`forward`,y=z({width:t.props.width,height:t.props.height,...n,direction:v,normalizationScale:v===`inverse`&&e===u.length-1?1/t.stats.elementCount:1,inputOffset:m.byteOffset%256/8,outputOffset:f.byteOffset%256/8}),b=[t.stats.workgroupCount[0],t.stats.workgroupCount[1],s];l.push(_(h({id:g,resources:[{buffer:m,usage:`storage-read`},{buffer:f,usage:`storage-write`}],workload:{commandCount:1,maximumWorkgroupCount:b[0]*b[1]*s,maximumInvocationCount:b[0]*b[1]*s*64},compile:({device:e})=>{let t=e.createBuffer({id:`${g}-parameters`,data:y,usage:p.UNIFORM}),n;try{n=new o(e,{id:g,source:k,shaderLayout:{bindings:[{name:`inputValues`,type:`read-only-storage`,group:0,location:0},{name:`outputValues`,type:`storage`,group:0,location:1},{name:`parameters`,type:`uniform`,group:0,location:2}]}})}catch(e){throw t.destroy(),e}return{encode:({computePass:e,getBuffer:i})=>n.dispatch(e,{bindings:{inputValues:r(m,i),outputValues:r(f,i),parameters:t},x:b[0],y:b[1],z:b[2]}),destroy:()=>{n.destroy(),t.destroy()}}}}),b)),d=f}return l}function P(e,t,n,r,i){let a=[],o=0;for(let s of n.data){for(let n=0;n<s.length;){let c=s.byteOffset+n*8,l=Math.min(s.length-n,Math.floor((e.device.limits.maxStorageBufferBindingSize-c%256)/8));if(l<1)throw Error(`GPUFFT2D chunk offset exceeds storage binding capacity`);let u=e.createDataView(s.buffer,{format:`float32x2`,byteOffset:c,length:l}),d=i?u:r,f=i?r:u,p=i?c%256/8:o+n,m=i?o+n:c%256/8,h=S(t,l,64,e.device.limits.maxComputeWorkgroupsPerDimension);a.push(b(e,{id:`${t}-${a.length}`,inputs:{inputValues:d},outputs:{outputValues:f},dispatch:h,workgroupSize:64,source:`@group(0) @binding(0) var<storage, read> inputValues: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<vec2f>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${x(h,64)}
  if (index < ${l}u) {
    outputValues[${m}u + index] = inputValues[${p}u + index];
  }
}`})),n+=l}o+=s.length}return a}function F(e,t){let n=B(t.width,t.height,t.batchCount??1);if(n)return{supported:!1,reason:n};let r=I(t.width,t.height,t.batchCount??1);if(e.type!==`webgpu`)return{supported:!1,reason:`GPUFFT2D requires WebGPU.`,stats:r};if(e.limits.maxStorageBuffersPerShaderStage<2)return{supported:!1,reason:`GPUFFT2D requires two compute storage buffers.`,stats:r};if(e.limits.maxUniformBuffersPerShaderStage<1)return{supported:!1,reason:`GPUFFT2D requires one compute uniform buffer.`,stats:r};if(e.limits.maxComputeInvocationsPerWorkgroup<64||e.limits.maxComputeWorkgroupSizeX<8||e.limits.maxComputeWorkgroupSizeY<8)return{supported:!1,reason:`GPUFFT2D requires 8 by 8 compute workgroups.`,stats:r};if(r.workgroupCount[0]>e.limits.maxComputeWorkgroupsPerDimension||r.workgroupCount[1]>e.limits.maxComputeWorkgroupsPerDimension||(t.input&&t.output?1:r.workgroupCount[2])>e.limits.maxComputeWorkgroupsPerDimension)return{supported:!1,reason:`GPUFFT2D workgroup count exceeds the device dispatch limit.`,stats:r};let i=t.input&&t.output?r.elementCount*8:r.complexBufferByteLength;return i>e.limits.maxStorageBufferBindingSize?{supported:!1,reason:`GPUFFT2D complex buffer exceeds maxStorageBufferBindingSize.`,stats:r}:i>e.limits.maxBufferSize?{supported:!1,reason:`GPUFFT2D complex buffer exceeds maxBufferSize.`,stats:r}:{supported:!0,stats:r}}function I(e,t,n=1){let r=B(e,t,n);if(r)throw Error(r);let i=Math.log2(e),a=Math.log2(t),o=i+a+2,s=e*t,c=s*n*2*Float32Array.BYTES_PER_ELEMENT;return Object.freeze({width:e,height:t,...n>1?{batchCount:n}:{},elementCount:s,complexBufferByteLength:c,horizontalStageCount:i,verticalStageCount:a,passCount:o,dispatchCountPerEncode:o,workgroupSize:Object.freeze([8,8,1]),workgroupCount:Object.freeze([Math.ceil(e/8),Math.ceil(t/8),n]),scratchBufferByteLength:c,parameterBufferCount:o,parameterBufferByteLength:o*48})}function L(e,t){let n=[];return R(n,`horizontal`,e),R(n,`vertical`,t),n}function R(e,t,n){for(let r of C(n))e.push({axis:t,transformSize:n,...r})}function z(e){let t=new ArrayBuffer(48),n=new Uint32Array(t),r=new Float32Array(t);return n[0]=e.width,n[1]=e.height,n[2]=e.axis===`horizontal`?0:1,n[3]=e.kind===`bit-reversal`?0:1,n[4]=e.transformSize,n[5]=e.stage,r[6]=e.direction===`forward`?-1:1,r[7]=e.normalizationScale,n[8]=e.inputOffset,n[9]=e.outputOffset,n}function B(e,t,n=1){let r=V(`width`,e);if(r)return r;let i=V(`height`,t);if(i)return i;if(!Number.isSafeInteger(n)||n<=0)return`GPUFFT2D batchCount must be a positive integer.`;if(e*t*n>4294967295)return`GPUFFT2D logical length must fit uint32 addressing.`}function V(e,t){return T(`GPUFFT2D`,e,t)}function H(e,t,n,r={}){D(e,t,Object.fromEntries(n.map((e,t)=>[`view${t}`,e])));for(let[n,i]of Object.entries(r))u(e.device,t,n,i,`reduce the raster dimensions or refinement window`),l(t,`${n} byteLength (maxBufferSize)`,i,0,e.device.limits.maxBufferSize)}function U(e,t,n,r,i=1){let a=F(e.device,{width:n,height:r,batchCount:i});c(t,a.supported,`grid dimensions exceed GPUFFT2D device limits: ${a.reason}`);let o=I(n,r,i);H(e,t,[],{"GPUFFT2D scratch":o.scratchBufferByteLength}),l(t,`GPUFFT2D parameter allocation byteLength (maxBufferSize)`,o.parameterBufferByteLength/o.parameterBufferCount,1,e.device.limits.maxBufferSize),l(t,`GPUFFT2D parameter binding byteLength (maxUniformBufferBindingSize)`,o.parameterBufferByteLength/o.parameterBufferCount,1,e.device.limits.maxUniformBufferBindingSize)}function W(e,t,n,r){for(let r of[...t,...n])s(r,[r.format],e);f(e,t,n,r)}function G(e,t,n,r){d(e,t,[...n,...r]),H(t,e,[...n,...r])}function K(e,t,n=!1){return{name:e,view:t,type:t.format.startsWith(`uint32`)?`u32`:t.format.startsWith(`sint32`)?`i32`:`f32`,access:n?`read_write`:`read`}}var q=`${t}${e}
fn finite(value: f32) -> bool { return isFiniteBits(value); }
fn missing() -> f32 { return getQuietNaN(); }`;function J(e,t,n=!1){let r=n?`columnCount`:`${t}u`;return`
struct LeastSquaresResult { coefficients: array<f32, ${t}>, residual: f32, valid: bool }
fn leastSquares(design: array<f32, ${e*t}>, response: array<f32, ${e}>${n?`, columnCount: u32`:``}) -> LeastSquaresResult {
  var basis = design;
  var upper: array<f32, ${t*t}>;
  var projected: array<f32, ${t}>;
  var result: LeastSquaresResult;
  result.valid = false;
  var maximumNorm = 0.0;
  for (var column = 0u; column < ${r}; column++) {
    // Original norm, orthogonal projections and final norm have disjoint lifetimes.
    var columnAccumulator = 0.0;
    for (var row = 0u; row < ${e}u; row++) { columnAccumulator += basis[column * ${e}u + row] * basis[column * ${e}u + row]; }
    maximumNorm = max(maximumNorm, sqrt(columnAccumulator));
    for (var orthogonalizationRound = 0u; orthogonalizationRound < 2u; orthogonalizationRound++) {
      for (var previous = 0u; previous < column; previous++) {
        columnAccumulator = 0.0;
        for (var row = 0u; row < ${e}u; row++) { columnAccumulator += basis[previous * ${e}u + row] * basis[column * ${e}u + row]; }
        upper[previous * ${t}u + column] += columnAccumulator;
        for (var row = 0u; row < ${e}u; row++) { basis[column * ${e}u + row] -= columnAccumulator * basis[previous * ${e}u + row]; }
      }
    }
    columnAccumulator = 0.0;
    for (var row = 0u; row < ${e}u; row++) { columnAccumulator += basis[column * ${e}u + row] * basis[column * ${e}u + row]; }
    columnAccumulator = sqrt(columnAccumulator);
    if (!finite(columnAccumulator) || columnAccumulator <= maximumNorm * 1e-6) { return result; }
    upper[column * ${t}u + column] = columnAccumulator;
    for (var row = 0u; row < ${e}u; row++) {
      basis[column * ${e}u + row] /= columnAccumulator;
      projected[column] += basis[column * ${e}u + row] * response[row];
    }
  }
  for (var reverse = 0u; reverse < ${r}; reverse++) {
    let column = ${r} - 1u - reverse;
    var value = projected[column];
    for (var following = column + 1u; following < ${r}; following++) { value -= upper[column * ${t}u + following] * result.coefficients[following]; }
    result.coefficients[column] = value / upper[column * ${t}u + column];
    if (!finite(result.coefficients[column])) { return result; }
  }
  result.residual = 0.0;
  for (var row = 0u; row < ${e}u; row++) {
    var prediction = 0.0;
    for (var column = 0u; column < ${r}; column++) { prediction += design[column * ${e}u + row] * result.coefficients[column]; }
    result.residual += (prediction - response[row]) * (prediction - response[row]);
  }
  result.valid = finite(result.residual);
  return result;
}`}function Y(e,t){let n=[O(e,{id:`${t.id}-count-clear`,operation:t.operation,bindings:[{name:`counted`,view:t.output,type:`u32`,access:`read_write`}],invocationCount:1,body:`counted[countedOffset] = 0u;`})];return t.count>0&&n.push(O(e,{id:`${t.id}-count`,operation:t.operation,bindings:[K(`countValues`,t.values),{name:`counted`,view:t.output,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.count,guardIndex:!1,declarations:`var<workgroup> workgroupCount: atomic<u32>;`,body:`if (localInvocationIndex == 0u) { atomicStore(&workgroupCount, 0u); }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let value = countValues[countValuesOffset + index];
    if (${t.condition}) { atomicAdd(&workgroupCount, 1u); }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    let total = atomicLoad(&workgroupCount);
    if (total != 0u) { atomicAdd(&counted[countedOffset], total); }
  }`})),n}var X=256;function Z(e,t){let n=Math.ceil(t.count/X);H(e,t.id,[...t.bindings.map(e=>e.view),t.outputValue,t.outputIndex],{"argmax group values and indices":n*4});let r=i(e,`${t.id}-group-values`,`float32`,n),a=i(e,`${t.id}-group-indices`,`uint32`,n),o=`var<workgroup> argmaxSharedValues: array<f32, ${X}>;
var<workgroup> argmaxSharedIndices: array<u32, ${X}>;`,s=`workgroupBarrier();
  for (var argmaxStride = ${X/2}u; argmaxStride > 0u; argmaxStride = argmaxStride >> 1u) {
    if (localInvocationIndex < argmaxStride) {
      let argmaxOtherValue = argmaxSharedValues[localInvocationIndex + argmaxStride];
      let argmaxOtherIndex = argmaxSharedIndices[localInvocationIndex + argmaxStride];
      let argmaxValue = argmaxSharedValues[localInvocationIndex];
      if (argmaxOtherValue > argmaxValue || (argmaxOtherValue == argmaxValue && argmaxOtherIndex < argmaxSharedIndices[localInvocationIndex])) {
        argmaxSharedValues[localInvocationIndex] = argmaxOtherValue;
        argmaxSharedIndices[localInvocationIndex] = argmaxOtherIndex;
      }
    }
    workgroupBarrier();
  }`;return[O(e,{id:`${t.id}-argmax-groups`,operation:t.operation,invocationCount:t.count,workgroupSize:X,guardIndex:!1,bindings:[...t.bindings,K(`groupValues`,r,!0),K(`groupIndices`,a,!0)],declarations:`${t.declarations??``}\n${o}`,body:`var argmaxBestValue = ${t.initialValue};
  var argmaxBestIndex = 0xffffffffu;
  if (index < INVOCATION_COUNT) {
    var candidateValue = ${t.initialValue};
    var candidateEligible = false;
    ${t.candidate}
    if (candidateEligible && candidateValue > ${t.initialValue}) {
      argmaxBestValue = candidateValue;
      argmaxBestIndex = index;
    }
  }
  argmaxSharedValues[localInvocationIndex] = argmaxBestValue;
  argmaxSharedIndices[localInvocationIndex] = argmaxBestIndex;
  ${s}
  if (localInvocationIndex == 0u && workgroupIndex < ${n}u) {
    groupValues[groupValuesOffset + workgroupIndex] = argmaxSharedValues[0];
    groupIndices[groupIndicesOffset + workgroupIndex] = argmaxSharedIndices[0];
  }`}),O(e,{id:`${t.id}-argmax-final`,operation:t.operation,invocationCount:X,workgroupSize:X,guardIndex:!1,bindings:[K(`groupValues`,r),K(`groupIndices`,a),K(`outputValue`,t.outputValue,!0),K(`outputIndex`,t.outputIndex,!0)],declarations:o,body:`var argmaxBestValue = ${t.initialValue};
  var argmaxBestIndex = 0xffffffffu;
  for (var argmaxGroup = localInvocationIndex; argmaxGroup < ${n}u; argmaxGroup += ${X}u) {
    let argmaxValue = groupValues[groupValuesOffset + argmaxGroup];
    let argmaxGroupIndex = groupIndices[groupIndicesOffset + argmaxGroup];
    if (argmaxValue > argmaxBestValue || (argmaxValue == argmaxBestValue && argmaxGroupIndex < argmaxBestIndex)) {
      argmaxBestValue = argmaxValue;
      argmaxBestIndex = argmaxGroupIndex;
    }
  }
  argmaxSharedValues[localInvocationIndex] = argmaxBestValue;
  argmaxSharedIndices[localInvocationIndex] = argmaxBestIndex;
  ${s}
  if (localInvocationIndex == 0u) {
    outputValue[outputValueOffset] = argmaxSharedValues[0];
    outputIndex[outputIndexOffset] = select(argmaxSharedIndices[0], 0u, argmaxSharedIndices[0] == 0xffffffffu);
  }`})]}export{K as a,U as c,j as d,A as f,J as i,H as l,Z as n,G as o,F as p,Y as r,W as s,q as t,M as u};