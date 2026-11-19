import{a as e,c as t,d as n,i as r,o as i,r as a,s as o,t as s,u as c}from"./kernel-C3tfQOn4.js";import{l,r as u,u as d}from"./gpu-command-graph-BdzdmV6u.js";import{n as f,r as p}from"./gpu-command-dispatch-metadata-Bpi5I1Ez.js";import{n as m,t as h}from"./gpu-dispatch-utils-CbRzFAgT.js";import{i as g,n as _,t as v}from"./gpu-fft-utils-C1LfmzeI.js";import{t as y}from"./gpu-grid-coordinate-DwQPYH7E.js";import{n as b,r as x,t as S}from"./gpu-subgroup-utils-QYBHgloD.js";import{n as C,t as w}from"./gpu-spatial-utils-BiKW3pHz.js";import{t as ee}from"./gpu-group-aggregation-CEYb6GJw.js";import{t as te}from"./gpu-histogram-fxACbUIT.js";import{t as ne}from"./gpu-reduction-tNgOVQ_C.js";import{t as re}from"./gpu-raster-buffer-to-texture-BDZQU4A5.js";import{E as T,S as E,T as D,a as ie,c as O,f as k,g as ae,i as oe,p as se,w as ce,x as le}from"./gpu-contributor-utils-BIwLQON5.js";import{s as ue,v as de}from"./spatial-context-DwZEd4EZ.js";import{_ as fe,b as pe,c as me,d as he,f as ge,g as A,i as _e,l as ve,n as j,o as ye,p as be,s as xe,u as Se,x as Ce,y as M}from"./density-scratch-D0AtmvQz.js";import{n as we,r as Te}from"./fixed-point-accumulator-Ce6W3HVY.js";import{t as Ee}from"./define-run-B-brkCe4.js";import{n as De}from"./inputs-CJKKIRKK.js";import{c as Oe,n as ke}from"./pick-Bo3t3hjv.js";import{zs as Ae}from"./entry-preview-D70TdSbK.js";var N=class{id;input;kernel;output;width;height;kernelWidth;kernelHeight;strategy;boundary;stats;constructor(e){this.id=e.id??`gpu-convolution`,this.input=e.input,this.kernel=e.kernel,this.output=e.output,this.width=e.width,this.height=e.height,this.kernelWidth=e.kernelWidth,this.kernelHeight=e.kernelHeight,this.strategy=e.strategy??`auto`,this.boundary=e.boundary??`zero`,this.stats=P(e);for(let e of[this.input,this.kernel,this.output])for(let t of p(e))n(t,[`float32`],this.id);if(this.input.length<this.stats.elementCount)throw Error(`${this.id} input must contain at least width * height rows`);if(this.kernel.length<this.stats.kernelElementCount)throw Error(`${this.id} kernel must contain at least kernelWidth * kernelHeight rows`);if(this.output.length<this.stats.elementCount)throw Error(`${this.id} output must contain at least width * height rows`);let t=new Set(p(this.input).map(e=>e.buffer)),i=new Set(p(this.kernel).map(e=>e.buffer)),a=p(this.output);if(a.some(e=>t.has(e.buffer)||i.has(e.buffer)))throw Error(`${this.id} output must use a separate buffer from input and kernel`);if(p(this.input).some(e=>i.has(e.buffer)))throw Error(`${this.id} input and kernel must use separate buffers`);for(let[e,t]of a.entries())if(a.slice(0,e).some(e=>r(e,t)))throw Error(`${this.id} output chunks must not overlap`);if(![`auto`,`direct`,`fft`].includes(this.strategy))throw Error(`${this.id} strategy must be auto, direct, or fft`)}getCommandNodes(e){let t=[];V(e,this.input,`${this.id} input`),V(e,this.kernel,`${this.id} kernel`),V(e,this.output,`${this.id} output`);let n=F(e.device,{width:this.width,height:this.height,kernelWidth:this.kernelWidth,kernelHeight:this.kernelHeight,strategy:this.strategy,boundary:this.boundary,input:this.input,kernel:this.kernel,output:this.output});if(!n.supported||!n.strategy)throw Error(n.reason);return n.strategy===`direct`?t.push(...je(e,this)):t.push(...Me(e,this)),t}};function P(e){Be(e);let t=e.boundary??`zero`,n=e.width*e.height,r=e.kernelWidth*e.kernelHeight;if(!Number.isSafeInteger(n)||n>4294967295)throw Error(`GPUConvolution width * height must fit in a uint32 index range.`);if(!Number.isSafeInteger(r)||r>4294967295)throw Error(`GPUConvolution kernel dimensions must fit in a uint32 index range.`);let i=z(e.width,e.kernelWidth,t),a=z(e.height,e.kernelHeight,t),o=i*a,s=Math.log2(i)+Math.log2(a)+2,c=o*2*Float32Array.BYTES_PER_ELEMENT;return Object.freeze({width:e.width,height:e.height,kernelWidth:e.kernelWidth,kernelHeight:e.kernelHeight,boundary:t,elementCount:n,kernelElementCount:r,directMultiplyAddCount:n*r,fftWidth:i,fftHeight:a,fftElementCount:o,fftTransformPassCount:s,fftDispatchCount:3*s+3,fftComplexBufferByteLength:c,fftLogicalTransientByteLength:9*c})}function F(e,t){let n;try{n=P(t)}catch(e){return{supported:!1,reason:e.message}}if(e.type!==`webgpu`)return{supported:!1,reason:`GPUConvolution requires WebGPU.`,stats:n};if(e.limits.maxStorageBuffersPerShaderStage<3)return{supported:!1,reason:`GPUConvolution requires three compute storage buffers.`,stats:n};if(e.limits.maxComputeInvocationsPerWorkgroup<256||e.limits.maxComputeWorkgroupSizeX<256)return{supported:!1,reason:`GPUConvolution requires 256 compute invocations.`,stats:n};let r=Math.max(n.elementCount,n.kernelElementCount)*Float32Array.BYTES_PER_ELEMENT,i=t.input&&t.kernel&&t.output;if(!i&&(r>e.limits.maxStorageBufferBindingSize||r>e.limits.maxBufferSize))return{supported:!1,reason:`GPUConvolution scalar data exceeds device buffer limits.`,stats:n};if(i)for(let[r,i]of[[t.input,n.elementCount],[t.kernel,n.kernelElementCount],[t.output,n.elementCount]]){let t=i;for(let i of p(r)){let r=Math.min(t,i.length);if(r&&(o(i).size-(i.length-r)*4>e.limits.maxStorageBufferBindingSize||i.buffer.byteLength>e.limits.maxBufferSize))return{supported:!1,reason:`GPUConvolution chunk exceeds device buffer limits.`,stats:n};if(t-=r,!t)break}}let a=n.elementCount;if(t.output){let e=n.elementCount;a=0;for(let n of p(t.output)){let t=Math.min(e,n.length);if(a=Math.max(a,t),e-=t,!e)break}}try{h(`GPUConvolution`,a,256,e.limits.maxComputeWorkgroupsPerDimension)}catch(e){return{supported:!1,reason:e.message,stats:n}}let s=t.strategy??`auto`;if(![`auto`,`direct`,`fft`].includes(s))return{supported:!1,reason:`GPUConvolution strategy must be auto, direct, or fft.`,stats:n};let c=Ve(e,t,n),l=s===`auto`?n.kernelElementCount<=4096||c?`direct`:`fft`:s;return l===`fft`&&c?{supported:!1,reason:c,stats:n}:{supported:!0,stats:n,strategy:l}}function je(e,t){let n=[],r=I(e,t.input,t.stats.elementCount),i=I(e,t.kernel,t.stats.kernelElementCount),a=I(e,t.output,t.stats.elementCount);for(let o of a){let a=h(t.id,o.length,256,e.device.limits.maxComputeWorkgroupsPerDimension),s=!0;for(let c of i)for(let i of r){let r=Le({...t,input:i.data,kernel:c.data,output:o.data,inputLogicalOffset:i.offset,kernelLogicalOffset:c.offset,outputLogicalOffset:o.offset,initializeOutput:s},a);n.push(...R(e,{id:`${t.id}-direct-${n.length}`,source:r,bindings:{inputValues:i.data,kernelValues:c.data},outputs:{outputValues:o.data},dispatchLayout:a,operation:`GPUConvolution.direct`,readOutputs:!s,readByteLength:o.length*c.length*8+(s?0:o.length*4),writeByteLength:o.length*4})),s=!1}}return n}function I(t,n,r){return d(p(e(t,n,r))).filter(e=>e.length)}function Me(e,t){let n=[],{fftElementCount:r}=t.stats,i=a(e,`${t.id}-fft-packed-input`,`float32x2`,r),o=a(e,`${t.id}-fft-packed-kernel`,`float32x2`,r),s=a(e,`${t.id}-fft-input-spectrum`,`float32x2`,r),c=a(e,`${t.id}-fft-kernel-spectrum`,`float32x2`,r),l=a(e,`${t.id}-fft-product-spectrum`,`float32x2`,r),u=a(e,`${t.id}-fft-inverse-spatial`,`float32x2`,r),d=a(e,`${t.id}-fft-input-scratch`,`float32x2`,r),f=a(e,`${t.id}-fft-kernel-scratch`,`float32x2`,r),p=a(e,`${t.id}-fft-inverse-scratch`,`float32x2`,r);return n.push(...Ne(e,t,i,o)),n.push(...L(e,`${t.id}-fft-input`,i,s,d,t.stats.fftWidth,t.stats.fftHeight,`forward`)),n.push(...L(e,`${t.id}-fft-kernel`,o,c,f,t.stats.fftWidth,t.stats.fftHeight,`forward`)),n.push(...Pe(e,t,s,c,l)),n.push(...L(e,`${t.id}-fft-inverse`,l,u,p,t.stats.fftWidth,t.stats.fftHeight,`inverse`)),n.push(...Fe(e,t,u)),n}function Ne(e,n,r,i){let a=[],o=I(e,n.input,n.stats.elementCount),s=I(e,n.kernel,n.stats.kernelElementCount),c=h(`${n.id} FFT pack`,n.stats.fftElementCount,256,e.device.limits.maxComputeWorkgroupsPerDimension);if(o.length>1||s.length>1){for(let[l,u,d]of[[`input`,o,r],[`kernel`,s,i]])for(let[r,i]of u.entries()){let o=l===`input`?`let sourceX = i32(coordinateX);
  let sourceY = i32(coordinateY);`:`var signedX = i32(coordinateX);
  var signedY = i32(coordinateY);
  if (coordinateX > ${Math.floor(n.kernelWidth/2)}u) {
    signedX -= i32(FFT_WIDTH);
  }
  if (coordinateY > ${Math.floor(n.kernelHeight/2)}u) {
    signedY -= i32(FFT_HEIGHT);
  }
  let sourceX = signedX + ${Math.floor(n.kernelWidth/2)};
  let sourceY = signedY + ${Math.floor(n.kernelHeight/2)};`,s=l===`input`?n.width:n.kernelWidth,u=l===`input`?n.height:n.kernelHeight,f=`const FFT_ELEMENT_COUNT: u32 = ${n.stats.fftElementCount}u;
const FFT_WIDTH: u32 = ${n.stats.fftWidth}u;
const FFT_HEIGHT: u32 = ${n.stats.fftHeight}u;
@group(0) @binding(0) var<storage, read> sourceValues: array<f32>;
@group(0) @binding(1) var<storage, read_write> packedValues: array<vec2f>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${m(c,256)}
  if (index >= FFT_ELEMENT_COUNT) {
    return;
  }
  ${r===0?`packedValues[index] = vec2f(0.0);`:``}
  let coordinateY = index / FFT_WIDTH;
  let coordinateX = index % FFT_WIDTH;
  ${o}
  if (sourceX >= 0 && sourceX < ${s} && sourceY >= 0 && sourceY < ${u}) {
    let sourceIndex = u32(sourceY) * ${s}u + u32(sourceX);
    if (sourceIndex >= ${i.offset}u && sourceIndex - ${i.offset}u < ${i.length}u) {
      packedValues[index] = vec2f(sourceValues[${t(i.data)}u + sourceIndex - ${i.offset}u], 0.0);
    }
  }
}`;a.push(...R(e,{id:`${n.id}-fft-pack-${l}-${r}`,source:f,bindings:{sourceValues:i.data},outputs:{packedValues:d},dispatchLayout:c,operation:`GPUConvolution.fft.pack`,readByteLength:i.length*4,writeByteLength:r===0?n.stats.fftComplexBufferByteLength:i.length*8}))}return a}let l=Re({...n,input:o[0].data,kernel:s[0].data},c);return a.push(...R(e,{id:`${n.id}-fft-pack`,source:l,bindings:{inputValues:o[0].data,kernelValues:s[0].data},outputs:{packedInput:r,packedKernel:i},dispatchLayout:c,operation:`GPUConvolution.fft.pack`,readByteLength:(n.stats.elementCount+n.stats.kernelElementCount)*Float32Array.BYTES_PER_ELEMENT,writeByteLength:2*n.stats.fftComplexBufferByteLength})),a}function Pe(e,t,n,r,i){let a=[],o=h(`${t.id} FFT multiply`,t.stats.fftElementCount,256,e.device.limits.maxComputeWorkgroupsPerDimension),s=`${v}
const ELEMENT_COUNT: u32 = ${t.stats.fftElementCount}u;
@group(0) @binding(0) var<storage, read> inputSpectrum: array<vec2f>;
@group(0) @binding(1) var<storage, read> kernelSpectrum: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> productSpectrum: array<vec2f>;
@compute @workgroup_size(256) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${m(o,256)}
  if (index >= ELEMENT_COUNT) { return; }
  productSpectrum[index] = multiplyComplex(inputSpectrum[index], kernelSpectrum[index]);
}`;return a.push(...R(e,{id:`${t.id}-fft-multiply`,source:s,bindings:{inputSpectrum:n,kernelSpectrum:r},outputs:{productSpectrum:i},dispatchLayout:o,operation:`GPUConvolution.fft.multiply`,readByteLength:2*t.stats.fftComplexBufferByteLength,writeByteLength:t.stats.fftComplexBufferByteLength})),a}function Fe(e,n,r){let i=[],a=I(e,n.output,n.stats.elementCount);for(let[o,s]of a.entries()){let a=h(`${n.id} FFT crop`,s.length,256,e.device.limits.maxComputeWorkgroupsPerDimension),c=`const ELEMENT_COUNT: u32 = ${s.length}u;
const WIDTH: u32 = ${n.width}u;
const FFT_WIDTH: u32 = ${n.stats.fftWidth}u;
@group(0) @binding(0) var<storage, read> inverseSpatial: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<f32>;
@compute @workgroup_size(256) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${m(a,256)}
  if (index >= ELEMENT_COUNT) {
    return;
  }
  let row = ${s.offset}u + index;
  let outputY = row / WIDTH;
  let outputX = row % WIDTH;
  outputValues[${t(s.data)}u + index] =
    inverseSpatial[outputY * FFT_WIDTH + outputX].x;
}`;i.push(...R(e,{id:`${n.id}-fft-crop-${o}`,source:c,bindings:{inverseSpatial:r},outputs:{outputValues:s.data},dispatchLayout:a,operation:`GPUConvolution.fft.crop`,readByteLength:s.length*2*Float32Array.BYTES_PER_ELEMENT,writeByteLength:s.length*Float32Array.BYTES_PER_ELEMENT}))}return i}function L(e,t,n,r,i,a,o,s){let c=[],l=[...g(a).map(e=>({axis:`horizontal`,...e})),...g(o).map(e=>({axis:`vertical`,...e}))],u=n;for(let[n,d]of l.entries()){let f=(l.length-n)%2==0?i:r;c.push(...Ie(e,{id:`${t}-${d.axis}-${d.kind}-${d.stage}`,input:u,output:f,width:a,height:o,direction:s,pass:d,finalPass:n===l.length-1})),u=f}return c}function Ie(e,t){let n=[],r=t.width*t.height,i=h(t.id,r,256,e.device.limits.maxComputeWorkgroupsPerDimension),a=ze(t,i);return n.push(...R(e,{id:t.id,source:a,bindings:{inputValues:t.input},outputs:{outputValues:t.output},dispatchLayout:i,operation:`GPUConvolution.fft.transform`,readByteLength:r*2*Float32Array.BYTES_PER_ELEMENT,writeByteLength:r*2*Float32Array.BYTES_PER_ELEMENT})),n}function R(e,t){let n=[],r=[...Object.entries(t.bindings),...Object.entries(t.outputs)];return n.push(u({id:t.id,workload:{operation:t.operation,commandCount:1,maximumWorkgroupCount:t.dispatchLayout.x*t.dispatchLayout.y*t.dispatchLayout.z,maximumInvocationCount:t.dispatchLayout.x*t.dispatchLayout.y*t.dispatchLayout.z*256,readByteLength:t.readByteLength,writeByteLength:t.writeByteLength},resources:[...Object.values(t.bindings).map(e=>({buffer:e,usage:`storage-read`})),...Object.values(t.outputs).map(e=>({buffer:e,usage:t.readOutputs?`storage-read-write`:`storage-write`}))],compile:({device:e})=>{let n=new s(e,{id:t.id,source:t.source,shaderLayout:{bindings:r.map(([e],n)=>({name:e,type:n<Object.keys(t.bindings).length?`read-only-storage`:`storage`,group:0,location:n}))}});return{encode:({computePass:e,getBuffer:a})=>{let o={};for(let[e,t]of r)o[e]=i(t,a);n.dispatch(e,{bindings:o,x:t.dispatchLayout.x,y:t.dispatchLayout.y,z:t.dispatchLayout.z})},destroy:()=>n.destroy()}}})),n}function Le(e,n){let r=e.boundary===`wrap`?`let wrappedX = ((sourceX % i32(WIDTH)) + i32(WIDTH)) % i32(WIDTH);
    let wrappedY = ((sourceY % i32(HEIGHT)) + i32(HEIGHT)) % i32(HEIGHT);
    let sourceIndex = u32(wrappedY) * WIDTH + u32(wrappedX);`:`if (sourceX < 0 || sourceX >= i32(WIDTH) || sourceY < 0 || sourceY >= i32(HEIGHT)) {
      continue;
    }
    let sourceIndex = u32(sourceY) * WIDTH + u32(sourceX);`;return`const ELEMENT_COUNT: u32 = ${Math.min(e.output.length,e.stats.elementCount)}u;
const WIDTH: u32 = ${e.width}u;
const HEIGHT: u32 = ${e.height}u;
const KERNEL_WIDTH: u32 = ${e.kernelWidth}u;
const KERNEL_HEIGHT: u32 = ${e.kernelHeight}u;
const KERNEL_CENTER_X: u32 = ${Math.floor(e.kernelWidth/2)}u;
const KERNEL_CENTER_Y: u32 = ${Math.floor(e.kernelHeight/2)}u;
const INPUT_OFFSET: u32 = ${t(e.input)}u;
const KERNEL_OFFSET: u32 = ${t(e.kernel)}u;
const OUTPUT_OFFSET: u32 = ${t(e.output)}u;
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> kernelValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;
@compute @workgroup_size(256) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${m(n,256)}
  if (index >= ELEMENT_COUNT) {
    return;
  }
  let outputIndex = ${e.outputLogicalOffset??0}u + index;
  let outputY = outputIndex / WIDTH;
  let outputX = outputIndex - outputY * WIDTH;
  var total = 0.0;
  for (var kernelLocalIndex = 0u; kernelLocalIndex < ${Math.min(e.kernel.length,e.stats.kernelElementCount)}u; kernelLocalIndex++) {
    let kernelIndex = ${e.kernelLogicalOffset??0}u + kernelLocalIndex;
    let kernelY = kernelIndex / KERNEL_WIDTH;
    let kernelX = kernelIndex % KERNEL_WIDTH;
    let sourceY = i32(outputY) - (i32(kernelY) - i32(KERNEL_CENTER_Y));
    let sourceX = i32(outputX) - (i32(kernelX) - i32(KERNEL_CENTER_X));
    let kernelValue = kernelValues[KERNEL_OFFSET + kernelLocalIndex];
    ${r}
    if (sourceIndex >= ${e.inputLogicalOffset??0}u &&
        sourceIndex - ${e.inputLogicalOffset??0}u < ${Math.min(e.input.length,e.stats.elementCount)}u) {
      total += inputValues[INPUT_OFFSET + sourceIndex - ${e.inputLogicalOffset??0}u] * kernelValue;
    }
  }
  outputValues[OUTPUT_OFFSET + index] ${e.initializeOutput===!1?`+=`:`=`} total;
}`}function Re(e,n){return`const FFT_ELEMENT_COUNT: u32 = ${e.stats.fftElementCount}u;
const WIDTH: u32 = ${e.width}u;
const HEIGHT: u32 = ${e.height}u;
const FFT_WIDTH: u32 = ${e.stats.fftWidth}u;
const FFT_HEIGHT: u32 = ${e.stats.fftHeight}u;
const KERNEL_WIDTH: u32 = ${e.kernelWidth}u;
const KERNEL_HEIGHT: u32 = ${e.kernelHeight}u;
const KERNEL_CENTER_X: i32 = ${Math.floor(e.kernelWidth/2)};
const KERNEL_CENTER_Y: i32 = ${Math.floor(e.kernelHeight/2)};
const INPUT_OFFSET: u32 = ${t(e.input)}u;
const KERNEL_OFFSET: u32 = ${t(e.kernel)}u;
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> kernelValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> packedInput: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> packedKernel: array<vec2f>;
@compute @workgroup_size(256) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${m(n,256)}
  if (index >= FFT_ELEMENT_COUNT) { return; }
  let coordinateY = index / FFT_WIDTH;
  let coordinateX = index - coordinateY * FFT_WIDTH;
  var inputValue = 0.0;
  if (coordinateX < WIDTH && coordinateY < HEIGHT) {
    inputValue = inputValues[INPUT_OFFSET + coordinateY * WIDTH + coordinateX];
  }
  var signedX = i32(coordinateX);
  var signedY = i32(coordinateY);
  if (coordinateX > u32(KERNEL_CENTER_X)) { signedX -= i32(FFT_WIDTH); }
  if (coordinateY > u32(KERNEL_CENTER_Y)) { signedY -= i32(FFT_HEIGHT); }
  let kernelX = signedX + KERNEL_CENTER_X;
  let kernelY = signedY + KERNEL_CENTER_Y;
  var kernelValue = 0.0;
  if (kernelX >= 0 && kernelX < i32(KERNEL_WIDTH) &&
      kernelY >= 0 && kernelY < i32(KERNEL_HEIGHT)) {
    kernelValue = kernelValues[KERNEL_OFFSET + u32(kernelY) * KERNEL_WIDTH + u32(kernelX)];
  }
  packedInput[index] = vec2f(inputValue, 0.0);
  packedKernel[index] = vec2f(kernelValue, 0.0);
}`}function ze(e,t){let n=e.width*e.height,r=e.pass.axis===`horizontal`,i=e.direction===`forward`?`-1.0`:`1.0`,a=e.direction===`inverse`&&e.finalPass?`${1/n}`:`1.0`,o=`const ELEMENT_COUNT: u32 = ${n}u;
const WIDTH: u32 = ${e.width}u;
const INPUT_OFFSET: u32 = ${He(e.input)}u;
const OUTPUT_OFFSET: u32 = ${He(e.output)}u;
@group(0) @binding(0) var<storage, read> inputValues: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<vec2f>;
${v}`,s=`let coordinateY = index / WIDTH;
  let coordinateX = index - coordinateY * WIDTH;`;if(e.pass.kind===`bit-reversal`){let n=r?`coordinateY * WIDTH + reverseLowBits(coordinateX, ${e.pass.stage}u)`:`reverseLowBits(coordinateY, ${e.pass.stage}u) * WIDTH + coordinateX`;return`${o}
@compute @workgroup_size(256) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${m(t,256)}
  if (index >= ELEMENT_COUNT) { return; }
  ${s}
  outputValues[OUTPUT_OFFSET + index] = inputValues[INPUT_OFFSET + ${n}];
}`}let c=1<<e.pass.stage,l=c>>1,u=r?`coordinateX`:`coordinateY`,d=r?`coordinateY * WIDTH + firstCoordinate`:`firstCoordinate * WIDTH + coordinateX`,f=r?`coordinateY * WIDTH + secondCoordinate`:`secondCoordinate * WIDTH + coordinateX`;return`${o}
@compute @workgroup_size(256) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${m(t,256)}
  if (index >= ELEMENT_COUNT) { return; }
  ${s}
  let butterflyOffset = ${u} & ${c-1}u;
  let twiddleIndex = butterflyOffset & ${l-1}u;
  let butterflyStart = ${u} - butterflyOffset;
  let firstCoordinate = butterflyStart + twiddleIndex;
  let secondCoordinate = firstCoordinate + ${l}u;
  let firstValue = inputValues[INPUT_OFFSET + ${d}];
  let secondValue = inputValues[INPUT_OFFSET + ${f}];
  let angle = ${i} * 2.0 * GPU_FFT_PI * f32(twiddleIndex) /
    f32(${c}u);
  let rotatedSecondValue = multiplyComplex(secondValue, vec2f(cos(angle), sin(angle)));
  let butterflyValue = select(
    firstValue + rotatedSecondValue,
    firstValue - rotatedSecondValue,
    butterflyOffset >= ${l}u
  );
  outputValues[OUTPUT_OFFSET + index] = butterflyValue * ${a};
}`}function Be(e){for(let[t,n]of[[`width`,e.width],[`height`,e.height],[`kernelWidth`,e.kernelWidth],[`kernelHeight`,e.kernelHeight]])if(!Number.isSafeInteger(n)||n<=0)throw Error(`GPUConvolution ${t} must be a positive integer.`);if(e.kernelWidth%2==0||e.kernelHeight%2==0)throw Error(`GPUConvolution kernel dimensions must be odd.`);if(e.boundary!==void 0&&e.boundary!==`zero`&&e.boundary!==`wrap`)throw Error(`GPUConvolution boundary must be zero or wrap.`)}function z(e,t,n){let r=n===`zero`?e+t-1:e;return Math.max(2,2**Math.ceil(Math.log2(r)))}function Ve(e,t,n){if(t.boundary===`wrap`){if(!B(t.width)||!B(t.height))return`GPUConvolution FFT wrap boundary requires power-of-two input dimensions.`;if(t.kernelWidth>t.width||t.kernelHeight>t.height)return`GPUConvolution FFT wrap boundary requires the kernel to fit inside the input field.`}if(n.fftWidth>2048||n.fftHeight>2048)return`GPUConvolution FFT dimensions must not exceed ${_}.`;if(e.limits.maxStorageBuffersPerShaderStage<4)return`GPUConvolution FFT strategy requires four compute storage buffers.`;if(n.fftComplexBufferByteLength>e.limits.maxStorageBufferBindingSize||n.fftComplexBufferByteLength>e.limits.maxBufferSize)return`GPUConvolution FFT scratch exceeds device buffer limits.`;try{h(`GPUConvolution FFT`,n.fftElementCount,256,e.limits.maxComputeWorkgroupsPerDimension)}catch(e){return e.message}}function B(e){return!(e&e-1)}function He(e){return e.byteOffset%256/(2*Float32Array.BYTES_PER_ELEMENT)}function V(e,t,n){if(p(t).some(t=>t.buffer.graph!==e))throw Error(`${n} belongs to a different GPUCommandGraph`)}var H=256,Ue=256,We=16,Ge=class{id;positions;output;gridSize;bounds;constructor(e){this.id=e.id??`gpu-grid-binning`,this.positions=e.positions,this.output=e.output,this.gridSize=e.gridSize,this.bounds=e.bounds;for(let e of G(this.positions))n(e,[`float32x2`],`${this.id} positions`);for(let e of p(this.output))c(e,`${this.id} output`);C(this.id,G(this.positions),p(this.output));let[t,r]=this.gridSize;if(!Number.isSafeInteger(t)||!Number.isSafeInteger(r)||t<=0||r<=0)throw Error(`${this.id} gridSize must contain two positive integers`);if(!Number.isSafeInteger(t*r)||t*r>4294967295||this.output.length!==t*r)throw Error(`${this.id} output.length must equal gridSize width * height`);if(G(this.positions).some(e=>p(this.output).some(t=>e.buffer===t.buffer)))throw Error(`${this.id} positions and output must use separate buffers`);if(Array.isArray(this.bounds)){let[e,t,n,r]=this.bounds;if(this.bounds.length!==4||!this.bounds.every(Number.isFinite)||e>n||t>r)throw Error(`${this.id} literal bounds must be finite [minX, minY, maxX, maxY]`)}else if(W(this.bounds)){if(n(this.bounds,[`float32x4`],`${this.id} bounds`),this.bounds.length!==1)throw Error(`${this.id} GPU bounds must contain one float32x4 row`);if(p(this.output).some(e=>e.buffer===this.bounds.buffer))throw Error(`${this.id} bounds and output must use separate buffers`)}}getCommandNodes(e){let t=[];if(G(this.positions).some(t=>t.buffer.graph!==e)||p(this.output).some(t=>t.buffer.graph!==e))throw Error(`${this.id} views must belong to the target graph`);if(W(this.bounds)&&this.bounds.buffer.graph!==e)throw Error(`${this.id} bounds must belong to the target graph`);let n=0;for(let[r,i]of p(this.output).entries()){if(i.length){let a=`${this.id}-output-${r}`;t.push(...Ke(e,a,i));for(let[r,o]of G(this.positions).entries())o.length&&t.push(...qe(e,{id:`${a}-input-${r}`,positions:o,output:i,outputStart:n,gridSize:this.gridSize,bounds:this.bounds}))}n+=i.length}return t}};function Ke(e,n,r){let i=[],a=`${n}-clear`,o=h(n,r.length,H,e.device.limits.maxComputeWorkgroupsPerDimension),s=`
const CELL_COUNT: u32 = ${r.length}u;
const OUTPUT_OFFSET: u32 = ${t(r)}u;
@group(0) @binding(0) var<storage, read_write> outputCounts: array<atomic<u32>>;
@compute @workgroup_size(${H}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${m(o,H)}
  if (index < CELL_COUNT) { atomicStore(&outputCounts[OUTPUT_OFFSET + index], 0u); }
}`;return i.push(...w(e,{id:a,source:s,resources:[{buffer:r,usage:`storage-write`}],bindings:{outputCounts:r},dispatch:o})),i}function qe(e,n){let r=[],[i,a]=n.gridSize,o=h(n.id,n.positions.length,H,e.device.limits.maxComputeWorkgroupsPerDimension),s=n.output.length<=Ue,c=s&&n.output.length<=We&&S(e.device)===`subgroups`,l=W(n.bounds),u=n.bounds,d=l?`@group(0) @binding(1) var<storage, read> boundsValues: array<f32>;`:``,f=l?2:1,p=l?`let minimumX = boundsValues[BOUNDS_OFFSET];
  let minimumY = boundsValues[BOUNDS_OFFSET + 1u];
  let maximumX = boundsValues[BOUNDS_OFFSET + 2u];
  let maximumY = boundsValues[BOUNDS_OFFSET + 3u];`:`let minimumX = ${U(u[0])};
  let minimumY = ${U(u[1])};
  let maximumX = ${U(u[2])};
  let maximumY = ${U(u[3])};`,g=c?x(`accepted`,`cellIndex`,`localCounts`,n.output.length):`  if (accepted) { atomicAdd(&localCounts[cellIndex], 1u); }`,_=s?`${g}
  workgroupBarrier();
  if (lane < CELL_COUNT) {
    atomicAdd(&outputCounts[OUTPUT_OFFSET + lane], atomicLoad(&localCounts[lane]));
  }`:`if (accepted) { atomicAdd(&outputCounts[OUTPUT_OFFSET + cellIndex], 1u); }`,v=`
${c?`enable subgroups;`:``}
const ELEMENT_COUNT: u32 = ${n.positions.length}u;
const WIDTH: u32 = ${i}u;
const HEIGHT: u32 = ${a}u;
const CELL_COUNT: u32 = ${n.output.length}u;
const OUTPUT_START: u32 = ${n.outputStart}u;
const POSITIONS_OFFSET: u32 = ${t(n.positions)}u;
${l?`const BOUNDS_OFFSET: u32 = ${t(n.bounds)}u;`:``}
const OUTPUT_OFFSET: u32 = ${t(n.output)}u;
@group(0) @binding(0) var<storage, read> positions: array<f32>;
${d}
@group(0) @binding(${f}) var<storage, read_write> outputCounts: array<atomic<u32>>;
${s?`var<workgroup> localCounts: array<atomic<u32>, ${n.output.length}>;`:``}
${c?b():``}

${y}

@compute @workgroup_size(${H}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(local_invocation_id) localId: vec3<u32>${c?`,
  @builtin(subgroup_invocation_id) subgroupInvocationId: u32`:``}
) {
  ${m(o,H)}
  let lane = localId.x;
  ${p}
  ${s?`if (lane < CELL_COUNT) { atomicStore(&localCounts[lane], 0u); }
  workgroupBarrier();`:``}
  var accepted = false;
  var cellIndex = 0u;
  if (index < ELEMENT_COUNT && maximumX >= minimumX && maximumY >= minimumY) {
    let x = positions[POSITIONS_OFFSET + index * 2u];
    let y = positions[POSITIONS_OFFSET + index * 2u + 1u];
    let finite = x == x && y == y && abs(x) <= 3.402823466e+38 && abs(y) <= 3.402823466e+38;
    let inX = x >= minimumX && x <= maximumX && (maximumX != minimumX || x == minimumX);
    let inY = y >= minimumY && y <= maximumY && (maximumY != minimumY || y == minimumY);
    if (finite && inX && inY) {
      let column = getCoordinate(x, minimumX, maximumX, WIDTH);
      let row = getCoordinate(y, minimumY, maximumY, HEIGHT);
      let globalCell = row * WIDTH + column;
      accepted = globalCell >= OUTPUT_START && globalCell - OUTPUT_START < CELL_COUNT;
      if (accepted) { cellIndex = globalCell - OUTPUT_START; }
    }
  }
  ${_}
}`,C=[{buffer:n.positions,usage:`storage-read`},...l?[{buffer:n.bounds,usage:`storage-read`}]:[],{buffer:n.output,usage:`storage-read-write`}];return r.push(...w(e,{id:`${n.id}-${s?`local`:`global`}`,source:v,resources:C,bindings:{positions:n.positions,...l?{boundsValues:n.bounds}:{},outputCounts:n.output},dispatch:o})),r}function U(e){return Number.isInteger(e)?`${e}.0`:`${e}`}function W(e){return!Array.isArray(e)}function G(e){return e instanceof l?e.data:[e]}var K=256,Je=16,Ye=class{id;positions;weights;output;operation;gridSize;bounds;constructor(e){this.id=e.id??`gpu-grid-aggregation`,this.positions=e.positions,this.weights=e.weights,this.output=e.output,this.operation=e.operation??`sum`,this.gridSize=e.gridSize,this.bounds=e.bounds;for(let e of Y(this.positions))n(e,[`float32x2`],`${this.id} positions`);for(let e of X(this.weights))n(e,[`float32`],`${this.id} weights`);if(this.positions.length!==this.weights.length)throw Error(`${this.id} positions and weights must contain the same number of rows`);for(let e of p(this.output))n(e,[`float32`],`${this.id} output`);C(this.id,[...Y(this.positions),...X(this.weights)],p(this.output));let[t,r]=this.gridSize;if(!Number.isSafeInteger(t)||!Number.isSafeInteger(r)||t<=0||r<=0)throw Error(`${this.id} gridSize must contain two positive integers`);if(!Number.isSafeInteger(t*r)||t*r>4294967295||this.output.length!==t*r)throw Error(`${this.id} output.length must equal gridSize width * height`);if(![`sum`,`min`,`max`,`mean`].includes(this.operation))throw Error(`${this.id} operation must be sum, min, max, or mean`);if(this.operation===`mean`&&this.positions.length>4294967295)throw Error(`${this.id} mean input length must fit in uint32 cell counts`);if(Y(this.positions).some(e=>p(this.output).some(t=>e.buffer===t.buffer))||X(this.weights).some(e=>p(this.output).some(t=>e.buffer===t.buffer)))throw Error(`${this.id} inputs and output must use separate buffers`);if(Array.isArray(this.bounds)){let[e,t,n,r]=this.bounds;if(this.bounds.length!==4||!this.bounds.every(Number.isFinite)||e>n||t>r)throw Error(`${this.id} literal bounds must be finite [minX, minY, maxX, maxY]`)}else if(J(this.bounds)){if(n(this.bounds,[`float32x4`],`${this.id} bounds`),this.bounds.length!==1)throw Error(`${this.id} GPU bounds must contain one float32x4 row`);if(p(this.output).some(e=>e.buffer===this.bounds.buffer))throw Error(`${this.id} bounds and output must use separate buffers`)}}getCommandNodes(e){let t=[],n=Y(this.positions),r=X(this.weights);if(n.some(t=>t.buffer.graph!==e)||r.some(t=>t.buffer.graph!==e)||p(this.output).some(t=>t.buffer.graph!==e))throw Error(`${this.id} views must belong to the target graph`);if(J(this.bounds)&&this.bounds.buffer.graph!==e)throw Error(`${this.id} bounds must belong to the target graph`);let i=f(e,[this.positions,this.weights]),o=0;for(let[n,r]of p(this.output).entries()){if(r.length){let s=`${this.id}-output-${n}`,c=this.operation===`mean`?a(e,`${s}-counts`,`uint32`,r.length):void 0;t.push(...Xe(e,s,r,this.operation,c));for(let[n,[a,l]]of i.entries())t.push(...Ze(e,{id:`${s}-input-${n}`,positions:a,weights:l,output:r,outputStart:o,operation:this.operation,counts:c,gridSize:this.gridSize,bounds:this.bounds}));this.operation!==`sum`&&t.push(...Qe(e,s,r,this.operation,c))}o+=r.length}return t}};function Xe(e,n,r,i,a){let o=[],s=h(n,r.length,K,e.device.limits.maxComputeWorkgroupsPerDimension),c=i===`min`?`0xffffffffu`:`0u`,l=a?`@group(0) @binding(1) var<storage, read_write> outputCounts: array<atomic<u32>>;`:``,u=a?`atomicStore(&outputCounts[${t(a)}u + index], 0u);`:``,d=`
const CELL_COUNT: u32 = ${r.length}u;
const OUTPUT_OFFSET: u32 = ${t(r)}u;
@group(0) @binding(0) var<storage, read_write> outputValues: array<atomic<u32>>;
${l}
@compute @workgroup_size(${K}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${m(s,K)}
  if (index < CELL_COUNT) {
    atomicStore(&outputValues[OUTPUT_OFFSET + index], ${c});
    ${u}
  }
}`;return o.push(...w(e,{id:i===`sum`?`${n}-clear`:`${n}-initialize`,source:d,resources:[{buffer:r,usage:`storage-write`},...a?[{buffer:a,usage:`storage-write`}]:[]],bindings:{outputValues:r,...a?{outputCounts:a}:{}},dispatch:s})),o}function Ze(e,n){let r=[],[i,a]=n.gridSize,o=h(n.id,n.positions.length,K,e.device.limits.maxComputeWorkgroupsPerDimension),s=n.output.length<=Je&&S(e.device)===`subgroups`,c=J(n.bounds),l=n.bounds,u=c?`@group(0) @binding(2) var<storage, read> boundsValues: array<f32>;`:``,d=c?3:2,f=n.counts?`@group(0) @binding(${d+1}) var<storage, read_write> outputCounts: array<atomic<u32>>;`:``,p=c?`let minimumX = boundsValues[BOUNDS_OFFSET];
  let minimumY = boundsValues[BOUNDS_OFFSET + 1u];
  let maximumX = boundsValues[BOUNDS_OFFSET + 2u];
  let maximumY = boundsValues[BOUNDS_OFFSET + 3u];`:`let minimumX = ${q(l[0])};
  let minimumY = ${q(l[1])};
  let maximumX = ${q(l[2])};
  let maximumY = ${q(l[3])};`,g=s?tt(n.operation,n.output.length,n.counts):`  if (accepted) {
    ${et(n.operation)}
    ${n.counts?`atomicAdd(&outputCounts[${t(n.counts)}u + cellIndex], 1u);`:``}
  }`,_=`
${s?`enable subgroups;`:``}
const ELEMENT_COUNT: u32 = ${n.positions.length}u;
const WIDTH: u32 = ${i}u;
const HEIGHT: u32 = ${a}u;
const OUTPUT_START: u32 = ${n.outputStart}u;
const CELL_COUNT: u32 = ${n.output.length}u;
const POSITIONS_OFFSET: u32 = ${t(n.positions)}u;
const WEIGHTS_OFFSET: u32 = ${t(n.weights)}u;
${c?`const BOUNDS_OFFSET: u32 = ${t(n.bounds)}u;`:``}
const OUTPUT_OFFSET: u32 = ${t(n.output)}u;
@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
${u}
@group(0) @binding(${d}) var<storage, read_write> outputValues: array<atomic<u32>>;
${f}

${y}

${$e(n.operation)}
${s?b():``}

@compute @workgroup_size(${K}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32${s?`,
  @builtin(subgroup_invocation_id) subgroupInvocationId: u32`:``}
) {
  ${m(o,K)}
  ${p}
  var accepted = false;
  var cellIndex = 0u;
  var weight = 0.0;
  if (index < ELEMENT_COUNT && maximumX >= minimumX && maximumY >= minimumY) {
    let x = positions[POSITIONS_OFFSET + index * 2u];
    let y = positions[POSITIONS_OFFSET + index * 2u + 1u];
    weight = weights[WEIGHTS_OFFSET + index];
    let finitePosition = x == x && y == y && abs(x) <= 3.402823466e+38 && abs(y) <= 3.402823466e+38;
    let finiteWeight = weight == weight && abs(weight) <= 3.402823466e+38;
    let inX = x >= minimumX && x <= maximumX && (maximumX != minimumX || x == minimumX);
    let inY = y >= minimumY && y <= maximumY && (maximumY != minimumY || y == minimumY);
    if (finitePosition && finiteWeight && inX && inY) {
      let column = getCoordinate(x, minimumX, maximumX, WIDTH);
      let row = getCoordinate(y, minimumY, maximumY, HEIGHT);
      let globalCell = row * WIDTH + column;
      accepted = globalCell >= OUTPUT_START && globalCell - OUTPUT_START < CELL_COUNT;
      if (accepted) { cellIndex = globalCell - OUTPUT_START; }
    }
  }
${g}
}`,v=[{buffer:n.positions,usage:`storage-read`},{buffer:n.weights,usage:`storage-read`},...c?[{buffer:n.bounds,usage:`storage-read`}]:[],{buffer:n.output,usage:`storage-read-write`},...n.counts?[{buffer:n.counts,usage:`storage-read-write`}]:[]];return r.push(...w(e,{id:`${n.id}-${n.operation}`,source:_,resources:v,bindings:{positions:n.positions,weights:n.weights,...c?{boundsValues:n.bounds}:{},outputValues:n.output,...n.counts?{outputCounts:n.counts}:{}},dispatch:o})),r}function Qe(e,n,r,i,a){let o=[],s=h(n,r.length,K,e.device.limits.maxComputeWorkgroupsPerDimension),c=a?`@group(0) @binding(1) var<storage, read> outputCounts: array<u32>;`:``,l=i===`mean`?`let count = outputCounts[${t(a)}u + index];
    if (count == 0u) {
      outputValues[OUTPUT_OFFSET + index] = 0x7fc00000u;
    } else {
      let sum = bitcast<f32>(outputValues[OUTPUT_OFFSET + index]);
      outputValues[OUTPUT_OFFSET + index] = bitcast<u32>(sum / f32(count));
    }`:`let orderedValue = outputValues[OUTPUT_OFFSET + index];
    if (orderedValue == ${i===`min`?`0xffffffffu`:`0u`}) {
      outputValues[OUTPUT_OFFSET + index] = 0x7fc00000u;
    } else {
      outputValues[OUTPUT_OFFSET + index] = bitcast<u32>(decodeOrderedFloat(orderedValue));
    }`,u=i===`mean`?``:`fn decodeOrderedFloat(value: u32) -> f32 {
  let bits = select(~value, value ^ 0x80000000u, (value & 0x80000000u) != 0u);
  return bitcast<f32>(bits);
}`,d=`
const CELL_COUNT: u32 = ${r.length}u;
const OUTPUT_OFFSET: u32 = ${t(r)}u;
@group(0) @binding(0) var<storage, read_write> outputValues: array<u32>;
${c}
${u}
@compute @workgroup_size(${K}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${m(s,K)}
  if (index < CELL_COUNT) {
    ${l}
  }
}`;return o.push(...w(e,{id:`${n}-finalize`,source:d,resources:[{buffer:r,usage:`storage-read-write`},...a?[{buffer:a,usage:`storage-read`}]:[]],bindings:{outputValues:r,...a?{outputCounts:a}:{}},dispatch:s})),o}function $e(e){return e===`min`||e===`max`?`fn encodeOrderedFloat(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}`:`fn atomicAddFloat(destination: ptr<storage, atomic<u32>, read_write>, value: f32) {
  var oldBits = atomicLoad(destination);
  loop {
    let newBits = bitcast<u32>(bitcast<f32>(oldBits) + value);
    let result = atomicCompareExchangeWeak(destination, oldBits, newBits);
    if (result.exchanged) { break; }
    oldBits = result.old_value;
  }
}`}function et(e,t=`cellIndex`,n=`weight`){return e===`min`||e===`max`?`${e===`min`?`atomicMin`:`atomicMax`}(&outputValues[OUTPUT_OFFSET + ${t}], encodeOrderedFloat(${n}));`:`atomicAddFloat(&outputValues[OUTPUT_OFFSET + ${t}], ${n});`}function tt(e,n,r){let i=e===`min`||e===`max`;return`
  var subgroupPending = accepted;
  for (var subgroupCell = 0u; subgroupCell < ${n}u; subgroupCell++) {
    let pendingBallot = subgroupBallot(subgroupPending);
    let hasPending = any(pendingBallot != vec4<u32>(0u));
    let leaderInvocation = getFirstBallotLane(pendingBallot);
    let leaderCell = subgroupShuffle(cellIndex, leaderInvocation);
    let matchingCell = hasPending && subgroupPending && cellIndex == leaderCell;
    let matchingBallot = subgroupBallot(matchingCell);
    let selectedWeight = ${i?`select(${e===`min`?`0xffffffffu`:`0u`}, encodeOrderedFloat(weight), matchingCell)`:`select(0.0, weight, matchingCell)`};
    let aggregatedWeight = ${e===`min`?`subgroupMin(selectedWeight)`:e===`max`?`subgroupMax(selectedWeight)`:`subgroupAdd(selectedWeight)`};
    if (hasPending && subgroupInvocationId == leaderInvocation) {
      ${i?`${e===`min`?`atomicMin`:`atomicMax`}(&outputValues[OUTPUT_OFFSET + leaderCell], aggregatedWeight);`:et(e,`leaderCell`,`aggregatedWeight`)}
      ${r?`atomicAdd(&outputCounts[${t(r)}u + leaderCell], getBallotLaneCount(matchingBallot));`:``}
    }
    subgroupPending = subgroupPending && !matchingCell;
  }`}function q(e){return Number.isInteger(e)?`${e}.0`:`${e}`}function J(e){return!Array.isArray(e)}function Y(e){return e instanceof l?e.data:[e]}function X(e){return e instanceof l?e.data:[e]}function nt(e,t,n,r,i,o){let s=[],c=[],u=[];for(let l=0;l<Math.ceil(n.length/o);l++){let d=`${t}-tile-${l}`,f=l*o,p=Math.min(f+o,n.length),m=[];for(let[t,l,h]of[[`positions`,n,2143289344],[`weights`,r,0],[`mask`,i,0]]){if(!l)continue;let n=a(e,`${d}-${t}`,l.format,o);t===`positions`&&c.push(n),t===`weights`&&u.push(n),m.push(n);let r=l.rowByteLength/4;s.push(A(e,{id:`${d}-${t}-pad`,operation:`GPUPointDensity`,invocationCount:o,bindings:[{name:`destination`,view:n,type:`u32`,access:`read_write`}],body:Array.from({length:r},(e,t)=>`destination[destinationOffset + ${r}u * index + ${t}u] = ${h}u;`).join(`
`)}));let i=0;for(let[a,o]of O(l).entries()){let c=i+o.length,l=Math.max(f,i),u=Math.min(p,c);if(l<u){let c=e.createDataView(o.buffer,{format:o.format,length:u-l,byteOffset:o.byteOffset+(l-i)*o.rowByteLength});s.push(A(e,{id:`${d}-${t}-gather-${a}`,operation:`GPUPointDensity`,invocationCount:u-l,bindings:[{name:`source`,view:c,type:`u32`,access:`read`},{name:`destination`,view:n,type:`u32`,access:`read_write`}],runtimeConstants:{DESTINATION_ROW_BASE:l-f},body:Array.from({length:r},(e,t)=>`destination[destinationOffset + ${r}u * (DESTINATION_ROW_BASE + index) + ${t}u] = source[sourceOffset + ${r}u * index + ${t}u];`).join(`
`)}))}i=c}}i&&s.push(A(e,{id:`${d}-mask`,operation:`GPUPointDensity`,invocationCount:o,bindings:[{name:`positions`,view:c[l],type:`u32`,access:`read_write`},{name:`mask`,view:m[m.length-1],type:`u32`,access:`read`}],body:`if (mask[maskOffset + index] == 0u) { positions[positionsOffset + 2u * index] = 0x7fc00000u; positions[positionsOffset + 2u * index + 1u] = 0x7fc00000u; }`}))}let d=c.length*o;return{positions:new l({id:`${t}-position-tiles`,name:`density point tiles`,format:`float32x2`,length:d,valueLength:d*2,stride:2,byteStride:8,rowByteLength:8,data:c}),weights:r?new l({id:`${t}-weight-tiles`,name:`density weight tiles`,format:`float32`,length:d,valueLength:d,stride:1,byteStride:4,rowByteLength:4,data:u}):void 0,nodes:s}}var rt=4294967295,Z=1073741824,it=32,at=4294967295,ot=Object.freeze({...ue,measure:`area`}),st=ie(12),ct=512;function lt(e){return e===`atomic`?`atomic`:`workgroup`}function Q(e){return e?e.separableKernel?[e.separableKernel.horizontal,e.separableKernel.vertical]:e.kernel?[e.kernel]:[]:[]}var ut=class{id;scratchTileSize;usesTiledScratch;props;spatialContext;binning;statistic;sumAccumulation;cellCount;estimatedCommandNodeCount;surface;constructor(e){this.id=e.id??`point-density`,this.props=e,this.spatialContext=de(this.id,e.spatialContext,ot),this.binning=e.binning??`grid`,this.statistic=e.statistic??`count`;let t=this.id,[n,r]=e.gridSize;this.cellCount=n*r,this.sumAccumulation=lt(e.sumAccumulation),this.surface={values:e.output.values,columns:n,rows:r};let{output:i,weights:a,smoothing:o,mask:s}=e,c=O(e.positions)[0],l=c?c.buffer.graph.device.limits:void 0,u=l?Math.min(l.maxStorageBufferBindingSize,l.maxBufferSize):8;if(this.scratchTileSize=e.scratchTileSize??2**Math.floor(Math.log2(Math.max(1,Math.min(Z,u/8)))),k(t,`scratchTileSize`,this.scratchTileSize,1,Z),ce(t,`scratchTileSize`,this.scratchTileSize),this.usesTiledScratch=e.scratchTileSize===void 0?this.binning===`hexagon`&&e.positions.length*4>u||!!s&&this.binning===`grid`&&e.positions.length*8>u:e.positions.length>this.scratchTileSize,this.estimatedCommandNodeCount=ct+8*(O(e.positions).length+(a?O(a).length:0)),this.usesTiledScratch&&(this.estimatedCommandNodeCount+=it*Math.ceil(e.positions.length/this.scratchTileSize)),E(t,`positions.length`,e.positions.length,rt),se(t,this.estimatedCommandNodeCount),![`grid`,`hexagon`].includes(this.binning))throw Error(`${t} binning must be grid or hexagon`);if(![`cube`,`d3`].includes(e.hexagonAssignment??`cube`))throw Error(`${t} hexagonAssignment must be cube or d3`);if(![`count`,`sum`,`mean`].includes(this.statistic))throw Error(`${t} statistic must be count, sum, or mean`);if(![`auto`,`workgroup`,`atomic`].includes(e.sumAccumulation??`auto`))throw Error(`${t} sumAccumulation must be auto, workgroup, or atomic`);if(k(t,`gridSize columns`,n),k(t,`gridSize rows`,r),this.cellCount>=4294967295)throw Error(`${t} gridSize cell count must be below 2^32 - 1`);a&&this.sumAccumulation===`workgroup`&&E(t,`gridSize cells`,this.cellCount,st);for(let n of O(e.positions))T(n,[`float32x2`],`${t} positions`);if(a){for(let e of O(a))T(e,[`float32`],`${t} weights`);if(a.length!==e.positions.length)throw Error(`${t} weights length must equal positions length`)}for(let[e,n]of[[this.statistic!==`count`,`statistic ${this.statistic}`],[!!i.sums,`output.sums`],[!!i.means,`output.means`]])if(e&&!a)throw Error(`${t} ${n} requires weights`);if(s&&(T(s,[`uint32`],`${t} mask`),s.length!==e.positions.length))throw Error(`${t} mask length must equal positions length`);if(dt(t,e.bounds),this.binning===`hexagon`){let n=e.hexagonRadius;if(n===void 0)throw Error(`${t} hexagon binning requires hexagonRadius`);if(typeof n==`number`)ae(t,`hexagonRadius`,n,0),D(t,Math.fround(n)>0,`hexagonRadius must be positive in float32`);else if(T(n,[`float32`],`${t} hexagonRadius`),n.length<1)throw Error(`${t} hexagonRadius must contain one float32 row`);if(o)throw Error(`${t} smoothing requires grid binning`)}else if(e.hexagonRadius!==void 0)throw Error(`${t} hexagonRadius requires hexagon binning`);if(o){for(let[e,n]of[[`smoothing.kernelWidth`,o.kernelWidth],[`smoothing.kernelHeight`,o.kernelHeight]])if(k(t,e,n,1,at),n%2==0)throw Error(`${t} smoothing kernel sizes must be positive odd integers`);let{separableKernel:e}=o;if(e){if(T(e.horizontal,[`float32`],`${t} smoothing.separableKernel.horizontal`),T(e.vertical,[`float32`],`${t} smoothing.separableKernel.vertical`),e.horizontal.length<o.kernelWidth||e.vertical.length<o.kernelHeight)throw Error(`${t} smoothing.separableKernel is shorter than kernelWidth or kernelHeight`)}else{if(!o.kernel)throw Error(`${t} smoothing requires kernel or separableKernel`);if(T(o.kernel,[`float32`],`${t} smoothing.kernel`),o.kernel.length<o.kernelWidth*o.kernelHeight)throw Error(`${t} smoothing.kernel is shorter than kernelWidth * kernelHeight`)}}let d=[[`values`,i.values,`float32`],[`counts`,i.counts,`uint32`],[`sums`,i.sums,`float32`],[`means`,i.means,`float32`]];for(let[e,n,r]of d)if(n&&(T(n,[r],`${t} output.${e}`),n.length!==this.cellCount))throw Error(`${t} output.${e} must have one row per cell`);if(i.extent&&(T(i.extent,[`float32`],`${t} output.extent`),i.extent.length!==2))throw Error(`${t} output.extent must have two rows`);for(let[e,n]of[[`histogram`,i.histogram],[`status.overflow`,i.status?.overflow]])if(n&&(T(n,[`uint32`],`${t} output.${e}`),n.length<1))throw Error(`${t} output.${e} must have at least one row`);if(i.texture&&(i.texture.format!==`r32float`||i.texture.width!==n||i.texture.height!==r))throw Error(`${t} output.texture must be r32float with gridSize extent`);Ce(t,{"output.values":i.values,"output.counts":i.counts,"output.sums":i.sums,"output.means":i.means,"output.extent":i.extent,"output.histogram":i.histogram,"output.status.overflow":i.status?.overflow});let f=[i.values,i.counts,i.sums,i.means,i.extent,i.histogram,i.status?.overflow].filter(e=>e!==void 0).map(e=>e.buffer),p=[...O(e.positions),...a?O(a):[],...s?[s]:[],...Array.isArray(e.bounds)?[]:[e.bounds],...typeof e.hexagonRadius==`object`?[e.hexagonRadius]:[],...Q(o)].map(e=>e.buffer);if(f.some(e=>p.includes(e)))throw Error(`${t} outputs must use separate buffers`)}getCommandNodes(e){let{id:t,props:n,statistic:r,sumAccumulation:i,cellCount:o}=this,{smoothing:s,output:c,gridSize:l}=n,u=n.positions,d=n.weights,f=n.mask,p=c;pe(e,t,[[`positions`,u],[`weights`,d],[`mask`,f],[`bounds`,Array.isArray(n.bounds)?void 0:n.bounds],[`hexagonRadius`,typeof n.hexagonRadius==`object`?n.hexagonRadius:void 0],...Q(s).map(e=>[`smoothing kernel`,e]),[`output.values`,p.values],[`output.counts`,p.counts],[`output.sums`,p.sums],[`output.means`,p.means],[`output.extent`,p.extent],[`output.histogram`,p.histogram],[`output.status.overflow`,p.status?.overflow]],{skipEmptyChunks:!0});let m=!!d&&(r!==`count`||!!(p.sums||p.means));if(M(e,t,[[`grid scratch`,o*4]]),(p.histogram||p.extent)&&M(e,t,[[`extent scratch`,8]]),p.histogram&&M(e,t,[[`histogram scratch`,p.histogram.length*4]]),m&&i===`workgroup`&&M(e,t,[[`fixed-point grid scratch`,o*4*3]]),this.usesTiledScratch?(M(e,t,[[`point tile scratch`,this.scratchTileSize*8]]),k(t,`point tile scratch byteLength`,this.scratchTileSize*8,0,e.device.limits.maxBufferSize)):this.binning===`hexagon`?M(e,t,[[`point key scratch`,u.length*4]]):m&&i===`workgroup`&&ye(e,t,u,4),f&&this.binning===`grid`&&!this.usesTiledScratch&&M(e,t,[[`masked position scratch`,u.length*8]]),s&&!s.separableKernel){let n=F(e.device,{width:l[0],height:l[1],kernelWidth:s.kernelWidth,kernelHeight:s.kernelHeight,strategy:s.strategy??`auto`,boundary:`zero`});(n.strategy===`fft`||s.strategy===`fft`)&&M(e,t,[[`smoothing FFT scratch`,n.stats?.fftComplexBufferByteLength??0]]),D(t,n.supported,`smoothing must fit device limits: ${n.reason??``}`)}let h=Array.isArray(n.bounds)?void 0:n.bounds,g=typeof n.hexagonRadius==`object`?n.hexagonRadius:void 0;if(le(t,e,[u,d,f,h,g,...Q(s),p.values,p.counts,p.sums,p.means,p.extent,p.histogram,p.status?.overflow]),p.texture&&p.texture.texture.graph!==e)throw Error(`${t} views must belong to the target graph`);let _=fe(e,t,`GPUPointDensity`,{values:p.values,counts:p.counts,sums:p.sums,means:p.means,extent:p.extent,histogram:p.histogram,overflow:p.status?.overflow},{overflow:1});p={...c,..._.views,status:c.status?{overflow:_.views.overflow}:void 0};let v=this.usesTiledScratch?nt(e,t,u,d,f,this.scratchTileSize):void 0;v&&(u=v.positions,d=v.weights,f=void 0);let y=h&&h.format===`float32`?e.createDataView(h.buffer,{format:`float32x4`,length:1,byteOffset:h.byteOffset}):n.bounds,b=p.counts??a(e,`${t}-counts-scratch`,`uint32`,o),x=m?p.sums??a(e,`${t}-sums-scratch`,`float32`,o):void 0,S=s?a(e,`${t}-unsmoothed-values`,`float32`,o):p.values,C=p.extent??(p.histogram?a(e,`${t}-extent-scratch`,`float32`,2):void 0),w=[...v?.nodes??[]];if(p.status?.overflow&&w.push(xe(e,`${t}-clear-overflow`,p.status.overflow)),this.binning===`hexagon`){let r=j(e,`${t}-hexagon-keys`,u,`uint32`,!this.usesTiledScratch),a=O(r),o=0;for(let[r,i]of O(u).entries())i.length>0&&w.push(Se(e,{id:`${t}-hexagon-keys-${r}`,positions:i,keys:a[this.usesTiledScratch?r:0],keyStart:this.usesTiledScratch?0:o,gridSize:l,bounds:y,radius:n.hexagonRadius,assignment:n.hexagonAssignment,mask:f,maskStart:o,overflow:p.status?.overflow})),o+=i.length;w.push(...new ee({id:`${t}-counts`,keys:r,output:b}).getCommandNodes(e)),x&&d&&(i===`atomic`?w.push(...new ee({id:`${t}-sums`,keys:r,values:d,output:x,operation:`sum`}).getCommandNodes(e)):w.push(...ft(e,`${t}-sums`,d,r,x,n.positions.length)))}else{let r=u;if(f){let n=a(e,`${t}-masked-positions`,`float32x2`,u.length),i=0;for(let[r,a]of O(u).entries())a.length>0&&w.push(he(e,{id:`${t}-mask-positions-${r}`,positions:a,mask:f,maskedPositions:n,rowStart:i})),i+=a.length;r=n}if(w.push(...new Ge({id:`${t}-counts`,positions:r,output:b,gridSize:l,bounds:y}).getCommandNodes(e)),x&&d){if(i===`atomic`)w.push(...new Ye({id:`${t}-sums`,positions:r,weights:d,output:x,operation:`sum`,gridSize:l,bounds:y}).getCommandNodes(e));else{let i=j(e,`${t}-grid-keys`,r,`uint32`),a=O(i);for(let[n,i]of O(r).entries())i.length>0&&w.push(ve(e,{id:`${t}-sums-grid-keys-${n}`,positions:i,keys:a[n],keyStart:0,gridSize:l,bounds:y}));w.push(...ft(e,`${t}-sums`,d,i,x,n.positions.length))}}}if(w.push(me(e,{id:`${t}-finalize`,statistic:r,counts:b,sums:x,values:S,means:p.means})),s?.separableKernel){let n=a(e,`${t}-smooth-rows`,`float32`,o),r={width:l[0],height:l[1],boundary:`zero`};w.push(...new N({...r,id:`${t}-smooth-horizontal`,input:S,kernel:s.separableKernel.horizontal,output:n,kernelWidth:s.kernelWidth,kernelHeight:1,strategy:`direct`}).getCommandNodes(e),...new N({...r,id:`${t}-smooth-vertical`,input:n,kernel:s.separableKernel.vertical,output:p.values,kernelWidth:1,kernelHeight:s.kernelHeight,strategy:`direct`}).getCommandNodes(e))}else s&&w.push(...new N({id:`${t}-smooth`,input:S,kernel:s.kernel,output:p.values,width:l[0],height:l[1],kernelWidth:s.kernelWidth,kernelHeight:s.kernelHeight,boundary:`zero`,strategy:s.strategy??`auto`}).getCommandNodes(e));let T=s?void 0:b;C&&w.push(...new ne({id:`${t}-extent`,input:p.values,mask:T,output:C,operation:`extent`}).getCommandNodes(e));let E=p.histogram;if(E&&C&&w.push(...new te({id:`${t}-histogram`,input:p.values,mask:T,output:E,domain:C}).getCommandNodes(e)),p.texture){let n=p.texture;w.push(...oe(e,()=>new re({id:`${t}-texture`,input:{id:`${t}-texture-band`,format:`float32`,storage:{kind:`buffer`,values:p.values}},output:n}).addToGraph(e)))}return w.push(..._.publishNodes),w}};function dt(e,t){if(Array.isArray(t)){let[n,r,i,a]=t;if(t.length!==4||!t.every(e=>Number.isFinite(Math.fround(e)))||n>i||r>a)throw Error(`${e} bounds must be finite float32 [minX, minY, maxX, maxY]`);return}let n=t;if(T(n,[`float32x4`,`float32`],`${e} bounds`),n.format===`float32x4`&&n.length!==1||n.format===`float32`&&n.length!==4)throw Error(`${e} bounds view must be one float32x4 row or four float32 rows`)}function ft(e,t,n,r,i,o=n.length){let s=a(e,`${t}-scale`,`uint32`,1),c=a(e,`${t}-fixed-point`,`uint32`,i.length*3),l=[...ge(e,{id:`${t}-scale`,values:n,scale:s}),we(e,{id:`${t}-clear`,operation:`GPUPointDensity`,words:c})];for(let[i,a]of _e(e,n,r).entries())l.push(be(e,{id:`${t}-workgroup-${i}`,weights:a.weights,keys:a.keys,keyStart:0,fixedPointSums:c,scale:s,rowCount:o}));return l.push(Te(e,{id:`${t}-decode`,operation:`GPUPointDensity`,words:c,scale:s,maximumTermCount:o,output:i})),l}function pt(e,t){if(!Number.isInteger(e)||e<0||e>32)throw Error(`Gaussian kernel radius must be an integer in [0, 32]`);let n=t??Math.max(e/2,.5);if(!Number.isFinite(n)||n<=0)throw Error(`Gaussian kernel sigma must be positive and finite`);let r=2*e+1,i=new Float64Array(r*r),a=0;for(let t=0;t<r;t++)for(let o=0;o<r;o++){let s=o-e,c=t-e,l=Math.exp(-(s*s+c*c)/(2*n*n));i[t*r+o]=l,a+=l}return Float32Array.from(i,e=>e/a)}var $=[`count`,`sum`,`mean`],mt=Ee({parameters:Ae,caption:`Events binned into cells; the dots are the input points.`,createInputs:({random:e,world:t})=>({...(e=>({...e,index:ke(e.positions)}))(De(e,t,900,4,.12)),width:t.width}),build(e,t,n){let r=n.gridSize,i=Math.round(r*t.width),a=Float32Array.from(t.positions,(e,t)=>t%2==1?1-e:e),o=e.upload(`positions`,`float32x2`,a),s=e.upload(`weights`,`float32`,t.values),c=e.allocate(`values`,`float32`,i*r),l=n.smoothing;return e.graph.add(new ut({positions:o.view,weights:s.view,bounds:[0,0,t.width,1],gridSize:[i,r],statistic:$[n.statistic],smoothing:l>0?{kernel:e.upload(`kernel`,`float32`,pt(l)).view,kernelWidth:2*l+1,kernelHeight:2*l+1}:void 0,output:{values:c.view}})),{outputs:{cells:c,columns:i,rows:r,dots:e.upload(`dots`,`float32x2`,t.positions),binned:e.upload(`binned`,`float32x2`,new Float32Array(t.count*2).fill(NaN))}}},interaction:{kind:`probe`,hint:`hover a cell`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=[0,0,t.width,1],o=Math.max(0,t.index.nearest(e.x,e.y)),s=Oe(n.columns,n.rows,a,t.positions[o*2],t.positions[o*2+1]),c=t.width/n.columns,l=1/n.rows,u=[s.column*c,1-(s.row+1)*l],d=new Float32Array(t.count*2).fill(NaN),f=0;for(let e=0;e<t.count;e++){let n=t.positions[e*2],r=t.positions[e*2+1];n<u[0]||n>=u[0]+c||r<u[1]||r>=u[1]+l||(d.set([n,r],f*2),f++)}n.binned.write(d);let p=i(n.cells)?.[s.index];return{glyphs:[{shape:`box`,minimum:u,maximum:[u[0]+c,u[1]+l]}],readout:`cell ${s.column}, ${s.row} · ${f} events · ${$[r.statistic]} ${p===void 0?`...`:p.toFixed(2)}`}}},draw:({outputs:e,values:t})=>[{layer:`raster`,values:e.cells,columns:e.columns,rows:e.rows,range:[[0,10],[0,5],[0,1]][t.statistic]},{layer:`dots`,context:!0,positions:e.dots,radius:2.4},{layer:`dots`,positions:e.binned,radius:4,color:[1,.9,.35,1]}]});export{mt as default};