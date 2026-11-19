import{P as e}from"./dist-BusamGlx.js";import{c as t,d as n,o as r,r as i,t as a}from"./kernel-C3tfQOn4.js";import{c as o,d as s,f as c,h as l,i as u,l as d,n as f,o as p,r as m,s as h,u as g}from"./raster-utils-CztuVGLY.js";import{T as _,g as v,i as y,x as b}from"./gpu-contributor-utils-BIwLQON5.js";import{u as x}from"./wgsl-common-DfyXiYB0.js";import{t as S}from"./construction-snapshot-DdkX9Pc0.js";import{t as C}from"./raster-view-validation-CSMxj-cg.js";var w=class{id;metadata;bands;pixelCount;graph;bandsById=new Map;constructor(e){if(this.id=e.id??`gpu-raster`,this.metadata=e.metadata,this.bands=e.bands,this.pixelCount=s(this.metadata,this.id),this.bands.length===0)throw Error(`${this.id} requires at least one band`);let t;for(let e of this.bands){if(this.bandsById.has(e.id))throw Error(`${this.id} band identifiers must be unique`);let n=g(e,this.metadata,`${this.id} ${e.id}`);if(t&&n!==t)throw Error(`${this.id} bands must belong to the same graph`);t=n,this.bandsById.set(e.id,e)}this.graph=t}getBand(e){let t=this.bandsById.get(e);if(!t)throw Error(`${this.id} does not contain band ${e}`);return t}getPixelWorldPosition(e,t){let n=this.metadata.pixelInterpretation===`area`?.5:0,[r,i,a,o,s,c]=this.metadata.affine;return[r*(e+n)+i*(t+n)+a,o*(e+n)+s*(t+n)+c]}getPixelArea(){let[e,t,,n,r]=this.metadata.affine;return Math.abs(e*r-t*n)}isCompatibleWith(e){return this.metadata.width===e.metadata.width&&this.metadata.height===e.metadata.height&&this.metadata.pixelInterpretation===e.metadata.pixelInterpretation&&this.metadata.level===e.metadata.level&&this.metadata.affine.every((t,n)=>t===e.metadata.affine[n])&&d(this.metadata.coordinateReferenceSystem,e.metadata.coordinateReferenceSystem)&&T(this.metadata.levelZeroOrigin,e.metadata.levelZeroOrigin)}assertCompatibleWith(e){if(!this.isCompatibleWith(e))throw Error(`${this.id} raster grids and coordinate systems must match`)}};function T(e,t){return e===t?!0:!e||!t?!1:e[0]===t[0]&&e[1]===t[1]}var E=class{id;input;output;outputValidity;applyCalibration;width;height;constructor(e){if(this.id=e.id??`gpu-raster-texture-to-buffer`,this.input=e.input,this.output=e.output,this.outputValidity=e.outputValidity,this.applyCalibration=e.applyCalibration??!1,this.input.storage.kind!==`texture`)throw Error(`${this.id} requires a texture-backed input band`);this.width=this.input.storage.view.width,this.height=this.input.storage.view.height;let t=this.width*this.height,n=g(this.input,this,this.id),r=this.applyCalibration?`float32`:this.input.format;if(c(this.output,r,t,`${this.id} output`),l(this.outputValidity,t,`${this.id} output validity`),this.output.buffer.graph!==n||this.outputValidity.buffer.graph!==n)throw Error(`${this.id} resources must belong to the same graph`);if(this.output.buffer===this.outputValidity.buffer||this.input.validity?.buffer===this.output.buffer||this.input.validity?.buffer===this.outputValidity.buffer)throw Error(`${this.id} source validity and outputs must use separate buffers`);this.applyCalibration&&(u(this.input.scale??1),u(this.input.offset??0))}addToGraph(e){if(this.input.storage.view.texture.graph!==e||this.output.buffer.graph!==e||this.outputValidity.buffer.graph!==e||this.input.validity&&this.input.validity.buffer.graph!==e)throw Error(`${this.id} resources must belong to the target graph`);f(e.device,this.output,`${this.id} output`),f(e.device,this.outputValidity,`${this.id} output validity`),this.input.validity&&f(e.device,this.input.validity,`${this.id} input validity`);let[t,n]=m(e.device,this.width,this.height,this.id),i=[{texture:this.input.storage.view,usage:`sampled`},{buffer:this.output,usage:`storage-write`},{buffer:this.outputValidity,usage:`storage-write`}];this.input.validity&&i.push({buffer:this.input.validity,usage:`storage-read`}),e.addComputePass({id:this.id,resources:i,compile:({device:e})=>{let i=[{name:`sourceTexture`,type:`texture`,group:0,location:0,sampleType:o(this.input.format)},{name:`outputValues`,type:`storage`,group:0,location:1},{name:`outputValidity`,type:`storage`,group:0,location:2}];this.input.validity&&i.push({name:`sourceValidity`,type:`read-only-storage`,group:0,location:3});let s=new a(e,{id:this.id,source:this.getShaderSource(),shaderLayout:{bindings:i}});return{encode:({computePass:e,getBuffer:i,getTextureView:a})=>{let o={sourceTexture:a(this.input.storage.view),outputValues:r(this.output,i),outputValidity:r(this.outputValidity,i)};this.input.validity&&(o.sourceValidity=r(this.input.validity,i));let c=o;s.dispatch(e,{bindings:c,x:t,y:n})},destroy:()=>s.destroy()}}})}getShaderSource(){let e=h(this.input.format),n=h(this.output.format),r=[`x`,`y`,`z`,`w`][this.input.storage.channel??0],i=this.input.validity,a=i?`@group(0) @binding(3) var<storage, read> sourceValidity: array<u32>;\nconst SOURCE_VALIDITY_OFFSET: u32 = ${t(i)}u;`:``,o=[];i&&o.push(`sourceValidity[SOURCE_VALIDITY_OFFSET + pixelIndex] != 0u`),this.input.format===`float32`&&o.push(`rawSample == rawSample && abs(rawSample) <= 3.402823466e+38`),this.input.noDataValue!==void 0&&!Number.isNaN(this.input.noDataValue)&&o.push(`rawSample != ${p(this.input.noDataValue,this.input.format)}`);let s=this.applyCalibration?`f32(rawSample) * ${u(this.input.scale??1)} + ${u(this.input.offset??0)}`:`rawSample`;this.applyCalibration&&o.push(`outputSample == outputSample && abs(outputSample) <= 3.402823466e+38`);let c=o.join(` && `)||`true`,l=this.applyCalibration?`let outputSample = ${s};`:``,d=this.output.format===`float32`?`bitcast<f32>(0x7fc00000u | (pixelIndex & 0u))`:this.output.format===`uint32`?`0u`:`0i`;return`
const WIDTH: u32 = ${this.width}u;
const HEIGHT: u32 = ${this.height}u;
const OUTPUT_OFFSET: u32 = ${t(this.output)}u;
const OUTPUT_VALIDITY_OFFSET: u32 = ${t(this.outputValidity)}u;
@group(0) @binding(0) var sourceTexture: texture_2d<${e}>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<${n}>;
@group(0) @binding(2) var<storage, read_write> outputValidity: array<u32>;
${a}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= WIDTH || globalId.y >= HEIGHT) { return; }
  let pixelIndex = globalId.y * WIDTH + globalId.x;
  let rawSample = textureLoad(sourceTexture, vec2<i32>(globalId.xy), 0).${r};
  ${l}
  let validSample = ${c};
  outputValidity[OUTPUT_VALIDITY_OFFSET + pixelIndex] = select(0u, 1u, validSample);
  if (validSample) {
    outputValues[OUTPUT_OFFSET + pixelIndex] = ${this.applyCalibration?`outputSample`:`rawSample`};
  } else {
    outputValues[OUTPUT_OFFSET + pixelIndex] = ${d};
  }
}`}},D=8,O=class{id;width;height;input;output;outputValidity;radius;horizontalRadius;verticalRadius;requiredHalo;kernel;borderMode;borderValue;noDataPolicy;normalize;kernelSum;constructor(e){if(this.id=e.id??`gpu-raster-neighborhood`,this.width=e.width,this.height=e.height,this.input=e.input,this.output=e.output,this.outputValidity=e.outputValidity,this.radius=e.radius,this.borderMode=e.borderMode??`clamp`,this.borderValue=e.borderValue??0,this.noDataPolicy=e.noDataPolicy??`propagate`,this.normalize=e.normalize??!1,!Number.isSafeInteger(this.width)||this.width<=0||!Number.isSafeInteger(this.height)||this.height<=0)throw Error(`${this.id} dimensions must be positive integers`);let t=this.width*this.height;if(!Number.isSafeInteger(t)||t>4294967295)throw Error(`${this.id} pixel count must fit in uint32`);let n=typeof this.radius==`number`?[this.radius,this.radius]:this.radius;if(!Array.isArray(n)||n.length!==2||!n.every(e=>Number.isSafeInteger(e)&&e>=0&&e<=D))throw Error(`${this.id} neighborhood radii must be integers from zero through eight`);this.horizontalRadius=n[0],this.verticalRadius=n[1],this.requiredHalo=Math.max(this.horizontalRadius,this.verticalRadius);let r=(this.horizontalRadius*2+1)*(this.verticalRadius*2+1);if(!Array.isArray(e.kernel)||e.kernel.length!==r)throw Error(`${this.id} kernel must contain exactly one coefficient per neighborhood`);for(let t of e.kernel){if(!Number.isFinite(t))throw Error(`${this.id} kernel coefficients must be finite`);u(t)}if(this.kernel=Object.freeze(e.kernel.map(e=>Math.fround(e))),this.kernelSum=this.kernel.reduce((e,t)=>e+t,0),u(this.kernelSum),![`clamp`,`reflect`,`constant`,`nodata`].includes(this.borderMode))throw Error(`${this.id} border mode must be clamp, reflect, constant, or nodata`);if(!Number.isFinite(this.borderValue))throw Error(`${this.id} constant border value must be finite`);if(u(this.borderValue),![`propagate`,`ignore-renormalize`].includes(this.noDataPolicy))throw Error(`${this.id} nodata policy must be propagate or ignore-renormalize`);if(typeof this.normalize!=`boolean`)throw Error(`${this.id} normalize must be a boolean`);if(this.noDataPolicy===`ignore-renormalize`&&this.kernel.some(e=>e<0))throw Error(`${this.id} ignore-renormalize requires nonnegative kernel coefficients`);if((this.normalize||this.noDataPolicy===`ignore-renormalize`)&&this.kernelSum===0)throw Error(`${this.id} normalized kernels must have a nonzero coefficient sum`);if(this.input.storage.kind!==`buffer`)throw Error(`${this.id} requires a buffer-backed input band`);let i=g(this.input,this,`${this.id} input`);if(c(this.output,`float32`,t,`${this.id} output`),l(this.outputValidity,t,`${this.id} output validity`),this.output.buffer.graph!==i||this.outputValidity.buffer.graph!==i)throw Error(`${this.id} resources must belong to the same graph`);let a=[this.input.storage.values.buffer,...this.input.validity?[this.input.validity.buffer]:[]];if(this.output.buffer===this.outputValidity.buffer||a.includes(this.output.buffer)||a.includes(this.outputValidity.buffer))throw Error(`${this.id} neighborhood inputs and outputs must use separate buffers`);u(this.input.scale??1),u(this.input.offset??0)}addToGraph(e){let t=[this.input.storage.values,this.output,this.outputValidity,...this.input.validity?[this.input.validity]:[]];for(let n of t){if(n.buffer.graph!==e)throw Error(`${this.id} resources must belong to the target graph`);f(e.device,n,`${this.id} ${n.buffer.id}`)}let n=m(e.device,this.width,this.height,this.id);if((8+this.horizontalRadius*2)*(8+this.verticalRadius*2)*Float32Array.BYTES_PER_ELEMENT*2>e.device.limits.maxComputeWorkgroupStorageSize)throw Error(`${this.id} neighborhood exceeds the device workgroup storage limit`);if(t.length>e.device.limits.maxStorageBuffersPerShaderStage)throw Error(`${this.id} neighborhood exceeds the device storage binding count`);let i=[{buffer:this.input.storage.values,usage:`storage-read`},{buffer:this.output,usage:`storage-write`},{buffer:this.outputValidity,usage:`storage-write`}];this.input.validity&&i.push({buffer:this.input.validity,usage:`storage-read`}),e.addComputePass({id:this.id,resources:i,compile:({device:e})=>{let t=[{name:`sourceValues`,type:`read-only-storage`,group:0,location:0},{name:`outputValues`,type:`storage`,group:0,location:1},{name:`outputValidity`,type:`storage`,group:0,location:2}];this.input.validity&&t.push({name:`sourceValidity`,type:`read-only-storage`,group:0,location:3});let i=new a(e,{id:this.id,source:this.getShaderSource(),shaderLayout:{bindings:t}});return{encode:({computePass:e,getBuffer:t})=>{let a={sourceValues:r(this.input.storage.values,t),outputValues:r(this.output,t),outputValidity:r(this.outputValidity,t)};this.input.validity&&(a.sourceValidity=r(this.input.validity,t));let o=a;i.dispatch(e,{bindings:o,x:n[0],y:n[1]})},destroy:()=>i.destroy()}}})}getShaderSource(){let e=8+this.horizontalRadius*2,n=e*(8+this.verticalRadius*2),r=this.input.validity?`@group(0) @binding(3) var<storage, read> sourceValidity: array<u32>;\nconst SOURCE_VALIDITY_OFFSET: u32 = ${t(this.input.validity)}u;`:``,i=[`isFiniteValue(sample)`];this.input.format===`float32`&&i.push(`isFiniteValue(rawSample)`),this.input.validity&&i.push(`sourceValidity[SOURCE_VALIDITY_OFFSET + sourceIndex] != 0u`),this.input.noDataValue!==void 0&&!Number.isNaN(this.input.noDataValue)&&i.push(`rawSample != ${p(this.input.noDataValue,this.input.format)}`);let a=this.kernel.map(u).join(`, `),o=k(this.borderMode,this.borderValue),s=this.noDataPolicy===`propagate`?`neighborhoodIsValid = false;`:``,c=this.normalize?`weightedSum / participatingWeight`:this.noDataPolicy===`ignore-renormalize`?`weightedSum * ${u(this.kernelSum)} / participatingWeight`:`weightedSum`;return`
const WIDTH: u32 = ${this.width}u;
const HEIGHT: u32 = ${this.height}u;
const RADIUS_X: i32 = ${this.horizontalRadius}i;
const RADIUS_Y: i32 = ${this.verticalRadius}i;
const TILE_WIDTH: u32 = ${e}u;
const TILE_PIXEL_COUNT: u32 = ${n}u;
const SOURCE_OFFSET: u32 = ${t(this.input.storage.values)}u;
const OUTPUT_OFFSET: u32 = ${t(this.output)}u;
const OUTPUT_VALIDITY_OFFSET: u32 = ${t(this.outputValidity)}u;
const KERNEL: array<f32, ${this.kernel.length}> = array<f32, ${this.kernel.length}>(${a});
@group(0) @binding(0) var<storage, read> sourceValues: array<${h(this.input.format)}>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputValidity: array<u32>;
${r}
var<workgroup> neighborhoodValues: array<f32, ${n}>;
var<workgroup> neighborhoodValidity: array<u32, ${n}>;

fn isFiniteValue(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823466e+38;
}

fn reflectCoordinate(value: i32, length: i32) -> i32 {
  if (length <= 1i) { return 0i; }
  let period = (length - 1i) * 2i;
  let reflected = ((value % period) + period) % period;
  return select(reflected, period - reflected, reflected >= length);
}

@compute @workgroup_size(8, 8)
fn main(
  @builtin(global_invocation_id) globalId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let localIndex = localId.y * 8u + localId.x;
  let workgroupOrigin = vec2<i32>(
    i32(workgroupId.x * 8u),
    i32(workgroupId.y * 8u)
  );

  for (
    var tileIndex = localIndex;
    tileIndex < TILE_PIXEL_COUNT;
    tileIndex += 64u
  ) {
    var sourceColumn = workgroupOrigin.x + i32(tileIndex % TILE_WIDTH) - RADIUS_X;
    var sourceRow = workgroupOrigin.y + i32(tileIndex / TILE_WIDTH) - RADIUS_Y;
    var sample = ${u(this.borderValue)};
    var validSample = false;
    var sampleSource = true;
    if (
      sourceColumn < 0i || sourceColumn >= i32(WIDTH) ||
      sourceRow < 0i || sourceRow >= i32(HEIGHT)
    ) {
      ${o}
    }
    if (sampleSource) {
      let sourceIndex = u32(sourceRow) * WIDTH + u32(sourceColumn);
      let rawSample = sourceValues[SOURCE_OFFSET + sourceIndex];
      sample = f32(rawSample) * ${u(this.input.scale??1)} + ${u(this.input.offset??0)};
      validSample = ${i.join(` && `)};
    }
    neighborhoodValues[tileIndex] = sample;
    neighborhoodValidity[tileIndex] = select(0u, 1u, validSample);
  }
  workgroupBarrier();

  if (globalId.x >= WIDTH || globalId.y >= HEIGHT) { return; }
  let pixelIndex = globalId.y * WIDTH + globalId.x;
  let centerIndex = (localId.y + u32(RADIUS_Y)) * TILE_WIDTH + localId.x + u32(RADIUS_X);
  var neighborhoodIsValid = neighborhoodValidity[centerIndex] != 0u;
  var weightedSum = 0.0;
  var participatingWeight = 0.0;
  for (var kernelRow = 0u; kernelRow < ${this.verticalRadius*2+1}u; kernelRow++) {
    for (var kernelColumn = 0u; kernelColumn < ${this.horizontalRadius*2+1}u; kernelColumn++) {
      let kernelIndex = kernelRow * ${this.horizontalRadius*2+1}u + kernelColumn;
      let coefficient = KERNEL[kernelIndex];
      if (coefficient != 0.0) {
        let tileIndex = (localId.y + kernelRow) * TILE_WIDTH + localId.x + kernelColumn;
        if (neighborhoodValidity[tileIndex] != 0u) {
          weightedSum += neighborhoodValues[tileIndex] * coefficient;
          participatingWeight += coefficient;
        } else {
          ${s}
        }
      }
    }
  }
  ${this.normalize||this.noDataPolicy===`ignore-renormalize`?`neighborhoodIsValid = neighborhoodIsValid && participatingWeight != 0.0;`:``}
  let result = ${c};
  neighborhoodIsValid = neighborhoodIsValid && isFiniteValue(result);
  let invalidValue = bitcast<f32>(0x7fc00000u | (pixelIndex & 0u));
  outputValues[OUTPUT_OFFSET + pixelIndex] = select(invalidValue, result, neighborhoodIsValid);
  outputValidity[OUTPUT_VALIDITY_OFFSET + pixelIndex] = select(0u, 1u, neighborhoodIsValid);
}`}};function k(e,t){switch(e){case`clamp`:return`sourceColumn = clamp(sourceColumn, 0i, i32(WIDTH) - 1i);
      sourceRow = clamp(sourceRow, 0i, i32(HEIGHT) - 1i);`;case`reflect`:return`sourceColumn = reflectCoordinate(sourceColumn, i32(WIDTH));
      sourceRow = reflectCoordinate(sourceRow, i32(HEIGHT));`;case`constant`:return`sample = ${u(t)};\n      validSample = true;\n      sampleSource = false;`;case`nodata`:return`sampleSource = false;`}}var A=x;function j(e,t){let{width:n,height:r,affine:i}=t;if(!Number.isSafeInteger(n)||!Number.isSafeInteger(r)||n<1||r<1||n*r>4294967295)throw Error(`${e} grid dimensions must be positive integers`);if(i.length!==6||!i.every(Number.isFinite))throw Error(`${e} grid affine must be six finite values`);return n*r}function M(e,t,n){if((n??`uniform`)!==`uniform`&&(t.affine[1]!==0||t.affine[3]!==0))throw Error(`${e} ${n} cell sizes need an axis-aligned grid`)}var N=1e-6;function P(e,t){let[n,r,,i,a]=t.affine,o=Math.hypot(n,i),s=Math.hypot(r,a),c=Math.fround(o),l=Math.fround(s);_(e,c>0&&Number.isFinite(c)&&l>0&&Number.isFinite(l),`grid.affine pixel-axis lengths must be positive and finite after float32 conversion`);let u=n/o*(r/s)+i/o*(a/s);if(!(Math.abs(u)<=1e-6))throw Error(`${e} grid.affine must have orthogonal pixel axes (normalized axis dot product within ${N} of zero): a sheared grid has two different diagonal step lengths; resample it to a rotated or axis-aligned grid`)}function F(e,t,r,i=`parameters`){if(n(t,[`float32`],`${e} ${i}`),t.length<r)throw Error(`${e} ${i} must contain at least ${r} values`)}function I(e,t,n){if(t.length<n)throw Error(`${e} target must hold ${n} values`)}function L(e,t,n){C(e,n.filter(e=>e!==void 0),t.filter(e=>e!==void 0))}function R(e){return[...e.storage.kind===`buffer`?[e.storage.values]:[],...e.validity?[e.validity]:[]]}function z(t,n,r){n.storage.kind===`texture`&&(_(t,(n.storage.view.texture.usage&e.SAMPLE)!==0,`input.storage.view.texture.usage must include Texture.SAMPLE`),n.scale!==void 0&&v(t,`input.scale`,n.scale),n.offset!==void 0&&v(t,`input.offset`,n.offset),new w({id:`${t} input`,metadata:r,bands:[n]}))}function B(e){let t=S(e,[`id`,`format`,`storage`,`validity`,`noDataValue`,`scale`,`offset`]),n=t.storage;return{...t,storage:n.kind===`buffer`?S(n,[`kind`,`values`]):S(n,[`kind`,`view`,`channel`])}}function V(e,t,n,r){b(e,t,R(n));let i=[...r,n.storage.kind===`texture`?n.storage.view:void 0];for(let n of i)if(n&&n.texture.graph!==t)throw Error(`${e} views must belong to the target graph`)}function H(e,t,n,r,i){if(n&&(!r.includes(n.format)||n.width!==i.width||n.height!==i.height))throw Error(`${e} ${t} must be ${r.join(` or `)} with the grid extent`)}function U(e,t,n,r,a){let{width:o,height:s}=r;if(n.storage.kind===`buffer`&&!a||n.storage.kind===`buffer`&&a===`reuse-canonical`&&n.format===`float32`&&n.validity&&n.scale===void 0&&n.offset===void 0&&n.noDataValue===void 0)return{band:n,nodes:[]};let c=o*s,l=i(e,`${t}-elevation-values`,`float32`,c),u=i(e,`${t}-elevation-validity`,`uint32`,c),d=n.storage.kind===`texture`?y(e,()=>new E({id:`${t}-elevation`,input:n,output:l,outputValidity:u,applyCalibration:!0}).addToGraph(e)):y(e,()=>new O({id:`${t}-elevation`,width:o,height:s,input:n,output:l,outputValidity:u,radius:0,kernel:[1],borderMode:`clamp`,noDataPolicy:`propagate`,normalize:!1}).addToGraph(e));return{band:{id:`${t}-elevation`,format:`float32`,storage:{kind:`buffer`,values:l},validity:u},nodes:d}}function W(e,t){if(!t)return[0,0];if(!(Number.isFinite(t[0])&&Number.isFinite(t[1])))throw Error(`${e} localOrigin must be finite`);return t}function G(e,t){let[n,r,i,a]=t;if(t.length!==4||!t.every(Number.isFinite)||!(i>n)||!(a>r))throw Error(`${e} extent must be finite with maxX > minX and maxY > minY`)}function K(e,t,n){if(!e.id)throw Error(`${n} band requires an identifier`);if(![`float32`,`uint32`,`sint32`].includes(e.format))throw Error(`${n} format must be float32, uint32 or sint32`);if(e.scale!==void 0&&!Number.isFinite(e.scale)||e.offset!==void 0&&!Number.isFinite(e.offset))throw Error(`${n} scale and offset must be finite`);if(e.storage.kind===`buffer`){if(e.storage.values.length<t.width*t.height)throw Error(`${n} values must cover the raster grid`)}else if(e.storage.kind===`texture`){let{view:r}=e.storage;if(r.width!==t.width||r.height!==t.height)throw Error(`${n} texture extent must match the raster grid`)}else throw Error(`${n} storage must be a buffer or texture`)}export{O as _,B as a,V as c,M as d,P as f,z as g,F as h,W as i,L as l,G as m,R as n,I as o,H as p,U as r,K as s,A as t,j as u,w as v};