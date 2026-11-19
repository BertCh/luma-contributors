import{a as e,c as t,d as n,i as r,o as i,r as a,t as o}from"./kernel-C3tfQOn4.js";import{t as s}from"./gpu-scan-B4NNTfF9.js";import{t as c}from"./gpu-uint64-scan-_k5PA9Xg.js";import{r as l,u}from"./gpu-command-graph-BdzdmV6u.js";import{t as d}from"./gpu-dispatch-utils-CbRzFAgT.js";import{r as f}from"./gpu-command-dispatch-metadata-Bpi5I1Ez.js";import{c as p,o as m}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as h}from"./raster-band-utils-BcTxc8_Y.js";import{S as g,u as _}from"./gpu-contributor-utils-BIwLQON5.js";var v=class{id;input;output;rows;columns;stats;constructor(e){this.id=e.id??`gpu-transpose`,this.input=e.input,this.output=e.output,this.rows=e.rows,this.columns=e.columns,this.stats=y(e.rows,e.columns);for(let[e,t]of[[`input`,this.input],[`output`,this.output]])for(let r of f(t))n(r,[`uint32`,`sint32`,`float32`],`${this.id} ${e}`);if(this.output.format!==this.input.format)throw Error(`${this.id} input and output formats must match`);if(this.input.length<this.stats.elementCount)throw Error(`${this.id} input must contain at least rows * columns rows`);if(this.output.length<this.stats.elementCount)throw Error(`${this.id} output must contain at least rows * columns rows`);let t=new Set(f(this.input).map(e=>e.buffer)),i=f(this.output);if(i.some(e=>t.has(e.buffer)))throw Error(`${this.id} input and output must use separate buffers`);for(let[e,t]of i.entries())if(i.slice(0,e).some(e=>r(e,t)))throw Error(`${this.id} output chunks must not overlap`)}getCommandNodes(t){let n=[];if(w(t,this.input,`${this.id} input`),w(t,this.output,`${this.id} output`),this.stats.elementCount===0)return n;T(t.device,this.id);let r=e(t,this.input,this.stats.elementCount),i=e(t,this.output,this.stats.elementCount),a=u(f(r)).filter(e=>e.length),o=u(f(i)).filter(e=>e.length);for(let[e,r]of a.entries())for(let[i,s]of o.entries()){let c=D(r.offset,r.length,s.offset,s.length,this.rows,this.columns);if(!c)continue;let l=S(c.tileCount,t.device.limits.maxComputeWorkgroupsPerDimension);n.push(...x(t,{...this,id:a.length===1&&o.length===1?this.id:`${this.id}-input-${e}-output-${i}`,input:r.data,output:s.data},l,c))}return n}};function y(e,t){C(`rows`,e),C(`columns`,t);let n=e*t;if(!Number.isSafeInteger(n)||n>4294967295)throw Error(`GPUTranspose rows * columns must fit in a uint32 index range`);let r=Math.ceil(e/16),i=Math.ceil(t/16);return Object.freeze({rows:e,columns:t,elementCount:n,tileRowCount:r,tileColumnCount:i,tileCount:r*i,workgroupSize:Object.freeze([16,16,1])})}function b(e,n,r={inputOffset:0,outputOffset:0,tileRowOffset:0,tileColumnOffset:0,tileColumnCount:e.stats.tileColumnCount,tileCount:e.stats.tileCount,elementCount:e.stats.elementCount}){let i=r.inputOffset===0&&e.input.length>=e.stats.elementCount,a=r.outputOffset===0&&e.output.length>=e.stats.elementCount;return`const ROWS: u32 = ${e.rows}u;
const COLUMNS: u32 = ${e.columns}u;
const TILE_COLUMN_COUNT: u32 = ${r.tileColumnCount}u;
const INPUT_START: u32 = ${r.inputOffset}u;
const INPUT_LENGTH: u32 = ${e.input.length}u;
const OUTPUT_START: u32 = ${r.outputOffset}u;
const OUTPUT_LENGTH: u32 = ${e.output.length}u;
const INPUT_OFFSET: u32 = ${t(e.input)}u;
const OUTPUT_OFFSET: u32 = ${t(e.output)}u;

// Move raw words so signed zeros, NaN payloads, and integer bits survive unchanged.
@group(0) @binding(0) var<storage, read> inputValues: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<u32>;
var<workgroup> tile: array<array<u32, 17>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_id) localIdentifier: vec3u
) {
  let tileIndex = (workgroupId.z * ${n.y}u + workgroupId.y) *
    ${n.x}u + workgroupId.x;
  if (tileIndex >= ${r.tileCount}u) { return; }
  let localTileRow = tileIndex / TILE_COLUMN_COUNT;
  let tileRow = localTileRow + ${r.tileRowOffset}u;
  let tileColumn = tileIndex - localTileRow * TILE_COLUMN_COUNT + ${r.tileColumnOffset}u;
  let inputRow = tileRow * 16u + localIdentifier.y;
  let inputColumn = tileColumn * 16u + localIdentifier.x;
  let inputIndex = inputRow * COLUMNS + inputColumn;
  let destinationIndex = inputColumn * ROWS + inputRow;
  if (inputRow < ROWS && inputColumn < COLUMNS
      ${i?``:`&& inputIndex >= INPUT_START && inputIndex - INPUT_START < INPUT_LENGTH`}
      ${a?``:`&& destinationIndex >= OUTPUT_START && destinationIndex - OUTPUT_START < OUTPUT_LENGTH`}) {
    tile[localIdentifier.y][localIdentifier.x] =
      inputValues[INPUT_OFFSET + inputIndex - INPUT_START];
  }
  workgroupBarrier();
  let outputRow = tileColumn * 16u + localIdentifier.y;
  let outputColumn = tileRow * 16u + localIdentifier.x;
  let outputIndex = outputRow * ROWS + outputColumn;
  let sourceIndex = outputColumn * COLUMNS + outputRow;
  if (outputRow < COLUMNS && outputColumn < ROWS
      ${a?``:`&& outputIndex >= OUTPUT_START && outputIndex - OUTPUT_START < OUTPUT_LENGTH`}
      ${i?``:`&& sourceIndex >= INPUT_START && sourceIndex - INPUT_START < INPUT_LENGTH`}) {
    outputValues[OUTPUT_OFFSET + outputIndex - OUTPUT_START] =
      tile[localIdentifier.x][localIdentifier.y];
  }
}`}function x(e,t,n,r){let a=[],s=b(t,n,r);return a.push(l({id:t.id,workload:{operation:`GPUTranspose`,commandCount:1,maximumWorkgroupCount:r.tileCount,maximumInvocationCount:r.tileCount*16*16,readByteLength:r.elementCount*Uint32Array.BYTES_PER_ELEMENT,writeByteLength:r.elementCount*Uint32Array.BYTES_PER_ELEMENT},resources:[{buffer:t.input,usage:`storage-read`},{buffer:t.output,usage:`storage-write`}],compile:({device:e})=>{let r=new o(e,{id:t.id,source:s,shaderLayout:{bindings:[{name:`inputValues`,type:`read-only-storage`,group:0,location:0},{name:`outputValues`,type:`storage`,group:0,location:1}]}});return{encode:({computePass:e,getBuffer:a})=>{let o={inputValues:i(t.input,a),outputValues:i(t.output,a)};r.dispatch(e,{bindings:o,x:n.x,y:n.y,z:n.z})},destroy:()=>r.destroy()}}})),a}function S(e,t){return d(`GPUTranspose`,e*16*16,256,t)}function C(e,t){if(!Number.isSafeInteger(t)||t<0)throw Error(`GPUTranspose ${e} must be a non-negative safe integer`)}function w(e,t,n){if(f(t).some(t=>t.buffer.graph!==e))throw Error(`${n} belongs to a different GPUCommandGraph`)}function T(e,t){if(e.limits.maxComputeInvocationsPerWorkgroup<256||e.limits.maxComputeWorkgroupSizeX<16||e.limits.maxComputeWorkgroupSizeY<16)throw Error(`${t} requires 16 by 16 compute workgroups`)}function E(e,t,n){let r=Math.floor(e/n),i=Math.floor((e+t-1)/n),a=e%n,o=(e+t-1)%n+1;if(r===i)return[{rowStart:r,rowEnd:r+1,columnStart:a,columnEnd:o}];let s=[{rowStart:r,rowEnd:r+1,columnStart:a,columnEnd:n},{rowStart:i,rowEnd:i+1,columnStart:0,columnEnd:o}];return r+1<i&&s.push({rowStart:r+1,rowEnd:i,columnStart:0,columnEnd:n}),s}function D(e,t,n,r,i,a){let o=i,s=0,c=a,l=0,u=0;for(let d of E(e,t,a))for(let e of E(n,r,i)){let t=Math.max(d.rowStart,e.columnStart),n=Math.min(d.rowEnd,e.columnEnd),r=Math.max(d.columnStart,e.rowStart),i=Math.min(d.columnEnd,e.rowEnd);t>=n||r>=i||(u+=(n-t)*(i-r),o=Math.min(o,t),s=Math.max(s,n),c=Math.min(c,r),l=Math.max(l,i))}if(!u)return;let d=Math.floor(o/16),f=Math.floor(c/16),p=Math.ceil(l/16)-f;return{inputOffset:e,outputOffset:n,tileRowOffset:d,tileColumnOffset:f,tileColumnCount:p,tileCount:(Math.ceil(s/16)-d)*p,elementCount:u}}function O(e){return _(e,12)}var k=`
${h}
const MAXIMUM_QUANTIZED: f32 = ${p(2147483520)};
fn add64(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  let low = left.x + right.x;
  return vec2<u32>(low, left.y + right.y + select(0u, 1u, low < left.x));
}
fn subtract64(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  return vec2<u32>(left.x - right.x, left.y - right.y - select(0u, 1u, left.x < right.x));
}
fn multiplyWide32(left: u32, right: u32) -> vec2<u32> {
  let left0 = left & 0xffffu;
  let left1 = left >> 16u;
  let right0 = right & 0xffffu;
  let right1 = right >> 16u;
  let product00 = left0 * right0;
  let product01 = left0 * right1;
  let product10 = left1 * right0;
  let product11 = left1 * right1;
  let middle = (product00 >> 16u) + (product01 & 0xffffu) + (product10 & 0xffffu);
  return vec2<u32>(
    (product00 & 0xffffu) | (middle << 16u),
    product11 + (product01 >> 16u) + (product10 >> 16u) + (middle >> 16u)
  );
}
fn multiply64By32(left: vec2<u32>, right: u32) -> vec2<u32> {
  let wide = multiplyWide32(left.x, right);
  return vec2<u32>(wide.x, wide.y + left.y * right);
}
fn negate64(value: vec2<u32>) -> vec2<u32> {
  return subtract64(vec2<u32>(0u, 0u), value);
}
fn unsigned64ToFloat(value: vec2<u32>) -> f32 {
  return f32(value.y) * 4294967296.0 + f32(value.x);
}
// Converts the magnitude, not the two's-complement words: f32(hi) * 2^32 + f32(lo) of a small
// negative value cancels -2^32 against a low word rounded up to 2^32.
fn signed64ToFloat(value: vec2<u32>) -> f32 {
  if (bitcast<i32>(value.y) < 0i) {
    return -unsigned64ToFloat(negate64(value));
  }
  return unsigned64ToFloat(value);
}
fn isQuantizable(elevation: f32) -> bool {
  return isFiniteValue(elevation) && abs(elevation * INVERSE_QUANTUM) <= MAXIMUM_QUANTIZED;
}
fn quantizeElevation(elevation: f32) -> i32 {
  return i32(round(elevation * INVERSE_QUANTUM));
}`;function A(e,t){let{id:n,width:r,height:i}=t,o=t.includeSquares??!0,l=o?3:2,u=o?2:1,d=l-1,f=r*i;g(n,`summed-area table cells`,f,O(e.device));let _=t=>({low:a(e,`${n}-summed-area-${t}-low`,`uint32`,l*f),high:a(e,`${n}-summed-area-${t}-high`,`uint32`,u*f)}),y=_(`first`),b=_(`second`),x=(t,n)=>e.createDataView(t.buffer,{format:`uint32`,length:f,byteOffset:t.byteOffset+n*f*4}),S=t.elevation.validity;if(!S)throw Error(`${n} summed-area table requires a canonical elevation band`);let C=t.originRelative?a(e,`${n}-anchor-index`,`uint32`,1):void 0,w=t.originRelative?a(e,`${n}-anchor`,`float32`,1):void 0,T=[];C&&w&&T.push(m(e,{id:`${n}-anchor-reset`,operation:`GPUTerrainTopographicPosition`,bindings:[{name:`anchorIndex`,view:C,type:`u32`,access:`read_write`}],invocationCount:1,body:`anchorIndex[anchorIndexOffset] = 0xffffffffu;`}),m(e,{id:`${n}-anchor-select`,operation:`GPUTerrainTopographicPosition`,bindings:[{name:`values`,view:t.elevation.storage.values,type:`f32`,access:`read`},{name:`validity`,view:S,type:`u32`,access:`read`},{name:`anchorIndex`,view:C,type:`atomic<u32>`,access:`read_write`}],invocationCount:f,declarations:h,body:`if (validity[validityOffset + index] != 0u && isFiniteValue(values[valuesOffset + index])) {
          atomicMin(&anchorIndex[anchorIndexOffset], index);
        }`}),m(e,{id:`${n}-anchor-publish`,operation:`GPUTerrainTopographicPosition`,bindings:[{name:`values`,view:t.elevation.storage.values,type:`f32`,access:`read`},{name:`anchorIndex`,view:C,type:`u32`,access:`read`},{name:`anchor`,view:w,type:`f32`,access:`read_write`}],invocationCount:1,body:`let first = anchorIndex[anchorIndexOffset];
          var origin = 0.0;
          if (first != 0xffffffffu) { origin = values[valuesOffset + first]; }
          // Snap only small anchors; large f32 values already lie on the quantum lattice.
          if (abs(origin) < ${p(t.quantum*2**23)}) {
            origin = round(origin * ${p(1/t.quantum)}) * ${p(t.quantum)};
          }
          anchor[anchorOffset] = origin;`})),T.push(m(e,{id:`${n}-summed-area-quantize`,operation:`GPUTerrainTopographicPosition`,variant:`quantize`,bindings:[{name:`elevationValues`,view:t.elevation.storage.values,type:`f32`,access:`read`},...w?[{name:`anchor`,view:w,type:`f32`,access:`read`}]:[],{name:`elevationValidity`,view:S,type:`u32`,access:`read`},{name:`lowPlanes`,view:y.low,type:`u32`,access:`read_write`},{name:`highPlanes`,view:y.high,type:`u32`,access:`read_write`}],invocationCount:f,declarations:`const PIXEL_COUNT: u32 = ${f}u;
const INVERSE_QUANTUM: f32 = ${p(1/t.quantum)};
${k}`,body:`let elevation = elevationValues[elevationValuesOffset + index]${w?` - anchor[anchorOffset]`:``};
  let isValid = elevationValidity[elevationValidityOffset + index] != 0u && isQuantizable(elevation);
  var quantized = 0i;
  if (isValid) {
    quantized = quantizeElevation(elevation);
  }
  lowPlanes[lowPlanesOffset + index] = bitcast<u32>(quantized);
  lowPlanes[lowPlanesOffset + ${d}u * PIXEL_COUNT + index] = select(0u, 1u, isValid);
  highPlanes[highPlanesOffset + index] = select(0u, 0xffffffffu, quantized < 0i);${o?`
  let magnitude = u32(abs(quantized));
  let square = multiplyWide32(magnitude, magnitude);
  lowPlanes[lowPlanesOffset + PIXEL_COUNT + index] = square.x;
  highPlanes[highPlanesOffset + PIXEL_COUNT + index] = square.y;`:``}`}));let E=(t,r,i)=>{for(let[a,s]of o?[[`sum`,0],[`square`,1]]:[[`sum`,0]])T.push(...new c({id:`${n}-summed-area-${t}-${a}`,inputLow:x(r.low,s),inputHigh:x(r.high,s),outputLow:x(i.low,s),outputHigh:x(i.high,s)}).getCommandNodes(e));T.push(...new s({id:`${n}-summed-area-${t}-count`,input:x(r.low,d),output:x(i.low,d),mode:`inclusive`}).getCommandNodes(e))};E(`rows`,y,b);for(let[t,a]of[[`low`,l],[`high`,u]])for(let o=0;o<a;o++)T.push(...new v({id:`${n}-summed-area-transpose-${t}-${o}`,input:x(b[t],o),output:x(y[t],o),rows:i,columns:r}).getCommandNodes(e));return E(`columns`,y,b),{lowTable:b.low,highTable:b.high,anchor:w,nodes:T}}var j=`
struct TerrainBoxSums {
  sum: vec2<u32>,
  square: vec2<u32>,
  count: u32,
}
fn readTablePlane64(plane: u32, linearIndex: i32) -> vec2<u32> {
  if (linearIndex < 0i) {
    return vec2<u32>(0u, 0u);
  }
  let offset = plane * PIXEL_COUNT + u32(linearIndex);
  return vec2<u32>(lowTable[lowTableOffset + offset], highTable[highTableOffset + offset]);
}
fn readTableCount(linearIndex: i32) -> u32 {
  if (linearIndex < 0i) {
    return 0u;
  }
  return lowTable[lowTableOffset + 2u * PIXEL_COUNT + u32(linearIndex)];
}
fn readColumnRange(column: i32, firstRow: i32, lastRow: i32) -> TerrainBoxSums {
  var upper: i32;
  var lower: i32;
  if (column >= 0i) {
    upper = column * i32(HEIGHT) + lastRow;
    lower = column * i32(HEIGHT) + firstRow - 1i;
  } else {
    let lastColumnStart = (i32(WIDTH) - 1i) * i32(HEIGHT);
    upper = lastColumnStart + lastRow - 1i;
    lower = lastColumnStart + max(firstRow, 1i) - 2i;
  }
  return TerrainBoxSums(
    subtract64(readTablePlane64(0u, upper), readTablePlane64(0u, lower)),
    subtract64(readTablePlane64(1u, upper), readTablePlane64(1u, lower)),
    readTableCount(upper) - readTableCount(lower)
  );
}
fn readBoxSums(firstColumn: i32, lastColumn: i32, firstRow: i32, lastRow: i32) -> TerrainBoxSums {
  let right = readColumnRange(lastColumn, firstRow, lastRow);
  let left = readColumnRange(firstColumn - 1i, firstRow, lastRow);
  return TerrainBoxSums(
    subtract64(right.sum, left.sum),
    subtract64(right.square, left.square),
    right.count - left.count
  );
}`,M=`
struct TerrainBoxSums {
  sum: vec2<u32>,
  count: u32,
}
fn readTableSum64(linearIndex: i32) -> vec2<u32> {
  if (linearIndex < 0i) {
    return vec2<u32>(0u, 0u);
  }
  return vec2<u32>(lowTable[lowTableOffset + u32(linearIndex)], highTable[highTableOffset + u32(linearIndex)]);
}
fn readTableCount(linearIndex: i32) -> u32 {
  if (linearIndex < 0i) {
    return 0u;
  }
  return lowTable[lowTableOffset + PIXEL_COUNT + u32(linearIndex)];
}
fn readColumnRange(column: i32, firstRow: i32, lastRow: i32) -> TerrainBoxSums {
  var upper: i32;
  var lower: i32;
  if (column >= 0i) {
    upper = column * i32(HEIGHT) + lastRow;
    lower = column * i32(HEIGHT) + firstRow - 1i;
  } else {
    let lastColumnStart = (i32(WIDTH) - 1i) * i32(HEIGHT);
    upper = lastColumnStart + lastRow - 1i;
    lower = lastColumnStart + max(firstRow, 1i) - 2i;
  }
  return TerrainBoxSums(
    subtract64(readTableSum64(upper), readTableSum64(lower)),
    readTableCount(upper) - readTableCount(lower)
  );
}
fn readBoxSums(firstColumn: i32, lastColumn: i32, firstRow: i32, lastRow: i32) -> TerrainBoxSums {
  let right = readColumnRange(lastColumn, firstRow, lastRow);
  let left = readColumnRange(firstColumn - 1i, firstRow, lastRow);
  return TerrainBoxSums(subtract64(right.sum, left.sum), right.count - left.count);
}`;export{O as a,A as i,M as n,k as r,j as t};