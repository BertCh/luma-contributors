import{c as e,o as t,r as n,t as r}from"./kernel-C3tfQOn4.js";import{l as i,r as a}from"./gpu-command-graph-BdzdmV6u.js";import{t as o}from"./gpu-reduction-tNgOVQ_C.js";import{O as s,T as c,c as l,f as u,v as d}from"./gpu-contributor-utils-BIwLQON5.js";import{Q as f}from"./dist-BusamGlx.js";import{t as p}from"./gpu-dispatch-utils-CbRzFAgT.js";import{a as m,i as h}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as g}from"./gpu-grid-coordinate-DwQPYH7E.js";import{c as _,o as v}from"./wgsl-kernel-nodes-DagJklH2.js";import{c as y}from"./wgsl-common-DfyXiYB0.js";import{a as b,n as x,o as S,r as C,t as w}from"./fixed-point-accumulator-Ce6W3HVY.js";import{t as T}from"./point-density-hexagon-CYSFRWN5.js";2*Math.sin(Math.PI/3);var E=`
fn getPointDensityD3HexagonCell(x: f32, y: f32, originX: f32, originY: f32, radius: f32) -> vec2<i32> {
  let normalizedY = (y - originY) / radius / 1.5;
  var row = floor(normalizedY + 0.5);
  let parity = f32(i32(row) & 1);
  let normalizedX = (x - originX) / radius / 1.7320508075688772 - parity * 0.5;
  var column = floor(normalizedX + 0.5);
  let deltaY = normalizedY - row;
  if (abs(deltaY) * 3.0 > 1.0) {
    let deltaX = normalizedX - column;
    let alternativeColumn = column + select(0.5, -0.5, normalizedX < column);
    let alternativeRow = row + select(1.0, -1.0, normalizedY < row);
    let alternativeDeltaX = normalizedX - alternativeColumn;
    let alternativeDeltaY = normalizedY - alternativeRow;
    if (deltaX * deltaX + deltaY * deltaY > alternativeDeltaX * alternativeDeltaX + alternativeDeltaY * alternativeDeltaY) {
      column = alternativeColumn + select(-0.5, 0.5, parity != 0.0);
      row = alternativeRow;
    }
  }
  return vec2<i32>(i32(column), i32(row));
}
`,D=16,O=4294967295,k=2147483648,A=4294967295;function j(e,t){let n=Object.values(t).flatMap(e=>e?[e.buffer.graph.createDataView(e.buffer,{format:`uint32`,length:e.length*e.rowByteLength/4,byteOffset:e.byteOffset})]:[]),r=n.reduce((e,t)=>e+t.length,0);h(e,{"output fields":new i({id:`${e}-output-fields`,name:`output fields`,format:`uint32`,length:r,valueLength:r,stride:1,byteStride:4,rowByteLength:4,data:n})})}function M(e,t,r,i,a={}){let o=e=>e.buffer.defaultBuffer?s(e.buffer.defaultBuffer):e.buffer,c=Object.entries(i),l=c.flatMap(([e,t])=>t&&c.some(([n,r])=>e!==n&&r&&o(t)===o(r))?[{name:e,view:t,length:a[e]??t.length}]:[]);l.length>0&&F(e,t,[]),N(e,t,l.map(({name:e,length:t})=>[`output.${e} staging scratch`,t*4]));let u={...i},d=[];for(let{name:i,view:a,length:o}of l){let s=n(e,`${t}-${i}-scratch`,a.format,o);u[i]=s,d.push(I(e,{id:`${t}-publish-${i}`,operation:r,variant:`publish-output`,invocationCount:o,bindings:[{name:`sourceWords`,view:s,type:`u32`,access:`read`},{name:`outputWords`,view:a,type:`u32`,access:`read_write`}],body:`outputWords[outputWordsOffset + index] = sourceWords[sourceWordsOffset + index];`}))}return{views:u,publishNodes:d}}function N(e,t,n){for(let[r,i]of n)d(e.device,t,r,i,`reduce ${r}`),u(t,`${r} byteLength`,i,0,e.device.limits.maxBufferSize)}function P(e,t,n,r){N(e,t,[[`index row and sort scratch`,Math.max(n,1)*4],[`index offset and scan scratch`,(r+1)*4]])}function F(e,t,n,r={}){r.dispatchMetadata!==!1&&(c(t,D*Uint32Array.BYTES_PER_ELEMENT<=e.device.limits.maxUniformBufferBindingSize,`density dispatch metadata must fit maxUniformBufferBindingSize`),u(t,`density dispatch metadata byteLength`,D*Uint32Array.BYTES_PER_ELEMENT,0,e.device.limits.maxBufferSize));for(let[i,a]of n){let n=(a?l(a):[]).filter(e=>!r.skipEmptyChunks||e.length>0);m(e,t,Object.fromEntries(n.map((e,t)=>[`${i} chunk ${t}`,e])));for(let r of n)u(t,`${i} buffer byteLength`,r.buffer.byteLength,0,e.device.limits.maxBufferSize)}}function I(n,i){let o=i.workgroupSize??64,s=Object.entries(i.runtimeConstants??{});c(i.id,i.bindings.length+s.length+1<=D,`density dispatch metadata must fit DENSITY_DISPATCH_WORD_COUNT`),c(i.id,i.bindings.length<=n.device.limits.maxStorageBuffersPerShaderStage,`density storage bindings must fit maxStorageBuffersPerShaderStage`),F(n,i.id,i.bindings.map(e=>[e.name,e.view]));let l=new Uint32Array(D);l[0]=i.invocationCount;let u=new Map([[`INVOCATION_COUNT`,`densityDispatch.words[0].x`]]),d=e=>`densityDispatch.words[${Math.floor(e/4)}][${e%4}]`;for(let[t,n]of i.bindings.entries())l[t+1]=e(n.view),u.set(`${n.name}Offset`,d(t+1));for(let[e,[t,n]]of s.entries()){let r=i.bindings.length+1+e;l[r]=n,u.set(t,d(r))}let m=e=>e.replace(/\b[A-Za-z_][A-Za-z_0-9]*\b/g,e=>u.get(e)??e),h=`struct DensityDispatch { words: array<vec4u, 4> };
@group(0) @binding(${i.bindings.length}) var<uniform> densityDispatch: DensityDispatch;
${i.bindings.map((e,t)=>`@group(0) @binding(${t}) var<storage, ${e.access}> ${e.name}: array<${e.type}>;`).join(`
`)}
${m(i.declarations??``)}
@compute @workgroup_size(${o})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
        @builtin(num_workgroups) workgroupCount: vec3u,
        @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${i.bindings.map(e=>`_ = &${e.name};`).join(`
  `)}
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex > ${Math.floor(O/o)}u) { return; }
  let index = workgroupIndex * ${o}u + localInvocationIndex;
  ${i.guardIndex===!1?``:`if (index >= densityDispatch.words[0].x) { return; }`}
  ${m(i.body)}
}`,g=p(i.id,Math.max(1,i.invocationCount),o,n.device.limits.maxComputeWorkgroupsPerDimension);return a({id:i.id,condition:i.condition,resources:[...i.bindings.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`})),...i.extraResources??[]],workload:{operation:i.operation,...i.variant?{variant:i.variant}:{},commandCount:1,maximumInvocationCount:i.invocationCount,maximumWorkgroupCount:g.x*g.y*g.z},compile:({device:e})=>{let n=e.createBuffer({id:`${i.id}-dispatch`,data:l,usage:f.UNIFORM|f.COPY_DST}),a=new r(e,{id:i.id,source:h,constants:i.constants,shaderLayout:{bindings:[...i.bindings.map((e,t)=>({name:e.name,group:0,location:t,type:e.access===`read`?`read-only-storage`:`storage`})),{name:`densityDispatch`,group:0,location:i.bindings.length,type:`uniform`}]}});return{encode:({computePass:e,getBuffer:r})=>{let o={densityDispatch:{buffer:n,offset:0,size:l.byteLength}};for(let e of i.bindings)o[e.name]=t(e.view,r);a.dispatch(e,{bindings:o,...g})},destroy:()=>{a.destroy(),n.destroy()}}}})}var L=`GPUPointDensity`;function R(e,t,n){return v(e,{id:t,operation:L,variant:`clear-overflow`,bindings:[{name:`overflowFlags`,view:n,type:`u32`,access:`read_write`}],invocationCount:1,body:`overflowFlags[overflowFlagsOffset] = 0u;`})}function z(e,t){let n=[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`keys`,view:t.keys,type:`u32`,access:`read_write`}];t.mask&&n.push({name:`rowMask`,view:t.mask,type:`u32`,access:`read`});let r=!Array.isArray(t.bounds);r&&n.push({name:`boundsValues`,view:t.bounds,type:`f32`,access:`read`}),typeof t.radius!=`number`&&n.push({name:`radiusValues`,view:t.radius,type:`f32`,access:`read`}),t.overflow&&n.push({name:`overflowFlags`,view:t.overflow,type:`atomic<u32>`,access:`read_write`});let i=r?void 0:t.bounds.map(_),a=i?`let minimumX = ${i[0]}; let minimumY = ${i[1]};
  let maximumX = ${i[2]}; let maximumY = ${i[3]};`:`let minimumX = boundsValues[boundsValuesOffset];
  let minimumY = boundsValues[boundsValuesOffset + 1u];
  let maximumX = boundsValues[boundsValuesOffset + 2u];
  let maximumY = boundsValues[boundsValuesOffset + 3u];`,o=typeof t.radius==`number`?`let radius = ${_(t.radius)};`:`let radius = radiusValues[radiusValuesOffset];`;return I(e,{id:t.id,operation:L,variant:`hexagon-keys`,bindings:n,invocationCount:t.positions.length,runtimeConstants:{KEY_START:t.keyStart,MASK_START:t.maskStart??0},declarations:`const COLUMNS: u32 = ${t.gridSize[0]}u;
const ROWS: u32 = ${t.gridSize[1]}u;
const INVALID_KEY: u32 = 0xffffffffu;
const MAXIMUM_FLOAT: f32 = 3.402823466e+38;
${t.assignment===`d3`?E:T}`,body:`${a}
  ${o}
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  var key = INVALID_KEY;
  let included = ${t.mask?`rowMask[rowMaskOffset + MASK_START + index] != 0u`:`true`};
  let finite = included && abs(x) <= MAXIMUM_FLOAT && abs(y) <= MAXIMUM_FLOAT;
  let finiteBounds = abs(minimumX) <= MAXIMUM_FLOAT && abs(minimumY) <= MAXIMUM_FLOAT &&
    abs(maximumX) <= MAXIMUM_FLOAT && abs(maximumY) <= MAXIMUM_FLOAT;
  let orderedBounds = minimumX <= maximumX && minimumY <= maximumY;
  let inside = x >= minimumX && x <= maximumX && y >= minimumY && y <= maximumY;
  let validRadius = radius > 0.0 && radius <= MAXIMUM_FLOAT;
  if (finite && finiteBounds && orderedBounds && inside && validRadius) {
    let cell = ${t.assignment===`d3`?`getPointDensityD3HexagonCell`:`getPointDensityHexagonCell`}(x, y, minimumX, minimumY, radius);
    if (cell.x >= 0 && cell.y >= 0 && u32(cell.x) < COLUMNS && u32(cell.y) < ROWS) {
      key = u32(cell.y) * COLUMNS + u32(cell.x);
    } else {
      ${t.overflow?`atomicStore(&overflowFlags[overflowFlagsOffset], 1u);`:``}
    }
  }
  keys[keysOffset + KEY_START + index] = key;`})}function B(e,t){return I(e,{id:t.id,operation:L,variant:`mask-positions`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`rowMask`,view:t.mask,type:`u32`,access:`read`},{name:`maskedPositions`,view:t.maskedPositions,type:`f32`,access:`read_write`}],invocationCount:t.positions.length,runtimeConstants:{ROW_START:t.rowStart},body:`var nanBits = 0x7fc00000u;
  let nan = bitcast<f32>(nanBits);
  let included = rowMask[rowMaskOffset + ROW_START + index] != 0u;
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let slot = maskedPositionsOffset + (ROW_START + index) * 2u;
  maskedPositions[slot] = select(nan, x, included);
  maskedPositions[slot + 1u] = select(nan, y, included);`})}function V(e,t){let n=[{name:`counts`,view:t.counts,type:`u32`,access:`read`}];t.sums&&n.push({name:`sums`,view:t.sums,type:`f32`,access:`read`}),n.push({name:`values`,view:t.values,type:`f32`,access:`read_write`}),t.means&&n.push({name:`means`,view:t.means,type:`f32`,access:`read_write`});let r=t.statistic===`count`?`f32(count)`:t.statistic===`sum`?`sum`:`mean`;return v(e,{id:t.id,operation:L,variant:`finalize`,bindings:n,invocationCount:t.values.length,body:`let count = counts[countsOffset + index];
  let sum = ${t.sums?`sums[sumsOffset + index]`:`f32(count)`};
  let mean = select(0.0, sum / f32(max(count, 1u)), count > 0u);
  values[valuesOffset + index] = ${r};
  ${t.means?`means[meansOffset + index] = mean;`:``}`})}function H(e,t){let n=[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`keys`,view:t.keys,type:`u32`,access:`read_write`}],r=!Array.isArray(t.bounds);r&&n.push({name:`boundsValues`,view:t.bounds,type:`f32`,access:`read`});let i=r?void 0:t.bounds.map(_),a=i?`let minimumX = ${i[0]}; let minimumY = ${i[1]};
  let maximumX = ${i[2]}; let maximumY = ${i[3]};`:`let minimumX = boundsValues[boundsValuesOffset];
  let minimumY = boundsValues[boundsValuesOffset + 1u];
  let maximumX = boundsValues[boundsValuesOffset + 2u];
  let maximumY = boundsValues[boundsValuesOffset + 3u];`;return I(e,{id:t.id,operation:L,variant:`grid-keys`,bindings:n,invocationCount:t.positions.length,runtimeConstants:{KEY_START:t.keyStart},declarations:`const COLUMNS: u32 = ${t.gridSize[0]}u;
const ROWS: u32 = ${t.gridSize[1]}u;
${y}
${g}`,body:`${a}
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  var key = 0xffffffffu;
  let finite = isFiniteBits(x) && isFiniteBits(y);
  let finiteBounds = isFiniteBits(minimumX) && isFiniteBits(minimumY) &&
    isFiniteBits(maximumX) && isFiniteBits(maximumY);
  let inX = x >= minimumX && x <= maximumX && (maximumX != minimumX || x == minimumX);
  let inY = y >= minimumY && y <= maximumY && (maximumY != minimumY || y == minimumY);
  if (finiteBounds && maximumX >= minimumX && maximumY >= minimumY && finite && inX && inY) {
    key = getCoordinate(y, minimumY, maximumY, ROWS) * COLUMNS + getCoordinate(x, minimumX, maximumX, COLUMNS);
  }
  keys[keysOffset + KEY_START + index] = key;`})}function U(e,t){return I(e,{id:t.id,operation:L,variant:`workgroup-sums`,bindings:[{name:`weights`,view:t.weights,type:`f32`,access:`read`},{name:`keys`,view:t.keys,type:`u32`,access:`read`},{name:`scale`,view:t.scale,type:`u32`,access:`read`},{name:`sums`,view:t.fixedPointSums,type:`atomic<u32>`,access:`read_write`}],workgroupSize:256,invocationCount:t.weights.length,guardIndex:!1,runtimeConstants:{KEY_START:t.keyStart},declarations:`const CELL_COUNT: u32 = ${t.fixedPointSums.length/3}u;
override HEADROOM_BITS: u32;
${y}
${w}
${b(`sums`)}
var<workgroup> sortKeys: array<u32, 256>;
var<workgroup> sortValues: array<vec3<u32>, 256>;`,constants:{HEADROOM_BITS:S(t.rowCount)},body:`let lane = localInvocationIndex;
  let unitExponent = getFixedPointUnitExponent(scale[scaleOffset], HEADROOM_BITS);
  var key = 0xffffffffu;
  var value = vec3<u32>(0u);
  if (index < INVOCATION_COUNT) {
    let weight = weights[weightsOffset + index];
    let rowKey = keys[keysOffset + KEY_START + index];
    if (isFiniteBits(weight) && rowKey < CELL_COUNT) {
      key = rowKey;
      value = quantizeFixedPoint(weight, unitExponent);
    }
  }
  sortKeys[lane] = key;
  sortValues[lane] = value;
  workgroupBarrier();
  // Every row sums the rows of its workgroup that share its key (exact integer adds). Only the
  // first row of each key issues the global add, so no sort or scan is needed.
  var runSum = vec3<u32>(0u);
  var isFirst = true;
  for (var other = 0u; other < 256u; other = other + 1u) {
    if (sortKeys[other] == key) {
      runSum = addFixedPoint(runSum, sortValues[other]);
      isFirst = isFirst && other >= lane;
    }
  }
  if (isFirst && key < CELL_COUNT) {
    addFixedPointToSums(key, runSum);
  }`})}function W(e,t){let n=t.operation??L;return[v(e,{id:`${t.id}-clear`,operation:n,variant:`fixed-point-scale-clear`,invocationCount:1,bindings:[{name:`fixedPointScale`,view:t.scale,type:`u32`,access:`read_write`}],body:`fixedPointScale[fixedPointScaleOffset] = 0u;`}),...l(t.values).flatMap((r,i)=>r.length===0?[]:[I(e,{id:`${t.id}-${i}`,operation:n,variant:`fixed-point-scale`,invocationCount:r.length,workgroupSize:256,guardIndex:!1,bindings:[{name:`fixedPointValues`,view:r,type:`f32`,access:`read`},{name:`fixedPointScale`,view:t.scale,type:`atomic<u32>`,access:`read_write`}],declarations:`${w}\nvar<workgroup> workgroupExponentBits: atomic<u32>;`,body:`if (localInvocationIndex == 0u) { atomicStore(&workgroupExponentBits, 0u); }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let exponentBits = getFixedPointExponentBits(fixedPointValues[fixedPointValuesOffset + index]);
    if (exponentBits != 0u) { atomicMax(&workgroupExponentBits, exponentBits); }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    let workgroupMaximum = atomicLoad(&workgroupExponentBits);
    if (workgroupMaximum != 0u) { atomicMax(&fixedPointScale[fixedPointScaleOffset], workgroupMaximum); }
  }`})])]}function G(e,t,r,a,o=!1){if(o||!(r instanceof i))return n(e,t,a,r.length);let s=r.data.map((r,i)=>n(e,`${t}-${i}`,a,r.length));return new i({id:t,name:t,format:a,length:r.length,valueLength:r.length,stride:1,byteStride:4,rowByteLength:4,data:s})}function K(e,t,n,r){let i=l(n);for(let n of i){let i=Math.max(1,n.length)*r;d(e.device,t,`point scratch chunk`,i,`reduce the point chunk length`),u(t,`point scratch chunk byteLength`,i,0,e.device.limits.maxBufferSize)}}function q(e,t,n){let r=l(t).filter(e=>e.length>0),i=l(n).filter(e=>e.length>0),a=[],o=0,s=0;for(let t of r){let n=0;for(;n<t.length;){let r=i[o],c=Math.min(t.length-n,r.length-s);a.push({weights:e.createDataView(t.buffer,{format:`float32`,length:c,byteOffset:t.byteOffset+n*4}),keys:e.createDataView(r.buffer,{format:`uint32`,length:c,byteOffset:r.byteOffset+s*4})}),n+=c,s+=c,s===r.length&&(o++,s=0)}}return a}function J(e,t,r,a){if(!(r instanceof i))return[...new o({id:t,input:r,output:a,operation:`sum`}).getCommandNodes(e)];let s=n(e,`${t}-scale`,`uint32`,1),c=n(e,`${t}-words`,`uint32`,3),u=n(e,`${t}-nonfinite`,`uint32`,1),d=[...W(e,{id:`${t}-scale`,operation:`GPUKernelDensity`,values:r,scale:s}),x(e,{id:`${t}-clear`,operation:`GPUKernelDensity`,words:c}),I(e,{id:`${t}-nonfinite-clear`,operation:`GPUKernelDensity`,invocationCount:1,bindings:[{name:`nonfinite`,view:u,type:`u32`,access:`read_write`}],body:`nonfinite[nonfiniteOffset] = 0u;`})];for(let[n,i]of l(r).entries())i.length!==0&&d.push(I(e,{id:`${t}-accumulate-${n}`,operation:`GPUKernelDensity`,invocationCount:i.length,guardIndex:!1,workgroupSize:256,bindings:[{name:`terms`,view:i,type:`f32`,access:`read`},{name:`scale`,view:s,type:`u32`,access:`read`},{name:`sums`,view:c,type:`atomic<u32>`,access:`read_write`},{name:`nonfinite`,view:u,type:`atomic<u32>`,access:`read_write`}],constants:{HEADROOM_BITS:S(r.length)},declarations:`override HEADROOM_BITS: u32;
${w}
${b(`sums`)}
var<workgroup> termSums: array<vec3u, 256>;
var<workgroup> nonfiniteFlags: atomic<u32>;`,body:`let lane = localInvocationIndex;
  if (lane == 0u) { atomicStore(&nonfiniteFlags, 0u); }
  workgroupBarrier();
  var contribution = vec3u(0u);
  if (index < INVOCATION_COUNT) {
    let value = terms[termsOffset + index];
    let bits = bitcast<u32>(value);
    if ((bits & 0x7fffffffu) < 0x7f800000u) {
      let unitExponent = getFixedPointUnitExponent(scale[scaleOffset], HEADROOM_BITS);
      contribution = quantizeFixedPoint(value, unitExponent);
    } else {
      let flags = select(select(2u, 4u, (bits >> 31u) != 0u), 1u, (bits & 0x7fffffu) != 0u);
      atomicOr(&nonfiniteFlags, flags);
    }
  }
  termSums[lane] = contribution;
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (lane < stride) { termSums[lane] = addFixedPoint(termSums[lane], termSums[lane + stride]); }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride /= 2u;
  }
  if (lane == 0u) {
    addFixedPointToSums(0u, termSums[0]);
    atomicOr(&nonfinite[nonfiniteOffset], atomicLoad(&nonfiniteFlags));
  }`}));return d.push(C(e,{id:`${t}-decode`,operation:`GPUKernelDensity`,words:c,scale:s,maximumTermCount:r.length,output:a}),I(e,{id:`${t}-nonfinite-finish`,operation:`GPUKernelDensity`,invocationCount:1,bindings:[{name:`nonfinite`,view:u,type:`u32`,access:`read`},{name:`result`,view:a,type:`f32`,access:`read_write`}],body:`let flags = nonfinite[nonfiniteOffset];
  // Non-finite f32 constants are invalid WGSL, including bitcasts inside runtime branches.
  // Select integer bits from runtime flags, then bitcast the mutable local value once.
  var nonfiniteBits = 0u;
  if ((flags & 1u) != 0u || (flags & 6u) == 6u) {
    nonfiniteBits = 0x7fc00000u;
  } else if ((flags & 2u) != 0u) {
    nonfiniteBits = 0x7f800000u;
  } else if ((flags & 4u) != 0u) {
    nonfiniteBits = 0xff800000u;
  }
  if (nonfiniteBits != 0u) {
    result[resultOffset] = bitcast<f32>(nonfiniteBits);
  }`})),d}function Y(e,t,n,r){return n instanceof i?[I(e,{id:`${t}-clear`,operation:`GPUKernelDensity`,invocationCount:1,bindings:[{name:`invalidCount`,view:r,type:`u32`,access:`read_write`}],body:`invalidCount[invalidCountOffset] = 0u;`}),...l(n).flatMap((n,i)=>n.length===0?[]:[I(e,{id:`${t}-${i}`,operation:`GPUKernelDensity`,invocationCount:n.length,guardIndex:!1,workgroupSize:256,bindings:[{name:`excluded`,view:n,type:`u32`,access:`read`},{name:`invalidCount`,view:r,type:`atomic<u32>`,access:`read_write`}],declarations:`var<workgroup> excludedTotal: atomic<u32>;`,body:`if (localInvocationIndex == 0u) { atomicStore(&excludedTotal, 0u); }
  workgroupBarrier();
  if (index < INVOCATION_COUNT && excluded[excludedOffset + index] != 0u) {
    atomicAdd(&excludedTotal, 1u);
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    atomicAdd(&invalidCount[invalidCountOffset], atomicLoad(&excludedTotal));
  }`})])]:[...new o({id:t,input:n,output:r,operation:`sum`}).getCommandNodes(e)]}function X(e,t){let n=Math.max(1,t)*4;return n>e.device.limits.maxStorageBufferBindingSize||n>e.device.limits.maxBufferSize}export{M as _,X as a,F as b,V as c,B as d,W as f,I as g,k as h,q as i,H as l,A as m,G as n,K as o,U as p,J as r,R as s,Y as t,z as u,P as v,j as x,N as y};