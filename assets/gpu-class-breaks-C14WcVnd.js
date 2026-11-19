import{t as e}from"./classification-kernel-node-C62eCEbJ.js";import{E as t,T as n,c as r,f as i,p as a,v as o,y as s}from"./gpu-contributor-utils-BIwLQON5.js";import{Q as c}from"./dist-BusamGlx.js";import{r as l}from"./kernel-C3tfQOn4.js";import{t as u}from"./gpu-dispatch-utils-CbRzFAgT.js";import{t as d}from"./gpu-sort-xRm8l5yn.js";import{l as f}from"./gpu-command-graph-BdzdmV6u.js";import{t as p}from"./column-classification-shared-BTIVlxdJ.js";import{t as m}from"./column-quantiles-nodes-CFuZJihC.js";import{c as h,s as g}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as _}from"./gpu-scan-B4NNTfF9.js";import{t as v}from"./gpu-uint64-scan-_k5PA9Xg.js";import{a as y,i as b,r as x,t as S}from"./column-classification-rows-D2hiB8L9.js";function C(e,t){n(e,t.naturalBreaksMode===void 0||[`approximate`,`exact`,`monotone`].includes(t.naturalBreaksMode),`naturalBreaksMode must be approximate, exact or monotone`),n(e,t.naturalBreaksExactStorage===void 0||[`auto`,`materialized`,`recompute`].includes(t.naturalBreaksExactStorage),`naturalBreaksExactStorage must be auto, materialized or recompute`),n(e,t.naturalBreaksExactStorage===void 0||t.naturalBreaksMode===`exact`,`naturalBreaksExactStorage requires exact mode`)}var w=2048,T={method:0,classCount:1,layerCursor:2,refinementCursor:3,refinementMoved:4,refinementDone:5},E={minimumKey:0,maximumKey:1,validCount:2,finiteCount:3},D={done:0,breakCount:1,thresholdKey:2,previousCount:3},O=`
struct Moments {
  count: f32,
  mean: f32,
  m2: f32,
  minimum: f32,
  maximum: f32,
}

fn mergeMoments(left: Moments, right: Moments) -> Moments {
  if (right.count == 0.0) {
    return left;
  }
  if (left.count == 0.0) {
    return right;
  }
  let count = left.count + right.count;
  let delta = right.mean - left.mean;
  let mean = left.mean + delta * (right.count / count);
  let m2 = left.m2 + right.m2 + delta * delta * (left.count * right.count / count);
  return Moments(count, mean, m2, min(left.minimum, right.minimum), max(left.maximum, right.maximum));
}

fn emptyMoments() -> Moments {
  return Moments(0.0, 0.0, 0.0, 0.0, 0.0);
}

var<workgroup> sharedMoments: array<Moments, 256>;
var<workgroup> gateShared: u32;

fn reduceSharedMoments(local: u32) {
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    workgroupBarrier();
    if (local < stride) {
      sharedMoments[local] = mergeMoments(sharedMoments[local], sharedMoments[local + stride]);
    }
  }
}
`,k=3;function A(e,t,n){return{condition:{id:`${e}-gate`,source:`gpu`,mode:`indirect`,buffer:t.buffer,byteOffset:t.byteOffset+n*k*4},extraResources:[{buffer:t,usage:`indirect`}]}}function j(e,t){let n=u(t.operation,Math.max(t.invocationCount,1),t.workgroupSize,e.device.limits.maxComputeWorkgroupsPerDimension),r=t.slot*k;return`dispatch[dispatchOffset + ${r}u] = select(0u, ${n.x}u, ${t.activeExpression});
  dispatch[dispatchOffset + ${r+1}u] = ${n.y}u;
  dispatch[dispatchOffset + ${r+2}u] = ${n.z}u;`}function M(t,n){let{id:r,operation:i,values:a,mask:o,state:s,partials:c,moments:u,slot:d,gate:f,indirectGates:m}=n,h=a.length,g=Math.ceil(h/w),_=f.kind===`head-tail`?f.headTail:void 0,v=f.kind===`head-tail`?`state[stateOffset + ${T.method}u] == ${f.methodCode}u && headTail[headTailOffset + ${D.done}u] == 0u`:`state[stateOffset + ${T.method}u] == ${f.methodCode}u`,y=l(t,`${r}-origin`,`float32`,1),b=[{name:`values`,view:a,type:`f32`,access:`read`},...o?[{name:`rowMask`,view:o,type:`u32`,access:`read`}]:[],{name:`state`,view:s,type:`u32`,access:`read`},..._?[{name:`headTail`,view:_,type:`u32`,access:`read`}]:[]],x=e(t,{id:`${r}-origin`,operation:i,variant:`moments-origin`,bindings:[...b,{name:`origin`,view:y,type:`f32`,access:`read_write`}],invocationCount:256,workgroupSize:256,workgroupCollective:!0,...m?.merge,declarations:`fn getMomentRowCount() -> u32 { return arrayLength(&values) - valuesOffset; }
${p}
var<workgroup> firstRows: array<u32, 256>;
var<workgroup> gateShared: u32;`,body:`let local = localInvocationIndex;
  if (local == 0u) {
    gateShared = select(0u, 1u, ${v});
  }
  if (workgroupUniformLoad(&gateShared) == 0u) {
    return;
  }
  ${_?`let thresholdKey = headTail[headTailOffset + ${D.thresholdKey}u];`:``}
  var firstRow = getMomentRowCount();
  for (var row = local; row < getMomentRowCount(); row += 256u) {
    let value = values[valuesOffset + row];
    var include = (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
    ${o?`include = include && rowMask[rowMaskOffset + row] != 0u;`:``}
    ${_?`include = include && getOrderedKey(value) > thresholdKey;`:``}
    if (include) {
      firstRow = row;
      break;
    }
  }
  firstRows[local] = firstRow;
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    workgroupBarrier();
    if (local < stride) {
      firstRows[local] = min(firstRows[local], firstRows[local + stride]);
    }
  }
  if (local == 0u) {
    var pivot = 0.0;
    if (firstRows[0] < getMomentRowCount()) {
      pivot = values[valuesOffset + firstRows[0]];
    }
    origin[originOffset] = pivot;
  }`}),S=[{name:`values`,view:a,type:`f32`,access:`read`},...o?[{name:`rowMask`,view:o,type:`u32`,access:`read`}]:[],{name:`state`,view:s,type:`u32`,access:`read`},..._?[{name:`headTail`,view:_,type:`u32`,access:`read`}]:[],{name:`origin`,view:y,type:`f32`,access:`read`},{name:`partials`,view:c,type:`f32`,access:`read_write`}],C=e(t,{id:`${r}-tile`,operation:i,variant:`moments-tile`,bindings:S,invocationCount:g*256,workgroupSize:256,workgroupCollective:!0,...m?.tile,declarations:`fn getMomentRowCount() -> u32 { return arrayLength(&values) - valuesOffset; }
const TILE_SIZE: u32 = ${w}u;
const ROWS_PER_INVOCATION: u32 = 8u;
${p}
${O}
var<workgroup> centeredValueBits: array<u32, ${w}>;`,body:`let local = localInvocationIndex;
  if (local == 0u) {
    gateShared = select(0u, 1u, ${v});
  }
  if (workgroupUniformLoad(&gateShared) == 0u) {
    return;
  }
  let tile = workgroupIndex;
  ${_?`let thresholdKey = headTail[headTailOffset + ${D.thresholdKey}u];`:``}
  let pivot = origin[originOffset];
  for (var step = 0u; step < ROWS_PER_INVOCATION; step = step + 1u) {
    // Uniform trip count: rows past the end are skipped by a predicate, never by a break, so
    // the barriers after the loop stay in uniform control flow.
    let row = min(tile * TILE_SIZE + step * 256u + local, getMomentRowCount() - 1u);
    let inRange = tile * TILE_SIZE + step * 256u + local < getMomentRowCount();
    let value = values[valuesOffset + row];
    var include = inRange && (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
    ${o?`include = include && rowMask[rowMaskOffset + row] != 0u;`:``}
    ${_?`include = include && getOrderedKey(value) > thresholdKey;`:``}
    centeredValueBits[step * 256u + local] = select(0x7fc00000u, bitcast<u32>(value - pivot), include);
  }
  workgroupBarrier();
  var accumulated = emptyMoments();
  for (var step = 0u; step < ROWS_PER_INVOCATION; step += 1u) {
    // Read another invocation's materialized subtraction, so the optimizer cannot reassociate
    // (value - pivot) - mean into value - (pivot + mean) at the original large offset.
    let tileRow = step * 256u + (local ^ 128u);
    let shifted = bitcast<f32>(centeredValueBits[tileRow]);
    // Only the NaN sentinel excludes a row; an overflowing finite difference still counts.
    if ((bitcast<u32>(shifted) & 0x7fffffffu) <= 0x7f800000u) {
      let value = values[valuesOffset + tile * TILE_SIZE + tileRow];
      accumulated = mergeMoments(accumulated, Moments(1.0, shifted, 0.0, value, value));
    }
  }
  sharedMoments[local] = accumulated;
  reduceSharedMoments(local);
  if (local == 0u) {
    let record = sharedMoments[0];
    let base = partialsOffset + tile * 6u;
    partials[base] = record.count;
    partials[base + 1u] = record.mean;
    partials[base + 2u] = record.m2;
    partials[base + 3u] = record.minimum;
    partials[base + 4u] = record.maximum;
    partials[base + 5u] = select(0.0, pivot, record.count > 0.0);
  }`}),E=[{name:`state`,view:s,type:`u32`,access:`read`},..._?[{name:`headTail`,view:_,type:`u32`,access:`read`}]:[],{name:`origin`,view:y,type:`f32`,access:`read`},{name:`partials`,view:c,type:`f32`,access:`read`},{name:`moments`,view:u,type:`f32`,access:`read_write`}];return[x,C,e(t,{id:`${r}-merge`,operation:i,variant:`moments-merge`,bindings:E,invocationCount:256,workgroupSize:256,workgroupCollective:!0,...m?.merge,declarations:`fn getMomentTileCount() -> u32 { return (arrayLength(&partials) - partialsOffset) / 6u; }
fn getMomentStepCount() -> u32 { return (getMomentTileCount() + 255u) / 256u; }
${O}`,body:`let local = localInvocationIndex;
  if (local == 0u) {
    gateShared = select(0u, 1u, ${v});
  }
  if (workgroupUniformLoad(&gateShared) == 0u) {
    return;
  }
  var accumulated = emptyMoments();
  for (var step = 0u; step < getMomentStepCount(); step = step + 1u) {
    let tile = step * 256u + local;
    if (tile < getMomentTileCount()) {
      let base = partialsOffset + tile * 6u;
      accumulated = mergeMoments(
        accumulated,
        Moments(partials[base], partials[base + 1u], partials[base + 2u], partials[base + 3u], partials[base + 4u])
      );
    }
  }
  sharedMoments[local] = accumulated;
  reduceSharedMoments(local);
  if (local == 0u) {
    let record = sharedMoments[0];
    let base = momentsOffset + ${d*6}u;
    moments[base] = record.count;
    moments[base + 1u] = select(0.0, origin[originOffset] + record.mean, record.count > 0.0);
    moments[base + 2u] = record.m2;
    moments[base + 3u] = record.minimum;
    moments[base + 4u] = record.maximum;
    moments[base + 5u] = 0.0;
  }`})]}function N(e){return Math.ceil(e/w)}var P=`
fn getBinIndex(value: f32, minimum: f32, width: f32) -> u32 {
  let difference = value - minimum;
  var bin = floor(difference / width);
  bin = clamp(bin, 0.0, f32(BIN_COUNT - 1u));
  if (difference < bin * width) {
    bin = bin - 1.0;
  } else if (difference >= (bin + 1.0) * width) {
    bin = bin + 1.0;
  }
  return u32(clamp(bin, 0.0, f32(BIN_COUNT - 1u)));
}
`,F=1024,I=`
fn addWide(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  let low = left.x + right.x;
  let carry = select(0u, 1u, low < left.x);
  return vec2<u32>(low, left.y + right.y + carry);
}

fn subtractWide(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  let low = left.x - right.x;
  let borrow = select(0u, 1u, left.x < right.x);
  return vec2<u32>(low, left.y - right.y - borrow);
}

// Full 32 x 32 -> 64 bit product from 16-bit limbs; no partial product exceeds 2^32.
fn multiplyWide(left: u32, right: u32) -> vec2<u32> {
  let leftLow = left & 0xffffu;
  let leftHigh = left >> 16u;
  let rightLow = right & 0xffffu;
  let rightHigh = right >> 16u;
  let lowProduct = leftLow * rightLow;
  let crossA = leftLow * rightHigh;
  let crossB = leftHigh * rightLow;
  let highProduct = leftHigh * rightHigh;
  let cross = crossA + crossB;
  let crossCarry = select(0u, 1u, cross < crossA);
  let low = lowProduct + ((cross & 0xffffu) << 16u);
  let lowCarry = select(0u, 1u, low < lowProduct);
  return vec2<u32>(low, highProduct + (cross >> 16u) + (crossCarry << 16u) + lowCarry);
}

// Product modulo 2^64.
fn multiplyWideLow(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  let product = multiplyWide(left.x, right.x);
  return vec2<u32>(product.x, product.y + left.x * right.y + left.y * right.x);
}

fn wideToUnsignedFloat(value: vec2<u32>) -> f32 {
  return f32(value.y) * 4294967296.0 + f32(value.x);
}

// A small negative number has a huge low word in two's complement, so convert the magnitude.
fn wideToSignedFloat(value: vec2<u32>) -> f32 {
  if (bitcast<i32>(value.y) < 0) {
    return -wideToUnsignedFloat(subtractWide(vec2<u32>(0u, 0u), value));
  }
  return wideToUnsignedFloat(value);
}

fn getClassCost(count: u32, sum: vec2<u32>, squares: vec2<u32>) -> f32 {
  if (count == 0u) {
    return 0.0;
  }
  let countFloat = f32(count);
  let center = i32(round(wideToSignedFloat(sum) / countFloat));
  let centerBits = bitcast<u32>(center);
  let centerWide = vec2<u32>(centerBits, select(0u, 0xffffffffu, (center < 0)));
  let centeredSum = bitcast<i32>(sum.x - centerBits * count);
  let crossTerm = multiplyWideLow(centerWide, sum);
  let centerSquared = multiplyWideLow(centerWide, centerWide);
  let centeredSquares = addWide(
    subtractWide(squares, addWide(crossTerm, crossTerm)),
    multiplyWideLow(centerSquared, vec2<u32>(count, 0u))
  );
  let centeredSumFloat = f32(centeredSum);
  let correction = centeredSumFloat * centeredSumFloat / countFloat;
  return max(wideToUnsignedFloat(centeredSquares) - correction, 0.0);
}
`,L={"equal-interval":0,quantile:1,"standard-deviation":2,"head-tail":3,"box-plot":4,"maximum-breaks":5,"natural-breaks":6,custom:7},R=Object.keys(L);function z(e){return 8+e+1}function B(e,t,n=new Float32Array(z(t))){let r=z(t);if(n.length<r)throw Error(`GPUClassBreaks parameter target must hold ${r} elements`);let{method:a}=e,o=L[a];if(o===void 0)throw Error(`GPUClassBreaks unknown method ${String(a)}`);let s=e.classCount??5;if(a===`box-plot`)s=6;else if(a===`custom`){let t=e.customEdges?.length??0;if(t<2)throw Error(`GPUClassBreaks custom needs at least two edges`);s=t-1}i(`GPUClassBreaks`,`classCount`,s,1,t);for(let[t,n]of[[`standardDeviationInterval`,e.standardDeviationInterval],[`headTailRatio`,e.headTailRatio],[`boxPlotHinge`,e.boxPlotHinge]])if(n!==void 0&&!Number.isFinite(n))throw Error(`GPUClassBreaks ${t} must be finite`);if(a===`custom`&&Array.from(e.customEdges).some(Number.isNaN))throw Error(`GPUClassBreaks custom edges must not be NaN`);if(n.fill(NaN,0,r),n[0]=o,n[1]=s,n[2]=e.standardDeviationInterval??1,n[3]=e.headTailRatio??.4,n[4]=e.boxPlotHinge??1.5,n[5]=0,n[6]=0,n[7]=0,a===`quantile`)for(let e=1;e<s;e++)n[8+e-1]=e/s;else if(a===`box-plot`)n[8]=.25,n[9]=.5,n[10]=.75;else if(a===`custom`){let t=e.customEdges;for(let e=0;e<t.length;e++)n[8+e]=t[e]}return n}var V=L,H=`3.0e38`,U=`0xffffffffu`,ee=99,W=16,G=256;function K(e,t,n,r,i){return`{
    let previousLow = atomicAdd(&${e}[${t}], ${r});
    let carry = select(0u, 1u, previousLow + ${r} < previousLow);
    let highPart = ${i} + carry;
    if (highPart != 0u) {
      atomicAdd(&${e}[${n}], highPart);
    }
  }`}var te=6,ne=3,re=3;function ie(e){return(te+ne*Math.max(e-1,0))*re+ne}function ae(t,n){let{operation:i,values:a,mask:o,maximumClassCount:s,binCount:u}=n,d=`${n.id}-natural-breaks`,f=(e,n)=>l(t,`${d}-${e}`,`uint32`,Math.max(n,1)),m=(e,n)=>l(t,`${d}-${e}`,`float32`,Math.max(n,1)),g=f(`histogram`,5*u),_=f(`prefix`,5*(u+1)),v=m(`totals`,s*u),y=f(`starts`,s*u),b={name:`state`,view:n.state,type:`u32`,access:`read`},x=r(a),S=o?r(o):void 0,C=`state[stateOffset + ${T.method}u] != ${V[`natural-breaks`]}u`,w=`${p}
const BIN_COUNT: u32 = ${u}u;
const INVERSE_BIN_COUNT: f32 = ${h(1/u)};
${P}`,D=`let minimum = decodeOrderedKey(extremes[extremesOffset + ${E.minimumKey}u]);
  let maximum = decodeOrderedKey(extremes[extremesOffset + ${E.maximumKey}u]);
  let width = (maximum - minimum) * INVERSE_BIN_COUNT;`,O=[],k=r=>{let a=`${d}${r}`,o=[];o.push(e(t,{id:`${a}-clear`,operation:i,variant:`natural-breaks-clear`,bindings:[b,{name:`histogram`,view:g,type:`u32`,access:`read_write`}],invocationCount:5*u,body:`if (${C}) {
      return;
    }
    histogram[histogramOffset + index] = 0u;`}));for(let[r,s]of x.entries()){if(s.length===0)continue;let c=S?.[r],l=c?[{name:`rowMask`,view:c,type:`u32`,access:`read`}]:[],d=s.length;o.push(e(t,{id:x.length===1?`${a}-histogram`:`${a}-histogram-chunk-${r}`,operation:i,variant:`natural-breaks-histogram`,bindings:[{name:`values`,view:s,type:`f32`,access:`read`},...l,b,{name:`extremes`,view:n.extremes,type:`u32`,access:`read`},{name:`histogram`,view:g,type:`atomic<u32>`,access:`read_write`}],invocationCount:d,declarations:`${w}
  ${I}`,body:`if (${C}) {
      return;
    }
    let value = values[valuesOffset + index];
    if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
      return;
    }
    ${c?`if (rowMask[rowMaskOffset + index] == 0u) {
    return;
  }`:``}
    // Rows outside [minimum, maximum] belong to other classes when the stage runs over one.
    let key = getOrderedKey(value);
    if (key < extremes[extremesOffset + ${E.minimumKey}u] || key > extremes[extremesOffset + ${E.maximumKey}u]) {
      return;
    }
    ${D}
    var bin = 0u;
    var offset = 0u;
    if (width > 0.0) {
      bin = getBinIndex(value, minimum, width);
      // Position inside the bin in 1/256 bins.
      let remainder = (value - minimum) - f32(bin) * width;
      offset = u32(clamp(floor(remainder * (256.0 / width)), 0.0, 255.0));
    }
    let position = bin * 256u + offset;
    let square = multiplyWide(position, position);
    atomicAdd(&histogram[histogramOffset + bin], 1u);
    ${K(`histogram`,`histogramOffset + ${u}u + bin`,`histogramOffset + ${2*u}u + bin`,`position`,`0u`)}
    ${K(`histogram`,`histogramOffset + ${3*u}u + bin`,`histogramOffset + ${4*u}u + bin`,`square.x`,`square.y`)}`}))}o.push(e(t,{id:`${a}-prefix`,operation:i,variant:`natural-breaks-prefix`,bindings:[b,{name:`histogram`,view:g,type:`u32`,access:`read`},{name:`prefix`,view:_,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const BIN_COUNT: u32 = ${u}u;
  const PREFIX_STRIDE: u32 = ${u+1}u;
  ${I}`,body:`if (${C}) {
      return;
    }
    var count = 0u;
    var sum = vec2<u32>(0u, 0u);
    var squares = vec2<u32>(0u, 0u);
    for (var bin = 0u; bin < BIN_COUNT; bin = bin + 1u) {
      prefix[prefixOffset + bin] = count;
      prefix[prefixOffset + PREFIX_STRIDE + bin] = sum.x;
      prefix[prefixOffset + 2u * PREFIX_STRIDE + bin] = sum.y;
      prefix[prefixOffset + 3u * PREFIX_STRIDE + bin] = squares.x;
      prefix[prefixOffset + 4u * PREFIX_STRIDE + bin] = squares.y;
      count = count + histogram[histogramOffset + bin];
      sum = addWide(sum, vec2<u32>(histogram[histogramOffset + BIN_COUNT + bin], histogram[histogramOffset + 2u * BIN_COUNT + bin]));
      squares = addWide(squares, vec2<u32>(histogram[histogramOffset + 3u * BIN_COUNT + bin], histogram[histogramOffset + 4u * BIN_COUNT + bin]));
    }
    prefix[prefixOffset + BIN_COUNT] = count;
    prefix[prefixOffset + PREFIX_STRIDE + BIN_COUNT] = sum.x;
    prefix[prefixOffset + 2u * PREFIX_STRIDE + BIN_COUNT] = sum.y;
    prefix[prefixOffset + 3u * PREFIX_STRIDE + BIN_COUNT] = squares.x;
    prefix[prefixOffset + 4u * PREFIX_STRIDE + BIN_COUNT] = squares.y;`}));let f=`const BIN_COUNT: u32 = ${u}u;
  const PREFIX_STRIDE: u32 = ${u+1}u;
  ${I}

  fn getRangeCost(first: u32, last: u32) -> f32 {
    let end = last + 1u;
    let count = prefix[prefixOffset + end] - prefix[prefixOffset + first];
    let sum = subtractWide(
      vec2<u32>(prefix[prefixOffset + PREFIX_STRIDE + end], prefix[prefixOffset + 2u * PREFIX_STRIDE + end]),
      vec2<u32>(prefix[prefixOffset + PREFIX_STRIDE + first], prefix[prefixOffset + 2u * PREFIX_STRIDE + first])
    );
    let squares = subtractWide(
      vec2<u32>(prefix[prefixOffset + 3u * PREFIX_STRIDE + end], prefix[prefixOffset + 4u * PREFIX_STRIDE + end]),
      vec2<u32>(prefix[prefixOffset + 3u * PREFIX_STRIDE + first], prefix[prefixOffset + 4u * PREFIX_STRIDE + first])
    );
    return getClassCost(count, sum, squares);
  }`,p={name:`prefix`,view:_,type:`u32`,access:`read`},m={name:`totals`,view:v,type:`f32`,access:`read_write`},h={name:`starts`,view:y,type:`u32`,access:`read_write`};o.push(e(t,{id:`${a}-layer-0`,operation:i,variant:`natural-breaks-layer`,bindings:[b,p,m,h],invocationCount:u,declarations:f,body:`if (${C}) {
      return;
    }
    totals[totalsOffset + index] = getRangeCost(0u, index);
    starts[startsOffset + index] = 0u;`}));let O=l(t,`${a}-layer-dispatch`,`uint32`,3,c.STORAGE|c.INDIRECT),k=`state[stateOffset + ${T.layerCursor}u]`,M=`state[stateOffset + ${T.method}u] == ${V[`natural-breaks`]}u && layer < state[stateOffset + ${T.classCount}u]`;for(let r=1;r<s;r++){let s=`${a}-layer-${r}`;o.push(e(t,{id:`${s}-advance`,operation:i,variant:`natural-breaks-layer-advance`,bindings:[{name:`state`,view:n.state,type:`u32`,access:`read_write`},{name:`dispatch`,view:O,type:`u32`,access:`read_write`}],invocationCount:1,body:`let layer = ${k} + 1u;
    ${k} = layer;
    ${j(t,{operation:i,slot:0,invocationCount:G,workgroupSize:G,activeExpression:`(${M})`})}`}),e(t,{id:s,operation:i,variant:`natural-breaks-layer`,workgroupCollective:!0,bindings:[b,p,m,h],invocationCount:G,workgroupSize:G,...A(s,O,0),declarations:`${f}
  const WORKGROUP_SIZE: u32 = ${G}u;
  // Best start of the last class for every solved end bin (opt_c). 4 B per bin: 8 KiB at 2048 bins.
  var<workgroup> optimum: array<u32, ${u}>;
  var<workgroup> candidateCost: array<f32, ${G}>;
  var<workgroup> candidateStart: array<u32, ${G}>;
  var<workgroup> sharedLayer: u32;

  // Candidates for the last class of a mid at 1-based position 'position' (end bin 'end'): from the
  // optimum of the solved neighbour on the left to that of the solved neighbour on the right.
  fn getSearchLow(layer: u32, position: u32, stride: u32, end: u32) -> u32 {
    if (position == stride) {
      return layer;
    }
    return max(optimum[end - stride], layer);
  }

  fn getSearchHigh(count: u32, position: u32, stride: u32, end: u32, low: u32) -> u32 {
    var high = BIN_COUNT - 1u;
    if (position + stride <= count) {
      high = optimum[end + stride];
    }
    // Rounding in the costs can break monotonicity by one step; never search an empty range.
    return min(end, max(high, low));
  }

  fn getCandidateCost(layer: u32, start: u32, end: u32) -> f32 {
    return totals[totalsOffset + (layer - 1u) * BIN_COUNT + start - 1u] + getRangeCost(start, end);
  }`,body:`let local = localInvocationIndex;
    if (local == 0u) {
      sharedLayer = ${k};
    }
    let layer = workgroupUniformLoad(&sharedLayer);
    if (layer >= BIN_COUNT) {
      return;
    }
    // Ends layer..BIN_COUNT-1 are positions 1..count of a perfect tree with 2^levels - 1 mids.
    let count = BIN_COUNT - layer;
    var levels = 0u;
    while ((1u << levels) - 1u < count) {
      levels = levels + 1u;
    }
    for (var level = 0u; level < levels; level = level + 1u) {
      let stride = 1u << (levels - 1u - level);
      let nodeCount = 1u << level;
      let teamSize = max(WORKGROUP_SIZE >> level, 1u);
      let teamCount = WORKGROUP_SIZE / teamSize;
      let team = local / teamSize;
      let lane = local % teamSize;
      if (teamSize == 1u) {
        // Deep levels: a thread per node, no cooperation and so no barrier in the loop.
        for (var node = team; node < nodeCount; node = node + teamCount) {
          let position = (2u * node + 1u) * stride;
          if (position <= count) {
            let end = layer + position - 1u;
            let low = getSearchLow(layer, position, stride, end);
            let high = getSearchHigh(count, position, stride, end, low);
            var bestCost = ${H};
            var bestStart = low;
            for (var start = low; start <= high; start = start + 1u) {
              let cost = getCandidateCost(layer, start, end);
              if (cost < bestCost) {
                bestCost = cost;
                bestStart = start;
              }
            }
            optimum[end] = bestStart;
            totals[totalsOffset + layer * BIN_COUNT + end] = bestCost;
            starts[startsOffset + layer * BIN_COUNT + end] = bestStart;
          }
        }
      } else {
        // Shallow levels: a team of lanes per node scans its range together and the lanes reduce
        // to the lowest start of least cost.
        let position = (2u * team + 1u) * stride;
        let valid = position <= count;
        let end = layer + position - 1u;
        var bestCost = ${H};
        var bestStart = 0xffffffffu;
        if (valid) {
          let low = getSearchLow(layer, position, stride, end);
          let high = getSearchHigh(count, position, stride, end, low);
          for (var start = low + lane; start <= high; start = start + teamSize) {
            let cost = getCandidateCost(layer, start, end);
            if (cost < bestCost) {
              bestCost = cost;
              bestStart = start;
            }
          }
        }
        candidateCost[local] = bestCost;
        candidateStart[local] = bestStart;
        for (var reduction = teamSize >> 1u; reduction > 0u; reduction = reduction >> 1u) {
          workgroupBarrier();
          if (lane < reduction) {
            let otherCost = candidateCost[local + reduction];
            let otherStart = candidateStart[local + reduction];
            if (otherCost < candidateCost[local] || (otherCost == candidateCost[local] && otherStart < candidateStart[local])) {
              candidateCost[local] = otherCost;
              candidateStart[local] = otherStart;
            }
          }
        }
        if (valid && lane == 0u) {
          optimum[end] = candidateStart[local];
          totals[totalsOffset + layer * BIN_COUNT + end] = candidateCost[local];
          starts[startsOffset + layer * BIN_COUNT + end] = candidateStart[local];
        }
      }
      // The next level reads the optima of this one.
      workgroupBarrier();
    }`}))}return o};return O.push(...k(``)),O.push(...oe(t,n,{prefix:d,u32:f,f32:m,starts:y,getCoarseNodes:k,gate:C,binDeclarations:w,extremesRead:D})),O}function oe(t,n,i){let{operation:a,values:o,mask:s,maximumClassCount:d,binCount:f,edges:m,edgeState:_}=n,v=r(o),y=s?r(s):void 0,{prefix:b,gate:x,binDeclarations:S,extremesRead:C}=i,w=`min(state[stateOffset + ${T.classCount}u], BIN_COUNT)`,D=`const BIN_COUNT: u32 = ${f}u;`,O=o.length,k=F,M=k+2,N=Math.max(d-1,1),P=`state[stateOffset + ${T.method}u]`,L={name:`state`,view:n.state,type:`u32`,access:`read`},R=[],z=i.u32(`extents`,3*d),B=i.f32(`bulk-edges`,d+1),re=i.u32(`backup`,8),ae=i.u32(`break-state`,3*(d+1)),oe=i.f32(`plan`,3*(d+1)),q=i.u32(`cells`,N*M*7),J=l(t,`${b}-refinement-dispatch`,`uint32`,ie(v.length),c.STORAGE|c.INDIRECT),se=N*M*7,Y=G,ce=(e,t)=>t===0?e:te+ne*(t-1)+(e===0?0:e===3?1:2),le=e=>e===0?0:e===3?1:2,X=(e,t)=>`dispatch[arrayLength(&dispatch) - 3u + ${le(e)}u] = select(0u, 1u, ${t});`,ue=(n,r)=>v.flatMap((i,o)=>{if(i.length===0)return[];let s=u(a,i.length,Y,t.device.limits.maxComputeWorkgroupsPerDimension);return[e(t,{id:`${n}-chunk-${o}`,operation:a,variant:`natural-breaks-arm-rows`,invocationCount:1,parameterValues:[ce(r,o)*3,le(r),s.x,s.y,s.z],bindings:[{name:`dispatch`,view:J,type:`u32`,access:`read_write`}],body:`let first = dispatchOffset + getChunkParameter(36u);
  let isEnabled = dispatch[arrayLength(&dispatch) - 3u + getChunkParameter(37u)] != 0u;
  dispatch[first] = select(0u, getChunkParameter(38u), isEnabled);
  dispatch[first + 1u] = getChunkParameter(39u);
  dispatch[first + 2u] = getChunkParameter(40u);`})]}),de=g(N),fe=g(se),Z=`const MAXIMUM_CLASS_COUNT: u32 = ${d}u;
${D}`,pe=`const HALF_WIDTH_BASE: u32 = 0u;
const MOVEMENT_BASE: u32 = ${d+1}u;
const LEVEL_BASE: u32 = ${2*(d+1)}u;`,me=`const CELL_COUNT: u32 = ${k}u;
const ELEMENT_COUNT: u32 = ${M}u;
const ELEMENT_STRIDE: u32 = 7u;
const NO_MINIMUM: u32 = ${U};`,he=2**18;if(R.push(e(t,{id:`${b}-backtrack`,operation:a,variant:`natural-breaks-backtrack`,bindings:[L,{name:`extremes`,view:n.extremes,type:`u32`,access:`read`},{name:`starts`,view:i.starts,type:`u32`,access:`read`},{name:`edges`,view:m,type:`f32`,access:`read_write`},{name:`edgeState`,view:_,type:`u32`,access:`read_write`},{name:`extents`,view:z,type:`u32`,access:`read_write`},{name:`dispatch`,view:J,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`${S}
const MAXIMUM_CLASS_COUNT: u32 = ${d}u;`,body:`let classCount = ${w};
  let isActive = !(${x}) && classCount >= 2u && extremes[extremesOffset + ${E.minimumKey}u] < extremes[extremesOffset + ${E.maximumKey}u];
  ${X(0,`isActive`)}
  if (${x}) {
    return;
  }
  ${C}
  var end = BIN_COUNT - 1u;
  for (var layer = classCount - 1u; layer > 0u; layer = layer - 1u) {
    let start = starts[startsOffset + layer * BIN_COUNT + end];
    // The class starting at bin 'start' begins at that bin's lower edge.
    edges[edgesOffset + layer] = minimum + f32(start) * width;
    end = start - 1u;
  }
  edgeState[edgeStateOffset] = classCount;
  for (var index = 0u; index < MAXIMUM_CLASS_COUNT; index = index + 1u) {
    extents[extentsOffset + index] = 0xffffffffu;
    extents[extentsOffset + MAXIMUM_CLASS_COUNT + index] = 0u;
    extents[extentsOffset + 2u * MAXIMUM_CLASS_COUNT + index] = 0u;
  }`})),d<2)return R;let ge=(n,r)=>v.flatMap((i,o)=>{if(i.length===0)return[];let s=v.length===1?n:`${n}-chunk-${o}`,c=y?.[o],l=c?[{name:`rowMask`,view:c,type:`u32`,access:`read`}]:[],u=c?`rowMask[rowMaskOffset + index] != 0u`:`true`;return[e(t,{id:s,operation:a,variant:`natural-breaks-extents`,bindings:[{name:`values`,view:i,type:`f32`,access:`read`},...l,L,{name:`edges`,view:m,type:`f32`,access:`read`},{name:`extents`,view:z,type:`atomic<u32>`,access:`read_write`}],invocationCount:i.length,workgroupSize:Y,...A(s,J,ce(r,o)),declarations:`${p}
${Z}`,body:`let value = values[valuesOffset + index];
  if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u || !(${u})) {
    return;
  }
  let classCount = ${w};
  let key = getOrderedKey(value);
  // The class is the number of cuts e[1..classCount-1] at or below the key.
  var low = 1u;
  var high = classCount;
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (getOrderedKey(edges[edgesOffset + middle]) <= key) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  atomicMin(&extents[extentsOffset + low - 1u], key);
  atomicMax(&extents[extentsOffset + MAXIMUM_CLASS_COUNT + low - 1u], key);
  atomicAdd(&extents[extentsOffset + 2u * MAXIMUM_CLASS_COUNT + low - 1u], 1u);`})]}),_e=n=>e(t,{id:n,operation:a,variant:`natural-breaks-snap`,bindings:[L,{name:`extents`,view:z,type:`u32`,access:`read`},{name:`edges`,view:m,type:`f32`,access:`read_write`},{name:`breakState`,view:ae,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`${p}
${Z}
${pe}`,body:`if (${x}) {
    return;
  }
  let classCount = ${w};
  for (var cut = 1u; cut < classCount; cut = cut + 1u) {
    let smallest = extents[extentsOffset + cut];
    if (smallest != 0xffffffffu) {
      edges[edgesOffset + cut] = decodeOrderedKey(smallest);
    }
  }
  for (var index = 0u; index <= MAXIMUM_CLASS_COUNT; index = index + 1u) {
    breakState[breakStateOffset + HALF_WIDTH_BASE + index] = 0u;
    breakState[breakStateOffset + MOVEMENT_BASE + index] = bitcast<u32>(${H});
    breakState[breakStateOffset + LEVEL_BASE + index] = 0u;
  }
  // No break lies beside the first and last class.
  breakState[breakStateOffset + MOVEMENT_BASE] = 0u;
  breakState[breakStateOffset + MOVEMENT_BASE + classCount] = 0u;`});R.push(...ue(`${b}-extents-arm`,0)),R.push(...ge(`${b}-extents`,0)),R.push(_e(`${b}-snap`)),R.push(e(t,{id:`${b}-bulk-switch`,operation:a,variant:`natural-breaks-bulk-switch`,bindings:[{name:`state`,view:n.state,type:`u32`,access:`read_write`},{name:`extremes`,view:n.extremes,type:`u32`,access:`read_write`},{name:`extents`,view:z,type:`u32`,access:`read`},{name:`backup`,view:re,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`${Z}
const NO_MINIMUM: u32 = ${U};`,body:`let classCount = ${w};
  let minimumKey = extremes[extremesOffset + ${E.minimumKey}u];
  let maximumKey = extremes[extremesOffset + ${E.maximumKey}u];
  backup[backupOffset] = ${P};
  backup[backupOffset + 1u] = state[stateOffset + ${T.classCount}u];
  backup[backupOffset + 2u] = minimumKey;
  backup[backupOffset + 3u] = maximumKey;
  backup[backupOffset + 4u] = 0u;
  backup[backupOffset + 5u] = 0u;
  backup[backupOffset + 6u] = 0u;
  var bulk = 0u;
  var bulkRows = 0u;
  var nonEmpty = 0u;
  for (var index = 0u; index < classCount; index = index + 1u) {
    if (extents[extentsOffset + index] != NO_MINIMUM) {
      nonEmpty = nonEmpty + 1u;
      let rows = extents[extentsOffset + 2u * MAXIMUM_CLASS_COUNT + index];
      if (rows > bulkRows) {
        bulkRows = rows;
        bulk = index;
      }
    }
  }
  let emptyCount = classCount - nonEmpty;
  let bulkLow = extents[extentsOffset + bulk];
  let bulkHigh = extents[extentsOffset + MAXIMUM_CLASS_COUNT + bulk];
  let isActive = ${P} == ${V[`natural-breaks`]}u && classCount >= 2u && minimumKey < maximumKey
    && emptyCount > 0u && bulkLow < bulkHigh;
  if (isActive) {
    backup[backupOffset + 4u] = bulk;
    backup[backupOffset + 5u] = 1u;
    backup[backupOffset + 6u] = emptyCount;
    extremes[extremesOffset + ${E.minimumKey}u] = bulkLow;
    extremes[extremesOffset + ${E.maximumKey}u] = bulkHigh;
    state[stateOffset + ${T.classCount}u] = min(1u + emptyCount, BIN_COUNT);
    ${`state[stateOffset + ${T.layerCursor}u]`} = 0u;
  } else {
    // Every node of the second stage returns at once.
    ${P} = 0xffffffffu;
  }`})),R.push(...i.getCoarseNodes(`-bulk`)),R.push(e(t,{id:`${b}-bulk-backtrack`,operation:a,variant:`natural-breaks-bulk-backtrack`,bindings:[L,{name:`extremes`,view:n.extremes,type:`u32`,access:`read`},{name:`starts`,view:i.starts,type:`u32`,access:`read`},{name:`bulkEdges`,view:B,type:`f32`,access:`read_write`}],invocationCount:1,declarations:S,body:`if (${x}) {
    return;
  }
  let classCount = ${w};
  ${C}
  var end = BIN_COUNT - 1u;
  for (var layer = classCount - 1u; layer > 0u; layer = layer - 1u) {
    let start = starts[startsOffset + layer * BIN_COUNT + end];
    bulkEdges[bulkEdgesOffset + layer] = minimum + f32(start) * width;
    end = start - 1u;
  }`})),R.push(e(t,{id:`${b}-bulk-assemble`,operation:a,variant:`natural-breaks-bulk-assemble`,bindings:[{name:`state`,view:n.state,type:`u32`,access:`read_write`},{name:`extremes`,view:n.extremes,type:`u32`,access:`read_write`},{name:`extents`,view:z,type:`u32`,access:`read_write`},{name:`edges`,view:m,type:`f32`,access:`read_write`},{name:`bulkEdges`,view:B,type:`f32`,access:`read`},{name:`backup`,view:re,type:`u32`,access:`read`},{name:`dispatch`,view:J,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`${p}
${Z}
const NO_MINIMUM: u32 = ${U};`,body:`let isActive = backup[backupOffset + 5u] == 1u;
  ${X(5,`isActive`)}
  ${P} = backup[backupOffset];
  state[stateOffset + ${T.classCount}u] = backup[backupOffset + 1u];
  extremes[extremesOffset + ${E.minimumKey}u] = backup[backupOffset + 2u];
  extremes[extremesOffset + ${E.maximumKey}u] = backup[backupOffset + 3u];
  if (!isActive) {
    return;
  }
  let classCount = min(backup[backupOffset + 1u], BIN_COUNT);
  let bulk = backup[backupOffset + 4u];
  let parts = min(1u + backup[backupOffset + 6u], BIN_COUNT);
  var emitted = 0u;
  var isFirst = true;
  for (var index = 0u; index < classCount; index = index + 1u) {
    if (extents[extentsOffset + index] == NO_MINIMUM) {
      continue;
    }
    // The smallest value of the class starts it; the first start is the column minimum.
    if (isFirst) {
      isFirst = false;
    } else {
      emitted = emitted + 1u;
      edges[edgesOffset + emitted] = decodeOrderedKey(extents[extentsOffset + index]);
    }
    if (index == bulk) {
      for (var part = 1u; part < parts; part = part + 1u) {
        emitted = emitted + 1u;
        edges[edgesOffset + emitted] = bulkEdges[bulkEdgesOffset + part];
      }
    }
  }
  for (var index = 0u; index < MAXIMUM_CLASS_COUNT; index = index + 1u) {
    extents[extentsOffset + index] = NO_MINIMUM;
    extents[extentsOffset + MAXIMUM_CLASS_COUNT + index] = 0u;
    extents[extentsOffset + 2u * MAXIMUM_CLASS_COUNT + index] = 0u;
  }`})),R.push(...ue(`${b}-extents-bulk-arm`,5)),R.push(...ge(`${b}-extents-bulk`,5)),R.push(_e(`${b}-snap-bulk`));let Q=`state[stateOffset + ${T.refinementCursor}u]`,ve=`(1u - (${`(${Q} - 1u)`} & 1u))`,$=k/2,ye=2/k;for(let r=0;r<W;r++){let i=`${b}-refine-${r}`;R.push(e(t,{id:`${i}-advance`,operation:a,variant:`natural-breaks-refine-advance`,bindings:[{name:`state`,view:n.state,type:`u32`,access:`read_write`},{name:`extremes`,view:n.extremes,type:`u32`,access:`read`},{name:`breakState`,view:ae,type:`u32`,access:`read`},{name:`dispatch`,view:J,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`${D}
${pe}`,body:`let cursor = ${Q};
  let classCount = ${w};
  var isActive = ${P} == ${V[`natural-breaks`]}u && classCount >= 2u
    && extremes[extremesOffset + ${E.minimumKey}u] < extremes[extremesOffset + ${E.maximumKey}u]
    && state[stateOffset + ${T.refinementDone}u] == 0u
    && cursor < ${W}u;
  if (isActive && cursor > 0u && (cursor & 1u) == 0u) {
    var zoomed = true;
    for (var cut = 1u; cut < classCount; cut = cut + 1u) {
      if (breakState[breakStateOffset + LEVEL_BASE + cut] < 2u) {
        zoomed = false;
      }
    }
    if (state[stateOffset + ${T.refinementMoved}u] == 0u && zoomed) {
      state[stateOffset + ${T.refinementDone}u] = 1u;
      isActive = false;
    }
    state[stateOffset + ${T.refinementMoved}u] = 0u;
  }
  ${Q} = cursor + 1u;
  ${[{slot:1,invocationCount:N,workgroupSize:de},{slot:2,invocationCount:se,workgroupSize:fe},{slot:3,invocationCount:O,workgroupSize:Y},{slot:4,invocationCount:N,workgroupSize:de}].map(e=>e.slot===3?X(3,`isActive`):j(t,{operation:a,...e,activeExpression:`isActive`})).join(`
  `)}`})),R.push(...ue(`${i}-rows-arm`,3)),R.push(e(t,{id:`${i}-plan`,operation:a,variant:`natural-breaks-refine-plan`,bindings:[L,{name:`edges`,view:m,type:`f32`,access:`read`},{name:`extents`,view:z,type:`u32`,access:`read`},{name:`breakState`,view:ae,type:`u32`,access:`read_write`},{name:`plan`,view:oe,type:`f32`,access:`read_write`}],invocationCount:N,workgroupSize:de,...A(`${i}-plan`,J,1),declarations:`${p}
${Z}
${pe}`,body:`let cut = index + 1u;
  if (cut >= ${w}) {
    return;
  }
  plan[planOffset + 3u * cut + 1u] = 0.0;
  if ((cut & 1u) != ${ve}) {
    return;
  }
  // The data extent of the two classes beside the cut.
  let lowKey = min(extents[extentsOffset + cut - 1u], extents[extentsOffset + cut]);
  let highKey = max(extents[extentsOffset + MAXIMUM_CLASS_COUNT + cut - 1u], extents[extentsOffset + MAXIMUM_CLASS_COUNT + cut]);
  if (lowKey == 0xffffffffu) {
    breakState[breakStateOffset + LEVEL_BASE + cut] = ${ee}u;
    return;
  }
  let lowValue = decodeOrderedKey(lowKey);
  let highValue = decodeOrderedKey(highKey);
  // A cut beside an empty class lies outside the data; centre the window inside the extent.
  let anchor = min(max(edges[edgesOffset + cut], lowValue), highValue);
  let span = max(anchor - lowValue, highValue - anchor);
  if (!(span > 0.0)) {
    breakState[breakStateOffset + LEVEL_BASE + cut] = ${ee}u;
    return;
  }
  // The whole extent until the break has been refined; then the zoomed window, widened to follow
  // neighbours that moved since the break's own last update.
  var halfWidth = span;
  let stored = bitcast<f32>(breakState[breakStateOffset + HALF_WIDTH_BASE + cut]);
  if (stored > 0.0) {
    let shift = max(
      bitcast<f32>(breakState[breakStateOffset + MOVEMENT_BASE + cut - 1u]),
      bitcast<f32>(breakState[breakStateOffset + MOVEMENT_BASE + cut + 1u])
    );
    halfWidth = min(max(stored, 2.0 * min(shift, span)), span);
  }
  plan[planOffset + 3u * cut] = anchor;
  plan[planOffset + 3u * cut + 1u] = halfWidth * ${h(ye)};
  plan[planOffset + 3u * cut + 2u] = ${h(he)} / span;`})),R.push(e(t,{id:`${i}-clear`,operation:a,variant:`natural-breaks-refine-clear`,bindings:[{name:`cells`,view:q,type:`u32`,access:`read_write`}],invocationCount:se,workgroupSize:fe,...A(`${i}-clear`,J,2),declarations:me,body:`// Words: count, sum, squares (low and high), smallest key, largest key.
  let word = index % ELEMENT_STRIDE;
  cells[cellsOffset + index] = select(0u, NO_MINIMUM, word == 5u);`}));for(let[n,r]of v.entries()){if(r.length===0)continue;let o=y?.[n],s=o?[{name:`rowMask`,view:o,type:`u32`,access:`read`}]:[],c=r.length,l=o?`rowMask[rowMaskOffset + index] != 0u`:`true`;R.push(e(t,{id:v.length===1?`${i}-accumulate`:`${i}-accumulate-chunk-${n}`,operation:a,variant:`natural-breaks-refine-accumulate`,bindings:[{name:`values`,view:r,type:`f32`,access:`read`},...s,L,{name:`edges`,view:m,type:`f32`,access:`read`},{name:`plan`,view:oe,type:`f32`,access:`read`},{name:`cells`,view:q,type:`atomic<u32>`,access:`read_write`}],invocationCount:c,workgroupSize:Y,...A(`${i}-accumulate-chunk-${n}`,J,ce(3,n)),declarations:`${p}
${I}
${D}
${me}
const QUANTIZATION: f32 = ${h(he)};`,body:`let value = values[valuesOffset + index];
  if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u || !(${l})) {
    return;
  }
  let classCount = ${w};
  let key = getOrderedKey(value);
  // The class of the row: the number of cuts e[1..classCount-1] at or below its key.
  var low = 1u;
  var high = classCount;
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (getOrderedKey(edges[edgesOffset + middle]) <= key) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let rowClass = low - 1u;
  // The one break of this half-round beside the class: below it (rowClass) or above it.
  let wanted = ${ve};
  var cut = 0u;
  if (rowClass >= 1u && (rowClass & 1u) == wanted) {
    cut = rowClass;
  } else if (rowClass + 1u < classCount && ((rowClass + 1u) & 1u) == wanted) {
    cut = rowClass + 1u;
  }
  if (cut == 0u) {
    return;
  }
  let cellWidth = plan[planOffset + 3u * cut + 1u];
  if (!(cellWidth > 0.0)) {
    return;
  }
  let offset = value - plan[planOffset + 3u * cut];
  // Cell of the row, with the current boundary at cell ${$}: rows of the class on the
  // right of the cut never fall left of it and the other way round.
  var cell = i32(floor(clamp(offset / cellWidth, -${2*k}.0, ${2*k}.0))) + ${$};
  if (rowClass == cut) {
    cell = max(cell, ${$});
  } else {
    cell = min(cell, ${$-1});
  }
  // Element 0 holds the rows left of the window, 1..CELL_COUNT the cells, the last the rows right.
  var element = u32(cell) + 1u;
  if (cell < 0) {
    element = 0u;
  } else if (cell >= ${k}) {
    element = ELEMENT_COUNT - 1u;
  }
  let quantized = i32(round(clamp(offset * plan[planOffset + 3u * cut + 2u], -QUANTIZATION, QUANTIZATION)));
  let magnitude = u32(abs(quantized));
  let square = multiplyWide(magnitude, magnitude);
  let base = cellsOffset + ((cut - 1u) * ELEMENT_COUNT + element) * ELEMENT_STRIDE;
  atomicAdd(&cells[base], 1u);
  ${K(`cells`,`base + 1u`,`base + 2u`,`bitcast<u32>(quantized)`,`select(0u, 0xffffffffu, (quantized < 0))`)}
  ${K(`cells`,`base + 3u`,`base + 4u`,`square.x`,`square.y`)}
  atomicMin(&cells[base + 5u], key);
  atomicMax(&cells[base + 6u], key);`}))}R.push(e(t,{id:`${i}-solve`,operation:a,variant:`natural-breaks-refine-solve`,bindings:[{name:`state`,view:n.state,type:`atomic<u32>`,access:`read_write`},{name:`edges`,view:m,type:`f32`,access:`read_write`},{name:`extents`,view:z,type:`u32`,access:`read_write`},{name:`breakState`,view:ae,type:`u32`,access:`read_write`},{name:`plan`,view:oe,type:`f32`,access:`read`},{name:`cells`,view:q,type:`u32`,access:`read`}],invocationCount:N,workgroupSize:de,...A(`${i}-solve`,J,4),declarations:`${p}
${I}
${Z}
${pe}
${me}`,body:`let cut = index + 1u;
  if (cut >= min(atomicLoad(&state[stateOffset + ${T.classCount}u]), BIN_COUNT)) {
    return;
  }
  let cellWidth = plan[planOffset + 3u * cut + 1u];
  if (!(cellWidth > 0.0)) {
    return;
  }
  let base = cellsOffset + (cut - 1u) * ELEMENT_COUNT * ELEMENT_STRIDE;
  var totalCount = 0u;
  var totalSum = vec2<u32>(0u, 0u);
  var totalSquares = vec2<u32>(0u, 0u);
  for (var element = 0u; element < ELEMENT_COUNT; element = element + 1u) {
    let at = base + element * ELEMENT_STRIDE;
    totalCount = totalCount + cells[at];
    totalSum = addWide(totalSum, vec2<u32>(cells[at + 1u], cells[at + 2u]));
    totalSquares = addWide(totalSquares, vec2<u32>(cells[at + 3u], cells[at + 4u]));
  }
  // Cost of every boundary 0..CELL_COUNT: elements 0..boundary are left of it. The current cut is
  // boundary ${$}; another boundary has to be strictly cheaper to replace it.
  var leftCount = cells[base];
  var leftSum = vec2<u32>(cells[base + 1u], cells[base + 2u]);
  var leftSquares = vec2<u32>(cells[base + 3u], cells[base + 4u]);
  var bestCost = ${H};
  var bestBoundary = ${$}u;
  var currentCost = ${H};
  var currentLeftCount = 0u;
  var cost = ${H};
  for (var boundary = 0u; boundary <= CELL_COUNT; boundary = boundary + 1u) {
    // An empty cell leaves the partition, and so the cost, as it was one boundary before.
    var isChanged = boundary == 0u || boundary == ${$}u;
    if (boundary > 0u) {
      let at = base + boundary * ELEMENT_STRIDE;
      isChanged = isChanged || cells[at] != 0u;
      leftCount = leftCount + cells[at];
      leftSum = addWide(leftSum, vec2<u32>(cells[at + 1u], cells[at + 2u]));
      leftSquares = addWide(leftSquares, vec2<u32>(cells[at + 3u], cells[at + 4u]));
    }
    if (isChanged) {
      cost = getClassCost(leftCount, leftSum, leftSquares)
        + getClassCost(totalCount - leftCount, subtractWide(totalSum, leftSum), subtractWide(totalSquares, leftSquares));
    }
    if (boundary == ${$}u) {
      currentCost = cost;
      currentLeftCount = leftCount;
    }
    if (cost < bestCost) {
      bestCost = cost;
      bestBoundary = boundary;
    }
  }
  if (currentCost <= bestCost) {
    bestBoundary = ${$}u;
  }
  // Extents of the two new classes: elements 0..bestBoundary on the left, the rest on the right.
  var leftMinimum = NO_MINIMUM;
  var leftMaximum = 0u;
  var rightMinimum = NO_MINIMUM;
  var rightMaximum = 0u;
  for (var element = 0u; element < ELEMENT_COUNT; element = element + 1u) {
    let at = base + element * ELEMENT_STRIDE;
    if (cells[at] != 0u) {
      if (element <= bestBoundary) {
        leftMinimum = min(leftMinimum, cells[at + 5u]);
        leftMaximum = max(leftMaximum, cells[at + 6u]);
      } else {
        rightMinimum = min(rightMinimum, cells[at + 5u]);
        rightMaximum = max(rightMaximum, cells[at + 6u]);
      }
    }
  }
  extents[extentsOffset + cut - 1u] = leftMinimum;
  extents[extentsOffset + MAXIMUM_CLASS_COUNT + cut - 1u] = leftMaximum;
  extents[extentsOffset + cut] = rightMinimum;
  extents[extentsOffset + MAXIMUM_CLASS_COUNT + cut] = rightMaximum;
  // The cut is the smallest value on its right: an actual data value.
  var movement = 0.0;
  if (rightMinimum != NO_MINIMUM) {
    let newCut = decodeOrderedKey(rightMinimum);
    movement = abs(newCut - edges[edgesOffset + cut]);
    edges[edgesOffset + cut] = newCut;
    if (movement > 0.0) {
      atomicOr(&state[stateOffset + ${T.refinementMoved}u], 1u);
    }
  }
  breakState[breakStateOffset + MOVEMENT_BASE + cut] = bitcast<u32>(movement);
  // Zoom around the chosen boundary, unless a class beside the break was empty (the window must
  // cover the extent again) or the optimum sits at the window edge (the cut may continue).
  let degenerate = currentLeftCount == 0u || totalCount - currentLeftCount == 0u;
  let atEdge = abs(i32(bestBoundary) - ${$}) >= ${$-4};
  let halfWidth = cellWidth * ${h($)};
  var nextHalfWidth = halfWidth * ${h(8/k)};
  var level = breakState[breakStateOffset + LEVEL_BASE + cut] + 1u;
  if (degenerate) {
    nextHalfWidth = 0.0;
    level = 0u;
  } else if (atEdge) {
    nextHalfWidth = halfWidth;
    level = level - 1u;
  }
  breakState[breakStateOffset + HALF_WIDTH_BASE + cut] = bitcast<u32>(nextHalfWidth);
  breakState[breakStateOffset + LEVEL_BASE + cut] = level;`}))}return R.push(e(t,{id:`${b}-finalize`,operation:a,variant:`natural-breaks-finalize`,bindings:[L,{name:`extents`,view:z,type:`u32`,access:`read`},{name:`edges`,view:m,type:`f32`,access:`read_write`},{name:`edgeState`,view:_,type:`u32`,access:`read_write`}],invocationCount:1,declarations:Z,body:`if (${x}) {
    return;
  }
  let classCount = edgeState[edgeStateOffset];
  var kept = 0u;
  for (var cut = 1u; cut < classCount; cut = cut + 1u) {
    if (extents[extentsOffset + cut] != 0xffffffffu) {
      kept = kept + 1u;
      edges[edgesOffset + kept] = edges[edgesOffset + cut];
    }
  }
  edgeState[edgeStateOffset] = kept + 1u;`})),R}var q=1024,J=8192,se=262144,Y=`fn getNormalizedValue(value: f32, scale: f32) -> f32 {
  let valueParts = frexp(value);
  let scaleParts = frexp(scale);
  return ldexp(valueParts.fract / scaleParts.fract, valueParts.exp - scaleParts.exp);
}`;function ce(t,n){let{values:r,mask:i,state:a,extremes:s,edges:f,edgeState:m,maximumClassCount:h}=n,g=`${n.id}-exact-natural-breaks`,_=r.length,v=_+1,y=u(`GPUClassBreaks`,_,64,t.device.limits.maxComputeWorkgroupsPerDimension),b=(t,n)=>e(t,{...n,parameterValues:[_,v,y.x,y.y,y.z]}),x=(e,n)=>l(t,`${g}-${e}`,`uint32`,n),S=x(`cursor`,1),C=l(t,`${g}-dispatch`,`uint32`,3,c.STORAGE|c.INDIRECT),w=x(`keys`,_),D=x(`rows`,_),O=x(`sorted-keys`,_),k=x(`sorted-rows`,_),j=n.exactStorage===`materialized`||n.exactStorage!==`recompute`&&_<=1024;o(t.device,g,`exact partitions`,(h+1)*v*4,`reduce rows or maximumClassCount`),j&&o(t.device,g,`exact costs`,_*_*4,`use naturalBreaksExactStorage: recompute`);let M=j?l(t,`${g}-costs`,`float32`,_*_):void 0,N=x(`partitions`,(h+1)*v),P=l(t,`${g}-scores`,`float32`,(j?h+1:2)*v),F=(e,t,n=`u32`,r=`read`)=>({name:e,view:t,type:n,access:r}),I=`state[stateOffset + ${T.method}u] != ${L[`natural-breaks`]}u`,R=`extremes[extremesOffset + ${E.finiteCount}u]`,z=`${p}\nfn getReferenceRowCount() -> u32 { return getChunkParameter(36u); }\nfn getReferenceStride() -> u32 { return getChunkParameter(37u); }`,B=[];B.push(b(t,{id:`${g}-keys`,operation:`GPUClassBreaks`,variant:`exact-keys`,invocationCount:_,bindings:[F(`values`,r,`f32`),...i?[F(`rowMask`,i)]:[],F(`keys`,w,`u32`,`read_write`),F(`rows`,D,`u32`,`read_write`)],declarations:z,body:`let value = values[valuesOffset + index];
  let include = (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u ${i?`&& rowMask[rowMaskOffset + index] != 0u`:``};
  keys[keysOffset + index] = select(0xffffffffu, getOrderedKey(value), include);
  rows[rowsOffset + index] = index;`})),B.push(...new d({id:`${g}-sort`,keys:w,values:D,outputKeys:O,outputValues:k,algorithm:`radix`}).getCommandNodes(t)),M&&B.push(b(t,{id:`${g}-costs`,operation:`GPUClassBreaks`,variant:`exact-costs`,invocationCount:_,bindings:[F(`state`,a),F(`extremes`,s),F(`sortedKeys`,O),F(`costs`,M,`f32`,`read_write`)],declarations:`${z}\n${Y}`,body:`if (${I}) { return; }
  let count = ${R};
  if (index >= count) { return; }
  let minimum = decodeOrderedKey(sortedKeys[sortedKeysOffset]);
  let maximum = decodeOrderedKey(sortedKeys[sortedKeysOffset + count - 1u]);
  let magnitude = max(abs(minimum), abs(maximum));
  // Power-of-two scaling preserves normal f32 mantissas and avoids overflow on opposite extremes.
  // Use the magnitude itself for subnormals; zero is the only scale replaced with one.
  let exponentScale = bitcast<f32>(bitcast<u32>(magnitude) & 0x7f800000u);
  let scale = select(select(1.0, magnitude, magnitude > 0.0), exponentScale, exponentScale > 0.0);
  let origin = getNormalizedValue(decodeOrderedKey(sortedKeys[sortedKeysOffset + index]), scale);
  var mean = 0.0;
  var variance = 0.0;
  for (var start = index + 1u; start > 0u;) {
    start = start - 1u;
    let value = getNormalizedValue(decodeOrderedKey(sortedKeys[sortedKeysOffset + start]), scale) - origin;
    let delta = value - mean;
    mean = mean + delta / f32(index - start + 1u);
    variance = variance + delta * (value - mean);
    costs[costsOffset + index * getReferenceRowCount() + start] = max(variance, 0.0);
  }`})),B.push(b(t,{id:`${g}-initialize`,operation:`GPUClassBreaks`,variant:`exact-initialize`,invocationCount:(h+1)*v,bindings:[F(`scores`,P,`f32`,`read_write`),F(`partitions`,N,`u32`,`read_write`),F(`cursor`,S,`u32`,`read_write`)],body:`if (index == 0u) { cursor[cursorOffset] = 0u; }
  // Normalized finite interval costs are at most 16 * getReferenceRowCount(), far below this sentinel.
  if (index < (arrayLength(&scores) - scoresOffset)) { scores[scoresOffset + index] = select(1e30, 0.0, index == 0u); }
  partitions[partitionsOffset + index] = 0u;`}));for(let e=1;e<=h;e++)B.push(b(t,{id:`${g}-advance-${e}`,operation:`GPUClassBreaks`,variant:`exact-advance`,invocationCount:1,bindings:[F(`cursor`,S,`u32`,`read_write`),F(`state`,a),F(`dispatch`,C,`u32`,`read_write`)],body:`let layer = cursor[cursorOffset] + 1u;
  cursor[cursorOffset] = layer;
  let isEnabled = !(${I}) && layer <= state[stateOffset + ${T.classCount}u];
  dispatch[dispatchOffset] = select(0u, getChunkParameter(38u), isEnabled);
  dispatch[dispatchOffset + 1u] = getChunkParameter(39u);
  dispatch[dispatchOffset + 2u] = getChunkParameter(40u);`})),B.push(b(t,{id:`${g}-layer-${e}`,operation:`GPUClassBreaks`,variant:`exact-layer`,invocationCount:_,workgroupSize:64,...A(`${g}-layer-${e}`,C,0),bindings:[F(`state`,a),F(`cursor`,S),F(`extremes`,s),F(`sortedKeys`,O),...M?[F(`costs`,M,`f32`)]:[],F(`scores`,P,`f32`,`read_write`),F(`partitions`,N,`u32`,`read_write`)],declarations:`${z}\n${Y}`,body:`if (${I}) { return; }
  let layer = cursor[cursorOffset];
  if (layer > state[stateOffset + ${T.classCount}u]) { return; }
  let end = index + 1u;
  if (end > ${R}) { return; }
  var best = 1e30;
  var split = 0u;
  ${M?`for (var start = 0u; start < end; start = start + 1u) {
    if (start > 0u && decodeOrderedKey(sortedKeys[sortedKeysOffset + start - 1u]) == decodeOrderedKey(sortedKeys[sortedKeysOffset + start])) { continue; }
    let candidate = scores[scoresOffset + (layer - 1u) * getReferenceStride() + start] + costs[costsOffset + index * getReferenceRowCount() + start];
    if (candidate < best) { best = candidate; split = start; }
  }
  scores[scoresOffset + layer * getReferenceStride() + end] = best;`:`
  let minimum = decodeOrderedKey(sortedKeys[sortedKeysOffset]);
  let maximum = decodeOrderedKey(sortedKeys[sortedKeysOffset + ${R} - 1u]);
  let magnitude = max(abs(minimum), abs(maximum));
  let exponentScale = bitcast<f32>(bitcast<u32>(magnitude) & 0x7f800000u);
  let scale = select(select(1.0, magnitude, magnitude > 0.0), exponentScale, exponentScale > 0.0);
  let origin = getNormalizedValue(decodeOrderedKey(sortedKeys[sortedKeysOffset + index]), scale);
  var mean = 0.0;
  var variance = 0.0;
  for (var start = end; start > 0u;) {
    start--;
    // Keep every update, including duplicates: weighting or reordering changes this recurrence.
    let value = getNormalizedValue(decodeOrderedKey(sortedKeys[sortedKeysOffset + start]), scale) - origin;
    let delta = value - mean;
    mean = mean + delta / f32(end - start);
    variance = variance + delta * (value - mean);
    if (start > 0u && decodeOrderedKey(sortedKeys[sortedKeysOffset + start - 1u]) == decodeOrderedKey(sortedKeys[sortedKeysOffset + start])) { continue; }
    let candidate = scores[scoresOffset + ((layer - 1u) & 1u) * getReferenceStride() + start] + max(variance, 0.0);
    // Descending traversal still chooses the same lowest split as the ascending materialized path.
    if (candidate < best || (candidate == best && start < split)) { best = candidate; split = start; }
  }
  if (index == 0u) { scores[scoresOffset + (layer & 1u) * getReferenceStride()] = 1e30; }
  scores[scoresOffset + (layer & 1u) * getReferenceStride() + end] = best;`}
  partitions[partitionsOffset + layer * getReferenceStride() + end] = split;`}));return B.push(b(t,{id:`${g}-backtrack`,operation:`GPUClassBreaks`,variant:`exact-backtrack`,invocationCount:1,bindings:[F(`state`,a),F(`extremes`,s),F(`sortedKeys`,O),F(`partitions`,N),F(`edges`,f,`f32`,`read_write`),F(`edgeState`,m,`u32`,`read_write`)],declarations:z,body:`if (${I}) { return; }
  var end = ${R};
  var distinct = select(0u, 1u, end > 0u);
  for (var row = 1u; row < end; row = row + 1u) {
    if (decodeOrderedKey(sortedKeys[sortedKeysOffset + row - 1u]) != decodeOrderedKey(sortedKeys[sortedKeysOffset + row])) { distinct = distinct + 1u; }
  }
  let classes = min(distinct, state[stateOffset + ${T.classCount}u]);
  edgeState[edgeStateOffset] = classes;
  for (var layer = classes; layer > 1u; layer = layer - 1u) {
    let split = partitions[partitionsOffset + layer * getReferenceStride() + end];
    edges[edgesOffset + layer - 1u] = decodeOrderedKey(sortedKeys[sortedKeysOffset + split]);
    end = split;
  }`})),B}function le(t,n){let{values:r,mask:i,state:a,extremes:s,edges:f,edgeState:m,maximumClassCount:h}=n,g=`${n.id}-monotone-natural-breaks`,y=r.length,b=y+1,x=(t,n)=>e(t,{...n,parameterValues:[y,b,...n.parameterValues??[]]}),S=u(g,y*64,64,t.device.limits.maxComputeWorkgroupsPerDimension);for(let[e,n]of[[`partitions`,(h+1)*b],[`tasks`,y*8],[`prefix moments`,y*5]])o(t.device,g,e,n*4,`reduce rows or maximumClassCount`);let C=(e,n)=>l(t,`${g}-${e}`,`uint32`,n),w=C(`keys`,y),D=C(`sorted-keys`,y),O=C(`rows`,y),k=C(`sorted-rows`,y),j=C(`moments`,y*5),M=C(`prefix`,y*5),N=C(`partitions`,(h+1)*b),P=l(t,`${g}-scores`,`float32`,b*2),F=C(`tasks`,y*8),R=C(`cursor`,5),z=(e,t,n=`u32`,r=`read`)=>({name:e,view:t,type:n,access:r}),B=`state[stateOffset + ${T.method}u] == ${L[`natural-breaks`]}u`,V=`extremes[extremesOffset + ${E.finiteCount}u]`,H=`${p}\n${I}\nfn getReferenceRowCount() -> u32 { return getChunkParameter(36u); }\nfn getReferenceStride() -> u32 { return getChunkParameter(37u); }`,U=[x(t,{id:`${g}-initialize`,operation:`GPUClassBreaks`,variant:`monotone-initialize`,invocationCount:Math.max(P.length,5),bindings:[z(`scores`,P,`f32`,`read_write`),z(`cursor`,R,`u32`,`read_write`)],body:`if (index < (arrayLength(&scores) - scoresOffset)) { scores[scoresOffset + index] = select(1e30, 0.0, index == 0u); }
  if (index < 5u) { cursor[cursorOffset + index] = 0u; }`}),x(t,{id:`${g}-keys`,operation:`GPUClassBreaks`,variant:`monotone-keys`,invocationCount:y,bindings:[z(`values`,r,`f32`),...i?[z(`rowMask`,i)]:[],z(`keys`,w,`u32`,`read_write`),z(`rows`,O,`u32`,`read_write`)],declarations:p,body:`let value = values[valuesOffset + index];
  let include = (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u ${i?`&& rowMask[rowMaskOffset + index] != 0u`:``};
  keys[keysOffset + index] = select(0xffffffffu, getOrderedKey(value), include);
  rows[rowsOffset + index] = index;`}),...new d({id:`${g}-sort`,keys:w,values:O,outputKeys:D,outputValues:k,algorithm:`radix`}).getCommandNodes(t),x(t,{id:`${g}-quantize`,operation:`GPUClassBreaks`,variant:`monotone-quantize`,invocationCount:y,bindings:[z(`sortedKeys`,D),z(`extremes`,s),z(`moments`,j,`u32`,`read_write`),z(`cursor`,R,`atomic<u32>`,`read_write`)],declarations:`${H}\n${Y}`,body:`let count = ${V};
  var position = 0u;
  var distinct = false;
  if (index < count) {
    let minimum = decodeOrderedKey(sortedKeys[sortedKeysOffset]);
    let maximum = decodeOrderedKey(sortedKeys[sortedKeysOffset + count - 1u]);
    let magnitude = max(abs(minimum), abs(maximum));
    let exponentScale = bitcast<f32>(bitcast<u32>(magnitude) & 0x7f800000u);
    let scale = select(select(1.0, magnitude, magnitude > 0.0), exponentScale, exponentScale > 0.0);
    let lower = getNormalizedValue(minimum, scale);
    let span = getNormalizedValue(maximum, scale) - lower;
    let value = decodeOrderedKey(sortedKeys[sortedKeysOffset + index]);
    if (span > 0.0) {
      position = u32(round(clamp((getNormalizedValue(value, scale) - lower) / span, 0.0, 1.0) * 1048575.0));
    }
    // Numeric equality makes negative and positive zero one value, even though their keys differ.
    distinct = index == 0u;
    if (index > 0u) { distinct = value != decodeOrderedKey(sortedKeys[sortedKeysOffset + index - 1u]); }
    if (distinct) { atomicAdd(&cursor[cursorOffset + 4u], 1u); }
  }
  let square = multiplyWide(position, position);
  moments[momentsOffset + index] = position;
  moments[momentsOffset + getReferenceRowCount() + index] = 0u;
  moments[momentsOffset + 2u * getReferenceRowCount() + index] = square.x;
  moments[momentsOffset + 3u * getReferenceRowCount() + index] = square.y;
  moments[momentsOffset + 4u * getReferenceRowCount() + index] = u32(distinct);`})],ee=(e,n)=>t.createDataView(e.buffer,{format:`uint32`,length:y,byteOffset:e.byteOffset+n*y*4}),W=C(`prefix-low`,y),G=C(`prefix-high`,y),K=l(t,`${g}-task-dispatch`,`uint32`,3,c.STORAGE|c.INDIRECT);for(let e of[0,2])U.push(...new v({id:`${g}-prefix-${e}`,inputLow:ee(j,e),inputHigh:ee(j,e+1),outputLow:W,outputHigh:G}).getCommandNodes(t),x(t,{id:`${g}-pack-prefix-${e}`,operation:`GPUClassBreaks`,variant:`monotone-pack-prefix`,invocationCount:y,bindings:[z(`low`,W),z(`high`,G),z(`prefix`,t.createDataView(M.buffer,{format:`uint32`,byteOffset:M.byteOffset+e*y*4,length:2*y}),`u32`,`read_write`)],body:`prefix[prefixOffset + index] = low[lowOffset + index];
  prefix[prefixOffset + getChunkParameter(36u) + index] = high[highOffset + index];`}));U.push(...new _({id:`${g}-distinct-prefix`,input:ee(j,4),output:ee(M,4),mode:`inclusive`}).getCommandNodes(t));let te=Math.floor(Math.log2(y))+1;for(let e=0;e<h;e++){U.push(x(t,{id:`${g}-layer-${e}`,operation:`GPUClassBreaks`,variant:`monotone-layer`,invocationCount:1,bindings:[z(`state`,a),z(`extremes`,s),z(`cursor`,R,`u32`,`read_write`),z(`tasks`,F,`u32`,`read_write`),z(`scores`,P,`f32`,`read_write`)],declarations:H,body:`let layer = cursor[cursorOffset] + 1u;
  let count = ${V};
  cursor[cursorOffset] = layer;
  cursor[cursorOffset + 1u] = 0u;
  cursor[cursorOffset + 2u] = select(0u, 1u, ${B} && count > 0u && layer <= min(cursor[cursorOffset + 4u], state[stateOffset + ${T.classCount}u]));
  cursor[cursorOffset + 3u] = count;
  scores[scoresOffset + (layer & 1u) * getReferenceStride()] = 1e30;
  tasks[tasksOffset] = 1u;
  tasks[tasksOffset + 1u] = count;
  tasks[tasksOffset + 2u] = layer - 1u;
  tasks[tasksOffset + 3u] = select(0u, count - 1u, count > 0u);`}));for(let n=0;n<te;n++)U.push(x(t,{id:`${g}-dispatch-${e}-${n}`,operation:`GPUClassBreaks`,invocationCount:1,parameterValues:[S.x,S.y],bindings:[z(`cursor`,R),z(`dispatch`,K,`u32`,`read_write`)],body:`let taskCount = min(cursor[cursorOffset + 2u], getChunkParameter(36u));
  let maximumX = getChunkParameter(38u);
  let maximumY = getChunkParameter(39u);
  var x = min(taskCount, maximumX);
  var y = 1u;
  var z = 1u;
  if (taskCount > maximumX) {
    x = maximumX;
    y = min((taskCount + maximumX - 1u) / maximumX, maximumY);
  }
  if (taskCount > maximumX * maximumY) {
    y = maximumY;
    z = (taskCount + maximumX * maximumY - 1u) / (maximumX * maximumY);
  }
  dispatch[dispatchOffset] = x;
  dispatch[dispatchOffset + 1u] = y;
  dispatch[dispatchOffset + 2u] = z;`}),x(t,{id:`${g}-search-${e}-${n}`,operation:`GPUClassBreaks`,variant:`monotone-search`,workgroupCollective:!0,invocationCount:y*64,workgroupSize:64,...A(`${g}-search-${e}-${n}`,K,0),bindings:[z(`sortedKeys`,D),z(`prefix`,M),z(`cursor`,R),z(`tasks`,F,`u32`,`read_write`),z(`scores`,P,`f32`,`read_write`),z(`partitions`,N,`u32`,`read_write`)],declarations:`${H}
var<workgroup> candidateCosts: array<f32, 64>;
var<workgroup> candidateSplits: array<u32, 64>;
var<workgroup> requiresFallback: u32;
fn readPrefix(field: u32, end: u32) -> vec2<u32> {
  if (end == 0u) { return vec2<u32>(0u); }
  return vec2<u32>(prefix[prefixOffset + field * getReferenceRowCount() + end - 1u], prefix[prefixOffset + (field + 1u) * getReferenceRowCount() + end - 1u]);
}
fn rangeCost(first: u32, end: u32) -> f32 {
  return getClassCost(end - first, subtractWide(readPrefix(0u, end), readPrefix(0u, first)), subtractWide(readPrefix(2u, end), readPrefix(2u, first)));
}`,body:`let task = index / 64u;
  let local = localInvocationIndex;
  let layer = cursor[cursorOffset];
  let level = cursor[cursorOffset + 1u];
  let taskCount = cursor[cursorOffset + 2u];
  let base = tasksOffset + (level & 1u) * getReferenceRowCount() * 4u + min(task, getReferenceRowCount() - 1u) * 4u;
  let first = tasks[base]; let last = tasks[base + 1u];
  let lower = tasks[base + 2u]; let upper = tasks[base + 3u];
  let end = first + (last - min(first, last)) / 2u;
  var best = 1e30; var split = lower;
  let feasible = task < taskCount && first <= last && prefix[prefixOffset + 4u * getReferenceRowCount() + min(end - 1u, getReferenceRowCount() - 1u)] >= layer;
  if (feasible) {
    let high = min(max(upper, lower), end - 1u);
    for (var start = lower + local; start <= high; start += 64u) {
      if (start > 0u && decodeOrderedKey(sortedKeys[sortedKeysOffset + start - 1u]) == decodeOrderedKey(sortedKeys[sortedKeysOffset + start])) { continue; }
      let candidate = scores[scoresOffset + ((layer - 1u) & 1u) * getReferenceStride() + start] + rangeCost(start, end);
      if (candidate < best || (candidate == best && start < split)) { best = candidate; split = start; }
    }
  }
  candidateCosts[local] = best; candidateSplits[local] = split;
  workgroupBarrier();
  for (var width = 32u; width > 0u; width >>= 1u) {
    if (local < width) {
      let otherCost = candidateCosts[local + width]; let otherSplit = candidateSplits[local + width];
      if (otherCost < candidateCosts[local] || (otherCost == candidateCosts[local] && otherSplit < candidateSplits[local])) {
        candidateCosts[local] = otherCost; candidateSplits[local] = otherSplit;
      }
    }
    workgroupBarrier();
  }
  if (local == 0u) { requiresFallback = u32(feasible && candidateCosts[0] >= 1e30); }
  let fallback = workgroupUniformLoad(&requiresFallback);
  if (fallback != 0u) {
    // Rounding can make the inherited search interval miss all valid boundaries. Recover a
    // valid partition by searching this endpoint exhaustively; this is not an optimality proof.
    best = 1e30; split = layer - 1u;
    for (var start = layer - 1u + local; start < end; start += 64u) {
      if (start > 0u && decodeOrderedKey(sortedKeys[sortedKeysOffset + start - 1u]) == decodeOrderedKey(sortedKeys[sortedKeysOffset + start])) { continue; }
      let candidate = scores[scoresOffset + ((layer - 1u) & 1u) * getReferenceStride() + start] + rangeCost(start, end);
      if (candidate < best || (candidate == best && start < split)) { best = candidate; split = start; }
    }
    candidateCosts[local] = best; candidateSplits[local] = split;
    workgroupBarrier();
    for (var width = 32u; width > 0u; width >>= 1u) {
      if (local < width) {
        let otherCost = candidateCosts[local + width]; let otherSplit = candidateSplits[local + width];
        if (otherCost < candidateCosts[local] || (otherCost == candidateCosts[local] && otherSplit < candidateSplits[local])) {
          candidateCosts[local] = otherCost; candidateSplits[local] = otherSplit;
        }
      }
      workgroupBarrier();
    }
  }
  if (local == 0u && task < taskCount) {
    let selected = candidateSplits[0];
    if (first <= last) {
      scores[scoresOffset + (layer & 1u) * getReferenceStride() + end] = candidateCosts[0];
      partitions[partitionsOffset + layer * getReferenceStride() + end] = selected;
    }
    if (taskCount * 2u <= getReferenceRowCount()) {
      let next = tasksOffset + ((level + 1u) & 1u) * getReferenceRowCount() * 4u + task * 8u;
      tasks[next] = first; tasks[next + 1u] = select(0u, end - 1u, first <= last);
      tasks[next + 2u] = lower; tasks[next + 3u] = max(lower, selected);
      tasks[next + 4u] = select(1u, end + 1u, first <= last); tasks[next + 5u] = select(0u, last, first <= last);
      tasks[next + 6u] = max(lower, selected); tasks[next + 7u] = max(upper, selected);
    }
  }`}),x(t,{id:`${g}-advance-${e}-${n}`,operation:`GPUClassBreaks`,variant:`monotone-advance`,invocationCount:1,bindings:[z(`cursor`,R,`u32`,`read_write`)],body:`cursor[cursorOffset + 1u]++;
  let count = cursor[cursorOffset + 2u] * 2u;
  cursor[cursorOffset + 2u] = select(0u, count, count <= cursor[cursorOffset + 3u]);`}))}return U.push(x(t,{id:`${g}-backtrack`,operation:`GPUClassBreaks`,variant:`monotone-backtrack`,invocationCount:1,bindings:[z(`state`,a),z(`sortedKeys`,D),z(`cursor`,R),z(`partitions`,N),z(`edges`,f,`f32`,`read_write`),z(`edgeState`,m,`u32`,`read_write`)],declarations:H,body:`if (!(${B})) { return; }
  var end = cursor[cursorOffset + 3u];
  let classes = min(cursor[cursorOffset + 4u], state[stateOffset + ${T.classCount}u]);
  edgeState[edgeStateOffset] = classes;
  for (var layer = classes; layer > 1u; layer--) {
    let split = partitions[partitionsOffset + layer * getReferenceStride() + end];
    edges[edgesOffset + layer - 1u] = decodeOrderedKey(sortedKeys[sortedKeysOffset + split]);
    end = split;
  }`})),U}var X=`GPUClassBreaks`,ue=64,de=2**24,fe=1024,Z=128,pe=38,me=2048,he=2048,ge=3,_e=4294967295,Q=L,ve=4,$=class{id;props;methods;naturalBreaksBinCount;quantileCount;estimatedCommandNodeCount;constructor(e){this.id=e.id??`class-breaks`,this.props=e;let o=this.id,{maximumClassCount:c,output:l}=e;if(i(o,`maximumClassCount`,c,1,ue),this.methods=[...new Set(e.methods??R)],this.methods.length===0)throw Error(`${o} needs at least one method`);for(let e of this.methods)if(Q[e]===void 0)throw Error(`${o} unknown method ${String(e)}`);if(this.methods.includes(`box-plot`)&&c<6)throw Error(`${o} box-plot needs maximumClassCount of at least 6`);this.naturalBreaksBinCount=e.naturalBreaksBinCount??me,i(o,`naturalBreaksBinCount`,this.naturalBreaksBinCount,2,he),this.quantileCount=Math.max(c-1,3),(this.methods.includes(`maximum-breaks`)||this.methods.includes(`natural-breaks`)&&e.naturalBreaksMode&&e.naturalBreaksMode!==`approximate`)&&t(e.values,[`float32`,`uint32`,`sint32`],`${o} values (compiled sort methods)`),b(o,`values`,e.values,[`float32`,`uint32`,`sint32`]);let u=e.values.length;if(C(o,e),e.naturalBreaksMode===`exact`&&i(o,`exact natural breaks values length`,u,1,e.naturalBreaksExactStorage===`materialized`?q:J),e.naturalBreaksMode===`monotone`&&i(o,`monotone natural breaks values length`,u,1,se),n(o,u>=1&&u<=de,`values length must be in [1, ${de}] (2^24): moment counts and quantile ranks are held in float32 and are exact only up to 2^24`),e.mask&&(b(o,`mask`,e.mask,[`uint32`],e.values),e.mask.length!==u))throw Error(`${o} mask length must equal values length`);t(e.parameters,[`float32`],`${o} parameters`);let d=z(c);if(e.parameters.length<d)throw Error(`${o} parameters must hold ${d} float32 values`);if(t(l.breaks,[`float32`],`${o} output.breaks`),l.breaks.length<c+1)throw Error(`${o} output.breaks must hold maximumClassCount + 1 rows`);if(t(l.classCount,[`uint32`],`${o} output.classCount`),l.classCount.length<1)throw Error(`${o} output.classCount must hold one row`);if(l.classCounts&&(t(l.classCounts,[`uint32`],`${o} output.classCounts`),l.classCounts.length<c))throw Error(`${o} output.classCounts must hold maximumClassCount rows`);x(o,[l.breaks,l.classCount,l.classCounts]),s(o,[l.breaks,l.classCount,l.classCounts],[e.values,e.mask,e.parameters]);let f=r(e.values).filter(e=>e.length>0).length;this.estimatedCommandNodeCount=32+8*f+4*c+(this.methods.includes(`natural-breaks`)?Z*c+pe*f:0)+(this.methods.includes(`maximum-breaks`)?2*fe:0)+(e.naturalBreaksMode&&e.naturalBreaksMode!==`approximate`?fe+4*c:0),a(o,this.estimatedCommandNodeCount)}getCommandNodes(t){let{id:n,props:i,methods:a,quantileCount:s}=this,{parameters:d,output:h,maximumClassCount:g}=i,_=[];a.includes(`natural-breaks`)&&(!i.naturalBreaksMode||i.naturalBreaksMode===`approximate`)&&o(t.device,n,`natural-breaks refinement dispatch scratch`,ie(r(i.values).length)*4,`use fewer source chunks`);let v=a.includes(`standard-deviation`)||a.includes(`head-tail`);(v||i.values.format!==`float32`)&&(o(t.device,n,`packed conversion/moment scratch`,i.values.length*4,`use fewer rows or omit moment methods`),o(t.device,n,`moment-partials`,N(i.values.length)*6*4,`use fewer rows`)),S(n,t,{values:i.values,mask:i.mask,parameters:d,"output.breaks":h.breaks,"output.classCount":h.classCount,"output.classCounts":h.classCounts});let b=i.values.length,x=g,C=a.includes(`quantile`)||a.includes(`box-plot`),w=a.includes(`natural-breaks`),O=this.naturalBreaksBinCount,k=i.naturalBreaksMode===`exact`,j=w&&(!i.naturalBreaksMode||i.naturalBreaksMode===`approximate`),P=k&&(i.naturalBreaksExactStorage===`materialized`||i.naturalBreaksExactStorage!==`recompute`&&b<=1024);y(t,n,{state:6,"packed value conversion/moment input":v||i.values.format!==`float32`?b:0,"packed mask conversion/moment input":i.mask&&(v||i.values.format!==`float32`)?b:0,"natural refinement dispatch":j?ie(r(i.values).length):0,extremes:4,edges:x+1,"quantile histogram":C?2*s*256:0,"quantile state":C?8*s+2:0,"quantile parameters":C?4+s:0,"head-tail state":a.includes(`head-tail`)?ve+Math.max(x-1,1):0,moments:v?12:0,"head-tail dispatch":a.includes(`head-tail`)?6:0,"moment partials":v?N(b)*6:0,"natural histogram":j?5*(O+1):0,"natural DP":j?x*O:0,"natural refinement cells":j?Math.max(x-1,1)*(F+2)*7:0,"sorted row scratch":a.includes(`maximum-breaks`)||w&&(k||i.naturalBreaksMode===`monotone`)?b:0,"exact partitions":w&&k?(x+1)*(b+1):0,"exact scores":w&&k?(P?x+1:2)*(b+1):0,"exact costs":w&&P?b*b:0,"monotone partitions":w&&i.naturalBreaksMode===`monotone`?(x+1)*(b+1):0,"monotone tasks":w&&i.naturalBreaksMode===`monotone`?b*8:0,"monotone task dispatch":w&&i.naturalBreaksMode===`monotone`?3:0});let I=i.values.length,L=i.values.format===`float32`?void 0:i.values.format,R=(i,a,o)=>{if(!(a instanceof f)&&a.format===o)return a;let s=l(t,`${n}-${i}`,o,a.length),c=0;for(let[l,u]of r(a).entries()){if(u.length>0){let r=t.createDataView(s.buffer,{format:o,length:u.length,byteOffset:c*4});_.push(e(t,{id:`${n}-${i}-${l}`,operation:X,variant:`pack-convert`,invocationCount:u.length,bindings:[{name:`source`,view:u,type:u.format===`float32`?`f32`:u.format===`uint32`?`u32`:`i32`,access:`read`},{name:`destination`,view:r,type:o===`float32`?`f32`:`u32`,access:`read_write`}],body:`destination[destinationOffset + index] = ${o===`float32`?`f32`:`u32`}(source[sourceOffset + index]);`}))}c+=u.length}return s},z=v||L?R(`float-values`,i.values,`float32`):void 0,B=i.mask&&(v||L)?R(`packed-mask`,i.mask,`uint32`):void 0,V=L?z:i.values,H=L?B:i.mask,U=r(V),ee=H?r(H):void 0,W=e=>a.includes(e),G=W(`quantile`)||W(`box-plot`),K=W(`standard-deviation`),te=W(`head-tail`),ne=g-1,re=(e,r)=>l(t,`${n}-${e}`,`uint32`,r),oe=(e,r)=>l(t,`${n}-${e}`,`float32`,r),q=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),J=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),se=H?`rowMask[rowMaskOffset + index] != 0u`:`true`,Y=re(`state`,6),ue=re(`extremes`,4),de=oe(`edges`,g+1),fe=re(`edge-state`,1),Z=G?re(`quantile-gate`,1):void 0,pe=G?oe(`quantile-parameters`,4+s):void 0,me=G?oe(`quantile-values`,s):void 0,he=K||te,$=+!!te+1,be=he?oe(`moments`,$*6):void 0,xe=he?oe(`moment-partials`,N(I)*6):void 0,Se=te?re(`head-tail`,ve+Math.max(ne,1)):void 0,Ce=te?l(t,`${n}-head-tail-dispatch`,`uint32`,6,c.STORAGE|c.INDIRECT):void 0,we=[N(I)*256,256].map(e=>u(X,e,256,t.device.limits.maxComputeWorkgroupsPerDimension)).flatMap(e=>[e.x,e.y,e.z]),Te=e=>[0,1].map(t=>`
  dispatch[dispatchOffset + ${t*3}u] = select(0u, getChunkParameter(${36+t*3}u), ${e});
  dispatch[dispatchOffset + ${t*3+1}u] = getChunkParameter(${37+t*3}u);
  dispatch[dispatchOffset + ${t*3+2}u] = getChunkParameter(${38+t*3}u);`).join(`
`),Ee=a.reduce((e,t)=>e|1<<Q[t],0),De=`state[stateOffset + ${T.method}u]`,Oe=`state[stateOffset + ${T.classCount}u]`,ke=[q(`params`,d,`f32`),J(`state`,Y),J(`extremes`,ue),J(`edgeState`,fe),...Z&&pe?[J(`quantileGate`,Z),J(`quantileParameters`,pe,`f32`)]:[],...Se?[J(`headTail`,Se)]:[],...Ce?[J(`dispatch`,Ce)]:[]];_.push(e(t,{id:`${n}-prepare`,operation:X,variant:`prepare`,parameterValues:we,bindings:ke,invocationCount:1,declarations:`const COMPILED_METHODS: u32 = ${Ee}u;
const MAXIMUM_CLASS_COUNT: u32 = ${g}u;
const QUANTILE_COUNT: u32 = ${s}u;
const HEADER: u32 = 8u;`,body:`let methodValue = params[paramsOffset];
  var method = ${_e}u;
  if (methodValue >= 0.0 && methodValue < 32.0 && floor(methodValue) == methodValue) {
    let code = u32(methodValue);
    if ((COMPILED_METHODS & (1u << code)) != 0u) {
      method = code;
    }
  }
  let classValue = params[paramsOffset + 1u];
  var classCount = 1u;
  if (classValue >= 1.0) {
    classCount = u32(min(classValue, f32(MAXIMUM_CLASS_COUNT)));
  }
  if (method == ${Q[`box-plot`]}u) {
    classCount = 6u;
  }
  state[stateOffset + ${T.method}u] = method;
  state[stateOffset + ${T.classCount}u] = classCount;
  state[stateOffset + ${T.layerCursor}u] = 0u;
  state[stateOffset + ${T.refinementCursor}u] = 0u;
  state[stateOffset + ${T.refinementMoved}u] = 0u;
  state[stateOffset + ${T.refinementDone}u] = 0u;
  extremes[extremesOffset + ${E.minimumKey}u] = 0xffffffffu;
  extremes[extremesOffset + ${E.maximumKey}u] = 0u;
  extremes[extremesOffset + ${E.validCount}u] = 0u;
  extremes[extremesOffset + ${E.finiteCount}u] = 0u;
  edgeState[edgeStateOffset] = 0u;
  ${Z?`quantileGate[quantileGateOffset] = select(0u, 1u, method == ${Q.quantile}u || method == ${Q[`box-plot`]}u);
  quantileParameters[quantileParametersOffset] = ${ge}.0;
  quantileParameters[quantileParametersOffset + 1u] = 0.0;
  quantileParameters[quantileParametersOffset + 2u] = 1.0;
  quantileParameters[quantileParametersOffset + 3u] = 0.0;
  for (var slot = 0u; slot < QUANTILE_COUNT; slot = slot + 1u) {
    quantileParameters[quantileParametersOffset + 4u + slot] = params[paramsOffset + HEADER + slot];
  }`:``}
  ${Se?`headTail[headTailOffset + ${D.done}u] = 0u;
  headTail[headTailOffset + ${D.breakCount}u] = 0u;
  headTail[headTailOffset + ${D.thresholdKey}u] = 0u;
  headTail[headTailOffset + ${D.previousCount}u] = 0u;
  ${Te(`method == ${Q[`head-tail`]}u`)}`:``}`}));for(let[r,i]of U.entries()){if(i.length===0)continue;let a=ee?.[r],o=a?[q(`rowMask`,a,`u32`)]:[],s=i.length;_.push(e(t,{id:U.length===1?`${n}-extremes`:`${n}-extremes-chunk-${r}`,operation:X,variant:`extremes`,bindings:[q(`values`,i,`f32`),...o,J(`extremes`,ue,`atomic<u32>`)],invocationCount:s,declarations:p,body:`let value = values[valuesOffset + index];
  if (isNanBits(value) || !(${se})) {
    return;
  }
  atomicAdd(&extremes[extremesOffset + ${E.validCount}u], 1u);
  if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
    return;
  }
  let key = getOrderedKey(value);
  atomicAdd(&extremes[extremesOffset + ${E.finiteCount}u], 1u);
  atomicMin(&extremes[extremesOffset + ${E.minimumKey}u], key);
  atomicMax(&extremes[extremesOffset + ${E.maximumKey}u], key);`}))}if(Z&&pe&&me&&_.push(...m(t,{id:`${n}-quantiles`,operation:X,values:V,mask:H,parameters:pe,quantileCount:s,output:{quantiles:me,validCount:re(`quantile-valid-count`,1)},gate:{view:Z,value:1}})),be&&xe&&K&&_.push(...M(t,{id:`${n}-moments`,operation:X,values:z,mask:B,state:Y,partials:xe,moments:be,slot:$-1,gate:{kind:`method`,methodCode:Q[`standard-deviation`]}})),be&&xe&&Se&&Ce)for(let r=0;r<ne;r++){let i=`${n}-head-tail-${r}`;_.push(...M(t,{id:i,operation:X,values:z,mask:B,state:Y,partials:xe,moments:be,slot:0,gate:{kind:`head-tail`,methodCode:Q[`head-tail`],headTail:Se},indirectGates:{tile:A(`${i}-tile`,Ce,0),merge:A(`${i}-merge`,Ce,1)}})),_.push(e(t,{id:`${i}-split`,operation:X,variant:`head-tail-split`,parameterValues:we,bindings:[q(`params`,d,`f32`),q(`state`,Y,`u32`),q(`moments`,be,`f32`),J(`headTail`,Se),J(`dispatch`,Ce)],invocationCount:1,declarations:p,body:ye(De,Oe,Te(`false`))}))}if(W(`maximum-breaks`)&&_.push(...this.getMaximumBreaksNodes(t,{values:V,state:Y,extremes:ue,edges:de,edgeState:fe})),W(`natural-breaks`)){let e={id:n,operation:X,state:Y,extremes:ue,edges:de,edgeState:fe,maximumClassCount:g,binCount:this.naturalBreaksBinCount};if(i.naturalBreaksMode===`exact`||i.naturalBreaksMode===`monotone`){let n=i.naturalBreaksMode===`exact`?ce:le;_.push(...n(t,{...e,values:V,mask:H,exactStorage:i.naturalBreaksExactStorage}))}else _.push(...ae(t,{...e,values:V,mask:H}))}let Ae=[q(`params`,d,`f32`),q(`state`,Y,`u32`),q(`extremes`,ue,`u32`),...me?[q(`quantileValues`,me,`f32`)]:[],...K&&be?[q(`moments`,be,`f32`)]:[],...Se?[q(`headTail`,Se,`u32`)]:[],J(`edges`,de,`f32`)];if(_.push(e(t,{id:`${n}-finish`,operation:X,variant:`finish`,bindings:[...Ae,J(`edgeState`,fe)],invocationCount:1,declarations:`${p}
const HEADER: u32 = 8u;`,body:this.getFinishBody({methodRead:De,classCountRead:Oe,hasQuantiles:!!me,momentsSlot:K?$-1:-1,hasHeadTail:!!Se})})),_.push(e(t,{id:`${n}-publish`,operation:X,variant:`publish`,bindings:[q(`state`,Y,`u32`),q(`extremes`,ue,`u32`),q(`edges`,de,`f32`),q(`edgeState`,fe,`u32`),J(`breaksOut`,h.breaks,`f32`),J(`classCountOut`,h.classCount)],invocationCount:1,declarations:`${p}
const MAXIMUM_CLASS_COUNT: u32 = ${g}u;`,body:`let method = ${De};
  let finiteCount = extremes[extremesOffset + ${E.finiteCount}u];
  let minimum = decodeOrderedKey(extremes[extremesOffset + ${E.minimumKey}u]);
  let maximum = decodeOrderedKey(extremes[extremesOffset + ${E.maximumKey}u]);
  var classes = edgeState[edgeStateOffset];
  let isCustom = method == ${Q.custom}u;
  if (method == ${_e}u) {
    classes = 0u;
  } else if (!isCustom && finiteCount == 0u) {
    classes = 0u;
  } else if (!isCustom && minimum == maximum) {
    classes = 1u;
  }
  classes = min(classes, MAXIMUM_CLASS_COUNT);
  // A NaN edge (a hand-written parameter view with a NaN probability, hinge, interval or custom
  // edge) makes the classification undefined: report 0 classes and NaN edges, the no-data result,
  // instead of clamping NaN to an end edge and assigning a silent wrong class.
  for (var edge = 0u; edge <= classes; edge = edge + 1u) {
    let isComputedEnd = !isCustom && (edge == 0u || edge == classes);
    if (!isComputedEnd && isNanBits(edges[edgesOffset + edge])) {
      classes = 0u;
    }
  }
  let nan = getNaN();
  for (var edge = 0u; edge <= MAXIMUM_CLASS_COUNT; edge = edge + 1u) {
    var value = nan;
    if (classes > 0u && edge <= classes) {
      if (isCustom) {
        value = edges[edgesOffset + edge];
      } else if (edge == 0u) {
        value = minimum;
      } else if (edge == classes) {
        value = maximum;
      } else {
        value = clamp(edges[edgesOffset + edge], minimum, maximum);
      }
    }
    breaksOut[breaksOutOffset + edge] = value;
  }
  classCountOut[classCountOutOffset] = classes;`})),h.classCounts){_.push(e(t,{id:`${n}-class-counts-clear`,operation:X,variant:`class-counts-clear`,bindings:[J(`classCounts`,h.classCounts)],invocationCount:g,body:`classCounts[classCountsOffset + index] = 0u;`}));for(let[r,i]of U.entries()){if(i.length===0)continue;let a=ee?.[r],o=a?[q(`rowMask`,a,`u32`)]:[],s=i.length;_.push(e(t,{id:U.length===1?`${n}-class-counts`:`${n}-class-counts-chunk-${r}`,operation:X,variant:`class-counts`,bindings:[q(`values`,i,`f32`),...o,q(`breaks`,h.breaks,`f32`),q(`classCountIn`,h.classCount,`u32`),J(`classCounts`,h.classCounts,`atomic<u32>`)],invocationCount:s,declarations:p,body:`let value = values[valuesOffset + index];
  let classes = classCountIn[classCountInOffset];
  if (classes == 0u || isNanBits(value) || !(${se})) {
    return;
  }
  let key = getOrderedKey(value);
  // Binary search for the number of inner edges e[1..classes-1] whose key is at or below key.
  var low = 1u;
  var high = classes;
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (getOrderedKey(breaks[breaksOffset + middle]) <= key) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  atomicAdd(&classCounts[classCountsOffset + low - 1u], 1u);`}))}}return _}getFinishBody(e){let{methodRead:t,classCountRead:n,hasQuantiles:r,momentsSlot:i,hasHeadTail:a}=e,o=`extremes[extremesOffset + ${E.minimumKey}u]`,s=`extremes[extremesOffset + ${E.maximumKey}u]`,c=i*6;return`let method = ${t};
  let classCount = ${n};
  let minimum = decodeOrderedKey(${o});
  let maximum = decodeOrderedKey(${s});
  if (method == ${Q[`equal-interval`]}u) {
    let width = (maximum - minimum) / f32(classCount);
    for (var edge = 1u; edge < classCount; edge = edge + 1u) {
      edges[edgesOffset + edge] = minimum + width * f32(edge);
    }
    edgeState[edgeStateOffset] = classCount;
  }
  ${r?`if (method == ${Q.quantile}u) {
    for (var edge = 1u; edge < classCount; edge = edge + 1u) {
      edges[edgesOffset + edge] = quantileValues[quantileValuesOffset + edge - 1u];
    }
    edgeState[edgeStateOffset] = classCount;
  }
  if (method == ${Q[`box-plot`]}u) {
    let hinge = params[paramsOffset + 4u];
    let lowerQuartile = quantileValues[quantileValuesOffset];
    let median = quantileValues[quantileValuesOffset + 1u];
    let upperQuartile = quantileValues[quantileValuesOffset + 2u];
    let spread = upperQuartile - lowerQuartile;
    edges[edgesOffset + 1u] = lowerQuartile - hinge * spread;
    edges[edgesOffset + 2u] = lowerQuartile;
    edges[edgesOffset + 3u] = median;
    edges[edgesOffset + 4u] = upperQuartile;
    edges[edgesOffset + 5u] = upperQuartile + hinge * spread;
    edgeState[edgeStateOffset] = 6u;
  }`:``}
  ${i>=0?`if (method == ${Q[`standard-deviation`]}u) {
    let count = moments[momentsOffset + ${c}u];
    let mean = moments[momentsOffset + ${c+1}u];
    // Sample standard deviation (n - 1), as mapclassify StdMean and R classInt 'sd'.
    let deviation = sqrt(moments[momentsOffset + ${c+2}u] / max(count - 1.0, 1.0));
    let interval = params[paramsOffset + 2u] * deviation;
    let center = f32(classCount - 2u) * 0.5;
    for (var edge = 1u; edge < classCount; edge = edge + 1u) {
      edges[edgesOffset + edge] = mean + (f32(edge - 1u) - center) * interval;
    }
    edgeState[edgeStateOffset] = classCount;
  }`:``}
  ${a?`if (method == ${Q[`head-tail`]}u) {
    let breakCount = headTail[headTailOffset + ${D.breakCount}u];
    for (var edge = 1u; edge <= breakCount; edge = edge + 1u) {
      edges[edgesOffset + edge] = bitcast<f32>(
        headTail[headTailOffset + 3u + edge]
      );
    }
    edgeState[edgeStateOffset] = breakCount + 1u;
  }`:``}
  if (method == ${Q.custom}u) {
    for (var edge = 0u; edge <= classCount; edge = edge + 1u) {
      edges[edgesOffset + edge] = params[paramsOffset + HEADER + edge];
    }
    edgeState[edgeStateOffset] = classCount;
  }`}getMaximumBreaksNodes(t,n){let{id:r,props:i}=this,{values:a}=n,{maximumClassCount:o}=i,s=i.mask,c=a.length,u=`${r}-maximum-breaks`,f=e=>l(t,`${u}-${e}`,`uint32`,c),m=f(`keys`),h=f(`rows`),g=f(`sorted-keys`),_=f(`sorted-rows`),v=f(`gap-keys`),y=f(`gap-positions`),b=f(`sorted-gap-keys`),x=f(`sorted-gap-positions`),S=`state[stateOffset + ${T.method}u] != ${Q[`maximum-breaks`]}u`,C=[];return C.push(e(t,{id:`${u}-keys`,operation:X,variant:`maximum-breaks-keys`,bindings:[{name:`values`,view:a,type:`f32`,access:`read`},...s?[{name:`rowMask`,view:s,type:`u32`,access:`read`}]:[],{name:`state`,view:n.state,type:`u32`,access:`read`},{name:`keys`,view:m,type:`u32`,access:`read_write`},{name:`rows`,view:h,type:`u32`,access:`read_write`}],invocationCount:c,declarations:p,body:`if (${S}) {
    return;
  }
  let value = values[valuesOffset + index];
  var include = (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
  ${s?`include = include && rowMask[rowMaskOffset + index] != 0u;`:``}
  // Excluded rows take the NaN key 0xffffffff, which no finite value has, and sort last.
  keys[keysOffset + index] = select(0xffffffffu, getOrderedKey(value), include);
  rows[rowsOffset + index] = index;`})),C.push(...new d({id:`${u}-sort`,keys:m,values:h,outputKeys:g,outputValues:_,algorithm:`radix`}).getCommandNodes(t)),C.push(e(t,{id:`${u}-gaps`,operation:X,variant:`maximum-breaks-gaps`,bindings:[{name:`state`,view:n.state,type:`u32`,access:`read`},{name:`extremes`,view:n.extremes,type:`u32`,access:`read`},{name:`sortedKeys`,view:g,type:`u32`,access:`read`},{name:`gapKeys`,view:v,type:`u32`,access:`read_write`},{name:`gapPositions`,view:y,type:`u32`,access:`read_write`}],invocationCount:c,declarations:p,body:`if (${S}) {
    return;
  }
  let finiteCount = extremes[extremesOffset + ${E.finiteCount}u];
  var gapKey = 0xffffffffu;
  if (index > 0u && index < finiteCount) {
    let lowerKey = sortedKeys[sortedKeysOffset + index - 1u];
    let upperKey = sortedKeys[sortedKeysOffset + index];
    if (upperKey != lowerKey) {
      let gap = decodeOrderedKey(upperKey) - decodeOrderedKey(lowerKey);
      // Inverted keys sort the largest gap first; the stable sort keeps lower positions first.
      gapKey = ~getOrderedKey(gap);
    }
  }
  gapKeys[gapKeysOffset + index] = gapKey;
  gapPositions[gapPositionsOffset + index] = index;`})),C.push(...new d({id:`${u}-gap-sort`,keys:v,values:y,outputKeys:b,outputValues:x,algorithm:`radix`}).getCommandNodes(t)),C.push(e(t,{id:`${u}-finish`,operation:X,variant:`maximum-breaks-finish`,bindings:[{name:`state`,view:n.state,type:`u32`,access:`read`},{name:`sortedKeys`,view:g,type:`u32`,access:`read`},{name:`sortedGapKeys`,view:b,type:`u32`,access:`read`},{name:`sortedGapPositions`,view:x,type:`u32`,access:`read`},{name:`edges`,view:n.edges,type:`f32`,access:`read_write`},{name:`edgeState`,view:n.edgeState,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`${p}
const MAXIMUM_CLASS_COUNT: u32 = ${o}u;`,body:`if (${S}) {
    return;
  }
  let classCount = state[stateOffset + ${T.classCount}u];
  var midpoints: array<f32, ${o}>;
  var found = 0u;
  for (var rank = 0u; rank + 1u < classCount && rank < arrayLength(&sortedGapKeys) - sortedGapKeysOffset; rank = rank + 1u) {
    if (sortedGapKeys[sortedGapKeysOffset + rank] == 0xffffffffu) {
      break;
    }
    let position = sortedGapPositions[sortedGapPositionsOffset + rank];
    let lower = decodeOrderedKey(sortedKeys[sortedKeysOffset + position - 1u]);
    let upper = decodeOrderedKey(sortedKeys[sortedKeysOffset + position]);
    // (upper - lower) * 0.5 is exact, so a fused multiply-add rounds the same way.
    let midpoint = lower + (upper - lower) * 0.5;
    // Insertion sort keeps the midpoints ascending.
    var slot = found;
    while (slot > 0u && midpoints[slot - 1u] > midpoint) {
      midpoints[slot] = midpoints[slot - 1u];
      slot = slot - 1u;
    }
    midpoints[slot] = midpoint;
    found = found + 1u;
  }
  for (var edge = 0u; edge < found; edge = edge + 1u) {
    edges[edgesOffset + edge + 1u] = midpoints[edge];
  }
  edgeState[edgeStateOffset] = found + 1u;`})),C}};function ye(e,t,n){let r=e=>`headTail[headTailOffset + ${D[e]}u]`,i=r(`done`);return`if (${e} != ${Q[`head-tail`]}u || ${i} != 0u) {
    return;
  }
  let round = ${r(`breakCount`)};
  let count = moments[momentsOffset];
  let mean = moments[momentsOffset + 1u];
  let minimum = moments[momentsOffset + 3u];
  let maximum = moments[momentsOffset + 4u];
  var ratio = params[paramsOffset + 3u];
  if (!(ratio >= 0.0)) {
    ratio = 0.4;
  }
  // At most k - 1 breaks; stop on an empty or single-valued head.
  var stop = round + 1u >= ${t} || count < 1.0 || !(minimum < maximum);
  if (!stop && round > 0u) {
    let previousCount = bitcast<f32>(${r(`previousCount`)});
    // The head kept from the previous split must be a minority (the 40% rule).
    stop = count > ratio * previousCount || count <= 1.0;
  }
  if (stop) {
    ${i} = 1u;
    ${n}
    return;
  }
  headTail[headTailOffset + ${ve}u + round] = bitcast<u32>(mean);
  ${r(`breakCount`)} = round + 1u;
  ${r(`thresholdKey`)} = getOrderedKey(mean);
  ${r(`previousCount`)} = bitcast<u32>(count);`}export{F as a,B as i,ie as n,N as o,z as r,$ as t};