import{o as e}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as t}from"./kernel-C3tfQOn4.js";import{D as n,E as r,T as i,f as a,x as o,y as s}from"./gpu-contributor-utils-BIwLQON5.js";import{a as c}from"./graph-chunk-dispatch-zBt8k2-o.js";import{c as l,r as u}from"./wgsl-common-DfyXiYB0.js";import{t as d}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{t as f}from"./uint64-wgsl-DCuW6-R6.js";import{t as p}from"./regression-output-dispatches-KKVTWkHP.js";import{i as m}from"./spatial-likelihood-allocations-BGEHbqkB.js";import{r as h}from"./rain-BPTWkQeX.js";function g(e,t=new Float32Array(4)){let n=Math.fround(e.miscoverage),r=Math.fround(e.selfWeight??1);if(!(n>0&&n<1))throw Error(`getGPUConformalPredictionParameterValues miscoverage must be in (0, 1)`);if(!(Number.isFinite(r)&&r>=0))throw Error(`getGPUConformalPredictionParameterValues selfWeight must be a non-negative finite number`);if(t.length<4)throw Error(`getGPUConformalPredictionParameterValues target needs 4 elements`);return t.set([n,r,0,0]),t}function _(e,t,i,o,s){a(e,`foldCount`,o,1,64);let c=t.observations.length;if(a(e,`calibration rows`,c,1),r(t.observations,[`float32`],`${e} calibration.observations`),r(t.predictions,[`float32`],`${e} calibration.predictions`),n(e,`calibration.predictions`,t.predictions,o*c),o>1&&!t.folds)throw Error(`${e} calibration.folds is required when foldCount is more than one`);for(let[i,a,o]of[[`folds`,t.folds,`uint32`],[`scales`,t.scales,`float32`],[`mask`,t.mask,`uint32`]])a&&(r(a,[o],`${e} calibration.${i}`),n(e,`calibration.${i}`,a,c));return r(i.predictions,[`float32`],`${e} query.predictions`),n(e,`query.predictions`,i.predictions,o*s),i.scales&&(r(i.scales,[`float32`],`${e} query.scales`),n(e,`query.scales`,i.scales,s)),c}function v(e,t){return[e.observations,e.predictions,e.folds,e.scales,e.mask,t.predictions,t.scales].filter(e=>e!==void 0)}function y(n,r,i,a){let o=i.observations.length,s=t(n,`${r}-fold-keys`,`uint32`,o),c=t(n,`${r}-score-keys`,`uint32`,o),u=t(n,`${r}-indices`,`uint32`,o);return{nodes:[e(n,{id:`${r}-scores`,operation:`GPUConformalPredictionIntervals`,variant:`scores`,invocationCount:o,bindings:[{name:`observations`,view:i.observations,type:`f32`,access:`read`},{name:`predictions`,view:i.predictions,type:`f32`,access:`read`},...i.folds?[{name:`folds`,view:i.folds,type:`u32`,access:`read`}]:[],...i.scales?[{name:`scales`,view:i.scales,type:`f32`,access:`read`}]:[],...i.mask?[{name:`mask`,view:i.mask,type:`u32`,access:`read`}]:[],{name:`foldKeys`,view:s,type:`u32`,access:`read_write`},{name:`scoreKeys`,view:c,type:`u32`,access:`read_write`},{name:`indices`,view:u,type:`u32`,access:`read_write`}],declarations:l,body:`let fold = ${i.folds?`folds[foldsOffset + index]`:`0u`};
  var valid = fold < ${a}u${i.mask?` && mask[maskOffset + index] != 0u`:``};
  let observation = observations[observationsOffset + index];
  let prediction = predictions[predictionsOffset + min(fold, ${a-1}u) * ${o}u + index];
  valid = valid && isFiniteBits(observation) && isFiniteBits(prediction);
  var score = abs(observation - prediction);
  ${i.scales?`let scale = scales[scalesOffset + index];
  valid = valid && isFiniteBits(scale) && scale > 0.0;
  score = score / scale;`:``}
  valid = valid && isFiniteBits(score);
  // A non-negative finite float32 sorts as its bit pattern; the score is recovered with bitcast<f32>.
  scoreKeys[scoreKeysOffset + index] = select(0u, bitcast<u32>(score), valid);
  foldKeys[foldKeysOffset + index] = select(${a}u, fold, valid);
  indices[indicesOffset + index] = index;`})],views:{foldKeys:s},scoreKeys:c,indices:u}}var b=4294967294;function x(e,t){let n=`getConformalPredictionRanks`;a(n,`calibrationCount`,e,0,b);let r=Math.fround(t);i(n,r>0&&r<1,`miscoverage must be in (0, 1) after rounding to float32`);let o=new Uint32Array(new Float32Array([r]).buffer)[0],s=o>>>23,c=o&8388607|(s===0?0:8388608),l=s===0?149:150-s,u=e+1,d=Number(BigInt(c)*BigInt(u)>>BigInt(l));return{lowerRank:d,upperRank:u-d}}var S=`${f}
fn getConformalPredictionRanks(calibrationCount: u32, miscoverage: f32) -> vec2<u32> {
  let bits = bitcast<u32>(miscoverage);
  let exponent = bits >> 23u;
  let significand = (bits & 0x7fffffu) | select(0x800000u, 0u, exponent == 0u);
  let shift = select(150u - exponent, 149u, exponent == 0u);
  let size = calibrationCount + 1u;
  let product = multiplyUint32Wide(significand, size);
  let lowerRank = shiftRightUint64(product, shift).y;
  return vec2<u32>(lowerRank, size - lowerRank);
}
`,C=`GPUConformalPredictionIntervals`,w=`
fn getInfinity() -> f32 { var bits = 0x7f800000u; return bitcast<f32>(bits); }
`,T=`
fn getOrderedKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
fn getKeyedValue(key: u32) -> f32 {
  return bitcast<f32>(select(~key, key & 0x7fffffffu, (key & 0x80000000u) != 0u));
}
`,E=class{props;id;foldCount;queryRows;constructor(e){this.props=e,this.id=e.id??`conformal-prediction-intervals`;let{id:t}=this;this.foldCount=e.foldCount??1,a(t,`foldCount`,this.foldCount,1,64),r(e.query.predictions,[`float32`],`${t} query.predictions`),a(t,`query rows`,e.query.predictions.length/this.foldCount,1),this.queryRows=e.query.predictions.length/this.foldCount,_(t,e.calibration,e.query,this.foldCount,this.queryRows),a(t,`calibration rows`,e.calibration.observations.length,1,b),D(t,e.parameters,e.output,this.queryRows),p(this.id,e.output,[[`intervals`,`status`]]),s(t,Object.values(e.output),[...v(e.calibration,e.query),e.parameters])}getCommandNodes(n){let{id:r,props:i,foldCount:a,queryRows:s}=this,f=i.calibration.observations.length;c(n,r,{...Object.fromEntries(v(i.calibration,i.query).map((e,t)=>[`input ${t}`,e])),parameters:i.parameters,...i.output}),m(n.device,r,{"conformal scores, keys, sorted rows and published sort/scan scratch":f,"conformal fold offsets":a+1}),o(r,n,[...v(i.calibration,i.query),i.parameters,...Object.values(i.output)]);let p=y(n,r,i.calibration,a),h=d(n,{id:`${r}-sort`,operation:C,indices:p.indices,keys:[{view:p.views.foldKeys,keyBits:Math.max(1,Math.ceil(Math.log2(a+1)))},{view:p.scoreKeys,keyBits:32}]}),g=t(n,`${r}-sorted-scores`,`float32`,f),_=t(n,`${r}-fold-offsets`,`uint32`,a+1);return[...p.nodes,...h.nodes,e(n,{id:`${r}-gather`,operation:C,variant:`sorted-scores`,invocationCount:f,bindings:[{name:`scoreKeys`,view:p.scoreKeys,type:`u32`,access:`read`},{name:`sortedIndices`,view:h.sortedIndices,type:`u32`,access:`read`},{name:`sortedScores`,view:g,type:`f32`,access:`read_write`}],body:`sortedScores[sortedScoresOffset + index] = bitcast<f32>(scoreKeys[scoreKeysOffset + sortedIndices[sortedIndicesOffset + index]]);`}),e(n,{id:`${r}-offsets`,operation:C,variant:`fold-offsets`,invocationCount:a+1,bindings:[{name:`sortedFolds`,view:h.sortedPrimaryKeys,type:`u32`,access:`read`},{name:`foldOffsets`,view:_,type:`u32`,access:`read_write`},{name:`calibrationCount`,view:i.output.calibrationCount,type:`u32`,access:`read_write`}],body:`var low = 0u;
  var high = ${f}u;
  for (var step = 0u; step < 32u; step++) {
    if (low >= high) { break; }
    let middle = low + (high - low) / 2u;
    if (sortedFolds[sortedFoldsOffset + middle] < index) { low = middle + 1u; } else { high = middle; }
  }
  foldOffsets[foldOffsetsOffset + index] = low;
  if (index == ${a}u) { calibrationCount[calibrationCountOffset] = low; }`}),e(n,{id:`${r}-intervals`,operation:C,variant:`order-statistics`,invocationCount:s,bindings:[{name:`parameters`,view:i.parameters,type:`f32`,access:`read`},{name:`sortedScores`,view:g,type:`f32`,access:`read`},{name:`foldOffsets`,view:_,type:`u32`,access:`read`},{name:`queryPredictions`,view:i.query.predictions,type:`f32`,access:`read`},...i.query.scales?[{name:`queryScales`,view:i.query.scales,type:`f32`,access:`read`}]:[],{name:`intervals`,view:i.output.intervals,type:`f32`,access:`read_write`},{name:`status`,view:i.output.status,type:`u32`,access:`read_write`}],declarations:`${l}
${u}
${w}
${T}
${S}
fn getShifted(prediction: f32, scale: f32, score: f32, sign: f32) -> f32 {
  return prediction + sign * (scale * score);
}
// Values of one fold at or below the threshold. Upper values rise with the sorted score and
// lower values fall, so each count is one binary search over the fold's sorted segment.
fn countAtOrBelow(fold: u32, threshold: f32, prediction: f32, scale: f32, sign: f32) -> u32 {
  let first = foldOffsets[foldOffsetsOffset + fold];
  let last = foldOffsets[foldOffsetsOffset + fold + 1u];
  var low = first;
  var high = last;
  for (var step = 0u; step < 32u; step++) {
    if (low >= high) { break; }
    let middle = low + (high - low) / 2u;
    let value = getShifted(prediction, scale, sortedScores[sortedScoresOffset + middle], sign);
    // Advance past the values that are at or below the threshold (rising values) or above it (falling values).
    let goRight = select(value > threshold, value <= threshold, sign > 0.0);
    if (goRight) { low = middle + 1u; } else { high = middle; }
  }
  return select(last - low, low - first, sign > 0.0);
}
fn getTotalCount(threshold: f32, query: u32, scale: f32, sign: f32) -> u32 {
  var total = 0u;
  for (var fold = 0u; fold < ${a}u; fold++) {
    total += countAtOrBelow(fold, threshold, queryPredictions[queryPredictionsOffset + fold * ${s}u + query], scale, sign);
  }
  return total;
}
// Smallest candidate value whose count reaches the rank, by bisection over ordered float32 keys.
fn getOrderStatistic(rank: u32, query: u32, scale: f32, sign: f32) -> f32 {
  // Keys of minus and plus infinity: NaN keys outside the range would break the monotone count.
  var low = 0x007fffffu;
  var high = 0xff800000u;
  for (var step = 0u; step < 32u; step++) {
    if (low >= high) { break; }
    let middle = low + (high - low) / 2u;
    if (getTotalCount(getKeyedValue(middle), query, scale, sign) >= rank) { high = middle; } else { low = middle + 1u; }
  }
  return getKeyedValue(low);
}`,body:`let miscoverage = parameters[parametersOffset];
  let count = foldOffsets[foldOffsetsOffset + ${a}u];
  let scale = ${i.query.scales?`queryScales[queryScalesOffset + index]`:`1.0`};
  var valid = isFiniteBits(scale) && scale > 0.0;
  for (var fold = 0u; fold < ${a}u; fold++) {
    valid = valid && isFiniteBits(queryPredictions[queryPredictionsOffset + fold * ${s}u + index]);
  }
  let lowerBase = intervalsOffset + index * 2u;
  var code = 0u;
  var lower = getNaN();
  var upper = getNaN();
  if (!valid) {
    code = 4u;
  } else if (count == 0u) {
    code = 8u;
  } else {
    let ranks = getConformalPredictionRanks(count, miscoverage);
    let lowerRank = ranks.x;
    let upperRank = ranks.y;
    if (upperRank > count) {
      upper = getInfinity();
      code |= 2u;
    } else {
      upper = getOrderStatistic(upperRank, index, scale, 1.0);
    }
    if (lowerRank < 1u) {
      lower = -getInfinity();
      code |= 1u;
    } else {
      lower = getOrderStatistic(lowerRank, index, scale, -1.0);
    }
  }
  intervals[lowerBase] = lower;
  intervals[lowerBase + 1u] = upper;
  status[statusOffset + index] = code;`})]}};function D(e,t,i,a){if(r(t,[`float32`],`${e} parameters`),t.length<4)throw Error(`${e} parameters must have at least 4 values`);r(i.intervals,[`float32`],`${e} output.intervals`),n(e,`output.intervals`,i.intervals,a*2),r(i.status,[`uint32`],`${e} output.status`),n(e,`output.status`,i.status,a)}var O=8,k=.2;function A(e,t,n=540){let r=h(e,t,n),i=Array.from({length:r.count},(e,t)=>t),a=i.map((e,t)=>t%3),o=i.filter((e,t)=>a[t]===0),s=e=>Math.log1p(r.rain[e]),c=e=>{let t=r.positions[e*2],n=r.positions[e*2+1],i=o.filter(t=>t!==e).map(e=>({other:e,distance:Math.hypot(r.positions[e*2]-t,r.positions[e*2+1]-n)})).sort((e,t)=>e.distance-t.distance).slice(0,O),a=i.reduce((e,t)=>e+s(t.other),0)/i.length;return{mean:a,spread:Math.sqrt(i.reduce((e,t)=>e+(s(t.other)-a)**2,0)/i.length)}},l=e=>{let t=i.filter((t,n)=>a[n]===e),n=new Float32Array(t.length*2);return t.forEach((e,t)=>n.set([r.positions[e*2],r.positions[e*2+1]],t*2)),{count:t.length,positions:n,observations:Float32Array.from(t,s),predictions:Float32Array.from(t,e=>c(e).mean),scales:Float32Array.from(t,e=>Math.max(c(e).spread,k))}};return{calibration:l(1),test:l(2),mapWidth:r.mapWidth,panel:r.panel}}function j(e,t){let n=0,r=0,i=0,a=0;for(let o=0;o<t.length;o++){let s=e[o*2],c=e[o*2+1];Number.isNaN(s)||Number.isNaN(c)||(a++,t[o]>=s&&t[o]<=c&&n++,Number.isFinite(s)&&Number.isFinite(c)&&(r+=c-s,i++))}return{coverage:a>0?n/a:NaN,meanWidth:i>0?r/i:NaN,boundedShare:a>0?i/a:0}}function M(e,t){return Float32Array.from({length:t},(t,n)=>{let r=e[n*2+1]-e[n*2];return Number.isFinite(r)?r:NaN})}export{w as a,x as c,g as d,_ as f,E as i,y as l,M as n,D as o,j as r,S as s,A as t,v as u};