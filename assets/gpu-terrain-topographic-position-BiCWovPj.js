import{c as e,o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as n,S as r,u as i,x as a}from"./gpu-contributor-utils-BIwLQON5.js";import{c as o,l as s,n as c,r as l,u}from"./raster-band-utils-BcTxc8_Y.js";import{a as d,i as f,r as p,t as m}from"./terrain-summed-area-table-BJldJnOP.js";function h(e){return i(e,4)}var g=class{id;props;requiredHalo;constructor(e){this.id=e.id??`terrain-topographic-position`,this.props=e;let{id:t}=this,{output:r}=e,i=u(t,e.grid),{scales:a}=e;if(!Array.isArray(a)||a.length<1||a.length>64)throw Error(`${t} scales must contain 1 to 64 entries`);for(let e of a){let n=e.innerRadius??0;if(!Number.isSafeInteger(e.radius)||e.radius<1||!Number.isSafeInteger(n)||n<0||n>=e.radius)throw Error(`${t} scales need integer radius >= 1 and 0 <= innerRadius < radius`)}let o=e.quantum??.00390625;if(!(o>0)||!Number.isFinite(o)||2**Math.round(Math.log2(o))!==o)throw Error(`${t} quantum must be a positive power of two`);if(this.requiredHalo=Math.max(...a.map(e=>e.radius)),!r.topographicPositionIndex&&!r.deviationFromMean&&!r.maximumDeviation&&!r.maximumDeviationRadius&&!r.outputValidity)throw Error(`${t} requires at least one output`);for(let[e,o,s]of[[`topographicPositionIndex`,r.topographicPositionIndex,i*a.length],[`deviationFromMean`,r.deviationFromMean,i*a.length],[`maximumDeviation`,r.maximumDeviation,i]])if(o&&(n(o,[`float32`],`${t} output.${e}`),o.length!==s))throw Error(`${t} output.${e} must contain ${s} values`);for(let[e,a]of[[`maximumDeviationRadius`,r.maximumDeviationRadius],[`outputValidity`,r.outputValidity]])if(a&&(n(a,[`uint32`],`${t} output.${e}`),a.length!==i))throw Error(`${t} output.${e} must contain one value per pixel`);s(t,[r.topographicPositionIndex,r.deviationFromMean,r.maximumDeviation,r.maximumDeviationRadius,r.outputValidity],c(e.elevation))}getCommandNodes(n){let{id:i,props:s}=this,{output:c,scales:u}=s,{width:g,height:_}=s.grid;o(i,n,s.elevation,[]),a(i,n,[c.topographicPositionIndex,c.deviationFromMean,c.maximumDeviation,c.maximumDeviationRadius,c.outputValidity]);let v=g*_;r(i,`grid cells`,v,d(n.device)),(c.topographicPositionIndex||c.deviationFromMean)&&r(i,`grid cells x scales`,v*s.scales.length,h(n.device));let y=s.quantum??.00390625,b=l(n,i,s.elevation,s.grid,!0),x=f(n,{id:i,width:g,height:_,elevation:b.band,quantum:y}),S=[{name:`lowTable`,view:x.lowTable,type:`u32`,access:`read`},{name:`highTable`,view:x.highTable,type:`u32`,access:`read`}],C=[[`tpiValues`,c.topographicPositionIndex,`f32`],[`deviationValues`,c.deviationFromMean,`f32`],[`maximumDeviationValues`,c.maximumDeviation,`f32`],[`maximumRadiusValues`,c.maximumDeviationRadius,`u32`],[`validityValues`,c.outputValidity,`u32`]];for(let[e,t,n]of C)t&&S.push({name:e,view:t,type:n,access:`read_write`});let w=u.map(e=>`vec2<i32>(${e.radius}i, ${e.innerRadius??0}i)`).join(`, `),T=t(n,{id:`${i}-evaluate`,operation:`GPUTerrainTopographicPosition`,variant:`evaluate`,bindings:S,invocationCount:v,declarations:`const WIDTH: u32 = ${g}u;
const HEIGHT: u32 = ${_}u;
const PIXEL_COUNT: u32 = ${v}u;
const SCALE_COUNT: u32 = ${u.length}u;
const QUANTUM: f32 = ${e(y)};
const INVERSE_QUANTUM: f32 = ${e(1/y)};
const SCALES = array<vec2<i32>, ${u.length}>(${w});
${p}
${m}
fn readClippedBox(column: i32, row: i32, radius: i32) -> TerrainBoxSums {
  return readBoxSums(
    max(column - radius, 0i),
    min(column + radius, i32(WIDTH) - 1i),
    max(row - radius, 0i),
    min(row + radius, i32(HEIGHT) - 1i)
  );
}
// Σ(q - q_c) over a box, exact modulo 2^64.
fn getCentredSum(box: TerrainBoxSums, magnitude: u32, isNegative: bool) -> vec2<u32> {
  var centreTotal = multiply64By32(vec2<u32>(box.count, 0u), magnitude);
  if (isNegative) {
    centreTotal = negate64(centreTotal);
  }
  return subtract64(box.sum, centreTotal);
}`,body:`let column = i32(index % WIDTH);
  let row = i32(index / WIDTH);
  // A single-cell box returns the centre's quantized elevation and validity exactly.
  let centre = readBoxSums(column, column, row, row);
  let isValid = centre.count == 1u;
  let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  let quantized = bitcast<i32>(centre.sum.x);
  let magnitude = u32(abs(quantized));
  let isNegative = quantized < 0i;
  let centreSquare = multiplyWide32(magnitude, magnitude);
  var maximumDeviation = invalidValue;
  var maximumAbsolute = -1.0;
  var maximumRadius = 0u;
  for (var scaleIndex = 0u; scaleIndex < SCALE_COUNT; scaleIndex++) {
    let scale = SCALES[scaleIndex];
    let outer = readClippedBox(column, row, scale.x);
    // Population statistics relative to the centre: d1 = Σ(q - q_c), d2 = Σ(q - q_c)².
    let centredSum = getCentredSum(outer, magnitude, isNegative);
    var twiceCentreSum = multiply64By32(outer.sum, magnitude);
    twiceCentreSum = add64(twiceCentreSum, twiceCentreSum);
    if (isNegative) {
      twiceCentreSum = negate64(twiceCentreSum);
    }
    let centredSquare = add64(
      subtract64(outer.square, twiceCentreSum),
      multiply64By32(centreSquare, outer.count)
    );
    let count = f32(outer.count);
    let meanOffset = signed64ToFloat(centredSum) / count;
    // Review C-16: d2 / n - (d1 / n)^2 in f32 cancels when the centre is far from the window mean.
    // Re-centre exactly on the integer s nearest that mean: sum (d - s)^2 = d2 - 2 s d1 + n s^2 in
    // modular 64-bit integers, so only the residual mean (|r| about 0.5 quanta) is squared in f32.
    let shiftValue = round(meanOffset);
    let shiftMagnitude = u32(abs(shiftValue));
    let shiftNegative = shiftValue < 0.0;
    var shiftedCross = multiply64By32(centredSum, shiftMagnitude);
    shiftedCross = add64(shiftedCross, shiftedCross);
    var shiftTotal = multiply64By32(vec2<u32>(outer.count, 0u), shiftMagnitude);
    if (shiftNegative) {
      shiftedCross = negate64(shiftedCross);
      shiftTotal = negate64(shiftTotal);
    }
    let shiftedSquare = add64(
      subtract64(centredSquare, shiftedCross),
      multiply64By32(multiplyWide32(shiftMagnitude, shiftMagnitude), outer.count)
    );
    let residualMean = signed64ToFloat(subtract64(centredSum, shiftTotal)) / count;
    let variance = unsigned64ToFloat(shiftedSquare) / count - residualMean * residualMean;
    var deviation = 0.0;
    if (variance > 0.0) {
      deviation = -meanOffset / sqrt(variance);
    }
    if (!isValid) {
      deviation = invalidValue;
    }
    ${c.deviationFromMean?`deviationValues[deviationValuesOffset + scaleIndex * PIXEL_COUNT + index] = deviation;`:``}
    if (isValid && abs(deviation) > maximumAbsolute) {
      maximumAbsolute = abs(deviation);
      maximumDeviation = deviation;
      maximumRadius = u32(scale.x);
    }
    ${c.topographicPositionIndex?`let inner = readClippedBox(column, row, scale.y);
    let annulus = TerrainBoxSums(
      subtract64(outer.sum, inner.sum),
      subtract64(outer.square, inner.square),
      outer.count - inner.count
    );
    var tpi = invalidValue;
    if (isValid && annulus.count > 0u) {
      tpi = -signed64ToFloat(getCentredSum(annulus, magnitude, isNegative)) / f32(annulus.count) * QUANTUM;
    }
    tpiValues[tpiValuesOffset + scaleIndex * PIXEL_COUNT + index] = tpi;`:``}
  }
  ${c.maximumDeviation?`maximumDeviationValues[maximumDeviationValuesOffset + index] = maximumDeviation;`:``}
  ${c.maximumDeviationRadius?`maximumRadiusValues[maximumRadiusValuesOffset + index] = maximumRadius;`:``}
  ${c.outputValidity?`validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);`:``}`});return[...b.nodes,...x.nodes,T]}};export{g as t};