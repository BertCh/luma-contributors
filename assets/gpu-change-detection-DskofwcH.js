import{a as e,i as t,n,t as r}from"./raster-time-series-chunks-b-CPab_8L5.js";import{E as i,c as a,f as o,p as s}from"./gpu-contributor-utils-BIwLQON5.js";import{h as c,o as l}from"./raster-band-utils-BcTxc8_Y.js";import{n as u,s as d}from"./wgsl-common-DfyXiYB0.js";import{a as f,i as p,o as m,r as h}from"./raster-irregular-date-axis-B315UxmE.js";function g(e,t=new Float32Array(5)){return l(`getGPUChangeDetectionParameterValues`,t,5),t[0]=e.beforeSlice,t[1]=e.afterSlice,t[2]=e.epsilon??1e-6,t[3]=e.alpha??.05,t[4]=e.splitSlice??1,t}var _=3e-7,v=`
fn quietNan() -> f32 {
  // A runtime value: WGSL rejects constant-expression NaN.
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}
fn isNanValue(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u;
}
// A sample is missing when it is not finite (NaN or an infinity): the rule of every
// raster-time-series and remote-sensing contributor (contributor contract section 7).
fn isMissingSample(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u;
}
`,y=`
fn logGammaPositive(argument: f32) -> f32 {
  var x = argument;
  var shiftProduct = 1.0;
  for (var step = 0u; step < 8u; step++) {
    if (x < 8.0) {
      shiftProduct *= x;
      x += 1.0;
    }
  }
  let inverse = 1.0 / x;
  let inverseSquared = inverse * inverse;
  let series = inverse * (1.0 / 12.0 - inverseSquared * (1.0 / 360.0 - inverseSquared * (1.0 / 1260.0)));
  return (x - 0.5) * log(x) - x + 0.9189385332 + series - log(shiftProduct);
}

fn betaContinuedFraction(a: f32, b: f32, x: f32) -> f32 {
  let tiny = 1e-30;
  let qab = a + b;
  let qap = a + 1.0;
  let qam = a - 1.0;
  var c = 1.0;
  var d = 1.0 - qab * x / qap;
  if (abs(d) < tiny) { d = tiny; }
  d = 1.0 / d;
  var h = d;
  for (var m = 1u; m <= 64u; m++) {
    let mf = f32(m);
    let m2 = 2.0 * mf;
    var aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (abs(d) < tiny) { d = tiny; }
    c = 1.0 + aa / c;
    if (abs(c) < tiny) { c = tiny; }
    d = 1.0 / d;
    h *= d * c;
    aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (abs(d) < tiny) { d = tiny; }
    c = 1.0 + aa / c;
    if (abs(c) < tiny) { c = tiny; }
    d = 1.0 / d;
    let delta = d * c;
    h *= delta;
    if (abs(delta - 1.0) < ${_}) {
      break;
    }
  }
  return h;
}

fn studentTTwoSidedP(t: f32, degreesOfFreedom: f32) -> f32 {
  let t2 = t * t;
  if (t2 == 0.0) {
    return 1.0;
  }
  if (t2 > 1e30) {
    return 0.0;
  }
  let a = 0.5 * degreesOfFreedom;
  let b = 0.5;
  let y = t2 / (degreesOfFreedom + t2);
  let x = degreesOfFreedom / (degreesOfFreedom + t2);
  let logX = -log(1.0 + t2 / degreesOfFreedom);
  let logFront = a * logX + b * log(y)
    - (logGammaPositive(a) + logGammaPositive(b) - logGammaPositive(a + b));
  var incomplete: f32;
  if (x < (a + 1.0) / (a + b + 2.0)) {
    incomplete = exp(logFront) * betaContinuedFraction(a, b, x) / a;
  } else {
    incomplete = 1.0 - exp(logFront) * betaContinuedFraction(b, a, y) / b;
  }
  return clamp(incomplete, 0.0, 1.0);
}
`,b=`GPUChangeDetection`,x=256,S=64,C=16,w=[`tStatistic`,`tDegreesOfFreedom`,`tPValue`,`senSlope`,`mannKendallS`,`mannKendallZ`,`mannKendallP`,`significance`],T=[`difference`,`logRatio`,`percentChange`,`tStatistic`,`tDegreesOfFreedom`,`tPValue`,`senSlope`,`mannKendallZ`,`mannKendallP`,`changeMagnitude`],E=class{id;props;chunkPlan;estimatedCommandNodeCount;bandCount;significanceSource;ticksPerYear;constructor(t){this.id=t.id??`change-detection`,this.props=t;let r=this.id,{cellCount:l,sliceCount:u,output:d}=t,p=t.bandCount??1;if(this.bandCount=p,this.significanceSource=t.significanceSource??`t-test`,this.ticksPerYear=t.dateAxis?f(r,t.dateAxis,u):0,o(r,`cellCount`,l,1,2147483647),o(r,`bandCount`,p,1,2147483647),o(r,`sliceCount`,u,2,x),l*u*p>2147483647)throw Error(`${r} cellCount * sliceCount * bandCount must not exceed 2^31 - 1`);if(this.significanceSource!==`t-test`&&this.significanceSource!==`mann-kendall`)throw Error(`${r} significanceSource must be 't-test' or 'mann-kendall'`);let m=(e,t,n)=>{for(let r of a(e))i(r,t,n)};if(m(t.slices,[`float32`],`${r} slices`),t.slices.length<l*u*p)throw Error(`${r} slices must hold cellCount * sliceCount * bandCount rows`);if(t.mask&&(m(t.mask,[`uint32`],`${r} mask`),t.mask.length<l))throw Error(`${r} mask must hold cellCount rows`);if(t.observationMask&&(m(t.observationMask,[`uint32`],`${r} observationMask`),t.observationMask.length<l*u))throw Error(`${r} observationMask must hold cellCount * sliceCount rows`);c(r,t.parameters,5);let h=Object.entries(d).filter(([,e])=>e);if(h.length===0)throw Error(`${r} needs at least one output`);for(let e of T)d[e]&&m(d[e],[`float32`],`${r} output.${e}`);d.mannKendallS&&m(d.mannKendallS,[`sint32`],`${r} output.mannKendallS`),d.significance&&m(d.significance,[`uint32`],`${r} output.significance`);for(let e of w)if(d[e]&&p!==1)throw Error(`${r} output.${e} requires bandCount 1`);if(d.senSlope&&u>S)throw Error(`${r} output.senSlope supports at most ${S} slices`);if((d.changeMagnitude||d.changeDirection)&&(p<2||p>C))throw Error(`${r} changeMagnitude and changeDirection need bandCount in [2, ${C}]`);d.changeDirection&&m(d.changeDirection,[p===2?`float32`:`uint32`],`${r} output.changeDirection`);for(let[e,t]of h){let n=e===`difference`||e===`logRatio`||e===`percentChange`?l*p:l;if(t.length<n)throw Error(`${r} output.${e} must hold ${n} rows`)}this.chunkPlan=n(r,l,{slices:{view:t.slices,entriesPerCell:u*p},mask:{view:t.mask,entriesPerCell:1},observationMask:{view:t.observationMask,entriesPerCell:u},...Object.fromEntries(h.map(([e,t])=>[e,{view:t,entriesPerCell:[`difference`,`logRatio`,`percentChange`].includes(e)?p:1}]))});let g=this.getDispatches(),_=g.length;this.estimatedCommandNodeCount=_*this.chunkPlan.estimatedCommandNodeCount,s(r,this.estimatedCommandNodeCount),e(r,this.getInputs(),Object.values(d).filter(e=>e!==void 0),this.chunkPlan,g)}getDispatches(){let{props:e}=this,{output:t}=e,n=e=>e.filter(e=>e!==void 0),r=n([e.slices,e.mask,e.observationMask]),i=[...r,e.parameters];return[{inputs:i,outputs:n([t.difference,t.logRatio,t.percentChange])},{inputs:i,outputs:n([t.changeMagnitude,t.changeDirection])},{inputs:i,outputs:n([t.tStatistic,t.tDegreesOfFreedom,t.tPValue,this.significanceSource===`t-test`?t.significance:void 0])},{inputs:n([...r,e.dateAxis?.acquisitionTimes]),outputs:n([t.senSlope])},{inputs:i,outputs:n([t.mannKendallS,t.mannKendallZ,t.mannKendallP,this.significanceSource===`mann-kendall`?t.significance:void 0])}].filter(e=>e.outputs.length>0)}getInputs(){return[...new Set(this.getDispatches().flatMap(e=>e.inputs))]}getCommandNodes(e){let{props:n,id:i,bandCount:a}=this,{output:o,cellCount:s,sliceCount:c}=n;t(i,e,this.getInputs(),Object.values(o).filter(e=>e!==void 0)),n.dateAxis&&o.senSlope&&m(i,e,n.dateAxis);let l=(e,t)=>({name:e,view:t,type:`f32`,access:`read`}),f=()=>[l(`slices`,n.slices),...n.mask?[{name:`cellMask`,view:n.mask,type:`u32`,access:`read`}]:[],...n.observationMask?[{name:`observationMask`,view:n.observationMask,type:`u32`,access:`read`}]:[]],g=(e,t,n)=>({name:e,view:t,type:n,access:`read_write`}),_=`const SLICE_COUNT: u32 = ${c}u;
const BAND_COUNT: u32 = ${a}u;
${v}
// Per-observation mask (all bands of one cell and slice); true when no view is bound.
fn isObservationValid(cell: u32, slice: u32) -> bool {
  return ${n.observationMask?`observationMask[observationMaskOffset + cell * SLICE_COUNT + slice] != 0u`:`true`};
}
// The sample at (cell, slice, band) is missing: not finite, or its observation is invalid.
fn isMissingAt(cell: u32, slice: u32, value: f32) -> bool {
  return isMissingSample(value) || !isObservationValid(cell, slice);
}`,x=n.mask?`if (cellMask[cellMaskOffset + cell] == 0u) {
    return;
  }`:``,S=(e,t=`0u`)=>`(cell * SLICE_COUNT + ${e}) * BAND_COUNT + ${t}`,C=[];if(o.difference||o.logRatio||o.percentChange){let t=[...f(),l(`params`,n.parameters)],c=[],u=[];o.difference&&(t.push(g(`differenceOut`,o.difference,`f32`)),c.push(`differenceOut[differenceOutOffset + index] = quietNan();`),u.push(`differenceOut[differenceOutOffset + index] = after - before;`)),o.logRatio&&(t.push(g(`logRatioOut`,o.logRatio,`f32`)),c.push(`logRatioOut[logRatioOutOffset + index] = quietNan();`),u.push(`if (after > -epsilon && before > -epsilon) {
    logRatioOut[logRatioOutOffset + index] = log((after + epsilon) / (before + epsilon));
  }`)),o.percentChange&&(t.push(g(`percentChangeOut`,o.percentChange,`f32`)),c.push(`percentChangeOut[percentChangeOutOffset + index] = quietNan();`),u.push(`if (before != 0.0) {
    percentChangeOut[percentChangeOutOffset + index] = 100.0 * (after - before) / abs(before);
  }`)),C.push(...r(e,this.chunkPlan,{id:`${i}-two-slice`,operation:b,variant:`two-slice`,bindings:t,invocationCount:s*a,declarations:_,body:`let cell = index / BAND_COUNT;
  let band = index % BAND_COUNT;
  ${c.join(`
  `)}
  ${x}
  let beforeSlice = params[paramsOffset];
  let afterSlice = params[paramsOffset + 1u];
  let epsilon = params[paramsOffset + 2u];
  if (!(beforeSlice >= 0.0 && beforeSlice < f32(SLICE_COUNT) && afterSlice >= 0.0 && afterSlice < f32(SLICE_COUNT))) {
    return;
  }
  let before = slices[slicesOffset + ${S(`u32(beforeSlice)`,`band`)}];
  let after = slices[slicesOffset + ${S(`u32(afterSlice)`,`band`)}];
  if (isMissingAt(cell, u32(beforeSlice), before) || isMissingAt(cell, u32(afterSlice), after)) {
    return;
  }
  ${u.join(`
  `)}`}))}if(o.changeMagnitude||o.changeDirection){let t=[...f(),l(`params`,n.parameters)],c=[],u=[];if(o.changeMagnitude&&(t.push(g(`magnitudeOut`,o.changeMagnitude,`f32`)),c.push(`magnitudeOut[magnitudeOutOffset + cell] = quietNan();`),u.push(`magnitudeOut[magnitudeOutOffset + cell] = sqrt(squaredNorm);`)),o.changeDirection){let e=a===2;t.push(g(`directionOut`,o.changeDirection,e?`f32`:`u32`)),c.push(`directionOut[directionOutOffset + cell] = ${e?`quietNan()`:`0xffffffffu`};`),u.push(e?`if (differences[0] != 0.0 || differences[1] != 0.0) {
    directionOut[directionOutOffset + cell] = guardedAtan2(differences[1], differences[0]);
  } else {
    directionOut[directionOutOffset + cell] = 0.0;
  }`:`directionOut[directionOutOffset + cell] = code;`)}C.push(...r(e,this.chunkPlan,{id:`${i}-multiband`,operation:b,variant:`multiband`,bindings:t,invocationCount:s,declarations:`${_}\n${d}`,body:`let cell = index;
  ${c.join(`
  `)}
  ${x}
  let beforeSlice = params[paramsOffset];
  let afterSlice = params[paramsOffset + 1u];
  if (!(beforeSlice >= 0.0 && beforeSlice < f32(SLICE_COUNT) && afterSlice >= 0.0 && afterSlice < f32(SLICE_COUNT))) {
    return;
  }
  var differences: array<f32, ${a}>;
  var squaredNorm = 0.0;
  var code = 0u;
  for (var band = 0u; band < BAND_COUNT; band++) {
    let before = slices[slicesOffset + ${S(`u32(beforeSlice)`,`band`)}];
    let after = slices[slicesOffset + ${S(`u32(afterSlice)`,`band`)}];
    if (isMissingAt(cell, u32(beforeSlice), before) || isMissingAt(cell, u32(afterSlice), after)) {
      return;
    }
    let difference = after - before;
    differences[band] = difference;
    squaredNorm += difference * difference;
    if (difference > 0.0) {
      code |= 1u << band;
    } else if (difference < 0.0) {
      code |= 1u << (16u + band);
    }
  }
  ${u.join(`
  `)}`}))}let w=o.tStatistic||o.tDegreesOfFreedom||o.tPValue,T=o.significance&&this.significanceSource===`t-test`;if(w||T){let t=[...f(),l(`params`,n.parameters)],a=[],c=[];for(let[e,n,r]of[[`tStatisticOut`,o.tStatistic,`t`],[`tDegreesOut`,o.tDegreesOfFreedom,`degreesOfFreedom`],[`tPValueOut`,o.tPValue,`pValue`]])n&&(t.push(g(e,n,`f32`)),a.push(`${e}[${e}Offset + cell] = quietNan();`),c.push(`${e}[${e}Offset + cell] = ${r};`));T&&(t.push(g(`significanceOut`,o.significance,`u32`)),a.push(`significanceOut[significanceOutOffset + cell] = 0u;`),c.push(`if (pValue < alpha) {
    significanceOut[significanceOutOffset + cell] = select(2u, 1u, t > 0.0);
  }`)),C.push(...r(e,this.chunkPlan,{id:`${i}-t-test`,operation:b,variant:`t-test`,bindings:t,invocationCount:s,declarations:`${_}\n${y}`,body:`let cell = index;
  ${a.join(`
  `)}
  ${x}
  let alpha = params[paramsOffset + 3u];
  let splitSlice = params[paramsOffset + 4u];
  if (!(splitSlice >= 1.0 && splitSlice <= f32(SLICE_COUNT - 1u))) {
    return;
  }
  let split = u32(splitSlice);
  var count1 = 0u;
  var count2 = 0u;
  var origin1 = 0.0;
  var origin2 = 0.0;
  var sum1 = 0.0;
  var sum2 = 0.0;
  for (var slice = 0u; slice < SLICE_COUNT; slice++) {
    let value = slices[slicesOffset + ${S(`slice`)}];
    if (isMissingAt(cell, slice, value)) {
      continue;
    }
    if (slice < split) {
      if (count1 == 0u) {
        origin1 = value;
      }
      count1++;
      sum1 += value - origin1;
    } else {
      if (count2 == 0u) {
        origin2 = value;
      }
      count2++;
      sum2 += value - origin2;
    }
  }
  if (count1 < 2u || count2 < 2u) {
    return;
  }
  // Center before accumulating: constant groups have exactly zero residuals even
  // when division by their non-power-of-two sample counts is rounded by the GPU.
  let mean1 = sum1 / f32(count1);
  let mean2 = sum2 / f32(count2);
  var squares1 = 0.0;
  var squares2 = 0.0;
  for (var slice = 0u; slice < SLICE_COUNT; slice++) {
    let value = slices[slicesOffset + ${S(`slice`)}];
    if (isMissingAt(cell, slice, value)) {
      continue;
    }
    if (slice < split) {
      let residual = (value - origin1) - mean1;
      squares1 += residual * residual;
    } else {
      let residual = (value - origin2) - mean2;
      squares2 += residual * residual;
    }
  }
  let errorSquared1 = squares1 / f32(count1 - 1u) / f32(count1);
  let errorSquared2 = squares2 / f32(count2 - 1u) / f32(count2);
  let errorSquared = errorSquared1 + errorSquared2;
  let denominator = errorSquared1 * errorSquared1 / f32(count1 - 1u)
    + errorSquared2 * errorSquared2 / f32(count2 - 1u);
  if (!(errorSquared > 0.0 && denominator > 0.0)) {
    return;
  }
  let t = ((origin2 - origin1) + (mean2 - mean1)) / sqrt(errorSquared);
  let degreesOfFreedom = errorSquared * errorSquared / denominator;
  let pValue = studentTTwoSidedP(t, degreesOfFreedom);
  ${c.join(`
  `)}`}))}if(o.senSlope){let t=c*(c-1)/2,a=Math.floor(t/2)+1;C.push(...r(e,this.chunkPlan,{id:`${i}-sen-slope`,operation:b,variant:`sen-slope`,workgroupSize:64,bindings:[...f(),...n.dateAxis?[h(n.dateAxis)]:[],g(`senOut`,o.senSlope,`f32`)],invocationCount:s,declarations:n.dateAxis?`${_}\n${p(this.ticksPerYear,c)}`:_,body:`let cell = index;
  senOut[senOutOffset + cell] = quietNan();
  ${x}
  var validCount = 0u;
  for (var slice = 0u; slice < SLICE_COUNT; slice++) {
    if (!isMissingAt(cell, slice, slices[slicesOffset + ${S(`slice`)}])) {
      validCount++;
    }
  }
  if (validCount < 2u) {
    return;
  }
  let pairCount = validCount * (validCount - 1u) / 2u;
  let heapSize = pairCount / 2u + 1u;
  var heap: array<f32, ${a}>;
  var size = 0u;
  for (var i = 0u; i < SLICE_COUNT; i++) {
    let valueI = slices[slicesOffset + ${S(`i`)}];
    if (isMissingAt(cell, i, valueI)) {
      continue;
    }
    for (var j = i + 1u; j < SLICE_COUNT; j++) {
      let valueJ = slices[slicesOffset + ${S(`j`)}];
      if (isMissingAt(cell, j, valueJ)) {
        continue;
      }
      let slope = (valueJ - valueI) / ${n.dateAxis?`rasterYearsBetween(j, i)`:`f32(j - i)`};
      if (size < heapSize) {
        var child = size;
        size++;
        heap[child] = slope;
        loop {
          if (child == 0u) { break; }
          let parent = (child - 1u) / 2u;
          if (heap[parent] >= heap[child]) { break; }
          let swapped = heap[parent];
          heap[parent] = heap[child];
          heap[child] = swapped;
          child = parent;
        }
      } else if (slope < heap[0]) {
        heap[0] = slope;
        var parent = 0u;
        loop {
          let left = 2u * parent + 1u;
          if (left >= size) { break; }
          var larger = left;
          if (left + 1u < size && heap[left + 1u] > heap[left]) { larger = left + 1u; }
          if (heap[parent] >= heap[larger]) { break; }
          let swapped = heap[parent];
          heap[parent] = heap[larger];
          heap[larger] = swapped;
          parent = larger;
        }
      }
    }
  }
  var median = heap[0];
  if (pairCount % 2u == 0u) {
    var secondLargest = heap[1];
    if (size > 2u) { secondLargest = max(secondLargest, heap[2]); }
    median = 0.5 * (heap[0] + secondLargest);
  }
  senOut[senOutOffset + cell] = median;`}))}let E=o.mannKendallS||o.mannKendallZ||o.mannKendallP,D=o.significance&&this.significanceSource===`mann-kendall`;if(E||D){let t=[...f()],a=[],c=[];D&&t.push(l(`params`,n.parameters)),o.mannKendallS&&(t.push(g(`sOut`,o.mannKendallS,`i32`)),a.push(`sOut[sOutOffset + cell] = 0;`),c.push(`sOut[sOutOffset + cell] = score;`)),o.mannKendallZ&&(t.push(g(`zOut`,o.mannKendallZ,`f32`)),a.push(`zOut[zOutOffset + cell] = quietNan();`),c.push(`zOut[zOutOffset + cell] = z;`)),o.mannKendallP&&(t.push(g(`pOut`,o.mannKendallP,`f32`)),a.push(`pOut[pOutOffset + cell] = quietNan();`),c.push(`pOut[pOutOffset + cell] = pValue;`)),D&&(t.push(g(`significanceOut`,o.significance,`u32`)),a.push(`significanceOut[significanceOutOffset + cell] = 0u;`),c.push(`if (pValue < params[paramsOffset + 3u] && score != 0) {
    significanceOut[significanceOutOffset + cell] = select(2u, 1u, score > 0);
  }`)),C.push(...r(e,this.chunkPlan,{id:`${i}-mann-kendall`,operation:b,variant:`mann-kendall`,workgroupSize:64,bindings:t,invocationCount:s,declarations:`${_}\n${u}`,body:`let cell = index;
  ${a.join(`
  `)}
  ${x}
  var validCount = 0u;
  var score = 0;
  var tieTerm = 0u;
  for (var i = 0u; i < SLICE_COUNT; i++) {
    let valueI = slices[slicesOffset + ${S(`i`)}];
    if (isMissingAt(cell, i, valueI)) {
      continue;
    }
    validCount++;
    var seenBefore = false;
    for (var k = 0u; k < i; k++) {
      // An earlier observation that is masked out does not start a tie group.
      if (isObservationValid(cell, k) && slices[slicesOffset + ${S(`k`)}] == valueI) {
        seenBefore = true;
      }
    }
    var tied = 1u;
    for (var j = i + 1u; j < SLICE_COUNT; j++) {
      let valueJ = slices[slicesOffset + ${S(`j`)}];
      if (isMissingAt(cell, j, valueJ)) {
        continue;
      }
      if (valueJ > valueI) {
        score += 1;
      } else if (valueJ < valueI) {
        score -= 1;
      } else {
        tied++;
      }
    }
    if (!seenBefore && tied > 1u) {
      tieTerm += tied * (tied - 1u) * (2u * tied + 5u);
    }
  }
  if (validCount < 2u) {
    return;
  }
  let varianceNumerator = validCount * (validCount - 1u) * (2u * validCount + 5u) - tieTerm;
  var z = 0.0;
  var pValue = 1.0;
  if (varianceNumerator > 0u) {
    let standardDeviation = sqrt(f32(varianceNumerator) / 18.0);
    let correctedScore = select(select(0, score + 1, score < 0), score - 1, score > 0);
    z = f32(correctedScore) / standardDeviation;
    pValue = erfcValue(abs(z) * 0.70710678118);
  }
  ${c.join(`
  `)}`}))}return C}};export{g as n,E as t};