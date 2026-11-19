import{a as e,i as t,n,r,t as i}from"./raster-time-series-chunks-b-CPab_8L5.js";import{T as a,f as o}from"./gpu-contributor-utils-BIwLQON5.js";import{t as s}from"./raster-kernel-utils-DcIN7G5I.js";import{o as c,r as l,t as u}from"./gpu-raster-streaming-accumulator-sRoa05Oe.js";import{t as d}from"./fixed-point-accumulator-Ce6W3HVY.js";var f=100;function p(e,t,n=new Float32Array(7)){let r=e.maximumStreamSliceCount??u;return n[0]=+!!e.reset,n[1]=e.activeSliceCount??t,n[2]=e.location??0,n[3]=l(e.maximumValueMagnitude,r),n[4]=l(f,r),n[5]=l(4*e.maximumValueMagnitude**2,r),n[6]=e.maximumValueMagnitude,n}var m=class{id;props;chunkPlan;estimatedCommandNodeCount;slotCount;constructor(t){this.props=t;let r=t.id??`raster-standardized-index-calibration`;this.id=r;let i=t.slotCount??1;this.slotCount=i,o(r,`cellCount`,t.cellCount),o(r,`chunkSliceCount`,t.chunkSliceCount,1,256),o(r,`slotCount`,i,1,366),a(r,t.distribution===`gamma`||t.distribution===`fisk`,`distribution must be 'gamma' or 'fisk'`);let s=t.cellCount*t.chunkSliceCount,c=t.cellCount*i;o(r,`cellCount times chunkSliceCount`,s),o(r,`cellCount times slotCount`,c),a(r,t.values.length>=s&&(!t.mask||t.mask.length>=s),`values and mask must hold cellCount times chunkSliceCount entries`),a(r,i===1||!!t.slotIndices&&t.slotIndices.length>=t.chunkSliceCount,`slotIndices must hold one entry per chunk slice when slotCount exceeds 1`),a(r,t.parameters.length>=7,`parameters must hold 7 entries`),a(r,t.state.counts.length>=c*3&&t.state.moments.length>=c*9&&t.fit.length>=c*3,`state.counts needs three, state.moments nine and fit three entries per slot`),this.chunkPlan=n(this.id,t.cellCount,{values:{view:t.values,entriesPerCell:t.chunkSliceCount},mask:{view:t.mask,entriesPerCell:t.chunkSliceCount},"state.counts":{view:t.state.counts,entriesPerCell:(t.slotCount??1)*3},"state.moments":{view:t.state.moments,entriesPerCell:(t.slotCount??1)*9},fit:{view:t.fit,entriesPerCell:(t.slotCount??1)*3}}),this.estimatedCommandNodeCount=this.chunkPlan.estimatedCommandNodeCount,e(this.id,this.getInputs(),this.getOutputs(),this.chunkPlan)}getInputs(){let{values:e,mask:t,slotIndices:n,parameters:r}=this.props;return[e,r,...t?[t]:[],...(this.props.slotCount??1)>1&&n?[n]:[]]}getOutputs(){let{state:e,fit:t}=this.props;return[e.counts,e.moments,t]}getCommandNodes(e){let{props:n,id:a,slotCount:o}=this,{chunkSliceCount:l,values:u,mask:f,slotIndices:p,parameters:m,state:h,fit:g}=n,_=n.zeroInflated??!1;t(a,e,this.getInputs(),this.getOutputs());let v=n.distribution===`gamma`?`let logarithmGap = log(mean) - logarithmMean;
    let shape = (1.0 + sqrt(1.0 + 4.0 * logarithmGap / 3.0)) / (4.0 * logarithmGap);
    let scale = mean / shape;`:`let variance = max(moment, 0.0) / f32(positiveCount);
    let scale = 2.0 * mean * mean * mean / (variance + 2.0 * mean * mean);
    let shape = 3.141592653589793 * mean / (1.7320508075688772 * sqrt(variance));`;return i(e,this.chunkPlan,{id:a,operation:`GPURasterStandardizedIndexCalibration`,invocationCount:n.cellCount,workgroupSize:64,bindings:[r(`values`,u),r(`parameters`,m),...f?[r(`mask`,f)]:[],...o>1&&p?[r(`slotIndices`,p)]:[],r(`counts`,h.counts,!0),r(`moments`,h.moments,!0),r(`fit`,g,!0)],declarations:`${s}\n${d}\n${c(`moments`)}`,body:`if (parameters[parametersOffset] != 0.0) {
  for (var slot = 0u; slot < ${o}u; slot++) {
    let entry = index * ${o}u + slot;
    counts[countsOffset + 3u * entry] = 0u; counts[countsOffset + 3u * entry + 1u] = 0u; counts[countsOffset + 3u * entry + 2u] = 0u;
    for (var triple = 0u; triple < 3u; triple++) { storeMoments(3u * entry + triple, vec3<u32>(0u)); }
  }
}
let activeSliceCount = u32(clamp(parameters[parametersOffset + 1u], 0.0, ${l}.0));
let location = parameters[parametersOffset + 2u];
let sumUnit = i32(parameters[parametersOffset + 3u]);
let logarithmUnit = i32(parameters[parametersOffset + 4u]);
let momentUnit = i32(parameters[parametersOffset + 5u]);
let maximumMagnitude = parameters[parametersOffset + 6u];
for (var slice = 0u; slice < activeSliceCount; slice++) {
  let source = index * ${l}u + slice;
  let value = values[valuesOffset + source];
  if (!finite(value) ${f?`|| mask[maskOffset + source] == 0u`:``}) { continue; }
  let slot = ${o>1?`slotIndices[slotIndicesOffset + slice]`:`0u`};
  if (slot >= ${o}u) { continue; }
  let entry = index * ${o}u + slot;
  let positive = value - location;
  if (positive > maximumMagnitude) { continue; }
  counts[countsOffset + 3u * entry] += 1u;
  ${_?`if (value == 0.0) { counts[countsOffset + 3u * entry + 1u] += 1u; continue; }`:``}
  if (!(positive > 0.0)) { continue; }
  let priorCount = counts[countsOffset + 3u * entry + 2u];
  counts[countsOffset + 3u * entry + 2u] = priorCount + 1u;
  let priorSum = loadMoments(3u * entry);
  let priorMean = getFixedPointFloat(priorSum, sumUnit) / f32(max(priorCount, 1u));
  let sum = addFixedPoint(priorSum, quantizeFixedPoint(positive, sumUnit));
  storeMoments(3u * entry, sum);
  storeMoments(3u * entry + 1u, addFixedPoint(loadMoments(3u * entry + 1u), quantizeFixedPoint(log(positive), logarithmUnit)));
  let updatedMean = getFixedPointFloat(sum, sumUnit) / f32(priorCount + 1u);
  let deviation = select((positive - priorMean) * (positive - updatedMean), 0.0, priorCount == 0u);
  storeMoments(3u * entry + 2u, addFixedPoint(loadMoments(3u * entry + 2u), quantizeFixedPoint(deviation, momentUnit)));
}
for (var slot = 0u; slot < ${o}u; slot++) {
  let entry = index * ${o}u + slot;
  let nonMissing = counts[countsOffset + 3u * entry];
  let zeroCount = counts[countsOffset + 3u * entry + 1u];
  let positiveCount = counts[countsOffset + 3u * entry + 2u];
  var fittedShape = missing();
  var fittedScale = missing();
  if (positiveCount >= 2u) {
    let mean = getFixedPointFloat(loadMoments(3u * entry), sumUnit) / f32(positiveCount);
    let logarithmMean = getFixedPointFloat(loadMoments(3u * entry + 1u), logarithmUnit) / f32(positiveCount);
    let moment = getFixedPointFloat(loadMoments(3u * entry + 2u), momentUnit);
    ${v}
    if (shape > 0.0 && scale > 0.0 && finite(shape) && finite(scale)) { fittedShape = shape; fittedScale = scale; }
  }
  fit[fitOffset + 3u * entry] = fittedShape;
  fit[fitOffset + 3u * entry + 1u] = fittedScale;
  fit[fitOffset + 3u * entry + 2u] = select(0.0, f32(zeroCount) / f32(nonMissing), nonMissing > 0u);
}`})}};export{p as n,m as t};