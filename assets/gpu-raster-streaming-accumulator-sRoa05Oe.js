import{a as e,i as t,n,r,t as i}from"./raster-time-series-chunks-b-CPab_8L5.js";import{o as a,t as o}from"./fixed-point-accumulator-Ce6W3HVY.js";import{T as s,f as c,g as l}from"./gpu-contributor-utils-BIwLQON5.js";import{t as u}from"./raster-kernel-utils-DcIN7G5I.js";function d(e,t=!0){let n=`${e[0].toUpperCase()}${e.slice(1)}`;return`
fn load${n}(entry: u32) -> vec3<u32> {
  return vec3<u32>(${e}[${e}Offset + 3u * entry], ${e}[${e}Offset + 3u * entry + 1u], ${e}[${e}Offset + 3u * entry + 2u]);
}
${t?`fn store${n}(entry: u32, words: vec3<u32>) {
  ${e}[${e}Offset + 3u * entry] = words.x;
  ${e}[${e}Offset + 3u * entry + 1u] = words.y;
  ${e}[${e}Offset + 3u * entry + 2u] = words.z;
}`:``}`}var f=2**24;function p(e,t,n=new Float32Array(5)){l(`getGPURasterStreamingAccumulatorParameterValues`,`maximumValueMagnitude`,e.maximumValueMagnitude,Number.MIN_VALUE);let r=e.maximumStreamSliceCount??f;return n[0]=+!!e.reset,n[1]=e.activeSliceCount??t,n[2]=m(e.maximumValueMagnitude,r),n[3]=m(4*e.maximumValueMagnitude**2,r),n[4]=e.maximumValueMagnitude,n}function m(e,t){return Math.max(new Uint32Array(new Float32Array([e]).buffer)[0]>>>23&255,1)-127+1+a(t)-95}function h(e,t){let n=new Float64Array(Math.floor(e.length/3));for(let r=0;r<n.length;r++){let i=BigInt(e[r*3]>>>0)|BigInt(e[r*3+1]>>>0)<<32n|BigInt(e[r*3+2]>>>0)<<64n,a=BigInt.asIntN(96,i);n[r]=Number(a)*2**t}return n}var g=class{id;props;chunkPlan;estimatedCommandNodeCount;slotCount;constructor(t){this.props=t;let r=t.id??`raster-streaming-accumulator`;this.id=r;let i=t.slotCount??1;this.slotCount=i,c(r,`cellCount`,t.cellCount),c(r,`chunkSliceCount`,t.chunkSliceCount,1,256),c(r,`slotCount`,i,1,366),c(r,`cellCount times chunkSliceCount`,t.cellCount*t.chunkSliceCount),c(r,`cellCount times slotCount`,t.cellCount*i);let a=t.cellCount*i;s(r,t.values.length>=t.cellCount*t.chunkSliceCount&&(!t.mask||t.mask.length>=t.cellCount*t.chunkSliceCount),`values and mask must hold cellCount times chunkSliceCount entries`),s(r,i===1||!!t.slotIndices&&t.slotIndices.length>=t.chunkSliceCount,`slotIndices must hold one entry per chunk slice when slotCount exceeds 1`),s(r,t.parameters.length>=5,`parameters must hold 5 entries`),s(r,t.state.counts.length>=a&&t.state.sums.length>=a*3,`state.counts must hold cellCount times slotCount entries and state.sums three times that`),s(r,(!t.state.moments||t.state.moments.length>=a*3)&&(!t.state.extrema||t.state.extrema.length>=a*2),`state.moments must hold three entries per slot and state.extrema two`),this.chunkPlan=n(this.id,t.cellCount,{values:{view:t.values,entriesPerCell:t.chunkSliceCount},mask:{view:t.mask,entriesPerCell:t.chunkSliceCount},"state.counts":{view:t.state.counts,entriesPerCell:t.slotCount??1},"state.sums":{view:t.state.sums,entriesPerCell:(t.slotCount??1)*3},"state.moments":{view:t.state.moments,entriesPerCell:(t.slotCount??1)*3},"state.extrema":{view:t.state.extrema,entriesPerCell:(t.slotCount??1)*2}}),this.estimatedCommandNodeCount=this.chunkPlan.estimatedCommandNodeCount,e(this.id,this.getInputs(),this.getOutputs(),this.chunkPlan)}getInputs(){let{values:e,mask:t,slotIndices:n,parameters:r}=this.props;return[e,r,...t?[t]:[],...(this.props.slotCount??1)>1&&n?[n]:[]]}getOutputs(){let{counts:e,sums:t,moments:n,extrema:r}=this.props.state;return[e,t,...n?[n]:[],...r?[r]:[]]}getCommandNodes(e){let{props:n,id:a,slotCount:s}=this,{values:c,mask:l,slotIndices:f,parameters:p,state:m,cellCount:h,chunkSliceCount:g}=n;t(a,e,this.getInputs(),this.getOutputs());let _=s>1?`slotIndices[slotIndicesOffset + slice]`:`0u`;return i(e,this.chunkPlan,{id:a,operation:`GPURasterStreamingAccumulator`,invocationCount:h,workgroupSize:64,bindings:[r(`values`,c),r(`parameters`,p),...l?[r(`mask`,l)]:[],...s>1&&f?[r(`slotIndices`,f)]:[],r(`counts`,m.counts,!0),r(`sums`,m.sums,!0),...m.moments?[r(`moments`,m.moments,!0)]:[],...m.extrema?[r(`extrema`,m.extrema,!0)]:[]],declarations:`${u}
${o}
${d(`sums`)}
${m.moments?d(`moments`):``}`,body:`let unitExponent = i32(parameters[parametersOffset + 2u]);
let momentUnitExponent = i32(parameters[parametersOffset + 3u]);
if (parameters[parametersOffset] != 0.0) {
  for (var slot = 0u; slot < ${s}u; slot++) {
    let entry = index * ${s}u + slot;
    counts[countsOffset + entry] = 0u;
    storeSums(entry, vec3<u32>(0u));
    ${m.moments?`storeMoments(entry, vec3<u32>(0u));`:``}
    ${m.extrema?`extrema[extremaOffset + 2u * entry] = 3.402823466e38; extrema[extremaOffset + 2u * entry + 1u] = -3.402823466e38;`:``}
  }
}
let activeSliceCount = u32(clamp(parameters[parametersOffset + 1u], 0.0, ${g}.0));
for (var slice = 0u; slice < activeSliceCount; slice++) {
  let source = index * ${g}u + slice;
  let value = values[valuesOffset + source];
  if (!finite(value) ${l?`|| mask[maskOffset + source] == 0u`:``}) { continue; }
  if (abs(value) > parameters[parametersOffset + 4u]) { continue; }
  let slot = ${_};
  if (slot >= ${s}u) { continue; }
  let entry = index * ${s}u + slot;
  let priorCount = counts[countsOffset + entry];
  counts[countsOffset + entry] = priorCount + 1u;
  let priorSum = loadSums(entry);
  let priorMean = getFixedPointFloat(priorSum, unitExponent) / f32(max(priorCount, 1u));
  let sum = addFixedPoint(priorSum, quantizeFixedPoint(value, unitExponent));
  storeSums(entry, sum);
  ${m.moments?`let updatedMean = getFixedPointFloat(sum, unitExponent) / f32(priorCount + 1u);
  let deviation = select((value - priorMean) * (value - updatedMean), 0.0, priorCount == 0u);
  storeMoments(entry, addFixedPoint(loadMoments(entry), quantizeFixedPoint(deviation, momentUnitExponent)));`:``}
  ${m.extrema?`extrema[extremaOffset + 2u * entry] = min(extrema[extremaOffset + 2u * entry], value);
  extrema[extremaOffset + 2u * entry + 1u] = max(extrema[extremaOffset + 2u * entry + 1u], value);`:``}
}`})}};export{p as a,h as i,g as n,d as o,m as r,f as t};