import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-gather-BJIYd708.js";import{t as n}from"./gpu-sort-xRm8l5yn.js";import{S as r}from"./gpu-contributor-utils-BIwLQON5.js";import{o as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{n as a}from"./sorted-segment-offsets-ChmOtA9F.js";var o={sum:`0.0`,min:`3.402823466e+38`,max:`-3.402823466e+38`},s=16777215,c=`var<workgroup> partialSums: array<f32, 256>;`;function l(e=`sum`){return`partialSums[localInvocationIndex] = partialSum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] = ${e===`sum`?`partialSums[localInvocationIndex] + partialSums[localInvocationIndex + stride]`:`${e}(partialSums[localInvocationIndex], partialSums[localInvocationIndex + stride])`};
    }
    workgroupBarrier();
  }`}function u(e){return[Math.min(e,s)*256,256]}function d(e){return Math.max(1,Math.floor(e).toString(2).length)}function f(r,o){let{id:s,operation:c,segmentCount:l,segmentKeys:u}=o,f=u.length,m=[],h=e(r,`${s}-sort-keys`,`uint32`,f),g=e(r,`${s}-sort-indices`,`uint32`,f),_=e(r,`${s}-sorted-keys`,`uint32`,f),v=e(r,`${s}-sorted-indices`,`uint32`,f);m.push(i(r,{id:`${s}-sort-prepare`,operation:c,variant:`sort-prepare`,bindings:[{name:`pointFeatureRows`,view:u,type:`u32`,access:`read`},{name:`sortKeys`,view:h,type:`u32`,access:`read_write`},{name:`sortIndices`,view:g,type:`u32`,access:`read_write`}],invocationCount:f,declarations:`const FEATURE_COUNT: u32 = ${l}u;`,body:`sortKeys[sortKeysOffset + index] = min(pointFeatureRows[pointFeatureRowsOffset + index], FEATURE_COUNT);
  sortIndices[sortIndicesOffset + index] = index;`})),m.push(...new n({id:`${s}-sort`,keys:h,values:g,outputKeys:_,outputValues:v,keyBits:d(l)}).getCommandNodes(r));let y=e(r,`${s}-segment-offsets`,`uint32`,l+1);m.push(...a(r,{id:s,operation:c,groupCount:l,sortedGroupKeys:_,segmentOffsets:y}));let b=[];o.sums&&o.sumContributions&&b.push({name:`sums`,contributions:o.sumContributions,output:o.sums}),o.weightSums&&o.weightContributions&&b.push({name:`weight-sums`,contributions:o.weightContributions,output:o.weightSums}),b.push(...o.reductions??[]);for(let n of b){let i=e(r,`${s}-sorted-${n.name}`,`float32`,f);m.push(...new t({id:`${s}-gather-${n.name}`,source:n.contributions,indices:v,output:i}).getCommandNodes(r),p(r,{id:`${s}-reduce-${n.name}`,operation:c,segmentCount:l,input:i,segmentOffsets:y,output:n.output}))}return m}function p(e,t){return r(t.id,`segmentCount + 1`,t.segmentCount+1,t.segmentOffsets.length),r(t.id,`segmentCount`,t.segmentCount,t.output.length),m(e,t)}function m(e,t){let n=t.reduction??`sum`,r=n===`sum`?`partialSum + next`:`${n}(partialSum, next)`;return i(e,{id:t.id,operation:t.operation,variant:`segment-${n}`,condition:t.condition,extraResources:t.extraResources,bindings:[{name:`segmentInput`,view:t.input,type:`f32`,access:`read`},{name:`segmentOffsets`,view:t.segmentOffsets,type:`u32`,access:`read`},{name:`segmentOutput`,view:t.output,type:`f32`,access:`read_write`}],workgroupSize:256,invocationCount:Math.min(t.segmentCount,s)*256,guardIndex:!1,declarations:`const INPUT_COUNT: u32 = ${t.input.length}u;
const SEGMENT_COUNT: u32 = ${t.segmentCount}u;
${c}`,body:`let firstSegment = index / 256u;
  let segmentStride = INVOCATION_COUNT / 256u;
  // Segment counts beyond one dispatch are reduced in strided passes with a uniform trip count.
  for (var segmentPass = 0u; segmentPass < ${Math.ceil(t.segmentCount/s)}u; segmentPass++) {
  let segment = firstSegment + segmentPass * segmentStride;
  let isInRange = index < INVOCATION_COUNT && segment < SEGMENT_COUNT;
  var partialSum = ${o[n]};
  var isEmpty = true;
  if (isInRange) {
    let begin = segmentOffsets[segmentOffsetsOffset + segment];
    let end = min(segmentOffsets[segmentOffsetsOffset + segment + 1u], INPUT_COUNT);
    for (var row = begin + localInvocationIndex; row < end; row += 256u) {
      let next = segmentInput[segmentInputOffset + row];
      partialSum = ${r};
    }
    isEmpty = begin >= end;
  }
  ${l(n)}
  if (isInRange && localInvocationIndex == 0u) {
    segmentOutput[segmentOutputOffset + segment] = select(partialSums[0], 0.0, isEmpty);
  }
  workgroupBarrier();
  }`})}export{d as a,l as i,p as n,f as o,u as r,c as t};