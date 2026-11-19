import{c as e,o as t,r as n,t as r}from"./kernel-C3tfQOn4.js";import{r as i}from"./gpu-command-graph-BdzdmV6u.js";import{t as a}from"./gpu-dispatch-utils-CbRzFAgT.js";import{a as o,s}from"./graph-chunk-dispatch-zBt8k2-o.js";import{v as c}from"./gpu-contributor-utils-BIwLQON5.js";var l=`
fn mergeFacilityScores(first: vec4<u32>, second: vec4<u32>) -> vec4<u32> {
  let preferSecond = second.x < first.x ||
    (second.x == first.x && (second.y < first.y ||
      (second.y == first.y && second.z < first.z)));
  let winner = select(first, second, preferSecond);
  return vec4<u32>(winner.xyz, first.w | second.w);
}`;function u(l,u){o(l,u.id,{scores:u.scores}),u.candidateCount>1&&(c(l.device,u.id,`reduction parameters`,16,`request a larger storage binding limit`),c(l.device,u.id,`reduced scores`,Math.ceil(u.candidateCount/64)*16,`reduce candidateCount`));let f=[],p=u.scores,m=u.candidateCount,h=0;for(;m>1;){let o=Math.ceil(m/64),c=a(u.operation,o*64,64,l.device.limits.maxComputeWorkgroupsPerDimension),g=n(l,`${u.id}-${h}`,`uint32`,o*4),_=s(l,`${u.id}-${h}-parameters`,Uint32Array.of(e(p),m,c.x,c.y));f.push(..._.nodes);let v=p;f.push(i({id:`${u.id}-${h}`,resources:[{buffer:v,usage:`storage-read`},{buffer:_.parameters,usage:`storage-read`},{buffer:g,usage:`storage-read-write`}],workload:{operation:u.operation,variant:`candidate-reduction`,commandCount:1,maximumWorkgroupCount:c.x*c.y*c.z,maximumInvocationCount:o*64},compile:({device:e})=>{let n=new r(e,{id:`${u.id}-reduction`,source:d,shaderLayout:{bindings:[{name:`candidateScores`,type:`read-only-storage`,group:0,location:0},{name:`parameters`,type:`read-only-storage`,group:0,location:1},{name:`reducedScores`,type:`storage`,group:0,location:2}]}});return{encode:({computePass:e,getBuffer:r})=>n.dispatch(e,{bindings:{candidateScores:t(v,r),parameters:t(_.parameters,r),reducedScores:t(g,r)},x:c.x,y:c.y,z:c.z}),destroy:()=>n.destroy()}}})),p=g,m=o,h++}return{nodes:f,winner:p}}var d=`
@group(0) @binding(0) var<storage, read> candidateScores: array<u32>;
@group(0) @binding(1) var<storage, read> parameters: array<u32>;
@group(0) @binding(2) var<storage, read_write> reducedScores: array<u32>;
${l}
var<workgroup> scores: array<vec4<u32>, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) localInvocationIndex: u32,
        @builtin(workgroup_id) workgroupId: vec3<u32>) {
  let workgroupIndex = (workgroupId.z * parameters[3] + workgroupId.y) * parameters[2] + workgroupId.x;
  let index = workgroupIndex * 64u + localInvocationIndex;
  var score = vec4<u32>(0xffffffffu, 0xffffffffu, 0xffffffffu, 0u);
  if (index < parameters[1]) {
    let offset = parameters[0] + index * 4u;
    score = vec4<u32>(candidateScores[offset], candidateScores[offset + 1u], candidateScores[offset + 2u], candidateScores[offset + 3u]);
  }
  scores[localInvocationIndex] = score;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (localInvocationIndex < stride) {
      scores[localInvocationIndex] = mergeFacilityScores(scores[localInvocationIndex], scores[localInvocationIndex + stride]);
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u && workgroupIndex < arrayLength(&reducedScores) / 4u) {
    let offset = workgroupIndex * 4u;
    reducedScores[offset] = scores[0].x;
    reducedScores[offset + 1u] = scores[0].y;
    reducedScores[offset + 2u] = scores[0].z;
    reducedScores[offset + 3u] = scores[0].w;
  }
}`;export{u as n,l as t};