import{t as e}from"./common-CshkX-d_.js";import{r as t}from"./kernel-C3tfQOn4.js";import{o as n}from"./wgsl-kernel-nodes-DagJklH2.js";var r=256,i=4096,a=r;function o(o,s){let{id:c,operation:l,variant:u,itemCount:d,output:f}=s,p=e(Math.ceil(d/i),1,a),m=`const ITEM_COUNT: u32 = ${d}u;
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
const PARTIAL_COUNT: u32 = ${p}u;
const WORKGROUP_THREADS: u32 = ${r}u;
${s.declarations}
var<workgroup> sharedMinima: array<vec2f, ${r}>;
var<workgroup> sharedMaxima: array<vec2f, ${r}>;`,h=`sharedMinima[localInvocationIndex] = localMinimum;
  sharedMaxima[localInvocationIndex] = localMaximum;
  workgroupBarrier();
  for (var stride = ${r/2}u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      sharedMinima[localInvocationIndex] = min(sharedMinima[localInvocationIndex], sharedMinima[localInvocationIndex + stride]);
      sharedMaxima[localInvocationIndex] = max(sharedMaxima[localInvocationIndex], sharedMaxima[localInvocationIndex + stride]);
    }
    workgroupBarrier();
  }`,g=e=>`if (localInvocationIndex == 0u) {
    let minimum = sharedMinima[0];
    let maximum = sharedMaxima[0];
    let hasValid = minimum.x <= maximum.x && minimum.y <= maximum.y;
    ${e}[${e}Offset] = select(0.0, minimum.x, hasValid);
    ${e}[${e}Offset + 1u] = select(0.0, minimum.y, hasValid);
    ${e}[${e}Offset + 2u] = select(0.0, maximum.x, hasValid);
    ${e}[${e}Offset + 3u] = select(0.0, maximum.y, hasValid);
  }`,_=`var localMinimum = vec2f(FLOAT32_MAXIMUM);
  var localMaximum = vec2f(-FLOAT32_MAXIMUM);
  // Grid-stride loop: consecutive threads read consecutive items on every trip.
  for (var row = index; row < ITEM_COUNT; row += PARTIAL_COUNT * WORKGROUP_THREADS) {
    let item = readItem(row);
    if (item.z > 0.5) {
      localMinimum = min(localMinimum, item.xy);
      localMaximum = max(localMaximum, item.xy);
    }
  }
  ${h}`;if(p===1)return[n(o,{id:`${c}-bounds`,operation:l,variant:u,bindings:[...s.bindings,{name:`boundsOut`,view:f,type:`f32`,access:`read_write`}],invocationCount:r,guardIndex:!1,declarations:m,body:`// Exactly one workgroup is dispatched, so index == localInvocationIndex.
  ${_}
  ${g(`boundsOut`)}`})];let v=t(o,`${c}-bounds-partials`,`float32`,p*4);return[n(o,{id:`${c}-bounds-blocks`,operation:l,variant:`${u}-blocks`,bindings:[...s.bindings,{name:`partials`,view:v,type:`f32`,access:`read_write`}],invocationCount:p*r,guardIndex:!1,declarations:m,body:`${_}
  if (localInvocationIndex == 0u) {
    let block = index / WORKGROUP_THREADS;
    partials[partialsOffset + block * 4u] = sharedMinima[0].x;
    partials[partialsOffset + block * 4u + 1u] = sharedMinima[0].y;
    partials[partialsOffset + block * 4u + 2u] = sharedMaxima[0].x;
    partials[partialsOffset + block * 4u + 3u] = sharedMaxima[0].y;
  }`}),n(o,{id:`${c}-bounds`,operation:l,variant:`${u}-merge`,bindings:[{name:`partials`,view:v,type:`f32`,access:`read`},{name:`boundsOut`,view:f,type:`f32`,access:`read_write`}],invocationCount:r,guardIndex:!1,declarations:`const PARTIAL_COUNT: u32 = ${p}u;
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
var<workgroup> sharedMinima: array<vec2f, ${r}>;
var<workgroup> sharedMaxima: array<vec2f, ${r}>;`,body:`// Exactly one workgroup is dispatched, so index == localInvocationIndex.
  var localMinimum = vec2f(FLOAT32_MAXIMUM);
  var localMaximum = vec2f(-FLOAT32_MAXIMUM);
  if (localInvocationIndex < PARTIAL_COUNT) {
    let base = partialsOffset + localInvocationIndex * 4u;
    localMinimum = vec2f(partials[base], partials[base + 1u]);
    localMaximum = vec2f(partials[base + 2u], partials[base + 3u]);
  }
  ${h}
  ${g(`boundsOut`)}`})]}export{o as t};