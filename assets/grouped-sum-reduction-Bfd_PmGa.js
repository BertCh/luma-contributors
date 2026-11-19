import{t as e}from"./common-CshkX-d_.js";import{r as t}from"./kernel-C3tfQOn4.js";import{f as n}from"./gpu-contributor-utils-BIwLQON5.js";import{o as r}from"./wgsl-kernel-nodes-DagJklH2.js";var i=256,a=64,o=1024,s=4;function c(t,n){return{partial:[e(Math.ceil(t/i),1,o)*i,i],fold:[n*i,i]}}function l(c,l){let{id:u,operation:d,rows:f,groupCount:p,outputs:m}=l,h=m.length;n(u,`groupCount`,p,1,256),n(u,`output count`,h,1,s);let g=e(Math.ceil(f/i),1,o),_=Math.ceil(f/g/i)*i,v=Math.min(a,Math.floor(i/p)),y=p*v,b=Array.from({length:h},(e,t)=>t),x=t(c,`${u}-partial-sums`,`float32`,h*p*g),S=l.countOutput?t(c,`${u}-partial-counts`,`uint32`,p*g):void 0,C=!!S,w=r(c,{id:`${u}-partial`,operation:d,variant:`grouped-partial`,bindings:[...l.bindings,{name:`partialSums`,view:x,type:`f32`,access:`read_write`},...S?[{name:`partialCounts`,view:S,type:`u32`,access:`read_write`}]:[]],workgroupSize:i,invocationCount:g*i,guardIndex:!1,...l.gates?.partial,declarations:`${l.declarations??``}
const GROUP_COUNT: u32 = ${p}u;
const NO_GROUP: u32 = 0xffffffffu;
const ROWS: u32 = ${f}u;
const CHUNK_SIZE: u32 = ${_}u;
const CHUNK_COUNT: u32 = ${g}u;
const TILE_COUNT: u32 = ${_/i}u;
const SLICES: u32 = ${v}u;
const LANES: u32 = ${y}u;
var<workgroup> tileGroups: array<u32, ${i}>;
var<workgroup> tileValues: array<f32, ${i*h}>;
var<workgroup> laneSums: array<f32, ${i*h}>;
var<workgroup> laneCounts: array<u32, ${i}>;`,body:`let chunk = index / ${i}u;
  let begin = chunk * CHUNK_SIZE;
  let end = min(begin + CHUNK_SIZE, ROWS);
  let lane = localInvocationIndex;
  let laneGroup = lane % GROUP_COUNT;
  let laneSlice = lane / GROUP_COUNT;
  ${b.map(e=>`var sum${e} = 0.0;`).join(`
  `)}
  var count = 0u;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let row = begin + tile * ${i}u + lane;
    var group = NO_GROUP;
    ${b.map(e=>`var value${e} = 0.0;`).join(`
    `)}
    if (row < end) {
      ${l.rowSnippet}
    }
    if (group >= GROUP_COUNT) {
      group = NO_GROUP;
    }
    tileGroups[lane] = group;
    ${b.map(e=>`tileValues[${e*i}u + lane] = value${e};`).join(`
    `)}
    workgroupBarrier();
    if (lane < LANES) {
      // Lane (slice, group) owns tile entries slice, slice + SLICES, ...: a fixed order.
      for (var entry = laneSlice; entry < ${i}u; entry += SLICES) {
        if (tileGroups[entry] == laneGroup) {
          count++;
          ${b.map(e=>`sum${e} += tileValues[${e*i}u + entry];`).join(`
          `)}
        }
      }
    }
    workgroupBarrier();
  }
  ${b.map(e=>`laneSums[${e*i}u + lane] = sum${e};`).join(`
  `)}
  laneCounts[lane] = count;
  workgroupBarrier();
  if (lane < GROUP_COUNT) {
    var chunkCountTotal = 0u;
    ${b.map(e=>`var chunkSum${e} = 0.0;`).join(`
    `)}
    for (var slice = 0u; slice < SLICES; slice++) {
      chunkCountTotal += laneCounts[slice * GROUP_COUNT + lane];
      ${b.map(e=>`chunkSum${e} += laneSums[${e*i}u + slice * GROUP_COUNT + lane];`).join(`
      `)}
    }
    ${b.map(e=>`partialSums[partialSumsOffset + (${e}u * GROUP_COUNT + lane) * CHUNK_COUNT + chunk] = chunkSum${e};`).join(`
    `)}
    ${C?`partialCounts[partialCountsOffset + lane * CHUNK_COUNT + chunk] = chunkCountTotal;`:``}
  }`}),T=l.finish;return[w,r(c,{id:`${u}-${l.foldSuffix??`fold`}`,operation:d,variant:`grouped-fold`,bindings:[{name:`partialSums`,view:x,type:`f32`,access:`read`},...S?[{name:`partialCounts`,view:S,type:`u32`,access:`read`}]:[],...m.map((e,t)=>({name:`output${t}`,view:e,type:`f32`,access:`read_write`})),...l.countOutput?[{name:`countOutput`,view:l.countOutput,type:`u32`,access:`read_write`}]:[],...T?.bindings??[]],workgroupSize:i,invocationCount:p*i,guardIndex:!1,...l.gates?.fold,declarations:`${T?.declarations??``}
const GROUP_COUNT: u32 = ${p}u;
const CHUNK_COUNT: u32 = ${g}u;
var<workgroup> reducedSums: array<f32, ${i*h}>;
var<workgroup> reducedCounts: array<u32, ${i}>;`,body:`let group = index / ${i}u;
  let lane = localInvocationIndex;
  ${b.map(e=>`var sum${e} = 0.0;`).join(`
  `)}
  var count = 0u;
  for (var chunk = lane; chunk < CHUNK_COUNT; chunk += ${i}u) {
    ${b.map(e=>`sum${e} += partialSums[partialSumsOffset + (${e}u * GROUP_COUNT + group) * CHUNK_COUNT + chunk];`).join(`
    `)}
    ${C?`count += partialCounts[partialCountsOffset + group * CHUNK_COUNT + chunk];`:``}
  }
  ${b.map(e=>`reducedSums[${e*i}u + lane] = sum${e};`).join(`
  `)}
  reducedCounts[lane] = count;
  workgroupBarrier();
  for (var stride = ${i/2}u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      ${b.map(e=>`reducedSums[${e*i}u + lane] += reducedSums[${e*i}u + lane + stride];`).join(`
      `)}
      reducedCounts[lane] += reducedCounts[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    ${b.map(e=>`sum${e} = reducedSums[${e*i}u];`).join(`
    `)}
    count = reducedCounts[0];
    ${b.map(e=>`output${e}[output${e}Offset + group] = sum${e};`).join(`
    `)}
    ${l.countOutput?`countOutput[countOutputOffset + group] = count;`:``}
    ${T?.body??``}
  }`})]}export{l as n,c as t};