import{a as e,o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as n}from"./permutation-random-SiwCiDNI.js";import{t as r}from"./permutation-shuffle-zu7WXW69.js";import{E as i,f as a}from"./gpu-contributor-utils-BIwLQON5.js";import{r as o}from"./kernel-C3tfQOn4.js";var s={observed:0,pairCount:1,timeClosePairs:2,expected:3,permutationMean:4,permutationVariance:5,greaterCount:6,lesserCount:7,pseudoPGreater:8,pseudoPLesser:9,pseudoPTwoSided:10,zScore:11};function c(e){let{seed:t,permutationCount:n}=e;if(!Number.isSafeInteger(t)||t<0)throw Error(`space-time seed must be a non-negative safe integer`);a(`space-time`,`permutationCount`,n,1,2**31-1);let r=e.timeThreshold??0;if(Number.isNaN(r)||r<0)throw Error(`space-time timeThreshold must be non-negative`);let i=new Uint32Array(4);return i[0]=t%2**32,i[1]=Math.floor(t/2**32),i[2]=n,i[3]=new Uint32Array(new Float32Array([r]).buffer)[0],i}function l(e,t,n,r){c(t);let i=Math.min(t.permutationCount,n);return{alternative:`greater`,requestedPermutationCount:i,...r===void 0?{permutationCount:i,stoppingRule:{method:`fixed-count`}}:{stoppingRule:{method:`besag-clifford`,stopAfterExceedances:r}},seed:t.seed,multipleTesting:`none`,testedRowsDefinition:e===`knox`?`all unordered event pairs listed in the spatial pair list, counted once`:`all unordered event pairs listed in the distance band, counted once`}}var u=64,d={cursor:0,exceedances:1,stopIndex:2,shiftDistance:4,shiftTime:5};function f(e){let t=64;for(;Math.ceil(e/t)>256;)t*=2;return t}function p(e,t,n,r){if(t===void 0){if(r)throw Error(`${e} output.permutationCount needs stopAfterExceedances`)}else{if(a(e,`stopAfterExceedances`,t,1,n),!r)throw Error(`${e} stopAfterExceedances needs output.permutationCount`);if(i(r,[`uint32`],`${e} output.permutationCount`),r.length<1)throw Error(`${e} output.permutationCount needs 1 row`)}}function m(n,r){let{id:i,operation:a,state:o,statistics:s,parameters:c,common:l}=r,p=f(r.slotCount),m=Math.ceil(r.slotCount/p),h=e(n,{id:`${i}-stop-dispatch`,operation:a,shapes:[[p*r.groupsPerSlot*r.workgroupSize,r.workgroupSize],[p,u]]}),g=e=>({name:`state`,view:o,type:`u32`,access:e}),_=r.statisticType===`u32`?`0u`:`0.0`,v=(e,t)=>{let{condition:n,extraResources:r}=h.getGate(t);return{id:e,condition:{...n,id:`${e}-gate`},extraResources:r}},y=d,b=[t(n,{id:`${i}-stop-reset`,operation:a,variant:`stop-reset`,bindings:[g(`read_write`),h.binding],invocationCount:1,body:`state[stateOffset + ${y.cursor}u] = 0u;
  state[stateOffset + ${y.exceedances}u] = 0u;
  state[stateOffset + ${y.stopIndex}u] = 0u;
  ${h.armWGSL}`})];for(let e=0;e<m;e++)b.push(r.createPartialsNode(v(`${i}-batch-${e}-partials`,0)),r.createReduceNode(v(`${i}-batch-${e}-reduce`,1)),t(n,{id:`${i}-batch-${e}-scan`,operation:a,variant:`stop-scan`,bindings:[{name:`parameters`,view:c,type:`u32`,access:`read`},{name:`statistics`,view:s,type:r.statisticType,access:`read`},g(`read_write`),h.binding],invocationCount:1,declarations:`${l}
const BATCH_SIZE: u32 = ${p}u;
const STOP_AFTER: u32 = ${r.stopAfterExceedances}u;`,body:`let permutations = readPermutationCount();
  let cursor = state[stateOffset + ${y.cursor}u];
  if (state[stateOffset + ${y.stopIndex}u] != 0u || cursor * BATCH_SIZE > permutations) {
    return;
  }
  let observed = statistics[statisticsOffset];
  var exceedances = state[stateOffset + ${y.exceedances}u];
  var stopIndex = 0u;
  for (var offset = 0u; offset < BATCH_SIZE; offset++) {
    let slot = cursor * BATCH_SIZE + offset;
    if (slot > permutations) {
      break;
    }
    if (slot > 0u && statistics[statisticsOffset + slot] >= observed) {
      exceedances++;
      if (exceedances == STOP_AFTER) {
        stopIndex = slot;
        break;
      }
    }
  }
  state[stateOffset + ${y.exceedances}u] = exceedances;
  state[stateOffset + ${y.stopIndex}u] = stopIndex;
  state[stateOffset + ${y.cursor}u] = cursor + 1u;
  if (stopIndex != 0u || (cursor + 1u) * BATCH_SIZE > permutations) {
    ${h.disarmWGSL}
  }`}));return b.push(t(n,{id:`${i}-stop-finalize`,operation:a,variant:`stop-finalize`,bindings:[{name:`parameters`,view:c,type:`u32`,access:`read`},g(`read`),{name:`statistics`,view:s,type:r.statisticType,access:`read_write`}],invocationCount:s.length,declarations:l,body:`let stopIndex = state[stateOffset + ${y.stopIndex}u];
  let used = select(readPermutationCount(), stopIndex, (stopIndex != 0u));
  if (index > used) {
    statistics[statisticsOffset + index] = ${_};
  }`})),b}function h(e,t){return o(e,`${t}-stop-state`,`uint32`,8)}var g=2**16,_=2**24-1;function v(e){return Math.ceil(e/64)}function y(e,t=256){let n=Math.max(1,Math.min(256,t,e));return[n,Math.ceil(e/n)]}function b(e,n){return t(e,{id:`${n.id}-pair-mask`,operation:n.operation,variant:`pair-mask`,bindings:[{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:n.neighbors,type:`u32`,access:`read`},{name:`pairMask`,view:n.pairMask,type:`u32`,access:`read_write`}],invocationCount:n.rows,declarations:`const ROWS: u32 = ${n.rows}u;`,body:`let start = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  for (var entry = start; entry < end; entry++) {
    let other = neighbors[neighborsOffset + entry];
    var counted = other < ROWS && other != index;
    if (counted) {
      for (var earlier = start; earlier < entry; earlier++) {
        if (neighbors[neighborsOffset + earlier] == other) {
          counted = false;
          break;
        }
      }
    }
    if (counted && other < index) {
      for (var mirror = offsets[offsetsOffset + other]; mirror < offsets[offsetsOffset + other + 1u]; mirror++) {
        if (neighbors[neighborsOffset + mirror] == index) {
          counted = false;
          break;
        }
      }
    }
    pairMask[pairMaskOffset + entry] = select(0u, 1u, counted);
  }`})}function x(e,t,r,i){return`
${n}
const ROWS: u32 = ${e}u;
const BLOCKS: u32 = ${r}u;
const ROWS_PER_BLOCK: u32 = ${i}u;
const MAXIMUM_PERMUTATIONS: u32 = ${t}u;
fn readSeedKey() -> vec2<u32> {
  return vec2<u32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
}
fn readPermutationCount() -> u32 {
  return clamp(parameters[parametersOffset + 2u], 1u, MAXIMUM_PERMUTATIONS);
}
fn readTimeThreshold() -> f32 {
  return bitcast<f32>(parameters[parametersOffset + 3u]);
}
`}function S(e,t){return`
${r(t)}
const SPACE_TIME_WORKGROUP_SIZE: u32 = 64u;
const GROUPS_PER_SLOT: u32 = ${v(e)}u;
fn getPermutedRow(row: u32, slot: u32) -> u32 {
  if (slot == 0u) {
    return row;
  }
  return getSwapOrNotPermutationIndex(row, ROWS);
}
`}var C=`let workgroup = index / SPACE_TIME_WORKGROUP_SIZE;
  let slot = workgroup / GROUPS_PER_SLOT;
  let block = (workgroup - slot * GROUPS_PER_SLOT) * SPACE_TIME_WORKGROUP_SIZE + localInvocationIndex;
  let permutations = readPermutationCount();
  if (slot > 0u && slot <= permutations) {
    loadSwapOrNotRounds(readSeedKey(), slot, ROWS, localInvocationIndex, SPACE_TIME_WORKGROUP_SIZE);
  }
  workgroupBarrier();
  let runsSlot = slot <= permutations && block < BLOCKS;`,w=`let workgroup = index / SPACE_TIME_WORKGROUP_SIZE;
  let localSlot = workgroup / GROUPS_PER_SLOT;
  let slot = state[stateOffset] * BATCH_SIZE + localSlot;
  let block = (workgroup - localSlot * GROUPS_PER_SLOT) * SPACE_TIME_WORKGROUP_SIZE + localInvocationIndex;
  let permutations = readPermutationCount();
  if (slot > 0u && slot <= permutations) {
    loadSwapOrNotRounds(readSeedKey(), slot, ROWS, localInvocationIndex, SPACE_TIME_WORKGROUP_SIZE);
  }
  workgroupBarrier();
  let runsSlot = slot <= permutations && block < BLOCKS;`;function T(e,t){let n=s,r=t!==void 0,i=t=>e===`u32`?`f32(${t})`:t;return`
  ${r?`let stopIndex = state[stateOffset + ${d.stopIndex}u];
  let permutations = select(readPermutationCount(), stopIndex, (stopIndex != 0u));
  permutationsUsed[permutationsUsedOffset] = permutations;`:`let permutations = readPermutationCount();`}
  let observed = statistics[statisticsOffset];
  var greater = 0u;
  var lesser = 0u;
  var sumDeviation = 0.0;
  for (var slot = 1u; slot <= permutations; slot++) {
    let value = statistics[statisticsOffset + slot];
    if (value >= observed) { greater++; }
    if (value <= observed) { lesser++; }
    sumDeviation += ${e===`u32`?`f32(i32(value) - i32(observed))`:`value - observed`};
  }
  let permutationCount = f32(permutations);
  let meanDeviation = sumDeviation / permutationCount;
  var sumSquares = 0.0;
  for (var slot = 1u; slot <= permutations; slot++) {
    let value = statistics[statisticsOffset + slot];
    let deviation = ${e===`u32`?`f32(i32(value) - i32(observed))`:`value - observed`} - meanDeviation;
    sumSquares += deviation * deviation;
  }
  let variance = select(0.0, sumSquares / (permutationCount - 1.0), permutations > 1u);
  let mean = ${i(`observed`)} + meanDeviation;
  let pGreater = ${r?`select(f32(greater + 1u) / (permutationCount + 1.0), min(1.0, ${t}.0 / f32(max(stopIndex, 1u))), (stopIndex != 0u))`:`f32(greater + 1u) / (permutationCount + 1.0)`};
  let pLesser = f32(lesser + 1u) / (permutationCount + 1.0);
  summary[summaryOffset + ${n.observed}u] = ${i(`observed`)};
  summary[summaryOffset + ${n.pairCount}u] = observedPairs;
  summary[summaryOffset + ${n.timeClosePairs}u] = timeClosePairs;
  summary[summaryOffset + ${n.expected}u] = expected;
  summary[summaryOffset + ${n.permutationMean}u] = mean;
  summary[summaryOffset + ${n.permutationVariance}u] = variance;
  summary[summaryOffset + ${n.greaterCount}u] = f32(greater);
  summary[summaryOffset + ${n.lesserCount}u] = f32(lesser);
  summary[summaryOffset + ${n.pseudoPGreater}u] = pGreater;
  summary[summaryOffset + ${n.pseudoPLesser}u] = pLesser;
  summary[summaryOffset + ${n.pseudoPTwoSided}u] = min(1.0, 2.0 * min(pGreater, pLesser));
  let zScore = (${i(`observed`)} - mean) / sqrt(variance);
  summary[summaryOffset + ${n.zScore}u] = select(bitcast<f32>(parameters[parametersOffset + 3u] | 0x7fc00000u), zScore, variance > 0.0);
`}export{l as _,b as a,v as c,d,h as f,s as g,p as h,C as i,S as l,f as m,g as n,y as o,m as p,_ as r,x as s,w as t,T as u,c as v};