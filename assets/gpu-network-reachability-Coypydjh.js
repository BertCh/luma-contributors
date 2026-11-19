import{a as e,o as t,s as n}from"./network-view-utils-B1U8oNKx.js";import{t as r}from"./construction-snapshot-DdkX9Pc0.js";import{r as i}from"./kernel-C3tfQOn4.js";import{t as a}from"./gpu-histogram-fxACbUIT.js";import{A as o,E as s,S as c,f as l,k as u,u as d}from"./gpu-contributor-utils-BIwLQON5.js";import{o as f}from"./wgsl-kernel-nodes-DagJklH2.js";import{Q as p}from"./dist-BusamGlx.js";var m=4;function h(e){return{controlWordCount:4+e+1,dispatchSlotCount:Math.floor(e/2)+1}}function g(e,t,n){let r=p.STORAGE|p.INDIRECT;return{nodeCount:n.nodeCount,queues:i(e,`${t}-queues`,`uint32`,2*n.nodeCount),queuedRound:i(e,`${t}-queued-round`,`uint32`,n.nodeCount),control:i(e,`${t}-control`,`uint32`,n.controlWordCount),dispatchEven:i(e,`${t}-dispatch-even`,`uint32`,n.dispatchSlotCount*64,r),dispatchOdd:i(e,`${t}-dispatch-odd`,`uint32`,n.dispatchSlotCount*64,r)}}function _(e){return e.stampBase+e.maxRounds+1}function v(e,t){return 256*(e.dispatchSlotOffset+Math.floor(t/2))}function y(e,t,n,r){let i=r%2==1?t.dispatchOdd:t.dispatchEven,a=e.createDataView(i.buffer,{format:`uint32`,length:64,byteOffset:i.byteOffset+v(n,r)});return[{name:`queues`,view:t.queues,type:`u32`,access:`read_write`},{name:`queuedRound`,view:t.queuedRound,type:`atomic<u32>`,access:`read_write`},{name:`control`,view:t.control,type:`atomic<u32>`,access:`read_write`},{name:`nextDispatch`,view:a,type:`atomic<u32>`,access:`read_write`}]}function b(e,t,n){return`
const FRONTIER_MAXIMUM_X: u32 = ${n}u;
const FRONTIER_TRUNCATED_WORD: u32 = ${e.controlWordOffset+0}u;
const FRONTIER_LIMIT_WORD: u32 = ${e.controlWordOffset+1}u;
const FRONTIER_PARAMETER_0_WORD: u32 = ${e.controlWordOffset+2}u;
const FRONTIER_PARAMETER_1_WORD: u32 = ${e.controlWordOffset+3}u;
const FRONTIER_ROUND_COUNTS_WORD: u32 = ${e.controlWordOffset+4}u;
const FRONTIER_STAMP_BASE: u32 = ${e.stampBase}u;
const FRONTIER_QUEUE_NODE_COUNT: u32 = ${t}u;
const FRONTIER_SLOT_ROUND_WORD: u32 = 3u;

// Round index (within the phase) of the bound nextDispatch slot.
fn frontierNextRound() -> u32 {
  return atomicLoad(&nextDispatch[nextDispatchOffset + FRONTIER_SLOT_ROUND_WORD]);
}

fn frontierPush(targetNode: u32) {
  let nextRound = frontierNextRound();
  if (nextRound >= atomicLoad(&control[controlOffset + FRONTIER_LIMIT_WORD])) {
    atomicStore(&control[controlOffset + FRONTIER_TRUNCATED_WORD], 1u);
    return;
  }
  let nextStamp = FRONTIER_STAMP_BASE + nextRound + 1u;
  if (atomicMax(&queuedRound[queuedRoundOffset + targetNode], nextStamp) >= nextStamp) {
    return;
  }
  let slot = atomicAdd(&control[controlOffset + FRONTIER_ROUND_COUNTS_WORD + nextRound], 1u);
  queues[queuesOffset + (nextRound & 1u) * FRONTIER_QUEUE_NODE_COUNT + slot] = targetNode;
  if (slot % 256u == 0u && slot / 256u < FRONTIER_MAXIMUM_X) {
    atomicAdd(&nextDispatch[nextDispatchOffset], 1u);
  }
}`}function x(e,t){let{controlWordCount:n,dispatchSlotCount:r}=h(e.maxRounds);return`if (index < ${n}u) {
    var controlValue = 0u;
    if (index == 1u) { controlValue = ${t.limit}; }
    if (index == 2u) { controlValue = ${t.parameter0??`0u`}; }
    if (index == 3u) { controlValue = ${t.parameter1??`0u`}; }
    control[controlOffset + ${e.controlWordOffset}u + index] = controlValue;
  }
  if (index < ${r*m}u) {
    let slot = index / ${m}u;
    let slotWord = index % ${m}u;
    let word = (${e.dispatchSlotOffset}u + slot) * 64u + slotWord;
    // x starts at 0 (pushes grow it), y and z at 1; word 3 is the slot's round index.
    var evenValue = select(1u, 0u, slotWord == 0u);
    var oddValue = evenValue;
    if (slotWord == 3u) {
      evenValue = 2u * slot;
      oddValue = 2u * slot + 1u;
    }
    dispatchEven[dispatchEvenOffset + word] = evenValue;
    dispatchOdd[dispatchOddOffset + word] = oddValue;
  }`}function S(e){let{controlWordCount:t,dispatchSlotCount:n}=h(e.maxRounds);return Math.max(t,n*m)}function C(e){let t=`controlOffset + ${e.controlWordOffset}u`;return`let frontierLimit = control[${t} + 1u];
  var frontierRoundCount = 0u;
  for (var roundIndex = 0u; roundIndex < ${e.maxRounds}u; roundIndex++) {
    if (roundIndex >= frontierLimit || control[${t} + 4u + roundIndex] == 0u) {
      break;
    }
    frontierRoundCount++;
  }
  frontierRoundCount = max(frontierRoundCount, min(frontierLimit, 1u));
  let frontierConverged = select(
    0u,
    1u,
    frontierLimit > 0u && control[${t} + 0u] == 0u
  );`}function w(e,t){let{state:n,phase:r,round:i,localIterations:a}=t,{nodeCount:o}=n,s=y(e,n,r,i+1),l=[...t.bindings,...s];c(t.operation,`${t.id} storage bindings`,l.length,8);let u=e.device.limits.maxComputeWorkgroupsPerDimension,d=i%2==0?n.dispatchEven:n.dispatchOdd,p=v(r,i);return f(e,{id:t.id,operation:t.operation,variant:t.variant,bindings:l,invocationCount:o,guardIndex:!1,condition:{id:`${t.id}-gate`,source:`gpu`,mode:`indirect`,buffer:d.buffer,byteOffset:d.byteOffset+p},extraResources:[{buffer:d,usage:`indirect`}],declarations:`${t.declarations??``}
${b(r,o,u)}
const FRONTIER_LOCAL_ITERATIONS: u32 = ${a}u;

var<workgroup> frontierLocalQueue: array<u32, 512>;
var<workgroup> frontierLocalCount: array<atomic<u32>, 2>;
var<workgroup> frontierSharedCount: u32;
var<private> frontierPushToLocal: bool;
var<private> frontierLocalWriteHalf: u32;

fn push(targetNode: u32) {
  if (frontierPushToLocal) {
    let slot = atomicAdd(&frontierLocalCount[frontierLocalWriteHalf], 1u);
    if (slot < 256u) {
      frontierLocalQueue[frontierLocalWriteHalf * 256u + slot] = targetNode;
      return;
    }
  }
  frontierPush(targetNode);
}

${t.visitSource}`,body:`// This round's index is one below the round of the slot it pushes into (rule 12: it is data).
  let currentRound = frontierNextRound() - 1u;
  let currentQueueBase = (currentRound & 1u) * FRONTIER_QUEUE_NODE_COUNT;
  if (localInvocationIndex == 0u) {
    frontierSharedCount = atomicLoad(&control[controlOffset + FRONTIER_ROUND_COUNTS_WORD + currentRound]);
  }
  let queueLength = workgroupUniformLoad(&frontierSharedCount);
  let dispatchedGroupCount = min(
    (queueLength + 255u) / 256u,
    FRONTIER_MAXIMUM_X
  );
  for (var chunk = workgroupId.x; chunk * 256u < queueLength; chunk += dispatchedGroupCount) {
    // Hop 0 reads the global queue chunk; hop k reads the local queue written by hop k - 1.
    frontierPushToLocal = FRONTIER_LOCAL_ITERATIONS > 1u;
    frontierLocalWriteHalf = 1u;
    let queueIndex = chunk * 256u + localInvocationIndex;
    if (queueIndex < queueLength) {
      visit(queues[queuesOffset + currentQueueBase + queueIndex]);
    }
    for (var hop = 1u; hop < FRONTIER_LOCAL_ITERATIONS; hop++) {
      workgroupBarrier();
      let readHalf = hop & 1u;
      if (localInvocationIndex == 0u) {
        frontierSharedCount = min(atomicLoad(&frontierLocalCount[readHalf]), 256u);
        atomicStore(&frontierLocalCount[readHalf], 0u);
      }
      let localLength = workgroupUniformLoad(&frontierSharedCount);
      if (localLength == 0u) { break; }
      // The last hop pushes straight to the next global round.
      frontierPushToLocal = hop + 1u < FRONTIER_LOCAL_ITERATIONS;
      frontierLocalWriteHalf = (hop + 1u) & 1u;
      if (localInvocationIndex < localLength) {
        visit(frontierLocalQueue[readHalf * 256u + localInvocationIndex]);
      }
    }
    workgroupBarrier();
  }
  ${t.epilogueSource??``}`})}function T(e){let t=r(e,[`id`,`topology`,`startNodes`,`startCosts`,`startCount`,`maximumIterations`,`localIterations`,`maximumTieIterations`,`activeIterations`,`costLimit`,`laneCount`,`costBreaks`,`output`,`nodeCount`,`costBreakCount`]),n=r(t.output,[`costs`,`predecessors`,`bands`,`bandCounts`,`unresolvedCount`,`status`]);return Object.freeze({...t,topology:t.topology&&r(t.topology,[`offsets`,`neighbors`,`weights`,`nodePositions`]),output:Object.freeze({...n,...n.status?{status:r(n.status,[`converged`,`iterationCount`,`iterationLimitReached`,`relaxationCount`,`reachedEdgeCount`])}:{}})})}var E=`GPUNetworkReachability`,D=`
const INFINITY_BITS: u32 = 0x7f800000u;
const NONE: u32 = 0xffffffffu;`,O=`
var<workgroup> counterLow: atomic<u32>;
var<workgroup> counterHigh: atomic<u32>;

// Adds to this workgroup's 64-bit partial count.
fn addCount(count: u32) {
  if (count == 0u) { return; }
  let previous = atomicAdd(&counterLow, count);
  if (previous + count < previous) { atomicAdd(&counterHigh, 1u); }
}`;function k(e){return`if (localInvocationIndex == 0u) {
    let partialLow = atomicLoad(&counterLow);
    let partialHigh = atomicLoad(&counterHigh);
    if (partialLow != 0u || partialHigh != 0u) {
      let previous = atomicAdd(&control[controlOffset + ${e}u], partialLow);
      var carry = partialHigh;
      if (previous + partialLow < previous) { carry += 1u; }
      if (carry != 0u) { atomicAdd(&control[controlOffset + ${e+1}u], carry); }
    }
  }`}function A(e,t){return t?`min(activeIterations[activeIterationsOffset], ${e}u)`:`${e}u`}function j(e,t){let{state:n,phase:r}=t,i=[{name:`costBits`,view:t.costs,type:`u32`,access:`read_write`},{name:`queuedRound`,view:n.queuedRound,type:`u32`,access:`read_write`},{name:`control`,view:n.control,type:`u32`,access:`read_write`},{name:`dispatchEven`,view:n.dispatchEven,type:`u32`,access:`read_write`},{name:`dispatchOdd`,view:n.dispatchOdd,type:`u32`,access:`read_write`}];return t.predecessors&&i.push({name:`predecessors`,view:t.predecessors,type:`u32`,access:`read_write`}),t.activeIterations&&i.push({name:`activeIterations`,view:t.activeIterations,type:`u32`,access:`read`}),t.costLimit&&i.push({name:`costLimit`,view:t.costLimit,type:`f32`,access:`read`}),f(e,{id:t.id,operation:E,variant:`initialize`,bindings:i,invocationCount:Math.max(t.nodeCount,S(r),t.counterWordOffset===void 0?0:4),declarations:`${D}
const NODE_COUNT: u32 = ${t.nodeCount}u;`,body:`${x(r,{limit:A(r.maxRounds,!!t.activeIterations),parameter0:t.costLimit?`bitcast<u32>(costLimit[costLimitOffset])`:void 0})}
  ${t.counterWordOffset===void 0?``:`if (index < 4u) { control[controlOffset + ${t.counterWordOffset}u + index] = 0u; }`}
  if (index < NODE_COUNT) {
    costBits[costBitsOffset + index] = INFINITY_BITS;
    // Stamps never need clearing within one encoding, but a previous encoding left them dirty.
    queuedRound[queuedRoundOffset + index] = 0u;
    ${t.predecessors?`predecessors[predecessorsOffset + index] = NONE;`:``}
  }`})}function M(e){return e===void 0?``:typeof e==`number`?`if (index >= ${e}u) { return; }`:`if (index >= sourceCount[sourceCountOffset]) { return; }`}function N(e,t){let n=[{name:`sources`,view:t.sources,type:`u32`,access:`read`}];return t.sourceCosts&&n.push({name:`sourceCosts`,view:t.sourceCosts,type:`f32`,access:`read`}),typeof t.sourceCount==`object`&&n.push({name:`sourceCount`,view:t.sourceCount,type:`u32`,access:`read`}),n.push({name:`costBits`,view:t.costs,type:`atomic<u32>`,access:`read_write`},...y(e,t.state,t.phase,0)),f(e,{id:t.id,operation:E,variant:`seed`,bindings:n,invocationCount:t.sources.length,declarations:`${D}
const NODE_COUNT: u32 = ${t.nodeCount}u;
${b(t.phase,t.nodeCount,e.device.limits.maxComputeWorkgroupsPerDimension)}`,body:`${M(t.sourceCount)}
  let node = sources[sourcesOffset + index];
  if (node >= NODE_COUNT) { return; }
  let cost = ${t.sourceCosts?`sourceCosts[sourceCostsOffset + index]`:`0.0`};
  if (!(cost >= 0.0)) { return; }
  ${t.hasCostLimit?`if (cost > bitcast<f32>(atomicLoad(&control[controlOffset + FRONTIER_PARAMETER_0_WORD]))) { return; }`:``}
  // Fold -0.0 so the u32 bit order of non-negative floats matches numeric order.
  let bits = select(bitcast<u32>(cost), 0u, cost == 0.0);
  if (bits >= INFINITY_BITS) { return; }
  atomicMin(&costBits[costBitsOffset + node], bits);
  frontierPush(node);`})}function P(e,t){let{csr:n,laneNodeCount:r,counterWordOffset:i}=t,a=i!==void 0;return w(e,{id:t.id,operation:E,variant:`${r===void 0?`relax`:`relax-lanes`}${a?`-counted`:``}`,epilogueSource:a?k(i):void 0,state:t.state,phase:t.phase,round:t.round,localIterations:t.localIterations,bindings:[{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:n.neighbors,type:`u32`,access:`read`},{name:`weights`,view:n.weights,type:`f32`,access:`read`},{name:`costBits`,view:t.costs,type:`atomic<u32>`,access:`read_write`}],declarations:`${D}
const NODE_COUNT: u32 = ${r??t.nodeCount}u;
const EDGE_COUNT: u32 = ${n.neighbors.length}u;${a?O:``}`,visitSource:`fn visit(node: u32) {
  let sourceCost = bitcast<f32>(atomicLoad(&costBits[costBitsOffset + node]));
  ${r===void 0?`let csrNode = node; let laneBase = 0u;`:`let laneBase = (node / NODE_COUNT) * NODE_COUNT; let csrNode = node - laneBase;`}
  let firstEdge = min(offsets[offsetsOffset + csrNode], EDGE_COUNT);
  let lastEdge = min(offsets[offsetsOffset + csrNode + 1u], EDGE_COUNT);
  ${t.hasCostLimit?`let costLimitValue = bitcast<f32>(atomicLoad(&control[controlOffset + FRONTIER_PARAMETER_0_WORD]));`:``}
  ${a?`addCount(select(0u, lastEdge - firstEdge, lastEdge > firstEdge));`:``}
  for (var edgeIndex = firstEdge; edgeIndex < lastEdge; edgeIndex++) {
    let targetCsrNode = neighbors[neighborsOffset + edgeIndex];
    let weight = weights[weightsOffset + edgeIndex];
    if (targetCsrNode >= NODE_COUNT || !(weight >= 0.0)) { continue; }
    let targetNode = laneBase + targetCsrNode;
    let candidate = sourceCost + weight;
    ${t.hasCostLimit?`if (candidate > costLimitValue) { continue; }`:``}
    let candidateBits = select(bitcast<u32>(candidate), 0u, candidate == 0.0);
    if (candidateBits >= INFINITY_BITS) { continue; }
    if (candidateBits < atomicLoad(&costBits[costBitsOffset + targetNode])) {
      if (atomicMin(&costBits[costBitsOffset + targetNode], candidateBits) > candidateBits) {
        push(targetNode);
      }
    }
  }
}`})}function F(e,t){let n=[{name:`control`,view:t.state.control,type:`u32`,access:`read`}],r=[];if(t.counters?.relaxationCount){n.push({name:`relaxationCount`,view:t.counters.relaxationCount,type:`u32`,access:`read_write`});let e=t.counters.wordOffset;r.push(`relaxationCount[relaxationCountOffset] = control[controlOffset + ${e}u];`,`relaxationCount[relaxationCountOffset + 1u] = control[controlOffset + ${e+1}u];`)}if(t.counters?.reachedEdgeCount){n.push({name:`reachedEdgeCount`,view:t.counters.reachedEdgeCount,type:`u32`,access:`read_write`});let e=t.counters.wordOffset+2;r.push(`reachedEdgeCount[reachedEdgeCountOffset] = control[controlOffset + ${e}u];`,`reachedEdgeCount[reachedEdgeCountOffset + 1u] = control[controlOffset + ${e+1}u];`)}return t.converged&&n.push({name:`converged`,view:t.converged,type:`u32`,access:`read_write`}),t.iterationCount&&n.push({name:`iterationCount`,view:t.iterationCount,type:`u32`,access:`read_write`}),f(e,{id:t.id,operation:E,variant:`finalize`,bindings:n,invocationCount:1,body:`${C(t.phase)}
  ${t.converged?`converged[convergedOffset] = frontierConverged${t.tiePhase?` & select(0u, 1u, control[controlOffset + ${t.tiePhase.controlWordOffset+0}u] == 0u)`:``};`:``}
  ${t.iterationCount?`iterationCount[iterationCountOffset] = frontierRoundCount;`:``}
  ${r.join(`
  `)}`})}function I(e,t){let{csr:n}=t;return f(e,{id:t.id,operation:E,variant:`reached-edges`,bindings:[{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:n.neighbors,type:`u32`,access:`read`},{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`control`,view:t.state.control,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.nodeCount,guardIndex:!1,declarations:`${D}
const NODE_COUNT: u32 = ${t.nodeCount}u;
const LANE_NODE_COUNT: u32 = ${t.laneNodeCount??t.nodeCount}u;
const EDGE_COUNT: u32 = ${n.neighbors.length}u;${O}`,body:`if (index < NODE_COUNT && bitcast<u32>(costs[costsOffset + index]) < INFINITY_BITS) {
    let csrNode = index % LANE_NODE_COUNT;
    let firstEdge = min(offsets[offsetsOffset + csrNode], EDGE_COUNT);
    let lastEdge = min(offsets[offsetsOffset + csrNode + 1u], EDGE_COUNT);
    var degree = 0u;
    for (var edgeIndex = firstEdge; edgeIndex < lastEdge; edgeIndex++) {
      if (neighbors[neighborsOffset + edgeIndex] < LANE_NODE_COUNT) { degree++; }
    }
    addCount(degree);
  }
  workgroupBarrier();
  ${k(t.counterWordOffset+2)}`})}function L(e,t){let{csr:n}=t,r=[{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:n.neighbors,type:`u32`,access:`read`},{name:`weights`,view:n.weights,type:`f32`,access:`read`},{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`predecessors`,view:t.predecessors,type:`atomic<u32>`,access:`read_write`}];return t.tieBits&&r.push({name:`tieBits`,view:t.tieBits,type:`atomic<u32>`,access:`read_write`}),t.labels&&r.push({name:`labels`,view:t.labels,type:`u32`,access:`read`}),f(e,{id:t.id,operation:E,variant:t.labels?`predecessors-labelled`:`predecessors`,bindings:r,invocationCount:t.nodeCount,declarations:`${D}
const NODE_COUNT: u32 = ${t.nodeCount}u;
const EDGE_COUNT: u32 = ${n.neighbors.length}u;`,body:`let sourceCost = costs[costsOffset + index];
  if (bitcast<u32>(sourceCost) >= INFINITY_BITS) { return; }
  let firstEdge = min(offsets[offsetsOffset + index], EDGE_COUNT);
  let lastEdge = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
  for (var edgeIndex = firstEdge; edgeIndex < lastEdge; edgeIndex++) {
    let targetNode = neighbors[neighborsOffset + edgeIndex];
    let weight = weights[weightsOffset + edgeIndex];
    if (targetNode >= NODE_COUNT || !(weight >= 0.0)) { continue; }
    let targetCost = costs[costsOffset + targetNode];
    if (bitcast<u32>(targetCost) >= INFINITY_BITS || sourceCost + weight != targetCost) { continue; }
    ${t.labels?`if (labels[labelsOffset + index] != labels[labelsOffset + targetNode]) { continue; }`:``}
    if (sourceCost < targetCost) {
      atomicMin(&predecessors[predecessorsOffset + targetNode], index);
    }${t.tieBits?` else if (sourceCost == targetCost) {
      atomicOr(&tieBits[tieBitsOffset + (edgeIndex >> 5u)], 1u << (edgeIndex & 31u));
    }`:``}
  }`})}function R(e,t){return f(e,{id:t.id,operation:E,variant:`bands`,bindings:[{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`thresholds`,view:t.thresholds,type:`f32`,access:`read`},{name:`bands`,view:t.bands,type:`u32`,access:`read_write`}],invocationCount:t.nodeCount,declarations:`${D}
const THRESHOLD_COUNT: u32 = ${t.thresholds.length}u;`,body:`let cost = costs[costsOffset + index];
  var band = NONE;
  if (bitcast<u32>(cost) < INFINITY_BITS) {
    var count = 0u;
    for (var thresholdIndex = 0u; thresholdIndex < THRESHOLD_COUNT; thresholdIndex++) {
      if (thresholds[thresholdsOffset + thresholdIndex] < cost) { count++; }
    }
    if (count < THRESHOLD_COUNT) { band = count; }
  }
  bands[bandsOffset + index] = band;`})}var z=`
fn isTieEdge(edgeIndex: u32) -> bool {
  return ((tieBits[tieBitsOffset + (edgeIndex >> 5u)] >> (edgeIndex & 31u)) & 1u) != 0u;
}

fn hasTieOutEdge(node: u32) -> bool {
  let firstEdge = min(offsets[offsetsOffset + node], EDGE_COUNT);
  let lastEdge = min(offsets[offsetsOffset + node + 1u], EDGE_COUNT);
  for (var edgeIndex = firstEdge; edgeIndex < lastEdge; edgeIndex++) {
    if (isTieEdge(edgeIndex)) { return true; }
  }
  return false;
}`;function B(e,t){let{state:n,tiePhase:r}=t,i=[{name:`levels`,view:t.levels,type:`u32`,access:`read_write`},{name:`tieBits`,view:t.tieBits,type:`u32`,access:`read_write`},{name:`control`,view:n.control,type:`u32`,access:`read_write`},{name:`dispatchEven`,view:n.dispatchEven,type:`u32`,access:`read_write`},{name:`dispatchOdd`,view:n.dispatchOdd,type:`u32`,access:`read_write`}];return t.unresolvedCount&&i.push({name:`unresolvedCount`,view:t.unresolvedCount,type:`u32`,access:`read_write`}),f(e,{id:t.id,operation:E,variant:`tie-initialize`,bindings:i,invocationCount:Math.max(t.nodeCount,t.tieBits.length,S(r)),declarations:`${D}
const NODE_COUNT: u32 = ${t.nodeCount}u;
const TIE_WORD_COUNT: u32 = ${t.tieBits.length}u;`,body:`${x(r,{limit:`${r.maxRounds}u`})}
  if (index < NODE_COUNT) { levels[levelsOffset + index] = NONE; }
  if (index < TIE_WORD_COUNT) { tieBits[tieBitsOffset + index] = 0u; }${t.unresolvedCount?`
  if (index == 0u) { unresolvedCount[unresolvedCountOffset] = 0u; }`:``}`})}function V(e,t){let n=[{name:`sources`,view:t.sources,type:`u32`,access:`read`}];return t.labels&&n.push({name:`labels`,view:t.labels,type:`u32`,access:`read`}),t.sourceCosts&&n.push({name:`sourceCosts`,view:t.sourceCosts,type:`f32`,access:`read`}),typeof t.sourceCount==`object`&&n.push({name:`sourceCount`,view:t.sourceCount,type:`u32`,access:`read`}),n.push({name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`levels`,view:t.levels,type:`u32`,access:`read_write`}),t.hasCostLimit&&n.push({name:`control`,view:t.state.control,type:`u32`,access:`read`}),f(e,{id:t.id,operation:E,variant:`tie-roots`,bindings:n,invocationCount:t.sources.length,declarations:`${D}
const NODE_COUNT: u32 = ${t.nodeCount}u;`,body:`${M(t.sourceCount)}
  let node = sources[sourcesOffset + index];
  if (node >= NODE_COUNT) { return; }
  let cost = ${t.sourceCosts?`sourceCosts[sourceCostsOffset + index]`:`0.0`};
  if (!(cost >= 0.0)) { return; }
  ${t.hasCostLimit?`if (cost > bitcast<f32>(control[controlOffset + ${t.phase.controlWordOffset+2}u])) { return; }`:``}
  let bits = select(bitcast<u32>(cost), 0u, cost == 0.0);
  if (bits >= INFINITY_BITS) { return; }
  ${t.labels?`if (labels[labelsOffset + node] != index) { return; }`:``}
  if (bits == bitcast<u32>(costs[costsOffset + node])) {
    levels[levelsOffset + node] = 0u;
  }`})}function H(e,t){return f(e,{id:t.id,operation:E,variant:`tie-seed`,bindings:[{name:`offsets`,view:t.offsets,type:`u32`,access:`read`},{name:`tieBits`,view:t.tieBits,type:`u32`,access:`read`},{name:`predecessors`,view:t.predecessors,type:`u32`,access:`read`},{name:`levels`,view:t.levels,type:`u32`,access:`read_write`},...y(e,t.state,t.tiePhase,0)],invocationCount:t.nodeCount,declarations:`${D}
const NODE_COUNT: u32 = ${t.nodeCount}u;
const EDGE_COUNT: u32 = ${t.edgeCount}u;
${z}
${b(t.tiePhase,t.nodeCount,e.device.limits.maxComputeWorkgroupsPerDimension)}`,body:`if (predecessors[predecessorsOffset + index] != NONE) {
    levels[levelsOffset + index] = 0u;
  }
  if (levels[levelsOffset + index] == 0u && hasTieOutEdge(index)) {
    frontierPush(index);
  }`})}function U(e,t){return w(e,{id:t.id,operation:E,variant:`tie-level`,state:t.state,phase:t.tiePhase,round:t.round,localIterations:t.localIterations,bindings:[{name:`offsets`,view:t.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:t.neighbors,type:`u32`,access:`read`},{name:`tieBits`,view:t.tieBits,type:`u32`,access:`read`},{name:`levels`,view:t.levels,type:`atomic<u32>`,access:`read_write`}],declarations:`${D}
const NODE_COUNT: u32 = ${t.nodeCount}u;
const EDGE_COUNT: u32 = ${t.edgeCount}u;
${z}`,visitSource:`fn visit(node: u32) {
  let level = atomicLoad(&levels[levelsOffset + node]);
  if (level == NONE) { return; }
  let nextLevel = level + 1u;
  let firstEdge = min(offsets[offsetsOffset + node], EDGE_COUNT);
  let lastEdge = min(offsets[offsetsOffset + node + 1u], EDGE_COUNT);
  for (var edgeIndex = firstEdge; edgeIndex < lastEdge; edgeIndex++) {
    if (!isTieEdge(edgeIndex)) { continue; }
    let targetNode = neighbors[neighborsOffset + edgeIndex];
    if (targetNode >= NODE_COUNT) { continue; }
    if (nextLevel < atomicLoad(&levels[levelsOffset + targetNode])) {
      if (atomicMin(&levels[levelsOffset + targetNode], nextLevel) > nextLevel && hasTieOutEdge(targetNode)) {
        push(targetNode);
      }
    }
  }
}`})}function W(e,t){return f(e,{id:t.id,operation:E,variant:`tie-predecessors`,bindings:[{name:`offsets`,view:t.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:t.neighbors,type:`u32`,access:`read`},{name:`tieBits`,view:t.tieBits,type:`u32`,access:`read`},{name:`levels`,view:t.levels,type:`u32`,access:`read`},{name:`predecessors`,view:t.predecessors,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.nodeCount,declarations:`${D}
const NODE_COUNT: u32 = ${t.nodeCount}u;
const EDGE_COUNT: u32 = ${t.edgeCount}u;
${z}`,body:`let level = levels[levelsOffset + index];
  if (level == NONE) { return; }
  let firstEdge = min(offsets[offsetsOffset + index], EDGE_COUNT);
  let lastEdge = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
  for (var edgeIndex = firstEdge; edgeIndex < lastEdge; edgeIndex++) {
    if (!isTieEdge(edgeIndex)) { continue; }
    let targetNode = neighbors[neighborsOffset + edgeIndex];
    if (targetNode >= NODE_COUNT) { continue; }
    let targetLevel = levels[levelsOffset + targetNode];
    if (targetLevel != 0u && targetLevel != NONE && level + 1u == targetLevel) {
      atomicMin(&predecessors[predecessorsOffset + targetNode], index);
    }
  }`})}function G(e,t){return f(e,{id:t.id,operation:E,variant:`unresolved-count`,bindings:[{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`levels`,view:t.levels,type:`u32`,access:`read`},{name:`predecessors`,view:t.predecessors,type:`u32`,access:`read`},{name:`unresolvedCount`,view:t.unresolvedCount,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.nodeCount,declarations:D,body:`if (bitcast<u32>(costs[costsOffset + index]) >= INFINITY_BITS) { return; }
  if (predecessors[predecessorsOffset + index] != NONE) { return; }
  if (levels[levelsOffset + index] == 0u) { return; }
  atomicAdd(&unresolvedCount[unresolvedCountOffset], 1u);`})}var K=1024,q=`GPUNetworkReachability`,J=1024;function Y(e){return d(e,8)}var X=64,Z=4,Q=32,ee=class{id;props;maximumIterations;localIterations;maximumTieIterations;laneCount;constructor(t){t=T(t),this.id=t.id??`network-reachability`,this.props=t;let r=t.output;n(this.id,r,[{outputs:{converged:r.status?.converged,iterationCount:r.status?.iterationCount,relaxationCount:r.status?.relaxationCount,reachedEdgeCount:r.status?.reachedEdgeCount}},{outputs:{predecessors:r.predecessors},inputs:{costs:r.costs}},{outputs:{bands:r.bands},inputs:{costs:r.costs}},{outputs:{unresolvedCount:r.unresolvedCount},inputs:{costs:r.costs,predecessors:r.predecessors}},{outputs:{bandCounts:r.bandCounts},inputs:{bands:r.bands}},{outputs:{iterationLimitReached:r.status?.iterationLimitReached},inputs:{converged:r.status?.converged}}]),this.maximumIterations=t.maximumIterations??X,this.localIterations=t.localIterations??Q,this.maximumTieIterations=t.maximumTieIterations??Z,this.laneCount=t.laneCount??1;let{id:i}=this,{output:a}=t,{nodeCount:c}=o(i,t.topology,{weights:!0});for(let[e,n]of[[`startNodes`,t.startNodes],[`activeIterations`,t.activeIterations],[`output.predecessors`,a.predecessors],[`output.bands`,a.bands],[`output.bandCounts`,a.bandCounts],[`output.status.converged`,a.status?.converged],[`output.unresolvedCount`,a.unresolvedCount],[`output.status.iterationCount`,a.status?.iterationCount],[`output.status.iterationLimitReached`,a.status?.iterationLimitReached],[`output.status.relaxationCount`,a.status?.relaxationCount],[`output.status.reachedEdgeCount`,a.status?.reachedEdgeCount]])n&&s(n,[`uint32`],`${i} ${e}`);typeof t.startCount==`object`&&s(t.startCount,[`uint32`],`${i} startCount`);for(let[e,n]of[[`startCosts`,t.startCosts],[`costLimit`,t.costLimit],[`output.costs`,a.costs],[`costBreaks`,t.costBreaks]])n&&s(n,[`float32`],`${i} ${e}`);let d=a.costs.length;if(d<1)throw Error(`${i} output.costs must contain at least one node`);if(l(i,`laneCount`,this.laneCount),this.laneCount>1&&(a.predecessors||a.unresolvedCount))throw Error(`${i} laneCount > 1 does not support output.predecessors or output.unresolvedCount`);if(d%this.laneCount!==0)throw Error(`${i} output.costs length must be a multiple of laneCount`);if(c!==d/this.laneCount)throw Error(`${i} topology must contain one node per output.costs row per lane`);if(t.startCosts&&t.startCosts.length!==t.startNodes.length)throw Error(`${i} startCosts length must equal startNodes length`);typeof t.startCount==`number`&&l(i,`startCount`,t.startCount,0,t.startNodes.length);for(let[e,n]of[[`startCount`,typeof t.startCount==`object`?t.startCount:void 0],[`activeIterations`,t.activeIterations],[`costLimit`,t.costLimit],[`output.status.converged`,a.status?.converged],[`output.unresolvedCount`,a.unresolvedCount],[`output.status.iterationCount`,a.status?.iterationCount],[`output.status.iterationLimitReached`,a.status?.iterationLimitReached]])if(n&&n.length!==1)throw Error(`${i} ${e} must contain exactly one row`);for(let[e,t]of[[`output.status.relaxationCount`,a.status?.relaxationCount],[`output.status.reachedEdgeCount`,a.status?.reachedEdgeCount]])if(t&&t.length!==2)throw Error(`${i} ${e} must contain exactly two rows`);for(let[e,t]of[[`output.predecessors`,a.predecessors],[`output.bands`,a.bands]])if(t&&t.length!==d)throw Error(`${i} ${e} length must equal the node count`);if(!!(a.bands||a.bandCounts)!=!!t.costBreaks)throw Error(`${i} costBreaks is required exactly when output.bands or output.bandCounts is given`);if(t.costBreaks&&t.costBreaks.length<1)throw Error(`${i} costBreaks must contain at least one break`);if(a.bandCounts&&a.bandCounts.length!==t.costBreaks?.length)throw Error(`${i} output.bandCounts length must equal costBreaks length`);if(l(i,`maximumIterations`,this.maximumIterations,1,K),l(i,`localIterations`,this.localIterations,1,64),l(i,`maximumTieIterations`,this.maximumTieIterations,1,J),a.unresolvedCount&&!a.predecessors)throw Error(`${i} output.unresolvedCount requires output.predecessors`);e(i,$(a),[...u(t.topology),t.startNodes,t.startCosts,typeof t.startCount==`object`?t.startCount:void 0,t.activeIterations,t.costLimit,t.costBreaks])}getCommandNodes(e){let{id:n,props:r,maximumIterations:o,localIterations:s}=this,{output:l}=r,{offsets:u,neighbors:d,weights:p}=r.topology;t(n,e,te(r));let m=l.costs.length;c(n,`output.costs rows`,m,Y(e.device));let v={offsets:u,neighbors:d,weights:p},y={maxRounds:o,stampBase:0,controlWordOffset:0,dispatchSlotOffset:0},b=h(o),x=h(this.maximumTieIterations),S={maxRounds:this.maximumTieIterations,stampBase:_(y),controlWordOffset:b.controlWordCount,dispatchSlotOffset:b.dispatchSlotCount},C=!!l.predecessors,w=!!l.status?.relaxationCount,T=!!l.status?.reachedEdgeCount,E=w||T,D=b.controlWordCount+(C?x.controlWordCount:0),O=E?D:void 0;t(n,e,[],{frontier:m*8,control:(D+(E?4:0))*4,dispatch:(b.dispatchSlotCount+(C?x.dispatchSlotCount:0))*64*4,tieBits:C?Math.max(1,Math.ceil(d.length/32))*4:4});let k=g(e,n,{nodeCount:m,controlWordCount:D+(E?4:0),dispatchSlotCount:b.dispatchSlotCount+(C?x.dispatchSlotCount:0)}),A=d.length,M=C?i(e,`${n}-tie-levels`,`uint32`,m):void 0,z=C?i(e,`${n}-tie-bits`,`uint32`,Math.max(1,Math.ceil(A/32))):void 0,K=!!r.costLimit,J=[j(e,{id:`${n}-initialize`,nodeCount:m,state:k,phase:y,costs:l.costs,predecessors:l.predecessors,activeIterations:r.activeIterations,costLimit:r.costLimit,counterWordOffset:O})];r.startNodes.length>0&&J.push(N(e,{id:`${n}-seed`,nodeCount:m,state:k,phase:y,sources:r.startNodes,sourceCosts:r.startCosts,sourceCount:r.startCount,hasCostLimit:K,costs:l.costs}));for(let t=0;t<o;t++)J.push(P(e,{id:`${n}-relax-${t}`,nodeCount:m,state:k,phase:y,round:t,localIterations:s,csr:v,hasCostLimit:K,costs:l.costs,laneNodeCount:this.laneCount>1?m/this.laneCount:void 0,counterWordOffset:w?O:void 0}));if(T&&O!==void 0&&J.push(I(e,{id:`${n}-reached-edges`,nodeCount:m,state:k,csr:v,costs:l.costs,laneNodeCount:this.laneCount>1?m/this.laneCount:void 0,counterWordOffset:O})),l.predecessors&&M&&z){J.push(B(e,{id:`${n}-tie-initialize`,nodeCount:m,state:k,tiePhase:S,levels:M,tieBits:z,unresolvedCount:l.unresolvedCount}),L(e,{id:`${n}-predecessors`,nodeCount:m,csr:v,costs:l.costs,predecessors:l.predecessors,tieBits:z})),r.startNodes.length>0&&J.push(V(e,{id:`${n}-tie-roots`,nodeCount:m,state:k,phase:y,sources:r.startNodes,sourceCosts:r.startCosts,sourceCount:r.startCount,hasCostLimit:K,costs:l.costs,levels:M})),J.push(H(e,{id:`${n}-tie-seed`,nodeCount:m,edgeCount:A,state:k,tiePhase:S,offsets:u,tieBits:z,predecessors:l.predecessors,levels:M}));for(let t=0;t<this.maximumTieIterations;t++)J.push(U(e,{id:`${n}-tie-level-${t}`,nodeCount:m,edgeCount:A,state:k,tiePhase:S,round:t,localIterations:s,offsets:u,neighbors:d,tieBits:z,levels:M}));J.push(W(e,{id:`${n}-tie-predecessors`,nodeCount:m,edgeCount:A,offsets:u,neighbors:d,tieBits:z,levels:M,predecessors:l.predecessors}))}l.predecessors&&M&&l.unresolvedCount&&J.push(G(e,{id:`${n}-unresolved-count`,nodeCount:m,costs:l.costs,levels:M,predecessors:l.predecessors,unresolvedCount:l.unresolvedCount}));let X=l.status?.iterationLimitReached,Z=l.status?.converged??(X?i(e,`${n}-converged-scratch`,`uint32`,1):void 0);if((Z||l.status?.iterationCount||l.status?.relaxationCount||l.status?.reachedEdgeCount)&&J.push(F(e,{id:`${n}-finalize`,state:k,phase:y,tiePhase:C?S:void 0,converged:Z,iterationCount:l.status?.iterationCount,counters:O===void 0?void 0:{wordOffset:O,relaxationCount:l.status?.relaxationCount,reachedEdgeCount:l.status?.reachedEdgeCount}})),X&&Z&&J.push(f(e,{id:`${n}-iteration-limit-reached`,operation:q,variant:`iteration-limit-reached`,bindings:[{name:`converged`,view:Z,type:`u32`,access:`read`},{name:`limitReached`,view:X,type:`u32`,access:`read_write`}],invocationCount:1,body:`limitReached[limitReachedOffset] = 1u - min(converged[convergedOffset], 1u);`})),r.costBreaks&&(l.bands||l.bandCounts)){let t=l.bands??i(e,`${n}-bands-scratch`,`uint32`,m);if(J.push(R(e,{id:`${n}-bands`,nodeCount:m,costs:l.costs,thresholds:r.costBreaks,bands:t})),l.bandCounts){let i=r.costBreaks.length;J.push(...new a({id:`${n}-band-counts`,input:t,output:l.bandCounts,domain:[0,i]}).getCommandNodes(e))}}return J}};function te(e){let{output:t}=e;return[...u(e.topology),e.startNodes,e.startCosts,typeof e.startCount==`object`?e.startCount:void 0,e.activeIterations,e.costLimit,e.costBreaks,...$(t)]}function $(e){return[e.costs,e.predecessors,e.bands,e.bandCounts,e.status?.converged,e.unresolvedCount,e.status?.iterationCount,e.status?.iterationLimitReached,e.status?.relaxationCount,e.status?.reachedEdgeCount]}export{S as _,B as a,V as c,g as d,y as f,b as g,_ as h,L as i,H as l,h as m,K as n,U as o,C as p,J as r,W as s,ee as t,w as u,x as v};