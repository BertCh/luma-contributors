import{t as e}from"./common-CshkX-d_.js";import{r as t}from"./kernel-C3tfQOn4.js";import{A as n,E as r,O as i,S as a,T as o,a as s,f as c,g as l,p as u,v as d,x as f}from"./gpu-contributor-utils-BIwLQON5.js";import{_ as p,d as m,f as h,g,h as _,m as v,n as y,t as b,u as x,v as S}from"./gpu-network-reachability-Coypydjh.js";import{t as C}from"./network-accessibility-lanes-D6-5poNo.js";import{c as w,o as T,t as E}from"./wgsl-kernel-nodes-DagJklH2.js";import{i as D,t as O}from"./network-view-utils-B1U8oNKx.js";import{a as k,i as A}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as j}from"./define-run-B-brkCe4.js";import{p as M}from"./inputs-CJKKIRKK.js";import{d as N,s as ee}from"./pointer-overlay-Qh8qflcA.js";import{g as te,i as ne,l as re,m as ie,p as P,v as F}from"./roads-DzovamHj.js";import{Ma as I}from"./entry-preview-D70TdSbK.js";var L=`GPUNetworkCentrality`;function R(e,t,n){let r=0+e+1,i=r+t,a=i+t,o=a+e+1,s=o+n;return{nodeCount:e,forwardEdgeCount:t,reverseEdgeCount:n,forwardOffsets:0,forwardNeighbors:r,forwardWeights:i,reverseOffsets:a,reverseNeighbors:o,reverseWeights:s,wordCount:s+n}}function z(e,t){return`
const INFINITY_BITS: u32 = 0x7f800000u;
const NONE: u32 = 0xffffffffu;
const NODE_COUNT: u32 = ${e.nodeCount}u;
const FORWARD_EDGE_COUNT: u32 = ${e.forwardEdgeCount}u;
const REVERSE_EDGE_COUNT: u32 = ${e.reverseEdgeCount}u;
const FORWARD_OFFSETS: u32 = ${e.forwardOffsets}u;
const FORWARD_NEIGHBORS: u32 = ${e.forwardNeighbors}u;
const FORWARD_WEIGHTS: u32 = ${e.forwardWeights}u;
const REVERSE_OFFSETS: u32 = ${e.reverseOffsets}u;
const REVERSE_NEIGHBORS: u32 = ${e.reverseNeighbors}u;
const REVERSE_WEIGHTS: u32 = ${e.reverseWeights}u;
// Scratch words: sigma bits, remaining-edge counter, delta bits, one word per expanded node each.
const SIGMA_BASE: u32 = 0u;
const REMAINING_BASE: u32 = ${t}u;
const DELTA_BASE: u32 = ${2*t}u;

// A tight edge lies on a shortest path: the f32 sum the relaxation computes equals the target
// cost, and either the cost rises or the weight is exactly zero (review network-paths-02). A
// self-loop never is tight. A positive weight that float32 absorbs (cost + weight == cost, for
// example 1 at a cost of 2^24) is not tight: it would be tight in both directions of an
// undirected road and form a cycle that no positive weight could remove (fix review
// fr-fields-02). So only a cycle of exactly-zero weights can make the tight graph cyclic; the
// delta initialize kernel reports it.
fn isTightEdge(sourceNode: u32, targetNode: u32, sourceCost: f32, weight: f32, targetCost: f32) -> bool {
  return sourceNode != targetNode && weight >= 0.0 && bitcast<u32>(targetCost) < INFINITY_BITS &&
    sourceCost + weight == targetCost && (sourceCost < targetCost || weight == 0.0);
}

// Path counts are held as a float32 mantissa in [1, 2) (scratch word SIGMA_BASE + node, 0 when
// the node has no count) and a base-2 exponent (exponents view), so a count is not bounded by the
// float32 range (review network-paths-03). 2^shift for a shift in [-126, 0], exactly; 0 below
// -126, where the scaled term is under the resolution of anything it is added to.
fn getSigmaScale(shift: i32) -> f32 {
  if (shift < -126) { return 0.0; }
  return bitcast<f32>(u32(min(shift, 0) + 127) << 23u);
}`}function B(e,t){let{layout:n}=t;return T(e,{id:t.id,operation:L,variant:`pack-adjacency`,bindings:[{name:`forwardOffsets`,view:t.forward.offsets,type:`u32`,access:`read`},{name:`forwardNeighbors`,view:t.forward.neighbors,type:`u32`,access:`read`},{name:`forwardWeights`,view:t.forward.weights,type:`u32`,access:`read`},{name:`reverseOffsets`,view:t.reverse.offsets,type:`u32`,access:`read`},{name:`reverseNeighbors`,view:t.reverse.neighbors,type:`u32`,access:`read`},{name:`reverseWeights`,view:t.reverse.weights,type:`u32`,access:`read`},{name:`adjacency`,view:t.adjacency,type:`u32`,access:`read_write`}],invocationCount:n.wordCount,declarations:`const FORWARD_NEIGHBORS: u32 = ${n.forwardNeighbors}u;
const FORWARD_WEIGHTS: u32 = ${n.forwardWeights}u;
const REVERSE_OFFSETS: u32 = ${n.reverseOffsets}u;
const REVERSE_NEIGHBORS: u32 = ${n.reverseNeighbors}u;
const REVERSE_WEIGHTS: u32 = ${n.reverseWeights}u;`,body:`var value = 0u;
  if (index < FORWARD_NEIGHBORS) {
    value = forwardOffsets[forwardOffsetsOffset + index];
  } else if (index < FORWARD_WEIGHTS) {
    value = forwardNeighbors[forwardNeighborsOffset + index - FORWARD_NEIGHBORS];
  } else if (index < REVERSE_OFFSETS) {
    value = forwardWeights[forwardWeightsOffset + index - FORWARD_WEIGHTS];
  } else if (index < REVERSE_NEIGHBORS) {
    value = reverseOffsets[reverseOffsetsOffset + index - REVERSE_OFFSETS];
  } else if (index < REVERSE_WEIGHTS) {
    value = reverseNeighbors[reverseNeighborsOffset + index - REVERSE_NEIGHBORS];
  } else {
    value = reverseWeights[reverseWeightsOffset + index - REVERSE_WEIGHTS];
  }
  adjacency[adjacencyOffset + index] = value;`})}function V(e,t){return T(e,{id:t.id,operation:L,variant:`expand-seeds`,bindings:[{name:`pivotNodes`,view:t.pivotNodes,type:`u32`,access:`read`},{name:`expandedSeeds`,view:t.expandedSeeds,type:`u32`,access:`read_write`}],invocationCount:t.batchCount*t.laneCount,declarations:`const NONE: u32 = 0xffffffffu;
const NODE_COUNT: u32 = ${t.nodeCount}u;
const LANE_COUNT: u32 = ${t.laneCount}u;
const PIVOT_COUNT: u32 = ${t.pivotNodes.length}u;
const SEED_STRIDE: u32 = ${t.seedStride}u;`,body:`let batch = index / LANE_COUNT;
  let lane = index - batch * LANE_COUNT;
  let row = batch * LANE_COUNT + lane;
  var seed = NONE;
  if (row < PIVOT_COUNT) {
    let pivot = pivotNodes[pivotNodesOffset + row];
    if (pivot < NODE_COUNT) { seed = lane * NODE_COUNT + pivot; }
  }
  expandedSeeds[expandedSeedsOffset + batch * SEED_STRIDE + lane] = seed;`})}function H(e,t){return T(e,{id:t.id,operation:L,variant:`batch-info`,bindings:[{name:`batchInfo`,view:t.batchInfo,type:`u32`,access:`read_write`}],invocationCount:t.batchCount,declarations:`const LANE_COUNT: u32 = ${t.laneCount}u;
const SLOT_WORDS: u32 = 64u;`,body:`batchInfo[batchInfoOffset + index * SLOT_WORDS] = index * LANE_COUNT;`})}function U(e,t){return T(e,{id:t.id,operation:L,variant:`zero`,bindings:[{name:`betweenness`,view:t.betweenness,type:`f32`,access:`read_write`}],invocationCount:t.betweenness.length,body:`betweenness[betweennessOffset + index] = 0.0;`})}function W(e,t){let n=[{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`batchInfo`,view:t.batchInfo,type:`u32`,access:`read`}];return t.closeness&&n.push({name:`closeness`,view:t.closeness,type:`f32`,access:`read_write`}),t.harmonicCloseness&&n.push({name:`harmonicCloseness`,view:t.harmonicCloseness,type:`f32`,access:`read_write`}),t.reachedCounts&&n.push({name:`reachedCounts`,view:t.reachedCounts,type:`u32`,access:`read_write`}),T(e,{id:t.id,operation:L,variant:`closeness`,bindings:n,invocationCount:t.laneCount*256,workgroupSize:256,guardIndex:!1,declarations:`const INFINITY_BITS: u32 = 0x7f800000u;
const NODE_COUNT: u32 = ${t.nodeCount}u;
const LANE_COUNT: u32 = ${t.laneCount}u;
const ROW_COUNT: u32 = ${t.rowCount}u;
var<workgroup> reachedPartial: array<u32, 256>;
var<workgroup> costPartial: array<f32, 256>;
var<workgroup> harmonicPartial: array<f32, 256>;`,body:`let lane = index / 256u;
  var reached = 0u;
  var costSum = 0.0;
  var harmonicSum = 0.0;
  if (lane < LANE_COUNT) {
    for (var node = localInvocationIndex; node < NODE_COUNT; node += 256u) {
      let cost = costs[costsOffset + lane * NODE_COUNT + node];
      if (bitcast<u32>(cost) < INFINITY_BITS) {
        reached++;
        costSum += cost;
        if (cost > 0.0) { harmonicSum += 1.0 / cost; }
      }
    }
  }
  reachedPartial[localInvocationIndex] = reached;
  costPartial[localInvocationIndex] = costSum;
  harmonicPartial[localInvocationIndex] = harmonicSum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (localInvocationIndex < stride) {
      reachedPartial[localInvocationIndex] += reachedPartial[localInvocationIndex + stride];
      costPartial[localInvocationIndex] += costPartial[localInvocationIndex + stride];
      harmonicPartial[localInvocationIndex] += harmonicPartial[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  let row = batchInfo[batchInfoOffset] + lane;
  if (localInvocationIndex == 0u && lane < LANE_COUNT && row < ROW_COUNT) {
    let totalCost = costPartial[0];
    ${t.closeness?`closeness[closenessOffset + row] = select(0.0, (f32(reachedPartial[0]) - 1.0) / totalCost, totalCost > 0.0);`:``}
    ${t.harmonicCloseness?`harmonicCloseness[harmonicClosenessOffset + row] = harmonicPartial[0];`:``}
    ${t.reachedCounts?`reachedCounts[reachedCountsOffset + row] = reachedPartial[0];`:``}
  }`})}function G(e,t){let{state:n,sigmaPhase:r,deltaPhase:i}=t;return T(e,{id:t.id,operation:L,variant:`reset`,bindings:[{name:`queuedRound`,view:n.queuedRound,type:`u32`,access:`read_write`},{name:`control`,view:n.control,type:`u32`,access:`read_write`},{name:`dispatchEven`,view:n.dispatchEven,type:`u32`,access:`read_write`},{name:`dispatchOdd`,view:n.dispatchOdd,type:`u32`,access:`read_write`}],invocationCount:Math.max(n.nodeCount,p(r),p(i)),body:`${S(r,{limit:`${r.maxRounds}u`})}
  ${S(i,{limit:`${i.maxRounds}u`})}
  if (index < ${n.nodeCount}u) {
    queuedRound[queuedRoundOffset + index] = 0u;
  }`})}function K(e,t){let{layout:n,expandedNodeCount:r,state:i,phase:a}=t;return T(e,{id:t.id,operation:L,variant:`sigma-initialize`,bindings:[{name:`adjacency`,view:t.adjacency,type:`u32`,access:`read`},{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`scratch`,view:t.scratch,type:`u32`,access:`read_write`},{name:`exponents`,view:t.exponents,type:`u32`,access:`read_write`},...h(e,i,a,0)],invocationCount:r,declarations:`${z(n,r)}
${g(a,r,e.device.limits.maxComputeWorkgroupsPerDimension)}`,body:`let laneBase = (index / NODE_COUNT) * NODE_COUNT;
  let csrNode = index - laneBase;
  let cost = costs[costsOffset + index];
  var tightInCount = 0u;
  if (bitcast<u32>(cost) < INFINITY_BITS) {
    let first = min(adjacency[adjacencyOffset + REVERSE_OFFSETS + csrNode], REVERSE_EDGE_COUNT);
    let last = min(adjacency[adjacencyOffset + REVERSE_OFFSETS + csrNode + 1u], REVERSE_EDGE_COUNT);
    for (var edge = first; edge < last; edge++) {
      let sourceCsrNode = adjacency[adjacencyOffset + REVERSE_NEIGHBORS + edge];
      if (sourceCsrNode >= NODE_COUNT) { continue; }
      let weight = bitcast<f32>(adjacency[adjacencyOffset + REVERSE_WEIGHTS + edge]);
      if (isTightEdge(sourceCsrNode, csrNode, costs[costsOffset + laneBase + sourceCsrNode], weight, cost)) {
        tightInCount++;
      }
    }
  }
  scratch[scratchOffset + REMAINING_BASE + index] = tightInCount;
  scratch[scratchOffset + SIGMA_BASE + index] = 0u;
  exponents[exponentsOffset + index] = 0u;
  // Reached nodes without a tight in-edge are the roots: the pivots, and nodes reached only
  // across a positive weight that float32 absorbs. A root has one path: mantissa 1, exponent 0.
  if (tightInCount == 0u && bitcast<u32>(cost) < INFINITY_BITS) {
    scratch[scratchOffset + SIGMA_BASE + index] = 0x3f800000u;
    frontierPush(index);
  }`})}function q(e,t){let{layout:n,expandedNodeCount:r}=t;return x(e,{id:t.id,operation:L,variant:`sigma`,state:t.state,phase:t.phase,round:t.round,localIterations:t.localIterations,bindings:[{name:`adjacency`,view:t.adjacency,type:`u32`,access:`read`},{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`scratch`,view:t.scratch,type:`atomic<u32>`,access:`read_write`},{name:`exponents`,view:t.exponents,type:`u32`,access:`read_write`}],declarations:`${z(n,r)}
fn loadScratchFloat(word: u32) -> f32 {
  return bitcast<f32>(atomicLoad(&scratch[scratchOffset + word]));
}`,visitSource:`fn visit(node: u32) {
  let laneBase = (node / NODE_COUNT) * NODE_COUNT;
  let csrNode = node - laneBase;
  let cost = costs[costsOffset + node];
  var tightInCount = 0u;
  // total * 2^totalExponent is the running path count.
  var total = 0.0;
  var totalExponent = 0;
  let firstReverse = min(adjacency[adjacencyOffset + REVERSE_OFFSETS + csrNode], REVERSE_EDGE_COUNT);
  let lastReverse = min(adjacency[adjacencyOffset + REVERSE_OFFSETS + csrNode + 1u], REVERSE_EDGE_COUNT);
  for (var edge = firstReverse; edge < lastReverse; edge++) {
    let sourceCsrNode = adjacency[adjacencyOffset + REVERSE_NEIGHBORS + edge];
    if (sourceCsrNode >= NODE_COUNT) { continue; }
    let weight = bitcast<f32>(adjacency[adjacencyOffset + REVERSE_WEIGHTS + edge]);
    if (!isTightEdge(sourceCsrNode, csrNode, costs[costsOffset + laneBase + sourceCsrNode], weight, cost)) { continue; }
    tightInCount++;
    let mantissa = loadScratchFloat(SIGMA_BASE + laneBase + sourceCsrNode);
    if (!(mantissa > 0.0)) { continue; }
    let exponent = bitcast<i32>(exponents[exponentsOffset + laneBase + sourceCsrNode]);
    if (total == 0.0) {
      total = mantissa;
      totalExponent = exponent;
    } else if (exponent > totalExponent) {
      total = total * getSigmaScale(totalExponent - exponent) + mantissa;
      totalExponent = exponent;
    } else {
      total += mantissa * getSigmaScale(exponent - totalExponent);
    }
  }
  // A root keeps the sigma 1 written by the initialize kernel.
  if (tightInCount > 0u) {
    var mantissaBits = 0u;
    if (total > 0.0) {
      // Normalise: the sum of at most 2^24 mantissas below 2 is a normal float32 of at least 1.
      let totalBits = bitcast<u32>(total);
      totalExponent += i32((totalBits >> 23u) & 0xffu) - 127;
      mantissaBits = (totalBits & 0x007fffffu) | 0x3f800000u;
    }
    exponents[exponentsOffset + node] = bitcast<u32>(totalExponent);
    atomicStore(&scratch[scratchOffset + SIGMA_BASE + node], mantissaBits);
  }
  let firstForward = min(adjacency[adjacencyOffset + FORWARD_OFFSETS + csrNode], FORWARD_EDGE_COUNT);
  let lastForward = min(adjacency[adjacencyOffset + FORWARD_OFFSETS + csrNode + 1u], FORWARD_EDGE_COUNT);
  for (var edge = firstForward; edge < lastForward; edge++) {
    let targetCsrNode = adjacency[adjacencyOffset + FORWARD_NEIGHBORS + edge];
    if (targetCsrNode >= NODE_COUNT) { continue; }
    let weight = bitcast<f32>(adjacency[adjacencyOffset + FORWARD_WEIGHTS + edge]);
    if (!isTightEdge(csrNode, targetCsrNode, cost, weight, costs[costsOffset + laneBase + targetCsrNode])) { continue; }
    if (atomicSub(&scratch[scratchOffset + REMAINING_BASE + laneBase + targetCsrNode], 1u) == 1u) {
      push(laneBase + targetCsrNode);
    }
  }
}`})}function J(e,t){let{layout:n,expandedNodeCount:r,state:i,phase:a}=t;return T(e,{id:t.id,operation:L,variant:`delta-initialize`,bindings:[{name:`adjacency`,view:t.adjacency,type:`u32`,access:`read`},{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`scratch`,view:t.scratch,type:`u32`,access:`read_write`},...h(e,i,a,0)],invocationCount:r,declarations:`${z(n,r)}
${g(a,r,e.device.limits.maxComputeWorkgroupsPerDimension)}`,body:`let laneBase = (index / NODE_COUNT) * NODE_COUNT;
  let csrNode = index - laneBase;
  let cost = costs[costsOffset + index];
  var tightOutCount = 0u;
  if (bitcast<u32>(cost) < INFINITY_BITS) {
    let first = min(adjacency[adjacencyOffset + FORWARD_OFFSETS + csrNode], FORWARD_EDGE_COUNT);
    let last = min(adjacency[adjacencyOffset + FORWARD_OFFSETS + csrNode + 1u], FORWARD_EDGE_COUNT);
    for (var edge = first; edge < last; edge++) {
      let targetCsrNode = adjacency[adjacencyOffset + FORWARD_NEIGHBORS + edge];
      if (targetCsrNode >= NODE_COUNT) { continue; }
      let weight = bitcast<f32>(adjacency[adjacencyOffset + FORWARD_WEIGHTS + edge]);
      if (isTightEdge(csrNode, targetCsrNode, cost, weight, costs[costsOffset + laneBase + targetCsrNode])) {
        tightOutCount++;
      }
    }
    // The sigma phase left this counter at the number of tight predecessors that never released
    // the node. Above zero, the node was never ordered: the sigma phase hit its round limit, or
    // the node lies on or after a cycle of zero-weight tight edges. The batch status node tells
    // the two apart.
    if (scratch[scratchOffset + REMAINING_BASE + index] != 0u) {
      atomicStore(&control[controlOffset + FRONTIER_PARAMETER_0_WORD], 1u);
    }
  }
  scratch[scratchOffset + REMAINING_BASE + index] = tightOutCount;
  scratch[scratchOffset + DELTA_BASE + index] = 0u;
  if (tightOutCount == 0u && bitcast<u32>(cost) < INFINITY_BITS) {
    frontierPush(index);
  }`})}function ae(e,t){let{layout:n,expandedNodeCount:r}=t;return x(e,{id:t.id,operation:L,variant:`delta`,state:t.state,phase:t.phase,round:t.round,localIterations:t.localIterations,bindings:[{name:`adjacency`,view:t.adjacency,type:`u32`,access:`read`},{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`scratch`,view:t.scratch,type:`atomic<u32>`,access:`read_write`},{name:`exponents`,view:t.exponents,type:`u32`,access:`read`}],declarations:`${z(n,r)}
fn loadScratchFloat(word: u32) -> f32 {
  return bitcast<f32>(atomicLoad(&scratch[scratchOffset + word]));
}`,visitSource:`fn visit(node: u32) {
  let laneBase = (node / NODE_COUNT) * NODE_COUNT;
  let csrNode = node - laneBase;
  let cost = costs[costsOffset + node];
  let nodeSigma = loadScratchFloat(SIGMA_BASE + node);
  let nodeExponent = bitcast<i32>(exponents[exponentsOffset + node]);
  var total = 0.0;
  let firstForward = min(adjacency[adjacencyOffset + FORWARD_OFFSETS + csrNode], FORWARD_EDGE_COUNT);
  let lastForward = min(adjacency[adjacencyOffset + FORWARD_OFFSETS + csrNode + 1u], FORWARD_EDGE_COUNT);
  for (var edge = firstForward; edge < lastForward; edge++) {
    let targetCsrNode = adjacency[adjacencyOffset + FORWARD_NEIGHBORS + edge];
    if (targetCsrNode >= NODE_COUNT) { continue; }
    let weight = bitcast<f32>(adjacency[adjacencyOffset + FORWARD_WEIGHTS + edge]);
    if (!isTightEdge(csrNode, targetCsrNode, cost, weight, costs[costsOffset + laneBase + targetCsrNode])) { continue; }
    let targetSigma = loadScratchFloat(SIGMA_BASE + laneBase + targetCsrNode);
    // A zero sigma only occurs when the sigma phase left the target unordered (round limit or a
    // zero-weight cycle).
    if (!(targetSigma > 0.0)) { continue; }
    // sigma(v) / sigma(w): the ratio of the mantissas times an exact power of two. A target has at
    // least the paths of each predecessor, so the exponent difference is never positive.
    let targetExponent = bitcast<i32>(exponents[exponentsOffset + laneBase + targetCsrNode]);
    let ratio = nodeSigma / targetSigma * getSigmaScale(nodeExponent - targetExponent);
    total += ratio * (1.0 + loadScratchFloat(DELTA_BASE + laneBase + targetCsrNode));
  }
  atomicStore(&scratch[scratchOffset + DELTA_BASE + node], bitcast<u32>(total));
  let firstReverse = min(adjacency[adjacencyOffset + REVERSE_OFFSETS + csrNode], REVERSE_EDGE_COUNT);
  let lastReverse = min(adjacency[adjacencyOffset + REVERSE_OFFSETS + csrNode + 1u], REVERSE_EDGE_COUNT);
  for (var edge = firstReverse; edge < lastReverse; edge++) {
    let sourceCsrNode = adjacency[adjacencyOffset + REVERSE_NEIGHBORS + edge];
    if (sourceCsrNode >= NODE_COUNT) { continue; }
    let weight = bitcast<f32>(adjacency[adjacencyOffset + REVERSE_WEIGHTS + edge]);
    if (!isTightEdge(sourceCsrNode, csrNode, costs[costsOffset + laneBase + sourceCsrNode], weight, cost)) { continue; }
    if (atomicSub(&scratch[scratchOffset + REMAINING_BASE + laneBase + sourceCsrNode], 1u) == 1u) {
      push(laneBase + sourceCsrNode);
    }
  }
}`})}function oe(e,t){return T(e,{id:t.id,operation:L,variant:`accumulate`,bindings:[{name:`scratch`,view:t.scratch,type:`u32`,access:`read`},{name:`seeds`,view:t.seeds,type:`u32`,access:`read`},{name:`betweenness`,view:t.betweenness,type:`f32`,access:`read_write`}],invocationCount:t.nodeCount,declarations:`const NODE_COUNT: u32 = ${t.nodeCount}u;
const LANE_COUNT: u32 = ${t.laneCount}u;
const DELTA_BASE: u32 = ${2*t.expandedNodeCount}u;
const BETWEENNESS_SCALE: f32 = ${w(t.betweennessScale)};`,body:`var total = 0.0;
  for (var lane = 0u; lane < LANE_COUNT; lane++) {
    let expanded = lane * NODE_COUNT + index;
    if (seeds[seedsOffset + lane] == expanded) { continue; }
    total += bitcast<f32>(scratch[scratchOffset + DELTA_BASE + expanded]);
  }
  betweenness[betweennessOffset + index] = betweenness[betweennessOffset + index] + BETWEENNESS_SCALE * total;`})}var Y=2;function se(e,t){return T(e,{id:t.id,operation:L,variant:`batch-status`,bindings:[{name:`control`,view:t.state.control,type:`u32`,access:`read`},{name:`batchStatus`,view:t.batchStatus,type:`u32`,access:`read_write`}],invocationCount:1,body:`let searchTruncated = batchStatus[batchStatusOffset] == 0u;
  let sigmaTruncated = control[controlOffset + ${t.sigmaPhase.controlWordOffset+0}u] != 0u;
  let deltaTruncated = control[controlOffset + ${t.deltaPhase.controlWordOffset+0}u] != 0u;
  let unordered = control[controlOffset + ${t.deltaPhase.controlWordOffset+2}u] != 0u;
  if (sigmaTruncated || deltaTruncated || unordered) {
    batchStatus[batchStatusOffset] = 0u;
  }
  batchStatus[batchStatusOffset + 1u] = select(0u, 1u, unordered && !sigmaTruncated);
  batchStatus[batchStatusOffset + ${Y}u] = select(0u, 1u, searchTruncated || sigmaTruncated || deltaTruncated);`})}function ce(e,t){let n=[{name:`batchStatus`,view:t.batchStatus,type:`u32`,access:`read`}];return t.converged&&n.push({name:`converged`,view:t.converged,type:`u32`,access:`read_write`}),t.zeroWeightCycle&&n.push({name:`zeroWeightCycle`,view:t.zeroWeightCycle,type:`u32`,access:`read_write`}),t.iterationLimitReached&&n.push({name:`iterationLimitReached`,view:t.iterationLimitReached,type:`u32`,access:`read_write`}),T(e,{id:t.id,operation:L,variant:`converged${t.converged?``:`-no-flag`}${t.zeroWeightCycle?`-cycle`:``}`,bindings:n,invocationCount:1,declarations:`const BATCH_COUNT: u32 = ${t.batchCount}u;
const SLOT_WORDS: u32 = 64u;`,body:`var allConverged = 1u;
  var anyCycle = 0u;
  var anyLimitReached = 0u;
  for (var batch = 0u; batch < BATCH_COUNT; batch++) {
    allConverged = min(allConverged, batchStatus[batchStatusOffset + batch * SLOT_WORDS]);
    anyCycle = max(anyCycle, batchStatus[batchStatusOffset + batch * SLOT_WORDS + 1u]);
    anyLimitReached = max(anyLimitReached, ${t.hasBetweenness?`batchStatus[batchStatusOffset + batch * SLOT_WORDS + ${Y}u]`:`1u - batchStatus[batchStatusOffset + batch * SLOT_WORDS]`});
  }
  ${t.converged?`converged[convergedOffset] = allConverged;`:``}
  ${t.zeroWeightCycle?`zeroWeightCycle[zeroWeightCycleOffset] = anyCycle;`:``}
  ${t.iterationLimitReached?`iterationLimitReached[iterationLimitReachedOffset] = anyLimitReached;`:``}`})}function le(e,t){return T(e,{id:t.id,operation:L,variant:`edge-accumulate`,invocationCount:t.layout.nodeCount,bindings:[{name:`adjacency`,view:t.adjacency,type:`u32`,access:`read`},{name:`scratch`,view:t.scratch,type:`u32`,access:`read`},{name:`exponents`,view:t.exponents,type:`u32`,access:`read`},{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`edges`,view:t.edgeBetweenness,type:`f32`,access:`read_write`}],declarations:z(t.layout,t.expandedNodeCount),body:`let first = min(adjacency[adjacencyOffset + FORWARD_OFFSETS + index], FORWARD_EDGE_COUNT);
  let last = min(adjacency[adjacencyOffset + FORWARD_OFFSETS + index + 1u], FORWARD_EDGE_COUNT);
  for (var edge = first; edge < last; edge++) {
    let destination = adjacency[adjacencyOffset + FORWARD_NEIGHBORS + edge];
    let weight = bitcast<f32>(adjacency[adjacencyOffset + FORWARD_WEIGHTS + edge]);
    var total = 0.0;
    if (destination < NODE_COUNT) {
      for (var lane = 0u; lane < ${t.laneCount}u; lane++) {
        let sourceRow = lane * NODE_COUNT + index;
        let targetRow = lane * NODE_COUNT + destination;
        if (isTightEdge(index, destination, costs[costsOffset + sourceRow], weight, costs[costsOffset + targetRow])) {
          let denominator = bitcast<f32>(scratch[scratchOffset + targetRow]);
          if (denominator > 0.0) {
            let ratio = bitcast<f32>(scratch[scratchOffset + sourceRow]) / denominator * getSigmaScale(
              bitcast<i32>(exponents[exponentsOffset + sourceRow]) -
              bitcast<i32>(exponents[exponentsOffset + targetRow])
            );
            total += ratio * (1.0 + bitcast<f32>(scratch[scratchOffset + DELTA_BASE + targetRow]));
          }
        }
      }
    }
    edges[edgesOffset + edge] += ${w(t.betweennessScale)} * total;
  }`})}function ue(e,t,n){let r=e=>Object.entries(e).flatMap(([e,t])=>t&&t.length>0?[{name:e,buffer:t.buffer.defaultBuffer?i(t.buffer.defaultBuffer):t.buffer,start:t.byteOffset,end:t.byteOffset+(t.length-1)*t.byteStride+t.rowByteLength}]:[]),a=r(t),s=r(n),c=(e,t)=>e.buffer===t.buffer&&e.start<t.end&&t.start<e.end;for(let[t,n]of a.entries()){for(let r of a.slice(t+1))o(e,!c(n,r),`outputs must use disjoint logical byte ranges for ${n.name} and ${r.name} (separate buffers or disjoint packed fields)`);for(let t of s)o(e,!c(n,t),`outputs must not overlap input logical byte ranges for ${n.name} and ${t.name} (separate buffers or disjoint packed fields)`)}}var de=16777215,fe=64,X=16,Z=1,pe=class{id;props;nodeCount;pivotCount;laneCount;batchCount;maximumIterations;localIterations;betweennessScale;estimatedCommandNodeCount;constructor(t,i=!1){this.id=t.id??`network-centrality`,this.props=t;let{id:d}=this,{output:f}=t,p=n(d,t.topology,{weights:!0}),m=n(d,t.reverseTopology,{weights:!0});o(d,p.nodeCount===m.nodeCount,`reverseTopology must have the node count of topology`),this.nodeCount=p.nodeCount,r(t.pivotNodes,[`uint32`],`${d} pivotNodes`),this.pivotCount=t.pivotNodes.length,o(d,this.pivotCount>=1,`pivotNodes must contain at least one pivot`),t.costLimit&&(r(t.costLimit,[`float32`],`${d} costLimit`),o(d,t.costLimit.length===1,`costLimit must contain exactly one row`)),o(d,!!(f.betweenness||f.edgeBetweenness||f.closeness||f.harmonicCloseness||f.reachedCounts),`output must contain betweenness, edgeBetweenness, closeness, harmonicCloseness or reachedCounts`);for(let[e,t,n]of[[`output.betweenness`,f.betweenness,this.nodeCount],[`output.edgeBetweenness`,f.edgeBetweenness,p.edgeCount],[`output.closeness`,f.closeness,this.pivotCount],[`output.harmonicCloseness`,f.harmonicCloseness,this.pivotCount]])t&&(r(t,[`float32`],`${d} ${e}`),o(d,t.length===n,`${e} must contain one row per ${e===`output.betweenness`?`node`:e===`output.edgeBetweenness`?`edge slot`:`pivot`}`));f.reachedCounts&&(r(f.reachedCounts,[`uint32`],`${d} output.reachedCounts`),o(d,f.reachedCounts.length===this.pivotCount,`output.reachedCounts must contain one row per pivot`)),o(d,t.sampling===void 0||t.sampling===`exact`||t.sampling===`sampled`,`sampling must be 'exact' or 'sampled'`);let h=f.status?.approximation;h&&(r(h,[`uint32`],`${d} output.status.approximation`),c(d,`output.status.approximation.length`,h.length,1,1));let g=f.status?.converged;g&&(r(g,[`uint32`],`${d} output.status.converged`),o(d,g.length===1,`output.status.converged must contain exactly one row`));let _=f.status?.iterationLimitReached;_&&(r(_,[`uint32`],`${d} output.status.iterationLimitReached`),o(d,_.length===1,`output.status.iterationLimitReached must contain exactly one row`));let v=f.status?.zeroWeightCycle;v&&(r(v,[`uint32`],`${d} output.status.zeroWeightCycle`),o(d,v.length===1,`output.status.zeroWeightCycle must contain exactly one row`),o(d,!!(f.betweenness||f.edgeBetweenness),`output.status.zeroWeightCycle needs output.betweenness or output.edgeBetweenness`));let b=!!(g||v||_);this.maximumIterations=t.maximumIterations??fe,this.localIterations=t.localIterations??X,c(d,`maximumIterations`,this.maximumIterations,1,y),c(d,`localIterations`,this.localIterations,1,64),this.betweennessScale=t.betweennessScale??1,l(d,`betweennessScale`,Math.fround(this.betweennessScale));let x=f.betweenness||f.edgeBetweenness?Math.min(s(8),s(12)):s(8),S=e(Math.floor(x/this.nodeCount),1,this.pivotCount);this.laneCount=t.laneCount??Math.min(S,C({rowCount:this.pivotCount,nodeCount:this.nodeCount,edgeCount:p.edgeCount})),c(d,`laneCount`,this.laneCount,1,this.pivotCount),a(d,`laneCount * nodeCount`,this.laneCount*this.nodeCount,x),this.batchCount=Math.ceil(this.pivotCount/this.laneCount);let w=!!(f.betweenness||f.edgeBetweenness),T=!!(f.closeness||f.harmonicCloseness||f.reachedCounts);T&&a(d,`laneCount for closeness outputs`,this.laneCount,de);let E=Math.min(this.maximumIterations,this.nodeCount),D=Number(!!f.betweenness)+Number(!!f.edgeBetweenness),O=E+2+Number(b)+Number(T)+(w?2*E+3+D+Number(b):0);this.estimatedCommandNodeCount=1+Number(T)+(w?1+D:0)+Number(b)+Number(!!h)+this.batchCount*O,u(d,this.estimatedCommandNodeCount),ue(d,{betweenness:f.betweenness,edgeBetweenness:f.edgeBetweenness,closeness:f.closeness,harmonicCloseness:f.harmonicCloseness,reachedCounts:f.reachedCounts,...f.status},{offsets:t.topology.offsets,neighbors:t.topology.neighbors,weights:t.topology.weights,reverseOffsets:t.reverseTopology.offsets,reverseNeighbors:t.reverseTopology.neighbors,reverseWeights:t.reverseTopology.weights,pivotNodes:t.pivotNodes,costLimit:t.costLimit}),A(d,{closeness:f.closeness,harmonicCloseness:f.harmonicCloseness,reachedCounts:f.reachedCounts},{strictAlignedBindings:!0}),i||Q(d,f)}getCommandNodes(e){let{id:n,props:r,nodeCount:i,pivotCount:a,laneCount:o,batchCount:s}=this,{output:c}=r;Q(n,c),this.validateGraphBindings(e);let l=!!(c.betweenness||c.edgeBetweenness),u=!!(c.closeness||c.harmonicCloseness||c.reachedCounts),d=o*i,f=Math.min(this.maximumIterations,i),p=[],h=O(o),g=t(e,`${n}-expanded-seeds`,`uint32`,s*h);p.push(V(e,{id:`${n}-expand-seeds`,nodeCount:i,laneCount:o,batchCount:s,seedStride:h,pivotNodes:r.pivotNodes,expandedSeeds:g}));let y=t(e,`${n}-costs`,`float32`,d),x=c.status?.converged||c.status?.zeroWeightCycle||c.status?.iterationLimitReached?t(e,`${n}-batch-converged`,`uint32`,s*64):void 0,S=u?t(e,`${n}-batch-info`,`uint32`,s*64):void 0;S&&p.push(H(e,{id:`${n}-batch-info`,laneCount:o,batchCount:s,batchInfo:S}));let C;if(l){let a=R(i,r.topology.neighbors.length,r.reverseTopology.neighbors.length),o=t(e,`${n}-adjacency`,`uint32`,a.wordCount),s=t(e,`${n}-scratch`,`uint32`,3*d),l=t(e,`${n}-sigma-exponents`,`uint32`,d),u=v(f),h={maxRounds:f,stampBase:0,controlWordOffset:0,dispatchSlotOffset:0},g={maxRounds:f,stampBase:_(h),controlWordOffset:u.controlWordCount,dispatchSlotOffset:u.dispatchSlotCount};C={adjacency:o,scratch:s,exponents:l,layout:a,state:m(e,`${n}-frontier`,{nodeCount:d,controlWordCount:2*u.controlWordCount,dispatchSlotCount:2*u.dispatchSlotCount}),sigmaPhase:h,deltaPhase:g},p.push(B(e,{id:`${n}-pack-adjacency`,layout:a,forward:r.topology,reverse:r.reverseTopology,adjacency:o}));for(let[t,r]of[[`betweenness`,c.betweenness],[`edge-betweenness`,c.edgeBetweenness]])r&&p.push(U(e,{id:`${n}-zero-${t}`,betweenness:r}))}for(let t=0;t<s;t++){let s=`${n}-batch-${t}`,l=D(e,g,t*h,o),u=x?D(e,x,t*64,1):void 0;if(p.push(...new b({id:`${s}-reachability`,topology:{offsets:r.topology.offsets,neighbors:r.topology.neighbors,weights:r.topology.weights},laneCount:o,startNodes:l,costLimit:r.costLimit,maximumIterations:f,localIterations:this.localIterations,output:{costs:y,status:{converged:u}}}).getCommandNodes(e)),S&&p.push(W(e,{id:`${s}-closeness`,nodeCount:i,laneCount:o,rowCount:a,costs:y,batchInfo:D(e,S,t*64,1),closeness:c.closeness,harmonicCloseness:c.harmonicCloseness,reachedCounts:c.reachedCounts})),C){let{adjacency:n,scratch:r,exponents:a,layout:u,state:m,sigmaPhase:h,deltaPhase:g}=C,_={layout:u,expandedNodeCount:d,state:m,adjacency:n,costs:y,scratch:r,exponents:a};p.push(G(e,{id:`${s}-reset`,state:m,sigmaPhase:h,deltaPhase:g}),K(e,{..._,id:`${s}-sigma-initialize`,phase:h}));for(let t=0;t<f;t++)p.push(q(e,{..._,id:`${s}-sigma-${t}`,phase:h,round:t,localIterations:Z}));p.push(J(e,{..._,id:`${s}-delta-initialize`,phase:g}));for(let t=0;t<f;t++)p.push(ae(e,{..._,id:`${s}-delta-${t}`,phase:g,round:t,localIterations:Z}));c.betweenness&&p.push(oe(e,{id:`${s}-accumulate`,nodeCount:i,laneCount:o,expandedNodeCount:d,betweennessScale:this.betweennessScale,scratch:r,seeds:l,betweenness:c.betweenness})),c.edgeBetweenness&&p.push(le(e,{id:`${s}-accumulate-edges`,layout:u,expandedNodeCount:d,laneCount:o,betweennessScale:this.betweennessScale,adjacency:n,scratch:r,exponents:a,costs:y,edgeBetweenness:c.edgeBetweenness})),x&&p.push(se(e,{id:`${s}-status`,state:m,sigmaPhase:h,deltaPhase:g,batchStatus:D(e,x,t*64,3)}))}}return x&&p.push(ce(e,{id:`${n}-converged`,batchCount:s,batchStatus:x,converged:c.status?.converged,iterationLimitReached:c.status?.iterationLimitReached,hasBetweenness:l,zeroWeightCycle:c.status?.zeroWeightCycle})),c.status?.approximation&&p.push(E(e,{id:`${n}-approximation`,operation:`GPUNetworkCentrality`,view:c.status.approximation,type:`u32`,value:r.sampling===`sampled`?`1u`:`0u`})),p}validateGraphBindings(e){let{id:t,props:n,nodeCount:r,laneCount:i,batchCount:a}=this,{output:o}=n,s={"topology.offsets":n.topology.offsets,"topology.neighbors":n.topology.neighbors,"topology.weights":n.topology.weights,"reverseTopology.offsets":n.reverseTopology.offsets,"reverseTopology.neighbors":n.reverseTopology.neighbors,"reverseTopology.weights":n.reverseTopology.weights,pivotNodes:n.pivotNodes,costLimit:n.costLimit,"output.betweenness":o.betweenness,"output.edgeBetweenness":o.edgeBetweenness,"output.closeness":o.closeness,"output.harmonicCloseness":o.harmonicCloseness,"output.reachedCounts":o.reachedCounts,...Object.fromEntries(Object.entries(o.status??{}).map(([e,t])=>[`output.status.${e}`,t]))};f(t,e,[...Object.values(s),n.topology.nodePositions,n.reverseTopology.nodePositions]),k(e,t,s);let c=i*r,l=Math.min(this.maximumIterations,r),u=v(l),p=!!(o.betweenness||o.edgeBetweenness),m=!!(o.status?.converged||o.status?.zeroWeightCycle||o.status?.iterationLimitReached),h=!!(o.closeness||o.harmonicCloseness||o.reachedCounts),g=[[`expanded seeds`,a*O(i)],[`costs`,c],[`reachability queues`,2*c],[`reachability queued rounds`,c],[`reachability control`,u.controlWordCount],[`reachability dispatch slots`,u.dispatchSlotCount*64]];h&&g.push([`batch info`,a*64]),m&&g.push([`batch status`,a*64]),p&&g.push([`adjacency`,R(r,n.topology.neighbors.length,n.reverseTopology.neighbors.length).wordCount],[`scratch`,3*c],[`sigma exponents`,c],[`dependency queues`,2*c],[`dependency queued rounds`,c],[`dependency control`,2*u.controlWordCount],[`dependency dispatch slots`,2*u.dispatchSlotCount*64]);for(let[n,r]of g)d(e.device,t,n,4*r,`reduce laneCount, pivot count, edge count or maximumIterations`)}};function Q(e,t){A(e,{converged:t.status?.converged,iterationLimitReached:t.status?.iterationLimitReached,zeroWeightCycle:t.status?.zeroWeightCycle},{strictAlignedBindings:!0})}var $=ee(I,`costLimit`),me=j({parameters:I,caption:`Betweenness per intersection, sampled from the pivots within the cost limit.`,createInputs:({random:e,world:t})=>{let n=ne(e,t),r=n.findNearestNode(t.width/2,.5);return{network:n,pivots:te(n,t,96),center:[n.nodes[r*2],n.nodes[r*2+1]]}},build(e,{network:t,pivots:n},r){let i=Math.round(r.pivotNodes),{topology:a}=M(e,t),o=e.allocate(`cost-limit`,`float32`,1),s=e.allocate(`betweenness`,`float32`,t.nodeCount);return e.graph.add(new pe({topology:a,reverseTopology:a,pivotNodes:e.upload(`pivot-nodes`,`uint32`,n.slice(0,i)).view,costLimit:o.view,betweennessScale:t.nodeCount/i,maximumIterations:48,output:{betweenness:s.view}})),{outputs:{...F(e,t),betweenness:s},update:e=>o.write(Float32Array.of(e.costLimit))}},rebuildOn:[`pivotNodes`],interaction:{kind:`reach`,hint:`reach out`,rest:({inputs:e,world:t,toUnit:n})=>n(re(e.center,$.default,t)),resolve(e,{inputs:t,outputs:n,peek:r}){let i=N($,Math.hypot(e.x-t.center[0],e.y-t.center[1])),a=r(n.betweenness),o=0;if(a)for(let e of a)Number.isFinite(e)&&(o=Math.max(o,e));return{values:{costLimit:i},glyphs:[{shape:`ring`,at:t.center,radius:i,dashed:!0},{shape:`line`,from:t.center,to:[e.x,e.y],dashed:!0}],readout:`costLimit ${i.toFixed(2)} · peak betweenness ${o.toFixed(0)}`}}},draw:({outputs:e})=>[P(e),ie(e,e.betweenness,{radius:2.2,range:[0,3e3]})]});export{me as default};