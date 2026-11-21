// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getSparseRowBindings,
  getSparseColumnBindings,
  SPARSE_ROW_COST_WGSL,
  type SparseTranspose
} from './location-allocation-sparse';
import type {GPULocationAllocation} from './gpu-location-allocation';
import {createFacilitySelectedCompaction} from './facility-selected-compaction';

const OPERATION = 'GPULocationAllocation';
const CHOOSE_WORKGROUP_SIZE = 256;
export type FacilitySwapContext = {
  declarations: string;
  swapState: GraphDataView<'uint32'>;
  scoreScale: GraphDataView<'uint32'>;
  weightBinding: WGSLKernelBinding[];
  coverageBinding: WGSLKernelBinding[];
  demandWeightExpression: string;
  transpose?: SparseTranspose;
};

/** @internal Build initialization, scoring, deterministic pair selection and gated swap application. */
export function createFacilitySwapNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  contributor: Pick<
    GPULocationAllocation,
    'id' | 'props' | 'demandCount' | 'maximumSwapIterations'
  >,
  context: FacilitySwapContext
): GPUCommandNode<Parameters>[] {
  const {id, props, demandCount} = contributor;
  const {output, facilityCount} = props;
  const sparse = props.sparseCosts;
  const {declarations, swapState, weightBinding, coverageBinding, demandWeightExpression} = context;
  const slots = props.maximumFacilityCount;
  const pairCount = slots * facilityCount;
  const coverageMode = props.operation === 'maximum-coverage';
  // Bound comparisons by the current objective when all positive-weight demand is reachable.
  // Larger costs cannot improve it; capping them also prevents sparse loss/extra cancellation
  // against an unused, enormous second-nearest cost. Public costs remain in caller units.
  const scoreScaleBinding: WGSLKernelBinding[] = [
    {name: 'scoreScale', view: context.scoreScale, type: 'u32', access: 'read'}
  ];
  const removalScalePrologue = coverageMode
    ? `let costScoreExponent = 0;
  let reachScoreExponent = getAllocationWeightExponent(scoreScale[scoreScaleOffset + 2u]);
  let costScoreLimit = getUnreachableCost();`
    : `// High bit requests the full input scale when a positive-weight row is unreachable.
  var requiredScale = 0u;
  for (var demand = localInvocationIndex; demand < DEMAND_COUNT; demand += SWAP_WORKGROUP_SIZE) {
    let weight = bitcast<u32>(${demandWeightExpression});
    let nearest = demandData[demandDataOffset + demand * 3u];
    if (weight != 0u && weight < 0x7f800000u) {
      var rowScale = 0u;
      if ((nearest & 0x7fffffffu) >= 0x7f800000u) {
        rowScale = 0x80000000u | scoreScale[scoreScaleOffset + 1u];
      } else if ((nearest & 0x7fffffffu) != 0u) {
        // Extra demand-count headroom bounds a whole sparse loss/extra sum of capped costs.
        rowScale = u32(max(0, getAllocationValueExponent(nearest) + getAllocationValueExponent(weight) + ${2 * Math.ceil(Math.log2(demandCount))} - 120));
      }
      requiredScale = max(requiredScale, rowScale);
    }
  }
  sharedScale[localInvocationIndex] = requiredScale;
  workgroupBarrier();
  for (var stride = SWAP_WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (localInvocationIndex < stride) {
      sharedScale[localInvocationIndex] = max(sharedScale[localInvocationIndex], sharedScale[localInvocationIndex + stride]);
    }
    workgroupBarrier();
  }
  let costScoreExponent = i32(sharedScale[0] & 0x7fffffffu);
  let reachScoreExponent = getAllocationWeightExponent(scoreScale[scoreScaleOffset + 2u]);
  var sumObjective = 0.0;
  for (var demand = localInvocationIndex; demand < DEMAND_COUNT; demand += SWAP_WORKGROUP_SIZE) {
    let nearest = bitcast<f32>(demandData[demandDataOffset + demand * 3u]);
    if (isReachable(nearest)) {
      sumObjective += getAllocationScoreCost(nearest, ${demandWeightExpression}, costScoreExponent);
    }
  }
  sharedB[localInvocationIndex] = sumObjective;
  workgroupBarrier();
  for (var stride = SWAP_WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (localInvocationIndex < stride) { sharedB[localInvocationIndex] += sharedB[localInvocationIndex + stride]; }
    workgroupBarrier();
  }
  let currentObjective = sharedB[0];
  var costScoreLimit = getUnreachableCost();
  if ((sharedScale[0] & 0x80000000u) == 0u) { costScoreLimit = 2.0 * currentObjective; }
  // All lanes must finish reading the objective before the next reduction reuses sharedB.
  workgroupBarrier();`;
  // Carry the exponents in the totals buffer so weighted coverage stays within eight bindings.
  const scoreScalePrologue = `let costScoreExponent = i32(swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS + 2u]);
  let reachScoreExponent = i32(swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS + 3u]);
  let costScoreLimit = swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS + 4u];`;
  const view = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `${id}-swap-${name}`, format, length);
  // Per demand, packed as three u32 words (a device allows only eight storage buffers per kernel).
  // p-median: nearest cost bits, second-nearest cost bits, slot of the nearest facility.
  // maximum coverage: number of selected facilities within the coverage distance, unused, and the
  // slot of the only covering facility (NONE unless the count is one).
  const demandData = view('demand-data', 'uint32', demandCount * 3);
  const selectedList = view('selected-list', 'uint32', slots);
  // Slot (rank among the selected facilities) per facility, NONE for an unselected candidate.
  const slotOfFacility = view('slot-of-facility', 'uint32', facilityCount);
  // Three values per slot: the loss(r) terms (reach weight, cost; coverage uses the first) and
  // 1 when the slot holds a facility; then objective scale, total weight, their exponents and cap.
  const swapTotals = view('totals', 'float32', 3 * slots + 5);
  // Two values per pair: first the extra(i, r) terms, then the final (primary, secondary)
  // improvement, -1 when the pair is not a swap.
  const pairScores = view('pair-scores', 'float32', 2 * pairCount);
  const threshold = coverageMode ? 'coverageDistance[coverageDistanceOffset]' : '0.0';
  // Sparse scoring cannot visit the rows a candidate does not list, so loss(r) is summed per slot
  // (one workgroup each, plus one for the totals). Dense scoring visits every row and adds each
  // row's net term directly, so one workgroup of totals is enough.
  const removalGroupCount = sparse ? slots + 1 : 1;
  // Slots: list, prepare, removal (totals), score, choose.
  const gate = createWGSLDispatchGate(graph, {
    id: `${id}-swap-dispatch`,
    operation: OPERATION,
    shapes: [
      [1],
      [demandCount],
      [removalGroupCount * CHOOSE_WORKGROUP_SIZE, CHOOSE_WORKGROUP_SIZE],
      [facilityCount],
      [CHOOSE_WORKGROUP_SIZE, CHOOSE_WORKGROUP_SIZE]
    ]
  });
  const createListNodes = createFacilitySelectedCompaction(graph, {
    id,
    facilityCount,
    slots,
    selected: output.selectedFacilities,
    status: output.status,
    swapState,
    swapDispatch: gate.view,
    selectedList,
    slotOfFacility
  });
  const gateBinding = gate.binding;
  // Rows of one demand row that are considered for the nearest / second-nearest selection.
  const considerSlot = `if (cost < nearestCost) {
      secondCost = nearestCost;
      nearestCost = cost;
      nearestSlot = slot;
    } else if (cost < secondCost) {
      secondCost = cost;
    }
    if (isReachable(cost) && cost <= threshold) {
      coveringCount++;
      coveringSlot = slot;
    }`;
  const denseSelectedLoop = `for (var slot = 0u; slot < SWAP_SLOTS; slot++) {
    let facility = selectedList[selectedListOffset + slot];
    if (facility == NONE) { break; }
    let cost = ${
      sparse
        ? 'getSparseRowCost(demand, facility)'
        : 'costs[costsOffset + demand * FACILITY_COUNT + facility]'
    };
    ${considerSlot}
  }`;
  // Sparse rows are scanned when they are no longer than the slot list, otherwise the slots are
  // looked up in the row; both visit the selected facilities in ascending order.
  const prepareLoop = sparse
    ? `let rowStart = rowOffsets[rowOffsetsOffset + demand];
  let rowEnd = rowOffsets[rowOffsetsOffset + demand + 1u];
  if (rowEnd - rowStart <= SWAP_SLOTS) {
    for (var entry = rowStart; entry < rowEnd; entry++) {
      let slot = slotOfFacility[slotOfFacilityOffset + rowFacilities[rowFacilitiesOffset + entry]];
      if (slot == NONE) { continue; }
      let cost = rowCosts[rowCostsOffset + entry];
      ${considerSlot}
    }
  } else {
    ${denseSelectedLoop}
  }`
    : denseSelectedLoop;
  const prepareBindings: WGSLKernelBinding[] = [
    ...(sparse
      ? [
          ...getSparseRowBindings(sparse),
          {name: 'slotOfFacility', view: slotOfFacility, type: 'u32', access: 'read'} as const
        ]
      : [{name: 'costs', view: props.costs!, type: 'f32', access: 'read'} as const]),
    {name: 'selectedList', view: selectedList, type: 'u32', access: 'read'},
    {name: 'demandData', view: demandData, type: 'u32', access: 'read_write'},
    ...coverageBinding
  ];
  // One loop over the entries (j, c) of candidate `i`: all demand rows (dense), or the candidate's
  // column of the transpose (sparse, ascending demand rows).
  const entryLoop = (body: string) =>
    sparse
      ? `let columnEnd = columnOffsets[columnOffsetsOffset + candidate + 1u];
  for (var entry = columnOffsets[columnOffsetsOffset + candidate]; entry < columnEnd; entry++) {
    let demand = columnEntries[columnEntriesOffset + 2u * entry];
    let c = bitcast<f32>(columnEntries[columnEntriesOffset + 2u * entry + 1u]);
    ${body}
  }`
      : `for (var demand = 0u; demand < DEMAND_COUNT; demand++) {
    let c = costs[costsOffset + demand * FACILITY_COUNT + candidate];
    ${body}
  }`;
  const scoreBindings: WGSLKernelBinding[] = [
    ...(sparse
      ? getSparseColumnBindings(context.transpose!)
      : [{name: 'costs', view: props.costs!, type: 'f32', access: 'read'} as const]),
    {name: 'slotOfFacility', view: slotOfFacility, type: 'u32', access: 'read'},
    {name: 'demandData', view: demandData, type: 'u32', access: 'read'},
    {name: 'swapTotals', view: swapTotals, type: 'f32', access: 'read'},
    ...(sparse
      ? []
      : [{name: 'selectedList', view: selectedList, type: 'u32', access: 'read'} as const]),
    {name: 'pairScores', view: pairScores, type: 'f32', access: 'read_write'},
    ...weightBinding,
    ...coverageBinding
  ];
  const sparseScoreBody = coverageMode
    ? entryLoop(`let weight = getAllocationScoreCost(1.0, ${demandWeightExpression}, reachScoreExponent);
    if (isReachable(c) && c <= threshold) {
      let count = demandData[demandDataOffset + demand * 3u];
      let sole = demandData[demandDataOffset + demand * 3u + 2u];
      if (count == 0u) {
        gainCover += weight;
      } else if (sole != NONE) {
        pairScores[pairScoresOffset + 2u * (sole * FACILITY_COUNT + candidate)] += weight;
      }
    }`)
    : entryLoop(`let weight = ${demandWeightExpression};
    let scoreWeight = getAllocationScoreCost(1.0, weight, reachScoreExponent);
    let nearest = bitcast<f32>(demandData[demandDataOffset + demand * 3u]);
    let second = bitcast<f32>(demandData[demandDataOffset + demand * 3u + 1u]);
    let slot = demandData[demandDataOffset + demand * 3u + 2u];
    if (c < nearest) {
      // gain(i): the candidate beats the nearest selected facility (or reaches an unreached row).
      if (isReachable(nearest)) {
        gainCost += getAllocationSwapCost(nearest, weight, costScoreExponent, costScoreLimit) - getAllocationSwapCost(c, weight, costScoreExponent, costScoreLimit);
      } else {
        gainReach += scoreWeight;
        gainCost -= getAllocationSwapCost(c, weight, costScoreExponent, costScoreLimit);
      }
    }
    if (slot != NONE && c < second) {
      // extra(i, r): the row loses its nearest facility r but the candidate beats the second.
      var extraReach = 0.0;
      var extraCost = 0.0;
      if (isReachable(second)) {
        extraCost = getAllocationSwapCost(second, weight, costScoreExponent, costScoreLimit) - max(getAllocationSwapCost(c, weight, costScoreExponent, costScoreLimit), getAllocationSwapCost(nearest, weight, costScoreExponent, costScoreLimit));
      } else {
        extraReach = scoreWeight;
        // min(nearest, c) - c - nearest = -max(nearest, c), without an overflowing intermediate.
        extraCost = -max(getAllocationSwapCost(nearest, weight, costScoreExponent, costScoreLimit), getAllocationSwapCost(c, weight, costScoreExponent, costScoreLimit));
      }
      let pair = pairScoresOffset + 2u * (slot * FACILITY_COUNT + candidate);
      pairScores[pair] += extraReach;
      pairScores[pair + 1u] += extraCost;
    }`);
  // Dense: every row is visited, so a row served by slot r adds its net term directly,
  // `ct(min(d1, c)) - ct(min(d2, c))` (cost) and the reach weight lost when neither d2 nor c
  // reaches it. Every term is a modest difference; no loss(r) and extra(i, r) cancel.
  const denseScoreBody = coverageMode
    ? entryLoop(`let weight = getAllocationScoreCost(1.0, ${demandWeightExpression}, reachScoreExponent);
    let count = demandData[demandDataOffset + demand * 3u];
    let sole = demandData[demandDataOffset + demand * 3u + 2u];
    if (count == 0u) {
      if (isReachable(c) && c <= threshold) { gainCover += weight; }
    } else if (sole != NONE && (!isReachable(c) || c > threshold)) {
      // The only covering facility leaves and the candidate does not cover the row.
      pairScores[pairScoresOffset + 2u * (sole * FACILITY_COUNT + candidate)] -= weight;
    }`)
    : entryLoop(`let weight = ${demandWeightExpression};
    let scoreWeight = getAllocationScoreCost(1.0, weight, reachScoreExponent);
    let nearest = bitcast<f32>(demandData[demandDataOffset + demand * 3u]);
    let second = bitcast<f32>(demandData[demandDataOffset + demand * 3u + 1u]);
    let slot = demandData[demandDataOffset + demand * 3u + 2u];
    if (c < nearest) {
      if (isReachable(nearest)) {
        gainCost += getAllocationSwapCost(nearest, weight, costScoreExponent, costScoreLimit) - getAllocationSwapCost(c, weight, costScoreExponent, costScoreLimit);
      } else {
        gainReach += scoreWeight;
        gainCost -= getAllocationSwapCost(c, weight, costScoreExponent, costScoreLimit);
      }
    } else if (slot != NONE) {
      // The row loses its nearest facility and falls back to the cheaper of the second and the
      // candidate (c >= nearest here, so the candidate does not beat the nearest).
      let kept = min(second, c);
      if (kept > nearest) {
        let pair = pairScoresOffset + 2u * (slot * FACILITY_COUNT + candidate);
        if (isReachable(kept)) {
          pairScores[pair + 1u] += getAllocationSwapCost(nearest, weight, costScoreExponent, costScoreLimit) - getAllocationSwapCost(kept, weight, costScoreExponent, costScoreLimit);
        } else {
          pairScores[pair] -= scoreWeight;
          pairScores[pair + 1u] += getAllocationSwapCost(nearest, weight, costScoreExponent, costScoreLimit);
        }
      }
    }`);
  const scoreBody = sparse ? sparseScoreBody : denseScoreBody;
  const lossReach = sparse ? 'swapTotals[swapTotalsOffset + 3u * slot]' : '0.0';
  const slotIsEmpty = sparse
    ? 'swapTotals[swapTotalsOffset + 3u * slot + 2u] < 0.5'
    : 'selectedList[selectedListOffset + slot] == NONE';
  const lossCost = sparse ? 'swapTotals[swapTotalsOffset + 3u * slot + 1u]' : '0.0';
  const scoreFinalize = coverageMode
    ? `let improvement = gainCover + ${lossReach} + pairScores[pairScoresOffset + pair];
    if (improvement > SWAP_TOLERANCE * scale) {
      pairScores[pairScoresOffset + pair] = improvement;
      pairScores[pairScoresOffset + pair + 1u] = 0.0;
    } else {
      pairScores[pairScoresOffset + pair] = -1.0;
      pairScores[pairScoresOffset + pair + 1u] = -1.0;
    }`
    : `let reachGain = gainReach + ${lossReach} + pairScores[pairScoresOffset + pair];
    let improvement = gainCost + ${lossCost} + pairScores[pairScoresOffset + pair + 1u];
    var primary = -1.0;
    var secondary = -1.0;
    if (reachGain > SWAP_TOLERANCE * totalWeight) {
      primary = reachGain;
      secondary = improvement;
    } else if (abs(reachGain) <= SWAP_TOLERANCE * totalWeight && improvement > SWAP_TOLERANCE * scale) {
      primary = 0.0;
      secondary = improvement;
    }
    pairScores[pairScoresOffset + pair] = primary;
    pairScores[pairScoresOffset + pair + 1u] = secondary;`;
  const removalBody = coverageMode
    ? `let weight = getAllocationScoreCost(1.0, ${demandWeightExpression}, reachScoreExponent);
    if (removalTarget == SWAP_SLOTS) {
      sumA += weight;
    } else if (demandData[demandDataOffset + demand * 3u + 2u] == removalTarget) {
      sumA -= weight;
    }`
    : `let weight = ${demandWeightExpression};
    let scoreWeight = getAllocationScoreCost(1.0, weight, reachScoreExponent);
    let nearest = bitcast<f32>(demandData[demandDataOffset + demand * 3u]);
    let second = bitcast<f32>(demandData[demandDataOffset + demand * 3u + 1u]);
    if (removalTarget == SWAP_SLOTS) {
      sumA += scoreWeight;
    } else if (demandData[demandDataOffset + demand * 3u + 2u] == removalTarget) {
      // loss(r): the row falls back to its second-nearest facility, or becomes unreachable.
      if (isReachable(second)) {
        sumB += getAllocationSwapCost(nearest, weight, costScoreExponent, costScoreLimit) - getAllocationSwapCost(second, weight, costScoreExponent, costScoreLimit);
      } else {
        sumA -= scoreWeight;
        sumB += getAllocationSwapCost(nearest, weight, costScoreExponent, costScoreLimit);
      }
    }`;
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-swap-init`,
      operation: OPERATION,
      variant: 'swap-init',
      invocationCount: 1,
      bindings: [
        {name: 'swapState', view: swapState, type: 'u32', access: 'read_write'},
        gateBinding
      ],
      declarations,
      body: `swapState[swapStateOffset] = 0u;
  swapState[swapStateOffset + 1u] = 0u;
  swapState[swapStateOffset + 2u] = NONE;
  ${gate.armWGSL}`
    })
  ];
  for (let iteration = 0; iteration < contributor.maximumSwapIterations; iteration++) {
    const iterationId = `${id}-swap-${iteration}`;
    nodes.push(
      ...createListNodes(iteration),
      createWGSLKernelNode(graph, {
        id: `${iterationId}-prepare`,
        operation: OPERATION,
        variant: `${props.operation}-swap-prepare${sparse ? '-sparse' : ''}`,
        invocationCount: demandCount,
        ...gate.getGate(1),
        bindings: prepareBindings,
        declarations: sparse ? `${declarations}\n${SPARSE_ROW_COST_WGSL}` : declarations,
        body: `let demand = index;
  var nearestCost = getUnreachableCost();
  var secondCost = getUnreachableCost();
  var nearestSlot = NONE;
  var coveringCount = 0u;
  var coveringSlot = NONE;
  let threshold = ${threshold};
  ${prepareLoop}
  demandData[demandDataOffset + demand * 3u] = ${coverageMode ? 'coveringCount' : 'bitcast<u32>(nearestCost)'};
  demandData[demandDataOffset + demand * 3u + 1u] = ${coverageMode ? '0u' : 'bitcast<u32>(secondCost)'};
  demandData[demandDataOffset + demand * 3u + 2u] = ${coverageMode ? 'select(NONE, coveringSlot, coveringCount == 1u)' : 'nearestSlot'};`
      }),
      createWGSLKernelNode(graph, {
        id: `${iterationId}-${sparse ? 'removal' : 'totals'}`,
        operation: OPERATION,
        variant: `${props.operation}-swap-${sparse ? 'removal' : 'totals'}`,
        invocationCount: removalGroupCount * CHOOSE_WORKGROUP_SIZE,
        workgroupSize: CHOOSE_WORKGROUP_SIZE,
        guardIndex: false,
        ...gate.getGate(2),
        bindings: [
          {name: 'demandData', view: demandData, type: 'u32', access: 'read'},
          {name: 'selectedList', view: selectedList, type: 'u32', access: 'read'},
          {name: 'swapTotals', view: swapTotals, type: 'f32', access: 'read_write'},
          ...scoreScaleBinding,
          ...weightBinding
        ],
        declarations: `${declarations}
const SWAP_WORKGROUP_SIZE: u32 = ${CHOOSE_WORKGROUP_SIZE}u;
var<workgroup> sharedA: array<f32, ${CHOOSE_WORKGROUP_SIZE}>;
var<workgroup> sharedB: array<f32, ${CHOOSE_WORKGROUP_SIZE}>;
${coverageMode ? '' : `var<workgroup> sharedScale: array<u32, ${CHOOSE_WORKGROUP_SIZE}>;`}`,
        // Workgroup r (< SWAP_SLOTS) sums loss(r) over the rows served by slot r; the last
        // workgroup sums the total demand weight (A) and the objective scale (B). Strided lane
        // partials and a fixed tree make the sums independent of scheduling. No early return:
        // every invocation must reach the barriers.
        body: `let removalTarget = ${sparse ? 'workgroupIndex' : 'SWAP_SLOTS'};
  ${removalScalePrologue}
  var sumA = 0.0;
  var sumB = ${coverageMode ? '0.0' : 'select(0.0, currentObjective, removalTarget == SWAP_SLOTS && localInvocationIndex == 0u)'};
  for (var demand = localInvocationIndex; demand < DEMAND_COUNT; demand += SWAP_WORKGROUP_SIZE) {
    ${removalBody}
  }
  sharedA[localInvocationIndex] = sumA;
  sharedB[localInvocationIndex] = sumB;
  workgroupBarrier();
  for (var stride = SWAP_WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      sharedA[localInvocationIndex] += sharedA[localInvocationIndex + stride];
      sharedB[localInvocationIndex] += sharedB[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    if (removalTarget == SWAP_SLOTS) {
      swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS] = ${coverageMode ? 'sharedA[0]' : 'sharedB[0]'};
      swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS + 1u] = sharedA[0];
      swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS + 2u] = f32(costScoreExponent);
      swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS + 3u] = f32(reachScoreExponent);
      swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS + 4u] = costScoreLimit;
    } else {
      swapTotals[swapTotalsOffset + 3u * removalTarget] = sharedA[0];
      swapTotals[swapTotalsOffset + 3u * removalTarget + 1u] = sharedB[0];
      swapTotals[swapTotalsOffset + 3u * removalTarget + 2u] =
        select(0.0, 1.0, selectedList[selectedListOffset + removalTarget] != NONE);
    }
  }`
      }),
      createWGSLKernelNode(graph, {
        id: `${iterationId}-score`,
        operation: OPERATION,
        variant: `${props.operation}-swap-score${sparse ? '-sparse' : ''}`,
        invocationCount: facilityCount,
        ...gate.getGate(3),
        bindings: scoreBindings,
        declarations,
        // One thread scores every pair of one candidate. Improvement is positive when the swap
        // helps: cost removed for p-median (after the reachable demand weight gained), covered
        // weight gained for maximum coverage. A pair is only kept when it beats the noise
        // tolerance. Only this thread touches the candidate's pair entries, so no atomics.
        body: `let candidate = index;
  ${scoreScalePrologue}
  let isCandidate = slotOfFacility[slotOfFacilityOffset + candidate] == NONE;
  let threshold = ${threshold};
  for (var slot = 0u; slot < SWAP_SLOTS; slot++) {
    let pair = pairScoresOffset + 2u * (slot * FACILITY_COUNT + candidate);
    pairScores[pair] = select(-1.0, 0.0, isCandidate);
    pairScores[pair + 1u] = select(-1.0, 0.0, isCandidate);
  }
  if (!isCandidate) {
    return;
  }
  var gainReach = 0.0;
  var gainCost = 0.0;
  var gainCover = 0.0;
  ${scoreBody}
  let scale = swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS];
  let totalWeight = swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS + 1u];
  for (var slot = 0u; slot < SWAP_SLOTS; slot++) {
    let pair = 2u * (slot * FACILITY_COUNT + candidate);
    if (${slotIsEmpty}) {
      pairScores[pairScoresOffset + pair] = -1.0;
      pairScores[pairScoresOffset + pair + 1u] = -1.0;
      continue;
    }
    ${scoreFinalize}
  }`
      }),
      createWGSLKernelNode(graph, {
        id: `${iterationId}-choose`,
        operation: OPERATION,
        variant: 'swap-choose',
        invocationCount: CHOOSE_WORKGROUP_SIZE,
        workgroupSize: CHOOSE_WORKGROUP_SIZE,
        guardIndex: false,
        ...gate.getGate(4),
        bindings: [
          {name: 'pairScores', view: pairScores, type: 'f32', access: 'read'},
          {name: 'swapState', view: swapState, type: 'u32', access: 'read_write'}
        ],
        declarations: `${declarations}
const PAIR_COUNT: u32 = ${pairCount}u;
const SWAP_WORKGROUP_SIZE: u32 = ${CHOOSE_WORKGROUP_SIZE}u;
var<workgroup> sharedScore: array<f32, ${CHOOSE_WORKGROUP_SIZE}>;
var<workgroup> sharedSecondary: array<f32, ${CHOOSE_WORKGROUP_SIZE}>;
var<workgroup> sharedPair: array<u32, ${CHOOSE_WORKGROUP_SIZE}>;
fn isBetterPair(
  score: f32, secondary: f32, pair: u32, bestScore: f32, bestSecondary: f32, bestPair: u32
) -> bool {
  if (score != bestScore) { return score > bestScore; }
  if (secondary != bestSecondary) { return secondary > bestSecondary; }
  return pair < bestPair;
}`,
        // Total order ((primary, secondary) descending, pair ascending), so the result is
        // independent of scheduling. Qualifying pairs have a non-negative primary; -1 marks
        // "no swap". No early return: every invocation must reach the barriers.
        body: `var bestPair = NONE;
var bestScore = -1.0;
var bestSecondary = -1.0;
if (swapState[swapStateOffset] == 0u) {
  for (var pair = localInvocationIndex; pair < PAIR_COUNT; pair += SWAP_WORKGROUP_SIZE) {
    let score = pairScores[pairScoresOffset + 2u * pair];
    let secondary = pairScores[pairScoresOffset + 2u * pair + 1u];
    if (isBetterPair(score, secondary, pair, bestScore, bestSecondary, bestPair)) {
      bestPair = pair;
      bestScore = score;
      bestSecondary = secondary;
    }
  }
}
sharedScore[localInvocationIndex] = bestScore;
sharedSecondary[localInvocationIndex] = bestSecondary;
sharedPair[localInvocationIndex] = bestPair;
workgroupBarrier();
for (var stride = SWAP_WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
  if (localInvocationIndex < stride &&
      isBetterPair(
        sharedScore[localInvocationIndex + stride],
        sharedSecondary[localInvocationIndex + stride],
        sharedPair[localInvocationIndex + stride],
        sharedScore[localInvocationIndex],
        sharedSecondary[localInvocationIndex],
        sharedPair[localInvocationIndex])) {
    sharedScore[localInvocationIndex] = sharedScore[localInvocationIndex + stride];
    sharedSecondary[localInvocationIndex] = sharedSecondary[localInvocationIndex + stride];
    sharedPair[localInvocationIndex] = sharedPair[localInvocationIndex + stride];
  }
  workgroupBarrier();
}
if (localInvocationIndex == 0u) {
  // A qualifying pair has a non-negative primary score; -1 marks "no swap".
  swapState[swapStateOffset + 2u] = select(NONE, sharedPair[0], sharedScore[0] >= 0.0);
}`
      }),
      createWGSLKernelNode(graph, {
        id: `${iterationId}-apply`,
        operation: OPERATION,
        variant: 'swap-apply',
        invocationCount: 1,
        bindings: [
          {name: 'selected', view: output.selectedFacilities, type: 'u32', access: 'read_write'},
          {name: 'selectedList', view: selectedList, type: 'u32', access: 'read'},
          {name: 'swapState', view: swapState, type: 'u32', access: 'read_write'},
          gateBinding
        ],
        declarations,
        body: `if (swapState[swapStateOffset] != 0u) {
  return;
}
let pair = swapState[swapStateOffset + 2u];
if (pair == NONE) {
  swapState[swapStateOffset] = 1u;
  ${gate.disarmWGSL}
  return;
}
selected[selectedOffset + selectedList[selectedListOffset + pair / FACILITY_COUNT]] = 0u;
selected[selectedOffset + pair % FACILITY_COUNT] = 1u;
swapState[swapStateOffset + 1u] = swapState[swapStateOffset + 1u] + 1u;`
      })
    );
  }
  return nodes;
}
