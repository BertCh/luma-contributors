// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCompletenessStatusPort} from '../../analysis-core/contracts/status';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createFacilityScoreReductionNodes} from './facility-score-reduction';
import {snapshotBoundedFacilityProps} from './location-allocation-construction';
import {
  validateSinglePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs,
  validateCapacity,
  validateMaximumCount,
  getStorageBindingElementLimit,
  validateRequirement,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {getPackedUint32Values} from '../../analysis-core/kernels/packed-readback';

/** Exhaustive subset generation uses bounded private selection and capacity arrays. */
const BOUNDED_ALLOCATION_MAXIMUM_FACILITIES = 32;
const BOUNDED_ALLOCATION_MAXIMUM_SELECTED_FACILITIES = 8;
/** Bounds the exhaustive subset enumeration and four-word score records. */
const BOUNDED_ALLOCATION_MAXIMUM_SUBSETS = 65536;
/** Demand-order capacity assignment remains a serial reference algorithm. */
const BOUNDED_ALLOCATION_MAXIMUM_CAPACITATED_DEMAND = 4096;
/** Cooperative scoring rounds invocation counts to 64 and dense cost indices are uint32. */
const BOUNDED_ALLOCATION_MAXIMUM_PARALLEL_DEMAND = 0xffffffc0;

/** Caller-owned enumeration results and an optional additive decision certificate destination. */
export type GPUBoundedFacilityAllocationOutput = {
  selectedFacilities: GraphDataView<'uint32'>;
  assignments: GraphDataView<'uint32'>;
  /** Infinity when some demand cannot be assigned; otherwise maximum cost or weighted sum. */
  objective: GraphDataView<'float32'>;
  /** Three words: unassigned row count, valid-input flag, global-optimum certificate.
   * Capacitated allocation never claims global optimality (fixed demand-order greedy assignment).
   */
  status: GraphDataView<'uint32'>;
  /** Optional caller-owned destination for `contributor.status.complete`. */
  ports?: {status: GPUCompletenessStatusPort};
};

export type GPUBoundedFacilityAllocationProps = {
  id?: string;
  operation: 'p-center' | 'capacitated-p-median';
  facilityCount: number;
  selectCount: number;
  /** Demand-major dense costs, nonnegative finite entries; other entries are unreachable. */
  costs: GraphDataView<'float32'>;
  /** Optional one-row positive completeness certificate of the supplied travel costs. */
  costsComplete?: GraphDataView<'uint32'>;
  /** Optional one-row search horizon. Both objectives require +Infinity when supplied. */
  costLimit?: GraphDataView<'float32'>;
  /** Nonnegative indivisible demand per row. Zero-demand rows consume no capacity. */
  demand: GraphDataView<'float32'>;
  /** Nonnegative finite capacity per candidate; ignored by p-center but still required. */
  capacities: GraphDataView<'float32'>;
  output: GPUBoundedFacilityAllocationOutput;
};

/** Enumerates all facility subsets (at most 65536), evaluating subsets independently on the GPU.
 * P-center is exact for the supplied f32 cost matrix. Capacitated p-median uses a deterministic
 * unsplit demand-order nearest-feasible assignment for each subset and chooses the best feasible
 * result, or the result with fewest unassigned rows. Failure to find a feasible assignment does
 * not certify infeasibility. Reorder demand explicitly to investigate greedy order sensitivity.
 * P-center demand rows cooperate within each subset; capacity assignment retains demand order.
 * A deterministic hierarchical reduction selects the lowest-ranked tied subset.
 * Bounds: 32 facilities, 8 selected, 65536 subsets. Capacitated mode supports 4096 demand rows;
 * p-center demand rows are bounded by the device storage binding limit of the dense cost matrix.
 * Exhaustive subset growth still limits large candidate sets.
 * Supplied costsComplete must be nonzero and a supplied costLimit must be +Infinity:
 * a finite horizon alone cannot certify either unbounded objective. status.complete distinguishes
 * a completed decision from local enumeration. Loss clears the global-optimum word but preserves
 * unassigned counts and input validity. Clearing loss recovers on the next graph submission.
 */
export class GPUBoundedFacilityAllocation implements GPUCommandNodeProducer {
  readonly id: string;
  readonly subsetCount: number;
  readonly props: GPUBoundedFacilityAllocationProps;
  readonly status: {complete: GraphDataView<'uint32'>};
  constructor(props: GPUBoundedFacilityAllocationProps) {
    props = snapshotBoundedFacilityProps(props);
    const {ports, ...output} = props.output;
    const complete = ports?.status.complete;
    this.props = props;
    this.id = props.id ?? 'bounded-facility-allocation';
    validateRequirement(
      this.id,
      ['p-center', 'capacitated-p-median'].includes(props.operation),
      'operation must be p-center or capacitated-p-median'
    );
    validateCapacity(
      this.id,
      'facilityCount',
      props.facilityCount,
      1,
      BOUNDED_ALLOCATION_MAXIMUM_FACILITIES
    );
    validateCapacity(
      this.id,
      'selectCount',
      props.selectCount,
      1,
      Math.min(BOUNDED_ALLOCATION_MAXIMUM_SELECTED_FACILITIES, props.facilityCount)
    );
    validateCapacity(
      this.id,
      'demandCount',
      props.demand.length,
      1,
      props.operation === 'p-center'
        ? Math.min(
            BOUNDED_ALLOCATION_MAXIMUM_PARALLEL_DEMAND,
            Math.floor(0xffffffff / props.facilityCount)
          )
        : BOUNDED_ALLOCATION_MAXIMUM_CAPACITATED_DEMAND
    );
    this.subsetCount = combinationCount(props.facilityCount, props.selectCount);
    validateCapacity(
      this.id,
      'subsetCount',
      this.subsetCount,
      1,
      BOUNDED_ALLOCATION_MAXIMUM_SUBSETS
    );
    for (const [name, view, format, length] of [
      ['costs', props.costs, 'float32', props.demand.length * props.facilityCount],
      ['demand', props.demand, 'float32', props.demand.length],
      ['capacities', props.capacities, 'float32', props.facilityCount],
      ['output.selectedFacilities', props.output.selectedFacilities, 'uint32', props.facilityCount],
      ['output.assignments', props.output.assignments, 'uint32', props.demand.length],
      ['output.objective', props.output.objective, 'float32', 1],
      ['output.status', props.output.status, 'uint32', 3]
    ] as const) {
      validateSinglePackedView(view, [format], `${this.id} ${name}`);
      validateRequirement(this.id, view.length === length, `${name} length must equal ${length}`);
    }
    for (const [name, view, format] of [
      ['costsComplete', props.costsComplete, 'uint32'],
      ['costLimit', props.costLimit, 'float32'],
      ['output.ports.status.complete', complete, 'uint32']
    ] as const) {
      if (!view) continue;
      validateSinglePackedView(view, [format], `${this.id} ${name}`);
      validateRequirement(
        this.id,
        view.length === 1,
        `${name} length must equal 1; received ${view.length}`
      );
    }
    const outputViews = [...Object.values(output), complete];
    const outputBuffers = outputViews.filter(view => view !== undefined).map(view => view.buffer);
    validateRequirement(
      this.id,
      !(new Set(outputBuffers).size !== outputBuffers.length),
      `outputs must use distinct buffers`
    );
    validateGraphOutputsDisjointFromInputs(this.id, outputViews, [
      props.costs,
      props.demand,
      props.capacities,
      props.costsComplete,
      props.costLimit
    ]);
    validateMaximumCount(
      this.id,
      'demandCount',
      props.demand.length,
      getStorageBindingElementLimit(
        (props.costs.buffer.graph as GPUCommandGraph).device,
        4 * props.facilityCount
      )
    );
    this.status = Object.freeze({
      complete:
        complete ??
        createTransientView(
          props.costs.buffer.graph as GPUCommandGraph,
          `${this.id}-complete`,
          'uint32',
          1
        )
    });
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, subsetCount} = this;
    const {ports, ...output} = props.output;
    validateMaximumCount(
      id,
      'demandCount',
      props.demand.length,
      getStorageBindingElementLimit(graph.device, 4 * props.facilityCount)
    );
    validateGraphViewsBelongToGraph(id, graph, [
      props.costs,
      props.demand,
      props.capacities,
      props.costsComplete,
      props.costLimit,
      this.status.complete,
      ...Object.values(output)
    ]);
    validateGraphChunkBindingRanges(graph, id, {
      costs: props.costs,
      demand: props.demand,
      capacities: props.capacities,
      costsComplete: props.costsComplete,
      costLimit: props.costLimit,
      'output.ports.status.complete': this.status.complete,
      ...Object.fromEntries(Object.entries(output).map(([name, view]) => [`output.${name}`, view]))
    });
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'subset scores',
      subsetCount * 16,
      'reduce facilityCount or selectCount'
    );
    const scores = createTransientView(graph, `${id}-scores`, 'uint32', subsetCount * 4);
    const binomials = Array.from({length: 33 * 9}, (_, position) =>
      combinationCount(Math.floor(position / 9), position % 9)
    );
    const declarations = `const FACILITY_COUNT: u32 = ${props.facilityCount}u; const SELECT_COUNT: u32 = ${props.selectCount}u; const DEMAND_COUNT: u32 = ${props.demand.length}u;
const CAPACITATED: bool = ${props.operation === 'capacitated-p-median'};
const BINOMIALS: array<u32,297> = array<u32,297>(${binomials.map(value => `${value}u`).join(',')});
var<private> selection: array<u32,8>;
var<private> remaining: array<f32,8>;
fn infiniteCost() -> f32 { var bits: u32 = 0x7f800000u; return bitcast<f32>(bits); }
fn initializeSubset(subset: u32) {
  var rank = subset; var first = 0u;
  for (var slot = 0u; slot < SELECT_COUNT; slot++) {
    for (var candidate = first; candidate < FACILITY_COUNT; candidate++) {
      let suffixCount = BINOMIALS[(FACILITY_COUNT - candidate - 1u) * 9u + SELECT_COUNT - slot - 1u];
      if (rank < suffixCount) { selection[slot] = candidate; remaining[slot] = capacities[capacitiesOffset + candidate]; first = candidate + 1u; break; }
      rank -= suffixCount;
    }
  }
}
fn chooseFacility(row: u32) -> u32 {
  let weight = demand[demandOffset + row]; var best = 0xffffffffu; var bestCost = infiniteCost();
  for (var slot = 0u; slot < SELECT_COUNT; slot++) {
    let cost = costs[costsOffset + row * FACILITY_COUNT + selection[slot]];
    if (cost >= 0.0 && cost < bestCost && (!CAPACITATED || remaining[slot] >= weight)) { best = slot; bestCost = cost; }
  }
  if (best != 0xffffffffu && CAPACITATED) { remaining[best] -= weight; }
  return best;
}
fn isFiniteNonnegative(value: f32) -> bool { return value >= 0.0 && (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }`;
    const inputBindings = [
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {name: 'demand', view: props.demand, type: 'f32', access: 'read'},
      {name: 'capacities', view: props.capacities, type: 'f32', access: 'read'}
    ] as const;
    const capacitated = props.operation === 'capacitated-p-median';
    const reduction = createFacilityScoreReductionNodes(graph, {
      id: `${id}-reduce`,
      operation: 'GPUBoundedFacilityAllocation',
      scores,
      candidateCount: subsetCount
    });
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-score`,
        operation: 'GPUBoundedFacilityAllocation',
        variant: props.operation,
        invocationCount: subsetCount * (capacitated ? 1 : 64),
        workgroupSize: 64,
        guardIndex: capacitated,
        bindings: [
          ...inputBindings,
          {name: 'scores', view: scores, type: 'u32', access: 'read_write'}
        ],
        declarations: capacitated
          ? declarations
          : `${declarations.replace('var<private> selection', 'var<workgroup> selection')}
var<workgroup> partialMissing: array<u32, 64>;
var<workgroup> partialMaximum: array<f32, 64>;
var<workgroup> partialInvalid: array<u32, 64>;`,
        body: capacitated
          ? `initializeSubset(index); var objective = 0.0; var unassigned = 0u; var valid = 1u;
  if (CAPACITATED) { for (var candidate = 0u; candidate < FACILITY_COUNT; candidate++) { if (!isFiniteNonnegative(capacities[capacitiesOffset + candidate])) { valid = 0u; } } }
  for (var row = 0u; row < DEMAND_COUNT; row++) {
    let weight = demand[demandOffset + row];
    if (!isFiniteNonnegative(weight)) { valid = 0u; unassigned++; continue; }
    let slot = chooseFacility(row);
    if (slot == 0xffffffffu) { unassigned++; continue; }
    let cost = costs[costsOffset + row * FACILITY_COUNT + selection[slot]];
    if (CAPACITATED) { objective += weight * cost; } else { objective = max(objective, cost); }
  }
  scores[scoresOffset + index * 4u] = unassigned;
  scores[scoresOffset + index * 4u + 1u] = bitcast<u32>(objective);
  scores[scoresOffset + index * 4u + 2u] = index;
  scores[scoresOffset + index * 4u + 3u] = 1u - valid;`
          : `if (localInvocationIndex == 0u && workgroupIndex < ${subsetCount}u) { initializeSubset(workgroupIndex); }
  workgroupBarrier();
  var missing = 0u; var maximum = 0.0; var invalid = 0u;
  if (workgroupIndex < ${subsetCount}u) {
    for (var row = localInvocationIndex; row < DEMAND_COUNT; row += 64u) {
      if (!isFiniteNonnegative(demand[demandOffset + row])) { missing++; invalid = 1u; continue; }
      let slot = chooseFacility(row);
      if (slot == 0xffffffffu) { missing++; continue; }
      maximum = max(maximum, costs[costsOffset + row * FACILITY_COUNT + selection[slot]]);
    }
  }
  partialMissing[localInvocationIndex] = missing;
  partialMaximum[localInvocationIndex] = maximum;
  partialInvalid[localInvocationIndex] = invalid;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (localInvocationIndex < stride) {
      partialMissing[localInvocationIndex] += partialMissing[localInvocationIndex + stride];
      partialMaximum[localInvocationIndex] = max(partialMaximum[localInvocationIndex], partialMaximum[localInvocationIndex + stride]);
      partialInvalid[localInvocationIndex] |= partialInvalid[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u && workgroupIndex < ${subsetCount}u) {
    scores[scoresOffset + workgroupIndex * 4u] = partialMissing[0];
    scores[scoresOffset + workgroupIndex * 4u + 1u] = bitcast<u32>(partialMaximum[0]);
    scores[scoresOffset + workgroupIndex * 4u + 2u] = workgroupIndex;
    scores[scoresOffset + workgroupIndex * 4u + 3u] = partialInvalid[0];
  }`
      }),
      ...reduction.nodes,
      createWGSLKernelNode(graph, {
        id: `${id}-choose`,
        operation: 'GPUBoundedFacilityAllocation',
        variant: 'choose',
        invocationCount: 1,
        workgroupSize: 2,
        bindings: [
          ...inputBindings,
          {name: 'winner', view: reduction.winner, type: 'u32', access: 'read'},
          {
            name: 'selected',
            view: props.output.selectedFacilities,
            type: 'u32',
            access: 'read_write'
          },
          ...(capacitated
            ? [
                {
                  name: 'assignments',
                  view: props.output.assignments,
                  type: 'u32',
                  access: 'read_write'
                } as const
              ]
            : []),
          {name: 'objective', view: props.output.objective, type: 'f32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations,
        body: `initializeSubset(winner[winnerOffset + 2u]);
  for (var candidate = 0u; candidate < FACILITY_COUNT; candidate++) { selected[selectedOffset + candidate] = 0u; }
  for (var slot = 0u; slot < SELECT_COUNT; slot++) { selected[selectedOffset + selection[slot]] = 1u; }
  ${
    capacitated
      ? `for (var row = 0u; row < DEMAND_COUNT; row++) {
    var assignment = 0xffffffffu;
    if (isFiniteNonnegative(demand[demandOffset + row])) { let slot = chooseFacility(row); if (slot != 0xffffffffu) { assignment = selection[slot]; } }
    assignments[assignmentsOffset + row] = assignment;
  }`
      : ''
  }
  let missing = winner[winnerOffset]; let valid = 1u - winner[winnerOffset + 3u];
  objective[objectiveOffset] = select(infiniteCost(), bitcast<f32>(winner[winnerOffset + 1u]), missing == 0u && valid == 1u);
  status[statusOffset] = missing; status[statusOffset + 1u] = valid;
  status[statusOffset + 2u] = select(0u, 1u, !CAPACITATED && valid == 1u);`
      }),
      ...(capacitated || props.demand.length === 0
        ? []
        : [
            createWGSLKernelNode(graph, {
              id: `${id}-assign`,
              operation: 'GPUBoundedFacilityAllocation',
              variant: 'p-center-assign',
              invocationCount: props.demand.length,
              bindings: [
                ...inputBindings,
                {name: 'winner', view: reduction.winner, type: 'u32', access: 'read'},
                {
                  name: 'assignments',
                  view: props.output.assignments,
                  type: 'u32',
                  access: 'read_write'
                }
              ],
              declarations,
              body: `initializeSubset(winner[winnerOffset + 2u]);
  var assignment = 0xffffffffu;
  if (isFiniteNonnegative(demand[demandOffset + index])) {
    let slot = chooseFacility(index);
    if (slot != 0xffffffffu) { assignment = selection[slot]; }
  }
  assignments[assignmentsOffset + index] = assignment;`
            })
          ]),
      createWGSLKernelNode(graph, {
        id: `${id}-cost-certificate`,
        operation: 'GPUBoundedFacilityAllocation',
        variant: 'cost-certificate',
        invocationCount: 2,
        bindings: [
          ...(props.costsComplete
            ? [
                {
                  name: 'costsComplete',
                  view: props.costsComplete,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          ...(props.costLimit
            ? [
                {
                  name: 'costLimit',
                  view: props.costLimit,
                  type: 'f32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
          {name: 'complete', view: this.status.complete, type: 'u32', access: 'read_write'}
        ],
        body: `if (index == 0u) {
  let whole = status[statusOffset + 1u] != 0u${props.costsComplete ? ' && costsComplete[costsCompleteOffset] != 0u' : ''}${props.costLimit ? ' && bitcast<u32>(costLimit[costLimitOffset]) == 0x7f800000u' : ''};
  complete[completeOffset] = u32(whole);
  if (!whole) { status[statusOffset + 2u] = 0u; }
}`
      })
    ];
  }
}

/** Decode the unchanged three-word diagnostic plus the separate decision completeness word. */
export function decodeGPUBoundedFacilityAllocationStatus(
  input: Uint32Array | Uint8Array | ArrayBuffer,
  completeInput: Uint32Array | Uint8Array | ArrayBuffer
): {unassignedDemandCount: number; validInput: number; globalOptimum: number; complete: number} {
  const words = getPackedUint32Values(input, 3);
  const complete = getPackedUint32Values(completeInput, 1)[0];
  validateRequirement(
    'decodeGPUBoundedFacilityAllocationStatus',
    words[1] <= 1 && words[2] <= 1 && complete <= 1,
    'validInput, globalOptimum and complete must be 0 or 1'
  );
  return {unassignedDemandCount: words[0], validInput: words[1], globalOptimum: words[2], complete};
}

function combinationCount(count: number, selected: number): number {
  if (selected > count) return 0;
  let combinations = 1;
  for (let index = 1; index <= selected; index++)
    combinations = (combinations * (count - index + 1)) / index;
  return Math.round(combinations);
}
