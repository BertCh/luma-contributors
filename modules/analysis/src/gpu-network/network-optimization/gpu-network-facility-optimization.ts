// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  withoutComplete,
  validateOptimizationCosts,
  createOptimizationCostCompletionNodes
} from './optimization-cost-completeness';
import {snapshotFacilityConstructionProps} from './facility-construction';
import {
  validateNetworkOptimizationBindings,
  validateNetworkOptimizationInputOutputBuffers
} from './network-optimization-bindings';

import {
  createTransientView,
  getViewBindingRange,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  createNetworkOptimizationStatusNodes,
  getNetworkOptimizationStatusViews,
  validateNetworkOptimizationStatus
} from './network-optimization-status';
import type {
  GPUInvalidStatusPort,
  GPUIterationStatusPort
} from '../../analysis-core/contracts/status';
import {
  createFacilityScoreReductionNodes,
  FACILITY_SCORE_ORDER_WGSL
} from '../../gpu-spatial-analysis/location-allocation/facility-score-reduction';
import {
  validateCapacity,
  validateMaximumCount,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';

/** Largest facility count: the exhaustive search enumerates 2^F subsets (65,536 at 16). */
export const MAXIMUM_FACILITY_COUNT = 16;
/** Largest total demand row count in every mode; the candidate bound can impose a smaller cap. */
export const MAXIMUM_DEMAND_COUNT = 64;
/**
 * Largest candidate count, 2^20: 2^F subsets, or F^D assignments of D demand rows to F facilities
 * for capacitated p-median. It bounds the work to 16 batches of 65,536 candidates, and it is the
 * candidate bound for capacitated p-median (20 rows at 2 facilities, 10 at 4), in conjunction
 * with the universal 64-demand-row bound (including one facility).
 */
export const MAXIMUM_CANDIDATE_COUNT = 1048576;
/** Words of the `summary` view: valid, feasible incumbent, enumeration complete, examined count. */
const SUMMARY_WORD_COUNT = 4;
/** Fixed score scratch bounds each candidate dispatch and its hierarchical reduction. */
const CANDIDATE_BATCH_CAPACITY = 65536;
/** Fixed control scratch: cursor, three winner words, invalid flag, valid flag, batch count. */
const STATE_WORD_COUNT = 7;

/** GPU-written exact-search result and named completion diagnostics. */
export type GPUNetworkFacilityOptimizationOutput = {
  selectedFacilities: GraphDataView<'uint32'>;
  assignments: GraphDataView<'uint32'>;
  objective: GraphDataView<'float32'>;
  /** [valid, feasibleIncumbent, enumerationComplete, examinedCandidates]. */
  summary: GraphDataView<'uint32'>;
  /**
   * invalidCount is 1 when the problem has invalid inputs or a feasible objective overflows.
   * iterationCount counts nonempty candidate-scoring batches, at most 16, even when an
   * overflowing objective invalidates the result. Rejected inputs execute zero scoring batches.
   * converged certifies complete enumeration, independently of feasibility; iterationLimitReached
   * reports incomplete enumeration of a valid problem. Invalid problems set neither stopping flag.
   */
  status?: GPUInvalidStatusPort &
    GPUIterationStatusPort & {
      /** 1 only when local work finished on complete costs within the required horizon.
       * A zero leaves incumbent outputs provisional; iteration/enumeration diagnostics remain independent. */
      complete?: GraphDataView<'uint32'>;
    };
};

export type GPUNetworkFacilityOptimizationProps = {
  id?: string;
  operation: 'backup-coverage' | 'p-dispersion' | 'capacitated-p-median';
  facilityCount: number;
  /** Omit only for backup coverage: minimizes facility count, then maximizes backed-up demand. */
  selectCount?: number;
  /** Demand-major costs, or facility-major facility distances for p-dispersion. +Infinity means unreachable. */
  costs: GraphDataView<'float32'>;
  /** One-row upstream completeness witness. Omitted means caller promises complete costs. */
  costsComplete?: GraphDataView<'uint32'>;
  /** One-row travel horizon; unbounded objectives require +Infinity. */
  costLimit?: GraphDataView<'float32'>;
  demand: GraphDataView<'float32'>;
  capacities: GraphDataView<'float32'>;
  /** One nonnegative finite catchment threshold; used only by backup coverage. */
  threshold: GraphDataView<'float32'>;
  maximumCandidates?: number;
  output: GPUNetworkFacilityOptimizationOutput;
};

/**
 * Exhaustive facility candidates evaluated independently in fixed-size GPU batches.
 * Per-candidate sums and capacities retain input row order; hierarchical reductions preserve the
 * lowest candidate on ties. Incomplete enumeration never certifies infeasibility/optimality.
 *
 * Reference tier (contributor contract rule 2): an exact exhaustive search. Candidates are
 * row-parallel in batches of 65,536, with at most 16 batches, but the number of candidates grows
 * exponentially: 2^facilityCount subsets for backup coverage and p-dispersion, and
 * facilityCount^demandRows assignments for capacitated p-median. Construction enforces all three
 * limits together in every mode: at most 16 facilities, at most 64 total demand rows and
 * `candidateCount <= 2^20` (1,048,576 total candidates). For capacitated p-median, one facility
 * accepts 64 demand rows and rejects 65, although both candidate spaces have size one. With
 * 2, 4 or 16 facilities the candidate bound allows 20, 10 or 5 demand rows respectively.
 * `maximumCandidates` only shortens the examined prefix; it never relaxes these limits.
 * The full 2^20-candidate search declares 130 commands without named status ports and 131 with
 * named iteration status ports: two fixed commands plus 16 batches of eight commands, plus optional status.
 * Requesting status.complete adds one separate cost-completion command.
 * For larger problems use
 * `GPULocationAllocation` (greedy with swap improvement).
 * Compile-time: operation, counts, selectCount, maximumCandidates and optional status views.
 * Per-frame: costs, demand, capacities and threshold contents. Every encoding rewrites all outputs.
 */
export class GPUNetworkFacilityOptimization implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUNetworkFacilityOptimizationProps;
  readonly candidateCount: number;
  /** Exact number of emitted nodes, including all candidate-reduction levels and status publication. */
  readonly estimatedCommandNodeCount: number;
  constructor(props: GPUNetworkFacilityOptimizationProps) {
    props = snapshotFacilityConstructionProps(props);
    this.id = props.id ?? 'network-facility-optimization';
    this.props = props;
    const {id} = this;
    validateOptimizationCosts(id, props);
    const limits = calculateNetworkFacilitySearchLimits(
      {
        operation: props.operation,
        facilityCount: props.facilityCount,
        demandCount: props.demand.length
      },
      id
    );
    validateCapacity(
      id,
      'facilityCount',
      props.facilityCount,
      limits.minimumFacilityCount,
      MAXIMUM_FACILITY_COUNT
    );
    if (props.selectCount === undefined) {
      validateRequirement(
        id,
        props.operation === 'backup-coverage',
        'selectCount may be omitted only for backup-coverage'
      );
    } else {
      validateCapacity(
        id,
        'selectCount',
        props.selectCount,
        props.operation === 'p-dispersion' ? 2 : 1,
        props.facilityCount
      );
    }
    validateMaximumCount(id, 'demand rows', props.demand.length, MAXIMUM_DEMAND_COUNT);
    this.candidateCount = limits.candidateCount;
    // Exhaustive enumeration: facilityCount ** demand rows for capacitated-p-median, otherwise
    // 2 ** facilityCount. Both this total candidate bound and the demand row bound apply.
    validateMaximumCount(
      id,
      'candidate count (facilityCount ** demand rows for capacitated-p-median, else 2 ** facilityCount)',
      this.candidateCount,
      MAXIMUM_CANDIDATE_COUNT
    );
    validateCapacity(
      id,
      'maximumCandidates',
      props.maximumCandidates ?? this.candidateCount,
      0,
      MAXIMUM_CANDIDATE_COUNT
    );
    const examined = Math.min(this.candidateCount, props.maximumCandidates ?? this.candidateCount);
    const batchCapacity = Math.max(1, Math.min(examined, CANDIDATE_BATCH_CAPACITY));
    let reductionNodeCount = 0;
    for (let remaining = batchCapacity; remaining > 1; remaining = Math.ceil(remaining / 64)) {
      // The shared reducer uploads runtime parameters, then dispatches one reduction kernel.
      reductionNodeCount += 2;
    }
    this.estimatedCommandNodeCount =
      2 +
      Math.ceil(Math.max(examined, 1) / batchCapacity) * (2 + reductionNodeCount) +
      Number(Object.values(withoutComplete(props.output.status) ?? {}).some(Boolean)) +
      Number(Boolean(props.output.status?.complete));
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    for (const view of [
      props.costs,
      props.demand,
      props.capacities,
      props.threshold,
      props.output.objective
    ])
      validateSinglePackedView(view, ['float32'], this.id);
    for (const view of [
      props.output.selectedFacilities,
      props.output.assignments,
      props.output.summary
    ])
      if (view) validateSinglePackedView(view, ['uint32'], this.id);
    validateRequirement(
      id,
      props.costs.length ===
        props.facilityCount *
          (props.operation === 'p-dispersion' ? props.facilityCount : props.demand.length),
      'costs must hold facilityCount times demand rows (facilityCount squared for p-dispersion)'
    );
    validateRequirement(
      id,
      props.capacities.length === props.facilityCount,
      'capacities must hold facilityCount rows'
    );
    validateRequirement(id, props.threshold.length === 1, 'threshold must hold one row');
    validateRequirement(
      id,
      props.output.selectedFacilities.length === props.facilityCount,
      'output.selectedFacilities must hold facilityCount rows'
    );
    validateRequirement(
      id,
      props.output.assignments.length === props.demand.length,
      'output.assignments must hold one row per demand row'
    );
    validateRequirement(
      id,
      props.output.objective.length === 1,
      'output.objective must hold one row'
    );
    validateRequirement(
      id,
      props.output.summary.length === SUMMARY_WORD_COUNT,
      `output.summary must hold ${SUMMARY_WORD_COUNT} rows`
    );
    const outputViews = [
      props.output.selectedFacilities,
      props.output.assignments,
      props.output.objective,
      props.output.summary
    ];
    validateNetworkOptimizationStatus(
      id,
      [...outputViews, ...[props.output.status?.complete].filter(view => view !== undefined)],
      withoutComplete(props.output.status)
    );
    validateNetworkOptimizationInputOutputBuffers(this.id, getFacilityOutputViews(props.output), [
      props.costs,
      props.demand,
      props.capacities,
      props.threshold,
      ...[props.costsComplete, props.costLimit].filter(view => view !== undefined)
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, candidateCount} = this;
    const examined = Math.min(candidateCount, props.maximumCandidates ?? candidateCount);
    const capacitated = props.operation === 'capacitated-p-median';
    const dispersion = props.operation === 'p-dispersion';
    const batchCapacity = Math.max(1, Math.min(examined, CANDIDATE_BATCH_CAPACITY));
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'candidate scores',
      batchCapacity * 16,
      'reduce maximumCandidates'
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'facility control state',
      STATE_WORD_COUNT * 4,
      'request a larger device binding limit'
    );
    for (const view of [
      props.costs,
      props.demand,
      props.capacities,
      props.threshold,
      ...getFacilityOutputViews(props.output)
    ]) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'input/output aligned binding range',
        getViewBindingRange(view).size,
        'reduce facilityCount, demand rows or the alignment prefix'
      );
    }
    validateNetworkOptimizationBindings(graph, id, {
      costs: props.costs.length ? props.costs : undefined,
      demand: props.demand.length ? props.demand : undefined,
      capacities: props.capacities,
      threshold: props.threshold,
      selectedFacilities: props.output.selectedFacilities,
      assignments: props.output.assignments.length ? props.output.assignments : undefined,
      objective: props.output.objective,
      summary: props.output.summary,
      ...props.output.status,
      costsComplete: props.costsComplete,
      costLimit: props.costLimit
    });
    // Seven words: batch start, winning primary/secondary/rank, arithmetic failure, valid input,
    // and nonempty batches actually scored. The last counter survives result invalidation.
    const state = createTransientView(
      graph,
      `${id}-${props.operation}-state`,
      'uint32',
      STATE_WORD_COUNT
    );
    const scores = createTransientView(
      graph,
      `${id}-${props.operation}-scores`,
      'uint32',
      batchCapacity * 4
    );
    const declarations = `const FACILITY_COUNT: u32 = ${props.facilityCount}u;
const DEMAND_COUNT: u32 = ${props.demand.length}u;
${IS_FINITE_VALUE_WGSL}`;
    // Zero-row views may have zero-byte logical buffers. WebGPU storage bindings require at
    // least one element, even when a zero-trip demand loop never dereferences the array. Keep
    // the public views empty and bind graph-owned placeholders for these unused arrays only.
    const demand = props.demand.length
      ? props.demand
      : createTransientView(graph, `${id}-empty-demand-input`, 'float32', 1);
    const costs = props.costs.length ? props.costs : demand;
    const assignments = props.output.assignments.length
      ? props.output.assignments
      : createTransientView(graph, `${id}-empty-assignments`, 'uint32', 1);
    const inputBindings = [
      {name: 'costs', view: costs, type: 'f32', access: 'read'},
      {name: 'demand', view: demand, type: 'f32', access: 'read'},
      {name: 'capacities', view: props.capacities, type: 'f32', access: 'read'},
      {name: 'threshold', view: props.threshold, type: 'f32', access: 'read'}
    ] as const;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-initialize`,
        operation: 'GPUNetworkFacilityOptimization',
        variant: 'initialize',
        invocationCount: 1,
        workgroupSize: 2,
        bindings: [
          ...inputBindings,
          {name: 'state', view: state, type: 'u32', access: 'read_write'}
        ],
        declarations,
        body: `state[stateOffset] = 0u;
  state[stateOffset + 1u] = 0xffffffffu;
  state[stateOffset + 2u] = 0xffffffffu;
  state[stateOffset + 3u] = 0xffffffffu;
  state[stateOffset + 4u] = 0u;
  state[stateOffset + 5u] = 0u;
  state[stateOffset + 6u] = 0u;
  for (var row = 0u; row < DEMAND_COUNT; row++) {
    if (!isFiniteValue(demand[demandOffset + row]) || demand[demandOffset + row] < 0.0) { return; }
  }
  ${capacitated ? 'for (var facility = 0u; facility < FACILITY_COUNT; facility++) { if (!isFiniteValue(capacities[capacitiesOffset + facility]) || capacities[capacitiesOffset + facility] < 0.0) { return; } }' : ''}
  ${!capacitated && !dispersion ? 'if (!isFiniteValue(threshold[thresholdOffset]) || threshold[thresholdOffset] < 0.0) { return; }' : ''}
  for (var entry = 0u; entry < ${props.costs.length}u; entry++) {
    let cost = costs[costsOffset + entry];
    if (cost < 0.0 || (bitcast<u32>(cost) & 0x7fffffffu) > 0x7f800000u) { return; }
  }
  ${dispersion ? 'for (var first = 0u; first < FACILITY_COUNT; first++) { for (var second = 0u; second < FACILITY_COUNT; second++) { if (costs[costsOffset + first * FACILITY_COUNT + second] != costs[costsOffset + second * FACILITY_COUNT + first]) { return; } } }' : ''}
  state[stateOffset + 5u] = 1u;`
      })
    ];
    // Reuse one score buffer and one batch cursor; increasing the batch count adds no pipelines.
    for (let batch = 0; batch < Math.ceil(Math.max(examined, 1) / batchCapacity); batch++) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-score-${batch}`,
          operation: 'GPUNetworkFacilityOptimization',
          variant: props.operation,
          invocationCount: batchCapacity,
          bindings: [
            ...inputBindings,
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {name: 'scores', view: scores, type: 'u32', access: 'read_write'}
          ],
          declarations,
          body: `let offset = scoresOffset + index * 4u;
  scores[offset] = 0xffffffffu; scores[offset + 1u] = 0xffffffffu;
  scores[offset + 2u] = 0xffffffffu; scores[offset + 3u] = 0u;
  let candidate = state[stateOffset] + index;
  if (candidate >= ${examined}u || state[stateOffset + 5u] == 0u) { return; }
  var mask = candidate; var score = 0.0; var feasible = true;
  ${
    capacitated
      ? `mask = 0u; var code = candidate; var used: array<f32, ${props.facilityCount}>;
  for (var row = 0u; row < DEMAND_COUNT; row++) {
    let facility = code % FACILITY_COUNT; code /= FACILITY_COUNT;
    let cost = costs[costsOffset + row * FACILITY_COUNT + facility]; let weight = demand[demandOffset + row];
    if (weight == 0.0) { continue; } mask |= 1u << facility;
    used[facility] += weight;
    if (!isFiniteValue(cost) || used[facility] > capacities[capacitiesOffset + facility]) { feasible = false; }
    score += weight * cost;
  }
  if (countOneBits(mask) > ${props.selectCount}u) { feasible = false; }`
      : `let selectedCount = countOneBits(mask);
  if (selectedCount == 0u${props.selectCount !== undefined ? ` || selectedCount != ${props.selectCount}u` : ''}) { return; }
  ${
    dispersion
      ? `var infinityBits = 0x7f800000u; score = bitcast<f32>(infinityBits);
  for (var first = 0u; first < FACILITY_COUNT; first++) { if ((mask & (1u << first)) == 0u) { continue; }
    for (var second = first + 1u; second < FACILITY_COUNT; second++) { if ((mask & (1u << second)) != 0u) { score = min(score, costs[costsOffset + first * FACILITY_COUNT + second]); } }
  }`
      : `for (var row = 0u; row < DEMAND_COUNT; row++) {
    var covered = 0u;
    for (var facility = 0u; facility < FACILITY_COUNT; facility++) {
      if ((mask & (1u << facility)) != 0u && costs[costsOffset + row * FACILITY_COUNT + facility] <= threshold[thresholdOffset]) { covered++; }
    }
    if (covered == 0u) { feasible = false; }
    if (covered >= 2u) { score += demand[demandOffset + row]; }
  }`
  }`
  }
  ${!dispersion ? 'if (!isFiniteValue(score) && feasible) { scores[offset + 3u] = 1u; return; }' : ''}
  if (!feasible) { return; }
  // Signed zero is a numeric tie, including in the maximum-dispersion objective.
  let scoreBits = select(bitcast<u32>(score), 0u, score == 0.0);
  scores[offset] = ${!capacitated && props.selectCount === undefined ? 'countOneBits(mask)' : '0u'};
  scores[offset + 1u] = ${capacitated ? 'scoreBits' : '~scoreBits'};
  scores[offset + 2u] = candidate;`
        })
      );
      const reduction = createFacilityScoreReductionNodes(graph, {
        id: `${id}-${props.operation}-reduce-${batch}`,
        operation: 'GPUNetworkFacilityOptimization',
        scores,
        candidateCount: batchCapacity
      });
      nodes.push(
        ...reduction.nodes,
        createWGSLKernelNode(graph, {
          id: `${id}-advance-${batch}`,
          operation: 'GPUNetworkFacilityOptimization',
          variant: 'advance',
          invocationCount: 1,
          workgroupSize: 2,
          bindings: [
            {name: 'state', view: state, type: 'u32', access: 'read_write'},
            {name: 'winner', view: reduction.winner, type: 'u32', access: 'read'}
          ],
          declarations: FACILITY_SCORE_ORDER_WGSL,
          body: `let current = vec4<u32>(state[stateOffset + 1u], state[stateOffset + 2u], state[stateOffset + 3u], state[stateOffset + 4u]);
  let next = vec4<u32>(winner[winnerOffset], winner[winnerOffset + 1u], winner[winnerOffset + 2u], winner[winnerOffset + 3u]);
  let best = mergeFacilityScores(current, next);
  state[stateOffset + 1u] = best.x; state[stateOffset + 2u] = best.y;
  state[stateOffset + 3u] = best.z; state[stateOffset + 4u] = best.w;
  if (state[stateOffset + 5u] != 0u && state[stateOffset] < ${examined}u) {
    state[stateOffset + 6u]++;
  }
  state[stateOffset] += ${batchCapacity}u;`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation: 'GPUNetworkFacilityOptimization',
        variant: 'publish',
        invocationCount: 1,
        workgroupSize: 2,
        bindings: [
          {name: 'demand', view: demand, type: 'f32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {
            name: 'selected',
            view: props.output.selectedFacilities,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'assignments', view: assignments, type: 'u32', access: 'read_write'},
          {name: 'objective', view: props.output.objective, type: 'f32', access: 'read_write'},
          {name: 'summary', view: props.output.summary, type: 'u32', access: 'read_write'}
        ],
        declarations,
        body: `for (var word = 0u; word < 4u; word++) { summary[summaryOffset + word] = 0u; }
  var invalidBits = 0x7fc00000u; objective[objectiveOffset] = bitcast<f32>(invalidBits);
  for (var facility = 0u; facility < FACILITY_COUNT; facility++) { selected[selectedOffset + facility] = 0u; }
  for (var row = 0u; row < DEMAND_COUNT; row++) { assignments[assignmentsOffset + row] = 0xffffffffu; }
  if (state[stateOffset + 5u] == 0u || state[stateOffset + 4u] != 0u) { return; }
  let bestCandidate = state[stateOffset + 3u];
  let haveBest = bestCandidate != 0xffffffffu;
  summary[summaryOffset] = 1u; summary[summaryOffset + 1u] = u32(haveBest);
  summary[summaryOffset + 2u] = ${examined === candidateCount ? '1u' : '0u'};
  summary[summaryOffset + 3u] = ${examined}u;
  if (!haveBest) { return; }
  var bestMask = bestCandidate;
  ${
    capacitated
      ? `bestMask = 0u; var code = bestCandidate;
  for (var row = 0u; row < DEMAND_COUNT; row++) {
    let facility = code % FACILITY_COUNT; code /= FACILITY_COUNT;
    if (demand[demandOffset + row] > 0.0) {
      assignments[assignmentsOffset + row] = facility; bestMask |= 1u << facility;
    }
  }
  var bestCount = countOneBits(bestMask);
  for (var facility = 0u; facility < FACILITY_COUNT && bestCount < ${props.selectCount}u; facility++) {
    if ((bestMask & (1u << facility)) == 0u) { bestMask |= 1u << facility; bestCount++; }
  }`
      : ''
  }
  for (var facility = 0u; facility < FACILITY_COUNT; facility++) { selected[selectedOffset + facility] = (bestMask >> facility) & 1u; }
  objective[objectiveOffset] = bitcast<f32>(${capacitated ? '' : '~'}state[stateOffset + 2u]);`
      })
    );
    nodes.push(
      ...createNetworkOptimizationStatusNodes(
        graph,
        id,
        'GPUNetworkFacilityOptimization',
        props.output.summary,
        withoutComplete(props.output.status),
        {
          invalidCount: '1u - summary[summaryOffset]',
          iterationCount: 'state[stateOffset + 6u]',
          converged: 'summary[summaryOffset] * summary[summaryOffset + 2u]',
          iterationLimitReached: 'summary[summaryOffset] * (1u - summary[summaryOffset + 2u])'
        },
        props.output.status?.iterationCount
          ? [{name: 'state', view: state, type: 'u32', access: 'read'}]
          : []
      )
    );
    nodes.push(
      ...createOptimizationCostCompletionNodes(graph, id, {
        costsComplete: props.costsComplete,
        costLimit: props.costLimit,
        summary: props.output.summary,
        complete: props.output.status?.complete,
        facility: true,
        threshold: props.operation === 'backup-coverage' ? props.threshold : undefined
      })
    );
    return nodes;
  }
}

/** Flatten optional named ports for graph ownership and alias checks. */
function getFacilityOutputViews(output: GPUNetworkFacilityOptimizationOutput): GraphDataView[] {
  return [
    output.selectedFacilities,
    output.assignments,
    output.objective,
    output.summary,
    ...getNetworkOptimizationStatusViews(output.status)
  ];
}

/** Check total exhaustive-search dimensions before allocating GPU inputs.
 * Partitioning demand changes the joint capacity/selection problem and cannot recover this optimum.
 * For preselected facilities use GPUNetworkServiceAreas; larger joint selection uses the separate
 * heuristic GPULocationAllocation contract.
 */
export function getNetworkFacilitySearchLimits(options: {
  operation: GPUNetworkFacilityOptimizationProps['operation'];
  facilityCount: number;
  demandCount: number;
}) {
  return calculateNetworkFacilitySearchLimits(options, 'getNetworkFacilitySearchLimits');
}

function calculateNetworkFacilitySearchLimits(
  options: {
    operation: GPUNetworkFacilityOptimizationProps['operation'];
    facilityCount: number;
    demandCount: number;
  },
  id: string
) {
  const {operation, facilityCount, demandCount} = options;
  const minimumFacilityCount = operation === 'p-dispersion' ? 2 : 1;
  validateRequirement(
    id,
    ['backup-coverage', 'p-dispersion', 'capacitated-p-median'].includes(operation),
    'operation must be backup-coverage, p-dispersion or capacitated-p-median'
  );
  validateCapacity(id, 'facilityCount', facilityCount);
  validateCapacity(id, 'demandCount', demandCount, 0);
  const candidateCount =
    operation === 'capacitated-p-median' ? facilityCount ** demandCount : 2 ** facilityCount;
  const limitingDimension =
    facilityCount < minimumFacilityCount || facilityCount > MAXIMUM_FACILITY_COUNT
      ? 'facilityCount'
      : demandCount > MAXIMUM_DEMAND_COUNT
        ? 'demandCount'
        : candidateCount > MAXIMUM_CANDIDATE_COUNT
          ? 'candidateCount'
          : undefined;
  return {
    candidateCount,
    minimumFacilityCount,
    maximumFacilityCount: MAXIMUM_FACILITY_COUNT,
    maximumDemandCount: MAXIMUM_DEMAND_COUNT,
    maximumCandidateCount: MAXIMUM_CANDIDATE_COUNT,
    limitingDimension,
    supported: limitingDimension === undefined
  };
}
