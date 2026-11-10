// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp} from '@math.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  createFacilityScoreReductionNodes,
  FACILITY_SCORE_ORDER_WGSL
} from '../../gpu-spatial-analysis/location-allocation/facility-score-reduction';
import {
  validateCapacity,
  validateMaximumCount,
  validatePackedView,
  validateRequirement,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Largest facility count: the exhaustive search enumerates 2^F subsets (65,536 at 16). */
const MAXIMUM_FACILITY_COUNT = 16;
/** Largest demand row count of any operation; capacitated p-median is limited far lower by the candidate count. */
const MAXIMUM_DEMAND_COUNT = 64;
/**
 * Largest candidate count, 2^20: 2^F subsets, or F^D assignments of D demand rows to F facilities
 * for capacitated p-median. It bounds the work to 16 batches of 65,536 candidates, and it is the
 * real limit on demand rows of capacitated p-median (20 rows at 2 facilities, 10 at 4).
 */
const MAXIMUM_CANDIDATE_COUNT = 1048576;
/** Words of the `status` view: valid, feasible incumbent, enumeration complete, examined count. */
const STATUS_WORD_COUNT = 4;

export type GPUNetworkFacilityOptimizationProps = {
  id?: string;
  operation: 'backup-coverage' | 'p-dispersion' | 'capacitated-p-median';
  facilityCount: number;
  /** Omit only for backup coverage: minimizes facility count, then maximizes backed-up demand. */
  selectCount?: number;
  /** Demand-major costs, or facility-major facility distances for p-dispersion. +Infinity means unreachable. */
  costs: GraphDataView<'float32'>;
  demand: GraphDataView<'float32'>;
  capacities: GraphDataView<'float32'>;
  /** One nonnegative finite catchment threshold; used only by backup coverage. */
  threshold: GraphDataView<'float32'>;
  maximumCandidates?: number;
  output: {
    selectedFacilities: GraphDataView<'uint32'>;
    assignments: GraphDataView<'uint32'>;
    objective: GraphDataView<'float32'>;
    /** [valid, feasibleIncumbent, enumerationComplete, examinedCandidates]. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Exhaustive facility candidates evaluated independently in fixed-size GPU batches.
 * Per-candidate sums and capacities retain input row order; hierarchical reductions preserve the
 * lowest candidate on ties. Incomplete enumeration never certifies infeasibility/optimality.
 *
 * Reference tier (contributor contract rule 2): an exact exhaustive search. Candidates are
 * row-parallel in batches of 65,536, with at most 16 batches, but the number of candidates grows
 * exponentially: 2^facilityCount subsets for backup coverage and p-dispersion, and
 * facilityCount^demandRows assignments for capacitated p-median. The real limit, enforced at
 * construction, is `candidateCount <= 2^20` (1,048,576) with at most 16 facilities. For capacitated
 * p-median that means demand rows are limited by the facilities, not by the 64-row ceiling that
 * applies to every operation: 20 rows at 2 facilities, 10 at 4, 5 at 16. For larger problems use
 * `GPULocationAllocation` (greedy with swap improvement).
 */
export class GPUNetworkFacilityOptimization implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUNetworkFacilityOptimizationProps;
  readonly candidateCount: number;
  constructor(props: GPUNetworkFacilityOptimizationProps) {
    this.id = props.id ?? 'network-facility-optimization';
    this.props = props;
    const {id} = this;
    validateCapacity(id, 'facilityCount', props.facilityCount, 1, MAXIMUM_FACILITY_COUNT);
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
    this.candidateCount =
      props.operation === 'capacitated-p-median'
        ? props.facilityCount ** props.demand.length
        : 2 ** props.facilityCount;
    // Exhaustive enumeration: facilityCount ** demand rows for capacitated-p-median, otherwise
    // 2 ** facilityCount. The product, not the demand row count, is the real limit.
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
    for (const view of [
      props.costs,
      props.demand,
      props.capacities,
      props.threshold,
      props.output.objective
    ])
      validatePackedView(view, ['float32'], this.id);
    for (const view of [
      props.output.selectedFacilities,
      props.output.assignments,
      props.output.status
    ])
      validatePackedView(view, ['uint32'], this.id);
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
      props.output.status.length === STATUS_WORD_COUNT,
      `output.status must hold ${STATUS_WORD_COUNT} rows`
    );
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.costs,
      props.demand,
      props.capacities,
      props.threshold
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, candidateCount} = this;
    const examined = Math.min(candidateCount, props.maximumCandidates ?? candidateCount);
    const capacitated = props.operation === 'capacitated-p-median';
    const dispersion = props.operation === 'p-dispersion';
    validateGraphViewsBelongToGraph(id, graph, [
      props.costs,
      props.demand,
      props.capacities,
      props.threshold,
      ...Object.values(props.output)
    ]);
    // Six words: batch start, winning primary/secondary/rank, arithmetic failure, valid input.
    const state = createTransientView(graph, `${id}-${props.operation}-state`, 'uint32', 6);
    const batchCapacity = clamp(examined, 1, 65536);
    const scores = createTransientView(
      graph,
      `${id}-${props.operation}-scores`,
      'uint32',
      batchCapacity * 4
    );
    const declarations = `const FACILITY_COUNT: u32 = ${props.facilityCount}u;
const DEMAND_COUNT: u32 = ${props.demand.length}u;
fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }`;
    const inputBindings = [
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {name: 'demand', view: props.demand, type: 'f32', access: 'read'},
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
  for (var row = 0u; row < DEMAND_COUNT; row++) {
    if (!finite(demand[demandOffset + row]) || demand[demandOffset + row] < 0.0) { return; }
  }
  ${capacitated ? 'for (var facility = 0u; facility < FACILITY_COUNT; facility++) { if (!finite(capacities[capacitiesOffset + facility]) || capacities[capacitiesOffset + facility] < 0.0) { return; } }' : ''}
  ${!capacitated && !dispersion ? 'if (!finite(threshold[thresholdOffset]) || threshold[thresholdOffset] < 0.0) { return; }' : ''}
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
    if (!finite(cost) || used[facility] > capacities[capacitiesOffset + facility]) { feasible = false; }
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
  ${!dispersion ? 'if (!finite(score) && feasible) { scores[offset + 3u] = 1u; return; }' : ''}
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
          {name: 'demand', view: props.demand, type: 'f32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {
            name: 'selected',
            view: props.output.selectedFacilities,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'assignments', view: props.output.assignments, type: 'u32', access: 'read_write'},
          {name: 'objective', view: props.output.objective, type: 'f32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations,
        body: `for (var word = 0u; word < 4u; word++) { status[statusOffset + word] = 0u; }
  var invalidBits = 0x7fc00000u; objective[objectiveOffset] = bitcast<f32>(invalidBits);
  for (var facility = 0u; facility < FACILITY_COUNT; facility++) { selected[selectedOffset + facility] = 0u; }
  for (var row = 0u; row < DEMAND_COUNT; row++) { assignments[assignmentsOffset + row] = 0xffffffffu; }
  if (state[stateOffset + 5u] == 0u || state[stateOffset + 4u] != 0u) { return; }
  let bestCandidate = state[stateOffset + 3u];
  let haveBest = bestCandidate != 0xffffffffu;
  status[statusOffset] = 1u; status[statusOffset + 1u] = u32(haveBest);
  status[statusOffset + 2u] = ${examined === candidateCount ? '1u' : '0u'};
  status[statusOffset + 3u] = ${examined}u;
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
    return nodes;
  }
}
