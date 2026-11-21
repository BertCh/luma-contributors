// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {snapshotLocationAllocationProps} from './location-allocation-construction';
import {createFacilitySwapNodes, type FacilitySwapContext} from './facility-swap-phases';
import {getFacilityCompactionLevelCounts} from './facility-selected-compaction';
import {validateSpatialOutputViews} from '../spatial-clustering/spatial-output-validation';
import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {IS_NAN_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateRequirement,
  validateCommandNodeBudget,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateSinglePackedView,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createSparseTransposeNodes,
  createSparseValidationNodes,
  getSparseColumnBindings,
  getSparseRowBindings,
  SPARSE_ROW_COST_WGSL,
  type GPULocationAllocationSparseCosts,
  type SparseTranspose
} from './location-allocation-sparse';

export type {GPULocationAllocationSparseCosts};

const OPERATION = 'GPULocationAllocation';

/** Workgroup size of the single-workgroup choose and finalize reductions. */
const CHOOSE_WORKGROUP_SIZE = 256;

/** Location-allocation objective implemented by {@link GPULocationAllocation}. */
export type GPULocationAllocationOperation = 'p-median' | 'maximum-coverage' | 'set-covering';

/**
 * Packed status words written by {@link GPULocationAllocation}.
 *
 * Words 4 to 6 describe the swap improvement phase and are all zero when it is disabled
 * (`maximumSwapIterations: 0`) or not applicable (`set-covering`).
 */
export const GPU_LOCATION_ALLOCATION_STATUS_LAYOUT = {
  selectedFacilityCount: 0,
  uncoveredDemandCount: 1,
  /**
   * `1` when the search finished: the facility budget was filled (p-median, maximum coverage) or
   * every demand row is covered (set covering). A supplied `costsComplete` must be nonzero;
   * independently, a supplied matrix `costLimit` must reach the requested cutoff.
   */
  complete: 2,
  invalidCount: 3,
  /** Number of improving swaps applied. */
  swapIterationCount: 4,
  /** `1` when the swap phase stopped because no strictly improving swap exists (local optimum). */
  converged: 5,
  /** `1` when `maximumSwapIterations` swaps were applied, so local optimality is not verified. */
  iterationLimitReached: 6,
  length: 7
} as const;

/** Default `maximumSwapIterations` of the p-median and maximum-coverage operations. */
export const GPU_LOCATION_ALLOCATION_DEFAULT_MAXIMUM_SWAP_ITERATIONS = 100;

/** Largest `maximumSwapIterations`: each round adds six nodes plus two per compaction level. */
export const GPU_LOCATION_ALLOCATION_MAXIMUM_SWAP_ITERATIONS = 1000;

/**
 * Command nodes the sparse transpose sort adds to
 * {@link GPULocationAllocation.estimatedCommandNodeCount}. A `GPUSort` adds a pass per key digit
 * and per factor of its workgroup size, so its node count depends on the entry count and the
 * facility count inside a published luma.gl class. Measured at most 35 for 3,000,000 entries with
 * 20 key bits, so 64 covers every accepted size.
 */
const SPARSE_TRANSPOSE_SORT_NODE_ALLOWANCE = 64;
/** Five indirect triples gate the list, prepare, removal, score and choose swap passes. */
const LOCATION_ALLOCATION_SWAP_DISPATCH_BYTE_LENGTH = 5 * 3 * 4;

/**
 * A swap must improve the objective by more than this fraction of the objective scale (the sum of
 * the finite weighted nearest-facility costs for p-median, the total demand weight for maximum
 * coverage) to be applied; for p-median a gain in reachable demand weight larger than this fraction
 * of the total weight takes precedence over cost. It keeps f32 summation noise from applying zero-gain swaps, so the
 * phase cannot cycle.
 */
export const GPU_LOCATION_ALLOCATION_SWAP_TOLERANCE = 1e-6;

/**
 * Largest `facilityCount` of {@link GPULocationAllocation} on `device`: the facility score table
 * holds two float32 values (8 bytes) per candidate in one storage binding, so 16777216 on a default
 * device (128 MiB bindings).
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPULocationAllocationMaximumFacilityCount(device: Device): number {
  return getStorageBindingElementLimit(device, 8);
}

/**
 * Largest `costs` row count (`facilityCount * demand rows`) of {@link GPULocationAllocation} on
 * `device`: the cost matrix holds 4 bytes per entry in one storage binding, so 33554432 on a default
 * device (128 MiB bindings).
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPULocationAllocationMaximumCostCount(device: Device): number {
  return getStorageBindingElementLimit(device, 4);
}

/**
 * Largest demand row count (`costs.length / facilityCount`) of {@link GPULocationAllocation} on
 * `device`: the swap phase packs three uint32 words (12 bytes) per demand row in one storage
 * binding, so 11184810 on a default device (128 MiB bindings). It applies to every operation.
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPULocationAllocationMaximumDemandCount(device: Device): number {
  return getStorageBindingElementLimit(device, 12);
}

/**
 * Largest `sparseCosts` entry count of {@link GPULocationAllocation} on `device`: the
 * facility-major transpose holds two uint32 words (8 bytes) per entry in one storage binding, so
 * 16777216 on a default device (128 MiB bindings).
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPULocationAllocationMaximumSparseEntryCount(device: Device): number {
  return getStorageBindingElementLimit(device, 8);
}

/**
 * Largest number of (selected slot, candidate) pairs one swap iteration scores on `device`: two
 * float32 scores (8 bytes) per pair in one storage binding, so 2^24 on a default device.
 */
function getMaximumSwapPairCount(device: Device): number {
  return getStorageBindingElementLimit(device, 8);
}

/** Caller-owned result views for {@link GPULocationAllocation}. */
export type GPULocationAllocationOutput = {
  /** One flag per candidate facility. */
  selectedFacilities: GraphDataView<'uint32'>;
  /** Selected facility per demand row, or `0xffffffff` when no selected facility is reachable. */
  assignments: GraphDataView<'uint32'>;
  /** Optional minimum selected-facility cost per demand row. */
  assignedCosts?: GraphDataView<'float32'>;
  /** One objective value: weighted cost for p-median, covered weight for maximum coverage, selected count for set covering. */
  objective: GraphDataView<'float32'>;
  /** Seven words indexed by {@link GPU_LOCATION_ALLOCATION_STATUS_LAYOUT}. */
  status: GraphDataView<'uint32'>;
};

/** Properties for {@link GPULocationAllocation}. */
export type GPULocationAllocationProps = {
  id?: string;
  /**
   * Row-major demand-by-facility cost matrix. Positive infinity means unreachable. At most
   * {@link getGPULocationAllocationMaximumCostCount} entries and
   * {@link getGPULocationAllocationMaximumDemandCount} demand rows. Exactly one of `costs` and
   * `sparseCosts` is required.
   */
  costs?: GraphDataView<'float32'>;
  /**
   * Sparse alternative to `costs`: CSR rows of (facility, cost) entries, where a pair that is not
   * listed is unreachable (`+Infinity`). It removes the dense `demand rows x facilityCount` limit;
   * the work and memory then scale with the entry count, at most
   * {@link getGPULocationAllocationMaximumSparseEntryCount} (at least one entry), and the demand
   * row count is `offsets.length - 1`, at most {@link getGPULocationAllocationMaximumDemandCount}.
   * Rows must list strictly ascending facilities below `facilityCount`; a violation counts in
   * `status` word 3 (`invalidCount`). Supported by every operation, with the same results as the
   * equivalent dense matrix for a finite `coverageDistance` (up to float32 summation order in
   * near-ties).
   */
  sparseCosts?: GPULocationAllocationSparseCosts;
  /**
   * Number of candidate facility columns in `costs` (or the facility index range of
   * `sparseCosts`). At most {@link getGPULocationAllocationMaximumFacilityCount}.
   */
  facilityCount: number;
  /** Optional non-negative demand weight per matrix row. Defaults to one. */
  demandWeights?: GraphDataView<'float32'>;
  /** Greedy objective. */
  operation: GPULocationAllocationOperation;
  /** Facility budget for p-median/MCLP and hard search cap for LSCP. */
  maximumFacilityCount: number;
  /**
   * Maximum number of Teitz-Bart vertex-substitution (swap) iterations run after the greedy
   * construction, an integer in `[0, 1000]`. Compile-time. Default 100 for `p-median` and
   * `maximum-coverage`; `0` disables the phase (pure greedy); `set-covering` has no swap phase and
   * rejects a positive value. Each iteration scores every (selected facility, unselected candidate)
   * swap in parallel with the Whitaker / Resende-Werneck gain decomposition (`O(facilityCount *
   * demandCount)` work, or the entry count with `sparseCosts`), applies the
   * single best strictly improving swap (ties go to the lowest selected facility, then the lowest
   * candidate), and stops when none improves. Requires
   * `maximumFacilityCount * facilityCount <= 2^24`.
   */
  maximumSwapIterations?: number;
  /** One per-frame distance threshold, required by maximum coverage and set covering. */
  coverageDistance?: GraphDataView<'float32'>;
  /**
   * Optional one-row completeness word of `costs` (or `sparseCosts`), for example the
   * `output.status.converged` of `GPUNetworkCostMatrix`: nonzero means every entry is final. A
   * matrix stopped at its round budget leaves `+Infinity` (or an overestimate) where it did not
   * finish, which reads as unreachable here. When supplied, status word `complete` is 0 unless
   * this word is nonzero and the `costLimit` rule holds (NET-101).
   */
  costsComplete?: GraphDataView<'uint32'>;
  /**
   * Optional one-row cost limit the matrix was built with (the `costLimit` view of
   * `GPUNetworkCostMatrix`). Evaluated independently of `costsComplete`: status word `complete` is 0 when the
   * requested cutoff exceeds it, that is `coverageDistance` for the coverage operations and
   * `+Infinity` for p-median, whose costs have no cutoff (any finite matrix limit can hide a
   * finite cost). Omit it when the producer ran without a limit, or when a p-median cutoff is part
   * of the problem.
   */
  costLimit?: GraphDataView<'float32'>;
  output: GPULocationAllocationOutput;
};

/** WGSL that flags an invalid demand weight or coverage distance in `status` word 3. */
function getWeightAndCoverageValidationWGSL(hasWeights: boolean, hasCoverage: boolean): string {
  return `${
    hasWeights
      ? `if (index < DEMAND_COUNT && isInvalid(demandWeights[demandWeightsOffset + index])) {
    atomicAdd(&status[statusOffset + 3u], 1u);
  }`
      : ''
  }
  ${
    hasCoverage
      ? `if (index == 0u && isInvalid(coverageDistance[coverageDistanceOffset])) {
    atomicAdd(&status[statusOffset + 3u], 1u);
  }`
      : ''
  }`;
}

/**
 * Deterministic bounded location-allocation over a caller-owned cost matrix.
 *
 * `p-median` greedily adds the facility with the largest reduction in weighted nearest-facility
 * cost. `maximum-coverage` greedily adds the facility covering the greatest remaining demand
 * weight within `coverageDistance`. `set-covering` greedily adds the facility covering the most
 * uncovered demand rows, whatever their weight (as in LSCP, `demandWeights` does not enter the
 * choice), until every reachable demand row is covered or `maximumFacilityCount` is reached. Ties
 * select the lowest facility row. All assignments and stopping evidence remain GPU-resident.
 *
 * Unreachable demand (a `+Infinity` cost to every selected facility) is never summed into the
 * p-median objective: the objective is the weighted cost of the reachable rows, unreachable rows
 * are counted in the `uncoveredDemandCount` status word (infeasibility is reported there, not in
 * `invalidCount`, which marks invalid input and stops the run), their assignment is `0xffffffff`
 * and their `assignedCosts` entry is `+Infinity`. Greedy and swap steps compare candidates
 * lexicographically: first the demand weight left unreachable, then the reachable cost, so one
 * unreachable row never swamps the float32 cost differences of the others.
 *
 * After the greedy construction, `p-median` and `maximum-coverage` run a Teitz-Bart
 * vertex-substitution improvement phase (see `maximumSwapIterations`): the best strictly improving
 * swap of a selected facility for an unselected candidate is applied until none exists. The result
 * is a swap-local optimum, not the integer-programming optimum. Greedy maximum coverage alone has
 * the `(1 - 1/e)` guarantee relative to the optimum (Nemhauser, Wolsey and Fisher); greedy
 * p-median has no constant-factor guarantee, which is why the swap phase is on by default. The swap
 * phase never makes the objective worse, so the greedy guarantee still holds. Swap progress is
 * reported in the `swapIterationCount`, `converged` and `iterationLimitReached` status words.
 *
 * Swaps are scored with the Whitaker / Resende-Werneck gain decomposition, one pass per candidate. The
 * costs are a dense matrix (`costs`) or sparse CSR rows (`sparseCosts`, unlisted pairs unreachable).
 *
 * One allocation graph is limited to 32,768 command nodes (`GPU_MAXIMUM_COMMAND_NODE_COUNT`).
 * The estimate starts at four nodes for dense costs or 73 for sparse costs. Sparse costs add
 * one validation node when weights are supplied or the operation is not p-median. Either route
 * adds one node when `costsComplete` or `costLimit` is supplied and one for p-median or a weighted
 * swap phase. Each requested selection adds three nodes. An enabled swap phase adds one setup
 * node and `(6 + 2 * L) * maximumSwapIterations` nodes. `L` counts the facility scan and each
 * subsequent ceil-division by 256 through the first level of at most 256 entries. The sparse
 * estimate includes a 64-node allowance for the transpose sort.
 *
 * Construction rejects larger graphs even when all buffers fit the device. A dense p-median
 * request with 8,000 facilities, 7,800 selections, one demand row and 1,000 swaps requires 33,406
 * nodes on a device with 512 MiB bindings. Lower `maximumSwapIterations` or `maximumFacilityCount`.
 *
 * Run them after `GPUNetworkCostMatrix` for network location-allocation or provide any dense cost matrix.
 */
export class GPULocationAllocation implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPULocationAllocationProps;
  readonly demandCount: number;
  /** Resolved `maximumSwapIterations` (0 when the phase is disabled or not applicable). */
  readonly maximumSwapIterations: number;
  /**
   * Upper bound of the command nodes `getCommandNodes` returns. Exact for dense costs.
   * The sparse transpose sort allowance can overestimate the true count by up to 64.
   */
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPULocationAllocationProps) {
    props = snapshotLocationAllocationProps(props);
    this.id = props.id ?? 'location-allocation';
    this.props = props;
    const {id} = this;
    const {output} = props;
    validateRequirement(
      id,
      ['p-median', 'maximum-coverage', 'set-covering'].includes(props.operation),
      `operation must be p-median, maximum-coverage or set-covering; received ${props.operation}`
    );
    validateRequirement(
      id,
      !(props.costs && props.sparseCosts),
      `costs and sparseCosts are mutually exclusive`
    );
    validateRequirement(id, !(!props.costs && !props.sparseCosts), `requires costs or sparseCosts`);
    validateCapacity(id, 'facilityCount', props.facilityCount);
    if (props.costs) {
      validateSinglePackedView(props.costs, ['float32'], `${id} costs`);
      validateRequirement(
        id,
        !(
          props.costs.length < props.facilityCount || props.costs.length % props.facilityCount !== 0
        ),
        `costs length must be a positive multiple of facilityCount`
      );
      this.demandCount = props.costs.length / props.facilityCount;
    } else {
      const sparse = props.sparseCosts!;
      validateSinglePackedView(sparse.offsets, ['uint32'], `${id} sparseCosts.offsets`);
      validateSinglePackedView(sparse.facilities, ['uint32'], `${id} sparseCosts.facilities`);
      validateSinglePackedView(sparse.costs, ['float32'], `${id} sparseCosts.costs`);
      validateRequirement(
        id,
        !(sparse.offsets.length < 2),
        `sparseCosts.offsets must hold demand rows + 1 offsets`
      );
      validateRequirement(
        id,
        !(sparse.facilities.length < 1 || sparse.facilities.length !== sparse.costs.length),
        `sparseCosts.facilities and costs must be equally long and not empty`
      );
      this.demandCount = sparse.offsets.length - 1;
    }
    validateCapacity(
      id,
      'maximumFacilityCount',
      props.maximumFacilityCount,
      1,
      props.facilityCount
    );
    const requestedSwaps = props.maximumSwapIterations;
    if (requestedSwaps !== undefined) {
      validateCapacity(
        id,
        'maximumSwapIterations',
        requestedSwaps,
        0,
        GPU_LOCATION_ALLOCATION_MAXIMUM_SWAP_ITERATIONS
      );
    }
    if (props.operation === 'set-covering') {
      validateRequirement(
        id,
        !requestedSwaps,
        `set-covering has no swap phase; omit maximumSwapIterations`
      );
      this.maximumSwapIterations = 0;
    } else {
      this.maximumSwapIterations =
        requestedSwaps ?? GPU_LOCATION_ALLOCATION_DEFAULT_MAXIMUM_SWAP_ITERATIONS;
    }
    if (props.demandWeights) {
      validateSinglePackedView(props.demandWeights, ['float32'], `${id} demandWeights`);
      validateRequirement(
        id,
        !(props.demandWeights.length !== this.demandCount),
        `demandWeights length must equal the demand row count`
      );
    }
    if (props.operation !== 'p-median') {
      validateRequirement(
        id,
        !!props.coverageDistance,
        `coverageDistance is required for ${props.operation}`
      );
      validateSinglePackedView(props.coverageDistance, ['float32'], `${id} coverageDistance`);
      validateRequirement(
        id,
        !(props.coverageDistance.length < 1),
        `coverageDistance must hold one value`
      );
    } else if (props.coverageDistance) {
      validateSinglePackedView(props.coverageDistance, ['float32'], `${id} coverageDistance`);
    }
    for (const [name, view, format] of [
      ['costsComplete', props.costsComplete, 'uint32'],
      ['costLimit', props.costLimit, 'float32']
    ] as const) {
      if (view) {
        validateSinglePackedView(view, [format], `${id} ${name}`);
        validateRequirement(id, !(view.length !== 1), `${name} must contain exactly one row`);
      }
    }
    validateSinglePackedView(
      output.selectedFacilities,
      ['uint32'],
      `${id} output.selectedFacilities`
    );
    validateSinglePackedView(output.assignments, ['uint32'], `${id} output.assignments`);
    validateSinglePackedView(output.status, ['uint32'], `${id} output.status`);
    validateSinglePackedView(output.objective, ['float32'], `${id} output.objective`);
    validateRequirement(
      id,
      !(output.selectedFacilities.length < props.facilityCount),
      `output.selectedFacilities must hold one flag per facility`
    );
    validateRequirement(
      id,
      !(output.assignments.length < this.demandCount),
      `output.assignments must hold one row per demand`
    );
    validateRequirement(id, !(output.objective.length < 1), `output.objective must hold one value`);
    validateRequirement(
      id,
      !(output.status.length < GPU_LOCATION_ALLOCATION_STATUS_LAYOUT.length),
      `output.status must hold ${GPU_LOCATION_ALLOCATION_STATUS_LAYOUT.length} words`
    );
    if (output.assignedCosts) {
      validateSinglePackedView(output.assignedCosts, ['float32'], `${id} output.assignedCosts`);
      validateRequirement(
        id,
        !(output.assignedCosts.length < this.demandCount),
        `output.assignedCosts must hold one row per demand`
      );
    }
    validateSpatialOutputViews(id, output, [
      [output.selectedFacilities, output.assignments, output.objective, output.status],
      {read: [output.selectedFacilities], write: [output.assignments, output.assignedCosts]}
    ]);
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.selectedFacilities,
        output.assignments,
        output.assignedCosts,
        output.objective,
        output.status
      ],
      [
        ...this.getCostViews(),
        props.demandWeights,
        props.coverageDistance,
        props.costsComplete,
        props.costLimit
      ]
    );
    const hasSwaps = this.maximumSwapIterations > 0;
    // Sparse setup: two structure validations, a weight and coverage validation when either exists,
    // and the transpose (row scatter, sort, segment offsets, gather). Dense setup: one validation.
    const setupNodeCount = props.sparseCosts
      ? 2 +
        Number(props.demandWeights !== undefined || props.operation !== 'p-median') +
        3 +
        SPARSE_TRANSPOSE_SORT_NODE_ALLOWANCE
      : 1;
    this.estimatedCommandNodeCount =
      // initialize, finalize, the costs completeness node and the assign node (two for sparse costs)
      2 +
      Number(Boolean(props.costsComplete || props.costLimit)) +
      Number(Boolean(props.sparseCosts)) +
      1 +
      setupNodeCount +
      Number(props.operation === 'p-median' || (hasSwaps && props.demandWeights !== undefined)) +
      // score, choose and update per greedy step
      3 * props.maximumFacilityCount +
      // Swap initialization; each round has five scoring/application nodes, a compaction
      // dispatch and two nodes per hierarchical scan level (scan, offsets/scatter).
      (hasSwaps
        ? 1 +
          (6 + 2 * getFacilityCompactionLevelCounts(props.facilityCount).length) *
            this.maximumSwapIterations
        : 0);
    // Every greedy step and swap iteration is unrolled; lower maximumFacilityCount or
    // maximumSwapIterations.
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  /** The cost views read by the kernels: the dense matrix, or the three CSR views. */
  private getCostViews(): GraphDataView[] {
    const {costs, sparseCosts} = this.props;
    return costs ? [costs] : [sparseCosts!.offsets, sparseCosts!.facilities, sparseCosts!.costs];
  }

  /**
   * Teitz-Bart vertex substitution scored with the Whitaker / Resende-Werneck gain decomposition.
   *
   * For a swap that removes selected facility `r` and adds candidate `i`, every demand row `j`
   * keeps its nearest selected facility unless `j` is served by `r` (p-median) or covered only by
   * `r` (maximum coverage); the improvement is therefore `gain(i) - loss(r) + extra(i, r)`:
   * `gain(i)` sums over all rows what `i` would win if nothing were removed, `loss(r)` is what the
   * rows served by `r` lose when it goes (their second-nearest facility takes over) and `extra(i,
   * r)` corrects the rows served by `r` that `i` also reaches. `gain` and `extra` only involve the
   * entries `(j, i)` that beat the row's nearest (second-nearest) cost, so one pass over candidate
   * `i`'s costs scores all of its `maximumFacilityCount` pairs: `O(facilityCount * demandCount)`
   * work per iteration instead of `O(maximumFacilityCount * facilityCount * demandCount)`.
   *
   * Each iteration: a hierarchical 256-lane prefix scan and parallel scatter build the ascending
   * selected-facility list and inverse slots, without a serial facility scan; per demand
   * row the nearest and second-nearest selected cost and the nearest slot (or the covering count
   * and the sole covering slot); one workgroup per slot summing `loss(r)` (plus one workgroup for
   * the objective scale); one thread per candidate scoring its pairs, each in a fixed ascending
   * demand order and without atomics; a single-workgroup argmax (ties to the lowest pair index,
   * i.e. lowest selected facility then lowest candidate); and the swap itself. Once no swap
   * improves, the apply kernel zeroes the swap dispatch and the list dispatch node disarms
   * every compaction level. A converged run costs two one-thread dispatches per remaining
   * iteration (rule 13).
   */
  private getSwapNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    context: FacilitySwapContext
  ): GPUCommandNode<Parameters>[] {
    return createFacilitySwapNodes(graph, this, context);
  }

  /**
   * Sparse input setup: validation of the CSR structure and of the weights and coverage distance,
   * then the facility-major transpose the facility-side loops (greedy scores, swap candidates) read.
   */
  private getSparseSetupNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    context: {
      declarations: string;
      weightBinding: WGSLKernelBinding[];
      coverageBinding: WGSLKernelBinding[];
      needsCoverage: boolean;
    }
  ): {nodes: GPUCommandNode<Parameters>[]; transpose: SparseTranspose} {
    const {id, props, demandCount} = this;
    const sparse = props.sparseCosts!;
    const {declarations, weightBinding, coverageBinding, needsCoverage} = context;
    const nodes: GPUCommandNode<Parameters>[] = [
      ...createSparseValidationNodes(graph, {
        id,
        operation: OPERATION,
        sparse,
        declarations,
        status: props.output.status
      })
    ];
    if (props.demandWeights || needsCoverage) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-validate-inputs`,
          operation: OPERATION,
          variant: 'validate-sparse-inputs',
          invocationCount: Math.max(demandCount, 1),
          bindings: [
            ...weightBinding,
            ...coverageBinding,
            {name: 'status', view: props.output.status, type: 'atomic<u32>', access: 'read_write'}
          ],
          declarations,
          body: getWeightAndCoverageValidationWGSL(props.demandWeights !== undefined, needsCoverage)
        })
      );
    }
    const {nodes: transposeNodes, transpose} = createSparseTransposeNodes(graph, {
      id,
      operation: OPERATION,
      sparse,
      facilityCount: props.facilityCount,
      declarations
    });
    nodes.push(...transposeNodes);
    return {nodes, transpose};
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, demandCount} = this;
    const {output, facilityCount} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      ...this.getCostViews(),
      props.demandWeights,
      props.coverageDistance,
      props.costsComplete,
      props.costLimit,
      output.selectedFacilities,
      output.assignments,
      output.assignedCosts,
      output.objective,
      output.status
    ]);
    validateGraphChunkBindingRanges(graph, id, {
      ...Object.fromEntries(this.getCostViews().map((view, index) => [`costs[${index}]`, view])),
      demandWeights: props.demandWeights,
      coverageDistance: props.coverageDistance,
      costsComplete: props.costsComplete,
      costLimit: props.costLimit,
      ...output
    });
    validateCapacity(
      id,
      'facilityCount',
      props.facilityCount,
      1,
      getGPULocationAllocationMaximumFacilityCount(graph.device)
    );
    if (props.costs) {
      validateMaximumCount(
        id,
        'costs rows',
        props.costs.length,
        getGPULocationAllocationMaximumCostCount(graph.device)
      );
    } else {
      validateMaximumCount(
        id,
        'sparseCosts entries',
        props.sparseCosts!.facilities.length,
        getGPULocationAllocationMaximumSparseEntryCount(graph.device)
      );
    }
    validateMaximumCount(
      id,
      'demand rows',
      demandCount,
      getGPULocationAllocationMaximumDemandCount(graph.device)
    );
    if (this.maximumSwapIterations > 0) {
      const maximumSwapPairs = getMaximumSwapPairCount(graph.device);
      validateRequirement(
        id,
        !(props.maximumFacilityCount * props.facilityCount > maximumSwapPairs),
        `swap phase needs maximumFacilityCount * facilityCount <= ${maximumSwapPairs}; reduce them or set maximumSwapIterations: 0`
      );
    }
    for (const [name, byteLength] of [
      ['demand scratch', demandCount * (this.maximumSwapIterations > 0 ? 12 : 4)],
      ['facility scores', facilityCount * 8],
      [
        'swap totals',
        this.maximumSwapIterations > 0 ? (3 * props.maximumFacilityCount + 5) * 4 : 0
      ],
      [
        'swap pair scores',
        this.maximumSwapIterations > 0 ? 8 * props.maximumFacilityCount * facilityCount : 0
      ],
      ['sparse transpose entries', props.sparseCosts ? props.sparseCosts.facilities.length * 8 : 0],
      ['sparse transpose offsets', props.sparseCosts ? (facilityCount + 1) * 4 : 0],
      [
        'swap dispatch',
        this.maximumSwapIterations > 0 ? LOCATION_ALLOCATION_SWAP_DISPATCH_BYTE_LENGTH : 0
      ]
    ] as const)
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        byteLength,
        'reduce demand or facility counts'
      );
    const bestCosts = createTransientView(graph, `${id}-best-costs`, 'float32', demandCount);
    const facilityScores = createTransientView(
      graph,
      `${id}-facility-scores`,
      'float32',
      2 * facilityCount
    );
    // Word 0 is the chosen facility; word 1 is the common product exponent; word 2 is max weight.
    const chosenFacility = createTransientView(graph, `${id}-chosen-facility`, 'uint32', 3);
    const demandObjectives = createTransientView(
      graph,
      `${id}-demand-objectives`,
      'float32',
      demandCount
    );
    const uncovered = createTransientView(graph, `${id}-uncovered`, 'uint32', demandCount);
    const mode =
      props.operation === 'p-median' ? 0 : props.operation === 'maximum-coverage' ? 1 : 2;
    const needsCoverage = mode !== 0;
    const swaps = this.maximumSwapIterations;
    const sparse = props.sparseCosts;
    const costBindings: WGSLKernelBinding[] = sparse
      ? getSparseRowBindings(sparse)
      : [{name: 'costs', view: props.costs!, type: 'f32', access: 'read'}];
    const declarations = `const DEMAND_COUNT: u32 = ${demandCount}u;
const FACILITY_COUNT: u32 = ${facilityCount}u;${
      sparse ? `\nconst SPARSE_ENTRY_COUNT: u32 = ${sparse.facilities.length}u;` : ''
    }
const MAXIMUM_FACILITIES: u32 = ${props.maximumFacilityCount}u;
const MODE: u32 = ${mode}u;
const NONE: u32 = 0xffffffffu;
${IS_NAN_BITS_WGSL}
fn isInvalid(value: f32) -> bool { return isNanBits(value) || value < 0.0; }
// Non-finite constant-expression bitcasts are invalid WGSL; a function-local var is runtime.
fn getUnreachableCost() -> f32 {
  var bits = 0x7f800000u;
  return bitcast<f32>(bits);
}
// Inspect the representation explicitly: every finite nonnegative cost is reachable.
fn isReachable(cost: f32) -> bool { return (bitcast<u32>(cost) & 0x7fffffffu) < 0x7f800000u; }
// Leave ordinary scores in caller units. At overflow risk, all candidates share the same
// binary exponent, with headroom for the complete demand sum and finite weighted products.
fn getAllocationValueExponent(bits: u32) -> i32 {
  let exponent = i32((bits >> 23u) & 255u);
  if (exponent != 0) { return exponent - 127; }
  return 31 - i32(countLeadingZeros(bits & 0x7fffffu)) - 149;
}
fn getAllocationWeightExponent(weightMaximum: u32) -> i32 {
  return max(0, getAllocationValueExponent(weightMaximum) + ${Math.ceil(Math.log2(demandCount))} - 120);
}
fn getAllocationProductExponent(cost: u32, weight: u32) -> u32 {
  // Exclude zero products; independent maxima can belong to different, noncontributing rows.
  if (cost == 0u || weight == 0u || cost >= 0x7f800000u || weight >= 0x7f800000u) { return 0u; }
  return u32(max(0, getAllocationValueExponent(cost) + getAllocationValueExponent(weight) + ${Math.ceil(Math.log2(demandCount))} - 120));
}
fn getAllocationScoreCost(cost: f32, weight: f32, exponent: i32) -> f32 {
  if (!isReachable(weight)) { return cost * weight; }
  if (exponent == 0) { return cost * weight; }
  if (cost == 0.0 || weight == 0.0) { return 0.0; }
  let costParts = frexp(cost);
  let weightParts = frexp(weight);
  return ldexp(costParts.fract * weightParts.fract, costParts.exp + weightParts.exp - exponent);
}
// A cost above twice the current objective cannot improve a fully reachable solution. Clip
// weighted costs before differencing, so a huge sparse loss and its extra term cannot erase
// the useful gain. Check the exponent before ldexp, avoiding an overflowing intermediate.
fn getAllocationSwapCost(cost: f32, weight: f32, exponent: i32, limit: f32) -> f32 {
  if (!isReachable(limit)) { return getAllocationScoreCost(cost, weight, exponent); }
  if (limit == 0.0 || cost == 0.0 || weight == 0.0) { return 0.0; }
  let costParts = frexp(cost);
  let weightParts = frexp(weight);
  let productExponent = costParts.exp + weightParts.exp - exponent;
  if (productExponent > getAllocationValueExponent(bitcast<u32>(limit)) + 2) { return limit; }
  return min(ldexp(costParts.fract * weightParts.fract, productExponent), limit);
}`;
    const swapDeclarations = `${declarations}
const MAXIMUM_SWAPS: u32 = ${swaps}u;
const SWAP_SLOTS: u32 = ${props.maximumFacilityCount}u;
const SWAP_TOLERANCE: f32 = ${GPU_LOCATION_ALLOCATION_SWAP_TOLERANCE};`;
    // Word 0: stopped (no improving swap, or invalid input), word 1: swaps applied, word 2: chosen pair.
    const swapState =
      swaps > 0 ? createTransientView(graph, `${id}-swap-state`, 'uint32', 3) : undefined!;
    const weightBinding: WGSLKernelBinding[] = props.demandWeights
      ? [{name: 'demandWeights', view: props.demandWeights, type: 'f32', access: 'read'}]
      : [];
    const coverageBinding: WGSLKernelBinding[] = needsCoverage
      ? [
          {
            name: 'coverageDistance',
            view: props.coverageDistance!,
            type: 'f32',
            access: 'read'
          }
        ]
      : [];
    const demandWeightExpression = props.demandWeights
      ? 'demandWeights[demandWeightsOffset + demand]'
      : '1.0';
    // Coverage modes stop choosing once no facility improves the objective: the choose kernel then
    // zeroes the indirect dispatch of the score and update kernels (slots 0 and 1), so the remaining
    // steps skip the demand-by-facility scan. p-median only stops on invalid input.
    const greedyGate = needsCoverage
      ? createWGSLDispatchGate(graph, {
          id: `${id}-greedy-dispatch`,
          operation: OPERATION,
          shapes: [[facilityCount], [demandCount]]
        })
      : null;
    const greedyGateBinding: WGSLKernelBinding[] = greedyGate ? [greedyGate.binding] : [];
    const sparseSetup = sparse
      ? this.getSparseSetupNodes(graph, {
          declarations,
          weightBinding,
          coverageBinding,
          needsCoverage
        })
      : undefined;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-initialize`,
        operation: OPERATION,
        variant: 'initialize',
        invocationCount: Math.max(
          facilityCount,
          demandCount,
          GPU_LOCATION_ALLOCATION_STATUS_LAYOUT.length
        ),
        bindings: [
          {
            name: 'selected',
            view: output.selectedFacilities,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'assignments', view: output.assignments, type: 'u32', access: 'read_write'},
          {name: 'bestCosts', view: bestCosts, type: 'f32', access: 'read_write'},
          {name: 'objective', view: output.objective, type: 'f32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'},
          {name: 'chosen', view: chosenFacility, type: 'u32', access: 'read_write'},
          ...greedyGateBinding
        ],
        declarations,
        body: `if (index < FACILITY_COUNT) {
    selected[selectedOffset + index] = 0u;
  }
  if (index < DEMAND_COUNT) {
    bestCosts[bestCostsOffset + index] = getUnreachableCost();
    assignments[assignmentsOffset + index] = NONE;
  }
  if (index < ${GPU_LOCATION_ALLOCATION_STATUS_LAYOUT.length}u) {
    status[statusOffset + index] = select(0u, DEMAND_COUNT, index == 1u);
  }
  if (index == 0u) {
    objective[objectiveOffset] = 0.0;
    // Any value but NONE: the first choose step is not stopped.
    chosen[chosenOffset] = 0u;
    chosen[chosenOffset + 1u] = 0u;
    chosen[chosenOffset + 2u] = ${props.demandWeights ? '0u' : '0x3f800000u'};
    ${greedyGate ? greedyGate.armWGSL : ''}
  }`
      }),
      ...(sparseSetup
        ? sparseSetup.nodes
        : [
            createWGSLKernelNode(graph, {
              id: `${id}-validate`,
              operation: OPERATION,
              variant: 'validate',
              invocationCount: Math.max(props.costs!.length, demandCount, 1),
              bindings: [
                {name: 'costs', view: props.costs!, type: 'f32', access: 'read'},
                ...weightBinding,
                ...coverageBinding,
                {name: 'status', view: output.status, type: 'atomic<u32>', access: 'read_write'}
              ],
              declarations,
              body: `if (index < DEMAND_COUNT * FACILITY_COUNT && isInvalid(costs[costsOffset + index])) {
    atomicAdd(&status[statusOffset + 3u], 1u);
  }
  ${getWeightAndCoverageValidationWGSL(props.demandWeights !== undefined, needsCoverage)}`
            })
          ])
    ];

    if (mode === 0 || (swaps > 0 && props.demandWeights !== undefined)) {
      const entryCount = sparse ? sparse.costs.length : props.costs!.length;
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-score-scale`,
          operation: OPERATION,
          variant: 'score-scale',
          invocationCount: mode === 0 ? Math.max(entryCount, demandCount) : demandCount,
          bindings: [
            ...(mode !== 0
              ? []
              : sparse
                ? [
                    {
                      name: 'columnEntries',
                      view: sparseSetup!.transpose.columnEntries,
                      type: 'u32',
                      access: 'read'
                    } as const
                  ]
                : [{name: 'costs', view: props.costs!, type: 'u32', access: 'read'} as const]),
            ...weightBinding,
            {name: 'chosen', view: chosenFacility, type: 'atomic<u32>', access: 'read_write'}
          ],
          declarations,
          body: `${
            mode !== 0
              ? ''
              : `if (index < ${entryCount}u) {
    let demand = ${sparse ? 'columnEntries[columnEntriesOffset + 2u * index]' : 'index / FACILITY_COUNT'};
    let bits = ${sparse ? 'columnEntries[columnEntriesOffset + 2u * index + 1u]' : 'costs[costsOffset + index]'};
    if (demand < DEMAND_COUNT) {
      let weight = bitcast<u32>(${demandWeightExpression});
      atomicMax(&chosen[chosenOffset + 1u], getAllocationProductExponent(bits, weight));
    }
  }`
          }
  ${
    props.demandWeights
      ? `if (index < DEMAND_COUNT) {
    let bits = bitcast<u32>(demandWeights[demandWeightsOffset + index]);
    if (bits < 0x7f800000u) { atomicMax(&chosen[chosenOffset + 2u], bits); }
  }`
      : ''
  }`
        })
      );
    }

    for (let step = 0; step < props.maximumFacilityCount; step++) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-score-${step}`,
          operation: OPERATION,
          variant: `${props.operation}-score`,
          invocationCount: facilityCount,
          ...(greedyGate ? greedyGate.getGate(0) : {}),
          bindings: [
            ...(sparseSetup
              ? getSparseColumnBindings(sparseSetup.transpose)
              : [{name: 'costs', view: props.costs!, type: 'f32', access: 'read'} as const]),
            {name: 'selected', view: output.selectedFacilities, type: 'u32', access: 'read'},
            {name: 'bestCosts', view: bestCosts, type: 'f32', access: 'read'},
            {name: 'scores', view: facilityScores, type: 'f32', access: 'read_write'},
            {name: 'status', view: output.status, type: 'u32', access: 'read'},
            ...(mode === 0
              ? [{name: 'chosen', view: chosenFacility, type: 'u32', access: 'read'} as const]
              : []),
            ...weightBinding,
            ...coverageBinding
          ],
          declarations,
          body: `let facility = index;
  let unavailableScore = select(-1.0, getUnreachableCost(), MODE == 0u);
  if (status[statusOffset + 3u] != 0u || selected[selectedOffset + facility] != 0u) {
    scores[scoresOffset + 2u * facility] = unavailableScore;
    scores[scoresOffset + 2u * facility + 1u] = unavailableScore;
    return;
  }
  let threshold = ${needsCoverage ? 'coverageDistance[coverageDistanceOffset]' : '0.0'};
  let costScoreExponent = ${mode === 0 ? 'i32(chosen[chosenOffset + 1u])' : '0'};
  let reachScoreExponent = ${mode === 0 ? 'getAllocationWeightExponent(chosen[chosenOffset + 2u])' : '0'};
  // p-median scores are lexicographic: (demand weight left unreachable, weighted cost of the
  // reachable rows). Coverage modes use only the first value: maximum coverage the demand weight
  // newly covered, set covering the number of rows newly covered (a row of weight 0 must still
  // be covered), counted in u32 so the count is exact.
  var score = 0.0;
  var reachableCost = 0.0;
  var newlyCoveredRowCount = 0u;
  ${
    sparseSetup
      ? `// Sparse: the facility only changes the rows it lists, so the score is the negated gain
  // (reachable weight won, cost saved), which orders candidates like the dense totals.
  let columnEnd = columnOffsets[columnOffsetsOffset + facility + 1u];
  for (var entry = columnOffsets[columnOffsetsOffset + facility]; entry < columnEnd; entry++) {
    let demand = columnEntries[columnEntriesOffset + 2u * entry];
    let candidate = bitcast<f32>(columnEntries[columnEntriesOffset + 2u * entry + 1u]);
    let current = bestCosts[bestCostsOffset + demand];
    if (MODE == 0u) {
      if (candidate < current) {
        if (isReachable(current)) {
          reachableCost += getAllocationScoreCost(current - candidate, ${demandWeightExpression}, costScoreExponent);
        } else {
          score += getAllocationScoreCost(1.0, ${demandWeightExpression}, reachScoreExponent);
          reachableCost -= getAllocationScoreCost(candidate, ${demandWeightExpression}, costScoreExponent);
        }
      }
    } else if ((!isReachable(current) || current > threshold) && isReachable(candidate) && candidate <= threshold) {
      score += ${demandWeightExpression};
      newlyCoveredRowCount += 1u;
    }
  }
  if (MODE == 0u) {
    score = -score;
    reachableCost = -reachableCost;
  }`
      : `for (var demand = 0u; demand < DEMAND_COUNT; demand++) {
    let current = bestCosts[bestCostsOffset + demand];
    let candidate = costs[costsOffset + demand * FACILITY_COUNT + facility];
    if (MODE == 0u) {
      let next = min(current, candidate);
      if (isReachable(next)) {
        reachableCost += getAllocationScoreCost(next, ${demandWeightExpression}, costScoreExponent);
      } else {
        score += getAllocationScoreCost(1.0, ${demandWeightExpression}, reachScoreExponent);
      }
    } else if ((!isReachable(current) || current > threshold) && isReachable(candidate) && candidate <= threshold) {
      score += ${demandWeightExpression};
      newlyCoveredRowCount += 1u;
    }
  }`
  }
  if (MODE == 2u) {
    score = f32(newlyCoveredRowCount);
  }
  scores[scoresOffset + 2u * facility] = score;
  scores[scoresOffset + 2u * facility + 1u] = reachableCost;`
        }),
        createWGSLKernelNode(graph, {
          id: `${id}-choose-${step}`,
          operation: OPERATION,
          variant: `${props.operation}-choose`,
          invocationCount: CHOOSE_WORKGROUP_SIZE,
          workgroupSize: CHOOSE_WORKGROUP_SIZE,
          guardIndex: false,
          bindings: [
            {name: 'selected', view: output.selectedFacilities, type: 'u32', access: 'read_write'},
            {name: 'scores', view: facilityScores, type: 'f32', access: 'read'},
            {name: 'chosen', view: chosenFacility, type: 'u32', access: 'read_write'},
            {name: 'status', view: output.status, type: 'u32', access: 'read_write'},
            ...greedyGateBinding
          ],
          declarations: `${declarations}
const CHOOSE_WORKGROUP_SIZE: u32 = ${CHOOSE_WORKGROUP_SIZE}u;
var<workgroup> sharedScore: array<f32, ${CHOOSE_WORKGROUP_SIZE}>;
var<workgroup> sharedSecondary: array<f32, ${CHOOSE_WORKGROUP_SIZE}>;
var<workgroup> sharedFacility: array<u32, ${CHOOSE_WORKGROUP_SIZE}>;
// p-median: lowest (unreachable weight, reachable cost); coverage: highest score.
fn isBetterCandidate(
  score: f32, secondary: f32, facility: u32,
  bestScore: f32, bestSecondary: f32, bestFacility: u32
) -> bool {
  if (MODE == 0u) {
    if (score != bestScore) { return score < bestScore; }
    if (secondary != bestSecondary) { return secondary < bestSecondary; }
    return facility < bestFacility;
  }
  return score > bestScore || (score == bestScore && facility < bestFacility);
}`,
          // One workgroup scans the facilities with a strided loop, then tree-reduces. The
          // (score, facility) order is total, so the lowest-facility tie rule is unchanged.
          // No early return: every invocation must reach the barriers.
          // A coverage step after one that chose nothing is stopped: its score kernel was skipped,
          // so the scores are stale. Every lane reads chosen before the first barrier and lane 0
          // writes it after the last one.
          body: `var bestFacility = NONE;
  var bestScore = select(-1.0, getUnreachableCost(), MODE == 0u);
  var bestSecondary = select(-1.0, getUnreachableCost(), MODE == 0u);
  let stopped = MODE != 0u && chosen[chosenOffset] == NONE;
  if (status[statusOffset + 3u] == 0u && !stopped) {
    for (var facility = localInvocationIndex; facility < FACILITY_COUNT; facility += CHOOSE_WORKGROUP_SIZE) {
      if (selected[selectedOffset + facility] != 0u) { continue; }
      let score = scores[scoresOffset + 2u * facility];
      let secondary = scores[scoresOffset + 2u * facility + 1u];
      if (isBetterCandidate(score, secondary, facility, bestScore, bestSecondary, bestFacility)) {
        bestFacility = facility;
        bestScore = score;
        bestSecondary = secondary;
      }
    }
  }
  sharedScore[localInvocationIndex] = bestScore;
  sharedSecondary[localInvocationIndex] = bestSecondary;
  sharedFacility[localInvocationIndex] = bestFacility;
  workgroupBarrier();
  for (var stride = CHOOSE_WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride &&
        isBetterCandidate(
          sharedScore[localInvocationIndex + stride],
          sharedSecondary[localInvocationIndex + stride],
          sharedFacility[localInvocationIndex + stride],
          sharedScore[localInvocationIndex],
          sharedSecondary[localInvocationIndex],
          sharedFacility[localInvocationIndex])) {
      sharedScore[localInvocationIndex] = sharedScore[localInvocationIndex + stride];
      sharedSecondary[localInvocationIndex] = sharedSecondary[localInvocationIndex + stride];
      sharedFacility[localInvocationIndex] = sharedFacility[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    bestFacility = sharedFacility[0];
    if (bestFacility != NONE && (MODE == 0u || sharedScore[0] > 0.0)) {
      selected[selectedOffset + bestFacility] = 1u;
      status[statusOffset] += 1u;
    } else {
      bestFacility = NONE;${
        greedyGate
          ? `
      ${greedyGate.disarmWGSL}`
          : ''
      }
    }
    chosen[chosenOffset] = bestFacility;
  }`
        }),
        createWGSLKernelNode(graph, {
          id: `${id}-update-${step}`,
          operation: OPERATION,
          variant: `${props.operation}-update`,
          invocationCount: demandCount,
          ...(greedyGate ? greedyGate.getGate(1) : {}),
          bindings: [
            ...costBindings,
            {name: 'bestCosts', view: bestCosts, type: 'f32', access: 'read_write'},
            {name: 'chosen', view: chosenFacility, type: 'u32', access: 'read'}
          ],
          declarations: sparse ? `${declarations}\n${SPARSE_ROW_COST_WGSL}` : declarations,
          body: `let facility = chosen[chosenOffset];
  if (facility != NONE) {
    let candidate = ${
      sparse
        ? 'getSparseRowCost(index, facility)'
        : 'costs[costsOffset + index * FACILITY_COUNT + facility]'
    };
    bestCosts[bestCostsOffset + index] = min(bestCosts[bestCostsOffset + index], candidate);
  }`
        })
      );
    }

    if (swaps > 0) {
      nodes.push(
        ...this.getSwapNodes(graph, {
          declarations: swapDeclarations,
          swapState,
          scoreScale: chosenFacility,
          weightBinding,
          coverageBinding,
          demandWeightExpression,
          transpose: sparseSetup?.transpose
        })
      );
    }

    const assignedCostsWGSL = output.assignedCosts
      ? 'assignedCosts[assignedCostsOffset + demand] = select(bitcast<f32>(0x7f800000u | (demand & 0u)), assignedCost, assigned != NONE);'
      : '';
    const objectiveWGSL = `let threshold = ${needsCoverage ? 'coverageDistance[coverageDistanceOffset]' : '0.0'};
  let isUncovered = assigned == NONE || (MODE != 0u && assignedCost > threshold);
  uncovered[uncoveredOffset + demand] = select(0u, 1u, isUncovered);
  let weight = ${demandWeightExpression};
  // p-median sums the reachable rows only; unreachable rows are counted in uncoveredDemandCount.
  demandObjectives[demandObjectivesOffset + demand] = select(
    select(0.0, weight, !isUncovered),
    select(weight * assignedCost, 0.0, isUncovered),
    MODE == 0u
  );`;
    const assignedCostsBinding: WGSLKernelBinding[] = output.assignedCosts
      ? [{name: 'assignedCosts', view: output.assignedCosts, type: 'f32', access: 'read_write'}]
      : [];
    const objectiveBindings: WGSLKernelBinding[] = [
      {name: 'demandObjectives', view: demandObjectives, type: 'f32', access: 'read_write'},
      {name: 'uncovered', view: uncovered, type: 'u32', access: 'read_write'},
      ...weightBinding,
      ...coverageBinding,
      ...assignedCostsBinding
    ];
    if (sparse) {
      // Sparse rows list their facilities in ascending order, so the first strictly cheaper
      // selected entry wins ties like the dense lowest-facility rule. Two kernels keep each
      // within the eight-binding limit.
      const assignedCostScratch = createTransientView(
        graph,
        `${id}-assigned-cost-scratch`,
        'float32',
        demandCount
      );
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-assign`,
          operation: OPERATION,
          variant: `${props.operation}-assign-sparse`,
          invocationCount: demandCount,
          bindings: [
            ...costBindings,
            {name: 'selected', view: output.selectedFacilities, type: 'u32', access: 'read'},
            {name: 'assignments', view: output.assignments, type: 'u32', access: 'read_write'},
            {
              name: 'assignedCostScratch',
              view: assignedCostScratch,
              type: 'f32',
              access: 'read_write'
            }
          ],
          declarations,
          body: `let demand = index;
  var assigned = NONE;
  var assignedCost = getUnreachableCost();
  let rowEnd = min(rowOffsets[rowOffsetsOffset + demand + 1u], SPARSE_ENTRY_COUNT);
  for (var entry = rowOffsets[rowOffsetsOffset + demand]; entry < rowEnd; entry++) {
    let facility = rowFacilities[rowFacilitiesOffset + entry];
    if (facility >= FACILITY_COUNT || selected[selectedOffset + facility] == 0u) { continue; }
    let cost = rowCosts[rowCostsOffset + entry];
    if (isReachable(cost) && (cost < assignedCost || (cost == assignedCost && facility < assigned))) {
      assignedCost = cost;
      assigned = facility;
    }
  }
  assignments[assignmentsOffset + demand] = assigned;
  assignedCostScratch[assignedCostScratchOffset + demand] = assignedCost;`
        }),
        createWGSLKernelNode(graph, {
          id: `${id}-objective`,
          operation: OPERATION,
          variant: `${props.operation}-objective`,
          invocationCount: demandCount,
          bindings: [
            {name: 'assignments', view: output.assignments, type: 'u32', access: 'read'},
            {name: 'assignedCostScratch', view: assignedCostScratch, type: 'f32', access: 'read'},
            ...objectiveBindings
          ],
          declarations,
          body: `let demand = index;
  let assigned = assignments[assignmentsOffset + demand];
  let assignedCost = assignedCostScratch[assignedCostScratchOffset + demand];
  ${assignedCostsWGSL}
  ${objectiveWGSL}`
        })
      );
    } else {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-assign`,
          operation: OPERATION,
          variant: `${props.operation}-assign`,
          invocationCount: demandCount,
          bindings: [
            ...costBindings,
            {name: 'selected', view: output.selectedFacilities, type: 'u32', access: 'read'},
            {name: 'assignments', view: output.assignments, type: 'u32', access: 'read_write'},
            ...objectiveBindings
          ],
          declarations,
          body: `let demand = index;
  var assigned = NONE;
  var assignedCost = getUnreachableCost();
  for (var facility = 0u; facility < FACILITY_COUNT; facility++) {
    if (selected[selectedOffset + facility] == 0u) { continue; }
    let cost = costs[costsOffset + demand * FACILITY_COUNT + facility];
    if (isReachable(cost) && (cost < assignedCost || (cost == assignedCost && facility < assigned))) {
      assignedCost = cost;
      assigned = facility;
    }
  }
  assignments[assignmentsOffset + demand] = assigned;
  ${assignedCostsWGSL}
  ${objectiveWGSL}`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        variant: `${props.operation}-finalize`,
        invocationCount: CHOOSE_WORKGROUP_SIZE,
        workgroupSize: CHOOSE_WORKGROUP_SIZE,
        guardIndex: false,
        bindings: [
          {name: 'demandObjectives', view: demandObjectives, type: 'f32', access: 'read'},
          {name: 'uncovered', view: uncovered, type: 'u32', access: 'read'},
          {name: 'objective', view: output.objective, type: 'f32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'},
          ...(swaps > 0
            ? [{name: 'swapState', view: swapState, type: 'u32' as const, access: 'read' as const}]
            : [])
        ],
        declarations: `${swapDeclarations}
const FINALIZE_WORKGROUP_SIZE: u32 = ${CHOOSE_WORKGROUP_SIZE}u;
var<workgroup> sharedObjective: array<f32, ${CHOOSE_WORKGROUP_SIZE}>;
var<workgroup> sharedUncovered: array<u32, ${CHOOSE_WORKGROUP_SIZE}>;`,
        // One workgroup sums strided partials, then tree-reduces in a fixed order. No early
        // return: every invocation must reach the barriers.
        body: `var uncoveredCount = 0u;
  var objectiveValue = 0.0;
  for (var demand = localInvocationIndex; demand < DEMAND_COUNT; demand += FINALIZE_WORKGROUP_SIZE) {
    uncoveredCount += uncovered[uncoveredOffset + demand];
    objectiveValue += demandObjectives[demandObjectivesOffset + demand];
  }
  sharedObjective[localInvocationIndex] = objectiveValue;
  sharedUncovered[localInvocationIndex] = uncoveredCount;
  workgroupBarrier();
  for (var stride = FINALIZE_WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      sharedObjective[localInvocationIndex] += sharedObjective[localInvocationIndex + stride];
      sharedUncovered[localInvocationIndex] += sharedUncovered[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u && status[statusOffset + 3u] == 0u) {
    let selectedCount = status[statusOffset];
    let totalUncovered = sharedUncovered[0];
    objective[objectiveOffset] = select(sharedObjective[0], f32(selectedCount), MODE == 2u);
    status[statusOffset + 1u] = totalUncovered;
    status[statusOffset + 2u] = select(
      0u,
      1u,
      select(selectedCount == MAXIMUM_FACILITIES, totalUncovered == 0u, MODE == 2u)
    );${
      swaps > 0
        ? `
    status[statusOffset + 4u] = swapState[swapStateOffset + 1u];
    status[statusOffset + 5u] = swapState[swapStateOffset];
    status[statusOffset + 6u] = select(
      0u,
      1u,
      swapState[swapStateOffset] == 0u && swapState[swapStateOffset + 1u] >= MAXIMUM_SWAPS
    );`
        : ''
    }
  }`
      })
    );
    if (props.costsComplete || props.costLimit) {
      // Clears the finished word unless the costs are complete and the matrix horizon reaches the
      // requested cutoff: coverageDistance, or +Infinity for p-median (NET-101). `<=` is false for
      // a NaN limit; a cleared word stays 0.
      const usesCoverage = props.operation !== 'p-median';
      const withinLimit = !props.costLimit
        ? ''
        : usesCoverage
          ? ' && coverageDistance[coverageDistanceOffset] <= costLimit[costLimitOffset]'
          : ' && bitcast<u32>(costLimit[costLimitOffset]) == 0x7f800000u';
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-costs-complete`,
          operation: OPERATION,
          variant: `${usesCoverage ? 'coverage' : 'p-median'}-${props.costLimit ? 'complete-limit' : 'complete'}`,
          invocationCount: 1,
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
            ...(props.costLimit && usesCoverage
              ? [
                  {
                    name: 'coverageDistance',
                    view: props.coverageDistance!,
                    type: 'f32' as const,
                    access: 'read' as const
                  }
                ]
              : []),
            {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
          ],
          body: `let isComplete = ${props.costsComplete ? 'costsComplete[costsCompleteOffset] != 0u' : 'true'}${withinLimit};
  status[statusOffset + 2u] = select(0u, status[statusOffset + 2u], isComplete);`
        })
      );
    }
    return nodes;
  }
}
