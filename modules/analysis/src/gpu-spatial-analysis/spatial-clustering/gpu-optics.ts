// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';
import {
  getGPUSphereRadius,
  resolveGPUSpatialContext,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  resolveGPUTemporalContext,
  type GPUTemporalContext
} from '../../analysis-core/contracts/temporal-context';
import {createOPTICSXiNodes} from './gpu-optics-xi';
import {createOPTICSGridFrontierNodes} from './optics-grid-frontier';
import {validateSpatialOutputViews} from './spatial-output-validation';
import type {
  GPUCandidateStatusPort,
  GPUInvalidStatusPort,
  GPUIterationStatusPort
} from '../../analysis-core/contracts/status';
import {GPUNeighborSearch} from '../neighbor-search/gpu-neighbor-search';
import {createNeighborSearchDerivedParameterNodes} from '../neighbor-search/neighbor-search-derived-parameters';
import {IS_FINITE_FLOAT_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  validatePointProcessPositive,
  validatePointProcessView
} from '../point-process/point-process-utils';

/** Preserve the contributor identity and exact port through the shared structured validator. */
function validateOPTICSPort(
  id: string,
  path: string,
  view: GraphDataView,
  format: 'uint32' | 'uint32x2' | 'float32' | 'float32x2',
  expectedRows: number
): void {
  validatePointProcessView(id, path, view, [format], expectedRows);
}

/** Time buckets of the core-distance search; only its speed depends on this. */
const OPTICS_TIME_BUCKET_COUNT = 8;

/** Native distance with a normal squared-magnitude frame, shared by all OPTICS routes. @internal */
export const OPTICS_PLANAR_DISTANCE_WGSL = `fn getPointDistance(first: vec2<f32>, second: vec2<f32>) -> f32 {
  let delta = first - second;
  let magnitude = max(abs(delta.x), abs(delta.y));
  if (magnitude == 0.0 || (bitcast<u32>(magnitude) & 0x7fffffffu) >= 0x7f800000u) { return magnitude; }
  let exponent = i32(max((bitcast<u32>(magnitude) >> 23u) & 255u, 1u)) - 127;
  let normalized = vec2<f32>(ldexp(delta.x, -exponent), ldexp(delta.y, -exponent));
  return ldexp(length(normalized), exponent);
}`;

/**
 * Largest row count of the exhaustive reachability step (`neighborIndex: 'brute-force'`).
 * Every visit scans every row.
 */
export const GPU_OPTICS_BRUTE_FORCE_MAXIMUM_ROWS = 4096;

/**
 * Largest row count of the grid reachability step (`neighborIndex: 'grid'`). At this size
 * the packed frontier points occupy 16 MiB. Neighbor lists and core-distance scratch have
 * separate device binding checks; frontierBatchSize bounds the batched command graph.
 */
export const GPU_OPTICS_GRID_MAXIMUM_ROWS = 2 ** 20;

/** Serial xi active-area and predecessor scans retain their reference-tier row bound. */
export const GPU_OPTICS_XI_MAXIMUM_ROWS = 65536;

/**
 * Command nodes besides the frontier visits/batches and xi batches: the two GPUNeighborSearch
 * graphs (which do not expose their counts; each is a few tens of nodes), parameter setup, core,
 * initialization and extraction. An allowance; a spec checks the real count stays below it.
 */
export const GPU_OPTICS_FIXED_COMMAND_NODE_ALLOWANCE = 256;

/** Xi batches visit at most this many ordering ranks before yielding to the next dispatch. */
export const GPU_OPTICS_XI_BATCH_SIZE = 64;

/** One gated xi extraction dispatch and one ungated scalar dispatch to update its gate. */
export const GPU_OPTICS_XI_NODES_PER_BATCH = 2;

/** Each indexed batch has a gated frontier dispatch and an ungated scalar gate update. */
export const GPU_OPTICS_FRONTIER_NODES_PER_BATCH = 2;

/** Amortizes dispatches while keeping a full million-row indexed graph below the shared budget. */
export const GPU_OPTICS_DEFAULT_FRONTIER_BATCH_SIZE = 256;

/**
 * Two published uint32 reductions (at most two levels plus finalize each), tie marking,
 * commit and relaxation: nine nodes per parallel visit through 65,536 rows. Full-size
 * graphs necessarily exceed the shared budget. Node specs measure the actual count at
 * both reduction boundaries; scratch and reduction templates are reused between visits.
 */
export const GPU_OPTICS_PARALLEL_NODES_PER_VISIT = 9;
// Preserve the measured legacy parallel-graph budget when increasing the grid row capacity.
// Larger reductions add levels and therefore permit fewer visits within the same budget.
export const GPU_OPTICS_MAXIMUM_COMMAND_NODE_COUNT =
  GPU_OPTICS_PARALLEL_NODES_PER_VISIT * GPU_OPTICS_XI_MAXIMUM_ROWS +
  GPU_OPTICS_XI_NODES_PER_BATCH * Math.ceil(GPU_OPTICS_XI_MAXIMUM_ROWS / GPU_OPTICS_XI_BATCH_SIZE) +
  GPU_OPTICS_FIXED_COMMAND_NODE_ALLOWANCE;

/** Published integer reduction levels each reduce at most 256 inputs to one value. */
const OPTICS_REDUCTION_WORKGROUP_SIZE = 256;

/** Includes both reduction finalizers, tie marking, commit and relaxation. */
const OPTICS_PARALLEL_FIXED_NODES_PER_VISIT = 5;

/** Exact visit-node count for the published reduction hierarchy, including its finalizers. */
function getOPTICSParallelNodesPerVisit(rows: number): number {
  let remaining = rows;
  let levels = 0;
  do {
    remaining = Math.ceil(remaining / OPTICS_REDUCTION_WORKGROUP_SIZE);
    levels++;
  } while (remaining > 1);
  return 2 * levels + OPTICS_PARALLEL_FIXED_NODES_PER_VISIT;
}

/** Self-inclusive k-nearest search stores at most 255 other samples per query. */
const OPTICS_MAXIMUM_MINIMUM_SAMPLES = 256;
/** Batches bound sequential work per dispatch. */
const OPTICS_MAXIMUM_FRONTIER_BATCH_SIZE = 256;
/** Radius-list offsets use signed 32-bit-compatible capacities. */
const OPTICS_MAXIMUM_NEIGHBOR_CAPACITY = 2 ** 31 - 1;
/** Leave headroom for a squared float32 radius and the grid frontier's widening margin. */
const OPTICS_MAXIMUM_DIRECT_SEARCH_EXPONENT = 63;
const OPTICS_MAXIMUM_DIRECT_SEARCH_RADIUS = 2 ** OPTICS_MAXIMUM_DIRECT_SEARCH_EXPONENT;

/** Named iteration ports shared by the contributor and its graph operations. @internal */
export const OPTICS_ITERATION_STATUS_FIELDS = [
  'iterationCount',
  'converged',
  'iterationLimitReached'
] as const;

/** Named termination reasons. */
export const GPU_OPTICS_REASON = {
  complete: 0,
  candidateOverflow: 1,
  iterationLimitReached: 2
} as const;

/** @internal Optional status outputs shared with the operation form. */
export const OPTICS_NAMED_STATUS_FIELDS = [
  ...OPTICS_ITERATION_STATUS_FIELDS,
  'reason',
  'invalidCount'
] as const;

/** Default radius-neighbor slots per row of the grid reachability step. */
export const GPU_OPTICS_DEFAULT_NEIGHBORS_PER_ROW = 64;

/**
 * How {@link GPUOPTICS} finds the rows within `maximumDistance` of each visited row:
 * `'brute-force'` tests every row, `'grid'` reads a `GPUNeighborSearch` radius list and keeps a
 * seed list, and `'auto'` uses the scan up to {@link GPU_OPTICS_BRUTE_FORCE_MAXIMUM_ROWS} rows
 * and the grid above it.
 */
export type GPUOPTICSNeighborIndex = 'auto' | 'brute-force' | 'grid';

/** Caller-owned exact OPTICS ordering and DBSCAN/xi extraction. */
export type GPUOPTICSOutput = {
  /** Input row IDs in visit order; unused slots are 0xffffffff. */
  ordering: GraphDataView<'uint32'>;
  /** Self-inclusive minimumSamples-th distance; infinity for non-core rows. */
  coreDistances: GraphDataView<'float32'>;
  /** Reachability indexed by input row; infinity for roots and unvisited rows. */
  reachability: GraphDataView<'float32'>;
  /** Predecessor input row, or 0xffffffff. */
  predecessors: GraphDataView<'uint32'>;
  /** Selected extraction indexed by input row; noise is 0xffffffff. */
  labels: GraphDataView<'uint32'>;
  /** Optional xi hierarchy, inclusive positions in ordering; capacity must equal input rows. */
  clusterIntervals?: GraphDataView<'uint32x2'>;
  /** Optional [number of xi intervals, overflow]. */
  clusterSummary?: GraphDataView<'uint32'>;
  /** Optional named one-row status ports. */
  ports?: {
    /**
     * candidateOverflow reports a truncated radius list. iterationCount is the visited
     * count; converged means all valid rows were visited. iterationLimitReached is set
     * only for an exhausted visit budget. invalidCount counts excluded position or time rows.
     */
    status: GPUCandidateStatusPort &
      Pick<GPUInvalidStatusPort, 'invalidCount'> &
      GPUIterationStatusPort & {
        /** GPU_OPTICS_REASON: completion, truncated candidates, or exhausted visit budget. */
        reason?: GraphDataView<'uint32'>;
      };
  };
};

/** The views a {@link GPUOPTICSOutput} writes, with the named status ports flattened. */
function getOPTICSOutputViews(output: GPUOPTICSOutput): GraphDataView[] {
  const {ports, ...views} = output;
  return [
    ...(Object.values(views) as (GraphDataView | undefined)[]),
    ...Object.values(ports?.status ?? {})
  ].filter((view): view is GraphDataView => view !== undefined);
}

/** Bounded exact planar or spherical OPTICS; all data-dependent work executes on the GPU. */
export type GPUOPTICSProps = {
  id?: string;
  /** Defaults to planar native distances. Geographic positions are lon/lat degrees with
   * great-circle distances in meters (or declared kilometers); antimeridian pairs are supported.
   * Absolute Float32 degrees retain their original quantization. */
  spatialContext?: GPUSpatialContext;
  positions: GraphDataView<'float32x2'>;
  /**
   * Optional relative times, one `float32` per row, which turn the ordering into ST-OPTICS: a pair
   * of rows is related only when `|t_a - t_b| <= maximumTimeDistance`, besides the spatial test
   * `distance <= maximumDistance`. Core distances, reachability and extraction use only related
   * pairs, so a pair outside the time window behaves as if it were farther than
   * `maximumDistance`. This is a hard filter on the neighbor graph, which equals the precomputed
   * matrix of `st-clustering` (distance `2 * eps1` outside the window) whenever
   * `maximumDistance < 2 * eps1`. A row with a non-finite time is invalid like a non-finite
   * position. Typically the output of `GPUTimeWordsRelative`. Requires `temporalContext` and
   * `maximumTimeDistance`. When absent, nothing changes. Compile-time presence; per-frame contents.
   */
  times?: GraphDataView<'float32'>;
  /**
   * Describes `times`: required with `times`, rejected without it. Only `'float32-relative'` times
   * are read; any unit, epoch (a bigint origin), time zone and axis. `unit` is the unit of
   * `maximumTimeDistance`. Nothing is converted.
   */
  temporalContext?: GPUTemporalContext;
  /** Largest time difference of related rows, in the `temporalContext` unit. Positive, finite. Compile-time. */
  maximumTimeDistance?: number;
  /** Includes the focus point, as in sklearn OPTICS. */
  minimumSamples: number;
  maximumDistance: number;
  /** DBSCAN extraction radius, positive and <= maximumDistance. Required only for DBSCAN extraction. */
  extractionDistance?: number;
  /** Default dbscan. Xi retains nested hierarchy and labels only disjoint innermost clusters. */
  extractionMethod?: 'dbscan' | 'xi';
  xi?: number;
  minimumClusterSize?: number;
  predecessorCorrection?: boolean;
  /**
   * Sequential visits per dispatch, default 256. Ignored by parallel ordering.
   * Every remaining sequential batch is declared; indexed batches are GPU-gated after completion.
   */
  frontierBatchSize?: number;
  /** Maximum visited rows, default input length. Incomplete extraction labels stay noise. */
  maximumIterations?: number;
  /**
   * How the reachability step finds rows within `maximumDistance`. Compile-time, default
   * `'auto'`; see {@link GPUOPTICSNeighborIndex}. Both paths produce the same ordering,
   * reachability and predecessors. `'brute-force'` accepts at most
   * {@link GPU_OPTICS_BRUTE_FORCE_MAXIMUM_ROWS} rows; `'grid'` at most
   * {@link GPU_OPTICS_GRID_MAXIMUM_ROWS}. Xi extraction retains its separate
   * {@link GPU_OPTICS_XI_MAXIMUM_ROWS} reference limit.
   */
  neighborIndex?: GPUOPTICSNeighborIndex;
  /**
   * Compile-time, default 'sequential' for compatibility and its smaller graph.
   * 'parallel' requires the grid index: published hierarchical reductions select the
   * exact (reachability, row) minimum, then separate dispatches commit and relax it.
   * Visits remain dependent; frontier work spans workgroups. Costs up to eleven nodes
   * per visit at the grid row limit and retains the legacy command-node budget. Use the
   * default batched sequential ordering for a complete million-row ordering.
   */
  orderingMethod?: 'sequential' | 'parallel';
  /**
   * Grid path only: radius-neighbor slots in total (pairs within `maximumDistance`, both
   * directions). Compile-time, default `rows * min(rows - 1, 64)`. Too small sets
   * `output.ports.status.candidateOverflow` and leaves the ordering incomplete.
   * The default can exceed the device's storage binding limit for large inputs; choose
   * a capacity matching the expected neighborhood density and the device's binding size.
   */
  neighborCapacity?: number;
  output: GPUOPTICSOutput;
};

/**
 * Exact bounded OPTICS with deterministic (reachability, input-row) frontier ties.
 * Core distances come from a `GPUNeighborSearch` k-nearest self join capped at `maximumDistance`
 * (and the time window, with `times`). The ordering follows OPTICS' sequential dependency in one
 * workgroup, one visit after another, so this is a reference-tier operator:
 *
 * - `neighborIndex: 'brute-force'` (the default up to 4096 rows): every visit scans every row to
 *   select the next row and to relax reachability, O(n²) work. Graph scratch is
 *   O(n * minimumSamples).
 * - `neighborIndex: 'grid'` (the default above 4096 rows, at most 1,048,576): a `GPUNeighborSearch`
 *   radius self join lists the rows within `maximumDistance` once; each visit relaxes only those
 *   rows and selects the next row from a seed list of the rows with finite reachability, falling
 *   back to the lowest unvisited row when the list is empty. Work per visit is the seed count plus
 *   the neighbor count. Scratch adds `neighborCapacity` slots and two row arrays.
 *
 * Both paths select by (reachability, row) and relax with the same distance test, so they give
 * identical outputs. `orderingMethod: 'parallel'` distributes the indexed frontier across
 * workgroups with published reductions and dispatch boundaries; 'sequential' retains the
 * original single-workgroup reference. Xi extraction accepts 65,536 rows, walks batches
 * of 64 ranks in one invocation, and remains reference-tier (nested scans are bounded
 * by the row cap). Prefer CPUOPTICSOperation for small CPU-resident collections.
 *
 * Compile-time: row count, `minimumSamples`, `maximumDistance`, `extractionDistance`, `xi`,
 * `minimumClusterSize`, `frontierBatchSize`, `maximumIterations`, `neighborIndex`,
 * `neighborCapacity`, `orderingMethod`, `maximumTimeDistance` and which optional views exist.
 * Per-frame: the contents of `positions` and `times`.
 */
export class GPUOPTICS implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUOPTICSProps;
  readonly spatialContext: GPUSpatialContext;
  /** Resolved context of `times` (ST-OPTICS), else undefined. */
  readonly temporalContext?: GPUTemporalContext;
  /** Resolved reachability path. */
  readonly neighborIndex: 'brute-force' | 'grid';
  /** Radius-neighbor slots of the grid path; 0 on the exhaustive path. */
  readonly neighborCapacity: number;
  /**
   * Upper bound: two per indexed sequential batch, one per exhaustive batch, or at most
   * eleven per parallel visit,
   * plus two per xi batch and GPU_OPTICS_FIXED_COMMAND_NODE_ALLOWANCE for other stages.
   */
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUOPTICSProps) {
    this.props = props;
    this.id = props.id ?? 'optics';
    validateRequirement(
      this.id,
      !('status' in props.output),
      'output.status was removed; use output.ports.status for iterationCount, converged and invalidCount'
    );
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext, {
      ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
      measure: 'distance'
    });
    // Validate the positions port before deriving the row count from it, so a missing port is
    // a structured diagnostic rather than a TypeError.
    const positions = props.positions as GraphDataView | undefined;
    validateOPTICSPort(this.id, `positions`, props.positions, 'float32x2', positions?.length ?? 0);
    const rows = props.positions.length;
    if (props.times) {
      validateRequirement(
        this.id,
        props.temporalContext !== undefined,
        'times need a temporalContext'
      );
      this.temporalContext = resolveGPUTemporalContext(this.id, props.temporalContext, {
        encodings: ['float32-relative']
      });
      validateOPTICSPort(this.id, `times`, props.times, 'float32', rows);
      validatePointProcessPositive(
        props.maximumTimeDistance ?? NaN,
        this.id,
        'maximumTimeDistance'
      );
    } else {
      validateRequirement(
        this.id,
        props.temporalContext === undefined && props.maximumTimeDistance === undefined,
        'temporalContext and maximumTimeDistance need times'
      );
    }
    const neighborIndex = props.neighborIndex ?? 'auto';
    validateRequirement(
      this.id,
      ['auto', 'brute-force', 'grid'].includes(neighborIndex),
      "neighborIndex must be 'auto', 'brute-force' or 'grid'"
    );
    this.neighborIndex =
      neighborIndex === 'grid' ||
      (neighborIndex === 'auto' && rows > GPU_OPTICS_BRUTE_FORCE_MAXIMUM_ROWS)
        ? 'grid'
        : 'brute-force';
    // Both frontiers share the indexed candidate and output envelope.
    validateCapacity(
      this.id,
      'positions rows',
      rows,
      1,
      this.neighborIndex === 'grid'
        ? GPU_OPTICS_GRID_MAXIMUM_ROWS
        : GPU_OPTICS_BRUTE_FORCE_MAXIMUM_ROWS
    );
    validateRequirement(
      this.id,
      ['sequential', 'parallel'].includes(props.orderingMethod ?? 'sequential'),
      "orderingMethod must be 'sequential' or 'parallel'"
    );
    validateRequirement(
      this.id,
      props.orderingMethod !== 'parallel' || this.neighborIndex === 'grid',
      "orderingMethod 'parallel' needs neighborIndex 'grid'"
    );
    this.neighborCapacity =
      this.neighborIndex === 'grid'
        ? (props.neighborCapacity ??
          Math.max(1, rows * Math.min(rows - 1, GPU_OPTICS_DEFAULT_NEIGHBORS_PER_ROW)))
        : 0;
    if (this.neighborIndex === 'grid') {
      validateCapacity(
        this.id,
        'neighborCapacity',
        this.neighborCapacity,
        1,
        OPTICS_MAXIMUM_NEIGHBOR_CAPACITY
      );
    } else {
      validateRequirement(
        this.id,
        props.neighborCapacity === undefined &&
          props.output.ports?.status.candidateOverflow === undefined,
        "neighborCapacity and output.ports.status.candidateOverflow need neighborIndex 'grid'"
      );
    }
    validateCapacity(
      this.id,
      'minimumSamples',
      props.minimumSamples,
      1,
      OPTICS_MAXIMUM_MINIMUM_SAMPLES
    );
    const iterations = props.maximumIterations ?? rows;
    validateCapacity(this.id, 'maximumIterations', iterations, 1, rows);
    validatePointProcessPositive(props.maximumDistance, this.id, 'maximumDistance');
    validateRequirement(
      this.id,
      props.extractionMethod === undefined || ['dbscan', 'xi'].includes(props.extractionMethod),
      "extractionMethod must be 'dbscan' or 'xi'"
    );
    if ((props.extractionMethod ?? 'dbscan') === 'dbscan') {
      validatePointProcessPositive(props.extractionDistance ?? NaN, this.id, 'extractionDistance');
      validateRequirement(
        this.id,
        props.extractionDistance! <= props.maximumDistance,
        'extractionDistance must be at most maximumDistance'
      );
    }
    const xi = props.xi ?? 0.05;
    validateRequirement(this.id, Number.isFinite(xi) && xi > 0 && xi < 1, 'xi must be in (0, 1)');
    const clusterSize = props.minimumClusterSize ?? props.minimumSamples;
    if (props.extractionMethod === 'xi') {
      validateCapacity(this.id, 'xi positions rows', rows, 1, GPU_OPTICS_XI_MAXIMUM_ROWS);
      validateCapacity(this.id, 'minimumClusterSize', clusterSize, 2, rows);
    }
    const batchSize = props.frontierBatchSize ?? GPU_OPTICS_DEFAULT_FRONTIER_BATCH_SIZE;
    validateCapacity(
      this.id,
      'frontierBatchSize',
      batchSize,
      1,
      OPTICS_MAXIMUM_FRONTIER_BATCH_SIZE
    );
    this.estimatedCommandNodeCount =
      (props.orderingMethod === 'parallel'
        ? getOPTICSParallelNodesPerVisit(rows) * iterations
        : Math.ceil(iterations / batchSize) *
          (this.neighborIndex === 'grid' ? GPU_OPTICS_FRONTIER_NODES_PER_BATCH : 1)) +
      (props.extractionMethod === 'xi'
        ? GPU_OPTICS_XI_NODES_PER_BATCH * Math.ceil(rows / GPU_OPTICS_XI_BATCH_SIZE)
        : 0) +
      GPU_OPTICS_FIXED_COMMAND_NODE_ALLOWANCE;
    validateCommandNodeBudget(
      this.id,
      this.estimatedCommandNodeCount,
      props.orderingMethod === 'parallel' ? GPU_OPTICS_MAXIMUM_COMMAND_NODE_COUNT : undefined
    );
    if (props.output.clusterIntervals || props.output.clusterSummary) {
      validateRequirement(
        this.id,
        props.extractionMethod === 'xi',
        "clusterIntervals and clusterSummary need extractionMethod 'xi'"
      );
    }
    if (props.output.clusterIntervals)
      validateOPTICSPort(
        this.id,
        `output.clusterIntervals`,
        props.output.clusterIntervals,
        'uint32x2',
        rows
      );
    if (props.output.clusterSummary)
      validateOPTICSPort(
        this.id,
        `output.clusterSummary`,
        props.output.clusterSummary,
        'uint32',
        2
      );
    for (const name of ['ordering', 'predecessors', 'labels'] as const) {
      validateOPTICSPort(this.id, `output.${name}`, props.output[name], 'uint32', rows);
    }
    for (const name of ['coreDistances', 'reachability'] as const) {
      validateOPTICSPort(this.id, `output.${name}`, props.output[name], 'float32', rows);
    }
    const status = props.output.ports?.status;
    const statusEntries = Object.entries(status ?? {});
    for (const [name, view] of statusEntries) {
      if (view) validateOPTICSPort(this.id, `output.ports.status.${name}`, view, 'uint32', 1);
    }
    if (status) {
      validateSpatialOutputViews(
        this.id,
        Object.fromEntries(
          statusEntries.map(([name, view]) => [`output.ports.status.${name}`, view])
        ),
        [
          {
            read: status.reason || status.iterationLimitReached ? [status.candidateOverflow] : [],
            write: OPTICS_NAMED_STATUS_FIELDS.map(name => status[name])
          }
        ]
      );
    }
    validateGraphOutputsDisjointFromInputs(
      this.id,
      getOPTICSOutputViews(props.output),
      [props.positions, props.times].filter(Boolean) as GraphDataView[]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, output, minimumSamples} = props;
    const rows = positions.length;
    const {times} = props;
    validateGraphViewsBelongToGraph(id, graph, [positions, times, ...getOPTICSOutputViews(output)]);
    for (const [name, byteLength] of [
      ['row scratch', rows * 4],
      ['neighbor offsets', (rows + 1) * 4],
      ['frontier points', this.neighborIndex === 'grid' ? rows * 16 : 0],
      ['frontier scratch', this.neighborIndex === 'grid' ? (rows * 2 + 4) * 4 : 0],
      ['xi areas', props.extractionMethod === 'xi' ? rows * 16 + 24 : 0],
      ['core neighbor scratch', rows * Math.max(0, minimumSamples - 1) * 4],
      ['neighborCapacity', this.neighborCapacity * 4]
    ] as const) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        byteLength,
        'lower the row count or capacity'
      );
    }
    // Rows are related only inside the time window; without times every pair is.
    const timeBindings = times
      ? [{name: 'times', view: times, type: 'f32' as const, access: 'read' as const}]
      : [];
    const processed = createTransientView(graph, `${id}-processed`, 'uint32', rows);
    // Retain the actual candidate loss even when its public port is omitted.
    const candidateOverflow =
      this.neighborIndex === 'grid'
        ? (output.ports?.status.candidateOverflow ??
          createTransientView(graph, `${id}-reach-neighbor-overflow`, 'uint32', 1))
        : undefined;
    const invalidRows = createTransientView(graph, `${id}-invalid-rows`, 'uint32', rows);
    // The visit count, completion flag and invalid row count also gate frontier and extraction.
    const frontierStatus = createTransientView(graph, `${id}-frontier-state`, 'uint32', 3);
    const geographic = this.spatialContext.coordinateSpace === 'longitude-latitude';
    const distanceSource = geographic
      ? `${GEODESIC_WGSL}
fn getPointDistance(first: vec2<f32>, second: vec2<f32>) -> f32 { return geodesicHaversineDistance(first, second, ${getWGSLFloatLiteral(getGPUSphereRadius(this.spatialContext))}); }`
      : OPTICS_PLANAR_DISTANCE_WGSL;
    const declarations = `${IS_FINITE_FLOAT_WGSL}
${distanceSource}
const ROWS = ${rows}u;
const MAXIMUM_DISTANCE = ${Math.fround(props.maximumDistance)}f;
fn point(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
fn validPoint(row: u32) -> bool {
  let value = point(row);
  return isFiniteFloat(value.x) && isFiniteFloat(value.y)${geographic ? ' && abs(value.x) <= 180.0 && abs(value.y) <= 90.0' : ''}${times ? ' && isFiniteFloat(times[timesOffset + row])' : ''};
}
${
  times
    ? `const MAXIMUM_TIME_DISTANCE = ${Math.fround(props.maximumTimeDistance!)}f;
fn isRelated(first: u32, second: u32) -> bool {
  return abs(times[timesOffset + first] - times[timesOffset + second]) <= MAXIMUM_TIME_DISTANCE;
}`
    : ''
}`;
    const otherSamples = minimumSamples - 1;
    // Both indexed searches use squared distances. Scale only as much as needed to keep
    // the radius square finite, then publish core distances in the original coordinate units.
    const positionScaleExponent =
      !geographic &&
      (otherSamples > 0 || this.neighborIndex === 'grid') &&
      props.maximumDistance > OPTICS_MAXIMUM_DIRECT_SEARCH_RADIUS
        ? OPTICS_MAXIMUM_DIRECT_SEARCH_EXPONENT - Math.ceil(Math.log2(props.maximumDistance))
        : 0;
    const searchRadius = props.maximumDistance * 2 ** positionScaleExponent;
    const searchPositions =
      positionScaleExponent === 0
        ? positions
        : createTransientView(graph, `${id}-search-positions`, 'float32x2', rows);
    const coreSearchNodes: GPUCommandNode<Parameters>[] = [];
    if (positionScaleExponent !== 0) {
      coreSearchNodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-normalize-search-positions`,
          operation: 'GPUOPTICS',
          invocationCount: rows,
          bindings: [
            {name: 'positions', view: positions, type: 'f32', access: 'read'},
            {name: 'normalized', view: searchPositions, type: 'f32', access: 'read_write'}
          ],
          body: `normalized[normalizedOffset + 2u * index] = ldexp(positions[positionsOffset + 2u * index], ${positionScaleExponent});
  normalized[normalizedOffset + 2u * index + 1u] = ldexp(positions[positionsOffset + 2u * index + 1u], ${positionScaleExponent});`
        })
      );
    }
    let neighborOffsets: GraphDataView<'uint32'> | undefined;
    let neighborDistances: GraphDataView<'float32'> | undefined;
    if (otherSamples > 0) {
      const search = createNeighborSearchDerivedParameterNodes<Parameters>(graph, {
        id: `${id}-core-neighbors-setup`,
        operation: 'GPUOPTICS',
        positions: searchPositions,
        radius: searchRadius,
        geographic,
        times,
        timeRadius: props.maximumTimeDistance
      });
      const slotCount = rows * otherSamples;
      neighborOffsets = createTransientView(
        graph,
        `${id}-core-neighbor-offsets`,
        'uint32',
        rows + 1
      );
      neighborDistances = createTransientView(
        graph,
        `${id}-core-neighbor-distances`,
        'float32',
        slotCount
      );
      coreSearchNodes.push(
        ...search.nodes,
        ...new GPUNeighborSearch({
          id: `${id}-core-neighbors`,
          mode: 'knn',
          neighborCount: otherSamples,
          spatialContext: this.spatialContext,
          positions: searchPositions,
          times,
          temporalContext: this.temporalContext,
          timeBucketCount: times ? OPTICS_TIME_BUCKET_COUNT : undefined,
          parameters: search.parameters,
          gridSize: search.gridSize,
          output: {
            offsets: neighborOffsets,
            neighbors: createTransientView(graph, `${id}-core-neighbor-ids`, 'uint32', slotCount),
            weights: createTransientView(
              graph,
              `${id}-core-neighbor-weights`,
              'float32',
              slotCount
            ),
            distances: neighborDistances,
            status: {
              overflow: createTransientView(graph, `${id}-core-neighbor-overflow`, 'uint32', 1)
            }
          }
        }).getCommandNodes(graph)
      );
    }
    const nodes: GPUCommandNode<Parameters>[] = [
      ...coreSearchNodes,
      createWGSLKernelNode(graph, {
        id: `${id}-core`,
        operation: 'GPUOPTICS',
        variant: 'core',
        invocationCount: rows,
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          ...timeBindings,
          ...(neighborOffsets && neighborDistances
            ? [
                {
                  name: 'neighborOffsets',
                  view: neighborOffsets,
                  type: 'u32' as const,
                  access: 'read' as const
                },
                {
                  name: 'neighborDistances',
                  view: neighborDistances,
                  type: 'f32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'core', view: output.coreDistances, type: 'f32', access: 'read_write'}
        ],
        declarations,
        // The search lists the other related rows within maximumDistance; the row itself is the
        // first sample, so minimumSamples - 1 of them reach the self-inclusive core distance.
        body: `var coreDistance = bitcast<f32>(0x7f800000u | (index & 0u));
  if (validPoint(index)) {
    ${
      neighborOffsets
        ? `let begin = neighborOffsets[neighborOffsetsOffset + index];
    let end = neighborOffsets[neighborOffsetsOffset + index + 1u];
    if (end - begin + 1u >= ${minimumSamples}u) {
      coreDistance = 0.0;
      for (var slot = begin; slot < end; slot++) {
        coreDistance = max(coreDistance, neighborDistances[neighborDistancesOffset + slot]);
      }${positionScaleExponent ? `\n      coreDistance = ldexp(coreDistance, ${-positionScaleExponent});` : ''}
    }`
        : 'coreDistance = 0.0;'
    }
  }
  core[coreOffset + index] = coreDistance;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-initialize`,
        operation: 'GPUOPTICS',
        variant: 'frontier-initialize',
        invocationCount: rows,
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          ...timeBindings,
          {name: 'ordering', view: output.ordering, type: 'u32', access: 'read_write'},
          {name: 'reachability', view: output.reachability, type: 'f32', access: 'read_write'},
          {name: 'predecessors', view: output.predecessors, type: 'u32', access: 'read_write'},
          {name: 'processed', view: processed, type: 'u32', access: 'read_write'},
          {name: 'invalidRows', view: invalidRows, type: 'u32', access: 'read_write'},
          {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'}
        ],
        declarations,
        body: `let infinity = bitcast<f32>(0x7f800000u | (index & 0u));
  let invalid = u32(!validPoint(index));
  invalidRows[invalidRowsOffset + index] = invalid;
  processed[processedOffset + index] = invalid;
  ordering[orderingOffset + index] = 0xffffffffu;
  predecessors[predecessorsOffset + index] = 0xffffffffu;
  reachability[reachabilityOffset + index] = infinity;
  labels[labelsOffset + index] = 0xffffffffu;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-initialize-status`,
        operation: 'GPUOPTICS',
        variant: 'frontier-initialize-status',
        invocationCount: 1,
        bindings: [{name: 'status', view: frontierStatus, type: 'u32', access: 'read_write'}],
        body: `status[statusOffset] = 0u;
  status[statusOffset + 1u] = 0u;`
      }),
      ...new GPUReduction({
        id: `${id}-invalid-count`,
        input: invalidRows,
        output: graph.createDataView(frontierStatus.buffer, {
          format: 'uint32',
          byteOffset: frontierStatus.byteOffset + 2 * Uint32Array.BYTES_PER_ELEMENT,
          length: 1
        }),
        operation: 'sum'
      }).getCommandNodes(graph)
    ];
    if (this.neighborIndex === 'grid') {
      nodes.push(
        ...createOPTICSGridFrontierNodes(
          graph,
          this,
          {positions: searchPositions, radius: searchRadius},
          distanceSource,
          timeBindings,
          processed,
          candidateOverflow!,
          frontierStatus
        )
      );
    }
    for (
      let batch = 0;
      this.neighborIndex === 'brute-force' &&
      batch <
        Math.ceil(
          (props.maximumIterations ?? rows) /
            (props.frontierBatchSize ?? GPU_OPTICS_DEFAULT_FRONTIER_BATCH_SIZE)
        );
      batch++
    ) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-frontier-${batch}`,
          operation: 'GPUOPTICS',
          variant: 'frontier-batch',
          invocationCount: 256,
          workgroupSize: 256,
          guardIndex: false,
          bindings: [
            {name: 'positions', view: positions, type: 'f32', access: 'read'},
            ...timeBindings,
            {name: 'core', view: output.coreDistances, type: 'f32', access: 'read'},
            {name: 'ordering', view: output.ordering, type: 'u32', access: 'read_write'},
            {name: 'reachability', view: output.reachability, type: 'f32', access: 'read_write'},
            {name: 'predecessors', view: output.predecessors, type: 'u32', access: 'read_write'},
            {name: 'processed', view: processed, type: 'u32', access: 'read_write'},
            {name: 'status', view: frontierStatus, type: 'u32', access: 'read_write'}
          ],
          declarations: `${declarations}
var<workgroup> candidateRows: array<u32, 256>;
var<workgroup> candidateDistances: array<f32, 256>;
var<workgroup> visited: u32;`,
          body: `let infinity = bitcast<f32>(0x7f800000u | (index & 0u));
  if (index == 0u) { visited = status[statusOffset]; }
  workgroupBarrier();
  for (var iteration = 0u; iteration < ${Math.min(props.frontierBatchSize ?? GPU_OPTICS_DEFAULT_FRONTIER_BATCH_SIZE, (props.maximumIterations ?? rows) - batch * (props.frontierBatchSize ?? GPU_OPTICS_DEFAULT_FRONTIER_BATCH_SIZE))}u; iteration++) {
    var selected = 0xffffffffu; var best = infinity;
    for (var row = index; row < ROWS; row += 256u) {
      let candidate = reachability[reachabilityOffset + row];
      if (processed[processedOffset + row] == 0u && (candidate < best || (candidate == best && row < selected))) {
        selected = row; best = candidate;
      }
    }
    candidateRows[index] = selected; candidateDistances[index] = best;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
      if (index < stride) {
        let otherDistance = candidateDistances[index + stride]; let otherRow = candidateRows[index + stride];
        if (otherDistance < candidateDistances[index] || (otherDistance == candidateDistances[index] && otherRow < candidateRows[index])) {
          candidateDistances[index] = otherDistance; candidateRows[index] = otherRow;
        }
      }
      workgroupBarrier();
    }
    let focus = workgroupUniformLoad(&candidateRows[0]);
    if (focus == 0xffffffffu) { break; }
    if (index == 0u) { ordering[orderingOffset + visited] = focus; visited++; processed[processedOffset + focus] = 1u; }
    storageBarrier(); workgroupBarrier();
    let coreDistance = core[coreOffset + focus];
    if (isFiniteFloat(coreDistance)) {
      for (var row = index; row < ROWS; row += 256u) {
        if (processed[processedOffset + row] != 0u) { continue; }
        ${times ? 'if (!isRelated(focus, row)) { continue; }\n        ' : ''}let pairDistance = getPointDistance(point(focus), point(row));
        if (pairDistance > MAXIMUM_DISTANCE) { continue; }
        let candidate = max(coreDistance, pairDistance);
        if (candidate < reachability[reachabilityOffset + row]) {
          reachability[reachabilityOffset + row] = candidate; predecessors[predecessorsOffset + row] = focus;
        }
      }
    }
    storageBarrier(); workgroupBarrier();
  }
  if (index == 0u) {
    status[statusOffset] = visited;
    status[statusOffset + 1u] = select(0u, 1u, visited + status[statusOffset + 2u] == ROWS);
  }`
        })
      );
    }
    const iterationStatus = OPTICS_NAMED_STATUS_FIELDS.filter(name => output.ports?.status[name]);
    const needsCandidateLoss = iterationStatus.some(
      name => name === 'reason' || name === 'iterationLimitReached'
    );
    if (iterationStatus.length) {
      const statusExpressions = {
        iterationCount: 'status[statusOffset]',
        converged: 'status[statusOffset + 1u]',
        invalidCount: 'status[statusOffset + 2u]',
        reason: `select(select(2u, 0u, status[statusOffset + 1u] != 0u), 1u, ${candidateOverflow ? 'overflow[overflowOffset] != 0u' : 'false'})`,
        iterationLimitReached: `u32(status[statusOffset + 1u] == 0u && status[statusOffset] == ${props.maximumIterations ?? rows}u${candidateOverflow ? ' && overflow[overflowOffset] == 0u' : ''})`
      };
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-iteration-status`,
          operation: 'GPUOPTICS',
          variant: 'iteration-status',
          invocationCount: 1,
          bindings: [
            {name: 'status', view: frontierStatus, type: 'u32', access: 'read'},
            ...(candidateOverflow && needsCandidateLoss
              ? [
                  {
                    name: 'overflow',
                    view: candidateOverflow,
                    type: 'u32' as const,
                    access: 'read' as const
                  }
                ]
              : []),
            ...iterationStatus.map(name => ({
              name,
              view: output.ports!.status[name]!,
              type: 'u32' as const,
              access: 'read_write' as const
            }))
          ],
          body: iterationStatus
            .map(name => `${name}[${name}Offset] = ${statusExpressions[name]};`)
            .join('\n')
        })
      );
    }
    if (props.extractionMethod === 'xi') {
      nodes.push(...createOPTICSXiNodes(graph, props, id, frontierStatus));
      return nodes;
    }
    const clusterStarts = createTransientView(graph, `${id}-cluster-starts`, 'uint32', rows);
    const clusterCounts = createTransientView(graph, `${id}-cluster-counts`, 'uint32', rows);
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-cluster-starts`,
        operation: 'GPUOPTICS',
        variant: 'dbscan-cluster-starts',
        invocationCount: rows,
        bindings: [
          {name: 'core', view: output.coreDistances, type: 'f32', access: 'read'},
          {name: 'ordering', view: output.ordering, type: 'u32', access: 'read'},
          {name: 'reachability', view: output.reachability, type: 'f32', access: 'read'},
          {name: 'status', view: frontierStatus, type: 'u32', access: 'read'},
          {name: 'starts', view: clusterStarts, type: 'u32', access: 'read_write'}
        ],
        body: `var startsCluster = false;
  if (status[statusOffset + 1u] != 0u && index < status[statusOffset]) {
    let row = ordering[orderingOffset + index];
    startsCluster = reachability[reachabilityOffset + row] > ${Math.fround(props.extractionDistance!)}f && core[coreOffset + row] <= ${Math.fround(props.extractionDistance!)}f;
  }
  starts[startsOffset + index] = u32(startsCluster);`
      }),
      ...new GPUScan({
        id: `${id}-cluster-scan`,
        input: clusterStarts,
        output: clusterCounts,
        mode: 'inclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-extract`,
        operation: 'GPUOPTICS',
        variant: 'dbscan-extraction',
        invocationCount: rows,
        bindings: [
          {name: 'starts', view: clusterStarts, type: 'u32', access: 'read'},
          {name: 'counts', view: clusterCounts, type: 'u32', access: 'read'},
          {name: 'ordering', view: output.ordering, type: 'u32', access: 'read'},
          {name: 'reachability', view: output.reachability, type: 'f32', access: 'read'},
          {name: 'status', view: frontierStatus, type: 'u32', access: 'read'},
          {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'}
        ],
        body: `if (status[statusOffset + 1u] == 0u || index >= status[statusOffset]) { return; }
  let row = ordering[orderingOffset + index];
  if (starts[startsOffset + index] != 0u || reachability[reachabilityOffset + row] <= ${Math.fround(props.extractionDistance!)}f) {
    labels[labelsOffset + row] = counts[countsOffset + index] - 1u;
  }`
      })
    );
    return nodes;
  }
}
