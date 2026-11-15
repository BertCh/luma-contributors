// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
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
import {createOPTICSParallelFrontierNodes} from './gpu-optics-parallel-frontier';
import type {
  GPUCandidateStatusPort,
  GPUIterationStatusPort
} from '../../analysis-core/contracts/status';
import {GPUNeighborSearch} from '../neighbor-search/gpu-neighbor-search';
import {createNeighborSearchDerivedParameterNodes} from '../neighbor-search/neighbor-search-derived-parameters';
import {IS_FINITE_FLOAT_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  validatePointProcessPositive,
  validatePointProcessView
} from '../point-process/point-process-utils';

/** Time buckets of the core-distance search; only its speed depends on this. */
const OPTICS_TIME_BUCKET_COUNT = 8;

/**
 * Largest row count of the exhaustive reachability step (`neighborIndex: 'brute-force'`).
 * Every visit scans every row.
 */
export const GPU_OPTICS_BRUTE_FORCE_MAXIMUM_ROWS = 4096;

/**
 * Largest row count of the grid reachability step (`neighborIndex: 'grid'`), including xi.
 * Bounds row scratch, serial xi scans and the unrolled parallel graph. Parallel selection
 * uses at most 256 workgroups; the sequential reference scans at most rows / 256 seeds per lane.
 */
export const GPU_OPTICS_GRID_MAXIMUM_ROWS = 65536;

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

/**
 * Two published uint32 reductions (at most two levels plus finalize each), tie marking,
 * commit and relaxation: nine nodes per parallel visit through 65,536 rows. Full-size
 * graphs necessarily exceed the shared budget. Node specs measure the actual count at
 * both reduction boundaries; scratch and reduction templates are reused between visits.
 */
export const GPU_OPTICS_PARALLEL_NODES_PER_VISIT = 9;
export const GPU_OPTICS_MAXIMUM_COMMAND_NODE_COUNT =
  GPU_OPTICS_PARALLEL_NODES_PER_VISIT * GPU_OPTICS_GRID_MAXIMUM_ROWS +
  GPU_OPTICS_XI_NODES_PER_BATCH *
    Math.ceil(GPU_OPTICS_GRID_MAXIMUM_ROWS / GPU_OPTICS_XI_BATCH_SIZE) +
  GPU_OPTICS_FIXED_COMMAND_NODE_ALLOWANCE;

/** Self-inclusive k-nearest search stores at most 255 other samples per query. */
const OPTICS_MAXIMUM_MINIMUM_SAMPLES = 256;
/** Batches bound sequential work per dispatch. */
const OPTICS_MAXIMUM_FRONTIER_BATCH_SIZE = 256;
/** Radius-list offsets use signed 32-bit-compatible capacities. */
const OPTICS_MAXIMUM_NEIGHBOR_CAPACITY = 2 ** 31 - 1;

/** Named iteration ports shared by the contributor and its graph operations. @internal */
export const OPTICS_ITERATION_STATUS_FIELDS = [
  'iterationCount',
  'converged',
  'iterationLimitReached'
] as const;

/** Default radius-neighbor slots per row of the grid reachability step. */
export const GPU_OPTICS_DEFAULT_NEIGHBORS_PER_ROW = 64;

/**
 * Relative widening of the grid search radius and time window. The search measures squared
 * distances (or its own haversine); the frontier re-tests every candidate with the exhaustive
 * step's distance, so a slightly wider candidate list keeps the relaxed set identical.
 */
const OPTICS_CANDIDATE_RADIUS_MARGIN = 1e-4;

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
  /** [visitedCount, complete, invalidRowCount]. Invalid positions are excluded. */
  status: GraphDataView<'uint32'>;
  /** Optional xi hierarchy, inclusive positions in ordering; capacity must equal input rows. */
  clusterIntervals?: GraphDataView<'uint32x2'>;
  /** Optional [number of xi intervals, overflow]. */
  clusterSummary?: GraphDataView<'uint32'>;
  /**
   * Optional named one-row status ports. `status` stays the packed `[visitedCount, complete,
   * invalidRowCount]` view, so the named ports sit under `ports.status`.
   */
  ports?: {
    /**
     * candidateOverflow reports a truncated radius list. iterationCount is the visited
     * count; converged means all valid rows were visited. iterationLimitReached is set
     * only for an exhausted visit budget, never for candidate overflow.
     */
    status: GPUCandidateStatusPort & GPUIterationStatusPort;
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
   * Sequential visits per dispatch, default 64. Ignored by parallel ordering.
   * Every remaining sequential batch is declared.
   */
  frontierBatchSize?: number;
  /** Maximum visited rows, default input length. Incomplete extraction labels stay noise. */
  maximumIterations?: number;
  /**
   * How the reachability step finds rows within `maximumDistance`. Compile-time, default
   * `'auto'`; see {@link GPUOPTICSNeighborIndex}. Both paths produce the same ordering,
   * reachability and predecessors. `'brute-force'` accepts at most
   * {@link GPU_OPTICS_BRUTE_FORCE_MAXIMUM_ROWS} rows; `'grid'` at most
   * {@link GPU_OPTICS_GRID_MAXIMUM_ROWS}, including xi extraction.
   */
  neighborIndex?: GPUOPTICSNeighborIndex;
  /**
   * Compile-time, default 'sequential' for compatibility and its smaller graph.
   * 'parallel' requires the grid index: published hierarchical reductions select the
   * exact (reachability, row) minimum, then separate dispatches commit and relax it.
   * Visits remain dependent; frontier work spans workgroups. Costs up to nine nodes
   * per visit, so graph construction/encoding may outweigh GPU parallelism.
   */
  orderingMethod?: 'sequential' | 'parallel';
  /**
   * Grid path only: radius-neighbor slots in total (pairs within `maximumDistance`, both
   * directions). Compile-time, default `rows * min(rows - 1, 64)`. Too small sets
   * `output.ports.status.candidateOverflow` and leaves the ordering incomplete.
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
 * - `neighborIndex: 'grid'` (the default above 4096 rows, at most 65,536): a `GPUNeighborSearch`
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
   * Upper bound: one per sequential frontier batch, or at most nine per parallel visit,
   * plus two per xi batch and GPU_OPTICS_FIXED_COMMAND_NODE_ALLOWANCE for other stages.
   */
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUOPTICSProps) {
    this.props = props;
    this.id = props.id ?? 'optics';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext, {
      ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
      measure: 'distance'
    });
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
      validatePointProcessView(props.times, 'float32', rows, `${this.id} times`);
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
      validateCapacity(this.id, 'minimumClusterSize', clusterSize, 2, rows);
    }
    const batchSize = props.frontierBatchSize ?? 64;
    validateCapacity(
      this.id,
      'frontierBatchSize',
      batchSize,
      1,
      OPTICS_MAXIMUM_FRONTIER_BATCH_SIZE
    );
    this.estimatedCommandNodeCount =
      (props.orderingMethod === 'parallel'
        ? GPU_OPTICS_PARALLEL_NODES_PER_VISIT * iterations
        : Math.ceil(iterations / batchSize)) +
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
      validatePointProcessView(
        props.output.clusterIntervals,
        'uint32x2',
        rows,
        `${this.id} output.clusterIntervals`
      );
    if (props.output.clusterSummary)
      validatePointProcessView(
        props.output.clusterSummary,
        'uint32',
        2,
        `${this.id} output.clusterSummary`
      );
    validatePointProcessView(props.positions, 'float32x2', rows, `${this.id} positions`);
    for (const name of ['ordering', 'predecessors', 'labels'] as const) {
      validatePointProcessView(props.output[name], 'uint32', rows, `${this.id} output.${name}`);
    }
    for (const name of ['coreDistances', 'reachability'] as const) {
      validatePointProcessView(props.output[name], 'float32', rows, `${this.id} output.${name}`);
    }
    validatePointProcessView(props.output.status, 'uint32', 3, `${this.id} output.status`);
    for (const [name, view] of Object.entries(props.output.ports?.status ?? {})) {
      if (view)
        validatePointProcessView(view, 'uint32', 1, `${this.id} output.ports.status.${name}`);
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
      ['row scratch', rows * 16 + 24],
      ['core neighbor scratch', rows * Math.max(1, minimumSamples - 1) * 4],
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
    const geographic = this.spatialContext.coordinateSpace === 'longitude-latitude';
    const distanceSource = geographic
      ? `${GEODESIC_WGSL}
fn getPointDistance(first: vec2<f32>, second: vec2<f32>) -> f32 { return geodesicHaversineDistance(first, second, ${getWGSLFloatLiteral(getGPUSphereRadius(this.spatialContext))}); }`
      : 'fn getPointDistance(first: vec2<f32>, second: vec2<f32>) -> f32 { return distance(first, second); }';
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
    const coreSearchNodes: GPUCommandNode<Parameters>[] = [];
    let neighborOffsets: GraphDataView<'uint32'> | undefined;
    let neighborDistances: GraphDataView<'float32'> | undefined;
    if (otherSamples > 0) {
      const search = createNeighborSearchDerivedParameterNodes<Parameters>(graph, {
        id: `${id}-core-neighbors-setup`,
        operation: 'GPUOPTICS',
        positions,
        radius: props.maximumDistance,
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
          k: otherSamples,
          spatialContext: this.spatialContext,
          positions,
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
      }
    }`
        : 'coreDistance = 0.0;'
    }
  }
  core[coreOffset + index] = coreDistance;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-initialize`,
        operation: 'GPUOPTICS',
        variant: 'frontier',
        invocationCount: 256,
        workgroupSize: 256,
        guardIndex: false,
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          ...timeBindings,
          {name: 'ordering', view: output.ordering, type: 'u32', access: 'read_write'},
          {name: 'reachability', view: output.reachability, type: 'f32', access: 'read_write'},
          {name: 'predecessors', view: output.predecessors, type: 'u32', access: 'read_write'},
          {name: 'processed', view: processed, type: 'u32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'},
          {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'}
        ],
        declarations: `${declarations}
var<workgroup> invalidCounts: array<u32, 256>;`,
        body: `let infinity = bitcast<f32>(0x7f800000u | (index & 0u));
  var invalidCount = 0u;
  for (var row = index; row < ROWS; row += 256u) {
    let invalid = !validPoint(row);
    invalidCount += u32(invalid);
    processed[processedOffset + row] = u32(invalid);
    ordering[orderingOffset + row] = 0xffffffffu;
    predecessors[predecessorsOffset + row] = 0xffffffffu;
    reachability[reachabilityOffset + row] = infinity;
    labels[labelsOffset + row] = 0xffffffffu;
  }
  invalidCounts[index] = invalidCount;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (index < stride) { invalidCounts[index] += invalidCounts[index + stride]; }
    workgroupBarrier();
  }
  if (index == 0u) {
    status[statusOffset] = 0u;
    status[statusOffset + 1u] = 0u;
    status[statusOffset + 2u] = invalidCounts[0];
  }`
      })
    ];
    if (this.neighborIndex === 'grid') {
      nodes.push(...this.getGridFrontierNodes(graph, distanceSource, timeBindings, processed));
    }
    for (
      let batch = 0;
      this.neighborIndex === 'brute-force' &&
      batch < Math.ceil((props.maximumIterations ?? rows) / (props.frontierBatchSize ?? 64));
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
            {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
          ],
          declarations: `${declarations}
var<workgroup> candidateRows: array<u32, 256>;
var<workgroup> candidateDistances: array<f32, 256>;
var<workgroup> visited: u32;`,
          body: `let infinity = bitcast<f32>(0x7f800000u | (index & 0u));
  if (index == 0u) { visited = status[statusOffset]; }
  workgroupBarrier();
  for (var iteration = 0u; iteration < ${Math.min(props.frontierBatchSize ?? 64, (props.maximumIterations ?? rows) - batch * (props.frontierBatchSize ?? 64))}u; iteration++) {
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
    const iterationStatus = OPTICS_ITERATION_STATUS_FIELDS.filter(
      name => output.ports?.status[name]
    );
    if (iterationStatus.length) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-iteration-status`,
          operation: 'GPUOPTICS',
          variant: 'iteration-status',
          invocationCount: 1,
          bindings: [
            {name: 'status', view: output.status, type: 'u32', access: 'read'},
            ...(output.ports?.status.candidateOverflow
              ? [
                  {
                    name: 'overflow',
                    view: output.ports.status.candidateOverflow,
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
            .map(
              name =>
                `${name}[${name}Offset] = ${
                  name === 'iterationCount'
                    ? 'status[statusOffset]'
                    : name === 'converged'
                      ? 'status[statusOffset + 1u]'
                      : `u32(status[statusOffset + 1u] == 0u && status[statusOffset] == ${props.maximumIterations ?? rows}u${output.ports?.status.candidateOverflow ? ' && overflow[overflowOffset] == 0u' : ''})`
                };`
            )
            .join('\n')
        })
      );
    }
    if (props.extractionMethod === 'xi') {
      nodes.push(...createOPTICSXiNodes(graph, props, id));
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
          {name: 'status', view: output.status, type: 'u32', access: 'read'},
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
          {name: 'status', view: output.status, type: 'u32', access: 'read'},
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

  /**
   * Grid reachability: a widened radius self join lists candidate rows once; every visit relaxes
   * its candidates after the exhaustive step's exact distance and time tests, and selects the next
   * row by (reachability, row) from the seed list (rows with finite reachability that are not yet
   * visited) or, when that list is empty, as the lowest unvisited row past a cursor. Every
   * unvisited row outside the seed list has infinite reachability, so the selection equals the
   * exhaustive argmin.
   *
   * Seven storage bindings per frontier batch: a packed `[x, y, time, coreDistance]` row copy, the
   * CSR offsets and IDs, the three outputs, and one scratch view holding the seed list, the row
   * states (unvisited, visited, or seed slot) and the frontier words.
   */
  private getGridFrontierNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    distanceSource: string,
    timeBindings: {name: string; view: GraphDataView; type: 'f32'; access: 'read'}[],
    processed: GraphDataView<'uint32'>
  ): GPUCommandNode<Parameters>[] {
    const {id, props, neighborCapacity} = this;
    const {positions, output, times} = props;
    const rows = positions.length;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'neighborCapacity',
      neighborCapacity * 4,
      'lower neighborCapacity or maximumDistance'
    );
    const geographic = this.spatialContext.coordinateSpace === 'longitude-latitude';
    const search = createNeighborSearchDerivedParameterNodes<Parameters>(graph, {
      id: `${id}-reach-neighbors-setup`,
      operation: 'GPUOPTICS',
      positions,
      radius: props.maximumDistance * (1 + OPTICS_CANDIDATE_RADIUS_MARGIN),
      geographic,
      times,
      timeRadius:
        props.maximumTimeDistance === undefined
          ? undefined
          : props.maximumTimeDistance * (1 + OPTICS_CANDIDATE_RADIUS_MARGIN)
    });
    const offsets = createTransientView(graph, `${id}-reach-neighbor-offsets`, 'uint32', rows + 1);
    const neighbors = createTransientView(
      graph,
      `${id}-reach-neighbor-ids`,
      'uint32',
      neighborCapacity
    );
    const overflow =
      output.ports?.status.candidateOverflow ??
      createTransientView(graph, `${id}-reach-neighbor-overflow`, 'uint32', 1);
    const points = createTransientView(graph, `${id}-frontier-points`, 'float32', rows * 4);
    // [seed rows (rows) | row states (rows) | seedCount, cursor, visited, overflow].
    const scratch = createTransientView(graph, `${id}-frontier-scratch`, 'uint32', rows * 2 + 4);
    const declarations = `${IS_FINITE_FLOAT_WGSL}
${distanceSource}
const ROWS = ${rows}u;
const MAXIMUM_DISTANCE = ${Math.fround(props.maximumDistance)}f;
const ROW_STATES = ${rows}u;
const SEED_COUNT = ${rows * 2}u;
const CURSOR = ${rows * 2 + 1}u;
const VISITED = ${rows * 2 + 2}u;
const OVERFLOW = ${rows * 2 + 3}u;
// A row state is UNVISITED, VISITED, or the row's slot in the seed list.
const UNVISITED = 0xffffffffu;
const VISITED_ROW = 0xfffffffeu;
fn point(row: u32) -> vec2<f32> {
  return vec2<f32>(points[pointsOffset + 4u * row], points[pointsOffset + 4u * row + 1u]);
}
${
  times
    ? `const MAXIMUM_TIME_DISTANCE = ${Math.fround(props.maximumTimeDistance!)}f;
fn isRelated(first: u32, second: u32) -> bool {
  return abs(points[pointsOffset + 4u * first + 2u] - points[pointsOffset + 4u * second + 2u]) <= MAXIMUM_TIME_DISTANCE;
}`
    : ''
}`;
    const nodes: GPUCommandNode<Parameters>[] = [
      ...search.nodes,
      ...new GPUNeighborSearch({
        id: `${id}-reach-neighbors`,
        mode: 'radius',
        spatialContext: this.spatialContext,
        positions,
        times,
        temporalContext: this.temporalContext,
        timeBucketCount: times ? OPTICS_TIME_BUCKET_COUNT : undefined,
        parameters: search.parameters,
        gridSize: search.gridSize,
        output: {
          offsets,
          neighbors,
          weights: createTransientView(
            graph,
            `${id}-reach-neighbor-weights`,
            'float32',
            neighborCapacity
          ),
          status: {overflow}
        }
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-frontier-prepare`,
        operation: 'GPUOPTICS',
        variant: 'grid-frontier-prepare',
        invocationCount: rows,
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          ...timeBindings,
          {name: 'core', view: output.coreDistances, type: 'f32', access: 'read'},
          {name: 'processed', view: processed, type: 'u32', access: 'read'},
          {name: 'overflow', view: overflow, type: 'u32', access: 'read'},
          {name: 'points', view: points, type: 'f32', access: 'read_write'},
          {name: 'scratch', view: scratch, type: 'u32', access: 'read_write'}
        ],
        body: `points[pointsOffset + 4u * index] = positions[positionsOffset + 2u * index];
  points[pointsOffset + 4u * index + 1u] = positions[positionsOffset + 2u * index + 1u];
  points[pointsOffset + 4u * index + 2u] = ${times ? 'times[timesOffset + index]' : '0.0'};
  points[pointsOffset + 4u * index + 3u] = core[coreOffset + index];
  // Invalid rows were marked processed by the initialize pass.
  scratch[scratchOffset + ${rows}u + index] = select(0xffffffffu, 0xfffffffeu, processed[processedOffset + index] != 0u);
  if (index == 0u) {
    scratch[scratchOffset + ${rows * 2}u] = 0u;
    scratch[scratchOffset + ${rows * 2 + 1}u] = 0u;
    scratch[scratchOffset + ${rows * 2 + 2}u] = 0u;
    scratch[scratchOffset + ${rows * 2 + 3}u] = overflow[overflowOffset];
  }`
      })
    ];
    if (props.orderingMethod === 'parallel') {
      nodes.push(
        ...createOPTICSParallelFrontierNodes(graph, props, {
          distanceSource,
          points,
          offsets,
          neighbors,
          overflow,
          processed
        })
      );
      return nodes;
    }
    const batchSize = props.frontierBatchSize ?? 64;
    const iterations = props.maximumIterations ?? rows;
    const bindings = [
      {name: 'points', view: points, type: 'f32' as const, access: 'read' as const},
      {name: 'offsets', view: offsets, type: 'u32' as const, access: 'read' as const},
      {name: 'neighbors', view: neighbors, type: 'u32' as const, access: 'read' as const},
      {
        name: 'ordering',
        view: output.ordering,
        type: 'u32' as const,
        access: 'read_write' as const
      },
      {
        name: 'reachability',
        view: output.reachability,
        type: 'f32' as const,
        access: 'read_write' as const
      },
      {
        name: 'predecessors',
        view: output.predecessors,
        type: 'u32' as const,
        access: 'read_write' as const
      },
      {name: 'scratch', view: scratch, type: 'u32' as const, access: 'read_write' as const}
    ];
    for (let batch = 0; batch < Math.ceil(iterations / batchSize); batch++) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-frontier-${batch}`,
          operation: 'GPUOPTICS',
          variant: 'grid-frontier-batch',
          invocationCount: 256,
          workgroupSize: 256,
          guardIndex: false,
          bindings,
          declarations: `${declarations}
var<workgroup> candidateRows: array<u32, 256>;
var<workgroup> candidateDistances: array<f32, 256>;
var<workgroup> visited: u32;
var<workgroup> seedCount: u32;
var<workgroup> cursor: u32;
var<workgroup> focusRow: u32;
var<workgroup> appendCount: atomic<u32>;
var<workgroup> overflowed: u32;`,
          body: `let infinity = bitcast<f32>(0x7f800000u | (index & 0u));
  if (index == 0u) {
    overflowed = scratch[scratchOffset + OVERFLOW];
    visited = scratch[scratchOffset + VISITED];
    seedCount = scratch[scratchOffset + SEED_COUNT];
    cursor = scratch[scratchOffset + CURSOR];
    atomicStore(&appendCount, 0u);
  }
  // A truncated candidate list would drop relaxations: leave the ordering incomplete.
  if (workgroupUniformLoad(&overflowed) != 0u) { return; }
  for (var iteration = 0u; iteration < ${Math.min(batchSize, iterations - batch * batchSize)}u; iteration++) {
    let count = workgroupUniformLoad(&seedCount);
    if (count > 0u) {
      var selected = 0xffffffffu; var best = infinity;
      for (var slot = index; slot < count; slot += 256u) {
        let row = scratch[scratchOffset + slot];
        let candidate = reachability[reachabilityOffset + row];
        if (candidate < best || (candidate == best && row < selected)) { selected = row; best = candidate; }
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
      if (index == 0u) { focusRow = candidateRows[0]; }
    } else {
      // Every unvisited row has infinite reachability: the next root is the lowest unvisited row.
      if (index == 0u) { focusRow = 0xffffffffu; }
      loop {
        let base = workgroupUniformLoad(&cursor);
        if (base >= ROWS) { break; }
        let row = base + index;
        candidateRows[index] = select(0xffffffffu, row, row < ROWS && scratch[scratchOffset + ROW_STATES + min(row, ROWS - 1u)] == UNVISITED);
        workgroupBarrier();
        for (var stride = 128u; stride > 0u; stride /= 2u) {
          if (index < stride) { candidateRows[index] = min(candidateRows[index], candidateRows[index + stride]); }
          workgroupBarrier();
        }
        if (index == 0u) {
          if (candidateRows[0] == 0xffffffffu) { cursor = base + 256u; } else { cursor = candidateRows[0]; focusRow = candidateRows[0]; }
        }
        if (workgroupUniformLoad(&focusRow) != 0xffffffffu) { break; }
      }
    }
    let focus = workgroupUniformLoad(&focusRow);
    if (focus == 0xffffffffu) { break; }
    if (index == 0u) {
      ordering[orderingOffset + visited] = focus; visited++;
      let slot = scratch[scratchOffset + ROW_STATES + focus];
      if (slot != UNVISITED) {
        // The focus is a seed: move the last seed into its slot.
        let last = scratch[scratchOffset + seedCount - 1u];
        scratch[scratchOffset + slot] = last; scratch[scratchOffset + ROW_STATES + last] = slot;
        seedCount--;
      }
      scratch[scratchOffset + ROW_STATES + focus] = VISITED_ROW;
    }
    storageBarrier(); workgroupBarrier();
    let coreDistance = points[pointsOffset + 4u * focus + 3u];
    let seedBase = workgroupUniformLoad(&seedCount);
    if (isFiniteFloat(coreDistance)) {
      let end = offsets[offsetsOffset + focus + 1u];
      for (var slot = offsets[offsetsOffset + focus] + index; slot < end; slot += 256u) {
        let row = neighbors[neighborsOffset + slot];
        let state = scratch[scratchOffset + ROW_STATES + row];
        if (state == VISITED_ROW) { continue; }
        ${times ? 'if (!isRelated(focus, row)) { continue; }\n        ' : ''}let pairDistance = getPointDistance(point(focus), point(row));
        if (pairDistance > MAXIMUM_DISTANCE) { continue; }
        let candidate = max(coreDistance, pairDistance);
        if (candidate < reachability[reachabilityOffset + row]) {
          reachability[reachabilityOffset + row] = candidate; predecessors[predecessorsOffset + row] = focus;
          if (state == UNVISITED) {
            let seedSlot = seedBase + atomicAdd(&appendCount, 1u);
            scratch[scratchOffset + seedSlot] = row; scratch[scratchOffset + ROW_STATES + row] = seedSlot;
          }
        }
      }
    }
    storageBarrier(); workgroupBarrier();
    if (index == 0u) { seedCount += atomicExchange(&appendCount, 0u); }
    workgroupBarrier();
  }
  if (index == 0u) {
    scratch[scratchOffset + SEED_COUNT] = seedCount;
    scratch[scratchOffset + CURSOR] = cursor;
    scratch[scratchOffset + VISITED] = visited;
  }`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-frontier-status`,
        operation: 'GPUOPTICS',
        variant: 'grid-frontier-status',
        invocationCount: 1,
        bindings: [
          {name: 'scratch', view: scratch, type: 'u32', access: 'read'},
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
        ],
        body: `let visited = scratch[scratchOffset + ${rows * 2 + 2}u];
  status[statusOffset] = visited;
  status[statusOffset + 1u] = select(0u, 1u, visited + status[statusOffset + 2u] == ${rows}u);`
      })
    );
    return nodes;
  }
}
