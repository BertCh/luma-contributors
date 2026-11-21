// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateSpatialOutputViews} from './spatial-output-validation';
import {ADDITION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/addition-rounding-error-wgsl';
import {PHILOX_PURPOSES, PHILOX_WGSL} from '../../analysis-core/kernels/philox-random';
import {ROUNDED_DIFFERENCE_WGSL} from '../spatial-regression/rounded-difference-wgsl';
import {
  createTransientView,
  GPUCompaction,
  GPUGroupAggregation,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateRequirement,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateGPUStorageBindingByteLength,
  validateCommandNodeBudget
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {GPU_DBSCAN_NOISE} from './spatial-clustering-parameters';
import {
  GET_QUIET_NAN_SEEDED_WGSL,
  IS_FINITE_FLOAT_WGSL,
  IS_NAN_BITS_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import type {GPUStatusScalar} from '../../analysis-core/contracts/status';

/** Planar contexts suitable for distance measurements. */
const KMEANS_SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
} as const);

const OPERATION = 'GPUSpatialKMeans';
const NO_ROW = '0xffffffffu';
/** Workgroup size of the per-iteration center reductions. */
const REDUCTION_WORKGROUP_SIZE = 256;
/** Largest number of lane slices that split one cluster's work inside a chunk. */
const MAXIMUM_SLICES = 16;
/** Largest number of point chunks (workgroups) the partial pass launches. */
const MAXIMUM_CHUNKS = 256;
/** Fixed preparation, optional compaction, origin and final publishing nodes. */
const KMEANS_FIXED_COMMAND_NODE_ALLOWANCE = 256;
/** Restart preparation/scoring plus five nodes per Lloyd step and four per k-means++ draw. */
const KMEANS_RESTART_COMMAND_NODE_ALLOWANCE = 64;
const KMEANS_NODES_PER_ITERATION = 5;
const KMEANS_NODES_PER_SEED = 4;

/** Largest `k`. */
export const GPU_SPATIAL_KMEANS_MAXIMUM_CLUSTERS = 256;

/** Largest `maximumIterations`: every Lloyd iteration adds several dispatches to the graph. */
export const GPU_SPATIAL_KMEANS_MAXIMUM_ITERATIONS = 64;

/** Largest `restartCount`: every restart adds a full seeding and iteration chain to the graph. */
export const GPU_SPATIAL_KMEANS_MAXIMUM_RESTARTS = 32;

/**
 * Seed of one {@link GPUSpatialKMeans} restart, derived deterministically from the user `seed` and the
 * restart index. Restart 0 uses `seed` itself, so `restartCount: 1` is the single seeded run;
 * later restarts use a murmur3-finalized mix of `seed` and the index. Exported so tests (and
 * callers) can reproduce any single restart with `restartCount: 1` and this seed.
 */
export function getGPUSpatialKMeansRestartSeed(seed: number, restart: number): number {
  if (restart === 0) {
    return seed >>> 0;
  }
  let hash = (seed + Math.imul(restart, 0x9e3779b9)) >>> 0;
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x85ebca6b) >>> 0;
  hash ^= hash >>> 13;
  hash = Math.imul(hash, 0xc2b2ae35) >>> 0;
  hash ^= hash >>> 16;
  return hash >>> 0;
}

/** How {@link GPUSpatialKMeans} picks the initial centers. */
export type GPUSpatialKMeansInitialization = 'first-valid' | 'kmeans++';

/** Number of `uint32` words in the per-frame `parameters` view of {@link GPUSpatialKMeans}. */
export const GPU_SPATIAL_KMEANS_PARAMETER_LENGTH = 2;

/** Per-frame settings of {@link GPUSpatialKMeans}, packed by {@link getGPUSpatialKMeansParameterValues}. */
export type GPUSpatialKMeansSettings = {
  /**
   * Seed of the `'kmeans++'` draws, an integer in `[0, 2^32 - 1]`. Default 0. Restart `r` is seeded
   * with `getGPUSpatialKMeansRestartSeed(seed, r)`.
   */
  seed?: number;
  /**
   * Convergence tolerance: an iteration converges when the largest center shift (Euclidean, in
   * position units) is at most this value. Finite and `>= 0`; default `0`, which stops only at an
   * exact fixed point and so gives the same result as running every iteration. A positive value
   * trades exactness for fewer iterations. After convergence the remaining iterations' assign,
   * reduction and relocation kernels are skipped through a GPU-written indirect dispatch, so each
   * one costs only a one-thread convergence check.
   */
  tolerance?: number;
};

/**
 * Packs {@link GPUSpatialKMeansSettings} into the {@link GPU_SPATIAL_KMEANS_PARAMETER_LENGTH} `uint32` words of the
 * `parameters` view: `[seed, float32 bits of tolerance]`.
 *
 * @throws If `seed` is not a uint32 or `tolerance` is negative or not finite.
 */
export function getGPUSpatialKMeansParameterValues(
  settings: GPUSpatialKMeansSettings = {}
): Uint32Array {
  const seed = settings.seed ?? 0;
  const tolerance = settings.tolerance ?? 0;
  validateCapacity('GPUSpatialKMeans', 'seed', seed, 0, 0xffffffff);
  // The float32 word must stay finite: a larger value would round to infinity and stop at once.
  validateRequirement(
    'GPUSpatialKMeans',
    Number.isFinite(Math.fround(tolerance)) && tolerance >= 0,
    'tolerance must be a finite float32 number >= 0'
  );
  const values = new Uint32Array(GPU_SPATIAL_KMEANS_PARAMETER_LENGTH);
  values[0] = seed;
  new Float32Array(values.buffer)[1] = tolerance;
  return values;
}

/** Caller-owned outputs of {@link GPUSpatialKMeans}. */
export type GPUSpatialKMeansOutput = {
  /** Labels, one row per point: cluster `0..k-1`, or `0xffffffff` for noise. */
  labels: GraphDataView<'uint32'>;
  /**
   * Centers, `k` rows. A center that receives no point in an iteration is relocated like
   * scikit-learn does (see {@link GPUSpatialKMeans}): it moves to one of the points farthest from their
   * assigned centers. Only when no such point exists (every point sits on its center) it moves to
   * the mean of the largest cluster. A center that could not be initialized (fewer than `k` valid
   * points with `'first-valid'`, or no valid point at all with `'kmeans++'`) is NaN, is never
   * relocated and stays empty. When `'kmeans++'` runs out of weight (every remaining valid point
   * coincides with a chosen center, for example fewer distinct points than `k`), the next center
   * is the lowest valid row again: a finite duplicate, whose empty cluster is then relocated as
   * above.
   */
  centers: GraphDataView<'float32x2'>;
  /** Optional member count per cluster, `k` rows. */
  sizes?: GraphDataView<'uint32'>;
  /**
   * Optional squared distance of each point to its center, one row per point; NaN for noise. An
   * explain column: its sum is the within-cluster inertia.
   */
  squaredDistances?: GraphDataView<'float32'>;
  /**
   * Optional one-row total inertia (sum of squared distances of the valid points to their final
   * centers) of the returned clustering, i.e. of the best restart when `restartCount > 1`. NaN
   * when there are no valid points.
   */
  inertia?: GraphDataView<'float32'>;
  /**
   * Optional one-row status words. With restarts they describe the returned (lowest-inertia)
   * restart.
   */
  status?: {
    /** Number of iterations that moved centers (1 to `maximumIterations`). */
    iterationCount?: GPUStatusScalar;
    /**
     * `1` when the last iteration shifted every center by at most `tolerance`, `0` when
     * `maximumIterations` ran out first. A non-finite shift (float32 overflow of the coordinate
     * sums, or an infinite center) never converges.
     */
    converged?: GPUStatusScalar;
    /** One when the chosen restart exhausted its budget without meeting tolerance. */
    iterationLimitReached?: GPUStatusScalar;
  };
};

/**
 * Properties for {@link GPUSpatialKMeans}.
 *
 * Per-frame (no recompile): the contents of `positions` and `parameters` (seed, tolerance).
 * Compile-time: `positions.length`, `clusterCount`, `maximumIterations`, `initialization`, `restartCount` and
 * which optional outputs are present.
 */
export type GPUSpatialKMeansProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'kmeans'`. */
  id?: string;
  /**
   * Coordinate space and metric of `positions`. Omitted means native planar coordinates. Distances
   * are Euclidean in the position units and centers are arithmetic means, so only planar contexts
   * are accepted; project longitude/latitude to a planar frame first.
   */
  spatialContext?: GPUSpatialContext;
  /** Packed planar points, one row per point. Rows with a non-finite coordinate are noise. */
  positions: GraphDataView<'float32x2'>;
  /** Number of clusters, an integer in `[1, 256]`. Compile-time. */
  clusterCount: number;
  /**
   * Maximum number of Lloyd iterations (assign then update), an integer in `[1, 64]`.
   * Compile-time. The graph always holds this many iterations, but once an iteration moves no
   * center by more than the per-frame `tolerance` the remaining ones are skipped on the GPU, so
   * the result is still a pure function of the inputs.
   */
  maximumIterations: number;
  /**
   * Per-frame settings: {@link GPU_SPATIAL_KMEANS_PARAMETER_LENGTH} `uint32` words written with
   * {@link getGPUSpatialKMeansParameterValues} (seed and tolerance).
   */
  parameters: GraphDataView<'uint32'>;
  /**
   * Accumulates each cluster in coordinates relative to its own origin, the data point it was
   * seeded from (or moved to by empty-cluster relocation), instead of the raw coordinates. Float32
   * sums of projected coordinates near 1e6 to 1e7 lose metres of precision over 1e5 or more points
   * per cluster, which shifts centers and can keep a `tolerance` of 0 from ever being reached;
   * offsets from a cluster's own origin keep its sums and distances at the scale of that cluster's
   * extent, so a small cluster next to a distant one keeps its spread too. Assignments, centers and
   * distances are computed in each cluster's frame and the origin is added back to
   * `output.centers` (one float32 rounding of the absolute coordinates remains). Defaults to true.
   * Set false explicitly to reproduce the historical uncentered arithmetic, which can lose
   * distances and inertia even for 255 projected points. Compile-time. It adds no kernel to a
   * multi-restart run (which already publishes centers); a single-restart run gains one
   * `publish-centers` kernel, and the scratch centers hold `2k + 1` extra floats pairs (origins and
   * a zero word).
   */
  centerCoordinates?: boolean;
  /**
   * `'kmeans++'` (default) draws the initial centers with D-squared weighting from seeded Philox,
   * so the same seed always gives the same centers on one device. `'first-valid'` uses the `k`
   * lowest-index points with finite coordinates, which is order-dependent and can place centers
   * next to each other; it has no randomness, so it cannot be combined with `restartCount > 1`.
   */
  initialization?: GPUSpatialKMeansInitialization;
  /**
   * Number of independent seeded initializations, each followed by its own Lloyd iterations, an
   * integer in `[1, 32]`. Compile-time; default 1 (scikit-learn's `n_init='auto'` for k-means++).
   * The run with the lowest total inertia is returned (ties go to the lowest restart index).
   * Restart `r` is seeded with `getGPUSpatialKMeansRestartSeed(seed, r)`. Each restart adds a full
   * seeding and iteration chain to the graph, so graph size and GPU work grow linearly with it.
   * Requires `initialization: 'kmeans++'` when above 1.
   */
  restartCount?: number;
  /** Caller-owned outputs. */
  output: GPUSpatialKMeansOutput;
};

/**
 * Deterministic k-means (Lloyd's algorithm) of planar points.
 *
 * See also the published `GPUKMeans` in `@luma.gl/graph/gpu-vector-search`, which clusters
 * N-dimensional embedding vectors with evenly spaced seeding. This class is a different contract:
 * 2D points, `k-means++` seeding by default, best-of-`restartCount` restarts, a convergence tolerance, a noise label for invalid
 * points and a per-point squared distance (inertia) output.
 *
 * Every choice that could vary between runs is fixed:
 * - Initial centers are `k-means++` (default) with seeded Philox draws or, with
 *   `initialization: 'first-valid'`, the first `k` valid points (lowest index first). Each
 *   `k-means++` round samples the next center with probability proportional to the
 *   squared distance to the nearest chosen center, by an exponential race `-ln(u) / w` whose
 *   minimum is found with integer atomics, lowest row first on ties. `u` is one Philox 4x32-10
 *   block per `(row, round)` keyed by the restart seed and the registered purpose
 *   `kMeansPlusPlusSeeding`, mapped to the open interval `(0, 1)`.
 * - Each of the `maximumIterations` rounds assigns every point to the nearest center, the lowest center
 *   ID winning ties, and moves each center to the mean of its members. Means use a fixed-order chunked
 *   reduction (no sort, no atomics), so sums are bitwise reproducible on one device. A final assignment
 *   against the last centers produces `labels`, `sizes` and `squaredDistances`.
 *
 * Results are reproducible for identical inputs on one device. `log` is not exactly specified by
 * WGSL, so `'kmeans++'` draws may differ between devices; `'first-valid'` has no such caveat apart
 * from f32 distance rounding.
 *
 * Empty clusters are relocated after every assignment step, following scikit-learn's
 * `_relocate_empty_clusters_dense`: with `E` empty clusters (lowest cluster ID first) the `E`
 * points with the largest squared distance to their assigned center, farthest first, become the
 * new centers of those clusters, and the donor clusters' counts and sums lose those points before
 * the means are taken, so the means stay consistent. Ties in distance go to the lowest row. The
 * selection is a single-workgroup iterated argmax with a fixed reduction order, so it is
 * deterministic and adds no readback. If the farthest distance is 0 (every point sits on its
 * center) nothing is relocated, and clusters that are still empty (that case, or a donor that lost
 * its only member) move to the mean of the largest cluster, lowest ID on ties. Differences from
 * scikit-learn: it orders the `E` farthest points with `argpartition`, whose order among them is
 * unspecified, and for a still-empty cluster before the largest one it copies the largest
 * cluster's unaveraged sum; this class always uses descending distance and the averaged mean.
 * Clusters whose center is NaN (never initialized) are not relocated.
 *
 * Cost per iteration: an `O(n k)` assignment, an `O(n k)` shared-memory reduction over `n`
 * points and, when a cluster is empty, an `O(n E)` relocation scan in one workgroup (three
 * dispatches); no sort, so memory does not grow with `maximumIterations`.
 * Non-goals: weighted points, 3D points, k-medoids.
 */
export class GPUSpatialKMeans implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Resolved outputs, including graph-owned stopping views after graph.add(). */
  output: GPUSpatialKMeansOutput;
  readonly estimatedCommandNodeCount: number;
  /** Validated properties. */
  readonly props: GPUSpatialKMeansProps;
  /** Resolved initialization. */
  readonly initialization: GPUSpatialKMeansInitialization;
  /** Resolved restart count. */
  readonly restartCount: number;
  /** Whether sums run in coordinates relative to a data origin (see `props.centerCoordinates`). */
  readonly centerCoordinates: boolean;
  /** Resolved spatial context (native planar when none was given). */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUSpatialKMeansProps) {
    const id = props.id ?? 'kmeans';
    validateRequirement(
      {id, contributor: 'GPUSpatialKMeans', messagePrefix: `${id}:`},
      !('k' in props),
      'k was renamed; use clusterCount'
    );
    this.id = id;
    props = {...props, output: {...props.output, status: {...props.output.status}}};
    this.props = props;
    this.output = props.output;
    this.initialization = props.initialization ?? 'kmeans++';
    this.restartCount = props.restartCount ?? 1;
    this.centerCoordinates = props.centerCoordinates ?? true;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      KMEANS_SPATIAL_CONTEXT_SUPPORT
    );
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    const rows = props.positions.length;
    validateRequirement(id, rows >= 1, 'positions must hold at least one row');
    validateCapacity(
      id,
      'clusterCount',
      props.clusterCount,
      1,
      GPU_SPATIAL_KMEANS_MAXIMUM_CLUSTERS
    );
    validateCapacity(
      id,
      'maximumIterations',
      props.maximumIterations,
      1,
      GPU_SPATIAL_KMEANS_MAXIMUM_ITERATIONS
    );
    validateRequirement(
      id,
      !(this.initialization !== 'first-valid' && this.initialization !== 'kmeans++'),
      `initialization must be 'first-valid' or 'kmeans++'`
    );
    validateCapacity(id, 'restartCount', this.restartCount, 1, GPU_SPATIAL_KMEANS_MAXIMUM_RESTARTS);
    validateRequirement(
      id,
      !(this.restartCount > 1 && this.initialization !== 'kmeans++'),
      `restartCount above 1 needs the 'kmeans++' initialization`
    );
    validateSinglePackedView(props.parameters, ['uint32'], `${id} parameters`);
    validateRequirement(
      id,
      !(props.parameters.length < GPU_SPATIAL_KMEANS_PARAMETER_LENGTH),
      `parameters must hold ${GPU_SPATIAL_KMEANS_PARAMETER_LENGTH} uint32`
    );
    const {output} = props;
    for (const [name, view] of [
      ['inertia', output.inertia],
      ['status.iterationCount', output.status?.iterationCount],
      ['status.converged', output.status?.converged],
      ['status.iterationLimitReached', output.status?.iterationLimitReached]
    ] as const) {
      if (view) {
        validateSinglePackedView(
          view,
          name === 'inertia' ? ['float32'] : ['uint32'],
          `${id} output.${name}`
        );
        validateRequirement(id, !(view.length !== 1), `output.${name} must hold one row`);
      }
    }
    validateSinglePackedView(output.labels, ['uint32'], `${id} output.labels`);
    validateRequirement(
      id,
      !(output.labels.length !== rows),
      `output.labels length must equal positions length`
    );
    validateSinglePackedView(output.centers, ['float32x2'], `${id} output.centers`);
    validateRequirement(
      id,
      !(output.centers.length !== props.clusterCount),
      `output.centers length must equal clusterCount`
    );
    if (output.sizes) {
      validateSinglePackedView(output.sizes, ['uint32'], `${id} output.sizes`);
      validateRequirement(
        id,
        !(output.sizes.length !== props.clusterCount),
        `output.sizes length must equal clusterCount`
      );
    }
    if (output.squaredDistances) {
      validateSinglePackedView(
        output.squaredDistances,
        ['float32'],
        `${id} output.squaredDistances`
      );
      validateRequirement(
        id,
        !(output.squaredDistances.length !== rows),
        `output.squaredDistances length must equal positions length`
      );
    }
    validateSpatialOutputViews(
      id,
      Object.fromEntries(getOutputs(output).map((view, index) => [`output[${index}]`, view])),
      [
        {
          read: [!this.centerCoordinates ? output.centers : undefined],
          write: [output.labels, output.squaredDistances]
        },
        {read: [output.labels], write: [output.sizes]},
        ...((props.restartCount ?? 1) === 1
          ? [
              {
                read: [output.labels],
                write: [output.sizes, !this.centerCoordinates ? output.centers : undefined]
              }
            ]
          : [])
      ]
    );
    this.estimatedCommandNodeCount =
      KMEANS_FIXED_COMMAND_NODE_ALLOWANCE +
      this.restartCount *
        (KMEANS_RESTART_COMMAND_NODE_ALLOWANCE +
          KMEANS_NODES_PER_ITERATION * props.maximumIterations +
          (this.initialization === 'kmeans++' ? KMEANS_NODES_PER_SEED * props.clusterCount : 0));
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateGraphOutputsDisjointFromInputs(id, getOutputs(output), [
      props.positions,
      props.parameters
    ]);
  }

  /** Returns the preparation, initialization, Lloyd iteration and final assignment nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, clusterCount, maximumIterations, parameters} = props;
    const output = {...props.output, status: {...props.output.status}};
    const {labels, centers} = output;
    validateGraphViewsBelongToGraph(id, graph, [positions, parameters, ...getOutputs(output)]);
    const rows = positions.length;
    validateGraphChunkBindingRanges(graph, id, {
      positions,
      parameters,
      ...Object.fromEntries(getOutputs(output).map((view, index) => [`output[${index}]`, view]))
    });
    const partialCount =
      clusterCount * Math.min(MAXIMUM_CHUNKS, Math.ceil(rows / REDUCTION_WORKGROUP_SIZE));
    for (const [name, byteLength] of [
      ['row scratch', rows * 4],
      ['partial reductions', partialCount * 4],
      ['cluster sums', clusterCount * 8],
      ['iteration dispatch', 48]
    ] as const)
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        byteLength,
        'reduce positions rows or clusterCount'
      );
    for (const name of ['converged', 'iterationCount', 'iterationLimitReached'] as const) {
      output.status[name] ??= createTransientView(graph, `${id}-status-${name}`, 'uint32', 1);
    }
    this.output = output;
    const restartCount = this.restartCount;
    const multiStart = restartCount > 1;
    const view = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);

    const centering = this.centerCoordinates;
    // Centered runs keep one origin per cluster beside its center. In the scratch centers, floats
    // [0, 2k) are the centers relative to their origins, [2k, 4k) the absolute origins and the
    // pair at 4k is zero: shaders read it as the runtime zero that keeps the origin subtraction
    // from being regrouped (`getRoundedDifference`).
    const centerSlotCount = centering ? 4 * clusterCount + 2 : 2 * clusterCount;
    const rawPosition = (row: string, axis: 0 | 1) =>
      `positions[positionsOffset + ${row} * 2u${axis === 1 ? ' + 1u' : ''}]`;
    const originOf = (array: string, cluster: string, axis: 0 | 1) =>
      `${array}[${array}Offset + ${2 * clusterCount}u + (${cluster}) * 2u${axis === 1 ? ' + 1u' : ''}]`;
    const runtimeZero = (array: string) =>
      `bitcast<u32>(${array}[${array}Offset + ${4 * clusterCount}u])`;
    // Coordinate `value` relative to the origin of `cluster` (unchanged when not centered).
    const localPosition = (value: string, array: string, cluster: string, axis: 0 | 1) =>
      centering
        ? `getRoundedDifference(${value}, ${originOf(array, cluster, axis)}, ${runtimeZero(array)})`
        : value;
    const validFlags = view('valid-flags', 'uint32', rows);
    const rowIds = view('row-ids', 'uint32', rows);
    // Uncentered partial sums read the valid-selected coordinates; centered ones read positions.
    const xs = centering ? undefined : view('xs', 'float32', rows);
    const ys = centering ? undefined : view('ys', 'float32', rows);
    const sizes = output.sizes ?? view('sizes', 'uint32', clusterCount);
    // With restarts every Lloyd run works on scratch centers and sizes; the best restart's centers
    // are copied to the caller's views at the end.
    const workCenters =
      multiStart || centering
        ? createTransientView(graph, `${id}-work-centers`, 'float32x2', centerSlotCount / 2)
        : centers;
    const workSizes = multiStart ? view('work-sizes', 'uint32', clusterCount) : sizes;
    // Squared distances are always produced when inertia is needed (restart scoring or output).
    const finalDistances =
      output.squaredDistances ??
      (multiStart || output.inertia ? view('squared-distances', 'float32', rows) : undefined);
    // Word 0: converged flag, word 1: iterations used, word 2: largest center shift (f32 bits;
    // non-negative floats order like their bit patterns, so atomicMax works).
    const state = view('state', 'uint32', 3);
    // Seeding cursor: word 0 is the restart index, word 1 the k-means++ round. One-thread nodes
    // advance it on the GPU, so the restart and round never reach the WGSL text (one pipeline per
    // kernel at any k and restartCount).
    const seedCursor = view('seed-cursor', 'uint32', 2);
    // Update step: a sort-free, atomic-free deterministic reduction. Points are cut into
    // `chunkCount` contiguous chunks of `chunkSize` rows (a multiple of the tile width); each
    // workgroup keeps per-cluster partial counts and sums for its chunk, then one workgroup per
    // cluster folds the chunk partials in a fixed order.
    const chunkCount = Math.min(MAXIMUM_CHUNKS, Math.ceil(rows / REDUCTION_WORKGROUP_SIZE));
    const chunkSize =
      Math.ceil(rows / chunkCount / REDUCTION_WORKGROUP_SIZE) * REDUCTION_WORKGROUP_SIZE;
    const slices = Math.min(MAXIMUM_SLICES, Math.floor(REDUCTION_WORKGROUP_SIZE / clusterCount));
    const partialCounts = view('partial-counts', 'uint32', clusterCount * chunkCount);
    const partialSumsX = view('partial-sums-x', 'float32', clusterCount * chunkCount);
    const partialSumsY = view('partial-sums-y', 'float32', clusterCount * chunkCount);
    // Per-iteration scratch of the empty-cluster relocation: each point's squared distance to the
    // center it was assigned to, and the per-cluster coordinate sums before the means are taken.
    const iterationDistances = view('iteration-distances', 'float32', rows);
    const clusterSums = createTransientView(graph, `${id}-cluster-sums`, 'float32x2', clusterCount);
    // Indirect dispatch triples of the gated Lloyd kernels (assign, partial, update, relocate).
    // Each restart arms them; the one-thread converge kernel zeroes every x once the centers stop
    // moving, so the remaining iterations launch nothing. The converge kernel itself stays ungated:
    // one buffer must not be both an indirect argument and storage-written in one dispatch.
    const iterationGate = createWGSLDispatchGate(graph, {
      id: `${id}-iteration-dispatch`,
      operation: OPERATION,
      shapes: [
        [rows],
        [chunkCount * REDUCTION_WORKGROUP_SIZE, REDUCTION_WORKGROUP_SIZE],
        [clusterCount * REDUCTION_WORKGROUP_SIZE, REDUCTION_WORKGROUP_SIZE],
        [REDUCTION_WORKGROUP_SIZE, REDUCTION_WORKGROUP_SIZE]
      ]
    });
    const sharedWGSL = `const K: u32 = ${clusterCount}u;
const NOISE: u32 = ${GPU_DBSCAN_NOISE}u;

// Mirrors getGPUSpatialKMeansRestartSeed: restart 0 uses the seed itself, later restarts a murmur3 mix.
// The seed is the per-frame parameters word 0, so the source is the same for every seed.
fn getRestartSeed(seed: u32, restart: u32) -> u32 {
  if (restart == 0u) {
    return seed;
  }
  var hash = seed + restart * 0x9e3779b9u;
  hash = hash ^ (hash >> 16u);
  hash = hash * 0x85ebca6bu;
  hash = hash ^ (hash >> 13u);
  hash = hash * 0xc2b2ae35u;
  return hash ^ (hash >> 16u);
}

${IS_FINITE_FLOAT_WGSL}
${IS_NAN_BITS_WGSL}

${GET_QUIET_NAN_SEEDED_WGSL}
${centering ? ROUNDED_DIFFERENCE_WGSL : ''}`;

    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-prepare`,
        operation: OPERATION,
        variant: 'prepare',
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'validFlags', view: validFlags, type: 'u32', access: 'read_write'},
          {name: 'rowIds', view: rowIds, type: 'u32', access: 'read_write'},
          ...(xs && ys
            ? [
                {name: 'xs', view: xs, type: 'f32' as const, access: 'read_write' as const},
                {name: 'ys', view: ys, type: 'f32' as const, access: 'read_write' as const}
              ]
            : []),
          {name: 'seedCursor', view: seedCursor, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: sharedWGSL,
        // The first restart-begin node increments the restart cursor to 0.
        body: `if (index == 0u) {
    seedCursor[seedCursorOffset] = 0xffffffffu;
  }
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let valid = isFiniteFloat(x) && isFiniteFloat(y);
  validFlags[validFlagsOffset + index] = select(0u, 1u, valid);
  rowIds[rowIdsOffset + index] = index;${
    xs
      ? `
  xs[xsOffset + index] = select(0.0, x, valid);
  ys[ysOffset + index] = select(0.0, y, valid);`
      : ''
  }`
      })
    ];

    const seeding = this.initialization === 'kmeans++';
    const minDistance = seeding ? view('min-distance', 'float32', rows) : undefined!;
    const keys = seeding ? view('keys', 'uint32', rows) : undefined!;
    const bestKey = seeding ? view('best-key', 'uint32', 1) : undefined!;
    const bestRow = seeding ? view('best-row', 'uint32', 1) : undefined!;
    const bestInertia = output.inertia ?? view('best-inertia', 'float32', 1);
    const bestCenters = multiStart
      ? createTransientView(graph, `${id}-best-centers`, 'float32x2', centerSlotCount / 2)
      : undefined;
    const bestState = multiStart ? view('best-state', 'uint32', 3) : undefined;
    const assignNode = (
      assignId: string,
      withDistances: boolean,
      centersView: GraphDataView<'float32x2'>,
      distancesView?: GraphDataView<'float32'>
    ): GPUCommandNode<Parameters> =>
      createWGSLKernelNode<Parameters>(graph, {
        id: assignId,
        operation: OPERATION,
        variant: withDistances ? 'assign-final' : 'assign',
        // Lloyd-iteration assignments are skipped once the centers stop moving.
        ...(withDistances ? {} : iterationGate.getGate(0)),
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'centers', view: centersView, type: 'f32', access: 'read'},
          {name: 'labels', view: labels, type: 'u32', access: 'read_write'},
          ...(withDistances
            ? []
            : [{name: 'state', view: state, type: 'u32' as const, access: 'read' as const}]),
          ...(distancesView
            ? [
                {
                  name: 'squaredDistances',
                  view: distancesView,
                  type: 'f32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: rows,
        declarations: sharedWGSL,
        body: `${
          withDistances
            ? ''
            : `// Converged: centers no longer move, so labels are already final.
  if (state[stateOffset] != 0u) {
    return;
  }
  `
        }let x = ${rawPosition('index', 0)};
  let y = ${rawPosition('index', 1)};
  var best = NOISE;
  var bestDistance = getQuietNaN(index);
  if (isFiniteFloat(x) && isFiniteFloat(y)) {
    // Strict comparison: the lowest center ID wins ties; NaN centers never win.
    for (var center = 0u; center < K; center++) {
      let deltaX = centers[centersOffset + center * 2u] - ${localPosition('x', 'centers', 'center', 0)};
      let deltaY = centers[centersOffset + center * 2u + 1u] - ${localPosition('y', 'centers', 'center', 1)};
      let distance = deltaX * deltaX + deltaY * deltaY;
      if (!isNanBits(distance) && (best == NOISE || distance < bestDistance)) {
        best = center;
        bestDistance = distance;
      }
    }
  }
  labels[labelsOffset + index] = best;
  ${distancesView ? 'squaredDistances[squaredDistancesOffset + index] = select(getQuietNaN(index), bestDistance, best != NOISE);' : ''}`
      });

    const addInitialization = (restartId: string) => {
      if (this.initialization === 'first-valid') {
        const compactIds = view('compact-ids', 'uint32', rows);
        const validCount = view('valid-count', 'uint32', 1);
        nodes.push(
          ...new GPUCompaction({
            id: `${restartId}-valid-rows`,
            input: rowIds,
            flags: validFlags,
            output: compactIds,
            count: validCount
          }).getCommandNodes(graph),
          createWGSLKernelNode<Parameters>(graph, {
            id: `${restartId}-initial-centers`,
            operation: OPERATION,
            variant: 'initial-centers',
            bindings: [
              {name: 'positions', view: positions, type: 'f32', access: 'read'},
              {name: 'compactIds', view: compactIds, type: 'u32', access: 'read'},
              {name: 'validCount', view: validCount, type: 'u32', access: 'read'},
              {name: 'centers', view: workCenters, type: 'f32', access: 'read_write'}
            ],
            invocationCount: clusterCount,
            declarations: sharedWGSL,
            body: centering
              ? `var x = getQuietNaN(index);
    var y = x;
    var originX = 0.0;
    var originY = 0.0;
    if (index < validCount[validCountOffset]) {
      let row = compactIds[compactIdsOffset + index];
      originX = ${rawPosition('row', 0)};
      originY = ${rawPosition('row', 1)};
      x = 0.0;
      y = 0.0;
    }
    centers[centersOffset + index * 2u] = x;
    centers[centersOffset + index * 2u + 1u] = y;
    ${originOf('centers', 'index', 0)} = originX;
    ${originOf('centers', 'index', 1)} = originY;
    if (index == 0u) {
      centers[centersOffset + ${4 * clusterCount}u] = 0.0;
      centers[centersOffset + ${4 * clusterCount + 1}u] = 0.0;
    }`
              : `let nan = getQuietNaN(index);
    var x = nan;
    var y = nan;
    if (index < validCount[validCountOffset]) {
      let row = compactIds[compactIdsOffset + index];
      x = ${rawPosition('row', 0)};
      y = ${rawPosition('row', 1)};
    }
    centers[centersOffset + index * 2u] = x;
    centers[centersOffset + index * 2u + 1u] = y;`
          })
        );
      } else {
        for (let round = 0; round < clusterCount; round++) {
          const roundId = `${restartId}-seed-${round}`;
          nodes.push(
            createWGSLKernelNode<Parameters>(graph, {
              id: `${roundId}-begin`,
              operation: OPERATION,
              variant: 'seed-begin',
              bindings: [
                {name: 'bestKey', view: bestKey, type: 'u32', access: 'read_write'},
                {name: 'bestRow', view: bestRow, type: 'u32', access: 'read_write'},
                {name: 'seedCursor', view: seedCursor, type: 'u32', access: 'read_write'}
              ],
              invocationCount: 1,
              // restart-begin leaves the round at 0xffffffff, so the first round is 0.
              body: `bestKey[bestKeyOffset] = ${NO_ROW};
    bestRow[bestRowOffset] = ${NO_ROW};
    seedCursor[seedCursorOffset + 1u] = seedCursor[seedCursorOffset + 1u] + 1u;`
            }),
            createWGSLKernelNode<Parameters>(graph, {
              id: `${roundId}-key`,
              operation: OPERATION,
              variant: 'seed-key',
              bindings: [
                {name: 'validFlags', view: validFlags, type: 'u32', access: 'read'},
                {name: 'minDistance', view: minDistance, type: 'f32', access: 'read'},
                {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
                {name: 'bestKey', view: bestKey, type: 'atomic<u32>', access: 'read_write'},
                {name: 'seedCursor', view: seedCursor, type: 'u32', access: 'read'},
                {name: 'parameters', view: parameters, type: 'u32', access: 'read'}
              ],
              invocationCount: rows,
              workgroupSize: REDUCTION_WORKGROUP_SIZE,
              guardIndex: false,
              declarations: `${sharedWGSL}
  var<workgroup> groupBestKey: atomic<u32>;

  ${PHILOX_WGSL}
  // The registered purpose of this stream, not a seed: the seed is a per-frame parameters word.
  const PHILOX_PURPOSE: u32 = ${PHILOX_PURPOSES.kMeansPlusPlusSeeding.id}u;`,
              // Each workgroup folds its minimum in shared memory first, so the global word sees one
              // atomic per workgroup instead of one per point.
              body: `if (localInvocationIndex == 0u) {
      atomicStore(&groupBestKey, 0xffffffffu);
    }
    workgroupBarrier();
    let round = seedCursor[seedCursorOffset + 1u];
    let restartSeed = getRestartSeed(parameters[parametersOffset], seedCursor[seedCursorOffset]);
    var key = 0xffffffffu;
    if (index < INVOCATION_COUNT && validFlags[validFlagsOffset + index] != 0u) {
      // Exponential race: the minimum of -ln(u) / w samples a row with probability proportional to w.
      // One Philox block per (row, round) under the registered purpose; the uniform is open, so
      // log() never sees 0 and never returns the 0 that would win the race whatever the weight.
      let weight = select(minDistance[minDistanceOffset + index], 1.0, round == 0u);
      let word = philox4x32(vec4<u32>(index, round, 0u, 0u), vec2<u32>(restartSeed, PHILOX_PURPOSE)).x;
      let uniformValue = philoxOpenUnitFloat(word);
      key = select(0x7f800000u, bitcast<u32>(-log(uniformValue) / weight), weight > 0.0);
      atomicMin(&groupBestKey, key);
      keys[keysOffset + index] = key;
    } else if (index < INVOCATION_COUNT) {
      keys[keysOffset + index] = key;
    }
    workgroupBarrier();
    if (localInvocationIndex == 0u) {
      atomicMin(&bestKey[bestKeyOffset], atomicLoad(&groupBestKey));
    }`
            }),
            createWGSLKernelNode<Parameters>(graph, {
              id: `${roundId}-row`,
              operation: OPERATION,
              variant: 'seed-row',
              bindings: [
                {name: 'keys', view: keys, type: 'u32', access: 'read'},
                {name: 'bestKey', view: bestKey, type: 'u32', access: 'read'},
                {name: 'bestRow', view: bestRow, type: 'atomic<u32>', access: 'read_write'}
              ],
              invocationCount: rows,
              body: `let best = bestKey[bestKeyOffset];
    if (best != 0xffffffffu && keys[keysOffset + index] == best) {
      atomicMin(&bestRow[bestRowOffset], index);
    }`
            }),
            createWGSLKernelNode<Parameters>(graph, {
              id: `${roundId}-update`,
              operation: OPERATION,
              variant: 'seed-update',
              bindings: [
                {name: 'positions', view: positions, type: 'f32', access: 'read'},
                {name: 'validFlags', view: validFlags, type: 'u32', access: 'read'},
                {name: 'bestRow', view: bestRow, type: 'u32', access: 'read'},
                {name: 'minDistance', view: minDistance, type: 'f32', access: 'read_write'},
                {name: 'centers', view: workCenters, type: 'f32', access: 'read_write'},
                {name: 'seedCursor', view: seedCursor, type: 'u32', access: 'read'}
              ],
              invocationCount: rows,
              declarations: sharedWGSL,
              body: `let round = seedCursor[seedCursorOffset + 1u];
    let row = bestRow[bestRowOffset];
    let hasCenter = row != 0xffffffffu;
    var centerX = getQuietNaN(index);
    var centerY = centerX;
    if (hasCenter) {
      centerX = ${rawPosition('row', 0)};
      centerY = ${rawPosition('row', 1)};
    }
    if (index == 0u) {${
      centering
        ? `
      // The drawn point becomes the cluster's origin and its relative center is zero.
      centers[centersOffset + round * 2u] = select(centerX, 0.0, hasCenter);
      centers[centersOffset + round * 2u + 1u] = select(centerY, 0.0, hasCenter);
      ${originOf('centers', 'round', 0)} = select(0.0, centerX, hasCenter);
      ${originOf('centers', 'round', 1)} = select(0.0, centerY, hasCenter);
      centers[centersOffset + ${4 * clusterCount}u] = 0.0;
      centers[centersOffset + ${4 * clusterCount + 1}u] = 0.0;`
        : `
      centers[centersOffset + round * 2u] = centerX;
      centers[centersOffset + round * 2u + 1u] = centerY;`
    }
    }
    if (hasCenter && validFlags[validFlagsOffset + index] != 0u) {
      let deltaX = ${rawPosition('index', 0)} - centerX;
      let deltaY = ${rawPosition('index', 1)} - centerY;
      let distance = deltaX * deltaX + deltaY * deltaY;
      minDistance[minDistanceOffset + index] =
        select(min(minDistance[minDistanceOffset + index], distance), distance, round == 0u);
    }`
            })
          );
        }
      }
    };
    const addIterations = (restartId: string) => {
      for (let iteration = 0; iteration < maximumIterations; iteration++) {
        const iterationId = `${restartId}-iteration-${iteration}`;
        nodes.push(
          assignNode(`${iterationId}-assign`, false, workCenters, iterationDistances),
          createWGSLKernelNode<Parameters>(graph, {
            id: `${iterationId}-partial`,
            operation: OPERATION,
            variant: 'partial',
            bindings: [
              {name: 'labels', view: labels, type: 'u32', access: 'read'},
              ...(xs && ys
                ? [
                    {name: 'xs', view: xs, type: 'f32' as const, access: 'read' as const},
                    {name: 'ys', view: ys, type: 'f32' as const, access: 'read' as const}
                  ]
                : [
                    {
                      name: 'positions',
                      view: positions,
                      type: 'f32' as const,
                      access: 'read' as const
                    },
                    {
                      name: 'centers',
                      view: workCenters,
                      type: 'f32' as const,
                      access: 'read' as const
                    }
                  ]),
              {name: 'state', view: state, type: 'u32', access: 'read'},
              {name: 'partialCounts', view: partialCounts, type: 'u32', access: 'read_write'},
              {name: 'partialSumsX', view: partialSumsX, type: 'f32', access: 'read_write'},
              {name: 'partialSumsY', view: partialSumsY, type: 'f32', access: 'read_write'}
            ],
            workgroupSize: REDUCTION_WORKGROUP_SIZE,
            invocationCount: chunkCount * REDUCTION_WORKGROUP_SIZE,
            guardIndex: false,
            ...iterationGate.getGate(1),
            declarations: `${sharedWGSL}
  const ROWS: u32 = ${rows}u;
  const CHUNK_SIZE: u32 = ${chunkSize}u;
  const CHUNK_COUNT: u32 = ${chunkCount}u;
  const TILE_COUNT: u32 = ${chunkSize / REDUCTION_WORKGROUP_SIZE}u;
  const SLICES: u32 = ${slices}u;
  const LANES: u32 = ${slices * clusterCount}u;
  var<workgroup> tileLabels: array<u32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> tileXs: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> tileYs: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> laneCounts: array<u32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> laneSumsX: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> laneSumsY: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;`,
            // Every barrier sits in uniform control flow: the tile loop has a constant trip count and
            // the converged / out-of-range cases only change the data (noise labels match no cluster).
            body: `let chunk = index / ${REDUCTION_WORKGROUP_SIZE}u;
    let converged = state[stateOffset] != 0u;
    let begin = chunk * CHUNK_SIZE;
    let end = min(begin + CHUNK_SIZE, ROWS);
    let lane = localInvocationIndex;
    let cluster = lane % K;
    let slice = lane / K;
    var count = 0u;
    var sumX = 0.0;
    var sumY = 0.0;
    for (var tile = 0u; tile < TILE_COUNT; tile++) {
      let row = begin + tile * ${REDUCTION_WORKGROUP_SIZE}u + lane;
      var label = NOISE;
      var x = 0.0;
      var y = 0.0;
      if (!converged && row < end) {
        label = labels[labelsOffset + row];${
          centering
            ? `
        // Offsets from the origin of the assigned cluster; noise rows add nothing.
        if (label < K) {
          x = ${localPosition(rawPosition('row', 0), 'centers', 'label', 0)};
          y = ${localPosition(rawPosition('row', 1), 'centers', 'label', 1)};
        }`
            : `
        x = xs[xsOffset + row];
        y = ys[ysOffset + row];`
        }
      }
      tileLabels[lane] = label;
      tileXs[lane] = x;
      tileYs[lane] = y;
      workgroupBarrier();
      if (lane < LANES) {
        // Lane (slice, cluster) owns tile entries slice, slice + SLICES, ...: a fixed order.
        for (var entry = slice; entry < ${REDUCTION_WORKGROUP_SIZE}u; entry += SLICES) {
          if (tileLabels[entry] == cluster) {
            count++;
            sumX += tileXs[entry];
            sumY += tileYs[entry];
          }
        }
      }
      workgroupBarrier();
    }
    laneCounts[lane] = count;
    laneSumsX[lane] = sumX;
    laneSumsY[lane] = sumY;
    workgroupBarrier();
    if (lane < K) {
      var chunkCountTotal = 0u;
      var chunkSumX = 0.0;
      var chunkSumY = 0.0;
      for (var sliceIndex = 0u; sliceIndex < SLICES; sliceIndex++) {
        chunkCountTotal += laneCounts[sliceIndex * K + lane];
        chunkSumX += laneSumsX[sliceIndex * K + lane];
        chunkSumY += laneSumsY[sliceIndex * K + lane];
      }
      let slot = lane * CHUNK_COUNT + chunk;
      partialCounts[partialCountsOffset + slot] = chunkCountTotal;
      partialSumsX[partialSumsXOffset + slot] = chunkSumX;
      partialSumsY[partialSumsYOffset + slot] = chunkSumY;
    }`
          }),
          createWGSLKernelNode<Parameters>(graph, {
            id: `${iterationId}-update`,
            operation: OPERATION,
            variant: 'update',
            bindings: [
              {name: 'partialCounts', view: partialCounts, type: 'u32', access: 'read'},
              {name: 'partialSumsX', view: partialSumsX, type: 'f32', access: 'read'},
              {name: 'partialSumsY', view: partialSumsY, type: 'f32', access: 'read'},
              {name: 'sizes', view: workSizes, type: 'u32', access: 'read_write'},
              {name: 'clusterSums', view: clusterSums, type: 'f32', access: 'read_write'},
              {name: 'state', view: state, type: 'u32', access: 'read'}
            ],
            workgroupSize: REDUCTION_WORKGROUP_SIZE,
            invocationCount: clusterCount * REDUCTION_WORKGROUP_SIZE,
            guardIndex: false,
            ...iterationGate.getGate(2),
            declarations: `const CHUNK_COUNT: u32 = ${chunkCount}u;
  var<workgroup> reducedCounts: array<u32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> reducedSumsX: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> reducedSumsY: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;`,
            // One workgroup per cluster: strided per-lane sums over the chunk partials, then a fixed
            // binary tree, so the sums are bitwise reproducible. Lane 0 publishes the member count
            // and coordinate sums; the relocate kernel turns them into means.
            body: `let cluster = index / ${REDUCTION_WORKGROUP_SIZE}u;
    let lane = localInvocationIndex;
    var count = 0u;
    var sumX = 0.0;
    var sumY = 0.0;
    for (var chunk = lane; chunk < CHUNK_COUNT; chunk += ${REDUCTION_WORKGROUP_SIZE}u) {
      let slot = cluster * CHUNK_COUNT + chunk;
      count += partialCounts[partialCountsOffset + slot];
      sumX += partialSumsX[partialSumsXOffset + slot];
      sumY += partialSumsY[partialSumsYOffset + slot];
    }
    reducedCounts[lane] = count;
    reducedSumsX[lane] = sumX;
    reducedSumsY[lane] = sumY;
    workgroupBarrier();
    for (var stride = ${REDUCTION_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
      if (lane < stride) {
        reducedCounts[lane] += reducedCounts[lane + stride];
        reducedSumsX[lane] += reducedSumsX[lane + stride];
        reducedSumsY[lane] += reducedSumsY[lane + stride];
      }
      workgroupBarrier();
    }
    // Converged: centers no longer move; the partials are stale and the final sizes come later.
    if (lane == 0u && state[stateOffset] == 0u) {
      sizes[sizesOffset + cluster] = reducedCounts[0];
      clusterSums[clusterSumsOffset + cluster * 2u] = reducedSumsX[0];
      clusterSums[clusterSumsOffset + cluster * 2u + 1u] = reducedSumsY[0];
    }`
          }),
          // Empty-cluster relocation and the center update, in one workgroup (k <= 256 lanes).
          // scikit-learn's `_relocate_empty_clusters_dense`: the points farthest from the center they
          // were assigned to (largest squared distance first, lowest row on ties) become the centers
          // of the empty clusters (lowest cluster ID gets the farthest point), and the donor
          // clusters' counts and sums lose those points. Selection is an iterated argmax that
          // excludes the previous winner by (distance, row) rank, so it needs no atomics.
          createWGSLKernelNode<Parameters>(graph, {
            id: `${iterationId}-relocate`,
            operation: OPERATION,
            variant: 'relocate',
            bindings: [
              {name: 'positions', view: positions, type: 'f32', access: 'read'},
              {name: 'labels', view: labels, type: 'u32', access: 'read'},
              {name: 'distances', view: iterationDistances, type: 'f32', access: 'read'},
              {name: 'clusterSums', view: clusterSums, type: 'f32', access: 'read'},
              {name: 'sizes', view: workSizes, type: 'u32', access: 'read_write'},
              {name: 'centers', view: workCenters, type: 'f32', access: 'read_write'},
              {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
            ],
            workgroupSize: REDUCTION_WORKGROUP_SIZE,
            invocationCount: REDUCTION_WORKGROUP_SIZE,
            guardIndex: false,
            ...iterationGate.getGate(3),
            declarations: `${sharedWGSL}
  const ROWS: u32 = ${rows}u;
  var<workgroup> activeFlag: u32;
  var<workgroup> stopFlag: u32;
  var<workgroup> emptyCount: u32;
  var<workgroup> pickDistance: f32;
  var<workgroup> pickRow: u32;
  var<workgroup> clusterCounts: array<u32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> clusterSumsX: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> clusterSumsY: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> emptyClusters: array<u32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> laneDistances: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> laneRows: array<u32, ${REDUCTION_WORKGROUP_SIZE}>;${
    centering
      ? `
  var<workgroup> clusterOriginsX: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> clusterOriginsY: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;`
      : ''
  }`,
            // Barriers sit in uniform control flow: every branch that contains one is decided by
            // `workgroupUniformLoad` of a flag only lane 0 writes.
            body: `let lane = localInvocationIndex;
  if (lane == 0u) {
    activeFlag = select(0u, 1u, atomicLoad(&state[stateOffset]) == 0u);
  }
  // Converged: the partials are stale and the centers no longer move.
  if (workgroupUniformLoad(&activeFlag) == 0u) {
    return;
  }
  ${
    centering
      ? `let zero = ${runtimeZero('centers')};
  `
      : ''
  }if (lane < K) {
    clusterCounts[lane] = sizes[sizesOffset + lane];
    clusterSumsX[lane] = clusterSums[clusterSumsOffset + lane * 2u];
    clusterSumsY[lane] = clusterSums[clusterSumsOffset + lane * 2u + 1u];${
      centering
        ? `
    clusterOriginsX[lane] = ${originOf('centers', 'lane', 0)};
    clusterOriginsY[lane] = ${originOf('centers', 'lane', 1)};`
        : ''
    }
  }
  workgroupBarrier();
  if (lane == 0u) {
    // Clusters whose center could not be initialized (NaN) are not empty clusters to relocate.
    var count = 0u;
    for (var cluster = 0u; cluster < K; cluster++) {
      let centerX = centers[centersOffset + cluster * 2u];
      if (clusterCounts[cluster] == 0u && isFiniteFloat(centerX)) {
        emptyClusters[count] = cluster;
        count++;
      }
    }
    emptyCount = count;
    stopFlag = 0u;
    pickDistance = 0.0;
    pickRow = 0u;
  }
  let pickLimit = workgroupUniformLoad(&emptyCount);
  for (var pickIndex = 0u; pickIndex < pickLimit; pickIndex++) {
    // scikit-learn returns without relocating when every point sits on its center.
    if (workgroupUniformLoad(&stopFlag) != 0u) {
      break;
    }
    let previousDistance = pickDistance;
    let previousRow = pickRow;
    var bestDistance = -1.0;
    var bestRow = NOISE;
    for (var row = lane; row < ROWS; row += ${REDUCTION_WORKGROUP_SIZE}u) {
      let distance = distances[distancesOffset + row];
      if (isFiniteFloat(distance) && labels[labelsOffset + row] != NOISE) {
        // Rank (distance descending, row ascending) strictly after the previous winner.
        let ranked = pickIndex == 0u || distance < previousDistance ||
          (distance == previousDistance && row > previousRow);
        if (ranked && distance > bestDistance) {
          bestDistance = distance;
          bestRow = row;
        }
      }
    }
    laneDistances[lane] = bestDistance;
    laneRows[lane] = bestRow;
    workgroupBarrier();
    for (var stride = ${REDUCTION_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
      if (lane < stride) {
        let otherDistance = laneDistances[lane + stride];
        let otherRow = laneRows[lane + stride];
        if (otherDistance > laneDistances[lane] ||
          (otherDistance == laneDistances[lane] && otherRow < laneRows[lane])) {
          laneDistances[lane] = otherDistance;
          laneRows[lane] = otherRow;
        }
      }
      workgroupBarrier();
    }
    if (lane == 0u) {
      let row = laneRows[0];
      let distance = laneDistances[0];
      if (row == NOISE || (pickIndex == 0u && distance == 0.0)) {
        stopFlag = 1u;
      } else {
        pickDistance = distance;
        pickRow = row;
        let emptyCluster = emptyClusters[pickIndex];
        let donor = labels[labelsOffset + row];
${
  centering
    ? `        // The donor loses the point in its own frame; the point becomes the empty
        // cluster's origin, so its relative sum is zero.
        let pointX = ${rawPosition('row', 0)};
        let pointY = ${rawPosition('row', 1)};
        clusterSumsX[donor] -= getRoundedDifference(pointX, clusterOriginsX[donor], zero);
        clusterSumsY[donor] -= getRoundedDifference(pointY, clusterOriginsY[donor], zero);
        clusterCounts[donor] -= 1u;
        clusterOriginsX[emptyCluster] = pointX;
        clusterOriginsY[emptyCluster] = pointY;
        clusterSumsX[emptyCluster] = 0.0;
        clusterSumsY[emptyCluster] = 0.0;
        clusterCounts[emptyCluster] = 1u;`
    : `        let x = ${rawPosition('row', 0)};
        let y = ${rawPosition('row', 1)};
        clusterSumsX[donor] -= x;
        clusterSumsY[donor] -= y;
        clusterCounts[donor] -= 1u;
        clusterSumsX[emptyCluster] = x;
        clusterSumsY[emptyCluster] = y;
        clusterCounts[emptyCluster] = 1u;`
}
      }
    }
    workgroupBarrier();
  }
  // A cluster still empty (every point on its center, or its only member was taken) moves to the
  // mean of the largest cluster, lowest ID on ties, like scikit-learn's \`_average_centers\`.
  var largest = 0u;
  for (var cluster = 1u; cluster < K; cluster++) {
    if (clusterCounts[cluster] > clusterCounts[largest]) {
      largest = cluster;
    }
  }
  if (lane < K) {
    let size = clusterCounts[lane];
    let previousX = centers[centersOffset + lane * 2u];
    let previousY = centers[centersOffset + lane * 2u + 1u];
    var nextX = previousX;
    var nextY = previousY;${
      centering
        ? `
    let previousOriginX = ${originOf('centers', 'lane', 0)};
    let previousOriginY = ${originOf('centers', 'lane', 1)};
    var nextOriginX = clusterOriginsX[lane];
    var nextOriginY = clusterOriginsY[lane];`
        : ''
    }
    if (size > 0u) {
      nextX = clusterSumsX[lane] / f32(size);
      nextY = clusterSumsY[lane] / f32(size);
    } else if (isFiniteFloat(previousX) && clusterCounts[largest] > 0u) {
      nextX = clusterSumsX[largest] / f32(clusterCounts[largest]);
      nextY = clusterSumsY[largest] / f32(clusterCounts[largest]);${
        centering
          ? `
      nextOriginX = clusterOriginsX[largest];
      nextOriginY = clusterOriginsY[largest];`
          : ''
      }
    }
    sizes[sizesOffset + lane] = size;${
      centering
        ? `
    // The absolute shift: the origin move plus the relative move, each rounded once.
    let deltaX = getRoundedDifference(nextOriginX, previousOriginX, zero) + getRoundedDifference(nextX, previousX, zero);
    let deltaY = getRoundedDifference(nextOriginY, previousOriginY, zero) + getRoundedDifference(nextY, previousY, zero);`
        : `
    let deltaX = nextX - previousX;
    let deltaY = nextY - previousY;`
    }
    let magnitude = max(abs(deltaX), abs(deltaY));
    var shift = magnitude;
    if (magnitude > 0.0) {
      let scaledX = deltaX / magnitude;
      let scaledY = deltaY / magnitude;
      shift = magnitude * sqrt(scaledX * scaledX + scaledY * scaledY);
    }
    // A never-initialized (NaN) center does not move. Any other non-finite shift (overflowed
    // coordinate sums or an infinite center) counts as +infinity, written as integer bits (an
    // infinite f32 value cannot appear in WGSL text), so the iteration cannot report convergence.
    if (isFiniteFloat(deltaX) && isFiniteFloat(deltaY) && isFiniteFloat(shift)) {
      atomicMax(&state[stateOffset + 2u], bitcast<u32>(shift));
    } else if (!isNanBits(previousX)) {
      atomicMax(&state[stateOffset + 2u], 0x7f800000u);
    }
    centers[centersOffset + lane * 2u] = nextX;
    centers[centersOffset + lane * 2u + 1u] = nextY;${
      centering
        ? `
    ${originOf('centers', 'lane', 0)} = nextOriginX;
    ${originOf('centers', 'lane', 1)} = nextOriginY;`
        : ''
    }
  }`
          }),
          createWGSLKernelNode<Parameters>(graph, {
            id: `${iterationId}-converge`,
            operation: OPERATION,
            variant: 'converge',
            bindings: [
              {name: 'state', view: state, type: 'u32', access: 'read_write'},
              {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
              iterationGate.binding
            ],
            invocationCount: 1,
            body: `if (state[stateOffset] != 0u) {
      return;
    }
    state[stateOffset + 1u] = state[stateOffset + 1u] + 1u;
    // The tolerance is the per-frame parameters word 1 (float32 bits).
    if (bitcast<f32>(state[stateOffset + 2u]) <= bitcast<f32>(parameters[parametersOffset + 1u])) {
      state[stateOffset] = 1u;
      ${iterationGate.disarmWGSL}
    }
    state[stateOffset + 2u] = 0u;`
          })
        );
      }
    };
    // Scores one restart in two stages (rule 14): each chunk workgroup writes a partial sum and
    // count of its rows' squared distances (every lane's strided sum carries its exact rounding
    // error, the lanes merge in a fixed binary tree), then one workgroup folds the chunk partials in a
    // fixed order and (with restarts) keeps the best: it copies this restart's centers and state when
    // its inertia is strictly lower than the best so far (restart 0, read from the cursor, always
    // wins the empty comparison).
    const scorePartialSums = view('score-partial-sums', 'float32', chunkCount);
    const scorePartialCounts = view('score-partial-counts', 'uint32', chunkCount);
    const addScore = (scoreId: string) => {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${scoreId}-partial`,
          operation: OPERATION,
          variant: 'score-partial',
          bindings: [
            {name: 'squaredDistances', view: finalDistances!, type: 'f32', access: 'read'},
            {name: 'partialSums', view: scorePartialSums, type: 'f32', access: 'read_write'},
            {name: 'partialCounts', view: scorePartialCounts, type: 'u32', access: 'read_write'}
          ],
          workgroupSize: REDUCTION_WORKGROUP_SIZE,
          invocationCount: chunkCount * REDUCTION_WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `${sharedWGSL}
${ADDITION_ROUNDING_ERROR_WGSL}
const ROWS: u32 = ${rows}u;
const CHUNK_SIZE: u32 = ${chunkSize}u;
var<workgroup> laneSums: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;
var<workgroup> laneCounts: array<u32, ${REDUCTION_WORKGROUP_SIZE}>;`,
          body: `let lane = localInvocationIndex;
  let chunk = index / ${REDUCTION_WORKGROUP_SIZE}u;
  let begin = chunk * CHUNK_SIZE;
  let end = min(begin + CHUNK_SIZE, ROWS);
  var sum = 0.0;
  // Rounding error of every addition, read off the integer significands. (The float Kahan form
  // (next - sum) - value is algebraically zero and a fast-math shader compiler removes it.)
  var roundingError = 0.0;
  var count = 0u;
  for (var row = begin + lane; row < end; row += ${REDUCTION_WORKGROUP_SIZE}u) {
    let value = squaredDistances[squaredDistancesOffset + row];
    if (isFiniteFloat(value)) {
      let next = sum + value;
      roundingError += getAdditionRoundingError(sum, value, next);
      sum = next;
      count++;
    }
  }
  laneSums[lane] = sum + roundingError;
  laneCounts[lane] = count;
  workgroupBarrier();
  for (var stride = ${REDUCTION_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      laneSums[lane] += laneSums[lane + stride];
      laneCounts[lane] += laneCounts[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    partialSums[partialSumsOffset + chunk] = laneSums[0];
    partialCounts[partialCountsOffset + chunk] = laneCounts[0];
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: scoreId,
          operation: OPERATION,
          variant: multiStart ? 'restart-score' : 'inertia',
          bindings: [
            {name: 'partialSums', view: scorePartialSums, type: 'f32', access: 'read'},
            {name: 'partialCounts', view: scorePartialCounts, type: 'u32', access: 'read'},
            {name: 'inertia', view: bestInertia, type: 'f32', access: 'read_write'},
            ...(multiStart
              ? [
                  {
                    name: 'centers',
                    view: workCenters,
                    type: 'f32' as const,
                    access: 'read' as const
                  },
                  {
                    name: 'bestCenters',
                    view: bestCenters!,
                    type: 'f32' as const,
                    access: 'read_write' as const
                  },
                  {name: 'state', view: state, type: 'u32' as const, access: 'read' as const},
                  {
                    name: 'bestState',
                    view: bestState!,
                    type: 'u32' as const,
                    access: 'read_write' as const
                  },
                  {
                    name: 'seedCursor',
                    view: seedCursor,
                    type: 'u32' as const,
                    access: 'read' as const
                  }
                ]
              : [])
          ],
          workgroupSize: REDUCTION_WORKGROUP_SIZE,
          invocationCount: REDUCTION_WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `${sharedWGSL}
const CHUNK_COUNT: u32 = ${chunkCount}u;
var<workgroup> laneSums: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;
var<workgroup> laneCounts: array<u32, ${REDUCTION_WORKGROUP_SIZE}>;
var<workgroup> improves: u32;`,
          // CHUNK_COUNT <= 256, so each lane folds at most one partial before the fixed tree.
          body: `let lane = localInvocationIndex;
  var sum = 0.0;
  var count = 0u;
  for (var chunk = lane; chunk < CHUNK_COUNT; chunk += ${REDUCTION_WORKGROUP_SIZE}u) {
    sum += partialSums[partialSumsOffset + chunk];
    count += partialCounts[partialCountsOffset + chunk];
  }
  laneSums[lane] = sum;
  laneCounts[lane] = count;
  workgroupBarrier();
  for (var stride = ${REDUCTION_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      laneSums[lane] += laneSums[lane + stride];
      laneCounts[lane] += laneCounts[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    let total = select(getQuietNaN(0u), laneSums[0], laneCounts[0] > 0u);${
      multiStart
        ? `
    let previous = inertia[inertiaOffset];
    let firstRestart = seedCursor[seedCursorOffset] == 0u;
    // Strictly lower wins; a finite score also replaces a NaN one.
    let better = firstRestart || total < previous || (!isFiniteFloat(previous) && isFiniteFloat(total));
    improves = select(0u, 1u, better);
    if (better) {
      inertia[inertiaOffset] = total;
      bestState[bestStateOffset] = state[stateOffset];
      bestState[bestStateOffset + 1u] = state[stateOffset + 1u];
    }`
        : `
    inertia[inertiaOffset] = total;`
    }
  }${
    multiStart
      ? `
  if (workgroupUniformLoad(&improves) != 0u) {
    for (var slot = lane; slot < ${centerSlotCount}u; slot += ${REDUCTION_WORKGROUP_SIZE}u) {
      bestCenters[bestCentersOffset + slot] = centers[centersOffset + slot];
    }
  }`
      : ''
  }`
        })
      );
    };
    for (let restart = 0; restart < restartCount; restart++) {
      const restartId = multiStart ? `${id}-restart-${restart}` : id;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${restartId}-begin`,
          operation: OPERATION,
          variant: 'restart-begin',
          bindings: [
            {name: 'state', view: state, type: 'u32', access: 'read_write'},
            {name: 'seedCursor', view: seedCursor, type: 'u32', access: 'read_write'},
            iterationGate.binding
          ],
          invocationCount: 1,
          // Clears the convergence state, advances the restart cursor (prepare leaves it at
          // 0xffffffff), resets the seeding round and re-arms the Lloyd iteration gate.
          body: `state[stateOffset] = 0u;
  state[stateOffset + 1u] = 0u;
  state[stateOffset + 2u] = 0u;
  seedCursor[seedCursorOffset] = seedCursor[seedCursorOffset] + 1u;
  seedCursor[seedCursorOffset + 1u] = 0xffffffffu;
  ${iterationGate.armWGSL}`
        })
      );
      addInitialization(restartId);
      addIterations(restartId);
      if (multiStart) {
        // Final assignment of this restart against its own centers, then score and keep the best.
        nodes.push(assignNode(`${restartId}-score-assign`, true, workCenters, finalDistances));
        addScore(`${restartId}-score`);
      }
    }
    // A centered run keeps relative centers and their origins for the final assignment and
    // publishes the absolute centers (each cluster's origin added back) to the caller's view.
    const relativeCenters = multiStart ? bestCenters! : workCenters;
    if (multiStart || centering) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-publish-centers`,
          operation: OPERATION,
          variant: centering ? 'publish-centers-absolute' : 'publish-centers',
          bindings: [
            {name: 'bestCenters', view: relativeCenters, type: 'f32', access: 'read'},
            {name: 'centers', view: centers, type: 'f32', access: 'read_write'}
          ],
          invocationCount: 2 * clusterCount,
          body: `centers[centersOffset + index] = bestCenters[bestCentersOffset + index]${
            centering ? ` + bestCenters[bestCentersOffset + ${2 * clusterCount}u + index]` : ''
          };`
        })
      );
    }
    nodes.push(
      assignNode(`${id}-final-assign`, true, centering ? relativeCenters : centers, finalDistances),
      ...new GPUGroupAggregation({
        id: `${id}-final-sizes`,
        keys: labels,
        output: sizes
      }).getCommandNodes(graph)
    );
    if (output.inertia && !multiStart) {
      addScore(`${id}-inertia`);
    }
    // State word 0 is the converged flag and word 1 the iterations used.
    for (const [name, view, word] of [
      ['iteration-count', output.status?.iterationCount, 1],
      ['converged', output.status?.converged, 0]
    ] as const) {
      if (view) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-publish-${name}`,
            operation: OPERATION,
            variant: 'publish-state-word',
            bindings: [
              {name: 'state', view: multiStart ? bestState! : state, type: 'u32', access: 'read'},
              {name: 'stateWord', view, type: 'u32', access: 'read_write'}
            ],
            invocationCount: 1,
            declarations: `const WORD: u32 = ${word}u;`,
            body: `stateWord[stateWordOffset] = state[stateOffset + WORD];`
          })
        );
      }
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish-iteration-limit`,
        operation: OPERATION,
        variant: 'publish-iteration-limit',
        bindings: [
          {name: 'state', view: multiStart ? bestState! : state, type: 'u32', access: 'read'},
          {
            name: 'limit',
            view: output.status!.iterationLimitReached!,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: 1,
        body: `limit[limitOffset] = select(0u, 1u, state[stateOffset] == 0u);`
      })
    );
    return nodes;
  }
}

/** Returns every caller-owned output view of a k-means contributor. */
function getOutputs(output: GPUSpatialKMeansOutput): (GraphDataView | undefined)[] {
  return [
    output.labels,
    output.centers,
    output.sizes,
    output.squaredDistances,
    output.inertia,
    output.status?.iterationCount,
    output.status?.converged,
    output.status?.iterationLimitReached
  ];
}
