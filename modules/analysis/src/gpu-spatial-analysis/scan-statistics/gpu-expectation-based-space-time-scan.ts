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
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {POISSON_OBSERVED_TERM_WGSL} from '../../analysis-core/kernels/wgsl-poisson-observed-term';
import {
  getGPUSphereRadius,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import type {GPUInferenceMetadata} from '../../analysis-core/contracts/index';
import {PHILOX_PURPOSES} from '../../analysis-core/kernels/philox-random';
import {PERMUTATION_RANDOM_WGSL} from '../../analysis-core/kernels/permutation-random';
import {
  getSeparationWGSL,
  type InterpolationSeparation
} from '../spatial-interpolation/separation-wgsl';
import {createScanNearestZonesKernelNode} from './scan-statistic-nearest-zones';
import {EXPECTATION_BASED_SCAN_POISSON_WGSL} from './expectation-based-scan-poisson-wgsl';

const OPERATION = 'GPUExpectationBasedSpaceTimeScan';
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
} as const);
const SCAN_STREAM = PHILOX_PURPOSES.expectationBasedScan.id;
/** Centers reduced per workgroup of the evaluate kernels. */
const EVALUATE_WORKGROUP_SIZE = 256;
/** Largest batch count table, in words (a 128 MiB binding). */
const MAXIMUM_TABLE_WORDS = 2 ** 25;
/** Largest replicates per batch. */
const MAXIMUM_BATCH_REPLICATES = 4096;
/** Largest unrolled batch count; each batch adds three graph nodes. */
const MAXIMUM_BATCHES = 1024;
/** Words of the state view: best center, exceedance count, next batch base, running batch base. */
const STATE_WORDS = 4;

/** Words of the `parameters` view: seed low and high, replicate count, window length limit. */
export const GPU_EXPECTATION_BASED_SCAN_PARAMETER_LENGTH = 4;
/** Largest `maximumNeighbors`: the sort and prefix scan of a window run in one invocation. */
export const GPU_EXPECTATION_BASED_SCAN_MAXIMUM_NEIGHBORS = 64;
/** Largest `maximumWindowLength`: per window length, every neighborhood is sorted once. */
export const GPU_EXPECTATION_BASED_SCAN_MAXIMUM_WINDOW_LENGTH = 32;
/** Largest `maximumReplicates`, as `GPU_SPATIAL_SCAN_STATISTIC_MAXIMUM_PERMUTATIONS`. */
export const GPU_EXPECTATION_BASED_SCAN_MAXIMUM_REPLICATES = 99999;
/** Rows of `clusterStatistics`: LLR, observed cases, expected cases, p-value. */
export const GPU_EXPECTATION_BASED_SCAN_STATISTIC_WORDS = 4;
/** Rows of `summary`: found, member count, window length, center zone, replicate count. */
export const GPU_EXPECTATION_BASED_SCAN_SUMMARY_LENGTH = 5;
/** Sentinel of unused `clusterZones` rows. */
export const GPU_EXPECTATION_BASED_SCAN_NO_ZONE = 0xffffffff;

/** Indices into `clusterStatistics`. */
export const GPU_EXPECTATION_BASED_SCAN_CLUSTER = {
  logLikelihoodRatio: 0,
  observedCases: 1,
  expectedCases: 2,
  pValue: 3
} as const;

/** Indices into `summary`. */
export const GPU_EXPECTATION_BASED_SCAN_SUMMARY = {
  found: 0,
  memberCount: 1,
  windowLength: 2,
  centerZone: 3,
  replicateCount: 4
} as const;

/** Per-frame settings of {@link GPUExpectationBasedSpaceTimeScan}. */
export type GPUExpectationBasedScanSettings = {
  /** Integer in `[0, 2^53)`. */
  seed: number;
  /** Monte Carlo replicates, at most the contributor's `maximumReplicates`. */
  replicateCount: number;
  /** Longest window in buckets, at most the compile-time capacity. Defaults to the capacity. */
  maximumWindowLength?: number;
};

/** Packs {@link GPUExpectationBasedScanSettings} into `[seedLow, seedHigh, replicateCount, maximumWindowLength]`. */
export function getGPUExpectationBasedScanParameterValues(
  settings: GPUExpectationBasedScanSettings
): Uint32Array {
  const {seed, replicateCount} = settings;
  validateRequirement(
    'expectation-based scan',
    Number.isSafeInteger(seed) && seed >= 0,
    'seed must be a non-negative safe integer'
  );
  validateCapacity('expectation-based scan', 'replicateCount', replicateCount, 1, 2 ** 31 - 1);
  const values = new Uint32Array(GPU_EXPECTATION_BASED_SCAN_PARAMETER_LENGTH);
  values[0] = seed % 2 ** 32;
  values[1] = Math.floor(seed / 2 ** 32);
  values[2] = replicateCount;
  values[3] =
    settings.maximumWindowLength === undefined
      ? 0
      : Math.max(1, Math.floor(settings.maximumWindowLength));
  return values;
}

/** Caller-owned outputs of {@link GPUExpectationBasedSpaceTimeScan}. */
export type GPUExpectationBasedSpaceTimeScanOutput = {
  /** `maximumNeighbors` uint32 rows: the zones of the most likely cluster, `GPU_EXPECTATION_BASED_SCAN_NO_ZONE` afterwards. */
  clusterZones: GraphDataView<'uint32'>;
  /** Four float32 rows, `GPU_EXPECTATION_BASED_SCAN_CLUSTER`. All zero when no cluster has a positive score. */
  clusterStatistics: GraphDataView<'float32'>;
  /** `maximumReplicates + 1` float32 rows: element 0 the observed maximum score, then one per replicate. */
  statistics: GraphDataView<'float32'>;
  /** Five uint32 rows, `GPU_EXPECTATION_BASED_SCAN_SUMMARY`. */
  summary: GraphDataView<'uint32'>;
  /** Optional per-zone float32: the best observed score over windows centered at the zone. */
  zoneStatistics?: GraphDataView<'float32'>;
};

/** Properties for {@link GPUExpectationBasedSpaceTimeScan}. */
export type GPUExpectationBasedSpaceTimeScanProps = {
  /** Prefix for generated node IDs. Defaults to `'expectation-based-scan'`. */
  id?: string;
  /** Finite zone centroids in the spatial context coordinates, one row per zone. */
  positions: GraphDataView<'float32x2'>;
  /** As `GPUSpatialScanStatisticProps.spatialContext`: omitted is planar. */
  spatialContext?: GPUSpatialContext;
  /** Observed counts `zones * timeBuckets` uint32 rows `zone * timeBuckets + bucket`. */
  cases: GraphDataView<'uint32'>;
  /**
   * Expected counts `zones * timeBuckets` float32 rows in the same layout: the baseline, for
   * example a moving average of earlier buckets. A cell with a non-positive baseline is excluded.
   */
  baseline: GraphDataView<'float32'>;
  /** Compile-time time buckets per zone. */
  timeBuckets: number;
  /** Compile-time size of the neighborhood of each center (the center and its nearest zones), at most 64. */
  maximumNeighbors: number;
  /** Compile-time longest window in buckets, ending at the last bucket, at most `min(timeBuckets, 32)`. */
  maximumWindowLength: number;
  /** Compile-time upper bound of the per-frame replicate count, at most 99999. */
  maximumReplicates: number;
  /** Per-frame parameters, `getGPUExpectationBasedScanParameterValues`. */
  parameters: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUExpectationBasedSpaceTimeScanOutput;
  /**
   * Lowers the replicates per batch, for tests of batch independence.
   * @internal
   */
  batchReplicates?: number;
};

/**
 * Expectation-based space-time scan with Fast Subset Scan (Neill 2012), in the prospective
 * form of surveillance: every time window ends at the last bucket.
 *
 * Score: for a set `S` of zones and the last `w` buckets, `C` observed and `B` expected counts
 * over `S x w`, the expectation-based Poisson statistic
 * `F = C ln(C / B) + B - C` when `C > B`, else 0 (Neill et al. 2005). The score compares the
 * counts with the supplied baseline, not with the other zones or buckets, so a uniform rise over
 * every zone is a cluster. The permutation model of `GPUSpatialScanStatistic` conditions on the zone
 * and bucket margins and cannot see that.
 *
 * Search: for each center and window length `w`, the neighborhood is the center and its
 * `maximumNeighbors - 1` nearest zones (the kNN table of the scan, separation-key order). By the
 * linear-time subset scanning property the best subset of a neighborhood is a prefix of the zones
 * sorted by `C_i / B_i`, so an exact search over all subsets of the neighborhood is one fixed
 * sorting network and one prefix scan. The subset need not contain the center. A zone with a
 * non-positive baseline over the window is excluded. Work is `(replicates + 1) * zones * maximumWindowLength * k log^2 k`.
 *
 * Inference: Monte Carlo replicates draw independent Poisson counts from the baseline with
 * Philox per (replicate, cell); each replicate repeats the same search, and the p-value is
 * `(1 + #{replicate maxima >= observed}) / (P + 1)`. Replicate draws depend only on the seed, so
 * results do not depend on the batch size. No adjustment for earlier analyses is applied;
 * convert with `getGPUScanRecurrenceInterval`. Secondary clusters are not computed.
 *
 * Compile-time: the zone count, `timeBuckets`, `maximumNeighbors`, `maximumWindowLength`,
 * `maximumReplicates` and `spatialContext`. Per-frame: the contents of `positions`, `cases`,
 * `baseline` and `parameters`.
 */
export class GPUExpectationBasedSpaceTimeScan implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUExpectationBasedSpaceTimeScanProps;
  /** Resolved coordinate space, metric and length unit. */
  readonly spatialContext: GPUSpatialContext;
  /** Number of zones. */
  readonly zones: number;
  /** Replicates per batch, derived from the count table budget. */
  readonly batchReplicates: number;
  /** Unrolled batches. */
  readonly batchCount: number;
  /** Command nodes: three per batch plus the fixed prologue and epilogue. */
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUExpectationBasedSpaceTimeScanProps) {
    const id = props.id ?? 'expectation-based-scan';
    this.id = id;
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    const {timeBuckets, maximumNeighbors, maximumWindowLength, maximumReplicates, parameters} =
      props;
    const {output} = props;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    this.zones = props.positions.length;
    validateCapacity(id, 'zones', this.zones);
    validateCapacity(id, 'timeBuckets', timeBuckets);
    validateCapacity(
      id,
      'maximumNeighbors',
      maximumNeighbors,
      1,
      Math.min(GPU_EXPECTATION_BASED_SCAN_MAXIMUM_NEIGHBORS, this.zones)
    );
    validateCapacity(
      id,
      'maximumWindowLength',
      maximumWindowLength,
      1,
      Math.min(GPU_EXPECTATION_BASED_SCAN_MAXIMUM_WINDOW_LENGTH, timeBuckets)
    );
    validateCapacity(
      id,
      'maximumReplicates',
      maximumReplicates,
      1,
      GPU_EXPECTATION_BASED_SCAN_MAXIMUM_REPLICATES
    );
    const cells = this.zones * timeBuckets;
    validatePackedView(props.cases, ['uint32'], `${id} cases`);
    validateViewLength(id, 'cases', props.cases, cells);
    validatePackedView(props.baseline, ['float32'], `${id} baseline`);
    validateViewLength(id, 'baseline', props.baseline, cells);
    validatePackedView(parameters, ['uint32'], `${id} parameters`);
    validateRequirement(
      id,
      parameters.length >= GPU_EXPECTATION_BASED_SCAN_PARAMETER_LENGTH,
      `parameters needs ${GPU_EXPECTATION_BASED_SCAN_PARAMETER_LENGTH} rows`
    );
    const budgetReplicates =
      Math.floor(MAXIMUM_TABLE_WORDS / (this.zones * maximumWindowLength)) - 1;
    validateRequirement(
      id,
      budgetReplicates >= 1,
      `zones * maximumWindowLength must be at most ${MAXIMUM_TABLE_WORDS / 2}`
    );
    this.batchReplicates = Math.min(
      maximumReplicates,
      MAXIMUM_BATCH_REPLICATES,
      budgetReplicates,
      props.batchReplicates ?? Infinity
    );
    this.batchCount = Math.ceil(maximumReplicates / this.batchReplicates);
    validateRequirement(
      id,
      this.batchCount <= MAXIMUM_BATCHES,
      `maximumReplicates needs ${this.batchCount} batches, more than ${MAXIMUM_BATCHES}`
    );
    this.estimatedCommandNodeCount = 3 * this.batchCount + 7;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validatePackedView(output.clusterZones, ['uint32'], `${id} clusterZones`);
    validateViewLength(id, 'clusterZones', output.clusterZones, maximumNeighbors);
    validatePackedView(output.clusterStatistics, ['float32'], `${id} clusterStatistics`);
    validateViewLength(
      id,
      'clusterStatistics',
      output.clusterStatistics,
      GPU_EXPECTATION_BASED_SCAN_STATISTIC_WORDS
    );
    validatePackedView(output.statistics, ['float32'], `${id} statistics`);
    validateViewLength(id, 'statistics', output.statistics, maximumReplicates + 1);
    validatePackedView(output.summary, ['uint32'], `${id} summary`);
    validateViewLength(id, 'summary', output.summary, GPU_EXPECTATION_BASED_SCAN_SUMMARY_LENGTH);
    if (output.zoneStatistics) {
      validatePackedView(output.zoneStatistics, ['float32'], `${id} zoneStatistics`);
      validateViewLength(id, 'zoneStatistics', output.zoneStatistics, this.zones);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.clusterZones,
        output.clusterStatistics,
        output.statistics,
        output.summary,
        output.zoneStatistics
      ],
      [props.positions, props.cases, props.baseline, parameters]
    );
  }

  /** CPU description of the inference, for legends. */
  getInferenceMetadata(settings: GPUExpectationBasedScanSettings): GPUInferenceMetadata {
    getGPUExpectationBasedScanParameterValues(settings);
    return {
      alternative: 'greater',
      permutationCount: Math.min(settings.replicateCount, this.props.maximumReplicates),
      seed: settings.seed,
      multipleTesting: 'none',
      testedRowsDefinition:
        'every center, every subset of the center and its nearest zones, and every window of the last buckets; p-values are Monte Carlo maxima over all of them; no adjustment for earlier analyses'
    };
  }

  /** Returns the nearest-zones, draw, evaluate and reporting nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, zones, batchReplicates, batchCount} = this;
    const {positions, cases, baseline, parameters, output} = props;
    const {timeBuckets, maximumNeighbors, maximumWindowLength, maximumReplicates} = props;
    const {clusterZones, clusterStatistics, statistics, summary, zoneStatistics} = output;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      cases,
      baseline,
      parameters,
      clusterZones,
      clusterStatistics,
      statistics,
      summary,
      zoneStatistics
    ]);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'cases',
      zones * timeBuckets * 4,
      'reduce zones * timeBuckets'
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'neighbor table',
      zones * maximumNeighbors * 4,
      'reduce zones * maximumNeighbors'
    );
    const separation: InterpolationSeparation = {
      context: this.spatialContext,
      geographic: this.spatialContext.coordinateSpace === 'longitude-latitude',
      sphereRadius: getGPUSphereRadius(this.spatialContext)
    };
    const windowCells = zones * maximumWindowLength;
    const tableSlots = batchReplicates + 1;
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const neighbors = u32('neighbors', zones * maximumNeighbors);
    const distances = createTransientView(
      graph,
      `${id}-distances`,
      'float32',
      zones * maximumNeighbors
    );
    const sortKeys = u32('sort-keys', zones * maximumNeighbors);
    const sortZones = u32('sort-zones', zones * maximumNeighbors);
    const counts = u32('counts', tableSlots * windowCells);
    const centerBest = u32('center-best', zones * 3);
    const state = u32('state', STATE_WORDS);
    const evaluateGroups = Math.ceil(zones / EVALUATE_WORKGROUP_SIZE);
    const drawInvocations = batchReplicates * windowCells;
    const evaluateInvocations = batchReplicates * evaluateGroups * EVALUATE_WORKGROUP_SIZE;
    const batchGate = createWGSLDispatchGate<Parameters>(graph, {
      id: `${id}-batch-dispatch`,
      operation: OPERATION,
      shapes: [[drawInvocations], [evaluateInvocations, EVALUATE_WORKGROUP_SIZE]]
    });
    const common = /* wgsl */ `
const ZONES: u32 = ${zones}u;
const BUCKETS: u32 = ${timeBuckets}u;
const NEIGHBORS: u32 = ${maximumNeighbors}u;
const WINDOW_CAPACITY: u32 = ${maximumWindowLength}u;
const WINDOW_CELLS: u32 = ${windowCells}u;
const MAXIMUM_REPLICATES: u32 = ${maximumReplicates}u;
const BATCH_REPLICATES: u32 = ${batchReplicates}u;
const NO_CENTER: u32 = 0xffffffffu;
fn readReplicateCount() -> u32 {
  return clamp(parameters[parametersOffset + 2u], 1u, MAXIMUM_REPLICATES);
}
fn readWindowLimit() -> u32 {
  let requested = parameters[parametersOffset + 3u];
  return select(clamp(requested, 1u, WINDOW_CAPACITY), WINDOW_CAPACITY, requested == 0u);
}`;
    const parametersBinding = {
      name: 'parameters',
      view: parameters,
      type: 'u32' as const,
      access: 'read' as const
    };
    const search = getSearchWGSL(maximumNeighbors);
    const createEvaluateNode = (observed: boolean, batch: number) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: observed ? `${id}-evaluate` : `${id}-batch-evaluate-${batch}`,
        operation: OPERATION,
        variant: observed ? 'evaluate' : 'batch-evaluate',
        ...(observed ? {} : batchGate.getGate(1)),
        bindings: [
          parametersBinding,
          {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
          {name: 'baseline', view: baseline, type: 'f32', access: 'read'},
          {name: 'counts', view: counts, type: 'u32', access: 'read'},
          {name: 'statistics', view: statistics, type: 'atomic<u32>', access: 'read_write'},
          {
            name: 'centerBest',
            view: centerBest,
            type: 'u32',
            access: observed ? 'read_write' : 'read'
          },
          ...(observed
            ? []
            : [{name: 'state', view: state, type: 'u32' as const, access: 'read' as const}])
        ],
        invocationCount:
          (observed ? 1 : batchReplicates) * evaluateGroups * EVALUATE_WORKGROUP_SIZE,
        workgroupSize: EVALUATE_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${common}
${POISSON_OBSERVED_TERM_WGSL}
${search}
const EVALUATE_GROUPS: u32 = ${evaluateGroups}u;
const EVALUATE_SIZE: u32 = ${EVALUATE_WORKGROUP_SIZE}u;
var<workgroup> maxima: array<f32, ${EVALUATE_WORKGROUP_SIZE}>;`,
        body: `let group = index / EVALUATE_SIZE;
  let localReplicate = group / EVALUATE_GROUPS;
  let center = (group - localReplicate * EVALUATE_GROUPS) * EVALUATE_SIZE + localInvocationIndex;
  ${
    observed
      ? `let slot = 0u;
  let statisticRow = 0u;
  let isLive = index < INVOCATION_COUNT;`
      : `let replicate = state[stateOffset + 3u] + localReplicate;
  let slot = localReplicate + 1u;
  let statisticRow = replicate + 1u;
  let isLive = index < INVOCATION_COUNT && replicate < readReplicateCount();`
  }
  var best = 0.0;
  if (isLive && center < ZONES) {
    let found = searchCenter(center, slot);
    best = found.score;
    ${
      observed
        ? `centerBest[centerBestOffset + center * 3u] = bitcast<u32>(best);
    centerBest[centerBestOffset + center * 3u + 1u] = found.windowLength;
    centerBest[centerBestOffset + center * 3u + 2u] = found.memberCount;`
        : ''
    }
  }
  maxima[localInvocationIndex] = best;
  workgroupBarrier();
  for (var stride = EVALUATE_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      maxima[localInvocationIndex] = max(maxima[localInvocationIndex], maxima[localInvocationIndex + stride]);
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u && isLive && maxima[0] > 0.0) {
    atomicMax(&statistics[statisticsOffset + statisticRow], bitcast<u32>(maxima[0]));
  }`
      });

    const nodes: GPUCommandNode<Parameters>[] = [
      createScanNearestZonesKernelNode<Parameters>(graph, {
        id: `${id}-nearest`,
        operation: OPERATION,
        positions,
        neighbors,
        distances,
        sortKeys,
        sortZones,
        zones,
        listLength: maximumNeighbors,
        separationWGSL: getSeparationWGSL(separation)
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-init`,
        operation: OPERATION,
        variant: 'init',
        bindings: [
          {name: 'cases', view: cases, type: 'u32', access: 'read'},
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'},
          {name: 'statistics', view: statistics, type: 'u32', access: 'read_write'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'}
        ],
        invocationCount: Math.max(windowCells, statistics.length, STATE_WORDS),
        declarations: `const WINDOW_CELLS: u32 = ${windowCells}u;
const WINDOW_CAPACITY: u32 = ${maximumWindowLength}u;
const BUCKETS: u32 = ${timeBuckets}u;
const STATISTIC_ROWS: u32 = ${statistics.length}u;`,
        body: `if (index < WINDOW_CELLS) {
    let zone = index / WINDOW_CAPACITY;
    let windowBucket = index % WINDOW_CAPACITY;
    counts[countsOffset + index] = cases[casesOffset + zone * BUCKETS + BUCKETS - WINDOW_CAPACITY + windowBucket];
  }
  if (index < STATISTIC_ROWS) {
    statistics[statisticsOffset + index] = 0u;
  }
  if (index < ${STATE_WORDS}u) {
    state[stateOffset + index] = select(0u, 0xffffffffu, index == 0u);
  }`
      }),
      createEvaluateNode(true, 0)
    ];

    for (let batch = 0; batch < batchCount; batch++) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-batch-control-${batch}`,
          operation: OPERATION,
          variant: 'batch-control',
          bindings: [
            parametersBinding,
            {name: 'state', view: state, type: 'u32', access: 'read_write'},
            batchGate.binding
          ],
          invocationCount: 1,
          declarations: common,
          body: `let base = state[stateOffset + 2u];
  state[stateOffset + 3u] = base;
  state[stateOffset + 2u] = base + BATCH_REPLICATES;
  if (base < readReplicateCount()) {
    ${batchGate.armWGSL}
  } else {
    ${batchGate.disarmWGSL}
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-draw-${batch}`,
          operation: OPERATION,
          variant: 'draw',
          ...batchGate.getGate(0),
          bindings: [
            parametersBinding,
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {name: 'baseline', view: baseline, type: 'f32', access: 'read'},
            {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
          ],
          invocationCount: drawInvocations,
          declarations: `${PERMUTATION_RANDOM_WGSL}
${EXPECTATION_BASED_SCAN_POISSON_WGSL}
${common}
const SCAN_STREAM: u32 = ${SCAN_STREAM}u;`,
          body: `let localReplicate = index / WINDOW_CELLS;
  let replicate = state[stateOffset + 3u] + localReplicate;
  if (replicate >= readReplicateCount()) {
    return;
  }
  let windowCell = index - localReplicate * WINDOW_CELLS;
  let zone = windowCell / WINDOW_CAPACITY;
  let windowBucket = windowCell % WINDOW_CAPACITY;
  let mean = baseline[baselineOffset + zone * BUCKETS + BUCKETS - WINDOW_CAPACITY + windowBucket];
  var stream = createPhiloxStream(vec2<u32>(parameters[parametersOffset], parameters[parametersOffset + 1u]), replicate, windowCell, SCAN_STREAM);
  counts[countsOffset + (localReplicate + 1u) * WINDOW_CELLS + windowCell] = drawPhiloxPoisson(&stream, mean);`
        }),
        createEvaluateNode(false, batch)
      );
    }

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-find-center`,
        operation: OPERATION,
        variant: 'find-center',
        bindings: [
          {name: 'centerBest', view: centerBest, type: 'u32', access: 'read'},
          {name: 'statistics', view: statistics, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: zones,
        // The observed maximum already holds the bits of the largest center score, so the lowest
        // center reaching it needs no separate reduction.
        body: `let bits = centerBest[centerBestOffset + index * 3u];
  if (bits != 0u && bits == statistics[statisticsOffset]) {
    atomicMin(&state[stateOffset], index);
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-exceedance`,
        operation: OPERATION,
        variant: 'exceedance',
        bindings: [
          parametersBinding,
          {name: 'statistics', view: statistics, type: 'f32', access: 'read'},
          {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: maximumReplicates,
        declarations: common,
        body: `if (index < readReplicateCount() && statistics[statisticsOffset + index + 1u] >= statistics[statisticsOffset]) {
    atomicAdd(&state[stateOffset + 1u], 1u);
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-cluster`,
        operation: OPERATION,
        variant: 'cluster',
        bindings: [
          parametersBinding,
          {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
          {name: 'baseline', view: baseline, type: 'f32', access: 'read'},
          {name: 'counts', view: counts, type: 'u32', access: 'read'},
          {name: 'statistics', view: statistics, type: 'f32', access: 'read'},
          {name: 'centerBest', view: centerBest, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'clusterStatistics', view: clusterStatistics, type: 'f32', access: 'read_write'}
        ],
        // The sort and prefix walk of the one best neighborhood, bounded by the capacity.
        // contract-allow: single-invocation
        invocationCount: 1,
        declarations: `${common}
${POISSON_OBSERVED_TERM_WGSL}
${search}`,
        body: `let center = state[stateOffset];
  for (var row = 0u; row < ${GPU_EXPECTATION_BASED_SCAN_STATISTIC_WORDS}u; row++) {
    clusterStatistics[clusterStatisticsOffset + row] = 0.0;
  }
  if (center != NO_CENTER) {
    let windowLength = centerBest[centerBestOffset + center * 3u + 1u];
    let memberCount = centerBest[centerBestOffset + center * 3u + 2u];
    let totals = getMemberTotals(center, 0u, windowLength, memberCount);
    clusterStatistics[clusterStatisticsOffset] = statistics[statisticsOffset];
    clusterStatistics[clusterStatisticsOffset + 1u] = totals.x;
    clusterStatistics[clusterStatisticsOffset + 2u] = totals.y;
    clusterStatistics[clusterStatisticsOffset + 3u] = f32(state[stateOffset + 1u] + 1u) / f32(readReplicateCount() + 1u);
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-cluster-zones`,
        operation: OPERATION,
        variant: 'cluster-zones',
        bindings: [
          parametersBinding,
          {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
          {name: 'baseline', view: baseline, type: 'f32', access: 'read'},
          {name: 'counts', view: counts, type: 'u32', access: 'read'},
          {name: 'centerBest', view: centerBest, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'clusterZones', view: clusterZones, type: 'u32', access: 'read_write'},
          {name: 'summary', view: summary, type: 'u32', access: 'read_write'}
        ],
        // Writes the member list of the one best neighborhood, bounded by the capacity.
        // contract-allow: single-invocation
        invocationCount: 1,
        declarations: `${common}
${POISSON_OBSERVED_TERM_WGSL}
${search}`,
        body: `let center = state[stateOffset];
  var windowLength = 0u;
  var memberCount = 0u;
  var ordered: array<u32, NEIGHBORS>;
  if (center != NO_CENTER) {
    windowLength = centerBest[centerBestOffset + center * 3u + 1u];
    memberCount = centerBest[centerBestOffset + center * 3u + 2u];
    ordered = getOrderedNeighbors(center, 0u, windowLength);
  }
  for (var row = 0u; row < NEIGHBORS; row++) {
    clusterZones[clusterZonesOffset + row] = select(0xffffffffu, neighbors[neighborsOffset + center * NEIGHBORS + ordered[row]], center != NO_CENTER && row < memberCount);
  }
  summary[summaryOffset] = select(0u, 1u, center != NO_CENTER);
  summary[summaryOffset + 1u] = memberCount;
  summary[summaryOffset + 2u] = windowLength;
  summary[summaryOffset + 3u] = select(0u, center, center != NO_CENTER);
  summary[summaryOffset + 4u] = readReplicateCount();`
      })
    );
    if (zoneStatistics) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-zone-statistics`,
          operation: OPERATION,
          variant: 'zone-statistics',
          bindings: [
            {name: 'centerBest', view: centerBest, type: 'u32', access: 'read'},
            {name: 'zoneStatistics', view: zoneStatistics, type: 'u32', access: 'read_write'}
          ],
          invocationCount: zones,
          body: `zoneStatistics[zoneStatisticsOffset + index] = centerBest[centerBestOffset + index * 3u];`
        })
      );
    }
    return nodes;
  }
}

/**
 * WGSL of the Fast Subset Scan search: for one (center, count slot) the best subset of the center's
 * neighborhood over window lengths `1..readWindowLimit()`. The merge exchange sort (Knuth 5.2.2 M,
 * Batcher) orders the neighborhood by `C_i / B_i` descending with ties by neighbor slot; the
 * prefix scan then finds the best subset of each window. The sort and scan bind `neighbors`,
 * `baseline` and `counts`.
 */
function getSearchWGSL(neighborCount: number): string {
  const highestBit = neighborCount > 1 ? 1 << Math.ceil(Math.log2(neighborCount) - 1) : 0;
  return /* wgsl */ `
struct SearchResult {
  score: f32,
  windowLength: u32,
  memberCount: u32
}
struct NeighborhoodSums {
  caseSums: array<f32, ${neighborCount}>,
  baselineSums: array<f32, ${neighborCount}>
}
fn addWindowBucket(center: u32, slot: u32, windowLength: u32, sums: ptr<function, NeighborhoodSums>) {
  let bucketFromEnd = windowLength - 1u;
  let windowBucket = WINDOW_CAPACITY - 1u - bucketFromEnd;
  let bucket = BUCKETS - 1u - bucketFromEnd;
  for (var neighbor = 0u; neighbor < NEIGHBORS; neighbor++) {
    let zone = neighbors[neighborsOffset + center * NEIGHBORS + neighbor];
    (*sums).caseSums[neighbor] += f32(counts[countsOffset + slot * WINDOW_CELLS + zone * WINDOW_CAPACITY + windowBucket]);
    (*sums).baselineSums[neighbor] += baseline[baselineOffset + zone * BUCKETS + bucket];
  }
}
fn getOrder(sums: ptr<function, NeighborhoodSums>) -> array<u32, NEIGHBORS> {
  var keys: array<f32, NEIGHBORS>;
  var order: array<u32, NEIGHBORS>;
  for (var neighbor = 0u; neighbor < NEIGHBORS; neighbor++) {
    order[neighbor] = neighbor;
    keys[neighbor] = select(-1.0, (*sums).caseSums[neighbor] / (*sums).baselineSums[neighbor], (*sums).baselineSums[neighbor] > 0.0);
  }
  ${neighborCount > 1 ? getMergeExchangeSortWGSL(highestBit) : ''}
  return order;
}
fn searchCenter(center: u32, slot: u32) -> SearchResult {
  var sums: NeighborhoodSums;
  var result = SearchResult(0.0, 0u, 0u);
  let windowLimit = readWindowLimit();
  for (var windowLength = 1u; windowLength <= windowLimit; windowLength++) {
    addWindowBucket(center, slot, windowLength, &sums);
    let order = getOrder(&sums);
    var caseTotal = 0.0;
    var baselineTotal = 0.0;
    for (var member = 0u; member < NEIGHBORS; member++) {
      let neighbor = order[member];
      if (!(sums.baselineSums[neighbor] > 0.0)) {
        break;
      }
      caseTotal += sums.caseSums[neighbor];
      baselineTotal += sums.baselineSums[neighbor];
      if (caseTotal > baselineTotal) {
        let score = getObservedTerm(caseTotal, baselineTotal);
        if (score > result.score) {
          result = SearchResult(score, windowLength, member + 1u);
        }
      }
    }
  }
  return result;
}
fn getOrderedNeighbors(center: u32, slot: u32, windowLength: u32) -> array<u32, NEIGHBORS> {
  var sums: NeighborhoodSums;
  for (var bucketLength = 1u; bucketLength <= windowLength; bucketLength++) {
    addWindowBucket(center, slot, bucketLength, &sums);
  }
  return getOrder(&sums);
}
fn getMemberTotals(center: u32, slot: u32, windowLength: u32, memberCount: u32) -> vec2<f32> {
  var sums: NeighborhoodSums;
  for (var bucketLength = 1u; bucketLength <= windowLength; bucketLength++) {
    addWindowBucket(center, slot, bucketLength, &sums);
  }
  var order = getOrder(&sums);
  var caseTotal = 0.0;
  var baselineTotal = 0.0;
  for (var member = 0u; member < memberCount; member++) {
    caseTotal += sums.caseSums[order[member]];
    baselineTotal += sums.baselineSums[order[member]];
  }
  return vec2<f32>(caseTotal, baselineTotal);
}`;
}

/**
 * Merge exchange network over `keys` and `order` (descending by key, ties by slot):
 * Knuth 5.2.2 Algorithm M, a comparator between `i` and `i + d` for `i & p == r`.
 */
function getMergeExchangeSortWGSL(highestBit: number): string {
  return /* wgsl */ `var increment = ${highestBit}u;
  loop {
    var bound = ${highestBit}u;
    var mergeBit = 0u;
    var gapWidth = increment;
    loop {
      for (var first = 0u; first + gapWidth < NEIGHBORS; first++) {
        if ((first & increment) == mergeBit) {
          let left = order[first];
          let right = order[first + gapWidth];
          if (keys[left] < keys[right] || (keys[left] == keys[right] && left > right)) {
            order[first] = right;
            order[first + gapWidth] = left;
          }
        }
      }
      if (bound == increment) {
        break;
      }
      gapWidth = bound - increment;
      bound = bound / 2u;
      mergeBit = increment;
    }
    increment = increment / 2u;
    if (increment == 0u) {
      break;
    }
  }`;
}
