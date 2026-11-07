// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createSegmentSumNode} from '../../analysis-core/kernels/sorted-segment-sums';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGPUStorageBindingByteLength,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createFloat32ValuesNodes,
  type GPUFloat32OrUint32Values,
  validateFloat32OrUint32Values
} from '../spatial-weights/float32-values';
import {
  type GPUIslandPolicy,
  type GPUSpatialWeights,
  getGPUSpatialWeightsNeighborLoopWGSL,
  isGPUSpatialWeightsRowStandardized,
  resolveGPUIslandPolicy
} from '../spatial-weights/spatial-weights';
import {
  GPU_DEFAULT_INFERENCE_ALTERNATIVE,
  type GPUInferenceAlternative,
  type GPUInferenceMetadata
} from '../../analysis-core/contracts/inference';
import {getSlotGroupingNodes} from '../spatial-weights/slot-grouping';
import {GPUChunkedSpatialWeightsTranspose} from '../spatial-weights/gpu-chunked-spatial-weights-transpose';
import {
  type GPUChunkedSpatialWeights,
  type GPUSpatialWeightsChunkDispatch,
  getGPUSpatialWeightsChunkDispatches,
  getGPUSpatialWeightsOrChunksViews,
  validateGPUSpatialWeightsOrChunks,
  wrapGPUSpatialWeightsChunkBody
} from '../spatial-weights/gpu-chunked-spatial-weights';
import {SPATIAL_AUTOCORRELATION_FLOAT_WGSL} from '../spatial-autocorrelation/spatial-autocorrelation-kernels';
import {
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  type GPUGlobalSpatialStatistic
} from './global-spatial-statistics-layout';

const OPERATION = 'GPUGlobalSpatialStatistics';

/** Rows reduced by one workgroup in the first level of the deterministic column sums. */
const BLOCK_ROWS = 4096;

/** Lanes of a workgroup that reduces per-row counters before one global atomic per counter. */
const REDUCTION_WORKGROUP_SIZE = 256;

const STATISTICS: readonly GPUGlobalSpatialStatistic[] = [
  'moran',
  'geary',
  'getisOrdG',
  'bivariateMoran',
  'joinCount'
];

/**
 * Per-row partial columns of the internal column-major matrix, summed in three fixed-order
 * batches: A (inclusion and raw sums) before centering, B after the pair and column passes, C
 * (deviations from the batch-B means) last.
 */
const COLUMN = {
  included: 0,
  x: 1,
  y: 2,
  centeredSquare: 3,
  centeredCube: 4,
  centeredFourth: 5,
  centeredSquareY: 6,
  black: 7,
  rowWeight: 8,
  s1: 9,
  moranCross: 10,
  gearySum: 11,
  gCross: 12,
  bivariateCross: 13,
  binaryOutDegree: 14,
  binaryS1: 15,
  degree: 16,
  binaryDegree: 17,
  columnLag: 18,
  island: 19,
  degreeSquare: 20,
  binaryDegreeSquare: 21,
  degreeDeviation: 22,
  binaryDegreeDeviation: 23,
  columnLagDeviation: 24,
  deviationX: 25,
  deviationY: 26
} as const;
const COLUMN_COUNT = 27;
const BATCH_A = {first: 0, count: 3} as const;
const BATCH_B = {first: 3, count: 19} as const;
const BATCH_C = {first: 22, count: 3} as const;
/** Sums of the first-pass deviations from the batch-A means (second-pass mean correction). */
const BATCH_D = {first: 25, count: 2} as const;

/**
 * Rows are counted in u32 but compared and divided as f32 in the finalize kernel, and degrees and
 * counts are summed as f32 columns, which is exact only below 2^24. The local statistics use the
 * same limit.
 */
const MAXIMUM_ROWS = 2 ** 24;

/**
 * Properties for {@link GPUGlobalSpatialStatistics}.
 *
 * Per-frame (no rebuild or recompile): the contents of `weights`, `values`, `secondValues` and
 * `mask`. Compile-time: the row count, the slot capacity, `statistics` and which optional views
 * exist.
 */
export type GPUGlobalSpatialStatisticsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'global-spatial-statistics'`. */
  id?: string;
  /**
   * Square spatial weights: neighbor IDs index the same rows as `offsets`. Chunked weights run the
   * weights-reading kernels once per chunk (and the symmetry lookup per row chunk and target
   * chunk) with per-row partials accumulated in chunk order; a whole CSR is a single dispatch.
   */
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  /**
   * How chunked weights find the reverse weight `w_ji` of a slot. `'transpose'` builds one whole
   * scratch CSR of `W'` (linear dispatch count in the chunk count) and binary-searches it once per
   * row chunk; it needs the summed slot capacity to fit one storage binding. `'pairwise'` searches
   * each target chunk with one kernel per (row chunk, target chunk) pair (quadratic dispatch count,
   * no scratch). `'auto'` (default) uses `'transpose'` when `capacity * 4` bytes fit
   * `device.limits.maxStorageBufferBindingSize`. Whole weights ignore this.
   */
  symmetryStrategy?: 'auto' | 'transpose' | 'pairwise';
  /**
   * Analysis values, one per row, `float32` or `uint32` (for example the `uint32` counts of a
   * points-in-polygons aggregation; converted to f32 by one node, exact below 2^24). Join counts
   * treat nonzero values as black.
   */
  values: GPUFloat32OrUint32Values;
  /**
   * Second variable `y` of bivariate Moran's I, one per row, `float32` or `uint32`. Required for
   * `'bivariateMoran'`.
   */
  secondValues?: GPUFloat32OrUint32Values;
  /** Optional row selection: nonzero includes the row. */
  mask?: GraphDataView<'uint32'>;
  /** Statistics to report. Compile-time; other result blocks hold quiet NaN. */
  statistics: readonly GPUGlobalSpatialStatistic[];
  /**
   * Treatment of islands: included rows with no active neighbor (a slot with positive weight to
   * another included row). `'exclude'` (default) removes them from `n`, the moments, `S0`, `S1`,
   * `S2` and from the neighbors of other rows, in a single pass without cascade (a row whose only
   * neighbors were islands keeps its place in `n` and contributes no weight). `'keep'` is esda's
   * convention: islands stay in `n` and the moments. `summary.islandCount` reports the islands
   * either way. Compile-time.
   */
  islands?: GPUIslandPolicy;
  /**
   * Tail of every analytic p-value in `output.summary` (Moran, Geary, General G, bivariate Moran and
   * join counts): `'two-sided'` (default), `'greater'` (upper tail of z, for example positive
   * autocorrelation of Moran's I) or `'lesser'` (lower tail of z).
   * `'directed'` and `'folded'` are permutation conventions and throw here. Compile-time.
   */
  alternative?: GPUInferenceAlternative;
  /** Caller-owned outputs. */
  output: GPUGlobalSpatialStatisticsOutput;
};

/** Caller-owned outputs of {@link GPUGlobalSpatialStatistics}. */
export type GPUGlobalSpatialStatisticsOutput = {
  /**
   * Every statistic block, at least `GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length` float32 values,
   * laid out by `GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT`.
   */
  summary: GraphDataView<'float32'>;
  /**
   * Optional exact ordered-pair join counts `[BB2, BW2, WW2]` (twice the join counts for
   * symmetric weights) as uint32.
   */
  joinCounts?: GraphDataView<'uint32'>;
};

/**
 * Global spatial autocorrelation from a {@link GPUSpatialWeights} CSR and a value column: Moran's
 * I, Geary's C, Getis-Ord General G, bivariate Moran's I and join counts, each with its analytic
 * expectation, variance, z-score and normal p-value (`alternative`, default two-sided).
 *
 * Neighbors: a slot is a neighbor when its weight is positive, it points to another analysis row
 * and it is not the row itself (zero-weight slots are not neighbors). Weights are used as given,
 * except when `weights.information.transform === 'row'`: each row is then re-standardized over its
 * active neighbors, `w'_ij = w_ij / sum_k w_ik`, everywhere (`S0`, `S1`, `S2`, the reverse
 * weights `w'_ji` and the cross products), so masked, missing and island neighbors drop out and
 * every remaining row still sums to 1 (identical to the input up to rounding without masking).
 *
 * Islands: an included row with no active neighbor. `islands: 'exclude'` (default) removes them
 * from the analysis (`n`, moments, weight sums); `'keep'` reproduces esda, where islands count in
 * `n`. The island rule looks at out-links only; for symmetric weights it equals the in-or-out
 * rule. Parity with esda needs `islands: 'keep'`.
 *
 * Truncated weights: when `weights.status.overflow` is nonzero the weights are incomplete (rows
 * after the overflow are empty), so every statistic block and the weight sums and moments of the
 * summary are NaN. Only `count` and `islandCount` stay informative.
 *
 * Definitions (PySAL esda), over the `n` analysis rows (mask nonzero, every provided value finite
 * and, for `'exclude'`, not an island; excluded rows are neither foci nor neighbors), with
 * `z = x - mean(x)`,
 * `S0 = sum w_ij`, `S1 = 1/2 sum (w_ij + w_ji)^2`, `S2 = sum_i (w_i. + w_.i)^2`:
 * - Moran: `I = (n / S0) sum w_ij z_i z_j / sum z_i^2`, `E = -1/(n-1)`; `esda.Moran` variances
 *   `VI_norm` and `VI_rand` (with the sample kurtosis).
 * - Geary: `C = (n-1) sum w_ij (x_i - x_j)^2 / (2 S0 sum z_i^2)`, `E = 1`; `esda.Geary`
 *   `VC_norm` and `VC_rand`.
 * - General G: `G = sum_{i != j} w_ij x_i x_j / sum_{i != j} x_i x_j`, `E = S0 / (n(n-1))`; the
 *   randomization variance equals `esda.G.VG` but is evaluated in centered form,
 *   `Var(x'Wx) = mean^2 Var(d'z) + 2 mean Cov(d'z, z'Wz) + Var(z'Wz)` with `d_i = w_i. + w_.i`,
 *   which avoids the catastrophic cancellation of `E[G^2] - E[G]^2` in f32. G needs non-negative
 *   ratio-scale data: if any analysis value is negative the whole G block is NaN. Both inference
 *   field triples report this same analytical variance and normal approximation (esda names it
 *   `VG` / `z_norm`); no permutations run here. esda.G binarizes weights: matching it requires
 *   binary input weights and `islands: 'keep'`. Other weights give the weighted generalization.
 * - Bivariate Moran (`esda.Moran_BV`, which permutes `y`):
 *   `I_BV = (n / S0) sum_ij w_ij zx_i zy_j / sqrt(sum zx^2 sum zy^2)`, which equals esda's value for
 *   row-standardized weights (`S0 = n`). Under randomization of `y`, `E = 0` and the exact
 *   variance is `(n / S0)^2 sum_j (b_j - mean b)^2 / ((n-1) sum zx^2)` with the column lag
 *   `b_j = sum_i w_ij zx_i`. esda itself reports only permutation inference.
 * - Join counts: weights binarized (`w > 0`), black `x != 0`; `BB = 1/2 sum b_ij x_i x_j`,
 *   `BW = 1/2 sum b_ij (x_i - x_j)^2`, `WW` likewise, as exact integer ordered-pair counts halved.
 *   Cliff-Ord randomization (nonfree sampling) moments of BB and BW, evaluated with the same
 *   centered quadratic-form identities (BB is General G's numerator of the indicator, BW is
 *   Geary's numerator). Asymmetric weights are handled through `S1` and `S2`, and counts may be
 *   half-integers.
 *
 * Variances are NaN when `n < 4` (the randomization formulas divide by `n - 3`), `S0 = 0`, or the
 * values are constant (General G has variance zero for positive constant data); z and p are NaN
 * where the variance is not positive. Constant values center
 * to exactly 0, so Moran's I and Geary's C are NaN for them (not a ratio of rounding residues); the
 * expectations of I and C are NaN below two rows. With no analysis row every statistic is NaN and
 * the join counts are 0.
 *
 * Precision: `n` is counted in u32 (atomic integer adds), the mean gets a second-pass correction
 * `mean += sum(x - mean) / n`, and deviations are formed from the corrected mean. Row counts must
 * be below 2^24 (degrees and counts are f32 columns, exact only there), the same limit as the
 * local statistics.
 *
 * Algorithm: one gather pass per row over its CSR slots (binary search of `w_ji` in row `j`), a
 * deterministic transpose (stable sort of slots by neighbor) for column sums, and per-row partials
 * summed by fixed-order two-level workgroup trees. Join counts use u32 atomics. No float atomics,
 * so results are bitwise reproducible. Integer-valued sums (counts, degrees) are exact below 2^24.
 */
export class GPUGlobalSpatialStatistics implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGlobalSpatialStatisticsProps;
  /** Resolved island policy. */
  readonly islands: GPUIslandPolicy;
  /** Resolved tail of the analytic p-values . */
  readonly alternative: 'two-sided' | 'greater' | 'lesser';

  constructor(props: GPUGlobalSpatialStatisticsProps) {
    const id = props.id ?? 'global-spatial-statistics';
    this.id = id;
    this.props = props;
    this.islands = resolveGPUIslandPolicy(id, props.islands);
    this.alternative = resolveAnalyticAlternative(id, props.alternative);
    const rows = validateGPUSpatialWeightsOrChunks(id, props.weights);
    if (rows >= MAXIMUM_ROWS) {
      throw new Error(`${id} weights must hold fewer than 2^24 rows`);
    }
    if (props.weights.status?.overflow) {
      validatePackedView(
        props.weights.status.overflow,
        ['uint32'],
        `${id} weights.status.overflow`
      );
    }
    if (validateFloat32OrUint32Values(props.values, `${id} values`) !== rows) {
      throw new Error(`${id} values needs one row per weights row`);
    }
    if (
      props.secondValues &&
      validateFloat32OrUint32Values(props.secondValues, `${id} secondValues`) !== rows
    ) {
      throw new Error(`${id} secondValues needs one row per weights row`);
    }
    if (props.mask) {
      validatePackedView(props.mask, ['uint32'], `${id} mask`);
      if (props.mask.length !== rows) {
        throw new Error(`${id} mask needs one row per weights row`);
      }
    }
    if (props.statistics.length === 0) {
      throw new Error(`${id} statistics must name at least one statistic`);
    }
    for (const statistic of props.statistics) {
      if (!STATISTICS.includes(statistic)) {
        throw new Error(`${id} unknown statistic ${statistic}`);
      }
    }
    if (props.statistics.includes('bivariateMoran') && !props.secondValues) {
      throw new Error(`${id} bivariateMoran requires secondValues`);
    }
    const {output} = props;
    validatePackedView(output.summary, ['float32'], `${id} output.summary`);
    if (output.summary.length < GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length) {
      throw new Error(
        `${id} output.summary needs ${GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length} values`
      );
    }
    if (output.joinCounts) {
      validatePackedView(output.joinCounts, ['uint32'], `${id} output.joinCounts`);
      if (output.joinCounts.length < 3) {
        throw new Error(`${id} output.joinCounts needs 3 values`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.summary, output.joinCounts],
      [
        ...getWeightsViews(props.weights),
        props.weights.status?.overflow,
        props.values,
        props.secondValues,
        props.mask
      ]
    );
  }

  /**
   * Describes how the analytic p-values in `output.summary` were produced, enough to build a
   * legend. Global statistics report p-values, never multiple-testing adjusted ones. Takes no
   * settings: the statistics have no packed per-frame parameters (the tail is compile time).
   */
  getInferenceMetadata(): GPUInferenceMetadata {
    return {
      alternative: this.alternative,
      permutationCount: 0,
      multipleTesting: 'none',
      testedRowsDefinition: `included rows with finite values${
        this.islands === 'exclude' ? ', islands excluded' : ' (islands kept, as esda)'
      }; one test per statistic`
    };
  }

  /** Returns the global-statistics nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {weights, mask, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getWeightsViews(weights),
      weights.status?.overflow,
      props.values,
      props.secondValues,
      mask,
      output.summary,
      output.joinCounts
    ]);
    // uint32 values (counts) are converted to f32 once; float32 values are used as given.
    const valueConversion = createFloat32ValuesNodes(graph, `${id}-values`, props.values);
    const secondValueConversion = props.secondValues
      ? createFloat32ValuesNodes(graph, `${id}-second-values`, props.secondValues)
      : undefined;
    const values = valueConversion.view;
    const secondValues = secondValueConversion?.view;
    const dispatches = getGPUSpatialWeightsChunkDispatches(weights);
    const isWhole = dispatches[0].isWhole;
    const rows = dispatches.reduce((total, dispatch) => total + dispatch.rowCount, 0);
    const capacity = dispatches.reduce(
      (total, dispatch) => total + dispatch.weights.neighbors.length,
      0
    );
    const overflow = weights.status?.overflow;
    const standardized = isGPUSpatialWeightsRowStandardized(weights);
    const keepIslands = this.islands === 'keep';
    const blockCount = Math.ceil(rows / BLOCK_ROWS);
    // Constants of a kernel; chunk kernels see their own CAPACITY, CHUNK_ROWS and WEIGHTS_ROW_START.
    const getConstantsWGSL = (
      dispatch?: GPUSpatialWeightsChunkDispatch
    ) => `const ROWS: u32 = ${rows}u;
const CAPACITY: u32 = ${dispatch ? dispatch.weights.neighbors.length : capacity}u;${
      dispatch && !dispatch.isWhole
        ? `\nconst CHUNK_ROWS: u32 = ${dispatch.rowCount}u;\nconst CHUNK_INDEX: u32 = ${dispatch.chunkIndex}u;\n${dispatch.constantsWGSL}`
        : ''
    }
${Object.entries(COLUMN)
  .map(([name, column]) => `const COLUMN_${toConstantCase(name)}: u32 = ${column}u;`)
  .join('\n')}
fn getMatrixIndex(column: u32, row: u32) -> u32 {
  return column * ROWS + row;
}
// total / count with one remainder correction. WGSL division may be 2.5 ulp off (Metal computes
// it through a reciprocal), so 24 / 12 can miss 2; the correction makes the quotient correctly
// rounded in practice, so constant values center to exactly 0 and their statistics are NaN, not
// the ratio of rounding residues.
fn getMeanQuotient(total: f32, count: f32) -> f32 {
  let quotient = total / count;
  return quotient + fma(-quotient, count, total) / count;
}
${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}`;
    const constantsWGSL = getConstantsWGSL();
    // Neighbors are the shared active slots (positive weight, another row, analysis row).
    const analysisNeighbor = 'isFiniteFloat(centered[centeredOffset + neighbor * 2u])';

    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'matrix',
      COLUMN_COUNT * rows * 4,
      'reduce the row count'
    );
    const matrix = createTransientView(graph, `${id}-matrix`, 'float32', COLUMN_COUNT * rows);
    const centered = createTransientView(graph, `${id}-centered`, 'float32', rows * 2);
    const totalsA = createTransientView(graph, `${id}-totals-a`, 'float32', BATCH_A.count);
    const totalsB = createTransientView(graph, `${id}-totals-b`, 'float32', BATCH_B.count);
    const totalsC = createTransientView(graph, `${id}-totals-c`, 'float32', BATCH_C.count);
    const totalsD = createTransientView(graph, `${id}-totals-d`, 'float32', BATCH_D.count);
    const baseStates = createTransientView(graph, `${id}-base-states`, 'uint32', rows);
    // [n, analysis rows with a negative value, BB2, BW2, WW2], u32 so counts stay exact.
    const counters = createTransientView(graph, `${id}-counters`, 'uint32', 5);
    const rowScale = createTransientView(graph, `${id}-row-scale`, 'float32', rows);
    // Chunked weights accumulate the in-link sums of the chunks here: [weight, degree, lag] per row.
    const columnParts = isWhole
      ? undefined
      : createTransientView(graph, `${id}-column-parts`, 'float32', rows * 3);

    const optionalInputs: WGSLKernelBinding[] = [
      ...(secondValues
        ? [
            {
              name: 'secondValues',
              view: secondValues,
              type: 'f32' as const,
              access: 'read' as const
            }
          ]
        : []),
      ...(mask ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}] : [])
    ];
    // Weights-reading kernels run once per chunk (a whole CSR is a single dispatch with the
    // original WGSL). Row bodies see global rows; offsets are indexed by the chunk-local row.
    const getRowLimitWGSL = (dispatch: GPUSpatialWeightsChunkDispatch) =>
      dispatch.isWhole ? 'ROWS' : 'WEIGHTS_ROW_START + CHUNK_ROWS';
    const getLocalRowWGSL = (dispatch: GPUSpatialWeightsChunkDispatch) =>
      dispatch.isWhole ? 'index' : '(index - WEIGHTS_ROW_START)';
    const getNeighborLoopWGSL = (
      dispatch: GPUSpatialWeightsChunkDispatch,
      isIncluded: string,
      action: string
    ) =>
      dispatch.isWhole
        ? getGPUSpatialWeightsNeighborLoopWGSL(isIncluded, action)
        : getGPUSpatialWeightsNeighborLoopWGSL(isIncluded, action, 'index', 'WEIGHTS_ROW_START');
    const getWeightBindings = (dispatch: GPUSpatialWeightsChunkDispatch): WGSLKernelBinding[] => [
      {name: 'offsets', view: dispatch.weights.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: dispatch.weights.neighbors, type: 'u32', access: 'read'},
      {name: 'weights', view: dispatch.weights.weights, type: 'f32', access: 'read'}
    ];

    const islandNodes = dispatches.map(dispatch =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-islands${dispatch.idSuffix}`,
        operation: OPERATION,
        variant: 'islands',
        bindings: [
          ...getWeightBindings(dispatch),
          {name: 'baseStates', view: baseStates, type: 'u32', access: 'read'},
          {name: 'counters', view: counters, type: 'atomic<u32>', access: 'read_write'},
          {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'}
        ],
        invocationCount: dispatch.rowCount,
        workgroupSize: REDUCTION_WORKGROUP_SIZE,
        guardIndex: false,
        // Workgroup counters (zero-initialized by WGSL) keep the global counters to one atomic per
        // workgroup instead of one per row.
        declarations: `${getConstantsWGSL(dispatch)}
var<workgroup> groupIncluded: atomic<u32>;
var<workgroup> groupNegative: atomic<u32>;`,
        body: wrapGPUSpatialWeightsChunkBody(
          dispatch,
          `if (index < ${getRowLimitWGSL(dispatch)}) {
  let base = baseStates[baseStatesOffset + index] != 0u;
  var hasNeighbor = false;
  if (base) {
    ${getNeighborLoopWGSL(dispatch, 'baseStates[baseStatesOffset + neighbor] != 0u', 'hasNeighbor = true;')}
  }
  let island = base && !hasNeighbor;
  let included = base && ${keepIslands ? 'true' : '!island'};
  if (included) {
    atomicAdd(&groupIncluded, 1u);
    if (matrix[matrixOffset + getMatrixIndex(COLUMN_X, index)] < 0.0) {
      atomicAdd(&groupNegative, 1u);
    }
  } else {
    // Missing rows and excluded islands leave the sums.
    matrix[matrixOffset + getMatrixIndex(COLUMN_X, index)] = 0.0;
    matrix[matrixOffset + getMatrixIndex(COLUMN_Y, index)] = 0.0;
  }
  matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, index)] = select(0.0, 1.0, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_ISLAND, index)] = select(0.0, 1.0, island);
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    atomicAdd(&counters[countersOffset], atomicLoad(&groupIncluded));
    atomicAdd(&counters[countersOffset + 1u], atomicLoad(&groupNegative));
  }`
        )
      })
    );

    // Row re-standardization: the reciprocal sum of the active weights of every analysis row.
    const rowScaleNodes = standardized
      ? dispatches.map(dispatch =>
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-row-scale${dispatch.idSuffix}`,
            operation: OPERATION,
            variant: 'row-scale',
            bindings: [
              ...getWeightBindings(dispatch),
              {name: 'centered', view: centered, type: 'f32', access: 'read'},
              {name: 'rowScale', view: rowScale, type: 'f32', access: 'read_write'}
            ],
            invocationCount: dispatch.rowCount,
            declarations: getConstantsWGSL(dispatch),
            // A row with no active weight keeps scale 0 (it contributes nothing).
            body: wrapGPUSpatialWeightsChunkBody(
              dispatch,
              `var sum = 0.0;
  if (isFiniteFloat(centered[centeredOffset + index * 2u])) {
    ${getNeighborLoopWGSL(dispatch, analysisNeighbor, 'sum += weight;')}
  }
  rowScale[rowScaleOffset + index] = select(0.0, 1.0 / sum, sum > 0.0);`
            )
          })
        )
      : [];

    // Deterministic transpose per chunk: the source row of every slot, then a stable sort by
    // neighbor, grouping the slots of the chunk by column.
    const transposes = dispatches.map(dispatch => {
      const chunkCapacity = dispatch.weights.neighbors.length;
      const slotRows = createTransientView(
        graph,
        `${id}-slot-rows${dispatch.idSuffix}`,
        'uint32',
        chunkCapacity
      );
      const transposeKeys = createTransientView(
        graph,
        `${id}-transpose-keys${dispatch.idSuffix}`,
        'uint32',
        chunkCapacity
      );
      const slotIds = createTransientView(
        graph,
        `${id}-slot-ids${dispatch.idSuffix}`,
        'uint32',
        chunkCapacity
      );
      const columnOffsets = createTransientView(
        graph,
        `${id}-column-offsets${dispatch.idSuffix}`,
        'uint32',
        rows + 1
      );
      // Deterministic transpose for the column sums: stable sort of the slots by neighbor, then
      // per-column offsets by binary search of the sorted keys (no atomic count, no scan).
      const grouping = getSlotGroupingNodes<Parameters>(graph, {
        id: `${id}-transpose${dispatch.idSuffix}`,
        operation: OPERATION,
        slotKeys: transposeKeys,
        slotIndices: slotIds,
        columns: rows,
        columnOffsets
      });
      const localRow = getLocalRowWGSL(dispatch);
      const rowEnd = dispatch.isWhole ? 'ROWS' : 'CHUNK_ROWS';
      const nodes = [
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-slot-rows${dispatch.idSuffix}`,
          operation: OPERATION,
          variant: 'slot-rows',
          bindings: [
            {name: 'offsets', view: dispatch.weights.offsets, type: 'u32', access: 'read'},
            {name: 'centered', view: centered, type: 'f32', access: 'read'},
            {name: 'slotRows', view: slotRows, type: 'u32', access: 'read_write'}
          ],
          invocationCount: dispatch.rowCount,
          declarations: getConstantsWGSL(dispatch),
          // Every used slot is written (excluded rows write the sentinel ROWS), so no clear pass.
          body: wrapGPUSpatialWeightsChunkBody(
            dispatch,
            `let source = select(ROWS, index, isFiniteFloat(centered[centeredOffset + index * 2u]));
  let begin = min(offsets[offsetsOffset + ${localRow}], CAPACITY);
  let end = min(offsets[offsetsOffset + ${localRow} + 1u], CAPACITY);
  for (var slot = begin; slot < end; slot++) {
    slotRows[slotRowsOffset + slot] = source;
  }`
          )
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-transpose-keys${dispatch.idSuffix}`,
          operation: OPERATION,
          variant: 'transpose-keys',
          bindings: [
            ...getWeightBindings(dispatch),
            {name: 'slotRows', view: slotRows, type: 'u32', access: 'read'},
            {name: 'centered', view: centered, type: 'f32', access: 'read'},
            {name: 'transposeKeys', view: transposeKeys, type: 'u32', access: 'read_write'},
            {name: 'slotIds', view: slotIds, type: 'u32', access: 'read_write'}
          ],
          invocationCount: chunkCapacity,
          declarations: getConstantsWGSL(dispatch),
          body: `let used = index < min(offsets[offsetsOffset + ${rowEnd}], CAPACITY);
  let source = select(ROWS, slotRows[slotRowsOffset + index], used);
  let neighbor = neighbors[neighborsOffset + index];
  var key = ROWS;
  if (source < ROWS && neighbor < ROWS && neighbor != source &&
      weights[weightsOffset + index] > 0.0 &&
      isFiniteFloat(centered[centeredOffset + neighbor * 2u])) {
    key = neighbor;
  }
  transposeKeys[transposeKeysOffset + index] = key;
  slotIds[slotIdsOffset + index] = index;`
        }),
        ...grouping.nodes
      ];
      return {nodes, slotRows, columnOffsets, sortedSlots: grouping.sortedSlots};
    });
    const transposeNodes = transposes.flatMap(transpose => transpose.nodes);

    // Pair kernels. Whole weights: one kernel per row with the reverse-slot lookup inline. Chunked
    // weights: one kernel per row chunk for the sums that need only the row's own slots, then one
    // reverse-slot kernel per (row chunk, target chunk) pair that searches the target chunk's CSR;
    // targets ascend so each row visits its slots in slot order and the sums continue.
    const getPairsBody = (dispatch: GPUSpatialWeightsChunkDispatch): string => {
      const isChunked = !dispatch.isWhole;
      return wrapGPUSpatialWeightsChunkBody(
        dispatch,
        `var blackBlack = 0u;
  var blackWhite = 0u;
  var whiteWhite = 0u;
  if (index < ${getRowLimitWGSL(dispatch)}) {
  let centeredX = centered[centeredOffset + index * 2u];
  var rowWeight = 0.0;
  var s1 = 0.0;
  var lag = 0.0;
  var gearySum = 0.0;
  var rawLag = 0.0;
  var lagY = 0.0;
  var outDegree = 0u;
  var binaryS1 = 0u;
  let x = values[valuesOffset + index];
  if (isFiniteFloat(centeredX)) {
    let black = x != 0.0;
    let scale = getRowScale(index);
    ${getNeighborLoopWGSL(
      dispatch,
      analysisNeighbor,
      `let neighborX = centered[centeredOffset + neighbor * 2u];
      let scaled = weight * scale;
      ${
        isChunked
          ? ''
          : `let reverse = findWeight(neighbor, index);
      s1 += select(scaled * scaled, 0.5 * (scaled + reverse) * (scaled + reverse), reverse >= 0.0);
      binaryS1 += select(1u, 2u, reverse > 0.0);`
      }
      rowWeight += scaled;
      lag += scaled * neighborX;
      let difference = centeredX - neighborX;
      gearySum += scaled * difference * difference;
      rawLag += scaled * values[valuesOffset + neighbor];
      lagY += scaled * centered[centeredOffset + neighbor * 2u + 1u];
      outDegree++;
      let neighborBlack = values[valuesOffset + neighbor] != 0.0;
      if (black && neighborBlack) {
        blackBlack++;
      } else if (black != neighborBlack) {
        blackWhite++;
      } else {
        whiteWhite++;
      }`
    )}
  }
  let included = isFiniteFloat(centeredX);
  matrix[matrixOffset + getMatrixIndex(COLUMN_ROW_WEIGHT, index)] = rowWeight;
  ${isChunked ? '' : 'matrix[matrixOffset + getMatrixIndex(COLUMN_S1, index)] = s1;'}
  matrix[matrixOffset + getMatrixIndex(COLUMN_MORAN_CROSS, index)] = select(0.0, centeredX * lag, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_GEARY_SUM, index)] = gearySum;
  matrix[matrixOffset + getMatrixIndex(COLUMN_G_CROSS, index)] = select(0.0, x * rawLag, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BIVARIATE_CROSS, index)] = select(0.0, centeredX * lagY, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_OUT_DEGREE, index)] = f32(outDegree);
  ${isChunked ? '' : 'matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_S1, index)] = f32(binaryS1);'}
  }
  atomicAdd(&groupBlackBlack, blackBlack);
  atomicAdd(&groupBlackWhite, blackWhite);
  atomicAdd(&groupWhiteWhite, whiteWhite);
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    atomicAdd(&counters[countersOffset + 2u], atomicLoad(&groupBlackBlack));
    atomicAdd(&counters[countersOffset + 3u], atomicLoad(&groupBlackWhite));
    atomicAdd(&counters[countersOffset + 4u], atomicLoad(&groupWhiteWhite));
  }`
      );
    };
    const rowScaleBinding: WGSLKernelBinding[] = standardized
      ? [{name: 'rowScale', view: rowScale, type: 'f32', access: 'read'}]
      : [];
    const pairNodes: GPUCommandNode<Parameters>[] = [];
    // Chunked weights: one whole scratch CSR of W' makes the reverse lookup one kernel per chunk.
    const useTranspose =
      !isWhole &&
      (props.symmetryStrategy === 'transpose' ||
        (props.symmetryStrategy !== 'pairwise' &&
          capacity * 4 <= graph.device.limits.maxStorageBufferBindingSize));
    const symmetryTranspose = useTranspose
      ? {
          offsets: createTransientView(graph, `${id}-symmetry-offsets`, 'uint32', rows + 1),
          neighbors: createTransientView(graph, `${id}-symmetry-neighbors`, 'uint32', capacity),
          weights: createTransientView(graph, `${id}-symmetry-weights`, 'float32', capacity)
        }
      : undefined;
    if (symmetryTranspose && !isWhole) {
      pairNodes.push(
        ...new GPUChunkedSpatialWeightsTranspose({
          id: `${id}-symmetry`,
          weights: weights as GPUChunkedSpatialWeights,
          output: symmetryTranspose
        }).getCommandNodes(graph)
      );
    }
    for (const rowDispatch of dispatches) {
      const isChunked = !rowDispatch.isWhole;
      pairNodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-pairs${rowDispatch.idSuffix}`,
          operation: OPERATION,
          variant: isChunked ? 'pairs-chunk' : 'pairs',
          bindings: [
            ...getWeightBindings(rowDispatch),
            {name: 'values', view: values, type: 'f32', access: 'read'},
            ...rowScaleBinding,
            {name: 'centered', view: centered, type: 'f32', access: 'read'},
            {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'},
            {name: 'counters', view: counters, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: rowDispatch.rowCount,
          workgroupSize: REDUCTION_WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `${getConstantsWGSL(rowDispatch)}
var<workgroup> groupBlackBlack: atomic<u32>;
var<workgroup> groupBlackWhite: atomic<u32>;
var<workgroup> groupWhiteWhite: atomic<u32>;
${getRowScaleWGSL(standardized)}
${
  isChunked
    ? ''
    : `
fn getRowEnd(row: u32) -> u32 {
  return min(offsets[offsetsOffset + row + 1u], CAPACITY);
}

// Effective weight w'_row,column by binary search of the ascending neighbor IDs of row, or -1 when
// the slot is absent or not active (zero weight).
fn findWeight(row: u32, column: u32) -> f32 {
  var low = min(offsets[offsetsOffset + row], CAPACITY);
  let end = getRowEnd(row);
  var high = end;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (neighbors[neighborsOffset + middle] < column) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < end && neighbors[neighborsOffset + low] == column) {
    let weight = weights[weightsOffset + low];
    return select(-1.0, weight * getRowScale(row), weight > 0.0);
  }
  return -1.0;
}`
}`,
          body: getPairsBody(rowDispatch)
        })
      );
      if (!isChunked) {
        continue;
      }
      if (useTranspose && symmetryTranspose) {
        pairNodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-reverse${rowDispatch.idSuffix}-transpose`,
            operation: OPERATION,
            variant: 'reverse-transpose',
            bindings: [
              ...getWeightBindings(rowDispatch),
              {name: 'targetOffsets', view: symmetryTranspose.offsets, type: 'u32', access: 'read'},
              {
                name: 'targetNeighbors',
                view: symmetryTranspose.neighbors,
                type: 'u32',
                access: 'read'
              },
              {
                name: 'targetWeights',
                view: symmetryTranspose.weights,
                type: 'f32',
                access: 'read'
              },
              ...rowScaleBinding,
              {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'}
            ],
            invocationCount: rowDispatch.rowCount,
            declarations: `${getConstantsWGSL(rowDispatch)}
const TARGET_CAPACITY: u32 = ${symmetryTranspose.neighbors.length}u;
${getRowScaleWGSL(standardized)}

// Effective reverse weight: the row of the transpose lists its source rows ascending. -1 when
// the slot is absent or not active (zero weight).
fn findReverseWeight(row: u32, column: u32) -> f32 {
  var low = min(targetOffsets[targetOffsetsOffset + row], TARGET_CAPACITY);
  let end = min(targetOffsets[targetOffsetsOffset + row + 1u], TARGET_CAPACITY);
  var high = end;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (targetNeighbors[targetNeighborsOffset + middle] < column) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < end && targetNeighbors[targetNeighborsOffset + low] == column) {
    let weight = targetWeights[targetWeightsOffset + low];
    return select(-1.0, weight * getRowScale(column), weight > 0.0);
  }
  return -1.0;
}`,
            body: wrapGPUSpatialWeightsChunkBody(
              rowDispatch,
              `var s1 = 0.0;
  var binaryS1 = 0u;
  if (matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, index)] != 0.0) {
    let scale = getRowScale(index);
    ${getNeighborLoopWGSL(
      rowDispatch,
      'matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, neighbor)] != 0.0',
      `let scaled = weight * scale;
      let reverse = findReverseWeight(index, neighbor);
      s1 += select(scaled * scaled, 0.5 * (scaled + reverse) * (scaled + reverse), reverse >= 0.0);
      binaryS1 += select(1u, 2u, reverse > 0.0);`
    )}
  }
  matrix[matrixOffset + getMatrixIndex(COLUMN_S1, index)] = s1;
  matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_S1, index)] = f32(binaryS1);`
            )
          })
        );
        continue;
      }
      for (const targetDispatch of dispatches) {
        pairNodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-reverse${rowDispatch.idSuffix}-target${targetDispatch.chunkIndex}`,
            operation: OPERATION,
            variant: 'reverse-chunk',
            bindings: [
              ...getWeightBindings(rowDispatch),
              {
                name: 'targetOffsets',
                view: targetDispatch.weights.offsets,
                type: 'u32',
                access: 'read'
              },
              {
                name: 'targetNeighbors',
                view: targetDispatch.weights.neighbors,
                type: 'u32',
                access: 'read'
              },
              {
                name: 'targetWeights',
                view: targetDispatch.weights.weights,
                type: 'f32',
                access: 'read'
              },
              ...rowScaleBinding,
              {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'}
            ],
            invocationCount: rowDispatch.rowCount,
            declarations: `${getConstantsWGSL(rowDispatch)}
const TARGET_START: u32 = ${targetDispatch.rowStart}u;
const TARGET_END: u32 = ${targetDispatch.rowStart + targetDispatch.rowCount}u;
const TARGET_CAPACITY: u32 = ${targetDispatch.weights.neighbors.length}u;
const TARGET_INDEX: u32 = ${targetDispatch.chunkIndex}u;
${getRowScaleWGSL(standardized)}

// Effective weight w'_row,column by binary search of the ascending neighbor IDs of a target chunk
// row, or -1 when the slot is absent or not active (zero weight).
fn findWeight(row: u32, column: u32) -> f32 {
  let local = row - TARGET_START;
  var low = min(targetOffsets[targetOffsetsOffset + local], TARGET_CAPACITY);
  let end = min(targetOffsets[targetOffsetsOffset + local + 1u], TARGET_CAPACITY);
  var high = end;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (targetNeighbors[targetNeighborsOffset + middle] < column) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < end && targetNeighbors[targetNeighborsOffset + low] == column) {
    let weight = targetWeights[targetWeightsOffset + low];
    return select(-1.0, weight * getRowScale(row), weight > 0.0);
  }
  return -1.0;
}`,
            // The first target starts the row's symmetry sums; later targets continue them.
            body: wrapGPUSpatialWeightsChunkBody(
              rowDispatch,
              `var s1 = 0.0;
  var binaryS1 = 0u;
  if (TARGET_INDEX > 0u) {
    s1 = matrix[matrixOffset + getMatrixIndex(COLUMN_S1, index)];
    binaryS1 = u32(matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_S1, index)]);
  }
  if (matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, index)] != 0.0) {
    let scale = getRowScale(index);
    ${getNeighborLoopWGSL(
      rowDispatch,
      'matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, neighbor)] != 0.0 && neighbor >= TARGET_START && neighbor < TARGET_END',
      `let scaled = weight * scale;
      let reverse = findWeight(neighbor, index);
      s1 += select(scaled * scaled, 0.5 * (scaled + reverse) * (scaled + reverse), reverse >= 0.0);
      binaryS1 += select(1u, 2u, reverse > 0.0);`
    )}
  }
  matrix[matrixOffset + getMatrixIndex(COLUMN_S1, index)] = s1;
  matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_S1, index)] = f32(binaryS1);`
            )
          })
        );
      }
    }

    // In-link sums per column. Whole weights finish the degree columns in one kernel; chunked
    // weights add each chunk's slots to `columnParts` in chunk order, then finish once.
    const columnNodes: GPUCommandNode<Parameters>[] = dispatches.map((dispatch, chunkIndex) => {
      const transpose = transposes[chunkIndex];
      const bindings: WGSLKernelBinding[] = [
        {name: 'columnOffsets', view: transpose.columnOffsets, type: 'u32', access: 'read'},
        {name: 'sortedSlots', view: transpose.sortedSlots, type: 'u32', access: 'read'},
        {name: 'slotRows', view: transpose.slotRows, type: 'u32', access: 'read'},
        {name: 'weights', view: dispatch.weights.weights, type: 'f32', access: 'read'},
        ...(standardized
          ? [{name: 'rowScale', view: rowScale, type: 'f32' as const, access: 'read' as const}]
          : []),
        {name: 'centered', view: centered, type: 'f32', access: 'read'}
      ];
      if (columnParts) {
        const previous = (offset: number) =>
          chunkIndex > 0 ? `columnParts[columnPartsOffset + index * 3u + ${offset}u]` : '0.0';
        return createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-columns${dispatch.idSuffix}`,
          operation: OPERATION,
          variant: 'columns-chunk',
          bindings: [
            ...bindings,
            {name: 'columnParts', view: columnParts, type: 'f32', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: `${getConstantsWGSL(dispatch)}
${getRowScaleWGSL(standardized)}`,
          // Slots of one column are in ascending slot order (stable sort), so sums are fixed-order.
          body: `let begin = columnOffsets[columnOffsetsOffset + index];
  let end = columnOffsets[columnOffsetsOffset + index + 1u];
  var columnWeight = ${previous(0)};
  var inDegree = ${previous(1)};
  var columnLag = ${previous(2)};
  for (var position = begin; position < end; position++) {
    let slot = sortedSlots[sortedSlotsOffset + position];
    let source = slotRows[slotRowsOffset + slot];
    let weight = weights[weightsOffset + slot] * getRowScale(source);
    columnWeight += weight;
    inDegree += 1.0;
    columnLag += weight * centered[centeredOffset + source * 2u];
  }
  columnParts[columnPartsOffset + index * 3u] = columnWeight;
  columnParts[columnPartsOffset + index * 3u + 1u] = inDegree;
  columnParts[columnPartsOffset + index * 3u + 2u] = columnLag;`
        });
      }
      return createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-columns`,
        operation: OPERATION,
        variant: 'columns',
        bindings: [...bindings, {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'}],
        invocationCount: rows,
        declarations: `${constantsWGSL}
${getRowScaleWGSL(standardized)}`,
        // Slots of one column are in ascending slot order (stable sort), so sums are fixed-order.
        body: `let included = isFiniteFloat(centered[centeredOffset + index * 2u]);
  let begin = columnOffsets[columnOffsetsOffset + index];
  let end = columnOffsets[columnOffsetsOffset + index + 1u];
  var columnWeight = 0.0;
  var inDegree = 0u;
  var columnLag = 0.0;
  for (var position = begin; position < end; position++) {
    let slot = sortedSlots[sortedSlotsOffset + position];
    // Only active slots are keyed into columns, so every weight here is positive.
    let source = slotRows[slotRowsOffset + slot];
    let weight = weights[weightsOffset + slot] * getRowScale(source);
    columnWeight += weight;
    inDegree++;
    columnLag += weight * centered[centeredOffset + source * 2u];
  }
  ${COLUMN_DEGREE_WGSL}`
      });
    });
    if (columnParts) {
      columnNodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-columns-finalize`,
          operation: OPERATION,
          variant: 'columns-finalize',
          bindings: [
            {name: 'centered', view: centered, type: 'f32', access: 'read'},
            {name: 'columnParts', view: columnParts, type: 'f32', access: 'read'},
            {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: constantsWGSL,
          body: `let included = isFiniteFloat(centered[centeredOffset + index * 2u]);
  let columnWeight = columnParts[columnPartsOffset + index * 3u];
  let inDegree = u32(columnParts[columnPartsOffset + index * 3u + 1u]);
  let columnLag = columnParts[columnPartsOffset + index * 3u + 2u];
  ${COLUMN_DEGREE_WGSL}`
        })
      );
    }

    const nodes: GPUCommandNode<Parameters>[] = [
      ...valueConversion.nodes,
      ...(secondValueConversion?.nodes ?? []),
      createFillNode<Parameters>(graph, {
        id: `${id}-counters-clear`,
        operation: OPERATION,
        view: counters,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-include`,
        operation: OPERATION,
        variant: 'include',
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          ...optionalInputs,
          {name: 'baseStates', view: baseStates, type: 'u32', access: 'read_write'},
          {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: constantsWGSL,
        body: `let x = values[valuesOffset + index];
  let y = ${secondValues ? 'secondValues[secondValuesOffset + index]' : '0.0'};
  let included = ${mask ? 'mask[maskOffset + index] != 0u &&' : ''} isFiniteFloat(x) && isFiniteFloat(y);
  baseStates[baseStatesOffset + index] = select(0u, 1u, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_X, index)] = select(0.0, x, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_Y, index)] = select(0.0, y, included);`
      }),
      // Islands: included rows with no active slot to another included row. Rows left in the
      // analysis (all included rows for 'keep', the non-islands for 'exclude') are counted in
      // u32; the others leave the sums.
      ...islandNodes,
      ...getColumnSumNodes<Parameters>(graph, {
        id: `${id}-sum-a`,
        matrix,
        rows,
        blockCount,
        ...BATCH_A,
        totals: totalsA
      }),
      // Second-pass mean: first-pass deviations from the batch-A mean, whose sum corrects it.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-deviate`,
        operation: OPERATION,
        variant: 'deviate',
        bindings: [
          {name: 'totalsA', view: totalsA, type: 'f32', access: 'read'},
          {name: 'counters', view: counters, type: 'u32', access: 'read'},
          {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: constantsWGSL,
        body: `let included = matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, index)] != 0.0;
  let count = f32(counters[countersOffset]);
  let x = matrix[matrixOffset + getMatrixIndex(COLUMN_X, index)];
  let y = matrix[matrixOffset + getMatrixIndex(COLUMN_Y, index)];
  matrix[matrixOffset + getMatrixIndex(COLUMN_DEVIATION_X, index)] =
    select(0.0, x - getMeanQuotient(totalsA[totalsAOffset + 1u], count), included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_DEVIATION_Y, index)] =
    select(0.0, y - getMeanQuotient(totalsA[totalsAOffset + 2u], count), included);`
      }),
      ...getColumnSumNodes<Parameters>(graph, {
        id: `${id}-sum-d`,
        matrix,
        rows,
        blockCount,
        ...BATCH_D,
        totals: totalsD
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-center`,
        operation: OPERATION,
        variant: 'center',
        bindings: [
          {name: 'totalsD', view: totalsD, type: 'f32', access: 'read'},
          {name: 'counters', view: counters, type: 'u32', access: 'read'},
          {name: 'centered', view: centered, type: 'f32', access: 'read_write'},
          {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: constantsWGSL,
        // Stored first-pass deviation minus the mean of the first-pass deviations. The deviation is
        // read back from memory, not recomputed as (x - mean1) - mean2: the shader compiler may
        // reassociate that to x - (mean1 + mean2), which rounds the corrected mean to the f32 grid
        // of x and defeats the second pass.
        body: `let included = matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, index)] != 0.0;
  let count = f32(counters[countersOffset]);
  let x = matrix[matrixOffset + getMatrixIndex(COLUMN_X, index)];
  let centeredX = matrix[matrixOffset + getMatrixIndex(COLUMN_DEVIATION_X, index)] - getMeanQuotient(totalsD[totalsDOffset], count);
  let centeredY = ${secondValues ? 'matrix[matrixOffset + getMatrixIndex(COLUMN_DEVIATION_Y, index)] - getMeanQuotient(totalsD[totalsDOffset + 1u], count)' : '0.0'};
  let square = centeredX * centeredX;
  centered[centeredOffset + index * 2u] = select(getQuietNaN(index), centeredX, included);
  centered[centeredOffset + index * 2u + 1u] = select(getQuietNaN(index), centeredY, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_CENTERED_SQUARE, index)] = select(0.0, square, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_CENTERED_CUBE, index)] = select(0.0, square * centeredX, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_CENTERED_FOURTH, index)] = select(0.0, square * square, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_CENTERED_SQUARE_Y, index)] = select(0.0, centeredY * centeredY, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BLACK, index)] =
    select(0.0, 1.0, included && x != 0.0);`
      }),
      ...rowScaleNodes,
      ...transposeNodes,
      ...pairNodes,
      ...columnNodes,
      ...getColumnSumNodes<Parameters>(graph, {
        id: `${id}-sum-b`,
        matrix,
        rows,
        blockCount,
        ...BATCH_B,
        totals: totalsB
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-deviations`,
        operation: OPERATION,
        variant: 'deviations',
        bindings: [
          {name: 'totalsB', view: totalsB, type: 'f32', access: 'read'},
          {name: 'counters', view: counters, type: 'u32', access: 'read'},
          {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `${constantsWGSL}
const BATCH_B_FIRST: u32 = ${BATCH_B.first}u;
fn readTotalB(column: u32) -> f32 {
  return totalsB[totalsBOffset + column - BATCH_B_FIRST];
}
fn getSquaredDeviation(column: u32, row: u32, count: f32) -> f32 {
  let deviation = matrix[matrixOffset + getMatrixIndex(column, row)] - readTotalB(column) / count;
  return deviation * deviation;
}`,
        body: `let included = matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, index)] != 0.0;
  let count = f32(counters[countersOffset]);
  matrix[matrixOffset + getMatrixIndex(COLUMN_DEGREE_DEVIATION, index)] =
    select(0.0, getSquaredDeviation(COLUMN_DEGREE, index, count), included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_DEGREE_DEVIATION, index)] =
    select(0.0, getSquaredDeviation(COLUMN_BINARY_DEGREE, index, count), included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_COLUMN_LAG_DEVIATION, index)] =
    select(0.0, getSquaredDeviation(COLUMN_COLUMN_LAG, index, count), included);`
      }),
      ...getColumnSumNodes<Parameters>(graph, {
        id: `${id}-sum-c`,
        matrix,
        rows,
        blockCount,
        ...BATCH_C,
        totals: totalsC
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        variant: 'finalize',
        bindings: [
          {name: 'totalsA', view: totalsA, type: 'f32', access: 'read'},
          {name: 'totalsB', view: totalsB, type: 'f32', access: 'read'},
          {name: 'totalsC', view: totalsC, type: 'f32', access: 'read'},
          {name: 'totalsD', view: totalsD, type: 'f32', access: 'read'},
          {name: 'counters', view: counters, type: 'u32', access: 'read'},
          {name: 'results', view: output.summary, type: 'f32', access: 'read_write'},
          ...(output.joinCounts
            ? [
                {
                  name: 'joinCounts',
                  view: output.joinCounts,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : []),
          ...(overflow
            ? [{name: 'overflow', view: overflow, type: 'u32' as const, access: 'read' as const}]
            : [])
        ],
        invocationCount: 1,
        declarations: `${constantsWGSL}
${GLOBAL_STATISTICS_FINALIZE_WGSL}
const BATCH_B_FIRST: u32 = ${BATCH_B.first}u;
const BATCH_C_FIRST: u32 = ${BATCH_C.first}u;
const BATCH_D_FIRST: u32 = ${BATCH_D.first}u;
fn readTotal(column: u32) -> f32 {
  if (column < BATCH_B_FIRST) {
    return totalsA[totalsAOffset + column];
  }
  if (column < BATCH_C_FIRST) {
    return totalsB[totalsBOffset + column - BATCH_B_FIRST];
  }
  if (column < BATCH_D_FIRST) {
    return totalsC[totalsCOffset + column - BATCH_C_FIRST];
  }
  return totalsD[totalsDOffset + column - BATCH_D_FIRST];
}
${getPValueWGSL(this.alternative)}
fn writeBlock(offset: u32, block: array<f32, 8>, enabled: bool) {
  for (var field = 0u; field < 8u; field++) {
    results[resultsOffset + offset + field] = select(getQuietNaN(field), block[field], enabled);
  }
}`,
        body: getFinalizeBody(props, Boolean(overflow))
      })
    ];
    return nodes;
  }
}

/** Validates the analytic tail . */
function resolveAnalyticAlternative(
  id: string,
  alternative: GPUInferenceAlternative | undefined
): 'two-sided' | 'greater' | 'lesser' {
  const resolved = alternative ?? GPU_DEFAULT_INFERENCE_ALTERNATIVE;
  switch (resolved) {
    case 'two-sided':
    case 'greater':
      return resolved;
    case 'lesser':
      return resolved;
    default:
      throw new Error(`${id} alternative must be 'two-sided', 'greater' or 'lesser'`);
  }
}

/** Converts a camelCase column name to CAPITAL_CASE for WGSL constants. */
function toConstantCase(name: string): string {
  return name.replace(/([A-Z])/g, '_$1').toUpperCase();
}

/**
 * Sums `count` matrix columns starting at `first` into `totals` with two fixed-order workgroup
 * tree levels: one segment per (column, block of rows), then one per column over its blocks.
 */
function getColumnSumNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    matrix: GraphDataView<'float32'>;
    rows: number;
    blockCount: number;
    first: number;
    count: number;
    totals: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, rows, blockCount, first, count} = props;
  const segmentCount = count * blockCount;
  const blockOffsets = createTransientView(
    graph,
    `${id}-block-offsets`,
    'uint32',
    segmentCount + 1
  );
  const columnOffsets = createTransientView(graph, `${id}-column-offsets`, 'uint32', count + 1);
  const partials = createTransientView(graph, `${id}-partials`, 'float32', segmentCount);
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-offsets`,
      operation: OPERATION,
      variant: 'sum-offsets',
      bindings: [
        {name: 'blockOffsets', view: blockOffsets, type: 'u32', access: 'read_write'},
        {name: 'columnOffsets', view: columnOffsets, type: 'u32', access: 'read_write'}
      ],
      invocationCount: segmentCount + 1,
      declarations: `const ROWS: u32 = ${rows}u;
const BLOCK_ROWS: u32 = ${BLOCK_ROWS}u;
const BLOCK_COUNT: u32 = ${blockCount}u;
const FIRST_COLUMN: u32 = ${first}u;
const COLUMN_COUNT: u32 = ${count}u;`,
      body: `let column = index / BLOCK_COUNT;
  let block = index % BLOCK_COUNT;
  blockOffsets[blockOffsetsOffset + index] = (FIRST_COLUMN + column) * ROWS + min(block * BLOCK_ROWS, ROWS);
  if (index <= COLUMN_COUNT) {
    columnOffsets[columnOffsetsOffset + index] = index * BLOCK_COUNT;
  }`
    }),
    createSegmentSumNode<Parameters>(graph, {
      id: `${id}-blocks`,
      operation: OPERATION,
      segmentCount,
      input: props.matrix,
      segmentOffsets: blockOffsets,
      output: partials
    }),
    createSegmentSumNode<Parameters>(graph, {
      id: `${id}-columns`,
      operation: OPERATION,
      segmentCount: count,
      input: partials,
      segmentOffsets: columnOffsets,
      output: props.totals
    })
  ];
}

/** Moment formulas evaluated by the single-invocation finalize kernel. */
const GLOBAL_STATISTICS_FINALIZE_WGSL = /* wgsl */ `
fn getFinite(value: f32) -> f32 {
  return select(getQuietNaN(0u), value, isFiniteFloat(value));
}

// Division that never relies on the implementation-defined result of x / 0.
fn divide(numerator: f32, denominator: f32) -> f32 {
  return select(getQuietNaN(0u), getFinite(numerator / denominator), denominator != 0.0 && isFiniteFloat(denominator));
}

fn getZScore(statistic: f32, expected: f32, variance: f32) -> f32 {
  let z = (statistic - expected) / sqrt(variance);
  return select(getQuietNaN(0u), z, variance > 0.0 && isFiniteFloat(z));
}

// esda.Moran: (expected, VI_norm, VI_rand).
fn getMoranMoments(n: f32, s0: f32, s1: f32, s2: f32, kurtosis: f32) -> vec3<f32> {
  let n2 = n * n;
  let s02 = s0 * s0;
  let expected = -1.0 / (n - 1.0);
  let varianceNormality = (n2 * s1 - n * s2 + 3.0 * s02) / ((n2 - 1.0) * s02) - expected * expected;
  let a = n * ((n2 - 3.0 * n + 3.0) * s1 - n * s2 + 3.0 * s02);
  let b = kurtosis * ((n2 - n) * s1 - 2.0 * n * s2 + 6.0 * s02);
  let varianceRandomization =
    (a - b) / ((n - 1.0) * (n - 2.0) * (n - 3.0) * s02) - expected * expected;
  return vec3<f32>(expected, varianceNormality, varianceRandomization);
}

// esda.Geary: (VC_norm, VC_rand).
fn getGearyMoments(n: f32, s0: f32, s1: f32, s2: f32, kurtosis: f32) -> vec2<f32> {
  let n2 = n * n;
  let s02 = s0 * s0;
  let varianceNormality = ((2.0 * s1 + s2) * (n - 1.0) - 4.0 * s02) / (2.0 * (n + 1.0) * s02);
  let a = (n - 1.0) * s1 * (n2 - 3.0 * n + 3.0 - (n - 1.0) * kurtosis);
  let b = 0.25 * (n - 1.0) * s2 * (n2 + 3.0 * n - 6.0 - (n2 - n + 2.0) * kurtosis);
  let c = s02 * (n2 - 3.0 - (n - 1.0) * (n - 1.0) * kurtosis);
  let varianceRandomization = (a - b + c) / (n * (n - 2.0) * (n - 3.0) * s02);
  return vec2<f32>(varianceNormality, varianceRandomization);
}

// Randomization variance of x'Wx (zero diagonal) with x = mean + z, sum z = 0:
// mean^2 Var(d'z) + 2 mean Cov(d'z, z'Wz) + Var(z'Wz), d_i = w_i. + w_.i, Vd = sum (d - mean d)^2.
fn getQuadraticFormVariance(
  n: f32, s0: f32, s1: f32, s2: f32, degreeDeviation: f32, mean: f32,
  sumSquares: f32, sumCubes: f32, sumFourths: f32
) -> f32 {
  // A constant input has an invariant quadratic form under every permutation.
  if (sumSquares == 0.0) { return 0.0; }
  let kurtosis = n * sumFourths / (sumSquares * sumSquares);
  let moranVariance = getMoranMoments(n, s0, s1, s2, kurtosis).z;
  let scale = s0 * sumSquares / n;
  let linearVariance = sumSquares * degreeDeviation / (n - 1.0);
  let covariance = -sumCubes * degreeDeviation / ((n - 1.0) * (n - 2.0));
  return mean * mean * linearVariance + 2.0 * mean * covariance + moranVariance * scale * scale;
}
`;

const COLUMN_DEGREE_WGSL = `let degree = matrix[matrixOffset + getMatrixIndex(COLUMN_ROW_WEIGHT, index)] + columnWeight;
  let binaryDegree = matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_OUT_DEGREE, index)] + f32(inDegree);
  matrix[matrixOffset + getMatrixIndex(COLUMN_DEGREE, index)] = select(0.0, degree, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_DEGREE, index)] = select(0.0, binaryDegree, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_COLUMN_LAG, index)] = select(0.0, columnLag, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_DEGREE_SQUARE, index)] = select(0.0, degree * degree, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_DEGREE_SQUARE, index)] =
    select(0.0, binaryDegree * binaryDegree, included);`;

/** Offsets, neighbors and weights views of either form of weights. */
function getWeightsViews(
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights
): (GraphDataView<'uint32'> | GraphDataView<'float32'>)[] {
  return getGPUSpatialWeightsOrChunksViews(weights).filter(
    (view, index) => index % 4 !== 3 && view !== undefined
  ) as (GraphDataView<'uint32'> | GraphDataView<'float32'>)[];
}

/** Row scale of the effective weights: the reciprocal active sum, or 1 for weights used as given. */
function getRowScaleWGSL(standardized: boolean): string {
  return standardized
    ? `fn getRowScale(row: u32) -> f32 {
  return rowScale[rowScaleOffset + row];
}`
    : `fn getRowScale(row: u32) -> f32 {
  return 1.0;
}`;
}

/** Normal p-value of a z-score for the tail, NaN where z is not finite. */
function getPValueWGSL(alternative: 'two-sided' | 'greater' | 'lesser'): string {
  const SQRT_HALF = '0.70710678118654752';
  const expressions = {
    'two-sided': 'getTwoSidedPValue(z)',
    greater: `min(0.5 * erfcValue(z * ${SQRT_HALF}), 1.0)`,
    lesser: `min(0.5 * erfcValue(-z * ${SQRT_HALF}), 1.0)`
  };
  // erfcValue comes with SPATIAL_AUTOCORRELATION_FLOAT_WGSL, included by the finalize kernel.
  // f32 exp underflows beyond |z| of about 9.3; p-values are floored at the smallest normal f32.
  return `const P_VALUE_FLOOR: f32 = 1.17549435e-38;
fn getPValue(z: f32) -> f32 {
  return select(getQuietNaN(0u), max(${expressions[alternative]}, P_VALUE_FLOOR), isFiniteFloat(z));
}`;
}

/** Builds the finalize kernel body, writing only the requested blocks. */
function getFinalizeBody(props: GPUGlobalSpatialStatisticsProps, hasOverflow: boolean): string {
  // Truncated weights (overflow) make every statistic NaN; General G also needs non-negative data.
  const enabled = (statistic: GPUGlobalSpatialStatistic) =>
    props.statistics.includes(statistic)
      ? statistic === 'getisOrdG'
        ? '!truncated && !hasNegative'
        : '!truncated'
      : 'false';
  const layout = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT;
  return /* wgsl */ `let n = f32(counters[countersOffset]);
  let hasNegative = counters[countersOffset + 1u] > 0u;
  let truncated = ${hasOverflow ? 'overflow[overflowOffset] != 0u' : 'false'};
  // Mean with the second-pass correction sum(x - mean) / n.
  let mean = divide(readTotal(COLUMN_X), n) + divide(readTotal(COLUMN_DEVIATION_X), n);
  let sumSquares = readTotal(COLUMN_CENTERED_SQUARE);
  let sumCubes = readTotal(COLUMN_CENTERED_CUBE);
  let sumFourths = readTotal(COLUMN_CENTERED_FOURTH);
  let s0 = readTotal(COLUMN_ROW_WEIGHT);
  let s1 = readTotal(COLUMN_S1);
  let s2 = readTotal(COLUMN_DEGREE_SQUARE);
  let blackCount = readTotal(COLUMN_BLACK);
  let degreeDeviation = readTotal(COLUMN_DEGREE_DEVIATION);
  let kurtosis = divide(n * sumFourths, sumSquares * sumSquares);
  let valid = n >= 4.0 && s0 > 0.0 && sumSquares > 0.0;

  results[resultsOffset + ${layout.summary}u] = n;
  results[resultsOffset + ${layout.summary + 1}u] = select(s0, getQuietNaN(1u), truncated);
  results[resultsOffset + ${layout.summary + 2}u] = select(s1, getQuietNaN(2u), truncated);
  results[resultsOffset + ${layout.summary + 3}u] = select(s2, getQuietNaN(3u), truncated);
  results[resultsOffset + ${layout.summary + 4}u] = select(getFinite(mean), getQuietNaN(4u), truncated);
  results[resultsOffset + ${layout.summary + 5}u] = select(divide(sumSquares, n), getQuietNaN(5u), truncated);
  results[resultsOffset + ${layout.summary + 6}u] = blackCount;
  results[resultsOffset + ${layout.summary + 7}u] = readTotal(COLUMN_ISLAND);

  // Moran's I.
  {
    let statistic = divide(n * readTotal(COLUMN_MORAN_CROSS), s0 * sumSquares);
    let moments = getMoranMoments(n, s0, s1, s2, kurtosis);
    let varianceNormality = select(getQuietNaN(1u), getFinite(moments.y), valid);
    let varianceRandomization = select(getQuietNaN(1u), getFinite(moments.z), valid);
    let zNormality = getZScore(statistic, moments.x, varianceNormality);
    let zRandomization = getZScore(statistic, moments.x, varianceRandomization);
    // E[I] = -1 / (n - 1) is undefined below two rows (n = 0 would give 1).
    let expected = select(getQuietNaN(1u), getFinite(moments.x), n >= 2.0);
    writeBlock(${layout.moran}u, array<f32, 8>(statistic, expected, varianceNormality,
      zNormality, getPValue(zNormality), varianceRandomization, zRandomization,
      getPValue(zRandomization)), ${enabled('moran')});
  }
  // Geary's C.
  {
    let statistic = divide((n - 1.0) * readTotal(COLUMN_GEARY_SUM), 2.0 * s0 * sumSquares);
    let moments = getGearyMoments(n, s0, s1, s2, kurtosis);
    let varianceNormality = select(getQuietNaN(1u), getFinite(moments.x), valid);
    let varianceRandomization = select(getQuietNaN(1u), getFinite(moments.y), valid);
    let zNormality = getZScore(statistic, 1.0, varianceNormality);
    let zRandomization = getZScore(statistic, 1.0, varianceRandomization);
    writeBlock(${layout.geary}u, array<f32, 8>(statistic, select(getQuietNaN(1u), 1.0, n >= 2.0),
      varianceNormality, zNormality,
      getPValue(zNormality), varianceRandomization, zRandomization, getPValue(zRandomization)),
      ${enabled('geary')});
  }
  // Getis-Ord General G.
  {
    let denominator = n * (n - 1.0) * mean * mean - sumSquares;
    let statistic = divide(readTotal(COLUMN_G_CROSS), denominator);
    let expected = divide(s0, n * (n - 1.0));
    let variance = select(getQuietNaN(1u), getFinite(getQuadraticFormVariance(n, s0, s1, s2,
      degreeDeviation, mean, sumSquares, sumCubes, sumFourths) / (denominator * denominator)),
      n >= 4.0 && s0 > 0.0 && denominator > 0.0);
    let z = getZScore(statistic, expected, variance);
    // esda calls these VG / z_norm; retain the randomization aliases for compatibility.
    writeBlock(${layout.getisOrdG}u, array<f32, 8>(statistic, expected, variance,
      z, getPValue(z), variance, z, getPValue(z)), ${enabled('getisOrdG')});
  }
  // Bivariate Moran's I, randomization of y.
  {
    let sumSquaresY = readTotal(COLUMN_CENTERED_SQUARE_Y);
    let scale = divide(n, s0);
    let statistic = divide(scale * readTotal(COLUMN_BIVARIATE_CROSS), sqrt(sumSquares * sumSquaresY));
    let variance = select(getQuietNaN(1u), getFinite(scale * scale *
      readTotal(COLUMN_COLUMN_LAG_DEVIATION) / ((n - 1.0) * sumSquares)), valid);
    let z = getZScore(statistic, 0.0, variance);
    writeBlock(${layout.bivariateMoran}u, array<f32, 8>(statistic, 0.0, getQuietNaN(2u),
      getQuietNaN(2u), getQuietNaN(2u), variance, z, getPValue(z)), ${enabled('bivariateMoran')});
  }
  // Join counts over the binarized weights.
  {
    let enabled = ${enabled('joinCount')};
    let blackBlack2 = counters[countersOffset + 2u];
    let blackWhite2 = counters[countersOffset + 3u];
    let whiteWhite2 = counters[countersOffset + 4u];
    ${
      props.output.joinCounts
        ? `joinCounts[joinCountsOffset] = select(0u, blackBlack2, enabled);
    joinCounts[joinCountsOffset + 1u] = select(0u, blackWhite2, enabled);
    joinCounts[joinCountsOffset + 2u] = select(0u, whiteWhite2, enabled);`
        : ''
    }
    let b0 = readTotal(COLUMN_BINARY_OUT_DEGREE);
    let b1 = readTotal(COLUMN_BINARY_S1);
    let b2 = readTotal(COLUMN_BINARY_DEGREE_SQUARE);
    let p = divide(blackCount, n);
    let q = 1.0 - p;
    let blackSquares = n * p * q;
    let blackCubes = blackSquares * (q - p);
    let blackFourths = blackSquares * (p * p * p + q * q * q);
    let binaryValid = n >= 4.0 && b0 > 0.0;
    // A single colour makes every join count constant: zero variance.
    let oneColour = blackCount == 0.0 || blackCount == n;
    let expectedBlackBlack = getFinite(0.5 * b0 * blackCount * (blackCount - 1.0) / (n * (n - 1.0)));
    let varianceBlackBlack = select(getQuietNaN(3u), getFinite(0.25 * getQuadraticFormVariance(n,
      b0, b1, b2, readTotal(COLUMN_BINARY_DEGREE_DEVIATION), p, blackSquares, blackCubes,
      blackFourths)), binaryValid);
    let varianceBlackBlackOrZero = select(varianceBlackBlack, 0.0, binaryValid && oneColour);
    let expectedBlackWhite = getFinite(b0 * blackSquares / (n - 1.0));
    let blackKurtosis = n * blackFourths / (blackSquares * blackSquares);
    let blackWhiteScale = b0 * blackSquares / (n - 1.0);
    let varianceBlackWhite = select(getQuietNaN(3u), getFinite(getGearyMoments(n, b0, b1, b2,
      blackKurtosis).y * blackWhiteScale * blackWhiteScale), binaryValid && !oneColour);
    let varianceBlackWhiteOrZero = select(varianceBlackWhite, 0.0, binaryValid && oneColour);
    let blackBlack = 0.5 * f32(blackBlack2);
    let blackWhite = 0.5 * f32(blackWhite2);
    let zBlackBlack = getZScore(blackBlack, expectedBlackBlack, varianceBlackBlackOrZero);
    let zBlackWhite = getZScore(blackWhite, expectedBlackWhite, varianceBlackWhiteOrZero);
    let block = array<f32, 12>(blackBlack, blackWhite, 0.5 * f32(whiteWhite2), 0.5 * b0,
      expectedBlackBlack, varianceBlackBlackOrZero, zBlackBlack, getPValue(zBlackBlack),
      expectedBlackWhite, varianceBlackWhiteOrZero, zBlackWhite, getPValue(zBlackWhite));
    for (var field = 0u; field < 12u; field++) {
      results[resultsOffset + ${layout.joinCount}u + field] = select(getQuietNaN(field), block[field], enabled);
    }
  }`;
}
