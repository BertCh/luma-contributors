// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createStatisticsKernelNode as createWGSLKernelNode} from '../spatial-weights/statistics-kernel-node';
import {
  validateCapacity,
  validateFiniteNumber,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotWeightsPort} from '../spatial-weights/weights-construction-snapshot';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  getCooperativeMannKendallBody,
  getSliceStateWGSL,
  getSliceCountingWGSL,
  getSerialMannKendallBody
} from './emerging-hot-spot-trend-wgsl';
import {getTheilSenSlopeWGSL} from '../../analysis-core/kernels/wgsl-theil-sen-median';
import type {
  GPUInferenceMetadata,
  GPUMultipleTestingMethod
} from '../../analysis-core/contracts/inference';
import {
  GPUMultipleTestingAdjustment,
  GPU_MULTIPLE_TESTING_MAXIMUM_ROWS,
  resolveGPUMultipleTestingMethod
} from '../multiple-testing/index';
import {SPATIAL_AUTOCORRELATION_FLOAT_WGSL} from '../spatial-autocorrelation/spatial-autocorrelation-kernels';
import {GET_QUIET_NAN_SEEDED_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {COMPENSATED_SUM_WGSL} from '../../analysis-core/kernels/compensated-sum-wgsl';
import {getGPUSpatialWeightsLossViews} from '../spatial-weights/gpu-chunked-spatial-weights';
import {
  getGPUSpatialWeightsChunkDispatches,
  getGPUSpatialWeightsOrChunksViews,
  validateGPUSpatialWeightsOrChunks,
  type GPUChunkedSpatialWeights,
  type GPUSpatialWeights
} from '../spatial-weights/index';
import {
  getGPUEmergingHotSpotsParameterValues,
  snapshotEmergingHotSpotSettings,
  type GPUEmergingHotSpotsSettings,
  GPU_EMERGING_HOT_SPOTS_MAXIMUM_RADIUS,
  GPU_EMERGING_HOT_SPOTS_MAXIMUM_CORRECTED_TREND_SLICE_COUNT,
  GPU_EMERGING_HOT_SPOTS_MAXIMUM_SLICE_COUNT,
  GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH,
  GPU_EMERGING_HOT_SPOTS_STATISTICS_LENGTH,
  GPU_EMERGING_HOT_SPOTS_CATEGORIES
} from './emerging-hot-spot-parameters';

import {validateSpatialStatisticsBindings} from '../spatial-weights/spatial-statistics-bindings';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {createSpatialStatisticsCollectiveNode} from '../spatial-weights/spatial-statistics-collective-node';
import {SPATIAL_STATISTICS_COUNTS_WGSL} from '../spatial-weights/spatial-statistics-counts-wgsl';
import {getChiSquareSurvival} from '../spatial-regression/ordinary-least-squares-statistics';

import {
  resolveStatisticsPopulationCount,
  type SpatialStatisticsPopulationOutput
} from '../spatial-weights/spatial-statistics-population-output';

const OPERATION = 'GPUEmergingHotSpots';
/** Bins reduced by one workgroup in the first level of the moment sums. */
const BIN_BLOCK = 4096;
/** Threads of the moment-sum workgroups. */
const MOMENT_WORKGROUP_SIZE = 256;
/** Threads per workgroup of the space-time Gi* kernel; each workgroup covers whole cells. */
const GI_WORKGROUP_SIZE = 256;
const EMERGING_SUM_WGSL = /* wgsl */ `${COMPENSATED_SUM_WGSL}
fn mergeEmergingSums(total: vec2<f32>, partial: vec2<f32>) -> vec2<f32> {
  return addCompensatedValue(addCompensatedValue(total, partial.x), partial.y);
}`;
/** Threads per cell in the cooperative Mann-Kendall pass. */
const MANN_KENDALL_WORKGROUP_SIZE = 64;
/** Below this series length a single invocation is cheaper than workgroup setup and reduction. */
const MANN_KENDALL_COOPERATIVE_SLICE_COUNT = 64;
/** Two-sided normal critical z of a 95% lag filter, the Hamed-Rao default. */
const DEFAULT_TREND_CORRECTION_CRITICAL_Z = 1.959963984540054;
/** Signed bin indices in the space-time kernels must remain in i32 range. */
const MAXIMUM_BIN_COUNT = 2 ** 31 - 1;
/** Retains the existing cell-capacity contract independently of the cube-wide bin/count bound. */
const MAXIMUM_CELL_COUNT = 2 ** 24 - 1;

/** Four legacy moments plus two exact 16-bit words for the included-bin count. */
const EMERGING_HOT_SPOTS_MOMENT_WORD_COUNT = 6;

/** Caller-owned outputs of {@link GPUEmergingHotSpots}. */
export type GPUEmergingHotSpotsOutput = SpatialStatisticsPopulationOutput & {
  // includedCount is exact; globalStatistics[0] retains the compatible float32 count.
  /** Caller-owned per-bin Gi* z-scores in cube layout; NaN for missing bins. */
  giZScores: GraphDataView<'float32'>;
  /** Optional caller-owned per-bin two-sided normal p-value of the Gi* z, cube layout, NaN for missing bins. */
  pValues?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned per-bin `multipleTesting`-adjusted p-value in cube layout; NaN for
   * missing bins. Equals `pValues` for `'none'`.
   */
  adjustedPValues?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned per-cell index of the last finite slice, or
   * `GPU_EMERGING_HOT_SPOTS_NO_SLICE` (`0xffffffff`) for a cell without any.
   */
  lastValidSlice?: GraphDataView<'uint32'>;
  /** Caller-owned per-cell Mann-Kendall z of the Gi* series (continuity corrected). */
  trendZ: GraphDataView<'float32'>;
  /** Caller-owned per-cell two-sided Mann-Kendall p-value. */
  trendP: GraphDataView<'float32'>;
  /** Caller-owned per-cell Mann-Kendall statistic `S`, an exact integer. */
  trendS: GraphDataView<'sint32'>;
  /** Caller-owned per-cell category code from `GPU_EMERGING_HOT_SPOTS_CATEGORIES` (0 to 17). */
  category: GraphDataView<'uint32'>;
  /** Caller-owned per-cell count of significant hot slices (`z > 0` and adjusted p at most `1 - confidenceLevel`). */
  hotSliceCount: GraphDataView<'uint32'>;
  /** Caller-owned per-cell count of significant cold slices (`z < 0` and adjusted p at most `1 - confidenceLevel`). */
  coldSliceCount: GraphDataView<'uint32'>;
  /** Optional caller-owned `[n, mean, variance, standardDeviation]` of the valid bins. */
  globalStatistics?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUEmergingHotSpots}.
 *
 * The spatial neighborhood is either a regular lattice (`gridWidth`, `gridHeight` and the
 * per-frame `radius`) or arbitrary spatial `weights` over the cells; exactly one of the two.
 *
 * Per-frame (no rebuild or recompile): the contents of `values`, `mask`, `weights` and `parameters`
 * (radius in lattice mode, temporal window, critical z, trend significance and persistence
 * thresholds, minimum trend length, confidence level). Compile-time: `gridWidth`, `gridHeight`,
 * `sliceCount`, `maximumRadius`, `selfWeight`, `multipleTesting`, `finalSlice`, the value format,
 * and which optional views are present.
 */
export type GPUEmergingHotSpotsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'emerging-hot-spots'`. */
  id?: string;
  /**
   * Dense space-time cube indexed `cell * sliceCount + slice` with `cell = row * gridWidth +
   * column` in lattice mode and `cell` = the weights row in weights mode, the layout of `GPUTemporalReduction` output. A NaN bin is missing and excluded. A
   * `uint32` view (for example `GPUTemporalReduction` `counts`) is read as `f32(count)`, every bin
   * valid. Cell identities must remain in weights/grid order and slices in one fixed period
   * order. Observed empty event bins are 0; unknown float32 bins are NaN. Adapt sparse-gather
   * sentinels at the producer boundary (for example, GPUKeyLookup unmatchedValue: 0 for counts).
   */
  values: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Lattice mode: lattice width in cells. Compile-time. Set together with `gridHeight`. */
  gridWidth?: number;
  /** Lattice mode: lattice height in cells. Compile-time. Set together with `gridWidth`. */
  gridHeight?: number;
  /**
   * Weights mode: square self-join spatial weights over the cells (rows of `weights` are cells),
   * for example `GPUNeighborSearch` output over H3 cell centers, optionally transformed. Used as
   * given. The spatial neighborhood of a bin is the cell's CSR row (weights `w_ij`) and the cell
   * itself (`selfWeight`), each over the current slice and the `temporalWindow` previous slices,
   * every bin with temporal weight 1. Fewer than 2^24 cells. The per-frame `radius` is ignored.
   * Exclusive with `gridWidth` and `gridHeight`.
   */
  weights?: GPUSpatialWeights | GPUChunkedSpatialWeights;
  /**
   * Weights mode: weight `w_ii` of the focal cell's own bins (the star of Gi*). Compile-time,
   * finite and non-negative. Defaults to `1`. Lattice mode always uses the focal cell with weight 1.
   */
  selfWeight?: number;
  /** Slices per cell, at most `GPU_EMERGING_HOT_SPOTS_MAXIMUM_SLICE_COUNT`. Compile-time. */
  sliceCount: number;
  /**
   * Largest per-frame `radius` in cells, at most `GPU_EMERGING_HOT_SPOTS_MAXIMUM_RADIUS`. Larger
   * per-frame radii are clamped. Compile-time. Defaults to 4.
   */
  maximumRadius?: number;
  /** Optional per-cell mask (one row per cell): zero excludes every bin of the cell. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Per-frame parameters: a packed float32 view of at least
   * `GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH` elements written with
   * `getGPUEmergingHotSpotsParameterValues` or {@link GPUEmergingHotSpots.getParameterValues}.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Compile-time multiple-testing correction of the per-bin hot and cold decision. Defaults to
   * `'benjamini-hochberg'` (ESRI applies FDR in emerging hot spot analysis; it was uncorrected).
   *
   * The two-sided normal p-value of every valid Gi* bin is adjusted over the whole family of valid
   * bins of the cube (never a display mask), and a bin is hot or cold when its adjusted p-value is
   * at most `1 - confidenceLevel`, with the sign of z. Bins within a temporal window overlap and
   * neighboring cells share observations, so the tests are dependent: `'benjamini-yekutieli'` is
   * the method that controls the false discovery rate under arbitrary dependence, at a cost in
   * power of about `ln m`. `'none'` reproduces the pre-correction rule `|z| >= criticalZ` exactly
   * and is the only method that accepts the `criticalZ` parameter. The family size is limited to
   * at most 2^31 bins when adjustment or p-value outputs are requested.
   */
  multipleTesting?: GPUMultipleTestingMethod;
  /**
   * Compile-time. How the cell's final slice is chosen. `'require'` (default): the final slice is
   * the last slice of the cube, and a cell whose last slice is missing (NaN) gets
   * `GPU_EMERGING_HOT_SPOTS_CATEGORIES.UNDETERMINED`. `'last-valid'`: the final slice is the last
   * finite one, so a stale cell can be labelled New or Persistent for a period it has no data in
   * (the previous behavior).
   */
  finalSlice?: 'require' | 'last-valid';
  /**
   * Compile-time autocorrelation correction of the Mann-Kendall variance. `'none'` (default) keeps the
   * unmodified test, bit for bit. `'hamed-rao'` (Hamed and Rao 1998, ranks of the Sen-detrended series,
   * lags with a significant autocorrelation only) and `'yue-wang'` (Yue and Wang 2004, the detrended
   * series, all lags) scale `Var(S)` by the effective sample size correction, so `trendZ` and `trendP`
   * are a test of trend rather than a screening threshold over serially correlated Gi* windows. The
   * correction is skipped (factor 1) for fewer than 3 finite slices and when the factor is not
   * positive. Needs `sliceCount <= GPU_EMERGING_HOT_SPOTS_MAXIMUM_CORRECTED_TREND_SLICE_COUNT`. `trendS` and
   * the category codes are unchanged in meaning.
   */
  trendAutocorrelationCorrection?: 'none' | 'hamed-rao' | 'yue-wang';
  /**
   * Compile-time largest autocorrelation lag of the correction, at least 1; lags beyond `n - 1`
   * are ignored. Defaults to `n - 1`, every lag. Hamed and Rao suggest 3, Yue and Wang 1.
   */
  trendCorrectionMaximumLag?: number;
  /** Compile-time `'hamed-rao'` lag filter: a lag counts when `|r| > criticalZ / sqrt(n)`. Defaults to 1.96 (95%). */
  trendCorrectionCriticalZ?: number;
  /** Caller-owned outputs. */
  output: GPUEmergingHotSpotsOutput;
};

/**
 * ArcGIS-style emerging hot spot analysis over a dense space-time cube with lattice or CSR neighborhoods.
 *
 * Passes, each deterministic and free of float atomics:
 * 1. Global `n`, mean and population standard deviation of valid bins: fixed workgroup trees
 *    over 4096-bin blocks, then another fixed workgroup tree combines their partials (mean first,
 *    then the sum of squared deviations). The mean gets a second-pass correction
 *    (`mean += sum(x - mean) / n`) and every `x_j - X` below is the stored first-pass deviation
 *    minus the mean of those deviations, the scheme of `GPUGlobalSpatialStatistics` and
 *    `GPUHotSpotAnalysis`, so a large common offset does not round the centered values. In the
 *    Gi* sums the residual is subtracted once per sum (`sum_j w_j d_j - r sum_j w_j` with `d_j`
 *    the stored deviations), so two windows whose values add up to the same total get bitwise
 *    equal z-scores and count as a tie in step 3.
 * 2. Space-time Gi* per bin, the focal bin included. Lattice mode uses binary weights:
 *    `z = sum_j (x_j - X) / (S * sqrt((n k - k^2) / (n - 1)))` where `j` covers valid bins of cells
 *    whose lattice offset satisfies `dx^2 + dy^2 <= radius^2` in the current and `temporalWindow`
 *    previous slices, and `k` counts them. Neighbors are visited in a fixed order (offset rows,
 *    offset columns, ascending slice). z is NaN for a missing bin, `n < 2`, `S = 0` or `k >= n`.
 *    Weights mode generalizes to `z = sum_j w_j (x_j - X) / (S * sqrt((n S1 - W^2) / (n - 1)))`
 *    with `W = sum_j w_j` and `S1 = sum_j w_j^2` over the same bins, where `w_j` is the spatial
 *    weight of the bin's cell (`selfWeight` for the focal cell, CSR weight for the others; the
 *    temporal weight is 1) and neighbors are visited in CSR slot order, then ascending slice.
 *    z is NaN for a missing bin, `n < 2`, `S = 0` or `n S1 <= W^2`. With binary weights the two
 *    modes agree.
 *    Scaled coefficient moments use a verified majority coefficient comparable to the focal
 *    weight as origin, falling back to the smallest comparable usable coefficient. Masked and
 *    non-finite bins cannot choose the origin. Temporal merges convert each slice's sums and
 *    coefficient moments to a common scale and origin before combining them. The centered
 *    coefficient squares are summed with compensation. Weighted and unweighted deviation sums
 *    carry their rounding corrections through the temporal merge. The weighted route also
 *    compensates the population residual mean used to center those sums.
 * 3. Per-cell Mann-Kendall test over the finite z series (NaN slices skipped): `S = sum_{i<j}
 *    sign(z_j - z_i)` in exact integers, `Var = [n(n-1)(2n+5) - sum_t t(t-1)(2t+5)] / 18` with tie
 *    groups of exactly equal f32 values, `z = (S - sign(S)) / sqrt(Var)` (0 when `S = 0` or
 *    `Var <= 0`), and the two-sided p-value `erfc(|z| / sqrt(2))` with the Numerical Recipes
 *    `erfcc` Chebyshev fit (relative error below 1.2e-7); `p = 1` when `Var <= 0`.
 * 3b. Between 2 and 3: two-sided normal p-values `erfc(|z| / sqrt(2))` of every valid Gi* bin,
 *    adjusted by {@link GPUMultipleTestingAdjustment} over all valid bins (`multipleTesting`).
 * 4. Classification into the 17 {@link GPU_EMERGING_HOT_SPOTS_CATEGORIES} plus `UNDETERMINED`.
 *
 * Classification rules, evaluated on the finite z series (NaN slices skipped). A slice is hot when
 * its adjusted p-value is at most `1 - confidenceLevel` and `z > 0`, and cold with `z < 0`
 * (without a correction, `multipleTesting: 'none'`, hot is `z >= criticalZ` and cold
 * `z <= -criticalZ`, the same two-sided critical values). `N` is the finite slice count, `H` and
 * `C` the hot and cold counts, and "90%" is `persistentFraction * N`. The trend is significant
 * when `trendP <= trendSignificanceLevel`. The final slice is the last slice of the cube
 * (`finalSlice: 'require'`, default) or the last finite one (`'last-valid'`).
 * The final slice picks the track, then rules apply in order:
 * - Final slice hot: New (`H = 1`), Consecutive (trailing hot run of at least 2, `H` equal to the
 *   run, `H < 90%`), Intensifying / Diminishing / Persistent (`H >= 90%`, significant upward /
 *   significant downward / no significant trend), Sporadic (`C = 0`), otherwise Oscillating.
 * - Final slice cold: the mirror with Intensifying cold on a significant downward trend (the cold
 *   intensifies) and Diminishing cold on a significant upward trend.
 * - Final slice neither: Historical hot (`H >= 90%`), Historical cold (`C >= 90%`), otherwise no
 *   pattern. Current ArcGIS Pro defines Sporadic hot and cold only with a statistically significant
 *   hot (cold) final time-step interval, which the final-slice-hot (cold) track above already
 *   covers, so a cell that fails that test with only hot (cold) history is `NO_PATTERN`. With
 *   `sporadicRequiresFinalSlice: false` the older rule also applies here: Sporadic hot (`H > 0`,
 *   `C = 0`) and Sporadic cold (`C > 0`, `H = 0`) without a significant final slice, to reproduce
 *   results computed before the rule changed.
 * Ordering decisions: New is tested before Oscillating, so a cell hot only in the final slice is New
 * even with cold history; a cell with a cold final slice and `C = 1` is New cold even when 90% of
 * its earlier slices were hot; mixed hot and cold history without a significant final slice has
 * no pattern.
 *
 * Undetermined cells (`UNDETERMINED`, 17): with `finalSlice: 'require'`, a cell with at least one
 * finite slice whose last slice is missing; and a cell that is persistent in one direction
 * (the Intensifying / Persistent / Diminishing branches) but has fewer finite slices than
 * `minimumTrendSliceCount` (default 10, ESRI guidance), because its trend class would rest on a
 * Mann-Kendall test that is not supported by enough data. Cells below the minimum trend length
 * have NaN `trendZ` and `trendP`; a missing final slice alone does not discard a trend supported
 * by earlier slices. They are never silently reported as Persistent. Cells without any finite
 * slice are `NO_PATTERN`.
 *
 * Neighborhood loss flags suppress every Gi* bin before downstream inference. Input population
 * moments and counts remain available; clearing the flags restores the neighborhood results on
 * the next frame. Construction snapshots topology descriptors, while buffer contents stay mutable.
 *
 * Incomplete weights (weights mode): when `weights.status.overflow` or
 * `weights.status.candidateOverflow` (outer or chunk-local) is nonzero, the neighborhoods are
 * truncated and nothing that depends on them is reported for that frame: `giZScores`, `pValues`,
 * `adjustedPValues`, `trendZ` and `trendP` are quiet NaN, `trendS`, `hotSliceCount` and
 * `coldSliceCount` are 0, `lastValidSlice` is `GPU_EMERGING_HOT_SPOTS_NO_SLICE` and every
 * `category` is `UNDETERMINED`. `globalStatistics` and the population count do not depend on the
 * weights and are kept. Clearing the flags restores ordinary results on the next frame.
 *
 * Statistical caveats. Consecutive Gi* z values share `temporalWindow` slices, so the series is
 * serially correlated and the Mann-Kendall p-value is anti-conservative (ESRI has the same
 * property); `trendSignificanceLevel` is a screening threshold, not an error rate. Bins within a
 * temporal window overlap and neighboring cells share observations, so the per-bin tests are
 * dependent: Benjamini-Hochberg is only guaranteed under positive dependence, and
 * `'benjamini-yekutieli'` is the method that is valid under arbitrary dependence.
 *
 * Legend contract: category codes never change meaning with `multipleTesting`, `confidenceLevel`
 * or `finalSlice`; those options only change which bins count as hot or cold and which cells are
 * undetermined. Use {@link GPUEmergingHotSpots.getInferenceMetadata} to label the legend.
 *
 * Determinism: all sums run in a fixed order and integer statistics are exact, so repeated
 * encodings on one device are bitwise identical.
 *
 * Non-goals: permutation p-values, building the weights, and time-step intervals other than one
 * slice per cube column (aggregate to the interval before building the cube, as the space-time
 * hot spots recipe does).
 */
export class GPUEmergingHotSpots implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUEmergingHotSpotsProps;
  private readonly populationCount?: GraphDataView<'uint32'>;
  /** Resolved correction method (`'benjamini-hochberg'` when the prop is omitted). */
  readonly multipleTesting: GPUMultipleTestingMethod;
  /** Resolved final slice rule (`'require'` when the prop is omitted). */
  readonly finalSlice: 'require' | 'last-valid';
  /** Resolved Mann-Kendall variance correction (`'none'` when the prop is omitted). */
  readonly trendAutocorrelationCorrection: 'none' | 'hamed-rao' | 'yue-wang';

  constructor(props: GPUEmergingHotSpotsProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'values',
      'weights',
      'selfWeight',
      'gridWidth',
      'gridHeight',
      'sliceCount',
      'maximumRadius',
      'mask',
      'parameters',
      'multipleTesting',
      'finalSlice',
      'trendAutocorrelationCorrection',
      'trendCorrectionMaximumLag',
      'trendCorrectionCriticalZ',
      'output'
    ]);
    props = {
      ...props,
      output: snapshotConstructionFields(props.output, [
        'giZScores',
        'pValues',
        'adjustedPValues',
        'lastValidSlice',
        'trendZ',
        'trendP',
        'trendS',
        'category',
        'hotSliceCount',
        'coldSliceCount',
        'globalStatistics',
        'includedCount',
        'count'
      ]),
      weights: props.weights ? snapshotWeightsPort(props.weights) : undefined
    };
    this.id = props.id ?? 'emerging-hot-spots';
    this.props = props;
    this.populationCount = resolveStatisticsPopulationCount(this.id, props.output);
    const id = this.id;
    const {output} = props;
    this.multipleTesting = resolveGPUMultipleTestingMethod(
      `${id} multipleTesting`,
      props.multipleTesting ?? 'benjamini-hochberg'
    );
    this.finalSlice = props.finalSlice ?? 'require';
    this.trendAutocorrelationCorrection = props.trendAutocorrelationCorrection ?? 'none';
    validateRequirement(
      id,
      this.finalSlice === 'require' || this.finalSlice === 'last-valid',
      `finalSlice must be 'require' or 'last-valid'; received ${this.finalSlice}`
    );
    validateRequirement(
      id,
      ['none', 'hamed-rao', 'yue-wang'].includes(this.trendAutocorrelationCorrection),
      `trendAutocorrelationCorrection must be 'none', 'hamed-rao' or 'yue-wang'; received ${this.trendAutocorrelationCorrection}`
    );
    if (this.trendAutocorrelationCorrection !== 'none') {
      validateCapacity(
        id,
        'sliceCount with trendAutocorrelationCorrection',
        props.sliceCount,
        1,
        GPU_EMERGING_HOT_SPOTS_MAXIMUM_CORRECTED_TREND_SLICE_COUNT
      );
    }
    if (props.trendCorrectionMaximumLag !== undefined) {
      validateCapacity(id, 'trendCorrectionMaximumLag', props.trendCorrectionMaximumLag, 1);
    }
    if (props.trendCorrectionCriticalZ !== undefined) {
      validateFiniteNumber(
        id,
        'trendCorrectionCriticalZ float32',
        Math.fround(props.trendCorrectionCriticalZ),
        0
      );
    }
    const {gridWidth, gridHeight, sliceCount} = props;
    const hasLattice = gridWidth !== undefined || gridHeight !== undefined;
    validateRequirement(
      id,
      hasLattice !== (props.weights !== undefined),
      `gridWidth and gridHeight, or weights needs exactly one route; received lattice=${hasLattice}, weights=${props.weights !== undefined}`
    );
    const selfWeight = props.selfWeight ?? 1;
    validateRequirement(
      id,
      Number.isFinite(Math.fround(selfWeight)) && selfWeight >= 0,
      `selfWeight must round to a finite float32 >= 0; received ${selfWeight}`
    );
    if (hasLattice) {
      validateCapacity(id, 'gridWidth', gridWidth ?? Number.NaN);
      validateCapacity(id, 'gridHeight', gridHeight ?? Number.NaN);
    }
    validateCapacity(id, 'sliceCount', sliceCount, 1, GPU_EMERGING_HOT_SPOTS_MAXIMUM_SLICE_COUNT);
    const cellCount = props.weights
      ? validateGPUSpatialWeightsOrChunks(id, props.weights)
      : gridWidth! * gridHeight!;
    validateRequirement(
      id,
      cellCount <= MAXIMUM_CELL_COUNT,
      `cellCount must be at most ${MAXIMUM_CELL_COUNT}; received ${cellCount}`
    );
    const binCount = cellCount * sliceCount;
    validateRequirement(
      id,
      binCount <= MAXIMUM_BIN_COUNT,
      `cellCount * sliceCount must be below 2^31; received ${binCount}`
    );
    validateCapacity(
      id,
      'maximumRadius',
      props.maximumRadius ?? 4,
      0,
      GPU_EMERGING_HOT_SPOTS_MAXIMUM_RADIUS
    );
    validateSinglePackedView(props.values, ['float32', 'uint32'], `${id} values`);
    validateRequirement(
      id,
      props.values.length === binCount,
      `values length must equal cellCount * sliceCount (${binCount}); received ${props.values.length}`
    );
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    validateRequirement(
      id,
      props.parameters.length >= GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH,
      `parameters length must be at least ${GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH}; received ${props.parameters.length}`
    );
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      validateRequirement(
        id,
        props.mask.length === cellCount,
        `mask length must equal the cell count (${cellCount}); received ${props.mask.length}`
      );
    }
    validateSinglePackedView(output.giZScores, ['float32'], `${id} giZScores`);
    validateRequirement(
      id,
      output.giZScores.length === binCount,
      `giZScores length must equal values length (${binCount}); received ${output.giZScores.length}`
    );
    for (const [name, view] of [
      ['pValues', output.pValues],
      ['adjustedPValues', output.adjustedPValues]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['float32'], `${id} ${name}`);
        validateRequirement(
          id,
          view.length === binCount,
          `${name} length must equal values length (${binCount}); received ${view.length}`
        );
      }
    }
    validateRequirement(
      id,
      !(this.multipleTesting !== 'none' || output.pValues || output.adjustedPValues) ||
        binCount < GPU_MULTIPLE_TESTING_MAXIMUM_ROWS,
      `multiple testing bins must be less than ${GPU_MULTIPLE_TESTING_MAXIMUM_ROWS}; received ${binCount}`
    );
    if (output.lastValidSlice) {
      validatePackedUint32View(output.lastValidSlice, `${id} lastValidSlice`);
      validateRequirement(
        id,
        output.lastValidSlice.length === cellCount,
        `lastValidSlice length must equal the cell count (${cellCount}); received ${output.lastValidSlice.length}`
      );
    }
    validateSinglePackedView(output.trendZ, ['float32'], `${id} trendZ`);
    validateSinglePackedView(output.trendP, ['float32'], `${id} trendP`);
    validateSinglePackedView(output.trendS, ['sint32'], `${id} trendS`);
    validatePackedUint32View(output.category, `${id} category`);
    validatePackedUint32View(output.hotSliceCount, `${id} hotSliceCount`);
    validatePackedUint32View(output.coldSliceCount, `${id} coldSliceCount`);
    for (const [name, view] of [
      ['trendZ', output.trendZ],
      ['trendP', output.trendP],
      ['trendS', output.trendS],
      ['category', output.category],
      ['hotSliceCount', output.hotSliceCount],
      ['coldSliceCount', output.coldSliceCount]
    ] as const) {
      validateRequirement(
        id,
        view.length === cellCount,
        `${name} length must equal the cell count (${cellCount}); received ${view.length}`
      );
    }
    if (output.globalStatistics) {
      validateSinglePackedView(output.globalStatistics, ['float32'], `${id} globalStatistics`);
      validateRequirement(
        id,
        output.globalStatistics.length >= GPU_EMERGING_HOT_SPOTS_STATISTICS_LENGTH,
        `globalStatistics length must be at least ${GPU_EMERGING_HOT_SPOTS_STATISTICS_LENGTH}; received ${output.globalStatistics.length}`
      );
    }
    if (this.populationCount) {
      validatePackedUint32View(
        this.populationCount,
        `${id} output.${output.includedCount ? 'includedCount' : 'count'}`
      );
      validateCapacity(
        id,
        `output.${output.includedCount ? 'includedCount' : 'count'} length`,
        this.populationCount.length,
        1,
        1
      );
    }
    validateDisjointWritableViews(id, output);
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.giZScores,
        output.pValues,
        output.adjustedPValues,
        output.lastValidSlice,
        output.trendZ,
        output.trendP,
        output.trendS,
        output.category,
        output.hotSliceCount,
        output.coldSliceCount,
        output.globalStatistics,
        this.populationCount
      ],
      [
        props.values,
        props.parameters,
        props.mask,
        ...(props.weights ? getGPUSpatialWeightsOrChunksViews(props.weights) : [])
      ]
    );
  }

  /**
   * Packs `settings` for this node's `parameters` view, with `settings.multipleTesting` taken
   * from the node. Throws on invalid settings or a `multipleTesting` that differs from the node.
   *
   * @param settings The per-frame settings to pack (the same object `getInferenceMetadata` reads).
   * @param target Optional destination of at least `GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH` elements.
   */
  getParameterValues(settings: GPUEmergingHotSpotsSettings, target?: Float32Array): Float32Array {
    return getGPUEmergingHotSpotsParameterValues(this._resolveSettings(settings), target, this.id);
  }

  /**
   * CPU-side description of the inference behind the hot and cold bins, for building a legend,
   * derived from the same `settings` that {@link getParameterValues} packs.
   * A custom unadjusted critical value uses its encoded two-sided normal tail. Otherwise the
   * level is `1 - confidenceLevel`. The independent trend level is unchanged.
   */
  getInferenceMetadata(settings: GPUEmergingHotSpotsSettings): GPUInferenceMetadata {
    const resolved = this._resolveSettings(settings);
    const encoded = getGPUEmergingHotSpotsParameterValues(resolved, undefined, this.id);
    return {
      alternative: 'two-sided',
      permutationCount: 0,
      multipleTesting: this.multipleTesting,
      significanceLevel:
        this.multipleTesting === 'none' && resolved.criticalZ !== undefined
          ? getChiSquareSurvival(encoded[2] * encoded[2], 1)
          : 1 - (resolved.confidenceLevel ?? 0.95),
      testedRowsDefinition: 'valid space-time bins'
    };
  }

  private _resolveSettings(settings: GPUEmergingHotSpotsSettings): GPUEmergingHotSpotsSettings {
    settings = snapshotEmergingHotSpotSettings(settings);
    validateRequirement(
      this.id,
      settings.multipleTesting === undefined || settings.multipleTesting === this.multipleTesting,
      `settings.multipleTesting must match the node (${this.multipleTesting}); received ${settings.multipleTesting}`
    );
    return {...settings, multipleTesting: this.multipleTesting};
  }

  /** Returns the emerging-hot-spot nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateSpatialStatisticsBindings(graph, id, props, {
      'centered bins': props.values.length * 4,
      moments: EMERGING_HOT_SPOTS_MOMENT_WORD_COUNT * 4
    });
    const {values, mask, parameters, weights, sliceCount, output} = props;
    const {giZScores} = output;
    const gridWidth = props.gridWidth ?? 1;
    const gridHeight = props.gridHeight ?? 1;
    const selfWeight = props.selfWeight ?? 1;
    validateGraphViewsBelongToGraph(id, graph, [
      values,
      mask,
      parameters,
      ...(weights ? getGPUSpatialWeightsOrChunksViews(weights) : []),
      giZScores,
      output.trendZ,
      output.trendP,
      output.pValues,
      output.adjustedPValues,
      output.lastValidSlice,
      output.trendS,
      output.category,
      output.hotSliceCount,
      output.coldSliceCount,
      output.globalStatistics
    ]);
    const {finalSlice, multipleTesting} = this;
    // Without a correction the classifier compares z with the critical z and needs no p-values.
    const usesAdjustment = multipleTesting !== 'none';
    const needsPValues =
      usesAdjustment || output.pValues !== undefined || output.adjustedPValues !== undefined;
    const maximumRadius = props.maximumRadius ?? 4;
    const dispatches = weights ? getGPUSpatialWeightsChunkDispatches(weights) : [];
    const cellCount = weights
      ? dispatches.reduce((total, dispatch) => total + dispatch.rowCount, 0)
      : gridWidth * gridHeight;
    const binCount = cellCount * sliceCount;
    // Rule 16: every per-bin view is one storage binding, so the cube must fit the device's
    // binding size (2^25 four-byte bins at the 128 MiB default), not only the 2^31 index range.
    const maximumBindingBins = Math.floor(graph.device.limits.maxStorageBufferBindingSize / 4);
    validateRequirement(
      id,
      binCount <= maximumBindingBins,
      `cube bin count must be at most ${maximumBindingBins} for maxStorageBufferBindingSize; received ${binCount}`
    );
    const cooperativeMannKendall = sliceCount >= MANN_KENDALL_COOPERATIVE_SLICE_COUNT;
    const blockCount = Math.ceil(binCount / BIN_BLOCK);
    const valueType = values.format === 'uint32' ? 'u32' : 'f32';
    const sumPartials = createTransientView(graph, `${id}-sum-partials`, 'float32', blockCount);
    const countPartials = createTransientView(graph, `${id}-count-partials`, 'uint32', blockCount);
    const squarePartials = createTransientView(
      graph,
      `${id}-square-partials`,
      'float32',
      blockCount
    );
    const statistics = createTransientView(
      graph,
      `${id}-statistics`,
      'float32',
      EMERGING_HOT_SPOTS_MOMENT_WORD_COUNT
    );
    // Mean centering, the scheme of GPUGlobalSpatialStatistics and the local statistics: the
    // first-pass deviation of every bin from the provisional mean is stored (NaN for a bin outside
    // the analysis), the mean of those deviations corrects the mean, and a centered value is the
    // stored deviation minus that residual. The Gi* sums add the stored deviations and subtract
    // the residual once per sum (residual * sum of weights), so windows whose values add up to the
    // same total keep bitwise equal z-scores, which the Mann-Kendall tie count relies on.
    const centeredValues = createTransientView(
      graph,
      `${id}-first-deviations`,
      'float32',
      binCount
    );
    const deviationPartials = createTransientView(
      graph,
      `${id}-deviation-partials`,
      'float32',
      blockCount * (weights ? 2 : 1)
    );
    const residualMean = createTransientView(graph, `${id}-residual-mean`, 'float32', 1);
    // Retain both parts through the population residual reduction. An error in that mean can
    // otherwise be amplified when it is multiplied by a dense window's sum of coefficients.
    const deviationSumZero = weights ? 'vec2<f32>(0.0)' : '0.0';
    const deviationReductionWGSL = /* wgsl */ `
${weights ? EMERGING_SUM_WGSL : ''}
var<workgroup> partialSums: array<${weights ? 'vec2<f32>' : 'f32'}, ${MOMENT_WORKGROUP_SIZE}>;`;
    const mergeDeviationLanes = weights
      ? 'partialSums[localInvocationIndex] = mergeEmergingSums(partialSums[localInvocationIndex], partialSums[localInvocationIndex + stride]);'
      : 'partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];';

    const constantDeclarations = /* wgsl */ `
const GRID_WIDTH: u32 = ${gridWidth}u;
const GRID_HEIGHT: u32 = ${gridHeight}u;
const CELL_COUNT: u32 = ${cellCount}u;
const SLICE_COUNT: u32 = ${sliceCount}u;
const BIN_COUNT: u32 = ${binCount}u;
const BIN_BLOCK: u32 = ${BIN_BLOCK}u;
const MAXIMUM_RADIUS: f32 = ${maximumRadius}.0;
const SELF_WEIGHT: f32 = ${getWGSLFloatLiteral(selfWeight)};
${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
`;
    // Kernels that read the raw cube.
    const declarations = /* wgsl */ `${constantDeclarations}
fn readValue(bin: u32) -> f32 {
  return f32(values[valuesOffset + bin]);
}

fn isBinValid(bin: u32, cell: u32) -> bool {
  ${mask ? 'if (mask[maskOffset + cell] == 0u) { return false; }' : ''}
  return isFiniteFloat(readValue(bin));
}
`;
    // Kernels that read the stored first-pass deviations: a bin is in the analysis when its
    // deviation is finite (the deviation pass wrote NaN for masked and non-finite bins).
    const centeredDeclarations = /* wgsl */ `${constantDeclarations}
fn readDeviation(bin: u32) -> f32 {
  return centeredValues[centeredValuesOffset + bin];
}

fn isBinValid(bin: u32, cell: u32) -> bool {
  return isFiniteFloat(readDeviation(bin));
}
`;
    const residualMeanBinding: WGSLKernelBinding = {
      name: 'residualMean',
      view: residualMean,
      type: 'f32',
      access: 'read'
    };
    const centeredBinding = (access: 'read' | 'read_write'): WGSLKernelBinding => ({
      name: 'centeredValues',
      view: centeredValues,
      type: 'f32',
      access
    });
    const valuesBinding: WGSLKernelBinding = {
      name: 'values',
      view: values,
      type: valueType,
      access: 'read'
    };
    const maskBindings: WGSLKernelBinding[] = mask
      ? [{name: 'mask', view: mask, type: 'u32', access: 'read'}]
      : [];
    const parametersBinding: WGSLKernelBinding = {
      name: 'parameters',
      view: parameters,
      type: 'f32',
      access: 'read'
    };
    const statisticsBinding = (access: 'read' | 'read_write'): WGSLKernelBinding => ({
      name: 'statistics',
      view: statistics,
      type: 'f32',
      access
    });
    // Space-time Gi* by temporal factoring. The window sum over slices [first, slice] of the
    // neighbor sums equals the sum over those slices of per-slice ("instant") neighbor sums, so
    // each workgroup computes the instant sums of its cells once (one thread per (cell, slice),
    // consecutive threads reading consecutive slices) into workgroup memory and every bin then adds
    // at most `window + 1` instants. The work is n * T * (k + window) instead of the
    // n * T * k * window of re-walking the neighbors for every window slice.
    const cellsPerGroup = Math.max(1, Math.floor(GI_WORKGROUP_SIZE / sliceCount));
    const getInstantWGSL = (localCell: string) =>
      weights
        ? `var weighted = vec2<f32>(0.0);
    var unweighted = vec2<f32>(0.0);
    var weightCount = 0u;
    var weightMoment = vec3<f32>(0.0);
    let bin = cell * SLICE_COUNT + slice;
    // A common power-of-two scale keeps squared coefficient differences normal. Choose the
    // verified majority coefficient when comparable, falling back to the smallest comparable
    // coefficient. Both choices are independent of CSR order. Sparse positive or negative
    // deviations then remain exceptional terms, rather than thousands of cancelling offsets.
    var maximumCoefficient = abs(SELF_WEIGHT);
    var majorityCandidate = SELF_WEIGHT;
    var coefficientCount = select(0u, 1u, SELF_WEIGHT != 0.0 && isBinValid(bin, cell));
    var majorityBalance = coefficientCount;
    for (var slot = offsets[offsetsOffset + ${localCell}]; slot < offsets[offsetsOffset + ${localCell} + 1u]; slot++) {
      let neighborCell = neighbors[neighborsOffset + slot];
      if (neighborCell < CELL_COUNT && neighborCell != cell && isBinValid(neighborCell * SLICE_COUNT + slice, neighborCell)) {
        let coefficient = weights[weightsOffset + slot];
        maximumCoefficient = max(maximumCoefficient, abs(coefficient));
        coefficientCount++;
        if (majorityBalance == 0u) {
          majorityCandidate = coefficient;
          majorityBalance = 1u;
        } else if (coefficient == majorityCandidate) {
          majorityBalance++;
        } else {
          majorityBalance--;
        }
      }
    }
    let coefficientScale = getEmergingCoefficientScale(maximumCoefficient);
    let scaledSelf = SELF_WEIGHT / coefficientScale;
    // Only usable coefficients choose this slice's origin. Save the scale alongside the origin
    // so temporal merging can convert both before combining the sums and central moments.
    // A distant coefficient can erase a tiny focal weight in (origin + relative mean).
    // Only a comparable neighbor coefficient is a safe origin; scaled magnitudes are < 4,
    // so this magnitude gate cannot overflow even at the largest finite input coefficient.
    coefficientOrigin = scaledSelf;
    var majorityCount = select(0u, 1u, SELF_WEIGHT != 0.0 && isBinValid(bin, cell) && SELF_WEIGHT == majorityCandidate);
    for (var slot = offsets[offsetsOffset + ${localCell}]; slot < offsets[offsetsOffset + ${localCell} + 1u]; slot++) {
      let neighborCell = neighbors[neighborsOffset + slot];
      if (neighborCell < CELL_COUNT && neighborCell != cell && isBinValid(neighborCell * SLICE_COUNT + slice, neighborCell)) {
        let candidateOrigin = weights[weightsOffset + slot] / coefficientScale;
        if (weights[weightsOffset + slot] == majorityCandidate) { majorityCount++; }
        if (scaledSelf > 0.0 && candidateOrigin >= scaledSelf * 0.5 && candidateOrigin <= scaledSelf * 2.0) {
          coefficientOrigin = min(coefficientOrigin, candidateOrigin);
        }
      }
    }
    // Boyer-Moore proposes a candidate, but only a strict, verified majority may replace the
    // deterministic fallback. An unverified candidate would reintroduce order dependence.
    let scaledMajority = majorityCandidate / coefficientScale;
    if (majorityCount > coefficientCount / 2u && scaledSelf > 0.0 && scaledMajority >= scaledSelf * 0.5 && scaledMajority <= scaledSelf * 2.0) {
      coefficientOrigin = scaledMajority;
    }
    // The focal cell has weight SELF_WEIGHT; its CSR row never lists the cell itself.
    if (SELF_WEIGHT != 0.0 && isBinValid(bin, cell)) {
      let centeredValue = readDeviation(bin);
      unweighted = addCompensatedValue(unweighted, centeredValue);
      let relativeSelf = scaledSelf - coefficientOrigin;
      weighted = addCompensatedValue(weighted, relativeSelf * centeredValue);
      weightCount++;
      weightMoment = addEmergingWeightMoment(weightMoment, weightCount, relativeSelf);
    }
    for (var slot = offsets[offsetsOffset + ${localCell}]; slot < offsets[offsetsOffset + ${localCell} + 1u]; slot++) {
      let neighborCell = neighbors[neighborsOffset + slot];
      if (neighborCell >= CELL_COUNT || neighborCell == cell) {
        continue;
      }
      let weight = weights[weightsOffset + slot];
      let neighborBin = neighborCell * SLICE_COUNT + slice;
      if (isBinValid(neighborBin, neighborCell)) {
        let relativeWeight = weight / coefficientScale - coefficientOrigin;
        let centeredValue = readDeviation(neighborBin);
        weighted = addCompensatedValue(weighted, relativeWeight * centeredValue);
        unweighted = addCompensatedValue(unweighted, centeredValue);
        weightCount++;
        weightMoment = addEmergingWeightMoment(weightMoment, weightCount, relativeWeight);
      }
    }
    instantWeighted[local] = weighted;
    instantWeight[local] = weightMoment.x;
    instantSquare[local] = getCompensated(weightMoment.yz);
    instantCount[local] = weightCount;
    instantUnweighted[local] = unweighted;
    instantScale[local] = coefficientScale;`
        : `var weighted = 0.0;
    var neighborCount = 0u;
    let column = i32(cell % GRID_WIDTH);
    let row = i32(cell / GRID_WIDTH);
    for (var deltaRow = -reach; deltaRow <= reach; deltaRow++) {
      let neighborRow = row + deltaRow;
      if (neighborRow < 0 || neighborRow >= i32(GRID_HEIGHT)) {
        continue;
      }
      for (var deltaColumn = -reach; deltaColumn <= reach; deltaColumn++) {
        let neighborColumn = column + deltaColumn;
        if (neighborColumn < 0 || neighborColumn >= i32(GRID_WIDTH) ||
            f32(deltaRow * deltaRow + deltaColumn * deltaColumn) > radiusSquared) {
          continue;
        }
        let neighborCell = u32(neighborRow) * GRID_WIDTH + u32(neighborColumn);
        let neighborBin = neighborCell * SLICE_COUNT + slice;
        if (isBinValid(neighborBin, neighborCell)) {
          weighted += readDeviation(neighborBin);
          neighborCount++;
        }
      }
    }
    instantWeighted[local] = weighted;
    instantWeight[local] = f32(neighborCount);
    instantSquare[local] = 0.0;
    instantCount[local] = neighborCount;`;
    const windowSumWGSL = weights
      ? `var weightedSum = vec2<f32>(0.0);
    var unweightedSum = vec2<f32>(0.0);
    var weightMean = 0.0;
    var weightSquares = 0.0;
    var supportCount = 0u;
    // Missing bins can change a slice's maximum usable weight. Convert to the largest scale
    // in the window so none of the ratios amplify a slice's moments.
    var windowScale = instantScale[local];
    for (var windowSlice = firstSlice; windowSlice < slice; windowSlice++) {
      let slot = cellInGroup * SLICE_COUNT + windowSlice;
      windowScale = max(windowScale, instantScale[slot]);
    }
    let windowOrigin = coefficientOrigin * (instantScale[local] / windowScale);
    for (var windowSlice = firstSlice; windowSlice <= slice; windowSlice++) {
      let slot = cellInGroup * SLICE_COUNT + windowSlice;
      let scaleRatio = instantScale[slot] / windowScale;
      let originDifference = instantOrigin[slot] * scaleRatio - windowOrigin;
      // Bring both parts of the sums into the window coordinates before rounding them together.
      let scaledWeighted = instantWeighted[slot] * scaleRatio;
      let sliceUnweighted = instantUnweighted[slot];
      let originAdjustment = sliceUnweighted * originDifference;
      weightedSum = mergeEmergingSums(mergeEmergingSums(weightedSum, scaledWeighted), originAdjustment);
      unweightedSum = mergeEmergingSums(unweightedSum, sliceUnweighted);
      let sliceCount = instantCount[slot];
      if (sliceCount > 0u) {
        let mergedCount = supportCount + sliceCount;
        let difference = (instantWeight[slot] * scaleRatio + originDifference) - weightMean;
        weightSquares += (instantSquare[slot] * scaleRatio) * scaleRatio + difference * difference * f32(supportCount) * f32(sliceCount) / f32(mergedCount);
        weightMean += difference * f32(sliceCount) / f32(mergedCount);
        supportCount = mergedCount;
      }
    }`
      : `var weightedSum = 0.0;
    var weightSum = 0.0;
    var supportCount = 0u;
    for (var windowSlice = firstSlice; windowSlice <= slice; windowSlice++) {
      let slot = cellInGroup * SLICE_COUNT + windowSlice;
      weightedSum += instantWeighted[slot];
      weightSum += instantWeight[slot];
      supportCount += instantCount[slot];
    }`;
    // Lattice coefficients are binary. Arbitrary coefficients retain centered moments, including
    // zero coefficients outside the support, rather than subtracting large squared sums.
    const getGiBody = () => {
      const localCell = '(cell - chunkRowStart)';
      return `let chunkRowStart = getChunkParameter(36u);
  let chunkRowCount = getChunkParameter(37u);
  let group = index / ${GI_WORKGROUP_SIZE}u;
  let local = localInvocationIndex;
  let cellInGroup = local / SLICE_COUNT;
  let slice = local % SLICE_COUNT;
  let cell = group * CELLS_PER_GROUP + cellInGroup + chunkRowStart;
  let isActive = index < INVOCATION_COUNT && cellInGroup < CELLS_PER_GROUP && cell < chunkRowStart + chunkRowCount;
  let rawRadius = ${weights ? '0.0' : 'parameters[parametersOffset]'};
  let rawWindow = parameters[parametersOffset + 1u];
  let count = (u32(statistics[statisticsOffset + 4u]) << 16u) | u32(statistics[statisticsOffset + 5u]);
  let variance = statistics[statisticsOffset + 2u];
  let deviation = statistics[statisticsOffset + 3u];
  let parametersValid = isFiniteFloat(rawRadius) && isFiniteFloat(rawWindow) && rawRadius >= 0.0 && rawWindow >= 0.0;
  let radius = min(rawRadius, MAXIMUM_RADIUS);
  let radiusSquared = radius * radius;
  let reach = i32(floor(radius));
  ${weights ? 'var coefficientOrigin = 0.0;' : ''}
  instantWeighted[local] = ${weights ? 'vec2<f32>(0.0)' : '0.0'};
  instantWeight[local] = 0.0;
  instantSquare[local] = 0.0;
  instantCount[local] = 0u;
  ${weights ? 'instantUnweighted[local] = vec2<f32>(0.0); instantOrigin[local] = 0.0; instantScale[local] = 0.0;' : ''}
  if (isActive && parametersValid) {
    ${getInstantWGSL(localCell)}
    ${weights ? 'instantOrigin[local] = coefficientOrigin;' : ''}
  }
  // Every invocation reaches the barrier.
  workgroupBarrier();
  if (isActive) {
    var zScore = getQuietNaN(index);
    if (parametersValid && isBinValid(cell * SLICE_COUNT + slice, cell)) {
      let windowSize = u32(min(floor(rawWindow), f32(SLICE_COUNT - 1u)));
      let firstSlice = slice - min(windowSize, slice);
      ${windowSumWGSL}
      ${
        weights
          ? `let outsideCount = count - min(count, supportCount);
      let supportMean = windowOrigin + weightMean;
      let zeroContribution = divideBySpatialCount(multiplyBySpatialCount(multiplyBySpatialCount(supportMean * supportMean, supportCount), outsideCount), max(count, 1u));
      let spread = divideBySpatialCount(multiplyBySpatialCount(weightSquares + zeroContribution, count), max(count, 2u) - 1u);`
          : 'let spread = divideBySpatialCount(multiplyBySpatialCount(weightSum, count - min(count, supportCount)), max(count, 2u) - 1u);'
      }
      if (count >= 2u && variance > 0.0 && spread > 0.0 && isFiniteFloat(spread)) {
        // sum_j w_j (x_j - X): the stored deviations from the provisional mean, corrected once by
        // the mean of those deviations.
        ${
          weights
            ? `let residualTotal = residualMean[residualMeanOffset] * f32(supportCount);
        var centeredSum = addCompensatedValue(weightedSum, -residualTotal * weightMean);
        // A common coefficient on the complete population contributes exactly zero. On a
        // subset it contributes that subset's centered sum, with omitted bins weighted zero.
        if (supportCount < count) {
          let centeredUnweighted = addCompensatedValue(unweightedSum, -residualTotal);
          centeredSum = addCompensatedValue(centeredSum, windowOrigin * centeredUnweighted.x);
          centeredSum = addCompensatedValue(centeredSum, windowOrigin * centeredUnweighted.y);
        }`
            : 'let centeredSum = weightedSum - residualMean[residualMeanOffset] * weightSum;'
        }
        zScore = ${weights ? 'getCompensated(centeredSum)' : 'centeredSum'} / (deviation * sqrt(spread));
      }
    }
    giZScores[giZScoresOffset + cell * SLICE_COUNT + slice] = zScore;
  }`;
    };
    const pValues =
      output.pValues ??
      (needsPValues
        ? createTransientView(graph, `${id}-p-values`, 'float32', binCount)
        : undefined);
    const adjustedPValues =
      output.adjustedPValues ??
      (needsPValues
        ? createTransientView(graph, `${id}-adjusted-p-values`, 'float32', binCount)
        : undefined);
    // Two-sided normal p-value of every valid Gi* bin, then the correction over all valid bins.
    const getPValueNodes = (): GPUCommandNode<Parameters>[] => [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-p-values`,
        operation: OPERATION,
        variant: 'two-sided-p-values',
        bindings: [
          {name: 'giZScores', view: giZScores, type: 'f32', access: 'read'},
          {
            name: 'pValues',
            view: pValues!,
            type: 'f32',
            access: 'read_write'
          }
        ],
        invocationCount: binCount,
        declarations: SPATIAL_AUTOCORRELATION_FLOAT_WGSL,
        body: `let zScore = giZScores[giZScoresOffset + index];
  // erfc(|z| / sqrt(2)); NaN z (missing bin) stays NaN and is excluded from the family.
  pValues[pValuesOffset + index] = select(
    getQuietNaN(index),
    // The shared analytic p-value, floored at the smallest normal f32 instead of reporting exact
    // zeros that would tie in the correction.
    getTwoSidedPValue(zScore),
    isFiniteFloat(zScore)
  );`
      }),
      ...new GPUMultipleTestingAdjustment({
        id: `${id}-adjustment`,
        pValues: pValues!,
        method: multipleTesting,
        output: {adjustedPValues: adjustedPValues!}
      }).getCommandNodes(graph)
    ];
    const lossViews = weights ? getGPUSpatialWeightsLossViews(weights) : [];
    const weightsLoss = lossViews.length
      ? createTransientView(graph, `${id}-weights-loss`, 'uint32', 1)
      : undefined;
    const nodes: GPUCommandNode<Parameters>[] = [];

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-block-sums`,
        operation: OPERATION,
        variant: 'block-sums',
        bindings: [
          valuesBinding,
          ...maskBindings,
          {
            name: 'sumPartials',
            view: sumPartials,
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'countPartials',
            view: countPartials,
            type: 'u32',
            access: 'read_write'
          }
        ],
        // One workgroup per blockIndex: strided per-thread partials (coalesced reads) and a fixed tree.
        workgroupSize: MOMENT_WORKGROUP_SIZE,
        invocationCount: blockCount * MOMENT_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${declarations}
var<workgroup> partialSums: array<f32, ${MOMENT_WORKGROUP_SIZE}>;
var<workgroup> partialCounts: array<u32, ${MOMENT_WORKGROUP_SIZE}>;`,
        // No early return: every invocation of a workgroup must reach the barriers.
        body: `let blockIndex = index / ${MOMENT_WORKGROUP_SIZE}u;
  let isInRange = index < INVOCATION_COUNT;
  var sum = 0.0;
  var count = 0u;
  if (isInRange) {
    let begin = blockIndex * BIN_BLOCK;
    let end = min(begin + BIN_BLOCK, BIN_COUNT);
    for (var bin = begin + localInvocationIndex; bin < end; bin += ${MOMENT_WORKGROUP_SIZE}u) {
      if (isBinValid(bin, bin / SLICE_COUNT)) {
        sum += readValue(bin);
        count++;
      }
    }
  }
  partialSums[localInvocationIndex] = sum;
  partialCounts[localInvocationIndex] = count;
  workgroupBarrier();
  for (var stride = ${MOMENT_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
      partialCounts[localInvocationIndex] += partialCounts[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    sumPartials[sumPartialsOffset + blockIndex] = partialSums[0];
    countPartials[countPartialsOffset + blockIndex] = partialCounts[0];
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-mean`,
        operation: OPERATION,
        variant: 'mean',
        bindings: [
          {
            name: 'sumPartials',
            view: sumPartials,
            type: 'f32',
            access: 'read'
          },
          {
            name: 'countPartials',
            view: countPartials,
            type: 'u32',
            access: 'read'
          },
          statisticsBinding('read_write')
        ],
        // One workgroup combines the blockIndex partials with a fixed tree instead of one serial loop.
        workgroupSize: MOMENT_WORKGROUP_SIZE,
        invocationCount: MOMENT_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${SPATIAL_STATISTICS_COUNTS_WGSL}
const BLOCK_COUNT: u32 = ${blockCount}u;
var<workgroup> partialSums: array<f32, ${MOMENT_WORKGROUP_SIZE}>;
var<workgroup> partialCounts: array<u32, ${MOMENT_WORKGROUP_SIZE}>;`,
        body: `var sum = 0.0;
  var count = 0u;
  for (var blockIndex = localInvocationIndex; blockIndex < BLOCK_COUNT; blockIndex += ${MOMENT_WORKGROUP_SIZE}u) {
    sum += sumPartials[sumPartialsOffset + blockIndex];
    count += countPartials[countPartialsOffset + blockIndex];
  }
  partialSums[localInvocationIndex] = sum;
  partialCounts[localInvocationIndex] = count;
  workgroupBarrier();
  for (var stride = ${MOMENT_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
      partialCounts[localInvocationIndex] += partialCounts[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    count = partialCounts[0];
    statistics[statisticsOffset] = f32(count);
    statistics[statisticsOffset + 4u] = f32(count >> 16u);
    statistics[statisticsOffset + 5u] = f32(count & 65535u);
    statistics[statisticsOffset + 1u] = select(0.0, divideBySpatialCount(partialSums[0], count), count > 0u);
    statistics[statisticsOffset + 2u] = 0.0;
    statistics[statisticsOffset + 3u] = 0.0;
  }`
      }),
      // Second-pass mean: the first-pass deviation of every bin from the provisional mean is
      // stored, and the blockIndex sums of the deviations measure the rounding error of that mean.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-block-deviations`,
        operation: OPERATION,
        variant: 'block-deviations',
        bindings: [
          valuesBinding,
          ...maskBindings,
          statisticsBinding('read'),
          centeredBinding('read_write'),
          {
            name: 'deviationPartials',
            view: deviationPartials,
            type: 'f32',
            access: 'read_write'
          }
        ],
        workgroupSize: MOMENT_WORKGROUP_SIZE,
        invocationCount: blockCount * MOMENT_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${declarations}${deviationReductionWGSL}`,
        body: `let blockIndex = index / ${MOMENT_WORKGROUP_SIZE}u;
  let isInRange = index < INVOCATION_COUNT;
  let provisionalMean = statistics[statisticsOffset + 1u];
  var sum = ${deviationSumZero};
  if (isInRange) {
    let begin = blockIndex * BIN_BLOCK;
    let end = min(begin + BIN_BLOCK, BIN_COUNT);
    for (var bin = begin + localInvocationIndex; bin < end; bin += ${MOMENT_WORKGROUP_SIZE}u) {
      let valid = isBinValid(bin, bin / SLICE_COUNT);
      let deviation = readValue(bin) - provisionalMean;
      centeredValues[centeredValuesOffset + bin] = select(getQuietNaN(bin), deviation, valid);
      let term = select(0.0, deviation, valid);
      ${weights ? 'sum = addCompensatedValue(sum, term);' : 'sum += term;'}
    }
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = ${MOMENT_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      ${mergeDeviationLanes}
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    ${
      weights
        ? `let destination = deviationPartialsOffset + 2u * blockIndex;
    deviationPartials[destination] = partialSums[0].x;
    deviationPartials[destination + 1u] = partialSums[0].y;`
        : 'deviationPartials[deviationPartialsOffset + blockIndex] = partialSums[0];'
    }
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-mean-correction`,
        operation: OPERATION,
        variant: 'mean-correction',
        bindings: [
          {
            name: 'deviationPartials',
            view: deviationPartials,
            type: 'f32',
            access: 'read'
          },
          statisticsBinding('read_write'),
          {
            name: 'residualMean',
            view: residualMean,
            type: 'f32',
            access: 'read_write'
          }
        ],
        workgroupSize: MOMENT_WORKGROUP_SIZE,
        invocationCount: MOMENT_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${SPATIAL_STATISTICS_COUNTS_WGSL}
const BLOCK_COUNT: u32 = ${blockCount}u;
${deviationReductionWGSL}`,
        body: `var sum = ${deviationSumZero};
  for (var blockIndex = localInvocationIndex; blockIndex < BLOCK_COUNT; blockIndex += ${MOMENT_WORKGROUP_SIZE}u) {
    ${
      weights
        ? `let source = deviationPartialsOffset + 2u * blockIndex;
    sum = mergeEmergingSums(sum, vec2<f32>(deviationPartials[source], deviationPartials[source + 1u]));`
        : 'sum += deviationPartials[deviationPartialsOffset + blockIndex];'
    }
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = ${MOMENT_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      ${mergeDeviationLanes}
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    let count = (u32(statistics[statisticsOffset + 4u]) << 16u) | u32(statistics[statisticsOffset + 5u]);
    // Mean of the first-pass deviations: added to the reported mean and subtracted from every
    // stored deviation by the next kernel.
    let residual = select(0.0, divideBySpatialCount(${weights ? 'getCompensated(partialSums[0])' : 'partialSums[0]'}, count), count > 0u);
    statistics[statisticsOffset + 1u] += residual;
    residualMean[residualMeanOffset] = residual;
  }`
      }),
      // Reduces the squared centered values of every blockIndex: the centered value is the stored
      // deviation minus the residual mean (read back from memory, not recomputed from the raw
      // value).
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-block-squares`,
        operation: OPERATION,
        variant: 'block-squares',
        bindings: [
          centeredBinding('read'),
          residualMeanBinding,
          {
            name: 'squarePartials',
            view: squarePartials,
            type: 'f32',
            access: 'read_write'
          }
        ],
        workgroupSize: MOMENT_WORKGROUP_SIZE,
        invocationCount: blockCount * MOMENT_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${centeredDeclarations}
var<workgroup> partialSums: array<f32, ${MOMENT_WORKGROUP_SIZE}>;`,
        body: `let blockIndex = index / ${MOMENT_WORKGROUP_SIZE}u;
  let isInRange = index < INVOCATION_COUNT;
  let residual = residualMean[residualMeanOffset];
  var sum = 0.0;
  if (isInRange) {
    let begin = blockIndex * BIN_BLOCK;
    let end = min(begin + BIN_BLOCK, BIN_COUNT);
    for (var bin = begin + localInvocationIndex; bin < end; bin += ${MOMENT_WORKGROUP_SIZE}u) {
      if (isBinValid(bin, bin / SLICE_COUNT)) {
        let centered = readDeviation(bin) - residual;
        sum += centered * centered;
      }
    }
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = ${MOMENT_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    squarePartials[squarePartialsOffset + blockIndex] = partialSums[0];
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-deviation`,
        operation: OPERATION,
        variant: 'deviation',
        bindings: [
          {
            name: 'squarePartials',
            view: squarePartials,
            type: 'f32',
            access: 'read'
          },
          statisticsBinding('read_write')
        ],
        workgroupSize: MOMENT_WORKGROUP_SIZE,
        invocationCount: MOMENT_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${SPATIAL_STATISTICS_COUNTS_WGSL}
const BLOCK_COUNT: u32 = ${blockCount}u;
var<workgroup> partialSums: array<f32, ${MOMENT_WORKGROUP_SIZE}>;`,
        body: `var sum = 0.0;
  for (var blockIndex = localInvocationIndex; blockIndex < BLOCK_COUNT; blockIndex += ${MOMENT_WORKGROUP_SIZE}u) {
    sum += squarePartials[squarePartialsOffset + blockIndex];
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = ${MOMENT_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    let count = (u32(statistics[statisticsOffset + 4u]) << 16u) | u32(statistics[statisticsOffset + 5u]);
    let variance = select(0.0, divideBySpatialCount(partialSums[0], count), count > 0u);
    statistics[statisticsOffset + 2u] = variance;
    statistics[statisticsOffset + 3u] = sqrt(variance);
  }`
      }),
      ...(output.globalStatistics || this.populationCount
        ? [
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-publish-moments`,
              operation: OPERATION,
              invocationCount: output.globalStatistics
                ? GPU_EMERGING_HOT_SPOTS_STATISTICS_LENGTH
                : 1,
              bindings: [
                statisticsBinding('read'),
                ...(output.globalStatistics
                  ? [
                      {
                        name: 'globalStatistics',
                        view: output.globalStatistics,
                        type: 'f32' as const,
                        access: 'read_write' as const
                      }
                    ]
                  : []),
                ...(this.populationCount
                  ? [
                      {
                        name: 'exactCount',
                        view: this.populationCount,
                        type: 'u32' as const,
                        access: 'read_write' as const
                      }
                    ]
                  : [])
              ],
              body: `${output.globalStatistics ? 'globalStatistics[globalStatisticsOffset + index] = statistics[statisticsOffset + index];' : ''}
          ${this.populationCount ? 'if (index == 0u) { exactCount[exactCountOffset] = (u32(statistics[statisticsOffset + 4u]) << 16u) | u32(statistics[statisticsOffset + 5u]); }' : ''}`
            })
          ]
        : []),
      // One Gi* kernel per weights chunk (a whole CSR or the lattice is a single kernel).
      ...(weights ? dispatches : [undefined]).map(dispatch =>
        createSpatialStatisticsCollectiveNode<Parameters>(graph, {
          id: `${id}-gi-star${dispatch?.idSuffix ?? ''}`,
          operation: OPERATION,
          variant: weights ? 'gi-star-weights' : 'gi-star',
          bindings: [
            centeredBinding('read'),
            residualMeanBinding,
            parametersBinding,
            ...(dispatch
              ? ([
                  {
                    name: 'offsets',
                    view: dispatch.weights.offsets,
                    type: 'u32',
                    access: 'read'
                  },
                  {
                    name: 'neighbors',
                    view: dispatch.weights.neighbors,
                    type: 'u32',
                    access: 'read'
                  },
                  {
                    name: 'weights',
                    view: dispatch.weights.weights,
                    type: 'f32',
                    access: 'read'
                  }
                ] satisfies WGSLKernelBinding[])
              : []),
            statisticsBinding('read'),
            {
              name: 'giZScores',
              view: giZScores,
              type: 'f32',
              access: 'read_write'
            }
          ],
          workgroupSize: GI_WORKGROUP_SIZE,
          invocationCount:
            Math.ceil((dispatch?.rowCount ?? cellCount) / cellsPerGroup) * GI_WORKGROUP_SIZE,
          parameterValues: [dispatch?.rowStart ?? 0, dispatch?.rowCount ?? cellCount],
          declarations: `${centeredDeclarations}
const CELLS_PER_GROUP: u32 = ${cellsPerGroup}u;
var<workgroup> instantWeighted: array<${weights ? 'vec2<f32>' : 'f32'}, ${GI_WORKGROUP_SIZE}>;
var<workgroup> instantWeight: array<f32, ${GI_WORKGROUP_SIZE}>;
var<workgroup> instantSquare: array<f32, ${GI_WORKGROUP_SIZE}>;
var<workgroup> instantCount: array<u32, ${GI_WORKGROUP_SIZE}>;
${
  weights
    ? `var<workgroup> instantUnweighted: array<vec2<f32>, ${GI_WORKGROUP_SIZE}>;
var<workgroup> instantOrigin: array<f32, ${GI_WORKGROUP_SIZE}>;
var<workgroup> instantScale: array<f32, ${GI_WORKGROUP_SIZE}>;
${EMERGING_SUM_WGSL}
fn getEmergingCoefficientScale(maximum: f32) -> f32 {
  // Normal power of two with a normal reciprocal, even if the backend lowers division to multiply.
  return bitcast<f32>(clamp(bitcast<u32>(maximum) & 0x7f800000u, 0x00800000u, 0x7e800000u));
}
// Central moments in scaled coefficient-offset coordinates: mean, square sum, square correction.
fn addEmergingWeightMoment(previous: vec3<f32>, count: u32, value: f32) -> vec3<f32> {
  let difference = value - previous.x;
  let mean = previous.x + difference / f32(count);
  let squares = addCompensatedValue(previous.yz, difference * (value - mean));
  return vec3<f32>(mean, squares);
}`
    : ''
}`,
          body: getGiBody()
        })
      ),
      // A lost edge/candidate makes every dependent neighborhood statistic unavailable.
      // Suppress before p-values and trends so no downstream category survives incomplete CSR.
      ...lossViews.map((loss, lossIndex) =>
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-incomplete-aggregate-${lossIndex}`,
          operation: OPERATION,
          variant: 'aggregate-weights-loss',
          invocationCount: 1,
          bindings: [
            {name: 'loss', view: loss, type: 'u32', access: 'read'},
            {name: 'weightsLoss', view: weightsLoss!, type: 'u32', access: 'read_write'}
          ],
          // First flag replaces last frame's aggregate; later flags combine in graph order.
          body: `weightsLoss[weightsLossOffset] ${lossIndex === 0 ? '=' : '|='} select(0u, 1u, loss[lossOffset] != 0u);`
        })
      ),
      ...(weightsLoss
        ? [
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-incomplete-gi`,
              operation: OPERATION,
              variant: 'incomplete-weights-gi',
              invocationCount: binCount,
              bindings: [
                {name: 'weightsLoss', view: weightsLoss, type: 'u32', access: 'read'},
                {name: 'giZScores', view: giZScores, type: 'f32', access: 'read_write'}
              ],
              declarations: GET_QUIET_NAN_SEEDED_WGSL,
              body: `if (weightsLoss[weightsLossOffset] != 0u) {
  giZScores[giZScoresOffset + index] = getQuietNaN(index);
}`
            })
          ]
        : []),
      ...(needsPValues ? getPValueNodes() : []),
      // Hot and cold slice counts and the last finite slice, per cell. A separate pass keeps the
      // Mann-Kendall kernel within the storage buffer limit.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-slice-counts`,
        operation: OPERATION,
        variant: 'slice-counts',
        bindings: [
          parametersBinding,
          {name: 'giZScores', view: giZScores, type: 'f32', access: 'read'},
          ...(usesAdjustment
            ? [
                {
                  name: 'adjustedPValues',
                  view: adjustedPValues!,
                  type: 'f32',
                  access: 'read'
                } satisfies WGSLKernelBinding
              ]
            : []),
          {
            name: 'hotSliceCount',
            view: output.hotSliceCount,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'coldSliceCount',
            view: output.coldSliceCount,
            type: 'u32',
            access: 'read_write'
          },
          ...(output.lastValidSlice
            ? [
                {
                  name: 'lastValidSlice',
                  view: output.lastValidSlice,
                  type: 'u32',
                  access: 'read_write'
                } satisfies WGSLKernelBinding
              ]
            : [])
        ],
        invocationCount: cellCount,
        declarations: `const SLICE_COUNT: u32 = ${sliceCount}u;
${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
${getSliceStateWGSL(usesAdjustment)}`,
        body: `let base = index * SLICE_COUNT;
  ${getSliceCountingWGSL({
    valueExpression: 'giZScores[giZScoresOffset + base + slice]',
    adjustedExpression: 'adjustedPValues[adjustedPValuesOffset + base + slice]',
    usesAdjustment,
    collectSeries: false
  })}
  hotSliceCount[hotSliceCountOffset + index] = hot;
  coldSliceCount[coldSliceCountOffset + index] = cold;
  ${output.lastValidSlice ? 'lastValidSlice[lastValidSliceOffset + index] = lastValid;' : ''}`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-mann-kendall`,
        operation: OPERATION,
        variant: cooperativeMannKendall ? 'mann-kendall-cooperative' : 'mann-kendall-classify',
        bindings: [
          parametersBinding,
          {name: 'giZScores', view: giZScores, type: 'f32', access: 'read'},
          ...(usesAdjustment
            ? [
                {
                  name: 'adjustedPValues',
                  view: adjustedPValues!,
                  type: 'f32',
                  access: 'read'
                } satisfies WGSLKernelBinding
              ]
            : []),
          ...(weightsLoss
            ? [
                {
                  name: 'weightsLoss',
                  view: weightsLoss,
                  type: 'u32',
                  access: 'read'
                } satisfies WGSLKernelBinding
              ]
            : []),
          {
            name: 'trendZ',
            view: output.trendZ,
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'trendP',
            view: output.trendP,
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'trendS',
            view: output.trendS,
            type: 'i32',
            access: 'read_write'
          },
          {
            name: 'category',
            view: output.category,
            type: 'u32',
            access: 'read_write'
          }
        ],
        // One workgroup owns one cell. Pair comparisons are striped across lanes instead of one
        // invocation holding a 256-value private array and serially visiting every pair. Apart
        // from dividing the long dependent chain across 64 lanes, this avoids private-array
        // register pressure and the driver-dependent scratch-memory spill it can cause.
        workgroupSize: cooperativeMannKendall ? MANN_KENDALL_WORKGROUP_SIZE : undefined,
        invocationCount: cooperativeMannKendall
          ? cellCount * MANN_KENDALL_WORKGROUP_SIZE
          : cellCount,
        guardIndex: !cooperativeMannKendall,
        declarations: cooperativeMannKendall
          ? `const CELL_COUNT: u32 = ${cellCount}u;
const SLICE_COUNT: u32 = ${sliceCount}u;
const MANN_KENDALL_LANES: u32 = ${MANN_KENDALL_WORKGROUP_SIZE}u;
var<workgroup> partialStatistics: array<i32, ${MANN_KENDALL_WORKGROUP_SIZE}>;
var<workgroup> partialTieTerms: array<u32, ${MANN_KENDALL_WORKGROUP_SIZE}>;
var<workgroup> sharedSeries: array<f32, ${sliceCount}>;
var<workgroup> tieCandidates: array<u32, ${sliceCount}>;
var<workgroup> equalBeforeBits: array<atomic<u32>, ${Math.ceil(sliceCount / 32)}>;
${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
${getSliceStateWGSL(usesAdjustment)}`
          : `const SLICE_COUNT: u32 = ${sliceCount}u;
${this.trendAutocorrelationCorrection !== 'none' ? getTheilSenSlopeWGSL(sliceCount) : ''}
${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
${getSliceStateWGSL(usesAdjustment)}`,
        body: cooperativeMannKendall
          ? getCooperativeMannKendallBody({
              sliceCount,
              usesAdjustment,
              finalSlice,
              hasWeightsLoss: Boolean(weightsLoss)
            })
          : getSerialMannKendallBody({
              sliceCount,
              hasWeightsLoss: Boolean(weightsLoss),
              usesAdjustment,
              finalSlice,
              trendCorrection: this.trendAutocorrelationCorrection,
              maximumLag: props.trendCorrectionMaximumLag ?? 0,
              criticalZ: props.trendCorrectionCriticalZ ?? DEFAULT_TREND_CORRECTION_CRITICAL_Z
            })
      })
    );
    return nodes;
  }
}
