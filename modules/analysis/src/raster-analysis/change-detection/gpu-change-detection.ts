// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  createRasterTimeSeriesNodes,
  planRasterTimeSeriesChunks,
  validateRasterTimeSeriesViews,
  validateRasterTimeSeriesGraph,
  type RasterWritableDispatch,
  type RasterTimeSeriesBinding as WGSLKernelBinding,
  type RasterTimeSeriesScalarView,
  type RasterTimeSeriesView,
  type RasterTimeSeriesChunkPlan
} from '../raster-time-series/raster-time-series-chunks-b';
import {
  validateCapacity,
  validateSinglePackedView,
  validateCommandNodeBudget,
  getGraphViewChunks
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateRasterParameters} from '../raster-band-utils';
import {GPU_CHANGE_DETECTION_PARAMETER_LENGTH} from './change-detection-parameters';
import {ERFC_VALUE_WGSL, GUARDED_ATAN2_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {NAN_WGSL, STUDENT_T_WGSL} from './change-detection-wgsl';
import type {RasterAlgebraAtLeastOne} from '../raster-algebra/raster-algebra-output';
import {
  getGPURasterDateAxisBinding,
  getGPURasterDateAxisDeclarations,
  resolveGPURasterIrregularDateAxis,
  validateGPURasterDateAxisGraph,
  type GPURasterIrregularDateAxis
} from '../raster-time-series/raster-irregular-date-axis';

const OPERATION = 'GPUChangeDetection';
/** Private temporal arrays and bounded per-cell statistics keep the resident series at 256 slices. */
const MAXIMUM_SLICE_COUNT = 256;
/** Exact pairwise Sen selection is quadratic and keeps a private heap per cell. */
const MAXIMUM_SEN_SLICE_COUNT = 64;
/** The direction code stores two 16-bit sign masks in one uint32. */
const MAXIMUM_DIRECTION_BAND_COUNT = 16;

/** Which statistic decides the `significance` class. */
export type GPUChangeDetectionSignificanceSource = 't-test' | 'mann-kendall';

/**
 * Caller-owned outputs of {@link GPUChangeDetection}. Every output is optional, at least one is
 * required, and each present output selects the work that is compiled in.
 *
 * Single-band outputs have one row per cell. `difference`, `logRatio` and `percentChange` have
 * `cellCount * bandCount` rows indexed `cell * bandCount + band`. Masked cells and cells without
 * enough valid slices write NaN (float outputs), 0 (`mannKendallS`, `significance`) or
 * `0xffffffff` (the multiband direction code).
 */
export type GPUChangeDetectionOutput<Rows extends RasterTimeSeriesScalarView = GraphDataView> =
  RasterAlgebraAtLeastOne<{
    /** `after - before`; NaN when either slice is NaN. */
    difference?: RasterTimeSeriesView<'float32', Rows>;
    /** `ln((after + epsilon) / (before + epsilon))`; NaN when either side is `<= -epsilon`. */
    logRatio?: RasterTimeSeriesView<'float32', Rows>;
    /** `100 * (after - before) / |before|`; NaN when `before` is 0. */
    percentChange?: RasterTimeSeriesView<'float32', Rows>;
    /** Welch t statistic `(mean(after group) - mean(before group)) / se`. Single band only. */
    tStatistic?: RasterTimeSeriesView<'float32', Rows>;
    /** Welch-Satterthwaite degrees of freedom. Single band only. */
    tDegreesOfFreedom?: RasterTimeSeriesView<'float32', Rows>;
    /** Two-sided Student-t p-value of the Welch statistic. Single band only. */
    tPValue?: RasterTimeSeriesView<'float32', Rows>;
    /** Theil-Sen slope: median of `(x_j - x_i) / (j - i)` over valid slice pairs. Single band only. */
    senSlope?: RasterTimeSeriesView<'float32', Rows>;
    /** Mann-Kendall S statistic, integer exact. Single band only. */
    mannKendallS?: RasterTimeSeriesView<'sint32', Rows>;
    /** Tie-corrected Mann-Kendall Z with continuity correction. Single band only. */
    mannKendallZ?: RasterTimeSeriesView<'float32', Rows>;
    /** Two-sided normal p-value of the Mann-Kendall Z. Single band only. */
    mannKendallP?: RasterTimeSeriesView<'float32', Rows>;
    /** Euclidean norm of the band difference vector. Requires `bandCount >= 2`. */
    changeMagnitude?: RasterTimeSeriesView<'float32', Rows>;
    /**
     * Change direction. Float32 `atan2(d1, d0)` in radians (0 for a zero vector) when `bandCount`
     * is 2; uint32 sign-pattern code when `bandCount` is 3 to 16: bit `b` is set when band `b`
     * increased and bit `16 + b` when it decreased.
     */
    changeDirection?: RasterTimeSeriesView<'float32', Rows> | RasterTimeSeriesView<'uint32', Rows>;
    /**
     * Class per cell: 0 no significant change (or not computable), 1 significant increase, 2
     * significant decrease. Increase and decrease follow the sign of the t statistic or of
     * Mann-Kendall S. Single band only; see `significanceSource`.
     */
    significance?: RasterTimeSeriesView<'uint32', Rows>;
  }>;

/**
 * Properties for {@link GPUChangeDetection}.
 *
 * Per-frame (no recompile): the contents of `parameters` and of `slices`, `mask` and `observationMask`.
 * Topology (needs a new graph): `cellCount`, `sliceCount`, `bandCount`, view lengths, which
 * outputs, the mask and the observation mask are present, and `significanceSource`.
 */
export type GPUChangeDetectionProps<Rows extends RasterTimeSeriesScalarView = GraphDataView> = {
  /** Prefix for generated node IDs. Defaults to `'change-detection'`. */
  id?: string;
  /**
   * Dense float32 stack indexed `(cell * sliceCount + slice) * bandCount + band` (cell-major, like
   * `GPUTemporalReduction`'s `cell * bucketCount + bucket`). A sample that is not finite (NaN or
   * an infinity) is missing.
   */
  slices: RasterTimeSeriesView<'float32', Rows>;
  /**
   * Optional per-observation mask, `cell * sliceCount + slice` (for example a per-date cloud
   * mask): zero makes that observation missing in every band, exactly as a NaN sample.
   */
  observationMask?: RasterTimeSeriesView<'uint32', Rows>;
  /** Optional packed `uint32` cell mask with one row per cell; zero skips the cell. */
  mask?: RasterTimeSeriesView<'uint32', Rows>;
  /** Per-frame parameters, float32 of at least 5 elements; see `getGPUChangeDetectionParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Number of cells, compile-time. */
  cellCount: number;
  /** Number of slices per cell, compile-time, in `[2, 256]` (`<= 64` for `senSlope`). */
  sliceCount: number;
  /** Number of bands per slice, compile-time. Defaults to 1. */
  bandCount?: number;
  /** Which statistic drives `output.significance`. Defaults to `'t-test'`. */
  significanceSource?: GPUChangeDetectionSignificanceSource;
  /**
   * Acquisition dates of an irregular stack. `senSlope` then uses elapsed years between acquisitions (slope
   * per year); other outputs depend only on slice order and are unaffected.
   */
  dateAxis?: GPURasterIrregularDateAxis;
  /** Caller-owned outputs; at least one. */
  output: GPUChangeDetectionOutput<Rows>;
};

const SINGLE_BAND_OUTPUTS = [
  'tStatistic',
  'tDegreesOfFreedom',
  'tPValue',
  'senSlope',
  'mannKendallS',
  'mannKendallZ',
  'mannKendallP',
  'significance'
] as const;
const FLOAT_OUTPUTS = [
  'difference',
  'logRatio',
  'percentChange',
  'tStatistic',
  'tDegreesOfFreedom',
  'tPValue',
  'senSlope',
  'mannKendallZ',
  'mannKendallP',
  'changeMagnitude'
] as const;

/**
 * Per-cell change detection over a dense stack of time slices, with the modes selected at
 * compile time by which outputs are present.
 *
 * - Two-slice comparisons (`difference`, `logRatio`, `percentChange`) between the `beforeSlice`
 *   and `afterSlice` parameters, per band.
 * - Stack Welch two-sample t-test between slices `[0, splitSlice)` and `[splitSlice, sliceCount)`
 *   (`tStatistic`, `tDegreesOfFreedom`, `tPValue`). NaN slices are dropped; each group needs at
 *   least 2 valid slices and a positive standard error, otherwise the outputs are NaN. Variances
 *   use the two-pass (mean, then centered sum of squares) algorithm. The two-sided p-value is
 *   `I_x(df / 2, 1 / 2)` with `x = df / (df + t^2)`, evaluated by the regularized incomplete beta
 *   continued fraction (at most 64 iterations, early exit at 3e-7) and a Stirling log-gamma, in
 *   float32; accuracy is about 1e-4 absolute for `df` up to 254.
 * - Theil-Sen slope (`senSlope`): the exact median of the `n (n - 1) / 2` pairwise slopes over
 *   the `n` valid slices, found by a deterministic in-thread bounded max-heap selection (no sort
 *   of all slopes, no atomics). Cost is `O(T^2 log T)` per cell, so `sliceCount` is limited to 64.
 * - Mann-Kendall (`mannKendallS`, `mannKendallZ`, `mannKendallP`): `S = sum sign(x_j - x_i)` over
 *   valid pairs in `i32`, tie-corrected variance `(n (n - 1) (2 n + 5) - sum t (t - 1) (2 t + 5))
 *   / 18` computed in `u32`, `Z = (S - sign(S)) / sqrt(var)`, and `p = erfc(|Z| / sqrt(2))` with a
 *   fractional error below 1.2e-7. With fewer than 2 valid slices Z and p are NaN; with zero
 *   variance (all values tied) Z is 0 and p is 1.
 * - Multiband change vector (`changeMagnitude`, `changeDirection`, `bandCount >= 2`) between the
 *   before and after slices.
 * - `significance`: 0, 1 (increase) or 2 (decrease) when the p-value of the t-test (default) or of
 *   Mann-Kendall (`significanceSource`) is below `alpha`; the sign comes from the statistic.
 *
 * Determinism: each cell is computed by one thread with fixed-order loops, so results do not
 * depend on GPU scheduling. Cell-major views accept matching chunks of complete cell records;
 * shared tables stay packed.
 */
export class GPUChangeDetection<Rows extends RasterTimeSeriesScalarView = GraphDataView>
  implements GPUCommandNodeProducer
{
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUChangeDetectionProps<Rows>;
  readonly chunkPlan: RasterTimeSeriesChunkPlan;
  readonly estimatedCommandNodeCount: number;
  /** Resolved band count. */
  readonly bandCount: number;
  /** Resolved significance source. */
  readonly significanceSource: GPUChangeDetectionSignificanceSource;
  private readonly ticksPerYear: number;

  constructor(props: GPUChangeDetectionProps<Rows> & {slices: Rows}) {
    props = snapshotConstructionFields(props, [
      'id',
      'slices',
      'parameters',
      'cellCount',
      'sliceCount',
      'bandCount',
      'mask',
      'observationMask',
      'significanceSource',
      'dateAxis',
      'output'
    ]);
    props = Object.freeze({
      ...props,
      output: snapshotConstructionFields(props.output ?? {}, [
        'difference',
        'logRatio',
        'percentChange',
        'tStatistic',
        'tDegreesOfFreedom',
        'tPValue',
        'senSlope',
        'mannKendallS',
        'mannKendallZ',
        'mannKendallP',
        'changeMagnitude',
        'changeDirection',
        'significance'
      ])
    });
    this.id = props.id ?? 'change-detection';
    this.props = props;
    const id = this.id;
    const {cellCount, sliceCount, output} = props;
    const bandCount = props.bandCount ?? 1;
    this.bandCount = bandCount;
    this.significanceSource = props.significanceSource ?? 't-test';
    this.ticksPerYear = props.dateAxis
      ? resolveGPURasterIrregularDateAxis(id, props.dateAxis, sliceCount)
      : 0;
    validateCapacity(id, 'cellCount', cellCount, 1, 0x7fffffff);
    validateCapacity(id, 'bandCount', bandCount, 1, 0x7fffffff);
    validateCapacity(id, 'sliceCount', sliceCount, 2, MAXIMUM_SLICE_COUNT);
    if (cellCount * sliceCount * bandCount > 0x7fffffff) {
      throw new Error(`${id} cellCount * sliceCount * bandCount must not exceed 2^31 - 1`);
    }
    if (this.significanceSource !== 't-test' && this.significanceSource !== 'mann-kendall') {
      throw new Error(`${id} significanceSource must be 't-test' or 'mann-kendall'`);
    }
    const validateCellViews = (
      view: RasterTimeSeriesScalarView,
      formats: readonly ('float32' | 'uint32' | 'sint32')[],
      name: string
    ) => {
      for (const chunk of getGraphViewChunks(view)) validateSinglePackedView(chunk, formats, name);
    };
    validateCellViews(props.slices, ['float32'], `${id} slices`);
    if (props.slices.length < cellCount * sliceCount * bandCount) {
      throw new Error(`${id} slices must hold cellCount * sliceCount * bandCount rows`);
    }
    if (props.mask) {
      validateCellViews(props.mask, ['uint32'], `${id} mask`);
      if (props.mask.length < cellCount) {
        throw new Error(`${id} mask must hold cellCount rows`);
      }
    }
    if (props.observationMask) {
      validateCellViews(props.observationMask, ['uint32'], `${id} observationMask`);
      if (props.observationMask.length < cellCount * sliceCount) {
        throw new Error(`${id} observationMask must hold cellCount * sliceCount rows`);
      }
    }
    validateRasterParameters(id, props.parameters, GPU_CHANGE_DETECTION_PARAMETER_LENGTH);
    const present = Object.entries(output).filter(([, view]) => view);
    if (present.length === 0) {
      throw new Error(`${id} needs at least one output`);
    }
    for (const name of FLOAT_OUTPUTS) {
      if (output[name]) {
        validateCellViews(output[name], ['float32'], `${id} output.${name}`);
      }
    }
    if (output.mannKendallS) {
      validateCellViews(output.mannKendallS, ['sint32'], `${id} output.mannKendallS`);
    }
    if (output.significance) {
      validateCellViews(output.significance, ['uint32'], `${id} output.significance`);
    }
    for (const name of SINGLE_BAND_OUTPUTS) {
      if (output[name] && bandCount !== 1) {
        throw new Error(`${id} output.${name} requires bandCount 1`);
      }
    }
    if (output.senSlope && sliceCount > MAXIMUM_SEN_SLICE_COUNT) {
      throw new Error(`${id} output.senSlope supports at most ${MAXIMUM_SEN_SLICE_COUNT} slices`);
    }
    if (output.changeMagnitude || output.changeDirection) {
      if (bandCount < 2 || bandCount > MAXIMUM_DIRECTION_BAND_COUNT) {
        throw new Error(
          `${id} changeMagnitude and changeDirection need bandCount in [2, ${MAXIMUM_DIRECTION_BAND_COUNT}]`
        );
      }
    }
    if (output.changeDirection) {
      validateCellViews(
        output.changeDirection,
        [bandCount === 2 ? 'float32' : 'uint32'],
        `${id} output.changeDirection`
      );
    }
    for (const [name, view] of present) {
      const required =
        name === 'difference' || name === 'logRatio' || name === 'percentChange'
          ? cellCount * bandCount
          : cellCount;
      if ((view as GraphDataView).length < required) {
        throw new Error(`${id} output.${name} must hold ${required} rows`);
      }
    }
    this.chunkPlan = planRasterTimeSeriesChunks(id, cellCount, {
      slices: {view: props.slices, entriesPerCell: sliceCount * bandCount},
      mask: {view: props.mask, entriesPerCell: 1},
      observationMask: {view: props.observationMask, entriesPerCell: sliceCount},
      ...Object.fromEntries(
        present.map(([name, view]) => [
          name,
          {
            view,
            entriesPerCell: ['difference', 'logRatio', 'percentChange'].includes(name)
              ? bandCount
              : 1
          }
        ])
      )
    });
    const dispatches = this.getDispatches();
    const statisticCount = dispatches.length;
    this.estimatedCommandNodeCount = statisticCount * this.chunkPlan.estimatedCommandNodeCount;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateRasterTimeSeriesViews(
      id,
      this.getInputs(),
      Object.values(output).filter(view => view !== undefined),
      this.chunkPlan,
      dispatches
    );
  }

  private getDispatches(): RasterWritableDispatch[] {
    const {props} = this;
    const {output} = props;
    const present = (views: readonly (RasterTimeSeriesScalarView | undefined)[]) =>
      views.filter((view): view is RasterTimeSeriesScalarView => view !== undefined);
    const common = present([props.slices, props.mask, props.observationMask]);
    const parameterInputs = [...common, props.parameters];
    return [
      {
        inputs: parameterInputs,
        outputs: present([output.difference, output.logRatio, output.percentChange])
      },
      {inputs: parameterInputs, outputs: present([output.changeMagnitude, output.changeDirection])},
      {
        inputs: parameterInputs,
        outputs: present([
          output.tStatistic,
          output.tDegreesOfFreedom,
          output.tPValue,
          this.significanceSource === 't-test' ? output.significance : undefined
        ])
      },
      {
        inputs: present([...common, props.dateAxis?.acquisitionTimes]),
        outputs: present([output.senSlope])
      },
      {
        inputs: parameterInputs,
        outputs: present([
          output.mannKendallS,
          output.mannKendallZ,
          output.mannKendallP,
          this.significanceSource === 'mann-kendall' ? output.significance : undefined
        ])
      }
    ].filter(dispatch => dispatch.outputs.length > 0);
  }

  private getInputs(): RasterTimeSeriesScalarView[] {
    return [...new Set(this.getDispatches().flatMap(dispatch => dispatch.inputs))];
  }

  /** Returns the two-slice, multiband, t-test, Sen slope and Mann-Kendall nodes that are needed. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, bandCount} = this;
    const {output, cellCount, sliceCount} = props;
    validateRasterTimeSeriesGraph(
      id,
      graph,
      this.getInputs(),
      Object.values(output).filter(view => view !== undefined)
    );
    if (props.dateAxis && output.senSlope)
      validateGPURasterDateAxisGraph(id, graph, props.dateAxis);
    const read = (name: string, view: RasterTimeSeriesScalarView): WGSLKernelBinding => ({
      name,
      view,
      type: 'f32',
      access: 'read'
    });
    const common = (): WGSLKernelBinding[] => [
      read('slices', props.slices),
      ...(props.mask
        ? [
            {
              name: 'cellMask',
              view: props.mask,
              type: 'u32',
              access: 'read'
            } as const
          ]
        : []),
      ...(props.observationMask
        ? [
            {
              name: 'observationMask',
              view: props.observationMask,
              type: 'u32',
              access: 'read'
            } as const
          ]
        : [])
    ];
    const write = (
      name: string,
      view: RasterTimeSeriesScalarView,
      type: 'f32' | 'u32' | 'i32'
    ): WGSLKernelBinding => ({name, view, type, access: 'read_write'});
    const declarations = `const SLICE_COUNT: u32 = ${sliceCount}u;
const BAND_COUNT: u32 = ${bandCount}u;
${NAN_WGSL}
// Per-observation mask (all bands of one cell and slice); true when no view is bound.
fn isObservationValid(cell: u32, slice: u32) -> bool {
  return ${props.observationMask ? 'observationMask[observationMaskOffset + cell * SLICE_COUNT + slice] != 0u' : 'true'};
}
// The sample at (cell, slice, band) is missing: not finite, or its observation is invalid.
fn isMissingAt(cell: u32, slice: u32, value: f32) -> bool {
  return isMissingSample(value) || !isObservationValid(cell, slice);
}`;
    const maskGuard = props.mask
      ? 'if (cellMask[cellMaskOffset + cell] == 0u) {\n    return;\n  }'
      : '';
    const sliceIndex = (slice: string, band = '0u') =>
      `(cell * SLICE_COUNT + ${slice}) * BAND_COUNT + ${band}`;
    const nodes: GPUCommandNode<Parameters>[] = [];

    // Two-slice comparisons, one thread per (cell, band).
    if (output.difference || output.logRatio || output.percentChange) {
      const bindings = [...common(), read('params', props.parameters)];
      const defaults: string[] = [];
      const compute: string[] = [];
      if (output.difference) {
        bindings.push(write('differenceOut', output.difference, 'f32'));
        defaults.push('differenceOut[differenceOutOffset + index] = quietNan();');
        compute.push('differenceOut[differenceOutOffset + index] = after - before;');
      }
      if (output.logRatio) {
        bindings.push(write('logRatioOut', output.logRatio, 'f32'));
        defaults.push('logRatioOut[logRatioOutOffset + index] = quietNan();');
        compute.push(`if (after > -epsilon && before > -epsilon) {
    logRatioOut[logRatioOutOffset + index] = log((after + epsilon) / (before + epsilon));
  }`);
      }
      if (output.percentChange) {
        bindings.push(write('percentChangeOut', output.percentChange, 'f32'));
        defaults.push('percentChangeOut[percentChangeOutOffset + index] = quietNan();');
        compute.push(`if (before != 0.0) {
    percentChangeOut[percentChangeOutOffset + index] = 100.0 * (after - before) / abs(before);
  }`);
      }
      nodes.push(
        ...createRasterTimeSeriesNodes<Parameters>(graph, this.chunkPlan, {
          id: `${id}-two-slice`,
          operation: OPERATION,
          variant: 'two-slice',
          bindings,
          invocationCount: cellCount * bandCount,
          declarations,
          body: `let cell = index / BAND_COUNT;
  let band = index % BAND_COUNT;
  ${defaults.join('\n  ')}
  ${maskGuard}
  let beforeSlice = params[paramsOffset];
  let afterSlice = params[paramsOffset + 1u];
  let epsilon = params[paramsOffset + 2u];
  if (!(beforeSlice >= 0.0 && beforeSlice < f32(SLICE_COUNT) && afterSlice >= 0.0 && afterSlice < f32(SLICE_COUNT))) {
    return;
  }
  let before = slices[slicesOffset + ${sliceIndex('u32(beforeSlice)', 'band')}];
  let after = slices[slicesOffset + ${sliceIndex('u32(afterSlice)', 'band')}];
  if (isMissingAt(cell, u32(beforeSlice), before) || isMissingAt(cell, u32(afterSlice), after)) {
    return;
  }
  ${compute.join('\n  ')}`
        })
      );
    }

    // Multiband change vector, one thread per cell.
    if (output.changeMagnitude || output.changeDirection) {
      const bindings = [...common(), read('params', props.parameters)];
      const defaults: string[] = [];
      const compute: string[] = [];
      if (output.changeMagnitude) {
        bindings.push(write('magnitudeOut', output.changeMagnitude, 'f32'));
        defaults.push('magnitudeOut[magnitudeOutOffset + cell] = quietNan();');
        compute.push('magnitudeOut[magnitudeOutOffset + cell] = sqrt(squaredNorm);');
      }
      if (output.changeDirection) {
        const isAngle = bandCount === 2;
        bindings.push(write('directionOut', output.changeDirection, isAngle ? 'f32' : 'u32'));
        defaults.push(
          `directionOut[directionOutOffset + cell] = ${isAngle ? 'quietNan()' : '0xffffffffu'};`
        );
        compute.push(
          isAngle
            ? `if (differences[0] != 0.0 || differences[1] != 0.0) {
    directionOut[directionOutOffset + cell] = guardedAtan2(differences[1], differences[0]);
  } else {
    directionOut[directionOutOffset + cell] = 0.0;
  }`
            : `directionOut[directionOutOffset + cell] = code;`
        );
      }
      nodes.push(
        ...createRasterTimeSeriesNodes<Parameters>(graph, this.chunkPlan, {
          id: `${id}-multiband`,
          operation: OPERATION,
          variant: 'multiband',
          bindings,
          invocationCount: cellCount,
          declarations: `${declarations}\n${GUARDED_ATAN2_WGSL}`,
          body: `let cell = index;
  ${defaults.join('\n  ')}
  ${maskGuard}
  let beforeSlice = params[paramsOffset];
  let afterSlice = params[paramsOffset + 1u];
  if (!(beforeSlice >= 0.0 && beforeSlice < f32(SLICE_COUNT) && afterSlice >= 0.0 && afterSlice < f32(SLICE_COUNT))) {
    return;
  }
  var differences: array<f32, ${bandCount}>;
  var squaredNorm = 0.0;
  var code = 0u;
  for (var band = 0u; band < BAND_COUNT; band++) {
    let before = slices[slicesOffset + ${sliceIndex('u32(beforeSlice)', 'band')}];
    let after = slices[slicesOffset + ${sliceIndex('u32(afterSlice)', 'band')}];
    if (isMissingAt(cell, u32(beforeSlice), before) || isMissingAt(cell, u32(afterSlice), after)) {
      return;
    }
    let difference = after - before;
    differences[band] = difference;
    squaredNorm += difference * difference;
    if (difference > 0.0) {
      code |= 1u << band;
    } else if (difference < 0.0) {
      code |= 1u << (16u + band);
    }
  }
  ${compute.join('\n  ')}`
        })
      );
    }

    // Welch t-test, one thread per cell.
    const hasTOutput = output.tStatistic || output.tDegreesOfFreedom || output.tPValue;
    const significanceFromTest = output.significance && this.significanceSource === 't-test';
    if (hasTOutput || significanceFromTest) {
      const bindings = [...common(), read('params', props.parameters)];
      const defaults: string[] = [];
      const compute: string[] = [];
      for (const [name, view, key] of [
        ['tStatisticOut', output.tStatistic, 't'],
        ['tDegreesOut', output.tDegreesOfFreedom, 'degreesOfFreedom'],
        ['tPValueOut', output.tPValue, 'pValue']
      ] as const) {
        if (view) {
          bindings.push(write(name, view, 'f32'));
          defaults.push(`${name}[${name}Offset + cell] = quietNan();`);
          compute.push(`${name}[${name}Offset + cell] = ${key};`);
        }
      }
      if (significanceFromTest) {
        bindings.push(write('significanceOut', output.significance!, 'u32'));
        defaults.push('significanceOut[significanceOutOffset + cell] = 0u;');
        compute.push(`if (pValue < alpha) {
    significanceOut[significanceOutOffset + cell] = select(2u, 1u, t > 0.0);
  }`);
      }
      nodes.push(
        ...createRasterTimeSeriesNodes<Parameters>(graph, this.chunkPlan, {
          id: `${id}-t-test`,
          operation: OPERATION,
          variant: 't-test',
          bindings,
          invocationCount: cellCount,
          declarations: `${declarations}\n${STUDENT_T_WGSL}`,
          body: `let cell = index;
  ${defaults.join('\n  ')}
  ${maskGuard}
  let alpha = params[paramsOffset + 3u];
  let splitSlice = params[paramsOffset + 4u];
  if (!(splitSlice >= 1.0 && splitSlice <= f32(SLICE_COUNT - 1u))) {
    return;
  }
  let split = u32(splitSlice);
  var count1 = 0u;
  var count2 = 0u;
  var origin1 = 0.0;
  var origin2 = 0.0;
  var sum1 = 0.0;
  var sum2 = 0.0;
  for (var slice = 0u; slice < SLICE_COUNT; slice++) {
    let value = slices[slicesOffset + ${sliceIndex('slice')}];
    if (isMissingAt(cell, slice, value)) {
      continue;
    }
    if (slice < split) {
      if (count1 == 0u) {
        origin1 = value;
      }
      count1++;
      sum1 += value - origin1;
    } else {
      if (count2 == 0u) {
        origin2 = value;
      }
      count2++;
      sum2 += value - origin2;
    }
  }
  if (count1 < 2u || count2 < 2u) {
    return;
  }
  // Center before accumulating: constant groups have exactly zero residuals even
  // when division by their non-power-of-two sample counts is rounded by the GPU.
  let mean1 = sum1 / f32(count1);
  let mean2 = sum2 / f32(count2);
  var squares1 = 0.0;
  var squares2 = 0.0;
  for (var slice = 0u; slice < SLICE_COUNT; slice++) {
    let value = slices[slicesOffset + ${sliceIndex('slice')}];
    if (isMissingAt(cell, slice, value)) {
      continue;
    }
    if (slice < split) {
      let residual = (value - origin1) - mean1;
      squares1 += residual * residual;
    } else {
      let residual = (value - origin2) - mean2;
      squares2 += residual * residual;
    }
  }
  let errorSquared1 = squares1 / f32(count1 - 1u) / f32(count1);
  let errorSquared2 = squares2 / f32(count2 - 1u) / f32(count2);
  let errorSquared = errorSquared1 + errorSquared2;
  let denominator = errorSquared1 * errorSquared1 / f32(count1 - 1u)
    + errorSquared2 * errorSquared2 / f32(count2 - 1u);
  if (!(errorSquared > 0.0 && denominator > 0.0)) {
    return;
  }
  let t = ((origin2 - origin1) + (mean2 - mean1)) / sqrt(errorSquared);
  let degreesOfFreedom = errorSquared * errorSquared / denominator;
  let pValue = studentTTwoSidedP(t, degreesOfFreedom);
  ${compute.join('\n  ')}`
        })
      );
    }

    // Theil-Sen slope, one thread per cell with a bounded max-heap selection.
    if (output.senSlope) {
      const maximumPairs = (sliceCount * (sliceCount - 1)) / 2;
      const heapCapacity = Math.floor(maximumPairs / 2) + 1;
      nodes.push(
        ...createRasterTimeSeriesNodes<Parameters>(graph, this.chunkPlan, {
          id: `${id}-sen-slope`,
          operation: OPERATION,
          variant: 'sen-slope',
          workgroupSize: 64,
          bindings: [
            ...common(),
            ...(props.dateAxis ? [getGPURasterDateAxisBinding(props.dateAxis)] : []),
            write('senOut', output.senSlope, 'f32')
          ],
          invocationCount: cellCount,
          declarations: props.dateAxis
            ? `${declarations}\n${getGPURasterDateAxisDeclarations(this.ticksPerYear, sliceCount)}`
            : declarations,
          body: `let cell = index;
  senOut[senOutOffset + cell] = quietNan();
  ${maskGuard}
  var validCount = 0u;
  for (var slice = 0u; slice < SLICE_COUNT; slice++) {
    if (!isMissingAt(cell, slice, slices[slicesOffset + ${sliceIndex('slice')}])) {
      validCount++;
    }
  }
  if (validCount < 2u) {
    return;
  }
  let pairCount = validCount * (validCount - 1u) / 2u;
  let heapSize = pairCount / 2u + 1u;
  var heap: array<f32, ${heapCapacity}>;
  var size = 0u;
  for (var i = 0u; i < SLICE_COUNT; i++) {
    let valueI = slices[slicesOffset + ${sliceIndex('i')}];
    if (isMissingAt(cell, i, valueI)) {
      continue;
    }
    for (var j = i + 1u; j < SLICE_COUNT; j++) {
      let valueJ = slices[slicesOffset + ${sliceIndex('j')}];
      if (isMissingAt(cell, j, valueJ)) {
        continue;
      }
      let slope = (valueJ - valueI) / ${props.dateAxis ? 'rasterYearsBetween(j, i)' : 'f32(j - i)'};
      if (size < heapSize) {
        var child = size;
        size++;
        heap[child] = slope;
        loop {
          if (child == 0u) { break; }
          let parent = (child - 1u) / 2u;
          if (heap[parent] >= heap[child]) { break; }
          let swapped = heap[parent];
          heap[parent] = heap[child];
          heap[child] = swapped;
          child = parent;
        }
      } else if (slope < heap[0]) {
        heap[0] = slope;
        var parent = 0u;
        loop {
          let left = 2u * parent + 1u;
          if (left >= size) { break; }
          var larger = left;
          if (left + 1u < size && heap[left + 1u] > heap[left]) { larger = left + 1u; }
          if (heap[parent] >= heap[larger]) { break; }
          let swapped = heap[parent];
          heap[parent] = heap[larger];
          heap[larger] = swapped;
          parent = larger;
        }
      }
    }
  }
  var median = heap[0];
  if (pairCount % 2u == 0u) {
    var secondLargest = heap[1];
    if (size > 2u) { secondLargest = max(secondLargest, heap[2]); }
    median = 0.5 * (heap[0] + secondLargest);
  }
  senOut[senOutOffset + cell] = median;`
        })
      );
    }

    // Mann-Kendall, one thread per cell.
    const hasMannKendallOutput = output.mannKendallS || output.mannKendallZ || output.mannKendallP;
    const significanceFromMannKendall =
      output.significance && this.significanceSource === 'mann-kendall';
    if (hasMannKendallOutput || significanceFromMannKendall) {
      const bindings = [...common()];
      const defaults: string[] = [];
      const compute: string[] = [];
      if (significanceFromMannKendall) {
        bindings.push(read('params', props.parameters));
      }
      if (output.mannKendallS) {
        bindings.push(write('sOut', output.mannKendallS, 'i32'));
        defaults.push('sOut[sOutOffset + cell] = 0;');
        compute.push('sOut[sOutOffset + cell] = score;');
      }
      if (output.mannKendallZ) {
        bindings.push(write('zOut', output.mannKendallZ, 'f32'));
        defaults.push('zOut[zOutOffset + cell] = quietNan();');
        compute.push('zOut[zOutOffset + cell] = z;');
      }
      if (output.mannKendallP) {
        bindings.push(write('pOut', output.mannKendallP, 'f32'));
        defaults.push('pOut[pOutOffset + cell] = quietNan();');
        compute.push('pOut[pOutOffset + cell] = pValue;');
      }
      if (significanceFromMannKendall) {
        bindings.push(write('significanceOut', output.significance!, 'u32'));
        defaults.push('significanceOut[significanceOutOffset + cell] = 0u;');
        compute.push(`if (pValue < params[paramsOffset + 3u] && score != 0) {
    significanceOut[significanceOutOffset + cell] = select(2u, 1u, score > 0);
  }`);
      }
      nodes.push(
        ...createRasterTimeSeriesNodes<Parameters>(graph, this.chunkPlan, {
          id: `${id}-mann-kendall`,
          operation: OPERATION,
          variant: 'mann-kendall',
          workgroupSize: 64,
          bindings,
          invocationCount: cellCount,
          declarations: `${declarations}\n${ERFC_VALUE_WGSL}`,
          body: `let cell = index;
  ${defaults.join('\n  ')}
  ${maskGuard}
  var validCount = 0u;
  var score = 0;
  var tieTerm = 0u;
  for (var i = 0u; i < SLICE_COUNT; i++) {
    let valueI = slices[slicesOffset + ${sliceIndex('i')}];
    if (isMissingAt(cell, i, valueI)) {
      continue;
    }
    validCount++;
    var seenBefore = false;
    for (var k = 0u; k < i; k++) {
      // An earlier observation that is masked out does not start a tie group.
      if (isObservationValid(cell, k) && slices[slicesOffset + ${sliceIndex('k')}] == valueI) {
        seenBefore = true;
      }
    }
    var tied = 1u;
    for (var j = i + 1u; j < SLICE_COUNT; j++) {
      let valueJ = slices[slicesOffset + ${sliceIndex('j')}];
      if (isMissingAt(cell, j, valueJ)) {
        continue;
      }
      if (valueJ > valueI) {
        score += 1;
      } else if (valueJ < valueI) {
        score -= 1;
      } else {
        tied++;
      }
    }
    if (!seenBefore && tied > 1u) {
      tieTerm += tied * (tied - 1u) * (2u * tied + 5u);
    }
  }
  if (validCount < 2u) {
    return;
  }
  let varianceNumerator = validCount * (validCount - 1u) * (2u * validCount + 5u) - tieTerm;
  var z = 0.0;
  var pValue = 1.0;
  if (varianceNumerator > 0u) {
    let standardDeviation = sqrt(f32(varianceNumerator) / 18.0);
    let correctedScore = select(select(0, score + 1, score < 0), score - 1, score > 0);
    z = f32(correctedScore) / standardDeviation;
    pValue = erfcValue(abs(z) * 0.70710678118);
  }
  ${compute.join('\n  ')}`
        })
      );
    }
    return nodes;
  }
}
