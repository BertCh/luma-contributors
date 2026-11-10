// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GraphDataView,
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  RASTER_FINITE_WGSL,
  getRasterLeastSquaresWGSL,
  rasterBinding,
  validateRasterGraph,
  validateRasterViews,
  type RasterScalarView
} from '../remote-sensing/raster-kernel-utils';
import {
  getGPURasterDateAxisBinding,
  getGPURasterDateAxisDeclarations,
  resolveGPURasterIrregularDateAxis,
  validateGPURasterDateAxisGraph,
  type GPURasterIrregularDateAxis
} from './raster-irregular-date-axis';
import {getRasterHarmonicTrendDesignWGSL} from './raster-harmonic-trend-design-wgsl';

/** Slices: private arrays hold one entry per slice. */
export const MAXIMUM_COLD_MONITOR_SLICE_COUNT = 256;
/** Bands: per-thread coefficient and residual arrays grow with the band count. */
export const MAXIMUM_COLD_MONITOR_BAND_COUNT = 8;
/** Segments per cell: the fixed-size output record is `maximumSegmentCount` long. */
export const MAXIMUM_COLD_MONITOR_SEGMENT_COUNT = 32;
/** Peek window: per-thread residual storage grows with it. */
export const MAXIMUM_COLD_MONITOR_CONSECUTIVE_COUNT = 16;
/** Per-thread working memory in floats (8 KiB), the bound shared with the seasonal-break fit. */
const COLD_MONITOR_WORKING_MEMORY_FLOAT_COUNT = 2048;
/** Per-cell multiply budget (2^24), which keeps one invocation under the watchdog time. */
const COLD_MONITOR_WORK_BUDGET = 16777216;
/** Coefficient columns in the output record: intercept, slope and three harmonics. */
export const GPU_RASTER_COLD_MONITOR_COEFFICIENT_COLUMN_COUNT = 8;
/** Scalar per-frame parameters before the per-band RMSE floors. */
export const GPU_RASTER_COLD_MONITOR_BASE_PARAMETER_LENGTH = 6;
/** Words of the per-segment record in `output.segments`. */
export const GPU_RASTER_COLD_MONITOR_SEGMENT_WORD_COUNT = 4;
/** `output.segments` break word when a segment ended without a confirmed change. */
export const GPU_RASTER_COLD_MONITOR_NO_BREAK = 0xffffffff;

/** Per-frame settings of {@link GPURasterCOLDMonitor}. */
export type GPURasterCOLDMonitorSettings = {
  /** Base period of the harmonics in fractional years; 1 for an annual cycle. */
  period?: number;
  /** Chi-squared quantile of the band count that flags one observation as anomalous (COLD: probability 0.99). */
  changeThreshold: number;
  /** Larger chi-squared quantile above which one observation is skipped as an outlier (COLD: `1 - 1e-6`). */
  outlierThreshold: number;
  /** Smallest mean cosine between consecutive normalized residual vectors of the peek window; defaults to `cos(45 deg)`. */
  minimumMeanCosine?: number;
  /** Smallest span of an initialization window in fractional years; defaults to 1. */
  minimumSpanYears?: number;
  /** A gap of at least this many years in an initialization window shifts it; defaults to 1. */
  maximumInitializationGapYears?: number;
  /** Floor per band for the RMSE that normalizes residuals, in band units. Replaces COLD's variogram floor. */
  minimumRMSE: readonly number[];
};

/**
 * Packs {@link GPURasterCOLDMonitorSettings}: `[period, changeThreshold, outlierThreshold,
 * minimumMeanCosine, minimumSpanYears, maximumInitializationGapYears, minimumRMSE...]`.
 */
export function getGPURasterCOLDMonitorParameterValues(
  settings: GPURasterCOLDMonitorSettings,
  target: Float32Array = new Float32Array(
    GPU_RASTER_COLD_MONITOR_BASE_PARAMETER_LENGTH + settings.minimumRMSE.length
  )
): Float32Array {
  const id = 'getGPURasterCOLDMonitorParameterValues';
  validateFiniteNumber(id, 'changeThreshold', settings.changeThreshold, 0);
  validateFiniteNumber(id, 'outlierThreshold', settings.outlierThreshold, 0);
  target[0] = settings.period ?? 1;
  target[1] = settings.changeThreshold;
  target[2] = settings.outlierThreshold;
  target[3] = settings.minimumMeanCosine ?? Math.SQRT1_2;
  target[4] = settings.minimumSpanYears ?? 1;
  target[5] = settings.maximumInitializationGapYears ?? 1;
  target.set(settings.minimumRMSE, GPU_RASTER_COLD_MONITOR_BASE_PARAMETER_LENGTH);
  return target;
}

/** Properties of {@link GPURasterCOLDMonitor}. */
export type GPURasterCOLDMonitorProps = {
  id?: string;
  cellCount: number;
  sliceCount: number;
  bandCount: number;
  /** Band-interleaved `(cell * sliceCount + slice) * bandCount + band`; nonfinite samples are missing. */
  values: GraphDataView<'float32'>;
  /** Per-observation QA, `cell * sliceCount + slice`; zero rejects the whole observation. */
  validity?: GraphDataView<'uint32'>;
  /** Float32, see {@link getGPURasterCOLDMonitorParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Acquisition dates; the model is defined on real elapsed years. */
  dateAxis: GPURasterIrregularDateAxis;
  /** Observations of an initialization window; defaults to 12, the COLD setting. */
  minimumObservations?: number;
  /** Anomalous observations in a row that confirm a change; 2 to 16, defaults to 6, the COLD setting. */
  consecutiveCount?: number;
  /** Largest model: 4, 6 or 8 coefficients; defaults to 8. */
  maximumCoefficientCount?: 4 | 6 | 8;
  /** Output segments per cell; 1 to 32, defaults to 8. */
  maximumSegmentCount?: number;
  /** All-bands refits per cell before the cell stops; defaults to 96. */
  maximumFitCount?: number;
  output: {
    /** `(cell * maximumSegmentCount + segment) * 4 + word`: start, end, break slice ({@link GPU_RASTER_COLD_MONITOR_NO_BREAK} if none), coefficient count. Unused segments hold `0xffffffff`. */
    segments: GraphDataView<'uint32'>;
    /** `((cell * maximumSegmentCount + segment) * bandCount + band) * 8 + column`: intercept at the axis midpoint, slope per year, then cosine and sine per harmonic. Unused columns are zero, unused segments NaN. */
    coefficients: GraphDataView<'float32'>;
    /** `((cell * maximumSegmentCount + segment) * bandCount + band) * 2 + field`: field 0 the RMSE of the last fit, field 1 the median residual over the confirming observations (NaN when no break). NaN for unused segments. */
    segmentStatistics: GraphDataView<'float32'>;
    /**
     * Three words per cell: `[segmentCount, status, fitCount]`. `segmentCount` is the published
     * segments, `fitCount` the all-bands fits used. `status` is a bit field. 1: no segment could
     * be initialized. 2: a fit was rank deficient and the cell stopped. 8: `maximumFitCount` was
     * reached and the cell stopped, publishing what it has (`iterationLimitReached`). 16: more
     * segments were needed than `maximumSegmentCount` (`overflow`). Zero means the series was
     * fully processed.
     */
    cellSummary: GraphDataView<'uint32'>;
  };
};

/**
 * A simplified COLD monitor (Zhu et al. 2020, in the CCDC family of Zhu and Woodcock 2014): per
 * cell, recursive multi-band seasonal-trend segmentation on an irregular date axis, by ordinary
 * least squares. An observation window initializes a segment when it passes the stability test;
 * later observations are tested against the model and a change is confirmed by
 * `consecutiveCount` anomalies with a mean included angle under the threshold.
 *
 * What it omits against pyxccd's `cold_detect`, each documented in
 * `docs/api/gpu-raster/raster-change-monitoring.md`: LASSO (ordinary least squares here), Tmask
 * screening, the variogram-based minimum RMSE (`minimumRMSE` is caller-supplied), the temporally
 * adjusted RMSE (one whole-segment RMSE), the per-observation model update (refits happen when
 * the observation count grows by a factor of 1.33), the backward lookback of the initialization,
 * snow and insufficient-clear procedures and QA-class handling (the caller's `validity` is all),
 * and change probability below a confirmed break.
 *
 * One band's fit reuses the shared QR of `getRasterLeastSquaresWGSL`, refactored per band. Fits
 * per cell are bounded by `maximumFitCount`.
 *
 * Per-frame: `parameters`. Compile-time: `cellCount`, `sliceCount`, `bandCount`,
 * `minimumObservations`, `consecutiveCount`, `maximumCoefficientCount`, `maximumSegmentCount`,
 * `maximumFitCount`, and which optional views exist.
 */
export class GPURasterCOLDMonitor implements GPUCommandNodeProducer {
  readonly id: string;
  readonly minimumObservations: number;
  readonly consecutiveCount: number;
  readonly maximumCoefficientCount: number;
  readonly maximumSegmentCount: number;
  readonly maximumFitCount: number;
  private readonly ticksPerYear: number;

  constructor(readonly props: GPURasterCOLDMonitorProps) {
    const id = props.id ?? 'raster-cold-monitor';
    this.id = id;
    this.minimumObservations = props.minimumObservations ?? 12;
    this.consecutiveCount = props.consecutiveCount ?? 6;
    this.maximumCoefficientCount = props.maximumCoefficientCount ?? 8;
    this.maximumSegmentCount = props.maximumSegmentCount ?? 8;
    this.maximumFitCount = props.maximumFitCount ?? 96;
    const {sliceCount: slices, bandCount: bands} = props;
    validateCapacity(id, 'cellCount', props.cellCount);
    validateCapacity(id, 'sliceCount', slices, 1, MAXIMUM_COLD_MONITOR_SLICE_COUNT);
    validateCapacity(id, 'bandCount', bands, 1, MAXIMUM_COLD_MONITOR_BAND_COUNT);
    validateCapacity(id, 'minimumObservations', this.minimumObservations, 5, slices);
    validateCapacity(
      id,
      'consecutiveCount',
      this.consecutiveCount,
      2,
      MAXIMUM_COLD_MONITOR_CONSECUTIVE_COUNT
    );
    validateRequirement(
      id,
      [4, 6, 8].includes(this.maximumCoefficientCount),
      'maximumCoefficientCount must be 4, 6 or 8'
    );
    validateCapacity(
      id,
      'maximumSegmentCount',
      this.maximumSegmentCount,
      1,
      MAXIMUM_COLD_MONITOR_SEGMENT_COUNT
    );
    const columns = this.maximumCoefficientCount;
    validateCapacity(
      id,
      'maximumFitCount',
      this.maximumFitCount,
      1,
      Math.floor(COLD_MONITOR_WORK_BUDGET / (slices * columns * columns * 2 * bands))
    );
    const memoryFloats = Math.max(
      slices * columns + columns * columns + 2 * columns + 8,
      slices * columns +
        3 * slices +
        Math.ceil(slices / 32) +
        2 * bands * GPU_RASTER_COLD_MONITOR_COEFFICIENT_COLUMN_COUNT +
        2 * bands * this.consecutiveCount +
        64
    );
    validateRequirement(
      id,
      memoryFloats <= COLD_MONITOR_WORKING_MEMORY_FLOAT_COUNT,
      `per-thread working memory (sliceCount ${slices}, ${columns} coefficients, ${bands} bands) must fit ${COLD_MONITOR_WORKING_MEMORY_FLOAT_COUNT} floats`
    );
    const observations = props.cellCount * slices;
    const segments = props.cellCount * this.maximumSegmentCount;
    validateCapacity(id, 'cellCount times sliceCount', observations);
    validateRequirement(
      id,
      props.values.length >= observations * bands,
      'values must hold cellCount times sliceCount times bandCount entries'
    );
    validateRequirement(
      id,
      !props.validity || props.validity.length >= observations,
      'validity must hold cellCount times sliceCount entries'
    );
    validateRequirement(
      id,
      props.parameters.length >= GPU_RASTER_COLD_MONITOR_BASE_PARAMETER_LENGTH + bands,
      `parameters must hold ${GPU_RASTER_COLD_MONITOR_BASE_PARAMETER_LENGTH + bands} entries`
    );
    validateRequirement(
      id,
      props.output.segments.length >= segments * GPU_RASTER_COLD_MONITOR_SEGMENT_WORD_COUNT,
      `output.segments must hold ${GPU_RASTER_COLD_MONITOR_SEGMENT_WORD_COUNT} entries per segment`
    );
    validateRequirement(
      id,
      props.output.coefficients.length >=
        segments * bands * GPU_RASTER_COLD_MONITOR_COEFFICIENT_COLUMN_COUNT,
      `output.coefficients must hold ${GPU_RASTER_COLD_MONITOR_COEFFICIENT_COLUMN_COUNT} entries per segment and band`
    );
    validateRequirement(
      id,
      props.output.segmentStatistics.length >= segments * bands * 2,
      'output.segmentStatistics must hold two entries per segment and band'
    );
    validateRequirement(
      id,
      props.output.cellSummary.length >= props.cellCount * 3,
      'output.cellSummary must hold three entries per cell'
    );
    this.ticksPerYear = resolveGPURasterIrregularDateAxis(id, props.dateAxis, slices);
    validateRasterViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterScalarView[] {
    const {values, validity, parameters} = this.props;
    return [values, parameters, ...(validity ? [validity] : [])];
  }

  private getOutputs(): RasterScalarView[] {
    const {segments, coefficients, segmentStatistics, cellSummary} = this.props.output;
    return [segments, coefficients, segmentStatistics, cellSummary];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {
      props,
      id,
      minimumObservations,
      consecutiveCount,
      maximumCoefficientCount,
      maximumSegmentCount,
      maximumFitCount
    } = this;
    const {
      sliceCount: slices,
      bandCount: bands,
      values,
      validity,
      parameters,
      dateAxis,
      output
    } = props;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    validateGPURasterDateAxisGraph(id, graph, dateAxis);
    const columns = GPU_RASTER_COLD_MONITOR_COEFFICIENT_COLUMN_COUNT;
    const maskWords = Math.ceil(slices / 32);
    const firstRMSE = GPU_RASTER_COLD_MONITOR_BASE_PARAMETER_LENGTH;
    const declarations = `${RASTER_FINITE_WGSL}
${getGPURasterDateAxisDeclarations(this.ticksPerYear, slices)}
${getRasterHarmonicTrendDesignWGSL(slices)}
${getRasterLeastSquaresWGSL(slices, maximumCoefficientCount, true)}
var<private> cellIndex: u32;
var<private> observationSlices: array<u32, ${slices}>;
var<private> observationTotal: u32;
var<private> includedMask: array<u32, ${maskWords}>;
var<private> includedTotal: u32;
var<private> fitCoefficients: array<f32, ${bands * columns}>;
var<private> fitRMSE: array<f32, ${bands}>;
var<private> fitsUsed: u32;
var<private> statusBits: u32;
fn observationValue(slice: u32, band: u32) -> f32 {
  return values[valuesOffset + (cellIndex * ${slices}u + slice) * ${bands}u + band];
}
fn harmonicPeriod() -> f32 { return parameters[parametersOffset]; }
fn minimumRMSEOf(band: u32) -> f32 { return parameters[parametersOffset + ${firstRMSE}u + band]; }
fn clearIncluded() {
  for (var word = 0u; word < ${maskWords}u; word++) { includedMask[word] = 0u; }
  includedTotal = 0u;
}
fn addIncluded(position: u32) {
  includedMask[position / 32u] |= 1u << (position % 32u);
  includedTotal++;
}
fn isIncluded(position: u32) -> bool {
  return (includedMask[position / 32u] & (1u << (position % 32u))) != 0u;
}
fn getCoefficientCountFor(count: u32) -> u32 {
  // Four coefficients below 18 observations, six below 24, eight above (Zhu et al. 2020).
  return min(${maximumCoefficientCount}u, select(select(4u, 6u, count >= 18u), 8u, count >= 24u));
}
fn predictBand(slice: u32, band: u32) -> f32 {
  var prediction = 0.0;
  for (var column = 0u; column < ${columns}u; column++) { prediction += fitCoefficients[band * ${columns}u + column] * getHarmonicTrendColumn(slice, column, harmonicPeriod()); }
  return prediction;
}
// Refits every band on the included observations; false stops the cell (budget or rank).
fn fitIncluded(columnCount: u32) -> bool {
  if (fitsUsed >= ${maximumFitCount}u) { statusBits |= 8u; return false; }
  fitsUsed++;
  var refreshed: array<f32, ${bands * columns}>;
  var refreshedRMSE: array<f32, ${bands}>;
  for (var band = 0u; band < ${bands}u; band++) {
    var design: array<f32, ${slices * maximumCoefficientCount}>;
    var response: array<f32, ${slices}>;
    for (var position = 0u; position < observationTotal; position++) {
      if (!isIncluded(position)) { continue; }
      let slice = observationSlices[position];
      response[slice] = observationValue(slice, band);
      for (var column = 0u; column < columnCount; column++) { design[column * ${slices}u + slice] = getHarmonicTrendColumn(slice, column, harmonicPeriod()); }
    }
    let fit = leastSquares(design, response, columnCount);
    if (!fit.valid) { statusBits |= 2u; return false; }
    for (var column = 0u; column < columnCount; column++) { refreshed[band * ${columns}u + column] = fit.coefficients[column]; }
    refreshedRMSE[band] = sqrt(fit.residual / f32(includedTotal - min(includedTotal, columnCount)));
  }
  fitCoefficients = refreshed;
  fitRMSE = refreshedRMSE;
  return true;
}
fn fillOutputRecord(segment: u32, startSlice: u32, endSlice: u32, breakSlice: u32, columnCount: u32, breakFound: bool, magnitudes: array<f32, ${bands}>) {
  let base = (cellIndex * ${maximumSegmentCount}u + segment);
  segments[segmentsOffset + base * 4u] = startSlice;
  segments[segmentsOffset + base * 4u + 1u] = endSlice;
  segments[segmentsOffset + base * 4u + 2u] = breakSlice;
  segments[segmentsOffset + base * 4u + 3u] = columnCount;
  for (var band = 0u; band < ${bands}u; band++) {
    let entry = base * ${bands}u + band;
    segmentStatistics[segmentStatisticsOffset + entry * 2u] = fitRMSE[band];
    segmentStatistics[segmentStatisticsOffset + entry * 2u + 1u] = select(missing(), magnitudes[band], breakFound);
    for (var column = 0u; column < ${columns}u; column++) { coefficients[coefficientsOffset + entry * ${columns}u + column] = fitCoefficients[band * ${columns}u + column]; }
  }
}
fn getMedian(sample: array<f32, ${consecutiveCount}>) -> f32 {
  var ordered = sample;
  for (var outer = 1u; outer < ${consecutiveCount}u; outer++) {
    let item = ordered[outer];
    var inner = outer;
    while (inner > 0u && ordered[inner - 1u] > item) { ordered[inner] = ordered[inner - 1u]; inner--; }
    ordered[inner] = item;
  }
  return ${consecutiveCount % 2 === 1 ? `ordered[${(consecutiveCount - 1) / 2}u]` : `0.5 * (ordered[${consecutiveCount / 2 - 1}u] + ordered[${consecutiveCount / 2}u])`};
}`;
    const body = `fillRasterSliceYears();
cellIndex = index;
let changeThreshold = parameters[parametersOffset + 1u];
let outlierThreshold = parameters[parametersOffset + 2u];
let minimumMeanCosine = parameters[parametersOffset + 3u];
let minimumSpanYears = parameters[parametersOffset + 4u];
let maximumInitializationGapYears = parameters[parametersOffset + 5u];
fitsUsed = 0u;
statusBits = 0u;
observationTotal = 0u;
for (var slice = 0u; slice < ${slices}u; slice++) {
  var usable = ${validity ? `validity[validityOffset + index * ${slices}u + slice] != 0u` : 'true'};
  for (var band = 0u; band < ${bands}u; band++) { usable = usable && finite(observationValue(slice, band)); }
  if (usable) { observationSlices[observationTotal] = slice; observationTotal++; }
}
var segment = 0u;
var cursor = 0u;
for (var unused = 0u; unused < ${maximumSegmentCount}u; unused++) {
  let base = (index * ${maximumSegmentCount}u + unused);
  for (var word = 0u; word < 4u; word++) { segments[segmentsOffset + base * 4u + word] = ${GPU_RASTER_COLD_MONITOR_NO_BREAK}u; }
  for (var band = 0u; band < ${bands}u; band++) {
    segmentStatistics[segmentStatisticsOffset + (base * ${bands}u + band) * 2u] = missing();
    segmentStatistics[segmentStatisticsOffset + (base * ${bands}u + band) * 2u + 1u] = missing();
    for (var column = 0u; column < ${columns}u; column++) { coefficients[coefficientsOffset + (base * ${bands}u + band) * ${columns}u + column] = missing(); }
  }
}
loop {
  if (cursor >= observationTotal) { break; }
  if (segment == ${maximumSegmentCount}u) { statusBits |= 16u; break; }
  var start = cursor;
  var windowEnd = 0u;
  var initialized = false;
  loop {
    if (start + ${minimumObservations}u > observationTotal || (statusBits & 8u) != 0u) { break; }
    var end = start + ${minimumObservations}u;
    loop {
      if (end > observationTotal) { break; }
      if (sliceYears[observationSlices[end - 1u]] - sliceYears[observationSlices[start]] >= minimumSpanYears) { break; }
      end++;
    }
    if (end > observationTotal) { break; }
    var largestGap = 0.0;
    for (var position = start + 1u; position < end; position++) {
      largestGap = max(largestGap, sliceYears[observationSlices[position]] - sliceYears[observationSlices[position - 1u]]);
    }
    if (largestGap >= maximumInitializationGapYears) { start++; continue; }
    clearIncluded();
    for (var position = start; position < end; position++) { addIncluded(position); }
    if (!fitIncluded(4u)) { break; }
    let firstSlice = observationSlices[start];
    let lastSlice = observationSlices[end - 1u];
    let spanYears = sliceYears[lastSlice] - sliceYears[firstSlice];
    var score = 0.0;
    for (var band = 0u; band < ${bands}u; band++) {
      let floorRMSE = max(fitRMSE[band], minimumRMSEOf(band));
      let term = (abs(fitCoefficients[band * ${columns}u + 1u]) * spanYears + abs(observationValue(firstSlice, band) - predictBand(firstSlice, band)) + abs(observationValue(lastSlice, band) - predictBand(lastSlice, band))) / floorRMSE;
      score += term * term;
    }
    if (score <= changeThreshold) { windowEnd = end; initialized = true; break; }
    start++;
  }
  if (!initialized) { break; }
  var columnCount = getCoefficientCountFor(includedTotal);
  if (!fitIncluded(columnCount)) { break; }
  var fittedCount = includedTotal;
  var position = windowEnd;
  var breakFound = false;
  var breakPosition = 0u;
  var breakMagnitudes: array<f32, ${bands}>;
  loop {
    if (position + ${consecutiveCount}u > observationTotal) { break; }
    var normalized: array<f32, ${consecutiveCount * bands}>;
    var residualSamples: array<f32, ${consecutiveCount * bands}>;
    var allExceed = true;
    var firstScore = 0.0;
    for (var offset = 0u; offset < ${consecutiveCount}u; offset++) {
      let peekSlice = observationSlices[position + offset];
      var peekScore = 0.0;
      for (var band = 0u; band < ${bands}u; band++) {
        let residual = observationValue(peekSlice, band) - predictBand(peekSlice, band);
        residualSamples[offset * ${bands}u + band] = residual;
        let value = residual / max(fitRMSE[band], minimumRMSEOf(band));
        normalized[offset * ${bands}u + band] = value;
        peekScore += value * value;
      }
      if (offset == 0u) { firstScore = peekScore; }
      allExceed = allExceed && peekScore > changeThreshold;
    }
    var cosineTotal = 0.0;
    for (var offset = 0u; offset + 1u < ${consecutiveCount}u; offset++) {
      var dotProduct = 0.0; var leftNorm = 0.0; var rightNorm = 0.0;
      for (var band = 0u; band < ${bands}u; band++) {
        let left = normalized[offset * ${bands}u + band];
        let right = normalized[(offset + 1u) * ${bands}u + band];
        dotProduct += left * right; leftNorm += left * left; rightNorm += right * right;
      }
      cosineTotal += dotProduct / max(sqrt(leftNorm * rightNorm), 1e-12);
    }
    if (allExceed && cosineTotal / f32(${consecutiveCount - 1}u) >= minimumMeanCosine) {
      breakFound = true;
      breakPosition = position;
      for (var band = 0u; band < ${bands}u; band++) {
        var sample: array<f32, ${consecutiveCount}>;
        for (var offset = 0u; offset < ${consecutiveCount}u; offset++) { sample[offset] = residualSamples[offset * ${bands}u + band]; }
        breakMagnitudes[band] = getMedian(sample);
      }
      break;
    }
    if (firstScore > outlierThreshold) { position++; continue; }
    addIncluded(position);
    position++;
    if (includedTotal * 100u >= fittedCount * 133u && includedTotal != fittedCount) {
      let refitColumnCount = getCoefficientCountFor(includedTotal);
      if (!fitIncluded(refitColumnCount)) { break; }
      columnCount = refitColumnCount;
      fittedCount = includedTotal;
    }
  }
  if (!breakFound) {
    for (var remaining = position; remaining < observationTotal; remaining++) { addIncluded(remaining); }
  }
  if (includedTotal != fittedCount) {
    let finalColumnCount = getCoefficientCountFor(includedTotal);
    if (fitIncluded(finalColumnCount)) { columnCount = finalColumnCount; }
    else if ((statusBits & 2u) != 0u) { break; }
  }
  var firstIncluded = 0u;
  var lastIncluded = 0u;
  var foundFirst = false;
  for (var scan = 0u; scan < observationTotal; scan++) {
    if (!isIncluded(scan)) { continue; }
    if (!foundFirst) { firstIncluded = scan; foundFirst = true; }
    lastIncluded = scan;
  }
  fillOutputRecord(segment, observationSlices[firstIncluded], observationSlices[lastIncluded], select(${GPU_RASTER_COLD_MONITOR_NO_BREAK}u, observationSlices[breakPosition], breakFound), columnCount, breakFound, breakMagnitudes);
  segment++;
  if ((statusBits & 8u) != 0u || !breakFound) { break; }
  cursor = breakPosition;
}
if (segment == 0u) { statusBits |= 1u; }
cellSummary[cellSummaryOffset + index * 3u] = segment;
cellSummary[cellSummaryOffset + index * 3u + 1u] = statusBits;
cellSummary[cellSummaryOffset + index * 3u + 2u] = fitsUsed;`;
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterCOLDMonitor',
        invocationCount: props.cellCount,
        workgroupSize: 8,
        bindings: [
          rasterBinding('values', values),
          rasterBinding('parameters', parameters),
          ...(validity ? [rasterBinding('validity', validity)] : []),
          getGPURasterDateAxisBinding(dateAxis),
          rasterBinding('segments', output.segments, true),
          rasterBinding('coefficients', output.coefficients, true),
          rasterBinding('segmentStatistics', output.segmentStatistics, true),
          rasterBinding('cellSummary', output.cellSummary, true)
        ],
        declarations,
        body
      })
    ];
  }
}
