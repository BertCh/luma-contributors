// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateParameterTarget} from '../raster-band-utils';
import type {
  GraphDataView,
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {
  createRasterTimeSeriesNodes,
  planRasterTimeSeriesChunks,
  rasterTimeSeriesBinding as rasterBinding,
  validateRasterTimeSeriesViews as validateRasterViews,
  validateRasterTimeSeriesGraph as validateRasterGraph,
  type RasterTimeSeriesScalarView as RasterScalarView,
  type RasterTimeSeriesView,
  type RasterTimeSeriesChunkPlan
} from './raster-time-series-chunks-b';
import {
  validateGraphPorts,
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {RASTER_FINITE_WGSL} from '../remote-sensing/raster-kernel-utils';
import {
  getGPURasterDateAxisBinding,
  getGPURasterDateAxisDeclarations,
  resolveGPURasterIrregularDateAxis,
  validateGPURasterDateAxisGraph,
  type GPURasterIrregularDateAxis
} from './raster-irregular-date-axis';

/** Slices: one invocation per cell runs a banded QR over a private array of this many columns. */
export const MAXIMUM_WHITTAKER_SLICE_COUNT = 256;
/** Reweighting passes are unrolled into the per-cell loop, so the count is capped. */
export const MAXIMUM_WHITTAKER_REWEIGHT_PASS_COUNT = 8;
/**
 * Largest accepted smoothing parameter. The float32 banded QR on centred values agrees with a
 * binary64 solve to one float32 step of the result at this value for every difference order
 * (256 slices; pinned by `rts02-whittaker-precision.spec.ts`).
 */
export const MAXIMUM_WHITTAKER_SMOOTHING_PARAMETER = 1e7;
/** Float32 elements of the per-frame parameters: smoothing parameter, minimum weight. */
export const GPU_RASTER_WHITTAKER_SMOOTHER_PARAMETER_LENGTH = 2;

/** Per-frame settings of {@link GPURasterWhittakerSmoother}. */
export type GPURasterWhittakerSmootherSettings = {
  /** Penalty weight lambda of the Whittaker smoother; larger is smoother. */
  smoothingParameter: number;
  /** Robust-weight floor, so a point is never removed outright. Defaults to `0.01`. */
  minimumWeight?: number;
};

/** Packs {@link GPURasterWhittakerSmootherSettings}: `[smoothingParameter, minimumWeight]`. */
export function getGPURasterWhittakerSmootherParameterValues(
  settings: GPURasterWhittakerSmootherSettings,
  target: Float32Array = new Float32Array(GPU_RASTER_WHITTAKER_SMOOTHER_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget(
    'getGPURasterWhittakerSmootherParameterValues',
    target,
    GPU_RASTER_WHITTAKER_SMOOTHER_PARAMETER_LENGTH
  );
  validateFiniteNumber(
    'getGPURasterWhittakerSmootherParameterValues',
    'smoothingParameter',
    settings.smoothingParameter,
    0,
    MAXIMUM_WHITTAKER_SMOOTHING_PARAMETER
  );
  target[0] = settings.smoothingParameter;
  target[1] = settings.minimumWeight ?? 0.01;
  return target;
}

/** Robust reweighting function for repeated passes. */
export type GPURasterWhittakerReweightFunction = 'bisquare' | 'upper-envelope';

/** Properties of {@link GPURasterWhittakerSmoother}. */
export type GPURasterWhittakerSmootherProps<Rows extends RasterScalarView = GraphDataView> = {
  id?: string;
  cellCount: number;
  sliceCount: number;
  /** Cell-major `cell * sliceCount + slice`; nonfinite samples have zero weight. */
  values: RasterTimeSeriesView<'float32', Rows>;
  /** Per-observation QA; zero gives a sample zero weight. */
  mask?: RasterTimeSeriesView<'uint32', Rows>;
  /** Per-observation quality weights in `[0, 1]`, for example from a QA band. */
  qualityWeights?: RasterTimeSeriesView<'float32', Rows>;
  /** Float32, see {@link getGPURasterWhittakerSmootherParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Difference order `d` in `[1, 3]`; defaults to 2. */
  differenceOrder?: number;
  /** Robust reweighting passes; 0 to 8, defaults to 3. */
  reweightPassCount?: number;
  /** Defaults to `'bisquare'`. */
  reweightFunction?: GPURasterWhittakerReweightFunction;
  /** Acquisition dates; without one, slices are unit-spaced regular slots. */
  dateAxis?: GPURasterIrregularDateAxis;
  output: {
    /** Smoothed curve at every slice, gaps included; NaN when status is nonzero. */
    smoothed: RasterTimeSeriesView<'float32', Rows>;
    /** Per cell: 0 fit, 1 insufficient support, 2 numerical failure. */
    status: RasterTimeSeriesView<'uint32', Rows>;
  };
};

/**
 * Weighted Whittaker smoother per cell: minimises `sum w (y - z)^2 + lambda |D z|^2`, with zero
 * weight on missing observations so gaps are filled. The solve is a banded Givens QR of the stacked
 * matrix `[sqrt(W); sqrt(lambda) D]` on the values minus their per-cell mean (the smoother is exact
 * on constants, so the mean is added back), not a factorisation of the normal equations: in
 * float32 the result then depends on the variation of the series, not on its offset or on
 * `lambda`. Status 2 means the triangular factor has a zero or non-finite diagonal, or the
 * solution is not finite (for example `lambda = 0` with a gap, or a non-finite `lambda`); it is
 * not raised for a system that is merely badly conditioned. With a few clustered observations
 * and a long gap, orders 2 and 3 continue the curve as a line or a parabola, and the float32
 * error grows with the gap: over 250 slices beyond 3 or 5 adjacent observations it stays within
 * 6e-4 (order 2) and 1.3e-2 (order 3) of the spread of the curve, with status 0 (measured against
 * an exact rational solve for `lambda` from 1 to 1e7). `D` is the
 * difference operator of order `d`; with a date axis it is the divided difference on acquisition
 * dates scaled by the mean spacing so a regular axis reduces to ordinary differences and `lambda`
 * keeps its usual meaning. Robust reweighting runs a fixed number of passes (Tukey bisquare on the
 * residual over 6 MAD, or the Chen upper-envelope scheme for cloud-depressed vegetation series).
 *
 * Per-frame: `parameters`. Compile-time: `cellCount`, `sliceCount`, `differenceOrder`,
 * `reweightPassCount`, `reweightFunction`, and which optional views exist.
 *
 * Cell-major views accept matching chunks of complete cell records; shared tables stay packed.
 */
export class GPURasterWhittakerSmoother<Rows extends RasterScalarView = GraphDataView>
  implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly props: GPURasterWhittakerSmootherProps<Rows>;
  readonly chunkPlan: RasterTimeSeriesChunkPlan;
  readonly estimatedCommandNodeCount: number;
  readonly differenceOrder: number;
  readonly reweightPassCount: number;
  readonly reweightFunction: GPURasterWhittakerReweightFunction;
  private readonly ticksPerYear: number;

  constructor(props: GPURasterWhittakerSmootherProps<Rows> & {values: Rows}) {
    validateGraphPorts(props.id ?? 'raster-whittaker-smoother', props, {
      values: {required: true, viewType: 'either', formats: ['float32']},
      mask: {required: false, viewType: 'either', formats: ['uint32']},
      qualityWeights: {required: false, viewType: 'either', formats: ['float32']},
      parameters: {required: true, viewType: 'data', formats: ['float32']},
      'dateAxis.acquisitionTimes': {
        required: Boolean(props.dateAxis),
        viewType: 'data',
        formats: ['uint32x2']
      },
      'output.smoothed': {required: true, viewType: 'either', formats: ['float32']},
      'output.status': {required: true, viewType: 'either', formats: ['uint32']}
    });
    this.props = props;
    const id = props.id ?? 'raster-whittaker-smoother';
    this.id = id;
    const differenceOrder = props.differenceOrder ?? 2;
    this.differenceOrder = differenceOrder;
    const reweightPassCount = props.reweightPassCount ?? 3;
    this.reweightPassCount = reweightPassCount;
    this.reweightFunction = props.reweightFunction ?? 'bisquare';
    validateCapacity(id, 'cellCount', props.cellCount);
    validateCapacity(id, 'sliceCount', props.sliceCount, 2, MAXIMUM_WHITTAKER_SLICE_COUNT);
    validateCapacity(id, 'differenceOrder', differenceOrder, 1, 3);
    validateRequirement(
      id,
      props.sliceCount > differenceOrder,
      `sliceCount must exceed differenceOrder ${differenceOrder}`
    );
    validateRequirement(
      id,
      Number.isInteger(reweightPassCount) &&
        reweightPassCount >= 0 &&
        reweightPassCount <= MAXIMUM_WHITTAKER_REWEIGHT_PASS_COUNT,
      `reweightPassCount must be an integer in [0, ${MAXIMUM_WHITTAKER_REWEIGHT_PASS_COUNT}]`
    );
    validateRequirement(
      id,
      this.reweightFunction === 'bisquare' || this.reweightFunction === 'upper-envelope',
      "reweightFunction must be 'bisquare' or 'upper-envelope'"
    );
    const observationCount = props.cellCount * props.sliceCount;
    validateCapacity(id, 'cellCount times sliceCount', observationCount);
    validateRequirement(
      id,
      [props.values, props.mask, props.qualityWeights].every(
        view => !view || view.length >= observationCount
      ) && props.output.smoothed.length >= observationCount,
      'values, mask, qualityWeights and output.smoothed must hold cellCount times sliceCount entries'
    );
    validateRequirement(
      id,
      props.output.status.length >= props.cellCount,
      'output.status must hold one entry per cell'
    );
    validateRequirement(
      id,
      props.parameters.length >= GPU_RASTER_WHITTAKER_SMOOTHER_PARAMETER_LENGTH,
      `parameters must hold ${GPU_RASTER_WHITTAKER_SMOOTHER_PARAMETER_LENGTH} entries`
    );
    this.ticksPerYear = props.dateAxis
      ? resolveGPURasterIrregularDateAxis(id, props.dateAxis, props.sliceCount)
      : 0;
    this.chunkPlan = planRasterTimeSeriesChunks(this.id, props.cellCount, {
      values: {view: props.values, entriesPerCell: props.sliceCount},
      mask: {view: props.mask, entriesPerCell: props.sliceCount},
      qualityWeights: {view: props.qualityWeights, entriesPerCell: props.sliceCount},
      'output.smoothed': {view: props.output.smoothed, entriesPerCell: props.sliceCount},
      'output.status': {view: props.output.status, entriesPerCell: 1}
    });
    this.estimatedCommandNodeCount = this.chunkPlan.estimatedCommandNodeCount;
    validateRasterViews(this.id, this.getInputs(), this.getOutputs(), this.chunkPlan);
  }

  private getInputs(): RasterScalarView[] {
    const {values, mask, qualityWeights, parameters} = this.props;
    return [
      ...(this.props.dateAxis ? [this.props.dateAxis.acquisitionTimes] : []),
      values,
      parameters,
      ...(mask ? [mask] : []),
      ...(qualityWeights ? [qualityWeights] : [])
    ];
  }

  private getOutputs(): RasterScalarView[] {
    return [this.props.output.smoothed, this.props.output.status];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, differenceOrder: order, reweightPassCount, reweightFunction} = this;
    const {sliceCount: slices, values, mask, qualityWeights, parameters, dateAxis, output} = props;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    if (dateAxis) validateGPURasterDateAxisGraph(id, graph, dateAxis);
    const bandStride = order + 1;
    const inputBindings = [
      rasterBinding('values', values),
      rasterBinding('parameters', parameters),
      ...(mask ? [rasterBinding('mask', mask)] : []),
      ...(qualityWeights ? [rasterBinding('qualityWeights', qualityWeights)] : [])
    ];
    const spanExpression = dateAxis
      ? '(rasterYearsBetween(row + offset + level, row + offset) / axisMeanSpacing)'
      : 'f32(level)';
    return createRasterTimeSeriesNodes<Parameters>(graph, this.chunkPlan, {
      id,
      operation: 'GPURasterWhittakerSmoother',
      invocationCount: props.cellCount,
      workgroupSize: 8,
      bindings: [
        ...inputBindings,
        ...(dateAxis ? [getGPURasterDateAxisBinding(dateAxis)] : []),
        rasterBinding('smoothed', output.smoothed, true),
        rasterBinding('status', output.status, true)
      ],
      declarations: `${RASTER_FINITE_WGSL}
${dateAxis ? getGPURasterDateAxisDeclarations(this.ticksPerYear, slices) : ''}
fn baseWeight(cell: u32, slice: u32) -> f32 {
  let source = cell * ${slices}u + slice;
  if (!finite(values[valuesOffset + source]) ${mask ? '|| mask[maskOffset + source] == 0u' : ''}) { return 0.0; }
  return ${qualityWeights ? 'clamp(qualityWeights[qualityWeightsOffset + source], 0.0, 1.0)' : '1.0'};
}
fn siftDown(heap: ptr<function, array<f32, ${slices}>>, start: u32, size: u32) {
  var parent = start;
  loop {
    let left = 2u * parent + 1u;
    if (left >= size) { break; }
    var largest = left;
    if (left + 1u < size && (*heap)[left + 1u] > (*heap)[left]) { largest = left + 1u; }
    if ((*heap)[parent] >= (*heap)[largest]) { break; }
    let swapped = (*heap)[parent]; (*heap)[parent] = (*heap)[largest]; (*heap)[largest] = swapped;
    parent = largest;
  }
}
fn getMedian(heap: ptr<function, array<f32, ${slices}>>, count: u32) -> f32 {
  for (var node = count / 2u; node > 0u; node--) { siftDown(heap, node - 1u, count); }
  for (var end = count; end > 1u; end--) {
    let swapped = (*heap)[0]; (*heap)[0] = (*heap)[end - 1u]; (*heap)[end - 1u] = swapped;
    siftDown(heap, 0u, end - 1u);
  }
  if (count % 2u == 1u) { return (*heap)[count / 2u]; }
  return 0.5 * ((*heap)[count / 2u - 1u] + (*heap)[count / 2u]);
}
// Rotates one row of the stacked least-squares matrix into the upper band R, where
// band[j * stride + k] holds R[j, j + k]. entries[k] is the value of the row in column row + k.
// Rows must arrive ordered by first column (weight row, then penalty row, for each slice), which
// keeps every rotation inside the band: a row is finished after at most order + 1 rotations.
fn insertRow(
  band: ptr<function, array<f32, ${slices * bandStride}>>,
  rightSide: ptr<function, array<f32, ${slices}>>,
  row: u32,
  entries: array<f32, ${bandStride}>,
  entryRightSide: f32
) {
  var work = entries;
  var workRightSide = entryRightSide;
  let last = min(row + ${order}u, ${slices - 1}u);
  for (var column = row; column <= last; column++) {
    let lead = work[0];
    let base = column * ${bandStride}u;
    let diagonal = (*band)[base];
    if (lead != 0.0 && diagonal == 0.0) {
      for (var term = 0u; term <= ${order}u; term++) { (*band)[base + term] = work[term]; }
      (*rightSide)[column] = workRightSide;
      return;
    }
    var cosine = 1.0;
    var sine = 0.0;
    if (lead != 0.0) {
      let radius = sqrt(diagonal * diagonal + lead * lead);
      cosine = diagonal / radius;
      sine = lead / radius;
      (*band)[base] = radius;
    }
    for (var term = 1u; term <= ${order}u; term++) {
      let upper = (*band)[base + term];
      let lower = work[term];
      (*band)[base + term] = cosine * upper + sine * lower;
      work[term - 1u] = cosine * lower - sine * upper;
    }
    work[${order}u] = 0.0;
    let upperRightSide = (*rightSide)[column];
    (*rightSide)[column] = cosine * upperRightSide + sine * workRightSide;
    workRightSide = cosine * workRightSide - sine * upperRightSide;
  }
}`,
      body: `let smoothing = parameters[parametersOffset];
let minimumWeight = parameters[parametersOffset + 1u];
var band: array<f32, ${slices * bandStride}>;
var solution: array<f32, ${slices}>;
var robust: array<f32, ${slices}>;
var workspace: array<f32, ${slices}>;
var support = 0u;
// The smoother is exact on constants (every difference of order one or more removes them), so the
// solve runs on values minus a per-cell reference and the reference is added back. Without this
// the float32 error grows with the magnitude of the data instead of its variation.
var reference = 0.0;
for (var slice = 0u; slice < ${slices}u; slice++) {
  robust[slice] = 1.0;
  if (baseWeight(index, slice) > 0.0) {
    if (support == 0u) { reference = values[valuesOffset + index * ${slices}u + slice]; }
    support++;
  }
}
if (support > 0u) {
  // Mean of the valid values, formed around the first one so the sum stays small.
  var offsetSum = 0.0;
  for (var slice = 0u; slice < ${slices}u; slice++) {
    if (baseWeight(index, slice) > 0.0) { offsetSum += values[valuesOffset + index * ${slices}u + slice] - reference; }
  }
  reference += offsetSum / f32(support);
}
let penaltyScale = sqrt(max(smoothing, 0.0));
${dateAxis ? `let axisMeanSpacing = rasterYearsBetween(${slices - 1}u, 0u) / ${slices - 1}.0;` : ''}
var code = 0u;
if (support < ${order + 1}u) { code = 1u; }
if (code == 0u) {
  for (var iteration = 0u; iteration < ${reweightPassCount + 1}u; iteration++) {
    for (var entry = 0u; entry < ${slices * bandStride}u; entry++) { band[entry] = 0.0; }
    for (var slice = 0u; slice < ${slices}u; slice++) { solution[slice] = 0.0; }
    // Givens QR of the stacked matrix [sqrt(W); sqrt(lambda) D] instead of a factorisation of
    // W + lambda D'D: the condition number is the square root of the normal equations', which is
    // what float32 needs at large lambda.
    for (var row = 0u; row < ${slices}u; row++) {
      let weight = baseWeight(index, row) * robust[row];
      if (weight > 0.0) {
        var weightRow: array<f32, ${bandStride}>;
        weightRow[0] = sqrt(weight);
        insertRow(&band, &solution, row, weightRow, weightRow[0] * (values[valuesOffset + index * ${slices}u + row] - reference));
      }
      if (row >= ${slices - order}u) { continue; }
      var previous: array<array<f32, ${bandStride}>, ${bandStride}>;
      for (var offset = 0u; offset < ${bandStride}u; offset++) { previous[offset][0] = 1.0; }
      for (var level = 1u; level <= ${order}u; level++) {
        var current: array<array<f32, ${bandStride}>, ${bandStride}>;
        for (var offset = 0u; offset <= ${order}u - level; offset++) {
          let span = ${spanExpression} / f32(level);
          for (var term = 0u; term <= level; term++) {
            var coefficient = 0.0;
            if (term > 0u) { coefficient += previous[offset + 1u][term - 1u]; }
            if (term < level) { coefficient -= previous[offset][term]; }
            current[offset][term] = coefficient / span;
          }
        }
        previous = current;
      }
      var penaltyRow = previous[0];
      for (var term = 0u; term <= ${order}u; term++) { penaltyRow[term] *= penaltyScale; }
      insertRow(&band, &solution, row, penaltyRow, 0.0);
    }
    // A diagonal of R that is zero or not finite means the system has no solution in float32
    // (for example lambda = 0 with a gap). The size of a diagonal against the largest one is not
    // tested: the penalty rows scale with sqrt(lambda), so that ratio is small for well-posed
    // series with few clustered observations, which this solve handles (see the class
    // documentation for the accuracy there).
    for (var row = 0u; row < ${slices}u; row++) {
      let diagonal = abs(band[row * ${bandStride}u]);
      if (!finite(diagonal) || !(diagonal > 0.0)) { code = 2u; }
    }
    if (code != 0u) { break; }
    // Back substitution on the upper band.
    for (var reverse = 0u; reverse < ${slices}u; reverse++) {
      let row = ${slices - 1}u - reverse;
      var remainder = solution[row];
      for (var above = row + 1u; above <= min(row + ${order}u, ${slices - 1}u); above++) { remainder -= band[row * ${bandStride}u + (above - row)] * solution[above]; }
      solution[row] = remainder / band[row * ${bandStride}u];
      if (!finite(solution[row])) { code = 2u; }
    }
    if (code != 0u) { break; }
    if (iteration == ${reweightPassCount}u) { break; }
    var residualCount = 0u;
    var largestNegative = 0.0;
    for (var slice = 0u; slice < ${slices}u; slice++) {
      if (baseWeight(index, slice) <= 0.0) { continue; }
      let residual = (values[valuesOffset + index * ${slices}u + slice] - reference) - solution[slice];
      workspace[residualCount] = abs(residual); residualCount++;
      if (residual < 0.0) { largestNegative = max(largestNegative, -residual); }
    }
    let scale = 6.0 * getMedian(&workspace, residualCount);
    for (var slice = 0u; slice < ${slices}u; slice++) {
      if (baseWeight(index, slice) <= 0.0) { continue; }
      let residual = (values[valuesOffset + index * ${slices}u + slice] - reference) - solution[slice];
      ${
        reweightFunction === 'bisquare'
          ? `var nextRobust = 1.0;
      if (scale > 0.0) { let ratio = abs(residual) / scale; nextRobust = select(minimumWeight, (1.0 - ratio * ratio) * (1.0 - ratio * ratio), ratio < 1.0); }`
          : `var nextRobust = 1.0;
      if (residual < 0.0 && largestNegative > 0.0) { nextRobust = 1.0 - abs(residual) / largestNegative; }`
      }
      robust[slice] = max(nextRobust, minimumWeight);
    }
  }
}
status[statusOffset + index] = code;
for (var slice = 0u; slice < ${slices}u; slice++) {
  smoothed[smoothedOffset + index * ${slices}u + slice] = select(missing(), solution[slice] + reference, code == 0u);
}`
    });
  }
}
