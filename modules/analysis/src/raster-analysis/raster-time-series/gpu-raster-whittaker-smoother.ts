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

/** Slices: one invocation per cell runs a banded Cholesky over a private array of this many columns. */
export const MAXIMUM_WHITTAKER_SLICE_COUNT = 256;
/** Reweighting passes are unrolled into the per-cell loop, so the count is capped. */
export const MAXIMUM_WHITTAKER_REWEIGHT_PASS_COUNT = 8;
/** float32 band Cholesky loses accuracy past this smoothing parameter (roadmap RTS-02). */
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
export type GPURasterWhittakerSmootherProps = {
  id?: string;
  cellCount: number;
  sliceCount: number;
  /** Cell-major `cell * sliceCount + slice`; nonfinite samples have zero weight. */
  values: GraphDataView<'float32'>;
  /** Per-observation QA; zero gives a sample zero weight. */
  validity?: GraphDataView<'uint32'>;
  /** Per-observation quality weights in `[0, 1]`, for example from a QA band. */
  qualityWeights?: GraphDataView<'float32'>;
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
    smoothed: GraphDataView<'float32'>;
    /** Per cell: 0 fit, 1 insufficient support, 2 numerical failure. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Weighted Whittaker smoother per cell: minimises `sum w (y - z)^2 + lambda |D z|^2` by a banded
 * Cholesky solve, with zero weight on missing observations so gaps are filled. `D` is the
 * difference operator of order `d`; with a date axis it is the divided difference on acquisition
 * dates scaled by the mean spacing so a regular axis reduces to ordinary differences and `lambda`
 * keeps its usual meaning. Robust reweighting runs a fixed number of passes (Tukey bisquare on the
 * residual over 6 MAD, or the Chen upper-envelope scheme for cloud-depressed vegetation series).
 *
 * Per-frame: `parameters`. Compile-time: `cellCount`, `sliceCount`, `differenceOrder`,
 * `reweightPassCount`, `reweightFunction`, and which optional views exist.
 */
export class GPURasterWhittakerSmoother implements GPUCommandNodeProducer {
  readonly id: string;
  readonly differenceOrder: number;
  readonly reweightPassCount: number;
  readonly reweightFunction: GPURasterWhittakerReweightFunction;
  private readonly ticksPerYear: number;

  constructor(readonly props: GPURasterWhittakerSmootherProps) {
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
      [props.values, props.validity, props.qualityWeights].every(
        view => !view || view.length >= observationCount
      ) && props.output.smoothed.length >= observationCount,
      'values, validity, qualityWeights and output.smoothed must hold cellCount times sliceCount entries'
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
    validateRasterViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterScalarView[] {
    const {values, validity, qualityWeights, parameters} = this.props;
    return [
      values,
      parameters,
      ...(validity ? [validity] : []),
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
    const {
      sliceCount: slices,
      values,
      validity,
      qualityWeights,
      parameters,
      dateAxis,
      output
    } = props;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    if (dateAxis) validateGPURasterDateAxisGraph(id, graph, dateAxis);
    const bandStride = order + 1;
    const inputBindings = [
      rasterBinding('values', values),
      rasterBinding('parameters', parameters),
      ...(validity ? [rasterBinding('validity', validity)] : []),
      ...(qualityWeights ? [rasterBinding('qualityWeights', qualityWeights)] : [])
    ];
    const spanExpression = dateAxis
      ? '(rasterYearsBetween(row + offset + level, row + offset) / axisMeanSpacing)'
      : 'f32(level)';
    return [
      createWGSLKernelNode<Parameters>(graph, {
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
  if (!finite(values[valuesOffset + source]) ${validity ? '|| validity[validityOffset + source] == 0u' : ''}) { return 0.0; }
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
}`,
        body: `let smoothing = parameters[parametersOffset];
let minimumWeight = parameters[parametersOffset + 1u];
var band: array<f32, ${slices * bandStride}>;
var solution: array<f32, ${slices}>;
var robust: array<f32, ${slices}>;
var workspace: array<f32, ${slices}>;
var support = 0u;
for (var slice = 0u; slice < ${slices}u; slice++) { robust[slice] = 1.0; if (baseWeight(index, slice) > 0.0) { support++; } }
${dateAxis ? `let axisMeanSpacing = rasterYearsBetween(${slices - 1}u, 0u) / ${slices - 1}.0;` : ''}
var code = 0u;
if (support < ${order + 1}u) { code = 1u; }
if (code == 0u) {
  for (var iteration = 0u; iteration < ${reweightPassCount + 1}u; iteration++) {
    for (var entry = 0u; entry < ${slices * bandStride}u; entry++) { band[entry] = 0.0; }
    for (var slice = 0u; slice < ${slices}u; slice++) {
      let weight = baseWeight(index, slice) * robust[slice];
      band[slice * ${bandStride}u] = weight;
      solution[slice] = select(0.0, weight * values[valuesOffset + index * ${slices}u + slice], weight > 0.0);
    }
    for (var row = 0u; row < ${slices - order}u; row++) {
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
      let coefficients = previous[0];
      for (var left = 0u; left <= ${order}u; left++) {
        for (var right = left; right <= ${order}u; right++) {
          band[(row + left) * ${bandStride}u + (right - left)] += smoothing * coefficients[left] * coefficients[right];
        }
      }
    }
    // LDL^T of the symmetric band: band[i, 0] holds D_i and band[i, k] holds L[i + k, i].
    for (var row = 0u; row < ${slices}u; row++) {
      var pivot = band[row * ${bandStride}u];
      let firstColumn = select(0u, row - ${order}u, row > ${order}u);
      for (var column = firstColumn; column < row; column++) {
        let factor = band[column * ${bandStride}u + (row - column)];
        pivot -= factor * factor * band[column * ${bandStride}u];
      }
      band[row * ${bandStride}u] = pivot;
      if (!(pivot > 0.0) || !finite(pivot)) { code = 2u; break; }
      for (var below = row + 1u; below <= min(row + ${order}u, ${slices - 1}u); below++) {
        var coupled = band[row * ${bandStride}u + (below - row)];
        let firstShared = select(0u, below - ${order}u, below > ${order}u);
        for (var column = firstShared; column < row; column++) {
          coupled -= band[column * ${bandStride}u + (below - column)] * band[column * ${bandStride}u + (row - column)] * band[column * ${bandStride}u];
        }
        band[row * ${bandStride}u + (below - row)] = coupled / pivot;
      }
    }
    if (code != 0u) { break; }
    for (var row = 1u; row < ${slices}u; row++) {
      let firstColumn = select(0u, row - ${order}u, row > ${order}u);
      for (var column = firstColumn; column < row; column++) { solution[row] -= band[column * ${bandStride}u + (row - column)] * solution[column]; }
    }
    for (var row = 0u; row < ${slices}u; row++) { solution[row] /= band[row * ${bandStride}u]; }
    for (var reverse = 1u; reverse < ${slices}u; reverse++) {
      let row = ${slices - 1}u - reverse;
      for (var below = row + 1u; below <= min(row + ${order}u, ${slices - 1}u); below++) { solution[row] -= band[row * ${bandStride}u + (below - row)] * solution[below]; }
    }
    if (iteration == ${reweightPassCount}u) { break; }
    var residualCount = 0u;
    var largestNegative = 0.0;
    for (var slice = 0u; slice < ${slices}u; slice++) {
      if (baseWeight(index, slice) <= 0.0) { continue; }
      let residual = values[valuesOffset + index * ${slices}u + slice] - solution[slice];
      workspace[residualCount] = abs(residual); residualCount++;
      if (residual < 0.0) { largestNegative = max(largestNegative, -residual); }
    }
    let scale = 6.0 * getMedian(&workspace, residualCount);
    for (var slice = 0u; slice < ${slices}u; slice++) {
      if (baseWeight(index, slice) <= 0.0) { continue; }
      let residual = values[valuesOffset + index * ${slices}u + slice] - solution[slice];
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
  smoothed[smoothedOffset + index * ${slices}u + slice] = select(missing(), solution[slice], code == 0u);
}`
      })
    ];
  }
}
