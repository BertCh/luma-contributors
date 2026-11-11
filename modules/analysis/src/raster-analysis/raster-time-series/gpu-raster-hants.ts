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

/** Slices: private design arrays hold one row per slice. */
export const MAXIMUM_HANTS_SLICE_COUNT = 256;
/** Per-thread QR working memory in floats (8 KiB), the bound shared with the seasonal-break fit. */
const HANTS_WORKING_MEMORY_FLOAT_COUNT = 2048;
/** Reject-and-refit passes are unrolled per cell, so the count is capped. */
export const MAXIMUM_HANTS_REJECTION_PASS_COUNT = 16;
/** Float32 elements of the per-frame parameters: period, fit-error tolerance, overdetermination. */
export const GPU_RASTER_HANTS_PARAMETER_LENGTH = 3;

/** Per-frame settings of {@link GPURasterHANTS}. */
export type GPURasterHANTSSettings = {
  /** Base period of the harmonics, in fractional years with a date axis, otherwise in slices. */
  period: number;
  /** Fit-error tolerance: observations further than this from the curve are rejected. */
  fitErrorTolerance: number;
  /** Observations kept above the `2 * harmonicCount + 1` minimum; defaults to 1. */
  degreeOfOverdetermination?: number;
};

/** Packs {@link GPURasterHANTSSettings}: `[period, fitErrorTolerance, degreeOfOverdetermination]`. */
export function getGPURasterHANTSParameterValues(
  settings: GPURasterHANTSSettings,
  target: Float32Array = new Float32Array(GPU_RASTER_HANTS_PARAMETER_LENGTH)
): Float32Array {
  validateFiniteNumber('getGPURasterHANTSParameterValues', 'period', settings.period, 0);
  target[0] = settings.period;
  target[1] = settings.fitErrorTolerance;
  target[2] = settings.degreeOfOverdetermination ?? 1;
  return target;
}

/** Properties of {@link GPURasterHANTS}. */
export type GPURasterHANTSProps = {
  id?: string;
  cellCount: number;
  sliceCount: number;
  /** Cell-major `cell * sliceCount + slice`; nonfinite samples are missing. */
  values: GraphDataView<'float32'>;
  /** Per-observation QA; zero rejects the sample. */
  mask?: GraphDataView<'uint32'>;
  /** Float32, see {@link getGPURasterHANTSParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Harmonics; coefficients per cell are `2 * harmonicCount + 1`. */
  harmonicCount: number;
  /** Which residual sign is rejected: `'low'` (cloud-depressed), `'high'`, or `'both'`; defaults to `'low'`. */
  rejectionSide?: 'low' | 'high' | 'both';
  /** Reject-and-refit passes; 0 to 16, defaults to 8. */
  rejectionPassCount?: number;
  /** Acquisition dates; without one, slices are unit-spaced regular slots. */
  dateAxis?: GPURasterIrregularDateAxis;
  output: {
    /** Fitted curve at every slice, gaps included; NaN when status is 1 or 2. */
    smoothed: GraphDataView<'float32'>;
    /** Per cell: mean, then cosine and sine per harmonic, centred on the axis midpoint. */
    coefficients: GraphDataView<'float32'>;
    /** Per cell: 0 converged, 1 insufficient support, 2 rank failure, 8 rejection passes exhausted (partial result published). */
    status: GraphDataView<'uint32'>;
    /** Optional per cell `[rejectedObservationCount, passesUsed]`. */
    diagnostics?: GraphDataView<'uint32'>;
  };
};

/**
 * Harmonic ANalysis of Time Series per cell (Roerink, Menenti and Verhoef 2000; GRASS `r.hants`):
 * a least-squares harmonic fit with a bounded reject-and-refit loop that drops the worst
 * observation beyond the fit-error tolerance on the selected side and refits, keeping at least
 * `2 * harmonicCount + 1 + degreeOfOverdetermination` observations. A cell that runs out of passes
 * with outliers remaining publishes its latest fit and reports status 8.
 *
 * Per-frame: `parameters`. Compile-time: `cellCount`, `sliceCount`, `harmonicCount`,
 * `rejectionSide`, `rejectionPassCount`, and which optional views exist.
 */
export class GPURasterHANTS implements GPUCommandNodeProducer {
  readonly id: string;
  readonly rejectionPassCount: number;
  readonly rejectionSide: 'low' | 'high' | 'both';
  private readonly ticksPerYear: number;

  constructor(readonly props: GPURasterHANTSProps) {
    const id = props.id ?? 'raster-hants';
    this.id = id;
    const passCount = props.rejectionPassCount ?? 8;
    this.rejectionPassCount = passCount;
    const side = props.rejectionSide ?? 'low';
    this.rejectionSide = side;
    validateCapacity(id, 'cellCount', props.cellCount);
    validateCapacity(id, 'sliceCount', props.sliceCount, 1, MAXIMUM_HANTS_SLICE_COUNT);
    validateCapacity(id, 'harmonicCount', props.harmonicCount, 1);
    const columns = 2 * props.harmonicCount + 1;
    validateRequirement(
      id,
      Number.isInteger(passCount) &&
        passCount >= 0 &&
        passCount <= MAXIMUM_HANTS_REJECTION_PASS_COUNT,
      `rejectionPassCount must be an integer in [0, ${MAXIMUM_HANTS_REJECTION_PASS_COUNT}]`
    );
    validateRequirement(
      id,
      side === 'low' || side === 'high' || side === 'both',
      "rejectionSide must be 'low', 'high' or 'both'"
    );
    validateRequirement(
      id,
      Math.max(
        props.sliceCount * columns + columns * columns + 2 * columns + 8,
        props.sliceCount * columns +
          props.sliceCount +
          Math.ceil(props.sliceCount / 32) +
          columns +
          64
      ) <= HANTS_WORKING_MEMORY_FLOAT_COUNT,
      `per-thread QR working memory (sliceCount ${props.sliceCount}, ${columns} coefficients) must fit ${HANTS_WORKING_MEMORY_FLOAT_COUNT} floats`
    );
    const observationCount = props.cellCount * props.sliceCount;
    validateCapacity(id, 'cellCount times sliceCount', observationCount);
    validateRequirement(
      id,
      [props.values, props.mask, props.output.smoothed].every(
        view => !view || view.length >= observationCount
      ),
      'values, mask and output.smoothed must hold cellCount times sliceCount entries'
    );
    validateRequirement(
      id,
      props.output.coefficients.length >= props.cellCount * columns,
      `output.coefficients must hold ${columns} entries per cell`
    );
    validateRequirement(
      id,
      props.output.status.length >= props.cellCount,
      'output.status must hold one entry per cell'
    );
    validateRequirement(
      id,
      !props.output.diagnostics || props.output.diagnostics.length >= props.cellCount * 2,
      'output.diagnostics must hold two entries per cell'
    );
    validateRequirement(
      id,
      props.parameters.length >= GPU_RASTER_HANTS_PARAMETER_LENGTH,
      `parameters must hold ${GPU_RASTER_HANTS_PARAMETER_LENGTH} entries`
    );
    this.ticksPerYear = props.dateAxis
      ? resolveGPURasterIrregularDateAxis(id, props.dateAxis, props.sliceCount)
      : 0;
    validateRasterViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterScalarView[] {
    const {values, mask, parameters} = this.props;
    return [values, parameters, ...(mask ? [mask] : [])];
  }

  private getOutputs(): RasterScalarView[] {
    const {smoothed, coefficients, status, diagnostics} = this.props.output;
    return [smoothed, coefficients, status, ...(diagnostics ? [diagnostics] : [])];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, rejectionPassCount, rejectionSide} = this;
    const {sliceCount: slices, harmonicCount, values, mask, parameters, dateAxis, output} = props;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    if (dateAxis) validateGPURasterDateAxisGraph(id, graph, dateAxis);
    const columns = 2 * harmonicCount + 1;
    const timeExpression = dateAxis
      ? 'rasterSliceYears(slice)'
      : `f32(slice) - ${(slices - 1) * 0.5}`;
    const rejectionMagnitude =
      rejectionSide === 'both'
        ? 'abs(residual)'
        : rejectionSide === 'low'
          ? '-residual'
          : 'residual';
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterHANTS',
        invocationCount: props.cellCount,
        workgroupSize: 8,
        bindings: [
          rasterBinding('values', values),
          rasterBinding('parameters', parameters),
          ...(mask ? [rasterBinding('mask', mask)] : []),
          ...(dateAxis ? [getGPURasterDateAxisBinding(dateAxis)] : []),
          rasterBinding('smoothed', output.smoothed, true),
          rasterBinding('coefficients', output.coefficients, true),
          rasterBinding('status', output.status, true),
          ...(output.diagnostics ? [rasterBinding('diagnostics', output.diagnostics, true)] : [])
        ],
        declarations: `${RASTER_FINITE_WGSL}
${dateAxis ? getGPURasterDateAxisDeclarations(this.ticksPerYear, slices) : ''}
${getRasterLeastSquaresWGSL(slices, columns)}
fn getHarmonicColumn(slice: u32, column: u32, period: f32) -> f32 {
  if (column == 0u) { return 1.0; }
  let harmonic = (column + 1u) / 2u;
  let angle = 6.283185307179586 * f32(harmonic) * (${timeExpression}) / period;
  return select(sin(angle), cos(angle), column % 2u == 1u);
}`,
        body: `let period = parameters[parametersOffset];
let tolerance = parameters[parametersOffset + 1u];
let overdetermination = u32(max(parameters[parametersOffset + 2u], 0.0));
let minimumSupport = ${columns}u + overdetermination;
var keptMask: array<u32, ${Math.ceil(slices / 32)}>;
var keptCount = 0u;
for (var slice = 0u; slice < ${slices}u; slice++) {
  let source = index * ${slices}u + slice;
  if (finite(values[valuesOffset + source]) ${mask ? '&& mask[maskOffset + source] != 0u' : ''}) { keptMask[slice / 32u] |= 1u << (slice % 32u); keptCount++; }
}
var code = 0u;
var fitted: LeastSquaresResult;
var rejected = 0u;
var passesUsed = 0u;
if (keptCount < minimumSupport) {
  code = 1u;
} else {
  var finished = false;
  for (var iteration = 0u; iteration <= ${rejectionPassCount}u; iteration++) {
    var design: array<f32, ${slices * columns}>;
    var response: array<f32, ${slices}>;
    for (var slice = 0u; slice < ${slices}u; slice++) {
      if ((keptMask[slice / 32u] & (1u << (slice % 32u))) == 0u) { continue; }
      response[slice] = values[valuesOffset + index * ${slices}u + slice];
      for (var column = 0u; column < ${columns}u; column++) { design[column * ${slices}u + slice] = getHarmonicColumn(slice, column, period); }
    }
    fitted = leastSquares(design, response);
    passesUsed = iteration + 1u;
    if (!fitted.valid) { code = 2u; finished = true; break; }
    var worst = 0xffffffffu;
    var worstMagnitude = tolerance;
    for (var slice = 0u; slice < ${slices}u; slice++) {
      if ((keptMask[slice / 32u] & (1u << (slice % 32u))) == 0u) { continue; }
      var prediction = 0.0;
      for (var column = 0u; column < ${columns}u; column++) { prediction += fitted.coefficients[column] * getHarmonicColumn(slice, column, period); }
      let residual = values[valuesOffset + index * ${slices}u + slice] - prediction;
      if (${rejectionMagnitude} > worstMagnitude) { worstMagnitude = ${rejectionMagnitude}; worst = slice; }
    }
    if (worst == 0xffffffffu || keptCount <= minimumSupport) { finished = true; break; }
    if (iteration == ${rejectionPassCount}u) { break; }
    keptMask[worst / 32u] &= ~(1u << (worst % 32u));
    keptCount--;
    rejected++;
  }
  if (!finished) { code = 8u; }
}
let published = code == 0u || code == 8u;
for (var column = 0u; column < ${columns}u; column++) {
  coefficients[coefficientsOffset + index * ${columns}u + column] = select(missing(), fitted.coefficients[column], published);
}
for (var slice = 0u; slice < ${slices}u; slice++) {
  var prediction = 0.0;
  if (published) {
    for (var column = 0u; column < ${columns}u; column++) { prediction += fitted.coefficients[column] * getHarmonicColumn(slice, column, period); }
  }
  smoothed[smoothedOffset + index * ${slices}u + slice] = select(missing(), prediction, published);
}
status[statusOffset + index] = code;
${output.diagnostics ? 'diagnostics[diagnosticsOffset + index * 2u] = rejected; diagnostics[diagnosticsOffset + index * 2u + 1u] = passesUsed;' : ''}`
      })
    ];
  }
}
