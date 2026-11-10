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

/** Slices: private design arrays hold one row per slice. */
export const MAXIMUM_HARMONIC_REGRESSION_SLICE_COUNT = 256;
/** Per-thread QR working memory in floats (8 KiB), the bound shared with the seasonal-break fit. */
const HARMONIC_REGRESSION_WORKING_MEMORY_FLOAT_COUNT = 2048;
/** Float32 elements of the per-frame parameters: period. */
export const GPU_RASTER_HARMONIC_REGRESSION_PARAMETER_LENGTH = 1;

/** Per-frame settings of {@link GPURasterHarmonicRegression}. */
export type GPURasterHarmonicRegressionSettings = {
  /** Base period of the harmonics in fractional years; 1 for an annual cycle. */
  period: number;
};

/** Packs {@link GPURasterHarmonicRegressionSettings}: `[period]`. */
export function getGPURasterHarmonicRegressionParameterValues(
  settings: GPURasterHarmonicRegressionSettings,
  target: Float32Array = new Float32Array(GPU_RASTER_HARMONIC_REGRESSION_PARAMETER_LENGTH)
): Float32Array {
  validateFiniteNumber(
    'getGPURasterHarmonicRegressionParameterValues',
    'period',
    settings.period,
    0
  );
  target[0] = settings.period;
  return target;
}

/** Properties of {@link GPURasterHarmonicRegression}. */
export type GPURasterHarmonicRegressionProps = {
  id?: string;
  cellCount: number;
  sliceCount: number;
  /** Cell-major `cell * sliceCount + slice`; nonfinite samples are missing. */
  values: GraphDataView<'float32'>;
  /** Per-observation QA; zero rejects the sample. */
  validity?: GraphDataView<'uint32'>;
  /** Float32, see {@link getGPURasterHarmonicRegressionParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Harmonics of the base period; coefficients per cell are `2 + 2 * harmonicCount`. */
  harmonicCount: number;
  /** Acquisition dates; the regression is defined on real elapsed years. */
  dateAxis: GPURasterIrregularDateAxis;
  output: {
    /** Per cell: intercept (at the axis midpoint), slope per year, then cosine and sine per harmonic. NaN unless status is 0. */
    coefficients: GraphDataView<'float32'>;
    /** Two words per cell: `[status, observationCount]`. `status`: 0 fitted, 1 at most `2 + 2 * harmonicCount` valid observations, 2 rank failure. */
    summary: GraphDataView<'uint32'>;
    /** Optional fitted curve at every slice, gaps included; NaN unless status is 0. */
    fitted?: GraphDataView<'float32'>;
    /** Optional per cell residual standard deviation with `count - coefficients` degrees of freedom. */
    rmse?: GraphDataView<'float32'>;
  };
};

/**
 * Trend and harmonic ordinary least squares per cell on an irregular acquisition axis: the
 * seasonal-trend model of CCDC and BFAST, without monitoring. Time is centred fractional years, so
 * the design stays well conditioned, and the fit is the shared QR of `getRasterLeastSquaresWGSL`.
 *
 * Per-frame: `parameters`. Compile-time: `cellCount`, `sliceCount`, `harmonicCount`, and which
 * optional views exist.
 */
export class GPURasterHarmonicRegression implements GPUCommandNodeProducer {
  readonly id: string;
  private readonly ticksPerYear: number;

  constructor(readonly props: GPURasterHarmonicRegressionProps) {
    const id = props.id ?? 'raster-harmonic-regression';
    this.id = id;
    validateCapacity(id, 'cellCount', props.cellCount);
    validateCapacity(
      id,
      'sliceCount',
      props.sliceCount,
      1,
      MAXIMUM_HARMONIC_REGRESSION_SLICE_COUNT
    );
    validateCapacity(id, 'harmonicCount', props.harmonicCount, 1);
    const columns = 2 + 2 * props.harmonicCount;
    validateRequirement(
      id,
      Math.max(
        props.sliceCount * columns + columns * columns + 2 * columns + 8,
        props.sliceCount * columns + 2 * props.sliceCount + columns + 64
      ) <= HARMONIC_REGRESSION_WORKING_MEMORY_FLOAT_COUNT,
      `per-thread QR working memory (sliceCount ${props.sliceCount}, ${columns} coefficients) must fit ${HARMONIC_REGRESSION_WORKING_MEMORY_FLOAT_COUNT} floats`
    );
    const observationCount = props.cellCount * props.sliceCount;
    validateCapacity(id, 'cellCount times sliceCount', observationCount);
    validateRequirement(
      id,
      [props.values, props.validity, props.output.fitted].every(
        view => !view || view.length >= observationCount
      ),
      'values, validity and output.fitted must hold cellCount times sliceCount entries'
    );
    validateRequirement(
      id,
      props.output.coefficients.length >= props.cellCount * columns,
      `output.coefficients must hold ${columns} entries per cell`
    );
    validateRequirement(
      id,
      !props.output.rmse || props.output.rmse.length >= props.cellCount,
      'output.rmse must hold one entry per cell'
    );
    validateRequirement(
      id,
      props.output.summary.length >= props.cellCount * 2,
      'output.summary must hold two entries per cell'
    );
    validateRequirement(
      id,
      props.parameters.length >= GPU_RASTER_HARMONIC_REGRESSION_PARAMETER_LENGTH,
      `parameters must hold ${GPU_RASTER_HARMONIC_REGRESSION_PARAMETER_LENGTH} entries`
    );
    this.ticksPerYear = resolveGPURasterIrregularDateAxis(id, props.dateAxis, props.sliceCount);
    validateRasterViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterScalarView[] {
    const {values, validity, parameters} = this.props;
    return [values, parameters, ...(validity ? [validity] : [])];
  }

  private getOutputs(): RasterScalarView[] {
    const {coefficients, summary, fitted, rmse} = this.props.output;
    return [coefficients, summary, ...(fitted ? [fitted] : []), ...(rmse ? [rmse] : [])];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {
      sliceCount: slices,
      harmonicCount,
      values,
      validity,
      parameters,
      dateAxis,
      output
    } = props;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    validateGPURasterDateAxisGraph(id, graph, dateAxis);
    const columns = 2 + 2 * harmonicCount;
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterHarmonicRegression',
        invocationCount: props.cellCount,
        workgroupSize: 8,
        bindings: [
          rasterBinding('values', values),
          rasterBinding('parameters', parameters),
          ...(validity ? [rasterBinding('validity', validity)] : []),
          getGPURasterDateAxisBinding(dateAxis),
          rasterBinding('coefficients', output.coefficients, true),
          rasterBinding('summary', output.summary, true),
          ...(output.fitted ? [rasterBinding('fitted', output.fitted, true)] : []),
          ...(output.rmse ? [rasterBinding('rmse', output.rmse, true)] : [])
        ],
        declarations: `${RASTER_FINITE_WGSL}
${getGPURasterDateAxisDeclarations(this.ticksPerYear, slices)}
${getRasterHarmonicTrendDesignWGSL(slices)}
${getRasterLeastSquaresWGSL(slices, columns)}`,
        body: `fillRasterSliceYears();
let period = parameters[parametersOffset];
var design: array<f32, ${slices * columns}>;
var response: array<f32, ${slices}>;
var count = 0u;
for (var slice = 0u; slice < ${slices}u; slice++) {
  let source = index * ${slices}u + slice;
  let value = values[valuesOffset + source];
  if (!finite(value) ${validity ? '|| validity[validityOffset + source] == 0u' : ''}) { continue; }
  response[slice] = value;
  count++;
  for (var column = 0u; column < ${columns}u; column++) { design[column * ${slices}u + slice] = getHarmonicTrendColumn(slice, column, period); }
}
var code = 1u;
var fit: LeastSquaresResult;
if (count > ${columns}u) {
  fit = leastSquares(design, response);
  code = select(2u, 0u, fit.valid);
}
for (var column = 0u; column < ${columns}u; column++) {
  coefficients[coefficientsOffset + index * ${columns}u + column] = select(missing(), fit.coefficients[column], code == 0u);
}
summary[summaryOffset + index * 2u] = code;
summary[summaryOffset + index * 2u + 1u] = count;
${
  output.fitted
    ? `for (var slice = 0u; slice < ${slices}u; slice++) {
  var prediction = 0.0;
  for (var column = 0u; column < ${columns}u; column++) { prediction += fit.coefficients[column] * getHarmonicTrendColumn(slice, column, period); }
  fitted[fittedOffset + index * ${slices}u + slice] = select(missing(), prediction, code == 0u);
}`
    : ''
}
${output.rmse ? `rmse[rmseOffset + index] = select(missing(), sqrt(fit.residual / f32(count - min(count, ${columns}u))), code == 0u);` : ''}`
      })
    ];
  }
}
