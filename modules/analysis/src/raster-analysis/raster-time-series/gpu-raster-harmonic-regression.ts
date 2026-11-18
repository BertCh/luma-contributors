// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import type {
  GraphDataView,
  GraphVectorView,
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {
  createRasterTimeSeriesKernelNode,
  planRasterTimeSeriesChunks,
  splitRasterTimeSeriesChunks,
  validateRasterTimeSeriesBindings,
  validateRasterTimeSeriesViews,
  type RasterTimeSeriesField,
  type RasterTimeSeriesScalarView
} from './raster-time-series-chunks';
import {
  GPU_CORE_MAX_STORAGE_BUFFER_BINDING_SIZE,
  validateCapacity,
  validateCommandNodeBudget,
  validateFiniteNumber,
  validateGPUStorageBindingByteLength,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  RASTER_FINITE_WGSL,
  getRasterLeastSquaresWGSL,
  rasterBinding
} from '../remote-sensing/raster-kernel-utils';
import {
  getGPURasterDateAxisBinding,
  getGPURasterDateAxisDeclarations,
  resolveGPURasterIrregularDateAxis,
  type GPURasterIrregularDateAxis
} from './raster-irregular-date-axis';
import {getRasterHarmonicTrendDesignWGSL} from './raster-harmonic-trend-design-wgsl';

/** Slices: private design arrays hold one row per slice. */
export const MAXIMUM_HARMONIC_REGRESSION_SLICE_COUNT = 256;
/** Storage-backed QR removes slice-sized private arrays; cap the per-cell serial QR walk at 4096. */
export const MAXIMUM_STORAGE_HARMONIC_REGRESSION_SLICE_COUNT = 4096;
/** Bound each basis batch to the core 128 MiB binding so packed callers never need extra scratch headroom. */
const HARMONIC_REGRESSION_BASIS_BATCH_BYTE_LENGTH = GPU_CORE_MAX_STORAGE_BUFFER_BINDING_SIZE;
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
export type GPURasterHarmonicRegressionProps<
  Rows extends GraphDataView<'float32'> | GraphVectorView<'float32'> = GraphDataView<'float32'>
> = {
  id?: string;
  cellCount: number;
  sliceCount: number;
  /** Cell-major `cell * sliceCount + slice`; nonfinite samples are missing. */
  values: Rows;
  /** Per-observation QA; zero rejects the sample. */
  mask?: RasterTimeSeriesField<'uint32', Rows>;
  /** Float32, see {@link getGPURasterHarmonicRegressionParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Harmonics of the base period; coefficients per cell are `2 + 2 * harmonicCount`. */
  harmonicCount: number;
  /** Storage-backed QR (default), or the original private-array QR with its 256-slice/memory limits. */
  mode?: 'storage' | 'reference';
  /** Acquisition dates; the regression is defined on real elapsed years. */
  dateAxis: GPURasterIrregularDateAxis;
  output: {
    /** Per cell: intercept (at the axis midpoint), slope per year, then cosine and sine per harmonic. NaN unless status is 0. */
    coefficients: RasterTimeSeriesField<'float32', Rows>;
    /** Two words per cell: `[status, observationCount]`. `status`: 0 fitted, 1 at most `2 + 2 * harmonicCount` valid observations, 2 rank failure. */
    summary: RasterTimeSeriesField<'uint32', Rows>;
    /** Optional fitted curve at every slice, gaps included; NaN unless status is 0. */
    fitted?: RasterTimeSeriesField<'float32', Rows>;
    /** Optional per cell residual standard deviation with `count - coefficients` degrees of freedom. */
    rmse?: RasterTimeSeriesField<'float32', Rows>;
  };
};

/**
 * Trend and harmonic ordinary least squares per cell on an irregular acquisition axis: the
 * seasonal-trend model of CCDC and BFAST, without monitoring. Time is centred fractional years, so
 * the design stays well conditioned. Both routes use the shared two-pass QR algorithm. The default
 * storage route holds the cell-local basis in device-checked, bounded batches and accepts up to 4096 slices;
 * `mode: 'reference'` retains the original private-array QR and its 256-slice/memory limits.
 * Route comparisons use absolute tolerance 5e-4 on unit-scale, well-conditioned fits; both use
 * the shared float32 rank test.
 *
 * Cell-major fields accept packed views or matching vector chunks of complete cell records.
 * Shared date, model and parameter tables remain packed.
 *
 * Per-frame: `parameters`. Compile-time: `cellCount`, `sliceCount`, `harmonicCount`, `mode`, and which
 * optional views exist.
 */
export class GPURasterHarmonicRegression<
  Rows extends GraphDataView<'float32'> | GraphVectorView<'float32'> = GraphDataView<'float32'>
> implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly estimatedCommandNodeCount: number;
  readonly mode: 'storage' | 'reference';
  private readonly chunkProps: readonly GPURasterHarmonicRegressionProps<Rows>[];
  private readonly ticksPerYear: number;

  constructor(readonly props: GPURasterHarmonicRegressionProps<Rows>) {
    const id = props.id ?? 'raster-harmonic-regression';
    this.id = id;
    this.mode = props.mode ?? 'storage';
    validateRequirement(
      id,
      this.mode === 'storage' || this.mode === 'reference',
      "mode must be 'storage' or 'reference'"
    );
    validateCapacity(id, 'cellCount', props.cellCount);
    validateCapacity(
      id,
      'sliceCount',
      props.sliceCount,
      1,
      this.mode === 'reference'
        ? MAXIMUM_HARMONIC_REGRESSION_SLICE_COUNT
        : MAXIMUM_STORAGE_HARMONIC_REGRESSION_SLICE_COUNT
    );
    validateCapacity(id, 'harmonicCount', props.harmonicCount, 1);
    const columns = 2 + 2 * props.harmonicCount;
    validateRequirement(
      id,
      (this.mode === 'reference'
        ? Math.max(
            props.sliceCount * columns + columns * columns + 2 * columns + 8,
            props.sliceCount * columns + 2 * props.sliceCount + columns + 64
          )
        : columns * columns + 2 * columns + 8) <= HARMONIC_REGRESSION_WORKING_MEMORY_FLOAT_COUNT,
      `per-thread QR working memory (sliceCount ${props.sliceCount}, ${columns} coefficients) must fit ${HARMONIC_REGRESSION_WORKING_MEMORY_FLOAT_COUNT} floats`
    );
    const observationCount = props.cellCount * props.sliceCount;
    validateCapacity(id, 'cellCount times sliceCount', observationCount);
    validateRequirement(
      id,
      [props.values, props.mask, props.output.fitted].every(
        view => !view || view.length >= observationCount
      ),
      'values, mask and output.fitted must hold cellCount times sliceCount entries'
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
    const recordWidths = {
      values: props.sliceCount,
      mask: props.sliceCount,
      'output.coefficients': columns,
      'output.summary': 2,
      'output.fitted': props.sliceCount,
      'output.rmse': 1
    };
    const chunks = planRasterTimeSeriesChunks(id, props, recordWidths);
    this.chunkProps =
      this.mode === 'storage'
        ? splitRasterTimeSeriesChunks(
            chunks,
            recordWidths,
            Math.floor(
              HARMONIC_REGRESSION_BASIS_BATCH_BYTE_LENGTH /
                (props.sliceCount * columns * Float32Array.BYTES_PER_ELEMENT)
            )
          )
        : chunks;
    if (this.mode === 'storage') {
      for (const chunk of this.chunkProps)
        validateCapacity(
          id,
          'QR basis scratch entries',
          chunk.cellCount * props.sliceCount * columns
        );
    }
    this.estimatedCommandNodeCount =
      this.chunkProps.length * (this.mode === 'storage' && props.output.fitted ? 2 : 1);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    const inputs = this.getInputs();
    const outputs = this.getOutputs();
    validateRasterTimeSeriesViews(id, inputs, outputs, [
      {
        inputs,
        outputs:
          this.mode === 'reference' ? outputs : outputs.filter(view => view !== props.output.fitted)
      },
      ...(this.mode === 'storage'
        ? this.chunkProps.flatMap(chunk =>
            chunk.output.fitted
              ? [
                  {
                    inputs: [
                      chunk.parameters,
                      chunk.dateAxis.acquisitionTimes,
                      chunk.output.coefficients,
                      chunk.output.summary
                    ],
                    outputs: [chunk.output.fitted]
                  }
                ]
              : []
          )
        : [])
    ]);
  }

  private getInputs(): RasterTimeSeriesScalarView[] {
    const {values, mask, parameters} = this.props;
    return [
      values,
      parameters,
      ...(mask ? [mask] : []),
      ...(this.props.dateAxis ? [this.props.dateAxis.acquisitionTimes] : [])
    ];
  }

  private getOutputs(): RasterTimeSeriesScalarView[] {
    const {coefficients, summary, fitted, rmse} = this.props.output;
    return [coefficients, summary, ...(fitted ? [fitted] : []), ...(rmse ? [rmse] : [])];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateRasterTimeSeriesBindings(graph, this.id, [...this.getInputs(), ...this.getOutputs()]);
    if (this.mode === 'storage') {
      validateCapacity(
        this.id,
        'storage binding count',
        6 + (this.props.mask ? 1 : 0) + (this.props.output.rmse ? 1 : 0),
        1,
        graph.device.limits.maxStorageBuffersPerShaderStage
      );
      for (const chunk of this.chunkProps)
        validateGPUStorageBindingByteLength(
          graph.device,
          this.id,
          'QR basis scratch',
          chunk.cellCount *
            this.props.sliceCount *
            (2 + 2 * this.props.harmonicCount) *
            Float32Array.BYTES_PER_ELEMENT,
          'reduce the cell count per chunk or sliceCount'
        );
    }
    return this.chunkProps.flatMap((props, chunkIndex) =>
      this.getChunkCommandNodes(graph, {
        ...props,
        id: `${this.id}-chunk-${chunkIndex}`
      } as unknown as GPURasterHarmonicRegressionProps)
    );
  }

  private getChunkCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    props: GPURasterHarmonicRegressionProps
  ): readonly GPUCommandNode<Parameters>[] {
    if (this.mode === 'storage') return this.getStorageCommandNodes(graph, props);
    const id = props.id!;
    const {sliceCount: slices, harmonicCount, values, mask, parameters, dateAxis, output} = props;
    const columns = 2 + 2 * harmonicCount;
    return [
      createRasterTimeSeriesKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterHarmonicRegression',
        invocationCount: props.cellCount,
        workgroupSize: 8,
        bindings: [
          rasterBinding('values', values),
          rasterBinding('parameters', parameters),
          ...(mask ? [rasterBinding('mask', mask)] : []),
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
  if (!finite(value) ${mask ? '|| mask[maskOffset + source] == 0u' : ''}) { continue; }
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

  private getStorageCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    props: GPURasterHarmonicRegressionProps
  ): readonly GPUCommandNode<Parameters>[] {
    const {sliceCount: slices, harmonicCount, values, mask, parameters, dateAxis, output} = props;
    const id = props.id!;
    const columns = 2 + 2 * harmonicCount;
    const basis = createTransientView(
      graph,
      `${id}-qr-basis`,
      'float32',
      props.cellCount * slices * columns,
      Buffer.STORAGE
    );
    // Only the fit needs the basis. Publishing the optional fitted curve separately keeps every
    // stage within WebGPU's eight storage bindings, including mask, dates and optional RMSE.
    const designDeclarations = `${RASTER_FINITE_WGSL}
${getGPURasterDateAxisDeclarations(this.ticksPerYear, slices)}
${getRasterHarmonicTrendDesignWGSL(slices)
  .replace(/var<private> sliceYears:[\s\S]*?fn getHarmonicTrendColumn/, 'fn getHarmonicTrendColumn')
  .replaceAll('sliceYears[slice]', 'rasterSliceYears(slice)')}`;
    const nodes = [
      createRasterTimeSeriesKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterHarmonicRegression',
        variant: 'storage-qr',
        invocationCount: props.cellCount,
        workgroupSize: 8,
        bindings: [
          rasterBinding('values', values),
          rasterBinding('parameters', parameters),
          ...(mask ? [rasterBinding('mask', mask)] : []),
          getGPURasterDateAxisBinding(dateAxis),
          rasterBinding('qrBasis', basis, true),
          rasterBinding('coefficients', output.coefficients, true),
          rasterBinding('summary', output.summary, true),
          ...(output.rmse ? [rasterBinding('rmse', output.rmse, true)] : [])
        ],
        declarations: `${designDeclarations}
fn isHarmonicObservationPresent(cell: u32, slice: u32) -> bool {
  let source = cell * ${slices}u + slice;
  return finite(values[valuesOffset + source]) ${mask ? '&& mask[maskOffset + source] != 0u' : ''};
}
fn getHarmonicResponse(cell: u32, slice: u32) -> f32 {
  if (!isHarmonicObservationPresent(cell, slice)) { return 0.0; }
  return values[valuesOffset + cell * ${slices}u + slice];
}
fn getMaskedHarmonicColumn(cell: u32, slice: u32, column: u32, period: f32) -> f32 {
  if (!isHarmonicObservationPresent(cell, slice)) { return 0.0; }
  return getHarmonicTrendColumn(slice, column, period);
}
${getStorageHarmonicRegressionQRWGSL(slices, columns)}`,
        body: `let period = parameters[parametersOffset];
let basisBase = qrBasisOffset + index * ${slices * columns}u;
var count = 0u;
for (var slice = 0u; slice < ${slices}u; slice++) {
  if (isHarmonicObservationPresent(index, slice)) { count++; }
  for (var column = 0u; column < ${columns}u; column++) {
    qrBasis[basisBase + column * ${slices}u + slice] = getMaskedHarmonicColumn(index, slice, column, period);
  }
}
var code = 1u;
var fit: LeastSquaresResult;
if (count > ${columns}u) {
  fit = leastSquares(index, period);
  code = select(2u, 0u, fit.valid);
}
for (var column = 0u; column < ${columns}u; column++) {
  coefficients[coefficientsOffset + index * ${columns}u + column] = select(missing(), fit.coefficients[column], code == 0u);
}
summary[summaryOffset + index * 2u] = code;
summary[summaryOffset + index * 2u + 1u] = count;
${output.rmse ? `rmse[rmseOffset + index] = select(missing(), sqrt(fit.residual / f32(count - min(count, ${columns}u))), code == 0u);` : ''}`
      })
    ];
    if (output.fitted)
      nodes.push(
        createRasterTimeSeriesKernelNode<Parameters>(graph, {
          id: `${id}-fitted`,
          operation: 'GPURasterHarmonicRegression',
          variant: 'fitted',
          invocationCount: props.cellCount,
          workgroupSize: 8,
          bindings: [
            rasterBinding('parameters', parameters),
            getGPURasterDateAxisBinding(dateAxis),
            rasterBinding('coefficients', output.coefficients),
            rasterBinding('summary', output.summary),
            rasterBinding('fitted', output.fitted, true)
          ],
          declarations: designDeclarations,
          body: `let period = parameters[parametersOffset];
let code = summary[summaryOffset + index * 2u];
for (var slice = 0u; slice < ${slices}u; slice++) {
  var prediction = 0.0;
  for (var column = 0u; column < ${columns}u; column++) {
    prediction += coefficients[coefficientsOffset + index * ${columns}u + column] * getHarmonicTrendColumn(slice, column, period);
  }
  fitted[fittedOffset + index * ${slices}u + slice] = select(missing(), prediction, code == 0u);
}`
        })
      );
    return nodes;
  }
}

/** Reuses the shared two-pass QR algorithm, adapting only its basis and response storage. */
function getStorageHarmonicRegressionQRWGSL(slices: number, columns: number): string {
  return getRasterLeastSquaresWGSL(slices, columns)
    .replace(
      `fn leastSquares(design: array<f32, ${slices * columns}>, response: array<f32, ${slices}>)`,
      'fn leastSquares(cell: u32, period: f32)'
    )
    .replace('var basis = design;', `let basisBase = qrBasisOffset + cell * ${slices * columns}u;`)
    .replace(/basis\[([^\]]+)\]/g, 'qrBasis[basisBase + $1]')
    .replaceAll('response[row]', 'getHarmonicResponse(cell, row)')
    .replaceAll(
      `design[column * ${slices}u + row]`,
      'getMaskedHarmonicColumn(cell, row, column, period)'
    );
}
