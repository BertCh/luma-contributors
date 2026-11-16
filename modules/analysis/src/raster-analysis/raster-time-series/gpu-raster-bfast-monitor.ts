// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

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
  validateRasterTimeSeriesBindings,
  validateRasterTimeSeriesViews,
  type RasterTimeSeriesField,
  type RasterTimeSeriesScalarView
} from './raster-time-series-chunks';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateFiniteNumber,
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

/** Slices: private arrays hold one entry per slice. */
export const MAXIMUM_BFAST_MONITOR_SLICE_COUNT = 256;
/** Per-thread QR working memory in floats (8 KiB), the bound shared with the seasonal-break fit. */
const BFAST_MONITOR_WORKING_MEMORY_FLOAT_COUNT = 2048;
/** Float32 elements of the per-frame parameters: period, bandwidth, critical value. */
export const GPU_RASTER_BFAST_MONITOR_PARAMETER_LENGTH = 3;
/** `output.breakSlice` when the boundary was never crossed. */
export const GPU_RASTER_BFAST_MONITOR_NO_BREAK = 0xffffffff;

/** Per-frame settings of {@link GPURasterBFASTMonitor}. */
export type GPURasterBFASTMonitorSettings = {
  /** Base period of the harmonics in fractional years; 1 for an annual cycle. */
  period?: number;
  /** MOSUM bandwidth as a fraction of the valid history observations, in `(0, 1]`; bfastmonitor uses 0.25. */
  bandwidth?: number;
  /**
   * Critical value `lambda` of the boundary `lambda * sqrt(2 * log+(t / n))`. It depends on the
   * bandwidth, the monitoring horizon and the significance level; choose it from the published
   * critical-value tables for those settings. No table is shipped.
   */
  criticalValue: number;
};

/** Packs {@link GPURasterBFASTMonitorSettings}: `[period, bandwidth, criticalValue]`. */
export function getGPURasterBFASTMonitorParameterValues(
  settings: GPURasterBFASTMonitorSettings,
  target: Float32Array = new Float32Array(GPU_RASTER_BFAST_MONITOR_PARAMETER_LENGTH)
): Float32Array {
  const id = 'getGPURasterBFASTMonitorParameterValues';
  validateFiniteNumber(id, 'criticalValue', settings.criticalValue, 0);
  target[0] = settings.period ?? 1;
  target[1] = settings.bandwidth ?? 0.25;
  target[2] = settings.criticalValue;
  return target;
}

/** Properties of {@link GPURasterBFASTMonitor}. */
export type GPURasterBFASTMonitorProps<
  Rows extends GraphDataView<'float32'> | GraphVectorView<'float32'> = GraphDataView<'float32'>
> = {
  id?: string;
  cellCount: number;
  sliceCount: number;
  /** Cell-major `cell * sliceCount + slice`; nonfinite samples are missing. */
  values: Rows;
  /** Per-observation QA; zero rejects the sample. */
  mask?: RasterTimeSeriesField<'uint32', Rows>;
  /** Float32, see {@link getGPURasterBFASTMonitorParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Acquisition dates; the model is defined on real elapsed years. */
  dateAxis: GPURasterIrregularDateAxis;
  /** Leading slices forming the stable history; monitoring runs over the remaining slices. */
  historySliceCount: number;
  /** Harmonics; defaults to 3, bfastmonitor's `order`. */
  harmonicCount?: number;
  /** Include the linear trend column; defaults to true. */
  includeTrend?: boolean;
  output: {
    /** Per cell: intercept, slope per year when `includeTrend`, then cosine and sine per harmonic. NaN unless status is 0. */
    coefficients: RasterTimeSeriesField<'float32', Rows>;
    /** Two floats per cell: `[magnitude, maximumBoundaryRatio]`. `magnitude` is the median residual over the monitoring observations (bfastmonitor's `magnitude`); the ratio is the largest `|MOSUM| / boundary`. Both NaN unless status is 0. */
    scores: RasterTimeSeriesField<'float32', Rows>;
    /** Three words per cell: `[breakSlice, status, historyObservationCount]`. `breakSlice` is the first monitoring slice whose MOSUM crossed the boundary, or {@link GPU_RASTER_BFAST_MONITOR_NO_BREAK}. `status`: 0 monitored, 1 too few valid history observations, 2 rank failure or zero residual scale. */
    cellSummary: RasterTimeSeriesField<'uint32', Rows>;
    /** Optional `cell * (sliceCount - historySliceCount) + monitoringSlice`: MOSUM per monitoring slice, NaN for missing observations. */
    process?: RasterTimeSeriesField<'float32', Rows>;
  };
};

/**
 * BFAST Monitor (Verbesselt, Zeileis and Herold 2012) per cell on an irregular date axis: a
 * harmonic and optionally trend least-squares fit on a fixed stable history, then OLS-MOSUM
 * monitoring of the residuals. With `n` valid history observations, `h = floor(bandwidth * n)` and
 * residual scale `sigma` on `n - k` degrees of freedom, the process at observation `t` is
 * `sum(residuals over the last h observations) / (sigma * sqrt(n))`, compared with
 * `lambda * sqrt(2 * log+(t / n))` where `log+` is `max(1, log)`. Windows reach back into the
 * history. Missing observations are dropped, not interpolated.
 *
 * bfast is GPL and R-only, so this is written from the published definitions; no code was copied.
 * Stable-history selection (ROC and Bai-Perron) and critical-value tables are not included: the
 * history is fixed and `lambda` is supplied.
 *
 * Cell-major fields accept packed views or matching vector chunks of complete cell records.
 * Shared date, model and parameter tables remain packed.
 *
 * Per-frame: `parameters`. Compile-time: `cellCount`, `sliceCount`, `historySliceCount`,
 * `harmonicCount`, `includeTrend`, and which optional views exist.
 */
export class GPURasterBFASTMonitor<
  Rows extends GraphDataView<'float32'> | GraphVectorView<'float32'> = GraphDataView<'float32'>
> implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly estimatedCommandNodeCount: number;
  private readonly chunkProps: readonly GPURasterBFASTMonitorProps<Rows>[];
  readonly harmonicCount: number;
  readonly includeTrend: boolean;
  readonly columnCount: number;
  private readonly ticksPerYear: number;

  constructor(readonly props: GPURasterBFASTMonitorProps<Rows>) {
    const id = props.id ?? 'raster-bfast-monitor';
    this.id = id;
    this.harmonicCount = props.harmonicCount ?? 3;
    this.includeTrend = props.includeTrend ?? true;
    this.columnCount = (this.includeTrend ? 2 : 1) + 2 * this.harmonicCount;
    const {sliceCount: slices, historySliceCount: history} = props;
    validateCapacity(id, 'cellCount', props.cellCount);
    validateCapacity(id, 'sliceCount', slices, 2, MAXIMUM_BFAST_MONITOR_SLICE_COUNT);
    validateCapacity(id, 'harmonicCount', this.harmonicCount, 1);
    validateCapacity(id, 'historySliceCount', history, 1, slices - 1);
    const columns = this.columnCount;
    validateRequirement(
      id,
      Math.max(
        history * columns + columns * columns + 2 * columns + 8,
        history * columns + 3 * slices + columns + 64
      ) <= BFAST_MONITOR_WORKING_MEMORY_FLOAT_COUNT,
      `per-thread QR working memory (historySliceCount ${history}, ${columns} coefficients) must fit ${BFAST_MONITOR_WORKING_MEMORY_FLOAT_COUNT} floats`
    );
    const observations = props.cellCount * slices;
    validateCapacity(id, 'cellCount times sliceCount', observations);
    validateRequirement(
      id,
      [props.values, props.mask].every(view => !view || view.length >= observations),
      'values and mask must hold cellCount times sliceCount entries'
    );
    validateRequirement(
      id,
      props.output.coefficients.length >= props.cellCount * columns,
      `output.coefficients must hold ${columns} entries per cell`
    );
    validateRequirement(
      id,
      props.output.scores.length >= props.cellCount * 2,
      'output.scores must hold two entries per cell'
    );
    validateRequirement(
      id,
      props.output.cellSummary.length >= props.cellCount * 3,
      'output.cellSummary must hold three entries per cell'
    );
    validateRequirement(
      id,
      !props.output.process || props.output.process.length >= props.cellCount * (slices - history),
      'output.process must hold one entry per cell and monitoring slice'
    );
    validateRequirement(
      id,
      props.parameters.length >= GPU_RASTER_BFAST_MONITOR_PARAMETER_LENGTH,
      `parameters must hold ${GPU_RASTER_BFAST_MONITOR_PARAMETER_LENGTH} entries`
    );
    this.ticksPerYear = resolveGPURasterIrregularDateAxis(id, props.dateAxis, slices);
    this.chunkProps = planRasterTimeSeriesChunks(id, props, {
      values: props.sliceCount,
      mask: props.sliceCount,
      'output.coefficients': this.columnCount,
      'output.scores': 2,
      'output.cellSummary': 3,
      'output.process': props.sliceCount - props.historySliceCount
    });
    this.estimatedCommandNodeCount = this.chunkProps.length * 1;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateRasterTimeSeriesViews(id, this.getInputs(), this.getOutputs());
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
    const {coefficients, scores, cellSummary, process} = this.props.output;
    return [coefficients, scores, cellSummary, ...(process ? [process] : [])];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateRasterTimeSeriesBindings(graph, this.id, [...this.getInputs(), ...this.getOutputs()]);
    return this.chunkProps.flatMap((props, chunkIndex) =>
      this.getChunkCommandNodes(graph, {
        ...props,
        id: `${this.id}-chunk-${chunkIndex}`
      } as unknown as GPURasterBFASTMonitorProps)
    );
  }

  private getChunkCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    props: GPURasterBFASTMonitorProps
  ): readonly GPUCommandNode<Parameters>[] {
    const {columnCount: columns, includeTrend} = this;
    const id = props.id!;
    const {
      sliceCount: slices,
      historySliceCount: history,
      values,
      mask,
      parameters,
      output
    } = props;
    const monitoringSlices = slices - history;
    return [
      createRasterTimeSeriesKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterBFASTMonitor',
        invocationCount: props.cellCount,
        workgroupSize: 8,
        bindings: [
          rasterBinding('values', values),
          rasterBinding('parameters', parameters),
          ...(mask ? [rasterBinding('mask', mask)] : []),
          getGPURasterDateAxisBinding(props.dateAxis),
          rasterBinding('coefficients', output.coefficients, true),
          rasterBinding('scores', output.scores, true),
          rasterBinding('cellSummary', output.cellSummary, true),
          ...(output.process ? [rasterBinding('process', output.process, true)] : [])
        ],
        declarations: `${RASTER_FINITE_WGSL}
${getGPURasterDateAxisDeclarations(this.ticksPerYear, slices)}
${getRasterHarmonicTrendDesignWGSL(slices, includeTrend)}
${getRasterLeastSquaresWGSL(history, columns)}
fn getLogPlus(value: f32) -> f32 { return select(1.0, log(value), value > 2.718281828459045); }
fn getSortedMedian(sample: ptr<function, array<f32, ${monitoringSlices}>>, count: u32) -> f32 {
  for (var outer = 1u; outer < count; outer++) {
    let item = (*sample)[outer];
    var inner = outer;
    while (inner > 0u && (*sample)[inner - 1u] > item) { (*sample)[inner] = (*sample)[inner - 1u]; inner--; }
    (*sample)[inner] = item;
  }
  if (count % 2u == 1u) { return (*sample)[count / 2u]; }
  return 0.5 * ((*sample)[count / 2u - 1u] + (*sample)[count / 2u]);
}`,
        body: `fillRasterSliceYears();
let period = parameters[parametersOffset];
let bandwidth = parameters[parametersOffset + 1u];
let criticalValue = parameters[parametersOffset + 2u];
var slicesUsed: array<u32, ${slices}>;
var total = 0u;
var historyTotal = 0u;
var design: array<f32, ${history * columns}>;
var response: array<f32, ${history}>;
for (var slice = 0u; slice < ${slices}u; slice++) {
  let source = index * ${slices}u + slice;
  let value = values[valuesOffset + source];
  if (!finite(value) ${mask ? '|| mask[maskOffset + source] == 0u' : ''}) { continue; }
  slicesUsed[total] = slice;
  total++;
  if (slice < ${history}u) {
    historyTotal++;
    response[slice] = value;
    for (var column = 0u; column < ${columns}u; column++) { design[column * ${history}u + slice] = getHarmonicTrendColumn(slice, column, period); }
  }
}
var code = 1u;
var fit: LeastSquaresResult;
var scale = 0.0;
if (historyTotal > ${columns}u) {
  fit = leastSquares(design, response);
  scale = sqrt(fit.residual / f32(historyTotal - ${columns}u));
  code = select(2u, 0u, fit.valid && finite(scale) && scale > 0.0);
}
for (var column = 0u; column < ${columns}u; column++) {
  coefficients[coefficientsOffset + index * ${columns}u + column] = select(missing(), fit.coefficients[column], code == 0u);
}
${
  output.process
    ? `for (var monitoringSlice = 0u; monitoringSlice < ${monitoringSlices}u; monitoringSlice++) { process[processOffset + index * ${monitoringSlices}u + monitoringSlice] = missing(); }`
    : ''
}
var breakAt = ${GPU_RASTER_BFAST_MONITOR_NO_BREAK}u;
var largestRatio = 0.0;
var monitoredMedian = missing();
if (code == 0u) {
  var residuals: array<f32, ${slices}>;
  var monitored: array<f32, ${monitoringSlices}>;
  var monitoredTotal = 0u;
  for (var position = 0u; position < total; position++) {
    let slice = slicesUsed[position];
    var prediction = 0.0;
    for (var column = 0u; column < ${columns}u; column++) { prediction += fit.coefficients[column] * getHarmonicTrendColumn(slice, column, period); }
    residuals[position] = values[valuesOffset + index * ${slices}u + slice] - prediction;
    if (position >= historyTotal) { monitored[monitoredTotal] = residuals[position]; monitoredTotal++; }
  }
  let bandwidthCount = max(1u, u32(floor(bandwidth * f32(historyTotal))));
  let processScale = scale * sqrt(f32(historyTotal));
  for (var position = historyTotal; position < total; position++) {
    var windowTotal = 0.0;
    let windowStart = position + 1u - min(bandwidthCount, position + 1u);
    for (var entry = windowStart; entry <= position; entry++) { windowTotal += residuals[entry]; }
    let processValue = windowTotal / processScale;
    let boundary = criticalValue * sqrt(2.0 * getLogPlus(f32(position + 1u) / f32(historyTotal)));
    largestRatio = max(largestRatio, abs(processValue) / boundary);
    if (breakAt == ${GPU_RASTER_BFAST_MONITOR_NO_BREAK}u && abs(processValue) > boundary) { breakAt = slicesUsed[position]; }
    ${output.process ? `process[processOffset + index * ${monitoringSlices}u + slicesUsed[position] - ${history}u] = processValue;` : ''}
  }
  if (monitoredTotal > 0u) { monitoredMedian = getSortedMedian(&monitored, monitoredTotal); }
}
scores[scoresOffset + index * 2u] = select(missing(), monitoredMedian, code == 0u);
scores[scoresOffset + index * 2u + 1u] = select(missing(), largestRatio, code == 0u);
cellSummary[cellSummaryOffset + index * 3u] = breakAt;
cellSummary[cellSummaryOffset + index * 3u + 1u] = code;
cellSummary[cellSummaryOffset + index * 3u + 2u] = historyTotal;`
      })
    ];
  }
}
