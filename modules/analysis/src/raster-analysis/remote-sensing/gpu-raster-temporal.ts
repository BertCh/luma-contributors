// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

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
} from '../raster-time-series/raster-time-series-chunks-b';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {RASTER_FINITE_WGSL, getRasterLeastSquaresWGSL} from './raster-kernel-utils';
import {
  getGPURasterDateAxisBinding,
  getGPURasterDateAxisDeclarations,
  resolveGPURasterIrregularDateAxis,
  validateGPURasterDateAxisGraph,
  type GPURasterIrregularDateAxis
} from '../raster-time-series/raster-irregular-date-axis';

export type GPURasterTemporalProps<Rows extends RasterScalarView = GraphDataView> = {
  id?: string;
  cellCount: number;
  sliceCount: number;
  /** Cell-major regular-time samples; nonfinite samples are missing. */
  values: RasterTimeSeriesView<'float32', Rows>;
  /** Per observation QA: zero rejects that acquisition for that cell. */
  mask?: RasterTimeSeriesView<'uint32', Rows>;
  operation: 'mean' | 'minimum' | 'maximum' | 'median' | 'savitzky-golay';
  /** SG only: odd 3–15 complete window, clipped or shifted at domain edges. */
  windowLength?: number;
  polynomialOrder?: number;
  edgeMode?: 'interp' | 'truncate';
  /**
   * Acquisition dates of an irregular stack. The Savitzky-Golay polynomial is fitted on real
   * elapsed time across the same window of neighbouring acquisitions; composites do not depend on
   * time and only carry the declaration. Slot-index behaviour is unchanged when this is absent.
   */
  dateAxis?: GPURasterIrregularDateAxis;
  output: {
    values: RasterTimeSeriesView<'float32', Rows>;
    counts: RasterTimeSeriesView<'uint32', Rows>;
    chosenSlice: RasterTimeSeriesView<'uint32', Rows>;
    status: RasterTimeSeriesView<'uint32', Rows>;
  };
};
/** Time slices: each invocation sorts or fits its slices in a private array of this length (256 floats is 1 KiB), and slice indices stay exact in float32. */
export const MAXIMUM_TEMPORAL_SLICE_COUNT = 256;
/** Savitzky-Golay window: the design matrix of window by (order + 1) floats lives in private memory. */
const MAXIMUM_SAVITZKY_GOLAY_WINDOW = 15;

/** Composites write one row per cell; SG writes cellCount*sliceCount rows. SG refits a
 * polynomial with QR over valid observations, rather than renormalizing signed coefficients.
 * Status 1=insufficient support, 2=rank failure. Dates use original indices; extrema take the
 * earliest tie and odd medians take the middle stable-sorted observation. Mean, even median
 * and SG have no unique source and write 0xffffffff. *
 * Cell-major views accept matching chunks of complete cell records; shared tables stay packed.
 */
export class GPURasterTemporal<Rows extends RasterScalarView = GraphDataView>
  implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly props: GPURasterTemporalProps<Rows>;
  readonly chunkPlan: RasterTimeSeriesChunkPlan;
  readonly estimatedCommandNodeCount: number;
  private readonly ticksPerYear: number;
  constructor(props: GPURasterTemporalProps<Rows> & {values: Rows}) {
    this.props = props;
    this.id = props.id ?? 'raster-temporal';
    this.ticksPerYear = props.dateAxis
      ? resolveGPURasterIrregularDateAxis(this.id, props.dateAxis, props.sliceCount)
      : 0;
    validateCapacity(this.id, 'cellCount', props.cellCount);
    validateCapacity(this.id, 'sliceCount', props.sliceCount, 1, MAXIMUM_TEMPORAL_SLICE_COUNT);
    validateCapacity(this.id, 'cellCount times sliceCount', props.cellCount * props.sliceCount);
    validateRequirement(
      this.id,
      ['mean', 'minimum', 'maximum', 'median', 'savitzky-golay'].includes(props.operation),
      "operation must be 'mean', 'minimum', 'maximum', 'median' or 'savitzky-golay'"
    );
    const smoothing = props.operation === 'savitzky-golay';
    if (smoothing) {
      const window = props.windowLength ?? 7;
      const order = props.polynomialOrder ?? 2;
      validateRequirement(
        this.id,
        Number.isInteger(window) &&
          window >= 3 &&
          window <= MAXIMUM_SAVITZKY_GOLAY_WINDOW &&
          window % 2 === 1 &&
          window <= props.sliceCount,
        `windowLength must be an odd integer in [3, ${MAXIMUM_SAVITZKY_GOLAY_WINDOW}] and at most sliceCount`
      );
      validateRequirement(
        this.id,
        Number.isInteger(order) && order >= 0 && order <= 3 && order < window,
        'polynomialOrder must be an integer in [0, 3] below windowLength'
      );
      validateRequirement(
        this.id,
        ['interp', 'truncate'].includes(props.edgeMode ?? 'interp'),
        "edgeMode must be 'interp' or 'truncate'"
      );
    }
    const inputCount = props.cellCount * props.sliceCount;
    const outputCount = smoothing ? inputCount : props.cellCount;
    validateRequirement(
      this.id,
      props.values.length >= inputCount && (!props.mask || props.mask.length >= inputCount),
      'values and mask must hold cellCount times sliceCount entries'
    );
    validateRequirement(
      this.id,
      Object.values(props.output).every(view => view.length >= outputCount),
      'output views must hold one entry per cell (per observation for savitzky-golay)'
    );
    this.chunkPlan = planRasterTimeSeriesChunks(this.id, props.cellCount, {
      values: {view: props.values, entriesPerCell: props.sliceCount},
      mask: {view: props.mask, entriesPerCell: props.sliceCount},
      'output.values': {
        view: props.output.values,
        entriesPerCell: props.operation === 'savitzky-golay' ? props.sliceCount : 1
      },
      'output.counts': {
        view: props.output.counts,
        entriesPerCell: props.operation === 'savitzky-golay' ? props.sliceCount : 1
      },
      'output.chosenSlice': {
        view: props.output.chosenSlice,
        entriesPerCell: props.operation === 'savitzky-golay' ? props.sliceCount : 1
      },
      'output.status': {
        view: props.output.status,
        entriesPerCell: props.operation === 'savitzky-golay' ? props.sliceCount : 1
      }
    });
    this.estimatedCommandNodeCount = this.chunkPlan.estimatedCommandNodeCount;
    validateRasterViews(this.id, this.getInputs(), this.getOutputs(), this.chunkPlan);
  }
  private getInputs(): RasterScalarView[] {
    return [
      ...(this.props.dateAxis && this.props.operation === 'savitzky-golay'
        ? [this.props.dateAxis.acquisitionTimes]
        : []),
      this.props.values,
      ...(this.props.mask ? [this.props.mask] : [])
    ];
  }
  private getOutputs(): RasterScalarView[] {
    return Object.values(this.props.output);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const smoothing = props.operation === 'savitzky-golay';
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    if (smoothing && props.dateAxis) validateGPURasterDateAxisGraph(id, graph, props.dateAxis);
    const useDates = smoothing && props.dateAxis !== undefined;
    const window = props.windowLength ?? 7;
    const order = props.polynomialOrder ?? 2;
    return createRasterTimeSeriesNodes<Parameters>(graph, this.chunkPlan, {
      id,
      operation: 'GPURasterTemporal',
      invocationCount: props.cellCount * (smoothing ? props.sliceCount : 1),
      workgroupSize: 64,
      bindings: [
        rasterBinding('values', props.values),
        ...(props.mask ? [rasterBinding('mask', props.mask)] : []),
        ...(useDates ? [getGPURasterDateAxisBinding(props.dateAxis!)] : []),
        ...Object.entries(props.output).map(([name, view]) =>
          rasterBinding(`output${name}`, view, true)
        )
      ],
      declarations:
        RASTER_FINITE_WGSL +
        (smoothing ? getRasterLeastSquaresWGSL(window, order + 1) : '') +
        (useDates
          ? `\n${getGPURasterDateAxisDeclarations(this.ticksPerYear, props.sliceCount)}`
          : ''),
      body: smoothing
        ? `let cell = index / ${props.sliceCount}u; let center = i32(index % ${props.sliceCount}u);
let first = ${props.edgeMode === 'truncate' ? `center - ${Math.floor(window / 2)}` : `clamp(center - ${Math.floor(window / 2)}, 0, ${props.sliceCount - window})`};
${
  useDates
    ? `let lastInRange = min(first + ${window} - 1, ${props.sliceCount - 1}); let firstInRange = max(first, 0); let axisScale = max(rasterYearsBetween(u32(lastInRange), u32(center)), rasterYearsBetween(u32(center), u32(firstInRange)));
`
    : ''
}var design: array<f32, ${window * (order + 1)}>; var response: array<f32, ${window}>; var count = 0u;
for (var position = 0u; position < ${window}u; position++) {
  let slice = first + i32(position); if (slice < 0 || slice >= ${props.sliceCount}) { continue; }
  let source = cell * ${props.sliceCount}u + u32(slice); let value = values[valuesOffset + source];
  if (!finite(value) ${props.mask ? '|| mask[maskOffset + source] == 0u' : ''}) { continue; }
  let coordinate = ${useDates ? 'rasterYearsBetween(u32(slice), u32(center)) / axisScale' : `f32(slice - center) / ${window}.0`}; var power = 1.0;
  for (var column = 0u; column < ${order + 1}u; column++) { design[column * ${window}u + position] = power; power *= coordinate; }
  response[position] = value; count++;
}
outputcounts[outputcountsOffset + index] = count; outputchosenSlice[outputchosenSliceOffset + index] = 0xffffffffu;
var value = missing(); var status = 1u;
if (count >= ${order + 1}u) { let fit = leastSquares(design, response); status = 2u; if (fit.valid) { value = fit.coefficients[0]; status = 0u; } }
outputvalues[outputvaluesOffset + index] = value; outputstatus[outputstatusOffset + index] = status;`
        : `var observations: array<f32, ${props.sliceCount}>; var dates: array<u32, ${props.sliceCount}>; var count = 0u; var total = 0.0;
for (var slice = 0u; slice < ${props.sliceCount}u; slice++) {
  let source = index * ${props.sliceCount}u + slice; let value = values[valuesOffset + source];
  if (!finite(value) ${props.mask ? '|| mask[maskOffset + source] == 0u' : ''}) { continue; }
  var position = count;
  loop { if (position == 0u) { break; } if (observations[position - 1u] <= value) { break; } observations[position] = observations[position - 1u]; dates[position] = dates[position - 1u]; position--; }
  observations[position] = value; dates[position] = slice; count++; total = total * (1.0 - 1.0 / f32(count)) + value / f32(count);
}
var result = missing(); var chosen = 0xffffffffu;
if (count > 0u) {
${props.operation === 'mean' ? 'result = total;' : props.operation === 'minimum' ? 'result = observations[0]; chosen = dates[0];' : props.operation === 'maximum' ? 'result = observations[count - 1u]; chosen = dates[count - 1u]; for (var position = 0u; position < count; position++) { if (observations[position] == result) { chosen = min(chosen, dates[position]); } }' : 'result = observations[count / 2u]; if (count % 2u == 0u) { result = result * 0.5 + observations[count / 2u - 1u] * 0.5; } else { chosen = dates[count / 2u]; }'}
}
outputvalues[outputvaluesOffset + index] = result; outputcounts[outputcountsOffset + index] = count; outputchosenSlice[outputchosenSliceOffset + index] = chosen; outputstatus[outputstatusOffset + index] = select(1u, 0u, count > 0u);`
    });
  }
}
