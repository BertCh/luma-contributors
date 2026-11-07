// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {
  GraphDataView,
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  RASTER_FINITE_WGSL,
  rasterBinding,
  validateRasterCount,
  validateRasterViews,
  validateRasterGraph,
  getRasterLeastSquaresWGSL
} from './raster-kernel-utils';

export type GPURasterTemporalProps = {
  id?: string;
  cellCount: number;
  sliceCount: number;
  /** Cell-major regular-time samples; nonfinite samples are missing. */
  values: GraphDataView<'float32'>;
  /** Per observation QA: zero rejects that acquisition for that cell. */
  validity?: GraphDataView<'uint32'>;
  operation: 'mean' | 'minimum' | 'maximum' | 'median' | 'savitzky-golay';
  /** SG only: odd 3–15 complete window, clipped or shifted at domain edges. */
  windowLength?: number;
  polynomialOrder?: number;
  edgeMode?: 'interp' | 'truncate';
  output: {
    values: GraphDataView<'float32'>;
    counts: GraphDataView<'uint32'>;
    chosenSlice: GraphDataView<'uint32'>;
    status: GraphDataView<'uint32'>;
  };
};
/** Composites write one row per cell; SG writes cellCount*sliceCount rows. SG refits a
 * polynomial with QR over valid observations, rather than renormalizing signed coefficients.
 * Status 1=insufficient support, 2=rank failure. Dates use original indices; extrema take the
 * earliest tie and odd medians take the middle stable-sorted observation. Mean, even median
 * and SG have no unique source and write 0xffffffff. */
export class GPURasterTemporal implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPURasterTemporalProps) {
    this.id = props.id ?? 'raster-temporal';
    validateRasterCount(props.cellCount);
    validateRasterCount(props.sliceCount, 256);
    validateRasterCount(props.cellCount * props.sliceCount);
    assert(['mean', 'minimum', 'maximum', 'median', 'savitzky-golay'].includes(props.operation));
    const smoothing = props.operation === 'savitzky-golay';
    if (smoothing) {
      const window = props.windowLength ?? 7;
      const order = props.polynomialOrder ?? 2;
      assert(
        Number.isInteger(window) &&
          window >= 3 &&
          window <= 15 &&
          window % 2 === 1 &&
          window <= props.sliceCount
      );
      assert(Number.isInteger(order) && order >= 0 && order <= 3 && order < window);
      assert(['interp', 'truncate'].includes(props.edgeMode ?? 'interp'));
    }
    const inputCount = props.cellCount * props.sliceCount;
    const outputCount = smoothing ? inputCount : props.cellCount;
    assert(
      props.values.length >= inputCount && (!props.validity || props.validity.length >= inputCount)
    );
    assert(Object.values(props.output).every(view => view.length >= outputCount));
    validateRasterViews(
      this.id,
      [props.values, ...(props.validity ? [props.validity] : [])],
      Object.values(props.output)
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const smoothing = props.operation === 'savitzky-golay';
    validateRasterGraph(
      id,
      graph,
      [props.values, ...(props.validity ? [props.validity] : [])],
      Object.values(props.output)
    );
    const window = props.windowLength ?? 7;
    const order = props.polynomialOrder ?? 2;
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterTemporal',
        invocationCount: props.cellCount * (smoothing ? props.sliceCount : 1),
        workgroupSize: 64,
        bindings: [
          rasterBinding('values', props.values),
          ...(props.validity ? [rasterBinding('validity', props.validity)] : []),
          ...Object.entries(props.output).map(([name, view]) =>
            rasterBinding(`output${name}`, view, true)
          )
        ],
        declarations:
          RASTER_FINITE_WGSL + (smoothing ? getRasterLeastSquaresWGSL(window, order + 1) : ''),
        body: smoothing
          ? `let cell = index / ${props.sliceCount}u; let center = i32(index % ${props.sliceCount}u);
let first = ${props.edgeMode === 'truncate' ? `center - ${Math.floor(window / 2)}` : `clamp(center - ${Math.floor(window / 2)}, 0, ${props.sliceCount - window})`};
var design: array<f32, ${window * (order + 1)}>; var response: array<f32, ${window}>; var count = 0u;
for (var position = 0u; position < ${window}u; position++) {
  let slice = first + i32(position); if (slice < 0 || slice >= ${props.sliceCount}) { continue; }
  let source = cell * ${props.sliceCount}u + u32(slice); let value = values[valuesOffset + source];
  if (!finite(value) ${props.validity ? '|| validity[validityOffset + source] == 0u' : ''}) { continue; }
  let coordinate = f32(slice - center) / ${window}.0; var power = 1.0;
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
  if (!finite(value) ${props.validity ? '|| validity[validityOffset + source] == 0u' : ''}) { continue; }
  var position = count;
  loop { if (position == 0u) { break; } if (observations[position - 1u] <= value) { break; } observations[position] = observations[position - 1u]; dates[position] = dates[position - 1u]; position--; }
  observations[position] = value; dates[position] = slice; count++; total = total * (1.0 - 1.0 / f32(count)) + value / f32(count);
}
var result = missing(); var chosen = 0xffffffffu;
if (count > 0u) {
${props.operation === 'mean' ? 'result = total;' : props.operation === 'minimum' ? 'result = observations[0]; chosen = dates[0];' : props.operation === 'maximum' ? 'result = observations[count - 1u]; chosen = dates[count - 1u]; for (var position = 0u; position < count; position++) { if (observations[position] == result) { chosen = min(chosen, dates[position]); } }' : 'result = observations[count / 2u]; if (count % 2u == 0u) { result = result * 0.5 + observations[count / 2u - 1u] * 0.5; } else { chosen = dates[count / 2u]; }'}
}
outputvalues[outputvaluesOffset + index] = result; outputcounts[outputcountsOffset + index] = count; outputchosenSlice[outputchosenSliceOffset + index] = chosen; outputstatus[outputstatusOffset + index] = select(1u, 0u, count > 0u);`
      })
    ];
  }
}
