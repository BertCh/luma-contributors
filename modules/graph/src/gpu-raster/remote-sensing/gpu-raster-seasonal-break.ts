// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {createExtendedSeasonalBreakNodes} from './seasonal-break-extended';
import type {
  GraphDataView,
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer
} from '@luma-contributors/graph/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  RASTER_FINITE_WGSL,
  rasterBinding,
  validateRasterCount,
  validateRasterViews,
  validateRasterGraph,
  getRasterLeastSquaresWGSL
} from './raster-kernel-utils';

export type GPURasterSeasonalBreakProps = {
  id?: string;
  cellCount: number;
  sliceCount: number;
  values: GraphDataView<'float32'>;
  validity?: GraphDataView<'uint32'>;
  /** Regular-time seasonal period measured in slices. */
  period: number;
  /** Minimum elapsed slices and valid observations on each side; at least 3. */
  minimumSegmentLength: number;
  minimumBICGain?: number;
  /** Dummy season zero is the reference category; integer periods 3–12. */
  seasonality?: 'harmonic' | 'dummy';
  /** Cyclic least-squares updates alternate trend, seasonal terms and jump until convergence. */
  fitMethod?: 'joint' | 'iterative';
  maximumIterations?: number;
  tolerance?: number;
  /** Fixed regular slots; interpolate only interior training gaps, never future monitoring data. */
  missingPolicy?: 'omit' | 'reject' | 'interpolate';
  /** Fixed-history monitoring; append observations by explicitly re-encoding the full buffer. */
  monitoring?: {historyLength: number; threshold: number; minimumConsecutive: number};
  output: {
    /** Harmonic: five coefficients. Dummy: period+2 (intercept,slope,phases 1..period-1,jump). */
    coefficients: GraphDataView<'float32'>;
    /** Per observation interleaved trend (including jump), seasonality, residual. */
    components: GraphDataView<'float32'>;
    /** Original acquisition index; 0xffffffff means no selected break or invalid model. */
    breakIndex: GraphDataView<'uint32'>;
    /** BIC(no break) - BIC(selected break), or zero when no break selected. */
    improvement: GraphDataView<'float32'>;
    /** 0=fit, 1=insufficient support, 2=rank/nonfinite failure, 4=rejected gap, 8=iteration limit. */
    status: GraphDataView<'uint32'>;
    /** Six floats/cell: iterations, converged, training support, residual scale, alarm index (-1 absent), maximum absolute monitoring score. Required for iterative/monitoring modes. */
    diagnostics?: GraphDataView<'float32'>;
  };
};
/** Offline single-harmonic trend plus at most one level shift. QR fits each eligible original
 * acquisition boundary. BIC charges two extra parameters (jump and selected date); earliest
 * boundary wins exact ties. Gaps are omitted without compressing time. This is a bounded
 * single-break model. Explicit extended options add dummy seasons, bounded cyclic decomposition
 * and fixed-history monitoring; no multi-break BFAST inference is claimed.
 */
export class GPURasterSeasonalBreak implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPURasterSeasonalBreakProps) {
    this.id = props.id ?? 'raster-seasonal-break';
    const historyLength = props.monitoring?.historyLength ?? props.sliceCount;
    const coefficientCount = props.seasonality === 'dummy' ? props.period + 2 : 5;
    // Keep the largest private QR workspace within the shared 8 KiB kernel-audit envelope.
    assert(
      historyLength * coefficientCount + coefficientCount ** 2 + 2 * coefficientCount + 8 <= 2048
    );
    assert(
      historyLength * coefficientCount +
        historyLength +
        Math.ceil(historyLength / 32) +
        coefficientCount +
        64 <=
        2048
    );
    if (props.seasonality === 'dummy') assert(Number.isInteger(props.period) && props.period <= 12);
    if (props.monitoring) {
      assert(
        Number.isInteger(historyLength) && historyLength >= 6 && historyLength < props.sliceCount
      );
      assert(
        Number.isFinite(Math.fround(props.monitoring.threshold)) &&
          Math.fround(props.monitoring.threshold) > 0
      );
      validateRasterCount(props.monitoring.minimumConsecutive, props.sliceCount - historyLength);
    }
    assert(
      Number.isInteger(props.maximumIterations ?? 128) &&
        (props.maximumIterations ?? 128) >= 1 &&
        (props.maximumIterations ?? 128) <= 256
    );
    assert(
      Number.isFinite(Math.fround(props.tolerance ?? 1e-5)) &&
        Math.fround(props.tolerance ?? 1e-5) > 0
    );
    assert(
      props.fitMethod !== 'iterative' ||
        historyLength *
          coefficientCount *
          (historyLength - 2 * props.minimumSegmentLength + 2) *
          (props.maximumIterations ?? 128) <=
          16777216
    );
    assert(!(props.fitMethod === 'iterative' || props.monitoring) || props.output.diagnostics);
    assert(!props.output.diagnostics || props.output.diagnostics.length >= props.cellCount * 6);
    validateRasterCount(props.cellCount);
    validateRasterCount(props.sliceCount, 256);
    validateRasterCount(props.cellCount * props.sliceCount * 3);
    assert(Number.isFinite(Math.fround(props.period)) && props.period > 2);
    validateRasterCount(props.minimumSegmentLength, Math.floor(historyLength / 2));
    assert(props.minimumSegmentLength >= 3);
    assert(
      Number.isFinite(Math.fround(props.minimumBICGain ?? 0)) && (props.minimumBICGain ?? 0) >= 0
    );
    assert(
      props.values.length >= props.cellCount * props.sliceCount &&
        (!props.validity || props.validity.length >= props.cellCount * props.sliceCount)
    );
    assert(
      props.output.coefficients.length >= props.cellCount * coefficientCount &&
        props.output.components.length >= props.cellCount * props.sliceCount * 3
    );
    assert(
      [props.output.breakIndex, props.output.improvement, props.output.status].every(
        view => view.length >= props.cellCount
      )
    );
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
    const slices = props.sliceCount;
    validateRasterGraph(
      id,
      graph,
      [props.values, ...(props.validity ? [props.validity] : [])],
      Object.values(props.output)
    );
    if (
      props.seasonality ||
      props.fitMethod ||
      props.missingPolicy ||
      props.monitoring ||
      props.output.diagnostics
    ) {
      return createExtendedSeasonalBreakNodes(graph, id, props);
    }
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterSeasonalBreak',
        invocationCount: props.cellCount,
        workgroupSize: 32,
        bindings: [
          rasterBinding('values', props.values),
          ...(props.validity ? [rasterBinding('validity', props.validity)] : []),
          ...Object.entries(props.output).map(([name, view]) => rasterBinding(name, view, true))
        ],
        declarations: RASTER_FINITE_WGSL + getRasterLeastSquaresWGSL(slices, 5, true),
        body: `var design: array<f32, ${slices * 5}>; var response: array<f32, ${slices}>; var valid: array<bool, ${slices}>; var count = 0u; var energy = 0.0;
for (var slice = 0u; slice < ${slices}u; slice++) {
  let source = index * ${slices}u + slice; let value = values[valuesOffset + source];
  valid[slice] = finite(value) ${props.validity ? '&& validity[validityOffset + source] != 0u' : ''};
  if (!valid[slice]) { continue; }
  count++; response[slice] = value; energy += value * value;
  let angle = 6.283185307179586 * f32(slice) / ${props.period};
  let row = array<f32, 4>(1.0, f32(slice) / ${slices}.0, cos(angle), sin(angle));
  for (var column = 0u; column < 4u; column++) { design[column * ${slices}u + slice] = row[column]; }
}
var chosen: array<f32, 5>; var boundary = 0xffffffffu; var gain = 0.0; var code = 1u;
if (!finite(energy)) { code = 2u; }
else if (count >= ${Math.max(5, props.minimumSegmentLength * 2)}u) {
  let baseline = leastSquares(design, response, 4u); code = 2u;
  if (baseline.valid) {
    code = 0u; for (var column = 0u; column < 4u; column++) { chosen[column] = baseline.coefficients[column]; }
    let residualFloor = max(1e-20, energy * 1e-12);
    let baselineScore = f32(count) * log(max(baseline.residual, residualFloor) / f32(count)) + 4.0 * log(f32(count));
    for (var split = ${props.minimumSegmentLength}u; split <= ${slices - props.minimumSegmentLength}u; split++) {
      var before = 0u; var after = 0u;
      for (var slice = 0u; slice < ${slices}u; slice++) { design[4u * ${slices}u + slice] = 0.0; if (valid[slice]) { if (slice < split) { before++; } else { after++; design[4u * ${slices}u + slice] = 1.0; } } }
      if (before < ${props.minimumSegmentLength}u || after < ${props.minimumSegmentLength}u) { continue; }
      let fit = leastSquares(design, response, 5u); if (!fit.valid) { continue; }
      let candidateGain = baselineScore - (f32(count) * log(max(fit.residual, residualFloor) / f32(count)) + 6.0 * log(f32(count)));
      if (candidateGain > max(gain, ${props.minimumBICGain ?? 0})) { gain = candidateGain; boundary = split; chosen = fit.coefficients; }
    }
  }
}
for (var column = 0u; column < 5u; column++) { coefficients[coefficientsOffset + index * 5u + column] = select(missing(), chosen[column], code == 0u); }
for (var slice = 0u; slice < ${slices}u; slice++) {
  let destination = (index * ${slices}u + slice) * 3u;
  var trend = missing(); var seasonal = missing(); var error = missing();
  if (code == 0u) { trend = chosen[0] + chosen[1] * f32(slice) / ${slices}.0 + select(0.0, chosen[4], slice >= boundary); let angle = 6.283185307179586 * f32(slice) / ${props.period}; seasonal = chosen[2] * cos(angle) + chosen[3] * sin(angle); if (valid[slice]) { error = response[slice] - trend - seasonal; } }
  components[componentsOffset + destination] = trend; components[componentsOffset + destination + 1u] = seasonal; components[componentsOffset + destination + 2u] = error;
}
breakIndex[breakIndexOffset + index] = boundary; improvement[improvementOffset + index] = select(missing(), gain, code == 0u); status[statusOffset + index] = code;`
      })
    ];
  }
}
