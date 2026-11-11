// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createExtendedSeasonalBreakNodes} from './seasonal-break-extended';
import type {
  GraphDataView,
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  RASTER_FINITE_WGSL,
  rasterBinding,
  validateRasterViews,
  validateRasterGraph,
  getRasterLeastSquaresWGSL
} from './raster-kernel-utils';
import {
  getGPURasterDateAxisBinding,
  getGPURasterDateAxisDeclarations,
  resolveGPURasterIrregularDateAxis,
  validateGPURasterDateAxisGraph,
  type GPURasterIrregularDateAxis
} from '../raster-time-series/raster-irregular-date-axis';

export type GPURasterSeasonalBreakProps = {
  id?: string;
  cellCount: number;
  sliceCount: number;
  values: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  /** Seasonal period in slices on regular slots, or in fractional years with `dateAxis`. */
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
  /**
   * Acquisition dates of an irregular stack: trend and harmonic columns use centred fractional
   * years (slope per year, period in years) and interpolation uses elapsed time. Harmonic
   * seasonality only; breaks and alarms are still reported as slice indices. Results on regular
   * slots are unchanged when this is absent.
   */
  dateAxis?: GPURasterIrregularDateAxis;
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
/** Time slices: private arrays hold one value per slice and slice indices stay exact in float32. */
export const MAXIMUM_SEASONAL_BREAK_SLICE_COUNT = 256;
/** Per-thread QR working memory in floats (8 KiB): beyond this, private arrays spill and occupancy collapses. */
const SEASONAL_BREAK_WORKING_MEMORY_FLOAT_COUNT = 2048;
/** Dummy-season period: each season adds a coefficient, which grows the dense per-thread QR. */
const MAXIMUM_DUMMY_SEASON_PERIOD = 12;
/** Iterative refits run inside one invocation, so the round count is capped. */
const MAXIMUM_SEASONAL_BREAK_ITERATIONS = 256;
/** Per-cell multiply budget (2^24) for iterative fitting, which keeps one invocation under the watchdog time. */
const ITERATIVE_FIT_WORK_BUDGET = 16777216;

/** Offline single-harmonic trend plus at most one level shift. QR fits each eligible original
 * acquisition boundary. BIC charges two extra parameters (jump and selected date); earliest
 * boundary wins exact ties. Gaps are omitted without compressing time. This is a bounded
 * single-break model. Explicit extended options add dummy seasons, bounded cyclic decomposition
 * and fixed-history monitoring; no multi-break BFAST inference is claimed.
 */
export class GPURasterSeasonalBreak implements GPUCommandNodeProducer {
  readonly id: string;
  private readonly ticksPerYear: number;
  constructor(readonly props: GPURasterSeasonalBreakProps) {
    this.id = props.id ?? 'raster-seasonal-break';
    this.ticksPerYear = props.dateAxis
      ? resolveGPURasterIrregularDateAxis(this.id, props.dateAxis, props.sliceCount)
      : 0;
    validateRequirement(
      this.id,
      !(props.dateAxis && props.seasonality === 'dummy'),
      'dateAxis requires harmonic seasonality; dummy seasons are defined on regular slots'
    );
    const historyLength = props.monitoring?.historyLength ?? props.sliceCount;
    const coefficientCount = props.seasonality === 'dummy' ? props.period + 2 : 5;
    // Keep the largest private QR workspace within the shared 8 KiB kernel-audit envelope.
    validateRequirement(
      this.id,
      historyLength * coefficientCount + coefficientCount ** 2 + 2 * coefficientCount + 8 <=
        SEASONAL_BREAK_WORKING_MEMORY_FLOAT_COUNT,
      `per-thread QR working memory (history ${historyLength}, ${coefficientCount} coefficients) must fit ${SEASONAL_BREAK_WORKING_MEMORY_FLOAT_COUNT} floats; reduce sliceCount, monitoring.historyLength or period`
    );
    validateRequirement(
      this.id,
      historyLength * coefficientCount +
        historyLength +
        Math.ceil(historyLength / 32) +
        coefficientCount +
        64 <=
        SEASONAL_BREAK_WORKING_MEMORY_FLOAT_COUNT,
      `per-thread design and mask working memory (history ${historyLength}, ${coefficientCount} coefficients) must fit ${SEASONAL_BREAK_WORKING_MEMORY_FLOAT_COUNT} floats`
    );
    if (props.seasonality === 'dummy') {
      validateRequirement(
        this.id,
        Number.isInteger(props.period) && props.period <= MAXIMUM_DUMMY_SEASON_PERIOD,
        `period must be an integer of at most ${MAXIMUM_DUMMY_SEASON_PERIOD} for dummy seasonality`
      );
    }
    if (props.monitoring) {
      validateRequirement(
        this.id,
        Number.isInteger(historyLength) && historyLength >= 6 && historyLength < props.sliceCount,
        'monitoring.historyLength must be an integer in [6, sliceCount)'
      );
      validateRequirement(
        this.id,
        Number.isFinite(Math.fround(props.monitoring.threshold)) &&
          Math.fround(props.monitoring.threshold) > 0,
        'monitoring.threshold must be a positive finite float32 value'
      );
      validateCapacity(
        this.id,
        'monitoring.minimumConsecutive',
        props.monitoring.minimumConsecutive,
        1,
        props.sliceCount - historyLength
      );
    }
    validateCapacity(
      this.id,
      'maximumIterations',
      props.maximumIterations ?? 128,
      1,
      MAXIMUM_SEASONAL_BREAK_ITERATIONS
    );
    validateRequirement(
      this.id,
      Number.isFinite(Math.fround(props.tolerance ?? 1e-5)) &&
        Math.fround(props.tolerance ?? 1e-5) > 0,
      'tolerance must be a positive finite float32 value'
    );
    validateRequirement(
      this.id,
      props.fitMethod !== 'iterative' ||
        historyLength *
          coefficientCount *
          (historyLength - 2 * props.minimumSegmentLength + 2) *
          (props.maximumIterations ?? 128) <=
          ITERATIVE_FIT_WORK_BUDGET,
      `iterative fitting work (history times coefficients times candidate breaks times maximumIterations) must be at most ${ITERATIVE_FIT_WORK_BUDGET} per cell`
    );
    validateRequirement(
      this.id,
      !(props.fitMethod === 'iterative' || props.monitoring) || !!props.output.diagnostics,
      'output.diagnostics is required for iterative fitting and monitoring'
    );
    validateRequirement(
      this.id,
      !props.output.diagnostics || props.output.diagnostics.length >= props.cellCount * 6,
      'output.diagnostics must hold six entries per cell'
    );
    validateCapacity(this.id, 'cellCount', props.cellCount);
    validateCapacity(
      this.id,
      'sliceCount',
      props.sliceCount,
      1,
      MAXIMUM_SEASONAL_BREAK_SLICE_COUNT
    );
    validateCapacity(
      this.id,
      'cellCount times sliceCount times three',
      props.cellCount * props.sliceCount * 3
    );
    validateRequirement(
      this.id,
      Number.isFinite(Math.fround(props.period)) && props.period > (props.dateAxis ? 0 : 2),
      props.dateAxis
        ? 'period must be a positive finite number of fractional years with dateAxis'
        : 'period must be a finite float32 value above 2'
    );
    validateCapacity(
      this.id,
      'minimumSegmentLength',
      props.minimumSegmentLength,
      3,
      Math.floor(historyLength / 2)
    );
    validateRequirement(
      this.id,
      Number.isFinite(Math.fround(props.minimumBICGain ?? 0)) && (props.minimumBICGain ?? 0) >= 0,
      'minimumBICGain must be a nonnegative finite float32 value'
    );
    validateRequirement(
      this.id,
      props.values.length >= props.cellCount * props.sliceCount &&
        (!props.mask || props.mask.length >= props.cellCount * props.sliceCount),
      'values and mask must hold cellCount times sliceCount entries'
    );
    validateRequirement(
      this.id,
      props.output.coefficients.length >= props.cellCount * coefficientCount &&
        props.output.components.length >= props.cellCount * props.sliceCount * 3,
      'output.coefficients must hold coefficientCount entries per cell and output.components three entries per observation'
    );
    validateRequirement(
      this.id,
      [props.output.breakIndex, props.output.improvement, props.output.status].every(
        view => view.length >= props.cellCount
      ),
      'output.breakIndex, output.improvement and output.status must hold cellCount entries'
    );
    validateRasterViews(
      this.id,
      [props.values, ...(props.mask ? [props.mask] : [])],
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
      [props.values, ...(props.mask ? [props.mask] : [])],
      Object.values(props.output)
    );
    if (props.dateAxis) validateGPURasterDateAxisGraph(id, graph, props.dateAxis);
    const axisDeclarations = props.dateAxis
      ? `\n${getGPURasterDateAxisDeclarations(this.ticksPerYear, slices)}`
      : '';
    if (
      props.seasonality ||
      props.fitMethod ||
      props.missingPolicy ||
      props.monitoring ||
      props.output.diagnostics
    ) {
      return createExtendedSeasonalBreakNodes(graph, id, props, axisDeclarations);
    }
    const sliceTime = props.dateAxis ? 'rasterSliceYears(slice)' : `f32(slice) / ${slices}.0`;
    const sliceAngle = props.dateAxis
      ? `6.283185307179586 * rasterSliceYears(slice) / ${props.period}`
      : `6.283185307179586 * f32(slice) / ${props.period}`;
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterSeasonalBreak',
        invocationCount: props.cellCount,
        workgroupSize: 32,
        bindings: [
          rasterBinding('values', props.values),
          ...(props.mask ? [rasterBinding('mask', props.mask)] : []),
          ...(props.dateAxis ? [getGPURasterDateAxisBinding(props.dateAxis)] : []),
          ...Object.entries(props.output).map(([name, view]) => rasterBinding(name, view, true))
        ],
        declarations:
          RASTER_FINITE_WGSL + getRasterLeastSquaresWGSL(slices, 5, true) + axisDeclarations,
        body: `var design: array<f32, ${slices * 5}>; var response: array<f32, ${slices}>; var valid: array<bool, ${slices}>; var count = 0u; var energy = 0.0;
for (var slice = 0u; slice < ${slices}u; slice++) {
  let source = index * ${slices}u + slice; let value = values[valuesOffset + source];
  valid[slice] = finite(value) ${props.mask ? '&& mask[maskOffset + source] != 0u' : ''};
  if (!valid[slice]) { continue; }
  count++; response[slice] = value; energy += value * value;
  let angle = ${sliceAngle};
  let row = array<f32, 4>(1.0, ${sliceTime}, cos(angle), sin(angle));
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
  if (code == 0u) { trend = chosen[0] + chosen[1] * ${sliceTime} + select(0.0, chosen[4], slice >= boundary); let angle = ${sliceAngle}; seasonal = chosen[2] * cos(angle) + chosen[3] * sin(angle); if (valid[slice]) { error = response[slice] - trend - seasonal; } }
  components[componentsOffset + destination] = trend; components[componentsOffset + destination + 1u] = seasonal; components[componentsOffset + destination + 2u] = error;
}
breakIndex[breakIndexOffset + index] = boundary; improvement[improvementOffset + index] = select(missing(), gain, code == 0u); status[statusOffset + index] = code;`
      })
    ];
  }
}
