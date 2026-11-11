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

/** Slices per cell: the curve times live in a private array and every season scans the curve. */
export const MAXIMUM_PHENOLOGY_SLICE_COUNT = 256;
/** Seasons per cell: the season loop is unrolled and each season has private metric slots. */
export const MAXIMUM_PHENOLOGY_SEASON_COUNT = 2;
/** Float32 elements of the per-frame parameters. */
export const GPU_RASTER_PHENOLOGY_METRICS_PARAMETER_LENGTH = 4;

/** Fields of one season in `output.metrics`, at `(cell * maximumSeasonCount + season) * 10 + field`. */
export const GPU_RASTER_PHENOLOGY_METRIC = Object.freeze({
  /** Time of the rising crossing of the start level. */
  startTime: 0,
  /** Time of the falling crossing of the end level. */
  endTime: 1,
  /** Time of the highest sample of the season. */
  peakTime: 2,
  /** `endTime - startTime`. */
  length: 3,
  /** Curve value at `peakTime`. */
  peakValue: 4,
  /** Lowest value of the whole curve. */
  baseValue: 5,
  /** `peakValue - baseValue`. */
  amplitude: 6,
  /** Rise per time unit from the start crossing to the peak. */
  leftSlope: 7,
  /** Change per time unit from the peak to the end crossing; negative for a falling limb. */
  rightSlope: 8,
  /** Area between the curve and `baseValue` from start to end (trapezoids on the piecewise-linear curve). */
  integral: 9
} as const);
/** Number of float32 fields per season in `output.metrics`. */
export const GPU_RASTER_PHENOLOGY_METRIC_COUNT = 10;

/** Bits of `output.flags`: the crossing was not found, so the season is cut at a boundary. */
export const GPU_RASTER_PHENOLOGY_FLAG = Object.freeze({
  /** The curve is above the start level at the first sample; `startTime` is the first sample. */
  startAtSeriesStart: 1,
  /** The curve is above the end level at the last sample; `endTime` is the last sample. */
  endAtSeriesEnd: 2,
  /** The start search ran into the other season; `startTime` is the first sample after it. */
  startAtOtherSeason: 4,
  /** The end search ran into the other season; `endTime` is the last sample before it. */
  endAtOtherSeason: 8
} as const);

/** Per-frame settings of {@link GPURasterPhenologyMetrics}. */
export type GPURasterPhenologyMetricsSettings = {
  /** Fraction of the curve amplitude above the base where a season starts, in `(0, 1)`; MR-VPP uses `0.25`. */
  startFraction: number;
  /** Fraction of the curve amplitude above the base where a season ends, in `(0, 1)`; MR-VPP uses `0.15`. */
  endFraction: number;
  /** Cells whose amplitude (maximum minus minimum) is below this get status 2 and no season. */
  minimumAmplitude: number;
  /**
   * A second season needs a peak at least this fraction of the amplitude above the base. It must be
   * at least both start and end fractions, so the shoulders of the first season are not a second one.
   * Defaults to the larger of the two.
   */
  secondSeasonMinimumFraction?: number;
};

/**
 * Packs {@link GPURasterPhenologyMetricsSettings}: `[startFraction, endFraction, minimumAmplitude,
 * secondSeasonMinimumFraction]`.
 */
export function getGPURasterPhenologyMetricsParameterValues(
  settings: GPURasterPhenologyMetricsSettings,
  target: Float32Array = new Float32Array(GPU_RASTER_PHENOLOGY_METRICS_PARAMETER_LENGTH)
): Float32Array {
  const name = 'getGPURasterPhenologyMetricsParameterValues';
  validateFiniteNumber(name, 'startFraction', settings.startFraction, Number.MIN_VALUE, 1);
  validateFiniteNumber(name, 'endFraction', settings.endFraction, Number.MIN_VALUE, 1);
  validateFiniteNumber(name, 'minimumAmplitude', settings.minimumAmplitude, 0);
  const larger = Math.max(settings.startFraction, settings.endFraction);
  const second = settings.secondSeasonMinimumFraction ?? larger;
  validateFiniteNumber(name, 'secondSeasonMinimumFraction', second, larger, 1);
  target.set([settings.startFraction, settings.endFraction, settings.minimumAmplitude, second]);
  return target;
}

/** Properties of {@link GPURasterPhenologyMetrics}. */
export type GPURasterPhenologyMetricsProps = {
  id?: string;
  cellCount: number;
  /** Slices in `[2, 256]`. */
  sliceCount: number;
  /** Seasons found per cell, 1 or 2. */
  maximumSeasonCount: number;
  /**
   * The smoothed curve `cell * sliceCount + slice`, for example `output.smoothed` of
   * {@link GPURasterWhittakerSmoother}. A non-finite sample gives the cell status 1.
   */
  smoothed: GraphDataView<'float32'>;
  /** Float32, see {@link getGPURasterPhenologyMetricsParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /**
   * Acquisition dates. Times in the output are then centred fractional years of the axis; without
   * it they are slice indices.
   */
  dateAxis?: GPURasterIrregularDateAxis;
  output: {
    /** `cellCount * maximumSeasonCount * 10` fields, see {@link GPU_RASTER_PHENOLOGY_METRIC}; NaN for an absent season. */
    metrics: GraphDataView<'float32'>;
    /** Per season, bits of {@link GPU_RASTER_PHENOLOGY_FLAG}. */
    flags: GraphDataView<'uint32'>;
    /** Seasons found per cell, ordered by peak time in the metrics. */
    seasonCount: GraphDataView<'uint32'>;
    /** Per cell: 0 fit, 1 non-finite curve, 2 amplitude below the minimum. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Phenology metrics on a smoothed curve, without any nonlinear fit. Per cell: base is the curve
 * minimum, amplitude the maximum minus the minimum. Season 1 is the highest sample (the earliest on
 * ties); it starts where the curve last drops below the start level before the peak and ends where
 * it first drops below the end level after it, both by linear interpolation. With two seasons, the
 * second is the highest sample outside the first season's span, accepted when its peak reaches
 * `secondSeasonMinimumFraction` of the amplitude; its crossings are searched only between the first
 * season and the series end, and report a flag where they meet the boundary. Seasons are ordered by
 * peak time. This follows the threshold definitions of the Copernicus MR-VPP product (start 25% and
 * end 15% of the amplitude) on the curve; it is not TIMESAT, and it does not fit double logistic or
 * asymmetric Gaussian functions, which are ill-conditioned in float32.
 *
 * Per-frame: `parameters` and the contents of the views. Compile-time: `cellCount`, `sliceCount`,
 * `maximumSeasonCount` and whether `dateAxis` exists.
 */
export class GPURasterPhenologyMetrics implements GPUCommandNodeProducer {
  readonly id: string;
  private readonly ticksPerYear: number;

  constructor(readonly props: GPURasterPhenologyMetricsProps) {
    const id = props.id ?? 'raster-phenology-metrics';
    this.id = id;
    validateCapacity(id, 'cellCount', props.cellCount);
    validateCapacity(id, 'sliceCount', props.sliceCount, 2, MAXIMUM_PHENOLOGY_SLICE_COUNT);
    validateCapacity(
      id,
      'maximumSeasonCount',
      props.maximumSeasonCount,
      1,
      MAXIMUM_PHENOLOGY_SEASON_COUNT
    );
    const observationCount = props.cellCount * props.sliceCount;
    validateCapacity(id, 'cellCount times sliceCount', observationCount);
    const seasonSlots = props.cellCount * props.maximumSeasonCount;
    validateRequirement(
      id,
      props.smoothed.length >= observationCount,
      'smoothed must hold cellCount times sliceCount entries'
    );
    validateRequirement(
      id,
      props.output.metrics.length >= seasonSlots * GPU_RASTER_PHENOLOGY_METRIC_COUNT &&
        props.output.flags.length >= seasonSlots,
      'output.metrics must hold 10 entries and output.flags one per cell and season'
    );
    validateRequirement(
      id,
      props.output.seasonCount.length >= props.cellCount &&
        props.output.status.length >= props.cellCount,
      'output.seasonCount and output.status must hold one entry per cell'
    );
    validateRequirement(
      id,
      props.parameters.length >= GPU_RASTER_PHENOLOGY_METRICS_PARAMETER_LENGTH,
      `parameters must hold ${GPU_RASTER_PHENOLOGY_METRICS_PARAMETER_LENGTH} entries`
    );
    this.ticksPerYear = props.dateAxis
      ? resolveGPURasterIrregularDateAxis(id, props.dateAxis, props.sliceCount)
      : 0;
    validateRasterViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterScalarView[] {
    return [this.props.smoothed, this.props.parameters];
  }

  private getOutputs(): RasterScalarView[] {
    const {metrics, flags, seasonCount, status} = this.props.output;
    return [metrics, flags, seasonCount, status];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {sliceCount: slices, maximumSeasonCount: seasons, smoothed, parameters, output} = props;
    const dateAxis = props.dateAxis;
    const fields = GPU_RASTER_PHENOLOGY_METRIC_COUNT;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    if (dateAxis) validateGPURasterDateAxisGraph(id, graph, dateAxis);
    // Time of a slice. With a date axis it is read from a function-scope table filled from the
    // axis. Without one it is the slice index, computed in place, so no table is needed. (A
    // local integer table written only by a loop with a constant trip count can read back as
    // zeros on the test GPU, contributor contract section 11; float tables, such as the one
    // filled from the axis, read right at every size tried.)
    const sliceTime = (slice: string) => (dateAxis ? `times[${slice}]` : `f32(${slice})`);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterPhenologyMetrics',
        invocationCount: props.cellCount,
        workgroupSize: 64,
        bindings: [
          rasterBinding('smoothed', smoothed),
          rasterBinding('parameters', parameters),
          ...(dateAxis ? [getGPURasterDateAxisBinding(dateAxis)] : []),
          rasterBinding('metrics', output.metrics, true),
          rasterBinding('flags', output.flags, true),
          rasterBinding('seasonCount', output.seasonCount, true),
          rasterBinding('status', output.status, true)
        ],
        declarations: `${RASTER_FINITE_WGSL}
${dateAxis ? getGPURasterDateAxisDeclarations(this.ticksPerYear, slices) : ''}
fn curve(cell: u32, slice: u32) -> f32 { return smoothed[smoothedOffset + cell * ${slices}u + slice]; }`,
        body: `let startFraction = parameters[parametersOffset];
let endFraction = parameters[parametersOffset + 1u];
let minimumAmplitude = parameters[parametersOffset + 2u];
let secondFraction = parameters[parametersOffset + 3u];
${dateAxis ? `var times: array<f32, ${slices}>;` : ''}
var lowest = curve(index, 0u);
var highest = lowest;
var allFinite = true;
for (var slice = 0u; slice < ${slices}u; slice++) {
  ${dateAxis ? 'times[slice] = rasterSliceYears(slice);' : ''}
  let value = curve(index, slice);
  if (!finite(value)) { allFinite = false; }
  lowest = min(lowest, value);
  highest = max(highest, value);
}
let amplitude = highest - lowest;
var code = 0u;
if (!allFinite) { code = 1u; }
else if (!(amplitude > 0.0) || amplitude < minimumAmplitude) { code = 2u; }
var found = 0u;
var season: array<array<f32, ${fields}>, ${seasons}>;
var seasonFlags: array<u32, ${seasons}>;
if (code == 0u) {
  let startLevel = lowest + startFraction * amplitude;
  let endLevel = lowest + endFraction * amplitude;
  var excludedLow = 1u;
  var excludedHigh = 0u;
  var hasExcluded = false;
  for (var candidateSeason = 0u; candidateSeason < ${seasons}u; candidateSeason++) {
    var peak = 0xffffffffu;
    var peakValue = 0.0;
    for (var slice = 0u; slice < ${slices}u; slice++) {
      if (hasExcluded && slice >= excludedLow && slice <= excludedHigh) { continue; }
      let value = curve(index, slice);
      if (peak == 0xffffffffu || value > peakValue) { peak = slice; peakValue = value; }
    }
    if (peak == 0xffffffffu) { break; }
    if (candidateSeason > 0u && peakValue < lowest + secondFraction * amplitude) { break; }
    var lower = 0u;
    var upper = ${slices - 1}u;
    if (hasExcluded) {
      if (peak < excludedLow) { upper = excludedLow - 1u; } else { lower = excludedHigh + 1u; }
    }
    var boundaryFlags = 0u;
    var startTime = ${sliceTime('lower')};
    var startValue = curve(index, lower);
    var leftIndex = lower;
    var crossing = peak;
    var crossed = false;
    while (crossing > lower) {
      crossing--;
      if (curve(index, crossing) < startLevel) { crossed = true; break; }
    }
    if (crossed) {
      let below = curve(index, crossing);
      let above = curve(index, crossing + 1u);
      startValue = startLevel;
      startTime = ${sliceTime('crossing')} + (startLevel - below) / (above - below) * (${sliceTime('crossing + 1u')} - ${sliceTime('crossing')});
      leftIndex = crossing + 1u;
    } else {
      boundaryFlags |= select(4u, 1u, lower == 0u);
    }
    var endTime = ${sliceTime('upper')};
    var endValue = curve(index, upper);
    var rightIndex = upper;
    crossing = peak;
    crossed = false;
    while (crossing < upper) {
      crossing++;
      if (curve(index, crossing) < endLevel) { crossed = true; break; }
    }
    if (crossed) {
      let above = curve(index, crossing - 1u);
      let below = curve(index, crossing);
      endValue = endLevel;
      endTime = ${sliceTime('crossing - 1u')} + (above - endLevel) / (above - below) * (${sliceTime('crossing')} - ${sliceTime('crossing - 1u')});
      rightIndex = crossing - 1u;
    } else {
      boundaryFlags |= select(8u, 2u, upper == ${slices - 1}u);
    }
    var integral = 0.0;
    var previousTime = startTime;
    var previousHeight = startValue - lowest;
    for (var slice = leftIndex; slice <= rightIndex; slice++) {
      let height = curve(index, slice) - lowest;
      integral += 0.5 * (height + previousHeight) * (${sliceTime('slice')} - previousTime);
      previousTime = ${sliceTime('slice')};
      previousHeight = height;
    }
    integral += 0.5 * ((endValue - lowest) + previousHeight) * (endTime - previousTime);
    let peakTime = ${sliceTime('peak')};
    var entry: array<f32, ${fields}>;
    entry[0] = startTime;
    entry[1] = endTime;
    entry[2] = peakTime;
    entry[3] = endTime - startTime;
    entry[4] = peakValue;
    entry[5] = lowest;
    entry[6] = peakValue - lowest;
    entry[7] = select(missing(), (peakValue - startValue) / (peakTime - startTime), peakTime > startTime);
    entry[8] = select(missing(), (endValue - peakValue) / (endTime - peakTime), endTime > peakTime);
    entry[9] = integral;
    season[candidateSeason] = entry;
    seasonFlags[candidateSeason] = boundaryFlags;
    hasExcluded = true;
    excludedLow = leftIndex;
    excludedHigh = rightIndex;
    found++;
  }
  ${
    seasons === 2
      ? `  if (found == 2u && season[1][2] < season[0][2]) {
    let swapped = season[0];
    season[0] = season[1];
    season[1] = swapped;
    let swappedFlags = seasonFlags[0];
    seasonFlags[0] = seasonFlags[1];
    seasonFlags[1] = swappedFlags;
  }`
      : ''
  }
}
status[statusOffset + index] = code;
seasonCount[seasonCountOffset + index] = found;
for (var slot = 0u; slot < ${seasons}u; slot++) {
  flags[flagsOffset + index * ${seasons}u + slot] = select(0u, seasonFlags[slot], slot < found);
  for (var field = 0u; field < ${fields}u; field++) {
    metrics[metricsOffset + (index * ${seasons}u + slot) * ${fields}u + field] = select(missing(), season[slot][field], slot < found);
  }
}`
      })
    ];
  }
}
