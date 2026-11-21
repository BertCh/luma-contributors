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
} from './raster-time-series-chunks-b';
import {
  validateGraphPorts,
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {RASTER_FINITE_WGSL} from '../remote-sensing/raster-kernel-utils';
import {FIXED_POINT_WGSL} from '../../analysis-core/kernels/fixed-point-accumulator';
import {
  getGPURasterDateAxisBinding,
  getGPURasterDateAxisDeclarations,
  resolveGPURasterIrregularDateAxis,
  validateGPURasterDateAxisGraph,
  type GPURasterIrregularDateAxis
} from './raster-irregular-date-axis';

/** Source slices: a per-cell merge scans them in order. */
export const MAXIMUM_RESAMPLE_SLICE_COUNT = 256;
/** Output grid points per cell: a per-cell merge scans them in order. */
export const MAXIMUM_RESAMPLE_GRID_COUNT = 4096;
/** Float32 elements of the per-frame parameters: grid start, grid step, maximum gap. */
export const GPU_RASTER_TEMPORAL_RESAMPLE_PARAMETER_LENGTH = 3;

/** Per-frame settings of {@link GPURasterTemporalResample}, all in centred fractional years. */
export type GPURasterTemporalResampleSettings = {
  gridStart: number;
  gridStep: number;
  /** Largest allowed span between the bracketing observations; defaults to no limit. */
  maximumGap?: number;
};

/** Packs {@link GPURasterTemporalResampleSettings}: `[gridStart, gridStep, maximumGap]`. */
export function getGPURasterTemporalResampleParameterValues(
  settings: GPURasterTemporalResampleSettings,
  target: Float32Array = new Float32Array(GPU_RASTER_TEMPORAL_RESAMPLE_PARAMETER_LENGTH)
): Float32Array {
  validateRequirement(
    'getGPURasterTemporalResampleParameterValues',
    target.length >= GPU_RASTER_TEMPORAL_RESAMPLE_PARAMETER_LENGTH,
    `target must hold at least ${GPU_RASTER_TEMPORAL_RESAMPLE_PARAMETER_LENGTH} float32 words; received ${target.length}`
  );
  validateFiniteNumber(
    'getGPURasterTemporalResampleParameterValues',
    'gridStep',
    settings.gridStep,
    0
  );
  target[0] = settings.gridStart;
  target[1] = settings.gridStep;
  target[2] = settings.maximumGap ?? 1e30;
  return target;
}

/** Properties of {@link GPURasterTemporalResample}. */
export type GPURasterTemporalResampleProps<Rows extends RasterScalarView = GraphDataView> = {
  id?: string;
  cellCount: number;
  sliceCount: number;
  /** Output grid points per cell, in `[1, 4096]`. */
  gridCount: number;
  /**
   * Optional exact final grid coordinate in centred fractional years, captured at construction.
   * Overrides the last step-derived coordinate and must not precede the prior grid point.
   */
  gridEnd?: number;
  /** Optional strictly increasing Int64 targets, one uint32x2 (low, high) row per grid point,
   * in the date axis unit and epoch. Preserves exact acquisition identity; overrides fractional
   * grid coordinates. gridEnd cannot be supplied with targetTimes. */
  targetTimes?: GraphDataView<'uint32x2'>;
  /** Cell-major `cell * sliceCount + slice`; nonfinite samples are missing. */
  values: RasterTimeSeriesView<'float32', Rows>;
  /** Per-observation QA; zero rejects the sample. */
  mask?: RasterTimeSeriesView<'uint32', Rows>;
  /** Float32, see {@link getGPURasterTemporalResampleParameterValues}. */
  parameters: GraphDataView<'float32'>;
  dateAxis: GPURasterIrregularDateAxis;
  output: {
    /** `cell * gridCount + point`; NaN where status is nonzero. */
    values: RasterTimeSeriesView<'float32'>;
    /** Per grid point: 0 interpolated, 1 outside the observed range, 2 gap larger than `maximumGap`. */
    status: RasterTimeSeriesView<'uint32'>;
  };
};

/**
 * Fills a regular time grid from irregular observations by linear interpolation between the
 * bracketing valid samples, in O(slices + grid) per cell with a merge in time. Grid dates use the
 * same centred fractional years as the date axis (`getGPURasterCentredFractionalYears`). A gap
 * filled by an earlier smoother reaches this node as a finite value, so a Whittaker or HANTS
 * output feeds a regular-time grid directly; nothing is extrapolated.
 *
 * The GPU route accepts 2–256 source slices and 1–4096 target points per cell.
 *
 * Per-frame: `parameters` and optional `targetTimes` contents. Compile-time: `cellCount`, `sliceCount`, `gridCount`, `gridEnd`, and which optional views exist.
 *
 * Cell-major views accept matching chunks of complete cell records; shared tables stay packed.
 */
export class GPURasterTemporalResample<Rows extends RasterScalarView = GraphDataView>
  implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly props: GPURasterTemporalResampleProps<Rows>;
  readonly chunkPlan: RasterTimeSeriesChunkPlan;
  readonly estimatedCommandNodeCount: number;
  private readonly ticksPerYear: number;
  private readonly gridEnd?: number;

  constructor(props: GPURasterTemporalResampleProps<Rows> & {values: Rows}) {
    this.props = props;
    const id = props.id ?? 'raster-temporal-resample';
    this.id = id;
    validateGraphPorts(id, props, {
      values: {required: true, viewType: 'either', formats: ['float32']},
      mask: {required: false, viewType: 'either', formats: ['uint32']},
      parameters: {required: true, viewType: 'data', formats: ['float32']},
      targetTimes: {required: false, viewType: 'data', formats: ['uint32x2']},
      'dateAxis.acquisitionTimes': {required: true, viewType: 'data', formats: ['uint32x2']},
      'output.values': {required: true, viewType: 'either', formats: ['float32']},
      'output.status': {required: true, viewType: 'either', formats: ['uint32']}
    });
    // Capture constructor options/maps while preserving graph resource identities.
    props = {
      ...props,
      output: {...props.output},
      ...(props.dateAxis ? {dateAxis: {...props.dateAxis}} : {})
    };
    this.props = props;
    this.gridEnd = props.gridEnd === undefined ? undefined : Math.fround(props.gridEnd);
    validateRequirement(
      id,
      this.gridEnd === undefined || Number.isFinite(this.gridEnd),
      'gridEnd must be finite float32 centred fractional years'
    );
    validateCapacity(id, 'cellCount', props.cellCount);
    validateCapacity(id, 'sliceCount', props.sliceCount, 2, MAXIMUM_RESAMPLE_SLICE_COUNT);
    validateCapacity(id, 'gridCount', props.gridCount, 1, MAXIMUM_RESAMPLE_GRID_COUNT);
    validateCapacity(id, 'cellCount times gridCount', props.cellCount * props.gridCount);
    const observationCount = props.cellCount * props.sliceCount;
    validateCapacity(id, 'cellCount times sliceCount', observationCount);
    validateRequirement(
      id,
      props.values.length >= observationCount &&
        (!props.mask || props.mask.length >= observationCount),
      'values and mask must hold cellCount times sliceCount entries'
    );
    validateRequirement(
      id,
      props.output.values.length >= props.cellCount * props.gridCount &&
        props.output.status.length >= props.cellCount * props.gridCount,
      'output.values and output.status must hold cellCount times gridCount entries'
    );
    validateRequirement(
      id,
      props.parameters.length >= GPU_RASTER_TEMPORAL_RESAMPLE_PARAMETER_LENGTH,
      `parameters must hold ${GPU_RASTER_TEMPORAL_RESAMPLE_PARAMETER_LENGTH} entries`
    );
    validateRequirement(
      id,
      !props.targetTimes || props.gridEnd === undefined,
      'gridEnd cannot be used with targetTimes; write exact targetTimes only'
    );
    validateRequirement(
      id,
      !props.targetTimes || props.targetTimes.length >= props.gridCount,
      'targetTimes must hold gridCount Int64 rows'
    );
    this.ticksPerYear = resolveGPURasterIrregularDateAxis(id, props.dateAxis, props.sliceCount);
    this.chunkPlan = planRasterTimeSeriesChunks(this.id, props.cellCount, {
      values: {view: props.values, entriesPerCell: props.sliceCount},
      mask: {view: props.mask, entriesPerCell: props.sliceCount},
      'output.values': {view: props.output.values, entriesPerCell: props.gridCount},
      'output.status': {view: props.output.status, entriesPerCell: props.gridCount}
    });
    this.estimatedCommandNodeCount = this.chunkPlan.estimatedCommandNodeCount;
    const inputs = this.getInputs();
    const outputs = this.getOutputs();
    // One interpolation dispatch per cell chunk binds all source and destination fields.
    validateRasterViews(this.id, inputs, outputs, this.chunkPlan, [{inputs, outputs}]);
  }

  private getInputs(): RasterScalarView[] {
    const {values, mask, parameters} = this.props;
    return [
      ...(this.props.dateAxis ? [this.props.dateAxis.acquisitionTimes] : []),
      values,
      parameters,
      ...(this.props.targetTimes ? [this.props.targetTimes] : []),
      ...(mask ? [mask] : [])
    ];
  }

  private getOutputs(): RasterScalarView[] {
    return [this.props.output.values, this.props.output.status];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {sliceCount: slices, gridCount, values, mask, parameters, dateAxis, output} = props;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    validateGPURasterDateAxisGraph(id, graph, dateAxis);
    return createRasterTimeSeriesNodes<Parameters>(graph, this.chunkPlan, {
      id,
      operation: 'GPURasterTemporalResample',
      invocationCount: props.cellCount,
      workgroupSize: 16,
      bindings: [
        rasterBinding('values', values),
        rasterBinding('parameters', parameters),
        ...(mask ? [rasterBinding('mask', mask)] : []),
        getGPURasterDateAxisBinding(dateAxis),
        ...(props.targetTimes
          ? [
              {
                name: 'targetTimes',
                view: props.targetTimes,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : []),
        rasterBinding('resampled', output.values, true),
        rasterBinding('resampleStatus', output.status, true)
      ],
      declarations: `${RASTER_FINITE_WGSL}
${getGPURasterDateAxisDeclarations(this.ticksPerYear, slices)}
${
  props.targetTimes
    ? ''
    : `${FIXED_POINT_WGSL}
// Integer subtraction forces each year coordinate to round BEFORE differencing. Without this
// boundary Metal can fuse a year conversion with subtraction, changing a close bracket's span.
fn rasterRoundedYearFraction(targetYear: f32, leftYear: f32, rightYear: f32) -> f32 {
  let exponentBits = max(getFixedPointExponentBits(targetYear),
    max(getFixedPointExponentBits(leftYear), getFixedPointExponentBits(rightYear)));
  let coordinateUnit = getFixedPointUnitExponent(exponentBits, 2u);
  let leftWords = quantizeFixedPoint(leftYear, coordinateUnit);
  let spanWords = addFixedPoint(quantizeFixedPoint(rightYear, coordinateUnit), negateFixedPoint(leftWords));
  let numeratorWords = addFixedPoint(quantizeFixedPoint(targetYear, coordinateUnit), negateFixedPoint(leftWords));
  let roundedSpan = getFixedPointFloat(spanWords, coordinateUnit);
  if (!(roundedSpan > 0.0)) { return 0.0; }
  return clamp(getFixedPointFloat(numeratorWords, coordinateUnit) / roundedSpan, 0.0, 1.0);
}`
}
fn isObserved(cell: u32, slice: u32) -> bool {
  let source = cell * ${slices}u + slice;
  return finite(values[valuesOffset + source]) ${mask ? '&& mask[maskOffset + source] != 0u' : ''};
}
fn advanceObserved(cell: u32, start: u32) -> u32 {
  var slice = start;
  while (slice < ${slices}u && !isObserved(cell, slice)) { slice++; }
  return slice;
}`,
      body: `let gridStart = parameters[parametersOffset];
let gridStep = parameters[parametersOffset + 1u];
let maximumGap = parameters[parametersOffset + 2u];
var left = ${slices}u;
var right = advanceObserved(index, 0u);
for (var point = 0u; point < ${gridCount}u; point++) {
  ${props.targetTimes ? 'let targetWords = vec2<u32>(targetTimes[targetTimesOffset + 2u * point], targetTimes[targetTimesOffset + 2u * point + 1u]);' : ''}
  let gridYears = ${
    this.gridEnd === undefined
      ? 'gridStart + f32(point) * gridStep'
      : `select(gridStart + f32(point) * gridStep, ${getWGSLFloatLiteral(this.gridEnd)}, point == ${gridCount - 1}u)`
  };
  while (right < ${slices}u && ${props.targetTimes ? 'timeWordsCompare(rasterAxisWords(right), targetWords) <= 0i' : 'rasterSliceYears(right) <= gridYears'}) {
    left = right;
    right = advanceObserved(index, right + 1u);
  }
  var value = missing();
  var code = 1u;
  if (left < ${slices}u && ${props.targetTimes ? 'timeWordsCompare(rasterAxisWords(left), targetWords) == 0i' : 'rasterSliceYears(left) == gridYears'}) {
    // An exact observation is usable even when the next observation is beyond maximumGap.
    value = values[valuesOffset + index * ${slices}u + left];
    code = 0u;
  } else if (left < ${slices}u && right < ${slices}u) {
    let span = rasterYearsBetween(right, left);
    if (span > maximumGap) {
      code = 2u;
    } else {
      ${
        props.targetTimes
          ? `let localSpan = timeWordsDifferenceToF32(rasterAxisWords(right), rasterAxisWords(left));
      let numerator = timeWordsDifferenceToF32(targetWords, rasterAxisWords(left));`
          : ''
      }
      ${
        props.targetTimes
          ? `var fraction = 0.0;
      if (localSpan > 0.0) { fraction = clamp(numerator / localSpan, 0.0, 1.0); }`
          : 'let fraction = rasterRoundedYearFraction(gridYears, rasterSliceYears(left), rasterSliceYears(right));'
      }
      value = mix(values[valuesOffset + index * ${slices}u + left], values[valuesOffset + index * ${slices}u + right], fraction);
      code = 0u;
    }
  }
  resampled[resampledOffset + index * ${gridCount}u + point] = value;
  resampleStatus[resampleStatusOffset + index * ${gridCount}u + point] = code;
}`
    });
  }
}
