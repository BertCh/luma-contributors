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
  validateFiniteNumber,
  validateCapacity,
  validateGraphPorts,
  validateOption,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {RASTER_FINITE_WGSL} from '../remote-sensing/raster-kernel-utils';
import {
  MAXIMUM_RASTER_STREAM_CHUNK_SLICE_COUNT,
  MAXIMUM_RASTER_STREAM_SLOT_COUNT,
  DEFAULT_MAXIMUM_RASTER_STREAM_SLICE_COUNT,
  getGPURasterFixedPointUnitExponent
} from './gpu-raster-streaming-accumulator';
import {
  validateRasterStreamFrame,
  validateRasterStreamParameterTarget,
  validateRasterStreamEncodingPlan,
  validateRasterStreamHeadroom,
  getRasterStreamPositiveFloat,
  validateRasterStreamPlanFrame,
  type GPURasterStreamFrameSettings
} from './raster-stream-parameters';
import {
  FIXED_POINT_WGSL,
  FIXED_POINT_MOMENTS_WGSL
} from '../../analysis-core/kernels/fixed-point-accumulator';
import {
  getRasterFixedPointStateWGSL,
  RASTER_CALIBRATION_MOMENTS_WGSL
} from './raster-fixed-point-state-wgsl';
import {RASTER_CLIMATE_ZERO_ATOM_WGSL} from './raster-climate-distribution-wgsl';

/** Distribution fitted per calendar slot by the Approximate Parameter closed form. */
export type GPURasterStandardizedIndexDistribution = 'gamma' | 'fisk';

/** Float32 elements of the per-frame parameters; see the packer. */
export const GPU_RASTER_STANDARDIZED_CALIBRATION_PARAMETER_LENGTH = 7;
/** Largest `|ln x|` of a positive float32, which bounds the logarithm sum's fixed-point terms. */
const MAXIMUM_LOG_MAGNITUDE = 100;

/**
 * Packs `[reset, activeSliceCount, location, sumUnit, logUnit, momentUnit, maximumValueMagnitude]`.
 * `maximumValueMagnitude` bounds `value - location`; samples above it are treated as missing.
 */
export type GPURasterStandardizedIndexCalibrationSettings = {
  reset: boolean;
  activeSliceCount?: number;
  location?: number;
  maximumValueMagnitude: number;
  maximumStreamSliceCount?: number;
};

/** Immutable model settings and units; replace only when resetting all retained state. */
export type GPURasterStandardizedIndexCalibrationEncodingPlan = Readonly<{
  kind: 'raster-standardized-index-calibration';
  chunkSliceCount: number;
  maximumStreamSliceCount: number;
  location: number;
  maximumValueMagnitude: number;
  sumUnitExponent: number;
  logarithmUnitExponent: number;
  momentUnitExponent: number;
}>;

export function createGPURasterStandardizedIndexCalibrationEncodingPlan(
  settings: Omit<GPURasterStandardizedIndexCalibrationSettings, 'reset' | 'activeSliceCount'>,
  chunkSliceCount: number
): GPURasterStandardizedIndexCalibrationEncodingPlan {
  const id = 'createGPURasterStandardizedIndexCalibrationEncodingPlan';
  validateRasterStreamFrame(id, chunkSliceCount, 0, MAXIMUM_RASTER_STREAM_CHUNK_SLICE_COUNT);
  validateFiniteNumber(id, 'location', settings.location ?? 0);
  const location = Math.fround(settings.location ?? 0);
  const maximumValueMagnitude = getRasterStreamPositiveFloat(
    id,
    'maximumValueMagnitude',
    settings.maximumValueMagnitude
  );
  const maximumStreamSliceCount =
    settings.maximumStreamSliceCount ?? DEFAULT_MAXIMUM_RASTER_STREAM_SLICE_COUNT;
  validateRasterStreamHeadroom(id, maximumStreamSliceCount);
  return Object.freeze({
    kind: 'raster-standardized-index-calibration',
    chunkSliceCount,
    maximumStreamSliceCount,
    location,
    maximumValueMagnitude,
    sumUnitExponent: getGPURasterFixedPointUnitExponent(
      maximumValueMagnitude,
      maximumStreamSliceCount
    ),
    logarithmUnitExponent: getGPURasterFixedPointUnitExponent(
      MAXIMUM_LOG_MAGNITUDE,
      maximumStreamSliceCount
    ),
    momentUnitExponent: getGPURasterFixedPointUnitExponent(
      4 * maximumValueMagnitude ** 2,
      maximumStreamSliceCount
    )
  });
}

/** Pack legacy settings with fixed encoding until reset, or reuse an immutable encoding plan. */
export function getGPURasterStandardizedIndexCalibrationParameterValues(
  settings: GPURasterStandardizedIndexCalibrationSettings,
  chunkSliceCount: number,
  target?: Float32Array
): Float32Array;
export function getGPURasterStandardizedIndexCalibrationParameterValues(
  plan: GPURasterStandardizedIndexCalibrationEncodingPlan,
  frame: GPURasterStreamFrameSettings,
  target?: Float32Array
): Float32Array;
export function getGPURasterStandardizedIndexCalibrationParameterValues(
  settings:
    | GPURasterStandardizedIndexCalibrationSettings
    | GPURasterStandardizedIndexCalibrationEncodingPlan,
  chunkOrFrame: number | GPURasterStreamFrameSettings,
  target: Float32Array = new Float32Array(GPU_RASTER_STANDARDIZED_CALIBRATION_PARAMETER_LENGTH)
): Float32Array {
  const id = 'getGPURasterStandardizedIndexCalibrationParameterValues';
  validateRasterStreamParameterTarget(
    id,
    target,
    GPU_RASTER_STANDARDIZED_CALIBRATION_PARAMETER_LENGTH
  );
  const legacy = typeof chunkOrFrame === 'number';
  const plan = legacy
    ? createGPURasterStandardizedIndexCalibrationEncodingPlan(
        settings as GPURasterStandardizedIndexCalibrationSettings,
        chunkOrFrame
      )
    : (settings as GPURasterStandardizedIndexCalibrationEncodingPlan);
  if (!legacy)
    validateRasterStreamEncodingPlan(id, plan, 'raster-standardized-index-calibration', () =>
      createGPURasterStandardizedIndexCalibrationEncodingPlan(plan, plan.chunkSliceCount)
    );
  const frame =
    typeof chunkOrFrame === 'number'
      ? (settings as GPURasterStandardizedIndexCalibrationSettings)
      : chunkOrFrame;
  validateRequirement(
    id,
    typeof frame.reset === 'boolean',
    `reset must be boolean; received ${String(frame.reset)}`
  );
  if (typeof chunkOrFrame !== 'number') validateRasterStreamPlanFrame(id, plan, frame);
  const activeSliceCount = frame.activeSliceCount ?? plan.chunkSliceCount;
  validateRasterStreamFrame(
    id,
    plan.chunkSliceCount,
    activeSliceCount,
    MAXIMUM_RASTER_STREAM_CHUNK_SLICE_COUNT,
    plan.maximumStreamSliceCount,
    frame.reset
  );
  target.set([
    frame.reset ? 1 : 0,
    activeSliceCount,
    plan.location,
    plan.sumUnitExponent,
    plan.logarithmUnitExponent,
    plan.momentUnitExponent,
    plan.maximumValueMagnitude
  ]);
  return target;
}

/** Resolve the explicit atom at original value zero, shared by calibration and evaluation. */
export function resolveGPURasterStandardizedIndexZeroAtom(
  id: string,
  settings: {zeroAtom?: boolean}
): boolean {
  validateRequirement(id, !('zeroInflated' in settings), 'zeroInflated was removed; use zeroAtom');
  validateRequirement(
    id,
    settings.zeroAtom === undefined || typeof settings.zeroAtom === 'boolean',
    `zeroAtom must be boolean; received ${String(settings.zeroAtom)}`
  );
  return settings.zeroAtom ?? false;
}

/** Properties of {@link GPURasterStandardizedIndexCalibration}. */
export type GPURasterStandardizedIndexCalibrationProps<
  Rows extends RasterScalarView = GraphDataView
> = {
  id?: string;
  cellCount: number;
  chunkSliceCount: number;
  /** Calendar slots per cell, usually 12 months. Defaults to 1. */
  slotCount?: number;
  /** `'gamma'` for SPI, `'fisk'` (log-logistic) for SPEI. */
  distribution: GPURasterStandardizedIndexDistribution;
  /** Model an atom at the original value zero, independent of location. Defaults to false. */
  zeroAtom?: boolean;
  /** Cell-major chunk; nonfinite samples are missing. */
  values: RasterTimeSeriesView<'float32', Rows>;
  /** Per-observation QA; zero rejects the sample. */
  mask?: RasterTimeSeriesView<'uint32', Rows>;
  /** Slot of each chunk slice, required when `slotCount` exceeds 1. */
  slotIndices?: GraphDataView<'uint32'>;
  /** Float32, see {@link getGPURasterStandardizedIndexCalibrationParameterValues}. */
  parameters: GraphDataView<'float32'>;
  state: {
    /** Per slot `[nonMissing, zeroCount, positiveCount]`, uint32; `positiveCount` holds the samples above the location entering the fit. */
    counts: RasterTimeSeriesView<'uint32', Rows>;
    /** Per slot 96-bit fixed-point sums of `x`, `ln x` and the squared deviations (nine `uint32` words per slot). */
    moments: RasterTimeSeriesView<'uint32', Rows>;
  };
  /** Per slot `[shape, scale, zeroProbability]`, float32; NaN shape marks an unusable slot. */
  fit: RasterTimeSeriesView<'float32', Rows>;
};

/**
 * Streams calibration slices into per-slot distribution statistics and writes the fit, so a
 * calibration period longer than one chunk (30 years of months exceeds the in-memory slice limit)
 * needs no resident stack. The fit is the Approximate Parameter method of `xclim` (Thom's gamma
 * estimator, the method-of-moments log-logistic): closed form, no iterative maximum likelihood. Sums
 * are exact 96-bit fixed-point integers, so means of long series keep float32 precision; see the
 * documentation for the float32 limits of this fit. Squared deviations center in fixed-point units
 * before float conversion and scale before multiplication; words 6–8 retain their squared units.
 * Shifted observations retain value - location low bits in the sum and centered update. Fisk fits
 * use a dimensionless deviation/mean ratio so a finite result does not require a finite f32 cube.
 *
 * At most 256 slices per calibration submission; declared stream headroom defaults to 2^24 slices.
 *
 * Per-frame: `parameters`. Compile-time: `cellCount`, `chunkSliceCount`, `slotCount`,
 * `distribution`, `zeroAtom`, and which optional views exist. Run it once per calibration
 * chunk, with `reset` on the first; the fit is rewritten after every chunk.
 *
 * Cell-major views accept matching chunks of complete cell records; shared tables stay packed.
 */
export class GPURasterStandardizedIndexCalibration<Rows extends RasterScalarView = GraphDataView>
  implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly props: GPURasterStandardizedIndexCalibrationProps<Rows>;
  readonly chunkPlan: RasterTimeSeriesChunkPlan;
  readonly estimatedCommandNodeCount: number;
  readonly slotCount: number;
  constructor(props: GPURasterStandardizedIndexCalibrationProps<Rows> & {values: Rows}) {
    const zeroAtom = resolveGPURasterStandardizedIndexZeroAtom(
      props.id ?? 'raster-standardized-index-calibration',
      props
    );
    this.props = {...props, zeroAtom, state: {...props.state}};
    const id = props.id ?? 'raster-standardized-index-calibration';
    this.id = id;
    validateGraphPorts(id, props, {
      values: {required: true, viewType: 'either', formats: ['float32']},
      mask: {required: false, viewType: 'either', formats: ['uint32']},
      slotIndices: {required: false, viewType: 'data', formats: ['uint32']},
      parameters: {required: true, viewType: 'data', formats: ['float32']},
      'state.counts': {required: true, viewType: 'either', formats: ['uint32']},
      'state.moments': {required: true, viewType: 'either', formats: ['uint32']},
      fit: {required: true, viewType: 'either', formats: ['float32']}
    });
    const slotCount = props.slotCount ?? 1;
    this.slotCount = slotCount;
    validateCapacity(id, 'cellCount', props.cellCount);
    validateCapacity(
      id,
      'chunkSliceCount',
      props.chunkSliceCount,
      1,
      MAXIMUM_RASTER_STREAM_CHUNK_SLICE_COUNT
    );
    validateCapacity(id, 'slotCount', slotCount, 1, MAXIMUM_RASTER_STREAM_SLOT_COUNT);
    validateOption(id, 'distribution', props.distribution, ['gamma', 'fisk']);
    const observationCount = props.cellCount * props.chunkSliceCount;
    const slotEntries = props.cellCount * slotCount;
    validateCapacity(id, 'cellCount times chunkSliceCount', observationCount);
    validateCapacity(id, 'cellCount times slotCount', slotEntries);
    validateRequirement(
      id,
      props.values.length >= observationCount &&
        (!props.mask || props.mask.length >= observationCount),
      'values and mask must hold cellCount times chunkSliceCount entries'
    );
    validateRequirement(
      id,
      slotCount === 1 || (!!props.slotIndices && props.slotIndices.length >= props.chunkSliceCount),
      'slotIndices must hold one entry per chunk slice when slotCount exceeds 1'
    );
    validateRequirement(
      id,
      props.parameters.length >= GPU_RASTER_STANDARDIZED_CALIBRATION_PARAMETER_LENGTH,
      `parameters must hold ${GPU_RASTER_STANDARDIZED_CALIBRATION_PARAMETER_LENGTH} entries`
    );
    validateRequirement(
      id,
      props.state.counts.length >= slotEntries * 3 &&
        props.state.moments.length >= slotEntries * 9 &&
        props.fit.length >= slotEntries * 3,
      'state.counts needs three, state.moments nine and fit three entries per slot'
    );
    this.chunkPlan = planRasterTimeSeriesChunks(this.id, props.cellCount, {
      values: {view: props.values, entriesPerCell: props.chunkSliceCount},
      mask: {view: props.mask, entriesPerCell: props.chunkSliceCount},
      'state.counts': {view: props.state.counts, entriesPerCell: (props.slotCount ?? 1) * 3},
      'state.moments': {view: props.state.moments, entriesPerCell: (props.slotCount ?? 1) * 9},
      fit: {view: props.fit, entriesPerCell: (props.slotCount ?? 1) * 3}
    });
    this.estimatedCommandNodeCount = this.chunkPlan.estimatedCommandNodeCount;
    validateRasterViews(this.id, this.getInputs(), this.getOutputs(), this.chunkPlan);
  }

  private getInputs(): RasterScalarView[] {
    const {values, mask, slotIndices, parameters} = this.props;
    return [
      values,
      parameters,
      ...(mask ? [mask] : []),
      ...((this.props.slotCount ?? 1) > 1 && slotIndices ? [slotIndices] : [])
    ];
  }

  private getOutputs(): RasterScalarView[] {
    const {state, fit} = this.props;
    return [state.counts, state.moments, fit];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, slotCount} = this;
    const {chunkSliceCount: chunk, values, mask, slotIndices, parameters, state, fit} = props;
    const zeroAtom = props.zeroAtom ?? false;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    const fitBody =
      props.distribution === 'gamma'
        ? `let mean = getFixedPointFloat(loadMoments(3u * entry), sumUnit) / f32(positiveCount);
    let logarithmMean = getFixedPointFloat(loadMoments(3u * entry + 1u), logarithmUnit) / f32(positiveCount);
    let logarithmGap = log(mean) - logarithmMean;
    let shape = (1.0 + sqrt(1.0 + 4.0 * logarithmGap / 3.0)) / (4.0 * logarithmGap);
    let scale = mean / shape;`
        : `let halfMomentExponent = momentUnit / 2;
    let scaledMoment = getFixedPointFloat(loadMoments(3u * entry + 2u), momentUnit - 2 * halfMomentExponent);
    let scaledMean = getFixedPointFloat(loadMoments(3u * entry), sumUnit - halfMomentExponent) / f32(positiveCount);
    let relativeDeviation = sqrt(max(scaledMoment, 0.0) / f32(positiveCount)) / scaledMean;
    let mean = ldexp(scaledMean, halfMomentExponent);
    let scale = mean / (1.0 + 0.5 * relativeDeviation * relativeDeviation);
    let shape = 3.141592653589793 / (1.7320508075688772 * relativeDeviation);`;
    return createRasterTimeSeriesNodes<Parameters>(graph, this.chunkPlan, {
      id,
      operation: 'GPURasterStandardizedIndexCalibration',
      invocationCount: props.cellCount,
      workgroupSize: 64,
      bindings: [
        rasterBinding('values', values),
        rasterBinding('parameters', parameters),
        ...(mask ? [rasterBinding('mask', mask)] : []),
        ...(slotCount > 1 && slotIndices ? [rasterBinding('slotIndices', slotIndices)] : []),
        rasterBinding('counts', state.counts, true),
        rasterBinding('moments', state.moments, true),
        rasterBinding('fit', fit, true)
      ],
      declarations: `${RASTER_FINITE_WGSL}\n${zeroAtom ? RASTER_CLIMATE_ZERO_ATOM_WGSL : ''}\n${FIXED_POINT_WGSL}\n${FIXED_POINT_MOMENTS_WGSL}\n${RASTER_CALIBRATION_MOMENTS_WGSL}\n${getRasterFixedPointStateWGSL('moments')}`,
      body: `if (parameters[parametersOffset] != 0.0) {
  for (var slot = 0u; slot < ${slotCount}u; slot++) {
    let entry = index * ${slotCount}u + slot;
    counts[countsOffset + 3u * entry] = 0u; counts[countsOffset + 3u * entry + 1u] = 0u; counts[countsOffset + 3u * entry + 2u] = 0u;
    for (var triple = 0u; triple < 3u; triple++) { storeMoments(3u * entry + triple, vec3<u32>(0u)); }
  }
}
let activeSliceCount = u32(clamp(parameters[parametersOffset + 1u], 0.0, ${chunk}.0));
let location = parameters[parametersOffset + 2u];
let sumUnit = i32(parameters[parametersOffset + 3u]);
let logarithmUnit = i32(parameters[parametersOffset + 4u]);
let momentUnit = i32(parameters[parametersOffset + 5u]);
let maximumMagnitude = parameters[parametersOffset + 6u];
for (var slice = 0u; slice < activeSliceCount; slice++) {
  let source = index * ${chunk}u + slice;
  let value = values[valuesOffset + source];
  if (!finite(value) ${mask ? '|| mask[maskOffset + source] == 0u' : ''}) { continue; }
  let slot = ${slotCount > 1 ? 'slotIndices[slotIndicesOffset + slice]' : '0u'};
  if (slot >= ${slotCount}u) { continue; }
  let entry = index * ${slotCount}u + slot;
  let positive = value - location;
  ${zeroAtom ? 'if (isRasterClimateOriginalZero(value)) { counts[countsOffset + 3u * entry] += 1u; counts[countsOffset + 3u * entry + 1u] += 1u; continue; }' : ''}
  if (positive > maximumMagnitude) { continue; }
  counts[countsOffset + 3u * entry] += 1u;
  if (!(positive > 0.0)) { continue; }
  let priorCount = counts[countsOffset + 3u * entry + 2u];
  counts[countsOffset + 3u * entry + 2u] = priorCount + 1u;
  let priorSum = loadMoments(3u * entry);
  let positiveWords = getRasterCalibrationObservation(value, location, sumUnit);
  let sum = addFixedPoint(priorSum, positiveWords);
  storeMoments(3u * entry, sum);
  storeMoments(3u * entry + 1u, addFixedPoint(loadMoments(3u * entry + 1u), quantizeFixedPoint(log(positive), logarithmUnit)));
  let momentIncrement = getRasterCalibrationMomentIncrement(priorSum, priorCount, positiveWords, sumUnit, momentUnit);
  storeMoments(3u * entry + 2u, addFixedPoint(loadMoments(3u * entry + 2u), momentIncrement));
}
for (var slot = 0u; slot < ${slotCount}u; slot++) {
  let entry = index * ${slotCount}u + slot;
  let nonMissing = counts[countsOffset + 3u * entry];
  let zeroCount = counts[countsOffset + 3u * entry + 1u];
  let positiveCount = counts[countsOffset + 3u * entry + 2u];
  var fittedShape = missing();
  var fittedScale = missing();
  if (positiveCount >= 2u) {
    ${fitBody}
    if (shape > 0.0 && scale > 0.0 && finite(shape) && finite(scale)) { fittedShape = shape; fittedScale = scale; }
  }
  fit[fitOffset + 3u * entry] = fittedShape;
  fit[fitOffset + 3u * entry + 1u] = fittedScale;
  fit[fitOffset + 3u * entry + 2u] = select(0.0, f32(zeroCount) / f32(nonMissing), nonMissing > 0u);
}`
    });
  }
}
