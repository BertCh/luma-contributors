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
  MAXIMUM_RASTER_STREAM_CHUNK_SLICE_COUNT,
  MAXIMUM_RASTER_STREAM_SLOT_COUNT,
  DEFAULT_MAXIMUM_RASTER_STREAM_SLICE_COUNT,
  getGPURasterFixedPointUnitExponent
} from './gpu-raster-streaming-accumulator';
import {FIXED_POINT_WGSL} from '../../analysis-core/kernels/fixed-point-accumulator';
import {getRasterFixedPointStateWGSL} from './raster-fixed-point-state-wgsl';

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
export function getGPURasterStandardizedIndexCalibrationParameterValues(
  settings: {
    reset: boolean;
    activeSliceCount?: number;
    location?: number;
    maximumValueMagnitude: number;
    maximumStreamSliceCount?: number;
  },
  chunkSliceCount: number,
  target: Float32Array = new Float32Array(GPU_RASTER_STANDARDIZED_CALIBRATION_PARAMETER_LENGTH)
): Float32Array {
  const slices = settings.maximumStreamSliceCount ?? DEFAULT_MAXIMUM_RASTER_STREAM_SLICE_COUNT;
  target[0] = settings.reset ? 1 : 0;
  target[1] = settings.activeSliceCount ?? chunkSliceCount;
  target[2] = settings.location ?? 0;
  target[3] = getGPURasterFixedPointUnitExponent(settings.maximumValueMagnitude, slices);
  target[4] = getGPURasterFixedPointUnitExponent(MAXIMUM_LOG_MAGNITUDE, slices);
  target[5] = getGPURasterFixedPointUnitExponent(4 * settings.maximumValueMagnitude ** 2, slices);
  target[6] = settings.maximumValueMagnitude;
  return target;
}

/** Properties of {@link GPURasterStandardizedIndexCalibration}. */
export type GPURasterStandardizedIndexCalibrationProps = {
  id?: string;
  cellCount: number;
  chunkSliceCount: number;
  /** Calendar slots per cell, usually 12 months. Defaults to 1. */
  slotCount?: number;
  /** `'gamma'` for SPI, `'fisk'` (log-logistic) for SPEI. */
  distribution: GPURasterStandardizedIndexDistribution;
  /** Treat exact zeros separately, as xclim does for precipitation (true for SPI). Defaults to false. */
  zeroInflated?: boolean;
  /** Cell-major chunk; nonfinite samples are missing. */
  values: GraphDataView<'float32'>;
  /** Per-observation QA; zero rejects the sample. */
  mask?: GraphDataView<'uint32'>;
  /** Slot of each chunk slice, required when `slotCount` exceeds 1. */
  slotIndices?: GraphDataView<'uint32'>;
  /** Float32, see {@link getGPURasterStandardizedIndexCalibrationParameterValues}. */
  parameters: GraphDataView<'float32'>;
  state: {
    /** Per slot `[nonMissing, zeroCount, positiveCount]`, uint32; `positiveCount` holds the samples above the location entering the fit. */
    counts: GraphDataView<'uint32'>;
    /** Per slot 96-bit fixed-point sums of `x`, `ln x` and the squared deviations (nine `uint32` words per slot). */
    moments: GraphDataView<'uint32'>;
  };
  /** Per slot `[shape, scale, zeroProbability]`, float32; NaN shape marks an unusable slot. */
  fit: GraphDataView<'float32'>;
};

/**
 * Streams calibration slices into per-slot distribution statistics and writes the fit, so a
 * calibration period longer than one chunk (30 years of months exceeds the in-memory slice limit)
 * needs no resident stack. The fit is the Approximate Parameter method of `xclim` (Thom's gamma
 * estimator, the method-of-moments log-logistic): closed form, no iterative maximum likelihood. Sums
 * are exact 96-bit fixed-point integers, so means of long series keep float32 precision; see the
 * documentation for the float32 limits of this fit.
 *
 * Per-frame: `parameters`. Compile-time: `cellCount`, `chunkSliceCount`, `slotCount`,
 * `distribution`, `zeroInflated`, and which optional views exist. Run it once per calibration
 * chunk, with `reset` on the first; the fit is rewritten after every chunk.
 */
export class GPURasterStandardizedIndexCalibration implements GPUCommandNodeProducer {
  readonly id: string;
  readonly slotCount: number;
  constructor(readonly props: GPURasterStandardizedIndexCalibrationProps) {
    const id = props.id ?? 'raster-standardized-index-calibration';
    this.id = id;
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
    validateRequirement(
      id,
      props.distribution === 'gamma' || props.distribution === 'fisk',
      "distribution must be 'gamma' or 'fisk'"
    );
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
    validateRasterViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterScalarView[] {
    const {values, mask, slotIndices, parameters} = this.props;
    return [values, parameters, ...(mask ? [mask] : []), ...(slotIndices ? [slotIndices] : [])];
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
    const zeroInflated = props.zeroInflated ?? false;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    const fitBody =
      props.distribution === 'gamma'
        ? `let logarithmGap = log(mean) - logarithmMean;
    let shape = (1.0 + sqrt(1.0 + 4.0 * logarithmGap / 3.0)) / (4.0 * logarithmGap);
    let scale = mean / shape;`
        : `let variance = max(moment, 0.0) / f32(positiveCount);
    let scale = 2.0 * mean * mean * mean / (variance + 2.0 * mean * mean);
    let shape = 3.141592653589793 * mean / (1.7320508075688772 * sqrt(variance));`;
    return [
      createWGSLKernelNode<Parameters>(graph, {
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
        declarations: `${RASTER_FINITE_WGSL}\n${FIXED_POINT_WGSL}\n${getRasterFixedPointStateWGSL('moments')}`,
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
  if (positive > maximumMagnitude) { continue; }
  counts[countsOffset + 3u * entry] += 1u;
  ${zeroInflated ? 'if (value == 0.0) { counts[countsOffset + 3u * entry + 1u] += 1u; continue; }' : ''}
  if (!(positive > 0.0)) { continue; }
  let priorCount = counts[countsOffset + 3u * entry + 2u];
  counts[countsOffset + 3u * entry + 2u] = priorCount + 1u;
  let priorSum = loadMoments(3u * entry);
  let priorMean = getFixedPointFloat(priorSum, sumUnit) / f32(max(priorCount, 1u));
  let sum = addFixedPoint(priorSum, quantizeFixedPoint(positive, sumUnit));
  storeMoments(3u * entry, sum);
  storeMoments(3u * entry + 1u, addFixedPoint(loadMoments(3u * entry + 1u), quantizeFixedPoint(log(positive), logarithmUnit)));
  let updatedMean = getFixedPointFloat(sum, sumUnit) / f32(priorCount + 1u);
  let deviation = select((positive - priorMean) * (positive - updatedMean), 0.0, priorCount == 0u);
  storeMoments(3u * entry + 2u, addFixedPoint(loadMoments(3u * entry + 2u), quantizeFixedPoint(deviation, momentUnit)));
}
for (var slot = 0u; slot < ${slotCount}u; slot++) {
  let entry = index * ${slotCount}u + slot;
  let nonMissing = counts[countsOffset + 3u * entry];
  let zeroCount = counts[countsOffset + 3u * entry + 1u];
  let positiveCount = counts[countsOffset + 3u * entry + 2u];
  var fittedShape = missing();
  var fittedScale = missing();
  if (positiveCount >= 2u) {
    let mean = getFixedPointFloat(loadMoments(3u * entry), sumUnit) / f32(positiveCount);
    let logarithmMean = getFixedPointFloat(loadMoments(3u * entry + 1u), logarithmUnit) / f32(positiveCount);
    let moment = getFixedPointFloat(loadMoments(3u * entry + 2u), momentUnit);
    ${fitBody}
    if (shape > 0.0 && scale > 0.0 && finite(shape) && finite(scale)) { fittedShape = shape; fittedScale = scale; }
  }
  fit[fitOffset + 3u * entry] = fittedShape;
  fit[fitOffset + 3u * entry + 1u] = fittedScale;
  fit[fitOffset + 3u * entry + 2u] = select(0.0, f32(zeroCount) / f32(nonMissing), nonMissing > 0u);
}`
      })
    ];
  }
}
