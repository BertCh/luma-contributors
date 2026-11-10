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
  FIXED_POINT_WGSL,
  getFixedPointHeadroomBits
} from '../../analysis-core/kernels/fixed-point-accumulator';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getRasterFixedPointStateWGSL} from './raster-fixed-point-state-wgsl';
import {
  RASTER_FINITE_WGSL,
  rasterBinding,
  validateRasterGraph,
  validateRasterViews,
  type RasterScalarView
} from '../remote-sensing/raster-kernel-utils';

/** Slices per dispatch: one invocation per cell loops over the chunk, so this is a per-thread bound. */
export const MAXIMUM_RASTER_STREAM_CHUNK_SLICE_COUNT = 256;
/** Calendar slots per cell (366 days of year): each slot adds a loop step at reset and per chunk. */
export const MAXIMUM_RASTER_STREAM_SLOT_COUNT = 366;
/** Float32 elements of the per-frame parameters: reset, active slice count, sum and moment unit exponents, maximum magnitude. */
export const GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH = 5;
/** Default stream length bound for the fixed-point headroom (2^24 slices, over 45,000 years of days). */
export const DEFAULT_MAXIMUM_RASTER_STREAM_SLICE_COUNT = 2 ** 24;

/** Per-frame settings of a streaming raster contributor. */
export type GPURasterStreamingAccumulatorSettings = {
  /** True on the first chunk of a stream, which clears the state first. */
  reset: boolean;
  /** Slices of this chunk that hold data; defaults to the whole chunk. */
  activeSliceCount?: number;
  /**
   * Largest absolute sample the stream can hold. It sets the fixed-point unit of the sums: declare
   * it tightly, because a sample above it is treated as missing, and the unit is about
   * `maximumValueMagnitude * 2^-70` at the default stream length.
   */
  maximumValueMagnitude: number;
  /** Longest stream in slices, for the fixed-point headroom. Defaults to 2^24. */
  maximumStreamSliceCount?: number;
};

/**
 * Packs the per-frame settings of {@link GPURasterStreamingAccumulator}.
 * Layout (float32): `[reset, activeSliceCount, sumUnitExponent, momentUnitExponent, maximumValueMagnitude]`.
 */
export function getGPURasterStreamingAccumulatorParameterValues(
  settings: GPURasterStreamingAccumulatorSettings,
  chunkSliceCount: number,
  target: Float32Array = new Float32Array(GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH)
): Float32Array {
  validateFiniteNumber(
    'getGPURasterStreamingAccumulatorParameterValues',
    'maximumValueMagnitude',
    settings.maximumValueMagnitude,
    Number.MIN_VALUE
  );
  const slices = settings.maximumStreamSliceCount ?? DEFAULT_MAXIMUM_RASTER_STREAM_SLICE_COUNT;
  target[0] = settings.reset ? 1 : 0;
  target[1] = settings.activeSliceCount ?? chunkSliceCount;
  target[2] = getGPURasterFixedPointUnitExponent(settings.maximumValueMagnitude, slices);
  target[3] = getGPURasterFixedPointUnitExponent(4 * settings.maximumValueMagnitude ** 2, slices);
  target[4] = settings.maximumValueMagnitude;
  return target;
}

/**
 * Unit exponent of a streamed fixed-point sum whose terms stay below `maximumMagnitude`, with
 * headroom for `maximumTermCount` terms. Pass the value with the sum's readback to decode it.
 */
export function getGPURasterFixedPointUnitExponent(
  maximumMagnitude: number,
  maximumTermCount: number
): number {
  const biasedExponent = Math.max(
    (new Uint32Array(new Float32Array([maximumMagnitude]).buffer)[0] >>> 23) & 0xff,
    1
  );
  return biasedExponent - 127 + 1 + getFixedPointHeadroomBits(maximumTermCount) - 95;
}

/** Decodes the 96-bit two's complement fixed-point sums of a readback, three `uint32` words each. */
export function getGPURasterFixedPointValues(
  words: ArrayLike<number>,
  unitExponent: number
): Float64Array {
  const values = new Float64Array(Math.floor(words.length / 3));
  for (let index = 0; index < values.length; index++) {
    const raw =
      BigInt(words[index * 3] >>> 0) |
      (BigInt(words[index * 3 + 1] >>> 0) << 32n) |
      (BigInt(words[index * 3 + 2] >>> 0) << 64n);
    const signed = BigInt.asIntN(96, raw);
    values[index] = Number(signed) * 2 ** unitExponent;
  }
  return values;
}

/** Caller-owned running state, indexed `cell * slotCount + slot`. */
export type GPURasterStreamingState = {
  /** Valid observations per slot. */
  counts: GraphDataView<'uint32'>;
  /** 96-bit fixed-point sum of the samples, three `uint32` words per slot; decode with {@link getGPURasterFixedPointValues}. */
  sums: GraphDataView<'uint32'>;
  /** Optional 96-bit fixed-point sum of squared deviations from the running mean (Welford), three words per slot. */
  moments?: GraphDataView<'uint32'>;
  /** Optional `(minimum, maximum)` per slot. */
  extrema?: GraphDataView<'float32'>;
};

/**
 * Properties of {@link GPURasterStreamingAccumulator}.
 *
 * Per-frame: the contents of `values`, `validity`, `slotIndices` and `parameters`. Compile-time:
 * `cellCount`, `chunkSliceCount`, `slotCount`, and which optional views and state fields exist.
 */
export type GPURasterStreamingAccumulatorProps = {
  id?: string;
  cellCount: number;
  /** Slices of each chunk, in `[1, 256]`. The stream itself has no slice limit. */
  chunkSliceCount: number;
  /** Calendar slots per cell, for per-slot climatologies. Defaults to 1. */
  slotCount?: number;
  /** Cell-major chunk `cell * chunkSliceCount + slice`; nonfinite samples are missing. */
  values: GraphDataView<'float32'>;
  /** Per-observation QA, same layout as `values`; zero rejects the sample. */
  validity?: GraphDataView<'uint32'>;
  /** Slot of each chunk slice, shared by all cells; required when `slotCount` exceeds 1. Slots out of range are skipped. */
  slotIndices?: GraphDataView<'uint32'>;
  /** Float32, see {@link getGPURasterStreamingAccumulatorParameterValues}. */
  parameters: GraphDataView<'float32'>;
  state: GPURasterStreamingState;
};

/**
 * Accumulates raster time slices chunk by chunk, so a stack is not limited by the 256-slice
 * private-array cap of the in-memory temporal contributors. Run it once per chunk, with `reset`
 * set on the first chunk of the stream; the cost is one per-cell pass over the chunk and memory is
 * O(cells x slots). Sums are exact 96-bit fixed-point integers (the shared fixed-point
 * accumulator, because float32 compensated-sum tricks are optimized away by shader compilers), so
 * totals over tens of thousands of slices stay exact to the declared unit. Each cell is one thread with fixed-order loops, so results do not depend on
 * scheduling.
 */
export class GPURasterStreamingAccumulator implements GPUCommandNodeProducer {
  readonly id: string;
  readonly slotCount: number;
  constructor(readonly props: GPURasterStreamingAccumulatorProps) {
    const id = props.id ?? 'raster-streaming-accumulator';
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
    validateCapacity(
      id,
      'cellCount times chunkSliceCount',
      props.cellCount * props.chunkSliceCount
    );
    validateCapacity(id, 'cellCount times slotCount', props.cellCount * slotCount);
    const entryCount = props.cellCount * slotCount;
    validateRequirement(
      id,
      props.values.length >= props.cellCount * props.chunkSliceCount &&
        (!props.validity || props.validity.length >= props.cellCount * props.chunkSliceCount),
      'values and validity must hold cellCount times chunkSliceCount entries'
    );
    validateRequirement(
      id,
      slotCount === 1 || (!!props.slotIndices && props.slotIndices.length >= props.chunkSliceCount),
      'slotIndices must hold one entry per chunk slice when slotCount exceeds 1'
    );
    validateRequirement(
      id,
      props.parameters.length >= GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH,
      `parameters must hold ${GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH} entries`
    );
    validateRequirement(
      id,
      props.state.counts.length >= entryCount && props.state.sums.length >= entryCount * 3,
      'state.counts must hold cellCount times slotCount entries and state.sums three times that'
    );
    validateRequirement(
      id,
      (!props.state.moments || props.state.moments.length >= entryCount * 3) &&
        (!props.state.extrema || props.state.extrema.length >= entryCount * 2),
      'state.moments must hold three entries per slot and state.extrema two'
    );
    validateRasterViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterScalarView[] {
    const {values, validity, slotIndices, parameters} = this.props;
    return [
      values,
      parameters,
      ...(validity ? [validity] : []),
      ...(slotIndices ? [slotIndices] : [])
    ];
  }

  private getOutputs(): RasterScalarView[] {
    const {counts, sums, moments, extrema} = this.props.state;
    return [counts, sums, ...(moments ? [moments] : []), ...(extrema ? [extrema] : [])];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, slotCount} = this;
    const {values, validity, slotIndices, parameters, state, cellCount, chunkSliceCount} = props;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    const slotExpression = slotCount > 1 ? 'slotIndices[slotIndicesOffset + slice]' : '0u';
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterStreamingAccumulator',
        invocationCount: cellCount,
        workgroupSize: 64,
        bindings: [
          rasterBinding('values', values),
          rasterBinding('parameters', parameters),
          ...(validity ? [rasterBinding('validity', validity)] : []),
          ...(slotCount > 1 && slotIndices ? [rasterBinding('slotIndices', slotIndices)] : []),
          rasterBinding('counts', state.counts, true),
          rasterBinding('sums', state.sums, true),
          ...(state.moments ? [rasterBinding('moments', state.moments, true)] : []),
          ...(state.extrema ? [rasterBinding('extrema', state.extrema, true)] : [])
        ],
        declarations: `${RASTER_FINITE_WGSL}
${FIXED_POINT_WGSL}
${getRasterFixedPointStateWGSL('sums')}
${state.moments ? getRasterFixedPointStateWGSL('moments') : ''}`,
        body: `let unitExponent = i32(parameters[parametersOffset + 2u]);
let momentUnitExponent = i32(parameters[parametersOffset + 3u]);
if (parameters[parametersOffset] != 0.0) {
  for (var slot = 0u; slot < ${slotCount}u; slot++) {
    let entry = index * ${slotCount}u + slot;
    counts[countsOffset + entry] = 0u;
    storeSums(entry, vec3<u32>(0u));
    ${state.moments ? 'storeMoments(entry, vec3<u32>(0u));' : ''}
    ${state.extrema ? 'extrema[extremaOffset + 2u * entry] = 3.402823466e38; extrema[extremaOffset + 2u * entry + 1u] = -3.402823466e38;' : ''}
  }
}
let activeSliceCount = u32(clamp(parameters[parametersOffset + 1u], 0.0, ${chunkSliceCount}.0));
for (var slice = 0u; slice < activeSliceCount; slice++) {
  let source = index * ${chunkSliceCount}u + slice;
  let value = values[valuesOffset + source];
  if (!finite(value) ${validity ? '|| validity[validityOffset + source] == 0u' : ''}) { continue; }
  if (abs(value) > parameters[parametersOffset + 4u]) { continue; }
  let slot = ${slotExpression};
  if (slot >= ${slotCount}u) { continue; }
  let entry = index * ${slotCount}u + slot;
  let priorCount = counts[countsOffset + entry];
  counts[countsOffset + entry] = priorCount + 1u;
  let priorSum = loadSums(entry);
  let priorMean = getFixedPointFloat(priorSum, unitExponent) / f32(max(priorCount, 1u));
  let sum = addFixedPoint(priorSum, quantizeFixedPoint(value, unitExponent));
  storeSums(entry, sum);
  ${
    state.moments
      ? `let updatedMean = getFixedPointFloat(sum, unitExponent) / f32(priorCount + 1u);
  let deviation = select((value - priorMean) * (value - updatedMean), 0.0, priorCount == 0u);
  storeMoments(entry, addFixedPoint(loadMoments(entry), quantizeFixedPoint(deviation, momentUnitExponent)));`
      : ''
  }
  ${
    state.extrema
      ? 'extrema[extremaOffset + 2u * entry] = min(extrema[extremaOffset + 2u * entry], value);\n  extrema[extremaOffset + 2u * entry + 1u] = max(extrema[extremaOffset + 2u * entry + 1u], value);'
      : ''
  }
}`
      })
    ];
  }
}
