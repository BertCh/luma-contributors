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
  FIXED_POINT_WGSL,
  FIXED_POINT_MOMENTS_WGSL,
  getFixedPointMagnitudeUnitExponent
} from '../../analysis-core/kernels/fixed-point-accumulator';
import {
  validateCapacity,
  getPackedUint32Values,
  type GPUPackedReadbackValues,
  validateGraphPorts,
  validateCommandNodeBudget,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getRasterStreamStatusWord,
  validateRasterStreamFrame,
  validateRasterStreamParameterTarget,
  validateRasterStreamEncodingPlan,
  validateRasterStreamHeadroom,
  getRasterStreamPositiveFloat,
  validateRasterStreamPlanFrame,
  type GPURasterStreamFrameSettings
} from './raster-stream-parameters';
export type {GPURasterStreamFrameSettings} from './raster-stream-parameters';
import {getRasterFixedPointStateWGSL} from './raster-fixed-point-state-wgsl';
import {RASTER_FINITE_WGSL} from '../remote-sensing/raster-kernel-utils';

/** Slices per dispatch: one invocation per cell loops over the chunk, so this is a per-thread bound. */
export const MAXIMUM_RASTER_STREAM_CHUNK_SLICE_COUNT = 256;
/** Calendar slots per cell (366 days of year): each slot adds a loop step at reset and per chunk. */
export const MAXIMUM_RASTER_STREAM_SLOT_COUNT = 366;
/** Float32 elements of the per-frame parameters: reset, active slice count, sum and moment unit exponents, maximum magnitude. */
export const GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH = 5;
/** Default stream length bound for the fixed-point headroom (2^24 slices, over 45,000 years of days). */
export const DEFAULT_MAXIMUM_RASTER_STREAM_SLICE_COUNT = 2 ** 24;
/** Maximum representable valid observations per cell and slot, independently of frame size/headroom. */
export const MAXIMUM_RASTER_STREAM_OBSERVATION_COUNT = 0xffffffff;

/** Read one per-cell/slot loss certificate: overflow is sticky until reset. */
export function decodeGPURasterStreamingAccumulatorStatus(input: GPUPackedReadbackValues): {
  complete: boolean;
  overflow: boolean;
} {
  const code = getRasterStreamStatusWord('decodeGPURasterStreamingAccumulatorStatus', input, 1);
  return {complete: code === 0, overflow: code === 1};
}

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

/** Immutable units and headroom of one retained stream; replace only when resetting state. */
export type GPURasterStreamingAccumulatorEncodingPlan = Readonly<{
  kind: 'raster-streaming-accumulator';
  chunkSliceCount: number;
  maximumValueMagnitude: number;
  maximumStreamSliceCount: number;
  sumUnitExponent: number;
  momentUnitExponent: number;
}>;

/** Capture scale once, then reuse this plan for every frame and for readback decoding. */
export function createGPURasterStreamingAccumulatorEncodingPlan(
  settings: Pick<
    GPURasterStreamingAccumulatorSettings,
    'maximumValueMagnitude' | 'maximumStreamSliceCount'
  >,
  chunkSliceCount: number
): GPURasterStreamingAccumulatorEncodingPlan {
  const id = 'createGPURasterStreamingAccumulatorEncodingPlan';
  validateRasterStreamFrame(id, chunkSliceCount, 0, MAXIMUM_RASTER_STREAM_CHUNK_SLICE_COUNT);
  const maximumValueMagnitude = getRasterStreamPositiveFloat(
    id,
    'maximumValueMagnitude',
    settings.maximumValueMagnitude
  );
  const maximumStreamSliceCount =
    settings.maximumStreamSliceCount ?? DEFAULT_MAXIMUM_RASTER_STREAM_SLICE_COUNT;
  validateRasterStreamHeadroom(id, maximumStreamSliceCount);
  return Object.freeze({
    kind: 'raster-streaming-accumulator',
    chunkSliceCount,
    maximumValueMagnitude,
    maximumStreamSliceCount,
    sumUnitExponent: getGPURasterFixedPointUnitExponent(
      maximumValueMagnitude,
      maximumStreamSliceCount
    ),
    momentUnitExponent: getGPURasterFixedPointUnitExponent(
      4 * maximumValueMagnitude ** 2,
      maximumStreamSliceCount
    )
  });
}

/** Packs `[reset, activeSliceCount, sumUnitExponent, momentUnitExponent, maximumValueMagnitude]`.
 * Legacy settings calls must keep the scale and headroom fixed until reset; prefer the plan overload.
 */
export function getGPURasterStreamingAccumulatorParameterValues(
  settings: GPURasterStreamingAccumulatorSettings,
  chunkSliceCount: number,
  target?: Float32Array
): Float32Array;
export function getGPURasterStreamingAccumulatorParameterValues(
  plan: GPURasterStreamingAccumulatorEncodingPlan,
  frame: GPURasterStreamFrameSettings,
  target?: Float32Array
): Float32Array;
export function getGPURasterStreamingAccumulatorParameterValues(
  settings: GPURasterStreamingAccumulatorSettings | GPURasterStreamingAccumulatorEncodingPlan,
  chunkOrFrame: number | GPURasterStreamFrameSettings,
  target: Float32Array = new Float32Array(GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH)
): Float32Array {
  const id = 'getGPURasterStreamingAccumulatorParameterValues';
  validateRasterStreamParameterTarget(
    id,
    target,
    GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH
  );
  const legacy = typeof chunkOrFrame === 'number';
  const plan = legacy
    ? createGPURasterStreamingAccumulatorEncodingPlan(
        settings as GPURasterStreamingAccumulatorSettings,
        chunkOrFrame
      )
    : (settings as GPURasterStreamingAccumulatorEncodingPlan);
  if (!legacy)
    validateRasterStreamEncodingPlan(id, plan, 'raster-streaming-accumulator', () =>
      createGPURasterStreamingAccumulatorEncodingPlan(plan, plan.chunkSliceCount)
    );
  const frame =
    typeof chunkOrFrame === 'number'
      ? (settings as GPURasterStreamingAccumulatorSettings)
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
    plan.sumUnitExponent,
    plan.momentUnitExponent,
    plan.maximumValueMagnitude
  ]);
  return target;
}

/**
 * Unit exponent of a streamed fixed-point sum whose terms stay below `maximumMagnitude`, with
 * headroom for `maximumTermCount` terms. Bounds stay binary64, including squared observation bounds
 * beyond float32 range. Reset retained state when adopting corrected units for an existing stream.
 * Pass the value with the sum's readback to decode it.
 */
export function getGPURasterFixedPointUnitExponent(
  maximumMagnitude: number,
  maximumTermCount: number
): number {
  return getFixedPointMagnitudeUnitExponent(maximumMagnitude, maximumTermCount);
}

/** Decodes 96-bit two's complement sums, three uint32 words each, in units of 2^unitExponent.
 * Accepts numeric words or bounded Uint8Array/ArrayBuffer GPU readback bytes (little endian).
 * Byte offset and length are honored; partial words and partial triples are rejected.
 */
export function getGPURasterFixedPointValues(
  input: GPUPackedReadbackValues,
  unitExponent: number
): Float64Array {
  const words = getPackedUint32Values(input);
  validateRequirement(
    'getGPURasterFixedPointValues',
    words.length % 3 === 0,
    `words must contain a whole number of three-word values (expected a multiple of 3 words); received ${words.length}`
  );
  const values = new Float64Array(words.length / 3);
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
export type GPURasterStreamingState<View extends RasterScalarView = GraphDataView> = {
  /** Valid observations per slot. */
  counts: RasterTimeSeriesView<'uint32', View>;
  /** 96-bit fixed-point sum of the samples, three `uint32` words per slot; decode with {@link getGPURasterFixedPointValues}. */
  sums: RasterTimeSeriesView<'uint32', View>;
  /** Optional 96-bit fixed-point sum of squared deviations from the running mean (Welford), three words per slot. */
  moments?: RasterTimeSeriesView<'uint32', View>;
  /** Optional sticky loss certificate per slot: 0 complete, 1 observation-count overflow. Supply this to detect count overflow; declared encoding headroom must still be honored. Reset clears it. */
  status?: RasterTimeSeriesView<'uint32', View>;
  /** Optional `(minimum, maximum)` per slot. */
  extrema?: RasterTimeSeriesView<'float32', View>;
};

/**
 * Properties of {@link GPURasterStreamingAccumulator}.
 *
 * Per-frame: the contents of `values`, `mask`, `slotIndices` and `parameters`. Compile-time:
 * `cellCount`, `chunkSliceCount`, `slotCount`, and which optional views and state fields exist.
 */
export type GPURasterStreamingAccumulatorProps<Rows extends RasterScalarView = GraphDataView> = {
  id?: string;
  cellCount: number;
  /** Slices of each chunk, in `[1, 256]`. Stream headroom defaults to 2^24; counts hold at most 2^32 - 1 valid observations per cell and slot. */
  chunkSliceCount: number;
  /** Calendar slots per cell, for per-slot climatologies. Defaults to 1. */
  slotCount?: number;
  /** Cell-major chunk `cell * chunkSliceCount + slice`; nonfinite samples are missing. */
  values: RasterTimeSeriesView<'float32', Rows>;
  /** Per-observation QA, same layout as `values`; zero rejects the sample. */
  mask?: RasterTimeSeriesView<'uint32', Rows>;
  /** Slot of each chunk slice, shared by all cells; required when `slotCount` exceeds 1. Slots out of range are skipped. */
  slotIndices?: GraphDataView<'uint32'>;
  /** Float32, see {@link getGPURasterStreamingAccumulatorParameterValues}. */
  parameters: GraphDataView<'float32'>;
  state: GPURasterStreamingState<Rows>;
};

/**
 * Accumulates raster time slices chunk by chunk, so a stack is not limited by the 256-slice
 * private-array cap of the in-memory temporal contributors. Run it once per chunk, with `reset`
 * set on the first chunk of the stream; the cost is one per-cell pass over the chunk and memory is
 * O(cells x slots). Sums are exact 96-bit fixed-point integers (the shared fixed-point
 * accumulator, because float32 compensated-sum tricks are optimized away by shader compilers), so
 * totals over tens of thousands of slices stay exact to the declared unit. Each cell is one thread with fixed-order loops, so results do not depend on
 * scheduling. Squared deviations center in fixed-point units before float conversion and scale before
 * multiplication, retaining the same three-word moment encoding in squared observation units.
 *
 * At most 256 slices per submission; declared stream headroom defaults to 2^24 slices. Per-cell,
 * per-slot counts saturate at 2^32 - 1 rather than wrap. Supply state.status to detect loss on replay;
 * overflow preserves prior statistics and remains set until reset. Status adds one per-slot pass,
 * keeping every kernel within eight storage bindings even with all optional statistics. Legacy counts alone cannot certify
 * completeness. Keep the encoding plan fixed between resets, including magnitude and headroom.
 * Cell-major views accept matching chunks of complete cell records; shared tables stay packed.
 */
export class GPURasterStreamingAccumulator<Rows extends RasterScalarView = GraphDataView>
  implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly props: GPURasterStreamingAccumulatorProps<Rows>;
  readonly chunkPlan: RasterTimeSeriesChunkPlan;
  readonly estimatedCommandNodeCount: number;
  readonly slotCount: number;
  constructor(props: GPURasterStreamingAccumulatorProps<Rows> & {values: Rows}) {
    this.props = {...props, state: {...props.state}};
    const id = props.id ?? 'raster-streaming-accumulator';
    this.id = id;
    validateGraphPorts(id, props, {
      values: {required: true, viewType: 'either', formats: ['float32']},
      mask: {required: false, viewType: 'either', formats: ['uint32']},
      slotIndices: {required: false, viewType: 'data', formats: ['uint32']},
      parameters: {required: true, viewType: 'data', formats: ['float32']},
      'state.counts': {required: true, viewType: 'either', formats: ['uint32']},
      'state.sums': {required: true, viewType: 'either', formats: ['uint32']},
      'state.moments': {required: false, viewType: 'either', formats: ['uint32']},
      'state.extrema': {required: false, viewType: 'either', formats: ['float32']},
      'state.status': {required: false, viewType: 'either', formats: ['uint32']}
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
        (!props.mask || props.mask.length >= props.cellCount * props.chunkSliceCount),
      'values and mask must hold cellCount times chunkSliceCount entries'
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
        (!props.state.extrema || props.state.extrema.length >= entryCount * 2) &&
        (!props.state.status || props.state.status.length >= entryCount),
      'state.moments must hold three entries per slot and state.extrema two; state.status needs one entry per slot'
    );
    this.chunkPlan = planRasterTimeSeriesChunks(this.id, props.cellCount, {
      values: {view: props.values, entriesPerCell: props.chunkSliceCount},
      mask: {view: props.mask, entriesPerCell: props.chunkSliceCount},
      'state.counts': {view: props.state.counts, entriesPerCell: props.slotCount ?? 1},
      'state.sums': {view: props.state.sums, entriesPerCell: (props.slotCount ?? 1) * 3},
      'state.moments': {view: props.state.moments, entriesPerCell: (props.slotCount ?? 1) * 3},
      'state.extrema': {view: props.state.extrema, entriesPerCell: (props.slotCount ?? 1) * 2},
      'state.status': {view: props.state.status, entriesPerCell: props.slotCount ?? 1}
    });
    this.estimatedCommandNodeCount =
      this.chunkPlan.estimatedCommandNodeCount * (props.state.status ? 2 : 1);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    const inputs = this.getInputs();
    const accumulationOutputs = [
      props.state.counts,
      props.state.sums,
      ...(props.state.moments ? [props.state.moments] : []),
      ...(props.state.extrema ? [props.state.extrema] : [])
    ];
    validateRasterViews(this.id, inputs, this.getOutputs(), this.chunkPlan, [
      ...(props.state.status
        ? [{inputs: [...inputs, props.state.counts], outputs: [props.state.status]}]
        : []),
      {inputs, outputs: accumulationOutputs}
    ]);
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
    const {counts, sums, moments, extrema, status} = this.props.state;
    return [
      counts,
      sums,
      ...(moments ? [moments] : []),
      ...(extrema ? [extrema] : []),
      ...(status ? [status] : [])
    ];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, slotCount} = this;
    const {values, mask, slotIndices, parameters, state, cellCount, chunkSliceCount} = props;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    const slotExpression = slotCount > 1 ? 'slotIndices[slotIndicesOffset + slice]' : '0u';
    // A separate per-slot pass preserves the eight-storage-binding route when every optional
    // statistic is requested. It reads the prior count before the accumulation pass changes it.
    const statusNodes = state.status
      ? createRasterTimeSeriesNodes<Parameters>(graph, this.chunkPlan, {
          id: `${id}-status`,
          operation: 'GPURasterStreamingAccumulatorStatus',
          invocationCount: cellCount * slotCount,
          workgroupSize: 64,
          bindings: [
            rasterBinding('values', values),
            rasterBinding('parameters', parameters),
            ...(mask ? [rasterBinding('mask', mask)] : []),
            ...(slotCount > 1 && slotIndices ? [rasterBinding('slotIndices', slotIndices)] : []),
            rasterBinding('counts', state.counts),
            rasterBinding('status', state.status, true)
          ],
          declarations: RASTER_FINITE_WGSL,
          body: `let cell = index / ${slotCount}u;
let outputSlot = index % ${slotCount}u;
var count = 0u;
var overflow = 0u;
if (parameters[parametersOffset] == 0.0) {
  count = counts[countsOffset + index];
  overflow = status[statusOffset + index];
}
let activeSliceCount = u32(clamp(parameters[parametersOffset + 1u], 0.0, ${chunkSliceCount}.0));
for (var slice = 0u; slice < activeSliceCount; slice++) {
  let source = cell * ${chunkSliceCount}u + slice;
  let value = values[valuesOffset + source];
  if (!finite(value) ${mask ? '|| mask[maskOffset + source] == 0u' : ''}) { continue; }
  if (abs(value) > parameters[parametersOffset + 4u]) { continue; }
  let slot = ${slotExpression};
  if (slot != outputSlot) { continue; }
  if (count == ${MAXIMUM_RASTER_STREAM_OBSERVATION_COUNT}u) { overflow = 1u; }
  else { count += 1u; }
}
status[statusOffset + index] = overflow;`
        })
      : [];
    const accumulationNodes = createRasterTimeSeriesNodes<Parameters>(graph, this.chunkPlan, {
      id,
      operation: 'GPURasterStreamingAccumulator',
      invocationCount: cellCount,
      workgroupSize: 64,
      bindings: [
        rasterBinding('values', values),
        rasterBinding('parameters', parameters),
        ...(mask ? [rasterBinding('mask', mask)] : []),
        ...(slotCount > 1 && slotIndices ? [rasterBinding('slotIndices', slotIndices)] : []),
        rasterBinding('counts', state.counts, true),
        rasterBinding('sums', state.sums, true),
        ...(state.moments ? [rasterBinding('moments', state.moments, true)] : []),
        ...(state.extrema ? [rasterBinding('extrema', state.extrema, true)] : [])
      ],
      declarations: `${RASTER_FINITE_WGSL}
${FIXED_POINT_WGSL}
${state.moments ? FIXED_POINT_MOMENTS_WGSL : ''}
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
  if (!finite(value) ${mask ? '|| mask[maskOffset + source] == 0u' : ''}) { continue; }
  if (abs(value) > parameters[parametersOffset + 4u]) { continue; }
  let slot = ${slotExpression};
  if (slot >= ${slotCount}u) { continue; }
  let entry = index * ${slotCount}u + slot;
  let priorCount = counts[countsOffset + entry];
  if (priorCount == ${MAXIMUM_RASTER_STREAM_OBSERVATION_COUNT}u) { continue; }
  counts[countsOffset + entry] = priorCount + 1u;
  let priorSum = loadSums(entry);
  let sum = addFixedPoint(priorSum, quantizeFixedPoint(value, unitExponent));
  storeSums(entry, sum);
  ${
    state.moments
      ? `let momentIncrement = getFixedPointMomentIncrement(priorSum, priorCount, value, unitExponent, momentUnitExponent);
  storeMoments(entry, addFixedPoint(loadMoments(entry), momentIncrement));`
      : ''
  }
  ${
    state.extrema
      ? 'extrema[extremaOffset + 2u * entry] = min(extrema[extremaOffset + 2u * entry], value);\n  extrema[extremaOffset + 2u * entry + 1u] = max(extrema[extremaOffset + 2u * entry + 1u], value);'
      : ''
  }
}`
    });
    return [...statusNodes, ...accumulationNodes];
  }
}
