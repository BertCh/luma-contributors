// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GraphDataView,
  GraphVectorView,
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {
  createRasterTimeSeriesKernelNode,
  planRasterTimeSeriesChunks,
  validateRasterTimeSeriesBindings,
  validateRasterTimeSeriesViews,
  type RasterTimeSeriesField,
  type RasterTimeSeriesScalarView
} from './raster-time-series-chunks';
import {
  validateCapacity,
  validateGraphPorts,
  validateOption,
  validateCommandNodeBudget,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {RASTER_FINITE_WGSL, rasterBinding} from '../remote-sensing/raster-kernel-utils';
import {
  DEFAULT_MAXIMUM_RASTER_STREAM_SLICE_COUNT,
  MAXIMUM_RASTER_STREAM_CHUNK_SLICE_COUNT,
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
import {FIXED_POINT_WGSL} from '../../analysis-core/kernels/fixed-point-accumulator';

/** Comparison of a sample against the threshold; missing samples never match it. */
export type GPURasterClimateCondition = 'above' | 'at-or-above' | 'below' | 'at-or-below';

/** Float32 elements of the per-frame parameters: reset, active slices, threshold, unit exponent. */
export const GPU_RASTER_CLIMATE_RUN_ACCUMULATOR_PARAMETER_LENGTH = 4;

/**
 * Packs `[reset, activeSliceCount, threshold, degreeDayUnitExponent]`. `maximumIncrement` bounds a
 * single degree-day term `max(0, value - threshold)`; it sets the fixed-point unit of the sum.
 */
export type GPURasterClimateRunAccumulatorSettings = {
  reset: boolean;
  activeSliceCount?: number;
  threshold: number;
  maximumIncrement: number;
  maximumStreamSliceCount?: number;
};

/** Immutable model settings and units; replace only when resetting all retained state. */
export type GPURasterClimateRunAccumulatorEncodingPlan = Readonly<{
  kind: 'raster-climate-run-accumulator';
  chunkSliceCount: number;
  maximumStreamSliceCount: number;
  threshold: number;
  maximumIncrement: number;
  degreeDayUnitExponent: number;
}>;

export function createGPURasterClimateRunAccumulatorEncodingPlan(
  settings: Omit<GPURasterClimateRunAccumulatorSettings, 'reset' | 'activeSliceCount'>,
  chunkSliceCount: number
): GPURasterClimateRunAccumulatorEncodingPlan {
  const id = 'createGPURasterClimateRunAccumulatorEncodingPlan';
  validateRasterStreamFrame(id, chunkSliceCount, 0, MAXIMUM_RASTER_STREAM_CHUNK_SLICE_COUNT);
  validateFiniteNumber(id, 'threshold', settings.threshold);
  const threshold = Math.fround(settings.threshold);
  const maximumIncrement = getRasterStreamPositiveFloat(
    id,
    'maximumIncrement',
    settings.maximumIncrement
  );
  const maximumStreamSliceCount =
    settings.maximumStreamSliceCount ?? DEFAULT_MAXIMUM_RASTER_STREAM_SLICE_COUNT;
  validateRasterStreamHeadroom(id, maximumStreamSliceCount);
  return Object.freeze({
    kind: 'raster-climate-run-accumulator',
    chunkSliceCount,
    maximumStreamSliceCount,
    threshold,
    maximumIncrement,
    degreeDayUnitExponent: getGPURasterFixedPointUnitExponent(
      maximumIncrement,
      maximumStreamSliceCount
    )
  });
}

/** Pack legacy settings with fixed encoding until reset, or reuse an immutable encoding plan. */
export function getGPURasterClimateRunAccumulatorParameterValues(
  settings: GPURasterClimateRunAccumulatorSettings,
  chunkSliceCount: number,
  target?: Float32Array
): Float32Array;
export function getGPURasterClimateRunAccumulatorParameterValues(
  plan: GPURasterClimateRunAccumulatorEncodingPlan,
  frame: GPURasterStreamFrameSettings,
  target?: Float32Array
): Float32Array;
export function getGPURasterClimateRunAccumulatorParameterValues(
  settings: GPURasterClimateRunAccumulatorSettings | GPURasterClimateRunAccumulatorEncodingPlan,
  chunkOrFrame: number | GPURasterStreamFrameSettings,
  target: Float32Array = new Float32Array(GPU_RASTER_CLIMATE_RUN_ACCUMULATOR_PARAMETER_LENGTH)
): Float32Array {
  const id = 'getGPURasterClimateRunAccumulatorParameterValues';
  validateRasterStreamParameterTarget(
    id,
    target,
    GPU_RASTER_CLIMATE_RUN_ACCUMULATOR_PARAMETER_LENGTH
  );
  const legacy = typeof chunkOrFrame === 'number';
  const plan = legacy
    ? createGPURasterClimateRunAccumulatorEncodingPlan(
        settings as GPURasterClimateRunAccumulatorSettings,
        chunkOrFrame
      )
    : (settings as GPURasterClimateRunAccumulatorEncodingPlan);
  if (!legacy)
    validateRasterStreamEncodingPlan(id, plan, 'raster-climate-run-accumulator', () =>
      createGPURasterClimateRunAccumulatorEncodingPlan(plan, plan.chunkSliceCount)
    );
  const frame =
    typeof chunkOrFrame === 'number'
      ? (settings as GPURasterClimateRunAccumulatorSettings)
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
  target.set([frame.reset ? 1 : 0, activeSliceCount, plan.threshold, plan.degreeDayUnitExponent]);
  return target;
}

/** Properties of {@link GPURasterClimateRunAccumulator}. */
export type GPURasterClimateRunAccumulatorProps<
  Rows extends GraphDataView<'float32'> | GraphVectorView<'float32'> = GraphDataView<'float32'>
> = {
  id?: string;
  cellCount: number;
  /** Slices of each chunk, in `[1, 256]`. */
  chunkSliceCount: number;
  /** Cell-major chunk; nonfinite samples are missing. */
  values: Rows;
  /** Per-observation QA; zero makes the sample missing. */
  mask?: RasterTimeSeriesField<'uint32', Rows>;
  /** Float32, see {@link getGPURasterClimateRunAccumulatorParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Run condition against the threshold. Defaults to `'at-or-above'`. */
  condition?: GPURasterClimateCondition;
  /** Run length at which a spell counts; defaults to 1. */
  minimumSpellLength?: number;
  state: {
    /** Per cell `[currentRun, longestRun, matchingSampleCount, spellCount, spellSampleCount, missingCount]`, uint32. */
    runs?: RasterTimeSeriesField<'uint32', Rows>;
    /** Per cell 96-bit fixed-point degree-day sum, three `uint32` words: `sum max(0, value - threshold)` for `'above'` conditions, `sum max(0, threshold - value)` for `'below'`; decode with `getGPURasterFixedPointValues`. */
    degreeDays?: RasterTimeSeriesField<'uint32', Rows>;
  };
};

/**
 * Streaming spell, exceedance and degree-day accumulator (the run and threshold indices of xclim).
 * Run it once per chunk with `reset` at the start of each period (typically each year, matching
 * xclim's `resample_before_rl`). Missing samples break the current run and are counted in
 * `missingCount`. Degree-day totals are exact 96-bit fixed-point sums, so a total over decades
 * loses nothing to float32 rounding.
 *
 * Cell-major fields accept packed views or matching vector chunks of complete cell records.
 * Shared date, model and parameter tables remain packed.
 *
 * Per-frame: `parameters` and the view contents. Compile-time: `cellCount`, `chunkSliceCount`,
 * `condition`, `minimumSpellLength`, and which optional views and state exist.
 */
export class GPURasterClimateRunAccumulator<
  Rows extends GraphDataView<'float32'> | GraphVectorView<'float32'> = GraphDataView<'float32'>
> implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly estimatedCommandNodeCount: number;
  private readonly chunkProps: readonly GPURasterClimateRunAccumulatorProps<Rows>[];
  readonly condition: GPURasterClimateCondition;
  readonly minimumSpellLength: number;
  constructor(readonly props: GPURasterClimateRunAccumulatorProps<Rows>) {
    props = {...props, state: {...props.state}};
    this.props = props;
    const id = props.id ?? 'raster-climate-run-accumulator';
    this.id = id;
    validateGraphPorts(id, props, {
      values: {required: true, viewType: 'either', formats: ['float32']},
      mask: {required: false, viewType: 'either', formats: ['uint32']},
      parameters: {required: true, viewType: 'data', formats: ['float32']},
      'state.runs': {required: false, viewType: 'either', formats: ['uint32']},
      'state.degreeDays': {required: false, viewType: 'either', formats: ['uint32']}
    });
    this.condition = props.condition ?? 'at-or-above';
    this.minimumSpellLength = props.minimumSpellLength ?? 1;
    validateCapacity(id, 'cellCount', props.cellCount);
    validateCapacity(
      id,
      'chunkSliceCount',
      props.chunkSliceCount,
      1,
      MAXIMUM_RASTER_STREAM_CHUNK_SLICE_COUNT
    );
    validateCapacity(id, 'minimumSpellLength', this.minimumSpellLength, 1, 0xffffffff);
    validateOption(id, 'condition', this.condition, [
      'above',
      'at-or-above',
      'below',
      'at-or-below'
    ]);
    validateRequirement(
      id,
      !!props.state.runs || !!props.state.degreeDays,
      'state needs runs or degreeDays'
    );
    const observationCount = props.cellCount * props.chunkSliceCount;
    validateCapacity(id, 'cellCount times chunkSliceCount', observationCount);
    validateRequirement(
      id,
      props.values.length >= observationCount &&
        (!props.mask || props.mask.length >= observationCount),
      'values and mask must hold cellCount times chunkSliceCount entries'
    );
    validateRequirement(
      id,
      (!props.state.runs || props.state.runs.length >= props.cellCount * 6) &&
        (!props.state.degreeDays || props.state.degreeDays.length >= props.cellCount * 3),
      'state.runs needs six entries per cell and state.degreeDays three'
    );
    validateRequirement(
      id,
      props.parameters.length >= GPU_RASTER_CLIMATE_RUN_ACCUMULATOR_PARAMETER_LENGTH,
      `parameters must hold ${GPU_RASTER_CLIMATE_RUN_ACCUMULATOR_PARAMETER_LENGTH} entries`
    );
    this.chunkProps = planRasterTimeSeriesChunks(id, props, {
      values: props.chunkSliceCount,
      mask: props.chunkSliceCount,
      'state.runs': 6,
      'state.degreeDays': 3
    });
    this.estimatedCommandNodeCount = this.chunkProps.length * 1;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateRasterTimeSeriesViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterTimeSeriesScalarView[] {
    const {values, mask, parameters} = this.props;
    return [values, parameters, ...(mask ? [mask] : [])];
  }

  private getOutputs(): RasterTimeSeriesScalarView[] {
    const {runs, degreeDays} = this.props.state;
    return [...(runs ? [runs] : []), ...(degreeDays ? [degreeDays] : [])];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateRasterTimeSeriesBindings(graph, this.id, [...this.getInputs(), ...this.getOutputs()]);
    return this.chunkProps.flatMap((props, chunkIndex) =>
      this.getChunkCommandNodes(graph, {
        ...props,
        id: `${this.id}-chunk-${chunkIndex}`
      } as unknown as GPURasterClimateRunAccumulatorProps)
    );
  }

  private getChunkCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    props: GPURasterClimateRunAccumulatorProps
  ): readonly GPUCommandNode<Parameters>[] {
    const {condition, minimumSpellLength} = this;
    const id = props.id!;
    const {chunkSliceCount: chunk, values, mask, parameters, state} = props;
    const conditionOperator = {
      above: '>',
      'at-or-above': '>=',
      below: '<',
      'at-or-below': '<='
    }[condition];
    const degreeDayValue = condition.endsWith('above')
      ? 'max(value - threshold, 0.0)'
      : 'max(threshold - value, 0.0)';
    return [
      createRasterTimeSeriesKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterClimateRunAccumulator',
        invocationCount: props.cellCount,
        workgroupSize: 64,
        bindings: [
          rasterBinding('values', values),
          rasterBinding('parameters', parameters),
          ...(mask ? [rasterBinding('mask', mask)] : []),
          ...(state.runs ? [rasterBinding('runs', state.runs, true)] : []),
          ...(state.degreeDays ? [rasterBinding('degreeDays', state.degreeDays, true)] : [])
        ],
        declarations: `${RASTER_FINITE_WGSL}\n${state.degreeDays ? FIXED_POINT_WGSL : ''}`,
        body: `if (parameters[parametersOffset] != 0.0) {
  ${state.runs ? 'for (var word = 0u; word < 6u; word++) { runs[runsOffset + 6u * index + word] = 0u; }' : ''}
  ${state.degreeDays ? 'degreeDays[degreeDaysOffset + 3u * index] = 0u; degreeDays[degreeDaysOffset + 3u * index + 1u] = 0u; degreeDays[degreeDaysOffset + 3u * index + 2u] = 0u;' : ''}
}
let activeSliceCount = u32(clamp(parameters[parametersOffset + 1u], 0.0, ${chunk}.0));
let threshold = parameters[parametersOffset + 2u];
let degreeDayUnit = i32(parameters[parametersOffset + 3u]);
for (var slice = 0u; slice < activeSliceCount; slice++) {
  let source = index * ${chunk}u + slice;
  let value = values[valuesOffset + source];
  let present = finite(value) ${mask ? '&& mask[maskOffset + source] != 0u' : ''};
  ${
    state.runs
      ? `let runWord = runsOffset + 6u * index;
  if (!present) { runs[runWord + 5u] += 1u; runs[runWord] = 0u; continue; }
  if (value ${conditionOperator} threshold) {
    let run = runs[runWord] + 1u;
    runs[runWord] = run;
    runs[runWord + 1u] = max(runs[runWord + 1u], run);
    runs[runWord + 2u] += 1u;
    if (run == ${minimumSpellLength}u) { runs[runWord + 3u] += 1u; runs[runWord + 4u] += ${minimumSpellLength}u; }
    else if (run > ${minimumSpellLength}u) { runs[runWord + 4u] += 1u; }
  } else {
    runs[runWord] = 0u;
  }`
      : 'if (!present) { continue; }'
  }
  ${
    state.degreeDays
      ? `let priorDegreeDays = vec3<u32>(degreeDays[degreeDaysOffset + 3u * index], degreeDays[degreeDaysOffset + 3u * index + 1u], degreeDays[degreeDaysOffset + 3u * index + 2u]);
  let degreeState = addFixedPoint(priorDegreeDays, quantizeFixedPoint(${degreeDayValue}, degreeDayUnit));
  degreeDays[degreeDaysOffset + 3u * index] = degreeState.x;
  degreeDays[degreeDaysOffset + 3u * index + 1u] = degreeState.y;
  degreeDays[degreeDaysOffset + 3u * index + 2u] = degreeState.z;`
      : ''
  }
}`
      })
    ];
  }
}
