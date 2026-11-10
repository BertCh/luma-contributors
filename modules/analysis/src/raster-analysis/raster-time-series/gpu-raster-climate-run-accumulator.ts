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
  DEFAULT_MAXIMUM_RASTER_STREAM_SLICE_COUNT,
  MAXIMUM_RASTER_STREAM_CHUNK_SLICE_COUNT,
  getGPURasterFixedPointUnitExponent
} from './gpu-raster-streaming-accumulator';
import {FIXED_POINT_WGSL} from '../../analysis-core/kernels/fixed-point-accumulator';

/** Comparison of a sample against the threshold; missing samples never match it. */
export type GPURasterClimateCondition = 'above' | 'at-or-above' | 'below' | 'at-or-below';

/** Float32 elements of the per-frame parameters: reset, active slices, threshold, unit exponent. */
export const GPU_RASTER_CLIMATE_RUN_ACCUMULATOR_PARAMETER_LENGTH = 4;

/**
 * Packs `[reset, activeSliceCount, threshold, degreeDayUnitExponent]`. `maximumIncrement` bounds a
 * single degree-day term `max(0, value - threshold)`; it sets the fixed-point unit of the sum.
 */
export function getGPURasterClimateRunAccumulatorParameterValues(
  settings: {
    reset: boolean;
    activeSliceCount?: number;
    threshold: number;
    maximumIncrement: number;
    maximumStreamSliceCount?: number;
  },
  chunkSliceCount: number,
  target: Float32Array = new Float32Array(GPU_RASTER_CLIMATE_RUN_ACCUMULATOR_PARAMETER_LENGTH)
): Float32Array {
  validateFiniteNumber(
    'getGPURasterClimateRunAccumulatorParameterValues',
    'threshold',
    settings.threshold
  );
  target[0] = settings.reset ? 1 : 0;
  target[1] = settings.activeSliceCount ?? chunkSliceCount;
  target[2] = settings.threshold;
  target[3] = getGPURasterFixedPointUnitExponent(
    settings.maximumIncrement,
    settings.maximumStreamSliceCount ?? DEFAULT_MAXIMUM_RASTER_STREAM_SLICE_COUNT
  );
  return target;
}

/** Properties of {@link GPURasterClimateRunAccumulator}. */
export type GPURasterClimateRunAccumulatorProps = {
  id?: string;
  cellCount: number;
  /** Slices of each chunk, in `[1, 256]`. */
  chunkSliceCount: number;
  /** Cell-major chunk; nonfinite samples are missing. */
  values: GraphDataView<'float32'>;
  /** Per-observation QA; zero makes the sample missing. */
  validity?: GraphDataView<'uint32'>;
  /** Float32, see {@link getGPURasterClimateRunAccumulatorParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Run condition against the threshold. Defaults to `'at-or-above'`. */
  condition?: GPURasterClimateCondition;
  /** Run length at which a spell counts; defaults to 1. */
  minimumSpellLength?: number;
  state: {
    /** Per cell `[currentRun, longestRun, matchingSampleCount, spellCount, spellSampleCount, missingCount]`, uint32. */
    runs?: GraphDataView<'uint32'>;
    /** Per cell 96-bit fixed-point degree-day sum, three `uint32` words: `sum max(0, value - threshold)` for `'above'` conditions, `sum max(0, threshold - value)` for `'below'`; decode with `getGPURasterFixedPointValues`. */
    degreeDays?: GraphDataView<'uint32'>;
  };
};

/**
 * Streaming spell, exceedance and degree-day accumulator (the run and threshold indices of xclim).
 * Run it once per chunk with `reset` at the start of each period (typically each year, matching
 * xclim's `resample_before_rl`). Missing samples break the current run and are counted in
 * `missingCount`. Degree-day totals are exact 96-bit fixed-point sums, so a total over decades
 * loses nothing to float32 rounding.
 *
 * Per-frame: `parameters` and the view contents. Compile-time: `cellCount`, `chunkSliceCount`,
 * `condition`, `minimumSpellLength`, and which optional views and state exist.
 */
export class GPURasterClimateRunAccumulator implements GPUCommandNodeProducer {
  readonly id: string;
  readonly condition: GPURasterClimateCondition;
  readonly minimumSpellLength: number;
  constructor(readonly props: GPURasterClimateRunAccumulatorProps) {
    const id = props.id ?? 'raster-climate-run-accumulator';
    this.id = id;
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
    validateRequirement(
      id,
      ['above', 'at-or-above', 'below', 'at-or-below'].includes(this.condition),
      "condition must be 'above', 'at-or-above', 'below' or 'at-or-below'"
    );
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
        (!props.validity || props.validity.length >= observationCount),
      'values and validity must hold cellCount times chunkSliceCount entries'
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
    validateRasterViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterScalarView[] {
    const {values, validity, parameters} = this.props;
    return [values, parameters, ...(validity ? [validity] : [])];
  }

  private getOutputs(): RasterScalarView[] {
    const {runs, degreeDays} = this.props.state;
    return [...(runs ? [runs] : []), ...(degreeDays ? [degreeDays] : [])];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, condition, minimumSpellLength} = this;
    const {chunkSliceCount: chunk, values, validity, parameters, state} = props;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
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
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterClimateRunAccumulator',
        invocationCount: props.cellCount,
        workgroupSize: 64,
        bindings: [
          rasterBinding('values', values),
          rasterBinding('parameters', parameters),
          ...(validity ? [rasterBinding('validity', validity)] : []),
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
  let present = finite(value) ${validity ? '&& validity[validityOffset + source] != 0u' : ''};
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
