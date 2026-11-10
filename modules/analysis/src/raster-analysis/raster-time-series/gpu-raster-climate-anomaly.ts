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
  GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH,
  type GPURasterStreamingState
} from './gpu-raster-streaming-accumulator';
import {FIXED_POINT_WGSL} from '../../analysis-core/kernels/fixed-point-accumulator';
import {getRasterFixedPointStateWGSL} from './raster-fixed-point-state-wgsl';

/** Properties of {@link GPURasterClimateAnomaly}. */
export type GPURasterClimateAnomalyProps = {
  id?: string;
  cellCount: number;
  slotCount: number;
  /** State from a {@link GPURasterStreamingAccumulator}; `moments` is needed for `output.standardDeviation` or a standardized anomaly. */
  state: Pick<GPURasterStreamingState, 'counts' | 'sums' | 'moments'>;
  /** The accumulator's parameters view, which holds the fixed-point unit exponents of `state`. */
  accumulatorParameters: GraphDataView<'float32'>;
  /** Slices of the optional anomaly chunk, in `[1, 256]`; required with `anomaly`. */
  chunkSliceCount?: number;
  /** Chunk to express as anomalies, cell-major; required with `output.anomaly`. */
  values?: GraphDataView<'float32'>;
  /** Slot of each chunk slice, with `output.anomaly`. */
  slotIndices?: GraphDataView<'uint32'>;
  /** Divide anomalies by the slot standard deviation (`ddof = 1`). Needs `state.moments`. Defaults to false. */
  standardized?: boolean;
  output: {
    /** Mean per `cell * slotCount + slot`; NaN when the slot has no observations. */
    climatologyMean?: GraphDataView<'float32'>;
    /** Sample standard deviation per slot (`ddof = 1`); NaN with fewer than two observations. Needs `state.moments`. */
    standardDeviation?: GraphDataView<'float32'>;
    /** `value - mean` (or z-score) per `cell * chunkSliceCount + slice`; NaN for missing input or an empty slot. */
    anomaly?: GraphDataView<'float32'>;
  };
};

/**
 * Reads the climatology held in a {@link GPURasterStreamingAccumulator}'s state: per-slot means
 * and sample standard deviations, and the anomaly of a chunk of slices against them. The state is
 * unchanged, so the climatology can serve any number of chunks.
 *
 * Per-frame: the view contents. Compile-time: `cellCount`, `slotCount`, `chunkSliceCount`,
 * `standardized`, and which optional views exist.
 */
export class GPURasterClimateAnomaly implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPURasterClimateAnomalyProps) {
    const id = props.id ?? 'raster-climate-anomaly';
    this.id = id;
    const {cellCount, slotCount, state, output, standardized} = props;
    validateCapacity(id, 'cellCount', cellCount);
    validateCapacity(id, 'slotCount', slotCount, 1, MAXIMUM_RASTER_STREAM_SLOT_COUNT);
    validateCapacity(id, 'cellCount times slotCount', cellCount * slotCount);
    validateRequirement(
      id,
      props.accumulatorParameters.length >= GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH,
      `accumulatorParameters must hold ${GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH} entries`
    );
    validateRequirement(
      id,
      !!(output.climatologyMean || output.standardDeviation || output.anomaly),
      'output needs at least one of climatologyMean, standardDeviation or anomaly'
    );
    const entries = cellCount * slotCount;
    validateRequirement(
      id,
      state.counts.length >= entries && state.sums.length >= entries * 3,
      'state.counts and state.sums must hold cellCount times slotCount entries (sums three times that)'
    );
    validateRequirement(
      id,
      !(output.standardDeviation || standardized) ||
        (!!state.moments && state.moments.length >= entries * 3),
      'state.moments with three entries per slot is required for standardDeviation or standardized anomalies'
    );
    validateRequirement(
      id,
      (!output.climatologyMean || output.climatologyMean.length >= entries) &&
        (!output.standardDeviation || output.standardDeviation.length >= entries),
      'output.climatologyMean and output.standardDeviation must hold cellCount times slotCount entries'
    );
    if (output.anomaly) {
      const chunk = props.chunkSliceCount ?? 0;
      validateCapacity(id, 'chunkSliceCount', chunk, 1, MAXIMUM_RASTER_STREAM_CHUNK_SLICE_COUNT);
      validateRequirement(
        id,
        !!props.values &&
          !!props.slotIndices &&
          props.values.length >= cellCount * chunk &&
          props.slotIndices.length >= chunk &&
          output.anomaly.length >= cellCount * chunk,
        'output.anomaly requires values and slotIndices holding one entry per chunk observation'
      );
    }
    validateRasterViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterScalarView[] {
    const {state, values, slotIndices, accumulatorParameters} = this.props;
    return [
      accumulatorParameters,
      state.counts,
      state.sums,
      ...(state.moments ? [state.moments] : []),
      ...(values ? [values] : []),
      ...(slotIndices ? [slotIndices] : [])
    ];
  }

  private getOutputs(): RasterScalarView[] {
    const {climatologyMean, standardDeviation, anomaly} = this.props.output;
    return [
      ...(climatologyMean ? [climatologyMean] : []),
      ...(standardDeviation ? [standardDeviation] : []),
      ...(anomaly ? [anomaly] : [])
    ];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {cellCount, slotCount, state, output, values, slotIndices} = props;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    const nodes: GPUCommandNode<Parameters>[] = [];
    const stateBindings = [
      rasterBinding('accumulatorParameters', props.accumulatorParameters),
      rasterBinding('counts', state.counts),
      rasterBinding('sums', state.sums),
      ...(state.moments ? [rasterBinding('moments', state.moments)] : [])
    ];
    const statisticsDeclarations = `${RASTER_FINITE_WGSL}
${FIXED_POINT_WGSL}
${getRasterFixedPointStateWGSL('sums', false)}
${state.moments ? getRasterFixedPointStateWGSL('moments', false) : ''}
fn slotMean(entry: u32) -> f32 {
  let count = counts[countsOffset + entry];
  if (count == 0u) { return missing(); }
  return getFixedPointFloat(loadSums(entry), i32(accumulatorParameters[accumulatorParametersOffset + 2u])) / f32(count);
}
${
  state.moments
    ? `fn slotVariance(entry: u32) -> f32 {
  let count = counts[countsOffset + entry];
  if (count < 2u) { return missing(); }
  return max(getFixedPointFloat(loadMoments(entry), i32(accumulatorParameters[accumulatorParametersOffset + 3u])) / f32(count - 1u), 0.0);
}`
    : ''
}`;
    if (output.climatologyMean || output.standardDeviation) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-climatology`,
          operation: 'GPURasterClimateAnomaly',
          variant: 'climatology',
          invocationCount: cellCount * slotCount,
          workgroupSize: 64,
          bindings: [
            ...stateBindings,
            ...(output.climatologyMean
              ? [rasterBinding('climatologyMean', output.climatologyMean, true)]
              : []),
            ...(output.standardDeviation
              ? [rasterBinding('standardDeviation', output.standardDeviation, true)]
              : [])
          ],
          declarations: statisticsDeclarations,
          body: `${output.climatologyMean ? 'climatologyMean[climatologyMeanOffset + index] = slotMean(index);' : ''}
${output.standardDeviation ? 'standardDeviation[standardDeviationOffset + index] = sqrt(slotVariance(index));' : ''}`
        })
      );
    }
    if (output.anomaly) {
      const chunk = props.chunkSliceCount!;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-anomaly`,
          operation: 'GPURasterClimateAnomaly',
          variant: 'anomaly',
          invocationCount: cellCount * chunk,
          workgroupSize: 64,
          bindings: [
            ...stateBindings,
            rasterBinding('values', values!),
            rasterBinding('slotIndices', slotIndices!),
            rasterBinding('anomalyOut', output.anomaly, true)
          ],
          declarations: statisticsDeclarations,
          body: `let cell = index / ${chunk}u;
let slice = index % ${chunk}u;
let value = values[valuesOffset + index];
let slot = slotIndices[slotIndicesOffset + slice];
var result = missing();
if (finite(value) && slot < ${slotCount}u) {
  let entry = cell * ${slotCount}u + slot;
  let difference = value - slotMean(entry);
  ${
    props.standardized
      ? `let deviation = sqrt(slotVariance(entry));
  if (deviation > 0.0) { result = difference / deviation; }`
      : 'result = difference;'
  }
}
anomalyOut[anomalyOutOffset + index] = result;`
        })
      );
    }
    return nodes;
  }
}
