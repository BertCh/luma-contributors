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
  validateCommandNodeBudget,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {RASTER_FINITE_WGSL, rasterBinding} from '../remote-sensing/raster-kernel-utils';
import {
  MAXIMUM_RASTER_STREAM_CHUNK_SLICE_COUNT,
  MAXIMUM_RASTER_STREAM_SLOT_COUNT,
  GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH
} from './gpu-raster-streaming-accumulator';
import {FIXED_POINT_WGSL} from '../../analysis-core/kernels/fixed-point-accumulator';
import {getRasterFixedPointStateWGSL} from './raster-fixed-point-state-wgsl';

/** Properties of {@link GPURasterClimateAnomaly}. */
export type GPURasterClimateAnomalyProps<
  Rows extends GraphDataView<'uint32'> | GraphVectorView<'uint32'> = GraphDataView<'uint32'>
> = {
  id?: string;
  cellCount: number;
  slotCount: number;
  /** State from a {@link GPURasterStreamingAccumulator}; `moments` is needed for `output.standardDeviation` or a standardized anomaly. */
  state: {
    counts: Rows;
    sums: RasterTimeSeriesField<'uint32', Rows>;
    moments?: RasterTimeSeriesField<'uint32', Rows>;
  };
  /** The accumulator's parameters view, which holds the fixed-point unit exponents of `state`. */
  accumulatorParameters: GraphDataView<'float32'>;
  /** Slices of the optional anomaly chunk, in `[1, 256]`; required with `anomaly`. */
  chunkSliceCount?: number;
  /** Chunk to express as anomalies, cell-major; required with `output.anomaly`. */
  values?: RasterTimeSeriesField<'float32', Rows>;
  /** Slot of each chunk slice, with `output.anomaly`. */
  slotIndices?: GraphDataView<'uint32'>;
  /** Divide anomalies by the slot standard deviation (`ddof = 1`). Needs `state.moments`. Defaults to false. */
  standardized?: boolean;
  output: {
    /** Mean per `cell * slotCount + slot`; NaN when the slot has no observations. */
    climatologyMean?: RasterTimeSeriesField<'float32', Rows>;
    /** Sample standard deviation per slot (`ddof = 1`); NaN with fewer than two observations. Needs `state.moments`. */
    standardDeviation?: RasterTimeSeriesField<'float32', Rows>;
    /** `value - mean` (or z-score) per `cell * chunkSliceCount + slice`; NaN for missing input or an empty slot. */
    anomaly?: RasterTimeSeriesField<'float32', Rows>;
  };
};

/**
 * Reads the climatology held in a raster streaming accumulator state: per-slot means
 * and sample standard deviations, and the anomaly of a chunk of slices against them. The state is
 * unchanged, so the climatology can serve any number of chunks.
 *
 * Cell-major fields accept packed views or matching vector chunks of complete cell records.
 * Shared date, model and parameter tables remain packed.
 *
 * Per-frame: the view contents. Compile-time: `cellCount`, `slotCount`, `chunkSliceCount`,
 * `standardized`, and which optional views exist.
 */
export class GPURasterClimateAnomaly<
  Rows extends GraphDataView<'uint32'> | GraphVectorView<'uint32'> = GraphDataView<'uint32'>
> implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly estimatedCommandNodeCount: number;
  private readonly chunkProps: readonly GPURasterClimateAnomalyProps<Rows>[];
  constructor(readonly props: GPURasterClimateAnomalyProps<Rows>) {
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
    this.chunkProps = planRasterTimeSeriesChunks(id, props, {
      'state.counts': props.slotCount,
      'state.sums': props.slotCount * 3,
      'state.moments': props.slotCount * 3,
      values: props.chunkSliceCount ?? 1,
      'output.climatologyMean': props.slotCount,
      'output.standardDeviation': props.slotCount,
      'output.anomaly': props.chunkSliceCount ?? 1
    });
    this.estimatedCommandNodeCount =
      this.chunkProps.length *
      ((props.output.climatologyMean || props.output.standardDeviation ? 1 : 0) +
        (props.output.anomaly ? 1 : 0));
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateRasterTimeSeriesViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterTimeSeriesScalarView[] {
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

  private getOutputs(): RasterTimeSeriesScalarView[] {
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
    validateRasterTimeSeriesBindings(graph, this.id, [...this.getInputs(), ...this.getOutputs()]);
    return this.chunkProps.flatMap((props, chunkIndex) =>
      this.getChunkCommandNodes(graph, {
        ...props,
        id: `${this.id}-chunk-${chunkIndex}`
      } as unknown as GPURasterClimateAnomalyProps)
    );
  }

  private getChunkCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    props: GPURasterClimateAnomalyProps
  ): readonly GPUCommandNode<Parameters>[] {
    const id = props.id!;
    const {cellCount, slotCount, state, output, values, slotIndices} = props;
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
        createRasterTimeSeriesKernelNode<Parameters>(graph, {
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
        createRasterTimeSeriesKernelNode<Parameters>(graph, {
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
