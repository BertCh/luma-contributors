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
  validateCapacity,
  validateGraphPorts,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {RASTER_FINITE_WGSL} from '../remote-sensing/raster-kernel-utils';
import {
  GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH,
  MAXIMUM_RASTER_STREAM_CHUNK_SLICE_COUNT
} from './gpu-raster-streaming-accumulator';

/** Window length: each output sample rescans the window on the thread, so it is a per-thread bound. */
export const MAXIMUM_ROLLING_WINDOW_LENGTH = 128;

/** Properties of {@link GPURasterRollingMean}. */
export type GPURasterRollingMeanProps<Rows extends RasterScalarView = GraphDataView> = {
  id?: string;
  cellCount: number;
  /** Slices of each chunk, in `[1, 256]`. */
  chunkSliceCount: number;
  /** Trailing window in slices, in `[1, 128]`. */
  windowLength: number;
  /** Cell-major chunk `cell * chunkSliceCount + slice`; nonfinite samples are missing. */
  values: RasterTimeSeriesView<'float32', Rows>;
  /** Per-observation QA; zero makes the sample missing. */
  mask?: RasterTimeSeriesView<'uint32', Rows>;
  /** Float32, see `getGPURasterStreamingAccumulatorParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned ring of the last `windowLength` samples per cell (float32), carried between chunks. */
  ring: RasterTimeSeriesView<'float32', Rows>;
  /** Caller-owned samples seen so far per cell (uint32). */
  seenCount: RasterTimeSeriesView<'uint32', Rows>;
  /** Rolling mean per chunk slice, `cell * chunkSliceCount + slice`; NaN until the window is full or when it holds a missing sample. */
  output: RasterTimeSeriesView<'float32', Rows>;
};

/**
 * Trailing rolling mean over a chunked stream, the aggregation SPI and SPEI apply before fitting.
 * Any missing sample in the window makes that output missing, matching xclim's
 * `rolling(...).mean(skipna=False)`. The window state lives in caller-owned buffers between chunks.
 *
 * Per-frame: `parameters` and the contents of the views. Compile-time: `cellCount`,
 * `chunkSliceCount`, `windowLength`, and which optional views exist.
 *
 * Cell-major views accept matching chunks of complete cell records; shared tables stay packed.
 */
export class GPURasterRollingMean<Rows extends RasterScalarView = GraphDataView>
  implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly props: GPURasterRollingMeanProps<Rows>;
  readonly chunkPlan: RasterTimeSeriesChunkPlan;
  readonly estimatedCommandNodeCount: number;
  constructor(props: GPURasterRollingMeanProps<Rows> & {values: Rows}) {
    this.props = {...props};
    const id = props.id ?? 'raster-rolling-mean';
    this.id = id;
    validateGraphPorts(id, props, {
      values: {required: true, viewType: 'either', formats: ['float32']},
      mask: {required: false, viewType: 'either', formats: ['uint32']},
      parameters: {required: true, viewType: 'data', formats: ['float32']},
      ring: {required: true, viewType: 'either', formats: ['float32']},
      seenCount: {required: true, viewType: 'either', formats: ['uint32']},
      output: {required: true, viewType: 'either', formats: ['float32']}
    });
    validateCapacity(id, 'cellCount', props.cellCount);
    validateCapacity(
      id,
      'chunkSliceCount',
      props.chunkSliceCount,
      1,
      MAXIMUM_RASTER_STREAM_CHUNK_SLICE_COUNT
    );
    validateCapacity(id, 'windowLength', props.windowLength, 1, MAXIMUM_ROLLING_WINDOW_LENGTH);
    const observationCount = props.cellCount * props.chunkSliceCount;
    validateCapacity(id, 'cellCount times chunkSliceCount', observationCount);
    validateRequirement(
      id,
      props.values.length >= observationCount &&
        (!props.mask || props.mask.length >= observationCount) &&
        props.output.length >= observationCount,
      'values, mask and output must hold cellCount times chunkSliceCount entries'
    );
    validateRequirement(
      id,
      props.ring.length >= props.cellCount * props.windowLength &&
        props.seenCount.length >= props.cellCount,
      'ring must hold cellCount times windowLength entries and seenCount one entry per cell'
    );
    validateRequirement(
      id,
      props.parameters.length >= GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH,
      `parameters must hold ${GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH} entries`
    );
    this.chunkPlan = planRasterTimeSeriesChunks(this.id, props.cellCount, {
      values: {view: props.values, entriesPerCell: props.chunkSliceCount},
      mask: {view: props.mask, entriesPerCell: props.chunkSliceCount},
      ring: {view: props.ring, entriesPerCell: props.windowLength},
      seenCount: {view: props.seenCount, entriesPerCell: 1},
      output: {view: props.output, entriesPerCell: props.chunkSliceCount}
    });
    this.estimatedCommandNodeCount = this.chunkPlan.estimatedCommandNodeCount;
    validateRasterViews(this.id, this.getInputs(), this.getOutputs(), this.chunkPlan);
  }

  private getInputs(): RasterScalarView[] {
    const {values, mask, parameters} = this.props;
    return [values, parameters, ...(mask ? [mask] : [])];
  }

  private getOutputs(): RasterScalarView[] {
    return [this.props.ring, this.props.seenCount, this.props.output];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {chunkSliceCount: chunk, windowLength: window, values, mask, parameters} = props;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    return createRasterTimeSeriesNodes<Parameters>(graph, this.chunkPlan, {
      id,
      operation: 'GPURasterRollingMean',
      invocationCount: props.cellCount,
      workgroupSize: 64,
      bindings: [
        rasterBinding('values', values),
        rasterBinding('parameters', parameters),
        ...(mask ? [rasterBinding('mask', mask)] : []),
        rasterBinding('ring', props.ring, true),
        rasterBinding('seenCount', props.seenCount, true),
        rasterBinding('rolling', props.output, true)
      ],
      declarations: RASTER_FINITE_WGSL,
      body: `var seen = seenCount[seenCountOffset + index];
if (parameters[parametersOffset] != 0.0) { seen = 0u; }
let activeSliceCount = u32(clamp(parameters[parametersOffset + 1u], 0.0, ${chunk}.0));
for (var slice = 0u; slice < ${chunk}u; slice++) {
  if (slice >= activeSliceCount) { rolling[rollingOffset + index * ${chunk}u + slice] = missing(); continue; }
  let source = index * ${chunk}u + slice;
  var value = values[valuesOffset + source];
  if (!finite(value) ${mask ? '|| mask[maskOffset + source] == 0u' : ''}) { value = missing(); }
  ring[ringOffset + index * ${window}u + seen % ${window}u] = value;
  seen++;
  var mean = missing();
  if (seen >= ${window}u) {
    var total = 0.0;
    var complete = true;
    for (var entry = 0u; entry < ${window}u; entry++) {
      let sample = ring[ringOffset + index * ${window}u + entry];
      if (!finite(sample)) { complete = false; break; }
      total += sample;
    }
    if (complete) { mean = total / ${window}.0; }
  }
  rolling[rollingOffset + index * ${chunk}u + slice] = mean;
}
seenCount[seenCountOffset + index] = seen;`
    });
  }
}
