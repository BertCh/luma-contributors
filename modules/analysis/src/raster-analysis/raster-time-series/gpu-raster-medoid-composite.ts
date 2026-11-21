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
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
export {
  prepareRasterStackInput,
  type PrepareRasterStackInputProps,
  type PreparedRasterStackInput,
  type RasterStackSourceLayout
} from './raster-stack-input';
import {RASTER_FINITE_WGSL} from '../remote-sensing/raster-kernel-utils';

/**
 * Slices per cell: one invocation compares every pair of valid observations, `O(slices^2 * bands)`,
 * and holds the mask of the slices in two 32-bit words.
 */
export const MAXIMUM_MEDOID_COMPOSITE_SLICE_COUNT = 64;
/** Bands per observation: the distance loop runs over the bands inside the pair loop. */
export const MAXIMUM_MEDOID_COMPOSITE_BAND_COUNT = 16;
/** Slice index published for a cell without a valid observation. */
export const GPU_RASTER_MEDOID_NO_SLICE = 0xffffffff;

/** Properties of {@link GPURasterMedoidComposite}. */
export type GPURasterMedoidCompositeProps<Rows extends RasterScalarView = GraphDataView> = {
  id?: string;
  cellCount: number;
  /** Slices in `[1, 64]`. */
  sliceCount: number;
  /** Bands per observation in `[1, 16]`. */
  bandCount: number;
  /**
   * Observations laid out `(cell * sliceCount + slice) * bandCount + band`. An observation with any
   * non-finite band is missing as a whole (hdmedians `nanmedoid`).
   */
  values: RasterTimeSeriesView<'float32', Rows>;
  /** Per-observation QA `cell * sliceCount + slice`; zero rejects the observation. */
  mask?: RasterTimeSeriesView<'uint32', Rows>;
  output: {
    /** The medoid observation `cell * bandCount + band`; NaN when status is nonzero. */
    composite: RasterTimeSeriesView<'float32', Rows>;
    /** Slice the medoid came from (provenance), `0xffffffff` when status is nonzero. */
    sliceIndex: RasterTimeSeriesView<'uint32', Rows>;
    /** Per cell: 0 composite, 1 no valid observation. */
    status: RasterTimeSeriesView<'uint32', Rows>;
  };
};

/**
 * Medoid composite per cell: the valid observation with the smallest sum of Euclidean distances (over
 * all bands) to the other valid observations, as hdmedians `nanmedoid`. The result is a real
 * acquisition, so its slice is the provenance of every band. Ties go to the earliest slice.
 *
 * Multiband distances use a common per-cell power-of-two scale chosen from differences before
 * squaring, independent of constant-band magnitudes. This preserves
 * comparisons for tiny (e.g. 1e-25) and large (e.g. 1e20) finite spectra. With at most one varying band, CPU and GPU use the
 * closed median interval and select its earliest acquisition exactly. Multiband distances and sums
 * are float32; observations whose sums differ by less than float32 rounding can swap places
 * against a binary64 reference.
 *
 * The GPU route accepts at most 64 slices and 16 bands per cell.
 *
 * Per-frame: the contents of the views. Compile-time: `cellCount`, `sliceCount`, `bandCount` and
 * whether `mask` exists.
 *
 * Cell-major views accept matching chunks of complete cell records; shared tables stay packed.
 */
export class GPURasterMedoidComposite<Rows extends RasterScalarView = GraphDataView>
  implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly props: GPURasterMedoidCompositeProps<Rows>;
  readonly chunkPlan: RasterTimeSeriesChunkPlan;
  readonly estimatedCommandNodeCount: number;
  constructor(props: GPURasterMedoidCompositeProps<Rows> & {values: Rows}) {
    this.props = props;
    const id = props.id ?? 'raster-medoid-composite';
    this.id = id;
    validateGraphPorts(id, props, {
      values: {required: true, viewType: 'either', formats: ['float32']},
      mask: {required: false, viewType: 'either', formats: ['uint32']},
      'output.composite': {required: true, viewType: 'either', formats: ['float32']},
      'output.sliceIndex': {required: true, viewType: 'either', formats: ['uint32']},
      'output.status': {required: true, viewType: 'either', formats: ['uint32']}
    });
    // Capture constructor options/maps while preserving graph resource identities.
    props = {...props, output: {...props.output}};
    this.props = props;
    validateCapacity(id, 'cellCount', props.cellCount);
    validateCapacity(id, 'sliceCount', props.sliceCount, 1, MAXIMUM_MEDOID_COMPOSITE_SLICE_COUNT);
    validateCapacity(id, 'bandCount', props.bandCount, 1, MAXIMUM_MEDOID_COMPOSITE_BAND_COUNT);
    const observationCount = props.cellCount * props.sliceCount;
    validateCapacity(id, 'cellCount times sliceCount', observationCount);
    validateRequirement(
      id,
      props.values.length >= observationCount * props.bandCount &&
        (!props.mask || props.mask.length >= observationCount),
      'values must hold cellCount times sliceCount times bandCount entries and mask cellCount times sliceCount'
    );
    validateRequirement(
      id,
      props.output.composite.length >= props.cellCount * props.bandCount &&
        props.output.sliceIndex.length >= props.cellCount &&
        props.output.status.length >= props.cellCount,
      'output.composite must hold cellCount times bandCount entries; sliceIndex and status one per cell'
    );
    this.chunkPlan = planRasterTimeSeriesChunks(this.id, props.cellCount, {
      values: {view: props.values, entriesPerCell: props.sliceCount * props.bandCount},
      mask: {view: props.mask, entriesPerCell: props.sliceCount},
      'output.composite': {view: props.output.composite, entriesPerCell: props.bandCount},
      'output.sliceIndex': {view: props.output.sliceIndex, entriesPerCell: 1},
      'output.status': {view: props.output.status, entriesPerCell: 1}
    });
    this.estimatedCommandNodeCount = this.chunkPlan.estimatedCommandNodeCount;
    validateRasterViews(this.id, this.getInputs(), this.getOutputs(), this.chunkPlan);
  }

  private getInputs(): RasterScalarView[] {
    return [this.props.values, ...(this.props.mask ? [this.props.mask] : [])];
  }

  private getOutputs(): RasterScalarView[] {
    const {composite, sliceIndex, status} = this.props.output;
    return [composite, sliceIndex, status];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {sliceCount: slices, bandCount: bands, values, mask, output} = props;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    return createRasterTimeSeriesNodes<Parameters>(graph, this.chunkPlan, {
      id,
      operation: 'GPURasterMedoidComposite',
      invocationCount: props.cellCount,
      workgroupSize: 64,
      bindings: [
        rasterBinding('values', values),
        ...(mask ? [rasterBinding('mask', mask)] : []),
        rasterBinding('composite', output.composite, true),
        rasterBinding('sliceIndex', output.sliceIndex, true),
        rasterBinding('status', output.status, true)
      ],
      declarations: `${RASTER_FINITE_WGSL}
fn isObservationValid(cell: u32, slice: u32) -> bool {
  let source = cell * ${slices}u + slice;
  ${mask ? `if (mask[maskOffset + source] == 0u) { return false; }` : ''}
  for (var band = 0u; band < ${bands}u; band++) {
    if (!finite(values[valuesOffset + source * ${bands}u + band])) { return false; }
  }
  return true;
}`,
      body: `var validMask = array<u32, 2>(0u, 0u);
for (var slice = 0u; slice < ${slices}u; slice++) {
  if (isObservationValid(index, slice)) { validMask[slice / 32u] |= 1u << (slice % 32u); }
}
var bestSlice = 0xffffffffu;
var validCount = 0u;
var firstSlice = 0u;
for (var slice = 0u; slice < ${slices}u; slice++) {
  if (((validMask[slice / 32u] >> (slice % 32u)) & 1u) == 0u) { continue; }
  if (validCount == 0u) { firstSlice = slice; }
  validCount++;
}
var varyingBands: array<u32, ${bands}>;
var varyingBandCount = 0u;
var scalarBand = 0u;
var magnitudeBits = 0u;
if (validCount > 0u) {
  for (var band = 0u; band < ${bands}u; band++) {
    let firstValue = values[valuesOffset + (index * ${slices}u + firstSlice) * ${bands}u + band];
    for (var slice = 0u; slice < ${slices}u; slice++) {
      if (((validMask[slice / 32u] >> (slice % 32u)) & 1u) == 0u) { continue; }
      let value = values[valuesOffset + (index * ${slices}u + slice) * ${bands}u + band];
      if (value != firstValue) { varyingBands[band] = 1u; }
      // Infinity from subtracting opposite extreme finite values has exponent 255, which
      // requests one extra scale bit. No infinity literal or reciprocal is needed.
      magnitudeBits = max(magnitudeBits, bitcast<u32>(value - firstValue) & 0x7fffffffu);
    }
    if (varyingBands[band] != 0u) { varyingBandCount++; scalarBand = band; }
  }
}
if (varyingBandCount <= 1u) {
  // Constant coordinates contribute zero distance. Use scalar ranks even in a multiband stack.
  for (var candidate = 0u; candidate < ${slices}u; candidate++) {
    if (((validMask[candidate / 32u] >> (candidate % 32u)) & 1u) == 0u) { continue; }
    let value = values[valuesOffset + (index * ${slices}u + candidate) * ${bands}u + scalarBand];
    var below = 0u;
    var above = 0u;
    for (var other = 0u; other < ${slices}u; other++) {
      if (((validMask[other / 32u] >> (other % 32u)) & 1u) == 0u) { continue; }
      let otherValue = values[valuesOffset + (index * ${slices}u + other) * ${bands}u + scalarBand];
      below += select(0u, 1u, otherValue < value);
      above += select(0u, 1u, otherValue > value);
    }
    if (below <= validCount / 2u && above <= validCount / 2u) { bestSlice = candidate; break; }
  }
} else {
  // Choose a common scale from differences, independent of absolute or constant-band offsets.
  let scaleExponent = 127 - i32((magnitudeBits >> 23u) & 255u);
  let firstExponent = clamp(scaleExponent, -126, 126);
  let firstScale = bitcast<f32>(u32(firstExponent + 127) << 23u);
  let secondScale = bitcast<f32>(u32(scaleExponent - firstExponent + 127) << 23u);
  var bestSum = 0.0;
  for (var candidate = 0u; candidate < ${slices}u; candidate++) {
    if (((validMask[candidate / 32u] >> (candidate % 32u)) & 1u) == 0u) { continue; }
    var sum = 0.0;
    for (var other = 0u; other < ${slices}u; other++) {
      if (other == candidate || ((validMask[other / 32u] >> (other % 32u)) & 1u) == 0u) { continue; }
      var squared = 0.0;
      for (var band = 0u; band < ${bands}u; band++) {
        if (varyingBands[band] == 0u) { continue; }
        let firstValue = values[valuesOffset + (index * ${slices}u + candidate) * ${bands}u + band];
        let secondValue = values[valuesOffset + (index * ${slices}u + other) * ${bands}u + band];
        var difference = firstValue - secondValue;
        if (finite(difference)) { difference = (difference * firstScale) * secondScale; }
        else { difference = (firstValue * firstScale) * secondScale - (secondValue * firstScale) * secondScale; }
        squared += difference * difference;
      }
      sum += sqrt(squared);
    }
    if (bestSlice == 0xffffffffu || sum < bestSum) { bestSlice = candidate; bestSum = sum; }
  }
}
sliceIndex[sliceIndexOffset + index] = bestSlice;
status[statusOffset + index] = select(0u, 1u, bestSlice == 0xffffffffu);
for (var band = 0u; band < ${bands}u; band++) {
  var component = missing();
  if (bestSlice != 0xffffffffu) { component = values[valuesOffset + (index * ${slices}u + bestSlice) * ${bands}u + band]; }
  composite[compositeOffset + index * ${bands}u + band] = component;
}`
    });
  }
}
