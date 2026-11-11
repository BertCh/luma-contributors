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
export type GPURasterMedoidCompositeProps = {
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
  values: GraphDataView<'float32'>;
  /** Per-observation QA `cell * sliceCount + slice`; zero rejects the observation. */
  mask?: GraphDataView<'uint32'>;
  output: {
    /** The medoid observation `cell * bandCount + band`; NaN when status is nonzero. */
    composite: GraphDataView<'float32'>;
    /** Slice the medoid came from (provenance), `0xffffffff` when status is nonzero. */
    sliceIndex: GraphDataView<'uint32'>;
    /** Per cell: 0 composite, 1 no valid observation. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Medoid composite per cell: the valid observation with the smallest sum of Euclidean distances (over
 * all bands) to the other valid observations, as hdmedians `nanmedoid`. The result is a real
 * acquisition, so its slice is the provenance of every band. Ties go to the earliest slice.
 *
 * Distances and sums are float32; two observations whose sums differ by less than float32 rounding
 * can swap places against a binary64 reference.
 *
 * Per-frame: the contents of the views. Compile-time: `cellCount`, `sliceCount`, `bandCount` and
 * whether `mask` exists.
 */
export class GPURasterMedoidComposite implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPURasterMedoidCompositeProps) {
    const id = props.id ?? 'raster-medoid-composite';
    this.id = id;
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
    validateRasterViews(id, this.getInputs(), this.getOutputs());
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
    return [
      createWGSLKernelNode<Parameters>(graph, {
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
var bestSum = 0.0;
for (var candidate = 0u; candidate < ${slices}u; candidate++) {
  if (((validMask[candidate / 32u] >> (candidate % 32u)) & 1u) == 0u) { continue; }
  var sum = 0.0;
  for (var other = 0u; other < ${slices}u; other++) {
    if (other == candidate || ((validMask[other / 32u] >> (other % 32u)) & 1u) == 0u) { continue; }
    var squared = 0.0;
    for (var band = 0u; band < ${bands}u; band++) {
      let difference = values[valuesOffset + (index * ${slices}u + candidate) * ${bands}u + band]
        - values[valuesOffset + (index * ${slices}u + other) * ${bands}u + band];
      squared += difference * difference;
    }
    sum += sqrt(squared);
  }
  if (bestSlice == 0xffffffffu || sum < bestSum) { bestSlice = candidate; bestSum = sum; }
}
sliceIndex[sliceIndexOffset + index] = bestSlice;
status[statusOffset + index] = select(0u, 1u, bestSlice == 0xffffffffu);
for (var band = 0u; band < ${bands}u; band++) {
  var component = missing();
  if (bestSlice != 0xffffffffu) { component = values[valuesOffset + (index * ${slices}u + bestSlice) * ${bands}u + band]; }
  composite[compositeOffset + index * ${bands}u + band] = component;
}`
      })
    ];
  }
}
