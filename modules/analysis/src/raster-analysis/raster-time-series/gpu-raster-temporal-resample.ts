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
  getGPURasterDateAxisBinding,
  getGPURasterDateAxisDeclarations,
  resolveGPURasterIrregularDateAxis,
  validateGPURasterDateAxisGraph,
  type GPURasterIrregularDateAxis
} from './raster-irregular-date-axis';

/** Source slices: a per-cell merge scans them in order. */
export const MAXIMUM_RESAMPLE_SLICE_COUNT = 256;
/** Output grid points per cell: a per-cell merge scans them in order. */
export const MAXIMUM_RESAMPLE_GRID_COUNT = 4096;
/** Float32 elements of the per-frame parameters: grid start, grid step, maximum gap. */
export const GPU_RASTER_TEMPORAL_RESAMPLE_PARAMETER_LENGTH = 3;

/** Per-frame settings of {@link GPURasterTemporalResample}, all in centred fractional years. */
export type GPURasterTemporalResampleSettings = {
  gridStart: number;
  gridStep: number;
  /** Largest allowed span between the bracketing observations; defaults to no limit. */
  maximumGap?: number;
};

/** Packs {@link GPURasterTemporalResampleSettings}: `[gridStart, gridStep, maximumGap]`. */
export function getGPURasterTemporalResampleParameterValues(
  settings: GPURasterTemporalResampleSettings,
  target: Float32Array = new Float32Array(GPU_RASTER_TEMPORAL_RESAMPLE_PARAMETER_LENGTH)
): Float32Array {
  validateFiniteNumber(
    'getGPURasterTemporalResampleParameterValues',
    'gridStep',
    settings.gridStep,
    0
  );
  target[0] = settings.gridStart;
  target[1] = settings.gridStep;
  target[2] = settings.maximumGap ?? 1e30;
  return target;
}

/** Properties of {@link GPURasterTemporalResample}. */
export type GPURasterTemporalResampleProps = {
  id?: string;
  cellCount: number;
  sliceCount: number;
  /** Output grid points per cell, in `[1, 4096]`. */
  gridCount: number;
  /** Cell-major `cell * sliceCount + slice`; nonfinite samples are missing. */
  values: GraphDataView<'float32'>;
  /** Per-observation QA; zero rejects the sample. */
  validity?: GraphDataView<'uint32'>;
  /** Float32, see {@link getGPURasterTemporalResampleParameterValues}. */
  parameters: GraphDataView<'float32'>;
  dateAxis: GPURasterIrregularDateAxis;
  output: {
    /** `cell * gridCount + point`; NaN where status is nonzero. */
    values: GraphDataView<'float32'>;
    /** Per grid point: 0 interpolated, 1 outside the observed range, 2 gap larger than `maximumGap`. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Fills a regular time grid from irregular observations by linear interpolation between the
 * bracketing valid samples, in O(slices + grid) per cell with a merge in time. Grid dates use the
 * same centred fractional years as the date axis (`getGPURasterCentredFractionalYears`). A gap
 * filled by an earlier smoother reaches this node as a finite value, so a Whittaker or HANTS
 * output feeds a regular-time grid directly; nothing is extrapolated.
 *
 * Per-frame: `parameters`. Compile-time: `cellCount`, `sliceCount`, `gridCount`, and which optional views exist.
 */
export class GPURasterTemporalResample implements GPUCommandNodeProducer {
  readonly id: string;
  private readonly ticksPerYear: number;

  constructor(readonly props: GPURasterTemporalResampleProps) {
    const id = props.id ?? 'raster-temporal-resample';
    this.id = id;
    validateCapacity(id, 'cellCount', props.cellCount);
    validateCapacity(id, 'sliceCount', props.sliceCount, 2, MAXIMUM_RESAMPLE_SLICE_COUNT);
    validateCapacity(id, 'gridCount', props.gridCount, 1, MAXIMUM_RESAMPLE_GRID_COUNT);
    validateCapacity(id, 'cellCount times gridCount', props.cellCount * props.gridCount);
    const observationCount = props.cellCount * props.sliceCount;
    validateCapacity(id, 'cellCount times sliceCount', observationCount);
    validateRequirement(
      id,
      props.values.length >= observationCount &&
        (!props.validity || props.validity.length >= observationCount),
      'values and validity must hold cellCount times sliceCount entries'
    );
    validateRequirement(
      id,
      props.output.values.length >= props.cellCount * props.gridCount &&
        props.output.status.length >= props.cellCount * props.gridCount,
      'output.values and output.status must hold cellCount times gridCount entries'
    );
    validateRequirement(
      id,
      props.parameters.length >= GPU_RASTER_TEMPORAL_RESAMPLE_PARAMETER_LENGTH,
      `parameters must hold ${GPU_RASTER_TEMPORAL_RESAMPLE_PARAMETER_LENGTH} entries`
    );
    this.ticksPerYear = resolveGPURasterIrregularDateAxis(id, props.dateAxis, props.sliceCount);
    validateRasterViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterScalarView[] {
    const {values, validity, parameters} = this.props;
    return [values, parameters, ...(validity ? [validity] : [])];
  }

  private getOutputs(): RasterScalarView[] {
    return [this.props.output.values, this.props.output.status];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {sliceCount: slices, gridCount, values, validity, parameters, dateAxis, output} = props;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    validateGPURasterDateAxisGraph(id, graph, dateAxis);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterTemporalResample',
        invocationCount: props.cellCount,
        workgroupSize: 16,
        bindings: [
          rasterBinding('values', values),
          rasterBinding('parameters', parameters),
          ...(validity ? [rasterBinding('validity', validity)] : []),
          getGPURasterDateAxisBinding(dateAxis),
          rasterBinding('resampled', output.values, true),
          rasterBinding('resampleStatus', output.status, true)
        ],
        declarations: `${RASTER_FINITE_WGSL}
${getGPURasterDateAxisDeclarations(this.ticksPerYear, slices)}
fn isObserved(cell: u32, slice: u32) -> bool {
  let source = cell * ${slices}u + slice;
  return finite(values[valuesOffset + source]) ${validity ? '&& validity[validityOffset + source] != 0u' : ''};
}
fn advanceObserved(cell: u32, start: u32) -> u32 {
  var slice = start;
  while (slice < ${slices}u && !isObserved(cell, slice)) { slice++; }
  return slice;
}`,
        body: `let gridStart = parameters[parametersOffset];
let gridStep = parameters[parametersOffset + 1u];
let maximumGap = parameters[parametersOffset + 2u];
var left = ${slices}u;
var right = advanceObserved(index, 0u);
for (var point = 0u; point < ${gridCount}u; point++) {
  let gridYears = gridStart + f32(point) * gridStep;
  while (right < ${slices}u && rasterSliceYears(right) <= gridYears) {
    left = right;
    right = advanceObserved(index, right + 1u);
  }
  var value = missing();
  var code = 1u;
  if (left < ${slices}u && right < ${slices}u) {
    let leftYears = rasterSliceYears(left);
    let span = rasterYearsBetween(right, left);
    if (span > maximumGap) {
      code = 2u;
    } else {
      let fraction = (gridYears - leftYears) / span;
      value = mix(values[valuesOffset + index * ${slices}u + left], values[valuesOffset + index * ${slices}u + right], fraction);
      code = 0u;
    }
  } else if (left < ${slices}u && rasterSliceYears(left) == gridYears) {
    value = values[valuesOffset + index * ${slices}u + left];
    code = 0u;
  }
  resampled[resampledOffset + index * ${gridCount}u + point] = value;
  resampleStatus[resampleStatusOffset + index * ${gridCount}u + point] = code;
}`
      })
    ];
  }
}
