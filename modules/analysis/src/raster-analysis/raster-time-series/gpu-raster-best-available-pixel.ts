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

/** Slices per cell: one invocation scores every slice in turn. */
export const MAXIMUM_BEST_AVAILABLE_PIXEL_SLICE_COUNT = 256;
/** Bands per observation of the optional composite. */
export const MAXIMUM_BEST_AVAILABLE_PIXEL_BAND_COUNT = 16;
/** Float32 elements of the per-frame parameters. */
export const GPU_RASTER_BEST_AVAILABLE_PIXEL_PARAMETER_LENGTH = 9;
/** Slice index published for a cell without a valid observation. */
export const GPU_RASTER_BEST_AVAILABLE_PIXEL_NO_SLICE = 0xffffffff;
/** Days of the Gregorian mean year, the period of the day-of-year distance. */
const DAYS_PER_YEAR = 365.2425;

/** Per-frame settings of {@link GPURasterBestAvailablePixel}. */
export type GPURasterBestAvailablePixelSettings = {
  /** Target day of year in `[0, 366]`. */
  targetDayOfYear: number;
  /** Standard deviation of the Gaussian day-of-year score, in days. */
  dayOfYearSigma: number;
  /** Weight of the day-of-year score; Griffiths et al. 2013 example `0.5`. */
  dayOfYearWeight: number;
  /** Target year in the unit of the `sliceAttributes` year column. */
  targetYear: number;
  /** Year distance at which the linear year score reaches zero. */
  maximumYearDistance: number;
  /** Weight of the year score; example `0.2`. */
  yearWeight: number;
  /** Cloud distance at or below which the cloud score is zero (pixels), example `10`. */
  minimumCloudDistance: number;
  /** Cloud distance at or above which the cloud score is one (pixels), example `100`. */
  maximumCloudDistance: number;
  /** Weight of the cloud-distance score; example `0.3`. */
  cloudWeight: number;
};

/**
 * Packs {@link GPURasterBestAvailablePixelSettings}: `[targetDayOfYear, dayOfYearSigma,
 * dayOfYearWeight, targetYear, maximumYearDistance, yearWeight, minimumCloudDistance,
 * maximumCloudDistance, cloudWeight]`.
 */
export function getGPURasterBestAvailablePixelParameterValues(
  settings: GPURasterBestAvailablePixelSettings,
  target: Float32Array = new Float32Array(GPU_RASTER_BEST_AVAILABLE_PIXEL_PARAMETER_LENGTH)
): Float32Array {
  const name = 'getGPURasterBestAvailablePixelParameterValues';
  validateFiniteNumber(name, 'targetDayOfYear', settings.targetDayOfYear, 0, 366);
  validateFiniteNumber(name, 'dayOfYearSigma', settings.dayOfYearSigma, Number.MIN_VALUE);
  validateFiniteNumber(name, 'dayOfYearWeight', settings.dayOfYearWeight, 0);
  validateFiniteNumber(name, 'targetYear', settings.targetYear);
  validateFiniteNumber(name, 'maximumYearDistance', settings.maximumYearDistance, Number.MIN_VALUE);
  validateFiniteNumber(name, 'yearWeight', settings.yearWeight, 0);
  validateFiniteNumber(name, 'minimumCloudDistance', settings.minimumCloudDistance, 0);
  validateFiniteNumber(
    name,
    'maximumCloudDistance',
    settings.maximumCloudDistance,
    settings.minimumCloudDistance + Number.MIN_VALUE
  );
  validateFiniteNumber(name, 'cloudWeight', settings.cloudWeight, 0);
  target.set([
    settings.targetDayOfYear,
    settings.dayOfYearSigma,
    settings.dayOfYearWeight,
    settings.targetYear,
    settings.maximumYearDistance,
    settings.yearWeight,
    settings.minimumCloudDistance,
    settings.maximumCloudDistance,
    settings.cloudWeight
  ]);
  return target;
}

/** Properties of {@link GPURasterBestAvailablePixel}. */
export type GPURasterBestAvailablePixelProps = {
  id?: string;
  cellCount: number;
  /** Slices in `[1, 256]`. */
  sliceCount: number;
  /**
   * Bands of the optional composite in `[0, 16]`; 0 selects the best slice only. With bands,
   * `values` and `output.composite` are required and an observation with a non-finite band is
   * missing.
   */
  bandCount: number;
  /** Observations `(cell * sliceCount + slice) * bandCount + band`; required when `bandCount > 0`. */
  values?: GraphDataView<'float32'>;
  /** Per-observation QA `cell * sliceCount + slice`; zero (for example a cloud or shadow) rejects the observation. */
  validity?: GraphDataView<'uint32'>;
  /**
   * Distance in pixels from each observation to the nearest cloud or shadow pixel of its slice,
   * `cell * sliceCount + slice`, for example from `GPUDistanceField` run on each slice's mask.
   * Positive infinity means no cloud in reach. Without it the cloud score is zero.
   */
  cloudDistance?: GraphDataView<'float32'>;
  /** Per-slice scene attributes: `[dayOfYear, year]` for slice `s` at `2 * s` and `2 * s + 1`. */
  sliceAttributes: GraphDataView<'float32'>;
  /** Float32, see {@link getGPURasterBestAvailablePixelParameterValues}. */
  parameters: GraphDataView<'float32'>;
  output: {
    /** Winning slice per cell, `0xffffffff` when no valid observation exists. */
    bestSlice: GraphDataView<'uint32'>;
    /** Winning score per cell, NaN when `bestSlice` is `0xffffffff`. */
    bestScore: GraphDataView<'float32'>;
    /** Optional composite `cell * bandCount + band` of the winning observation; NaN without one. */
    composite?: GraphDataView<'float32'>;
  };
};

/**
 * Best-available-pixel compositing (Griffiths et al. 2013): every valid observation gets
 * `score = wDay * exp(-0.5 (dDay / sigma)^2) + wYear * max(0, 1 - |year - targetYear| / maximumYearDistance)
 * + wCloud * clamp((cloudDistance - minimum) / (maximum - minimum), 0, 1)`, where `dDay` is the circular
 * distance in days to the target day of year, and the cell takes the highest-scoring slice (the
 * earliest on ties). A missing cloud distance scores zero. The output is a real acquisition.
 *
 * Per-frame: `parameters`, `sliceAttributes` and the contents of the views. Compile-time:
 * `cellCount`, `sliceCount`, `bandCount` and which optional views exist.
 */
export class GPURasterBestAvailablePixel implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPURasterBestAvailablePixelProps) {
    const id = props.id ?? 'raster-best-available-pixel';
    this.id = id;
    validateCapacity(id, 'cellCount', props.cellCount);
    validateCapacity(
      id,
      'sliceCount',
      props.sliceCount,
      1,
      MAXIMUM_BEST_AVAILABLE_PIXEL_SLICE_COUNT
    );
    validateCapacity(id, 'bandCount', props.bandCount, 0, MAXIMUM_BEST_AVAILABLE_PIXEL_BAND_COUNT);
    const observationCount = props.cellCount * props.sliceCount;
    validateCapacity(id, 'cellCount times sliceCount', observationCount);
    validateRequirement(
      id,
      (props.bandCount === 0) === (props.values === undefined) &&
        (props.bandCount === 0) === (props.output.composite === undefined),
      'values and output.composite are required exactly when bandCount is above zero'
    );
    validateRequirement(
      id,
      (!props.values || props.values.length >= observationCount * props.bandCount) &&
        (!props.validity || props.validity.length >= observationCount) &&
        (!props.cloudDistance || props.cloudDistance.length >= observationCount),
      'values must hold cellCount times sliceCount times bandCount entries; validity and cloudDistance cellCount times sliceCount'
    );
    validateRequirement(
      id,
      props.sliceAttributes.length >= 2 * props.sliceCount,
      'sliceAttributes must hold two entries per slice'
    );
    validateRequirement(
      id,
      props.parameters.length >= GPU_RASTER_BEST_AVAILABLE_PIXEL_PARAMETER_LENGTH,
      `parameters must hold ${GPU_RASTER_BEST_AVAILABLE_PIXEL_PARAMETER_LENGTH} entries`
    );
    validateRequirement(
      id,
      props.output.bestSlice.length >= props.cellCount &&
        props.output.bestScore.length >= props.cellCount &&
        (!props.output.composite ||
          props.output.composite.length >= props.cellCount * props.bandCount),
      'output.bestSlice and bestScore must hold one entry per cell and composite cellCount times bandCount'
    );
    validateRasterViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterScalarView[] {
    const {values, validity, cloudDistance, sliceAttributes, parameters} = this.props;
    return [
      sliceAttributes,
      parameters,
      ...(values ? [values] : []),
      ...(validity ? [validity] : []),
      ...(cloudDistance ? [cloudDistance] : [])
    ];
  }

  private getOutputs(): RasterScalarView[] {
    const {bestSlice, bestScore, composite} = this.props.output;
    return [bestSlice, bestScore, ...(composite ? [composite] : [])];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {
      sliceCount: slices,
      bandCount: bands,
      values,
      validity,
      cloudDistance,
      sliceAttributes,
      parameters,
      output
    } = props;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterBestAvailablePixel',
        invocationCount: props.cellCount,
        workgroupSize: 64,
        bindings: [
          ...(values ? [rasterBinding('values', values)] : []),
          ...(validity ? [rasterBinding('validity', validity)] : []),
          ...(cloudDistance ? [rasterBinding('cloudDistance', cloudDistance)] : []),
          rasterBinding('sliceAttributes', sliceAttributes),
          rasterBinding('parameters', parameters),
          rasterBinding('bestSlice', output.bestSlice, true),
          rasterBinding('bestScore', output.bestScore, true),
          ...(output.composite ? [rasterBinding('composite', output.composite, true)] : [])
        ],
        declarations: `${RASTER_FINITE_WGSL}
fn isObservationValid(cell: u32, slice: u32) -> bool {
  let source = cell * ${slices}u + slice;
  ${validity ? `if (validity[validityOffset + source] == 0u) { return false; }` : ''}
  ${
    values
      ? `for (var band = 0u; band < ${bands}u; band++) {
    if (!finite(values[valuesOffset + source * ${bands}u + band])) { return false; }
  }`
      : ''
  }
  return true;
}`,
        body: `let targetDay = parameters[parametersOffset];
let daySigma = parameters[parametersOffset + 1u];
let dayWeight = parameters[parametersOffset + 2u];
let targetYear = parameters[parametersOffset + 3u];
let maximumYearDistance = parameters[parametersOffset + 4u];
let yearWeight = parameters[parametersOffset + 5u];
let minimumCloud = parameters[parametersOffset + 6u];
let maximumCloud = parameters[parametersOffset + 7u];
let cloudWeight = parameters[parametersOffset + 8u];
var winner = 0xffffffffu;
var winnerScore = 0.0;
for (var slice = 0u; slice < ${slices}u; slice++) {
  if (!isObservationValid(index, slice)) { continue; }
  let day = sliceAttributes[sliceAttributesOffset + 2u * slice];
  let year = sliceAttributes[sliceAttributesOffset + 2u * slice + 1u];
  var dayDistance = abs(day - targetDay);
  dayDistance = min(dayDistance, ${DAYS_PER_YEAR} - dayDistance);
  let dayScore = exp(-0.5 * (dayDistance / daySigma) * (dayDistance / daySigma));
  let yearScore = max(0.0, 1.0 - abs(year - targetYear) / maximumYearDistance);
  var cloudScore = 0.0;
  ${
    cloudDistance
      ? `let distance = cloudDistance[cloudDistanceOffset + index * ${slices}u + slice];
  if (distance >= maximumCloud) { cloudScore = 1.0; }
  else if (distance > minimumCloud) { cloudScore = (distance - minimumCloud) / (maximumCloud - minimumCloud); }`
      : ''
  }
  let score = dayWeight * dayScore + yearWeight * yearScore + cloudWeight * cloudScore;
  if (winner == 0xffffffffu || score > winnerScore) { winner = slice; winnerScore = score; }
}
bestSlice[bestSliceOffset + index] = winner;
bestScore[bestScoreOffset + index] = select(winnerScore, missing(), winner == 0xffffffffu);
${
  output.composite
    ? `for (var band = 0u; band < ${bands}u; band++) {
  var component = missing();
  if (winner != 0xffffffffu) { component = values[valuesOffset + (index * ${slices}u + winner) * ${bands}u + band]; }
  composite[compositeOffset + index * ${bands}u + band] = component;
}`
    : ''
}`
      })
    ];
  }
}
