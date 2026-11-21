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
  type GPUPackedReadbackValues,
  validateCapacity,
  validateFiniteNumber,
  validateGraphPorts,
  validateOption,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateRasterStreamFrame,
  validateRasterStreamParameterTarget,
  getRasterStreamStatusWord
} from './raster-stream-parameters';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';
import {GET_NORMAL_QUANTILE_WGSL} from '../../analysis-core/kernels/normal-quantile-wgsl';
import {RASTER_FINITE_WGSL} from '../remote-sensing/raster-kernel-utils';
import {
  RASTER_REGULARIZED_GAMMA_TAILS_WGSL,
  RASTER_CLIMATE_ZERO_ATOM_WGSL,
  RASTER_CLIMATE_SHIFTED_RATIO_WGSL
} from './raster-climate-distribution-wgsl';
import {MAXIMUM_RASTER_STREAM_SLOT_COUNT} from './gpu-raster-streaming-accumulator';
import {
  resolveGPURasterStandardizedIndexZeroAtom,
  type GPURasterStandardizedIndexDistribution
} from './gpu-raster-standardized-index-calibration';

/** Independent sample invocations need no per-cell scratch; activeSliceCount is exact in float32 through 2^24. */
export const MAXIMUM_STANDARDIZED_INDEX_CHUNK_SLICE_COUNT = 2 ** 24;

/** Standardized index magnitude limit, the float64 bound xclim applies (`+-8.21`). */
export const GPU_RASTER_STANDARDIZED_INDEX_LIMIT = 8.21;
/** Float32 elements of the per-frame parameters: active slice count, location. */
export const GPU_RASTER_STANDARDIZED_INDEX_PARAMETER_LENGTH = 2;

/** Packs the active slice count and the fixed location (`floc`). */
export function getGPURasterStandardizedIndexParameterValues(
  settings: {activeSliceCount?: number; location?: number},
  chunkSliceCount: number,
  target: Float32Array = new Float32Array(GPU_RASTER_STANDARDIZED_INDEX_PARAMETER_LENGTH)
): Float32Array {
  const id = 'getGPURasterStandardizedIndexParameterValues';
  validateRasterStreamParameterTarget(id, target, GPU_RASTER_STANDARDIZED_INDEX_PARAMETER_LENGTH);
  const activeSliceCount = settings.activeSliceCount ?? chunkSliceCount;
  validateRasterStreamFrame(
    id,
    chunkSliceCount,
    activeSliceCount,
    MAXIMUM_STANDARDIZED_INDEX_CHUNK_SLICE_COUNT
  );
  validateFiniteNumber(id, 'location', settings.location ?? 0);
  target.set([activeSliceCount, settings.location ?? 0]);
  return target;
}

/** Named interpretation of a standardized-index status word. */
export function decodeGPURasterStandardizedIndexStatus(input: GPUPackedReadbackValues): {
  complete: boolean;
  invalidInput: boolean;
  calibrationUnavailable: boolean;
  iterationLimit: boolean;
} {
  const code = getRasterStreamStatusWord('decodeGPURasterStandardizedIndexStatus', input, 3);
  return {
    complete: code === 0,
    invalidInput: code === 1,
    calibrationUnavailable: code === 2,
    iterationLimit: code === 3
  };
}

/** Per-observation results, indexed cell * chunkSliceCount + slice. */
export type GPURasterStandardizedIndexOutput<Rows extends RasterScalarView = GraphDataView> = {
  /** Standardized index in [-8.21, 8.21]; NaN where status is nonzero. */
  index: RasterTimeSeriesView<'float32', Rows>;
  /** 0 valid, 1 missing input/inactive slice, 2 calibration unusable, 3 evaluation iteration limit. */
  status: RasterTimeSeriesView<'uint32', Rows>;
};

/** Properties of {@link GPURasterStandardizedIndex}. */
export type GPURasterStandardizedIndexProps<Rows extends RasterScalarView = GraphDataView> = {
  id?: string;
  cellCount: number;
  /** Samples per cell, at most 2^24: one independent invocation per sample. */
  chunkSliceCount: number;
  slotCount?: number;
  distribution: GPURasterStandardizedIndexDistribution;
  /** Atom at original value zero; must match calibration. Defaults to false. */
  zeroAtom?: boolean;
  /** Chunk to evaluate, cell-major; nonfinite samples are missing. */
  values: RasterTimeSeriesView<'float32', Rows>;
  mask?: RasterTimeSeriesView<'uint32', Rows>;
  slotIndices?: GraphDataView<'uint32'>;
  /** `[shape, scale, zeroProbability]` per slot, from {@link GPURasterStandardizedIndexCalibration}. */
  fit: RasterTimeSeriesView<'float32', Rows>;
  /** Float32, see {@link getGPURasterStandardizedIndexParameterValues}. */
  parameters: GraphDataView<'float32'>;
  output: GPURasterStandardizedIndexOutput<Rows>;
};

/**
 * Standardized index of a chunk from a fit written by {@link GPURasterStandardizedIndexCalibration}
 * (SPI with `gamma`, SPEI with `fisk`, xclim's APP fit and zero handling). The probability of an
 * observation is evaluated through whichever tail is small, so `1 - p` is never formed near one and
 * the quantile keeps its precision on both sides. Results are clipped to xclim's `+-8.21`.
 *
 * Per-frame: `parameters` and the view contents. Compile-time: `cellCount`, `chunkSliceCount`,
 * `slotCount`, `distribution`, `zeroAtom`, and which optional views exist.
 *
 * Cell-major views accept matching chunks of complete cell records; shared tables stay packed.
 */
export class GPURasterStandardizedIndex<Rows extends RasterScalarView = GraphDataView>
  implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly props: GPURasterStandardizedIndexProps<Rows>;
  readonly chunkPlan: RasterTimeSeriesChunkPlan;
  readonly estimatedCommandNodeCount: number;
  readonly slotCount: number;
  constructor(props: GPURasterStandardizedIndexProps<Rows> & {values: Rows}) {
    const zeroAtom = resolveGPURasterStandardizedIndexZeroAtom(
      props.id ?? 'raster-standardized-index',
      props
    );
    this.props = {...props, zeroAtom, output: {...props.output}};
    const id = props.id ?? 'raster-standardized-index';
    this.id = id;
    validateGraphPorts(id, props, {
      values: {required: true, viewType: 'either', formats: ['float32']},
      mask: {required: false, viewType: 'either', formats: ['uint32']},
      slotIndices: {required: false, viewType: 'data', formats: ['uint32']},
      parameters: {required: true, viewType: 'data', formats: ['float32']},
      fit: {required: true, viewType: 'either', formats: ['float32']},
      'output.index': {required: true, viewType: 'either', formats: ['float32']},
      'output.status': {required: true, viewType: 'either', formats: ['uint32']}
    });
    const slotCount = props.slotCount ?? 1;
    this.slotCount = slotCount;
    validateCapacity(id, 'cellCount', props.cellCount);
    validateCapacity(
      id,
      'chunkSliceCount',
      props.chunkSliceCount,
      1,
      MAXIMUM_STANDARDIZED_INDEX_CHUNK_SLICE_COUNT
    );
    validateCapacity(id, 'slotCount', slotCount, 1, MAXIMUM_RASTER_STREAM_SLOT_COUNT);
    validateOption(id, 'distribution', props.distribution, ['gamma', 'fisk']);
    const observationCount = props.cellCount * props.chunkSliceCount;
    validateCapacity(id, 'cellCount times chunkSliceCount', observationCount);
    validateRequirement(
      id,
      props.values.length >= observationCount &&
        (!props.mask || props.mask.length >= observationCount) &&
        props.output.index.length >= observationCount &&
        props.output.status.length >= observationCount,
      'values, mask, output.index and output.status must hold cellCount times chunkSliceCount entries'
    );
    validateRequirement(
      id,
      props.fit.length >= props.cellCount * slotCount * 3,
      'fit must hold three entries per slot'
    );
    validateRequirement(
      id,
      slotCount === 1 || (!!props.slotIndices && props.slotIndices.length >= props.chunkSliceCount),
      'slotIndices must hold one entry per chunk slice when slotCount exceeds 1'
    );
    validateRequirement(
      id,
      props.parameters.length >= GPU_RASTER_STANDARDIZED_INDEX_PARAMETER_LENGTH,
      `parameters must hold ${GPU_RASTER_STANDARDIZED_INDEX_PARAMETER_LENGTH} entries`
    );
    this.chunkPlan = planRasterTimeSeriesChunks(this.id, props.cellCount, {
      values: {view: props.values, entriesPerCell: props.chunkSliceCount},
      mask: {view: props.mask, entriesPerCell: props.chunkSliceCount},
      fit: {view: props.fit, entriesPerCell: (props.slotCount ?? 1) * 3},
      'output.index': {view: props.output.index, entriesPerCell: props.chunkSliceCount},
      'output.status': {view: props.output.status, entriesPerCell: props.chunkSliceCount}
    });
    this.estimatedCommandNodeCount = this.chunkPlan.estimatedCommandNodeCount;
    validateRasterViews(this.id, this.getInputs(), this.getOutputs(), this.chunkPlan);
  }

  private getInputs(): RasterScalarView[] {
    const {values, mask, slotIndices, fit, parameters} = this.props;
    return [
      values,
      fit,
      parameters,
      ...(mask ? [mask] : []),
      ...((this.props.slotCount ?? 1) > 1 && slotIndices ? [slotIndices] : [])
    ];
  }

  private getOutputs(): RasterScalarView[] {
    return [this.props.output.index, this.props.output.status];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, slotCount} = this;
    const {chunkSliceCount: chunk, values, mask, slotIndices, fit, parameters, output} = props;
    const zeroAtom = props.zeroAtom ?? false;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    const tails =
      props.distribution === 'gamma'
        ? 'getConvergedRegularizedGammaTails(fitShape, getRasterClimateShiftedRatio(value, parameters[parametersOffset + 1u], fitScale))'
        : 'getFiskTails(fitShape, getRasterClimateShiftedRatio(value, parameters[parametersOffset + 1u], fitScale))';
    return createRasterTimeSeriesNodes<Parameters>(graph, this.chunkPlan, {
      id,
      operation: 'GPURasterStandardizedIndex',
      invocationCount: props.cellCount * chunk,
      workgroupSize: 64,
      bindings: [
        rasterBinding('values', values),
        rasterBinding('fit', fit),
        rasterBinding('parameters', parameters),
        ...(mask ? [rasterBinding('mask', mask)] : []),
        ...(slotCount > 1 && slotIndices ? [rasterBinding('slotIndices', slotIndices)] : []),
        rasterBinding('indexOut', output.index, true),
        rasterBinding('statusOut', output.status, true)
      ],
      declarations: `${RASTER_FINITE_WGSL}
${SCALED_QUOTIENT_WGSL}
${RASTER_CLIMATE_SHIFTED_RATIO_WGSL}
${zeroAtom ? RASTER_CLIMATE_ZERO_ATOM_WGSL : ''}
${props.distribution === 'gamma' ? RASTER_REGULARIZED_GAMMA_TAILS_WGSL : ''}
${GET_NORMAL_QUANTILE_WGSL}
fn getFiskTails(shape: f32, ratio: f32) -> vec2<f32> {
  if (!(ratio > 0.0)) { return vec2<f32>(0.0, 1.0); }
  let logistic = shape * log(ratio);
  return vec2<f32>(1.0 / (1.0 + exp(-logistic)), 1.0 / (1.0 + exp(logistic)));
}`,
      body: `let cell = index / ${chunk}u;
let slice = index % ${chunk}u;
let source = cell * ${chunk}u + slice;
var result = missing();
var code = 1u;
let activeSliceCount = u32(clamp(parameters[parametersOffset], 0.0, ${chunk}.0));
let value = values[valuesOffset + source];
if (slice < activeSliceCount && finite(value) ${mask ? '&& mask[maskOffset + source] != 0u' : ''}) {
  let slot = ${slotCount > 1 ? 'slotIndices[slotIndicesOffset + slice]' : '0u'};
  code = 2u;
  if (slot < ${slotCount}u) {
    let entry = cell * ${slotCount}u + slot;
    let fitShape = fit[fitOffset + 3u * entry];
    let fitScale = fit[fitOffset + 3u * entry + 1u];
    let zeroProbability = fit[fitOffset + 3u * entry + 2u];
    if (finite(fitShape) && finite(fitScale) && fitShape > 0.0 && fitScale > 0.0 ${zeroAtom ? '&& finite(zeroProbability) && zeroProbability >= 0.0 && zeroProbability <= 1.0' : ''}) {
      code = 3u;
      let tails = ${tails};
      if (finite(tails.x) && finite(tails.y)) {
        code = 0u;
        var lowerTail = tails.x;
        var upperTail = tails.y;
        ${
          zeroAtom
            ? `lowerTail = (1.0 - zeroProbability) * tails.x + select(0.0, zeroProbability, !isRasterClimateOriginalNegative(value));
        upperTail = (1.0 - zeroProbability) * tails.y + select(0.0, zeroProbability, isRasterClimateOriginalNegative(value));`
            : ''
        }
        result = select(getNormalQuantileValue(lowerTail), -getNormalQuantileValue(upperTail), upperTail < lowerTail);
        result = clamp(result, -${GPU_RASTER_STANDARDIZED_INDEX_LIMIT}, ${GPU_RASTER_STANDARDIZED_INDEX_LIMIT});
      }
    }
  }
}
indexOut[indexOutOffset + index] = result;
statusOut[statusOutOffset + index] = code;`
    });
  }
}
