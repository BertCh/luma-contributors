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
import {GET_NORMAL_QUANTILE_WGSL} from '../../analysis-core/kernels/normal-quantile-wgsl';
import {
  RASTER_FINITE_WGSL,
  rasterBinding,
  validateRasterGraph,
  validateRasterViews,
  type RasterScalarView
} from '../remote-sensing/raster-kernel-utils';
import {RASTER_REGULARIZED_GAMMA_TAILS_WGSL} from './raster-climate-distribution-wgsl';
import {
  MAXIMUM_RASTER_STREAM_CHUNK_SLICE_COUNT,
  MAXIMUM_RASTER_STREAM_SLOT_COUNT
} from './gpu-raster-streaming-accumulator';
import type {GPURasterStandardizedIndexDistribution} from './gpu-raster-standardized-index-calibration';

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
  target[0] = settings.activeSliceCount ?? chunkSliceCount;
  target[1] = settings.location ?? 0;
  return target;
}

/** Properties of {@link GPURasterStandardizedIndex}. */
export type GPURasterStandardizedIndexProps = {
  id?: string;
  cellCount: number;
  chunkSliceCount: number;
  slotCount?: number;
  distribution: GPURasterStandardizedIndexDistribution;
  /** Must match the calibration. */
  zeroInflated?: boolean;
  /** Chunk to evaluate, cell-major; nonfinite samples are missing. */
  values: GraphDataView<'float32'>;
  validity?: GraphDataView<'uint32'>;
  slotIndices?: GraphDataView<'uint32'>;
  /** `[shape, scale, zeroProbability]` per slot, from {@link GPURasterStandardizedIndexCalibration}. */
  fit: GraphDataView<'float32'>;
  /** Float32, see {@link getGPURasterStandardizedIndexParameterValues}. */
  parameters: GraphDataView<'float32'>;
  output: {
    /** `cell * chunkSliceCount + slice`, standardized index in `[-8.21, 8.21]`, NaN where status is nonzero. */
    index: GraphDataView<'float32'>;
    /** 0 valid, 1 missing input, 2 calibration not usable for the slot. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Standardized index of a chunk from a fit written by {@link GPURasterStandardizedIndexCalibration}
 * (SPI with `gamma`, SPEI with `fisk`, xclim's APP fit and zero handling). The probability of an
 * observation is evaluated through whichever tail is small, so `1 - p` is never formed near one and
 * the quantile keeps its precision on both sides. Results are clipped to xclim's `+-8.21`.
 *
 * Per-frame: `parameters` and the view contents. Compile-time: `cellCount`, `chunkSliceCount`,
 * `slotCount`, `distribution`, `zeroInflated`, and which optional views exist.
 */
export class GPURasterStandardizedIndex implements GPUCommandNodeProducer {
  readonly id: string;
  readonly slotCount: number;
  constructor(readonly props: GPURasterStandardizedIndexProps) {
    const id = props.id ?? 'raster-standardized-index';
    this.id = id;
    const slotCount = props.slotCount ?? 1;
    this.slotCount = slotCount;
    validateCapacity(id, 'cellCount', props.cellCount);
    validateCapacity(
      id,
      'chunkSliceCount',
      props.chunkSliceCount,
      1,
      MAXIMUM_RASTER_STREAM_CHUNK_SLICE_COUNT
    );
    validateCapacity(id, 'slotCount', slotCount, 1, MAXIMUM_RASTER_STREAM_SLOT_COUNT);
    validateRequirement(
      id,
      props.distribution === 'gamma' || props.distribution === 'fisk',
      "distribution must be 'gamma' or 'fisk'"
    );
    const observationCount = props.cellCount * props.chunkSliceCount;
    validateCapacity(id, 'cellCount times chunkSliceCount', observationCount);
    validateRequirement(
      id,
      props.values.length >= observationCount &&
        (!props.validity || props.validity.length >= observationCount) &&
        props.output.index.length >= observationCount &&
        props.output.status.length >= observationCount,
      'values, validity, output.index and output.status must hold cellCount times chunkSliceCount entries'
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
    validateRasterViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterScalarView[] {
    const {values, validity, slotIndices, fit, parameters} = this.props;
    return [
      values,
      fit,
      parameters,
      ...(validity ? [validity] : []),
      ...(slotIndices ? [slotIndices] : [])
    ];
  }

  private getOutputs(): RasterScalarView[] {
    return [this.props.output.index, this.props.output.status];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, slotCount} = this;
    const {chunkSliceCount: chunk, values, validity, slotIndices, fit, parameters, output} = props;
    const zeroInflated = props.zeroInflated ?? false;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    const tails =
      props.distribution === 'gamma'
        ? 'getRegularizedGammaTails(fitShape, positive / fitScale)'
        : 'getFiskTails(fitShape, positive / fitScale)';
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterStandardizedIndex',
        invocationCount: props.cellCount * chunk,
        workgroupSize: 64,
        bindings: [
          rasterBinding('values', values),
          rasterBinding('fit', fit),
          rasterBinding('parameters', parameters),
          ...(validity ? [rasterBinding('validity', validity)] : []),
          ...(slotCount > 1 && slotIndices ? [rasterBinding('slotIndices', slotIndices)] : []),
          rasterBinding('indexOut', output.index, true),
          rasterBinding('statusOut', output.status, true)
        ],
        declarations: `${RASTER_FINITE_WGSL}
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
if (slice < activeSliceCount && finite(value) ${validity ? '&& validity[validityOffset + source] != 0u' : ''}) {
  let slot = ${slotCount > 1 ? 'slotIndices[slotIndicesOffset + slice]' : '0u'};
  code = 2u;
  if (slot < ${slotCount}u) {
    let entry = cell * ${slotCount}u + slot;
    let fitShape = fit[fitOffset + 3u * entry];
    let fitScale = fit[fitOffset + 3u * entry + 1u];
    let zeroProbability = fit[fitOffset + 3u * entry + 2u];
    if (finite(fitShape) && finite(fitScale)) {
      code = 0u;
      var lowerTail = zeroProbability;
      var upperTail = 1.0 - zeroProbability;
      ${
        zeroInflated
          ? `if (value != 0.0) {
        let positive = value - parameters[parametersOffset + 1u];
        let tails = ${tails};
        lowerTail = zeroProbability + (1.0 - zeroProbability) * tails.x;
        upperTail = (1.0 - zeroProbability) * tails.y;
      }`
          : `let positive = value - parameters[parametersOffset + 1u];
      let tails = ${tails};
      lowerTail = tails.x;
      upperTail = tails.y;`
      }
      result = select(getNormalQuantileValue(lowerTail), -getNormalQuantileValue(upperTail), upperTail < lowerTail);
      result = clamp(result, -${GPU_RASTER_STANDARDIZED_INDEX_LIMIT}, ${GPU_RASTER_STANDARDIZED_INDEX_LIMIT});
    }
  }
}
indexOut[indexOutOffset + index] = result;
statusOut[statusOutOffset + index] = code;`
      })
    ];
  }
}
