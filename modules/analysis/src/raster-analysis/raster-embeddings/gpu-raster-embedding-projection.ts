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
  validateCapacity,
  validateFiniteNumber,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {RASTER_FINITE_WGSL, rasterBinding} from '../remote-sensing/raster-kernel-utils';
import {
  MAXIMUM_RASTER_EMBEDDING_DIMENSIONS,
  validateRasterEmbeddingShape,
  validateRasterEmbeddingVectors
} from './raster-embedding-common';

import {
  createRasterEmbeddingKernelNodes,
  getRasterEmbeddingChunkProps,
  planRasterEmbeddingChunks,
  validateRasterEmbeddingBindings,
  validateRasterEmbeddingViews,
  type PackedEmbeddingProps,
  type RasterEmbeddingChunk,
  type RasterEmbeddingKernel,
  type RasterEmbeddingScalarView,
  type RasterEmbeddingView
} from './raster-embedding-chunks';

/** Components projected per pixel, held in private memory. */
export const MAXIMUM_RASTER_EMBEDDING_COMPONENT_COUNT = 16;

/** Float32 elements of the per-frame parameters: low and high of each of the three RGB channels. */
export const GPU_RASTER_EMBEDDING_PROJECTION_PARAMETER_LENGTH = 6;

/** Per-frame settings of {@link GPURasterEmbeddingProjection}. */
export type GPURasterEmbeddingProjectionSettings = {
  /**
   * `[low, high]` of the first three components, mapped to `[0, 1]` and clamped. Use
   * `getRasterEmbeddingStretchRanges(eigenvalues)` for a two-sigma stretch.
   */
  stretchRanges: readonly (readonly [number, number])[];
};

/** Packs {@link GPURasterEmbeddingProjectionSettings}: `[low0, high0, low1, high1, low2, high2]`. */
export function getGPURasterEmbeddingProjectionParameterValues(
  settings: GPURasterEmbeddingProjectionSettings,
  target: Float32Array = new Float32Array(GPU_RASTER_EMBEDDING_PROJECTION_PARAMETER_LENGTH)
): Float32Array {
  const id = 'getGPURasterEmbeddingProjectionParameterValues';
  validateRequirement(
    id,
    settings.stretchRanges.length >= 3,
    'stretchRanges must hold three ranges'
  );
  for (let channel = 0; channel < 3; channel++) {
    const [low, high] = settings.stretchRanges[channel];
    validateFiniteNumber(id, 'stretch low', low);
    validateFiniteNumber(id, 'stretch high', high);
    validateRequirement(id, Math.fround(high) > Math.fround(low), 'stretch high must exceed low');
    target[channel * 2] = low;
    target[channel * 2 + 1] = high;
  }
  return target;
}

/** Caller-owned outputs of {@link GPURasterEmbeddingProjection}; at least one is required. */
export type GPURasterEmbeddingProjectionOutput<
  FloatRows extends RasterEmbeddingView<'float32'> = GraphDataView<'float32'>
> = {
  /** Scores on the components, `pixel * componentCount + component`; NaN for invalid pixels. */
  projected?: FloatRows;
  /** The first three scores stretched to `[0, 1]`, `pixel * 3 + channel`; NaN for invalid pixels. Needs `parameters`. */
  rgb?: FloatRows;
};

/** Properties of {@link GPURasterEmbeddingProjection}. */
export type GPURasterEmbeddingProjectionProps<
  FloatRows extends RasterEmbeddingView<'float32'> = GraphDataView<'float32'>,
  UintRows extends RasterEmbeddingView<'uint32'> = GraphDataView<'uint32'>
> = {
  id?: string;
  pixelCount: number;
  /** Vector length, in `[1, 1024]`. */
  dimensions: number;
  vectors: FloatRows;
  /** Per-pixel QA; zero makes the pixel invalid. */
  mask?: UintRows;
  /** Components to project on, in `[1, 16]`; at least 3 with `output.rgb`. */
  componentCount: number;
  /** Mean vector subtracted first, `dimensions` entries (from {@link GPURasterEmbeddingCovariance}). */
  mean: GraphDataView<'float32'>;
  /** Unit components, `component * dimensions + band`, from `getRasterEmbeddingPrincipalComponents`. */
  components: GraphDataView<'float32'>;
  /** Float32, see {@link getGPURasterEmbeddingProjectionParameterValues}. Needed with `output.rgb`. */
  parameters?: GraphDataView<'float32'>;
  output: GPURasterEmbeddingProjectionOutput<FloatRows>;
};

/**
 * Projects an embedding raster on principal components: `(x - mean) . component` per pixel, and
 * optionally the first three scores stretched to an RGB image. This is the visualisation that
 * turns a 64-band embedding into a map. The components are computed on the host from
 * {@link GPURasterEmbeddingCovariance}, so the heavy pass is on the GPU and the eigenproblem
 * is not.
 *
 * Pixel fields accept GraphVectorView chunks with matching complete-pixel boundaries; shared tables stay packed.
 * Each chunk has its own workgroup count and uses the same pipeline.
 *
 * Per-frame: `parameters`, `mean`, `components`. Compile-time: `pixelCount`, `dimensions`,
 * `componentCount` and which optional views exist.
 */
export class GPURasterEmbeddingProjection<
  FloatRows extends RasterEmbeddingView<'float32'> = GraphDataView<'float32'>,
  UintRows extends RasterEmbeddingView<'uint32'> = GraphDataView<'uint32'>
> implements GPUCommandNodeProducer
{
  readonly id: string;
  private readonly chunks: readonly RasterEmbeddingChunk[];
  readonly estimatedCommandNodeCount: number;

  constructor(readonly props: GPURasterEmbeddingProjectionProps<FloatRows, UintRows>) {
    const id = props.id ?? 'raster-embedding-projection';
    this.id = id;
    const {pixelCount, dimensions, output} = props;
    validateRasterEmbeddingShape(
      id,
      pixelCount,
      dimensions,
      MAXIMUM_RASTER_EMBEDDING_DIMENSIONS,
      false
    );
    validateCapacity(
      id,
      'componentCount',
      props.componentCount,
      output.rgb ? 3 : 1,
      MAXIMUM_RASTER_EMBEDDING_COMPONENT_COUNT
    );
    validateRasterEmbeddingVectors(
      id,
      'vectors',
      props.vectors,
      pixelCount,
      dimensions,
      props.mask
    );
    validateRequirement(
      id,
      props.mean.length === dimensions &&
        props.components.length === props.componentCount * dimensions,
      'mean must hold dimensions entries and components componentCount times dimensions'
    );
    validateRequirement(
      id,
      Boolean(output.projected || output.rgb),
      'output needs at least one view'
    );
    validateRequirement(
      id,
      (!output.projected || output.projected.length === pixelCount * props.componentCount) &&
        (!output.rgb || output.rgb.length === pixelCount * 3),
      'output.projected must hold pixelCount times componentCount entries and output.rgb pixelCount times 3'
    );
    validateRequirement(
      id,
      !output.rgb ||
        (props.parameters !== undefined &&
          props.parameters.length >= GPU_RASTER_EMBEDDING_PROJECTION_PARAMETER_LENGTH),
      'output.rgb needs parameters'
    );
    if (props.mean) validateSinglePackedView(props.mean, ['float32'], `${id} mean`);
    if (props.components)
      validateSinglePackedView(props.components, ['float32'], `${id} components`);
    if (props.parameters)
      validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    validateRasterEmbeddingViews(id, this.getInputs(), this.getOutputs());
    this.chunks = planRasterEmbeddingChunks(
      id,
      pixelCount,
      {
        vectors: {view: props.vectors, format: 'float32', components: dimensions},
        mask: {view: props.mask, format: 'uint32', components: 1},
        'output.projected': {
          view: props.output.projected,
          format: 'float32',
          components: props.componentCount
        },
        'output.rgb': {view: props.output.rgb, format: 'float32', components: 3}
      },
      'vectors',
      1
    );
    this.estimatedCommandNodeCount =
      1 + this.chunks.filter(chunk => chunk.pixelCount > 0).length * 1;
  }

  private getInputs(): RasterEmbeddingScalarView[] {
    const {vectors, mask, mean, components, parameters} = this.props;
    return [
      vectors,
      mean,
      components,
      ...(mask ? [mask] : []),
      ...(parameters && this.props.output.rgb ? [parameters] : [])
    ];
  }

  private getOutputs(): RasterEmbeddingScalarView[] {
    const {projected, rgb} = this.props.output;
    return [...(projected ? [projected] : []), ...(rgb ? [rgb] : [])];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateRasterEmbeddingBindings(
      graph,
      this.id,
      this.getInputs(),
      this.getOutputs(),
      this.estimatedCommandNodeCount - 1,
      this.props,
      3 +
        Number(Boolean(this.props.mask)) +
        Number(Boolean(this.props.parameters && this.props.output.rgb)) +
        Number(Boolean(this.props.output.projected)) +
        Number(Boolean(this.props.output.rgb)),
      this.chunks
    );
    const kernels = this.chunks.flatMap((chunk, chunkIndex) =>
      chunk.pixelCount === 0
        ? []
        : this.getChunkKernels(graph, getRasterEmbeddingChunkProps(this.props, chunk), chunkIndex)
    );
    return createRasterEmbeddingKernelNodes(graph, this.id, kernels);
  }

  private getChunkKernels<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    props: PackedEmbeddingProps<GPURasterEmbeddingProjectionProps>,
    chunkIndex: number
  ): RasterEmbeddingKernel[] {
    const id = `${this.id}-chunk-${chunkIndex}`;
    const {pixelCount, dimensions, componentCount, output} = props;
    return [
      {
        id,
        operation: 'GPURasterEmbeddingProjection',
        invocationCount: pixelCount,
        pixelCount,
        bindings: [
          rasterBinding('vectors', props.vectors),
          rasterBinding('mean', props.mean),
          rasterBinding('components', props.components),
          ...(props.mask ? [rasterBinding('mask', props.mask)] : []),
          ...(props.parameters && output.rgb
            ? [rasterBinding('parameters', props.parameters)]
            : []),
          ...(output.projected ? [rasterBinding('projected', output.projected, true)] : []),
          ...(output.rgb ? [rasterBinding('rgb', output.rgb, true)] : [])
        ],
        declarations: RASTER_FINITE_WGSL,
        body: `var isValid = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'};
for (var band = 0u; band < ${dimensions}u; band++) {
  let value = vectors[vectorsOffset + index * ${dimensions}u + band];
  if (!finite(value)) { isValid = false; }
}
for (var component = 0u; component < ${componentCount}u; component++) {
  var score = 0.0;
  for (var band = 0u; band < ${dimensions}u; band++) {
    score += (vectors[vectorsOffset + index * ${dimensions}u + band] - mean[meanOffset + band]) * components[componentsOffset + component * ${dimensions}u + band];
  }
  score = select(missing(), score, isValid);
  ${output.projected ? `projected[projectedOffset + index * ${componentCount}u + component] = score;` : ''}
  ${
    output.rgb
      ? `if (component < 3u) {
    let low = parameters[parametersOffset + component * 2u];
    let high = parameters[parametersOffset + component * 2u + 1u];
    rgb[rgbOffset + index * 3u + component] = select(missing(), clamp((score - low) / (high - low), 0.0, 1.0), isValid);
  }`
      : ''
  }
}`
      }
    ];
  }
}
