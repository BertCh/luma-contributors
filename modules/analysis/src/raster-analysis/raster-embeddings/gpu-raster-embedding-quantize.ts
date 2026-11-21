// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GraphDataView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {
  validateGPUStorageBindingByteLength,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {RASTER_FINITE_WGSL, rasterBinding} from '../remote-sensing/raster-kernel-utils';
import {
  validateRasterEmbeddingQuantized,
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
  type RasterEmbeddingView
} from './raster-embedding-chunks';

/** Caller-owned packed AlphaEarth output of {@link GPURasterEmbeddingQuantize}. */
export type GPURasterEmbeddingQuantizeOutput<
  UintRows extends RasterEmbeddingView<'uint32'> = GraphDataView<'uint32'>
> = {
  /** Four little-endian signed bytes per uint32, pixel-major. Unused final bytes are zero. */
  quantized: UintRows;
  /** Optional one flag per pixel: 1 for finite, unmasked vectors, else 0. */
  validity?: UintRows;
};

/** Properties of {@link GPURasterEmbeddingQuantize}. */
export type GPURasterEmbeddingQuantizeProps<
  FloatRows extends RasterEmbeddingView<'float32'> = GraphDataView<'float32'>,
  UintRows extends RasterEmbeddingView<'uint32'> = GraphDataView<'uint32'>
> = {
  id?: string;
  pixelCount: number;
  /** Vector length in [1, 1024]; the validity kernel walks one bounded vector per pixel. */
  dimensions: number;
  /** Pixel-major float32 components. Finite values outside [-1, 1] saturate. */
  vectors: FloatRows;
  /** Optional per-pixel QA; zero invalidates the entire vector. */
  mask?: UintRows;
  output: GPURasterEmbeddingQuantizeOutput<UintRows>;
};

/**
 * Encodes an embedding raster as packed AlphaEarth int8 components:
 * `clip(roundToEven(sign(value) * sqrt(abs(value)) * 127.5), -127, 127)`.
 * Source: AlphaEarth Foundations, supplementary S22.1 (power 2, scale 127.5),
 * https://arxiv.org/html/2507.22291v2#S22.SS1 (verified 2026-11-14).
 * A masked pixel or any non-finite component writes -128 to every band. Finite out-of-range
 * components saturate; vectors are not implicitly normalized. Use a normalized overview as input.
 *
 * Two dispatches: validate each bounded vector on the GPU, then one invocation per packed word.
 * Word ownership avoids races when dimensions are not divisible by four. A pixelCount-word
 * validity scratch is allocated only when the caller omits output.validity. Every binding is
 * checked against the device limit before any node or scratch is created; tile larger rasters.
 * Pixel fields accept GraphVectorView chunks with matching complete-pixel boundaries; shared tables stay packed.
 * Each chunk has its own workgroup count and uses the same pipeline.
 *
 * Per-frame: view contents. Compile-time: pixelCount, dimensions, optional view presence.
 */
export class GPURasterEmbeddingQuantize<
  FloatRows extends RasterEmbeddingView<'float32'> = GraphDataView<'float32'>,
  UintRows extends RasterEmbeddingView<'uint32'> = GraphDataView<'uint32'>
> implements GPUCommandNodeProducer
{
  readonly id: string;
  private readonly chunks: readonly RasterEmbeddingChunk[];
  readonly estimatedCommandNodeCount: number;

  constructor(readonly props: GPURasterEmbeddingQuantizeProps<FloatRows, UintRows>) {
    const id = props.id ?? 'raster-embedding-quantize';
    this.id = id;
    const {pixelCount, dimensions, vectors, mask, output} = props;
    validateRasterEmbeddingShape(
      id,
      pixelCount,
      dimensions,
      MAXIMUM_RASTER_EMBEDDING_DIMENSIONS,
      false
    );
    validateRasterEmbeddingVectors(id, 'vectors', vectors, pixelCount, dimensions, mask);
    validateRasterEmbeddingQuantized(
      id,
      'output.quantized',
      output.quantized,
      pixelCount,
      dimensions
    );
    if (output.validity) {
      validateViewLength(id, 'output.validity', output.validity, pixelCount);
    }
    this.chunks = planRasterEmbeddingChunks(
      id,
      pixelCount,
      {
        vectors: {view: props.vectors, format: 'float32', components: dimensions},
        mask: {view: props.mask, format: 'uint32', components: 1},
        'output.quantized': {
          view: props.output.quantized,
          format: 'uint32',
          components: dimensions,
          componentsPerWord: 4
        },
        'output.validity': {view: props.output.validity, format: 'uint32', components: 1}
      },
      'vectors',
      2
    );
    validateRasterEmbeddingViews(id, this.getInputs(), this.getOutputs(), [
      {inputs: this.getInputs(), outputs: output.validity ? [output.validity] : []},
      {inputs: [vectors], outputs: this.getOutputs()}
    ]);
    this.estimatedCommandNodeCount =
      1 + this.chunks.filter(chunk => chunk.pixelCount > 0).length * 2;
  }

  private getInputs() {
    const {vectors, mask} = this.props;
    return [vectors, ...(mask ? [mask] : [])];
  }

  private getOutputs() {
    const {quantized, validity} = this.props.output;
    return [quantized, ...(validity ? [validity] : [])];
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
      3,
      this.chunks
    );
    if (!this.props.output.validity) {
      for (const chunk of this.chunks)
        validateGPUStorageBindingByteLength(
          graph.device,
          this.id,
          'validity scratch',
          Math.max(chunk.pixelCount, 1) * 4,
          'reduce the embedding chunk pixelCount'
        );
    }
    const kernels = this.chunks.flatMap((chunk, chunkIndex) =>
      chunk.pixelCount === 0
        ? []
        : this.getChunkKernels(graph, getRasterEmbeddingChunkProps(this.props, chunk), chunkIndex)
    );
    return createRasterEmbeddingKernelNodes(graph, this.id, kernels);
  }

  private getChunkKernels<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    props: PackedEmbeddingProps<GPURasterEmbeddingQuantizeProps>,
    chunkIndex: number
  ): RasterEmbeddingKernel[] {
    const id = `${this.id}-chunk-${chunkIndex}`;
    const {pixelCount, dimensions, vectors, mask, output} = props;
    const validity =
      output.validity ?? createTransientView(graph, `${id}-validity`, 'uint32', pixelCount);
    return [
      {
        id: `${id}-validity`,
        operation: 'GPURasterEmbeddingQuantize',
        invocationCount: pixelCount,
        pixelCount,
        bindings: [
          rasterBinding('vectors', vectors),
          ...(mask ? [rasterBinding('mask', mask)] : []),
          rasterBinding('validity', validity, true)
        ],
        declarations: RASTER_FINITE_WGSL,
        body: `var isValid = ${mask ? 'mask[maskOffset + index] != 0u' : 'true'};
for (var band = 0u; band < ${dimensions}u; band++) {
  isValid = isValid && finite(vectors[vectorsOffset + index * ${dimensions}u + band]);
}
validity[validityOffset + index] = select(0u, 1u, isValid);`
      },
      {
        id: `${id}-pack`,
        operation: 'GPURasterEmbeddingQuantize',
        invocationCount: output.quantized.length,
        pixelCount,
        bindings: [
          rasterBinding('vectors', vectors),
          // This caller output may share a physical buffer with quantized. Keep both bindings
          // writable in this usage scope; their aligned ranges were checked at construction.
          rasterBinding('validity', validity, true),
          rasterBinding('quantized', output.quantized, true)
        ],
        body: `var word = 0u;
for (var lane = 0u; lane < 4u; lane++) {
  let element = index * 4u + lane;
  if (element < embeddingDispatch.words[0u][1u] * ${dimensions}u) {
    var code = -128i;
    if (validity[validityOffset + element / ${dimensions}u] != 0u) {
      let value = vectors[vectorsOffset + element];
      let magnitude = sqrt(min(abs(value), 1.0)) * 127.5;
      code = i32(clamp(round(sign(value) * magnitude), -127.0, 127.0));
    }
    word |= (bitcast<u32>(code) & 255u) << (lane * 8u);
  }
}
quantized[quantizedOffset + index] = word;`
      }
    ];
  }
}
