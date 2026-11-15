// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getViewBindingRange,
  type GraphDataView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {
  validateGPUStorageBindingByteLength,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  RASTER_FINITE_WGSL,
  rasterBinding,
  validateRasterGraph,
  validateRasterViews
} from '../remote-sensing/raster-kernel-utils';
import {
  validateRasterEmbeddingQuantized,
  validateRasterEmbeddingShape,
  validateRasterEmbeddingVectors
} from './raster-embedding-common';

/** Caller-owned packed AlphaEarth output of {@link GPURasterEmbeddingQuantize}. */
export type GPURasterEmbeddingQuantizeOutput = {
  /** Four little-endian signed bytes per uint32, pixel-major. Unused final bytes are zero. */
  quantized: GraphDataView<'uint32'>;
  /** Optional one flag per pixel: 1 for finite, unmasked vectors, else 0. */
  validity?: GraphDataView<'uint32'>;
};

/** Properties of {@link GPURasterEmbeddingQuantize}. */
export type GPURasterEmbeddingQuantizeProps = {
  id?: string;
  pixelCount: number;
  /** Vector length in [1, 128]; the validity kernel walks one bounded vector per pixel. */
  dimensions: number;
  /** Pixel-major float32 components. Finite values outside [-1, 1] saturate. */
  vectors: GraphDataView<'float32'>;
  /** Optional per-pixel QA; zero invalidates the entire vector. */
  mask?: GraphDataView<'uint32'>;
  output: GPURasterEmbeddingQuantizeOutput;
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
 * Per-frame: view contents. Compile-time: pixelCount, dimensions, optional view presence.
 */
export class GPURasterEmbeddingQuantize implements GPUCommandNodeProducer {
  readonly id: string;

  constructor(readonly props: GPURasterEmbeddingQuantizeProps) {
    const id = props.id ?? 'raster-embedding-quantize';
    this.id = id;
    const {pixelCount, dimensions, vectors, mask, output} = props;
    validateRasterEmbeddingShape(id, pixelCount, dimensions);
    validateRasterEmbeddingVectors(id, 'vectors', vectors, pixelCount, dimensions, mask);
    validateRasterEmbeddingQuantized(
      id,
      'output.quantized',
      output.quantized,
      pixelCount,
      dimensions
    );
    if (output.validity) {
      validateSinglePackedView(output.validity, ['uint32'], `${id} output.validity`);
      validateViewLength(id, 'output.validity', output.validity, pixelCount);
    }
    validateRasterViews(id, this.getInputs(), this.getOutputs());
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
    const {id, props} = this;
    const {pixelCount, dimensions, vectors, mask, output} = props;
    for (const [name, view] of Object.entries({
      vectors,
      mask,
      'output.quantized': output.quantized,
      'output.validity': output.validity
    })) {
      if (!view) continue;
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        getViewBindingRange(view).size,
        'tile the embedding raster'
      );
    }
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'validity',
      pixelCount * 4,
      'tile the embedding raster'
    );
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    const validity =
      output.validity ?? createTransientView(graph, `${id}-validity`, 'uint32', pixelCount);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-validity`,
        operation: 'GPURasterEmbeddingQuantize',
        invocationCount: pixelCount,
        workgroupSize: 64,
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
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-pack`,
        operation: 'GPURasterEmbeddingQuantize',
        invocationCount: output.quantized.length,
        workgroupSize: 64,
        bindings: [
          rasterBinding('vectors', vectors),
          rasterBinding('validity', validity),
          rasterBinding('quantized', output.quantized, true)
        ],
        body: `var word = 0u;
for (var lane = 0u; lane < 4u; lane++) {
  let element = index * 4u + lane;
  if (element < ${pixelCount * dimensions}u) {
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
      })
    ];
  }
}
