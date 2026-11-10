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
  RASTER_FINITE_WGSL,
  rasterBinding,
  validateRasterGraph,
  validateRasterViews
} from '../remote-sensing/raster-kernel-utils';
import {
  getRasterEmbeddingDecodeWGSL,
  validateRasterEmbeddingQuantized,
  validateRasterEmbeddingShape,
  validateRasterEmbeddingVectors
} from './raster-embedding-common';

/** Caller-owned outputs of {@link GPURasterEmbeddingDequantize}. */
export type GPURasterEmbeddingDequantizeOutput = {
  /** Decoded vectors, `pixel * dimensions + band`; every component of an invalid pixel is NaN. */
  vectors: GraphDataView<'float32'>;
  /** One `uint32` per pixel: 0 where any band is NoData (-128), else 1. */
  validity?: GraphDataView<'uint32'>;
};

/** Properties of {@link GPURasterEmbeddingDequantize}. */
export type GPURasterEmbeddingDequantizeProps = {
  id?: string;
  /** Pixels of the raster, row-major. */
  pixelCount: number;
  /** Bands (vector length), in `[1, 128]`. */
  dimensions: number;
  /**
   * Signed 8-bit components packed four to a `uint32` word, little-endian, pixel-major
   * (`pixel * dimensions + band`), `ceil(pixelCount * dimensions / 4)` words. An `Int8Array`
   * uploaded as is has this layout. Band-sequential rasters are interleaved by the host adapter.
   */
  quantized: GraphDataView<'uint32'>;
  /** Divide each decoded vector by its length, so dot products are cosines. Defaults to false. */
  normalize?: boolean;
  output: GPURasterEmbeddingDequantizeOutput;
};

/**
 * Decodes AlphaEarth Satellite Embedding V1 int8 rasters: each component is
 * `sign(q) * (q / 127.5)^2` and `q = -128` is NoData (a pixel with NoData in any band is invalid).
 * Reading the integers as ordinary values, or averaging them (`gdaladdo -r average`), is wrong for
 * this product: the quantization is nonlinear, which is also why the overview rule averages the
 * decoded vectors ({@link GPURasterEmbeddingPyramidLevel}).
 *
 * The decoded vectors are close to unit length but not exactly, because of rounding to 8 bits;
 * set `normalize` to make dot products exact cosines. Other quantization schemes are not covered:
 * the TESSERA int8 scale is not verified.
 *
 * Per-frame: the contents of the views. Compile-time: `pixelCount`, `dimensions`, `normalize` and
 * which optional views exist.
 */
export class GPURasterEmbeddingDequantize implements GPUCommandNodeProducer {
  readonly id: string;

  constructor(readonly props: GPURasterEmbeddingDequantizeProps) {
    const id = props.id ?? 'raster-embedding-dequantize';
    this.id = id;
    validateRasterEmbeddingShape(id, props.pixelCount, props.dimensions);
    validateRasterEmbeddingQuantized(
      id,
      'quantized',
      props.quantized,
      props.pixelCount,
      props.dimensions
    );
    validateRasterEmbeddingVectors(
      id,
      'output.vectors',
      props.output.vectors,
      props.pixelCount,
      props.dimensions,
      props.output.validity
    );
    validateRasterViews(id, [props.quantized], this.getOutputs());
  }

  private getOutputs() {
    const {vectors, validity} = this.props.output;
    return [vectors, ...(validity ? [validity] : [])];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {pixelCount, dimensions, quantized, output} = props;
    validateRasterGraph(id, graph, [quantized], this.getOutputs());
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterEmbeddingDequantize',
        invocationCount: pixelCount,
        workgroupSize: 64,
        bindings: [
          rasterBinding('quantized', quantized),
          rasterBinding('vectors', output.vectors, true),
          ...(output.validity ? [rasterBinding('validity', output.validity, true)] : [])
        ],
        declarations: `${RASTER_FINITE_WGSL}${getRasterEmbeddingDecodeWGSL('quantized')}`,
        body: `var values: array<f32, ${dimensions}>;
var squaredLength = 0.0;
var isValid = true;
for (var band = 0u; band < ${dimensions}u; band++) {
  let byte = readEmbeddingByte(index * ${dimensions}u + band);
  if (byte == -128) { isValid = false; }
  let value = getEmbeddingValue(byte);
  values[band] = value;
  squaredLength += value * value;
}
${
  props.normalize
    ? 'let scale = select(1.0, inverseSqrt(squaredLength), squaredLength > 0.0);'
    : 'let scale = 1.0;'
}
for (var band = 0u; band < ${dimensions}u; band++) {
  vectors[vectorsOffset + index * ${dimensions}u + band] = select(missing(), values[band] * scale, isValid);
}
${output.validity ? 'validity[validityOffset + index] = select(0u, 1u, isValid);' : ''}`
      })
    ];
  }
}
