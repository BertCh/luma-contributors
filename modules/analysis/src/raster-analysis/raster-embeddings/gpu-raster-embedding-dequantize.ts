// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GraphDataView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';
import {RASTER_FINITE_WGSL, rasterBinding} from '../remote-sensing/raster-kernel-utils';
import {
  getRasterEmbeddingDecodeWGSL,
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

/** Caller-owned outputs of {@link GPURasterEmbeddingDequantize}. */
export type GPURasterEmbeddingDequantizeOutput<
  FloatRows extends RasterEmbeddingView<'float32'> = GraphDataView<'float32'>,
  UintRows extends RasterEmbeddingView<'uint32'> = GraphDataView<'uint32'>
> = {
  /** Decoded vectors, `pixel * dimensions + band`; every component of an invalid pixel is NaN. */
  vectors: FloatRows;
  /** One `uint32` per pixel: 0 for a masked or invalid pixel, else 1. */
  validity?: UintRows;
};

/** Properties of {@link GPURasterEmbeddingDequantize}. */
export type GPURasterEmbeddingDequantizeProps<
  FloatRows extends RasterEmbeddingView<'float32'> = GraphDataView<'float32'>,
  UintRows extends RasterEmbeddingView<'uint32'> = GraphDataView<'uint32'>
> = {
  id?: string;
  /** Pixels of the raster, row-major. */
  pixelCount: number;
  /** Bands (vector length), in `[1, 1024]`. */
  dimensions: number;
  /** Defaults to AlphaEarth. TESSERA accepts all signed bytes, including -128. */
  format?: 'alphaearth-int8' | 'tessera-int8' | 'float16';
  /**
   * Signed 8-bit components packed four to a `uint32` word, little-endian, pixel-major
   * (`pixel * dimensions + band`), `ceil(pixelCount * dimensions / 4)` words. An `Int8Array`
   * uploaded as is has this layout. For `float16`, two IEEE binary16 components occupy each word,
   * low half first. Pad the last word; padding is ignored. Interleave band-sequential files on the host.
   */
  quantized: UintRows;
  /** Required only for TESSERA: one float32 multiplier per pixel (not per component). */
  scales?: FloatRows;
  /** Optional per-pixel QA; zero invalidates the whole vector. */
  mask?: UintRows;
  /** Divide each decoded vector by its length, so dot products are cosines. Defaults to false. */
  normalize?: boolean;
  output: GPURasterEmbeddingDequantizeOutput<FloatRows, UintRows>;
};

/**
 * Decodes AlphaEarth int8, TESSERA scaled int8, or IEEE binary16 embedding rasters.
 * For AlphaEarth Satellite Embedding V1, each component is
 * `sign(q) * (q / 127.5)^2` and `q = -128` is NoData (a pixel with NoData in any band is invalid).
 * Reading the integers as ordinary values, or averaging them (`gdaladdo -r average`), is wrong for
 * this product: the quantization is nonlinear, which is also why the overview rule averages the
 * decoded vectors ({@link GPURasterEmbeddingPyramidLevel}).
 *
 * The decoded vectors are close to unit length but not exactly, because of rounding to 8 bits;
 * set `normalize` to make dot products exact cosines.
 *
 * TESSERA decodes `f32(q) * scales[pixel]`, with no reserved signed byte, matching
 * https://github.com/ucam-eo/geotessera/blob/main/geotessera/core.py#L30-L61
 * (`dequantize_embedding`, verified 2026-11-14). A non-finite scale or decoded component
 * invalidates the whole pixel. Zero and negative finite scales retain their arithmetic meaning.
 * Float16 uses IEEE binary16: `(-1)^sign * (1024 + fraction) * 2^(exponent - 25)` for
 * normal values and `(-1)^sign * fraction * 2^-24` for subnormals, per
 * https://www.w3.org/TR/WGSL/#floating-point-types . Exponent 31 invalidates the pixel.
 * Integer-field decoding preserves subnormals without requiring `shader-f16`.
 * AlphaEarth's formula is documented at https://arxiv.org/html/2507.22291v2#S22.SS1 .
 *
 * Pixel fields accept GraphVectorView chunks with matching complete-pixel boundaries; shared tables stay packed.
 * Each chunk has its own workgroup count and uses the same pipeline.
 *
 * Per-frame: the contents of the views. Compile-time: `pixelCount`, `dimensions`, `format`, `normalize` and
 * which optional views exist.
 */
export class GPURasterEmbeddingDequantize<
  FloatRows extends RasterEmbeddingView<'float32'> = GraphDataView<'float32'>,
  UintRows extends RasterEmbeddingView<'uint32'> = GraphDataView<'uint32'>
> implements GPUCommandNodeProducer
{
  readonly id: string;
  private readonly chunks: readonly RasterEmbeddingChunk[];
  readonly estimatedCommandNodeCount: number;

  constructor(readonly props: GPURasterEmbeddingDequantizeProps<FloatRows, UintRows>) {
    const id = props.id ?? 'raster-embedding-dequantize';
    this.id = id;
    const {pixelCount, dimensions} = props;
    validateRasterEmbeddingShape(
      id,
      props.pixelCount,
      props.dimensions,
      MAXIMUM_RASTER_EMBEDDING_DIMENSIONS,
      false
    );
    const format = props.format ?? 'alphaearth-int8';
    validateRequirement(
      id,
      ['alphaearth-int8', 'tessera-int8', 'float16'].includes(format),
      'format must be alphaearth-int8, tessera-int8 or float16'
    );
    validateRequirement(
      id,
      Boolean(props.scales) === (format === 'tessera-int8'),
      'scales must be supplied exactly when format is tessera-int8'
    );
    if (props.scales) {
      validateRequirement(
        id,
        props.scales.length === props.pixelCount,
        'scales must hold pixelCount entries'
      );
    }
    if (props.mask) {
      validateRequirement(
        id,
        props.mask.length === props.pixelCount,
        'mask must hold pixelCount entries'
      );
    }
    if (format === 'float16') {
      validateRequirement(
        id,
        props.quantized.length === Math.ceil((props.pixelCount * props.dimensions) / 2),
        'quantized must hold ceil(pixelCount times dimensions / 2) words'
      );
    } else
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
      props.dimensions
    );
    if (props.output.validity) {
      validateViewLength(id, 'output.validity', props.output.validity, props.pixelCount);
    }
    const inputs = this.getInputs();
    const outputs = this.getOutputs();
    validateRasterEmbeddingViews(id, inputs, outputs, [{inputs, outputs}]);
    this.chunks = planRasterEmbeddingChunks(
      id,
      pixelCount,
      {
        'output.vectors': {view: props.output.vectors, format: 'float32', components: dimensions},
        quantized: {
          view: props.quantized,
          format: 'uint32',
          components: dimensions,
          componentsPerWord: props.format === 'float16' ? 2 : 4
        },
        scales: {view: props.scales, format: 'float32', components: 1},
        mask: {view: props.mask, format: 'uint32', components: 1},
        'output.validity': {view: props.output.validity, format: 'uint32', components: 1}
      },
      'output.vectors',
      1
    );
    this.estimatedCommandNodeCount =
      1 + this.chunks.filter(chunk => chunk.pixelCount > 0).length * 1;
  }

  private getInputs() {
    const {quantized, scales, mask} = this.props;
    return [quantized, ...(scales ? [scales] : []), ...(mask ? [mask] : [])];
  }

  private getOutputs() {
    const {vectors, validity} = this.props.output;
    return [vectors, ...(validity ? [validity] : [])];
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
      2 +
        Number(Boolean(this.props.mask)) +
        Number(Boolean(this.props.scales)) +
        Number(Boolean(this.props.output.validity)),
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
    props: PackedEmbeddingProps<GPURasterEmbeddingDequantizeProps>,
    chunkIndex: number
  ): RasterEmbeddingKernel[] {
    const id = `${this.id}-chunk-${chunkIndex}`;
    const {pixelCount, dimensions, quantized, output} = props;
    const format = props.format ?? 'alphaearth-int8';
    return [
      {
        id,
        operation: 'GPURasterEmbeddingDequantize',
        invocationCount: pixelCount,
        pixelCount,
        bindings: [
          rasterBinding('quantized', quantized),
          ...(props.scales ? [rasterBinding('scales', props.scales)] : []),
          ...(props.mask ? [rasterBinding('mask', props.mask)] : []),
          rasterBinding('vectors', output.vectors, true),
          ...(output.validity ? [rasterBinding('validity', output.validity, true)] : [])
        ],
        declarations: `${RASTER_FINITE_WGSL}${SCALED_QUOTIENT_WGSL}${getRasterEmbeddingDecodeWGSL('quantized')}
fn readEmbeddingHalf(element: u32) -> f32 {
  let word = quantized[quantizedOffset + (element >> 1u)];
  let halfBits = (word >> ((element & 1u) * 16u)) & 65535u;
  let exponent = (halfBits >> 10u) & 31u;
  let fraction = halfBits & 1023u;
  if (exponent == 31u) { return missing(); }
  var magnitude = f32(fraction) * 5.960464477539063e-8;
  if (exponent != 0u) { magnitude = bitcast<f32>(((exponent + 112u) << 23u) | (fraction << 13u)); }
  return bitcast<f32>(bitcast<u32>(magnitude) | ((halfBits & 32768u) << 16u));
}`,
        body: `var largestMagnitude = 0.0;
var isValid = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'};
${props.scales ? 'let pixelScale = scales[scalesOffset + index];\nisValid = isValid && finite(pixelScale);' : ''}
for (var band = 0u; band < ${dimensions}u; band++) {
  ${
    format === 'float16'
      ? `let value = readEmbeddingHalf(index * ${dimensions}u + band);`
      : `
  let byte = readEmbeddingByte(index * ${dimensions}u + band);
  ${format === 'alphaearth-int8' ? 'if (byte == -128) { isValid = false; }' : ''}
  let value = ${format === 'tessera-int8' ? 'f32(byte) * pixelScale' : 'getEmbeddingValue(byte)'};`
  }
  isValid = isValid && finite(value);
  vectors[vectorsOffset + index * ${dimensions}u + band] = value;
  largestMagnitude = max(largestMagnitude, abs(value));
}
${
  props.normalize
    ? `var squaredLength = 0.0;
if (isValid && largestMagnitude > 0.0) {
  for (var band = 0u; band < ${dimensions}u; band++) {
    vectors[vectorsOffset + index * ${dimensions}u + band] = getScaledQuotient(vectors[vectorsOffset + index * ${dimensions}u + band], largestMagnitude, 1.0, 0);
    squaredLength += vectors[vectorsOffset + index * ${dimensions}u + band] * vectors[vectorsOffset + index * ${dimensions}u + band];
  }
}
let scale = inverseSqrt(select(1.0, squaredLength, squaredLength > 0.0));`
    : 'let scale = 1.0;'
}
for (var band = 0u; band < ${dimensions}u; band++) {
  vectors[vectorsOffset + index * ${dimensions}u + band] = select(missing(), vectors[vectorsOffset + index * ${dimensions}u + band] * scale, isValid);
}
${output.validity ? 'validity[validityOffset + index] = select(0u, 1u, isValid);' : ''}`
      }
    ];
  }
}
