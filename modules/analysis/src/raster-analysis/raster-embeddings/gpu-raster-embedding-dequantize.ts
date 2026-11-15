// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getViewBindingRange,
  type GraphDataView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateGPUStorageBindingByteLength,
  validateRequirement,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
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
  /** One `uint32` per pixel: 0 for a masked or invalid pixel, else 1. */
  validity?: GraphDataView<'uint32'>;
};

/** Properties of {@link GPURasterEmbeddingDequantize}. */
export type GPURasterEmbeddingDequantizeProps = {
  id?: string;
  /** Pixels of the raster, row-major. */
  pixelCount: number;
  /** Bands (vector length), in `[1, 128]`. */
  dimensions: number;
  /** Defaults to AlphaEarth. TESSERA accepts all signed bytes, including -128. */
  format?: 'alphaearth-int8' | 'tessera-int8' | 'float16';
  /**
   * Signed 8-bit components packed four to a `uint32` word, little-endian, pixel-major
   * (`pixel * dimensions + band`), `ceil(pixelCount * dimensions / 4)` words. An `Int8Array`
   * uploaded as is has this layout. For `float16`, two IEEE binary16 components occupy each word,
   * low half first. Pad the last word; padding is ignored. Interleave band-sequential files on the host.
   */
  quantized: GraphDataView<'uint32'>;
  /** Required only for TESSERA: one float32 multiplier per pixel (not per component). */
  scales?: GraphDataView<'float32'>;
  /** Optional per-pixel QA; zero invalidates the whole vector. */
  mask?: GraphDataView<'uint32'>;
  /** Divide each decoded vector by its length, so dot products are cosines. Defaults to false. */
  normalize?: boolean;
  output: GPURasterEmbeddingDequantizeOutput;
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
 * Per-frame: the contents of the views. Compile-time: `pixelCount`, `dimensions`, `format`, `normalize` and
 * which optional views exist.
 */
export class GPURasterEmbeddingDequantize implements GPUCommandNodeProducer {
  readonly id: string;

  constructor(readonly props: GPURasterEmbeddingDequantizeProps) {
    const id = props.id ?? 'raster-embedding-dequantize';
    this.id = id;
    validateRasterEmbeddingShape(id, props.pixelCount, props.dimensions);
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
      validateSinglePackedView(props.scales, ['float32'], `${id} scales`);
      validateRequirement(
        id,
        props.scales.length === props.pixelCount,
        'scales must hold pixelCount entries'
      );
    }
    if (props.mask) {
      validateSinglePackedView(props.mask, ['uint32'], `${id} mask`);
      validateRequirement(
        id,
        props.mask.length === props.pixelCount,
        'mask must hold pixelCount entries'
      );
    }
    if (format === 'float16') {
      validateSinglePackedView(props.quantized, ['uint32'], `${id} quantized`);
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
      validateSinglePackedView(props.output.validity, ['uint32'], `${id} output.validity`);
      validateViewLength(id, 'output.validity', props.output.validity, props.pixelCount);
    }
    validateRasterViews(id, this.getInputs(), this.getOutputs());
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
    const {id, props} = this;
    const {pixelCount, dimensions, quantized, output} = props;
    for (const [name, view] of Object.entries({
      quantized,
      scales: props.scales,
      mask: props.mask,
      'output.vectors': output.vectors,
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
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    const format = props.format ?? 'alphaearth-int8';
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterEmbeddingDequantize',
        invocationCount: pixelCount,
        workgroupSize: 64,
        bindings: [
          rasterBinding('quantized', quantized),
          ...(props.scales ? [rasterBinding('scales', props.scales)] : []),
          ...(props.mask ? [rasterBinding('mask', props.mask)] : []),
          rasterBinding('vectors', output.vectors, true),
          ...(output.validity ? [rasterBinding('validity', output.validity, true)] : [])
        ],
        declarations: `${RASTER_FINITE_WGSL}${getRasterEmbeddingDecodeWGSL('quantized')}
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
        body: `var values: array<f32, ${dimensions}>;
var largestMagnitude = 0.0;
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
  values[band] = value;
  largestMagnitude = max(largestMagnitude, abs(value));
}
${
  props.normalize
    ? `var squaredLength = 0.0;
if (isValid && largestMagnitude > 0.0) {
  for (var band = 0u; band < ${dimensions}u; band++) {
    values[band] /= largestMagnitude;
    squaredLength += values[band] * values[band];
  }
}
let scale = inverseSqrt(select(1.0, squaredLength, squaredLength > 0.0));`
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
