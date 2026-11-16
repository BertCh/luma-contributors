// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GraphDataView,
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  RASTER_FINITE_WGSL,
  rasterBinding,
  validateRasterViews,
  type RasterScalarView
} from '../remote-sensing/raster-kernel-utils';
import {
  getRasterEmbeddingDecodeWGSL,
  validateRasterEmbeddingQuantized,
  validateRasterEmbeddingShape,
  validateRasterEmbeddingVectors
} from './raster-embedding-common';

/**
 * Source pixels per overview pixel along one axis. Each output thread loops over its footprint and
 * the vector length, so the footprint (at most 16 x 16 pixels) is a per-thread bound.
 */
export const MAXIMUM_RASTER_EMBEDDING_PYRAMID_SCALE = 16;

/** Caller-owned outputs of {@link GPURasterEmbeddingPyramidLevel}. */
export type GPURasterEmbeddingPyramidLevelOutput = {
  /** Unit-length overview vectors, `pixel * dimensions + band`, row-major over the overview; NaN where no source pixel is valid. */
  vectors: GraphDataView<'float32'>;
  /** One `uint32` per overview pixel: 1 where the footprint holds a valid source pixel. */
  validity?: GraphDataView<'uint32'>;
  /** Valid source pixels in each footprint. */
  validCount?: GraphDataView<'uint32'>;
  /**
   * Length of the mean before renormalization, in `[0, 1]`: 1 where the footprint is uniform and
   * smaller where its pixels disagree. Useful to mask overviews that average unlike things.
   */
  meanLength?: GraphDataView<'float32'>;
};

/** Properties of {@link GPURasterEmbeddingPyramidLevel}. */
export type GPURasterEmbeddingPyramidLevelProps = {
  id?: string;
  sourceWidth: number;
  sourceHeight: number;
  /** Vector length, in `[1, 128]`. */
  dimensions: number;
  /** Source pixels per overview pixel, one integer for both axes or `[horizontal, vertical]`, each in `[1, 16]`. */
  scale: number | readonly [number, number];
  /** Decoded float32 vectors. Give this or `quantized`. */
  vectors?: GraphDataView<'float32'>;
  /**
   * Packed int8 AlphaEarth components, decoded in the kernel by `sign(q) * (q / 127.5)^2`; a pixel
   * with `-128` in any band is invalid. Give this or `vectors`.
   */
  quantized?: GraphDataView<'uint32'>;
  /** Source QA; zero excludes the pixel. */
  mask?: GraphDataView<'uint32'>;
  output: GPURasterEmbeddingPyramidLevelOutput;
};

/**
 * Overview level of an embedding raster in the unit-vector-mean mode: the decoded vectors of each
 * footprint are averaged and the mean is renormalized to unit length, as the AlphaEarth
 * documentation prescribes. The generic mean overview (`GPURasterOverview`, `gdaladdo -r
 * average`) is wrong for these data twice over: on the raw int8 values the nonlinear quantization
 * is averaged, and on decoded values the mean of unit vectors is shorter than one, which biases
 * every later dot product.
 *
 * Footprints at odd edges are partial, as in `GPURasterOverview`. Each level here is computed from
 * the vectors given, so build coarser levels from this level's output.
 *
 * Per-frame: the contents of the views. Compile-time: the sizes, `scale`, `dimensions` and which
 * optional views exist.
 */
export class GPURasterEmbeddingPyramidLevel implements GPUCommandNodeProducer {
  readonly id: string;
  /** Overview width in pixels. */
  readonly width: number;
  /** Overview height in pixels. */
  readonly height: number;
  private readonly horizontalScale: number;
  private readonly verticalScale: number;

  constructor(readonly props: GPURasterEmbeddingPyramidLevelProps) {
    const id = props.id ?? 'raster-embedding-pyramid-level';
    this.id = id;
    const {sourceWidth, sourceHeight, dimensions, output} = props;
    validateCapacity(id, 'sourceWidth', sourceWidth);
    validateCapacity(id, 'sourceHeight', sourceHeight);
    const [horizontalScale, verticalScale] =
      typeof props.scale === 'number' ? [props.scale, props.scale] : props.scale;
    validateCapacity(
      id,
      'horizontal scale',
      horizontalScale,
      1,
      MAXIMUM_RASTER_EMBEDDING_PYRAMID_SCALE
    );
    validateCapacity(
      id,
      'vertical scale',
      verticalScale,
      1,
      MAXIMUM_RASTER_EMBEDDING_PYRAMID_SCALE
    );
    this.horizontalScale = horizontalScale;
    this.verticalScale = verticalScale;
    this.width = Math.ceil(sourceWidth / horizontalScale);
    this.height = Math.ceil(sourceHeight / verticalScale);
    const sourcePixelCount = sourceWidth * sourceHeight;
    const outputPixelCount = this.width * this.height;
    validateRasterEmbeddingShape(id, sourcePixelCount, dimensions);
    validateRequirement(
      id,
      (props.vectors === undefined) !== (props.quantized === undefined),
      'needs exactly one of vectors and quantized'
    );
    if (props.vectors) {
      validateRasterEmbeddingVectors(
        id,
        'vectors',
        props.vectors,
        sourcePixelCount,
        dimensions,
        props.mask
      );
    } else {
      validateRasterEmbeddingQuantized(
        id,
        'quantized',
        props.quantized!,
        sourcePixelCount,
        dimensions
      );
      validateRequirement(
        id,
        !props.mask || props.mask.length === sourcePixelCount,
        'mask must hold sourceWidth times sourceHeight entries'
      );
    }
    validateRasterEmbeddingVectors(
      id,
      'output.vectors',
      output.vectors,
      outputPixelCount,
      dimensions
    );
    validateRequirement(
      id,
      (!output.validity || output.validity.length === outputPixelCount) &&
        (!output.validCount || output.validCount.length === outputPixelCount) &&
        (!output.meanLength || output.meanLength.length === outputPixelCount),
      'overview outputs must hold ceil(sourceWidth / scale) times ceil(sourceHeight / scale) entries'
    );
    validateRasterViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterScalarView[] {
    const {vectors, quantized, mask} = this.props;
    return [
      ...(vectors ? [vectors] : []),
      ...(quantized ? [quantized] : []),
      ...(mask ? [mask] : [])
    ];
  }

  private getOutputs(): RasterScalarView[] {
    const {vectors, validity, validCount, meanLength} = this.props.output;
    return [
      vectors,
      ...(validity ? [validity] : []),
      ...(validCount ? [validCount] : []),
      ...(meanLength ? [meanLength] : [])
    ];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, width, height, horizontalScale, verticalScale} = this;
    const {sourceWidth, sourceHeight, dimensions, output} = props;
    validateGraphChunkBindingRanges(graph, id, {
      vectors: props.vectors,
      quantized: props.quantized,
      mask: props.mask,
      'output.vectors': output.vectors,
      'output.validity': output.validity,
      'output.validCount': output.validCount,
      'output.meanLength': output.meanLength
    });
    const source = props.vectors ?? props.quantized!;
    const sourceName = props.vectors ? 'vectors' : 'quantized';
    // Reads component `band` of source pixel `pixel`, or sets `pixelValid` false for NoData.
    const readComponent = props.vectors
      ? `let value = vectors[vectorsOffset + pixel * ${dimensions}u + band];
        if (!finite(value)) { pixelValid = false; }`
      : `let byte = readEmbeddingByte(pixel * ${dimensions}u + band);
        if (byte == -128) { pixelValid = false; }
        let value = getEmbeddingValue(byte);`;
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterEmbeddingPyramidLevel',
        variant: 'unit-vector-mean',
        invocationCount: width * height,
        workgroupSize: 64,
        bindings: [
          rasterBinding(sourceName, source),
          ...(props.mask ? [rasterBinding('mask', props.mask)] : []),
          rasterBinding('overview', output.vectors, true),
          ...(output.validity ? [rasterBinding('overviewValidity', output.validity, true)] : []),
          ...(output.validCount ? [rasterBinding('validCount', output.validCount, true)] : []),
          ...(output.meanLength ? [rasterBinding('meanLength', output.meanLength, true)] : [])
        ],
        declarations: `${RASTER_FINITE_WGSL}${props.quantized ? getRasterEmbeddingDecodeWGSL('quantized') : ''}`,
        body: `let outColumn = index % ${width}u;
let outRow = index / ${width}u;
var total: array<f32, ${dimensions}>;
var validPixels = 0u;
for (var row = outRow * ${verticalScale}u; row < min((outRow + 1u) * ${verticalScale}u, ${sourceHeight}u); row++) {
  for (var column = outColumn * ${horizontalScale}u; column < min((outColumn + 1u) * ${horizontalScale}u, ${sourceWidth}u); column++) {
    let pixel = row * ${sourceWidth}u + column;
    var pixelValid = ${props.mask ? 'mask[maskOffset + pixel] != 0u' : 'true'};
    var decoded: array<f32, ${dimensions}>;
    for (var band = 0u; band < ${dimensions}u; band++) {
      ${readComponent}
      decoded[band] = value;
    }
    if (pixelValid) {
      for (var band = 0u; band < ${dimensions}u; band++) { total[band] += decoded[band]; }
      validPixels++;
    }
  }
}
var squaredLength = 0.0;
for (var band = 0u; band < ${dimensions}u; band++) {
  let mean = total[band] / f32(max(validPixels, 1u));
  total[band] = mean;
  squaredLength += mean * mean;
}
let meanLengthValue = sqrt(squaredLength);
let isValid = validPixels > 0u && meanLengthValue > 0.0;
for (var band = 0u; band < ${dimensions}u; band++) {
  overview[overviewOffset + index * ${dimensions}u + band] = select(missing(), total[band] / meanLengthValue, isValid);
}
${output.validity ? 'overviewValidity[overviewValidityOffset + index] = select(0u, 1u, isValid);' : ''}
${output.validCount ? 'validCount[validCountOffset + index] = validPixels;' : ''}
${output.meanLength ? 'meanLength[meanLengthOffset + index] = select(missing(), meanLengthValue, validPixels > 0u);' : ''}`
      })
    ];
  }
}
