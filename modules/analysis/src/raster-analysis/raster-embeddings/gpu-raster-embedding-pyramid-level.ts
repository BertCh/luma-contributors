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
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateRequirement,
  validateSinglePackedView
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
  /** Unit-length overview vectors, `pixel * dimensions + band`, row-major over the overview; NaN for empty or exactly cancelling footprints. */
  vectors: GraphDataView<'float32'>;
  /** One `uint32` per overview pixel: 1 where the displayed vector is usable (nonempty and nonzero). */
  validity?: GraphDataView<'uint32'>;
  /** Valid original source pixels in each footprint, including support whose vectors cancel. */
  validCount?: GraphDataView<'uint32'>;
  /** Unnormalized vector sums, same layout as vectors; zero for empty or exactly cancelling footprints. Pair with validCount to build coarser levels. */
  vectorSums?: GraphDataView<'float32'>;
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
  /** Decoded float32 vectors. Give exactly one of this, `quantized`, or `aggregation`. */
  vectors?: GraphDataView<'float32'>;
  /**
   * Packed int8 AlphaEarth components, decoded in the kernel by `sign(q) * (q / 127.5)^2`; a pixel
   * with `-128` in any band is invalid. Give exactly one of this, `vectors`, or `aggregation`.
   */
  quantized?: GraphDataView<'uint32'>;
  /**
   * Previous-level unnormalized sums and original-pixel support counts. Give exactly one of
   * this, `vectors`, or `quantized`. Counts must sum to at most 2^32 - 1 per output footprint.
   * A zero-length child vector still contributes its count; do not mask with its vector validity.
   */
  aggregation?: {
    vectorSums: GraphDataView<'float32'>;
    validCount: GraphDataView<'uint32'>;
  };
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
 * Footprints at odd edges are partial, as in `GPURasterOverview`. Build each overview from the
 * original vectors, or chain `output.vectorSums` and `output.validCount` into `aggregation`.
 * Chaining normalized `output.vectors` changes pixel weights at odd edges, with masks, and when
 * child means have different lengths. Sums/counts preserve those weights (up to float32 rounding),
 * including the support of a child whose vectors cancel exactly. `validity` describes the
 * normalized vector only; count and sums remain usable even when that vector is invalid.
 * Normalization scales by the largest absolute sum component before squaring, so a tiny
 * representable nonzero sum is not mistaken for exact cancellation.
 *
 * Per-frame: the contents of the views. Compile-time: the sizes, `scale`, `dimensions` and which
 * optional views exist.
 * These compile-time values and port maps are captured at construction; view contents remain
 * per-frame inputs.
 */
export class GPURasterEmbeddingPyramidLevel implements GPUCommandNodeProducer {
  readonly id: string;
  /** Overview width in pixels. */
  readonly width: number;
  /** Overview height in pixels. */
  readonly height: number;
  private readonly horizontalScale: number;
  private readonly verticalScale: number;
  readonly props: GPURasterEmbeddingPyramidLevelProps;

  constructor(props: GPURasterEmbeddingPyramidLevelProps) {
    props = Object.freeze({
      ...snapshotConstructionFields(props, [
        'id',
        'sourceWidth',
        'sourceHeight',
        'dimensions',
        'scale',
        'vectors',
        'quantized',
        'aggregation',
        'mask',
        'output'
      ]),
      scale:
        typeof props.scale === 'number'
          ? props.scale
          : (Object.freeze([...props.scale]) as readonly [number, number]),
      aggregation:
        props.aggregation &&
        snapshotConstructionFields(props.aggregation, ['vectorSums', 'validCount']),
      output: snapshotConstructionFields(props.output, [
        'vectors',
        'vectorSums',
        'validity',
        'validCount',
        'meanLength'
      ])
    });
    this.props = props;
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
      Number(props.vectors !== undefined) +
        Number(props.quantized !== undefined) +
        Number(props.aggregation !== undefined) ===
        1,
      'needs exactly one of vectors, quantized, and aggregation'
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
    } else if (props.aggregation) {
      validateRasterEmbeddingVectors(
        id,
        'aggregation.vectorSums',
        props.aggregation.vectorSums,
        sourcePixelCount,
        dimensions,
        props.mask
      );
      validateSinglePackedView(
        props.aggregation.validCount,
        ['uint32'],
        `${id} aggregation.validCount`
      );
      validateRequirement(
        id,
        props.aggregation.validCount.length === sourcePixelCount,
        'aggregation.validCount must hold sourceWidth times sourceHeight entries'
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
    if (output.vectorSums) {
      validateRasterEmbeddingVectors(
        id,
        'output.vectorSums',
        output.vectorSums,
        outputPixelCount,
        dimensions
      );
    }
    validateRequirement(
      id,
      (!output.validity || output.validity.length === outputPixelCount) &&
        (!output.validCount || output.validCount.length === outputPixelCount) &&
        (!output.meanLength || output.meanLength.length === outputPixelCount),
      'overview outputs must hold ceil(sourceWidth / scale) times ceil(sourceHeight / scale) entries'
    );
    const inputs = this.getInputs();
    const outputs = this.getOutputs();
    validateRasterViews(id, inputs, outputs, [{inputs, outputs}]);
  }

  private getInputs(): RasterScalarView[] {
    const {vectors, quantized, aggregation, mask} = this.props;
    return [
      ...(vectors ? [vectors] : []),
      ...(quantized ? [quantized] : []),
      ...(aggregation ? [aggregation.vectorSums, aggregation.validCount] : []),
      ...(mask ? [mask] : [])
    ];
  }

  private getOutputs(): RasterScalarView[] {
    const {vectors, vectorSums, validity, validCount, meanLength} = this.props.output;
    return [
      vectors,
      ...(vectorSums ? [vectorSums] : []),
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
      'aggregation.vectorSums': props.aggregation?.vectorSums,
      'aggregation.validCount': props.aggregation?.validCount,
      mask: props.mask,
      'output.vectors': output.vectors,
      'output.vectorSums': output.vectorSums,
      'output.validity': output.validity,
      'output.validCount': output.validCount,
      'output.meanLength': output.meanLength
    });
    const source = props.vectors ?? props.aggregation?.vectorSums ?? props.quantized!;
    const sourceName = props.vectors ? 'vectors' : props.aggregation ? 'sourceSums' : 'quantized';
    // Reads component `band` of source pixel `pixel`, or sets `pixelValid` false for NoData.
    const readComponent =
      props.vectors || props.aggregation
        ? `let value = ${sourceName}[${sourceName}Offset + pixel * ${dimensions}u + band];
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
          ...(props.aggregation
            ? [rasterBinding('sourceValidCount', props.aggregation.validCount)]
            : []),
          ...(props.mask ? [rasterBinding('mask', props.mask)] : []),
          rasterBinding('overview', output.vectors, true),
          ...(output.vectorSums ? [rasterBinding('vectorSums', output.vectorSums, true)] : []),
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
    let pixelCount = ${props.aggregation ? 'sourceValidCount[sourceValidCountOffset + pixel]' : '1u'};
    var pixelValid = ${props.mask ? 'mask[maskOffset + pixel] != 0u' : 'true'};
    var decoded: array<f32, ${dimensions}>;
    for (var band = 0u; band < ${dimensions}u; band++) {
      ${readComponent}
      decoded[band] = value;
    }
    if (pixelValid && pixelCount > 0u) {
      for (var band = 0u; band < ${dimensions}u; band++) { total[band] += decoded[band]; }
      validPixels += pixelCount;
    }
  }
}
var largestComponent = 0.0;
for (var band = 0u; band < ${dimensions}u; band++) {
  ${output.vectorSums ? `vectorSums[vectorSumsOffset + index * ${dimensions}u + band] = total[band];` : ''}
  largestComponent = max(largestComponent, abs(total[band]));
}
let componentScale = select(1.0, largestComponent, largestComponent > 0.0);
var squaredLength = 0.0;
for (var band = 0u; band < ${dimensions}u; band++) {
  let scaled = total[band] / componentScale;
  squaredLength += scaled * scaled;
}
let scaledLength = sqrt(squaredLength);
let meanLengthValue = largestComponent * (scaledLength / f32(max(validPixels, 1u)));
let isValid = validPixels > 0u && largestComponent > 0.0 && finite(scaledLength);
for (var band = 0u; band < ${dimensions}u; band++) {
  overview[overviewOffset + index * ${dimensions}u + band] = select(missing(), (total[band] / componentScale) / max(scaledLength, 1.0), isValid);
}
${output.validity ? 'overviewValidity[overviewValidityOffset + index] = select(0u, 1u, isValid);' : ''}
${output.validCount ? 'validCount[validCountOffset + index] = validPixels;' : ''}
${output.meanLength ? 'meanLength[meanLengthOffset + index] = select(missing(), meanLengthValue, validPixels > 0u);' : ''}`
      })
    ];
  }
}
