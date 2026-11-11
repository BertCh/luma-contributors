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
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  FIXED_POINT_WGSL,
  getFixedPointAtomicAddWGSL
} from '../../analysis-core/kernels/fixed-point-accumulator';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  RASTER_FINITE_WGSL,
  rasterBinding,
  validateRasterGraph,
  validateRasterViews,
  type RasterScalarView
} from '../remote-sensing/raster-kernel-utils';
import {getRasterFixedPointStateWGSL} from '../raster-time-series/raster-fixed-point-state-wgsl';
import {getGPURasterFixedPointUnitExponent} from '../raster-time-series/gpu-raster-streaming-accumulator';
import {createRasterEmbeddingUsableMask} from './raster-embedding-usable-mask';
import {
  validateRasterEmbeddingShape,
  validateRasterEmbeddingVectors
} from './raster-embedding-common';

/**
 * Zones per dispatch. The fixed-point accumulator holds `(dimensions + 1)` 96-bit sums per zone,
 * so 65,536 zones of 128 bands are 101 MB, inside a 128 MiB storage binding.
 */
export const MAXIMUM_RASTER_EMBEDDING_ZONE_COUNT = 65536;

/** Zone id of a pixel in no zone. */
export const RASTER_EMBEDDING_NO_ZONE = 0xffffffff;

/** Pixel-to-zone assignments as compact tuples, the form `GPURasterFractionalCoverage` writes. */
export type GPURasterEmbeddingZoneTuples = {
  /** Zone of each tuple; a value at or above `zoneCount` is ignored. */
  zoneIds: GraphDataView<'uint32'>;
  /** Pixel of each tuple; a value at or above `pixelCount` is ignored. */
  cellIds: GraphDataView<'uint32'>;
  /** Coverage weight of each tuple, in `(0, 1]`. */
  fractions: GraphDataView<'float32'>;
  /** The `status` of `GPURasterFractionalCoverage`: `count` is the number of tuples written, at most the tuple capacity. */
  status: {count: GraphDataView<'uint32'>};
};

/** Caller-owned outputs of {@link GPURasterEmbeddingZoneMean}. */
export type GPURasterEmbeddingZoneMeanOutput = {
  /** Mean vector of each zone, `zone * dimensions + band`; NaN for a zone without usable pixels. */
  means: GraphDataView<'float32'>;
  /** Total coverage weight of the usable pixels of each zone. */
  weights?: GraphDataView<'float32'>;
  /**
   * Length of the weighted mean before renormalization, in `[0, 1]` for unit vectors: 1 when every
   * pixel of the zone points the same way, near 0 when they cancel. A coherence measure.
   */
  meanLengths?: GraphDataView<'float32'>;
};

/** Properties of {@link GPURasterEmbeddingZoneMean}. */
export type GPURasterEmbeddingZoneMeanProps = {
  id?: string;
  pixelCount: number;
  /** Vector length, in `[1, 128]`. */
  dimensions: number;
  /** Zones, in `[1, 65536]`. */
  zoneCount: number;
  vectors: GraphDataView<'float32'>;
  /** Per-pixel QA; zero excludes the pixel. A pixel with any non-finite component is excluded too. */
  mask?: GraphDataView<'uint32'>;
  /** One zone id per pixel (`0xffffffff` for none); weight one. Give this or `tuples`. */
  zoneRaster?: GraphDataView<'uint32'>;
  /** Weighted pixel-to-zone tuples. Give this or `zoneRaster`. */
  tuples?: GPURasterEmbeddingZoneTuples;
  /**
   * Largest absolute component of the vectors, which sets the fixed-point unit of the sums.
   * Defaults to 1, the bound of decoded AlphaEarth vectors; a larger component is not exact.
   */
  maximumComponentMagnitude?: number;
  /** Divide each mean by its length (the AlphaEarth rule and the right query for cosine search). Defaults to true. */
  normalize?: boolean;
  output: GPURasterEmbeddingZoneMeanOutput;
};

/**
 * Mean embedding vector of each zone, weighted by coverage and renormalized to unit length: the
 * descriptor of a polygon for similarity search and classification. Pixels come from a zone raster
 * (weight one) or from the exact fractional coverage tuples of `GPURasterFractionalCoverage`
 * (`zoneIds`, `cellIds`, `fractions`, `status.count`), so a border pixel shared by two polygons
 * contributes to each in proportion to its overlap.
 *
 * Sums are 96-bit fixed-point integers added with atomics, so the result does not depend on the
 * order of threads; the only rounding is the final conversion to float32. A zone whose usable
 * weight is zero gets NaN.
 *
 * Per-frame: the contents of the views. Compile-time: `pixelCount`, `dimensions`, `zoneCount`,
 * the tuple capacity, `normalize` and which optional views exist.
 */
export class GPURasterEmbeddingZoneMean implements GPUCommandNodeProducer {
  readonly id: string;
  private readonly unitExponent: number;

  constructor(readonly props: GPURasterEmbeddingZoneMeanProps) {
    const id = props.id ?? 'raster-embedding-zone-mean';
    this.id = id;
    const {pixelCount, dimensions, zoneCount, output} = props;
    validateRasterEmbeddingShape(id, pixelCount, dimensions);
    validateCapacity(id, 'zoneCount', zoneCount, 1, MAXIMUM_RASTER_EMBEDDING_ZONE_COUNT);
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
      (props.zoneRaster === undefined) !== (props.tuples === undefined),
      'needs exactly one of zoneRaster and tuples'
    );
    validateRequirement(
      id,
      !props.zoneRaster || props.zoneRaster.length === pixelCount,
      'zoneRaster must hold pixelCount entries'
    );
    if (props.tuples) {
      const {zoneIds, cellIds, fractions, status} = props.tuples;
      validateRequirement(
        id,
        zoneIds.length === cellIds.length &&
          zoneIds.length === fractions.length &&
          status.count.length >= 1,
        'tuples zoneIds, cellIds and fractions must have equal length and status.count one entry'
      );
      validateCapacity(id, 'tuple capacity', zoneIds.length);
      validateRequirement(
        id,
        zoneIds.length * dimensions <= 0xffffffff,
        'tuple capacity times dimensions must stay below 2^32'
      );
    }
    validateFiniteNumber(
      id,
      'maximumComponentMagnitude',
      props.maximumComponentMagnitude ?? 1,
      Number.MIN_VALUE
    );
    validateRequirement(
      id,
      output.means.length === zoneCount * dimensions &&
        (!output.weights || output.weights.length === zoneCount) &&
        (!output.meanLengths || output.meanLengths.length === zoneCount),
      'output.means must hold zoneCount times dimensions entries, weights and meanLengths zoneCount'
    );
    // Each pixel adds at most one term to a sum, and a tuple list adds one per tuple.
    this.unitExponent = getGPURasterFixedPointUnitExponent(
      Math.max(props.maximumComponentMagnitude ?? 1, 1),
      props.tuples ? props.tuples.zoneIds.length : pixelCount
    );
    validateRasterViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterScalarView[] {
    const {vectors, mask, zoneRaster, tuples} = this.props;
    return [
      vectors,
      ...(mask ? [mask] : []),
      ...(zoneRaster ? [zoneRaster] : []),
      ...(tuples ? [tuples.zoneIds, tuples.cellIds, tuples.fractions, tuples.status.count] : [])
    ];
  }

  private getOutputs(): RasterScalarView[] {
    const {means, weights, meanLengths} = this.props.output;
    return [means, ...(weights ? [weights] : []), ...(meanLengths ? [meanLengths] : [])];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, unitExponent} = this;
    const {pixelCount, dimensions, zoneCount, output, tuples, zoneRaster} = props;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    const operation = 'GPURasterEmbeddingZoneMean';
    const entryCount = zoneCount * (dimensions + 1);
    const accumulator = createTransientView(graph, `${id}-sums`, 'uint32', entryCount * 3);
    const {usable, node: usableNode} = createRasterEmbeddingUsableMask(graph, {
      id,
      operation,
      pixelCount,
      dimensions,
      vectors: props.vectors,
      mask: props.mask
    });
    const rawMeans = output.means;
    const termCount = tuples ? tuples.zoneIds.length : pixelCount;
    const accumulate = tuples
      ? createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-accumulate`,
          operation,
          variant: 'tuples',
          invocationCount: termCount * dimensions,
          workgroupSize: 64,
          bindings: [
            rasterBinding('vectors', props.vectors),
            rasterBinding('usable', usable),
            rasterBinding('zoneIds', tuples.zoneIds),
            rasterBinding('cellIds', tuples.cellIds),
            rasterBinding('fractions', tuples.fractions),
            rasterBinding('tupleCount', tuples.status.count),
            {name: 'sums', view: accumulator, type: 'atomic<u32>', access: 'read_write'}
          ],
          declarations: `${FIXED_POINT_WGSL}${getFixedPointAtomicAddWGSL('sums')}`,
          body: `let tupleIndex = index / ${dimensions}u;
let band = index % ${dimensions}u;
if (tupleIndex >= min(tupleCount[tupleCountOffset], ${termCount}u)) { return; }
let zone = zoneIds[zoneIdsOffset + tupleIndex];
let pixel = cellIds[cellIdsOffset + tupleIndex];
if (zone >= ${zoneCount}u || pixel >= ${pixelCount}u || usable[usableOffset + pixel] == 0u) { return; }
let fraction = fractions[fractionsOffset + tupleIndex];
addFixedPointToSums(zone * ${dimensions + 1}u + band, quantizeFixedPoint(fraction * vectors[vectorsOffset + pixel * ${dimensions}u + band], ${unitExponent}));
if (band == 0u) {
  addFixedPointToSums(zone * ${dimensions + 1}u + ${dimensions}u, quantizeFixedPoint(fraction, ${unitExponent}));
}`
        })
      : createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-accumulate`,
          operation,
          variant: 'zone-raster',
          invocationCount: termCount * dimensions,
          workgroupSize: 64,
          bindings: [
            rasterBinding('vectors', props.vectors),
            rasterBinding('usable', usable),
            rasterBinding('zones', zoneRaster!),
            {name: 'sums', view: accumulator, type: 'atomic<u32>', access: 'read_write'}
          ],
          declarations: `${FIXED_POINT_WGSL}${getFixedPointAtomicAddWGSL('sums')}`,
          body: `let pixel = index / ${dimensions}u;
let band = index % ${dimensions}u;
let zone = zones[zonesOffset + pixel];
if (zone >= ${zoneCount}u || usable[usableOffset + pixel] == 0u) { return; }
addFixedPointToSums(zone * ${dimensions + 1}u + band, quantizeFixedPoint(vectors[vectorsOffset + pixel * ${dimensions}u + band], ${unitExponent}));
if (band == 0u) {
  addFixedPointToSums(zone * ${dimensions + 1}u + ${dimensions}u, quantizeFixedPoint(1.0, ${unitExponent}));
}`
        });
    const nodes: GPUCommandNode<Parameters>[] = [
      usableNode,
      createFillNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation,
        view: accumulator,
        type: 'u32',
        value: '0u'
      }),
      accumulate,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-means`,
        operation,
        variant: 'means',
        invocationCount: zoneCount * dimensions,
        workgroupSize: 64,
        bindings: [
          rasterBinding('sums', accumulator),
          rasterBinding('means', rawMeans, true),
          ...(output.weights ? [rasterBinding('weights', output.weights, true)] : [])
        ],
        declarations: `${RASTER_FINITE_WGSL}${FIXED_POINT_WGSL}${getRasterFixedPointStateWGSL('sums', false)}`,
        body: `let zone = index / ${dimensions}u;
let band = index % ${dimensions}u;
let weight = getFixedPointFloat(loadSums(zone * ${dimensions + 1}u + ${dimensions}u), ${unitExponent});
let total = getFixedPointFloat(loadSums(zone * ${dimensions + 1}u + band), ${unitExponent});
means[meansOffset + index] = select(missing(), total / weight, weight > 0.0);
${output.weights ? `if (band == 0u) { weights[weightsOffset + zone] = weight; }` : ''}`
      })
    ];
    if (props.normalize !== false || output.meanLengths) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-normalize`,
          operation,
          variant: 'normalize',
          invocationCount: zoneCount,
          workgroupSize: 64,
          bindings: [
            rasterBinding('means', rawMeans, true),
            ...(output.meanLengths ? [rasterBinding('meanLengths', output.meanLengths, true)] : [])
          ],
          declarations: RASTER_FINITE_WGSL,
          body: `var squaredLength = 0.0;
for (var band = 0u; band < ${dimensions}u; band++) {
  let value = means[meansOffset + index * ${dimensions}u + band];
  squaredLength += value * value;
}
let meanLength = sqrt(squaredLength);
${output.meanLengths ? 'meanLengths[meanLengthsOffset + index] = select(missing(), meanLength, finite(meanLength));' : ''}
${
  props.normalize !== false
    ? `if (finite(meanLength) && meanLength > 0.0) {
  for (var band = 0u; band < ${dimensions}u; band++) {
    means[meansOffset + index * ${dimensions}u + band] /= meanLength;
  }
}`
    : ''
}`
        })
      );
    }
    return nodes;
  }
}
