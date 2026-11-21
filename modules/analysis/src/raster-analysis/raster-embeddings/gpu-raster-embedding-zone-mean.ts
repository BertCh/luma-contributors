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
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  FIXED_POINT_WGSL,
  getFixedPointAtomicAddWGSL
} from '../../analysis-core/kernels/fixed-point-accumulator';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  RASTER_FINITE_WGSL,
  rasterBinding,
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
  /**
   * The `status` of `GPURasterFractionalCoverage`: `count` is the number of tuples written, at most
   * the tuple capacity. Pass the producer's whole status: when `requiredCount`, `overflow` or
   * `candidateOverflow` are given, a frame where `requiredCount != count`, `overflow != 0` or
   * `candidateOverflow != 0` (the tuple list is a truncated prefix) suppresses every zone. A
   * `count` above the tuple capacity always suppresses every zone.
   */
  status: {
    count: GraphDataView<'uint32'>;
    requiredCount?: GraphDataView<'uint32'>;
    overflow?: GraphDataView<'uint32'>;
    candidateOverflow?: GraphDataView<'uint32'>;
  };
  /**
   * The `zoneValidity` of `GPURasterFractionalCoverage`, at least `zoneCount` entries: a zone
   * whose entry is zero (failed triangulation or unknown support) is suppressed.
   */
  zoneValidity?: GraphDataView<'uint32'>;
};

/** Caller-owned outputs of {@link GPURasterEmbeddingZoneMean}. */
export type GPURasterEmbeddingZoneMeanOutput = {
  /**
   * Mean vector of each zone, `zone * dimensions + band`; NaN for a zone without usable pixels
   * and for a zone suppressed by incomplete tuples (see {@link GPURasterEmbeddingZoneTuples}).
   */
  means: GraphDataView<'float32'>;
  /**
   * Total coverage weight of the usable pixels of each zone: 0 for a zone without usable pixels,
   * NaN for a zone suppressed by incomplete tuples.
   */
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
 * weight is zero gets NaN. With tuples, the producer's completeness status and zone validity are
 * honoured when given: a truncated tuple prefix suppresses every zone (NaN means, weights and
 * mean lengths) instead of publishing a partial mean. Normalization scales each mean by its
 * largest component before squaring, so a tiny nonzero mean still normalizes.
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
      const {zoneIds, cellIds, fractions, status, zoneValidity} = props.tuples;
      validateRequirement(
        id,
        zoneIds.length === cellIds.length &&
          zoneIds.length === fractions.length &&
          status.count.length >= 1,
        'tuples zoneIds, cellIds and fractions must have equal length and status.count one entry'
      );
      for (const name of ['requiredCount', 'overflow', 'candidateOverflow'] as const) {
        const view = status[name];
        validateRequirement(
          id,
          !view || view.length >= 1,
          `tuples.status.${name} must hold one entry`
        );
      }
      validateRequirement(
        id,
        !zoneValidity || zoneValidity.length >= zoneCount,
        `tuples.zoneValidity must hold at least zoneCount (${zoneCount}) entries`
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
    validateRasterViews(id, this.getInputs(), this.getOutputs(), [
      {
        inputs: props.tuples?.zoneValidity ? [props.tuples.zoneValidity] : [],
        outputs: [output.means, ...(output.weights ? [output.weights] : [])]
      },
      {inputs: [], outputs: [output.means, ...(output.meanLengths ? [output.meanLengths] : [])]}
    ]);
  }

  private getInputs(): RasterScalarView[] {
    const {vectors, mask, zoneRaster, tuples} = this.props;
    return [
      vectors,
      ...(mask ? [mask] : []),
      ...(zoneRaster ? [zoneRaster] : []),
      ...(tuples
        ? [tuples.zoneIds, tuples.cellIds, tuples.fractions, ...getTupleCompletenessViews(tuples)]
        : [])
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
    const entryCount = zoneCount * (dimensions + 1);
    validateGraphChunkBindingRanges(graph, id, {
      vectors: props.vectors,
      mask: props.mask,
      zoneRaster,
      'tuples.zoneIds': tuples?.zoneIds,
      'tuples.cellIds': tuples?.cellIds,
      'tuples.fractions': tuples?.fractions,
      'tuples.status.count': tuples?.status.count,
      'tuples.status.requiredCount': tuples?.status.requiredCount,
      'tuples.status.overflow': tuples?.status.overflow,
      'tuples.status.candidateOverflow': tuples?.status.candidateOverflow,
      'tuples.zoneValidity': tuples?.zoneValidity,
      'output.means': output.means,
      'output.weights': output.weights,
      'output.meanLengths': output.meanLengths
    });
    // Both derived storage views start at byte zero; createTransientView requires Buffer.STORAGE.
    for (const [name, elementCount] of Object.entries({
      'accumulator scratch': entryCount * 3,
      'usable scratch': pixelCount
    })) {
      const byteLength = elementCount * Uint32Array.BYTES_PER_ELEMENT;
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        byteLength,
        'reduce zoneCount, dimensions or pixelCount'
      );
      validateCapacity(id, `${name} byteLength`, byteLength, 1, graph.device.limits.maxBufferSize);
    }
    const operation = 'GPURasterEmbeddingZoneMean';
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
    // One word: 1 when the tuple list is the producer's complete list for this frame.
    const complete = tuples ? createTransientView(graph, `${id}-complete`, 'uint32', 1) : null;
    const completeNode =
      tuples && complete
        ? createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-complete`,
            operation,
            variant: 'tuple-completeness',
            invocationCount: 1,
            bindings: [
              rasterBinding('tupleCount', tuples.status.count),
              ...(tuples.status.requiredCount
                ? [rasterBinding('requiredCount', tuples.status.requiredCount)]
                : []),
              ...(tuples.status.overflow
                ? [rasterBinding('overflow', tuples.status.overflow)]
                : []),
              ...(tuples.status.candidateOverflow
                ? [rasterBinding('candidateOverflow', tuples.status.candidateOverflow)]
                : []),
              rasterBinding('complete', complete, true)
            ],
            body: `let count = tupleCount[tupleCountOffset];
complete[completeOffset] = u32(count <= ${termCount}u${
              tuples.status.requiredCount ? ' && requiredCount[requiredCountOffset] == count' : ''
            }${tuples.status.overflow ? ' && overflow[overflowOffset] == 0u' : ''}${
              tuples.status.candidateOverflow
                ? ' && candidateOverflow[candidateOverflowOffset] == 0u'
                : ''
            });`
          })
        : null;
    const zoneValidity = tuples?.zoneValidity;
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
      ...(completeNode ? [completeNode] : []),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-means`,
        operation,
        variant: complete ? 'means-guarded' : 'means',
        invocationCount: zoneCount * dimensions,
        workgroupSize: 64,
        bindings: [
          rasterBinding('sums', accumulator),
          ...(complete ? [rasterBinding('complete', complete)] : []),
          ...(zoneValidity ? [rasterBinding('zoneValidity', zoneValidity)] : []),
          rasterBinding('means', rawMeans, true),
          ...(output.weights ? [rasterBinding('weights', output.weights, true)] : [])
        ],
        declarations: `${RASTER_FINITE_WGSL}${FIXED_POINT_WGSL}${getRasterFixedPointStateWGSL('sums', false)}`,
        body: `let zone = index / ${dimensions}u;
let band = index % ${dimensions}u;
let supported = ${complete ? 'complete[completeOffset] != 0u' : 'true'}${
          zoneValidity ? ' && zoneValidity[zoneValidityOffset + zone] != 0u' : ''
        };
let weight = getFixedPointFloat(loadSums(zone * ${dimensions + 1}u + ${dimensions}u), ${unitExponent});
let total = getFixedPointFloat(loadSums(zone * ${dimensions + 1}u + band), ${unitExponent});
means[meansOffset + index] = select(missing(), total / weight, supported && weight > 0.0);
${output.weights ? `if (band == 0u) { weights[weightsOffset + zone] = select(missing(), weight, supported); }` : ''}`
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
          // Scale by the largest component before squaring: the square of a tiny nonzero mean
          // underflows float32, which would leave it unnormalized with a zero length.
          body: `var largest = 0.0;
var finiteMean = true;
for (var band = 0u; band < ${dimensions}u; band++) {
  let value = means[meansOffset + index * ${dimensions}u + band];
  finiteMean = finiteMean && finite(value);
  largest = max(largest, abs(select(0.0, value, finite(value))));
}
var squaredScaledLength = 0.0;
if (finiteMean && largest > 0.0) {
  for (var band = 0u; band < ${dimensions}u; band++) {
    let scaled = means[meansOffset + index * ${dimensions}u + band] / largest;
    squaredScaledLength += scaled * scaled;
  }
}
let scaledLength = sqrt(squaredScaledLength);
let meanLength = largest * scaledLength;
let definedLength = finiteMean && finite(meanLength);
${output.meanLengths ? 'meanLengths[meanLengthsOffset + index] = select(missing(), meanLength, definedLength);' : ''}
${
  props.normalize !== false
    ? `if (definedLength && largest > 0.0) {
  for (var band = 0u; band < ${dimensions}u; band++) {
    let offset = meansOffset + index * ${dimensions}u + band;
    means[offset] = (means[offset] / largest) / scaledLength;
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

/** Tuple views that carry the producer's count, completeness and zone validity. */
function getTupleCompletenessViews(
  tuples: GPURasterEmbeddingZoneTuples
): GraphDataView<'uint32'>[] {
  const {count, requiredCount, overflow, candidateOverflow} = tuples.status;
  return [count, requiredCount, overflow, candidateOverflow, tuples.zoneValidity].filter(
    (view): view is GraphDataView<'uint32'> => view !== undefined
  );
}
