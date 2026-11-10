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
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  RASTER_FINITE_WGSL,
  rasterBinding,
  validateRasterGraph,
  validateRasterViews,
  type RasterScalarView
} from '../remote-sensing/raster-kernel-utils';
import {createRasterEmbeddingUsableMask} from './raster-embedding-usable-mask';
import {
  validateRasterEmbeddingShape,
  validateRasterEmbeddingVectors
} from './raster-embedding-common';

/**
 * Pixel ranges reduced in parallel. Each range is summed in order by one thread per statistic, so
 * the result is the same on every run; more ranges shorten the serial run of each thread and
 * enlarge the partial covariance (`ranges * dimensions^2` floats).
 */
export const MAXIMUM_RASTER_EMBEDDING_COVARIANCE_RANGE_COUNT = 256;

/** Caller-owned outputs of {@link GPURasterEmbeddingCovariance}. */
export type GPURasterEmbeddingCovarianceOutput = {
  /** Mean vector of the usable pixels, `dimensions` entries; NaN when no pixel is usable. */
  mean: GraphDataView<'float32'>;
  /** Sample covariance (divisor `n - 1`), `dimensions x dimensions`, row-major and symmetric. */
  covariance: GraphDataView<'float32'>;
  /** One `uint32`: the number of usable pixels `n`. */
  usableCount?: GraphDataView<'uint32'>;
};

/** Properties of {@link GPURasterEmbeddingCovariance}. */
export type GPURasterEmbeddingCovarianceProps = {
  id?: string;
  pixelCount: number;
  /** Vector length, in `[1, 128]`. */
  dimensions: number;
  vectors: GraphDataView<'float32'>;
  /** Per-pixel QA; zero excludes the pixel. A pixel with any non-finite component is excluded too. */
  validity?: GraphDataView<'uint32'>;
  /** Pixel ranges, in `[1, 256]`; defaults to `min(256, pixelCount)`. */
  rangeCount?: number;
  output: GPURasterEmbeddingCovarianceOutput;
};

/**
 * Mean vector and sample covariance of an embedding raster over its usable pixels, the GPU half
 * of PCA by the covariance method. The covariance is two-pass (the mean first, then centred
 * cross products) in fixed-order partial sums, so it is reproducible and does not lose precision
 * to a large mean. Feed it to `getRasterEmbeddingPrincipalComponents` on the CPU (a
 * `dimensions x dimensions` eigenproblem, milliseconds) and project with
 * {@link GPURasterEmbeddingProjection}.
 *
 * Cost: two passes of `pixelCount x dimensions` for the mean and `pixelCount x dimensions^2 / 2`
 * for the covariance; 1M pixels of 64 bands is about 2e9 multiply-adds.
 *
 * Per-frame: the contents of the views. Compile-time: `pixelCount`, `dimensions`, `rangeCount`
 * and which optional views exist.
 */
export class GPURasterEmbeddingCovariance implements GPUCommandNodeProducer {
  readonly id: string;
  readonly rangeCount: number;

  constructor(readonly props: GPURasterEmbeddingCovarianceProps) {
    const id = props.id ?? 'raster-embedding-covariance';
    this.id = id;
    const {pixelCount, dimensions, output} = props;
    validateRasterEmbeddingShape(id, pixelCount, dimensions);
    this.rangeCount =
      props.rangeCount ?? Math.min(MAXIMUM_RASTER_EMBEDDING_COVARIANCE_RANGE_COUNT, pixelCount);
    validateCapacity(
      id,
      'rangeCount',
      this.rangeCount,
      1,
      MAXIMUM_RASTER_EMBEDDING_COVARIANCE_RANGE_COUNT
    );
    validateRasterEmbeddingVectors(
      id,
      'vectors',
      props.vectors,
      pixelCount,
      dimensions,
      props.validity
    );
    validateRequirement(
      id,
      output.mean.length === dimensions &&
        output.covariance.length === dimensions * dimensions &&
        (!output.usableCount || output.usableCount.length === 1),
      'output.mean must hold dimensions entries, output.covariance dimensions squared and output.usableCount one'
    );
    validateRasterViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterScalarView[] {
    const {vectors, validity} = this.props;
    return [vectors, ...(validity ? [validity] : [])];
  }

  private getOutputs(): RasterScalarView[] {
    const {mean, covariance, usableCount} = this.props.output;
    return [mean, covariance, ...(usableCount ? [usableCount] : [])];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rangeCount} = this;
    const {pixelCount, dimensions, output} = props;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    const operation = 'GPURasterEmbeddingCovariance';
    const pixelsPerRange = Math.ceil(pixelCount / rangeCount);
    const {usable, node: usableNode} = createRasterEmbeddingUsableMask(graph, {
      id,
      operation,
      pixelCount,
      dimensions,
      vectors: props.vectors,
      validity: props.validity
    });
    const partialSums = createTransientView(
      graph,
      `${id}-partial-sums`,
      'float32',
      rangeCount * dimensions
    );
    const partialCounts = createTransientView(graph, `${id}-partial-counts`, 'uint32', rangeCount);
    const partialCovariance = createTransientView(
      graph,
      `${id}-partial-covariance`,
      'float32',
      rangeCount * dimensions * dimensions
    );
    const mean = output.mean;
    const usableCount =
      output.usableCount ?? createTransientView(graph, `${id}-usable-count`, 'uint32', 1);
    const rangeStart = `let rangeStart = range * ${pixelsPerRange}u;
let rangeEnd = min(rangeStart + ${pixelsPerRange}u, ${pixelCount}u);`;
    return [
      usableNode,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-partial-sums`,
        operation,
        variant: 'partial-sums',
        invocationCount: rangeCount * dimensions,
        workgroupSize: 64,
        bindings: [
          rasterBinding('vectors', props.vectors),
          rasterBinding('usable', usable),
          rasterBinding('partialSums', partialSums, true),
          rasterBinding('partialCounts', partialCounts, true)
        ],
        body: `let range = index / ${dimensions}u;
let band = index % ${dimensions}u;
${rangeStart}
var total = 0.0;
var count = 0u;
for (var pixel = rangeStart; pixel < rangeEnd; pixel++) {
  if (usable[usableOffset + pixel] != 0u) {
    total += vectors[vectorsOffset + pixel * ${dimensions}u + band];
    count++;
  }
}
partialSums[partialSumsOffset + index] = total;
if (band == 0u) { partialCounts[partialCountsOffset + range] = count; }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-mean`,
        operation,
        variant: 'mean',
        invocationCount: dimensions,
        workgroupSize: 64,
        declarations: RASTER_FINITE_WGSL,
        bindings: [
          rasterBinding('partialSums', partialSums),
          rasterBinding('partialCounts', partialCounts),
          rasterBinding('mean', mean, true),
          rasterBinding('usableCount', usableCount, true)
        ],
        body: `var total = 0.0;
var count = 0u;
for (var range = 0u; range < ${rangeCount}u; range++) {
  total += partialSums[partialSumsOffset + range * ${dimensions}u + index];
  count += partialCounts[partialCountsOffset + range];
}
mean[meanOffset + index] = select(missing(), total / f32(count), count > 0u);
if (index == 0u) { usableCount[usableCountOffset] = count; }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-partial-covariance`,
        operation,
        variant: 'partial-covariance',
        invocationCount: rangeCount * dimensions * dimensions,
        workgroupSize: 64,
        bindings: [
          rasterBinding('vectors', props.vectors),
          rasterBinding('usable', usable),
          rasterBinding('mean', mean),
          rasterBinding('partialCovariance', partialCovariance, true)
        ],
        body: `let range = index / ${dimensions * dimensions}u;
let pair = index % ${dimensions * dimensions}u;
let row = pair / ${dimensions}u;
let column = pair % ${dimensions}u;
${rangeStart}
var total = 0.0;
// Only the upper triangle is computed; the reduction mirrors it.
if (column >= row) {
  let rowMean = mean[meanOffset + row];
  let columnMean = mean[meanOffset + column];
  for (var pixel = rangeStart; pixel < rangeEnd; pixel++) {
    if (usable[usableOffset + pixel] != 0u) {
      total += (vectors[vectorsOffset + pixel * ${dimensions}u + row] - rowMean) *
        (vectors[vectorsOffset + pixel * ${dimensions}u + column] - columnMean);
    }
  }
}
partialCovariance[partialCovarianceOffset + index] = total;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-covariance`,
        operation,
        variant: 'covariance',
        invocationCount: dimensions * dimensions,
        workgroupSize: 64,
        declarations: RASTER_FINITE_WGSL,
        bindings: [
          rasterBinding('partialCovariance', partialCovariance),
          rasterBinding('usableCount', usableCount),
          rasterBinding('covariance', output.covariance, true)
        ],
        body: `let row = min(index / ${dimensions}u, index % ${dimensions}u);
let column = max(index / ${dimensions}u, index % ${dimensions}u);
var total = 0.0;
for (var range = 0u; range < ${rangeCount}u; range++) {
  total += partialCovariance[partialCovarianceOffset + range * ${dimensions * dimensions}u + row * ${dimensions}u + column];
}
let count = usableCount[usableCountOffset];
covariance[covarianceOffset + index] = select(missing(), total / f32(count - 1u), count > 1u);`
      })
    ];
  }
}
