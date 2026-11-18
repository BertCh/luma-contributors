// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView, GPUCommandGraph, GPUCommandNode} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {GPUKMeans} from '@luma.gl/graph/gpu-vector-search';
import {
  captureGraphCommandNodes,
  validateCapacity
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createRasterEmbeddingMatrix,
  validateRasterEmbeddingShape,
  validateRasterEmbeddingVectors
} from './raster-embedding-common';

/** Largest `maximumIterations`: every Lloyd iteration adds assignment and reduction passes to the graph. */
export const MAXIMUM_RASTER_EMBEDDING_KMEANS_ITERATIONS = 64;

/** Caller-owned outputs of {@link GPURasterEmbeddingKMeans}. */
export type GPURasterEmbeddingKMeansOutput = {
  /** Cluster of each pixel in `[0, clusterCount)`; `0xffffffff` for invalid pixels. */
  labels: GraphDataView<'uint32'>;
  /** Cluster centres, `cluster * dimensions + band`. An empty cluster keeps its previous centre. */
  centroids: GraphDataView<'float32'>;
  /** Pixels in each cluster, `clusterCount` entries. */
  counts: GraphDataView<'uint32'>;
  /** Optional `[executedIterations, changedLabels, converged]`. */
  status?: GraphDataView<'uint32'>;
};

/** Properties of {@link GPURasterEmbeddingKMeans}. */
export type GPURasterEmbeddingKMeansProps = {
  id?: string;
  pixelCount: number;
  /** Vector length, in `[1, 128]`. */
  dimensions: number;
  vectors: GraphDataView<'float32'>;
  /** Per-pixel QA; zero excludes the pixel. A pixel with a non-finite component is excluded too. */
  mask?: GraphDataView<'uint32'>;
  /** Number of clusters, at least 1. */
  clusterCount: number;
  /** Lloyd iterations encoded in the graph, in `[1, 64]`; later ones do nothing once labels stop changing. Defaults to 10. */
  maximumIterations?: number;
  output: GPURasterEmbeddingKMeansOutput;
};

/**
 * Unsupervised clustering of an embedding raster: k-means (Lloyd) on the vectors, as Google
 * suggests for AlphaEarth. It runs the published `GPUKMeans` of `@luma.gl/graph/gpu-vector-search`
 * on the raster's vectors, so seeding is that class's deterministic evenly spaced valid rows
 * (not k-means++) and the centres are Euclidean means. Normalize the vectors first
 * (`GPURasterEmbeddingDequantize` with `normalize`) for a clustering by direction. The 2-D
 * `GPUSpatialKMeans` of `gpu-spatial-analysis` is a different contract for planar points.
 *
 * Per-frame: the contents of the views. Compile-time: `pixelCount`, `dimensions`, `clusterCount`,
 * `maximumIterations` and which optional views exist.
 */
export class GPURasterEmbeddingKMeans implements GPUCommandNodeProducer {
  readonly id: string;
  private readonly clustering: GPUKMeans;
  private readonly commandNodes = new WeakMap<object, readonly GPUCommandNode[]>();

  constructor(readonly props: GPURasterEmbeddingKMeansProps) {
    const id = props.id ?? 'raster-embedding-kmeans';
    this.id = id;
    const {pixelCount, dimensions, output} = props;
    validateRasterEmbeddingShape(id, pixelCount, dimensions);
    validateRasterEmbeddingVectors(
      id,
      'vectors',
      props.vectors,
      pixelCount,
      dimensions,
      props.mask
    );
    validateCapacity(id, 'clusterCount', props.clusterCount);
    validateCapacity(
      id,
      'maximumIterations',
      props.maximumIterations ?? 10,
      1,
      MAXIMUM_RASTER_EMBEDDING_KMEANS_ITERATIONS
    );
    this.clustering = new GPUKMeans({
      id,
      dataset: createRasterEmbeddingMatrix(props.vectors, pixelCount, dimensions, props.mask),
      clusterCount: props.clusterCount,
      centroids: output.centroids,
      labels: output.labels,
      counts: output.counts,
      status: output.status,
      maxIterations: props.maximumIterations
    });
  }

  /** Returns the published k-means passes without scheduling them. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const cached = this.commandNodes.get(graph);
    if (cached) return cached as readonly GPUCommandNode<Parameters>[];
    const nodes = Object.freeze(
      captureGraphCommandNodes(graph, () => this.clustering.addToGraph(graph))
    );
    this.commandNodes.set(graph, nodes as readonly GPUCommandNode[]);
    return nodes;
  }
}
