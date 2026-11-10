// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSimilaritySearch} from '@luma.gl/graph/gpu-vector-search';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {MAXIMUM_RASTER_EMBEDDING_QUERY_COUNT} from './gpu-raster-embedding-similarity';
import {
  createRasterEmbeddingMatrix,
  validateRasterEmbeddingShape,
  validateRasterEmbeddingVectors
} from './raster-embedding-common';

/** Caller-owned outputs of {@link GPURasterEmbeddingNearestPixels}. */
export type GPURasterEmbeddingNearestPixelsOutput = {
  /** Pixel indices of the best matches, `query * resultCount + rank`; `0xffffffff` where fewer than `resultCount` pixels exist. */
  pixelIds: GraphDataView<'uint32'>;
  /** Cosine similarity of each match, `query * resultCount + rank`, best first. */
  scores: GraphDataView<'float32'>;
  /** Matches found per query. */
  resultCounts: GraphDataView<'uint32'>;
};

/** Properties of {@link GPURasterEmbeddingNearestPixels}. */
export type GPURasterEmbeddingNearestPixelsProps = {
  id?: string;
  pixelCount: number;
  /** Vector length, in `[1, 128]`. */
  dimensions: number;
  vectors: GraphDataView<'float32'>;
  /** Per-pixel QA; zero excludes the pixel from the candidates. A pixel with a non-finite component is excluded too. */
  validity?: GraphDataView<'uint32'>;
  /** Number of query vectors, in `[1, 64]`. */
  queryCount: number;
  /** Query vectors, `query * dimensions + band`. */
  queries: GraphDataView<'float32'>;
  /** Matches per query. */
  resultCount: number;
  output: GPURasterEmbeddingNearestPixelsOutput;
};

/**
 * The `resultCount` pixels of an embedding raster most similar to each query, ranked by cosine
 * similarity: "find places like this one". It runs the published exact `GPUSimilaritySearch` of
 * `@luma.gl/graph/gpu-vector-search` over the raster's vectors, so equal scores order by pixel
 * index and nothing like a query-by-pixel distance matrix is stored. For a similarity value at
 * every pixel (a heat map) use {@link GPURasterEmbeddingSimilarity}.
 *
 * Per-frame: the contents of the views. Compile-time: `pixelCount`, `dimensions`, `queryCount`,
 * `resultCount`.
 */
export class GPURasterEmbeddingNearestPixels {
  readonly id: string;
  private readonly search: GPUSimilaritySearch;

  constructor(readonly props: GPURasterEmbeddingNearestPixelsProps) {
    const id = props.id ?? 'raster-embedding-nearest-pixels';
    this.id = id;
    const {pixelCount, dimensions, output} = props;
    validateRasterEmbeddingShape(id, pixelCount, dimensions);
    validateRasterEmbeddingVectors(
      id,
      'vectors',
      props.vectors,
      pixelCount,
      dimensions,
      props.validity
    );
    validateCapacity(id, 'queryCount', props.queryCount, 1, MAXIMUM_RASTER_EMBEDDING_QUERY_COUNT);
    validateCapacity(id, 'resultCount', props.resultCount);
    validateRequirement(
      id,
      props.queries.length === props.queryCount * dimensions,
      'queries must hold queryCount times dimensions entries'
    );
    this.search = new GPUSimilaritySearch({
      id,
      dataset: createRasterEmbeddingMatrix(props.vectors, pixelCount, dimensions, props.validity),
      queries: createRasterEmbeddingMatrix(props.queries, props.queryCount, dimensions),
      outputIds: output.pixelIds,
      outputScores: output.scores,
      resultCounts: output.resultCounts,
      k: props.resultCount,
      metric: 'cosine'
    });
  }

  /** Adds the published search passes. */
  addToGraph<Parameters>(graph: GPUCommandGraph<Parameters>): void {
    this.search.addToGraph(graph);
  }
}
