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
import {
  RASTER_EMBEDDING_NO_LABEL,
  validateRasterEmbeddingShape,
  validateRasterEmbeddingVectors
} from './raster-embedding-common';

/** Queries per dispatch: each pixel thread loops over every query, so the count is a per-thread bound. */
export const MAXIMUM_RASTER_EMBEDDING_QUERY_COUNT = 64;

/** Float32 elements of the per-frame parameters: the match threshold. */
export const GPU_RASTER_EMBEDDING_SIMILARITY_PARAMETER_LENGTH = 1;

/** Per-frame settings of {@link GPURasterEmbeddingSimilarity}. */
export type GPURasterEmbeddingSimilaritySettings = {
  /** A pixel matches when its best similarity is at least this value. */
  matchThreshold: number;
};

/** Packs {@link GPURasterEmbeddingSimilaritySettings}: `[matchThreshold]`. */
export function getGPURasterEmbeddingSimilarityParameterValues(
  settings: GPURasterEmbeddingSimilaritySettings,
  target: Float32Array = new Float32Array(GPU_RASTER_EMBEDDING_SIMILARITY_PARAMETER_LENGTH)
): Float32Array {
  validateFiniteNumber(
    'getGPURasterEmbeddingSimilarityParameterValues',
    'matchThreshold',
    settings.matchThreshold
  );
  target[0] = settings.matchThreshold;
  return target;
}

/** Caller-owned outputs of {@link GPURasterEmbeddingSimilarity}; at least one is required. */
export type GPURasterEmbeddingSimilarityOutput = {
  /** Similarity of every pixel to every query, `pixel * queryCount + query`; NaN for invalid pixels. */
  similarity?: GraphDataView<'float32'>;
  /** Largest similarity over the queries; NaN for invalid pixels. */
  maximum?: GraphDataView<'float32'>;
  /** Index of the most similar query (the lowest on ties); `0xffffffff` for invalid pixels. */
  bestQuery?: GraphDataView<'uint32'>;
  /** 1 where `maximum >= matchThreshold`, else 0. Needs `parameters`. */
  matches?: GraphDataView<'uint32'>;
};

/** Properties of {@link GPURasterEmbeddingSimilarity}. */
export type GPURasterEmbeddingSimilarityProps = {
  id?: string;
  pixelCount: number;
  /** Vector length, in `[1, 128]`. */
  dimensions: number;
  /** Decoded vectors, `pixel * dimensions + band`; a pixel with a non-finite component is invalid. */
  vectors: GraphDataView<'float32'>;
  /** Per-pixel QA; zero makes the pixel invalid. */
  validity?: GraphDataView<'uint32'>;
  /** Number of query vectors, in `[1, 64]`. */
  queryCount: number;
  /**
   * Query vectors, `query * dimensions + band`. Take them from clicked pixels or from
   * {@link GPURasterEmbeddingZoneMean} (mean of a polygon, renormalized). Contents are per-frame.
   */
  queries: GraphDataView<'float32'>;
  /**
   * `'cosine'` (default) divides by both lengths; `'dot'` is the inner product, which for unit
   * vectors such as AlphaEarth's is the cosine without the divisions.
   */
  metric?: 'cosine' | 'dot';
  /** Float32, see {@link getGPURasterEmbeddingSimilarityParameterValues}. Needed with `output.matches`. */
  parameters?: GraphDataView<'float32'>;
  output: GPURasterEmbeddingSimilarityOutput;
};

/**
 * Similarity of every pixel of an embedding raster to up to 64 query vectors: cosine or dot
 * product, the best query per pixel, the maximum over the queries and a threshold mask. This is
 * the raster form of similarity search; for the pixels nearest to a query as a ranked list use
 * {@link GPURasterEmbeddingNearestPixels}.
 *
 * Cosine follows the published `GPUSimilaritySearch`: two zero vectors have similarity one and
 * exactly one zero vector gives zero. Queries must be finite.
 *
 * Per-frame: `parameters` and the contents of the views. Compile-time: `pixelCount`,
 * `dimensions`, `queryCount`, `metric` and which optional views exist.
 */
export class GPURasterEmbeddingSimilarity implements GPUCommandNodeProducer {
  readonly id: string;

  constructor(readonly props: GPURasterEmbeddingSimilarityProps) {
    const id = props.id ?? 'raster-embedding-similarity';
    this.id = id;
    const {pixelCount, dimensions, output} = props;
    validateRasterEmbeddingShape(id, pixelCount, dimensions);
    validateCapacity(id, 'queryCount', props.queryCount, 1, MAXIMUM_RASTER_EMBEDDING_QUERY_COUNT);
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
      props.queries.length === props.queryCount * dimensions,
      'queries must hold queryCount times dimensions entries'
    );
    validateRequirement(
      id,
      Boolean(output.similarity || output.maximum || output.bestQuery || output.matches),
      'output needs at least one view'
    );
    validateRequirement(
      id,
      (!output.similarity || output.similarity.length === pixelCount * props.queryCount) &&
        (!output.maximum || output.maximum.length === pixelCount) &&
        (!output.bestQuery || output.bestQuery.length === pixelCount) &&
        (!output.matches || output.matches.length === pixelCount),
      'output.similarity must hold pixelCount times queryCount entries and the other outputs pixelCount'
    );
    validateRequirement(
      id,
      !output.matches ||
        (props.parameters !== undefined &&
          props.parameters.length >= GPU_RASTER_EMBEDDING_SIMILARITY_PARAMETER_LENGTH),
      'output.matches needs parameters'
    );
    validateRasterViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterScalarView[] {
    const {vectors, validity, queries, parameters} = this.props;
    return [vectors, queries, ...(validity ? [validity] : []), ...(parameters ? [parameters] : [])];
  }

  private getOutputs(): RasterScalarView[] {
    const {similarity, maximum, bestQuery, matches} = this.props.output;
    return [similarity, maximum, bestQuery, matches].filter(
      (view): view is NonNullable<typeof view> => view !== undefined
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {pixelCount, dimensions, queryCount, output} = props;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    const isCosine = (props.metric ?? 'cosine') === 'cosine';
    const writeInvalid = `${output.similarity ? `for (var query = 0u; query < ${queryCount}u; query++) { similarity[similarityOffset + index * ${queryCount}u + query] = missing(); }` : ''}
${output.maximum ? 'maximum[maximumOffset + index] = missing();' : ''}
${output.bestQuery ? `bestQuery[bestQueryOffset + index] = ${RASTER_EMBEDDING_NO_LABEL}u;` : ''}
${output.matches ? 'matches[matchesOffset + index] = 0u;' : ''}`;
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterEmbeddingSimilarity',
        variant: isCosine ? 'cosine' : 'dot',
        invocationCount: pixelCount,
        workgroupSize: 64,
        bindings: [
          rasterBinding('vectors', props.vectors),
          rasterBinding('queries', props.queries),
          ...(props.validity ? [rasterBinding('validity', props.validity)] : []),
          ...(props.parameters ? [rasterBinding('parameters', props.parameters)] : []),
          ...(output.similarity ? [rasterBinding('similarity', output.similarity, true)] : []),
          ...(output.maximum ? [rasterBinding('maximum', output.maximum, true)] : []),
          ...(output.bestQuery ? [rasterBinding('bestQuery', output.bestQuery, true)] : []),
          ...(output.matches ? [rasterBinding('matches', output.matches, true)] : [])
        ],
        declarations: RASTER_FINITE_WGSL,
        body: `var vector: array<f32, ${dimensions}>;
var squaredLength = 0.0;
var isValid = ${props.validity ? 'validity[validityOffset + index] != 0u' : 'true'};
for (var band = 0u; band < ${dimensions}u; band++) {
  let value = vectors[vectorsOffset + index * ${dimensions}u + band];
  if (!finite(value)) { isValid = false; }
  vector[band] = value;
  squaredLength += value * value;
}
if (!isValid) {
  ${writeInvalid}
  return;
}
var best = -3.0e38;
var bestIndex = 0u;
for (var query = 0u; query < ${queryCount}u; query++) {
  var dotProduct = 0.0;
  var queryLength = 0.0;
  for (var band = 0u; band < ${dimensions}u; band++) {
    let component = queries[queriesOffset + query * ${dimensions}u + band];
    dotProduct += vector[band] * component;
    queryLength += component * component;
  }
  ${
    isCosine
      ? `let denominator = sqrt(squaredLength * queryLength);
  let similarityValue = select(select(0.0, 1.0, squaredLength == 0.0 && queryLength == 0.0), dotProduct / denominator, denominator > 0.0);`
      : 'let similarityValue = dotProduct;'
  }
  ${output.similarity ? `similarity[similarityOffset + index * ${queryCount}u + query] = similarityValue;` : ''}
  if (similarityValue > best) { best = similarityValue; bestIndex = query; }
}
${output.maximum ? 'maximum[maximumOffset + index] = best;' : ''}
${output.bestQuery ? 'bestQuery[bestQueryOffset + index] = bestIndex;' : ''}
${output.matches ? 'matches[matchesOffset + index] = select(0u, 1u, best >= parameters[parametersOffset]);' : ''}`
      })
    ];
  }
}
