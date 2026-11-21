// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GraphDataView,
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';
import {RASTER_FINITE_WGSL, rasterBinding} from '../remote-sensing/raster-kernel-utils';
import {
  RASTER_EMBEDDING_NO_LABEL,
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
  type RasterEmbeddingScalarView,
  type RasterEmbeddingView
} from './raster-embedding-chunks';

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
  validateRequirement(
    'getGPURasterEmbeddingSimilarityParameterValues',
    target.length >= 1,
    'target must hold at least 1 float32 elements; received ' + target.length
  );
  validateFiniteNumber(
    'getGPURasterEmbeddingSimilarityParameterValues',
    'matchThreshold',
    settings.matchThreshold
  );
  target[0] = settings.matchThreshold;
  return target;
}

/** Caller-owned outputs of {@link GPURasterEmbeddingSimilarity}; at least one is required. */
export type GPURasterEmbeddingSimilarityOutput<
  FloatRows extends RasterEmbeddingView<'float32'> = GraphDataView<'float32'>,
  UintRows extends RasterEmbeddingView<'uint32'> = GraphDataView<'uint32'>
> = {
  /** Similarity of every pixel to every query, `pixel * queryCount + query`; NaN for invalid pixels or queries. */
  similarity?: FloatRows;
  /** Largest eligible similarity over the queries; NaN for invalid pixels or no eligible query. */
  maximum?: FloatRows;
  /** Index of the most similar query (the lowest on ties); `0xffffffff` for invalid pixels or no eligible query. */
  bestQuery?: UintRows;
  /** 1 where `maximum >= matchThreshold`, else 0. Needs `parameters`. */
  matches?: UintRows;
};

/** Properties of {@link GPURasterEmbeddingSimilarity}. */
export type GPURasterEmbeddingSimilarityProps<
  FloatRows extends RasterEmbeddingView<'float32'> = GraphDataView<'float32'>,
  UintRows extends RasterEmbeddingView<'uint32'> = GraphDataView<'uint32'>
> = {
  id?: string;
  pixelCount: number;
  /** Vector length, in `[1, 1024]`. */
  dimensions: number;
  /** Decoded vectors, `pixel * dimensions + band`; a pixel with a non-finite component is invalid. */
  vectors: FloatRows;
  /** Per-pixel QA; zero makes the pixel invalid. */
  mask?: UintRows;
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
  output: GPURasterEmbeddingSimilarityOutput<FloatRows, UintRows>;
};

/**
 * Similarity of every pixel of an embedding raster to up to 64 query vectors: cosine or dot
 * product, the best query per pixel, the maximum over the queries and a threshold mask. This is
 * the raster form of similarity search; for the pixels nearest to a query as a ranked list use
 * {@link GPURasterEmbeddingNearestPixels}.
 *
 * Cosine follows the published `GPUSimilaritySearch`: two zero vectors have similarity one and
 * exactly one zero vector gives zero. Non-finite queries have NaN scores and cannot win; no eligible query yields NaN maximum,
 * no-label bestQuery and zero matches. Only finite computed scores can win; overflowed dot
 * products remain infinite in the score table but are ineligible. Dot products accumulate in float32 band order; decisions
 * within accumulation rounding error of a threshold or tie can depend on band order and backend.
 *
 * Pixel fields accept GraphVectorView chunks with matching complete-pixel boundaries; shared tables stay packed.
 * Each chunk has its own workgroup count and uses the same pipeline.
 *
 * Per-frame: `parameters` and the contents of the views. Compile-time: `pixelCount`,
 * `dimensions`, `queryCount`, `metric` and which optional views exist.
 * Pixel vectors accept 1–1,024 dimensions per complete pixel chunk; queryCount is at most 64 per dispatch.
 */
export class GPURasterEmbeddingSimilarity<
  FloatRows extends RasterEmbeddingView<'float32'> = GraphDataView<'float32'>,
  UintRows extends RasterEmbeddingView<'uint32'> = GraphDataView<'uint32'>
> implements GPUCommandNodeProducer
{
  readonly id: string;
  private readonly chunks: readonly RasterEmbeddingChunk[];
  readonly estimatedCommandNodeCount: number;

  constructor(readonly props: GPURasterEmbeddingSimilarityProps<FloatRows, UintRows>) {
    props = {...props, output: {...props.output}};
    this.props = props;
    const id = props.id ?? 'raster-embedding-similarity';
    this.id = id;
    validateRequirement(
      id,
      props.metric === undefined || ['cosine', 'dot'].includes(props.metric),
      'metric must be one of cosine, dot'
    );
    const {pixelCount, dimensions, output} = props;
    validateRasterEmbeddingShape(
      id,
      pixelCount,
      dimensions,
      MAXIMUM_RASTER_EMBEDDING_DIMENSIONS,
      false
    );
    validateCapacity(id, 'queryCount', props.queryCount, 1, MAXIMUM_RASTER_EMBEDDING_QUERY_COUNT);
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
    if (props.queries) validateSinglePackedView(props.queries, ['float32'], `${id} queries`);
    if (props.parameters)
      validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    const inputs = this.getInputs();
    const outputs = this.getOutputs();
    this.chunks = planRasterEmbeddingChunks(
      id,
      pixelCount,
      {
        vectors: {view: props.vectors, format: 'float32', components: dimensions},
        mask: {view: props.mask, format: 'uint32', components: 1},
        'output.similarity': {
          view: props.output.similarity,
          format: 'float32',
          components: props.queryCount
        },
        'output.maximum': {view: props.output.maximum, format: 'float32', components: 1},
        'output.bestQuery': {view: props.output.bestQuery, format: 'uint32', components: 1},
        'output.matches': {view: props.output.matches, format: 'uint32', components: 1}
      },
      'vectors',
      1
    );
    validateRasterEmbeddingViews(id, inputs, outputs, [{inputs, outputs}]);
    this.estimatedCommandNodeCount =
      1 + this.chunks.filter(chunk => chunk.pixelCount > 0).length * 1;
  }

  private getInputs(): RasterEmbeddingScalarView[] {
    const {vectors, mask, queries, parameters} = this.props;
    return [
      vectors,
      queries,
      ...(mask ? [mask] : []),
      ...(parameters && this.props.output.matches ? [parameters] : [])
    ];
  }

  private getOutputs(): RasterEmbeddingScalarView[] {
    const {similarity, maximum, bestQuery, matches} = this.props.output;
    return [similarity, maximum, bestQuery, matches].filter(
      (view): view is NonNullable<typeof view> => view !== undefined
    );
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
        Number(Boolean(this.props.parameters && this.props.output.matches)) +
        Object.values(this.props.output).filter(Boolean).length,
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
    props: PackedEmbeddingProps<GPURasterEmbeddingSimilarityProps>,
    chunkIndex: number
  ): RasterEmbeddingKernel[] {
    const id = `${this.id}-chunk-${chunkIndex}`;
    const {pixelCount, dimensions, queryCount, output} = props;
    const isCosine = (props.metric ?? 'cosine') === 'cosine';
    const writeInvalid = `${output.similarity ? `for (var query = 0u; query < ${queryCount}u; query++) { similarity[similarityOffset + index * ${queryCount}u + query] = missing(); }` : ''}
${output.maximum ? 'maximum[maximumOffset + index] = missing();' : ''}
${output.bestQuery ? `bestQuery[bestQueryOffset + index] = ${RASTER_EMBEDDING_NO_LABEL}u;` : ''}
${output.matches ? 'matches[matchesOffset + index] = 0u;' : ''}`;
    return [
      {
        id,
        operation: 'GPURasterEmbeddingSimilarity',
        variant: isCosine ? 'cosine' : 'dot',
        invocationCount: pixelCount,
        pixelCount,
        bindings: [
          rasterBinding('vectors', props.vectors),
          rasterBinding('queries', props.queries),
          ...(props.mask ? [rasterBinding('mask', props.mask)] : []),
          ...(props.parameters && output.matches
            ? [rasterBinding('parameters', props.parameters)]
            : []),
          ...(output.similarity ? [rasterBinding('similarity', output.similarity, true)] : []),
          ...(output.maximum ? [rasterBinding('maximum', output.maximum, true)] : []),
          ...(output.bestQuery ? [rasterBinding('bestQuery', output.bestQuery, true)] : []),
          ...(output.matches ? [rasterBinding('matches', output.matches, true)] : [])
        ],
        declarations: `${RASTER_FINITE_WGSL}\n${SCALED_QUOTIENT_WGSL}`,
        body: `var squaredLength = 0.0;
var pixelScale = 0.0;
var isValid = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'};
for (var band = 0u; band < ${dimensions}u; band++) {
  let value = vectors[vectorsOffset + index * ${dimensions}u + band];
  if (!finite(value)) { isValid = false; }
  ${isCosine ? 'pixelScale = max(pixelScale, abs(value));' : 'squaredLength += value * value;'}
}
${
  isCosine
    ? `for (var band = 0u; band < ${dimensions}u; band++) {
  let value = getScaledQuotient(vectors[vectorsOffset + index * ${dimensions}u + band], select(1.0, pixelScale, pixelScale > 0.0), 1.0, 0);
  squaredLength += value * value;
}`
    : ''
}
if (!isValid) {
  ${writeInvalid}
  return;
}
var best = missing();
var bestIndex = ${RASTER_EMBEDDING_NO_LABEL}u;
for (var query = 0u; query < ${queryCount}u; query++) {
  var queryValid = true;
  for (var band = 0u; band < ${dimensions}u; band++) {
    if (!finite(queries[queriesOffset + query * ${dimensions}u + band])) { queryValid = false; }
  }
  if (!queryValid) {
    ${output.similarity ? `similarity[similarityOffset + index * ${queryCount}u + query] = missing();` : ''}
    continue;
  }
  var dotProduct = 0.0;
  var queryLength = 0.0;
  ${
    isCosine
      ? `var queryScale = 0.0;
  for (var band = 0u; band < ${dimensions}u; band++) {
    queryScale = max(queryScale, abs(queries[queriesOffset + query * ${dimensions}u + band]));
  }`
      : ''
  }
  for (var band = 0u; band < ${dimensions}u; band++) {
    let component = ${isCosine ? `getScaledQuotient(queries[queriesOffset + query * ${dimensions}u + band], select(1.0, queryScale, queryScale > 0.0), 1.0, 0)` : `queries[queriesOffset + query * ${dimensions}u + band]`};
    let pixelComponent = ${isCosine ? `getScaledQuotient(vectors[vectorsOffset + index * ${dimensions}u + band], select(1.0, pixelScale, pixelScale > 0.0), 1.0, 0)` : `vectors[vectorsOffset + index * ${dimensions}u + band]`};
    dotProduct += pixelComponent * component;
    queryLength += component * component;
  }
  ${
    isCosine
      ? `let denominator = sqrt(squaredLength * queryLength);
  let similarityValue = select(select(0.0, 1.0, squaredLength == 0.0 && queryLength == 0.0), dotProduct / denominator, denominator > 0.0);`
      : 'let similarityValue = dotProduct;'
  }
  ${output.similarity ? `similarity[similarityOffset + index * ${queryCount}u + query] = similarityValue;` : ''}
  if (finite(similarityValue) && (bestIndex == ${RASTER_EMBEDDING_NO_LABEL}u || similarityValue > best)) { best = similarityValue; bestIndex = query; }
}
${output.maximum ? 'maximum[maximumOffset + index] = best;' : ''}
${output.bestQuery ? 'bestQuery[bestQueryOffset + index] = bestIndex;' : ''}
${output.matches ? `matches[matchesOffset + index] = select(0u, 1u, bestIndex != ${RASTER_EMBEDDING_NO_LABEL}u && best >= parameters[parametersOffset]);` : ''}`
      }
    ];
  }
}
