// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GraphDataView,
  GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateCommandNodeBudget,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';
import {RASTER_FINITE_WGSL, rasterBinding} from '../remote-sensing/raster-kernel-utils';
import {
  RASTER_EMBEDDING_NO_LABEL,
  MAXIMUM_RASTER_EMBEDDING_ELEMENT_COUNT,
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

/** Explicit brute-force reference route scans at most this many training rows per invocation. */
export const MAXIMUM_RASTER_EMBEDDING_TRAINING_COUNT = 4096;
/** Tile size bounds each scan invocation; top-k state persists in checked storage scratch. */
export const RASTER_EMBEDDING_TRAINING_TILE_SIZE = 256;
/** Training sample IDs and flattened component addresses are uint32; bindings impose the tighter bound. */
export const MAXIMUM_RASTER_EMBEDDING_TILED_TRAINING_COUNT = MAXIMUM_RASTER_EMBEDDING_ELEMENT_COUNT;
/** Neighbours kept per pixel in private memory. */
export const MAXIMUM_RASTER_EMBEDDING_NEIGHBOR_COUNT = 16;
/** Classes counted per pixel in private memory. */
export const MAXIMUM_RASTER_EMBEDDING_KNN_CLASS_COUNT = 32;

/** Caller-owned outputs of {@link GPURasterEmbeddingKNearestClassifier}. */
export type GPURasterEmbeddingKNearestClassifierOutput<
  FloatRows extends RasterEmbeddingView<'float32'> = GraphDataView<'float32'>,
  UintRows extends RasterEmbeddingView<'uint32'> = GraphDataView<'uint32'>
> = {
  /** Predicted class per pixel; `0xffffffff` for invalid pixels. */
  labels: UintRows;
  /** Optional vote share of the predicted class, `votes / filledNeighborCount`; NaN for invalid pixels. */
  confidence?: FloatRows;
  /** Optional votes / filledNeighborCount per `pixel * classCount + class`; NaN for invalid pixels. */
  probabilities?: FloatRows;
};

/** Properties of {@link GPURasterEmbeddingKNearestClassifier}. */
export type GPURasterEmbeddingKNearestClassifierProps<
  FloatRows extends RasterEmbeddingView<'float32'> = GraphDataView<'float32'>,
  UintRows extends RasterEmbeddingView<'uint32'> = GraphDataView<'uint32'>
> = {
  id?: string;
  pixelCount: number;
  /** Vector length, in `[1, 1024]`. */
  dimensions: number;
  vectors: FloatRows;
  /** Per-pixel QA; zero makes the pixel invalid. */
  mask?: UintRows;
  /** Labelled vectors, `sample * dimensions + band`; non-finite rows are excluded. Contents are per-frame. */
  trainingVectors: GraphDataView<'float32'>;
  /** Class of each labelled vector, in `[0, classCount)`. Contents are per-frame. */
  trainingLabels: GraphDataView<'uint32'>;
  /** Number of labelled vectors; the packed table must fit the device storage binding. */
  trainingCount: number;
  /** Tiled exact scan (default), or the bounded original single-dispatch reference kernel. */
  searchStructure?: 'tiled' | 'brute-force';
  /** Number of classes, in `[1, 32]`. */
  classCount: number;
  /** Maximum neighbours that vote, in `[1, 16]` and at most `trainingCount`. */
  neighborCount: number;
  /**
   * `'euclidean'` (scikit-learn's default) or `'cosine'`, the distance `1 - cosine similarity`.
   * Two zero vectors have similarity 1 (distance 0) and exactly one zero vector gives similarity 0
   * (distance 1), as in the published `GPUSimilaritySearch`. Defaults to `'euclidean'`.
   */
  metric?: 'euclidean' | 'cosine';
  output: GPURasterEmbeddingKNearestClassifierOutput<FloatRows, UintRows>;
};

/**
 * Exact k-nearest-neighbour classification of an embedding raster against a labelled set,
 * with the semantics of scikit-learn's `KNeighborsClassifier` (uniform weights): up to
 * `neighborCount` nearest labelled vectors each vote, the class with most votes wins and a tie goes
 * to the smallest class. Distance ties between labelled vectors go to the lower sample index.
 * Non-finite training rows and non-finite computed distances are excluded. Vote shares divide by
 * the number of filled neighbours. Masked or non-finite pixels, and pixels with no filled neighbours,
 * have label `0xffffffff` and NaN confidence and probabilities.
 * Training stays on the host: pass the labelled set as views.
 *
 * The default tiled route scans at most 256 training rows per invocation and preserves the
 * sorted top-k in two `pixelCount * neighborCount * 4`-byte scratch bindings per pixel chunk.
 * It is exhaustive: arithmetic cost remains `pixelCount x trainingCount x dimensions`.
 * `searchStructure: 'brute-force'` keeps the original reference route and its 4096-row limit.
 *
 * Pixel fields accept GraphVectorView chunks with matching complete-pixel boundaries; shared tables stay packed.
 * Each chunk has its own workgroup count and uses the same pipeline.
 *
 * Per-frame: the contents of the views. Compile-time: `pixelCount`, `dimensions`,
 * `trainingCount`, `classCount`, `neighborCount`, `metric` and which optional views exist.
 */
export class GPURasterEmbeddingKNearestClassifier<
  FloatRows extends RasterEmbeddingView<'float32'> = GraphDataView<'float32'>,
  UintRows extends RasterEmbeddingView<'uint32'> = GraphDataView<'uint32'>
> implements GPUCommandNodeProducer
{
  readonly id: string;
  private readonly chunks: readonly RasterEmbeddingChunk[];
  readonly estimatedCommandNodeCount: number;

  constructor(readonly props: GPURasterEmbeddingKNearestClassifierProps<FloatRows, UintRows>) {
    const id = props.id ?? 'raster-embedding-knn-classifier';
    this.id = id;
    validateRequirement(
      id,
      props.metric === undefined || ['euclidean', 'cosine'].includes(props.metric),
      'metric must be one of euclidean, cosine'
    );
    const {pixelCount, dimensions, output} = props;
    validateRasterEmbeddingShape(
      id,
      pixelCount,
      dimensions,
      MAXIMUM_RASTER_EMBEDDING_DIMENSIONS,
      false
    );
    validateRequirement(
      id,
      props.searchStructure === undefined ||
        ['tiled', 'brute-force'].includes(props.searchStructure),
      'searchStructure must be tiled or brute-force'
    );
    validateRequirement(
      id,
      props.trainingCount * dimensions <= MAXIMUM_RASTER_EMBEDDING_ELEMENT_COUNT,
      'trainingCount times dimensions must stay below 2^32'
    );
    validateCapacity(
      id,
      'trainingCount',
      props.trainingCount,
      1,
      props.searchStructure === 'brute-force'
        ? MAXIMUM_RASTER_EMBEDDING_TRAINING_COUNT
        : MAXIMUM_RASTER_EMBEDDING_TILED_TRAINING_COUNT
    );
    validateCapacity(
      id,
      'classCount',
      props.classCount,
      1,
      MAXIMUM_RASTER_EMBEDDING_KNN_CLASS_COUNT
    );
    validateCapacity(
      id,
      'neighborCount',
      props.neighborCount,
      1,
      Math.min(MAXIMUM_RASTER_EMBEDDING_NEIGHBOR_COUNT, props.trainingCount)
    );
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
      props.trainingVectors.length === props.trainingCount * dimensions &&
        props.trainingLabels.length === props.trainingCount,
      'trainingVectors must hold trainingCount times dimensions entries and trainingLabels trainingCount'
    );
    validateRequirement(
      id,
      output.labels.length === pixelCount &&
        (!output.confidence || output.confidence.length === pixelCount) &&
        (!output.probabilities || output.probabilities.length === pixelCount * props.classCount),
      'outputs must hold pixelCount entries (probabilities pixelCount times classCount)'
    );
    if (props.trainingVectors)
      validateSinglePackedView(props.trainingVectors, ['float32'], `${id} trainingVectors`);
    if (props.trainingLabels)
      validateSinglePackedView(props.trainingLabels, ['uint32'], `${id} trainingLabels`);
    validateRasterEmbeddingViews(
      id,
      this.getInputs(),
      this.getOutputs(),
      props.searchStructure === 'brute-force'
        ? undefined
        : [{inputs: [props.trainingLabels], outputs: this.getOutputs()}]
    );
    this.chunks = planRasterEmbeddingChunks(
      id,
      pixelCount,
      {
        vectors: {view: props.vectors, format: 'float32', components: dimensions},
        mask: {view: props.mask, format: 'uint32', components: 1},
        'output.labels': {view: props.output.labels, format: 'uint32', components: 1},
        'output.confidence': {view: props.output.confidence, format: 'float32', components: 1},
        'output.probabilities': {
          view: props.output.probabilities,
          format: 'float32',
          components: props.classCount
        }
      },
      'vectors',
      1
    );
    const stages =
      props.searchStructure === 'brute-force'
        ? 1
        : Math.ceil(props.trainingCount / RASTER_EMBEDDING_TRAINING_TILE_SIZE) + 1;
    this.estimatedCommandNodeCount =
      1 + this.chunks.filter(chunk => chunk.pixelCount > 0).length * stages;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  private getInputs(): RasterEmbeddingScalarView[] {
    const {vectors, mask, trainingVectors, trainingLabels} = this.props;
    return [vectors, trainingVectors, trainingLabels, ...(mask ? [mask] : [])];
  }

  private getOutputs(): RasterEmbeddingScalarView[] {
    const {labels, confidence, probabilities} = this.props.output;
    return [labels, ...(confidence ? [confidence] : []), ...(probabilities ? [probabilities] : [])];
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
      this.props.searchStructure === 'brute-force'
        ? 4 +
            Number(Boolean(this.props.mask)) +
            Number(Boolean(this.props.output.confidence)) +
            Number(Boolean(this.props.output.probabilities))
        : Math.max(
            4 + Number(Boolean(this.props.mask)),
            3 +
              Number(Boolean(this.props.output.confidence)) +
              Number(Boolean(this.props.output.probabilities))
          ),
      this.chunks
    );
    if (this.props.searchStructure !== 'brute-force') {
      for (const chunk of this.chunks)
        validateGPUStorageBindingByteLength(
          graph.device,
          this.id,
          'nearest-neighbor scratch',
          Math.max(chunk.pixelCount * this.props.neighborCount, 1) * 4,
          'reduce the embedding chunk pixelCount or neighborCount'
        );
    }
    const kernels = this.chunks.flatMap((chunk, chunkIndex) =>
      chunk.pixelCount === 0
        ? []
        : this.getChunkKernels(graph, getRasterEmbeddingChunkProps(this.props, chunk), chunkIndex)
    );
    return createRasterEmbeddingKernelNodes(graph, this.id, kernels);
  }

  private getChunkKernels<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    props: PackedEmbeddingProps<GPURasterEmbeddingKNearestClassifierProps>,
    chunkIndex: number
  ): RasterEmbeddingKernel[] {
    const id = `${this.id}-chunk-${chunkIndex}`;
    const {pixelCount, dimensions, trainingCount, classCount, neighborCount, output} = props;
    const isCosine = props.metric === 'cosine';
    if (props.searchStructure !== 'brute-force') return this.getTiledKernels(graph, props, id);
    return [
      {
        id,
        operation: 'GPURasterEmbeddingKNearestClassifier',
        variant: isCosine ? 'cosine' : 'euclidean',
        invocationCount: pixelCount,
        pixelCount,
        bindings: [
          rasterBinding('vectors', props.vectors),
          rasterBinding('trainingVectors', props.trainingVectors),
          rasterBinding('trainingLabels', props.trainingLabels),
          ...(props.mask ? [rasterBinding('mask', props.mask)] : []),
          rasterBinding('labels', output.labels, true),
          ...(output.confidence ? [rasterBinding('confidence', output.confidence, true)] : []),
          ...(output.probabilities
            ? [rasterBinding('probabilities', output.probabilities, true)]
            : [])
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
  labels[labelsOffset + index] = ${RASTER_EMBEDDING_NO_LABEL}u;
  ${output.confidence ? 'confidence[confidenceOffset + index] = missing();' : ''}
  ${output.probabilities ? `for (var classIndex = 0u; classIndex < ${classCount}u; classIndex++) { probabilities[probabilitiesOffset + index * ${classCount}u + classIndex] = missing(); }` : ''}
  return;
}
var nearestDistance: array<f32, ${neighborCount}>;
var nearestSample: array<u32, ${neighborCount}>;
var filledNeighborCount = 0u;
for (var sampleIndex = 0u; sampleIndex < ${trainingCount}u; sampleIndex++) {
  var dotProduct = 0.0;
  var sampleLength = 0.0;
  var isSampleValid = true;
  ${
    isCosine
      ? `var sampleScale = 0.0;
  for (var band = 0u; band < ${dimensions}u; band++) {
    let component = trainingVectors[trainingVectorsOffset + sampleIndex * ${dimensions}u + band];
    isSampleValid = isSampleValid && finite(component);
    sampleScale = max(sampleScale, abs(component));
  }`
      : ''
  }
  for (var band = 0u; band < ${dimensions}u; band++) {
    let pixelComponent = ${isCosine ? `getScaledQuotient(vectors[vectorsOffset + index * ${dimensions}u + band], select(1.0, pixelScale, pixelScale > 0.0), 1.0, 0)` : `vectors[vectorsOffset + index * ${dimensions}u + band]`};
    let component = ${isCosine ? `getScaledQuotient(trainingVectors[trainingVectorsOffset + sampleIndex * ${dimensions}u + band], select(1.0, sampleScale, sampleScale > 0.0), 1.0, 0)` : `trainingVectors[trainingVectorsOffset + sampleIndex * ${dimensions}u + band]`};
    if (!finite(component)) { isSampleValid = false; }
    ${isCosine ? 'dotProduct += pixelComponent * component;\n    sampleLength += component * component;' : 'dotProduct += (pixelComponent - component) * (pixelComponent - component);'}
  }
  if (!isSampleValid) { continue; }
  ${
    isCosine
      ? `let denominator = sqrt(squaredLength * sampleLength);
  // Zero vectors as in the published GPUSimilaritySearch: two zero vectors have similarity 1,
  // exactly one zero vector gives 0.
  let neighborDistance = 1.0 - select(select(0.0, 1.0, squaredLength == 0.0 && sampleLength == 0.0), dotProduct / denominator, denominator > 0.0);`
      : 'let neighborDistance = dotProduct;'
  }
  if (!finite(neighborDistance)) { continue; }
  // Strict comparison keeps the lower sample index on equal distances.
  if (filledNeighborCount < ${neighborCount}u || neighborDistance < nearestDistance[${neighborCount - 1}u]) {
    var slot = min(filledNeighborCount, ${neighborCount - 1}u);
    while (slot > 0u && neighborDistance < nearestDistance[slot - 1u]) {
      nearestDistance[slot] = nearestDistance[slot - 1u];
      nearestSample[slot] = nearestSample[slot - 1u];
      slot--;
    }
    nearestDistance[slot] = neighborDistance;
    nearestSample[slot] = sampleIndex;
    filledNeighborCount = min(filledNeighborCount + 1u, ${neighborCount}u);
  }
}
if (filledNeighborCount == 0u) {
  labels[labelsOffset + index] = ${RASTER_EMBEDDING_NO_LABEL}u;
  ${output.confidence ? 'confidence[confidenceOffset + index] = missing();' : ''}
  ${output.probabilities ? `for (var classIndex = 0u; classIndex < ${classCount}u; classIndex++) { probabilities[probabilitiesOffset + index * ${classCount}u + classIndex] = missing(); }` : ''}
  return;
}
var votes: array<u32, ${classCount}>;
for (var slot = 0u; slot < filledNeighborCount; slot++) {
  let classIndex = min(trainingLabels[trainingLabelsOffset + nearestSample[slot]], ${classCount - 1}u);
  votes[classIndex] += 1u;
}
var best = 0u;
for (var classIndex = 1u; classIndex < ${classCount}u; classIndex++) {
  if (votes[classIndex] > votes[best]) { best = classIndex; }
}
labels[labelsOffset + index] = best;
${output.confidence ? 'confidence[confidenceOffset + index] = f32(votes[best]) / f32(filledNeighborCount);' : ''}
${output.probabilities ? `for (var classIndex = 0u; classIndex < ${classCount}u; classIndex++) { probabilities[probabilitiesOffset + index * ${classCount}u + classIndex] = f32(votes[classIndex]) / f32(filledNeighborCount); }` : ''}`
      }
    ];
  }

  private getTiledKernels<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    props: PackedEmbeddingProps<GPURasterEmbeddingKNearestClassifierProps>,
    id: string
  ): RasterEmbeddingKernel[] {
    const {pixelCount, dimensions, trainingCount, classCount, neighborCount, output} = props;
    const isCosine = props.metric === 'cosine';
    const savedSamples = createTransientView(
      graph,
      `${id}-nearest-samples`,
      'uint32',
      pixelCount * neighborCount
    );
    const savedDistances = createTransientView(
      graph,
      `${id}-nearest-distances`,
      'float32',
      pixelCount * neighborCount
    );
    const kernels: RasterEmbeddingKernel[] = [];
    for (
      let tileBase = 0;
      tileBase < trainingCount;
      tileBase += RASTER_EMBEDDING_TRAINING_TILE_SIZE
    ) {
      kernels.push({
        id: `${id}-training-${tileBase / RASTER_EMBEDDING_TRAINING_TILE_SIZE}`,
        operation: 'GPURasterEmbeddingKNearestClassifier',
        variant: 'tiled-scan',
        invocationCount: pixelCount,
        pixelCount,
        tileBase,
        tileCount: Math.min(RASTER_EMBEDDING_TRAINING_TILE_SIZE, trainingCount - tileBase),
        bindings: [
          rasterBinding('vectors', props.vectors),
          rasterBinding('trainingVectors', props.trainingVectors),
          ...(props.mask ? [rasterBinding('mask', props.mask)] : []),
          rasterBinding('savedSamples', savedSamples, true),
          rasterBinding('savedDistances', savedDistances, true)
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
var nearestDistance: array<f32, ${neighborCount}>;
var nearestSample: array<u32, ${neighborCount}>;
var filledNeighborCount = 0u;
for (var slot = 0u; slot < ${neighborCount}u; slot++) {
  nearestSample[slot] = 0xffffffffu;
  if (embeddingDispatch.words[0u][2u] != 0u) {
    nearestSample[slot] = savedSamples[savedSamplesOffset + index * ${neighborCount}u + slot];
    nearestDistance[slot] = savedDistances[savedDistancesOffset + index * ${neighborCount}u + slot];
    if (nearestSample[slot] != 0xffffffffu) { filledNeighborCount++; }
  }
}
for (var tileRow = 0u; tileRow < embeddingDispatch.words[0u][3u]; tileRow++) {
  if (!isValid) { break; }
  let sampleIndex = embeddingDispatch.words[0u][2u] + tileRow;
  var dotProduct = 0.0;
  var sampleLength = 0.0;
  var isSampleValid = true;
  ${
    isCosine
      ? `var sampleScale = 0.0;
  for (var band = 0u; band < ${dimensions}u; band++) {
    let component = trainingVectors[trainingVectorsOffset + sampleIndex * ${dimensions}u + band];
    isSampleValid = isSampleValid && finite(component);
    sampleScale = max(sampleScale, abs(component));
  }`
      : ''
  }
  for (var band = 0u; band < ${dimensions}u; band++) {
    let pixelComponent = ${isCosine ? `getScaledQuotient(vectors[vectorsOffset + index * ${dimensions}u + band], select(1.0, pixelScale, pixelScale > 0.0), 1.0, 0)` : `vectors[vectorsOffset + index * ${dimensions}u + band]`};
    let component = ${isCosine ? `getScaledQuotient(trainingVectors[trainingVectorsOffset + sampleIndex * ${dimensions}u + band], select(1.0, sampleScale, sampleScale > 0.0), 1.0, 0)` : `trainingVectors[trainingVectorsOffset + sampleIndex * ${dimensions}u + band]`};
    if (!finite(component)) { isSampleValid = false; }
    ${isCosine ? 'dotProduct += pixelComponent * component;\n    sampleLength += component * component;' : 'dotProduct += (pixelComponent - component) * (pixelComponent - component);'}
  }
  if (!isSampleValid) { continue; }
  ${
    isCosine
      ? `let denominator = sqrt(squaredLength * sampleLength);
  // Zero vectors as in the published GPUSimilaritySearch: two zero vectors have similarity 1,
  // exactly one zero vector gives 0.
  let neighborDistance = 1.0 - select(select(0.0, 1.0, squaredLength == 0.0 && sampleLength == 0.0), dotProduct / denominator, denominator > 0.0);`
      : 'let neighborDistance = dotProduct;'
  }
  if (!finite(neighborDistance)) { continue; }
  // Strict comparison keeps the lower sample index on equal distances.
  if (filledNeighborCount < ${neighborCount}u || neighborDistance < nearestDistance[${neighborCount - 1}u]) {
    var slot = min(filledNeighborCount, ${neighborCount - 1}u);
    while (slot > 0u && neighborDistance < nearestDistance[slot - 1u]) {
      nearestDistance[slot] = nearestDistance[slot - 1u];
      nearestSample[slot] = nearestSample[slot - 1u];
      slot--;
    }
    nearestDistance[slot] = neighborDistance;
    nearestSample[slot] = sampleIndex;
    filledNeighborCount = min(filledNeighborCount + 1u, ${neighborCount}u);
  }
}
for (var slot = 0u; slot < ${neighborCount}u; slot++) {
  savedSamples[savedSamplesOffset + index * ${neighborCount}u + slot] = nearestSample[slot];
  savedDistances[savedDistancesOffset + index * ${neighborCount}u + slot] = nearestDistance[slot];
}
`
      });
    }
    kernels.push({
      id: `${id}-vote`,
      operation: 'GPURasterEmbeddingKNearestClassifier',
      variant: 'tiled-vote',
      invocationCount: pixelCount,
      pixelCount,
      bindings: [
        rasterBinding('savedSamples', savedSamples),
        rasterBinding('trainingLabels', props.trainingLabels),
        rasterBinding('labels', output.labels, true),
        ...(output.confidence ? [rasterBinding('confidence', output.confidence, true)] : []),
        ...(output.probabilities
          ? [rasterBinding('probabilities', output.probabilities, true)]
          : [])
      ],
      declarations: `${RASTER_FINITE_WGSL}\n${SCALED_QUOTIENT_WGSL}`,
      body: `var filledNeighborCount = 0u;
for (var slot = 0u; slot < ${neighborCount}u; slot++) {
  if (savedSamples[savedSamplesOffset + index * ${neighborCount}u + slot] != 0xffffffffu) { filledNeighborCount++; }
}
if (filledNeighborCount == 0u) {
  labels[labelsOffset + index] = ${RASTER_EMBEDDING_NO_LABEL}u;
  ${output.confidence ? 'confidence[confidenceOffset + index] = missing();' : ''}
  ${output.probabilities ? `for (var classIndex = 0u; classIndex < ${classCount}u; classIndex++) { probabilities[probabilitiesOffset + index * ${classCount}u + classIndex] = missing(); }` : ''}
  return;
}
var votes: array<u32, ${classCount}>;
for (var slot = 0u; slot < filledNeighborCount; slot++) {
  let classIndex = min(trainingLabels[trainingLabelsOffset + savedSamples[savedSamplesOffset + index * ${neighborCount}u + slot]], ${classCount - 1}u);
  votes[classIndex] += 1u;
}
var best = 0u;
for (var classIndex = 1u; classIndex < ${classCount}u; classIndex++) {
  if (votes[classIndex] > votes[best]) { best = classIndex; }
}
labels[labelsOffset + index] = best;
${output.confidence ? 'confidence[confidenceOffset + index] = f32(votes[best]) / f32(filledNeighborCount);' : ''}
${output.probabilities ? `for (var classIndex = 0u; classIndex < ${classCount}u; classIndex++) { probabilities[probabilitiesOffset + index * ${classCount}u + classIndex] = f32(votes[classIndex]) / f32(filledNeighborCount); }` : ''}`
    });
    return kernels;
  }
}
