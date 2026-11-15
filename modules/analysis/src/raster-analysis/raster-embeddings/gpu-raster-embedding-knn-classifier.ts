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

/**
 * Labelled vectors a pixel is compared with. Every pixel thread scans the whole set, so the set
 * is a per-thread bound; a few thousand labelled points are what a trained analyst provides.
 */
export const MAXIMUM_RASTER_EMBEDDING_TRAINING_COUNT = 4096;
/** Neighbours kept per pixel in private memory. */
export const MAXIMUM_RASTER_EMBEDDING_NEIGHBOR_COUNT = 16;
/** Classes counted per pixel in private memory. */
export const MAXIMUM_RASTER_EMBEDDING_KNN_CLASS_COUNT = 32;

/** Caller-owned outputs of {@link GPURasterEmbeddingKNearestClassifier}. */
export type GPURasterEmbeddingKNearestClassifierOutput = {
  /** Predicted class per pixel; `0xffffffff` for invalid pixels. */
  labels: GraphDataView<'uint32'>;
  /** Optional vote share of the predicted class, `votes / filledNeighborCount`; NaN for invalid pixels. */
  confidence?: GraphDataView<'float32'>;
  /** Optional votes / filledNeighborCount per `pixel * classCount + class`; NaN for invalid pixels. */
  probabilities?: GraphDataView<'float32'>;
};

/** Properties of {@link GPURasterEmbeddingKNearestClassifier}. */
export type GPURasterEmbeddingKNearestClassifierProps = {
  id?: string;
  pixelCount: number;
  /** Vector length, in `[1, 128]`. */
  dimensions: number;
  vectors: GraphDataView<'float32'>;
  /** Per-pixel QA; zero makes the pixel invalid. */
  mask?: GraphDataView<'uint32'>;
  /** Labelled vectors, `sample * dimensions + band`; non-finite rows are excluded. Contents are per-frame. */
  trainingVectors: GraphDataView<'float32'>;
  /** Class of each labelled vector, in `[0, classCount)`. Contents are per-frame. */
  trainingLabels: GraphDataView<'uint32'>;
  /** Number of labelled vectors, in `[1, 4096]`. */
  trainingCount: number;
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
  output: GPURasterEmbeddingKNearestClassifierOutput;
};

/**
 * Brute-force k-nearest-neighbour classification of an embedding raster against a labelled set,
 * with the semantics of scikit-learn's `KNeighborsClassifier` (uniform weights): up to
 * `neighborCount` nearest labelled vectors each vote, the class with most votes wins and a tie goes
 * to the smallest class. Distance ties between labelled vectors go to the lower sample index.
 * Non-finite training rows and non-finite computed distances are excluded. Vote shares divide by
 * the number of filled neighbours. Masked or non-finite pixels, and pixels with no filled neighbours,
 * have label `0xffffffff` and NaN confidence and probabilities.
 * Training stays on the host: pass the labelled set as views.
 *
 * Cost is `pixelCount x trainingCount x dimensions`; a 4096-point set on a million pixels of 64
 * bands is about 2.7e11 multiply-adds, so choose the set size to the raster.
 *
 * Per-frame: the contents of the views. Compile-time: `pixelCount`, `dimensions`,
 * `trainingCount`, `classCount`, `neighborCount`, `metric` and which optional views exist.
 */
export class GPURasterEmbeddingKNearestClassifier implements GPUCommandNodeProducer {
  readonly id: string;

  constructor(readonly props: GPURasterEmbeddingKNearestClassifierProps) {
    const id = props.id ?? 'raster-embedding-knn-classifier';
    this.id = id;
    const {pixelCount, dimensions, output} = props;
    validateRasterEmbeddingShape(id, pixelCount, dimensions);
    validateCapacity(
      id,
      'trainingCount',
      props.trainingCount,
      1,
      MAXIMUM_RASTER_EMBEDDING_TRAINING_COUNT
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
    validateRasterViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterScalarView[] {
    const {vectors, mask, trainingVectors, trainingLabels} = this.props;
    return [vectors, trainingVectors, trainingLabels, ...(mask ? [mask] : [])];
  }

  private getOutputs(): RasterScalarView[] {
    const {labels, confidence, probabilities} = this.props.output;
    return [labels, ...(confidence ? [confidence] : []), ...(probabilities ? [probabilities] : [])];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {pixelCount, dimensions, trainingCount, classCount, neighborCount, output} = props;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    const isCosine = props.metric === 'cosine';
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterEmbeddingKNearestClassifier',
        variant: isCosine ? 'cosine' : 'euclidean',
        invocationCount: pixelCount,
        workgroupSize: 64,
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
        declarations: RASTER_FINITE_WGSL,
        body: `var vector: array<f32, ${dimensions}>;
var squaredLength = 0.0;
var isValid = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'};
for (var band = 0u; band < ${dimensions}u; band++) {
  let value = vectors[vectorsOffset + index * ${dimensions}u + band];
  if (!finite(value)) { isValid = false; }
  vector[band] = value;
  squaredLength += value * value;
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
  for (var band = 0u; band < ${dimensions}u; band++) {
    let component = trainingVectors[trainingVectorsOffset + sampleIndex * ${dimensions}u + band];
    if (!finite(component)) { isSampleValid = false; }
    ${isCosine ? 'dotProduct += vector[band] * component;\n    sampleLength += component * component;' : 'dotProduct += (vector[band] - component) * (vector[band] - component);'}
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
      })
    ];
  }
}
