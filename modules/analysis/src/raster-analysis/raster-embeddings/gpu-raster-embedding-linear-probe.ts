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
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
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

/** Classes whose logits are held per pixel in private memory. */
export const MAXIMUM_RASTER_EMBEDDING_PROBE_CLASS_COUNT = 64;

/** Caller-owned outputs of {@link GPURasterEmbeddingLinearProbe}. */
export type GPURasterEmbeddingLinearProbeOutput<
  FloatRows extends RasterEmbeddingView<'float32'> = GraphDataView<'float32'>,
  UintRows extends RasterEmbeddingView<'uint32'> = GraphDataView<'uint32'>
> = {
  /** Predicted class (argmax; the lowest class on ties) per pixel; `0xffffffff` for invalid pixels. */
  labels: UintRows;
  /** Optional probability of the predicted class; NaN for invalid pixels. */
  confidence?: FloatRows;
  /** Optional probability of every class, `pixel * classCount + class`; NaN for invalid pixels. */
  probabilities?: FloatRows;
};

/** Properties of {@link GPURasterEmbeddingLinearProbe}. */
export type GPURasterEmbeddingLinearProbeProps<
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
  /** Number of classes, in `[1, 64]`. */
  classCount: number;
  /** Host-trained weights, `class * dimensions + band` (scikit-learn's `coef_`). Contents are per-frame. */
  weights: GraphDataView<'float32'>;
  /** Host-trained intercepts, one per class (`intercept_`). Contents are per-frame. */
  bias: GraphDataView<'float32'>;
  output: GPURasterEmbeddingLinearProbeOutput<FloatRows, UintRows>;
};

/**
 * Float32 logit accumulation can change near-tie labels with band order or backend reassociation.
 * Keep the decision margin larger than accumulation rounding error for stable labels.
 *
 * Inference of a linear probe on an embedding raster: `softmax(W x + b)` per pixel, the
 * prediction of a multinomial `LogisticRegression` (scikit-learn `predict_proba`, with the argmax
 * as `predict`). Training stays on the host. A binary scikit-learn model has one row of `coef_`
 * and one intercept; pass the two rows `[0, w]` and the intercepts `[0, b]`, which give the same
 * probabilities.
 *
 * Pixel fields accept GraphVectorView chunks with matching complete-pixel boundaries; shared tables stay packed.
 * Each chunk has its own workgroup count and uses the same pipeline.
 *
 * Per-frame: the contents of the views. Compile-time: `pixelCount`, `dimensions`, `classCount`
 * and which optional views exist.
 * Pixel vectors accept 1–1,024 dimensions per complete pixel chunk; classCount is at most 64.
 */
export class GPURasterEmbeddingLinearProbe<
  FloatRows extends RasterEmbeddingView<'float32'> = GraphDataView<'float32'>,
  UintRows extends RasterEmbeddingView<'uint32'> = GraphDataView<'uint32'>
> implements GPUCommandNodeProducer
{
  readonly id: string;
  private readonly chunks: readonly RasterEmbeddingChunk[];
  readonly estimatedCommandNodeCount: number;

  constructor(readonly props: GPURasterEmbeddingLinearProbeProps<FloatRows, UintRows>) {
    props = {...props, output: {...props.output}};
    this.props = props;
    const id = props.id ?? 'raster-embedding-linear-probe';
    this.id = id;
    const {pixelCount, dimensions, output} = props;
    validateRasterEmbeddingShape(
      id,
      pixelCount,
      dimensions,
      MAXIMUM_RASTER_EMBEDDING_DIMENSIONS,
      false
    );
    validateCapacity(
      id,
      'classCount',
      props.classCount,
      1,
      MAXIMUM_RASTER_EMBEDDING_PROBE_CLASS_COUNT
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
      props.weights.length === props.classCount * dimensions &&
        props.bias.length === props.classCount,
      'weights must hold classCount times dimensions entries and bias classCount'
    );
    validateRequirement(
      id,
      output.labels.length === pixelCount &&
        (!output.confidence || output.confidence.length === pixelCount) &&
        (!output.probabilities || output.probabilities.length === pixelCount * props.classCount),
      'outputs must hold pixelCount entries (probabilities pixelCount times classCount)'
    );
    if (props.weights) validateSinglePackedView(props.weights, ['float32'], `${id} weights`);
    if (props.bias) validateSinglePackedView(props.bias, ['float32'], `${id} bias`);
    const inputs = this.getInputs();
    const outputs = this.getOutputs();
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
    validateRasterEmbeddingViews(id, inputs, outputs, [{inputs, outputs}]);
    this.estimatedCommandNodeCount =
      1 + this.chunks.filter(chunk => chunk.pixelCount > 0).length * 1;
  }

  private getInputs(): RasterEmbeddingScalarView[] {
    const {vectors, mask, weights, bias} = this.props;
    return [vectors, weights, bias, ...(mask ? [mask] : [])];
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
      4 +
        Number(Boolean(this.props.mask)) +
        Number(Boolean(this.props.output.confidence)) +
        Number(Boolean(this.props.output.probabilities)),
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
    props: PackedEmbeddingProps<GPURasterEmbeddingLinearProbeProps>,
    chunkIndex: number
  ): RasterEmbeddingKernel[] {
    const id = `${this.id}-chunk-${chunkIndex}`;
    const {pixelCount, dimensions, classCount, output} = props;
    return [
      {
        id,
        operation: 'GPURasterEmbeddingLinearProbe',
        invocationCount: pixelCount,
        pixelCount,
        bindings: [
          rasterBinding('vectors', props.vectors),
          rasterBinding('weights', props.weights),
          rasterBinding('bias', props.bias),
          ...(props.mask ? [rasterBinding('mask', props.mask)] : []),
          rasterBinding('labels', output.labels, true),
          ...(output.confidence ? [rasterBinding('confidence', output.confidence, true)] : []),
          ...(output.probabilities
            ? [rasterBinding('probabilities', output.probabilities, true)]
            : [])
        ],
        declarations: RASTER_FINITE_WGSL,
        body: `var isValid = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'};
for (var band = 0u; band < ${dimensions}u; band++) {
  let value = vectors[vectorsOffset + index * ${dimensions}u + band];
  if (!finite(value)) { isValid = false; }
}
if (!isValid) {
  labels[labelsOffset + index] = ${RASTER_EMBEDDING_NO_LABEL}u;
  ${output.confidence ? 'confidence[confidenceOffset + index] = missing();' : ''}
  ${output.probabilities ? `for (var classIndex = 0u; classIndex < ${classCount}u; classIndex++) { probabilities[probabilitiesOffset + index * ${classCount}u + classIndex] = missing(); }` : ''}
  return;
}
var logits: array<f32, ${classCount}>;
var best = 0u;
for (var classIndex = 0u; classIndex < ${classCount}u; classIndex++) {
  var logit = bias[biasOffset + classIndex];
  for (var band = 0u; band < ${dimensions}u; band++) {
    logit += weights[weightsOffset + classIndex * ${dimensions}u + band] * vectors[vectorsOffset + index * ${dimensions}u + band];
  }
  logits[classIndex] = logit;
  if (logit > logits[best]) { best = classIndex; }
}
let largest = logits[best];
var total = 0.0;
for (var classIndex = 0u; classIndex < ${classCount}u; classIndex++) {
  logits[classIndex] = exp(logits[classIndex] - largest);
  total += logits[classIndex];
}
labels[labelsOffset + index] = best;
${output.confidence ? 'confidence[confidenceOffset + index] = logits[best] / total;' : ''}
${output.probabilities ? `for (var classIndex = 0u; classIndex < ${classCount}u; classIndex++) { probabilities[probabilitiesOffset + index * ${classCount}u + classIndex] = logits[classIndex] / total; }` : ''}`
      }
    ];
  }
}
