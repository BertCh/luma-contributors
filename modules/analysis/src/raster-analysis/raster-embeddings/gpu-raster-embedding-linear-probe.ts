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

/** Classes whose logits are held per pixel in private memory. */
export const MAXIMUM_RASTER_EMBEDDING_PROBE_CLASS_COUNT = 64;

/** Caller-owned outputs of {@link GPURasterEmbeddingLinearProbe}. */
export type GPURasterEmbeddingLinearProbeOutput = {
  /** Predicted class (argmax; the lowest class on ties) per pixel; `0xffffffff` for invalid pixels. */
  labels: GraphDataView<'uint32'>;
  /** Optional probability of the predicted class; NaN for invalid pixels. */
  confidence?: GraphDataView<'float32'>;
  /** Optional probability of every class, `pixel * classCount + class`; NaN for invalid pixels. */
  probabilities?: GraphDataView<'float32'>;
};

/** Properties of {@link GPURasterEmbeddingLinearProbe}. */
export type GPURasterEmbeddingLinearProbeProps = {
  id?: string;
  pixelCount: number;
  /** Vector length, in `[1, 128]`. */
  dimensions: number;
  vectors: GraphDataView<'float32'>;
  /** Per-pixel QA; zero makes the pixel invalid. */
  mask?: GraphDataView<'uint32'>;
  /** Number of classes, in `[1, 64]`. */
  classCount: number;
  /** Host-trained weights, `class * dimensions + band` (scikit-learn's `coef_`). Contents are per-frame. */
  weights: GraphDataView<'float32'>;
  /** Host-trained intercepts, one per class (`intercept_`). Contents are per-frame. */
  bias: GraphDataView<'float32'>;
  output: GPURasterEmbeddingLinearProbeOutput;
};

/**
 * Inference of a linear probe on an embedding raster: `softmax(W x + b)` per pixel, the
 * prediction of a multinomial `LogisticRegression` (scikit-learn `predict_proba`, with the argmax
 * as `predict`). Training stays on the host. A binary scikit-learn model has one row of `coef_`
 * and one intercept; pass the two rows `[0, w]` and the intercepts `[0, b]`, which give the same
 * probabilities.
 *
 * Per-frame: the contents of the views. Compile-time: `pixelCount`, `dimensions`, `classCount`
 * and which optional views exist.
 */
export class GPURasterEmbeddingLinearProbe implements GPUCommandNodeProducer {
  readonly id: string;

  constructor(readonly props: GPURasterEmbeddingLinearProbeProps) {
    const id = props.id ?? 'raster-embedding-linear-probe';
    this.id = id;
    const {pixelCount, dimensions, output} = props;
    validateRasterEmbeddingShape(id, pixelCount, dimensions);
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
    validateRasterViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterScalarView[] {
    const {vectors, mask, weights, bias} = this.props;
    return [vectors, weights, bias, ...(mask ? [mask] : [])];
  }

  private getOutputs(): RasterScalarView[] {
    const {labels, confidence, probabilities} = this.props.output;
    return [labels, ...(confidence ? [confidence] : []), ...(probabilities ? [probabilities] : [])];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {pixelCount, dimensions, classCount, output} = props;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterEmbeddingLinearProbe',
        invocationCount: pixelCount,
        workgroupSize: 64,
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
        body: `var vector: array<f32, ${dimensions}>;
var isValid = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'};
for (var band = 0u; band < ${dimensions}u; band++) {
  let value = vectors[vectorsOffset + index * ${dimensions}u + band];
  if (!finite(value)) { isValid = false; }
  vector[band] = value;
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
    logit += weights[weightsOffset + classIndex * ${dimensions}u + band] * vector[band];
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
      })
    ];
  }
}
