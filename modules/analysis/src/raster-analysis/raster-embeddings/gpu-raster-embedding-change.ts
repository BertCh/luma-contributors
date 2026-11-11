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
import {GUARDED_ATAN2_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
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
  validateRasterEmbeddingShape,
  validateRasterEmbeddingVectors
} from './raster-embedding-common';

/** Float32 elements of the per-frame parameters: the similarity threshold. */
export const GPU_RASTER_EMBEDDING_CHANGE_PARAMETER_LENGTH = 1;

/** Per-frame settings of {@link GPURasterEmbeddingChange}. */
export type GPURasterEmbeddingChangeSettings = {
  /** A pixel is changed when its cosine similarity between the two years is below this value. */
  similarityThreshold: number;
};

/** Packs {@link GPURasterEmbeddingChangeSettings}: `[similarityThreshold]`. */
export function getGPURasterEmbeddingChangeParameterValues(
  settings: GPURasterEmbeddingChangeSettings,
  target: Float32Array = new Float32Array(GPU_RASTER_EMBEDDING_CHANGE_PARAMETER_LENGTH)
): Float32Array {
  validateFiniteNumber(
    'getGPURasterEmbeddingChangeParameterValues',
    'similarityThreshold',
    settings.similarityThreshold,
    -1,
    1
  );
  target[0] = settings.similarityThreshold;
  return target;
}

/** Caller-owned outputs of {@link GPURasterEmbeddingChange}; at least one is required. */
export type GPURasterEmbeddingChangeOutput = {
  /** Cosine similarity of the two years' vectors, in `[-1, 1]`; NaN where either year is invalid. */
  similarity?: GraphDataView<'float32'>;
  /**
   * Angle between the vectors in radians, in `[0, pi]`; NaN where invalid. Computed as
   * `2 atan2(|u - v|, |u + v|)` of the two unit vectors, not as `acos(similarity)`, so angles far
   * below the float32 step of the cosine (about 5e-4 rad) are resolved. Two zero vectors give 0;
   * exactly one zero vector gives `pi / 2`.
   */
  angle?: GraphDataView<'float32'>;
  /** Length of the change vector `|after - before|`; NaN where invalid. */
  magnitude?: GraphDataView<'float32'>;
  /** 1 where `similarity < similarityThreshold`, else 0. Needs `parameters`. */
  changed?: GraphDataView<'uint32'>;
};

/** Properties of {@link GPURasterEmbeddingChange}. */
export type GPURasterEmbeddingChangeProps = {
  id?: string;
  pixelCount: number;
  /** Vector length, in `[1, 128]`. */
  dimensions: number;
  /** Decoded vectors of the earlier year; a pixel with a non-finite component is invalid. */
  before: GraphDataView<'float32'>;
  /** Decoded vectors of the later year. */
  after: GraphDataView<'float32'>;
  /** Per-pixel QA of the earlier year; zero makes the pixel invalid. */
  beforeMask?: GraphDataView<'uint32'>;
  /** Per-pixel QA of the later year. */
  afterMask?: GraphDataView<'uint32'>;
  /** Float32, see {@link getGPURasterEmbeddingChangeParameterValues}. Needed with `output.changed`. */
  parameters?: GraphDataView<'float32'>;
  output: GPURasterEmbeddingChangeOutput;
};

/**
 * Year-to-year change of an embedding raster, the use Google recommends for AlphaEarth: the dot
 * product between the years (cosine similarity of the decoded vectors), its angle, the length of
 * the change vector and a threshold mask. Pair the similarity raster with
 * `GPURasterFalseDiscoveryRateMask` or a percentile threshold when a map must show only
 * significant change.
 *
 * Per-frame: `parameters` and the contents of the views. Compile-time: `pixelCount`,
 * `dimensions` and which optional views exist.
 */
export class GPURasterEmbeddingChange implements GPUCommandNodeProducer {
  readonly id: string;

  constructor(readonly props: GPURasterEmbeddingChangeProps) {
    const id = props.id ?? 'raster-embedding-change';
    this.id = id;
    const {pixelCount, dimensions, output} = props;
    validateRasterEmbeddingShape(id, pixelCount, dimensions);
    validateRasterEmbeddingVectors(
      id,
      'before',
      props.before,
      pixelCount,
      dimensions,
      props.beforeMask
    );
    validateRasterEmbeddingVectors(
      id,
      'after',
      props.after,
      pixelCount,
      dimensions,
      props.afterMask
    );
    validateRequirement(
      id,
      Boolean(output.similarity || output.angle || output.magnitude || output.changed),
      'output needs at least one view'
    );
    validateRequirement(
      id,
      this.getOutputs().every(view => view.length === pixelCount),
      'outputs must hold pixelCount entries'
    );
    validateRequirement(
      id,
      !output.changed ||
        (props.parameters !== undefined &&
          props.parameters.length >= GPU_RASTER_EMBEDDING_CHANGE_PARAMETER_LENGTH),
      'output.changed needs parameters'
    );
    validateRasterViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterScalarView[] {
    const {before, after, beforeMask, afterMask, parameters} = this.props;
    return [
      before,
      after,
      ...(beforeMask ? [beforeMask] : []),
      ...(afterMask ? [afterMask] : []),
      ...(parameters ? [parameters] : [])
    ];
  }

  private getOutputs(): RasterScalarView[] {
    const {similarity, angle, magnitude, changed} = this.props.output;
    return [similarity, angle, magnitude, changed].filter(
      (view): view is NonNullable<typeof view> => view !== undefined
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {pixelCount, dimensions, output} = props;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    // The threshold kernel reads the similarity, so it exists even when the caller did not ask for it.
    const similarity =
      output.similarity ??
      (output.changed
        ? createTransientView(graph, `${id}-similarity`, 'float32', pixelCount)
        : undefined);
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterEmbeddingChange',
        invocationCount: pixelCount,
        workgroupSize: 64,
        bindings: [
          rasterBinding('before', props.before),
          rasterBinding('after', props.after),
          ...(props.beforeMask ? [rasterBinding('beforeMask', props.beforeMask)] : []),
          ...(props.afterMask ? [rasterBinding('afterMask', props.afterMask)] : []),
          ...(similarity ? [rasterBinding('similarity', similarity, true)] : []),
          ...(output.angle ? [rasterBinding('angle', output.angle, true)] : []),
          ...(output.magnitude ? [rasterBinding('magnitude', output.magnitude, true)] : [])
        ],
        declarations: `${RASTER_FINITE_WGSL}\n${GUARDED_ATAN2_WGSL}`,
        body: `var dotProduct = 0.0;
var beforeLength = 0.0;
var afterLength = 0.0;
var differenceLength = 0.0;
var isValid = ${
          [
            props.beforeMask ? 'beforeMask[beforeMaskOffset + index] != 0u' : '',
            props.afterMask ? 'afterMask[afterMaskOffset + index] != 0u' : ''
          ]
            .filter(Boolean)
            .join(' && ') || 'true'
        };
for (var band = 0u; band < ${dimensions}u; band++) {
  let left = before[beforeOffset + index * ${dimensions}u + band];
  let right = after[afterOffset + index * ${dimensions}u + band];
  if (!finite(left) || !finite(right)) { isValid = false; }
  dotProduct += left * right;
  beforeLength += left * left;
  afterLength += right * right;
  differenceLength += (right - left) * (right - left);
}
let denominator = sqrt(beforeLength * afterLength);
let cosine = select(select(0.0, 1.0, beforeLength == 0.0 && afterLength == 0.0), clamp(dotProduct / denominator, -1.0, 1.0), denominator > 0.0);
${similarity ? 'similarity[similarityOffset + index] = select(missing(), cosine, isValid);' : ''}
${
  output.angle
    ? `// acos of a float32 cosine cannot resolve angles below about 5e-4 rad (the cosine is within one
// float32 step of 1). The angle is taken from the chord and the sum of the two unit vectors,
// 2 atan2(|u - v|, |u + v|), which keeps its relative accuracy at every angle.
var angleValue = acos(cosine);
if (beforeLength > 0.0 && afterLength > 0.0) {
  let beforeScale = 1.0 / sqrt(beforeLength);
  let afterScale = 1.0 / sqrt(afterLength);
  var chordSquared = 0.0;
  var sumSquared = 0.0;
  for (var band = 0u; band < ${dimensions}u; band++) {
    let left = before[beforeOffset + index * ${dimensions}u + band] * beforeScale;
    let right = after[afterOffset + index * ${dimensions}u + band] * afterScale;
    chordSquared += (left - right) * (left - right);
    sumSquared += (left + right) * (left + right);
  }
  angleValue = 2.0 * guardedAtan2(sqrt(chordSquared), sqrt(sumSquared));
}
angle[angleOffset + index] = select(missing(), angleValue, isValid);`
    : ''
}
${output.magnitude ? 'magnitude[magnitudeOffset + index] = select(missing(), sqrt(differenceLength), isValid);' : ''}`
      })
    ];
    if (output.changed && similarity) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-threshold`,
          operation: 'GPURasterEmbeddingChange',
          variant: 'threshold',
          invocationCount: pixelCount,
          bindings: [
            rasterBinding('similarity', similarity),
            rasterBinding('parameters', props.parameters!),
            rasterBinding('changed', output.changed, true)
          ],
          // NaN (an invalid pixel) compares false.
          body: 'changed[changedOffset + index] = select(0u, 1u, similarity[similarityOffset + index] < parameters[parametersOffset]);'
        })
      );
    }
    return nodes;
  }
}
