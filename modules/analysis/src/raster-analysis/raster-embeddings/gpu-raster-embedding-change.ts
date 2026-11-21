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
import {GUARDED_ATAN2_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  validateGPUStorageBindingByteLength,
  validateFiniteNumber,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';
import {RASTER_FINITE_WGSL, rasterBinding} from '../remote-sensing/raster-kernel-utils';
import {
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
  validateRequirement(
    'getGPURasterEmbeddingChangeParameterValues',
    target.length >= 1,
    'target must hold at least 1 float32 elements; received ' + target.length
  );
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
export type GPURasterEmbeddingChangeOutput<
  FloatRows extends RasterEmbeddingView<'float32'> = GraphDataView<'float32'>,
  UintRows extends RasterEmbeddingView<'uint32'> = GraphDataView<'uint32'>
> = {
  /** Cosine similarity of the two years' vectors, in `[-1, 1]`; NaN where either year is invalid. */
  similarity?: FloatRows;
  /**
   * Angle between the vectors in radians, in `[0, pi]`; NaN where invalid. Computed as
   * `2 atan2(|u - v|, |u + v|)` of the two unit vectors, not as `acos(similarity)`, so angles far
   * below the float32 step of the cosine (about 5e-4 rad) are resolved. Two zero vectors give 0;
   * exactly one zero vector gives `pi / 2`.
   */
  angle?: FloatRows;
  /** Length of the change vector `|after - before|`; NaN where invalid. */
  magnitude?: FloatRows;
  /** 1 where `similarity < similarityThreshold`, else 0. Needs `parameters`. */
  changed?: UintRows;
};

/** Properties of {@link GPURasterEmbeddingChange}. */
export type GPURasterEmbeddingChangeProps<
  FloatRows extends RasterEmbeddingView<'float32'> = GraphDataView<'float32'>,
  UintRows extends RasterEmbeddingView<'uint32'> = GraphDataView<'uint32'>
> = {
  id?: string;
  pixelCount: number;
  /** Vector length, in `[1, 1024]`. */
  dimensions: number;
  /** Decoded vectors of the earlier year; a pixel with a non-finite component is invalid. */
  before: FloatRows;
  /** Decoded vectors of the later year. */
  after: FloatRows;
  /** Per-pixel QA of the earlier year; zero makes the pixel invalid. */
  beforeMask?: UintRows;
  /** Per-pixel QA of the later year. */
  afterMask?: UintRows;
  /** Float32, see {@link getGPURasterEmbeddingChangeParameterValues}. Needed with `output.changed`. */
  parameters?: GraphDataView<'float32'>;
  output: GPURasterEmbeddingChangeOutput<FloatRows, UintRows>;
};

/**
 * Year-to-year change of an embedding raster, the use Google recommends for AlphaEarth: the dot
 * product between the years (cosine similarity of the decoded vectors), its angle, the length of
 * the change vector and a threshold mask. Pair the similarity raster with
 * `GPURasterFalseDiscoveryRateMask` or a percentile threshold when a map must show only
 * significant change.
 *
 * Pixel fields accept GraphVectorView chunks with matching complete-pixel boundaries; shared tables stay packed.
 * Each chunk has its own workgroup count and uses the same pipeline.
 *
 * Per-frame: `parameters` and the contents of the views. Compile-time: `pixelCount`,
 * `dimensions` and which optional views exist.
 * Pixel vectors accept 1–1,024 dimensions per complete pixel chunk.
 */
export class GPURasterEmbeddingChange<
  FloatRows extends RasterEmbeddingView<'float32'> = GraphDataView<'float32'>,
  UintRows extends RasterEmbeddingView<'uint32'> = GraphDataView<'uint32'>
> implements GPUCommandNodeProducer
{
  readonly id: string;
  private readonly chunks: readonly RasterEmbeddingChunk[];
  readonly estimatedCommandNodeCount: number;

  constructor(readonly props: GPURasterEmbeddingChangeProps<FloatRows, UintRows>) {
    props = {...props, output: {...props.output}};
    this.props = props;
    const id = props.id ?? 'raster-embedding-change';
    this.id = id;
    const {pixelCount, dimensions, output} = props;
    validateRasterEmbeddingShape(
      id,
      pixelCount,
      dimensions,
      MAXIMUM_RASTER_EMBEDDING_DIMENSIONS,
      false
    );
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
    if (props.parameters)
      validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    this.chunks = planRasterEmbeddingChunks(
      id,
      pixelCount,
      {
        before: {view: props.before, format: 'float32', components: dimensions},
        after: {view: props.after, format: 'float32', components: dimensions},
        beforeMask: {view: props.beforeMask, format: 'uint32', components: 1},
        afterMask: {view: props.afterMask, format: 'uint32', components: 1},
        'output.similarity': {view: props.output.similarity, format: 'float32', components: 1},
        'output.angle': {view: props.output.angle, format: 'float32', components: 1},
        'output.magnitude': {view: props.output.magnitude, format: 'float32', components: 1},
        'output.changed': {view: props.output.changed, format: 'uint32', components: 1}
      },
      'before',
      output.changed ? 2 : 1
    );
    validateRasterEmbeddingViews(id, this.getInputs(), this.getOutputs(), [
      {
        inputs: [props.before, props.after, props.beforeMask, props.afterMask].filter(
          (view): view is NonNullable<typeof view> => view !== undefined
        ),
        outputs: [output.similarity, output.angle, output.magnitude].filter(
          (view): view is NonNullable<typeof view> => view !== undefined
        )
      },
      ...(output.changed
        ? [
            {
              inputs: [props.parameters!],
              outputs: [output.similarity, output.changed].filter(
                (view): view is NonNullable<typeof view> => view !== undefined
              )
            }
          ]
        : [])
    ]);
    this.estimatedCommandNodeCount =
      1 + this.chunks.filter(chunk => chunk.pixelCount > 0).length * (output.changed ? 2 : 1);
  }

  private getInputs(): RasterEmbeddingScalarView[] {
    const {before, after, beforeMask, afterMask, parameters} = this.props;
    return [
      before,
      after,
      ...(beforeMask ? [beforeMask] : []),
      ...(afterMask ? [afterMask] : []),
      ...(parameters && this.props.output.changed ? [parameters] : [])
    ];
  }

  private getOutputs(): RasterEmbeddingScalarView[] {
    const {similarity, angle, magnitude, changed} = this.props.output;
    return [similarity, angle, magnitude, changed].filter(
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
        Number(Boolean(this.props.beforeMask)) +
        Number(Boolean(this.props.afterMask)) +
        Number(Boolean(this.props.output.similarity || this.props.output.changed)) +
        Number(Boolean(this.props.output.angle)) +
        Number(Boolean(this.props.output.magnitude)),
      this.chunks
    );
    if (!this.props.output.similarity && this.props.output.changed) {
      for (const chunk of this.chunks)
        validateGPUStorageBindingByteLength(
          graph.device,
          this.id,
          'similarity scratch',
          Math.max(chunk.pixelCount, 1) * 4,
          'reduce the embedding chunk pixelCount'
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
    props: PackedEmbeddingProps<GPURasterEmbeddingChangeProps>,
    chunkIndex: number
  ): RasterEmbeddingKernel[] {
    const id = `${this.id}-chunk-${chunkIndex}`;
    const {pixelCount, dimensions, output} = props;
    // The threshold kernel reads the similarity, so it exists even when the caller did not ask for it.
    const similarity =
      output.similarity ??
      (output.changed
        ? createTransientView(graph, `${id}-similarity`, 'float32', pixelCount)
        : undefined);
    const nodes: RasterEmbeddingKernel[] = [
      {
        id,
        operation: 'GPURasterEmbeddingChange',
        invocationCount: pixelCount,
        pixelCount,
        bindings: [
          rasterBinding('before', props.before),
          rasterBinding('after', props.after),
          ...(props.beforeMask ? [rasterBinding('beforeMask', props.beforeMask)] : []),
          ...(props.afterMask ? [rasterBinding('afterMask', props.afterMask)] : []),
          ...(similarity ? [rasterBinding('similarity', similarity, true)] : []),
          ...(output.angle ? [rasterBinding('angle', output.angle, true)] : []),
          ...(output.magnitude ? [rasterBinding('magnitude', output.magnitude, true)] : [])
        ],
        declarations: `${RASTER_FINITE_WGSL}\n${GUARDED_ATAN2_WGSL}\n${SCALED_QUOTIENT_WGSL}`,
        body: `var dotProduct = 0.0;
var beforeLength = 0.0;
var afterLength = 0.0;
var differenceLength = 0.0;
var largestBefore = 0.0;
var largestAfter = 0.0;
var largestDifference = 0.0;
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
  largestBefore = max(largestBefore, abs(left));
  largestAfter = max(largestAfter, abs(right));
  largestDifference = max(largestDifference, abs(right - left));
}
let beforeDivisor = select(1.0, largestBefore, largestBefore > 0.0);
let afterDivisor = select(1.0, largestAfter, largestAfter > 0.0);
let differenceDivisor = select(1.0, largestDifference, largestDifference > 0.0 && finite(largestDifference));
for (var band = 0u; band < ${dimensions}u; band++) {
  let beforeValue = before[beforeOffset + index * ${dimensions}u + band];
  let afterValue = after[afterOffset + index * ${dimensions}u + band];
  let left = getScaledQuotient(beforeValue, beforeDivisor, 1.0, 0);
  let right = getScaledQuotient(afterValue, afterDivisor, 1.0, 0);
  dotProduct += left * right;
  beforeLength += left * left;
  afterLength += right * right;
  let difference = getScaledQuotient(afterValue - beforeValue, differenceDivisor, 1.0, 0);
  differenceLength += difference * difference;
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
  var chordSquared = 0.0;
  var sumSquared = 0.0;
  for (var band = 0u; band < ${dimensions}u; band++) {
    let left = getScaledQuotient(before[beforeOffset + index * ${dimensions}u + band], beforeDivisor, sqrt(beforeLength), 0);
    let right = getScaledQuotient(after[afterOffset + index * ${dimensions}u + band], afterDivisor, sqrt(afterLength), 0);
    chordSquared += (left - right) * (left - right);
    sumSquared += (left + right) * (left + right);
  }
  angleValue = 2.0 * guardedAtan2(sqrt(chordSquared), sqrt(sumSquared));
}
angle[angleOffset + index] = select(missing(), angleValue, isValid);`
    : ''
}
${output.magnitude ? 'magnitude[magnitudeOffset + index] = select(missing(), sqrt(differenceLength) * largestDifference, isValid);' : ''}`
      }
    ];
    if (output.changed && similarity) {
      nodes.push({
        id: `${id}-threshold`,
        operation: 'GPURasterEmbeddingChange',
        variant: 'threshold',
        invocationCount: pixelCount,
        pixelCount,
        bindings: [
          // Similarity and changed can be disjoint writable ranges of one physical buffer.
          rasterBinding('similarity', similarity, true),
          rasterBinding('parameters', props.parameters!),
          rasterBinding('changed', output.changed, true)
        ],
        // NaN (an invalid pixel) compares false.
        body: 'changed[changedOffset + index] = select(0u, 1u, similarity[similarityOffset + index] < parameters[parametersOffset]);'
      });
    }
    return nodes;
  }
}
