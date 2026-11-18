// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  validatePackedView,
  type GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {Kernel} from '@luma.gl/graph';
import {
  createWGSLKernelNode,
  type WGSLKernelProps
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  getGraphViewChunks,
  validateCapacity,
  validateCommandNodeBudget,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

import {
  createGraphChunkDispatchParameters,
  GRAPH_CHUNK_MAXIMUM_ROW_COUNT,
  planGraphChunkDispatch,
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';

// Fixed workgroups and the pre-multiplication guard preserve uint32 indices at large dispatches.
const GAUSSIAN_CHUNK_WORKGROUP_SIZE = 256;
// A 256-thread workgroup must be guarded before its index multiplication can wrap uint32.
const GAUSSIAN_MAXIMUM_LINEAR_WORKGROUP_COUNT = 0x1000000;

/** Properties for {@link GPUGaussianPredictiveScores}. */
export type GPUGaussianPredictiveScoresProps<
  Rows extends GraphDataView<'float32'> | GraphVectorView<'float32'> = GraphDataView<'float32'>
> = {
  /** Prefix for generated node IDs. Defaults to `'gaussian-predictive-scores'`. */
  id?: string;
  /** Predictive mean per site, for example the kriging prediction. */
  means: Rows;
  /**
   * Predictive variance per site in squared value units, for example the kriging variance. Zero
   * is a point forecast and scores `|y - mean|`; negative or NaN variance scores NaN.
   */
  variances: Rows;
  /** Observed value per site. */
  observations: Rows;
  /** Caller-owned outputs, one row per site (the length of `means`). */
  output: {
    /** Closed-form CRPS of the normal predictive distribution, `crps_norm(y, mean, sd)`. */
    crps: Rows;
  };
};

/**
 * Closed-form continuous ranked probability score of a Gaussian predictive distribution, for
 * kriging predictions `N(mean, variance)` (scoringRules 1.1.3 `crps_norm`, Gneiting et al. 2005):
 * with `z = |y - mean| / sd`, `CRPS = |y - mean| erf(z / sqrt(2)) +
 * sd * (2 phi(z) - 1 / sqrt(pi))`. Large errors use `|y - mean| - sd / sqrt(pi)` before
 * forming the standardized error, so a representable score cannot overflow that ratio.
 *
 * `2 Phi(z) - 1 = erf(z / sqrt(2))` is evaluated as the all-positive series
 * `erf(x) = 2 / sqrt(pi) * exp(-x^2) * sum_n 2^n x^(2n+1) / (2n + 1)!!` for `|x| <= 4` (no
 * cancellation in f32) and 1 beyond, which truncates `erf` by less than `1.6e-8`. One thread per
 * site; there are no reductions and no atomics. Lower is better.
 *
 * Chunked inputs and output must have identical ordered chunk lengths. Packed inputs preserve
 * the existing capacity contract: variances, observations and output may exceed means.length;
 * only the first means.length rows are scored.
 */
export class GPUGaussianPredictiveScores<
  Rows extends GraphDataView<'float32'> | GraphVectorView<'float32'> = GraphDataView<'float32'>
> implements GPUCommandNodeProducer
{
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGaussianPredictiveScoresProps<Rows>;
  /** Upper bound including the chunk parameter upload. */
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;
  private readonly chunked: boolean;

  constructor(props: GPUGaussianPredictiveScoresProps<Rows>) {
    this.id = props.id ?? 'gaussian-predictive-scores';
    this.props = props;
    const {id} = this;
    for (const [name, view] of [
      ['means', props.means],
      ['variances', props.variances],
      ['observations', props.observations],
      ['output.crps', props.output.crps]
    ] as const) {
      for (const chunk of getGraphViewChunks(view))
        validatePackedView(chunk, ['float32'], `${id} ${name}`);
    }
    const count = props.means.length;
    validateCapacity(id, 'means.length', count, 1, GRAPH_CHUNK_MAXIMUM_ROW_COUNT);
    validateRequirement(
      id,
      props.variances.length >= count &&
        props.observations.length >= count &&
        props.output.crps.length >= count,
      'variances, observations and output.crps need one row per mean'
    );
    this.chunked = [props.means, props.variances, props.observations, props.output.crps].some(
      view => !('buffer' in view)
    );
    this.chunkPlan = planGraphChunkDispatch(
      id,
      this.chunked
        ? {
            means: props.means,
            variances: props.variances,
            observations: props.observations,
            crps: props.output.crps
          }
        : {means: props.means}
    );
    validateDisjointWritableViews(id, {'output.crps': props.output.crps});
    this.estimatedCommandNodeCount = 1 + this.chunkPlan.nonemptyChunkCount;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.crps],
      [props.means, props.variances, props.observations]
    );
  }

  /** Returns the scoring kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphChunkDispatchBindings(graph, id, this.chunkPlan);
    validateGraphChunkBindingRanges(graph, id, {
      means: props.means,
      variances: props.variances,
      observations: props.observations,
      'output.crps': props.output.crps
    });
    const kernelProps: WGSLKernelProps<Parameters> = {
      id: `${id}-crps`,
      operation: 'GPUGaussianPredictiveScores',
      variant: 'crps-normal',
      bindings: [
        {name: 'means', view: getGraphViewChunks(props.means)[0], type: 'f32', access: 'read'},
        {
          name: 'variances',
          view: getGraphViewChunks(props.variances)[0],
          type: 'f32',
          access: 'read'
        },
        {
          name: 'observations',
          view: getGraphViewChunks(props.observations)[0],
          type: 'f32',
          access: 'read'
        },
        {
          name: 'crps',
          view: getGraphViewChunks(props.output.crps)[0],
          type: 'f32',
          access: 'read_write'
        }
      ],
      invocationCount: props.means.length,
      declarations: `const INVERSE_SQRT_PI: f32 = 0.5641895835;
const SQRT_TWO_OVER_PI: f32 = 0.7978845608;
// erf(x) for x >= 0: the positive series below 4, 1 above.
fn errorFunction(x: f32) -> f32 {
  if (x > 4.0) { return 1.0; }
  var term = x;
  var sum = x;
  let doubledSquare = 2.0 * x * x;
  for (var n = 1u; n < 160u; n++) {
    term = term * doubledSquare / f32(2u * n + 1u);
    sum += term;
    if (term < 1.0e-9 * sum) { break; }
  }
  return 1.1283791671 * exp(-x * x) * sum;
}`,
      body: `let variance = variances[variancesOffset + index];
  let error = observations[observationsOffset + index] - means[meansOffset + index];
  var score = bitcast<f32>(0x7fc00000u | (index & 0u));
  if (variance == 0.0) {
    score = abs(error);
  } else if (variance > 0.0) {
    let deviation = sqrt(variance);
    let absoluteError = abs(error);
    // Match the erf cutoff before division; the ratio may exceed the float32 range.
    if (absoluteError > 5.6568542495 * deviation) {
      score = absoluteError - deviation * INVERSE_SQRT_PI;
    } else {
      let standardizedError = absoluteError / deviation;
      score = absoluteError * errorFunction(standardizedError * 0.7071067812) +
        deviation * (SQRT_TWO_OVER_PI * exp(-0.5 * standardizedError * standardizedError) - INVERSE_SQRT_PI);
    }
  }
  crps[crpsOffset + index] = score;`
    };
    if (!this.chunked) return [createWGSLKernelNode(graph, kernelProps)];
    const layouts = this.chunkPlan.chunks.map(chunk =>
      getBoundedDispatchLayout(
        id,
        chunk.length,
        GAUSSIAN_CHUNK_WORKGROUP_SIZE,
        graph.device.limits.maxComputeWorkgroupsPerDimension
      )
    );
    const source = `
@group(0) @binding(0) var<storage, read> chunkParameters: array<u32>;
${kernelProps.bindings.map((binding, bindingIndex) => `@group(0) @binding(${bindingIndex + 1}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`).join('\n')}
${kernelProps.declarations}
@compute @workgroup_size(${GAUSSIAN_CHUNK_WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
        @builtin(num_workgroups) workgroupCount: vec3u,
        @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${kernelProps.bindings.map(binding => `_ = &${binding.name};`).join('\n  ')}
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${GAUSSIAN_MAXIMUM_LINEAR_WORKGROUP_COUNT}u) { return; }
  let index = workgroupIndex * ${GAUSSIAN_CHUNK_WORKGROUP_SIZE}u + localInvocationIndex;
  let rowCount = chunkParameters[1];
  if (index >= rowCount) { return; }
  ${kernelProps.bindings.map(binding => `let ${binding.name}Offset = arrayLength(&${binding.name}) - rowCount;`).join('\n  ')}
  ${kernelProps.body}
}`;
    const {slots, nodes: uploadNodes} = createGraphChunkDispatchParameters(graph, {
      id: `${id}-chunks`,
      plan: this.chunkPlan
    });
    return [
      ...uploadNodes,
      ...this.chunkPlan.chunks
        .filter(chunk => chunk.length > 0)
        .map(chunk => {
          const parameters = slots[chunk.chunkIndex];
          const chunkBindings = kernelProps.bindings.map(binding => ({
            ...binding,
            view: chunk.views[binding.name]
          }));
          return createGPUComputeCommandNode<Parameters>({
            id: `${id}-crps-${chunk.chunkIndex}`,
            workload: {
              operation: 'GPUGaussianPredictiveScores',
              variant: 'crps-normal',
              commandCount: 1,
              maximumInvocationCount: chunk.length,
              maximumWorkgroupCount:
                layouts[chunk.chunkIndex].x *
                layouts[chunk.chunkIndex].y *
                layouts[chunk.chunkIndex].z
            },
            resources: [
              {buffer: parameters, usage: 'storage-read'},
              ...chunkBindings.map(binding => ({
                buffer: binding.view,
                usage:
                  binding.access === 'read'
                    ? ('storage-read' as const)
                    : ('storage-read-write' as const)
              }))
            ],
            compile: ({device}) => {
              const kernel = new Kernel(device, {
                id: 'gaussian-predictive-scores-chunk',
                source,
                shaderLayout: {
                  bindings: [
                    {name: 'chunkParameters', group: 0, location: 0, type: 'read-only-storage'},
                    ...chunkBindings.map((binding, bindingIndex) => ({
                      name: binding.name,
                      group: 0,
                      location: bindingIndex + 1,
                      type:
                        binding.access === 'read'
                          ? ('read-only-storage' as const)
                          : ('storage' as const)
                    }))
                  ]
                }
              });
              return {
                encode: ({computePass, getBuffer}) =>
                  kernel.dispatch(computePass, {
                    bindings: {
                      chunkParameters: getViewBinding(parameters, getBuffer),
                      ...Object.fromEntries(
                        chunkBindings.map(binding => [
                          binding.name,
                          getViewBinding(binding.view, getBuffer)
                        ])
                      )
                    },
                    ...layouts[chunk.chunkIndex]
                  }),
                destroy: () => kernel.destroy()
              };
            }
          });
        })
    ];
  }
}
