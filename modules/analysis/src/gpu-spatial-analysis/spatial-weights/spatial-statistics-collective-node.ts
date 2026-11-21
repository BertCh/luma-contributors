// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  getViewBindingRange,
  getViewElementOffset,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  validateGraphChunkGroupKernelSupport,
  GRAPH_CHUNK_GROUP_PARAMETER_WORD_COUNT,
  GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET,
  type GraphChunkGroupKernelProps
} from '../../analysis-core/kernels/graph-chunk-groups';
import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement,
  validateSinglePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';

/**
 * Cooperative statistics workgroups with per-node dispatch dimensions and uniform bounds.
 * Unlike the independent-row helper, only a whole workgroup can return before the caller's
 * barriers. Chunk bounds, binding prefixes and batch indices never specialize the pipeline.
 * @internal
 */
export function createSpatialStatisticsCollectiveNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GraphChunkGroupKernelProps<Parameters> & {
    /** GPU-written, 256-byte scalar state, shared by fixed statistics stages. @internal */
    control?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const workgroupSize = props.workgroupSize ?? 256;
  validateGraphChunkGroupKernelSupport(graph, props.id, props.bindings.length, workgroupSize);
  validateGraphChunkBindingRanges(
    graph,
    props.id,
    Object.fromEntries(props.bindings.map(binding => [binding.name, binding.view]))
  );
  if (props.control) {
    validateSinglePackedView(props.control, ['uint32'], `${props.id} control`);
    validateCapacity(
      props.id,
      'control.length',
      props.control.length,
      GRAPH_CHUNK_GROUP_PARAMETER_WORD_COUNT
    );
    const range = getViewBindingRange(props.control);
    validateRequirement(
      props.id,
      range.offset === props.control.byteOffset,
      'control.byteOffset must start at an aligned uniform binding boundary; use an aligned 64-word control view'
    );
    validateMaximumCount(
      props.id,
      'control binding byteLength',
      range.size,
      graph.device.limits.maxUniformBufferBindingSize
    );
    validateMaximumCount(
      props.id,
      'control uniform bindings',
      2,
      graph.device.limits.maxUniformBuffersPerShaderStage
    );
    validateGraphViewsBelongToGraph(props.id, graph, [props.control]);
    validateGraphOutputsDisjointFromInputs(
      props.id,
      props.bindings
        .filter(binding => binding.access === 'read_write')
        .map(binding => binding.view),
      [props.control]
    );
  }
  validateCapacity(props.id, 'invocationCount', props.invocationCount);
  validateMaximumCount(
    props.id,
    'parameterValues length',
    props.parameterValues?.length ?? 0,
    GRAPH_CHUNK_GROUP_PARAMETER_WORD_COUNT - GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET
  );
  const values = new Uint32Array(GRAPH_CHUNK_GROUP_PARAMETER_WORD_COUNT);
  values[1] = props.invocationCount;
  for (const [bindingIndex, binding] of props.bindings.entries())
    values[4 + bindingIndex] = getViewElementOffset(binding.view);
  for (const [index, value] of (props.parameterValues ?? []).entries()) {
    validateCapacity(props.id, `parameterValues[${index}]`, value, 0);
    values[GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET + index] = value;
  }
  const layout = getBoundedDispatchLayout(
    props.id,
    props.invocationCount,
    workgroupSize,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const source = `${props.bindings.map((binding, index) => `@group(0) @binding(${index}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;\nvar<private> ${binding.name}Offset: u32;`).join('\n')}
struct StatisticsDispatchParameters { words: array<vec4<u32>, ${GRAPH_CHUNK_GROUP_PARAMETER_WORD_COUNT / 4}> }
@group(0) @binding(${props.bindings.length}) var<uniform> statisticsDispatchParameters: StatisticsDispatchParameters;
fn getChunkParameter(word: u32) -> u32 { return statisticsDispatchParameters.words[word / 4u][word % 4u]; }
${
  props.control
    ? `@group(0) @binding(${props.bindings.length + 1}) var<uniform> statisticsControl: StatisticsDispatchParameters;
fn getStatisticsControl(word: u32) -> u32 { return statisticsControl.words[word / 4u][word % 4u]; }`
    : ''
}
${props.declarations ?? ''}
@compute @workgroup_size(${workgroupSize})
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(num_workgroups) workgroupCount: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32) {
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  // A module-scope private load is nonuniform in WGSL, even after a uniform assignment.
  // Keep the dispatch bound immutable and local so the whole-workgroup return preserves barriers.
  let INVOCATION_COUNT = getChunkParameter(1u);
  if (workgroupIndex >= (INVOCATION_COUNT - 1u) / ${workgroupSize}u + 1u) { return; }
  let index = workgroupIndex * ${workgroupSize}u + localInvocationIndex;
  ${props.bindings.map((binding, index) => `${binding.name}Offset = getChunkParameter(${4 + index}u);\n  _ = &${binding.name};`).join('\n  ')}
  ${props.control ? '// Statically access this optional uniform even in stages that do not read its values.\n  _ = &statisticsControl;' : ''}
  ${props.body}
}`;
  return createGPUComputeCommandNode<Parameters>({
    id: props.id,
    condition: props.condition,
    resources: [
      ...props.bindings.map(binding => ({
        buffer: binding.view,
        usage:
          binding.access === 'read' ? ('storage-read' as const) : ('storage-read-write' as const)
      })),
      ...(props.control ? [{buffer: props.control, usage: 'uniform' as const}] : []),
      ...(props.extraResources ?? [])
    ],
    workload: {
      operation: props.operation,
      ...(props.variant ? {variant: props.variant} : {}),
      commandCount: 1,
      maximumInvocationCount: props.invocationCount,
      maximumWorkgroupCount: layout.x * layout.y * layout.z,
      readByteLength: props.bindings
        .filter(binding => binding.access === 'read')
        .reduce((total, binding) => total + binding.view.length * binding.view.rowByteLength, 0),
      writeByteLength: props.bindings
        .filter(binding => binding.access !== 'read')
        .reduce((total, binding) => total + binding.view.length * binding.view.rowByteLength, 0)
    },
    compile: ({device}) => {
      const parameters = device.createBuffer({
        id: `${props.id}-parameters`,
        data: values,
        usage: Buffer.UNIFORM | Buffer.COPY_DST
      });
      let kernel: Kernel;
      try {
        kernel = new Kernel(device, {
          id: props.pipelineId ?? props.id,
          source,
          shaderLayout: {
            bindings: [
              ...props.bindings.map((binding, location) => ({
                name: binding.name,
                location,
                group: 0,
                type:
                  binding.access === 'read' ? ('read-only-storage' as const) : ('storage' as const)
              })),
              {
                name: 'statisticsDispatchParameters',
                location: props.bindings.length,
                group: 0,
                type: 'uniform' as const
              },
              ...(props.control
                ? [
                    {
                      name: 'statisticsControl',
                      location: props.bindings.length + 1,
                      group: 0,
                      type: 'uniform' as const
                    }
                  ]
                : [])
            ]
          }
        });
      } catch (error) {
        parameters.destroy();
        throw error;
      }
      return {
        encode: ({computePass, getBuffer}) =>
          kernel.dispatch(computePass, {
            bindings: Object.fromEntries([
              ...props.bindings.map(binding => [
                binding.name,
                getViewBinding(binding.view, getBuffer)
              ]),
              ...(props.control
                ? [['statisticsControl', getViewBinding(props.control, getBuffer)]]
                : []),
              [
                'statisticsDispatchParameters',
                {buffer: parameters, offset: 0, size: values.byteLength}
              ]
            ]),
            ...layout
          }),
        destroy: () => {
          kernel.destroy();
          parameters.destroy();
        }
      };
    }
  });
}
