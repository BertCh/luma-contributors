// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {
  createGraphChunkGroupKernelNode,
  validateGraphChunkGroupKernelSupport,
  GRAPH_CHUNK_GROUP_PARAMETER_WORD_COUNT,
  GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET,
  type GraphChunkGroupKernelProps
} from '../../analysis-core/kernels/graph-chunk-groups';

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Full-workgroup histogram and moment kernels use WebGPU's standard 256-thread tile. */
const CLASSIFICATION_COLLECTIVE_WORKGROUP_SIZE = 256;

/** Collective histogram/reduction kernels need a workgroup-uniform guard before any barrier. */
export type ClassificationKernelProps<Parameters> = GraphChunkGroupKernelProps<Parameters> & {
  workgroupCollective?: boolean;
};

/** Uses shared row dispatch, or a uniform whole-workgroup dispatch for classification collectives. */
export function createClassificationKernelNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: ClassificationKernelProps<Parameters>
): GPUCommandNode<Parameters> {
  if (!props.workgroupCollective) return createGraphChunkGroupKernelNode(graph, props);
  // The shared independent-row helper has a lane-dependent early return. A collective cannot use
  // that guard before workgroupBarrier; compose the published Kernel with a whole-group guard.
  const workgroupSize = props.workgroupSize ?? CLASSIFICATION_COLLECTIVE_WORKGROUP_SIZE;
  validateGraphChunkGroupKernelSupport(graph, props.id, props.bindings.length, workgroupSize);
  validateGraphChunkBindingRanges(
    graph,
    props.id,
    Object.fromEntries(props.bindings.map(binding => [binding.name, binding.view]))
  );
  validateRequirement(
    props.id,
    props.invocationCount % workgroupSize === 0,
    'collective invocationCount must hold complete workgroups'
  );
  const layout = getBoundedDispatchLayout(
    props.operation,
    props.invocationCount,
    workgroupSize,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const values = new Uint32Array(GRAPH_CHUNK_GROUP_PARAMETER_WORD_COUNT);
  values.set([0, props.invocationCount, props.invocationCount, 0]);
  for (const [bindingIndex, binding] of props.bindings.entries())
    values[4 + bindingIndex] = getViewElementOffset(binding.view);
  values.set(props.parameterValues ?? [], GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET);
  const source = `${props.bindings.map((binding, index) => `@group(0) @binding(${index}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;\nvar<private> ${binding.name}Offset: u32;`).join('\n')}
struct ClassificationWorkgroupParameters { words: array<vec4<u32>, ${GRAPH_CHUNK_GROUP_PARAMETER_WORD_COUNT / 4}> }
@group(0) @binding(${props.bindings.length}) var<uniform> classificationParameters: ClassificationWorkgroupParameters;
fn getChunkParameter(word: u32) -> u32 { return classificationParameters.words[word / 4u][word % 4u]; }
${props.declarations ?? ''}
@compute @workgroup_size(${workgroupSize})
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(num_workgroups) workgroupCount: vec3<u32>, @builtin(local_invocation_index) localInvocationIndex: u32) {
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= getChunkParameter(1u) / ${workgroupSize}u) { return; }
  let index = workgroupIndex * ${workgroupSize}u + localInvocationIndex;
  ${props.bindings.map((binding, bindingIndex) => `${binding.name}Offset = getChunkParameter(${4 + bindingIndex}u);`).join('\n')}
  ${props.bindings.map(binding => `_ = &${binding.name};`).join('\n')}
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
        .reduce((sum, binding) => sum + binding.view.length * binding.view.rowByteLength, 0),
      writeByteLength: props.bindings
        .filter(binding => binding.access !== 'read')
        .reduce((sum, binding) => sum + binding.view.length * binding.view.rowByteLength, 0)
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
          id: props.id,
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
                name: 'classificationParameters',
                location: props.bindings.length,
                group: 0,
                type: 'uniform' as const
              }
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
              ['classificationParameters', {buffer: parameters, offset: 0, size: values.byteLength}]
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
