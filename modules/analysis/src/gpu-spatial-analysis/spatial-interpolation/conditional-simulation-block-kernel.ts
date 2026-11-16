// luma.gl contributors
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
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateGraphChunkGroupKernelSupport} from '../../analysis-core/kernels/graph-chunk-groups';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {WGSLKernelProps} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Header plus offsets for the blocked simulation's at most eight storage bindings, padded to vec4u. */
const SIMULATION_BLOCK_PARAMETER_WORD_COUNT = 12;
/** Each cooperative update is a complete 16 by 16 tile; every lane must reach its barriers. */
const SIMULATION_BLOCK_WORKGROUP_SIZE = 256;

/**
 * A cooperative conditional-covariance update with runtime block position and binding offsets.
 * The row-oriented chunk helper always guards individual lanes, so it cannot host barriers.
 * This adapter guards complete workgroups and uses the public Kernel/command-node APIs.
 */
export function createConditionalSimulationBlockNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: Omit<WGSLKernelProps<Parameters>, 'constants' | 'guardIndex'> & {blockStart: number}
): GPUCommandNode<Parameters> {
  validateGraphChunkGroupKernelSupport(
    graph,
    props.id,
    props.bindings.length,
    SIMULATION_BLOCK_WORKGROUP_SIZE
  );
  validateGraphChunkBindingRanges(
    graph,
    props.id,
    Object.fromEntries(props.bindings.map(binding => [binding.name, binding.view]))
  );
  validateRequirement(
    props.id,
    props.bindings.length + 4 <= SIMULATION_BLOCK_PARAMETER_WORD_COUNT &&
      props.invocationCount % SIMULATION_BLOCK_WORKGROUP_SIZE === 0,
    'blocked update needs at most eight storage bindings and complete 256-invocation tiles'
  );
  const parameters = new Uint32Array(SIMULATION_BLOCK_PARAMETER_WORD_COUNT);
  parameters[0] = props.blockStart;
  parameters[1] = props.invocationCount / SIMULATION_BLOCK_WORKGROUP_SIZE;
  props.bindings.forEach((binding, index) => {
    parameters[4 + index] = getViewElementOffset(binding.view);
  });
  const layout = getBoundedDispatchLayout(
    props.id,
    props.invocationCount,
    SIMULATION_BLOCK_WORKGROUP_SIZE,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const source = `${props.bindings
    .map(
      (binding, location) =>
        `@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;\nvar<private> ${binding.name}Offset: u32;`
    )
    .join('\n')}
struct SimulationBlockParameters { words: array<vec4u, ${SIMULATION_BLOCK_PARAMETER_WORD_COUNT / 4}> }
@group(0) @binding(${props.bindings.length}) var<uniform> simulationBlockParameters: SimulationBlockParameters;
fn getChunkParameter(word: u32) -> u32 { return simulationBlockParameters.words[word / 4u][word % 4u]; }
${props.declarations ?? ''}
@compute @workgroup_size(${SIMULATION_BLOCK_WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
  @builtin(num_workgroups) workgroupCount: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32) {
  // Address-only references retain every declared binding in WebGPU's inferred layout.
  ${props.bindings.map(binding => `_ = &${binding.name};`).join('\n  ')}
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  // All lanes in a padded workgroup return together. Never guard local invocation indices.
  if (workgroupIndex >= getChunkParameter(1u)) { return; }
  let index = workgroupIndex * ${SIMULATION_BLOCK_WORKGROUP_SIZE}u + localInvocationIndex;
  ${props.bindings.map((binding, index) => `${binding.name}Offset = getChunkParameter(${4 + index}u);`).join('\n  ')}
  ${props.body}
}`;
  return createGPUComputeCommandNode({
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
      maximumWorkgroupCount: layout.x * layout.y * layout.z
    },
    compile: ({device}) => {
      const parameterBuffer = device.createBuffer({
        id: `${props.id}-block-parameters`,
        data: parameters,
        usage: Buffer.UNIFORM | Buffer.COPY_DST
      });
      let kernel: Kernel;
      try {
        kernel = new Kernel(device, {
          id: `${props.operation}-${props.variant}`,
          source,
          shaderLayout: {
            bindings: [
              ...props.bindings.map((binding, location) => ({
                name: binding.name,
                type:
                  binding.access === 'read' ? ('read-only-storage' as const) : ('storage' as const),
                group: 0,
                location
              })),
              {
                name: 'simulationBlockParameters',
                type: 'uniform',
                group: 0,
                location: props.bindings.length
              }
            ]
          }
        });
      } catch (error) {
        parameterBuffer.destroy();
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
              [
                'simulationBlockParameters',
                {buffer: parameterBuffer, offset: 0, size: parameters.byteLength}
              ]
            ]),
            ...layout
          }),
        destroy: () => {
          kernel.destroy();
          parameterBuffer.destroy();
        }
      };
    }
  });
}
