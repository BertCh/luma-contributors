// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Kernel} from '@luma.gl/graph';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import type {WGSLKernelProps} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Fixed lane count makes row shaders independent of tile length and bounded dispatch shape. */
const DORLING_ROW_WORKGROUP_SIZE = 256;

/**
 * One Dorling row tile. Its own invocation count determines only node dispatch dimensions;
 * num_workgroups flattens those dimensions and the GPU state supplies the tile base and length.
 * All bindings and their offsets are identical across tiles. No tile pipeline constants exist.
 */
export function createDorlingRowTileNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: WGSLKernelProps<Parameters>
): GPUCommandNode<Parameters> {
  const layout = getBoundedDispatchLayout(
    props.operation,
    props.invocationCount,
    DORLING_ROW_WORKGROUP_SIZE,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const bindings = props.bindings;
  const source = `${bindings
    .map(
      (binding, location) => `
const ${binding.name}Offset: u32 = ${getViewElementOffset(binding.view)}u;
@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
    )
    .join('\n')}
${props.declarations ?? ''}
@compute @workgroup_size(${DORLING_ROW_WORKGROUP_SIZE})
fn main(@builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(num_workgroups) workgroupCount: vec3u) {
  ${bindings.map(binding => `_ = &${binding.name};`).join('\n  ')}
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex > 0x00ffffffu) { return; }
  let index = workgroupIndex * ${DORLING_ROW_WORKGROUP_SIZE}u + localInvocationIndex;
  ${props.body}
}`;
  return createGPUComputeCommandNode({
    id: props.id,
    resources: bindings.map(binding => ({
      buffer: binding.view,
      usage: binding.access === 'read' ? 'storage-read' : 'storage-read-write'
    })),
    workload: {
      operation: props.operation,
      variant: props.variant,
      commandCount: 1,
      maximumWorkgroupCount: layout.x * layout.y * layout.z,
      maximumInvocationCount: props.invocationCount
    },
    compile: ({device}) => {
      const kernel = new Kernel(device, {
        id: props.id,
        source,
        shaderLayout: {
          bindings: bindings.map((binding, location) => ({
            name: binding.name,
            type: binding.access === 'read' ? 'read-only-storage' : 'storage',
            group: 0,
            location
          }))
        }
      });
      return {
        encode: ({computePass, getBuffer}) =>
          kernel.dispatch(computePass, {
            bindings: Object.fromEntries(
              bindings.map(binding => [binding.name, getViewBinding(binding.view, getBuffer)])
            ),
            x: layout.x,
            y: layout.y,
            z: layout.z
          }),
        destroy: () => kernel.destroy()
      };
    }
  });
}
