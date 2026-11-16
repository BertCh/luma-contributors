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

/**
 * GWR batch dispatch: bounds come from the shared GPU cursor, while dispatch dimensions belong
 * to each node. num_workgroups linearizes the actual dispatch without pipeline specialization.
 * Callers preflight bindings and guard padded workgroup lanes with their runtime row/run bound.
 * @internal
 */
export function createGWRRuntimeKernelNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: WGSLKernelProps<Parameters>
): GPUCommandNode<Parameters> {
  const workgroupSize = props.workgroupSize ?? 256;
  const layout = getBoundedDispatchLayout(
    props.operation,
    Math.max(props.invocationCount, 1),
    workgroupSize,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const source = `${props.bindings
    .map(
      (
        binding,
        location
      ) => `const ${binding.name}Offset: u32 = ${getViewElementOffset(binding.view)}u;
@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
    )
    .join('\n')}
${props.declarations ?? ''}
@compute @workgroup_size(${workgroupSize})
fn main(@builtin(local_invocation_index) localInvocationIndex: u32,
        @builtin(workgroup_id) workgroupId: vec3u,
        @builtin(num_workgroups) workgroupCount: vec3u) {
  ${props.bindings.map(binding => `_ = &${binding.name};`).join('\n  ')}
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  let index = workgroupIndex * ${workgroupSize}u + localInvocationIndex;
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
      maximumWorkgroupCount: layout.x * layout.y * layout.z
    },
    compile: ({device}) => {
      const kernel = new Kernel(device, {
        id: props.id,
        source,
        shaderLayout: {
          bindings: props.bindings.map((binding, location) => ({
            name: binding.name,
            type: binding.access === 'read' ? ('read-only-storage' as const) : ('storage' as const),
            group: 0,
            location
          }))
        }
      });
      return {
        encode: ({computePass, getBuffer}) =>
          kernel.dispatch(computePass, {
            bindings: Object.fromEntries(
              props.bindings.map(binding => [
                binding.name,
                getViewBinding(binding.view, () => getBuffer(binding.view))
              ])
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
