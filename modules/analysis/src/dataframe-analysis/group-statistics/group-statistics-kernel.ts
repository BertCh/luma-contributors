// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Binding} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {validateMaximumCount} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {WGSLKernelProps} from '../../analysis-core/kernels/wgsl-kernel-nodes';

// Eight storage offsets plus dispatch, row-base and tree-level metadata occupy four vec4 words.
export const GROUP_DISPATCH_PARAMETER_BYTE_LENGTH = 64;
// One uniform record supports the core WebGPU limit without consuming a ninth storage binding.
const MAXIMUM_GROUP_STORAGE_BINDINGS = 8;
// Fixed across chunks and tree levels so neither the WGSL nor pipeline specialization changes.
const GROUP_WORKGROUP_SIZE = 256;
// A 256-lane workgroup index below 2^24 cannot overflow the uint32 linear invocation index.
const MAXIMUM_GROUP_LINEAR_WORKGROUP_COUNT = 2 ** 24;

/**
 * Analysis dispatch adapter using the published Kernel and command-node APIs. Unlike the shared
 * static kernel helper, it binds row bases, byte-offset prefixes, invocation counts and tree
 * widths as uniform data. The compiled node owns and destroys its immutable uniform record.
 * Storage ranges and the uniform size are preflighted by GPUGroupStatistics before any nodes.
 * @internal
 */
export function createGroupStatisticsKernelNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: Pick<
    WGSLKernelProps<Parameters>,
    | 'id'
    | 'operation'
    | 'variant'
    | 'bindings'
    | 'invocationCount'
    | 'body'
    | 'declarations'
    | 'guardIndex'
  > & {
    rowOffset?: number;
    halfWidth?: number;
  }
): GPUCommandNode<Parameters> {
  validateMaximumCount(
    props.id,
    'kernel storage bindings',
    props.bindings.length,
    MAXIMUM_GROUP_STORAGE_BINDINGS
  );
  const dispatch = getBoundedDispatchLayout(
    props.operation,
    Math.max(props.invocationCount, 1),
    GROUP_WORKGROUP_SIZE,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const parameters = new Uint32Array(GROUP_DISPATCH_PARAMETER_BYTE_LENGTH / 4);
  parameters.set([props.invocationCount, dispatch.x, dispatch.y, props.rowOffset ?? 0]);
  parameters[12] = props.halfWidth ?? 0;
  const replacements = new Map<string, string>([
    ['INVOCATION_COUNT', 'groupParameters[0].x'],
    ['GROUP_ROW_OFFSET', 'groupParameters[0].w'],
    ['GROUP_HALF_WIDTH', 'groupParameters[3].x']
  ]);
  for (const [location, binding] of props.bindings.entries()) {
    parameters[4 + location] = getViewElementOffset(binding.view);
    replacements.set(
      `${binding.name}Offset`,
      `groupParameters[${1 + Math.floor(location / 4)}][${location % 4}]`
    );
  }
  const replaceParameters = (source: string) =>
    source.replace(/\b[A-Za-z_][A-Za-z_0-9]*\b/g, identifier =>
      replacements.has(identifier) ? `(${replacements.get(identifier)})` : identifier
    );
  const source = `
${props.bindings.map((binding, location) => `@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`).join('\n')}
@group(0) @binding(${props.bindings.length}) var<uniform> groupParameters: array<vec4u, 4>;
${replaceParameters(props.declarations ?? '')}
@compute @workgroup_size(${GROUP_WORKGROUP_SIZE})
fn main(@builtin(local_invocation_index) localInvocationIndex: u32,
        @builtin(workgroup_id) workgroupId: vec3u) {
  ${props.bindings.map(binding => `_ = &${binding.name};`).join('\n  ')}
  let workgroupIndex = (workgroupId.z * groupParameters[0].z + workgroupId.y) * groupParameters[0].y + workgroupId.x;
  if (workgroupIndex >= ${MAXIMUM_GROUP_LINEAR_WORKGROUP_COUNT}u) { return; }
  let index = workgroupIndex * ${GROUP_WORKGROUP_SIZE}u + localInvocationIndex;
  ${props.guardIndex === false ? '' : 'if (index >= groupParameters[0].x) { return; }'}
  ${replaceParameters(props.body)}
}`;
  return createGPUComputeCommandNode<Parameters>({
    id: props.id,
    resources: props.bindings.map(binding => ({
      buffer: binding.view,
      usage: binding.access === 'read' ? 'storage-read' : 'storage-read-write'
    })),
    workload: {
      operation: props.operation,
      ...(props.variant ? {variant: props.variant} : {}),
      commandCount: 1,
      maximumInvocationCount: props.invocationCount,
      maximumWorkgroupCount: dispatch.x * dispatch.y * dispatch.z
    },
    compile: ({device}) => {
      const uniform = device.createBuffer({
        id: `${props.id}-parameters`,
        usage: Buffer.UNIFORM | Buffer.COPY_DST,
        data: parameters
      });
      const kernel = new Kernel(device, {
        id: props.id,
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
            {name: 'groupParameters', type: 'uniform', group: 0, location: props.bindings.length}
          ]
        }
      });
      return {
        encode: ({computePass, getBuffer}) => {
          const bindings: Record<string, Binding> = {groupParameters: uniform};
          for (const binding of props.bindings)
            bindings[binding.name] = getViewBinding(binding.view, getBuffer);
          kernel.dispatch(computePass, {bindings, x: dispatch.x, y: dispatch.y, z: dispatch.z});
        },
        destroy: () => {
          kernel.destroy();
          uniform.destroy();
        }
      };
    }
  });
}
