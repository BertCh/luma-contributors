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
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';

// Each invocation visits at most 256 rows, preserving the original serial float32 sum order.
export const UTILIZATION_PREFIX_BLOCK_SIZE = 256;
// Four levels cover the published sort's 2^31-row maximum; three preserve the original tree.
const MINIMUM_PREFIX_LEVEL_COUNT = 3;
// A 256-lane workgroup below this bound cannot wrap its uint32 linear invocation index.
const MAXIMUM_LINEAR_WORKGROUP_COUNT = 2 ** 24;
// Count, three binding offsets, and four domain-specific words fit two uniform vec4 words.
export const UTILIZATION_PREFIX_PARAMETER_BYTE_LENGTH = 32;
// Aligned starts keep each level's storage binding separate from its neighbours.
const PREFIX_ALIGNMENT_ROWS = 64;

/** Recursive tree shape and aligned locations of its upper prefixes. @internal */
export function getUtilizationPrefixLayout(cellCount: number) {
  const lengths = [cellCount];
  while (
    lengths.length < MINIMUM_PREFIX_LEVEL_COUNT ||
    lengths[lengths.length - 1] > UTILIZATION_PREFIX_BLOCK_SIZE
  ) {
    lengths.push(Math.ceil(lengths[lengths.length - 1] / UTILIZATION_PREFIX_BLOCK_SIZE));
  }
  const upperOffsets: number[] = [];
  let upperLength = 0;
  for (const length of lengths.slice(1)) {
    upperLength = Math.ceil(upperLength / PREFIX_ALIGNMENT_ROWS) * PREFIX_ALIGNMENT_ROWS;
    upperOffsets.push(upperLength);
    upperLength += length;
  }
  return {lengths, upperOffsets, upperLength};
}

/** One bounded serial prefix invocation per block, with level sizes supplied as data. @internal */
export function createUtilizationPrefixNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    input: GraphDataView<'float32'>;
    prefix: GraphDataView<'float32'>;
    totals: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createUtilizationTreeNode(graph, {
    id: props.id,
    variant: 'prefix',
    count: Math.ceil(props.input.length / UTILIZATION_PREFIX_BLOCK_SIZE),
    bindings: [
      {name: 'inputValues', view: props.input, type: 'f32', access: 'read'},
      {name: 'prefixValues', view: props.prefix, type: 'f32', access: 'read_write'},
      {name: 'totals', view: props.totals, type: 'f32', access: 'read_write'}
    ],
    settings: [props.input.length],
    body: `
  var sum = 0.0;
  let end = min((index + 1u) * ${UTILIZATION_PREFIX_BLOCK_SIZE}u, treeParameters[1].x);
  for (var row = index * ${UTILIZATION_PREFIX_BLOCK_SIZE}u; row < end; row = row + 1u) {
    prefixValues[treeParameters[0].z + row] = sum;
    sum = sum + inputValues[treeParameters[0].y + row];
  }
  totals[treeParameters[0].w + index] = sum;`
  });
}

/** Adds upper prefixes from the lowest level upward, retaining the legacy association. @internal */
export function createUtilizationOffsetNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    localPrefix: GraphDataView<'float32'>;
    upperPrefix: GraphDataView<'float32'>;
    upperOffsets: readonly number[];
  }
): GPUCommandNode<Parameters> {
  return createUtilizationTreeNode(graph, {
    id: props.id,
    variant: 'offsets',
    count: props.localPrefix.length,
    bindings: [
      {name: 'localPrefix', view: props.localPrefix, type: 'f32', access: 'read_write'},
      {name: 'upperPrefix', view: props.upperPrefix, type: 'f32', access: 'read'}
    ],
    settings: [props.upperOffsets.length, ...props.upperOffsets],
    body: `
  var preceding = localPrefix[treeParameters[0].y + index];
  var parent = index;
  for (var level = 0u; level < treeParameters[1].x; level = level + 1u) {
    parent = parent / ${UTILIZATION_PREFIX_BLOCK_SIZE}u;
    preceding = preceding + upperPrefix[treeParameters[0].z + treeParameters[1][level + 1u] + parent];
  }
  localPrefix[treeParameters[0].y + index] = preceding;`
  });
}

// The shared static kernel helper specializes invocation counts. Tree levels instead use an
// immutable uniform record with the published Kernel API, so all levels share one pipeline.
function createUtilizationTreeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    variant: string;
    count: number;
    bindings: readonly WGSLKernelBinding[];
    settings: readonly number[];
    body: string;
  }
): GPUCommandNode<Parameters> {
  const layout = getBoundedDispatchLayout(
    props.id,
    props.count,
    UTILIZATION_PREFIX_BLOCK_SIZE,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const parameters = new Uint32Array(UTILIZATION_PREFIX_PARAMETER_BYTE_LENGTH / 4);
  parameters[0] = props.count;
  props.bindings.forEach((binding, location) => {
    parameters[location + 1] = getViewElementOffset(binding.view);
  });
  parameters.set(props.settings, 4);
  const source = `
${props.bindings.map((binding, location) => `@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`).join('\n')}
@group(0) @binding(${props.bindings.length}) var<uniform> treeParameters: array<vec4u, 2>;
@compute @workgroup_size(${UTILIZATION_PREFIX_BLOCK_SIZE})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
        @builtin(num_workgroups) workgroupCount: vec3u,
        @builtin(local_invocation_index) localInvocationIndex: u32) {
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${MAXIMUM_LINEAR_WORKGROUP_COUNT}u) { return; }
  let index = workgroupIndex * ${UTILIZATION_PREFIX_BLOCK_SIZE}u + localInvocationIndex;
  if (index >= treeParameters[0].x) { return; }
  ${props.body}
}`;
  return createGPUComputeCommandNode<Parameters>({
    id: props.id,
    resources: props.bindings.map(binding => ({
      buffer: binding.view,
      usage: binding.access === 'read' ? 'storage-read' : 'storage-read-write'
    })),
    workload: {
      operation: 'GPUUtilizationVolumeIsopleths',
      variant: props.variant,
      commandCount: 1,
      maximumInvocationCount: props.count,
      maximumWorkgroupCount: layout.x * layout.y * layout.z
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
              group: 0,
              location,
              type:
                binding.access === 'read' ? ('read-only-storage' as const) : ('storage' as const)
            })),
            {name: 'treeParameters', group: 0, location: props.bindings.length, type: 'uniform'}
          ]
        }
      });
      return {
        encode: ({computePass, getBuffer}) => {
          const bindings: Record<string, Binding> = {treeParameters: uniform};
          for (const binding of props.bindings) {
            bindings[binding.name] = getViewBinding(binding.view, getBuffer);
          }
          kernel.dispatch(computePass, {bindings, ...layout});
        },
        destroy: () => {
          kernel.destroy();
          uniform.destroy();
        }
      };
    }
  });
}
