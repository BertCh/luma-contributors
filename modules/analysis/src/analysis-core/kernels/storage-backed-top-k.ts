// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Binding} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  getViewBindingRange,
  getViewElementOffset,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {validateGPUStorageBindingByteLength, validateRequirement} from './gpu-contributor-utils';
import type {WGSLKernelBinding, WGSLKernelProps} from './wgsl-kernel-nodes';

/** WGSL accessors for an invocation-owned, interleaved `(row, float32 distance bits)` list. @internal */
export type StorageTopKAccessors = {
  readRow(slot: string): string;
  readDistance(slot: string): string;
  writeRow(slot: string, value: string): string;
  writeDistance(slot: string, value: string): string;
};

/**
 * Generates storage access for bounded spatial top-k insertion lists. `base` is the first row
 * word of this invocation's list. Every invocation must own a disjoint range of `2 * capacity`
 * words, and the caller must check the binding size before declaring graph resources.
 *
 * Keeping float32 distances as bits preserves the private-list route's comparisons and ties;
 * storage replaces only the list's backing memory, without adding dispatches per candidate or k.
 * Binding names and expressions are internal WGSL fragments, fixed for all dispatch batches.
 * @internal
 */
export function getStorageTopKAccessors(binding: string, base: string): StorageTopKAccessors {
  const row = (slot: string) => `${binding}[(${base}) + (${slot}) * 2u]`;
  const distance = (slot: string) => `${binding}[(${base}) + (${slot}) * 2u + 1u]`;
  return {
    readRow: row,
    readDistance: slot => `bitcast<f32>(${distance(slot)})`,
    writeRow: (slot, value) => `${row(slot)} = ${value};`,
    writeDistance: (slot, value) => `${distance(slot)} = bitcast<u32>(${value});`
  };
}

/** Fixed workgroup shape avoids specializing pipelines for short batches. */
const TOP_K_WORKGROUP_SIZE = 64;
/** Count, eight scalar binding offsets, row base and batch index in three uniform vec4 rows. */
const TOP_K_PARAMETER_WORD_COUNT = 12;
/** The standard WebGPU storage binding limit bounds the runtime offset table. */
const TOP_K_MAXIMUM_BINDINGS = 8;

/** Generated kernel facts used by the spatial top-k audit specs. @internal */
export type StorageTopKNodeRecord = {
  source: string;
  settings: Uint32Array;
  bindings: readonly WGSLKernelBinding[];
  layout: {x: number; y: number; z: number};
};

const TOP_K_NODE_RECORDS = new WeakMap<object, StorageTopKNodeRecord>();

/** Returns immutable dispatch facts without compiling or allocating GPU resources. @internal */
export function getStorageTopKNodeRecord(node: object): StorageTopKNodeRecord | undefined {
  return TOP_K_NODE_RECORDS.get(node);
}

/**
 * Records a spatial top-k pass using the published Kernel uniform API. Counts and aligned binding
 * offsets are runtime settings; each node dispatches only its own batch's workgroups. No count,
 * row base, binding offset or workgroup layout is a pipeline constant. Callers preflight all views
 * and derived scratch before calling this helper. @internal
 */
export function createStorageTopKNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: Pick<
    WGSLKernelProps<Parameters>,
    'id' | 'operation' | 'variant' | 'bindings' | 'invocationCount' | 'declarations' | 'body'
  > & {firstRow?: number; batchIndex?: number}
): GPUCommandNode<Parameters> {
  validateRequirement(
    props.id,
    props.bindings.length <=
      Math.min(TOP_K_MAXIMUM_BINDINGS, graph.device.limits.maxStorageBuffersPerShaderStage),
    'bindings must fit TOP_K_MAXIMUM_BINDINGS (8) and the device storage binding limit'
  );
  const settings = new Uint32Array(TOP_K_PARAMETER_WORD_COUNT);
  settings[0] = props.invocationCount;
  settings[9] = props.firstRow ?? 0;
  settings[10] = props.batchIndex ?? 0;
  let declarations = props.declarations ?? '';
  let body = props.body;
  for (const [bindingIndex, binding] of props.bindings.entries()) {
    validateGPUStorageBindingByteLength(
      graph.device,
      props.id,
      binding.name,
      getViewBindingRange(binding.view).size,
      'reduce the bound view length'
    );
    const word = bindingIndex + 1;
    settings[word] = getViewElementOffset(binding.view);
    const offset = `selectionSettings[${Math.floor(word / 4)}].${'xyzw'[word % 4]}`;
    const pattern = new RegExp(`\\b${binding.name}Offset\\b`, 'g');
    declarations = declarations.replace(pattern, offset);
    body = body.replace(pattern, offset);
  }
  const source = `${props.bindings
    .map(
      (binding, bindingIndex) =>
        `@group(0) @binding(${bindingIndex}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
    )
    .join('\n')}
@group(0) @binding(${props.bindings.length}) var<uniform> selectionSettings: array<vec4<u32>, 3>;
${declarations}
@compute @workgroup_size(${TOP_K_WORKGROUP_SIZE})
fn main(@builtin(local_invocation_index) localInvocationIndex: u32,
        @builtin(workgroup_id) workgroupId: vec3<u32>,
        @builtin(num_workgroups) workgroupCounts: vec3<u32>) {
  ${props.bindings.map(binding => `_ = &${binding.name};`).join('\n')}
  let index = ((workgroupId.z * workgroupCounts.y + workgroupId.y) * workgroupCounts.x + workgroupId.x) * ${TOP_K_WORKGROUP_SIZE}u + localInvocationIndex;
  if (index >= selectionSettings[0].x) { return; }
  let batchFirstRow = selectionSettings[2].y;
  let batchIndex = selectionSettings[2].z;
  ${body}
}`;
  const layout = getBoundedDispatchLayout(
    props.operation,
    Math.max(1, props.invocationCount),
    TOP_K_WORKGROUP_SIZE,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const node = createGPUComputeCommandNode<Parameters>({
    id: props.id,
    resources: props.bindings.map(binding => ({
      buffer: binding.view,
      usage: binding.access === 'read' ? ('storage-read' as const) : ('storage-read-write' as const)
    })),
    workload: {
      operation: props.operation,
      ...(props.variant ? {variant: props.variant} : {}),
      maximumInvocationCount: props.invocationCount,
      maximumWorkgroupCount: layout.x * layout.y * layout.z
    },
    compile: ({device}) => {
      const configuration = device.createBuffer({
        id: `${props.id}-selection-settings`,
        data: settings,
        usage: Buffer.UNIFORM | Buffer.COPY_DST
      });
      let kernel: Kernel;
      try {
        kernel = new Kernel(device, {
          id: props.id,
          source,
          shaderLayout: {
            bindings: [
              ...props.bindings.map((binding, bindingIndex) => ({
                name: binding.name,
                type:
                  binding.access === 'read' ? ('read-only-storage' as const) : ('storage' as const),
                group: 0,
                location: bindingIndex
              })),
              {
                name: 'selectionSettings',
                type: 'uniform',
                group: 0,
                location: props.bindings.length
              }
            ]
          }
        });
      } catch (error) {
        // The uniform is node-owned; release it when the pipeline is refused.
        configuration.destroy();
        throw error;
      }
      return {
        encode: ({computePass, getBuffer}) => {
          const bindings: Record<string, Binding> = {selectionSettings: configuration};
          for (const binding of props.bindings)
            bindings[binding.name] = getViewBinding(binding.view, getBuffer);
          kernel.dispatch(computePass, {bindings, x: layout.x, y: layout.y, z: layout.z});
        },
        destroy: () => {
          kernel.destroy();
          configuration.destroy();
        }
      };
    }
  });
  TOP_K_NODE_RECORDS.set(node, {source, settings, bindings: props.bindings, layout});
  return node;
}
