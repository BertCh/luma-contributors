// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Kernel} from '@luma.gl/graph';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  registerCompiledWGSLKernelNodeRecord,
  registerWGSLKernelNodeRecord,
  type WGSLKernelNodeRecord,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** A full workgroup folds cell rows; the same shader serves every batch length. */
export const CELL_ROW_WORKGROUP_SIZE = 256;
/** Prevent uint32 multiplication wrapping in padded multidimensional dispatches. */
const CELL_ROW_MAXIMUM_LINEAR_WORKGROUP_COUNT = 0x1000000;

/** Runtime offsets are derived from the exact bound range and logical component counts. @internal */
export type CellRowKernelBinding = WGSLKernelBinding & {
  /** Components per row; global scratch uses totalLength rather than this chunk's length. */
  components: number;
  scope?: 'chunk' | 'total';
};

/**
 * Declares a cell-row dispatch with chunk-local workgroups and no pipeline overrides. Slots hold
 * [rowBase, length, totalLength]. Every binding prefix is found from arrayLength, so unaligned
 * subviews do not specialize the source either. The caller preflights views and slot storage.
 * @internal
 */
export function createCellRowKernelNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    /** Stable pipeline label across chunks; defaults to the command node ID. */
    kernelId?: string;
    operation: string;
    variant: string;
    length: number;
    parameters: GraphDataView<'uint32'>;
    bindings: readonly CellRowKernelBinding[];
    declarations?: string;
    body: string;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'chunkParameters', view: props.parameters, type: 'u32', access: 'read'},
    ...props.bindings
  ];
  const declarations = bindings
    .map(
      (binding, location) =>
        `@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
    )
    .join('\n');
  const offsets = props.bindings
    .map(
      binding =>
        `let ${binding.name}Offset = arrayLength(&${binding.name}) - ${binding.scope === 'total' ? 'chunkParameters[2]' : 'rowCount'} * ${binding.components}u;`
    )
    .join('\n  ');
  const source = `${declarations}
${props.declarations ?? ''}
@compute @workgroup_size(${CELL_ROW_WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
  @builtin(num_workgroups) workgroupCount: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32) {
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${CELL_ROW_MAXIMUM_LINEAR_WORKGROUP_COUNT}u) { return; }
  let index = workgroupIndex * ${CELL_ROW_WORKGROUP_SIZE}u + localInvocationIndex;
  let rowCount = chunkParameters[1];
  if (index >= rowCount) { return; }
  let chunkParametersOffset = 0u;
  ${offsets}
  ${props.body}
}`;
  const layout = getBoundedDispatchLayout(
    props.operation,
    Math.max(props.length, 1),
    CELL_ROW_WORKGROUP_SIZE,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const kernelRecord: WGSLKernelNodeRecord = {
    id: props.id,
    operation: props.operation,
    variant: props.variant,
    source,
    body: props.body,
    declarations: props.declarations,
    entryPoint: 'main',
    workgroupSize: [CELL_ROW_WORKGROUP_SIZE, 1, 1],
    workgroups: [layout.x, layout.y, layout.z],
    invocationCount: props.length,
    constants: {},
    bindings,
    textures: [],
    indirect: false
  };
  registerWGSLKernelNodeRecord(graph, kernelRecord);
  return createGPUComputeCommandNode<Parameters>({
    id: props.id,
    resources: bindings.map(binding => ({
      buffer: binding.view,
      usage: binding.access === 'read' ? 'storage-read' : 'storage-read-write'
    })),
    workload: {
      operation: props.operation,
      variant: props.variant,
      commandCount: 1,
      maximumInvocationCount: props.length,
      maximumWorkgroupCount: layout.x * layout.y * layout.z
    },
    compile: function compileRecordedKernel({device}) {
      const kernel = new Kernel(device, {
        id: props.kernelId ?? props.id,
        source,
        shaderLayout: {
          bindings: bindings.map((binding, location) => ({
            name: binding.name,
            group: 0,
            location,
            type: binding.access === 'read' ? 'read-only-storage' : 'storage'
          }))
        }
      });
      registerCompiledWGSLKernelNodeRecord(graph, kernelRecord, compileRecordedKernel);
      return {
        encode: ({computePass, getBuffer}) =>
          kernel.dispatch(computePass, {
            bindings: Object.fromEntries(
              bindings.map(binding => [binding.name, getViewBinding(binding.view, getBuffer)])
            ),
            ...layout
          }),
        destroy: () => kernel.destroy()
      };
    }
  });
}
