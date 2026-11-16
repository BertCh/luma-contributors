// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Kernel} from '@luma.gl/graph';
import type {Device} from '@luma.gl/core';
import {
  validateGPUStorageBindingByteLength,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GPUCommandNode,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {getCoreDefaultBuffer} from '../../analysis-core/network-topology';
import type {WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Fixed workgroup width; tile lengths affect node dispatch geometry, never pipeline constants. */
const FRACTIONAL_TILE_WORKGROUP_SIZE = 256;

/**
 * Domain tile kernel with a runtime bound and node-local dispatch geometry. All scratch subviews
 * start at the same aligned offset; arrayLength observes the actual final tile length. @internal
 */
export function createRasterFractionalTileNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    invocationCount: number;
    bindings: readonly WGSLKernelBinding[];
    rowCount: string;
    /** Cooperative kernels use this uniform workgroup bound instead of a per-invocation guard. */
    workgroupCount?: string;
    declarations?: string;
    body: string;
  }
): GPUCommandNode<Parameters> {
  const layout = getBoundedDispatchLayout(
    props.operation,
    Math.max(props.invocationCount, 1),
    FRACTIONAL_TILE_WORKGROUP_SIZE,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const source = `${props.bindings
    .map(
      (
        binding,
        location
      ) => `const ${binding.name}Offset:u32=${getViewElementOffset(binding.view)}u;
@group(0) @binding(${location}) var<storage,${binding.access}> ${binding.name}:array<${binding.type}>;`
    )
    .join('\n')}
${props.declarations ?? ''}
${props.workgroupCount ? 'var<workgroup> analysisDispatchWorkgroups: u32;' : ''}
@compute @workgroup_size(${FRACTIONAL_TILE_WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) workgroupId:vec3<u32>,@builtin(num_workgroups) workgroupCount:vec3<u32>,@builtin(local_invocation_index) localInvocationIndex:u32){
${props.bindings.map(binding => `_ = &${binding.name};`).join('\n')}
let workgroupIndex=(workgroupId.z*workgroupCount.y+workgroupId.y)*workgroupCount.x+workgroupId.x;
let index=workgroupIndex*${FRACTIONAL_TILE_WORKGROUP_SIZE}u+localInvocationIndex;
${
  props.workgroupCount
    ? `if(localInvocationIndex==0u){analysisDispatchWorkgroups=${props.workgroupCount};}
let dispatchWorkgroups=workgroupUniformLoad(&analysisDispatchWorkgroups);
if(workgroupIndex>=dispatchWorkgroups){return;}`
    : `if(index>=${props.rowCount}){return;}`
}
${props.body}
}`;
  return createGPUComputeCommandNode<Parameters>({
    id: props.id,
    resources: props.bindings.map(binding => ({
      buffer: binding.view,
      usage: binding.access === 'read' ? 'storage-read' : 'storage-read-write'
    })),
    workload: {
      operation: props.operation,
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
              props.bindings.map(binding => [binding.name, getViewBinding(binding.view, getBuffer)])
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

/** Checks writable ranges and per-dispatch physical storage usage scopes. @internal */
export function validateRasterFractionalWritableViews(
  id: string,
  outputs: object,
  inputs: readonly (GraphDataView | undefined)[],
  dispatches?: readonly (
    | readonly (GraphDataView | undefined)[]
    | {
        writes: readonly (GraphDataView | undefined)[];
        reads?: readonly (GraphDataView | undefined)[];
      }
  )[]
): void {
  const getBuffer = (view: GraphDataView) =>
    view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer;
  const validateStorageUsage = (write: GraphDataView, read: GraphDataView) =>
    validateRequirement(
      id,
      getBuffer(write) !== getBuffer(read),
      'read-only and writable storage bindings must not share a physical buffer in one dispatch'
    );
  const writable: Record<string, GraphDataView> = {};
  for (const [name, value] of Object.entries(outputs)) {
    if (value instanceof GraphDataView) {
      writable[`output.${name}`] = value;
    } else if (name === 'status') {
      for (const [statusName, view] of Object.entries(value ?? {})) {
        if (view instanceof GraphDataView) writable[`output.status.${statusName}`] = view;
      }
    }
  }
  if (dispatches) {
    const entries = Object.entries(writable);
    // Different dispatches may share an alignment prefix, but never logical writable bytes.
    for (const [index, [name, view]] of entries.entries()) {
      for (const [otherName, other] of [
        ...entries.slice(index + 1),
        ...inputs.flatMap((input, inputIndex) =>
          input ? [[`input.${inputIndex}`, input] as const] : []
        )
      ]) {
        if (getBuffer(view) !== getBuffer(other) || view.length === 0 || other.length === 0)
          continue;
        const end = view.byteOffset + (view.length - 1) * view.byteStride + view.rowByteLength;
        const otherEnd =
          other.byteOffset + (other.length - 1) * other.byteStride + other.rowByteLength;
        validateRequirement(
          id,
          end <= other.byteOffset || otherEnd <= view.byteOffset,
          `${name} and ${otherName} logical byte ranges must not overlap`
        );
      }
    }
    // Check alignment only for bindings that actually coexist. Read-only bindings may alias.
    const writableViews = new Set(Object.values(writable));
    for (const dispatch of dispatches) {
      const dispatchViews =
        'writes' in dispatch ? [...dispatch.writes, ...(dispatch.reads ?? [])] : dispatch;
      const dispatchWritable = 'writes' in dispatch ? new Set(dispatch.writes) : writableViews;
      const bindings = dispatchViews.filter((view): view is GraphDataView => Boolean(view));
      for (const [index, view] of bindings.entries()) {
        if (!dispatchWritable.has(view)) continue;
        for (const [otherIndex, other] of bindings.entries()) {
          if (index === otherIndex) continue;
          if (!dispatchWritable.has(other)) validateStorageUsage(view, other);
          validateDisjointWritableViews(id, {
            [`binding.${index}`]: view,
            [`binding.${otherIndex}`]: other
          });
        }
      }
    }
    return;
  }
  validateDisjointWritableViews(id, writable);
  // Read-only inputs may alias each other. Check each against all writable ranges separately.
  for (const [index, input] of inputs.entries()) {
    if (input) {
      for (const view of Object.values(writable)) validateStorageUsage(view, input);
      validateDisjointWritableViews(id, {...writable, [`input.${index}`]: input});
    }
  }
}

/** Preflights both the aligned binding and its complete backing allocation. @internal */
export function validateRasterFractionalBindingByteLength(
  device: Device,
  id: string,
  name: string,
  bindingByteLength: number,
  hint: string,
  bufferByteLength: number = bindingByteLength
): void {
  validateGPUStorageBindingByteLength(device, id, name, bindingByteLength, hint);
  validateMaximumCount(
    id,
    `${name} buffer byteLength (device.limits.maxBufferSize)`,
    Math.max(bufferByteLength, 4),
    device.limits.maxBufferSize
  );
}
