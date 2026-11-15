// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateRequirement
} from '../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../analysis-core/kernels/wgsl-kernel-nodes';

type PointCloudKernelView =
  | GraphDataView<'float32x3'>
  | GraphDataView<'uint32'>
  | GraphDataView<'float32'>;

/** Properties of one packed point-cloud row kernel. */
export type PointCloudKernelProps = {
  id: string;
  /** Number of logical rows. Each row sees `index` in the WGSL body. */
  length: number;
  /** Read-only packed views; `readName(row)` is declared for each, with `Name` capitalized. */
  inputs: Record<string, PointCloudKernelView>;
  /** Writable packed views; `readName(row)` and `writeName(row, value)` are declared. */
  outputs: Record<string, PointCloudKernelView>;
  /** Shared WGSL helper declarations used by the row body. */
  declarations?: string;
  /** One whole workgroup owns each logical row; the body must synchronize every active lane. */
  cooperative?: {workgroupSize: number; declarations: string};
  /** WGSL statements run per row with `index` and `localInvocationIndex` in scope. */
  body: string;
};

/**
 * Builds the compute nodes of one packed point-cloud row kernel on top of the analysis WGSL kernel
 * builder. Output views must not alias each other; the device rejects overlapping writable bindings. Generates typed `read*` and `write*` accessors over `float32`, `uint32` and
 * `float32x3` views, plus a `safeNormal` helper for finite unit normals.
 */
export function getPointCloudKernelNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: PointCloudKernelProps
): GPUCommandNode<Parameters>[] {
  const inputEntries = Object.entries(props.inputs);
  const outputEntries = Object.entries(props.outputs);
  const entries = [...inputEntries, ...outputEntries];
  validateGraphViewsBelongToGraph(
    props.id,
    graph,
    entries.map(([, view]) => view)
  );
  for (const [, view] of entries) {
    validatePackedView(view, ['float32x3', 'uint32', 'float32'], props.id);
  }
  if (!props.length) {
    return [];
  }
  const bindings: WGSLKernelBinding[] = entries.map(([name, view], bindingIndex) => ({
    name,
    view,
    type: view.format === 'uint32' ? 'u32' : 'f32',
    access: bindingIndex >= inputEntries.length ? 'read_write' : 'read'
  }));
  const accessors = entries
    .map(([name, view], bindingIndex) => {
      const isOutput = bindingIndex >= inputEntries.length;
      const scalar = view.format === 'uint32' ? 'u32' : 'f32';
      const isVector = view.format === 'float32x3';
      const valueType = isVector ? 'vec3f' : scalar;
      const address = `${name}Offset + row * ${isVector ? 3 : 1}u`;
      const title = name[0].toUpperCase() + name.slice(1);
      const read = isVector
        ? `vec3f(${name}[offset], ${name}[offset + 1u], ${name}[offset + 2u])`
        : `${name}[offset]`;
      const write = isVector
        ? `${name}[offset] = value.x; ${name}[offset + 1u] = value.y; ${name}[offset + 2u] = value.z;`
        : `${name}[offset] = value;`;
      return `fn read${title}(row: u32) -> ${valueType} { let offset = ${address}; return ${read}; }
${isOutput ? `fn write${title}(row: u32, value: ${valueType}) { let offset = ${address}; ${write} }` : ''}`;
    })
    .join('\n');
  const workgroupSize = props.cooperative?.workgroupSize;
  if (workgroupSize !== undefined) {
    validateRequirement(
      props.id,
      Number.isInteger(workgroupSize) &&
        workgroupSize >= 2 &&
        (workgroupSize & (workgroupSize - 1)) === 0 &&
        workgroupSize <= graph.device.limits.maxComputeInvocationsPerWorkgroup,
      `cooperative workgroupSize must be a power of two in [2, ${graph.device.limits.maxComputeInvocationsPerWorkgroup}]`
    );
    // Cooperative rows dispatch one workgroup each; the invocation count must stay a uint32.
    validateMaximumCount(
      props.id,
      'cooperative rows x workgroupSize',
      props.length * workgroupSize,
      0xffffffff
    );
  }
  const declarations = `${accessors}
fn safeNormal(value: vec3f) -> vec3f {
  let scale = max(max(abs(value.x), abs(value.y)), abs(value.z));
  if (!(scale > 0.0) || scale > 3.402823e38) { return vec3f(0.0); }
  let scaled = value / scale;
  let magnitude = length(scaled);
  if (!(magnitude > 0.0)) { return vec3f(0.0); }
  return scaled / magnitude;
}
${props.declarations ?? ''}
${props.cooperative?.declarations ?? ''}
fn runPointCloudRow(index: u32, localInvocationIndex: u32) {
  ${props.body}
}`;
  return [
    createWGSLKernelNode(graph, {
      id: props.id,
      operation: 'GPUPointCloud',
      bindings,
      // A cooperative row owns a whole workgroup, so the workgroup index is the row index.
      invocationCount: workgroupSize ? props.length * workgroupSize : props.length,
      ...(workgroupSize ? {workgroupSize, guardIndex: false} : {}),
      declarations,
      body: workgroupSize
        ? `if (workgroupIndex >= ${props.length}u) { return; }
  runPointCloudRow(workgroupIndex, localInvocationIndex);`
        : 'runPointCloudRow(index, localInvocationIndex);'
    })
  ];
}
