// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateGPUStorageBindingByteLength,
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '../analysis-core/kernels/gpu-contributor-utils';
import {validateWGSLIdentifier} from '../analysis-core/kernels/wgsl-identifiers';
import {getCoreDefaultBuffer} from '../analysis-core/network-topology';
import {SCALED_QUOTIENT_WGSL} from '../analysis-core/kernels/scaled-quotient-wgsl';
import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../analysis-core/kernels/graph-chunk-dispatch';
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
  validateGraphChunkBindingRanges(graph, props.id, {...props.inputs, ...props.outputs});
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
  const declarations = `${getPointCloudKernelDeclarations(props)}
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

/** Shared typed accessors and finite normal helper for packed and chunked point-cloud kernels. */
export function getPointCloudKernelDeclarations(
  props: Pick<PointCloudKernelProps, 'inputs' | 'outputs'>,
  offsets?: Readonly<Record<string, string>>
): string {
  const inputEntries = Object.entries(props.inputs);
  const entries = [...inputEntries, ...Object.entries(props.outputs)];
  const accessors = entries
    .map(([name, view], bindingIndex) => {
      validateWGSLIdentifier('pointcloud', 'binding', name);
      const isOutput = bindingIndex >= inputEntries.length;
      const scalar = view.format === 'uint32' ? 'u32' : 'f32';
      const isVector = view.format === 'float32x3';
      const valueType = isVector ? 'vec3f' : scalar;
      const address = `${offsets?.[name] ?? `${name}Offset`} + row * ${isVector ? 3 : 1}u`;
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
  return `${accessors}
${SCALED_QUOTIENT_WGSL}
fn getPointCloudQuotient2(value: vec2f, divisor: f32) -> vec2f {
  return vec2f(getScaledQuotient(value.x, divisor, 1.0, 0), getScaledQuotient(value.y, divisor, 1.0, 0));
}
fn getPointCloudQuotient3(value: vec3f, divisor: f32) -> vec3f {
  return vec3f(getScaledQuotient(value.x, divisor, 1.0, 0), getScaledQuotient(value.y, divisor, 1.0, 0), getScaledQuotient(value.z, divisor, 1.0, 0));
}
fn safeNormal(value: vec3f) -> vec3f {
  let scale = max(max(abs(value.x), abs(value.y)), abs(value.z));
  if (!(scale > 0.0) || scale > 3.402823e38) { return vec3f(0.0); }
  let scaled = getPointCloudQuotient3(value, scale);
  let magnitude = length(scaled);
  if (!(magnitude > 0.0)) { return vec3f(0.0); }
  return scaled / magnitude;
}
`;
}

type PointCloudNamedViews = Readonly<Record<string, GraphDataView | undefined>>;
type PointCloudDispatchViews = {inputs?: PointCloudNamedViews; outputs: PointCloudNamedViews};

/** Logical separation independent of dispatch. Project packed fields to uint32 metadata chunks
 * of one vector; the shared validator then compares logical ranges across chunk indices, including
 * imported handles sharing a physical buffer. No views, buffers or graph nodes are declared.
 */
export function validatePointCloudLogicalViews(id: string, views: PointCloudNamedViews): void {
  const fields = Object.entries(views).filter((entry): entry is [string, GraphDataView] =>
    Boolean(entry[1])
  );
  const data: GraphDataView<'uint32'>[] = fields.map(([, view]) => ({
    ...view,
    format: 'uint32',
    byteStride: 4,
    rowByteLength: 4,
    length: view.length ? ((view.length - 1) * view.byteStride + view.rowByteLength) / 4 : 0
  }));
  const length = data.reduce((total, view) => total + view.length, 0);
  try {
    validateDisjointWritableViews(id, {
      pointCloudFields: new GraphVectorView({
        id: `${id}-logical-fields`,
        name: 'logical fields',
        format: 'uint32',
        length,
        valueLength: length,
        stride: 1,
        byteStride: 4,
        rowByteLength: 4,
        data
      })
    });
  } catch (error) {
    if (!(error instanceof Error)) throw error;
    throw new Error(
      error.message.replace(
        /pointCloudFields chunk (\d+)/g,
        (_, index: string) => fields[Number(index)]?.[0] ?? index
      )
    );
  }
}

/** Construction-time packed-view contract, with actual co-bound read/write groups. */
export function validatePointCloudPackedViews(
  id: string,
  inputs: PointCloudNamedViews,
  outputs: PointCloudNamedViews,
  dispatches: readonly PointCloudDispatchViews[] = [{inputs, outputs}]
): void {
  for (const [name, view] of Object.entries({...inputs, ...outputs})) {
    if (view) validateSinglePackedView(view, [view.format], `${id} ${name}`);
  }
  validatePointCloudLogicalViews(id, outputs);
  for (const [inputName, input] of Object.entries(inputs)) {
    if (input) validatePointCloudLogicalViews(id, {...outputs, [`input ${inputName}`]: input});
  }
  for (const dispatch of dispatches) {
    const writableViews = Object.fromEntries(
      Object.entries(dispatch.outputs).filter(([, view]) => view && view.length > 0)
    );
    if (!Object.keys(writableViews).length) continue;
    validateDisjointWritableViews(id, writableViews);
    // WebGPU usage scopes forbid a physical buffer being both read-only and writable storage,
    // even when its binding ranges do not intersect. Writable bindings may share disjoint ranges.
    const writableBuffers = new Set(
      Object.values(writableViews).map(view => getPointCloudBufferIdentity(view!))
    );
    for (const [name, input] of Object.entries(dispatch.inputs ?? {})) {
      if (input)
        validateRequirement(
          id,
          !writableBuffers.has(getPointCloudBufferIdentity(input)),
          `input ${name} and writable outputs must not share a physical storage buffer in one dispatch`
        );
    }
  }
}

function getPointCloudBufferIdentity(view: GraphDataView) {
  return view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer;
}

/** Preflight every derived scratch binding before a contributor declares resources or nodes. */
export function validatePointCloudScratchBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  byteLengths: Readonly<Record<string, number>>
): void {
  for (const [name, byteLength] of Object.entries(byteLengths)) {
    validateCapacity(
      id,
      `${name} allocation byteLength`,
      Math.max(byteLength, 4),
      0,
      graph.device.limits.maxBufferSize
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      Math.max(byteLength, 4),
      'reduce the row count or grid dimensions'
    );
  }
}
