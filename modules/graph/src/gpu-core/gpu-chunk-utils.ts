// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUVectorFormatInfo, type GPUVectorFormat} from '@luma-contributors/graph/gpu-data';
import {type GPUCommandGraph, type GraphDataView, GraphVectorView} from './gpu-command-graph';
import type {GPUCommandNode} from './gpu-command-node';
import {
  createGPUKernelNode,
  getGPUKernelBindingDeclarations,
  getGPUKernelOverrideDeclarations,
  type GPUKernelBinding
} from './gpu-kernel-node';
import {getGraphVectorData} from './graph-vector-view-utils';
import {getViewBindingRange} from './graph-data-view-utils';
import type {GPUBoundedDispatchLayout} from './gpu-dispatch-utils';

/** Borrows a logical interval, retaining whole chunks and their storage identities. @internal */
export function getGraphDataRange<
  Parameters,
  Format extends Exclude<GPUVectorFormat, `value-list<${string}>` | `vertex-list<${string}>`>
>(
  graph: GPUCommandGraph<Parameters>,
  input: GraphDataView<Format> | GraphVectorView<Format>,
  start: number,
  length: number
): GraphVectorView<Format> {
  if (
    !Number.isSafeInteger(start) ||
    !Number.isSafeInteger(length) ||
    start < 0 ||
    length < 0 ||
    start + length > input.length
  )
    throw new Error('Graph range must fit the source');
  const data: GraphDataView<Format>[] = [];
  const chunks =
    input instanceof GraphVectorView
      ? input.chunks
      : [{offset: 0, length: input.length, data: input}];
  // Seek to the first intersecting chunk in O(log C), then visit only the requested range.
  let firstChunk = 0;
  let lastChunk = chunks.length;
  while (firstChunk < lastChunk) {
    const middle = Math.floor((firstChunk + lastChunk) / 2);
    if (chunks[middle].offset + chunks[middle].length <= start) firstChunk = middle + 1;
    else lastChunk = middle;
  }
  for (
    let index = firstChunk;
    index < chunks.length && chunks[index].offset < start + length;
    index++
  ) {
    const {data: chunk, offset} = chunks[index];
    const first = Math.max(start, offset);
    const last = Math.min(start + length, offset + chunk.length);
    if (first < last)
      data.push(
        first === offset && last === offset + chunk.length
          ? chunk
          : graph.createDataView(chunk.buffer, {
              format: chunk.format,
              length: last - first,
              byteOffset: chunk.byteOffset + (first - offset) * chunk.byteStride,
              byteStride: chunk.byteStride,
              rowByteLength: chunk.rowByteLength
            })
      );
  }
  return new GraphVectorView({
    id: `graph-range-${start}-${length}`,
    name: `graph-range-${start}-${length}`,
    stride:
      input instanceof GraphVectorView
        ? input.stride
        : getGPUVectorFormatInfo(input.format).components *
          (getGPUVectorFormatInfo(input.format).listSize ?? 1),
    format: input.format,
    length,
    valueLength: length * (getGPUVectorFormatInfo(input.format).listSize ?? 1),
    byteStride: input.byteStride,
    rowByteLength: input.rowByteLength,
    data
  });
}

/** Validates graph ownership, including empty chunks, and disjoint writable storage. @internal */
export function validateChunkViews<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  inputs: readonly (GraphDataView | GraphVectorView)[],
  outputs: readonly (GraphDataView | GraphVectorView)[]
): void {
  if ([...inputs, ...outputs].some(view => view.length > 0xffffffff))
    throw new Error('Logical lengths must fit uint32 addressing');
  const reads = inputs.flatMap(getGraphVectorData);
  const writes = outputs.flatMap(getGraphVectorData);
  for (const view of [...reads, ...writes]) {
    if (view.buffer.graph !== graph) throw new Error('Views must belong to the target graph');
  }
  const inputBuffers = new Set(reads.map(input => input.buffer));
  const outputRanges = new Map<GraphDataView['buffer'], GraphDataView[]>();
  for (const output of writes) {
    if (inputBuffers.has(output.buffer))
      throw new Error('Outputs must use separate buffers from inputs');
    if (!output.length) continue;
    const ranges = outputRanges.get(output.buffer) ?? [];
    ranges.push(output);
    outputRanges.set(output.buffer, ranges);
  }
  for (const ranges of outputRanges.values()) {
    ranges.sort((left, right) => left.byteOffset - right.byteOffset);
    let previousEnd = 0;
    for (const range of ranges) {
      if (range.byteOffset < previousEnd) throw new Error('Writable chunks must not overlap');
      previousEnd = range.byteOffset + (range.length - 1) * range.byteStride + range.rowByteLength;
    }
  }
}

/** Bounds a count of workgroups without imposing a one-dimensional dispatch limit. @internal */
export function getChunkDispatch(count: number, maximum: number): GPUBoundedDispatchLayout {
  if (
    !Number.isSafeInteger(count) ||
    count < 1 ||
    count > 0xffffffff ||
    !Number.isSafeInteger(maximum) ||
    maximum < 1
  )
    throw new Error('Invalid chunk dispatch');
  const x = Math.min(count, maximum);
  const y = Math.min(Math.ceil(count / x), maximum);
  const z = Math.ceil(count / x / y);
  if (z > maximum) throw new Error('Chunk operation exceeds the 3D dispatch limit');
  return {x, y, z};
}

/**
 * Compiles one bounded kernel with caller-owned or explicit algorithm-scratch bindings. Inputs bind
 * read-only and outputs read-write, in declaration order.
 *
 * Pass either a complete `source`, or a `shaderBody` with the shader text after the declarations: the
 * override and storage-binding declarations are then generated from `constants` and the bindings,
 * so binding order cannot drift. Problem-size values belong in `constants` so one shader serves
 * every size. @internal
 */
export function createChunkNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    inputs?: Record<string, GraphDataView>;
    outputs: Record<string, GraphDataView>;
    dispatch: GPUBoundedDispatchLayout;
    workgroupSize?: number;
    /** Pipeline override constants referenced by the shader. */
    constants?: Readonly<Record<string, number>>;
    /** Types of constants that are not `u32`. Only used with `shaderBody`. */
    constantTypes?: Readonly<Record<string, 'u32' | 'i32' | 'f32'>>;
    /** Output names declared as `array<atomic<...>>`. Only used with `shaderBody`. */
    atomic?: readonly string[];
    /** WGSL element types by binding name; defaults come from the view format. Only used with `shaderBody`. */
    elementTypes?: Readonly<Record<string, 'u32' | 'i32' | 'f32'>>;
    /** Operator name recorded in annotations. Defaults to `createChunkNode`. */
    operation?: string;
    /** Module identifier recorded in annotations. */
    moduleId?: string;
    /** Logical rows handled by the pass, recorded on recording graphs only. */
    invocationCount?: number;
    /** Complete WGSL. Mutually exclusive with `shaderBody`; wins when both are set. */
    source?: string;
    /** WGSL after the generated declarations. */
    shaderBody?: string;
  }
): GPUCommandNode<Parameters> {
  const bindings: GPUKernelBinding[] = [
    ...Object.entries(props.inputs ?? {}).map(
      ([name, view]): GPUKernelBinding => ({
        name,
        view,
        access: 'read',
        elementType: props.elementTypes?.[name]
      })
    ),
    ...Object.entries(props.outputs).map(
      ([name, view]): GPUKernelBinding => ({
        name,
        view,
        access: 'read_write',
        atomic: props.atomic?.includes(name),
        elementType: props.elementTypes?.[name]
      })
    )
  ];
  for (const {view} of bindings) {
    if (getViewBindingRange(view).size > graph.device.limits.maxStorageBufferBindingSize)
      throw new Error('Active chunks must fit a storage binding');
  }
  const source =
    props.source ??
    `${getGPUKernelOverrideDeclarations(props.constants ?? {}, props.constantTypes)}
${getGPUKernelBindingDeclarations(bindings)}
${props.shaderBody ?? ''}`;
  return createGPUKernelNode(graph, {
    id: props.id,
    operation: props.operation ?? 'createChunkNode',
    moduleId: props.moduleId ?? 'gpu-core/gpu-chunk-utils',
    invocationCount: props.invocationCount,
    source,
    constants: props.constants,
    bindings,
    workgroupSize: props.workgroupSize,
    dispatchLayout: props.dispatch
  });
}
