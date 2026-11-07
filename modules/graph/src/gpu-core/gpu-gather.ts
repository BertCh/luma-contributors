// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type GPUCommandNode} from './gpu-command-node';
import {
  getGPUVectorFormatInfo,
  getGPUVectorChunks,
  isValueListGPUVectorFormat,
  isVertexListGPUVectorFormat,
  type GPUVectorFormat
} from '@luma-contributors/graph/gpu-data';
import {GPUCommandGraph, type GraphDataView, type GraphVectorView} from './gpu-command-graph';
import {
  getBoundedDispatchLayout,
  getBoundedDispatchOverrides,
  getBoundedInvocationIndexOverrideSource
} from './gpu-dispatch-utils';
import {
  createGPUKernelNode,
  getGPUKernelBindingDeclarations,
  getGPUKernelOverrideDeclarations,
  type GPUKernelBinding
} from './gpu-kernel-node';
import {
  getGraphDataPrefix,
  getViewBindingRange,
  getViewElementOffset,
  validatePackedUint32View
} from './graph-data-view-utils';
import {alignGraphVectorViews, getGraphVectorData} from './graph-vector-view-utils';

const GATHER_WORKGROUP_SIZE = 256;
const UINT32_BYTE_LENGTH = Uint32Array.BYTES_PER_ELEMENT;

/** Fixed-width graph data formats accepted by {@link GPUGather}. */
export type GPUGatherFormat = Exclude<
  GPUVectorFormat,
  `vertex-list<${string}>` | `value-list<${string}>`
>;

/** Properties for one graph-native indexed gather. */
export type GPUGatherProps<T extends GPUGatherFormat = GPUGatherFormat> = {
  /** Prefix for the generated graph node. */
  id?: string;
  /** Word-aligned fixed-width source rows, optionally strided; indices address global rows. */
  source: GraphDataView<T> | GraphVectorView<T>;
  /** Packed uint32 source-row indices, independently partitioned from source and output. */
  indices: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Caller-owned packed destination with the source format and capacity for every index. */
  output: GraphDataView<T> | GraphVectorView<T>;
};

/**
 * Gathers packed fixed-width rows through uint32 indices.
 *
 * Each output row receives `source[indices[i]]`. Out-of-range indices write an all-zero row.
 * The current kernel copies rows as 32-bit words, so the row byte length must be a multiple of 4.
 */
export class GPUGather<T extends GPUGatherFormat = GPUGatherFormat> {
  readonly id: string;
  readonly source: GraphDataView<T> | GraphVectorView<T>;
  readonly indices: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  readonly output: GraphDataView<T> | GraphVectorView<T>;
  readonly wordsPerRow: number;

  constructor(props: GPUGatherProps<T>) {
    this.id = props.id ?? 'gpu-gather';
    this.source = props.source;
    this.indices = props.indices;
    this.output = props.output;

    for (const chunk of getGraphVectorData(this.indices)) {
      validatePackedUint32View(chunk, `${this.id} indices`);
    }
    for (const view of [this.source, this.output]) {
      for (const chunk of getGraphVectorData(view))
        validateGatherView(chunk, this.id, view === this.output);
    }

    if (this.source.format !== this.output.format) {
      throw new Error(`${this.id} source and output must use the same format`);
    }
    if (this.output.length < this.indices.length) {
      throw new Error(`${this.id} output must contain at least indices.length rows`);
    }
    const inputBuffers = new Set([
      ...getGraphVectorData(this.source).map(chunk => chunk.buffer),
      ...getGraphVectorData(this.indices).map(chunk => chunk.buffer)
    ]);
    if (getGraphVectorData(this.output).some(chunk => inputBuffers.has(chunk.buffer))) {
      throw new Error(`${this.id} output must use a separate buffer`);
    }

    this.wordsPerRow = getGPUVectorFormatInfo(this.source.format).byteLength / UINT32_BYTE_LENGTH;
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    return getGatherCommandNodes(graph, this);
  }
}

/** Shares fixed-width gather lowering with the uint32 invalid-value specialization. @internal */
export function getGatherCommandNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  gather: GPUGather,
  invalidValue = 0,
  operation = 'GPUGather'
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  for (const view of [gather.source, gather.indices, gather.output]) {
    for (const chunk of getGraphVectorData(view)) {
      if (chunk.buffer.graph !== graph) {
        throw new Error(`${gather.id} views must belong to the target graph`);
      }
    }
  }
  if (gather.indices.length === 0) return nodes;

  const output = getGraphDataPrefix(graph, gather.output, gather.indices.length);
  const spans = alignGraphVectorViews(graph, [gather.indices, output]);
  const sources = getGPUVectorChunks(getGraphVectorData(gather.source)).filter(
    chunk => chunk.length
  );
  // With no source rows, use an output-only fill pass; empty buffers need not be bound.
  const sourcePasses = sources.length ? sources : [undefined];
  for (const [spanIndex, [indices, destination]] of spans.entries()) {
    const dispatchLayout = getBoundedDispatchLayout(
      operation,
      indices.length,
      GATHER_WORKGROUP_SIZE,
      graph.device.limits.maxComputeWorkgroupsPerDimension
    );
    for (const [sourceIndex, chunk] of sourcePasses.entries()) {
      const sourceView = chunk?.data;
      for (const view of [destination, ...(sourceView ? [sourceView, indices] : [])]) {
        if (getViewBindingRange(view).size > graph.device.limits.maxStorageBufferBindingSize) {
          throw new Error('Active chunks must fit a storage binding');
        }
      }
      const id =
        spans.length === 1 && sourcePasses.length === 1
          ? gather.id
          : `${gather.id}-span-${spanIndex}-source-${sourceIndex}`;
      const bindings: GPUKernelBinding[] = [
        {name: 'outputWords', view: destination, access: 'write', elementType: 'u32'}
      ];
      if (sourceView) {
        bindings.push(
          {name: 'sourceWords', view: sourceView, access: 'read', elementType: 'u32'},
          {name: 'indices', view: indices, access: 'read'}
        );
      }
      // Counts, offsets, and the dispatch shape are pipeline constants shared by every size.
      const constants: Record<string, number> = {
        INDEX_COUNT: indices.length,
        WORDS_PER_ROW: gather.wordsPerRow,
        OUTPUT_OFFSET: getViewElementOffset(destination),
        INVALID_VALUE: invalidValue,
        ...(sourceView && {
          SOURCE_LENGTH: sourceView.length,
          SOURCE_ROW_OFFSET: chunk?.offset ?? 0,
          SOURCE_OFFSET: getViewElementOffset(sourceView),
          SOURCE_STRIDE_WORDS: sourceView.byteStride / UINT32_BYTE_LENGTH,
          INDEX_OFFSET: getViewElementOffset(indices)
        }),
        ...getBoundedDispatchOverrides(dispatchLayout)
      };
      nodes.push(
        createGPUKernelNode(graph, {
          id,
          operation,
          moduleId: `${operation}/${sourceView ? 'copy-rows' : 'fill-invalid'}`,
          ...(operation === 'GPUGather' && {formula: 'gpu-core.gather'}),
          source: makeShaderSource({
            bindings,
            constants,
            hasSource: Boolean(sourceView),
            initialize: sourceIndex === 0
          }),
          constants,
          bindings,
          workgroupSize: GATHER_WORKGROUP_SIZE,
          dispatchLayout,
          invocationCount: indices.length,
          workload: {
            readByteLength: sourceView
              ? indices.length * (UINT32_BYTE_LENGTH + sourceView.rowByteLength)
              : 0,
            writeByteLength: indices.length * destination.rowByteLength
          }
        })
      );
    }
  }
  return nodes;
}

function validateGatherView(view: GraphDataView, name: string, packed: boolean): void {
  if (isVertexListGPUVectorFormat(view.format) || isValueListGPUVectorFormat(view.format)) {
    throw new Error(`${name} must use a fixed-width GPU data format`);
  }
  const formatInfo = getGPUVectorFormatInfo(view.format);
  if (
    (packed
      ? view.byteStride !== formatInfo.byteLength
      : view.byteStride < formatInfo.byteLength || view.byteStride % UINT32_BYTE_LENGTH !== 0) ||
    view.rowByteLength !== formatInfo.byteLength ||
    view.byteOffset % UINT32_BYTE_LENGTH !== 0 ||
    view.rowByteLength % UINT32_BYTE_LENGTH !== 0
  ) {
    throw new Error(`${name} must be packed, uint32-aligned GPU data with 32-bit-word rows`);
  }
}

function makeShaderSource(props: {
  bindings: readonly GPUKernelBinding[];
  constants: Readonly<Record<string, number>>;
  hasSource: boolean;
  initialize: boolean;
}): string {
  const {bindings, constants, hasSource, initialize} = props;
  return `${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}
@compute @workgroup_size(${GATHER_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexOverrideSource(GATHER_WORKGROUP_SIZE).body}
  if (index >= INDEX_COUNT) { return; }
  let outputBase = OUTPUT_OFFSET + index * WORDS_PER_ROW;
${
  hasSource
    ? `
  let sourceIndex = indices[INDEX_OFFSET + index];
  if (sourceIndex >= SOURCE_ROW_OFFSET && sourceIndex - SOURCE_ROW_OFFSET < SOURCE_LENGTH) {
    let sourceBase = SOURCE_OFFSET + (sourceIndex - SOURCE_ROW_OFFSET) * SOURCE_STRIDE_WORDS;
    for (var word = 0u; word < WORDS_PER_ROW; word++) {
      outputWords[outputBase + word] = sourceWords[sourceBase + word];
    }
    return;
  }
`
    : ''
}
${
  initialize
    ? `
  for (var word = 0u; word < WORDS_PER_ROW; word++) {
    outputWords[outputBase + word] = INVALID_VALUE;
  }
`
    : ''
}
}`;
}
