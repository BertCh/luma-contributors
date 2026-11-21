// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type VertexFormat} from '@luma.gl/core';
import {
  createGPUCopyCommandNode,
  createTransientView,
  getViewBindingRange,
  type GraphVectorView,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCopyCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  getGraphViewChunks,
  isGraphVectorView,
  validateCapacity,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateGraphViewsBelongToGraph,
  validateRequirement
} from './gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from './graph-chunk-dispatch';

// WebGPU storage-binding sizes and buffer-copy offsets/sizes must be multiples of four bytes.
const GRAPH_CHUNK_GATHER_COPY_BYTE_ALIGNMENT = 4;

/**
 * Packs analysis input chunks with the published command-graph copy API, without a shader.
 *
 * Call at the start of `getCommandNodes`, before creating other nodes or scratch. Append the
 * returned nodes before consumers of `view`; this function does not schedule them on `graph`.
 * A packed data view is borrowed without copying. A vector (including a single-chunk vector)
 * gets graph-owned transient storage and one copy node per nonempty chunk, in source order.
 * An empty vector has a zero-length view backed by one row of scratch and emits no copies.
 *
 * Only packed fixed-width vertex formats with a positive multiple of four bytes per row are
 * supported, including on the borrowed and empty routes. Each nonempty vector
 * chunk's buffer must include `Buffer.COPY_SRC`; a borrowed packed source must include
 * `Buffer.STORAGE`. Chunk totals and copy ranges are validated before scratch is declared.
 * The destination includes storage and both copy
 * usages, so it can feed a packed-only reduction or a later copy. Copies run on every encoding;
 * the compiled graph owns the scratch, while source buffers remain caller-owned.
 *
 * The total must fit one device storage binding, including one backing row for empty input.
 * Callers still validate their accepted formats at construction and include the returned node
 * count in their contributor's command-node budget (in addition to their other nodes).
 *
 * @param graph Target command graph.
 * @param props Contributor-prefixed unique ID and the analysis input to pack.
 */
export function createGraphChunkGather<Format extends VertexFormat, Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; source: GraphDataView<Format> | GraphVectorView<Format>}
): {view: GraphDataView<Format>; nodes: GPUCopyCommandNode<Parameters>[]} {
  const {id, source} = props;
  const chunks = getGraphViewChunks(source);
  validateCapacity(id, 'source.length', source.length, 0);
  validateRequirement(
    id,
    chunks.reduce((total, chunk) => total + chunk.length, 0) === source.length,
    'source.length must equal its chunk total'
  );
  validateRequirement(
    id,
    Number.isSafeInteger(source.rowByteLength) &&
      source.rowByteLength > 0 &&
      source.rowByteLength % GRAPH_CHUNK_GATHER_COPY_BYTE_ALIGNMENT === 0,
    `source.rowByteLength must be a positive multiple of GRAPH_CHUNK_GATHER_COPY_BYTE_ALIGNMENT (${GRAPH_CHUNK_GATHER_COPY_BYTE_ALIGNMENT})`
  );
  validateGraphViewsBelongToGraph(id, graph, [source]);
  for (const [chunkIndex, chunk] of chunks.entries()) {
    validatePackedView(chunk, [source.format], `${id} source chunk ${chunkIndex}`);
    validateRequirement(
      id,
      chunk.rowByteLength === source.rowByteLength,
      `source chunk ${chunkIndex} rowByteLength must equal source.rowByteLength`
    );
  }
  const isVector = isGraphVectorView(source);
  const byteLength = isVector
    ? Math.max(source.length, 1) * source.rowByteLength
    : getViewBindingRange(source).size;
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'source packed scratch',
    byteLength,
    'reduce source rows or use a chunk-native contributor'
  );
  if (!isVector) {
    validateGraphChunkBindingRanges(graph, id, {source});
    return {view: source, nodes: []};
  }
  validateCapacity(
    id,
    'packed scratch byteLength',
    byteLength,
    0,
    graph.device.limits.maxBufferSize
  );
  const nonemptyChunks = chunks.filter(chunk => chunk.length > 0);
  validateCommandNodeBudget(id, nonemptyChunks.length);
  for (const [chunkIndex, chunk] of nonemptyChunks.entries()) {
    validateRequirement(
      id,
      chunk.byteOffset % GRAPH_CHUNK_GATHER_COPY_BYTE_ALIGNMENT === 0 &&
        chunk.byteOffset + chunk.length * chunk.rowByteLength <= chunk.buffer.byteLength,
      `source nonempty chunk ${chunkIndex} copy range must be four-byte aligned and fit its buffer byteLength`
    );
    validateRequirement(
      id,
      (chunk.buffer.usage & Buffer.COPY_SRC) !== 0,
      `source nonempty chunk ${chunkIndex} buffer usage must include Buffer.COPY_SRC`
    );
  }
  const view = createTransientView(
    graph,
    `${id}-packed`,
    source.format,
    source.length,
    Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  );
  let destinationOffset = 0;
  const nodes = nonemptyChunks.map((chunk, chunkIndex) => {
    const size = chunk.length * chunk.rowByteLength;
    const chunkDestinationOffset = destinationOffset;
    destinationOffset += size;
    return createGPUCopyCommandNode<Parameters>({
      id: `${id}-copy-${chunkIndex}`,
      resources: [
        {buffer: chunk, usage: 'copy-source'},
        {buffer: view, usage: 'copy-destination'}
      ],
      workload: {
        operation: 'AnalysisChunkGather',
        commandCount: 1,
        maximumInvocationCount: 0,
        readByteLength: size,
        writeByteLength: size
      },
      compile: () => ({
        encode: ({commandEncoder, getBuffer}) => {
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: getBuffer(chunk),
            sourceOffset: chunk.byteOffset,
            destinationBuffer: getBuffer(view),
            destinationOffset: chunkDestinationOffset,
            size
          });
        }
      })
    });
  });
  return {view, nodes};
}
