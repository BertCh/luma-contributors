// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createGraphChunkGroupKernelNode} from '../../analysis-core/kernels/graph-chunk-groups';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** One bit per bound keeps classification scratch independent of the packed result capacity. */
export const BOUNDS_FILTER_ROWS_PER_WORD = 32;
/** Eight bit words per count keep the global uint32 scan below 64 MiB for uint32 source rows. */
export const BOUNDS_FILTER_WORDS_PER_BLOCK = 8;
/** Bounded bit counting replaces row-sized identity, flag, offset and compact-ID tables. */
export const BOUNDS_FILTER_ROWS_PER_BLOCK =
  BOUNDS_FILTER_ROWS_PER_WORD * BOUNDS_FILTER_WORDS_PER_BLOCK;

/** Device-sized scratch for construction-validated chunk lengths; declares no resources. @internal */
export function planBoundsFilterCompaction(
  device: Device,
  id: string,
  chunkLengths: readonly number[]
) {
  let firstRow = 0;
  let firstBlock = 0;
  const chunks = chunkLengths.map(length => {
    const wordCount = Math.ceil(length / BOUNDS_FILTER_ROWS_PER_WORD);
    const blockCount = Math.ceil(length / BOUNDS_FILTER_ROWS_PER_BLOCK);
    const chunk = {length, firstRow, firstBlock, wordCount, blockCount};
    firstRow += length;
    firstBlock += blockCount;
    return chunk;
  });
  const sizes = {
    'compaction block counts/offsets': Math.max(firstBlock, 1) * 4,
    'compaction total': 4,
    ...Object.fromEntries(
      chunks.map((chunk, index) => [
        `compaction bit mask chunk ${index}`,
        Math.max(chunk.wordCount, 1) * 4
      ])
    )
  };
  for (const [name, byteLength] of Object.entries(sizes)) {
    validateGPUStorageBindingByteLength(device, id, name, byteLength, 'use smaller bounds chunks');
    validateCapacity(
      id,
      `${name} byteLength (device.limits.maxBufferSize)`,
      byteLength,
      0,
      device.limits.maxBufferSize
    );
  }
  return {chunks, blockCount: firstBlock};
}

/** Stable, bounded packed result from chunk-local bit masks and one small global count scan. @internal */
export function getBoundsFilterCompactionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    plan: ReturnType<typeof planBoundsFilterCompaction>;
    masks: readonly GraphDataView<'uint32'>[];
    output: GraphDataView<'uint32'>;
    total: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, plan, masks, output, total} = props;
  const counts = createTransientView(graph, `${id}-block-counts`, 'uint32', plan.blockCount);
  const offsets = createTransientView(graph, `${id}-block-offsets`, 'uint32', plan.blockCount);
  const nodes: GPUCommandNode<Parameters>[] = [];
  for (const [chunkIndex, chunk] of plan.chunks.entries()) {
    if (chunk.length === 0) continue;
    nodes.push(
      createGraphChunkGroupKernelNode(graph, {
        id: `${id}-count-${chunkIndex}`,
        operation: 'GPUBoundsFilter',
        variant: 'count-bound-bits',
        invocationCount: chunk.blockCount,
        parameterValues: [chunk.firstBlock, chunk.wordCount],
        bindings: [
          {name: 'boundBits', view: masks[chunkIndex], type: 'u32', access: 'read'},
          {name: 'blockCounts', view: counts, type: 'u32', access: 'read_write'}
        ],
        body: `var count = 0u;
  for (var word = 0u; word < ${BOUNDS_FILTER_WORDS_PER_BLOCK}u; word++) {
    let position = index * ${BOUNDS_FILTER_WORDS_PER_BLOCK}u + word;
    if (position < getChunkParameter(37u)) {
      count += countOneBits(boundBits[boundBitsOffset + position]);
    }
  }
  blockCounts[blockCountsOffset + getChunkParameter(36u) + index] = count;`
      })
    );
  }
  // There is one primitive scan for the encoding, never one specialization per source chunk.
  // A bit contributes at most one and the constructor bounds the source total by uint32.
  nodes.push(
    ...new GPUScan({id: `${id}-scan`, input: counts, output: offsets}).getCommandNodes(graph),
    createGraphChunkGroupKernelNode(graph, {
      id: `${id}-total`,
      operation: 'GPUBoundsFilter',
      invocationCount: 1,
      parameterValues: [plan.blockCount],
      bindings: [
        {name: 'blockCounts', view: counts, type: 'u32', access: 'read'},
        {name: 'blockOffsets', view: offsets, type: 'u32', access: 'read'},
        {name: 'total', view: total, type: 'u32', access: 'read_write'}
      ],
      body: `let last = getChunkParameter(36u) - 1u;
  total[totalOffset] = blockOffsets[blockOffsetsOffset + last] + blockCounts[blockCountsOffset + last];`
    })
  );
  for (const [chunkIndex, chunk] of plan.chunks.entries()) {
    if (chunk.length === 0) continue;
    nodes.push(
      createGraphChunkGroupKernelNode(graph, {
        id: `${id}-scatter-${chunkIndex}`,
        operation: 'GPUBoundsFilter',
        variant: 'scatter-bound-bits',
        invocationCount: chunk.wordCount,
        parameterValues: [chunk.firstBlock, chunk.firstRow, output.length],
        bindings: [
          {name: 'boundBits', view: masks[chunkIndex], type: 'u32', access: 'read'},
          {name: 'blockOffsets', view: offsets, type: 'u32', access: 'read'},
          {name: 'visibleIds', view: output, type: 'u32', access: 'read_write'}
        ],
        body: `let boundBlock = index / ${BOUNDS_FILTER_WORDS_PER_BLOCK}u;
  var position = blockOffsets[blockOffsetsOffset + getChunkParameter(36u) + boundBlock];
  for (var word = boundBlock * ${BOUNDS_FILTER_WORDS_PER_BLOCK}u; word < index; word++) {
    position += countOneBits(boundBits[boundBitsOffset + word]);
  }
  var bits = boundBits[boundBitsOffset + index];
  loop {
    if (bits == 0u || position >= getChunkParameter(38u)) { break; }
    let bit = firstTrailingBit(bits);
    visibleIds[visibleIdsOffset + position] = getChunkParameter(37u) + index * ${BOUNDS_FILTER_ROWS_PER_WORD}u + bit;
    position++;
    bits &= bits - 1u;
  }`
      })
    );
  }
  return nodes;
}
