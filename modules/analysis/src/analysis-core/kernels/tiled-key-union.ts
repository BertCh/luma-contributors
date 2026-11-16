// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createGraphChunkGroupKernelNode,
  validateGraphChunkGroupKernelSupport,
  GRAPH_CHUNK_GROUP_PARAMETER_WORD_COUNT,
  GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET
} from './graph-chunk-groups';
import {getBinarySearchWGSL} from './binary-search-wgsl';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView
} from './gpu-contributor-utils';
import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from './graph-chunk-dispatch';
import type {GPUCardinalityPort, GPUOverflowStatusPort} from '../contracts/status';
import type {WGSLKernelProps, WGSLKernelBinding} from './wgsl-kernel-nodes';

/** Every dispatch owns one fixed uniform slot for runtime capacity, length and binding offsets. @internal */
export const TILED_KEY_UNION_PARAMETER_BYTE_LENGTH =
  GRAPH_CHUNK_GROUP_PARAMETER_WORD_COUNT * Uint32Array.BYTES_PER_ELEMENT;
/** Prefixes and global ranks are uint32; the sum of all padded tile lengths must not wrap. */
const TILED_KEY_UNION_MAXIMUM_ROW_COUNT = 0xffffffff;

/** Preflight global uint32 ranks before callers allocate their tile tables. @internal */
export function validateTiledKeyUnionRowCount(id: string, paddedRowCount: number): void {
  validateCapacity(id, 'padded tile rows', paddedRowCount, 0, TILED_KEY_UNION_MAXIMUM_ROW_COUNT);
}

/** Generic sorted unsigned 64-bit keys, stored low word first, with counts per unique key. @internal */
export type TiledKeyUnionTable = {keys: GraphDataView<'uint32x2'>; counts: GraphDataView<'uint32'>};
/** Full packed tile table, including zero-count padding after its valid prefix. @internal */
export type TiledKeyUnionInput = TiledKeyUnionTable & {count: GraphDataView<'uint32'>};
/** Uniform/storage support check for callers to run before allocating any tile scratch. @internal */
export function validateTiledKeyUnionKernelSupport<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  extraCountLength = 0
): void {
  validateGraphChunkGroupKernelSupport(graph, id, Math.max(7, 4 + extraCountLength));
}

const TILED_KEY_SEARCH_WGSL = `${getBinarySearchWGSL('otherKeys', 'u64', 'UnionKeys')}
fn findKey(key: vec2u) -> u32 {
  return lowerBoundUnionKeys(0u,
    min((arrayLength(&otherKeys) - otherKeysOffset) / 2u, otherCount[otherCountOffset]), key.yx);
}`;

/**
 * Exact sorted unsigned key union, without concatenating tile records into a global binding.
 * Tiles have the same padded width, keeping published scans and sorting pipelines reusable.
 * Each tile has unique keys in its valid prefix. Counts add across tiles and saturate at uint32
 * maximum; flags/prefixes/ranks cannot wrap because padded tile rows are bounded above.
 * Callers preflight their complete graph before allocating tile scratch. This helper validates
 * its own inputs, outputs and uniform support again before declaring its first resource.
 * @internal
 */
export function getTiledKeyUnionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  tiles: readonly TiledKeyUnionInput[],
  output: TiledKeyUnionTable,
  status: GPUCardinalityPort & GPUOverflowStatusPort,
  extraCounts: readonly GraphDataView<'uint32'>[] = []
): {nodes: GPUCommandNode<Parameters>[]; ranks: GraphDataView<'uint32'>[]} {
  validateTiledKeyUnionKernelSupport(graph, id, extraCounts.length);
  const outputs = {
    keys: output.keys,
    counts: output.counts,
    ...Object.fromEntries(Object.entries(status).map(([name, view]) => [`status.${name}`, view])),
    ...Object.fromEntries(extraCounts.map((view, index) => [`extraCounts[${index}]`, view]))
  };
  // Keys and counts publish separately; only status and extra counts are co-bound.
  validateDisjointWritableViews(id, outputs, {separateDispatches: true});
  validateDisjointWritableViews(
    id,
    Object.fromEntries(
      Object.entries(outputs).filter(([name]) => name !== 'keys' && name !== 'counts')
    )
  );
  const inputs = tiles.flatMap(tile => [tile.keys, tile.counts, tile.count]);
  validateGraphOutputsDisjointFromInputs(id, Object.values(outputs), inputs);
  validateGraphChunkBindingRanges(graph, id, {
    ...outputs,
    ...Object.fromEntries(inputs.map((view, index) => [`input[${index}]`, view]))
  });
  validateSinglePackedView(output.keys, ['uint32x2'], `${id} output.keys`);
  validateSinglePackedView(output.counts, ['uint32'], `${id} output.counts`);
  validateCapacity(
    id,
    'output.keys rows',
    output.keys.length,
    0,
    TILED_KEY_UNION_MAXIMUM_ROW_COUNT
  );
  for (const [name, view] of Object.entries(outputs)) {
    if (name === 'keys' || name === 'counts') continue;
    validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
    validateCapacity(id, `${name} rows`, view.length, 1);
  }
  validateCapacity(
    id,
    'output.counts rows',
    output.counts.length,
    output.keys.length,
    output.keys.length
  );
  let totalRows = 0;
  for (const [index, tile] of tiles.entries()) {
    validateSinglePackedView(tile.keys, ['uint32x2'], `${id} tiles[${index}].keys`);
    validateSinglePackedView(tile.counts, ['uint32'], `${id} tiles[${index}].counts`);
    validateSinglePackedView(tile.count, ['uint32'], `${id} tiles[${index}].count`);
    validateCapacity(id, `tiles[${index}].keys rows`, tile.keys.length);
    validateCapacity(
      id,
      `tiles[${index}].padded rows`,
      tile.keys.length,
      tiles[0].keys.length,
      tiles[0].keys.length
    );
    validateCapacity(
      id,
      `tiles[${index}].counts rows`,
      tile.counts.length,
      tile.keys.length,
      tile.keys.length
    );
    validateCapacity(id, `tiles[${index}].count rows`, tile.count.length, 1);
    const scratchByteLength = tile.keys.length * Uint32Array.BYTES_PER_ELEMENT;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      `tiles[${index}] flag/prefix/rank scratch`,
      scratchByteLength,
      'reduce the padded tile width'
    );
    validateCapacity(
      id,
      `tiles[${index}] scratch byteLength`,
      scratchByteLength,
      0,
      graph.device.limits.maxBufferSize
    );
    totalRows += tile.keys.length;
    validateTiledKeyUnionRowCount(id, totalRows);
  }
  const nodes: GPUCommandNode<Parameters>[] = [];
  const read = (name: string, view: GraphDataView): WGSLKernelBinding => ({
    name,
    view,
    type: 'u32',
    access: 'read'
  });
  const write = (name: string, view: GraphDataView): WGSLKernelBinding => ({
    name,
    view,
    type: 'u32',
    access: 'read_write'
  });
  const kernel = (props: Omit<WGSLKernelProps<Parameters>, 'operation'>) => {
    nodes.push(
      createGraphChunkGroupKernelNode(graph, {
        ...props,
        operation: 'AnalysisTiledKeyUnion',
        parameterValues: [output.keys.length],
        declarations: `fn getUnionCapacity() -> u32 { return getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET}u); }\n${props.declarations ?? ''}`,
        body: props.body.replace(/\bCAPACITY\b/g, 'getUnionCapacity()')
      })
    );
  };
  const flags = tiles.map((tile, index) =>
    createTransientView(graph, `${id}-owned-${index}`, 'uint32', tile.keys.length)
  );
  const prefixes = tiles.map((tile, index) =>
    createTransientView(graph, `${id}-prefix-${index}`, 'uint32', tile.keys.length)
  );
  const ranks = tiles.map((tile, index) =>
    createTransientView(graph, `${id}-rank-${index}`, 'uint32', tile.keys.length)
  );
  const total = createTransientView(graph, `${id}-total`, 'uint32', 1);
  kernel({
    id: `${id}-total-clear`,
    invocationCount: 1,
    bindings: [write('total', total)],
    body: 'total[totalOffset] = 0u;'
  });
  for (const [tileIndex, tile] of tiles.entries()) {
    kernel({
      id: `${id}-owned-${tileIndex}`,
      invocationCount: tile.keys.length,
      bindings: [read('tileCounts', tile.counts), write('owned', flags[tileIndex])],
      body: 'owned[ownedOffset + index] = select(0u, 1u, tileCounts[tileCountsOffset + index] != 0u);'
    });
    for (let previous = 0; previous < tileIndex; previous++) {
      kernel({
        id: `${id}-deduplicate-${tileIndex}-${previous}`,
        invocationCount: tile.keys.length,
        bindings: [
          read('keys', tile.keys),
          read('otherKeys', tiles[previous].keys),
          read('otherCount', tiles[previous].count),
          write('owned', flags[tileIndex])
        ],
        declarations: TILED_KEY_SEARCH_WGSL,
        body: `if (owned[ownedOffset + index] == 0u) { return; }
  let key = vec2u(keys[keysOffset + 2u * index], keys[keysOffset + 2u * index + 1u]);
  let position = findKey(key);
  if (position < min((arrayLength(&otherKeys) - otherKeysOffset) / 2u, otherCount[otherCountOffset]) && all(key == vec2u(otherKeys[otherKeysOffset + 2u * position], otherKeys[otherKeysOffset + 2u * position + 1u]))) {
    owned[ownedOffset + index] = 0u;
  }`
      });
    }
    nodes.push(
      ...new GPUScan({
        id: `${id}-scan-${tileIndex}`,
        input: flags[tileIndex],
        output: prefixes[tileIndex],
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );
    kernel({
      id: `${id}-rank-start-${tileIndex}`,
      invocationCount: tile.keys.length,
      bindings: [read('prefix', prefixes[tileIndex]), write('rank', ranks[tileIndex])],
      body: 'rank[rankOffset + index] = prefix[prefixOffset + index];'
    });
    kernel({
      id: `${id}-total-${tileIndex}`,
      invocationCount: 1,
      bindings: [
        read('prefix', prefixes[tileIndex]),
        read('owned', flags[tileIndex]),
        write('total', total)
      ],
      body: `let last = arrayLength(&owned) - ownedOffset - 1u;
  total[totalOffset] += prefix[prefixOffset + last] + owned[ownedOffset + last];`
    });
  }
  kernel({
    id: `${id}-clear-keys`,
    invocationCount: output.keys.length,
    bindings: [write('keysOut', output.keys)],
    body: 'keysOut[keysOutOffset + 2u * index] = 0xffffffffu; keysOut[keysOutOffset + 2u * index + 1u] = 0xffffffffu;'
  });
  kernel({
    id: `${id}-clear-counts`,
    invocationCount: output.counts.length,
    bindings: [write('countsOut', output.counts)],
    body: 'countsOut[countsOutOffset + index] = 0u;'
  });
  for (const [tileIndex, tile] of tiles.entries()) {
    for (const [otherIndex, other] of tiles.entries()) {
      if (otherIndex === tileIndex) continue;
      kernel({
        id: `${id}-rank-${tileIndex}-${otherIndex}`,
        invocationCount: tile.keys.length,
        bindings: [
          read('keys', tile.keys),
          read('otherKeys', other.keys),
          read('otherCount', other.count),
          read('tileCounts', tile.counts),
          read('otherPrefix', prefixes[otherIndex]),
          read('otherOwned', flags[otherIndex]),
          write('rank', ranks[tileIndex])
        ],
        declarations: TILED_KEY_SEARCH_WGSL,
        body: `if (tileCounts[tileCountsOffset + index] == 0u) { return; }
  let position = findKey(vec2u(keys[keysOffset + 2u * index], keys[keysOffset + 2u * index + 1u]));
  let length = arrayLength(&otherOwned) - otherOwnedOffset;
  var below = otherPrefix[otherPrefixOffset + min(position, length - 1u)];
  if (position == length) { below += otherOwned[otherOwnedOffset + length - 1u]; }
  rank[rankOffset + index] += below;`
      });
    }
    kernel({
      id: `${id}-publish-keys-${tileIndex}`,
      invocationCount: tile.keys.length,
      bindings: [
        read('keys', tile.keys),
        read('rank', ranks[tileIndex]),
        read('owned', flags[tileIndex]),
        write('keysOut', output.keys)
      ],
      body: `let destination = rank[rankOffset + index];
  if (owned[ownedOffset + index] != 0u && destination < CAPACITY) {
    keysOut[keysOutOffset + 2u * destination] = keys[keysOffset + 2u * index];
    keysOut[keysOutOffset + 2u * destination + 1u] = keys[keysOffset + 2u * index + 1u];
  }`
    });
    kernel({
      id: `${id}-publish-counts-${tileIndex}`,
      invocationCount: tile.keys.length,
      bindings: [
        read('tileCounts', tile.counts),
        read('rank', ranks[tileIndex]),
        write('countsOut', output.counts)
      ],
      body: `let destination = rank[rankOffset + index];
  if (tileCounts[tileCountsOffset + index] != 0u && destination < CAPACITY) {
    let previous = countsOut[countsOutOffset + destination];
    let increment = tileCounts[tileCountsOffset + index];
    countsOut[countsOutOffset + destination] = previous + min(increment, 0xffffffffu - previous);
  }`
    });
  }
  kernel({
    id: `${id}-publish-status`,
    invocationCount: 1,
    bindings: [
      read('total', total),
      write('count', status.count),
      write('overflow', status.overflow),
      ...(status.requiredCount ? [write('requiredCount', status.requiredCount)] : []),
      ...extraCounts.map((view, index) => write(`extraCount${index}`, view))
    ],
    body: `let required = total[totalOffset];
  count[countOffset] = min(required, CAPACITY);
  overflow[overflowOffset] = select(0u, 1u, required > CAPACITY);
  ${status.requiredCount ? 'requiredCount[requiredCountOffset] = required;' : ''}
  ${extraCounts.map((_, index) => `extraCount${index}[extraCount${index}Offset] = min(required, CAPACITY);`).join('\n')}`
  });
  return {nodes, ranks};
}
