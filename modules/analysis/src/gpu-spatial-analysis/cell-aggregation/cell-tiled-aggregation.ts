// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPUCellAggregation, type GPUCellAggregationProps} from './gpu-cell-aggregation';
import {createCellLevelKernelNode} from './cell-level-kernel';
import {getTiledKeyUnionNodes} from '../../analysis-core/kernels/tiled-key-union';
import {getGraphViewChunks} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getFixedPointWGSL, ORDERED_KEY_WGSL} from '../../analysis-core/cell-key-wgsl';
import {type GPUCellTable} from './cell-table';
import type {
  WGSLKernelBinding,
  WGSLKernelProps
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Fixed-width local sorts followed by an exact union of all local cell keys and integer sums. */
export function getTiledCellAggregationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUCellAggregationProps,
  id: string,
  tileLength: number,
  sumScale: number
): GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  // Keep all caller columns out of read-only working bindings: disjoint columns may share
  // one physical buffer, whose read-only and writable storage usages cannot coexist.
  const callerOutput = props.output;
  const stagedOutput: GPUCellTable = {...callerOutput};
  for (const field of ['cells', 'counts', 'sums', 'sumValues', 'minimums', 'maximums'] as const) {
    const view = callerOutput[field];
    if (view)
      Object.assign(stagedOutput, {
        [field]: createTransientView(graph, `${id}-working-${field}`, view.format, view.length)
      });
  }
  props = {...props, output: stagedOutput};

  const read = (
    name: string,
    view: GraphDataView,
    type: 'u32' | 'f32' = 'u32'
  ): WGSLKernelBinding => ({name, view, type, access: 'read'});
  const write = (
    name: string,
    view: GraphDataView,
    type: 'u32' | 'f32' = 'u32'
  ): WGSLKernelBinding => ({name, view, type, access: 'read_write'});
  const kernel = (
    properties: Omit<WGSLKernelProps<Parameters>, 'operation'>,
    destinationRowBase = 0
  ) => {
    const result = createCellLevelKernelNode(
      graph,
      {...properties, operation: 'GPUCellAggregation'},
      {CAPACITY: props.output.cells.length, DESTINATION_ROW_BASE: destinationRowBase}
    );
    nodes.push(...result.uploads, result.node);
  };
  const rows = (props.positions ?? props.cells)!;
  const wantsSums = Boolean(props.output.sums || props.output.sumValues);
  const tables: GPUCellTable[] = [];
  for (let tileIndex = 0; tileIndex < Math.ceil(rows.length / tileLength); tileIndex++) {
    const tileId = `${id}-tile-${tileIndex}`;
    const begin = tileIndex * tileLength;
    const end = Math.min(begin + tileLength, rows.length);
    const inputs: {
      positions?: GraphDataView<'float32x2'>;
      cells?: GraphDataView<'uint32x2'>;
      values?: GraphDataView<'float32'>;
      mask?: GraphDataView<'uint32'>;
    } = {};
    for (const [name, source] of [
      ['positions', props.positions],
      ['cells', props.cells],
      ['values', props.values],
      ['mask', props.mask]
    ] as const) {
      if (!source) continue;
      const destination = createTransientView(
        graph,
        `${tileId}-${name}`,
        source.format,
        tileLength
      );
      Object.assign(inputs, {[name]: destination});
      const components = source.rowByteLength / 4;
      kernel({
        id: `${tileId}-${name}-pad`,
        invocationCount: tileLength,
        bindings: [write('destination', destination)],
        body: Array.from(
          {length: components},
          (_, component) =>
            `destination[destinationOffset + ${components}u * index + ${component}u] = ${name === 'mask' ? '0u' : name === 'cells' ? '0xffffffffu' : '0x7fc00000u'};`
        ).join('\n')
      });
      let sourceBegin = 0;
      for (const [chunkIndex, chunk] of getGraphViewChunks(source).entries()) {
        const sourceEnd = sourceBegin + chunk.length;
        const first = Math.max(begin, sourceBegin);
        const last = Math.min(end, sourceEnd);
        if (first < last) {
          const span = graph.createDataView(chunk.buffer, {
            format: chunk.format,
            length: last - first,
            byteOffset: chunk.byteOffset + (first - sourceBegin) * chunk.rowByteLength
          });
          kernel(
            {
              id: `${tileId}-${name}-gather-${chunkIndex}`,
              invocationCount: last - first,
              bindings: [read('source', span), write('destination', destination)],
              body: Array.from(
                {length: components},
                (_, component) =>
                  `destination[destinationOffset + ${components}u * (DESTINATION_ROW_BASE + index) + ${component}u] = source[sourceOffset + ${components}u * index + ${component}u];`
              ).join('\n')
            },
            first - begin
          );
        }
        sourceBegin = sourceEnd;
      }
    }
    const table: GPUCellTable = {
      cells: createTransientView(graph, `${tileId}-cells-out`, 'uint32x2', tileLength),
      counts: createTransientView(graph, `${tileId}-counts-out`, 'uint32', tileLength),
      ...(wantsSums
        ? {sums: createTransientView(graph, `${tileId}-sums-out`, 'uint32x2', tileLength)}
        : {}),
      ...(props.output.minimums
        ? {minimums: createTransientView(graph, `${tileId}-minimums-out`, 'float32', tileLength)}
        : {}),
      ...(props.output.maximums
        ? {maximums: createTransientView(graph, `${tileId}-maximums-out`, 'float32', tileLength)}
        : {}),
      status: {
        count: createTransientView(graph, `${tileId}-count`, 'uint32', 1),
        overflow: createTransientView(graph, `${tileId}-overflow`, 'uint32', 1)
      }
    };
    nodes.push(
      ...new GPUCellAggregation({
        ...props,
        ...inputs,
        id: tileId,
        scratchTileSize: undefined,
        scratchMode: 'packed',
        output: table,
        extraCounts: undefined
      }).getCommandNodes(graph)
    );
    tables.push(table);
  }
  const union = getTiledKeyUnionNodes(
    graph,
    `${id}-union`,
    tables.map(table => ({keys: table.cells, counts: table.counts, count: table.status.count})),
    {keys: props.output.cells, counts: props.output.counts},
    props.output.status,
    props.extraCounts
  );
  nodes.push(...union.nodes);
  const output = props.output;
  const sums = wantsSums
    ? (output.sums ??
      createTransientView(graph, `${id}-merged-sums`, 'uint32x2', output.cells.length))
    : undefined;
  if (sums) {
    kernel({
      id: `${id}-sums-clear`,
      invocationCount: sums.length,
      bindings: [write('sums', sums)],
      body: 'sums[sumsOffset + 2u * index] = 0u; sums[sumsOffset + 2u * index + 1u] = 0u;'
    });
    for (const [tileIndex, table] of tables.entries()) {
      kernel({
        id: `${id}-sums-merge-${tileIndex}`,
        invocationCount: tileLength,
        bindings: [
          read('tileCounts', table.counts),
          read('rank', union.ranks[tileIndex]),
          read('sourceSums', table.sums!),
          write('sums', sums)
        ],
        declarations: getFixedPointWGSL(sumScale),
        body: `let destination = rank[rankOffset + index];
  if (tileCounts[tileCountsOffset + index] == 0u || destination >= CAPACITY) { return; }
  let previous = vec2u(sums[sumsOffset + 2u * destination + 1u], sums[sumsOffset + 2u * destination]);
  let contribution = vec2u(sourceSums[sourceSumsOffset + 2u * index + 1u], sourceSums[sourceSumsOffset + 2u * index]);
  let combined = cellAddI64(previous, contribution);
  sums[sumsOffset + 2u * destination] = combined.y;
  sums[sumsOffset + 2u * destination + 1u] = combined.x;`
      });
    }
    if (output.sumValues)
      kernel({
        id: `${id}-sums-finish`,
        invocationCount: sums.length,
        bindings: [read('sums', sums), write('values', output.sumValues, 'f32')],
        declarations: getFixedPointWGSL(sumScale),
        body: 'values[valuesOffset + index] = cellI64ToF32(vec2u(sums[sumsOffset + 2u * index + 1u], sums[sumsOffset + 2u * index])) / SUM_SCALE;'
      });
  }
  for (const field of ['minimums', 'maximums'] as const) {
    const destination = output[field];
    if (!destination) continue;
    const ordered = createTransientView(graph, `${id}-${field}-keys`, 'uint32', destination.length);
    const minimum = field === 'minimums';
    kernel({
      id: `${id}-${field}-clear`,
      invocationCount: destination.length,
      bindings: [write('ordered', ordered)],
      body: `ordered[orderedOffset + index] = ${minimum ? '0xffffffffu' : '0u'};`
    });
    for (const [tileIndex, table] of tables.entries())
      kernel({
        id: `${id}-${field}-merge-${tileIndex}`,
        invocationCount: tileLength,
        bindings: [
          read('tileCounts', table.counts),
          read('rank', union.ranks[tileIndex]),
          read('sourceValues', table[field]!, 'f32'),
          write('ordered', ordered)
        ],
        declarations: ORDERED_KEY_WGSL,
        body: `let destination = rank[rankOffset + index];
  if (tileCounts[tileCountsOffset + index] != 0u && destination < CAPACITY) {
    ordered[orderedOffset + destination] = ${minimum ? 'min' : 'max'}(ordered[orderedOffset + destination], getOrderedKey(sourceValues[sourceValuesOffset + index]));
  }`
      });
    kernel({
      id: `${id}-${field}-finish`,
      invocationCount: destination.length,
      bindings: [
        read('ordered', ordered),
        read('counts', output.counts),
        write('values', destination, 'f32')
      ],
      declarations: ORDERED_KEY_WGSL,
      body: 'values[valuesOffset + index] = select(getNaN(), decodeOrderedKey(ordered[orderedOffset + index]), counts[countsOffset + index] != 0u);'
    });
  }
  for (const field of ['cells', 'counts', 'sums', 'sumValues', 'minimums', 'maximums'] as const) {
    const source = stagedOutput[field];
    const destination = callerOutput[field];
    if (!source || !destination) continue;
    const components = source.rowByteLength / 4;
    kernel({
      id: `${id}-publish-${field}`,
      invocationCount: source.length,
      bindings: [read('source', source), write('destination', destination)],
      body: Array.from(
        {length: components},
        (_, component) =>
          `destination[destinationOffset + ${components}u * index + ${component}u] = source[sourceOffset + ${components}u * index + ${component}u];`
      ).join('\n')
    });
  }
  return nodes;
}
