// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {getGraphViewChunks} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getTiledKeyUnionNodes} from '../../analysis-core/kernels/tiled-key-union';
import {getGroupStructureNodes} from './group-statistics-sort';
import {createGroupStatisticsKernelNode} from './group-statistics-kernel';
import {readBinding, writeBinding} from './group-statistics-common';
import {getColumnReductionNodes} from './group-statistics-moments';
import {getGroupMomentTreeNodes} from './group-statistics-moment-tree';
import type {GPUGroupStatisticsProps, GPUGroupStatistic} from './gpu-group-statistics';

type TiledGroupKeyRows =
  | GraphDataView<'uint32' | 'uint32x2'>
  | GraphVectorView<'uint32' | 'uint32x2'>;
type TiledGroupValueRows = GraphDataView<'float32'> | GraphVectorView<'float32'>;

/** Fixed local sorts; integer global reductions use the original unit and total-row headroom. */
export function getTiledGroupStatisticsNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUGroupStatisticsProps<TiledGroupKeyRows, TiledGroupValueRows>,
  id: string,
  tileLength: number,
  sumScale: number,
  variance: 'sample' | 'population'
): GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const capacity = props.output.keys.length;
  const kernel = (properties: Omit<ParametersOfGroupKernel<Parameters>, 'operation'>) => {
    nodes.push(
      createGroupStatisticsKernelNode(graph, {...properties, operation: 'GPUGroupStatistics'})
    );
  };
  const gather = (
    source:
      | GraphDataView<'uint32' | 'uint32x2' | 'float32'>
      | GraphVectorView<'uint32' | 'uint32x2' | 'float32'>,
    name: string,
    tileIndex: number,
    padding: number
  ): GraphDataView => {
    const destination = createTransientView(
      graph,
      `${id}-tile-${tileIndex}-${name}`,
      source.format,
      tileLength
    );
    const components = source.rowByteLength / 4;
    kernel({
      id: `${id}-tile-${tileIndex}-${name}-pad`,
      invocationCount: tileLength,
      bindings: [writeBinding('destination', destination)],
      body: Array.from(
        {length: components},
        (_, component) =>
          `destination[destinationOffset + ${components}u * index + ${component}u] = ${padding}u;`
      ).join('\n')
    });
    const begin = tileIndex * tileLength;
    const end = Math.min(begin + tileLength, source.length);
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
        kernel({
          id: `${id}-tile-${tileIndex}-${name}-gather-${chunkIndex}`,
          invocationCount: last - first,
          rowOffset: first - begin,
          bindings: [readBinding('source', span), writeBinding('destination', destination)],
          body: Array.from(
            {length: components},
            (_, component) =>
              `destination[destinationOffset + ${components}u * (GROUP_ROW_OFFSET + index) + ${component}u] = source[sourceOffset + ${components}u * index + ${component}u];`
          ).join('\n')
        });
      }
      sourceBegin = sourceEnd;
    }
    return destination;
  };
  const tiles = Array.from({length: Math.ceil(props.keys.length / tileLength)}, (_, tileIndex) => {
    const tileId = `${id}-tile-${tileIndex}`;
    const keys = gather(props.keys, 'keys', tileIndex, 0xffffffff) as GraphDataView<
      'uint32' | 'uint32x2'
    >;
    const mask = props.mask
      ? (gather(props.mask, 'mask', tileIndex, 0) as GraphDataView<'uint32'>)
      : undefined;
    if (props.keyCount !== undefined)
      kernel({
        id: `${tileId}-dense-domain`,
        invocationCount: tileLength,
        bindings: [writeBinding('keys', keys)],
        body: `if (keys[keysOffset + index] >= ${props.keyCount}u) { keys[keysOffset + index] = 0xffffffffu; }`
      });
    const output = {
      keys: createTransientView(graph, `${tileId}-group-keys`, props.keys.format, tileLength),
      counts: createTransientView(graph, `${tileId}-group-counts`, 'uint32', tileLength),
      status: {
        count: createTransientView(graph, `${tileId}-count`, 'uint32', 1),
        overflow: createTransientView(graph, `${tileId}-overflow`, 'uint32', 1)
      }
    };
    const local = getGroupStructureNodes(graph, {
      id: tileId,
      operation: 'GPUGroupStatistics',
      keys,
      mask,
      output
    });
    nodes.push(...local.nodes);
    // Cell-key union uses the same unsigned lexicographic order, with two all-ones words as sentinel.
    const cells = createTransientView(graph, `${tileId}-union-keys`, 'uint32x2', tileLength);
    kernel({
      id: `${tileId}-union-keys`,
      invocationCount: tileLength,
      bindings: [
        readBinding('keys', output.keys),
        readBinding('counts', output.counts),
        writeBinding('cells', cells)
      ],
      body: `let occupied = counts[countsOffset + index] != 0u;
  cells[cellsOffset + 2u * index] = keys[keysOffset + ${props.keys.format === 'uint32x2' ? '2u * index' : 'index'}];
  cells[cellsOffset + 2u * index + 1u] = ${props.keys.format === 'uint32x2' ? 'keys[keysOffset + 2u * index + 1u]' : 'select(0xffffffffu, 0u, occupied)'};`
    });
    const values = props.columns.map(
      (column, columnIndex) =>
        gather(
          column.values,
          `column-${columnIndex}`,
          tileIndex,
          0x7fc00000
        ) as GraphDataView<'float32'>
    );
    return {structure: local.structure, cells, counts: output.counts, values};
  });
  const globalCells = createTransientView(graph, `${id}-union-keys`, 'uint32x2', capacity);
  let ranks: GraphDataView<'uint32'>[];
  if (props.keyCount === undefined) {
    const union = getTiledKeyUnionNodes(
      graph,
      `${id}-union`,
      tiles.map(tile => ({
        keys: tile.cells,
        counts: tile.counts,
        count: tile.structure.groupTotal
      })),
      {keys: globalCells, counts: props.output.counts},
      props.output.status
    );
    nodes.push(...union.nodes);
    ranks = union.ranks;
  } else {
    kernel({
      id: `${id}-dense-clear`,
      invocationCount: capacity,
      bindings: [writeBinding('cells', globalCells), writeBinding('counts', props.output.counts)],
      body: 'cells[cellsOffset + 2u * index] = index; cells[cellsOffset + 2u * index + 1u] = 0u; counts[countsOffset + index] = 0u;'
    });
    ranks = tiles.map((tile, tileIndex) => {
      const rank = createTransientView(graph, `${id}-rank-${tileIndex}`, 'uint32', tileLength);
      kernel({
        id: `${id}-dense-merge-${tileIndex}`,
        invocationCount: tileLength,
        bindings: [
          readBinding('keys', tile.cells),
          readBinding('tileCounts', tile.counts),
          writeBinding('rank', rank),
          writeBinding('counts', props.output.counts)
        ],
        body: `let key = keys[keysOffset + 2u * index];
  rank[rankOffset + index] = key;
  if (tileCounts[tileCountsOffset + index] != 0u) { counts[countsOffset + key] += tileCounts[tileCountsOffset + index]; }`
      });
      return rank;
    });
    kernel({
      id: `${id}-dense-status`,
      invocationCount: 1,
      bindings: [
        writeBinding('count', props.output.status.count),
        writeBinding('overflow', props.output.status.overflow),
        ...(props.output.status.requiredCount
          ? [writeBinding('requiredCount', props.output.status.requiredCount)]
          : [])
      ],
      body: `count[countOffset] = ${capacity}u; overflow[overflowOffset] = 0u; ${props.output.status.requiredCount ? `requiredCount[requiredCountOffset] = ${capacity}u;` : ''}`
    });
  }
  kernel({
    id: `${id}-keys-publish`,
    invocationCount: capacity,
    bindings: [readBinding('cells', globalCells), writeBinding('keys', props.output.keys)],
    body:
      props.keys.format === 'uint32x2'
        ? 'keys[keysOffset + 2u * index] = cells[cellsOffset + 2u * index]; keys[keysOffset + 2u * index + 1u] = cells[cellsOffset + 2u * index + 1u];'
        : 'keys[keysOffset + index] = cells[cellsOffset + 2u * index];'
  });
  const rowGroups = tiles.map((tile, tileIndex) => {
    const output = createTransientView(
      graph,
      `${id}-row-groups-${tileIndex}`,
      'uint32',
      tileLength
    );
    kernel({
      id: `${id}-row-groups-${tileIndex}`,
      invocationCount: tileLength,
      bindings: [
        readBinding('localGroups', tile.structure.rowGroups),
        readBinding('rank', ranks[tileIndex]),
        writeBinding('groups', output)
      ],
      body: `let localGroup = localGroups[localGroupsOffset + index];
  var group = 0xffffffffu;
  if (localGroup != 0xffffffffu) { group = rank[rankOffset + localGroup]; }
  groups[groupsOffset + index] = select(0xffffffffu, group, group < ${capacity}u);`
    });
    return output;
  });
  for (const [columnIndex, column] of props.columns.entries()) {
    const columnId = `${id}-column-${columnIndex}`;
    const requested = new Set(column.statistics);
    const valueChunks = tiles.map((tile, tileIndex) => ({
      valueBits: graph.createDataView(tile.values[columnIndex].buffer, {
        format: 'uint32',
        length: tileLength
      }),
      rowOffset: 0,
      rowGroups: rowGroups[tileIndex]
    }));
    nodes.push(
      ...getColumnReductionNodes(graph, {
        id: columnId,
        operation: 'GPUGroupStatistics',
        structure: {
          ...tiles[0].structure,
          rowCount: props.keys.length,
          capacity,
          rowGroups: rowGroups[0]
        },
        valueChunks,
        sumScale,
        scaleMode: props.scaleMode,
        variance,
        needs: {
          sums: requested.has('sum') || requested.has('mean'),
          minimum: requested.has('minimum'),
          maximum: requested.has('maximum'),
          moments: ['variance', 'standardDeviation', 'skewness', 'kurtosis'].some(statistic =>
            requested.has(statistic as GPUGroupStatistic)
          ),
          zScore: false
        },
        output: column.output,
        getMomentNodes: (means, output) => {
          const momentNodes: GPUCommandNode<Parameters>[] = [];
          const append = (properties: Omit<ParametersOfGroupKernel<Parameters>, 'operation'>) =>
            momentNodes.push(
              createGroupStatisticsKernelNode(graph, {
                ...properties,
                operation: 'GPUGroupStatistics'
              })
            );
          append({
            id: `${columnId}-moments-clear`,
            invocationCount: capacity,
            bindings: [writeBinding('moments', output, 'f32')],
            body: 'for (var component = 0u; component < 4u; component++) { moments[momentsOffset + 4u * index + component] = 0.0; }'
          });
          for (const [tileIndex, tile] of tiles.entries()) {
            const localMeans = createTransientView(
              graph,
              `${columnId}-means-${tileIndex}`,
              'float32',
              tileLength * 2
            );
            const localMoments = createTransientView(
              graph,
              `${columnId}-moments-${tileIndex}`,
              'float32',
              tileLength * 4
            );
            append({
              id: `${columnId}-means-${tileIndex}`,
              invocationCount: tileLength,
              bindings: [
                readBinding('rank', ranks[tileIndex]),
                readBinding('means', means, 'f32'),
                writeBinding('localMeans', localMeans, 'f32')
              ],
              body: `let group = rank[rankOffset + index];
  var mean = 0.0; var exponent = 0.0;
  if (group < ${capacity}u) { mean = means[meansOffset + 2u * group]; exponent = means[meansOffset + 2u * group + 1u]; }
  localMeans[localMeansOffset + 2u * index] = mean;
  localMeans[localMeansOffset + 2u * index + 1u] = exponent;`
            });
            momentNodes.push(
              ...getGroupMomentTreeNodes(graph, {
                id: `${columnId}-tile-${tileIndex}`,
                operation: 'GPUGroupStatistics',
                structure: tile.structure,
                valueChunks: [{valueBits: valueChunks[tileIndex].valueBits, rowOffset: 0}],
                means: localMeans,
                output: localMoments,
                retainScaledOddMoments: true
              })
            );
            append({
              id: `${columnId}-moments-merge-${tileIndex}`,
              invocationCount: tileLength,
              bindings: [
                readBinding('rank', ranks[tileIndex]),
                readBinding('counts', tile.counts),
                readBinding('localMoments', localMoments, 'f32'),
                writeBinding('moments', output, 'f32')
              ],
              body: `let group = rank[rankOffset + index];
  if (group < ${capacity}u && counts[countsOffset + index] != 0u) {
    for (var component = 0u; component < 4u; component++) {
      moments[momentsOffset + 4u * group + component] += localMoments[localMomentsOffset + 4u * index + component];
    }
  }`
            });
          }
          // Restore signed terms only after opposite-sign tiles have cancelled. Restoring
          // each tile separately can overflow a cubic partial even when the final sum is zero.
          append({
            id: `${columnId}-moments-restore`,
            invocationCount: capacity,
            bindings: [readBinding('means', means, 'f32'), writeBinding('moments', output, 'f32')],
            body: `let exponent = i32(means[meansOffset + 2u * index + 1u]);
  let first = momentsOffset + 4u * index;
  let third = first + 2u;
  moments[first] = ldexp(moments[first], exponent);
  moments[third] = ldexp(ldexp(ldexp(moments[third], exponent), exponent), exponent);`
          });
          return momentNodes;
        }
      }).nodes
    );
  }
  return nodes;
}

type ParametersOfGroupKernel<Context> = Parameters<
  typeof createGroupStatisticsKernelNode<Context>
>[1];
