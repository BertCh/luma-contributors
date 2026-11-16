// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createGraphChunkGroupKernelNode,
  GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET
} from '../../analysis-core/kernels/graph-chunk-groups';
import type {WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Bounded Lorenz walks; published GPUScan supports uint32, not weighted float population/income. */
export const INEQUALITY_PREFIX_TILE_ROWS = 256;

type LorenzLevel = {
  keys: GraphDataView<'uint32'>;
  length: number;
  pairs?: GraphDataView<'float32'>;
  weights?: GraphDataView<'float32'>;
  incomes?: GraphDataView<'float32'>;
  tailKeys: GraphDataView<'uint32'>;
  tailPairs: GraphDataView<'float32'>;
};
const read = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
  name,
  view,
  type,
  access: 'read'
});
const write = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
  name,
  view,
  type,
  access: 'read_write'
});

/**
 * Inclusive population and income coordinates in value-sorted zone segments. Each invocation
 * walks at most 256 rows; recursive tail summaries and reverse carries propagate zone prefixes.
 * Every level uses the same summary/carry sources with runtime lengths and binding offsets.
 * This domain helper keeps two float32 planes (each fits the existing scalar-row binding bound).
 * @internal
 */
export function createInequalityLorenzPrefixNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    sortedZones: GraphDataView<'uint32'>;
    sortedRows: GraphDataView<'uint32'>;
    values: GraphDataView<'float32'>;
    weights?: GraphDataView<'float32'>;
    zoneCount: number;
  }
): {
  weights: GraphDataView<'float32'>;
  incomes: GraphDataView<'float32'>;
  nodes: GPUCommandNode<Parameters>[];
} {
  const {id, sortedZones, sortedRows, values, weights, zoneCount} = props;
  const rowCount = sortedRows.length;
  const nodes: GPUCommandNode<Parameters>[] = [];
  const scalar = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'float32', length);
  const keys = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', length);
  const prefixWeights = scalar('weights', rowCount);
  const prefixIncomes = scalar('incomes', rowCount);
  const tileCount = Math.ceil(rowCount / INEQUALITY_PREFIX_TILE_ROWS);
  const first: LorenzLevel = {
    keys: sortedZones,
    length: rowCount,
    weights: prefixWeights,
    incomes: prefixIncomes,
    tailKeys: keys('tail-keys-0', tileCount),
    tailPairs: scalar('tail-pairs-0', tileCount * 2)
  };
  const levels: LorenzLevel[] = [first];
  nodes.push(
    createGraphChunkGroupKernelNode(graph, {
      id: `${id}-rows`,
      operation: 'GPUInequality',
      variant: 'lorenz-prefix-rows',
      invocationCount: tileCount,
      parameterValues: [rowCount, zoneCount],
      bindings: [
        read('zoneKeys', sortedZones, 'u32'),
        read('sortedRows', sortedRows, 'u32'),
        read('values', values, 'f32'),
        ...(weights ? [read('weights', weights, 'f32')] : []),
        write('prefixWeights', prefixWeights, 'f32'),
        write('prefixIncomes', prefixIncomes, 'f32'),
        write('tailKeys', first.tailKeys, 'u32'),
        write('tailPairs', first.tailPairs, 'f32')
      ],
      body: `let firstRow = index * ${INEQUALITY_PREFIX_TILE_ROWS}u;
  let endRow = min(firstRow + ${INEQUALITY_PREFIX_TILE_ROWS}u, getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET}u));
  var previousZone = 0xffffffffu;
  var population = 0.0;
  var income = 0.0;
  for (var position = firstRow; position < endRow; position++) {
    let zone = zoneKeys[zoneKeysOffset + position];
    if (zone != previousZone) { population = 0.0; income = 0.0; }
    if (zone < getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET + 1}u)) {
      let row = sortedRows[sortedRowsOffset + position];
      let weight = ${weights ? 'weights[weightsOffset + row]' : '1.0'};
      population += weight;
      income += weight * values[valuesOffset + row];
    }
    prefixWeights[prefixWeightsOffset + position] = population;
    prefixIncomes[prefixIncomesOffset + position] = income;
    previousZone = zone;
  }
  tailKeys[tailKeysOffset + index] = previousZone;
  tailPairs[tailPairsOffset + 2u * index] = population;
  tailPairs[tailPairsOffset + 2u * index + 1u] = income;`
    })
  );
  let inputKeys = first.tailKeys;
  let inputPairs = first.tailPairs;
  // Scan a singleton summary too: the same four pipeline families exist at every hierarchy height.
  do {
    const levelIndex = levels.length;
    const length = inputKeys.length;
    const tails = Math.ceil(length / INEQUALITY_PREFIX_TILE_ROWS);
    const level: LorenzLevel = {
      keys: inputKeys,
      length,
      pairs: scalar(`prefix-pairs-${levelIndex}`, length * 2),
      tailKeys: keys(`tail-keys-${levelIndex}`, tails),
      tailPairs: scalar(`tail-pairs-${levelIndex}`, tails * 2)
    };
    nodes.push(
      createGraphChunkGroupKernelNode(graph, {
        id: `${id}-summaries-${levelIndex}`,
        operation: 'GPUInequality',
        variant: 'lorenz-prefix-summaries',
        invocationCount: tails,
        parameterValues: [length],
        bindings: [
          read('zoneKeys', inputKeys, 'u32'),
          read('inputPairs', inputPairs, 'f32'),
          write('prefixPairs', level.pairs!, 'f32'),
          write('tailKeys', level.tailKeys, 'u32'),
          write('tailPairs', level.tailPairs, 'f32')
        ],
        body: `let firstRow = index * ${INEQUALITY_PREFIX_TILE_ROWS}u;
  let endRow = min(firstRow + ${INEQUALITY_PREFIX_TILE_ROWS}u, getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET}u));
  var previousZone = 0xffffffffu;
  var population = 0.0;
  var income = 0.0;
  for (var position = firstRow; position < endRow; position++) {
    let zone = zoneKeys[zoneKeysOffset + position];
    if (zone != previousZone) { population = 0.0; income = 0.0; }
    population += inputPairs[inputPairsOffset + 2u * position];
    income += inputPairs[inputPairsOffset + 2u * position + 1u];
    prefixPairs[prefixPairsOffset + 2u * position] = population;
    prefixPairs[prefixPairsOffset + 2u * position + 1u] = income;
    previousZone = zone;
  }
  tailKeys[tailKeysOffset + index] = previousZone;
  tailPairs[tailPairsOffset + 2u * index] = population;
  tailPairs[tailPairsOffset + 2u * index + 1u] = income;`
      })
    );
    levels.push(level);
    inputKeys = level.tailKeys;
    inputPairs = level.tailPairs;
  } while (inputKeys.length > 1);
  for (let levelIndex = levels.length - 1; levelIndex >= 0; levelIndex--) {
    const level = levels[levelIndex];
    const parent = levels[levelIndex + 1];
    const parentKeys = parent?.keys ?? level.tailKeys;
    const parentPairs = parent?.pairs ?? level.tailPairs;
    nodes.push(
      createGraphChunkGroupKernelNode(graph, {
        id: `${id}-carry-${levelIndex}`,
        operation: 'GPUInequality',
        variant: level.pairs ? 'lorenz-summary-carry' : 'lorenz-row-carry',
        invocationCount: level.length,
        bindings: [
          read('zoneKeys', level.keys, 'u32'),
          read('parentKeys', parentKeys, 'u32'),
          read('parentPairs', parentPairs, 'f32'),
          ...(level.pairs
            ? [write('prefixPairs', level.pairs, 'f32')]
            : [
                write('prefixWeights', level.weights!, 'f32'),
                write('prefixIncomes', level.incomes!, 'f32')
              ])
        ],
        body: `let block = index / ${INEQUALITY_PREFIX_TILE_ROWS}u;
  if (block > 0u && zoneKeys[zoneKeysOffset + index] == parentKeys[parentKeysOffset + block - 1u]) {
    let population = parentPairs[parentPairsOffset + 2u * (block - 1u)];
    let income = parentPairs[parentPairsOffset + 2u * (block - 1u) + 1u];
    ${
      level.pairs
        ? `prefixPairs[prefixPairsOffset + 2u * index] += population;
    prefixPairs[prefixPairsOffset + 2u * index + 1u] += income;`
        : `prefixWeights[prefixWeightsOffset + index] += population;
    prefixIncomes[prefixIncomesOffset + index] += income;`
    }
  }`
      })
    );
  }
  return {weights: prefixWeights, incomes: prefixIncomes, nodes};
}
