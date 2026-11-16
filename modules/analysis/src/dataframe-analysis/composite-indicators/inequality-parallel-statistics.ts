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
import {createSegmentSumNode} from '../../analysis-core/kernels/sorted-segment-sums';
import {getSortedSegmentOffsetNodes} from '../../analysis-core/kernels/sorted-segment-offsets';
import {INEQUALITY_HELPERS_WGSL, INEQUALITY_STATS_STRIDE} from './inequality-wgsl';
import {createInequalityLorenzPrefixNodes} from './inequality-lorenz-prefix';
import type {WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Six row terms: x log x, Lorenz area, Theil T, log ratios, Atkinson powers, absolute deviations. */
const INEQUALITY_TERM_COUNT = 6;
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

/** Domain prefix coordinates, row terms, shared tree reductions and binary-search Lorenz cuts. @internal */
export function createInequalityParallelStatisticsNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    sortedZones: GraphDataView<'uint32'>;
    sortedRows: GraphDataView<'uint32'>;
    values: GraphDataView<'float32'>;
    weights?: GraphDataView<'float32'>;
    zoneCount: number;
    parameters: GraphDataView<'float32'>;
    stats: GraphDataView<'float32'>;
    counts: GraphDataView<'uint32'>;
    lorenz: GraphDataView<'float32'>;
    knotCount: number;
  }
): GPUCommandNode<Parameters>[] {
  const {
    id,
    sortedZones,
    sortedRows,
    values,
    weights,
    zoneCount,
    parameters,
    stats,
    counts,
    lorenz,
    knotCount
  } = props;
  const prefix = createInequalityLorenzPrefixNodes(graph, {
    id: `${id}-lorenz-prefix`,
    sortedZones,
    sortedRows,
    values,
    weights,
    zoneCount
  });
  const offsets = createTransientView(graph, `${id}-zone-offsets`, 'uint32', zoneCount + 1);
  const terms = createTransientView(graph, `${id}-row-terms`, 'float32', values.length);
  const totals = createTransientView(graph, `${id}-term-totals`, 'float32', zoneCount);
  const declarations = `const STATS_STRIDE: u32 = ${INEQUALITY_STATS_STRIDE}u;\n${INEQUALITY_HELPERS_WGSL}`;
  const nodes: GPUCommandNode<Parameters>[] = [
    ...prefix.nodes,
    ...getSortedSegmentOffsetNodes(graph, {
      id,
      operation: 'GPUInequality',
      groupCount: zoneCount,
      sortedGroupKeys: sortedZones,
      segmentOffsets: offsets
    })
  ];
  nodes.push(
    createGraphChunkGroupKernelNode(graph, {
      id: `${id}-zone-stats`,
      operation: 'GPUInequality',
      variant: 'zone-totals',
      invocationCount: zoneCount,
      bindings: [
        read('zoneOffsets', offsets, 'u32'),
        read('sortedRows', sortedRows, 'u32'),
        read('values', values, 'f32'),
        read('prefixWeights', prefix.weights, 'f32'),
        read('prefixIncomes', prefix.incomes, 'f32'),
        write('stats', stats, 'f32'),
        write('counts', counts, 'u32')
      ],
      declarations,
      body: `let begin = zoneOffsets[zoneOffsetsOffset + index];
  let end = zoneOffsets[zoneOffsetsOffset + index + 1u];
  let base = statsOffset + index * STATS_STRIDE;
  let nan = getNaN();
  for (var slot = 0u; slot < STATS_STRIDE; slot++) { stats[base + slot] = nan; }
  counts[countsOffset + index] = end - begin;
  stats[base + 7u] = 0.0; stats[base + 8u] = 0.0; stats[base + 9u] = 0.0;
  if (begin == end) { return; }
  let population = prefixWeights[prefixWeightsOffset + end - 1u];
  let income = prefixIncomes[prefixIncomesOffset + end - 1u];
  stats[base] = getScaledQuotient(income, population, 1.0, 0);
  stats[base + 7u] = population; stats[base + 8u] = income;
  let hasZero = values[valuesOffset + sortedRows[sortedRowsOffset + begin]] == 0.0;
  stats[base + 3u] = select(0.0, nan, hasZero);`
    })
  );
  for (let term = 0; term < INEQUALITY_TERM_COUNT; term++) {
    nodes.push(
      createGraphChunkGroupKernelNode(graph, {
        id: `${id}-terms-${term}`,
        operation: 'GPUInequality',
        variant: 'inequality-row-term',
        invocationCount: values.length,
        parameterValues: [term, zoneCount],
        bindings: [
          read('sortedRows', sortedRows, 'u32'),
          read('zoneKeys', sortedZones, 'u32'),
          read('values', values, 'f32'),
          ...(weights ? [read('weights', weights, 'f32')] : []),
          read('stats', stats, 'f32'),
          read('prefixIncomes', prefix.incomes, 'f32'),
          read('params', parameters, 'f32'),
          write('terms', terms, 'f32')
        ],
        declarations,
        body: `let zone = zoneKeys[zoneKeysOffset + index];
  terms[termsOffset + index] = 0.0;
  if (zone >= getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET + 1}u)) { return; }
  let row = sortedRows[sortedRowsOffset + index];
  let value = values[valuesOffset + row];
  let weight = ${weights ? 'weights[weightsOffset + row]' : '1.0'};
  let base = statsOffset + zone * STATS_STRIDE;
  let population = stats[base + 7u]; let income = stats[base + 8u];
  let term = getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET}u);
  var contribution = 0.0;
  if (term == 0u) {
    if (value > 0.0) { contribution = weight * value * log(value); }
  } else if (income > 0.0) {
    let ratio = getScaledQuotient(value, stats[base], 1.0, 0);
    if (term == 1u) {
      contribution = (getScaledQuotient(weight, population, 1.0, 0)) * (2.0 * (getScaledQuotient(prefixIncomes[prefixIncomesOffset + index], income, 1.0, 0)) - weight * getScaledQuotient(value, income, 1.0, 0));
    } else if (term == 5u) { contribution = weight * abs(value - stats[base]);
    } else if (value > 0.0) {
      let logRatio = log(ratio);
      if (term == 2u) { contribution = weight * ratio * logRatio; }
      if (term == 3u) { contribution = weight * logRatio; }
      if (term == 4u) { contribution = weight * exp((1.0 - params[paramsOffset]) * logRatio); }
    }
  }
  terms[termsOffset + index] = contribution;`
      })
    );
    nodes.push(
      createSegmentSumNode(graph, {
        id: `${id}-term-sum-${term}`,
        operation: 'GPUInequality',
        input: terms,
        segmentOffsets: offsets,
        segmentCount: zoneCount,
        output: totals
      })
    );
    nodes.push(
      createGraphChunkGroupKernelNode(graph, {
        id: `${id}-term-finish-${term}`,
        operation: 'GPUInequality',
        variant: 'inequality-term-finish',
        invocationCount: zoneCount,
        parameterValues: [term],
        bindings: [
          read('totals', totals, 'f32'),
          read('counts', counts, 'u32'),
          read('params', parameters, 'f32'),
          write('stats', stats, 'f32')
        ],
        declarations,
        body: `if (counts[countsOffset + index] == 0u) { return; }
  let base = statsOffset + index * STATS_STRIDE;
  let term = getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET}u);
  let total = totals[totalsOffset + index];
  if (term == 0u) { stats[base + 9u] = total; return; }
  let population = stats[base + 7u]; let income = stats[base + 8u];
  if (!(income > 0.0)) { return; }
  if (term == 1u) { stats[base + 1u] = 1.0 - total; }
  if (term == 2u) { stats[base + 2u] = getScaledQuotient(total, population, 1.0, 0); }
  if (term == 3u) { if (isFiniteBits(stats[base + 3u])) { stats[base + 3u] = -getScaledQuotient(total, population, 1.0, 0); } }
  if (term == 4u) {
    let power = 1.0 - params[paramsOffset]; var atkinson = getNaN();
    if (power == 0.0) { atkinson = 1.0 - exp(-stats[base + 3u]); }
    else if (power > 0.0 || isFiniteBits(stats[base + 3u])) { atkinson = 1.0 - pow(getScaledQuotient(total, population, 1.0, 0), 1.0 / power); }
    stats[base + 4u] = select(getNaN(), atkinson, isFiniteBits(atkinson));
  }
  if (term == 5u) { stats[base + 5u] = getScaledQuotient(total, income, 2.0, 0); }`
      })
    );
  }
  nodes.push(
    createGraphChunkGroupKernelNode(graph, {
      id: `${id}-lorenz-cuts`,
      operation: 'GPUInequality',
      variant: 'lorenz-cuts',
      invocationCount: zoneCount * knotCount,
      parameterValues: [knotCount],
      bindings: [
        read('zoneOffsets', offsets, 'u32'),
        read('prefixWeights', prefix.weights, 'f32'),
        read('prefixIncomes', prefix.incomes, 'f32'),
        read('params', parameters, 'f32'),
        write('stats', stats, 'f32'),
        write('lorenz', lorenz, 'f32')
      ],
      declarations: `${declarations}
fn readLorenzCut(zone: u32, cut: f32) -> f32 {
  let begin = zoneOffsets[zoneOffsetsOffset + zone]; let end = zoneOffsets[zoneOffsetsOffset + zone + 1u];
  let population = stats[statsOffset + zone * STATS_STRIDE + 7u]; let income = stats[statsOffset + zone * STATS_STRIDE + 8u];
  if (begin == end || !(income > 0.0)) { return getNaN(); }
  if (cut <= 0.0) { return 0.0; } if (cut >= 1.0) { return 1.0; }
  let threshold = cut * population; var low = begin; var high = end;
  while (low < high) { let middle = low + (high - low) / 2u;
    if (prefixWeights[prefixWeightsOffset + middle] < threshold) { low = middle + 1u; } else { high = middle; }
  }
  let position = min(low, end - 1u); var previousPopulation = 0.0; var previousIncome = 0.0;
  if (position > begin) { previousPopulation = prefixWeights[prefixWeightsOffset + position - 1u]; previousIncome = prefixIncomes[prefixIncomesOffset + position - 1u]; }
  let nextPopulation = prefixWeights[prefixWeightsOffset + position]; let nextIncome = prefixIncomes[prefixIncomesOffset + position];
  let fraction = select(1.0, clamp(getScaledQuotient(threshold - previousPopulation, nextPopulation - previousPopulation, 1.0, 0), 0.0, 1.0), nextPopulation > previousPopulation);
  return getScaledQuotient(previousIncome + fraction * (nextIncome - previousIncome), income, 1.0, 0);
}`,
      body: `let knotCount = getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET}u);
  let zone = index / knotCount; let knot = index % knotCount;
  lorenz[lorenzOffset + index] = readLorenzCut(zone, f32(knot) / f32(knotCount - 1u));
  if (knot == 0u) {
    let bottom = readLorenzCut(zone, params[paramsOffset + 2u]);
    let top = readLorenzCut(zone, 1.0 - params[paramsOffset + 1u]);
    stats[statsOffset + zone * STATS_STRIDE + 6u] = select(getNaN(), getScaledQuotient(1.0 - top, bottom, 1.0, 0), bottom > 0.0);
  }`
    })
  );
  return nodes;
}
