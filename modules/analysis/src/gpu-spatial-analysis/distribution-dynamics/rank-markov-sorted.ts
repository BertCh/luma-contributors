// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createLoadBalancedSegmentSumNodes} from '../../analysis-core/kernels/load-balanced-for-each';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import type {GPURankMarkovProps} from './gpu-rank-markov';

/** Stable value sort followed by a stable period sort; no shader varies by period. @internal */
export function createSortedRankNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPURankMarkovProps,
  valid: GraphDataView<'uint32'>
): GPUCommandNode<Parameters>[] {
  const length = props.values.length;
  const rows = length / props.periodCount;
  const keys = createTransientView(graph, `${id}-value-keys`, 'uint32', length);
  const indices = createTransientView(graph, `${id}-indices`, 'uint32', length);
  const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', length);
  const byValue = createTransientView(graph, `${id}-by-value`, 'uint32', length);
  const periodKeys = createTransientView(graph, `${id}-period-keys`, 'uint32', length);
  const byPeriod = createTransientView(graph, `${id}-by-period`, 'uint32', length);
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-rank-keys`,
      operation: 'GPURankMarkov',
      variant: 'rank-keys',
      invocationCount: length,
      bindings: [
        {name: 'values', view: props.values, type: 'f32', access: 'read'},
        {name: 'valid', view: valid, type: 'u32', access: 'read'},
        {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
        {name: 'indices', view: indices, type: 'u32', access: 'read_write'}
      ],
      body: `let base = index / ${rows}u * ${rows}u;
let row = ${props.tieOrder === 'row-id' ? `index % ${rows}u` : `${rows - 1}u - index % ${rows}u`};
let value = values[valuesOffset + base + row];
// Pairwise comparisons treat -0 and +0 as equal: use one key for both.
let bits = select(bitcast<u32>(value), 0u, value == 0.0);
let ascending = select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
keys[keysOffset + index] = select(0xffffffffu, ~ascending, valid[validOffset + row] != 0u);
indices[indicesOffset + index] = base + row;`
    }),
    ...new GPUSort({
      id: `${id}-value-sort`,
      keys,
      values: indices,
      outputKeys: sortedKeys,
      outputValues: byValue,
      algorithm: 'radix'
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-period-keys`,
      operation: 'GPURankMarkov',
      variant: 'period-keys',
      invocationCount: length,
      bindings: [
        {name: 'indices', view: byValue, type: 'u32', access: 'read'},
        {name: 'keys', view: periodKeys, type: 'u32', access: 'read_write'}
      ],
      body: `keys[keysOffset + index] = indices[indicesOffset + index] / ${rows}u;`
    }),
    ...new GPUSort({
      id: `${id}-period-sort`,
      keys: periodKeys,
      values: byValue,
      outputKeys: sortedKeys,
      outputValues: byPeriod,
      algorithm: 'radix',
      keyBits: Math.max(1, Math.ceil(Math.log2(props.periodCount)))
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-rank-scatter`,
      operation: 'GPURankMarkov',
      variant: 'rank-scatter',
      invocationCount: length,
      bindings: [
        {name: 'indices', view: byPeriod, type: 'u32', access: 'read'},
        {name: 'valid', view: valid, type: 'u32', access: 'read'},
        {name: 'ranks', view: props.output.ranks, type: 'u32', access: 'read_write'}
      ],
      body: `let original = indices[indicesOffset + index];
ranks[ranksOffset + original] = select(0xffffffffu, index % ${rows}u, valid[validOffset + original % ${rows}u] != 0u);`
    })
  ];
}

/** Adjacent-period displacement and population sums use fixed workgroup trees. @internal */
export function createRankMobilityNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPURankMarkovProps
): GPUCommandNode<Parameters>[] {
  if (!props.output.mobility) return [];
  const rows = props.values.length / props.periodCount;
  const periods = props.periodCount - 1;
  const offsets = createTransientView(graph, `${id}-mobility-offsets`, 'uint32', periods + 1);
  const totals = createTransientView(graph, `${id}-mobility-totals`, 'uint32', periods * 2);
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-mobility-offsets`,
      operation: 'GPURankMarkov',
      variant: 'mobility-offsets',
      invocationCount: periods + 1,
      bindings: [{name: 'offsets', view: offsets, type: 'u32', access: 'read_write'}],
      body: `offsets[offsetsOffset + index] = index * ${rows}u;`
    }),
    ...createLoadBalancedSegmentSumNodes(graph, {
      id: `${id}-mobility-reduce`,
      operation: 'GPURankMarkov',
      offsets,
      itemCapacity: rows * periods,
      accumulatorType: 'vec2<u32>',
      output: totals,
      bindings: [{name: 'ranks', view: props.output.ranks, type: 'u32', access: 'read'}],
      // At most 65,535 ranks, each displacement < rowCount: every subtotal fits uint32.
      declarations: `fn getLoadBalancedContribution(period: u32, item: u32, lane: u32) -> vec2<u32> {
  let first = ranks[ranksOffset + item];
  let second = ranks[ranksOffset + item + ${rows}u];
  let included = first < ${rows}u && second < ${rows}u;
  return select(vec2<u32>(0u), vec2<u32>(max(first, second) - min(first, second), 1u), included);
}`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-mobility`,
      operation: 'GPURankMarkov',
      variant: 'mobility',
      invocationCount: periods,
      bindings: [
        {name: 'totals', view: totals, type: 'u32', access: 'read'},
        {name: 'mobility', view: props.output.mobility, type: 'f32', access: 'read_write'}
      ],
      declarations: `${GET_NAN_WGSL}\n${RANK_MOBILITY_MEAN_WGSL}`,
      body: `let population = totals[totalsOffset + index * 2u + 1u];
mobility[mobilityOffset + index] = getRankMobilityMean(totals[totalsOffset + index * 2u], population);`
    })
  ];
}

/** Round the integer displacement/population ratio once, as the CPU oracle does.
 * Metal may turn f32 division into reciprocal multiplication, changing the final ULP.
 * Integer long division retains the remainder through nearest-even significand rounding.
 * Population <= 65,535 bounds doubled remainders and the loop (at most 39 steps). */
const RANK_MOBILITY_MEAN_WGSL = /* wgsl */ `
fn getRankMobilityMean(displacement: u32, population: u32) -> f32 {
  if (population < 2u) { return getNaN(); }
  if (displacement == 0u) { return 0.0; }
  var significand = displacement / population;
  var remainder = displacement % population;
  var exponent = 150u;
  while (significand < 0x800000u) {
    remainder *= 2u;
    significand = significand * 2u + remainder / population;
    remainder %= population;
    exponent -= 1u;
  }
  let doubledRemainder = remainder * 2u;
  if (doubledRemainder > population ||
      (doubledRemainder == population && (significand & 1u) != 0u)) {
    significand += 1u;
  }
  if (significand == 0x1000000u) { significand >>= 1u; exponent += 1u; }
  return bitcast<f32>((exponent << 23u) | (significand & 0x7fffffu));
}`;
