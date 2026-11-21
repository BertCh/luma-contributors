// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeSubgraphOracle, resolveSubgraphOracleState} from './network-subgraph-filter-oracle';
import {getGPUNetworkSubgraphFilterParameterValues as pack} from '../../../src/gpu-network/network-subgraph-filter/network-subgraph-filter-parameters';

const layout = {nodeColumnCount: 1, edgeColumnCount: 1, hasEdgeTimes: true};

it('decimal bounds preserve half-open attribute and closed time membership (PCK-04)', () => {
  // Exact binary32 neighbors: 0x3f333333 < 0.7 < 0x3f333334;
  // 0x3dcccccc < 0.1 < 0x3dcccccd. Expectations come from those ordered bit patterns.
  const packed = pack(layout, {
    nodeAttributeRanges: [[0, 0.7]],
    edgeAttributeRanges: [[0.1, Infinity]],
    closedEdgeTimeWindow: [0, 0.1]
  });
  expect(packed[1]).toBe(0.7000000476837158);
  expect(0.699999988079071 >= packed[0] && 0.699999988079071 < packed[1]).toBe(true);
  expect(packed[4]).toBe(0.10000000149011612);
  expect(packed[5]).toBe(Infinity);
  expect(packed[9]).toBe(0.09999999403953552);
  expect(0.10000000149011612 >= packed[8] && 0.10000000149011612 <= packed[9]).toBe(false);
  const negative = pack(layout, {
    nodeAttributeRanges: [[-0.7, -0.1]],
    closedEdgeTimeWindow: [-0.7, -0.1]
  });
  expect(Array.from(negative.slice(0, 2))).toEqual([-0.699999988079071, -0.09999999403953552]);
  expect(Array.from(negative.slice(8, 10))).toEqual([-0.699999988079071, -0.10000000149011612]);
});

it('rejects each retired range key before packing predicates', () => {
  for (const [removed, replacement] of [
    ['nodeRanges', 'nodeAttributeRanges'],
    ['edgeRanges', 'edgeAttributeRanges'],
    ['edgeTimeWindow', 'closedEdgeTimeWindow']
  ]) {
    expect(() => pack(layout, {[removed]: [0, 1]})).toThrow(
      new RegExp(removed + ' was removed; use ' + replacement)
    );
  }
  expect(() => pack(layout, {nodeAttributeRanges: [[NaN, 1]]})).toThrow(/nodeAttributeRanges/);
  expect(
    pack(layout, {nodeAttributeRanges: [[0, 0]], closedEdgeTimeWindow: [-Infinity, Infinity]})[9]
  ).toBe(Infinity);
});

it('subgraph fixture oracle honors new state spellings and mathematical bounds (S1)', () => {
  const result = computeSubgraphOracle({
    nodeCount: 2,
    offsets: Uint32Array.of(0, 1, 1),
    neighbors: Uint32Array.of(1),
    directed: true,
    vertexColumns: [Float32Array.of(0.699999988079071, 0.699999988079071)],
    edgeColumns: [Float32Array.of(0.5)],
    edgeTimes: Float32Array.of(0.10000000149011612),
    ...resolveSubgraphOracleState({
      nodeAttributeRanges: [[0, 0.7]],
      edgeAttributeRanges: [[0, 1]],
      closedEdgeTimeWindow: [0, 0.1]
    })
  });
  // These representable nodes are below 0.7; the edge time is above 0.1.
  expect(Array.from(result.vertexMask)).toEqual([1, 1]);
  expect(Array.from(result.edgeMask)).toEqual([0]);
  expect(
    resolveSubgraphOracleState({
      nodeAttributeRanges: [[0, 1]],
      edgeAttributeRanges: [[0, 2]],
      closedEdgeTimeWindow: [0, 3]
    })
  ).toEqual({vertexRanges: [[0, 1]], edgeRanges: [[0, 2]], edgeTimeWindow: [0, 3]});
});
