// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {CPUMapMatchingInput} from '../../../src/gpu-network/map-matching/cpu-map-matching-operation';
import {getGPUMapMatchingParameterValues} from '../../../src/gpu-network/map-matching/gpu-map-matching';
import type {MapMatchingOperationProps} from '../../../src/gpu-network/map-matching/gpu-map-matching-operation';

export function createMatchingOutputOracle(routeNodeBudget: number): {
  input: CPUMapMatchingInput;
  props: MapMatchingOperationProps;
} {
  // Directed chain 0->...->10 with edge lengths 2. Two tracks go from x=.5 to 19.5;
  // their fractions .25/.75 give offsets .5/1.5. Track three ends at unmatched (1,5).
  // Five of six fixes match. Each long transition drops node 9 once at budget 8:
  // breakCount=3 (two budget breaks and one unmatched fix), routeOverflow=2.
  // At budget 16 only the unmatched break remains: breakCount=1, routeOverflow=0.
  return {
    input: {
      topology: {
        offsets: Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10),
        neighbors: Uint32Array.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        nodePositions: Float32Array.of(
          0,
          0,
          2,
          0,
          4,
          0,
          6,
          0,
          8,
          0,
          10,
          0,
          12,
          0,
          14,
          0,
          16,
          0,
          18,
          0,
          20,
          0
        )
      },
      positions: Float32Array.of(0.5, 0, 19.5, 0, 0.5, 0, 19.5, 0, 0.5, 0, 1, 5),
      trackOffsets: Uint32Array.of(0, 2, 4, 6),
      parameters: getGPUMapMatchingParameterValues({
        sigma: 1,
        beta: 1,
        searchRadius: 0.1,
        routeFactor: 2,
        routeSlack: 2
      })
    },
    props: {
      nodeCount: 11,
      edgeCount: 10,
      pointCount: 6,
      trackCount: 3,
      topology: {offsets: 'offsets', neighbors: 'neighbors', nodePositions: 'nodePositions'},
      positions: 'positions',
      trackOffsets: 'trackOffsets',
      parameters: 'parameters',
      candidateCount: 1,
      routeNodeBudget,
      cellSize: 2,
      bounds: {minimum: [-1, -1], maximum: [21, 6]},
      output: {matchedEdges: 'matchedEdges'}
    }
  };
}

export function getMatchingOutputExpected(routeNodeBudget: number) {
  return {
    matchedEdges: [0, 9, 0, 9, 0, 4294967295],
    matchedOffsets: [0.5, 1.5, 0.5, 1.5, 0.5, -1],
    matchedCount: [5],
    breakCount: [routeNodeBudget === 8 ? 3 : 1],
    routeOverflow: [routeNodeBudget === 8 ? 2 : 0]
  };
}
