// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {prepareGPUNetworkRestrictionStates} from '../../../src/gpu-network/network-optimization/gpu-network-time-routing';

it('packs restriction-state rows with their GPU formats and counts', () => {
  const prepared = prepareGPUNetworkRestrictionStates({
    nodeCount: 3,
    edges: [
      [0, 1],
      [1, 2]
    ],
    restrictions: [[0, 1]]
  });

  // States 0, 1 and 2 have empty history at their corresponding vertices.
  // Edge 0 takes state 0 to state 3 (vertex 1 with history [0]).
  // Edge 1 takes empty-history state 1 to state 2, and is forbidden from state 3.
  expect(prepared).toEqual({
    nodeCount: 3,
    stateCount: 4,
    transitionCount: 2,
    formats: {
      stateVertices: 'uint32',
      transitions: 'uint32x2',
      originalEdges: 'uint32',
      initialStates: 'uint32'
    },
    stateVertices: Uint32Array.of(0, 1, 2, 1),
    transitions: Uint32Array.of(0, 3, 1, 2),
    originalEdges: Uint32Array.of(0, 1),
    initialStates: Uint32Array.of(0, 1, 2)
  });
});

it('rejects positional arguments with the options object migration', () => {
  expect(() =>
    Reflect.apply(prepareGPUNetworkRestrictionStates, undefined, [2, [{source: 0, target: 1}], []])
  ).toThrow(/^prepareGPUNetworkRestrictionStates .*\{nodeCount, edges, restrictions\}/);
});

it('rejects object-form edges with the endpoint tuple migration', () => {
  expect(() =>
    Reflect.apply(prepareGPUNetworkRestrictionStates, undefined, [
      {nodeCount: 2, edges: [{source: 0, target: 1}], restrictions: []}
    ])
  ).toThrow(/^prepareGPUNetworkRestrictionStates .*edges\[0\].*\[source, target\]/);
});
