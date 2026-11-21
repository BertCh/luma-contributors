// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeLineSimplificationOnCPU} from '../../../src/gpu-spatial-analysis/line-simplification/cpu-line-simplification-operation';
import {getGPULineSimplificationParameterValues} from '../../../src/gpu-spatial-analysis/line-simplification/line-simplification-parameters';

it.each([0, 2])('CPU endpoint selection has independent cardinality at capacity %s', capacity => {
  // Two endpoints are mandatory; there are no interior decisions. One terminal
  // round certifies completion. Zero slots changes written count, not requirements.
  const result = computeLineSimplificationOnCPU(
    {
      positions: Float32Array.of(0, 0, 1, 0),
      lineOffsets: Uint32Array.of(0, 2),
      parameters: getGPULineSimplificationParameterValues({tolerance: 0})
    },
    {capacity, maximumIterations: 1, finishSpanLimit: 1}
  );
  expect(Array.from(result.count)).toEqual([capacity]);
  expect(Array.from(result.requiredCount)).toEqual([2]);
  expect(Array.from(result.lineCounts)).toEqual([2]);
  expect(Array.from(result.overflow)).toEqual([Number(capacity === 0)]);
  expect(Array.from(result.converged)).toEqual([1]);
  expect(Array.from(result.iterationCount)).toEqual([1]);
});

it.each([
  'douglas-peucker',
  'visvalingam'
] as const)('CPU %s has named unresolved work after one round', method => {
  // DP root chord y=0 ties rows 1/3 at distance 1, so row 1 wins. Child
  // interval [1,4] still contains rows 2/3, beyond finishSpanLimit=1.
  // Visvalingam starts with two tied area-1/2 interiors; removing row 1
  // leaves row 2 for a later round. Both fixtures require more than one round.
  const positions =
    method === 'douglas-peucker'
      ? Float32Array.of(0, 0, 1, 1, 2, 0, 3, 1, 4, 0)
      : Float32Array.of(0, 0, 1, 1, 2, 1, 3, 0);
  const result = computeLineSimplificationOnCPU(
    {positions, lineOffsets: Uint32Array.of(0, positions.length / 2)},
    {method, maximumIterations: 1, finishSpanLimit: 1}
  );
  expect(Array.from(result.converged)).toEqual([0]);
  expect(Array.from(result.iterationCount)).toEqual([1]);
});

it('CPU Visvalingam stops after two removals within a three-round budget', () => {
  // Interiors initially tie at area 1/2; row 1 wins. The remaining row 2
  // has triangle (0,0),(2,1),(3,0), area 3/2, and is the second/final removal.
  const result = computeLineSimplificationOnCPU(
    {
      positions: Float32Array.of(0, 0, 1, 1, 2, 1, 3, 0),
      lineOffsets: Uint32Array.of(0, 4)
    },
    {method: 'visvalingam', maximumIterations: 3}
  );
  expect(Array.from(result.importance)).toEqual([Infinity, 0.5, 1.5, Infinity]);
  expect(Array.from(result.converged)).toEqual([1]);
  expect(Array.from(result.iterationCount)).toEqual([2]);
});
