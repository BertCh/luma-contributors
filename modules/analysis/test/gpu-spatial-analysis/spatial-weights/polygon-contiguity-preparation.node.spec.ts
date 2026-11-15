// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {computePolygonContiguityOnCPU} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {makeExactPolygonTopology} from './polygon-contiguity-preparation-fixture';

test('CPU route shares an exact whole edge but refuses a binary64 gap hidden by Float32', () => {
  const topology = makeExactPolygonTopology();
  expect(topology.equality).toBe('exact-source-xy');
  expect(topology.topologyEncoding).toBe('source-xy-key-ids');
  expect([...topology.geometryMask]).toEqual([1, 1, 1]);
  for (const criterion of ['queen', 'rook'] as const) {
    const result = computePolygonContiguityOnCPU(topology, {criterion, capacity: 2});
    expect([...result.offsets]).toEqual([0, 1, 2, 2]);
    expect([...result.neighbors]).toEqual([1, 0]);
    expect([...result.weights]).toEqual([1, 1]);
    expect([...result.requiredCount]).toEqual([2]);
    expect([...result.overflow]).toEqual([0]);
    expect([...result.candidateOverflow]).toEqual([0]);
    const truncated = computePolygonContiguityOnCPU(topology, {criterion, capacity: 1});
    expect([...truncated.offsets]).toEqual([0, 1, 1, 1]);
    expect([...truncated.requiredCount]).toEqual([2]);
    expect([...truncated.overflow]).toEqual([1]);
  }
});

test('all empty source rows stay aligned without passing degenerate offset arrays to the twin', () => {
  const topology = makeExactPolygonTopology(true);
  const result = computePolygonContiguityOnCPU(topology, {criterion: 'rook', capacity: 0});
  expect([...topology.geometryMask]).toEqual([0, 0, 0]);
  expect([...result.offsets]).toEqual([0, 0, 0, 0]);
  expect([...result.requiredCount]).toEqual([0]);
  expect([...result.overflow]).toEqual([0]);
});
