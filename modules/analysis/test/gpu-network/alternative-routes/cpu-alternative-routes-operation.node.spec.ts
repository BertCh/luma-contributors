// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  CPUAlternativeRoutesOperation,
  computeAlternativeRoutesOnCPU
} from '../../../src/gpu-network/alternative-routes/cpu-alternative-routes-operation';
import {
  GPUAlternativeRoutesOperation,
  type AlternativeRoutesOperationProps
} from '../../../src/gpu-network/alternative-routes/gpu-alternative-routes-operation';
import {
  enumerateAlternativeRoutes,
  getAlternativeRoutesInput
} from './alternative-routes-operation-test-utils';

for (const routeCount of [1, 4, 8])
  it(`matches exhaustive enumeration and sorting for top ${routeCount} edge paths`, () => {
    const input = getAlternativeRoutesInput();
    const result = computeAlternativeRoutesOnCPU(input, {routeCount});
    expect(result).toEqual(enumerateAlternativeRoutes(input, routeCount));
    expect(Array.from(result.edges.subarray(0, 5))).toEqual([0, 6, 9, 10, 0xffffffff]);
    if (routeCount === 8) expect(Array.from(result.counts)).toEqual([7, 2]);
    // No path may repeat a vertex, even on the zero-weight cycle.
    for (let route = 0; route < result.lengths.length; route++) {
      const path = result.vertices.subarray(route * 5, route * 5 + result.lengths[route]);
      expect(new Set(path).size).toBe(path.length);
    }
  });

it('certifies trivial/invalid endpoints and drops GPU route-count limits', () => {
  const original = getAlternativeRoutesInput();
  const input = {...original, sources: Uint32Array.of(0, 99, 4), targets: Uint32Array.of(0, 4, 4)};
  const result = computeAlternativeRoutesOnCPU(input, {routeCount: 33});
  expect(result).toEqual(enumerateAlternativeRoutes(input, 33));
  expect(Array.from(result.counts)).toEqual([1, 0, 1]);
  expect(Array.from(result.complete)).toEqual([1, 1, 1]);
});

it('rejects malformed arrays and validates both operation output shapes at construction', () => {
  const input = getAlternativeRoutesInput();
  expect(() =>
    computeAlternativeRoutesOnCPU({...input, targets: Uint32Array.of(1)}, {routeCount: 4})
  ).toThrow(/targets/);
  expect(() =>
    computeAlternativeRoutesOnCPU(
      {...input, topology: {...input.topology, weights: new Float32Array(1)}},
      {routeCount: 4}
    )
  ).toThrow(/weights/);
  const props: AlternativeRoutesOperationProps = {
    id: 'check-alternatives',
    nodeCount: 5,
    queryCount: 2,
    routeCount: 4,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    sources: 'sources',
    targets: 'targets',
    output: {
      vertices: 'vertices',
      edges: 'edges',
      lengths: 'lengths',
      costs: 'costs',
      counts: 'counts',
      complete: 'complete'
    }
  };
  for (const Operation of [CPUAlternativeRoutesOperation, GPUAlternativeRoutesOperation])
    expect(() => new Operation({...props, routeCount: 0})).toThrow(/^check-alternatives/);
  expect(
    () =>
      new CPUAlternativeRoutesOperation({
        ...props,
        nodeCount: 257,
        routeCount: 33,
        maximumExpansions: 1,
        maximumFrontierSize: 1
      })
  ).not.toThrow();
});
