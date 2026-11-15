// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeKShortestPathsOnCPU,
  CPUKShortestPathsOperation,
  GPUKShortestPathsOperation
} from '../../../src/gpu-network/k-shortest-paths/index';
import {getFixtureInput} from './k-shortest-paths-test-utils';
import reference from '../../oracles/fixtures/k-shortest-paths.json';

for (const fixture of reference.cases) {
  for (const routeCount of [1, 4, 16]) {
    it(`CPU Yen matches NetworkX shortest_simple_paths: ${fixture.name}, k=${routeCount}`, () => {
      const pathLength = fixture.offsets.length;
      const input = getFixtureInput(fixture);
      const result = computeKShortestPathsOnCPU(input, {routeCount, maximumPathLength: pathLength});
      const count = Math.min(routeCount, fixture.routes.length);
      expect(Array.from(result.counts)).toEqual([count]);
      expect(Array.from(result.costs.slice(0, count))).toEqual(
        fixture.routes.slice(0, count).map(route => route.cost)
      );
      expect(Array.from(result.converged)).toEqual([1]);
      expect(Array.from(result.iterationLimitReached)).toEqual([0]);
      expect(Array.from(result.candidateOverflow)).toEqual([0]);
      const keys = new Set<string>();
      for (let route = 0; route < routeCount; route++) {
        const length = result.lengths[route];
        const vertices = Array.from(
          result.vertices.slice(route * pathLength, route * pathLength + length)
        );
        const edges = Array.from(
          result.edges.slice(route * pathLength, route * pathLength + Math.max(0, length - 1))
        );
        if (route < count) {
          expect(new Set(vertices).size).toBe(length);
          expect(vertices[0]).toBe(fixture.source);
          expect(vertices[length - 1]).toBe(fixture.target);
          for (const [slot, edge] of edges.entries()) {
            expect(edge).toBeGreaterThanOrEqual(fixture.offsets[vertices[slot]]);
            expect(edge).toBeLessThan(fixture.offsets[vertices[slot] + 1]);
            expect(fixture.neighbors[edge]).toBe(vertices[slot + 1]);
          }
          expect(edges.reduce((cost, edge) => cost + fixture.weights[edge], 0)).toBe(
            result.costs[route]
          );
          expect(keys.has(edges.join(','))).toBe(false);
          keys.add(edges.join(','));
        } else {
          expect(length).toBe(0);
          expect(result.costs[route]).toBe(Infinity);
        }
        expect(
          Array.from(
            result.vertices.slice(route * pathLength + length, (route + 1) * pathLength)
          ).every(value => value === 0xffffffff)
        ).toBe(true);
        expect(
          Array.from(
            result.edges.slice(
              route * pathLength + Math.max(0, length - 1),
              (route + 1) * pathLength
            )
          ).every(value => value === 0xffffffff)
        ).toBe(true);
      }
    });
  }
}

it('retains certified routes when path or candidate budgets stop Yen', () => {
  const input = {
    topology: {
      offsets: Uint32Array.of(0, 2, 3, 4, 4),
      neighbors: Uint32Array.of(3, 1, 2, 3),
      weights: Float32Array.of(1, 1, 1, 1)
    },
    sources: Uint32Array.of(0),
    targets: Uint32Array.of(3)
  };
  const result = computeKShortestPathsOnCPU(input, {routeCount: 4, maximumPathLength: 3});
  expect(Array.from(result.counts)).toEqual([1]);
  expect(Array.from(result.lengths)).toEqual([2, 0, 0, 0]);
  expect(result.iterationLimitReached[0]).toBe(1);
  expect(result.converged[0]).toBe(0);
  expect(result.candidateOverflow[0]).toBe(0);
  const branching = getFixtureInput(reference.cases[1]);
  const overflow = computeKShortestPathsOnCPU(branching, {
    routeCount: 4,
    maximumPathLength: 8,
    candidateCapacity: 1
  });
  expect(overflow.counts[0]).toBe(1);
  expect(overflow.candidateOverflow[0]).toBe(1);
  expect(overflow.iterationLimitReached[0]).toBe(0);
  expect(overflow.converged[0]).toBe(0);
});

it('ignores invalid endpoints, padded slots, invalid neighbors and impassable weights', () => {
  const input = {
    topology: {
      offsets: Uint32Array.of(0, 6, 6),
      neighbors: Uint32Array.of(1, 1, 1, 1, 1, 99, 0),
      weights: Float32Array.of(-1, NaN, Infinity, 2, 3, 0, 0)
    },
    sources: Uint32Array.of(0, 0, 99),
    targets: Uint32Array.of(1, 0, 1)
  };
  const result = computeKShortestPathsOnCPU(input, {routeCount: 4, maximumPathLength: 3});
  expect(Array.from(result.counts)).toEqual([2, 1, 0]);
  expect(Array.from(result.costs.slice(0, 4))).toEqual([2, 3, Infinity, Infinity]);
  expect(result.converged[0]).toBe(1);
});

it('validates CPU arrays and construction props, and declares identical twin ports', () => {
  const input = getFixtureInput(reference.cases[0]);
  expect(() =>
    computeKShortestPathsOnCPU({...input, targets: new Uint32Array(0)}, {routeCount: 2})
  ).toThrow(/targets/);
  expect(() =>
    computeKShortestPathsOnCPU(
      {...input, topology: {...input.topology, offsets: Uint32Array.of(0, 3, 2)}},
      {routeCount: 2}
    )
  ).toThrow(/topology.offsets/);
  expect(() =>
    computeKShortestPathsOnCPU(
      {...input, topology: {...input.topology, weights: new Float32Array(0)}},
      {routeCount: 2}
    )
  ).toThrow(/topology.weights/);
  const props = {
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
      status: {
        converged: 'converged',
        iterationLimitReached: 'iterationLimitReached',
        candidateOverflow: 'candidateOverflow',
        iterationCount: 'iterationCount'
      }
    }
  };
  const gpu = new GPUKShortestPathsOperation(props);
  const cpu = new CPUKShortestPathsOperation(props);
  expect(cpu.inputs).toEqual(gpu.inputs);
  const positionsProps = {...props, topology: {...props.topology, nodePositions: 'positions'}};
  expect(new GPUKShortestPathsOperation(positionsProps).inputs['nodePositions']).toBe('positions');
  expect(new CPUKShortestPathsOperation(positionsProps).inputs).toEqual(
    new GPUKShortestPathsOperation(positionsProps).inputs
  );
  expect(cpu.contract.outputs).toEqual(gpu.contract.outputs);
  expect(gpu.outputs['counts']).toEqual({
    value: 'counts',
    format: 'uint32',
    length: props.queryCount
  });
  expect(gpu.outputs).not.toHaveProperty('count');
  expect(cpu.contract.outputs).toHaveProperty('counts');
  expect(cpu.contract.outputs).not.toHaveProperty('count');
  expect(() => new CPUKShortestPathsOperation({...props, candidateCapacity: 0})).toThrow(
    /k-shortest-paths candidateCapacity/
  );
});
