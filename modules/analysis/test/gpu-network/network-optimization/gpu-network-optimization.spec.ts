// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GPUNetworkFlow} from '../../../src/gpu-network/network-optimization/gpu-network-flow';
import {GPUNetworkLouvain} from '../../../src/gpu-network/network-optimization/gpu-network-louvain';
import {GPUNetworkTimeRouting} from '../../../src/gpu-network/network-optimization/gpu-network-time-routing';
import {GPUNetworkPointRouting} from '../../../src/gpu-network/network-optimization/gpu-network-point-routing';
import {GPUNetworkFacilityOptimization} from '../../../src/gpu-network/network-optimization/gpu-network-facility-optimization';
import {GPUNetworkRationalAccess} from '../../../src/gpu-network/network-optimization/gpu-network-rational-access';
import {createOptimizationHarness} from './optimization-test-harness';
import fixture from '../../oracles/fixtures/network-movement-coverage.json';

it('GPUNetworkFlow recomputes mutable capacities without retaining previous flow or certificates', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    flow: harness.output('uint32', 1),
    cut: harness.output('uint32', 2),
    summary: harness.output('uint32', 8)
  };
  const capacities = harness.input('uint32', Uint32Array.of(3));
  const result = await harness.run(
    new GPUNetworkFlow({
      nodeCount: 2,
      edges: harness.input('uint32x2', Uint32Array.of(0, 1)),
      capacities,
      parameters: harness.input('uint32', Uint32Array.of(0, 1, 0xffffffff)),
      output
    }),
    first => {
      expect(first(output.flow)).toEqual([3]);
      harness.rewrite(capacities, Uint32Array.of(0));
    }
  );
  expect(result(output.flow)).toEqual([0]);
  expect(result(output.summary).slice(0, 3)).toEqual([1, 1, 0]);
  expect(result(output.summary)[7]).toBe(1);
});

it('GPUNetworkFacilityOptimization distinguishes exhausted enumeration from infeasibility and implements backup/dispersion objectives', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const operation of ['backup-coverage', 'p-dispersion'] as const) {
    const harness = createOptimizationHarness(device);
    const output = {
      selectedFacilities: harness.output('uint32', 3),
      assignments: harness.output('uint32', 3),
      objective: harness.output('float32', 1),
      summary: harness.output('uint32', 4)
    };
    const result = await harness.run(
      new GPUNetworkFacilityOptimization({
        operation,
        facilityCount: 3,
        selectCount: operation === 'p-dispersion' ? 2 : undefined,
        costs: harness.input(
          'float32',
          new Float32Array(
            operation === 'p-dispersion' ? [0, 1, 5, 1, 0, 4, 5, 4, 0] : [0, 2, 0, 0, 0, 2, 2, 0, 0]
          )
        ),
        demand: harness.input('float32', Float32Array.of(1, 1, 1)),
        capacities: harness.input('float32', Float32Array.of(3, 3, 3)),
        threshold: harness.input('float32', Float32Array.of(1)),
        output
      })
    );
    expect(result(output.summary).slice(0, 3)).toEqual([1, 1, 1]);
    expect(result(output.objective)).toEqual([operation === 'p-dispersion' ? 5 : 1]);
    expect(result(output.selectedFacilities).reduce((sum, selected) => sum + selected, 0)).toBe(2);
  }
  const harness = createOptimizationHarness(device);
  const output = {
    selectedFacilities: harness.output('uint32', 1),
    assignments: harness.output('uint32', 1),
    objective: harness.output('float32', 1),
    summary: harness.output('uint32', 4)
  };
  const result = await harness.run(
    new GPUNetworkFacilityOptimization({
      operation: 'backup-coverage',
      facilityCount: 1,
      maximumCandidates: 0,
      costs: harness.input('float32', Float32Array.of(0)),
      demand: harness.input('float32', Float32Array.of(1)),
      capacities: harness.input('float32', Float32Array.of(1)),
      threshold: harness.input('float32', Float32Array.of(1)),
      output
    })
  );
  expect(result(output.summary)).toEqual([1, 0, 0, 0]);
  expect(result(output.objective)[0]).toBeNaN();
});

it('GPUNetworkLouvain matches the pinned NetworkX self-loop objective', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const reference = fixture.communities[1];
  const harness = createOptimizationHarness(device);
  const output = {
    communities: harness.output('uint32', 4),
    objectiveTrace: harness.output('float32', 129),
    summary: harness.output('uint32', 4)
  };
  const result = await harness.run(
    new GPUNetworkLouvain({
      nodeCount: 4,
      weights: harness.input('float32', Float32Array.from(reference.weights)),
      parameters: harness.input('float32', Float32Array.of(1, 0)),
      output
    })
  );
  const trace = result(output.objectiveTrace).filter(Number.isFinite);
  expect(trace.at(-1)).toBeCloseTo(reference.modularity, 5);
  expect(result(output.summary).slice(0, 2)).toEqual([1, 1]);
});

it('GPUNetworkFlow matches pinned NetworkX 3.5 max-flow/min-cost cases and independent conservation checks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const entry of fixture.flow) {
    const harness = createOptimizationHarness(device);
    const output = {
      flow: harness.output('uint32', entry.edges.length),
      cut: harness.output('uint32', entry.nodeCount),
      summary: harness.output('uint32', 8)
    };
    const result = await harness.run(
      new GPUNetworkFlow({
        nodeCount: entry.nodeCount,
        edges: harness.input('uint32x2', Uint32Array.from(entry.edges.flat())),
        capacities: harness.input('uint32', Uint32Array.from(entry.capacities)),
        costs: harness.input('uint32', Uint32Array.from(entry.costs)),
        parameters: harness.input('uint32', new Uint32Array([0, entry.nodeCount - 1, 0xffffffff])),
        output
      })
    );
    const flow = result(output.flow);
    const summary = result(output.summary);
    const balance = new Array<number>(entry.nodeCount).fill(0);
    flow.forEach((value, edge) => {
      expect(value, entry.name).toBeLessThanOrEqual(entry.capacities[edge]);
      balance[entry.edges[edge][0]] -= value;
      balance[entry.edges[edge][1]] += value;
    });
    expect(summary.slice(0, 3), entry.name).toEqual([1, 1, entry.maximumFlow]);
    expect(summary.slice(6), entry.name).toEqual([entry.maximumFlow, 1]);
    expect(new Float32Array(Uint32Array.of(summary[5]).buffer)[0], entry.name).toBe(
      entry.minimumCost
    );
    expect(balance.slice(1, -1), entry.name).toEqual(new Array(entry.nodeCount - 2).fill(0));
  }
});

it('GPUNetworkFlow preserves conservation and certifies the residual cut, including antiparallel edges', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    flow: harness.output('uint32', 6),
    cut: harness.output('uint32', 4),
    summary: harness.output('uint32', 8)
  };
  const result = await harness.run(
    new GPUNetworkFlow({
      nodeCount: 4,
      edges: harness.input('uint32x2', new Uint32Array([0, 1, 0, 2, 1, 2, 2, 1, 1, 3, 2, 3])),
      capacities: harness.input('uint32', new Uint32Array([3, 2, 1, 1, 2, 3])),
      parameters: harness.input('uint32', new Uint32Array([0, 3, 0xffffffff])),
      output
    })
  );
  const flow = result(output.flow);
  const summary = result(output.summary);
  expect(summary.slice(0, 4)).toEqual([1, 1, 5, 0]);
  expect(summary.slice(6)).toEqual([5, 1]);
  expect(flow[0] + flow[3]).toBe(flow[2] + flow[4]);
  expect(flow[1] + flow[2]).toBe(flow[3] + flow[5]);
});

it('GPUNetworkFlow reports a feasible incomplete incumbent when its augmentation budget is zero', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    flow: harness.output('uint32', 1),
    cut: harness.output('uint32', 2),
    summary: harness.output('uint32', 8)
  };
  const result = await harness.run(
    new GPUNetworkFlow({
      nodeCount: 2,
      maximumAugmentations: 0,
      edges: harness.input('uint32x2', new Uint32Array([0, 1])),
      capacities: harness.input('uint32', new Uint32Array([3])),
      parameters: harness.input('uint32', new Uint32Array([0, 1, 2])),
      output
    })
  );
  expect(result(output.summary).slice(0, 4)).toEqual([1, 0, 0, 2]);
  expect(result(output.flow)).toEqual([0]);
});

it('GPUNetworkFlow cancels an earlier path through a reverse residual edge for minimum cost', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    flow: harness.output('uint32', 8),
    cut: harness.output('uint32', 6),
    summary: harness.output('uint32', 8)
  };
  const result = await harness.run(
    new GPUNetworkFlow({
      nodeCount: 6,
      edges: harness.input(
        'uint32x2',
        new Uint32Array([0, 1, 0, 2, 1, 3, 1, 4, 2, 3, 2, 4, 3, 5, 4, 5])
      ),
      capacities: harness.input('uint32', new Uint32Array(8).fill(1)),
      costs: harness.input('uint32', new Uint32Array([0, 0, 1, 2, 2, 100, 0, 0])),
      parameters: harness.input('uint32', new Uint32Array([0, 5, 2])),
      output
    })
  );
  expect(result(output.flow)).toEqual([1, 1, 0, 1, 1, 0, 1, 1]);
  expect(result(output.summary).slice(0, 4)).toEqual([1, 1, 2, 0]);
  expect(new Float32Array(Uint32Array.of(result(output.summary)[5]).buffer)[0]).toBe(4);
});

it('GPUNetworkLouvain keeps disconnected pairs together and a nondecreasing objective trace', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    communities: harness.output('uint32', 4),
    objectiveTrace: harness.output('float32', 17),
    summary: harness.output('uint32', 4)
  };
  const result = await harness.run(
    new GPUNetworkLouvain({
      nodeCount: 4,
      maximumLevels: 4,
      maximumIterations: 4,
      weights: harness.input(
        'float32',
        new Float32Array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
      ),
      parameters: harness.input('float32', new Float32Array([1, 0])),
      output
    })
  );
  const labels = result(output.communities);
  expect(labels[0]).toBe(labels[1]);
  expect(labels[2]).toBe(labels[3]);
  expect(labels[0]).not.toBe(labels[2]);
  expect(result(output.summary).slice(0, 2)).toEqual([1, 1]);
  const trace = result(output.objectiveTrace).filter(Number.isFinite);
  expect(trace.at(-1)).toBeCloseTo(0.5, 6);
  for (let index = 1; index < trace.length; index++)
    expect(trace[index]).toBeGreaterThanOrEqual(trace[index - 1]);
});

it('GPUNetworkTimeRouting evaluates costs at arrival time rather than freezing departure costs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    arrivals: harness.output('float32', 3),
    predecessorTransitions: harness.output('uint32', 3),
    summary: harness.output('uint32', 4)
  };
  const result = await harness.run(
    new GPUNetworkTimeRouting({
      stateVertices: harness.input('uint32', new Uint32Array([0, 1, 2])),
      transitions: harness.input('uint32x2', new Uint32Array([0, 1, 1, 2, 0, 2])),
      costs: harness.input('float32x2', new Float32Array([1, 0, 1, 1, 10, 0])),
      endpoints: harness.input('uint32', new Uint32Array([0, 2])),
      departure: harness.input('float32', new Float32Array([3])),
      output
    })
  );
  expect(result(output.arrivals)).toEqual([3, 4, 9]);
  expect(result(output.summary).slice(0, 3)).toEqual([1, 1, 2]);
});

it('GPUNetworkPointRouting handles fractional same-edge travel and forbidden incoming turns', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    costs: harness.output('float32', 3),
    status: harness.output('uint32', 3),
    predecessorEdges: harness.output('uint32', 9)
  };
  const result = await harness.run(
    new GPUNetworkPointRouting({
      costs: harness.input('float32', new Float32Array([10, 20, 30])),
      turns: harness.input('uint32x2', new Uint32Array([0, 1, 1, 2])),
      queries: harness.input('uint32x2', new Uint32Array([0, 0, 0, 2, 2, 0])),
      fractions: harness.input('float32x2', new Float32Array([0.2, 0.7, 0.5, 0.5, 0.2, 0.7])),
      output
    })
  );
  expect(result(output.costs)[0]).toBeCloseTo(5);
  expect(result(output.costs).slice(1)).toEqual([40, Infinity]);
  expect(result(output.status)).toEqual([0, 0, 0]);
});

it('GPUNetworkPointRouting preserves and resets same-edge cycle provenance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    costs: harness.output('float32', 3),
    status: harness.output('uint32', 3),
    predecessorEdges: harness.output('uint32', 15)
  };
  const fractions = harness.input('float32x2', Float32Array.of(0.8, 0.2, 0.8, 0.5, 0.2, 0.8));
  const costs = [10, 2, 3, 100, 1];
  const result = await harness.run(
    new GPUNetworkPointRouting({
      costs: harness.input('float32', Float32Array.from(costs)),
      turns: harness.input('uint32x2', Uint32Array.of(0, 1, 1, 2, 2, 0, 0, 3, 3, 0)),
      queries: harness.input('uint32x2', Uint32Array.of(0, 0, 0, 2, 4, 0)),
      fractions,
      output
    }),
    first => {
      expect(first(output.status)).toEqual([0, 0, 0]);
      expect(first(output.costs)[0]).toBeCloseTo(9);
      expect(first(output.costs)[1]).toBeCloseTo(5.5);
      expect(first(output.costs)[2]).toBe(Infinity);
      // Following at least one predecessor distinguishes a returning route from direct travel.
      const predecessors = first(output.predecessorEdges);
      const route = [0];
      let current = 0;
      do {
        current = predecessors[current];
        route.unshift(current);
        expect(route.length).toBeLessThanOrEqual(costs.length + 1);
      } while (current !== 0);
      expect(route).toEqual([0, 1, 2, 0]);
      const reconstructedCost =
        0.2 * costs[route[0]] +
        route.slice(1, -1).reduce((total, edge) => total + costs[edge], 0) +
        0.2 * costs[route.at(-1)!];
      expect(first(output.costs)[0]).toBeCloseTo(reconstructedCost);
      expect(predecessors[5 + 2]).toBe(1);
      expect(predecessors[10]).toBe(0xffffffff);
      // Reusing the compiled graph must clear the previous query's returning source predecessor.
      harness.rewrite(fractions, Float32Array.of(0.2, 0.8, 0.8, 0.5, 0.2, 0.8));
    }
  );
  expect(result(output.costs)[0]).toBeCloseTo(6);
  expect(result(output.predecessorEdges)[0]).toBe(0xffffffff);
});

it('GPUNetworkFacilityOptimization finds the feasible unsplit assignment that demand-order greedy misses', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    selectedFacilities: harness.output('uint32', 2),
    assignments: harness.output('uint32', 2),
    objective: harness.output('float32', 1),
    summary: harness.output('uint32', 4)
  };
  const result = await harness.run(
    new GPUNetworkFacilityOptimization({
      operation: 'capacitated-p-median',
      facilityCount: 2,
      selectCount: 2,
      costs: harness.input('float32', new Float32Array([1, 2, 1, Infinity])),
      demand: harness.input('float32', new Float32Array([1, 1])),
      capacities: harness.input('float32', new Float32Array([1, 1])),
      threshold: harness.input('float32', new Float32Array([0])),
      output
    })
  );
  expect(result(output.assignments)).toEqual([1, 0]);
  expect(result(output.objective)).toEqual([3]);
  expect(result(output.summary)).toEqual([1, 1, 1, 4]);
});

it('GPUNetworkRationalAccess balances tied facilities while preserving integral demand and unreachable rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    assignments: harness.output('float32', 4),
    scores: harness.output('float32', 2),
    unservedDemand: harness.output('float32', 2),
    summary: harness.output('uint32', 4)
  };
  const result = await harness.run(
    new GPUNetworkRationalAccess({
      costs: harness.input('float32', new Float32Array([0, 0, Infinity, Infinity])),
      demand: harness.input('float32', new Float32Array([10, 3])),
      supply: harness.input('float32', new Float32Array([5, 5])),
      parameters: harness.input('float32', new Float32Array([1, 1, 1, 1, 50, 0.0001])),
      initialLimitCycles: 0,
      maximumCycles: 4,
      output
    })
  );
  expect(result(output.assignments)).toEqual([5, 5, 0, 0]);
  expect(result(output.unservedDemand)).toEqual([0, 3]);
  expect(result(output.scores)[0]).toBeCloseTo(1);
  expect(result(output.scores)[1]).toBeNaN();
  expect(result(output.summary).slice(0, 2)).toEqual([1, 1]);
});
