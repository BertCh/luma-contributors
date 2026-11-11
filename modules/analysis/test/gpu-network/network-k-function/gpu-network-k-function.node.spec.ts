// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUNetworkKFunctionParameterValues,
  GPUNetworkKFunction
} from '../../../src/gpu-network/network-k-function/index';
import {GPU_NETWORK_SNAPPING_MAXIMUM_UNINDEXED_WORK} from '../../../src/gpu-network/network-accessibility/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Record<string, unknown> = {},
  output: Record<string, unknown> = {},
  topology: Record<string, unknown> = {}
) {
  serial++;
  return {
    id: 'k',
    positions: createTransientView(graph, `points-${serial}`, 'float32x2', 10),
    topology: {
      offsets: createTransientView(graph, `offsets-${serial}`, 'uint32', 7),
      neighbors: createTransientView(graph, `neighbors-${serial}`, 'uint32', 12),
      weights: createTransientView(graph, `weights-${serial}`, 'float32', 12),
      nodePositions: createTransientView(graph, `positions-${serial}`, 'float32x2', 6),
      ...topology
    },
    maximumDistance: createTransientView(graph, `max-distance-${serial}`, 'float32', 1),
    networkLength: createTransientView(graph, `network-length-${serial}`, 'float32', 1),
    parameters: createTransientView(graph, `parameters-${serial}`, 'uint32', 4),
    bandCount: 8,
    simulationCount: 3,
    rowsPerBlock: 15,
    output: {kValues: createTransientView(graph, `k-values-${serial}`, 'float32', 32), ...output},
    ...overrides
  };
}

it('GPUNetworkKFunction schedules snapping, simulation, one search block per row block and outputs', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const nodes = new GPUNetworkKFunction({
    ...createProps(
      graph,
      {maximumIterations: 2},
      {
        pairCounts: createTransientView(graph, 'pair-counts', 'uint32', 32),
        envelope: createTransientView(graph, 'envelope', 'float32', 24),
        status: {
          converged: createTransientView(graph, 'converged', 'uint32', 1)
        }
      }
    )
  }).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  // 40 rows in blocks of 15 rows: three blocks (15, 15, 10).
  expect(ids.filter(id => /^k-block-\d+-count$/.test(id))).toEqual([
    'k-block-0-count',
    'k-block-1-count',
    'k-block-2-count'
  ]);
  expect(ids).toContain('k-simulate');
  expect(ids).toContain('k-quantize');
  expect(ids.slice(-3)).toEqual(['k-finalize', 'k-envelope', 'k-converged']);
});

it('GPUNetworkKFunction skips the simulation nodes without simulations', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const nodes = new GPUNetworkKFunction({
    ...createProps(
      graph,
      {simulationCount: 0, rowsPerBlock: 10},
      {kValues: createTransientView(graph, 'k-values-single', 'float32', 8)}
    )
  }).getCommandNodes(graph);
  expect(nodes.some(node => node.id === 'k-simulate')).toBe(false);
});

it('GPUNetworkKFunction validates views and parameters', () => {
  const create = (overrides: Record<string, unknown>) => () =>
    new GPUNetworkKFunction({
      ...createProps(new GPUCommandGraph(createNullWebGPUDevice()), overrides)
    });
  expect(create({})).not.toThrow();
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(
    () =>
      new GPUNetworkKFunction(
        createProps(graph, {}, {kValues: createTransientView(graph, 'short', 'float32', 8)})
      )
  ).toThrow(/output.kValues length must be 32/);
  expect(
    () =>
      new GPUNetworkKFunction(
        createProps(
          graph,
          {},
          {},
          {weights: createTransientView(graph, 'short-weights', 'float32', 3)}
        )
      )
  ).toThrow(/topology.weights length/);
  const missingPositions = createProps(graph);
  expect(
    () =>
      new GPUNetworkKFunction({
        ...missingPositions,
        topology: {...missingPositions.topology, nodePositions: undefined as never}
      })
  ).toThrow(/topology.nodePositions is required/);
  expect(create({bandCount: 1})).toThrow(/bandCount/);
  expect(() => getGPUNetworkKFunctionParameterValues({seed: -1, activeSimulations: 1})).toThrow(
    /seed/
  );
  expect(
    Array.from(getGPUNetworkKFunctionParameterValues({seed: 2 ** 32 + 5, activeSimulations: 7}))
  ).toEqual([5, 1, 7, 0]);
});

it('GPUNetworkKFunction bounds its command node count by the primitive allowance', () => {
  // Nodes this class creates are counted exactly; snapping, reductions and the length scan are
  // counted as a flat allowance of 192.
  for (const configuration of [
    {overrides: {maximumIterations: 2}, hasConverged: true},
    {overrides: {simulationCount: 0, rowsPerBlock: 10, maximumIterations: 5}, hasConverged: false},
    {overrides: {simulationCount: 20, rowsPerBlock: 7, laneCount: 3, maximumIterations: 4}},
    {overrides: {simulationCount: 4, rowsPerBlock: 40}, hasConverged: true}
  ] as {overrides: Record<string, unknown>; hasConverged?: boolean}[]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    // createProps uses 8 bands and 3 simulations unless overridden.
    const patternCount = 1 + ((configuration.overrides['simulationCount'] as number) ?? 3);
    const contributor = new GPUNetworkKFunction(
      createProps(graph, configuration.overrides, {
        pairCounts: createTransientView(graph, 'budget-pair-counts', 'uint32', 8 * patternCount),
        envelope: createTransientView(graph, 'budget-envelope', 'float32', 24),
        status: configuration.hasConverged
          ? {converged: createTransientView(graph, 'budget-converged', 'uint32', 1)}
          : undefined,
        kValues: createTransientView(graph, 'budget-k-values', 'float32', 8 * patternCount)
      }) as never
    );
    const nodeCount = contributor.getCommandNodes(graph).length;
    const label = JSON.stringify(configuration);
    expect(contributor.estimatedCommandNodeCount, label).toBeGreaterThanOrEqual(nodeCount);
    expect(contributor.estimatedCommandNodeCount - nodeCount, label).toBeLessThanOrEqual(192);
  }
});

it('GPUNetworkKFunction rejects configurations above the command node budget', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  // 10 events and 99 simulations in blocks of one row: 1,000 blocks of 64 rounds and more.
  expect(
    () =>
      new GPUNetworkKFunction(
        createProps(
          graph,
          {simulationCount: 600, rowsPerBlock: 1, bandCount: 2},
          {kValues: createTransientView(graph, 'budget-reject-k', 'float32', 2 * 601)}
        ) as never
      )
  ).toThrow(/^k estimated command node count must be at most 32768$/);
});

it('GPUNetworkKFunction schedules events times edges above the unindexed snapping limit', () => {
  // 2^21 events on 128 edges are 2^28 point-edge tests, above the 2^27 a direct GPUNetworkSnapping
  // accepts without candidateCapacity. This configuration built before the limit existed, so the
  // K function must still construct and schedule it, with no error under the nested snapping id.
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const events = 2 ** 21;
  const edgeCount = 128;
  const props = createProps(
    graph,
    {
      positions: createTransientView(graph, 'many-events', 'float32x2', events),
      simulationCount: 0,
      rowsPerBlock: 2047,
      maximumIterations: 1
    },
    {kValues: createTransientView(graph, 'many-k-values', 'float32', 8)},
    {
      neighbors: createTransientView(graph, 'many-neighbors', 'uint32', edgeCount),
      weights: createTransientView(graph, 'many-weights', 'float32', edgeCount)
    }
  );
  const contributor = new GPUNetworkKFunction(props);
  expect(contributor.props.positions.length * edgeCount).toBeGreaterThan(
    GPU_NETWORK_SNAPPING_MAXIMUM_UNINDEXED_WORK
  );
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.some(node => node.id === 'k-snap-scan')).toBe(true);
});
