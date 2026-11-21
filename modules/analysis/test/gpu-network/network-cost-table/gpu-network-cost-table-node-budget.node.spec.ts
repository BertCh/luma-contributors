// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {readFileSync} from 'node:fs';
import {GPUNetworkCostTable} from '../../../src/gpu-network/network-cost-table/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const EDGE_COUNT = 24;
let viewCount = 0;

type Configuration = {
  nodeCount: number;
  laneCount?: number;
  maximumIterations?: number;
  hasConverged: boolean;
  hasBuiltCostLimit?: boolean;
};

function createContributor(graph: GPUCommandGraph, configuration: Configuration) {
  const {nodeCount} = configuration;
  const createView = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `budget-view-${viewCount++}`, format, length);
  return new GPUNetworkCostTable({
    id: 'table',
    topology: {
      offsets: createView('uint32', nodeCount + 1),
      neighbors: createView('uint32', EDGE_COUNT),
      weights: createView('float32', EDGE_COUNT)
    },
    costLimit: createView('float32', 1),
    laneCount: configuration.laneCount,
    maximumIterations: configuration.maximumIterations,
    output: {
      builtCostLimit: configuration.hasBuiltCostLimit ? createView('float32', 1) : undefined,
      offsets: createView('uint32', nodeCount + 1),
      targets: createView('uint32', 100),
      costs: createView('float32', 100),
      status: configuration.hasConverged ? {converged: createView('uint32', 1)} : undefined
    }
  });
}

it('GPUNetworkCostTable estimates its command node count exactly', () => {
  for (const configuration of [
    {
      nodeCount: 3,
      laneCount: 1,
      maximumIterations: 1,
      hasConverged: false,
      hasBuiltCostLimit: true
    },
    {
      nodeCount: 35,
      laneCount: 10,
      maximumIterations: 4,
      hasConverged: true,
      hasBuiltCostLimit: true
    },
    {nodeCount: 20, laneCount: 20, hasConverged: false},
    {nodeCount: 50, laneCount: 7, maximumIterations: 3, hasConverged: true},
    {nodeCount: 33, hasConverged: false}
  ] satisfies Configuration[]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const contributor = createContributor(graph, configuration);
    expect(contributor.estimatedCommandNodeCount, JSON.stringify(configuration)).toBe(
      contributor.getCommandNodes(graph).length
    );
  }
});

it('GPUNetworkCostTable rejects configurations above the command node budget', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  // 600 one-lane batches of 64 rounds and about 7 further nodes each: about 42,000 nodes.
  expect(() =>
    createContributor(graph, {nodeCount: 600, laneCount: 1, hasConverged: false})
  ).toThrow(/^table estimated command node count must be at most 32768$/);
  expect(() =>
    createContributor(graph, {nodeCount: 400, laneCount: 1, hasConverged: false})
  ).not.toThrow();
});

it('the documented large-network configuration fits both command and frontier limits', () => {
  const documentation = readFileSync('docs/api/gpu-network/operators.md', 'utf8');
  const configuration = documentation.match(
    /For example, ([\d,]+) nodes with (\d+) lanes and (\d+) rounds/
  );
  expect(configuration, 'a concrete constructible cost-table configuration').not.toBeNull();
  const nodeCount = Number(configuration![1].replaceAll(',', ''));
  const laneCount = Number(configuration![2]);
  const maximumIterations = Number(configuration![3]);
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  try {
    const contributor = createContributor(graph, {
      nodeCount,
      laneCount,
      maximumIterations,
      hasConverged: true
    });
    expect(contributor.estimatedCommandNodeCount).toBeLessThanOrEqual(32768);
    expect(8 * laneCount * nodeCount).toBeLessThanOrEqual(128 * 1024 * 1024);
    // At the previous advertised 100,000 nodes, even the largest binding-safe lane count
    // exceeds the command budget. All smaller lane counts need at least as many batches.
    const maximumLanes = Math.floor((128 * 1024 * 1024) / (8 * 100_000));
    expect(() =>
      createContributor(graph, {nodeCount: 100_000, laneCount: maximumLanes, hasConverged: true})
    ).toThrow(/estimated command node count/);
  } finally {
    graph.device.destroy();
  }
});

it('counts the captured horizon at the 32768-command boundary', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  try {
    // 2 + 762 * (4 + 2 + 37) = 32768 before adding the capture command.
    expect(() =>
      createContributor(graph, {
        nodeCount: 762,
        laneCount: 1,
        maximumIterations: 37,
        hasConverged: false,
        hasBuiltCostLimit: true
      })
    ).toThrow(/estimated command node count/);
    expect(
      createContributor(graph, {
        nodeCount: 762,
        laneCount: 1,
        maximumIterations: 37,
        hasConverged: false
      }).estimatedCommandNodeCount
    ).toBe(32768);
  } finally {
    graph.device.destroy();
  }
});
