// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUNetworkCostTable} from '../../../src/gpu-network/network-cost-table/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const EDGE_COUNT = 24;
let viewCount = 0;

type Configuration = {
  nodeCount: number;
  laneCount?: number;
  maximumIterations?: number;
  hasConverged: boolean;
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
      offsets: createView('uint32', nodeCount + 1),
      targets: createView('uint32', 100),
      costs: createView('float32', 100),
      status: configuration.hasConverged ? {converged: createView('uint32', 1)} : undefined
    }
  });
}

it('GPUNetworkCostTable estimates its command node count exactly', () => {
  for (const configuration of [
    {nodeCount: 35, laneCount: 10, maximumIterations: 4, hasConverged: true},
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
