// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNetworkCentrality,
  type GPUNetworkCentralityOutput
} from '../../../src/gpu-network/network-centrality/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let viewCount = 0;
const NODE_COUNT = 40;
const EDGE_COUNT = 100;

type Configuration = {
  pivotCount: number;
  laneCount: number;
  maximumIterations: number;
  outputs: 'all' | 'closeness' | 'edgeBetweenness';
};

function createContributor(graph: GPUCommandGraph, configuration: Configuration) {
  const createView = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${viewCount++}`, format, length);
  const createTopology = () => ({
    offsets: createView('uint32', NODE_COUNT + 1),
    neighbors: createView('uint32', EDGE_COUNT),
    weights: createView('float32', EDGE_COUNT)
  });
  const {pivotCount} = configuration;
  const output: GPUNetworkCentralityOutput =
    configuration.outputs === 'all'
      ? {
          betweenness: createView('float32', NODE_COUNT),
          edgeBetweenness: createView('float32', EDGE_COUNT),
          closeness: createView('float32', pivotCount),
          harmonicCloseness: createView('float32', pivotCount),
          reachedCounts: createView('uint32', pivotCount),
          status: {converged: createView('uint32', 1)}
        }
      : configuration.outputs === 'closeness'
        ? {closeness: createView('float32', pivotCount)}
        : {edgeBetweenness: createView('float32', EDGE_COUNT)};
  return new GPUNetworkCentrality({
    topology: createTopology(),
    reverseTopology: createTopology(),
    pivotNodes: createView('uint32', pivotCount),
    laneCount: configuration.laneCount,
    maximumIterations: configuration.maximumIterations,
    output
  });
}

it('GPUNetworkCentrality estimates its command node count exactly', () => {
  for (const configuration of [
    {pivotCount: 5, laneCount: 2, maximumIterations: 3, outputs: 'all'},
    {pivotCount: 7, laneCount: 7, maximumIterations: 64, outputs: 'closeness'},
    {pivotCount: 9, laneCount: 4, maximumIterations: 1024, outputs: 'edgeBetweenness'}
  ] as const) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const contributor = createContributor(graph, configuration);
    expect(contributor.estimatedCommandNodeCount, JSON.stringify(configuration)).toBe(
      contributor.getCommandNodes(graph).length
    );
  }
});

it('GPUNetworkCentrality caps rounds at the node count in its estimate', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const capped = createContributor(graph, {
    pivotCount: 4,
    laneCount: 4,
    maximumIterations: 1024,
    outputs: 'all'
  });
  const exact = createContributor(graph, {
    pivotCount: 4,
    laneCount: 4,
    maximumIterations: NODE_COUNT,
    outputs: 'all'
  });
  expect(capped.estimatedCommandNodeCount).toBe(exact.estimatedCommandNodeCount);
});

it('GPUNetworkCentrality rejects configurations above the command node cap', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  // 40 batches of about 3 * 40 nodes are far below the cap; one lane per pivot over many pivots is not.
  expect(() =>
    createContributor(graph, {
      pivotCount: 40,
      laneCount: 1,
      maximumIterations: 64,
      outputs: 'all'
    })
  ).not.toThrow();
  expect(() =>
    createContributor(graph, {
      pivotCount: 2000,
      laneCount: 1,
      maximumIterations: 64,
      outputs: 'all'
    })
  ).toThrow(/estimated command node count must be at most 32768/);
});
