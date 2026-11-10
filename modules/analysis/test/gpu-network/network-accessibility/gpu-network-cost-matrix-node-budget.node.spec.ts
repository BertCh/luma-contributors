// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUNetworkCostMatrix} from '../../../src/gpu-network/network-accessibility/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const NODE_COUNT = 8;
const EDGE_COUNT = 12;
let viewCount = 0;

type Configuration = {
  rowCount: number;
  laneCount?: number;
  maximumIterations?: number;
  targetCount?: number;
  hasConverged: boolean;
};

function createContributor(graph: GPUCommandGraph, configuration: Configuration) {
  const {rowCount, targetCount} = configuration;
  const createView = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `matrix-budget-view-${viewCount++}`, format, length);
  return new GPUNetworkCostMatrix({
    id: 'matrix',
    topology: {
      offsets: createView('uint32', NODE_COUNT + 1),
      neighbors: createView('uint32', EDGE_COUNT),
      weights: createView('float32', EDGE_COUNT)
    },
    startNodes: createView('uint32', rowCount),
    targetNodes: targetCount === undefined ? undefined : createView('uint32', targetCount),
    laneCount: configuration.laneCount,
    maximumIterations: configuration.maximumIterations,
    output: {
      costs: createView('float32', rowCount * (targetCount ?? NODE_COUNT)),
      status: configuration.hasConverged ? {converged: createView('uint32', 1)} : undefined
    }
  });
}

it('GPUNetworkCostMatrix estimates its command node count exactly', () => {
  for (const configuration of [
    // Batches write in place or through a copy depending on alignment and the last short batch.
    {rowCount: 5, laneCount: 2, maximumIterations: 3, hasConverged: true},
    {rowCount: 8, laneCount: 8, maximumIterations: 5, hasConverged: false},
    {rowCount: 16, laneCount: 8, hasConverged: true},
    {rowCount: 7, laneCount: 3, maximumIterations: 4, targetCount: 3, hasConverged: true},
    {rowCount: 6, maximumIterations: 2, hasConverged: false}
  ] satisfies Configuration[]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const contributor = createContributor(graph, configuration);
    expect(contributor.estimatedCommandNodeCount, JSON.stringify(configuration)).toBe(
      contributor.getCommandNodes(graph).length
    );
  }
});

it('GPUNetworkCostMatrix rejects configurations above the command node budget', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  // 600 one-lane batches of 64 rounds each: about 40,000 nodes.
  expect(() =>
    createContributor(graph, {rowCount: 600, laneCount: 1, hasConverged: false})
  ).toThrow(/^matrix estimated command node count must be at most 32768$/);
  expect(() =>
    createContributor(graph, {rowCount: 400, laneCount: 1, hasConverged: false})
  ).not.toThrow();
});
