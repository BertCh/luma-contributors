// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUGraphAlgorithmsBreadthFirstSearch,
  GPUGraphAlgorithmsConnectedComponents
} from '@luma.gl/graph/gpu-graph-algorithms';
import {expect, it} from 'vitest';
import {GPUNetworkAnalyticsColumns} from '../../src/gpu-network/network-analysis/gpu-network-analytics-columns';
import {readUint32, submitGraph} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import {createTopologyFixture} from './upstream-drafts.test-utils';

// Baseline probes for the proposal's fixtures, not an implementation of the proposed CSR view API.
// The physical topology is prepopulated, so do not add its source-column rebuild to these graphs.

it('R1 uses two hops for both triangle submissions and distinguishes the one-hop default', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createTopologyFixture(device, [0, 2, 3, 3], [1, 0xffffffff, 2]);
  const seeds = fixture.createVector('seeds', [0]);
  const distances = fixture.createVector('distances', [0, 0, 0]);
  const predecessors = fixture.createVector('predecessors', [0, 0, 0]);
  const createSearch = (maxDepth?: number) => {
    const graph = new GPUCommandGraph<undefined>(device);
    new GPUGraphAlgorithmsBreadthFirstSearch({
      topology: fixture.topology,
      seeds: seeds.vector,
      distances: distances.vector,
      predecessors: predecessors.vector,
      ...(maxDepth === undefined ? {} : {maxDepth})
    }).addToGraph(graph);
    return graph.compile();
  };
  try {
    const defaultSearch = createSearch();
    try {
      submitGraph(device, defaultSearch, undefined);
      expect(await readUint32(distances.buffer, 3)).toEqual([0, 1, 0xffffffff]);
    } finally {
      defaultSearch.destroy();
    }
    const search = createSearch(2);
    try {
      // slotMask [1,0,1], materialized as absent neighbors on the existing physical route.
      // This host fixture adaptation does not claim to exercise GPU-resident mask consumption.
      submitGraph(device, search, undefined);
      expect(await readUint32(distances.buffer, 3)).toEqual([0, 1, 2]);
      // slotMask [0,1,0], same compiled graph, same offsets and seeds, same two-hop budget.
      fixture.forward.neighbors.buffer.write(Uint32Array.of(0xffffffff, 2, 0xffffffff));
      submitGraph(device, search, undefined);
      expect(await readUint32(distances.buffer, 3)).toEqual([0, 0xffffffff, 1]);
    } finally {
      search.destroy();
    }
  } finally {
    fixture.destroy();
  }
});

it('R3 retains zero degree and a converged singleton with two sentinel neighbors', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createTopologyFixture(device, [0, 2], [0xffffffff, 0xffffffff]);
  const degree = fixture.createVector('degree', [99]);
  const labels = fixture.createVector('labels', [99]);
  const converged = fixture.createVector('converged', [99]);
  const graph = new GPUCommandGraph<undefined>(device);
  const view = (value: ReturnType<typeof fixture.createVector>) =>
    graph.importGPUVector(value.vector.name, value.vector).data[0];
  graph.add(
    new GPUNetworkAnalyticsColumns({
      topology: {
        offsets: view(fixture.forward.offsets),
        neighbors: view(fixture.forward.neighbors)
      },
      degree: {output: view(degree)},
      components: {
        output: view(labels),
        maximumIterations: 2,
        status: {converged: view(converged)}
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(degree.buffer, 1)).toEqual([0]);
    expect(await readUint32(labels.buffer, 1)).toEqual([0]);
    expect(await readUint32(converged.buffer, 1)).toEqual([1]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('R4 reads overflow each submission even when stored adjacency is identical', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createTopologyFixture(device, [0, 1, 1], [1]);
  const labels = fixture.createVector('labels', [99, 99]);
  const converged = fixture.createVector('converged', [99]);
  const graph = new GPUCommandGraph<undefined>(device);
  new GPUGraphAlgorithmsConnectedComponents({
    topology: fixture.topology,
    output: labels.vector,
    converged: converged.vector,
    iterations: 4
  }).addToGraph(graph);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(labels.buffer, 2)).toEqual([0, 0]);
    expect(await readUint32(converged.buffer, 1)).toEqual([1]);
    fixture.forward.overflow.buffer.write(Uint32Array.of(1));
    submitGraph(device, compiled, undefined);
    expect(await readUint32(labels.buffer, 2)).toEqual([0xffffffff, 0xffffffff]);
    expect(await readUint32(converged.buffer, 1)).toEqual([0]);
    fixture.forward.overflow.buffer.write(Uint32Array.of(0));
    submitGraph(device, compiled, undefined);
    expect(await readUint32(labels.buffer, 2)).toEqual([0, 0]);
    expect(await readUint32(converged.buffer, 1)).toEqual([1]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
