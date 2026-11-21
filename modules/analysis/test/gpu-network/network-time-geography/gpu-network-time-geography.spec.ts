// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GPUCommandGraph, GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import fixture from '../../oracles/fixtures/network-time-geography-mob09.json';
import {GPUNetworkReachability} from '../../../src/gpu-network/network-reachability/index';
import {
  GPUNetworkPotentialPathArea,
  GPUNetworkTravelTimePercentiles
} from '../../../src/gpu-network/network-time-geography/index';
import {createOptimizationHarness} from '../network-optimization/optimization-test-harness';
import {requireTestDevice} from '../../../src/testing/require-test-device';

type Edge = [number, number, number];

/** Forward and reverse CSR (edges sorted by source, as the fixture lists them). */
function buildCsr(nodeCount: number, edges: Edge[], reverse: boolean) {
  const rows = edges
    .map(([source, target, weight], edge) => ({
      from: reverse ? target : source,
      to: reverse ? source : target,
      weight,
      edge
    }))
    .sort((first, second) => first.from - second.from || first.edge - second.edge);
  const offsets = new Uint32Array(nodeCount + 1);
  for (const row of rows) offsets[row.from + 1]++;
  for (let node = 0; node < nodeCount; node++) offsets[node + 1] += offsets[node];
  return {
    offsets,
    neighbors: Uint32Array.from(rows.map(row => row.to)),
    weights: Float32Array.from(rows.map(row => row.weight))
  };
}

const infinityOf = (values: (number | null)[]) =>
  Float32Array.from(values.map(value => value ?? Infinity));

it('potential path area includes literal zero slack at the budget boundary (NET-121)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    edgeSlack: harness.output('float32', 3),
    edgeInPrism: harness.output('uint32', 3),
    nodeInPrism: harness.output('uint32', 3)
  };
  const read = await harness.run(
    new GPUNetworkPotentialPathArea({
      topology: {
        offsets: harness.input('uint32', Uint32Array.of(0, 2, 3, 3)),
        neighbors: harness.input('uint32', Uint32Array.of(1, 1, 2)),
        weights: harness.input('float32', Float32Array.of(3, 7, 4))
      },
      forwardCosts: harness.input('float32', Float32Array.of(2, 5, 9)),
      reverseCosts: harness.input('float32', Float32Array.of(7, 4, 0)),
      budget: harness.input('float32', Float32Array.of(9)),
      output
    })
  );
  // Slack = 9 - (forward[source] + edge cost + reverse[target]); node sums all equal 9.
  expect(read(output.edgeSlack)).toEqual([0, -4, 0]);
  expect(read(output.edgeInPrism)).toEqual([1, 0, 1]);
  expect(read(output.nodeInPrism)).toEqual([1, 1, 1]);
});

for (const [caseIndex, scene] of fixture.cases.entries()) {
  it(`GPUNetworkPotentialPathArea matches networkx on case ${caseIndex} (${scene.nodeCount} nodes)`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const edges = scene.edges as Edge[];
    const csr = buildCsr(scene.nodeCount, edges, false);
    const harness = createOptimizationHarness(device);
    const output = {
      edgeInPrism: harness.output('uint32', edges.length),
      edgeSlack: harness.output('float32', edges.length),
      nodeInPrism: harness.output('uint32', scene.nodeCount),
      summary: harness.output('uint32', 2)
    };
    const result = await harness.run(
      new GPUNetworkPotentialPathArea({
        topology: {
          offsets: harness.input('uint32', csr.offsets),
          neighbors: harness.input('uint32', csr.neighbors),
          weights: harness.input('float32', csr.weights)
        },
        forwardCosts: harness.input('float32', infinityOf(scene.forward)),
        reverseCosts: harness.input('float32', infinityOf(scene.backward)),
        budget: harness.input('float32', Float32Array.of(scene.budget)),
        output
      })
    );
    // networkx lists the fixture's edges in CSR order of the sorted-by-source generator output.
    const expectedEdges = scene.edgeInPrism;
    expect(result(output.edgeInPrism)).toEqual(expectedEdges);
    expect(result(output.nodeInPrism)).toEqual(scene.nodeInPrism);
    expect(result(output.summary)).toEqual([
      scene.nodeInPrism.reduce((total, flag) => total + flag, 0),
      expectedEdges.reduce((total, flag) => total + flag, 0)
    ]);
    // Teeth: a proper part of the network, and slack agrees in sign with membership.
    expect(expectedEdges.some(flag => flag === 0)).toBe(true);
    result(output.edgeSlack).forEach((slack, edge) =>
      expect(slack >= 0, `edge ${edge} slack ${slack}`).toBe(expectedEdges[edge] === 1)
    );
  });
}

it('GPUNetworkPotentialPathArea composes with a forward and a reverse GPUNetworkReachability tree', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = fixture.cases[1];
  const edges = scene.edges as Edge[];
  const forwardCsr = buildCsr(scene.nodeCount, edges, false);
  const reverseCsr = buildCsr(scene.nodeCount, edges, true);
  const harness = createOptimizationHarness(device);
  const topology = (csr: ReturnType<typeof buildCsr>) => ({
    offsets: harness.input('uint32', csr.offsets),
    neighbors: harness.input('uint32', csr.neighbors),
    weights: harness.input('float32', csr.weights)
  });
  const forwardTopology = topology(forwardCsr);
  const forwardCosts = harness.output('float32', scene.nodeCount);
  const reverseCosts = harness.output('float32', scene.nodeCount);
  const output = {
    edgeInPrism: harness.output('uint32', edges.length),
    edgeSlack: harness.output('float32', edges.length),
    nodeInPrism: harness.output('uint32', scene.nodeCount)
  };
  const forwardTree = new GPUNetworkReachability({
    id: 'forward-tree',
    topology: forwardTopology,
    startNodes: harness.input('uint32', Uint32Array.of(scene.origin)),
    maximumIterations: 256,
    output: {costs: forwardCosts}
  });
  const reverseTree = new GPUNetworkReachability({
    id: 'reverse-tree',
    topology: topology(reverseCsr),
    startNodes: harness.input('uint32', Uint32Array.of(scene.destination)),
    maximumIterations: 256,
    output: {costs: reverseCosts}
  });
  const prism = new GPUNetworkPotentialPathArea({
    topology: forwardTopology,
    forwardCosts,
    reverseCosts,
    budget: harness.input('float32', Float32Array.of(scene.budget)),
    output
  });
  const chain = {
    id: 'trees-then-prism',
    getCommandNodes: <Parameters>(graph: GPUCommandGraph<Parameters>) => [
      ...forwardTree.getCommandNodes(graph),
      ...reverseTree.getCommandNodes(graph),
      ...prism.getCommandNodes(graph)
    ]
  } as GPUCommandNodeProducer;
  const result = await harness.run(chain);
  expect(result(output.edgeInPrism)).toEqual(scene.edgeInPrism);
  expect(result(output.nodeInPrism)).toEqual(scene.nodeInPrism);
});

/** Binary64 FIFO Dijkstra with arrival `t + base + slope * t` (float32 inputs), per vertex. */
function referenceArrivals(
  nodeCount: number,
  edges: Edge[],
  slopes: number[],
  origin: number,
  departure: number
): number[] {
  const arrival = new Array(nodeCount).fill(Infinity);
  const done = new Array(nodeCount).fill(false);
  arrival[origin] = departure;
  for (let round = 0; round < nodeCount; round++) {
    let node = -1;
    for (let candidate = 0; candidate < nodeCount; candidate++)
      if (!done[candidate] && (node < 0 || arrival[candidate] < arrival[node])) node = candidate;
    if (node < 0 || arrival[node] === Infinity) break;
    done[node] = true;
    edges.forEach(([source, target, base], edge) => {
      if (source !== node) return;
      arrival[target] = Math.min(
        arrival[target],
        arrival[node] + base + slopes[edge] * arrival[node]
      );
    });
  }
  return arrival;
}

async function runPercentiles(
  nodeCount: number,
  edges: Edge[],
  slopes: number[],
  options: {
    origin: number;
    start: number;
    step: number;
    departureCount: number;
    percentiles: number[];
  }
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    travelTimes: harness.output('float32', nodeCount * options.percentiles.length),
    reachedCounts: harness.output('uint32', nodeCount),
    status: harness.output('uint32', 2 * options.departureCount)
  };
  const result = await harness.run(
    new GPUNetworkTravelTimePercentiles({
      stateVertices: harness.input(
        'uint32',
        Uint32Array.from({length: nodeCount}, (_, node) => node)
      ),
      transitions: harness.input(
        'uint32x2',
        Uint32Array.from(edges.flatMap(([from, to]) => [from, to]))
      ),
      costs: harness.input(
        'float32x2',
        Float32Array.from(edges.flatMap(([, , base], edge) => [base, slopes[edge]]))
      ),
      initialState: harness.input('uint32', Uint32Array.of(options.origin)),
      vertexCount: nodeCount,
      departureCount: options.departureCount,
      percentiles: options.percentiles,
      window: harness.input('float32', Float32Array.of(options.start, options.step)),
      maximumRelaxations: nodeCount,
      output
    })
  );
  return {
    travelTimes: result(output.travelTimes),
    reachedCounts: result(output.reachedCounts),
    status: result(output.status)
  };
}

it('GPUNetworkTravelTimePercentiles of static durations equal the networkx travel times', async () => {
  const scene = fixture.cases[0];
  const edges = scene.edges as Edge[];
  const percentiles = [10, 50, 100];
  const result = await runPercentiles(
    scene.nodeCount,
    edges,
    edges.map(() => 0),
    {origin: scene.origin, start: 0, step: 1, departureCount: 5, percentiles}
  );
  expect(result.status.every((flag, index) => flag === 1 || index % 2 === 0)).toBe(true);
  scene.forward.forEach((cost, vertex) => {
    for (let column = 0; column < percentiles.length; column++) {
      const value = result.travelTimes[vertex * percentiles.length + column];
      if (cost === null) expect(value).toBe(Infinity);
      else expect(value, `vertex ${vertex} p${percentiles[column]}`).toBeCloseTo(cost, 3);
    }
    expect(result.reachedCounts[vertex]).toBe(cost === null ? 0 : 5);
  });
});

it('GPUNetworkTravelTimePercentiles follows FIFO time-dependent durations over a departure window', async () => {
  const nodeCount = 30;
  // Deterministic scene: a ring plus chords, base durations and nonnegative slopes (congestion
  // that grows with the clock), so a later departure never arrives earlier.
  const edges: Edge[] = [];
  const slopes: number[] = [];
  for (let node = 0; node < nodeCount; node++) {
    for (const hop of [1, 3, 7]) {
      edges.push([
        node,
        (node + hop) % nodeCount,
        Math.fround(2 + ((node * 7 + hop * 3) % 11) * 0.5)
      ]);
      slopes.push(Math.fround(((node + hop) % 5) * 0.02));
    }
  }
  const options = {
    origin: 0,
    start: 10,
    step: 3,
    departureCount: 12,
    percentiles: [25, 50, 90, 100]
  };
  const result = await runPercentiles(nodeCount, edges, slopes, options);
  expect(result.status.every((flag, index) => flag === 1 || index % 2 === 0)).toBe(true);
  let timeDependent = 0;
  for (let vertex = 0; vertex < nodeCount; vertex++) {
    const times = Array.from({length: options.departureCount}, (_, departure) => {
      const start = options.start + departure * options.step;
      return referenceArrivals(nodeCount, edges, slopes, options.origin, start)[vertex] - start;
    }).sort((first, second) => first - second);
    if (times[0] !== times.at(-1)) timeDependent++;
    options.percentiles.forEach((percentile, column) => {
      // Hand-selected 1-based ranks for [25, 50, 90, 100]% of twelve departures.
      expect(
        result.travelTimes[vertex * options.percentiles.length + column],
        `vertex ${vertex} p${percentile}`
      ).toBeCloseTo(times[[3, 6, 11, 12][column] - 1], 2);
    });
  }
  // Teeth: the departure window changes the travel time at most vertices.
  expect(timeDependent).toBeGreaterThan(nodeCount / 2);
});
