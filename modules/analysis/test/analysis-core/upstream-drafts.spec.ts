// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPUCommandGraph,
  GPUHistogram,
  GPUGridIndex,
  GPUGridIndexQuery
} from '@luma.gl/graph/gpu-core';
import {
  GPUGraphAlgorithmsBreadthFirstSearch,
  GPUGraphAlgorithmsConnectedComponents
} from '@luma.gl/graph/gpu-graph-algorithms';
import {expect, it} from 'vitest';
import {GPUNetworkAnalyticsColumns} from '../../src/gpu-network/network-analysis/gpu-network-analytics-columns';
import {readUint32, submitGraph} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {createFittingTestGraph} from './fitting/fitting-test-graph';
import {createTopologyFixture} from './upstream-drafts.test-utils';
import {GRID_QUERY_INITIAL_CANDIDATES} from './kernel-device-probe-fixtures';

type HistogramEdgeFixture = {
  name: string;
  values: Float32Array;
  domain: readonly [number, number];
  expected: number[];
  alpha16Observed?: number[];
  alpha16ObservedView?: number[];
};

// Independent NumPy histogram counts with explicit exact edges (range(50) or [0, 41, 82, 123]).
// Interior bins are half-open and the last is closed. Adjacent values are IEEE-754 words.
const HISTOGRAM_EDGE_FIXTURES: HistogramEdgeFixture[] = [
  {
    name: 'integer-edges-49',
    values: Float32Array.from({length: 50}, (_, index) => index),
    domain: [0, 49] as const,
    expected: [...Array<number>(48).fill(1), 2],
    alpha16ObservedView: [
      1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 0, 1, 1,
      1, 1, 1, 2, 0, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 2, 0, 2
    ]
  },
  {
    name: 'integer-edges-3',
    values: Float32Array.of(0, 41, 82, 123),
    domain: [0, 123] as const,
    expected: [1, 1, 2],
    alpha16Observed: [2, 1, 1]
  },
  {
    name: 'adjacent-1',
    values: new Float32Array(Uint32Array.of(0x3f7fffff, 0x3f800000, 0x3f800001).buffer),
    domain: [0, 49] as const,
    expected: [1, 2, ...Array<number>(47).fill(0)]
  },
  {
    name: 'adjacent-27',
    values: new Float32Array(Uint32Array.of(0x41d7ffff, 0x41d80000, 0x41d80001).buffer),
    domain: [0, 49] as const,
    expected: [...Array<number>(26).fill(0), 1, 2, ...Array<number>(21).fill(0)],
    alpha16ObservedView: [...Array<number>(26).fill(0), 2, 1, ...Array<number>(21).fill(0)]
  },
  {
    name: 'adjacent-41',
    values: new Float32Array(Uint32Array.of(0x4223ffff, 0x42240000, 0x42240001).buffer),
    domain: [0, 123] as const,
    expected: [1, 2, 0],
    alpha16Observed: [2, 1, 0]
  },
  {
    name: 'adjacent-82',
    values: new Float32Array(Uint32Array.of(0x42a3ffff, 0x42a40000, 0x42a40001).buffer),
    domain: [0, 123] as const,
    expected: [0, 1, 2],
    alpha16Observed: [0, 2, 1]
  }
];

async function readHistogramEdgeCounts(
  fixture: HistogramEdgeFixture,
  domainForm: 'literal' | 'view'
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const test = createFittingTestGraph(device);
  try {
    const input = test.floats('input', fixture.values, 1);
    const domain = test.floats('domain', fixture.domain, 1);
    const output = test.uints(
      'counts',
      fixture.expected.map(() => 0xdeadbeef)
    );
    test.graph.add(
      new GPUHistogram({
        input,
        output,
        domain: domainForm === 'literal' ? fixture.domain : domain
      })
    );
    test.run();
    const counts = await test.readUints(output);
    // Preserve full readbacks even when one fixture fails; the lead supplies version/adapter data.
    console.log(
      JSON.stringify({
        finding: 'LUMA-101',
        domainForm,
        name: fixture.name,
        values: Array.from(fixture.values),
        domain: fixture.domain,
        counts,
        expected: fixture.expected
      })
    );
    expect(counts).toHaveLength(fixture.expected.length);
    expect(counts.reduce((total, count) => total + count, 0)).toBe(fixture.values.length);
    return counts;
  } finally {
    test.destroy();
  }
}

for (const domainForm of ['literal', 'view'] as const) {
  for (const fixture of HISTOGRAM_EDGE_FIXTURES) {
    it(`LUMA-101 ${domainForm}: ${fixture.name} compares every bin`, async () => {
      const counts = await readHistogramEdgeCounts(fixture, domainForm);
      // The alpha.16 defect vectors were recorded on one adapter.
      const observed =
        domainForm === 'view'
          ? (fixture.alpha16ObservedView ?? fixture.alpha16Observed)
          : fixture.alpha16Observed;
      if (observed) {
        expect(counts, 'alpha.16 edge-placement defect').toEqual(observed);
        expect(counts).not.toEqual(fixture.expected);
      } else {
        expect(counts).toEqual(fixture.expected);
      }
    });
  }
}

it.each([
  {point: -76, query: -76, expected: [0], count: [1]},
  {point: -75.5, query: -75.5, expected: [0], count: [1]},
  {point: -76, query: 100.5, expected: [0xffffffff], count: [0]}
])('LUMA-103 bounds query $query against indexed $point', async fixture => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const test = createFittingTestGraph(device);
  try {
    const scalars = test.floats('positions', [fixture.point, 0.5]);
    const positions = test.graph.createDataView(scalars.buffer, {format: 'float32x2', length: 1});
    const indexCount = test.uints('index-count', [0xdeadbeef]);
    const indexOverflow = test.uints('index-overflow', [0xdeadbeef]);
    const objectIds = test.uints('index-ids', [0xdeadbeef]);
    const index = new GPUGridIndex({
      positions,
      gridSize: [360, 1],
      bounds: [-180, 0, 180, 1],
      cellOffsets: test.uints('offsets', Array<number>(361).fill(0xdeadbeef)),
      objectIds,
      count: indexCount,
      overflow: indexOverflow
    });
    // Unselected ranks scatter nothing; the empty-query sentinel is caller-owned.
    const output = test.uints('candidates', GRID_QUERY_INITIAL_CANDIDATES);
    const count = test.uints('candidate-count', [0xdeadbeef]);
    const overflow = test.uints('candidate-overflow', [0xdeadbeef]);
    test.graph.add(index);
    test.graph.add(
      new GPUGridIndexQuery({
        index,
        kind: 'bounds',
        query: test.floats('query', [fixture.query, 0.5, fixture.query, 0.5]),
        output,
        count,
        overflow
      })
    );
    test.run();
    const actual = {
      indexCount: await test.readUints(indexCount),
      indexOverflow: await test.readUints(indexOverflow),
      objectIds: await test.readUints(objectIds),
      candidates: await test.readUints(output),
      count: await test.readUints(count),
      overflow: await test.readUints(overflow)
    };
    console.log(JSON.stringify({finding: 'LUMA-103', ...fixture, actual}));
    expect(actual).toEqual({
      indexCount: [1],
      indexOverflow: [0],
      objectIds: [0],
      candidates: fixture.expected,
      count: fixture.count,
      overflow: [0]
    });
  } finally {
    test.destroy();
  }
});

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
