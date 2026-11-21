// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUKShortestPaths,
  decodeGPUKShortestPaths,
  GPUKShortestPathsOperation,
  CPUKShortestPathsOperation,
  type KShortestPathsOperationProps,
  type CPUKShortestPathsInput
} from '../../../src/gpu-network/k-shortest-paths/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  runOperationGraph,
  type OperationGraphInput,
  type OperationGraphRows
} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  getFixtureInput,
  getSpurInterruptionInput,
  getCandidateSpurInterruptionInput
} from './k-shortest-paths-test-utils';
import reference from '../../oracles/fixtures/k-shortest-paths.json';
import {
  expectTraversalPadding,
  expectYenEnumerationStatus
} from '../alternative-routes/traversal-enumeration-assertions';

const OUTPUT = {
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
};
const OUTPUT_NAMES = [
  'vertices',
  'edges',
  'lengths',
  'costs',
  'counts',
  ...Object.values(OUTPUT.status)
];

it('network:H2 every spur certificate precedes local truncation on both backends', async () => {
  for (const candidateStop of [false, true]) {
    for (const reverse of [false, true])
      for (const cpu of [false, true]) {
        const input = candidateStop
          ? getCandidateSpurInterruptionInput()
          : getSpurInterruptionInput();
        const maximumPathLength = candidateStop ? 4 : 3;
        const firstCost = candidateStop ? 3 : 2;
        if (reverse) {
          input.sources.reverse();
          input.targets.reverse();
        }
        const rows = await run(
          input,
          {
            routeCount: 2,
            maximumPathLength,
            maximumIterations: 1,
            candidateCapacity: candidateStop ? 1 : 256
          },
          cpu
        );
        expect(rows['counts']).toEqual(Uint32Array.of(1, 1));
        expect(rows['lengths']).toEqual(
          Uint32Array.from(reverse ? [2, 0, maximumPathLength, 0] : [maximumPathLength, 0, 2, 0])
        );
        expect(rows['costs']).toEqual(
          Float32Array.from(
            reverse ? [1, Infinity, firstCost, Infinity] : [firstCost, Infinity, 1, Infinity]
          )
        );
        expect(rows['converged']).toEqual(Uint32Array.of(0));
        expect(rows['iterationLimitReached']).toEqual(Uint32Array.of(1));
        expect(rows['candidateOverflow']).toEqual(Uint32Array.of(0));
        checkPadding(rows, 2, maximumPathLength, 2);
      }
  }
});

it('network:G5 a shared shortest-search interruption vetoes every active query proposal', async () => {
  const offsets = Uint32Array.from({length: 73}, (_, row) => Math.min(row, 69));
  offsets[71] = 70;
  offsets[72] = 70;
  const neighbors = Uint32Array.from({length: 70}, (_, row) => (row < 69 ? row + 1 : 71));
  for (const sources of [Uint32Array.of(0, 70), Uint32Array.of(70, 0)]) {
    const targets = sources[0] === 0 ? Uint32Array.of(69, 71) : Uint32Array.of(71, 69);
    for (const cpu of [false, true]) {
      const rows = await run(
        {topology: {offsets, neighbors, weights: new Float32Array(70).fill(1)}, sources, targets},
        {routeCount: 1, maximumPathLength: 72, maximumIterations: 1},
        cpu
      );
      expect(rows['counts']).toEqual(Uint32Array.of(0, 0));
      expect(rows['converged']).toEqual(Uint32Array.of(0));
      expect(rows['iterationLimitReached']).toEqual(Uint32Array.of(1));
      expect(rows['candidateOverflow']).toEqual(Uint32Array.of(0));
      checkPadding(rows, 1, 72, 2);
    }
  }
});

async function run(
  input: CPUKShortestPathsInput,
  options: Partial<
    Pick<
      KShortestPathsOperationProps,
      'routeCount' | 'maximumPathLength' | 'candidateCapacity' | 'maximumIterations'
    >
  > = {},
  cpu = false
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: KShortestPathsOperationProps = {
    queryCount: input.sources.length,
    routeCount: 8,
    maximumPathLength: 16,
    maximumIterations: 16,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    sources: 'sources',
    targets: 'targets',
    output: OUTPUT,
    ...options
  };
  const inputs: Record<string, OperationGraphInput> = {
    offsets: {format: 'uint32', values: input.topology.offsets},
    neighbors: {format: 'uint32', values: input.topology.neighbors},
    weights: {format: 'float32', values: input.topology.weights},
    sources: {format: 'uint32', values: input.sources},
    targets: {format: 'uint32', values: input.targets}
  };
  return runOperationGraph(device, {
    inputs,
    operations: [
      cpu ? new CPUKShortestPathsOperation(props) : new GPUKShortestPathsOperation(props)
    ],
    outputs: OUTPUT_NAMES
  });
}

function checkPadding(
  rows: Record<string, OperationGraphRows>,
  routeCount: number,
  pathLength: number,
  queryCount = 1
) {
  expectTraversalPadding(
    {
      counts: rows['counts'],
      lengths: rows['lengths'],
      costs: rows['costs'],
      vertices: rows['vertices'],
      edges: rows['edges']
    },
    queryCount,
    routeCount,
    pathLength
  );
}

for (const fixture of reference.cases) {
  it(`GPU Yen matches NetworkX shortest_simple_paths: ${fixture.name}`, async () => {
    const pathLength = fixture.name === 'long-chain' ? 300 : 16;
    const routeCount = fixture.name === 'long-chain' ? 2 : 8;
    const rows = await run(getFixtureInput(fixture), {routeCount, maximumPathLength: pathLength});
    const count = Math.min(routeCount, fixture.routes.length);
    expect(Array.from(rows['counts'])).toEqual([count]);
    expect(Array.from(rows['costs'].slice(0, count))).toEqual(
      fixture.routes.slice(0, count).map(route => route.cost)
    );
    expect(Array.from(rows['converged'])).toEqual([1]);
    expect(Array.from(rows['iterationLimitReached'])).toEqual([0]);
    expect(Array.from(rows['candidateOverflow'])).toEqual([0]);
    const keys = new Set<string>();
    for (let route = 0; route < count; route++) {
      const length = rows['lengths'][route];
      const vertices = Array.from(
        rows['vertices'].slice(route * pathLength, route * pathLength + length)
      );
      const edges = Array.from(
        rows['edges'].slice(route * pathLength, route * pathLength + length - 1)
      );
      expect(vertices[0]).toBe(fixture.source);
      expect(vertices[length - 1]).toBe(fixture.target);
      expect(new Set(vertices).size).toBe(length);
      for (const [slot, edge] of edges.entries()) {
        expect(edge).toBeGreaterThanOrEqual(fixture.offsets[vertices[slot]]);
        expect(edge).toBeLessThan(fixture.offsets[vertices[slot] + 1]);
        expect(fixture.neighbors[edge]).toBe(vertices[slot + 1]);
      }
      expect(edges.reduce((cost, edge) => cost + fixture.weights[edge], 0)).toBe(
        rows['costs'][route]
      );
      expect(keys.has(edges.join(','))).toBe(false);
      keys.add(edges.join(','));
    }
    checkPadding(rows, routeCount, pathLength);
  });
}

for (const edgeCase of [false, true]) {
  it(`Yen graph twins compare every output, including statuses (edge case ${edgeCase})`, async () => {
    const input = getFixtureInput(reference.cases[1]);
    if (edgeCase) {
      input.sources = Uint32Array.of(0, 0, 999);
      input.targets = Uint32Array.of(0, 6, 6);
    }
    const options = {routeCount: 8, maximumPathLength: 8};
    const gpu = await run(input, options);
    const cpu = await run(input, options, true);
    // All costs are small binary fractions, so no tolerance is needed for any output word.
    for (const name of OUTPUT_NAMES)
      expect(Array.from(gpu[name]), name).toEqual(Array.from(cpu[name]));
  });
}

it('decodes GPU readbacks for a route, a zero-edge query and an invalid endpoint', async () => {
  const options = {queryCount: 3, routeCount: 2, maximumPathLength: 3};
  const rows = await run(
    {
      topology: {
        offsets: Uint32Array.of(0, 1, 1),
        neighbors: Uint32Array.of(1),
        weights: Float32Array.of(2)
      },
      sources: Uint32Array.of(0, 1, 99),
      targets: Uint32Array.of(1, 1, 1)
    },
    options
  );
  const decoded = decodeGPUKShortestPaths(
    {
      vertices: rows['vertices'] as Uint32Array,
      edges: rows['edges'] as Uint32Array,
      lengths: rows['lengths'] as Uint32Array,
      costs: rows['costs'] as Float32Array,
      counts: rows['counts'] as Uint32Array,
      status: {
        converged: rows['converged'] as Uint32Array,
        iterationLimitReached: rows['iterationLimitReached'] as Uint32Array,
        candidateOverflow: rows['candidateOverflow'] as Uint32Array,
        iterationCount: rows['iterationCount'] as Uint32Array
      }
    },
    options
  );
  expect(decoded.queries).toEqual([
    {count: 1, routes: [{length: 2, cost: 2, vertices: [0, 1], edges: [0]}]},
    {count: 1, routes: [{length: 1, cost: 0, vertices: [1], edges: []}]},
    {count: 0, routes: []}
  ]);
  expect(decoded.status).toEqual({
    converged: true,
    iterationLimitReached: false,
    candidateOverflow: false,
    iterationCount: 2
  });
});

it('publishes retained paths and distinct path, shortest-search and candidate budget flags', async () => {
  const branching = getFixtureInput(reference.cases[1]);
  const overflow = await run(branching, {candidateCapacity: 1});
  expect(Array.from(overflow['counts'])).toEqual([1]);
  expect(Array.from(overflow['candidateOverflow'])).toEqual([1]);
  expect(Array.from(overflow['converged'])).toEqual([0]);
  expect(Array.from(overflow['iterationLimitReached'])).toEqual([0]);
  checkPadding(overflow, 8, 16);
  const input = {
    topology: {
      offsets: Uint32Array.of(0, 2, 3, 4, 4),
      neighbors: Uint32Array.of(3, 1, 2, 3),
      weights: Float32Array.of(1, 1, 1, 1)
    },
    sources: Uint32Array.of(0),
    targets: Uint32Array.of(3)
  };
  const overlong = await run(input, {maximumPathLength: 3});
  expect(Array.from(overlong['counts'])).toEqual([1]);
  expect(Array.from(overlong['iterationLimitReached'])).toEqual([1]);
  expect(Array.from(overlong['converged'])).toEqual([0]);
  expect(Array.from(overlong['candidateOverflow'])).toEqual([0]);
  checkPadding(overlong, 8, 3);
  for (const cpu of [false, true]) {
    const limited = await run(
      getFixtureInput(reference.cases[6]),
      {
        routeCount: 2,
        maximumPathLength: 300,
        maximumIterations: 1
      },
      cpu
    );
    expect(Array.from(limited['counts'])).toEqual([0]);
    expect(Array.from(limited['iterationLimitReached'])).toEqual([1]);
    expect(Array.from(limited['converged'])).toEqual([0]);
    checkPadding(limited, 2, 300);
  }
});

it('skips impassable edges and unused CSR capacity across expanded lanes', async () => {
  const input = {
    topology: {
      offsets: Uint32Array.of(0, 6, 6),
      neighbors: Uint32Array.of(1, 1, 1, 1, 1, 99, 0),
      weights: Float32Array.of(-1, NaN, Infinity, 2, 3, 0, 0)
    },
    sources: Uint32Array.of(0, 0, 99),
    targets: Uint32Array.of(1, 0, 1)
  };
  const gpu = await run(input, {routeCount: 4, maximumPathLength: 3});
  const cpu = await run(input, {routeCount: 4, maximumPathLength: 3}, true);
  for (const name of OUTPUT_NAMES)
    expect(Array.from(gpu[name]), name).toEqual(Array.from(cpu[name]));
  expect(Array.from(gpu['counts'])).toEqual([2, 1, 0]);
  checkPadding(gpu, 4, 3, 3);
});

it('reuses a compiled graph after sources and weights change, clearing pools, gates and padding', async () => {
  const testDevice = await getWebGPUTestDevice();
  requireTestDevice(testDevice);
  const device = testDevice;
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  let nextView = 0;
  const input = <Format extends 'uint32' | 'float32'>(
    values: Uint32Array | Float32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `input-${nextView++}`, buffer, format, values.length)
    };
  };
  const output = <Format extends 'uint32' | 'float32'>(length: number, format: Format) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, `output-${nextView++}`, buffer, format, length)};
  };
  const sources = input(Uint32Array.of(0), 'uint32');
  const weights = input(Float32Array.of(1, 3, 1), 'float32');
  const vertices = output(12, 'uint32');
  const counts = output(1, 'uint32');
  const converged = output(1, 'uint32');
  const iterationLimitReached = output(1, 'uint32');
  graph.add(
    new GPUKShortestPaths({
      topology: {
        offsets: input(Uint32Array.of(0, 2, 3, 3), 'uint32').view,
        neighbors: input(Uint32Array.of(1, 2, 2), 'uint32').view,
        weights: weights.view
      },
      sources: sources.view,
      targets: input(Uint32Array.of(2), 'uint32').view,
      routeCount: 4,
      maximumPathLength: 3,
      maximumIterations: 8,
      output: {
        vertices: vertices.view,
        edges: output(12, 'uint32').view,
        lengths: output(4, 'uint32').view,
        costs: output(4, 'float32').view,
        counts: counts.view,
        status: {
          converged: converged.view,
          iterationLimitReached: iterationLimitReached.view,
          candidateOverflow: output(1, 'uint32').view,
          iterationCount: output(1, 'uint32').view
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    await submitGraph(device, compiled, undefined);
    expect(Array.from(await readUint32(counts.buffer, 1))).toEqual([2]);
    sources.buffer.write(Uint32Array.of(99));
    await submitGraph(device, compiled, undefined);
    expect(Array.from(await readUint32(counts.buffer, 1))).toEqual([0]);
    expect(Array.from(await readUint32(vertices.buffer, 12))).toEqual(
      new Array(12).fill(0xffffffff)
    );
    sources.buffer.write(Uint32Array.of(0));
    weights.buffer.write(Float32Array.of(-1, -1, -1));
    await submitGraph(device, compiled, undefined);
    expect(Array.from(await readUint32(counts.buffer, 1))).toEqual([0]);
    weights.buffer.write(Float32Array.of(1, 3, 1));
    await submitGraph(device, compiled, undefined);
    expect(Array.from(await readUint32(counts.buffer, 1))).toEqual([2]);
    expect(Array.from(await readUint32(converged.buffer, 1))).toEqual([1]);
    expect(Array.from(await readUint32(iterationLimitReached.buffer, 1))).toEqual([0]);
  } finally {
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});

it('an overlong one-slot path cannot corrupt an adjacent invalid-endpoint query', async () => {
  const input = {
    topology: {
      offsets: Uint32Array.of(0, 2, 2),
      neighbors: Uint32Array.of(1, 1),
      weights: Float32Array.of(2, 1)
    },
    sources: Uint32Array.of(0, 99),
    targets: Uint32Array.of(1, 0)
  };
  const options = {routeCount: 2, maximumPathLength: 1, candidateCapacity: 1};
  const gpu = await run(input, options);
  const cpu = await run(input, options, true);
  expect(Array.from(gpu['counts'])).toEqual([0, 0]);
  expect(Array.from(gpu['lengths'])).toEqual([0, 0, 0, 0]);
  expect(Array.from(gpu['costs'])).toEqual([Infinity, Infinity, Infinity, Infinity]);
  expect(Array.from(gpu['iterationLimitReached'])).toEqual([1]);
  expect(Array.from(gpu['converged'])).toEqual([0]);
  for (const name of OUTPUT_NAMES)
    expect(Array.from(gpu[name]), name).toEqual(Array.from(cpu[name]));
  checkPadding(gpu, 2, 1, 2);
});

it('shares round shaders across independently exhausted queries', async () => {
  // Three parallel edges yield costs 1, 2, 3. Identity and invalid queries finish before them.
  const rows = await run(
    {
      topology: {
        offsets: Uint32Array.of(0, 3, 3),
        neighbors: Uint32Array.of(1, 1, 1),
        weights: Float32Array.of(3, 1, 2)
      },
      sources: Uint32Array.of(0, 0, 9),
      targets: Uint32Array.of(1, 0, 1)
    },
    {routeCount: 4, maximumPathLength: 2}
  );
  expect(Array.from(rows['counts'])).toEqual([3, 1, 0]);
  expect(Array.from(rows['costs'])).toEqual([
    1,
    2,
    3,
    Infinity,
    0,
    Infinity,
    Infinity,
    Infinity,
    Infinity,
    Infinity,
    Infinity,
    Infinity
  ]);
  expect(Array.from(rows['edges'].slice(0, 6))).toEqual([
    1, 0xffffffff, 2, 0xffffffff, 0, 0xffffffff
  ]);
  expect(Array.from(rows['converged'])).toEqual([1]);
  expect(Array.from(rows['iterationLimitReached'])).toEqual([0]);
  expect(Array.from(rows['candidateOverflow'])).toEqual([0]);
  checkPadding(rows, 4, 2, 3);
});

it('Yen complete enumeration and budget prefixes match independent simple edge-slot DFS', async () => {
  const {enumerateAlternativeRoutes, getTraversalEnumerationInput, readTraversalRoutes} =
    await import('../alternative-routes/alternative-routes-operation-test-utils');
  const input = getTraversalEnumerationInput();
  const expected = enumerateAlternativeRoutes(input, 8);
  const all = readTraversalRoutes(expected, 6);
  expect(Array.from(expected.counts)).toEqual([6, 0]);
  for (const options of [
    {maximumPathLength: 5, candidateCapacity: 64},
    {maximumPathLength: 2, candidateCapacity: 64},
    {maximumPathLength: 5, candidateCapacity: 1}
  ]) {
    const rows = await run(input, {...options, routeCount: 8});
    const count = rows['counts'][0];
    const routes = readTraversalRoutes(
      {edges: rows['edges'], lengths: rows['lengths'], costs: rows['costs']},
      count,
      options.maximumPathLength
    );
    // Yen promises cost order, not global edge-lexicographic ordering among cost ties.
    expect(routes.map(route => route.cost)).toEqual(all.slice(0, count).map(route => route.cost));
    for (const [routeIndex, route] of routes.entries()) {
      expect(all).toContainEqual(route);
      expect(
        Array.from(
          rows['vertices'].slice(
            routeIndex * options.maximumPathLength,
            routeIndex * options.maximumPathLength + route.edges.length + 1
          )
        )
      ).toEqual([0, ...route.edges.map(edge => input.topology.neighbors[edge])]);
    }
    expect(new Set(routes.map(route => JSON.stringify(route.edges))).size).toBe(count);
    expectYenEnumerationStatus(
      {
        counts: rows['counts'],
        converged: rows['converged'],
        iterationLimitReached: rows['iterationLimitReached'],
        candidateOverflow: rows['candidateOverflow']
      },
      options.maximumPathLength,
      options.candidateCapacity
    );
    if (options.maximumPathLength === 5 && options.candidateCapacity === 64) {
      expect(routes.map(route => route.edges.join(',')).sort()).toEqual(
        all.map(route => route.edges.join(',')).sort()
      );
    }
    checkPadding(rows, 8, options.maximumPathLength, 2);
  }
});
