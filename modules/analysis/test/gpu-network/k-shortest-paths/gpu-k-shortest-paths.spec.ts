// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUKShortestPaths,
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
import {getFixtureInput} from './k-shortest-paths-test-utils';
import reference from '../../oracles/fixtures/k-shortest-paths.json';

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
  pathLength: number
) {
  for (let route = 0; route < rows['lengths'].length; route++) {
    const length = rows['lengths'][route];
    expect(
      Array.from(
        rows['vertices'].slice(route * pathLength + length, (route + 1) * pathLength)
      ).every(value => value === 0xffffffff)
    ).toBe(true);
    expect(
      Array.from(
        rows['edges'].slice(route * pathLength + Math.max(0, length - 1), (route + 1) * pathLength)
      ).every(value => value === 0xffffffff)
    ).toBe(true);
    if (route % routeCount >= rows['counts'][Math.floor(route / routeCount)]) {
      expect(length).toBe(0);
      expect(rows['costs'][route]).toBe(Infinity);
    }
  }
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
  const limited = await run(getFixtureInput(reference.cases[6]), {
    routeCount: 2,
    maximumPathLength: 300,
    maximumIterations: 1
  });
  expect(Array.from(limited['counts'])).toEqual([0]);
  expect(Array.from(limited['iterationLimitReached'])).toEqual([1]);
  expect(Array.from(limited['converged'])).toEqual([0]);
  checkPadding(limited, 2, 300);
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
  checkPadding(gpu, 4, 3);
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
  checkPadding(gpu, 2, 1);
});
