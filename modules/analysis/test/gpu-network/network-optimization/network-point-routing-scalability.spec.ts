// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPUNetworkPointRouting,
  type GPUNetworkPointRoutingProps
} from '../../../src/gpu-network/network-optimization/gpu-network-point-routing';
import {
  computeNetworkPointRoutingOnCPU,
  type CPUNetworkPointRoutingInput
} from '../../../src/gpu-network/network-optimization/cpu-network-point-routing-operation';
import {GPUNetworkSnapping} from '../../../src/gpu-network/network-accessibility/gpu-network-snapping';
import {NETWORK_POINT_ROUTING_SNAPPING_CANDIDATE_CAPACITY} from './network-point-routing-scalability-test-utils';
import {
  getGPUCommandNodeKernelSources,
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  getPointRoutingInput,
  enumeratePointRoutingCosts
} from './network-point-routing-operation-test-utils';

type CompilationMetrics = {
  shaderCount: number;
  pipelineCount: number;
  compilationMilliseconds: number;
};

async function runPointRouting(
  input: CPUNetworkPointRoutingInput,
  options: Pick<GPUNetworkPointRoutingProps, 'mode' | 'maximumRelaxations'> = {},
  secondInput?: CPUNetworkPointRoutingInput,
  onCompiled?: (metrics: CompilationMetrics) => void
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const inputs = {
    turns: createInputBuffer(device, input.turns.length ? input.turns : new Uint32Array(2)),
    costs: createInputBuffer(device, input.costs),
    queries: createInputBuffer(device, input.queries),
    fractions: createInputBuffer(device, input.fractions)
  };
  buffers.push(...Object.values(inputs));
  const queryCount = input.queries.length / 2;
  const outputBuffers = {
    costs: createOutputBuffer(device, queryCount),
    status: createOutputBuffer(device, queryCount),
    predecessorEdges: createOutputBuffer(device, queryCount * input.costs.length)
  };
  buffers.push(...Object.values(outputBuffers));
  const output = {
    costs: importGraphBuffer(graph, 'output-costs', outputBuffers.costs, 'float32', queryCount),
    status: importGraphBuffer(graph, 'output-status', outputBuffers.status, 'uint32', queryCount),
    predecessorEdges: importGraphBuffer(
      graph,
      'output-parents',
      outputBuffers.predecessorEdges,
      'uint32',
      queryCount * input.costs.length
    )
  };
  const contributor = new GPUNetworkPointRouting({
    turns: importGraphBuffer(graph, 'turns', inputs.turns, 'uint32x2', input.turns.length / 2),
    costs: importGraphBuffer(graph, 'costs', inputs.costs, 'float32'),
    queries: importGraphBuffer(graph, 'queries', inputs.queries, 'uint32x2'),
    fractions: importGraphBuffer(graph, 'fractions', inputs.fractions, 'float32x2'),
    ...options,
    output
  });
  const nodes = contributor.getCommandNodes(graph);
  let shaderCount = 0;
  let pipelineCount = 0;
  if (onCompiled) {
    const kernels = getGPUCommandNodeKernelSources(nodes);
    shaderCount = new Set(kernels.map(kernel => kernel.source)).size;
    pipelineCount = new Set(
      kernels.map(kernel =>
        JSON.stringify([
          kernel.source,
          Object.entries(kernel.constants).sort(),
          kernel.shaderLayout
        ])
      )
    ).size;
  }
  for (const node of nodes) graph.add(node);
  const compilationStarted = performance.now();
  // Include asynchronous WebGPU shader/pipeline compilation in the measured time.
  const compiled = await graph.compileAsync();
  onCompiled?.({
    shaderCount,
    pipelineCount,
    compilationMilliseconds: performance.now() - compilationStarted
  });
  try {
    const results = [];
    for (const values of [input, ...(secondInput ? [secondInput] : [])]) {
      for (const name of ['turns', 'costs', 'queries', 'fractions'] as const) {
        if (values[name].length) inputs[name].write(values[name]);
      }
      submitGraph(device, compiled);
      results.push({
        costs: await readFloat32(outputBuffers.costs, queryCount),
        status: await readUint32(outputBuffers.status, queryCount),
        predecessorEdges: await readUint32(
          outputBuffers.predecessorEdges,
          queryCount * input.costs.length
        )
      });
    }
    return results;
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

/** Reconstruct a terminal path, allowing a same-edge return only at its final source visit. */
function reconstructPath(
  input: CPUNetworkPointRoutingInput,
  predecessorEdges: readonly number[],
  query = 0
) {
  const edgeCount = input.costs.length;
  const source = input.queries[query * 2];
  const destination = input.queries[query * 2 + 1];
  const parents = predecessorEdges.slice(query * edgeCount, (query + 1) * edgeCount);
  if (source === destination && parents[source] === 0xffffffff) return [source];
  const reversePath = [destination];
  const visited = new Set(reversePath);
  let cursor = destination;
  for (let slot = 0; slot < edgeCount; slot++) {
    const parent = parents[cursor];
    expect(parent, 'finite route must have a source-rooted predecessor chain').toBeLessThan(
      edgeCount
    );
    reversePath.push(parent);
    if (parent === source) return reversePath.reverse();
    expect(visited.has(parent), 'predecessor cycle').toBe(false);
    visited.add(parent);
    cursor = parent;
  }
  throw new Error('point routing predecessor path exceeds the edge count');
}

function computePathCost(input: CPUNetworkPointRoutingInput, path: readonly number[], query = 0) {
  const sourceFraction = input.fractions[query * 2];
  const targetFraction = input.fractions[query * 2 + 1];
  if (path.length === 1)
    return Math.fround((targetFraction - sourceFraction) * input.costs[path[0]]);
  let cost = Math.fround((1 - sourceFraction) * input.costs[path[0]]);
  for (const edge of path.slice(1, -1)) cost = Math.fround(cost + input.costs[edge]);
  return Math.fround(cost + Math.fround(targetFraction * input.costs[path[path.length - 1]]));
}

it('publishes the reference cost and matching path after full-destination float32 absorption', async () => {
  const input = {
    costs: Float32Array.of(2, 1, 67108864, 1),
    turns: Uint32Array.of(3, 1, 3, 0, 1, 2, 0, 2),
    queries: Uint32Array.of(3, 2),
    fractions: Float32Array.of(0, 0)
  };
  const [result] = await runPointRouting(input);
  const [reference] = await runPointRouting(input, {mode: 'reference'});
  expect(result.costs).toEqual([2]);
  expect(result.status).toEqual([0]);
  const path = reconstructPath(input, result.predecessorEdges);
  expect(path).toEqual([3, 1, 2]);
  expect(computePathCost(input, path)).toBe(reference.costs[0]);
  expect(result.predecessorEdges).toEqual(reference.predecessorEdges);
});

for (const edgeCount of [64, 65, 66, 67]) {
  it(`unit-cost ${edgeCount}-edge chain publishes reconstructable partial routes at one frontier round`, async () => {
    const turns = new Uint32Array((edgeCount - 1) * 2);
    for (let edge = 0; edge < edgeCount - 1; edge++) turns.set([edge, edge + 1], edge * 2);
    const input = {
      costs: new Float32Array(edgeCount).fill(1),
      turns,
      queries: Uint32Array.of(0, edgeCount - 1),
      fractions: Float32Array.of(0, 0)
    };
    const [result] = await runPointRouting(input, {maximumRelaxations: 1});
    expect(result.status).toEqual([edgeCount >= 65 ? 2 : 0]);
    if (edgeCount <= 66) {
      expect(result.costs).toEqual([edgeCount - 1]);
      expect(computePathCost(input, reconstructPath(input, result.predecessorEdges))).toBe(
        result.costs[0]
      );
    } else expect(result.costs).toEqual([Infinity]);
  });

  it(`zero-cost ${edgeCount}-edge plateau keeps terminal paths cycle-safe across exhaustion and same-edge returns`, async () => {
    const turns = new Uint32Array(edgeCount * 2);
    for (let edge = 0; edge < edgeCount; edge++)
      turns.set([edge, (edge + 1) % edgeCount], edge * 2);
    const costs = new Float32Array(edgeCount);
    costs[0] = 4;
    const input = {
      costs,
      turns,
      queries: Uint32Array.of(0, edgeCount - 1, 0, 0),
      fractions: Float32Array.of(0.75, 0, 0.75, 0.25)
    };
    const [result] = await runPointRouting(input, {maximumRelaxations: 1});
    expect(result.status).toEqual(Array(2).fill(edgeCount >= 65 ? 2 : 0));
    for (let query = 0; query < 2; query++) {
      if (Number.isFinite(result.costs[query])) {
        const path = reconstructPath(input, result.predecessorEdges, query);
        expect(computePathCost(input, path, query)).toBe(result.costs[query]);
        for (let slot = 0; slot < path.length - 1; slot++)
          expect(path[slot + 1]).toBe((path[slot] + 1) % edgeCount);
      }
    }
    if (edgeCount <= 64) expect(result.costs).toEqual([1, 2]);
    if (edgeCount === 67) expect(result.costs).toEqual([Infinity, Infinity]);
  });
}

it('avoids a destination-descendant cycle when zero-cost partial parents tie', async () => {
  const input = {
    costs: Float32Array.of(0, 0, 0, 1),
    turns: Uint32Array.of(3, 2, 2, 0, 0, 2),
    queries: Uint32Array.of(3, 2),
    fractions: Float32Array.of(0, 0)
  };
  const [result] = await runPointRouting(input);
  expect(result.status).toEqual([0]);
  expect(result.costs).toEqual([1]);
  expect(reconstructPath(input, result.predecessorEdges)).toEqual([3, 2]);
});

it('agrees with the reference and exhaustive CPU paths, including forward and returning same-edge routes', async () => {
  const input = getPointRoutingInput();
  const [scalable] = await runPointRouting(input);
  const [reference] = await runPointRouting(input, {mode: 'reference'});
  const cpu = computeNetworkPointRoutingOnCPU(input);
  expect(scalable.costs).toEqual(Array.from(enumeratePointRoutingCosts(input)));
  expect(scalable).toEqual(reference);
  expect(scalable.costs).toEqual(Array.from(cpu.costs));
  expect(scalable.status).toEqual(Array.from(cpu.status));
  expect(scalable.predecessorEdges).toEqual(Array.from(cpu.predecessorEdges));
});

it('agrees with reference costs within the documented tolerance for non-binary fractions and weights', async () => {
  const input = {
    ...getPointRoutingInput(),
    costs: Float32Array.of(0.13, 0.37, 0.73, 0.19, 0.29, 0.53),
    fractions: Float32Array.of(0.13, 0.79, 0.87, 0.23, 0.17, 0.83, 0.11, 0.67, 0, 1)
  };
  const [scalable] = await runPointRouting(input);
  const [reference] = await runPointRouting(input, {mode: 'reference'});
  const expectedCosts = enumeratePointRoutingCosts(input);
  expect(scalable.status).toEqual(reference.status);
  for (let query = 0; query < scalable.costs.length; query++) {
    if (!Number.isFinite(reference.costs[query])) {
      expect(scalable.costs[query]).toBe(reference.costs[query]);
    } else {
      const tolerance = Math.max(1e-6, 1e-5 * Math.abs(reference.costs[query]));
      expect(Math.abs(scalable.costs[query] - reference.costs[query])).toBeLessThanOrEqual(
        tolerance
      );
      expect(Math.abs(scalable.costs[query] - expectedCosts[query])).toBeLessThanOrEqual(tolerance);
    }
  }
});

it('documents deterministic tie changes while retaining the explicit turn-order reference', async () => {
  const input = {
    costs: Float32Array.of(1, 1, 1, 1),
    turns: Uint32Array.of(0, 2, 0, 1, 2, 3, 1, 3),
    queries: Uint32Array.of(0, 3),
    fractions: Float32Array.of(0, 1)
  };
  const [scalable] = await runPointRouting(input);
  const [reference] = await runPointRouting(input, {mode: 'reference'});
  expect(scalable.costs).toEqual(reference.costs);
  expect(scalable.status).toEqual([0]);
  expect(scalable.predecessorEdges[3]).toBe(1);
  expect(reference.predecessorEdges[3]).toBe(2);
});

it('keeps zero-cost plateau predecessors acyclic, including a legal same-edge return', async () => {
  const input = {
    costs: Float32Array.of(4, 0, 0, 4),
    turns: Uint32Array.of(0, 1, 1, 2, 2, 1, 2, 3, 3, 0),
    queries: Uint32Array.of(0, 3, 0, 0),
    fractions: Float32Array.of(0.5, 0.5, 0.75, 0.25)
  };
  const [result] = await runPointRouting(input);
  expect(result.costs).toEqual(Array.from(enumeratePointRoutingCosts(input)));
  expect(result.status).toEqual([0, 0]);
  expect(result.predecessorEdges).toEqual([0xffffffff, 0, 1, 2, 3, 0, 1, 2]);
});

it('resets invalid-network status and scratch on repeated encodings', async () => {
  const input = getPointRoutingInput();
  const turns = input.turns.slice();
  turns[1] = 99;
  const invalid = {...input, turns};
  const [first, second] = await runPointRouting(invalid, {}, input);
  expect(first.status).toEqual([1, 1, 1, 1, 1]);
  expect(first.costs).toEqual(Array(5).fill(Infinity));
  expect(first.predecessorEdges).toEqual(Array(30).fill(0xffffffff));
  expect(second.status).toEqual([0, 0, 0, 0, 0]);
  expect(second.costs).toEqual(Array.from(enumeratePointRoutingCosts(input)));
});

it('rewrites closed-edge availability and endpoint status on repeated encodings', async () => {
  const input = getPointRoutingInput();
  const closed = {...input, costs: Float32Array.of(8, -1, 6, 2, 3, 5)};
  const [first, second] = await runPointRouting(closed, {}, input);
  expect(first.status).toEqual([0, 0, 0, 1, 0]);
  expect(first.costs).toEqual([11, 12, 4, Infinity, Infinity]);
  expect(first.predecessorEdges[1]).toBe(0xffffffff);
  expect(first.predecessorEdges.slice(18, 24)).toEqual(new Array(6).fill(0xffffffff));
  expect(second.status).toEqual([0, 0, 0, 0, 0]);
  expect(second.costs).toEqual([11, 12, 4, 19, Infinity]);
  expect(second.predecessorEdges[1]).toBe(0);
});

it('preserves invalid queries, malformed turns, overflowing full labels, and zero-round results', async () => {
  const maximumFloat32 = new Float32Array(Uint32Array.of(0x7f7fffff).buffer)[0];
  for (const input of [
    {
      costs: Float32Array.of(1, 1),
      turns: Uint32Array.of(0, 1),
      queries: Uint32Array.of(99, 1),
      fractions: Float32Array.of(0, 1)
    },
    {
      costs: Float32Array.of(1, 1),
      turns: Uint32Array.of(0, 99),
      queries: Uint32Array.of(0, 1),
      fractions: Float32Array.of(0, 1)
    },
    {
      costs: Float32Array.of(1, 1),
      turns: Uint32Array.of(0, 1),
      queries: Uint32Array.of(0, 1),
      fractions: Float32Array.of(NaN, 1)
    },
    {
      costs: Float32Array.of(maximumFloat32, maximumFloat32),
      turns: Uint32Array.of(0, 1),
      queries: Uint32Array.of(0, 1),
      fractions: Float32Array.of(0, 0)
    }
  ]) {
    const [result] = await runPointRouting(input);
    expect(result.costs).toEqual([Infinity]);
    expect(result.status).toEqual([1]);
    expect(result.predecessorEdges).toEqual([0xffffffff, 0xffffffff]);
  }
  const input = getPointRoutingInput();
  const [zero] = await runPointRouting(input, {maximumRelaxations: 0});
  const [reference] = await runPointRouting(input, {mode: 'reference', maximumRelaxations: 0});
  expect(zero).toEqual(reference);
  const [isolated] = await runPointRouting({
    costs: Float32Array.of(2),
    turns: new Uint32Array(),
    queries: Uint32Array.of(0, 0),
    fractions: Float32Array.of(0.25, 0.75)
  });
  expect(isolated.costs).toEqual([1]);
  expect(isolated.status).toEqual([0]);
});

it('publishes partial results and exhaustion for an insufficient frontier budget', async () => {
  const edgeCount = 1024;
  const turns = new Uint32Array((edgeCount - 1) * 2);
  for (let edge = 0; edge < edgeCount - 1; edge++) turns.set([edge, edge + 1], edge * 2);
  const [result] = await runPointRouting(
    {
      costs: new Float32Array(edgeCount).fill(1),
      turns,
      queries: Uint32Array.of(0, edgeCount - 1),
      fractions: Float32Array.of(0.25, 0.75)
    },
    {maximumRelaxations: 1}
  );
  expect(result.status).toEqual([2]);
  expect(result.costs).toEqual([Infinity]);
  expect(result.predecessorEdges[1]).toBe(0);
});

it('routes a grid with over 100k directed edges and reconstructs paths from predecessors', async () => {
  const width = 160;
  const height = 160;
  const outgoing: number[][] = Array.from({length: width * height}, () => []);
  const sources: number[] = [];
  const targets: number[] = [];
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const source = row * width + column;
      for (const [offsetColumn, offsetRow] of [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1]
      ]) {
        const targetColumn = column + offsetColumn;
        const targetRow = row + offsetRow;
        if (targetColumn >= 0 && targetColumn < width && targetRow >= 0 && targetRow < height) {
          outgoing[source].push(sources.length);
          sources.push(source);
          targets.push(targetRow * width + targetColumn);
        }
      }
    }
  }
  const turnRows: number[] = [];
  for (let incoming = 0; incoming < sources.length; incoming++) {
    for (const next of outgoing[targets[incoming]]) turnRows.push(incoming, next);
  }
  const source = 0;
  const destination = outgoing[width * height - 2].find(
    edge => targets[edge] === width * height - 1
  )!;
  expect(sources.length).toBeGreaterThanOrEqual(100000);
  const input = {
    costs: new Float32Array(sources.length).fill(1),
    turns: Uint32Array.from(turnRows),
    queries: Uint32Array.of(source, destination, source, source, source, source),
    fractions: Float32Array.of(0.25, 0.75, 0.25, 0.75, 0.75, 0.25)
  };
  let firstMetrics: CompilationMetrics | undefined;
  for (const queryCount of [1, 10, 100]) {
    const queries = new Uint32Array(queryCount * 2);
    const fractions = new Float32Array(queryCount * 2);
    const expectedCosts: number[] = [];
    for (let query = 0; query < queryCount; query++) {
      const fixture = query % 3;
      queries.set(input.queries.subarray(fixture * 2, fixture * 2 + 2), query * 2);
      fractions.set(input.fractions.subarray(fixture * 2, fixture * 2 + 2), query * 2);
      expectedCosts.push([width + height - 2.5, 0.5, 1.5][fixture]);
    }
    const batch = {...input, queries, fractions};
    const [result] = await runPointRouting(batch, {maximumRelaxations: 16}, undefined, metrics => {
      console.info('point routing 101760-edge grid compilation', {queryCount, ...metrics});
      expect(metrics.compilationMilliseconds).toBeGreaterThanOrEqual(0);
      firstMetrics ??= metrics;
      expect(metrics.shaderCount).toBe(firstMetrics.shaderCount);
      expect(metrics.pipelineCount).toBe(firstMetrics.pipelineCount);
    });
    expect(result.status).toEqual(Array(queryCount).fill(0));
    expect(result.costs).toEqual(expectedCosts);
    for (let query = 0; query < queryCount; query++) {
      const path = reconstructPath(batch, result.predecessorEdges, query);
      expect(computePathCost(batch, path, query)).toBe(result.costs[query]);
      for (let slot = 0; slot < path.length - 1; slot++)
        expect(targets[path[slot]]).toBe(sources[path[slot + 1]]);
    }
  }
}, 120000);

it('same-edge travel keeps the full explored predecessor forest outside its terminal path', async () => {
  const input = {
    costs: Float32Array.of(8, 4),
    turns: Uint32Array.of(0, 1),
    queries: Uint32Array.of(0, 1, 0, 0),
    fractions: Float32Array.of(0.25, 0.5, 0.25, 0.75)
  };
  for (const mode of ['scalable', 'reference'] as const) {
    const [result] = await runPointRouting(input, {mode});
    expect(result.costs).toEqual([8, 4]);
    expect(result.status).toEqual([0, 0]);
    expect(result.predecessorEdges).toEqual([0xffffffff, 0, 0xffffffff, 0]);
    expect(reconstructPath(input, result.predecessorEdges, 1)).toEqual([0]);
  }
});

for (const {candidateCapacity, complete} of [
  {candidateCapacity: 2, complete: false},
  {candidateCapacity: NETWORK_POINT_ROUTING_SNAPPING_CANDIDATE_CAPACITY, complete: true}
]) {
  it(`composes indexed point snapping with turn-aware routing (global candidate capacity ${candidateCapacity})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const input = <Format extends 'uint32' | 'float32' | 'float32x2' | 'uint32x2'>(
      name: string,
      format: Format,
      values: Float32Array | Uint32Array
    ) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return importGraphBuffer(graph, name, buffer, format);
    };
    const costs = input('edge-costs', 'float32', Float32Array.of(8, 4));
    const snappedEdgesBuffer = createOutputBuffer(device, 4);
    const snappedFractionsBuffer = createOutputBuffer(device, 4);
    const snapOverflowBuffer = createOutputBuffer(device, 1);
    const predecessorBuffer = createOutputBuffer(device, 4);
    buffers.push(snappedEdgesBuffer, snappedFractionsBuffer, snapOverflowBuffer, predecessorBuffer);
    const snappedEdges = importGraphBuffer(graph, 'snapped-edges', snappedEdgesBuffer, 'uint32');
    const snappedFractions = importGraphBuffer(
      graph,
      'snapped-fractions',
      snappedFractionsBuffer,
      'float32'
    );
    const queries = createTransientView(graph, 'queries', 'uint32x2', 2);
    const fractions = createTransientView(graph, 'fractions', 'float32x2', 2);
    graph.add(
      new GPUNetworkSnapping({
        positions: input('points', 'float32x2', Float32Array.of(2, 0, 10, 0, 2, 0, 6, 0)),
        topology: {
          offsets: input('offsets', 'uint32', Uint32Array.of(0, 1, 2, 2)),
          neighbors: input('neighbors', 'uint32', Uint32Array.of(1, 2)),
          weights: costs,
          nodePositions: input('nodes', 'float32x2', Float32Array.of(0, 0, 8, 0, 12, 0))
        },
        candidateCapacity,
        maximumSnapDistance: input('snap-radius', 'float32', Float32Array.of(1)),
        output: {
          snappedEdges,
          snapFractions: snappedFractions,
          status: {
            overflow: importGraphBuffer(graph, 'snap-overflow', snapOverflowBuffer, 'uint32')
          }
        }
      })
    );
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'pair-snapped-points',
        operation: 'PointRoutingTest',
        variant: 'pair',
        invocationCount: 4,
        bindings: [
          {name: 'edges', view: snappedEdges, type: 'u32', access: 'read'},
          {name: 'snappedFractions', view: snappedFractions, type: 'f32', access: 'read'},
          {name: 'queries', view: queries, type: 'u32', access: 'read_write'},
          {name: 'fractions', view: fractions, type: 'f32', access: 'read_write'}
        ],
        body: 'queries[queriesOffset + index] = edges[edgesOffset + index]; fractions[fractionsOffset + index] = snappedFractions[snappedFractionsOffset + index];'
      })
    );
    const outputBuffer = createOutputBuffer(device, 2);
    buffers.push(outputBuffer);
    const statusBuffer = createOutputBuffer(device, 2);
    buffers.push(statusBuffer);
    graph.add(
      new GPUNetworkPointRouting({
        costs,
        turns: input('turns', 'uint32x2', Uint32Array.of(0, 1)),
        queries,
        fractions,
        output: {
          costs: importGraphBuffer(graph, 'output-costs', outputBuffer, 'float32'),
          status: importGraphBuffer(graph, 'status', statusBuffer, 'uint32'),
          predecessorEdges: importGraphBuffer(graph, 'parents', predecessorBuffer, 'uint32')
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled);
      // Candidate capacity is global, not per point. All four points need one candidate each.
      expect(await readUint32(snapOverflowBuffer)).toEqual([complete ? 0 : 1]);
      expect(await readUint32(snappedEdgesBuffer)).toEqual(
        complete ? [0, 1, 0, 0] : [0, 1, 0xffffffff, 0xffffffff]
      );
      expect(await readFloat32(snappedFractionsBuffer)).toEqual(
        complete ? [0.25, 0.5, 0.25, 0.75] : [0.25, 0.5, -1, -1]
      );
      expect(await readFloat32(outputBuffer)).toEqual(complete ? [8, 4] : [8, Infinity]);
      expect(await readUint32(statusBuffer)).toEqual(complete ? [0, 0] : [0, 1]);
      // This is the full explored forest: the direct same-edge query still reaches edge 1
      // from edge 0, so its non-terminal predecessor remains populated.
      expect(await readUint32(predecessorBuffer)).toEqual(
        complete ? [0xffffffff, 0, 0xffffffff, 0] : [0xffffffff, 0, 0xffffffff, 0xffffffff]
      );
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}
