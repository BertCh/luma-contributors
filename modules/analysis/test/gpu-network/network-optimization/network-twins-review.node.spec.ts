// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  computeNetworkPointRoutingOnCPU,
  CPUNetworkPointRoutingOperation
} from '../../../src/gpu-network/network-optimization/cpu-network-point-routing-operation';
import {GPUNetworkPointRouting} from '../../../src/gpu-network/network-optimization/gpu-network-point-routing';
import {GPUNetworkFlow} from '../../../src/gpu-network/network-optimization/gpu-network-flow';
import {CPUNetworkStructureOperation} from '../../../src/gpu-network/network-structure/cpu-network-structure-operation';
import {prepareGPUNetworkPointRoutingTurns} from '../../../src/gpu-network/network-optimization/prepare-point-routing-turns';
import {GPUNetworkLineGraph} from '../../../src/gpu-network/network-line-graph/gpu-network-line-graph';
import {getGPUNetworkLineGraphParameterValues} from '../../../src/gpu-network/network-line-graph/network-line-graph-parameters';
import {
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {setupPointRouting} from './network-point-routing-scalability-test-utils';
import {getRoutingReviewFixtures, traceRoutingReviewPath} from './network-twins-review-fixtures';

for (const fixture of getRoutingReviewFixtures()) {
  it(`${fixture.name}: CPU and interpreted reference WGSL`, () => {
    const result = computeNetworkPointRoutingOnCPU(fixture.input, {
      mode: 'reference',
      maximumRelaxations: fixture.maximumRelaxations
    });
    expect(Array.from(result.costs)).toEqual(fixture.expectedCosts);
    expect(Array.from(result.status)).toEqual(fixture.expectedStatus);
    for (const [query, path] of (fixture.expectedPaths ?? []).entries())
      expect(
        traceRoutingReviewPath(
          result.predecessorEdges,
          fixture.input.queries[query * 2],
          fixture.input.queries[query * 2 + 1],
          fixture.input.costs.length,
          query
        )
      ).toEqual(path);
    const {graph, props} = setupPointRouting({
      edgeCount: fixture.input.costs.length,
      turnCount: fixture.input.turns.length / 2,
      queryCount: fixture.expectedCosts.length,
      mode: 'reference',
      maximumRelaxations: fixture.maximumRelaxations ?? 8
    });
    const penalties = createTransientView(
      graph,
      'penalties',
      'float32',
      fixture.input.turns.length / 2
    );
    const memory = createCPUGraphMemory();
    memory.write(props.costs, fixture.input.costs);
    memory.write(props.turns, fixture.input.turns);
    memory.write(props.queries, fixture.input.queries);
    memory.write(props.fractions, fixture.input.fractions);
    memory.write(penalties, fixture.input.turnPenalties!);
    executeCommandNodesOnCPU(
      new GPUNetworkPointRouting({...props, turnPenalties: penalties}).getCommandNodes(graph),
      memory
    );
    expect(Array.from(memory.readFloat32(props.output.costs))).toEqual(fixture.expectedCosts);
    expect(Array.from(memory.readUint32(props.output.status))).toEqual(fixture.expectedStatus);
    for (const [query, path] of (fixture.expectedPaths ?? []).entries())
      expect(
        traceRoutingReviewPath(
          memory.readUint32(props.output.predecessorEdges),
          fixture.input.queries[query * 2],
          fixture.input.queries[query * 2 + 1],
          fixture.input.costs.length,
          query
        )
      ).toEqual(path);
  });
}

it('R1 explicit reference flow validates supplied CSR even for zero demand', () => {
  for (const valid of [false, true]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const input = {
      edges: createTransientView(graph, 'edges', 'uint32x2', 2),
      capacities: createTransientView(graph, 'capacities', 'uint32', 2),
      parameters: createTransientView(graph, 'parameters', 'uint32', 3)
    };
    const topology = {
      offsets: createTransientView(graph, 'offsets', 'uint32', 4),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 2)
    };
    const output = {
      flow: createTransientView(graph, 'flow', 'uint32', 2),
      cut: createTransientView(graph, 'cut', 'uint32', 3),
      summary: createTransientView(graph, 'summary', 'uint32', 8),
      status: {invalidCount: createTransientView(graph, 'invalid', 'uint32', 1)}
    };
    const memory = createCPUGraphMemory();
    memory.write(input.edges, Uint32Array.of(0, 1, 1, 2));
    memory.write(input.capacities, Uint32Array.of(1, 1));
    memory.write(input.parameters, Uint32Array.of(0, 2, 0));
    memory.write(topology.offsets, Uint32Array.of(0, 1, 2, 2));
    memory.write(topology.neighbors, Uint32Array.of(valid ? 1 : 2, 2));
    expect(
      () =>
        new GPUNetworkFlow({
          ...input,
          topology: {
            ...topology,
            offsets: createTransientView(graph, 'short-offsets', 'uint32', 2)
          },
          output,
          nodeCount: 3,
          solver: 'reference'
        })
    ).toThrow(/topology/);
    executeCommandNodesOnCPU(
      new GPUNetworkFlow({
        ...input,
        topology,
        output,
        nodeCount: 3,
        solver: 'reference'
      }).getCommandNodes(graph),
      memory
    );
    expect(memory.readUint32(output.status.invalidCount)).toEqual(Uint32Array.of(Number(!valid)));
    if (!valid) expect(memory.readUint32(output.summary)).toEqual(new Uint32Array(8));
    else expect(memory.readUint32(output.summary)[0]).toBe(1);
  }
});

function vector(
  values: Uint32Array | Float32Array,
  format: GPUCPUVector['format'] = 'uint32'
): GPUCPUVector {
  const length = values.length / (format.endsWith('x2') ? 2 : 1);
  return {format, length, chunks: [{values, length}], pack: () => values};
}

it('R4 default CPU SCC remains exact for a 1025-vertex chain', async () => {
  const nodeCount = 1025;
  const rows = await new CPUNetworkStructureOperation({
    nodeCount,
    edgeCount: nodeCount - 1,
    topology: {offsets: 'offsets', neighbors: 'neighbors'},
    output: {strongComponents: 'labels'}
  }).execute({
    inputs: {
      offsets: vector(
        Uint32Array.from({length: nodeCount + 1}, (_, row) => Math.min(row, nodeCount - 1))
      ),
      neighbors: vector(Uint32Array.from({length: nodeCount - 1}, (_, row) => row + 1))
    }
  });
  expect(rows['labels']).toEqual(Uint32Array.from({length: nodeCount}, (_, row) => row));
});

it('R5 CPU operation forwards a zero budget with penalties', async () => {
  const fixture = getRoutingReviewFixtures().find(fixture => fixture.name.startsWith('R5'))!;
  const rows = await new CPUNetworkPointRoutingOperation({
    edgeCount: 2,
    queryCount: 2,
    mode: 'reference',
    maximumRelaxations: 0,
    turns: 'turns',
    turnPenalties: 'penalties',
    costs: 'costs',
    queries: 'queries',
    fractions: 'fractions',
    output: {costs: 'output', status: 'status', predecessorEdges: 'parents'}
  }).execute({
    inputs: {
      turns: vector(fixture.input.turns, 'uint32x2'),
      turnPenalties: vector(fixture.input.turnPenalties!, 'float32'),
      edgeCosts: vector(fixture.input.costs, 'float32'),
      queries: vector(fixture.input.queries, 'uint32x2'),
      fractions: vector(fixture.input.fractions, 'float32x2')
    }
  });
  expect(rows['output']).toEqual(Float32Array.from(fixture.expectedCosts));
  expect(rows['status']).toEqual(Uint32Array.from(fixture.expectedStatus));
  expect(rows['parents']).toEqual(new Uint32Array(4).fill(0xffffffff));
});

it('R6 the intermediate penalty distinguishes destination-only charging', () => {
  const fixture = getRoutingReviewFixtures().find(fixture => fixture.name.startsWith('R6'))!;
  expect(computeNetworkPointRoutingOnCPU(fixture.input).costs[0]).toBe(13);
  expect(
    computeNetworkPointRoutingOnCPU({...fixture.input, turnPenalties: Float32Array.of(0, 0)})
      .costs[0]
  ).toBe(6);
});

it('R2 preparation preserves open-road routes when another edge is closed', () => {
  for (const closure of [-1, NaN]) {
    const costs = Float32Array.of(2, 4, closure);
    const prepared = prepareGPUNetworkPointRoutingTurns({
      costs,
      offsets: Uint32Array.of(0, 2, 2, 2),
      neighbors: Uint32Array.of(1, 2),
      weights: Float32Array.of(5, -1),
      overflow: 0
    });
    expect(
      computeNetworkPointRoutingOnCPU({
        ...prepared,
        costs,
        queries: Uint32Array.of(0, 1),
        fractions: Float32Array.of(0.5, 0.5)
      }).costs[0]
    ).toBe(4);
  }
});

it('R7 explicit line-graph penalties survive full-arc float32 absorption', () => {
  const input = {
    costs: Float32Array.of(0, 0, 2 ** 28),
    offsets: Uint32Array.of(0, 1, 2, 2),
    neighbors: Uint32Array.of(2, 2),
    weights: Float32Array.of(2 ** 28 + 7, 2 ** 28 + 1),
    turnPenalties: Float32Array.of(7, 1),
    overflow: 0
  };
  expect(prepareGPUNetworkPointRoutingTurns(input).turnPenalties).toEqual(Float32Array.of(7, 1));
});

it('R7 line-graph producer writes the requested separate penalty column within eight bindings', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const output = {
    topology: {
      offsets: createTransientView(graph, 'line-offsets', 'uint32', 4),
      neighbors: createTransientView(graph, 'line-neighbors', 'uint32', 3),
      weights: createTransientView(graph, 'line-weights', 'float32', 3)
    },
    turnPenalties: createTransientView(graph, 'line-penalties', 'float32', 3),
    status: {overflow: createTransientView(graph, 'overflow', 'uint32', 1)}
  };
  const contributor = new GPUNetworkLineGraph({
    topology: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 5),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 3),
      weights: createTransientView(graph, 'weights', 'float32', 3),
      nodePositions: createTransientView(graph, 'positions', 'float32x2', 4)
    },
    parameters: createTransientView(graph, 'parameters', 'float32', 8),
    output
  });
  const nodes = contributor.getCommandNodes(graph);
  const kernels = getGPUCommandNodeKernelSources(nodes);
  expect(kernels.some(kernel => /linePenalties\[.*\]\s*=/.test(kernel.source))).toBe(true);
  for (const kernel of kernels)
    expect(getStorageBindingCount(kernel.source)).toBeLessThanOrEqual(8);
  const memory = createCPUGraphMemory();
  memory.write(contributor.props.topology.offsets, Uint32Array.of(0, 1, 2, 3, 3));
  memory.write(contributor.props.topology.neighbors, Uint32Array.of(2, 2, 3));
  memory.write(contributor.props.topology.nodePositions, Float32Array.of(-1, 0, 1, 0, 0, 0, 0, 1));
  memory.write(
    contributor.props.parameters,
    getGPUNetworkLineGraphParameterValues({leftTurnCost: 7, rightTurnCost: 1})
  );
  memory.write(output.topology.offsets, Uint32Array.of(0, 1, 2, 2));
  memory.write(output.topology.neighbors, Uint32Array.of(2, 2, 0xffffffff));
  executeCommandNodesOnCPU(
    nodes.filter(node => /-(edge-vectors|clear-turn-penalties|fill-turn-penalties)$/.test(node.id)),
    memory
  );
  expect(memory.readFloat32(output.turnPenalties)).toEqual(Float32Array.of(7, 1, -1));
});
