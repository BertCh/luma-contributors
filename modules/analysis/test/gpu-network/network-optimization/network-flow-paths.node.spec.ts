// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {GPUCommandGraph, createTransientView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createNetworkFlowPathNodes,
  type GPUNetworkFlowPathOutput
} from '../../../src/gpu-network/network-optimization/network-flow-paths';
import {
  computeNetworkFlowOnCPU,
  CPUNetworkFlowOperation
} from '../../../src/gpu-network/network-optimization/cpu-network-flow-operation';
import {
  GPUNetworkFlowOperation,
  type NetworkFlowOperationProps
} from '../../../src/gpu-network/network-optimization/gpu-network-flow-operation';
import {GPUNetworkFlow} from '../../../src/gpu-network/network-optimization/gpu-network-flow';
import {prepareGPUNetworkFlowDemands} from '../../../src/gpu-network/network-optimization/prepare-flow-demands';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {
  createNullWebGPUDevice,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

const DIAMOND = {
  nodeCount: 4,
  edges: [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 3]
  ] as const,
  capacities: [1, 1, 1, 1],
  demands: [-2, 0, 0, 2]
};

it('NET-10 decomposes both unit diamond paths with their volumes and super arcs', () => {
  const prepared = prepareGPUNetworkFlowDemands(DIAMOND);
  const result = computeNetworkFlowOnCPU(prepared, {...prepared, pathCapacity: 3});
  expect(result.paths).toEqual({
    edgeRows: Uint32Array.of(
      4,
      0,
      2,
      5,
      0xffffffff,
      4,
      1,
      3,
      5,
      0xffffffff,
      ...new Array(5).fill(0xffffffff)
    ),
    lengths: Uint32Array.of(4, 4, 0),
    volumes: Uint32Array.of(1, 1, 0),
    status: {
      count: Uint32Array.of(2),
      requiredCount: Uint32Array.of(2),
      overflow: Uint32Array.of(0),
      converged: Uint32Array.of(1),
      iterationLimitReached: Uint32Array.of(0),
      invalidCount: Uint32Array.of(0)
    }
  });
});

it('NET-10 one augmentation publishes one partial path and propagates incompleteness', () => {
  const prepared = prepareGPUNetworkFlowDemands(DIAMOND);
  const result = computeNetworkFlowOnCPU(prepared, {
    ...prepared,
    maximumAugmentations: 1,
    pathCapacity: 2
  });
  expect(result.paths?.status.count).toEqual(Uint32Array.of(1));
  expect(result.paths?.volumes).toEqual(Uint32Array.of(1, 0));
  expect(result.paths?.edgeRows.slice(0, 4)).toEqual(Uint32Array.of(4, 0, 2, 5));
  expect(result.paths?.status.converged).toEqual(Uint32Array.of(0));
  expect(result.paths?.status.iterationLimitReached).toEqual(Uint32Array.of(1));
});

it('NET-10 zero demand returns a complete empty decomposition', () => {
  const prepared = prepareGPUNetworkFlowDemands({...DIAMOND, demands: [0, 0, 0, 0]});
  const result = computeNetworkFlowOnCPU(prepared, {
    ...prepared,
    maximumAugmentations: 0,
    pathCapacity: 2
  });
  expect(result.paths?.status.count).toEqual(Uint32Array.of(0));
  expect(result.paths?.lengths).toEqual(Uint32Array.of(0, 0));
  expect(result.paths?.volumes).toEqual(Uint32Array.of(0, 0));
  expect(result.paths?.status.converged).toEqual(Uint32Array.of(1));
  expect(result.paths?.status.iterationLimitReached).toEqual(Uint32Array.of(0));
});

it('reports path capacity loss separately and still counts both required paths', () => {
  const prepared = prepareGPUNetworkFlowDemands(DIAMOND);
  const result = computeNetworkFlowOnCPU(prepared, {...prepared, pathCapacity: 1});
  expect(result.paths?.status.count).toEqual(Uint32Array.of(1));
  expect(result.paths?.edgeRows).toEqual(Uint32Array.of(4, 0, 2, 5, 0xffffffff));
  expect(result.paths?.status).toEqual({
    count: Uint32Array.of(1),
    requiredCount: Uint32Array.of(2),
    overflow: Uint32Array.of(1),
    converged: Uint32Array.of(1),
    iterationLimitReached: Uint32Array.of(0),
    invalidCount: Uint32Array.of(0)
  });
});

it('decomposes the final rerouted flow instead of returning augmentation history', () => {
  const result = computeNetworkFlowOnCPU(
    {
      edges: Uint32Array.of(0, 1, 0, 2, 1, 3, 1, 4, 2, 3, 2, 4, 3, 5, 4, 5),
      capacities: new Uint32Array(8).fill(1),
      costs: Uint32Array.of(0, 0, 1, 2, 2, 100, 0, 0),
      parameters: Uint32Array.of(0, 5, 2)
    },
    {nodeCount: 6, pathCapacity: 2}
  );
  // The first augmentation used edge 2; its flow is canceled by the second augmentation.
  expect(result.flow[2]).toBe(0);
  expect(result.paths?.edgeRows).toEqual(
    Uint32Array.of(0, 3, 7, 0xffffffff, 0xffffffff, 1, 4, 6, 0xffffffff, 0xffffffff)
  );
  expect(result.paths?.volumes).toEqual(Uint32Array.of(1, 1));
});

it('breaks nonunique decompositions by BFS queue order then ascending edge row', () => {
  const result = computeNetworkFlowOnCPU(
    {
      edges: Uint32Array.of(0, 1, 0, 2, 1, 3, 2, 3, 3, 4, 3, 5, 4, 6, 5, 6),
      capacities: new Uint32Array(8).fill(1),
      parameters: Uint32Array.of(0, 6, 2)
    },
    {nodeCount: 7, pathCapacity: 2}
  );
  // At vertex 3 either source branch could be paired with either sink branch.
  expect(result.paths?.edgeRows).toEqual(
    Uint32Array.of(0, 2, 4, 6, 0xffffffff, 0xffffffff, 1, 3, 5, 7, 0xffffffff, 0xffffffff)
  );
  expect(result.paths?.volumes).toEqual(Uint32Array.of(1, 1));
});

it('uses bottleneck volumes rather than expanding each integral unit into a path', () => {
  const prepared = prepareGPUNetworkFlowDemands({
    ...DIAMOND,
    capacities: [3, 2, 3, 2],
    demands: [-5, 0, 0, 5]
  });
  const result = computeNetworkFlowOnCPU(prepared, {...prepared, pathCapacity: 2});
  expect(result.paths?.status.count).toEqual(Uint32Array.of(2));
  expect(result.paths?.volumes).toEqual(Uint32Array.of(3, 2));
});

it('rejects malformed problems with cleared paths and invalid status', () => {
  const result = computeNetworkFlowOnCPU(
    {
      edges: Uint32Array.of(0, 99),
      capacities: Uint32Array.of(1),
      parameters: Uint32Array.of(0, 1, 1)
    },
    {nodeCount: 2, pathCapacity: 1}
  );
  expect(result.paths).toEqual({
    edgeRows: Uint32Array.of(0xffffffff),
    lengths: Uint32Array.of(0),
    volumes: Uint32Array.of(0),
    status: {
      count: Uint32Array.of(0),
      requiredCount: Uint32Array.of(0),
      overflow: Uint32Array.of(0),
      converged: Uint32Array.of(0),
      iterationLimitReached: Uint32Array.of(0),
      invalidCount: Uint32Array.of(1)
    }
  });
});

const PATH_PROPS: NetworkFlowOperationProps = {
  nodeCount: 6,
  edgeCount: 6,
  pathCapacity: 2,
  edges: 'edges',
  capacities: 'capacities',
  parameters: 'parameters',
  output: {
    flow: 'flow',
    cut: 'cut',
    summary: 'summary',
    paths: {
      edgeRows: 'pathEdges',
      lengths: 'pathLengths',
      volumes: 'pathVolumes',
      status: {
        count: 'pathCount',
        requiredCount: 'pathRequired',
        overflow: 'pathOverflow',
        converged: 'pathConverged',
        iterationLimitReached: 'pathLimited',
        invalidCount: 'pathInvalid'
      }
    }
  }
};

it('both operation forms expose the same path layouts and the CPU callback writes every port', async () => {
  const prepared = prepareGPUNetworkFlowDemands(DIAMOND);
  const props = {...PATH_PROPS, maximumAugmentations: 1};
  const cpu = new CPUNetworkFlowOperation(props);
  const gpu = new GPUNetworkFlowOperation(props);
  expect(cpu.outputs).toEqual(expect.objectContaining({pathEdges: expect.anything()}));
  expect(gpu.outputs['pathEdgeRows'].length).toBe(10);
  const vector = (values: Uint32Array, format: 'uint32' | 'uint32x2' = 'uint32'): GPUCPUVector => ({
    format,
    length: values.length / (format === 'uint32x2' ? 2 : 1),
    chunks: [{values, length: values.length / (format === 'uint32x2' ? 2 : 1)}],
    pack: () => values
  });
  const result = await cpu.execute({
    inputs: {
      edges: vector(prepared.edges, 'uint32x2'),
      capacities: vector(prepared.capacities),
      parameters: vector(prepared.parameters)
    }
  });
  expect(result['pathEdges']).toEqual(
    Uint32Array.of(4, 0, 2, 5, 0xffffffff, ...new Array(5).fill(0xffffffff))
  );
  expect(result['pathLengths']).toEqual(Uint32Array.of(4, 0));
  expect(result['pathVolumes']).toEqual(Uint32Array.of(1, 0));
  expect(result['pathCount']).toEqual(Uint32Array.of(1));
  expect(result['pathRequired']).toEqual(Uint32Array.of(1));
  expect(result['pathOverflow']).toEqual(Uint32Array.of(0));
  expect(result['pathConverged']).toEqual(Uint32Array.of(0));
  expect(result['pathLimited']).toEqual(Uint32Array.of(1));
  expect(result['pathInvalid']).toEqual(Uint32Array.of(0));
});

it('validates path bounds and output ownership in both operation forms', () => {
  const shapes = {
    edges: {format: 'uint32x2' as const, length: 6},
    capacities: {format: 'uint32' as const, length: 6},
    parameters: {format: 'uint32' as const, length: 3}
  };
  expect(() =>
    constructNetworkOperationContributor(new GPUNetworkFlowOperation(PATH_PROPS), shapes)
  ).not.toThrow();
  for (const Operation of [GPUNetworkFlowOperation, CPUNetworkFlowOperation]) {
    for (const overrides of [
      {pathCapacity: 0},
      {pathCapacity: 1025},
      {nodeCount: 129},
      {edgeCount: 1025}
    ])
      expect(() => new Operation({...PATH_PROPS, ...overrides})).toThrow(
        /network-flow.*(pathCapacity|output.paths)/
      );
    expect(
      () => new Operation({...PATH_PROPS, output: {...PATH_PROPS.output, paths: undefined}})
    ).toThrow(/pathCapacity.*output.paths/);
    expect(
      () =>
        new Operation({
          ...PATH_PROPS,
          output: {
            ...PATH_PROPS.output,
            paths: {...PATH_PROPS.output.paths!, edgeRows: 'capacities'}
          }
        })
    ).toThrow();
  }
});

it('propagates the controlling parallel phase and relaxation budgets to path status', () => {
  for (const costed of [false, true]) {
    const prepared = prepareGPUNetworkFlowDemands({
      ...DIAMOND,
      costs: costed ? [0, 0, 0, 0] : undefined
    });
    for (const budget of [0, 1]) {
      const result = computeNetworkFlowOnCPU(prepared, {
        ...prepared,
        solver: 'parallel',
        pathCapacity: 2,
        maximumAugmentations: 0,
        mode: costed ? 'primal-dual' : undefined,
        maximumCostPhases: costed ? budget : undefined,
        maximumRelabels: costed ? undefined : budget
      });
      expect(result.paths?.status.count).toEqual(Uint32Array.of(budget * 2));
      expect(result.paths?.volumes).toEqual(Uint32Array.of(budget, budget));
      expect(result.paths?.status.converged).toEqual(Uint32Array.of(budget));
      expect(result.paths?.status.iterationLimitReached).toEqual(Uint32Array.of(1 - budget));
    }
    const limited = computeNetworkFlowOnCPU(prepared, {
      ...prepared,
      solver: 'parallel',
      maximumRelaxationRounds: 1,
      pathCapacity: 2
    });
    expect(limited.paths?.status.count).toEqual(Uint32Array.of(0));
    expect(limited.paths?.status.converged).toEqual(Uint32Array.of(0));
    expect(limited.paths?.status.iterationLimitReached).toEqual(Uint32Array.of(1));
  }
});

it('includes the two optional decomposition nodes in the integrated graph estimate', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const solver of ['reference', 'parallel'] as const) {
      const graph = new GPUCommandGraph(device);
      const operation = new GPUNetworkFlowOperation({
        ...PATH_PROPS,
        solver,
        maximumAugmentations: 1
      });
      const shapes = {
        edges: {format: 'uint32x2' as const, length: 6},
        capacities: {format: 'uint32' as const, length: 6},
        parameters: {format: 'uint32' as const, length: 3},
        ...operation.outputs
      };
      const views = Object.fromEntries(
        Object.entries(shapes).map(([name, shape]) => [
          name,
          createTransientView(graph, name, shape.format, shape.length)
        ])
      );
      const contributor = operation.createContributor(
        name => views[name] as never
      ) as GPUNetworkFlow;
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      expect(nodes.slice(-2).map(node => node.id)).toEqual([
        'network-flow-paths',
        'network-flow-path-status'
      ]);
    }
  } finally {
    device.destroy();
  }
});

it('interprets the actual decomposition and status WGSL against hand-worked CPU fixtures', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const scene of [
      'diamond',
      'partial',
      'overflow',
      'zero',
      'invalid',
      'circulation'
    ] as const) {
      const prepared = prepareGPUNetworkFlowDemands({
        ...DIAMOND,
        demands: scene === 'zero' ? [0, 0, 0, 0] : DIAMOND.demands
      });
      const pathCapacity = scene === 'overflow' ? 1 : 3;
      if (scene === 'invalid') prepared.edges[0] = 99;
      const expected = computeNetworkFlowOnCPU(prepared, {
        ...prepared,
        pathCapacity,
        maximumAugmentations: scene === 'partial' ? 1 : 8
      });
      // An independent feasible flow with a positive cycle must omit that circulation.
      if (scene === 'circulation') {
        prepared.nodeCount = 4;
        prepared.edges = Uint32Array.of(0, 1, 1, 3, 1, 2, 2, 1);
        prepared.parameters = Uint32Array.of(0, 3, 2);
        expected.flow = Uint32Array.of(2, 2, 1, 1);
        expected.summary = Uint32Array.of(1, 1, 2, 0, 1, 0, 2, 1);
        expected.paths = {
          edgeRows: Uint32Array.of(0, 1, 0xffffffff, ...new Array(6).fill(0xffffffff)),
          lengths: Uint32Array.of(2, 0, 0),
          volumes: Uint32Array.of(2, 0, 0),
          status: {
            count: Uint32Array.of(1),
            requiredCount: Uint32Array.of(1),
            overflow: Uint32Array.of(0),
            converged: Uint32Array.of(1),
            iterationLimitReached: Uint32Array.of(0),
            invalidCount: Uint32Array.of(0)
          }
        };
      }
      const graph = new GPUCommandGraph(device);
      recordWGSLKernelNodes(graph);
      const arrays = new Map<GraphDataView, Uint32Array>();
      const values = (view: GraphDataView): Uint32Array => {
        if (!arrays.has(view))
          arrays.set(
            view,
            new Uint32Array(view.length * (view.format === 'uint32x2' ? 2 : 1)).fill(0xdeadbeef)
          );
        return arrays.get(view)!;
      };
      const edges = createTransientView(graph, 'edges', 'uint32x2', prepared.edges.length / 2);
      const flow = createTransientView(graph, 'flow', 'uint32', expected.flow.length);
      const parameters = createTransientView(graph, 'parameters', 'uint32', 3);
      const summary = createTransientView(graph, 'summary', 'uint32', 8);
      const output: GPUNetworkFlowPathOutput = {
        edgeRows: createTransientView(
          graph,
          'pathEdges',
          'uint32',
          pathCapacity * (prepared.nodeCount - 1)
        ),
        lengths: createTransientView(graph, 'pathLengths', 'uint32', pathCapacity),
        volumes: createTransientView(graph, 'pathVolumes', 'uint32', pathCapacity),
        status: {
          count: createTransientView(graph, 'pathCount', 'uint32', 1),
          requiredCount: createTransientView(graph, 'pathRequired', 'uint32', 1),
          overflow: createTransientView(graph, 'pathOverflow', 'uint32', 1),
          invalidCount: createTransientView(graph, 'pathInvalid', 'uint32', 1),
          converged: createTransientView(graph, 'pathConverged', 'uint32', 1),
          iterationLimitReached: createTransientView(graph, 'pathLimited', 'uint32', 1)
        }
      };
      values(edges).set(prepared.edges);
      values(flow).set(expected.flow);
      values(parameters).set(prepared.parameters);
      values(summary).set(expected.summary);
      for (const node of createNetworkFlowPathNodes(
        graph,
        'flow',
        prepared.nodeCount,
        edges,
        flow,
        parameters,
        summary,
        output
      )) {
        const record = getWGSLKernelNodeRecord(graph, node.id)!;
        expect(
          findReservedIdentifiers(stripWGSLComments(record.source)),
          `${node.id} WGSL identifiers`
        ).toEqual([]);
        expect(() => new WgslReflect(record.source)).not.toThrow();
        expect(getStorageBindingCount(record.source)).toBeLessThanOrEqual(8);
        const source = record.source
          .replace(/^\s*_\s*=[^;]*;/gm, '')
          .replace(/@workgroup_size\(\s*(\d+)u?\s*\)/g, '@workgroup_size($1, 1, 1)');
        new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
          'main',
          [...record.workgroups],
          {
            0: Object.fromEntries(
              record.bindings.map((binding, index) => [index, values(binding.view)])
            )
          },
          {constants: record.constants}
        );
      }
      for (const name of ['edgeRows', 'lengths', 'volumes'] as const)
        expect(values(output[name]), `${scene} ${name}`).toEqual(expected.paths![name]);
      for (const name of [
        'count',
        'requiredCount',
        'overflow',
        'invalidCount',
        'converged',
        'iterationLimitReached'
      ] as const)
        expect(values(output.status[name]), `${scene} ${name}`).toEqual(
          expected.paths!.status[name]
        );
    }
  } finally {
    device.destroy();
  }
});
