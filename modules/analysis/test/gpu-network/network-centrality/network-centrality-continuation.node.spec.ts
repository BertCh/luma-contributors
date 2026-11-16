// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, createTransientView, type GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createCentralityContinuationGraphs,
  runCentralityContinuationPhases
} from '../../../src/gpu-network/network-centrality/network-centrality-continuation';
import {
  createNullWebGPUDevice,
  createInputBuffer,
  createOutputBuffer,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {
  getDeclaredNodes,
  type KernelAuditContext
} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {buildCentralityCSRs} from './network-centrality-oracle';

function build(maximumIterations: number, roundsPerStep = 7) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createInputBuffer>[] = [];
  let serial = 0;
  const input = (values: Uint32Array | Float32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `input-${serial++}`,
      buffer,
      values instanceof Float32Array ? 'float32' : 'uint32'
    );
  };
  const output = (format: 'float32' | 'uint32', length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return importGraphBuffer(graph, `output-${serial++}`, buffer, format, length);
  };
  const nodeCount = 2051;
  const {forward, reverse} = buildCentralityCSRs(
    nodeCount,
    Array.from({length: nodeCount - 1}, (_, node) => [node, node + 1, 1])
  );
  const topology = (network: typeof forward) => ({
    offsets: input(network.offsets) as never,
    neighbors: input(network.neighbors) as never,
    weights: input(network.weights) as never
  });
  const props = {
    topology: topology(forward),
    reverseTopology: topology(reverse),
    pivotNodes: input(Uint32Array.of(0)) as never,
    maximumIterations,
    roundsPerStep,
    localIterations: 1,
    output: {
      betweenness: output('float32', nodeCount) as never,
      edgeBetweenness: output('float32', nodeCount - 1) as never,
      closeness: output('float32', 1) as never,
      harmonicCloseness: output('float32', 1) as never,
      reachedCounts: output('uint32', 1) as never,
      status: {
        converged: output('uint32', 1) as never,
        zeroWeightCycle: output('uint32', 1) as never,
        iterationLimitReached: output('uint32', 1) as never,
        approximation: output('uint32', 1) as never
      }
    }
  };
  return {
    device,
    graph,
    props,
    destroy() {
      for (const buffer of buffers) buffer.destroy();
      device.destroy();
    }
  };
}

for (const separateHandles of [false, true]) {
  it(`encodes packed continuation fields with ${separateHandles ? 'separate' : 'shared'} caller import handles`, () => {
    const configuration = build(4096, 1);
    const packed = createOutputBuffer(configuration.device, 4352);
    const shared = importGraphBuffer(configuration.graph, 'packed-output', packed, 'uint32');
    let serial = 0;
    const field = <Format extends 'float32' | 'uint32'>(
      format: Format,
      length: number,
      byteOffset: number
    ) => {
      const handle = separateHandles
        ? importGraphBuffer(configuration.graph, `packed-field-${serial++}`, packed, format).buffer
        : shared.buffer;
      return configuration.graph.createDataView(handle, {format, length, byteOffset});
    };
    const output = {
      betweenness: field('float32', 2051, 0),
      edgeBetweenness: field('float32', 2050, 8204),
      closeness: field('float32', 1, 16640),
      harmonicCloseness: field('float32', 1, 16896),
      reachedCounts: field('uint32', 1, 17152),
      status: {
        converged: field('uint32', 1, 17156),
        zeroWeightCycle: field('uint32', 1, 17160),
        iterationLimitReached: field('uint32', 1, 17164),
        approximation: field('uint32', 1, 17168)
      }
    };
    const schedule = createCentralityContinuationGraphs(configuration.graph, {
      ...configuration.props,
      output
    });
    try {
      for (const graph of Object.values(schedule.graphs)) {
        // Keep real resource declarations and luma.gl import resolution. Only GPU execution is
        // replaced: NullDevice has no compute pipeline or pass implementation.
        const nodes = getDeclaredNodes({graph} as KernelAuditContext) as GPUCommandNode[];
        for (const node of nodes) {
          if (node.type === 'compute')
            node.compile = () => ({
              encode({computePass}) {
                computePass.dispatch(1);
              }
            });
          else node.compile = () => ({encode() {}});
        }
        const compiled = graph.compile();
        const encoder = configuration.device.createCommandEncoder();
        const endPass = vi.fn();
        vi.spyOn(encoder, 'beginComputePass').mockReturnValue({
          props: {},
          end: endPass,
          dispatch() {},
          dispatchIndirect() {},
          pushDebugGroup() {},
          popDebugGroup() {}
        } as unknown as ReturnType<typeof encoder.beginComputePass>);
        try {
          expect(() => compiled.encode(encoder, {parameters: undefined})).not.toThrow();
          expect(endPass).toHaveBeenCalled();
          const packedViews = nodes.flatMap(node =>
            (node.resources ?? []).flatMap(resource =>
              'buffer' in resource &&
              'buffer' in resource.buffer &&
              resource.buffer.buffer.defaultBuffer === packed
                ? [resource.buffer]
                : []
            )
          );
          expect(new Set(packedViews.map(view => view.buffer)).size).toBeLessThanOrEqual(1);
          if (graph === schedule.graphs['publish']) {
            // Every independent logical field survives rebinding at its original byte offset.
            expect(new Set(packedViews.map(view => view.byteOffset))).toEqual(
              new Set([0, 8204, 16640, 16896, 17152, 17156, 17160, 17164, 17168])
            );
          }
        } finally {
          compiled.destroy();
          encoder.destroy();
        }
      }
    } finally {
      schedule.destroy();
      packed.destroy();
      configuration.destroy();
    }
  });
}

it('continues search and both dependency phases on a path longer than 1024 hops', async () => {
  const nodeCount = 2051;
  const distances = new Float64Array(nodeCount).fill(Infinity);
  const sigma = new Float64Array(nodeCount);
  const delta = new Float64Array(nodeCount);
  let frontier: number[] = [];
  let steps = 0;
  const completed = await runCentralityContinuationPhases({
    maximumIterations: nodeCount,
    roundsPerStep: 64,
    hasBetweenness: true,
    initialize(phase) {
      frontier = [phase === 'delta' ? nodeCount - 1 : 0];
      if (phase === 'search') distances[0] = 0;
      if (phase === 'sigma') sigma[0] = 1;
    },
    async execute(phase, roundCount, carry) {
      steps++;
      if (steps % 33 === 1) expect(carry).toBe(false);
      for (let round = 0; round < roundCount; round++) {
        const next: number[] = [];
        for (const node of frontier) {
          if (phase === 'search' && node + 1 < nodeCount) {
            distances[node + 1] = distances[node] + 1;
            next.push(node + 1);
          }
          if (phase === 'sigma' && node + 1 < nodeCount) {
            sigma[node + 1] = sigma[node];
            next.push(node + 1);
          }
          if (phase === 'delta' && node > 0) {
            delta[node - 1] = (sigma[node - 1] / sigma[node]) * (1 + delta[node]);
            next.push(node - 1);
          }
        }
        frontier = next;
      }
      return frontier.length;
    }
  });
  expect(completed).toEqual([true, true, true]);
  expect(steps).toBe(3 * Math.ceil(nodeCount / 64));
  for (let node = 0; node < nodeCount; node++) {
    expect(distances[node]).toBe(node);
    expect(sigma[node]).toBe(1);
    expect(delta[node]).toBe(nodeCount - node - 1);
  }
});

it('keeps graph, shader and pipeline counts independent of the total round budget', () => {
  const configurations = [build(64), build(4096)];
  const schedules = configurations.map(configuration =>
    createCentralityContinuationGraphs(configuration.graph, configuration.props)
  );
  try {
    const collect = (index: number) =>
      Object.values(schedules[index].graphs).flatMap(graph =>
        getDeclaredNodes({graph} as KernelAuditContext)
      );
    const first = collect(0),
      long = collect(1);
    expect(long.length).toBe(first.length);
    expect(long.length).toBe(schedules[1].estimatedCommandNodeCount);
    expect(getGPUCommandNodePipelineKeys(long)).toEqual(getGPUCommandNodePipelineKeys(first));
    const unique = new Map(
      getGPUCommandNodeKernelSources(long).map(kernel => [
        kernel.source + JSON.stringify(kernel.constants),
        kernel
      ])
    );
    for (const kernel of unique.values())
      expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    expect(new Set(getGPUCommandNodeKernelSources(first).map(kernel => kernel.source))).toEqual(
      new Set([...unique.values()].map(kernel => kernel.source))
    );
    for (const phase of ['search', 'sigma', 'delta']) {
      const nodes = getDeclaredNodes({graph: schedules[1].graphs[phase]} as KernelAuditContext);
      expect(
        getGPUCommandNodePipelineKeys(nodes, {idPattern: new RegExp(`-${phase}-\\d+$`)}).size
      ).toBe(1);
    }
  } finally {
    for (const schedule of schedules) schedule.destroy();
    for (const configuration of configurations) configuration.destroy();
  }
});

it('preflights derived persistent bindings before allocating any device buffer', () => {
  const configuration = build(4096);
  Object.defineProperty(configuration.device.limits, 'maxStorageBufferBindingSize', {
    value: 30_000
  });
  const allocate = vi.spyOn(configuration.device, 'createBuffer');
  try {
    expect(() =>
      createCentralityContinuationGraphs(configuration.graph, configuration.props)
    ).toThrow(/adjacency needs/);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    configuration.destroy();
  }
});

it('retains ownership and shape checks for unused vertex-only geometry in continuation', () => {
  const configuration = build(4096);
  const positions = configuration.graph.createDataView(
    configuration.graph.createTransientBuffer({
      id: 'unused-vertex-geometry',
      byteLength: 2051 * 8,
      usage: Buffer.VERTEX
    }),
    {format: 'float32x2', length: 2051}
  );
  const schedule = createCentralityContinuationGraphs(configuration.graph, {
    ...configuration.props,
    topology: {...configuration.props.topology, nodePositions: positions}
  });
  try {
    expect(Object.keys(schedule.graphs)).toHaveLength(8);
  } finally {
    schedule.destroy();
    configuration.destroy();
  }
});

it('counts optional-output graphs exactly and validates numeric bounds before allocation', () => {
  const configuration = build(4096, 1);
  const {output} = configuration.props;
  try {
    for (const subset of [
      {closeness: output.closeness, status: {approximation: output.status.approximation}},
      {edgeBetweenness: output.edgeBetweenness, status: {converged: output.status.converged}},
      output
    ]) {
      const schedule = createCentralityContinuationGraphs(configuration.graph, {
        ...configuration.props,
        output: subset
      });
      try {
        const count = Object.values(schedule.graphs).reduce(
          (total, graph) => total + getDeclaredNodes({graph} as KernelAuditContext).length,
          0
        );
        expect(count).toBe(schedule.estimatedCommandNodeCount);
      } finally {
        schedule.destroy();
      }
    }
    const allocate = vi.spyOn(configuration.device, 'createBuffer');
    for (const maximumIterations of [0, Infinity, 0xffffffff])
      expect(() =>
        createCentralityContinuationGraphs(configuration.graph, {
          ...configuration.props,
          maximumIterations
        })
      ).toThrow(/maximumIterations/);
    for (const roundsPerStep of [0, NaN, 1024])
      expect(() =>
        createCentralityContinuationGraphs(configuration.graph, {
          ...configuration.props,
          roundsPerStep
        })
      ).toThrow(/roundsPerStep/);
    expect(() =>
      createCentralityContinuationGraphs(configuration.graph, {
        ...configuration.props,
        betweennessScale: Infinity
      })
    ).toThrow(/betweennessScale/);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    configuration.destroy();
  }
});

it('rejects the closeness invocation overflow before persistent allocation or graph mutation', () => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  const topology = {
    offsets: createTransientView(graph, 'offsets', 'uint32', 2),
    neighbors: createTransientView(graph, 'neighbors', 'uint32', 0),
    weights: createTransientView(graph, 'weights', 'float32', 0)
  };
  const props = {
    topology,
    reverseTopology: topology,
    pivotNodes: createTransientView(graph, 'pivots', 'uint32', 16_777_216),
    laneCount: 16_777_216,
    output: {closeness: createTransientView(graph, 'closeness', 'float32', 16_777_216)}
  };
  const mutations = [
    vi.spyOn(device, 'createBuffer'),
    vi.spyOn(graph, 'createTransientBuffer'),
    vi.spyOn(graph, 'createDataView'),
    vi.spyOn(graph, 'add')
  ];
  expect(() => createCentralityContinuationGraphs(graph, props)).toThrow(
    /laneCount for closeness outputs must be at most 16777215/
  );
  for (const mutation of mutations) expect(mutation).not.toHaveBeenCalled();
});

for (const failure of ['allocation', 'declaration'] as const) {
  it(`cleans up all owned resources after a continuation ${failure} error`, () => {
    const configuration = build(4096);
    const created: Buffer[] = [];
    const allocate = configuration.device.createBuffer.bind(configuration.device);
    const allocationSpy = vi
      .spyOn(configuration.device, 'createBuffer')
      .mockImplementation(props => {
        if (failure === 'allocation' && created.length === 2)
          throw new Error('injected allocation failure');
        const buffer = allocate(props);
        vi.spyOn(buffer, 'destroy');
        created.push(buffer);
        return buffer;
      });
    const declarationSpy =
      failure === 'declaration'
        ? vi.spyOn(GPUCommandGraph.prototype, 'add').mockImplementation(() => {
            throw new Error('injected declaration failure');
          })
        : undefined;
    try {
      expect(() =>
        createCentralityContinuationGraphs(configuration.graph, configuration.props)
      ).toThrow(/injected .* failure/);
      expect(created.length).toBeGreaterThan(0);
      for (const buffer of created) expect(buffer.destroy).toHaveBeenCalledTimes(1);
    } finally {
      allocationSpy.mockRestore();
      declarationSpy?.mockRestore();
      configuration.destroy();
    }
  });
}

it('checks actual CSR STORAGE usage before persistent allocation', () => {
  const configuration = build(4096);
  const weights = configuration.device.createBuffer({byteLength: 2050 * 4, usage: Buffer.VERTEX});
  const view = importGraphBuffer(configuration.graph, 'vertex-only-weights', weights, 'float32');
  const allocate = vi.spyOn(configuration.device, 'createBuffer');
  try {
    expect(() =>
      createCentralityContinuationGraphs(configuration.graph, {
        ...configuration.props,
        topology: {...configuration.props.topology, weights: view}
      })
    ).toThrow(/usage must include Buffer.STORAGE/);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    weights.destroy();
    configuration.destroy();
  }
});
