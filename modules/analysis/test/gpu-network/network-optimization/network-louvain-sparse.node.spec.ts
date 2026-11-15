// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  WgslReflect,
  WgslParser,
  WgslExec,
  Function as WGSLFunction,
  Let,
  Var,
  Const
} from 'wgsl_reflect';
import {GPUNetworkLouvain} from '../../../src/gpu-network/network-optimization/gpu-network-louvain';
import {computeNetworkLouvainOnCPU} from '../../../src/gpu-network/network-optimization/cpu-network-louvain-operation';
import {GPUNetworkLouvainOperation} from '../../../src/gpu-network/network-optimization/gpu-network-louvain-operation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function createContributor(
  nodeCount: number,
  edgeCount: number,
  maximumLevels = 1,
  maximumIterations = 1
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUNetworkLouvain({
    id: 'sparse-louvain',
    nodeCount,
    maximumLevels,
    maximumIterations,
    topology: {
      offsets: createTransientView(graph, 'offsets', 'uint32', nodeCount + 1),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', edgeCount),
      weights: createTransientView(graph, 'weights', 'float32', edgeCount)
    },
    parameters: createTransientView(graph, 'parameters', 'float32', 2),
    output: {
      communities: createTransientView(graph, 'communities', 'uint32', nodeCount),
      objectiveTrace: createTransientView(
        graph,
        'trace',
        'float32',
        1 + maximumLevels * maximumIterations
      ),
      summary: createTransientView(graph, 'summary', 'uint32', 4),
      status: {
        converged: createTransientView(graph, 'converged', 'uint32', 1),
        iterationLimitReached: createTransientView(graph, 'limited', 'uint32', 1)
      }
    }
  });
  return {device, graph, contributor};
}

it('sparse kernels parse, audit, stay within eight bindings and gate published primitives', () => {
  const {device, graph, contributor} = createContributor(64, 192, 2, 2);
  try {
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
      expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
      // Reflection parses duplicate declarations without WGSL's scope validation. Reject a
      // repeated local name in a function's outer block before it reaches pipeline creation.
      for (const statement of new WgslParser().parse(kernel.source)) {
        if (!(statement instanceof WGSLFunction)) continue;
        const names = statement.body
          .filter(local => local instanceof Let || local instanceof Var || local instanceof Const)
          .map(local => (local as Let | Var | Const).name);
        expect(
          new Set(names).size,
          `${kernel.id} ${statement.name}: duplicate local declaration`
        ).toBe(names.length);
      }
    }
    const roundNodes = nodes.filter(
      node =>
        node.id.includes('level-0-round-0') &&
        !node.id.endsWith('-gate') &&
        !node.id.endsWith('-control')
    );
    expect(roundNodes.length).toBeGreaterThan(20);
    expect(
      roundNodes.every(
        node =>
          node.condition?.source === 'gpu' ||
          node.description?.startsWith('GPU-gated Louvain primitive')
      )
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

it('command allowances cover empty and maximum sparse shapes without GPU allocation', () => {
  for (const [nodeCount, edgeCount] of [
    [1, 0],
    [65535, 1048576]
  ]) {
    const {device, graph, contributor} = createContributor(nodeCount, edgeCount, 2, 1);
    try {
      const commands = contributor.getCommandNodes(graph);
      const initial = commands.filter(
        command =>
          !command.id.includes('level-') &&
          !command.id.includes('aggregate-') &&
          !command.id.endsWith('-publish-summary') &&
          !command.id.endsWith('-status')
      );
      expect(initial.length).toBeLessThanOrEqual(256);
      expect(commands.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      expect(
        commands.filter(command => command.id.includes('level-0-round-0')).length
      ).toBeLessThanOrEqual(224);
      expect(
        commands.filter(command => command.id.includes('aggregate-0')).length
      ).toBeLessThanOrEqual(112);
    } finally {
      device.destroy();
    }
  }
});

it('construction enforces pair-key, slot, round and command budgets by name', () => {
  expect(() => createContributor(65536, 1)).toThrow(/sparse-louvain.*nodeCount/);
  expect(() => createContributor(2, 1048577)).toThrow(/sparse-louvain.*topology.neighbors.length/);
  expect(() => createContributor(2, 2, 0)).toThrow(/sparse-louvain.*maximumLevels/);
  expect(() => createContributor(2, 2, 32, 64)).toThrow(/sparse-louvain.*command node count/);
});

it('the sparse twin moves coarsened pairs after individual moves stabilize', () => {
  const result = computeNetworkLouvainOnCPU(
    {
      topology: {
        offsets: Uint32Array.of(0, 3, 6, 9, 12),
        neighbors: Uint32Array.of(1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2),
        weights: Float32Array.of(2, 1.25, 1.25, 2, 1.25, 1.25, 1.25, 1.25, 2, 1.25, 1.25, 2)
      },
      parameters: Float32Array.of(1, 1e-5)
    },
    {nodeCount: 4}
  );
  expect(new Set(result.communities).size).toBe(1);
  expect(result.summary[0]).toBe(1);
  expect(result.summary[1]).toBe(1);
  expect(result.summary[2]).toBeGreaterThanOrEqual(3);
  expect(result.objectiveTrace[0]).toBe(-0.25);
  expect(result.objectiveTrace[result.summary[3]]).toBeCloseTo(0, 12);
});

it('CSR padding/empty slots are ignored and malformed offsets fail closed consistently', () => {
  const input = {
    topology: {
      offsets: Uint32Array.of(0, 1, 2, 2),
      neighbors: Uint32Array.of(1, 0, 0xffffffff, 0),
      weights: Float32Array.of(2, 2, NaN, -1)
    },
    parameters: Float32Array.of(1, 0)
  };
  const result = computeNetworkLouvainOnCPU(input, {nodeCount: 3});
  expect(result.communities[0]).toBe(result.communities[1]);
  expect(result.communities[2]).toBe(2);
  expect(result.summary[1]).toBe(1);
  expect(result.objectiveTrace[result.summary[3]]).toBeCloseTo(0, 12);
  input.topology.offsets = Uint32Array.of(0, 2, 1, 4);
  expect(Array.from(computeNetworkLouvainOnCPU(input, {nodeCount: 3}).summary)).toEqual([
    0, 0, 0, 0
  ]);
});

it('sparse operation exposes CSR inputs and named status outputs', () => {
  const operation = new GPUNetworkLouvainOperation({
    nodeCount: 64,
    topology: {offsets: 'offsets', neighbors: 'targets', weights: 'weights'},
    parameters: 'parameters',
    output: {
      communities: 'communities',
      objectiveTrace: 'trace',
      summary: 'summary',
      status: {iterationCount: 'rounds', iterationLimitReached: 'limited'}
    }
  });
  expect(operation.inputs).toEqual({
    offsets: 'offsets',
    neighbors: 'targets',
    weights: 'weights',
    parameters: 'parameters'
  });
  expect(operation.outputs['iterationLimitReached']).toEqual({
    value: 'limited',
    format: 'uint32',
    length: 1
  });
});

it('matches NetworkX Louvain modularity and partitions beyond the legacy 32-node limit', async () => {
  const {default: fixture} = await import('../../oracles/fixtures/network-louvain-sparse.json');
  for (const scene of fixture.cases) {
    for (const seed of [undefined, 42, 0xffffffff]) {
      const result = computeNetworkLouvainOnCPU(
        {
          topology: {
            offsets: Uint32Array.from(scene.offsets),
            neighbors: Uint32Array.from(scene.neighbors),
            weights: Float32Array.from(scene.weights)
          },
          parameters: Float32Array.of(scene.resolution, 1e-5)
        },
        {nodeCount: scene.nodeCount, directed: scene.directed, seed}
      );
      expect(result.summary[0], scene.name).toBe(1);
      expect(result.summary[1], scene.name).toBe(1);
      expect(result.objectiveTrace[result.summary[3]], scene.name).toBeCloseTo(scene.modularity, 6);
      for (let first = 0; first < scene.nodeCount; first++)
        for (let second = 0; second < scene.nodeCount; second++)
          expect(result.communities[first] === result.communities[second], scene.name).toBe(
            scene.communities[first] === scene.communities[second]
          );
      for (let row = 1; row <= result.summary[3]; row++)
        expect(result.objectiveTrace[row], scene.name).toBeGreaterThanOrEqual(
          result.objectiveTrace[row - 1] - 1e-7
        );
      expect(result.objectiveTrace.subarray(result.summary[3] + 1).every(Number.isNaN)).toBe(true);
    }
  }
});

it('returns an improving partition at the round or level budget, and breaks exact ties by priority', () => {
  const input = {
    topology: {
      offsets: Uint32Array.of(0, 1, 2),
      neighbors: Uint32Array.of(1, 0),
      weights: Float32Array.of(2, 2)
    },
    parameters: Float32Array.of(1, 0)
  };
  for (const seed of [0, 1]) {
    const result = computeNetworkLouvainOnCPU(input, {
      nodeCount: 2,
      maximumLevels: 1,
      maximumIterations: 1,
      seed
    });
    expect(Array.from(result.communities)).toEqual(seed === 0 ? [1, 1] : [0, 0]);
    expect(Array.from(result.summary)).toEqual([1, 0, 1, 1]);
    expect(Array.from(result.objectiveTrace)).toEqual([-0.5, 0]);
  }
  const limited = computeNetworkLouvainOnCPU(input, {
    nodeCount: 2,
    maximumLevels: 1,
    maximumIterations: 4
  });
  expect(Array.from(limited.summary)).toEqual([1, 0, 1, 2]);
  const noMoves = computeNetworkLouvainOnCPU(
    {...input, parameters: Float32Array.of(1, 1)},
    {nodeCount: 2}
  );
  expect(Array.from(noMoves.communities)).toEqual([0, 1]);
  expect(Array.from(noMoves.summary)).toEqual([1, 1, 1, 1]);
});

it('packs per-frame controls and validates finite float32 ranges', async () => {
  const {getGPUNetworkLouvainParameterValues} = await import(
    '../../../src/gpu-network/network-optimization/gpu-network-louvain'
  );
  expect(Array.from(getGPUNetworkLouvainParameterValues())).toEqual([1, 0]);
  expect(
    Array.from(getGPUNetworkLouvainParameterValues({resolution: 0.5, minimumGain: 0.125}))
  ).toEqual([0.5, 0.125]);
  for (const value of [-1, NaN, Infinity, 1e40]) {
    expect(() => getGPUNetworkLouvainParameterValues({resolution: value})).toThrow(
      /network-louvain.*resolution/
    );
    expect(() => getGPUNetworkLouvainParameterValues({minimumGain: value})).toThrow(
      /network-louvain.*minimumGain/
    );
  }
});

it('the public dispatch adapter preserves primitive geometry, bindings and pipeline across encodings', async () => {
  const {Buffer} = await import('@luma.gl/core');
  const {gateNetworkLouvainPrimitive} = await import(
    '../../../src/gpu-network/network-optimization/network-louvain-dispatch'
  );
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  Object.defineProperty(device, 'createComputePipeline', {
    value: (props: {id?: string}) => ({id: props.id ?? '', hash: '', destroy() {}})
  });
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cachePipelines: false, _cacheShaders: false}
  });
  const state = createTransientView(graph, 'adapter-state', 'uint32', 8);
  const stateBuffer = device.createBuffer({byteLength: 32, usage: Buffer.STORAGE});
  const pipeline = {id: 'primitive-pipeline'} as import('@luma.gl/core').ComputePipeline;
  const primitiveBindings = {source: stateBuffer};
  let currentPipeline = '';
  let currentBindings: unknown;
  const calls: unknown[] = [];
  const computePass = {
    setPipeline(value: import('@luma.gl/core').ComputePipeline) {
      currentPipeline = value.id;
    },
    setBindings(value: unknown) {
      currentBindings = value;
    },
    dispatch(x: number, y = 1, z = 1) {
      calls.push(['direct', x, y, z]);
    },
    dispatchIndirect() {
      calls.push(['indirect', currentPipeline, currentBindings === primitiveBindings]);
    }
  } as unknown as import('@luma.gl/core').ComputePass;
  const command: import('@luma.gl/graph/gpu-core').GPUComputeCommandNode<void> = {
    id: 'primitive',
    type: 'compute',
    compile: () => ({
      encode({computePass: pass}) {
        pass.setPipeline(pipeline);
        pass.setBindings(primitiveBindings);
        pass.dispatch(7, 3, 2);
      }
    })
  };
  const executable = gateNetworkLouvainPrimitive(
    command,
    state,
    'state[stateOffset] != 0u'
  ).compile({device});
  try {
    for (let run = 0; run < 2; run++)
      executable.encode({computePass, getBuffer: () => stateBuffer} as unknown as Parameters<
        typeof executable.encode
      >[0]);
    expect(calls).toEqual([
      ['direct', 1, 1, 1],
      ['indirect', 'primitive-pipeline', true],
      ['direct', 1, 1, 1],
      ['indirect', 'primitive-pipeline', true]
    ]);
  } finally {
    executable.destroy?.();
    stateBuffer.destroy();
    device.destroy();
  }
});

it('executes generated acceptance WGSL against hand-derived winning and losing reservations', () => {
  const {device, graph, contributor} = createContributor(4, 4);
  try {
    const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    const kernel = kernels.find(candidate => candidate.id.endsWith('level-0-round-0-accept'))!;
    // Interpreter compatibility only: it does not implement phony assignments and interprets
    // a single workgroup-size string as separate digits. Neither rewrite changes the kernel.
    const source = kernel.source
      .replace(/^\s*_\s*=[^;]*;/gm, '')
      .replace(/@workgroup_size\(\s*(\d+)u?\s*\)/g, '@workgroup_size($1, 1, 1)');
    const none = 0xffffffff;
    for (const scene of [
      {
        name: 'no proposals',
        labels: [0, 1, 2, 3],
        proposals: [none, none, none, none],
        reservations: [none, none, none, none],
        expected: [0, 1, 2, 3],
        changed: [0, 0, 0, 0]
      },
      {
        name: 'wins both communities',
        labels: [0, 0, 2, 3],
        proposals: [2, none, none, none],
        reservations: [0, none, 0, none],
        expected: [2, 0, 2, 3],
        changed: [1, 0, 0, 0]
      },
      {
        name: 'loses original community',
        labels: [0, 1, 2, 3],
        proposals: [2, none, none, none],
        reservations: [1, none, 0, none],
        expected: [0, 1, 2, 3],
        changed: [0, 0, 0, 0]
      },
      {
        name: 'loses proposed community',
        labels: [0, 1, 2, 3],
        proposals: [2, none, none, none],
        reservations: [0, none, 2, none],
        expected: [0, 1, 2, 3],
        changed: [0, 0, 0, 0]
      },
      {
        name: 'two disjoint moves',
        labels: [0, 1, 2, 3],
        proposals: [1, none, 3, none],
        reservations: [0, 0, 2, 2],
        expected: [1, 1, 3, 3],
        changed: [1, 0, 1, 0]
      },
      {
        name: 'only one member of a community commits',
        labels: [1, 1, 3, 3],
        proposals: [3, 3, 1, none],
        reservations: [none, 0, none, 0],
        expected: [3, 1, 3, 3],
        changed: [1, 0, 0, 0]
      }
    ]) {
      const accepted = new Uint32Array(4).fill(none);
      const changed = new Uint32Array(4).fill(none);
      const interpreter = new WgslExec(new WgslParser().parse(source));
      interpreter.dispatchWorkgroups(
        'main',
        1,
        {
          0: {
            0: Uint32Array.from(scene.proposals),
            1: Uint32Array.from(scene.labels),
            2: Uint32Array.from(scene.reservations),
            3: accepted,
            4: changed
          }
        },
        {constants: kernel.constants}
      );
      expect(Array.from(accepted), scene.name).toEqual(scene.expected);
      expect(Array.from(changed), scene.name).toEqual(scene.changed);
    }
  } finally {
    device.destroy();
  }
});

it('rejects aliased Louvain primary and status outputs at construction in both input modes', () => {
  const {device, graph, contributor} = createContributor(4, 4);
  const importedOutputBuffer = device.createBuffer({byteLength: 16, usage: Buffer.STORAGE});
  try {
    const importedCommunities = importGraphBuffer(
      graph,
      'imported-communities',
      importedOutputBuffer,
      'uint32',
      4
    );
    const importedSummary = importGraphBuffer(
      graph,
      'imported-summary',
      importedOutputBuffer,
      'uint32',
      4
    );
    const base = contributor.props;
    const shared = base.output.communities;
    const partialSummary = graph.createDataView(
      createTransientView(graph, 'overlapping-output-buffer', 'uint32', 8).buffer,
      {format: 'uint32', byteOffset: 4, length: 4}
    );
    const partialCommunities = graph.createDataView(partialSummary.buffer, {
      format: 'uint32',
      length: 4
    });
    const sharedStatus = createTransientView(graph, 'shared-status', 'uint32', 1);
    const communityStatus = graph.createDataView(shared.buffer, {
      format: 'uint32',
      byteOffset: 4,
      length: 1
    });
    const summaryStatus = graph.createDataView(base.output.summary.buffer, {
      format: 'uint32',
      length: 1
    });
    const traceStatus = graph.createDataView(base.output.objectiveTrace.buffer, {
      format: 'uint32',
      length: 1
    });
    for (const inputs of [
      {topology: base.topology!, weights: undefined},
      {
        topology: undefined,
        weights: createTransientView(graph, 'dense-alias-weights', 'float32', 16)
      }
    ]) {
      for (const output of [
        {...base.output, summary: shared},
        {...base.output, communities: importedCommunities, summary: importedSummary},
        {...base.output, communities: partialCommunities, summary: partialSummary},
        {...base.output, status: {converged: communityStatus}},
        {...base.output, status: {converged: summaryStatus}},
        {...base.output, status: {converged: traceStatus}},
        {...base.output, status: {converged: sharedStatus, iterationCount: sharedStatus}}
      ]) {
        expect(() => new GPUNetworkLouvain({...base, ...inputs, output})).toThrow(
          /sparse-louvain.*output.*distinct buffers/
        );
      }
      expect(() => new GPUNetworkLouvain({...base, ...inputs})).not.toThrow();
    }
  } finally {
    importedOutputBuffer.destroy();
    device.destroy();
  }
});

for (const failurePoint of [
  'first-buffer',
  'second-buffer',
  'gate-kernel',
  'second-buffer-cleanup-throws'
] as const) {
  it(`dispatch compilation releases partial resources and preserves the error (${failurePoint})`, async () => {
    const {vi} = await import('vitest');
    const {gateNetworkLouvainPrimitive} = await import(
      '../../../src/gpu-network/network-optimization/network-louvain-dispatch'
    );
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const state = createTransientView(graph, 'failure-state', 'uint32', 8);
    Object.defineProperty(device, 'props', {
      value: {...device.props, _cachePipelines: false, _cacheShaders: false}
    });
    const failure = new Error(`injected ${failurePoint}`);
    const primitiveDestroy = vi.fn();
    const originalCreateBuffer = device.createBuffer.bind(device);
    const bufferDestructors: ReturnType<typeof vi.spyOn>[] = [];
    const createBuffer = vi.spyOn(device, 'createBuffer').mockImplementation(props => {
      if (
        failurePoint === 'first-buffer' ||
        (failurePoint.startsWith('second-buffer') && bufferDestructors.length === 1)
      )
        throw failure;
      const buffer = originalCreateBuffer(props);
      const originalDestroy = buffer.destroy.bind(buffer);
      bufferDestructors.push(
        vi.spyOn(buffer, 'destroy').mockImplementation(() => {
          originalDestroy();
          if (failurePoint === 'second-buffer-cleanup-throws')
            throw new Error('injected destructor failure');
        })
      );
      return buffer;
    });
    const createShader = vi.spyOn(device, 'createShader').mockImplementation(() => {
      throw failure;
    });
    const command: import('@luma.gl/graph/gpu-core').GPUComputeCommandNode<void> = {
      id: 'failing-primitive',
      type: 'compute',
      compile: () => ({encode() {}, destroy: primitiveDestroy})
    };
    try {
      let caught: unknown;
      try {
        gateNetworkLouvainPrimitive(command, state, 'state[stateOffset] != 0u').compile({device});
      } catch (error) {
        caught = error;
      }
      expect(caught).toBe(failure);
      expect(primitiveDestroy).toHaveBeenCalledTimes(1);
      expect(bufferDestructors.length).toBe(
        failurePoint === 'first-buffer' ? 0 : failurePoint === 'gate-kernel' ? 2 : 1
      );
      for (const destroy of bufferDestructors) expect(destroy).toHaveBeenCalledTimes(1);
    } finally {
      createShader.mockRestore();
      createBuffer.mockRestore();
      for (const destroy of bufferDestructors) destroy.mockRestore();
      device.destroy();
    }
  });
}
