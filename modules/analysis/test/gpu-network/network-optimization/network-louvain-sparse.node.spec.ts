// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
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
  createVectorView,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {
  createNetworkLouvainQuotientFixture,
  createNetworkLouvainMixedDuplicateFixture
} from './network-louvain-scalability-fixtures';
import {getSparseNetworkLouvainNodes} from '../../../src/gpu-network/network-optimization/network-louvain-sparse';
import {
  auditWGSLModule,
  WGSL_RESERVED_WORDS
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

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

it('command allowances cover empty, old and binding-sized sparse shapes without GPU allocation', () => {
  for (const [nodeCount, edgeCount] of [
    [1, 0],
    [65535, 1048576],
    [65536, 1048577],
    [8388608, 8388608]
  ]) {
    const {device, graph, contributor} = createContributor(nodeCount, edgeCount, 2, 1);
    try {
      const commands = contributor.getCommandNodes(graph);
      const firstRound = commands.findIndex(
        command => command.id === 'sparse-louvain-level-0-round-0-gate'
      );
      expect(firstRound).toBeGreaterThan(0);
      expect(firstRound).toBeLessThanOrEqual(352);
      expect(commands.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      expect(
        commands.filter(command => command.id.includes('level-0-round-0')).length
      ).toBeLessThanOrEqual(320);
      expect(
        commands.filter(command => command.id.includes('aggregate-0')).length
      ).toBeLessThanOrEqual(208);
    } finally {
      device.destroy();
    }
  }
});

it('construction enforces uint32 addressing, round and command budgets by name', () => {
  expect(() => createContributor(1073741824, 1)).toThrow(/sparse-louvain.*nodeCount/);
  expect(() => createContributor(2, 1073741824)).toThrow(
    /sparse-louvain.*topology.neighbors.length/
  );
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

it('checks two-word edge and four-word node scratch before creating command resources', () => {
  for (const [nodeCount, edgeCount, scratch] of [
    [65, 1, 'node'],
    [2, 65, 'edge']
  ] as const) {
    const {device, graph, contributor} = createContributor(nodeCount, edgeCount);
    Object.defineProperty(device, 'limits', {
      value: {...device.limits, maxStorageBufferBindingSize: 1024}
    });
    const createBuffer = vi.spyOn(graph, 'createTransientBuffer');
    try {
      expect(() => contributor.getCommandNodes(graph)).toThrow(
        new RegExp(`sparse-louvain sparse ${scratch} scratch needs 1040 bytes.*limit 1024`)
      );
      expect(createBuffer).not.toHaveBeenCalled();
    } finally {
      device.destroy();
    }
  }
});

it('topology selects the sparse route at small and large sizes; dense retains its 32-node cap', () => {
  for (const nodeCount of [4, 65536]) {
    const {device, graph, contributor} = createContributor(nodeCount, 2);
    try {
      const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
      expect(kernels.some(kernel => kernel.id.endsWith('-initialize-edges'))).toBe(true);
      const {topology: unusedTopology, ...props} = contributor.props;
      const denseProps = {
        ...props,
        weights: createTransientView(graph, 'dense-weights', 'float32', nodeCount ** 2)
      };
      if (nodeCount > 32) {
        expect(() => new GPUNetworkLouvain(denseProps)).toThrow(/sparse-louvain.*nodeCount.*32/);
      } else {
        expect(new GPUNetworkLouvain(denseProps).estimatedCommandNodeCount).toBe(2);
      }
    } finally {
      device.destroy();
    }
  }
});

it('two-word initialization keeps high vertex IDs and padding separate', () => {
  const {device, graph, contributor} = createContributor(131072, 3);
  try {
    const kernel = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).find(
      candidate => candidate.id.endsWith('-initialize-edges')
    )!;
    const source = kernel.source
      .replace(/^\s*_\s*=[^;]*;/gm, '')
      .replace(/@workgroup_size\(\s*(\d+)u?\s*\)/g, '@workgroup_size($1, 1, 1)');
    const keys = new Uint32Array(12);
    const weights = new Float32Array(6);
    new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
      'main',
      1,
      {
        0: {
          0: Uint32Array.of(32768, 65536, 131071),
          1: Uint32Array.of(0, 65537, 131071),
          2: Float32Array.of(1, 1, 0),
          3: Float32Array.of(2),
          4: keys,
          5: weights
        }
      },
      {constants: kernel.constants}
    );
    // The first pair would overflow the old packed key to zero at this node count.
    expect(Array.from(keys)).toEqual([
      0, 32768, 32768, 0, 65537, 65536, 65536, 65537, 131072, 131072, 131072, 131072
    ]);
    expect(Array.from(weights)).toEqual([0.5, 0.5, 0.5, 0.5, 0, 0]);
  } finally {
    device.destroy();
  }
});

it('command allowances cover the uint32 addressing bounds and the default large-graph schedule', () => {
  for (const [nodeCount, edgeCount, maximumLevels, maximumIterations] of [
    [1073741823, 1073741823, 2, 1],
    [8388608, 8388608, 8, 16]
  ]) {
    const {device, graph, contributor} = createContributor(
      nodeCount,
      edgeCount,
      maximumLevels,
      maximumIterations
    );
    // Describe the theoretical uint32 shape only; no buffers are allocated. Real devices
    // reject it at the binding check. The 128 MiB shape uses the actual default limit.
    if (nodeCount === 1073741823) {
      Object.defineProperty(device, 'limits', {
        value: {...device.limits, maxStorageBufferBindingSize: 16 * nodeCount}
      });
    }
    try {
      const commands = contributor.getCommandNodes(graph);
      expect(commands.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      const firstRound = commands.findIndex(
        command => command.id === 'sparse-louvain-level-0-round-0-gate'
      );
      expect(firstRound).toBeGreaterThan(0);
      expect(firstRound).toBeLessThanOrEqual(352);
      if (nodeCount === 1073741823) expect(firstRound).toBe(348);
      expect(
        commands.filter(command => command.id.includes('level-0-round-0')).length
      ).toBeLessThanOrEqual(320);
      expect(
        commands.filter(command => command.id.includes('aggregate-0')).length
      ).toBeLessThanOrEqual(208);
    } finally {
      device.destroy();
    }
  }
});

it('two-word symmetry lookup distinguishes reverse keys above the old packing limit', () => {
  const {device, graph, contributor} = createContributor(131072, 3);
  try {
    const kernel = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).find(
      candidate => candidate.id.endsWith('-symmetry-validate')
    )!;
    const source = kernel.source
      .replace(/^\s*_\s*=[^;]*;/gm, '')
      .replace(/@workgroup_size\(\s*(\d+)u?\s*\)/g, '@workgroup_size($1, 1, 1)');
    const keys = Uint32Array.of(
      32768,
      0,
      0,
      32768,
      65537,
      65536,
      65536,
      65537,
      131072,
      131072,
      131072,
      131072
    );
    for (const asymmetric of [false, true]) {
      const failures = new Uint32Array(6);
      new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
        'main',
        1,
        {
          0: {
            0: keys,
            1: Float32Array.of(1, 1, asymmetric ? 2 : 1, 1, 0, 0),
            2: failures
          }
        },
        {constants: kernel.constants}
      );
      expect(Array.from(failures)).toEqual(asymmetric ? [0, 0, 1, 1, 0, 0] : [0, 0, 0, 0, 0, 0]);
    }
  } finally {
    device.destroy();
  }
});

it('audits every generated identifier against the full W3C reserved-word list in both key encodings', () => {
  // The shared audit list was checked against WGSL 16.2 (2026-11-06), rather than the
  // interpreter's token list. Keep the caller-requested `sample` guard as well.
  for (const reserved of ['target', 'active', 'filter', 'module', 'meta', 'NULL', 'yield']) {
    expect(WGSL_RESERVED_WORDS.has(reserved), reserved).toBe(true);
  }
  for (const [nodeCount, encoding] of [
    [32, 'dense'],
    [32, 'dense-seeded'],
    [64, 'packed-reference'],
    [65535, 'two-word'],
    [65536, 'two-word'],
    [65537, 'two-word']
  ] as const) {
    const {device, graph, contributor} = createContributor(nodeCount, 384, 2, 2);
    try {
      const props = contributor.props;
      if (!props.topology) throw new Error('expected CSR props');
      const nodes =
        encoding === 'dense' || encoding === 'dense-seeded'
          ? new GPUNetworkLouvain({
              ...props,
              topology: undefined,
              weights: createTransientView(
                graph,
                'dense-reserved-word-weights',
                'float32',
                nodeCount * nodeCount
              ),
              seed: encoding === 'dense-seeded' ? 42 : undefined
            }).getCommandNodes(graph)
          : getSparseNetworkLouvainNodes(graph, contributor.id, props, encoding);
      for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
        const tokens =
          kernel.source
            .replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '')
            .match(/\b[A-Za-z_][A-Za-z_0-9]*\b/g) ?? [];
        expect(
          tokens.filter(token => WGSL_RESERVED_WORDS.has(token) || token === 'sample'),
          kernel.id
        ).toEqual([]);
      }
    } finally {
      device.destroy();
    }
  }
});

it('the internal packed-key reference retains the old named node limit', () => {
  const {device, graph, contributor} = createContributor(65536, 1);
  try {
    const props = contributor.props;
    if (!props.topology) throw new Error('expected CSR props');
    expect(() =>
      getSparseNetworkLouvainNodes(graph, contributor.id, props, 'packed-reference')
    ).toThrow(/sparse-louvain.*packed-reference nodeCount.*65535/);
  } finally {
    device.destroy();
  }
});

it('weighted scalability fixtures require an improving move after quotient aggregation', () => {
  const fixture = createNetworkLouvainQuotientFixture(16);
  const input = {topology: fixture, parameters: Float32Array.of(fixture.resolution, 1e-8)};
  const firstLevel = computeNetworkLouvainOnCPU(input, {
    nodeCount: 16,
    maximumLevels: 1,
    maximumIterations: 4
  });
  const multilevel = computeNetworkLouvainOnCPU(input, {
    nodeCount: 16,
    maximumLevels: 3,
    maximumIterations: 4
  });
  expect(new Set(firstLevel.communities).size).toBe(8);
  expect(new Set(multilevel.communities).size).toBe(4);
  expect(Array.from(multilevel.summary)).toEqual([1, 1, 3, 5]);
  expect(multilevel.objectiveTrace[3]).toBeGreaterThan(multilevel.objectiveTrace[2]);
  expect(multilevel.objectiveTrace[5]).toBeCloseTo(0, 7);

  const duplicates = createNetworkLouvainMixedDuplicateFixture();
  expect(duplicates.offsets[1]).toBe(3 * 384);
  expect(Math.max(...duplicates.weights) / Math.min(...duplicates.weights)).toBeGreaterThan(10000);
  expect(duplicates.weights.reduce((sum, weight) => sum + weight, 0)).toBe(18);
  for (const nodeCount of [65535, 65536, 65537, 131072]) {
    const large = createNetworkLouvainQuotientFixture(nodeCount, {
      duplicateArcs: true,
      selfLoops: true
    });
    expect(large.offsets[nodeCount]).toBe(large.neighbors.length);
    expect(large.neighbors.includes(nodeCount - 1)).toBe(true);
  }
});

it('distinct shader and pipeline keys remain constant as sparse rounds and levels grow', () => {
  const sourceCounts: number[] = [];
  const pipelineCounts: number[] = [];
  for (const [maximumLevels, maximumIterations] of [
    [2, 2],
    [3, 4],
    [4, 6]
  ]) {
    const {device, graph, contributor} = createContributor(
      64,
      192,
      maximumLevels,
      maximumIterations
    );
    try {
      const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
      const sources = new Set(kernels.map(kernel => kernel.source));
      const pipelines = new Set(
        kernels.map(kernel =>
          JSON.stringify([
            kernel.source,
            Object.entries(kernel.constants).sort(([left], [right]) => left.localeCompare(right)),
            kernel.shaderLayout
          ])
        )
      );
      sourceCounts.push(sources.size);
      pipelineCounts.push(pipelines.size);
      for (const kernel of kernels)
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    } finally {
      device.destroy();
    }
  }
  expect(new Set(sourceCounts).size, `shader counts: ${sourceCounts}`).toBe(1);
  expect(new Set(pipelineCounts).size, `pipeline counts: ${pipelineCounts}`).toBe(1);
});

it('dense schedule budgets also use bound lengths without source or pipeline variants', () => {
  const sources = new Set<string>();
  const pipelines = new Set<string>();
  for (const [maximumLevels, maximumIterations] of [
    [2, 2],
    [3, 4],
    [4, 6]
  ]) {
    const {device, graph, contributor} = createContributor(4, 4, maximumLevels, maximumIterations);
    try {
      const dense = new GPUNetworkLouvain({
        ...contributor.props,
        topology: undefined,
        weights: createTransientView(graph, 'dense-schedule-weights', 'float32', 16)
      });
      for (const kernel of getGPUCommandNodeKernelSources(dense.getCommandNodes(graph))) {
        sources.add(kernel.source);
        pipelines.add(JSON.stringify([kernel.source, kernel.constants, kernel.shaderLayout]));
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
      }
    } finally {
      device.destroy();
    }
  }
  // One dense solver and one named-status publisher, independent of schedule lengths.
  expect(sources.size).toBe(2);
  expect(pipelines.size).toBe(2);
});

it('checks aligned input/output ranges and bounded dispatch scratch without changing the graph', () => {
  for (const field of [
    'parameters',
    'offsets',
    'neighbors',
    'weights',
    'nodePositions',
    'communities',
    'objectiveTrace',
    'summary',
    'status',
    'dispatch'
  ] as const) {
    const {device, graph, contributor} = createContributor(2, 2);
    const prefixed = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      format: Format,
      length: number
    ) => {
      const buffer = createTransientView(graph, `prefixed-${field}`, 'uint32', 80).buffer;
      return graph.createDataView(buffer, {format, byteOffset: 252, length});
    };
    const base = contributor.props;
    if (!base.topology) throw new Error('expected CSR props');
    const props = {
      ...base,
      parameters: field === 'parameters' ? prefixed('float32', 2) : base.parameters,
      topology: {
        ...base.topology,
        offsets: field === 'offsets' ? prefixed('uint32', 3) : base.topology.offsets,
        neighbors: field === 'neighbors' ? prefixed('uint32', 2) : base.topology.neighbors,
        weights: field === 'weights' ? prefixed('float32', 2) : base.topology.weights,
        nodePositions: field === 'nodePositions' ? prefixed('float32x2', 2) : undefined
      },
      output: {
        ...base.output,
        communities: field === 'communities' ? prefixed('uint32', 2) : base.output.communities,
        objectiveTrace:
          field === 'objectiveTrace' ? prefixed('float32', 2) : base.output.objectiveTrace,
        summary: field === 'summary' ? prefixed('uint32', 4) : base.output.summary,
        status: field === 'status' ? {converged: prefixed('uint32', 1)} : base.output.status
      }
    };
    const candidate = new GPUNetworkLouvain(props);
    Object.defineProperty(device, 'limits', {
      value: {
        ...device.limits,
        maxStorageBufferBindingSize: field === 'dispatch' ? 1024 : 255
      }
    });
    const declarations = [
      vi.spyOn(graph, 'createTransientBuffer'),
      vi.spyOn(graph, 'createDataView'),
      vi.spyOn(graph, 'importBuffer'),
      vi.spyOn(graph, 'getUniqueResourceId'),
      vi.spyOn(graph, 'add')
    ];
    try {
      expect(() => candidate.getCommandNodes(graph)).toThrow(
        new RegExp(
          field === 'dispatch'
            ? 'sparse dispatch gate allowance.*3840 bytes'
            : `${field === 'status' ? 'output.status.converged' : field}.*needs (256|260|264|268) bytes`
        )
      );
      for (const declaration of declarations) expect(declaration).not.toHaveBeenCalled();
    } finally {
      device.destroy();
    }
  }
});

it('dense aligned binding failures also precede all resource declarations', () => {
  const {device, graph, contributor} = createContributor(4, 4);
  const buffer = createTransientView(graph, 'prefixed-dense-buffer', 'uint32', 80).buffer;
  const dense = new GPUNetworkLouvain({
    ...contributor.props,
    topology: undefined,
    weights: graph.createDataView(buffer, {format: 'float32', byteOffset: 252, length: 16})
  });
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBufferBindingSize: 300}
  });
  const createBuffer = vi.spyOn(graph, 'createTransientBuffer');
  const createDataView = vi.spyOn(graph, 'createDataView');
  try {
    expect(() => dense.getCommandNodes(graph)).toThrow(/sparse-louvain weights needs 316 bytes/);
    expect(createBuffer).not.toHaveBeenCalled();
    expect(createDataView).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('measures the graph-node budget near the contributor maximum without allocating GPU storage', () => {
  for (const [maximumLevels, maximumIterations] of [
    [8, 16],
    [16, 12]
  ]) {
    const {device, graph, contributor} = createContributor(
      1073741823,
      1073741823,
      maximumLevels,
      maximumIterations
    );
    Object.defineProperty(device, 'limits', {
      value: {...device.limits, maxStorageBufferBindingSize: 16 * 1073741823}
    });
    try {
      const commands = contributor.getCommandNodes(graph);
      const initialization = commands.findIndex(
        command => command.id === 'sparse-louvain-level-0-round-0-gate'
      );
      const round = commands.filter(command =>
        command.id.includes('sparse-louvain-level-0-round-0')
      ).length;
      const aggregation = commands.filter(command =>
        command.id.includes('sparse-louvain-aggregate-0')
      ).length;
      expect(initialization).toBe(348);
      expect(round).toBe(264);
      expect(aggregation).toBe(163);
      const expected = maximumLevels === 8 ? 35283 : 53483;
      expect(commands.length).toBe(expected);
      expect(commands.length).toBe(
        initialization +
          maximumLevels * maximumIterations * round +
          (maximumLevels - 1) * aggregation +
          2
      );
      expect(expected).toBeGreaterThan(32768);
      expect(commands.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    } finally {
      device.destroy();
    }
  }
});

it('runtime round/level control WGSL produces independently specified stopping states', () => {
  const {device, graph, contributor} = createContributor(4, 4, 3, 4);
  try {
    const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    const round = kernels.find(kernel => kernel.id.endsWith('level-0-round-0-control'))!;
    const aggregate = kernels.find(kernel => kernel.id.endsWith('aggregate-0-control'))!;
    const run = (kernel: typeof round, state: Uint32Array, anyChanged?: number) => {
      const source = kernel.source
        .replace(/^\s*_\s*=[^;]*;/gm, '')
        .replace(/@workgroup_size\(\s*(\d+)u?\s*\)/g, '@workgroup_size($1, 1, 1)');
      new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
        'main',
        1,
        {
          0:
            anyChanged === undefined
              ? {0: state}
              : {
                  0: state,
                  1: Uint32Array.of(anyChanged),
                  2: new Uint32Array(4),
                  3: new Uint32Array(3)
                }
        },
        {constants: kernel.constants}
      );
    };
    for (const scene of [
      {
        name: 'initially stable',
        before: [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        changed: 0,
        after: [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
      },
      {
        name: 'round budget exhausted after improvement',
        before: [1, 1, 1, 0, 1, 0, 2, 4, 3, 1],
        changed: 1,
        after: [0, 1, 1, 0, 1, 0, 2, 5, 4, 1]
      },
      {
        name: 'last level stabilizes after changes',
        before: [1, 1, 1, 0, 1, 0, 3, 4, 1, 2],
        changed: 0,
        after: [0, 0, 1, 0, 1, 0, 3, 5, 2, 2]
      },
      {
        name: 'inactive control is inert',
        before: [0, 1, 1, 0, 1, 0, 2, 4, 3, 1],
        changed: 1,
        after: [0, 1, 1, 0, 1, 0, 2, 4, 3, 1]
      }
    ]) {
      const state = Uint32Array.from(scene.before);
      run(round, state, scene.changed);
      expect(Array.from(state), scene.name).toEqual(scene.after);
    }
    const state = Uint32Array.of(1, 1, 1, 0, 1, 0, 1, 1, 1, 0);
    run(round, state, 0);
    expect(Array.from(state)).toEqual([1, 0, 1, 0, 1, 0, 1, 2, 2, 0]);
    run(aggregate, state);
    expect(Array.from(state)).toEqual([1, 1, 0, 0, 1, 0, 1, 2, 0, 1]);
    run(round, state, 0);
    expect(Array.from(state)).toEqual([0, 0, 0, 0, 1, 1, 2, 3, 1, 1]);
  } finally {
    device.destroy();
  }
});

it('rejects chunked writable outputs at construction before inspecting buffer identities', () => {
  const {device, graph, contributor} = createContributor(4, 4);
  try {
    for (const field of ['communities', 'objectiveTrace', 'summary', 'status'] as const) {
      const format = field === 'objectiveTrace' ? 'float32' : 'uint32';
      const chunked = createVectorView(`chunked-${field}`, format, [
        createTransientView(graph, `chunked-${field}-first`, format, 1),
        createTransientView(graph, `chunked-${field}-second`, format, 1)
      ]);
      const output =
        field === 'status'
          ? {...contributor.props.output, status: {converged: chunked}}
          : {...contributor.props.output, [field]: chunked};
      expect(
        () =>
          new GPUNetworkLouvain({
            ...contributor.props,
            output: output as unknown as typeof contributor.props.output
          })
      ).toThrow(new RegExp(`sparse-louvain.*${field === 'status' ? '' : field}.*one packed`));
    }
  } finally {
    device.destroy();
  }
});
