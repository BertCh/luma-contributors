// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {
  GPUCommandGraph,
  GraphVectorView,
  createTransientView,
  getViewBindingRange,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPUNetworkFlow,
  type GPUNetworkFlowProps
} from '../../../src/gpu-network/network-optimization/gpu-network-flow';
import {GPUNetworkFlowOperation} from '../../../src/gpu-network/network-optimization/gpu-network-flow-operation';
import {createNetworkFlowStorageReadbackNode} from './network-flow-storage-test-utils';
import {expectNetworkFlowFeasible} from './network-flow-feasibility-test-utils';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {computeNetworkFlowOnCPU} from '../../../src/gpu-network/network-optimization/cpu-network-flow-operation';
import primalDualOracle from './network-flow-primal-dual-oracle.json';
import movementOracle from '../../oracles/fixtures/network-movement-coverage.json';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getPrivateMemoryBytes,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

function createContributor(
  nodeCount = 6,
  edgeCount = 8,
  options: Partial<GPUNetworkFlowProps> = {}
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const props: GPUNetworkFlowProps = {
    id: 'parallel-flow',
    solver: 'parallel',
    nodeCount,
    edges: createTransientView(graph, 'edges', 'uint32x2', edgeCount),
    capacities: createTransientView(graph, 'capacities', 'uint32', edgeCount),
    costs: createTransientView(graph, 'costs', 'uint32', edgeCount),
    parameters: createTransientView(graph, 'parameters', 'uint32', 3),
    maximumRelaxationRounds: Math.min(nodeCount, 8),
    output: {
      flow: createTransientView(graph, 'flow', 'uint32', edgeCount),
      cut: createTransientView(graph, 'cut', 'uint32', nodeCount),
      summary: createTransientView(graph, 'summary', 'uint32', 8),
      exactCost: createTransientView(graph, 'exact-cost', 'uint32x2', 1),
      status: {
        invalidCount: createTransientView(graph, 'invalid', 'uint32', 1),
        converged: createTransientView(graph, 'converged', 'uint32', 1),
        iterationCount: createTransientView(graph, 'iterations', 'uint32', 1),
        iterationLimitReached: createTransientView(graph, 'limited', 'uint32', 1)
      }
    },
    ...options
  };
  return {device, graph, props, contributor: new GPUNetworkFlow(props)};
}

it('parallel flow kernels parse and stay within eight bindings without private vertex tables', () => {
  const {device, graph, contributor} = createContributor();
  try {
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
      expect(getPrivateMemoryBytes(kernel.source), kernel.id).toBeLessThanOrEqual(64);
    }
    for (const node of nodes.filter(node =>
      /-(relax-\d+|double-\d+|mark-\d+|augment)$/.test(node.id)
    )) {
      expect(node.condition?.source, node.id).toBe('gpu');
    }
  } finally {
    device.destroy();
  }
});

it('actual command counts fit the estimate at empty, 100k and maximum supported shapes', () => {
  for (const [nodeCount, edgeCount] of [
    [2, 0],
    [100_000, 400_000],
    [30_000_000, 30_000_000]
  ]) {
    const {device, graph, contributor} = createContributor(nodeCount, edgeCount, {
      maximumAugmentations: 0,
      maximumRelaxationRounds: 1
    });
    try {
      // No allocation: measure the public primitive's schedule with a sufficiently large limit.
      Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 2 ** 31});
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    } finally {
      device.destroy();
    }
  }
});

it('keeps reference caps, selects large graphs automatically and rejects unsafe schedules/bindings', () => {
  const small = createContributor(6, 8, {solver: 'auto', maximumRelaxationRounds: undefined});
  expect(small.contributor.solver).toBe('reference');
  small.device.destroy();
  const large = createContributor(100_000, 200_000, {solver: 'auto'});
  expect(large.contributor.solver).toBe('parallel');
  large.device.destroy();
  expect(() => createContributor(129, 8, {solver: 'reference'})).toThrow(
    /parallel-flow nodeCount.*128/
  );
  expect(() => createContributor(6, 1025, {solver: 'reference'})).toThrow(
    /parallel-flow edges rows.*1024/
  );
  expect(() => createContributor(30_000_001, 0)).toThrow(/parallel-flow nodeCount.*30000000/);
  expect(() => createContributor(6, 8, {maximumRelaxationRounds: 8193})).toThrow(
    /parallel-flow maximumRelaxationRounds.*8192/
  );
  expect(() =>
    createContributor(6, 8, {maximumAugmentations: 100, maximumRelaxationRounds: 1024})
  ).toThrow(/parallel-flow estimated command node count.*32768/);
  const oversized = createContributor(9_000_000, 1);
  try {
    expect(() => oversized.contributor.getCommandNodes(oversized.graph)).toThrow(
      /parallel-flow.*scratch.*limit/
    );
    expect(
      getWGSLKernelNodeRecord(oversized.graph, 'parallel-flow-initialize-state')
    ).toBeUndefined();
  } finally {
    oversized.device.destroy();
  }
});

/** Executes author-written WGSL in command order. The published GPUScan is replaced by a CPU
 * exclusive scan in this interpreter-only harness; headless specs exercise the real primitive.
 * This catches state/parity, predecessor, exact objective and budget errors without a GPU. */
function executeScene(options: {
  nodeCount: number;
  edges: number[];
  capacities: number[];
  costs?: number[];
  required?: number;
  maximumAugmentations?: number;
  maximumRelaxationRounds?: number;
  maximumRelabels?: number;
  uncosted?: boolean;
  mode?: GPUNetworkFlowProps['mode'];
  solver?: GPUNetworkFlowProps['solver'];
  maximumCostPhases?: number;
  source?: number;
  sink?: number;
}) {
  const scene = createContributor(options.nodeCount, options.capacities.length, {
    ...(options.uncosted ? {costs: undefined} : {}),
    solver: options.solver ?? 'parallel',
    ...(options.mode ? {mode: options.mode} : {}),
    maximumCostPhases: options.maximumCostPhases,
    maximumRelabels: options.maximumRelabels,
    maximumAugmentations: options.maximumAugmentations,
    maximumRelaxationRounds: options.maximumRelaxationRounds
  });
  const {device, graph, props, contributor} = scene;
  const data = new Map<GraphDataView, Uint32Array>();
  const buffers = new Map<GraphDataView['buffer'], Uint32Array>();
  const values = (view: GraphDataView): Uint32Array => {
    if (!data.has(view)) {
      const words = view.length * (view.format === 'uint32x2' ? 2 : 1);
      const array = new Uint32Array(Math.max(words, 1));
      data.set(view, array);
      buffers.set(view.buffer, array);
    }
    return data.get(view)!;
  };
  values(props.edges).set(options.edges);
  values(props.capacities).set(options.capacities);
  if (props.costs)
    values(props.costs).set(options.costs ?? new Array(options.capacities.length).fill(0));
  values(props.parameters).set([
    options.source ?? 0,
    options.sink ?? options.nodeCount - 1,
    options.required ?? 0xffffffff
  ]);
  try {
    const nodes = contributor.getCommandNodes(graph);
    let scanned = false;
    for (const node of nodes) {
      const record = getWGSLKernelNodeRecord(graph, node.id);
      if (!record) {
        if (!scanned && node.id.includes('residual-scan')) {
          const counts = getWGSLKernelNodeRecord(
            graph,
            'parallel-flow-initialize-counts'
          )!.bindings.find(binding => binding.name === 'counts')!.view;
          const cursorRecord = getWGSLKernelNodeRecord(graph, 'parallel-flow-initialize-cursors')!;
          const offsets = cursorRecord.bindings.find(binding => binding.name === 'offsets')!.view;
          let total = 0;
          for (let row = 0; row <= options.nodeCount; row++) {
            values(offsets)[row] = total;
            total += values(counts)[row];
          }
          scanned = true;
        }
        continue;
      }
      if (node.condition?.source === 'gpu' && node.condition.mode === 'indirect') {
        if (buffers.get(node.condition.buffer)?.[(node.condition.byteOffset ?? 0) / 4] === 0)
          continue;
      }
      const bindings = Object.fromEntries(
        record.bindings.map((binding, index) => [index, values(binding.view)])
      );
      const source = record.source
        .replace(/^\s*_\s*=[^;]*;/gm, '')
        .replace(/@workgroup_size\(\s*(\d+)u?\s*\)/g, '@workgroup_size($1, 1, 1)');
      new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
        'main',
        [...record.workgroups],
        {0: bindings},
        {constants: record.constants}
      );
    }
    return {
      flow: Array.from(values(props.output.flow)),
      cut: Array.from(values(props.output.cut)),
      summary: Array.from(values(props.output.summary)),
      exactCost: Array.from(values(props.output.exactCost!)),
      converged: values(props.output.status!.converged!)[0],
      limited: values(props.output.status!.iterationLimitReached!)[0],
      invalid: values(props.output.status!.invalidCount!)[0]
    };
  } finally {
    device.destroy();
  }
}

it('the shared GPU conservation oracle accepts zero and positive integral flow exactly', () => {
  const scene = {
    nodeCount: 3,
    edges: [2, 1, 1, 0],
    capacities: [2, 2],
    source: 2,
    sink: 0
  };
  // These call the same assertion as the headless budget cases. The zero case failed
  // there because the old oracle compared an initialized +0 balance with -value (-0).
  expectNetworkFlowFeasible(scene, [0, 0], 0);
  expectNetworkFlowFeasible(scene, [1, 1], 1);
  expectNetworkFlowFeasible(scene, [2, 2], 2);
  expectNetworkFlowFeasible({nodeCount: 3, edges: [], capacities: []}, [], 0);
});

it('the shared conservation oracle still rejects capacity, integral and balance violations', () => {
  const scene = {nodeCount: 3, edges: [0, 1, 1, 2], capacities: [1, 1]};
  for (const [flow, value] of [
    [[1, 1], 0],
    [[0, 0], 1],
    [[1, 0], 1],
    [[0, 1], 1],
    [[2, 2], 2],
    [[-1, -1], -1],
    [[0.5, 0.5], 0.5],
    [[1], 1]
  ] as const) {
    expect(() => expectNetworkFlowFeasible(scene, flow, value)).toThrow();
  }
  // Both terminal balances are correct here, but the two interior balances are not.
  expect(() =>
    expectNetworkFlowFeasible({nodeCount: 4, edges: [0, 1, 2, 3], capacities: [1, 1]}, [1, 1], 1)
  ).toThrow();
});

it('generated WGSL reroutes through reverse arcs to the hand-computed minimum-cost maximum flow', () => {
  const result = executeScene({
    nodeCount: 6,
    edges: [0, 1, 0, 2, 1, 3, 1, 4, 2, 3, 2, 4, 3, 5, 4, 5],
    capacities: new Array(8).fill(1),
    costs: [0, 0, 1, 2, 2, 100, 0, 0]
  });
  expect(result.flow).toEqual([1, 1, 0, 1, 1, 0, 1, 1]);
  expect(result.summary.slice(0, 5)).toEqual([1, 1, 2, 0, 2]);
  expect(result.exactCost).toEqual([4, 0]);
  expect(result.summary.slice(6)).toEqual([2, 1]);
  expect([result.converged, result.limited, result.invalid]).toEqual([1, 0, 0]);
});

it('publishes feasible partial flow at either iteration budget and certifies completion on the final search', () => {
  const scene = {nodeCount: 4, edges: [0, 1, 1, 3, 0, 2, 2, 3], capacities: [1, 1, 1, 1]};
  const limited = executeScene({...scene, maximumAugmentations: 1});
  expect(limited.flow).toEqual([1, 1, 0, 0]);
  expectNetworkFlowFeasible(scene, limited.flow, limited.summary[2]);
  expect(limited.summary.slice(0, 5)).toEqual([1, 0, 1, 0, 1]);
  expect([limited.converged, limited.limited]).toEqual([0, 1]);
  const shallow = executeScene({...scene, maximumRelaxationRounds: 1});
  expect(shallow.flow).toEqual([0, 0, 0, 0]);
  expect(shallow.summary[2]).toBe(0);
  expectNetworkFlowFeasible(scene, shallow.flow, shallow.summary[2]);
  expect([shallow.converged, shallow.limited]).toEqual([0, 1]);
  expect(shallow.summary[7]).toBe(0);
  const stopped = executeScene({...scene, maximumAugmentations: 0});
  expect(stopped.flow).toEqual([0, 0, 0, 0]);
  expect(stopped.summary[2]).toBe(0);
  expect([stopped.converged, stopped.limited, stopped.invalid]).toEqual([0, 1, 0]);
  expectNetworkFlowFeasible(scene, stopped.flow, stopped.summary[2]);
  const finished = executeScene({...scene, maximumAugmentations: 2});
  expect(finished.flow).toEqual([1, 1, 1, 1]);
  expectNetworkFlowFeasible(scene, finished.flow, finished.summary[2]);
  expect(finished.summary[7]).toBe(1);
  expect([finished.converged, finished.limited]).toEqual([1, 0]);
  const finite = executeScene({...scene, required: 1, maximumRelaxationRounds: 3});
  expect(finite.summary.slice(0, 4)).toEqual([1, 1, 1, 0]);
  expectNetworkFlowFeasible(scene, finite.flow, finite.summary[2]);
  expect(finite.cut).toEqual([1, 1, 1, 1]);
});

it('handles exact wide costs, invalid contents, disconnected graphs and zero augmentation budgets', () => {
  const wide = executeScene({nodeCount: 2, edges: [0, 1], capacities: [0x800000], costs: [32767]});
  const expected = 0x800000n * 32767n;
  expect(wide.exactCost).toEqual([Number(expected & 0xffffffffn), Number(expected >> 32n)]);
  const invalid = executeScene({nodeCount: 2, edges: [0, 3], capacities: [1]});
  expect([invalid.invalid, invalid.converged, invalid.limited]).toEqual([1, 0, 0]);
  const disconnected = executeScene({
    nodeCount: 3,
    edges: [0, 1],
    capacities: [2],
    maximumAugmentations: 0
  });
  expect(disconnected.summary.slice(0, 5)).toEqual([1, 1, 0, 0, 0]);
  expect(disconnected.summary[7]).toBe(1);
  const limited = executeScene({
    nodeCount: 2,
    edges: [0, 1],
    capacities: [2],
    maximumAugmentations: 0
  });
  expect([limited.limited, limited.converged]).toEqual([1, 0]);
});

it('parallel capacity totals use the full unsigned output range and reject overflow', () => {
  const large = executeScene({nodeCount: 2, edges: [0, 1], capacities: [0x80000000]});
  expect(large.flow).toEqual([0x80000000]);
  expect(large.summary[2]).toBe(0x80000000);
  expect([large.converged, large.invalid]).toEqual([1, 0]);
  const overflow = executeScene({
    nodeCount: 2,
    edges: [0, 1, 0, 1],
    capacities: [0x80000000, 0x80000000]
  });
  expect([overflow.invalid, overflow.converged, overflow.limited]).toEqual([1, 0, 0]);
});

it('generated parallel WGSL agrees with the BigInt CPU oracle on varied small residual graphs', () => {
  let randomState = 1729;
  const random = (maximum: number): number => {
    randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
    return randomState % maximum;
  };
  for (let index = 0; index < 12; index++) {
    const nodeCount = 3 + random(4);
    const edges = [0, 1, 1, nodeCount - 1];
    const capacities = [2, 2];
    const costs = [random(8), random(8)];
    for (let edge = 0; edge < 10; edge++) {
      edges.push(random(nodeCount), random(nodeCount));
      capacities.push(random(3));
      costs.push(random(8));
    }
    const required = index % 3 === 0 ? 1 : index % 3 === 1 ? 100 : 0xffffffff;
    const result = executeScene({
      nodeCount,
      edges,
      capacities,
      costs,
      required,
      maximumAugmentations: 8
    });
    const expected = computeNetworkFlowOnCPU(
      {
        edges: Uint32Array.from(edges),
        capacities: Uint32Array.from(capacities),
        costs: Uint32Array.from(costs),
        parameters: Uint32Array.of(0, nodeCount - 1, required)
      },
      {nodeCount}
    );
    expect(result.summary.slice(0, 4), `scene ${index}`).toEqual(
      Array.from(expected.summary.slice(0, 4))
    );
    expect(result.exactCost, `scene ${index}`).toEqual(Array.from(expected.exactCost));
    expect([result.converged, result.limited, result.invalid]).toEqual([1, 0, 0]);
  }
});

it('a zero finite request is complete even when the optional residual cut search is shallow', () => {
  const zero = executeScene({
    nodeCount: 4,
    edges: [0, 1, 1, 2, 2, 3],
    capacities: [1, 1, 1],
    required: 0,
    maximumRelaxationRounds: 1
  });
  expect(zero.flow).toEqual([0, 0, 0]);
  expect(zero.summary.slice(0, 5)).toEqual([1, 1, 0, 0, 0]);
  expect([zero.converged, zero.limited, zero.invalid]).toEqual([1, 0, 0]);
  expect(zero.summary[7]).toBe(0);
});

it('all generated identifiers avoid the full W3C reserved list, including the reference route', () => {
  for (const options of [
    {},
    {costs: undefined},
    {solver: 'reference' as const, maximumRelaxationRounds: undefined}
  ]) {
    const {device, graph, contributor} = createContributor(6, 8, options);
    try {
      const readback = createTransientView(graph, 'reserved-readback', 'uint32', 8);
      const nodes = [
        ...contributor.getCommandNodes(graph),
        createNetworkFlowStorageReadbackNode(
          graph,
          'reserved-readback-copy',
          contributor.props.capacities,
          readback
        )
      ];
      for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
        const code = kernel.source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
        const forbidden = [...new Set(code.match(/[A-Za-z_][A-Za-z_0-9]*/g))].filter(word =>
          WGSL_RESERVED_WORDS.has(word)
        );
        expect(forbidden, kernel.id).toEqual([]);
      }
    } finally {
      device.destroy();
    }
  }
});

it('shader and pipeline variants stay constant as iteration counts grow (packed input rejects chunks)', () => {
  for (const uncosted of [false, true]) {
    const counts: number[] = [];
    for (const rounds of [2, 3, 4]) {
      const {device, graph, contributor} = createContributor(6, 8, {
        ...(uncosted
          ? {costs: undefined, maximumRelabels: rounds}
          : {maximumAugmentations: rounds}),
        maximumRelaxationRounds: rounds
      });
      try {
        const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
        counts.push(
          new Set(kernels.map(kernel => JSON.stringify([kernel.source, kernel.constants]))).size
        );
      } finally {
        device.destroy();
      }
    }
    expect(new Set(counts).size, `pipeline counts ${counts}`).toBe(1);
  }
});

it('parallel preflow pushes complete all paths of a fan in a single batch', () => {
  const fan = executeScene({
    nodeCount: 8,
    edges: [0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 1, 7, 2, 7, 3, 7, 4, 7, 5, 7, 6, 7],
    capacities: new Array(12).fill(1),
    uncosted: true,
    maximumRelabels: 1,
    maximumRelaxationRounds: 3
  });
  expect(fan.flow).toEqual(new Array(12).fill(1));
  expect(fan.summary.slice(0, 5)).toEqual([1, 1, 6, 0, 1]);
  expect(fan.summary.slice(6)).toEqual([6, 1]);
  expect([fan.converged, fan.limited, fan.invalid]).toEqual([1, 0, 0]);
});

it('returns stranded excess before publishing a budget-limited preflow phase', () => {
  const result = executeScene({
    nodeCount: 5,
    edges: [0, 1, 0, 2, 1, 3, 2, 3, 3, 4],
    capacities: [4, 4, 4, 4, 1],
    uncosted: true,
    maximumRelabels: 1,
    maximumRelaxationRounds: 5
  });
  expect(result.summary[2]).toBe(1);
  const balances = new Int32Array(5);
  const edges = [0, 1, 0, 2, 1, 3, 2, 3, 3, 4];
  result.flow.forEach((amount, edge) => {
    balances[edges[2 * edge]] -= amount;
    balances[edges[2 * edge + 1]] += amount;
  });
  expect(Array.from(balances)).toEqual([-1, 0, 0, 0, 1]);
  expect(result.converged).toBe(1);
});

it('rejects every pair of overlapping writable output fields at construction', () => {
  const {device, graph, props} = createContributor();
  try {
    const shared = createTransientView(graph, 'overlap-output-storage', 'uint32', 128);
    const fields = [
      'flow',
      'cut',
      'summary',
      'exactCost',
      'invalidCount',
      'converged',
      'iterationCount',
      'iterationLimitReached'
    ] as const;
    const statusFields = new Set<string>(fields.slice(4));
    for (let first = 0; first < fields.length; first++) {
      for (let second = first + 1; second < fields.length; second++) {
        const output = {...props.output, status: {...props.output.status}};
        for (const name of [fields[first], fields[second]]) {
          const original = statusFields.has(name)
            ? output.status[name as keyof typeof output.status]!
            : output[name as keyof Omit<typeof output, 'status'>]!;
          const overlap = graph.createDataView(shared.buffer, {
            format: original.format,
            length: original.length,
            byteOffset: 4
          });
          if (statusFields.has(name))
            output.status[name as keyof typeof output.status] = overlap as GraphDataView<'uint32'>;
          else Object.assign(output, {[name]: overlap});
        }
        expect(
          () => new GPUNetworkFlow({...props, output}),
          `${fields[first]} / ${fields[second]}`
        ).toThrow(/parallel-flow.*distinct buffers/);
      }
    }
  } finally {
    device.destroy();
  }
});

it('preflights actual aligned ranges before declaring scratch or nodes', () => {
  const {device, graph, props} = createContributor();
  try {
    const storage = createTransientView(graph, 'offset-capacities', 'uint32', 80);
    const capacities = graph.createDataView(storage.buffer, {
      format: 'uint32',
      length: 8,
      byteOffset: 252
    });
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 64});
    const contributor = new GPUNetworkFlow({...props, capacities});
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /parallel-flow view 1 aligned binding range.*284.*limit 64/
    );
    expect(getWGSLKernelNodeRecord(graph, 'parallel-flow-initialize-state')).toBeUndefined();
  } finally {
    device.destroy();
  }
});

it('rejects chunked input at construction as chunk count grows', () => {
  const {device, graph, props} = createContributor();
  try {
    for (const chunkCount of [2, 4, 8]) {
      const chunkLength = 8 / chunkCount;
      const chunks = Array.from({length: chunkCount}, (_, index) =>
        createTransientView(graph, `chunk-${chunkCount}-${index}`, 'uint32x2', chunkLength)
      );
      const edges = new GraphVectorView({
        id: 'chunked-edges',
        name: 'chunked-edges',
        format: 'uint32x2',
        length: 8,
        valueLength: 8,
        byteStride: 8,
        stride: 1,
        rowByteLength: 8,
        data: chunks
      });
      expect(() => new GPUNetworkFlow({...props, edges: edges as never})).toThrow(
        /parallel-flow.*one packed view/
      );
    }
  } finally {
    device.destroy();
  }
});

it('operation forwards solver budgets and resolves shared topology ports', () => {
  const {device, graph, props} = createContributor();
  const operation = new GPUNetworkFlowOperation({
    nodeCount: 6,
    edgeCount: 8,
    solver: 'parallel',
    maximumRelaxationRounds: 7,
    maximumRelabels: 2,
    edges: 'edges',
    capacities: 'capacities',
    parameters: 'parameters',
    topology: {offsets: 'offsets', neighbors: 'neighbors'},
    output: {flow: 'flow', cut: 'cut', summary: 'summary'}
  });
  try {
    const topologyOffsets = createTransientView(graph, 'operation-offsets', 'uint32', 7);
    const topologyNeighbors = createTransientView(graph, 'operation-neighbors', 'uint32', 8);
    const views = {
      ...props.output,
      edges: props.edges,
      capacities: props.capacities,
      parameters: props.parameters,
      topologyOffsets,
      topologyNeighbors
    };
    const contributor = operation.createContributor(
      name => views[name as keyof typeof views] as never
    ) as GPUNetworkFlow;
    expect(contributor.solver).toBe('parallel');
    expect(contributor.props.maximumRelaxationRounds).toBe(7);
    expect(contributor.props.maximumRelabels).toBe(2);
    expect(contributor.props.topology).toEqual({
      offsets: topologyOffsets,
      neighbors: topologyNeighbors
    });
    expect(operation.inputs['topologyOffsets']).toBe('offsets');
  } finally {
    operation.destroy();
    device.destroy();
  }
});

it('commits feasible partial batches when the global relabel budget is reached', () => {
  const scene = {
    nodeCount: 5,
    edges: [0, 1, 0, 2, 1, 4, 2, 3, 3, 4],
    capacities: [1, 1, 1, 1, 1],
    uncosted: true,
    maximumRelaxationRounds: 5
  };
  const partial = executeScene({...scene, maximumRelabels: 1});
  expect(partial.summary.slice(0, 5)).toEqual([1, 0, 1, 0, 1]);
  expect(partial.flow).toEqual([1, 0, 1, 0, 0]);
  expect([partial.converged, partial.limited]).toEqual([0, 1]);
  const complete = executeScene({...scene, maximumRelabels: 2});
  expect(complete.flow).toEqual([1, 1, 1, 1, 1]);
  expect(complete.summary.slice(0, 5)).toEqual([1, 1, 2, 0, 2]);
});

it('global-relabel preflow batches match CPU max flow on small cyclic and antiparallel graphs', () => {
  let randomState = 937;
  const random = (maximum: number) => {
    randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
    return randomState % maximum;
  };
  for (let scene = 0; scene < 10; scene++) {
    const nodeCount = 4 + random(3);
    const edges = [0, 1, 1, nodeCount - 1];
    const capacities = [2, 2];
    for (let edge = 0; edge < 12; edge++) {
      edges.push(random(nodeCount), random(nodeCount));
      capacities.push(random(4));
    }
    const required = scene % 2 ? 0xffffffff : 1;
    const result = executeScene({
      nodeCount,
      edges,
      capacities,
      required,
      uncosted: true,
      maximumRelabels: nodeCount
    });
    const expected = computeNetworkFlowOnCPU(
      {
        edges: Uint32Array.from(edges),
        capacities: Uint32Array.from(capacities),
        parameters: Uint32Array.of(0, nodeCount - 1, required)
      },
      {nodeCount}
    );
    expect(result.summary.slice(0, 4), `scene ${scene}`).toEqual(
      Array.from(expected.summary.slice(0, 4))
    );
    expect([result.converged, result.limited, result.invalid]).toEqual([1, 0, 0]);
    expectNetworkFlowFeasible({nodeCount, edges, capacities}, result.flow, result.summary[2]);
  }
});

it('uncosted command estimates cover maximum shapes and the 100k fan without GPU allocation', () => {
  for (const [nodeCount, edgeCount, rounds] of [
    [100000, 199996, 3],
    [30000000, 30000000, 1]
  ]) {
    const {device, graph, contributor} = createContributor(nodeCount, edgeCount, {
      costs: undefined,
      maximumRelabels: 1,
      maximumRelaxationRounds: rounds
    });
    try {
      Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 2 ** 31});
      expect(contributor.getCommandNodes(graph).length).toBeLessThanOrEqual(
        contributor.estimatedCommandNodeCount
      );
    } finally {
      device.destroy();
    }
  }
});

it('rejects physical output aliasing even through distinct graph buffer handles', () => {
  const {device, graph, props} = createContributor();
  const buffer = device.createBuffer({byteLength: 512, usage: Buffer.STORAGE});
  try {
    const first = graph.importBuffer(
      {id: 'physical-first', byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    const second = graph.importBuffer(
      {id: 'physical-second', byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    const flow = graph.createDataView(first, {format: 'uint32', length: 8, byteOffset: 260});
    const summary = graph.createDataView(second, {format: 'uint32', length: 8, byteOffset: 264});
    expect(() => new GPUNetworkFlow({...props, output: {...props.output, flow, summary}})).toThrow(
      /parallel-flow.*distinct buffers/
    );
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('rejects logically disjoint outputs whose aligned storage binding ranges overlap', () => {
  const {device, graph, props} = createContributor();
  try {
    const shared = createTransientView(graph, 'aligned-overlap-output-storage', 'uint32', 128);
    const flow = graph.createDataView(shared.buffer, {format: 'uint32', length: 8, byteOffset: 4});
    const summary = graph.createDataView(shared.buffer, {
      format: 'uint32',
      length: 8,
      byteOffset: 64
    });
    expect(flow.byteOffset + 4 * flow.length).toBeLessThanOrEqual(summary.byteOffset);
    const flowRange = getViewBindingRange(flow);
    const summaryRange = getViewBindingRange(summary);
    expect(flowRange.offset).toBe(summaryRange.offset);
    expect(flowRange.offset + flowRange.size).toBeGreaterThan(summaryRange.offset);
    expect(() => new GPUNetworkFlow({...props, output: {...props.output, flow, summary}})).toThrow(
      /parallel-flow.*distinct buffers/
    );
  } finally {
    device.destroy();
  }
});

for (const nodeCount of [257, 1025]) {
  it(`costed path-doubling WGSL marks all ${nodeCount} vertices and reaches the nonzero source`, () => {
    const {device, graph, contributor} = createContributor(nodeCount, nodeCount - 1, {
      maximumAugmentations: 1,
      maximumRelaxationRounds: 1
    });
    try {
      const commands = contributor
        .getCommandNodes(graph)
        .filter(node => /search-0-(double|mark)-\d+$/.test(node.id));
      const arrays = new Map<GraphDataView, Uint32Array>();
      const values = (view: GraphDataView) => {
        if (!arrays.has(view)) arrays.set(view, new Uint32Array(view.length));
        return arrays.get(view)!;
      };
      const first = getWGSLKernelNodeRecord(graph, commands[0].id)!;
      const path = values(first.bindings.find(binding => binding.name === 'inputPath')!.view);
      for (let vertex = 0; vertex < nodeCount; vertex++) {
        path[4 * vertex] = Math.min(vertex + 1, nodeCount - 1);
        path[4 * vertex + 1] = vertex === nodeCount - 1 ? 0xffffffff : 1;
        path[4 * vertex + 2] = vertex;
        path[4 * vertex + 3] = vertex === 0 ? 1 : 0;
      }
      // A descending chain has source nodeCount - 1, sink 0, bottleneck 1, and every vertex
      // belongs to its unique path. Execute only the real doubling and marking kernels; the
      // expected path is a closed form, independent of a second path-building implementation.
      for (const command of commands) {
        const record = getWGSLKernelNodeRecord(graph, command.id)!;
        const source = record.source
          .replace(/^\s*_\s*=[^;]*;/gm, '')
          .replace(/@workgroup_size\(\s*(\d+)u?\s*\)/g, '@workgroup_size($1, 1, 1)');
        const bindings = Object.fromEntries(
          record.bindings.map((binding, index) => [index, values(binding.view)])
        );
        new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
          'main',
          [...record.workgroups],
          {0: bindings},
          {constants: record.constants}
        );
      }
      const last = getWGSLKernelNodeRecord(graph, commands[commands.length - 1].id)!;
      const result = values(last.bindings.find(binding => binding.name === 'outputPath')!.view);
      expect(Array.from({length: nodeCount}, (_, vertex) => result[4 * vertex + 3])).toEqual(
        new Array(nodeCount).fill(1)
      );
      expect(result[0]).toBe(nodeCount - 1);
      expect(result[1]).toBe(1);
    } finally {
      device.destroy();
    }
  });
}

it('primal-dual batches solve the independent NetworkX multigraph fixtures', () => {
  for (const scene of primalDualOracle.scenes) {
    const result = executeScene({...scene, mode: 'primal-dual', maximumCostPhases: 12});
    expect(result.summary.slice(0, 4), scene.name).toEqual([
      1,
      1,
      scene.expectedValue,
      scene.expectedUnserved
    ]);
    expect(result.exactCost, scene.name).toEqual([scene.expectedCost, 0]);
    expect([result.converged, result.limited, result.invalid], scene.name).toEqual([1, 0, 0]);
    expectNetworkFlowFeasible(scene, result.flow, scene.expectedValue);
  }
});

it('one primal-dual phase sends every equal-cost fan path while the explicit SSP budget stays bounded', () => {
  const scene = {
    nodeCount: 12,
    edges: Array.from({length: 10}, (_, index) => [0, index + 1, index + 1, 11]).flat(),
    capacities: new Array(20).fill(1),
    costs: new Array(20).fill(2),
    maximumAugmentations: 1,
    maximumRelaxationRounds: 3
  };
  const batched = executeScene({...scene, mode: 'primal-dual', maximumCostPhases: 1});
  expect(batched.flow).toEqual(new Array(20).fill(1));
  expect(batched.summary.slice(0, 5)).toEqual([1, 1, 10, 0, 1]);
  expect(batched.summary.slice(6)).toEqual([10, 1]);
  expect(batched.exactCost).toEqual([40, 0]);
  const single = executeScene({...scene, mode: 'successive-shortest-path'});
  expect(single.summary.slice(0, 5)).toEqual([1, 0, 1, 0, 1]);
  expect(single.exactCost).toEqual([4, 0]);
  expectNetworkFlowFeasible(scene, single.flow, 1);
});

it('primal-dual prefers cheaper longer paths and publishes a cost-optimal feasible partial batch', () => {
  const scene = {
    nodeCount: 5,
    edges: [0, 4, 0, 1, 1, 2, 2, 4, 0, 3, 3, 4],
    capacities: [5, 3, 3, 3, 2, 2],
    costs: [9, 1, 1, 1, 3, 2],
    mode: 'primal-dual' as const,
    required: 4
  };
  const partial = executeScene({...scene, maximumCostPhases: 1});
  expect(partial.flow).toEqual([0, 3, 3, 3, 0, 0]);
  expect(partial.summary.slice(0, 5)).toEqual([1, 0, 3, 1, 1]);
  expect(partial.exactCost).toEqual([9, 0]);
  expect([partial.converged, partial.limited]).toEqual([0, 1]);
  expectNetworkFlowFeasible(scene, partial.flow, 3);
  const complete = executeScene({...scene, maximumCostPhases: 2});
  expect(complete.flow).toEqual([0, 3, 3, 3, 1, 1]);
  expect(complete.summary.slice(0, 5)).toEqual([1, 1, 4, 0, 2]);
  expect(complete.exactCost).toEqual([14, 0]);
  expect([complete.converged, complete.limited]).toEqual([1, 0]);
});

it('primal-dual returns stranded excess through cost-tight reverse arcs with a nonzero source', () => {
  const scene = {
    nodeCount: 5,
    source: 4,
    sink: 0,
    edges: [4, 1, 4, 2, 1, 3, 2, 3, 3, 0],
    capacities: [4, 4, 4, 4, 1],
    costs: [2, 2, 1, 1, 3],
    mode: 'primal-dual' as const,
    maximumCostPhases: 1
  };
  const result = executeScene(scene);
  expect(result.summary.slice(0, 5)).toEqual([1, 1, 1, 0, 1]);
  expect(result.exactCost).toEqual([6, 0]);
  expect(result.summary.slice(6)).toEqual([1, 1]);
  expectNetworkFlowFeasible(scene, result.flow, 1);
});

it('primal-dual checks shallow searches, zero budgets, invalid inputs and full unsigned capacities', () => {
  const scene = {
    nodeCount: 4,
    edges: [0, 1, 1, 2, 2, 3],
    capacities: [1, 1, 1],
    costs: [2, 3, 4],
    mode: 'primal-dual' as const
  };
  for (const options of [{maximumRelaxationRounds: 1}, {maximumCostPhases: 0}]) {
    const limited = executeScene({...scene, ...options});
    expect(limited.flow).toEqual([0, 0, 0]);
    expect([limited.converged, limited.limited, limited.invalid]).toEqual([0, 1, 0]);
  }
  const zero = executeScene({...scene, required: 0, maximumRelaxationRounds: 1});
  expect(zero.summary.slice(0, 5)).toEqual([1, 1, 0, 0, 0]);
  expect(zero.summary[7]).toBe(0);
  const invalid = executeScene({...scene, edges: [0, 9, 1, 2, 2, 3]});
  expect(invalid.flow).toEqual([0, 0, 0]);
  expect([invalid.converged, invalid.limited, invalid.invalid]).toEqual([0, 0, 1]);
  const wide = executeScene({
    nodeCount: 2,
    edges: [0, 1, 0, 1],
    capacities: [0x80000000, 0x7fffffff],
    costs: [0, 0],
    mode: 'primal-dual',
    maximumCostPhases: 1
  });
  expect(wide.summary.slice(0, 5)).toEqual([1, 1, 0xffffffff, 0, 1]);
  expect(wide.flow).toEqual([0x80000000, 0x7fffffff]);
  expect(wide.exactCost).toEqual([0, 0]);
});

it('primal-dual sources/pipelines stay constant across phase/round budgets and meet kernel audit limits', () => {
  const variantCounts: number[] = [];
  for (const budget of [2, 3, 4]) {
    const {device, graph, contributor} = createContributor(6, 8, {
      mode: 'primal-dual',
      maximumCostPhases: budget,
      maximumRelaxationRounds: budget
    });
    try {
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      const kernels = getGPUCommandNodeKernelSources(nodes);
      variantCounts.push(
        new Set(kernels.map(kernel => JSON.stringify([kernel.source, kernel.constants]))).size
      );
      for (const kernel of kernels) {
        expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
        expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
        expect(getPrivateMemoryBytes(kernel.source), kernel.id).toBeLessThanOrEqual(64);
        const code = kernel.source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
        expect(
          [...new Set(code.match(/[A-Za-z_][A-Za-z_0-9]*/g))].filter(word =>
            WGSL_RESERVED_WORDS.has(word)
          ),
          kernel.id
        ).toEqual([]);
      }
    } finally {
      device.destroy();
    }
  }
  expect(new Set(variantCounts).size).toBe(1);
});

it('primal-dual validates its mode/phase budget and preflights all doubled label scratch before nodes', () => {
  expect(() => createContributor(6, 8, {mode: 'invalid' as never})).toThrow(/mode must be/);
  expect(() => createContributor(6, 8, {mode: 'primal-dual', costs: undefined})).toThrow(
    /mode requires costs/
  );
  expect(() => createContributor(6, 8, {mode: 'primal-dual', maximumCostPhases: 4097})).toThrow(
    /maximumCostPhases.*4096/
  );
  expect(() =>
    createContributor(6, 8, {mode: 'successive-shortest-path', maximumCostPhases: 1})
  ).toThrow(/maximumCostPhases requires costed primal-dual/);
  const {device, graph, contributor} = createContributor(100, 1, {
    mode: 'primal-dual',
    maximumCostPhases: 1,
    maximumRelaxationRounds: 1
  });
  try {
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 512});
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /push-relabel heights\/excess.*800.*limit 512/
    );
    expect(getWGSLKernelNodeRecord(graph, 'parallel-flow-initialize-state')).toBeUndefined();
  } finally {
    device.destroy();
  }
});

it('operation forwards primal-dual mode and cost phase budget', () => {
  const {device, graph, props} = createContributor();
  const operation = new GPUNetworkFlowOperation({
    nodeCount: 6,
    edgeCount: 8,
    mode: 'primal-dual',
    maximumCostPhases: 2,
    maximumRelaxationRounds: 6,
    edges: 'edges',
    capacities: 'capacities',
    costs: 'costs',
    parameters: 'parameters',
    output: {flow: 'flow', cut: 'cut', summary: 'summary'}
  });
  try {
    const views = {...props.output, ...props};
    const contributor = operation.createContributor(
      name => views[name as keyof typeof views] as never
    ) as GPUNetworkFlow;
    expect(contributor.solver).toBe('parallel');
    expect(contributor.props.mode).toBe('primal-dual');
    expect(contributor.props.maximumCostPhases).toBe(2);
    expect(contributor.getCommandNodes(graph).some(node => /relabel-0-push-0$/.test(node.id))).toBe(
      true
    );
  } finally {
    operation.destroy();
    device.destroy();
  }
});

it('maximumCostPhases opts auto into primal-dual and estimates its maximum shape without allocation', () => {
  for (const [nodeCount, edgeCount] of [
    [2, 0],
    [100_000, 400_000],
    [30_000_000, 30_000_000]
  ]) {
    const {device, graph, contributor} = createContributor(nodeCount, edgeCount, {
      solver: 'auto',
      maximumCostPhases: 1,
      maximumRelaxationRounds: 1
    });
    try {
      Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 2 ** 31});
      const nodes = contributor.getCommandNodes(graph);
      expect(contributor.solver).toBe('parallel');
      expect(nodes.some(node => /relabel-0-push-0$/.test(node.id))).toBe(true);
      expect(nodes.some(node => /search-0-path-initialize$/.test(node.id))).toBe(false);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    } finally {
      device.destroy();
    }
  }
});

for (const maximumAugmentations of [undefined, 0, 1, 4]) {
  it(`omitted mode preserves the legacy 129-node result with augmentation budget ${maximumAugmentations ?? 'default'}`, () => {
    const scene = {
      nodeCount: 129,
      solver: 'auto' as const,
      edges: [0, 128, 0, 128, 0, 128, 0, 128],
      capacities: [1, 1, 1, 1],
      costs: [1, 2, 3, 4],
      required: 4,
      maximumAugmentations
    };
    const result = executeScene(scene);
    const value = maximumAugmentations === undefined ? 4 : maximumAugmentations;
    expect(result.summary.slice(0, 5)).toEqual([1, Number(value === 4), value, 4 - value, value]);
    expect(result.exactCost).toEqual([(value * (value + 1)) / 2, 0]);
    expect(result.flow).toEqual(scene.capacities.map((_, index) => Number(index < value)));
    expect([result.converged, result.limited, result.invalid]).toEqual([
      Number(value === 4),
      Number(value < 4),
      0
    ]);
    expectNetworkFlowFeasible(scene, result.flow, value);
  });
}

it('legacy defaults and explicit primal-dual preserve all pinned NetworkX flow objectives', () => {
  for (const fixture of movementOracle.flow) {
    const scene = {
      nodeCount: fixture.nodeCount,
      edges: fixture.edges.flat(),
      capacities: fixture.capacities,
      costs: fixture.costs
    };
    // The interpreter cannot select from private bool arrays in the serial GPU reference.
    // Check its CPU twin here; the headless cases exercise the real GPU reference as well.
    const reference = computeNetworkFlowOnCPU(
      {
        edges: Uint32Array.from(scene.edges),
        capacities: Uint32Array.from(scene.capacities),
        costs: Uint32Array.from(scene.costs),
        parameters: Uint32Array.of(0, scene.nodeCount - 1, 0xffffffff)
      },
      scene
    );
    expect(Array.from(reference.summary.slice(0, 4)), fixture.name).toEqual([
      1,
      1,
      fixture.maximumFlow,
      0
    ]);
    expect(Array.from(reference.exactCost), fixture.name).toEqual([fixture.minimumCost, 0]);
    for (const options of [{solver: 'parallel' as const}, {mode: 'primal-dual' as const}]) {
      const result = executeScene({...scene, ...options});
      expect(result.summary.slice(0, 4), fixture.name).toEqual([1, 1, fixture.maximumFlow, 0]);
      expect(result.exactCost, fixture.name).toEqual([fixture.minimumCost, 0]);
      expect([result.converged, result.limited, result.invalid], fixture.name).toEqual([1, 0, 0]);
      expectNetworkFlowFeasible(scene, result.flow, fixture.maximumFlow);
    }
  }
});

it('default costed parallel flow and CPU reference complete all 12 new NetworkX fixtures', () => {
  for (const scene of primalDualOracle.scenes) {
    const reference = computeNetworkFlowOnCPU(
      {
        edges: Uint32Array.from(scene.edges),
        capacities: Uint32Array.from(scene.capacities),
        costs: Uint32Array.from(scene.costs),
        parameters: Uint32Array.of(0, scene.nodeCount - 1, scene.required)
      },
      scene
    );
    expect(Array.from(reference.summary.slice(0, 4)), scene.name).toEqual([
      1,
      1,
      scene.expectedValue,
      scene.expectedUnserved
    ]);
    expect(Array.from(reference.exactCost), scene.name).toEqual([scene.expectedCost, 0]);
    const result = executeScene(scene);
    expect(result.summary.slice(0, 4), scene.name).toEqual([
      1,
      1,
      scene.expectedValue,
      scene.expectedUnserved
    ]);
    expect(result.exactCost, scene.name).toEqual([scene.expectedCost, 0]);
    expect([result.converged, result.limited, result.invalid], scene.name).toEqual([1, 0, 0]);
    expectNetworkFlowFeasible(scene, result.flow, scene.expectedValue);
  }
});
