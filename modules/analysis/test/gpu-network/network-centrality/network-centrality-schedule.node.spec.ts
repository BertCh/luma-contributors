// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUNetworkCentrality,
  GPUNetworkCentralityScheduleStep,
  planNetworkCentralitySchedule,
  type GPUNetworkCentralityProps
} from '../../../src/gpu-network/network-centrality/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  auditWGSLModule,
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

import {
  createKernelAuditContext,
  getDeclaredNodes,
  recordShaderModules
} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {NETWORK_CENTRALITY_SCHEDULE_KERNEL_AUDIT_CASES} from './network-centrality-kernel-audit-cases';

function createProps(
  graph: GPUCommandGraph,
  nodeCount = 40,
  pivotCount = 2
): GPUNetworkCentralityProps {
  let viewCount = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `schedule-view-${viewCount++}`, format, length);
  const topology = () => ({
    offsets: view('uint32', nodeCount + 1),
    neighbors: view('uint32', 100),
    weights: view('float32', 100)
  });
  return {
    id: 'centrality',
    topology: topology(),
    reverseTopology: topology(),
    pivotNodes: view('uint32', pivotCount),
    laneCount: pivotCount,
    maximumIterations: 40,
    sampling: 'sampled',
    output: {
      betweenness: view('float32', nodeCount),
      edgeBetweenness: view('float32', 100),
      closeness: view('float32', pivotCount),
      harmonicCloseness: view('float32', pivotCount),
      reachedCounts: view('uint32', pivotCount),
      status: {
        converged: view('uint32', 1),
        zeroWeightCycle: view('uint32', 1),
        iterationLimitReached: view('uint32', 1),
        approximation: view('uint32', 1)
      }
    }
  };
}

it('plans lazy fixed-capacity submissions including a padded tail and uint32 total', () => {
  const plan = planNetworkCentralitySchedule({pivotCount: 7, laneCount: 3});
  expect(plan.stepCount).toBe(3);
  expect([...plan]).toEqual([
    {pivotOffset: 0, pivotCount: 3, laneCount: 3, reset: 1},
    {pivotOffset: 3, pivotCount: 3, laneCount: 3, reset: 0},
    {pivotOffset: 6, pivotCount: 1, laneCount: 3, reset: 0}
  ]);
  const large = planNetworkCentralitySchedule({pivotCount: 0xffffffff, laneCount: 1});
  expect(large.stepCount).toBe(0xffffffff);
  expect(large[Symbol.iterator]().next().value).toMatchObject({pivotOffset: 0, pivotCount: 1});
  for (const props of [
    {pivotCount: 0, laneCount: 1},
    {pivotCount: 1, laneCount: 2},
    {pivotCount: Infinity, laneCount: 1}
  ]) {
    expect(() => planNetworkCentralitySchedule(props)).toThrow(/network-centrality-schedule/);
  }
});

it('bounds one reusable graph independently of a source count above the old command budget', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph, 40, 2000);
  expect(() => new GPUNetworkCentrality({...props, laneCount: 1})).toThrow(/32768/);
  const schedule = planNetworkCentralitySchedule({pivotCount: 2000, laneCount: 1});
  expect(schedule.stepCount).toBe(2000);
  const fixed = {
    ...props,
    pivotNodes: graph.createDataView(props.pivotNodes.buffer, {format: 'uint32', length: 1}),
    laneCount: 1,
    output: {betweenness: props.output.betweenness, status: props.output.status}
  };
  const contributor = new GPUNetworkCentralityScheduleStep({
    ...fixed,
    reset: createTransientView(graph, 'reset', 'uint32', 1)
  });
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.length).toBe(contributor.estimatedCommandNodeCount);
  expect(nodes.length).toBeLessThan(150);
});

it('keeps the shader and pipeline count constant over source batches, tails and rounds', () => {
  const build = (roundCount: number) => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const contributor = new GPUNetworkCentralityScheduleStep({
      ...createProps(graph),
      maximumIterations: roundCount,
      reset: createTransientView(graph, 'reset', 'uint32', 1)
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBe(contributor.estimatedCommandNodeCount);
    for (const [id, source] of getParsedKernelSources(nodes)) {
      expect(findReservedIdentifiers(stripWGSLComments(source)), id).toEqual([]);
      expect(source, id).not.toMatch(/\b(?:nan|inf)\b/i);
    }
    return {
      pipelines: getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}).size,
      sources: new Set(getParsedKernelSources(nodes).values())
    };
  };
  const first = build(2);
  expect(build(3).pipelines).toBe(first.pipelines);
  expect(build(40).pipelines).toBe(first.pipelines);
  // Every host step uses the same compiled graph, including the final padded batch. No step
  // index, reset value or unpadded length is a constructor prop or shader constant.
  for (const step of planNetworkCentralitySchedule({pivotCount: 7, laneCount: 2})) {
    expect(step.laneCount).toBe(2);
    expect(build(2).sources).toEqual(first.sources);
  }
});

for (const failure of ['adjacency', 'scratch', 'dispatch', 'input-alignment', 'status'] as const) {
  it(`preflights ${failure} before declaring any graph resource or node`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const props = createProps(
      graph,
      failure === 'status' ? 1 : failure === 'dispatch' || failure === 'scratch' ? 100 : 40,
      failure === 'scratch' ? 4 : 1
    );
    if (failure === 'adjacency' || failure === 'scratch') props.maximumIterations = 1;
    if (failure === 'input-alignment') {
      const buffer = graph.createTransientBuffer({
        id: 'aligned-input',
        byteLength: 1280,
        usage: Buffer.STORAGE
      });
      props.pivotNodes = graph.createDataView(buffer, {
        format: 'uint32',
        byteOffset: 252,
        length: 200
      });
      props.laneCount = 1;
      props.output = {betweenness: props.output.betweenness};
    }
    if (failure === 'status') {
      for (const [index, topology] of [props.topology, props.reverseTopology].entries()) {
        topology.neighbors = createTransientView(graph, `empty-neighbors-${index}`, 'uint32', 0);
        topology.weights = createTransientView(graph, `empty-weights-${index}`, 'float32', 0);
      }
      const buffer = graph.createTransientBuffer({
        id: 'aligned-status',
        byteLength: 256,
        usage: Buffer.STORAGE
      });
      props.output = {
        betweenness: props.output.betweenness,
        status: {
          approximation: graph.createDataView(buffer, {
            format: 'uint32',
            byteOffset: 252,
            length: 1
          })
        }
      };
    }
    const contributor = new GPUNetworkCentrality(props);
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {
      value:
        failure === 'adjacency'
          ? 1500
          : failure === 'scratch'
            ? 4000
            : failure === 'dispatch'
              ? 6000
              : failure === 'input-alignment'
                ? 1000
                : 128
    });
    const mutations = [
      vi.spyOn(graph, 'createTransientBuffer'),
      vi.spyOn(graph, 'createDataView'),
      vi.spyOn(graph, 'add')
    ];
    const bindingName =
      failure === 'input-alignment'
        ? 'pivotNodes chunk 0'
        : failure === 'status'
          ? 'output.status.approximation chunk 0'
          : failure === 'dispatch'
            ? 'dependency dispatch slots'
            : failure;
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      new RegExp(`${bindingName} needs .* bytes in one binding`)
    );
    for (const mutation of mutations) expect(mutation).not.toHaveBeenCalled();
  });
}

it('rejects reset/output aliasing and validates sampling status at construction', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph);
  expect(
    () => new GPUNetworkCentralityScheduleStep({...props, reset: props.output.status!.converged!})
  ).toThrow(/logical byte ranges|aligned binding/);
  expect(() => new GPUNetworkCentrality({...props, sampling: 'invalid' as never})).toThrow(
    /sampling/
  );
  expect(
    () =>
      new GPUNetworkCentrality({
        ...props,
        output: {
          ...props.output,
          status: {approximation: createTransientView(graph, 'wrong-status', 'uint32', 2)}
        }
      })
  ).toThrow(/approximation.length/);
});

it('rejects logical reset/total separation when aligned binding windows overlap', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph);
  const shared = graph.createTransientBuffer({
    id: 'reset-and-total',
    byteLength: 256,
    usage: Buffer.STORAGE
  });
  const reset = graph.createDataView(shared, {format: 'uint32', byteOffset: 0, length: 1});
  props.output.status!.converged = graph.createDataView(shared, {
    format: 'uint32',
    byteOffset: 4,
    length: 1
  });
  expect(() => new GPUNetworkCentralityScheduleStep({...props, reset})).toThrow(
    /logical byte ranges|aligned binding/
  );
});

for (const auditCase of NETWORK_CENTRALITY_SCHEDULE_KERNEL_AUDIT_CASES) {
  it(`kernel audit: ${auditCase.exportName} ${auditCase.name}`, () => {
    const context = createKernelAuditContext(auditCase.exportName);
    try {
      auditCase.build(context);
      const nodes = getDeclaredNodes(context);
      expect(nodes.length).toBeGreaterThan(3000);
      expect(nodes.length).toBeLessThan(32768);
      const audited = new Set<string>();
      recordShaderModules(nodes, module => {
        const key = module.source + JSON.stringify(module.constants);
        if (audited.has(key)) return;
        audited.add(key);
        expect(auditWGSLModule(module.source, module.constants), module.id).toEqual([]);
      });
      expect(audited.size).toBeGreaterThan(10);
    } finally {
      context.device.destroy();
    }
  });
}

it('keeps vertex-only optional positions out of storage preflight for both routes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph, 3, 1);
  props.maximumIterations = 3;
  props.output = {closeness: props.output.closeness};
  for (const [index, topology] of [props.topology, props.reverseTopology].entries()) {
    const buffer = graph.createTransientBuffer({
      id: `positions-${index}`,
      byteLength: 24,
      usage: Buffer.VERTEX
    });
    topology.nodePositions = graph.createDataView(buffer, {format: 'float32x2', length: 3});
  }
  expect(new GPUNetworkCentrality(props).getCommandNodes(graph)).toHaveLength(8);
  expect(() =>
    new GPUNetworkCentralityScheduleStep({
      ...props,
      reset: createTransientView(graph, 'vertex-reset', 'uint32', 1)
    }).getCommandNodes(graph)
  ).not.toThrow();
});

it('rejects overflowing closeness lane dispatches at construction without mutations', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const topology = {
    offsets: view('offsets', 'uint32', 2),
    neighbors: view('neighbors', 'uint32', 0),
    weights: view('weights', 'float32', 0)
  };
  const props = {
    topology,
    reverseTopology: topology,
    pivotNodes: view('pivots', 'uint32', 16_777_216),
    laneCount: 16_777_216,
    maximumIterations: 1,
    output: {closeness: view('closeness', 'float32', 16_777_216)}
  };
  const reset = view('reset-large', 'uint32', 1);
  const mutations = [
    vi.spyOn(graph, 'createTransientBuffer'),
    vi.spyOn(graph, 'createDataView'),
    vi.spyOn(graph, 'add'),
    vi.spyOn(device, 'createBuffer')
  ];
  expect(() => new GPUNetworkCentrality(props)).toThrow(
    /laneCount for closeness outputs must be at most 16777215/
  );
  expect(() => new GPUNetworkCentralityScheduleStep({...props, reset})).toThrow(
    /laneCount for closeness outputs must be at most 16777215/
  );
  for (const mutation of mutations) expect(mutation).not.toHaveBeenCalled();
});

it('accepts disjoint packed outputs across dispatches while checking simultaneous bindings', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph, 3, 1);
  props.maximumIterations = 3;
  const buffer = graph.createTransientBuffer({
    id: 'packed-outputs',
    byteLength: 2048,
    usage: Buffer.STORAGE
  });
  const view = <Format extends 'uint32' | 'float32'>(
    format: Format,
    byteOffset: number,
    length: number
  ) => graph.createDataView(buffer, {format, byteOffset, length});
  const output = {
    betweenness: view('float32', 0, 3),
    edgeBetweenness: view('float32', 12, 100),
    closeness: view('float32', 448, 1),
    harmonicCloseness: view('float32', 512, 1),
    reachedCounts: view('uint32', 768, 1),
    status: {
      converged: view('uint32', 452, 1),
      zeroWeightCycle: view('uint32', 456, 1),
      iterationLimitReached: view('uint32', 460, 1),
      approximation: view('uint32', 464, 1)
    }
  };
  // Geometry is unused; its range may overlap writable outputs.
  props.topology.nodePositions = graph.createDataView(buffer, {format: 'float32x2', length: 3});
  const schedule = new GPUNetworkCentralityScheduleStep({
    ...props,
    output,
    reset: view('uint32', 1024, 1)
  });
  expect(schedule.getCommandNodes(graph).length).toBe(schedule.estimatedCommandNodeCount);
  const direct = new GPUNetworkCentrality({
    ...props,
    output: {
      ...output,
      status: {converged: output.status.converged, approximation: output.status.approximation}
    }
  });
  expect(direct.getCommandNodes(graph).length).toBe(direct.estimatedCommandNodeCount);
  expect(() => new GPUNetworkCentrality({...props, output})).toThrow(/aligned binding/);
  expect(
    () =>
      new GPUNetworkCentralityScheduleStep({
        ...props,
        output: {...output, harmonicCloseness: view('float32', 480, 1)},
        reset: view('uint32', 1024, 1)
      })
  ).toThrow(/aligned binding/);
});

for (const field of ['pivotNodes', 'closeness', 'reset'] as const) {
  it(`preflights missing STORAGE usage on ${field} before declaring scratch`, () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const props = createProps(graph, 3, 1);
    props.output = {closeness: props.output.closeness};
    const buffer = graph.createTransientBuffer({
      id: `vertex-${field}`,
      byteLength: 4,
      usage: Buffer.VERTEX
    });
    const invalidWord = graph.createDataView(buffer, {format: 'uint32', length: 1});
    const invalidFloat = graph.createDataView(buffer, {format: 'float32', length: 1});
    if (field === 'pivotNodes') props.pivotNodes = invalidWord;
    if (field === 'closeness') props.output.closeness = invalidFloat;
    const reset =
      field === 'reset' ? invalidWord : createTransientView(graph, 'storage-reset', 'uint32', 1);
    const contributor = new GPUNetworkCentralityScheduleStep({...props, reset});
    const mutations = [
      vi.spyOn(graph, 'createTransientBuffer'),
      vi.spyOn(graph, 'createDataView'),
      vi.spyOn(graph, 'add')
    ];
    expect(() => contributor.getCommandNodes(graph)).toThrow(/usage must include Buffer.STORAGE/);
    for (const mutation of mutations) expect(mutation).not.toHaveBeenCalled();
  });
}
