// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  REVERSE_RESIDUAL_SCENE,
  expectEnumeratedIntegralFlow
} from './network-flow-integral-enumeration.test-utils';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUNetworkFlow,
  type GPUNetworkFlowProps
} from '../../../src/gpu-network/network-optimization/gpu-network-flow';
import {computeNetworkFlowOnCPU} from '../../../src/gpu-network/network-optimization/cpu-network-flow-operation';
import {GPUNetworkFlowOperation} from '../../../src/gpu-network/network-optimization/gpu-network-flow-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createOptimizationHarness} from './optimization-test-harness';
import gridOracle from './network-flow-grid-oracle.json';
import fanOracle from './network-flow-fan-oracle.json';
import primalDualOracle from './network-flow-primal-dual-oracle.json';
import movementOracle from '../../oracles/fixtures/network-movement-coverage.json';
import {createNetworkFlowStorageReadbackNode} from './network-flow-storage-test-utils';
import {expectNetworkFlowFeasible} from './network-flow-feasibility-test-utils';

type Scene = {
  nodeCount: number;
  edges: Uint32Array;
  capacities: Uint32Array;
  costs?: Uint32Array;
  required?: number;
  topology?: boolean;
  offsets?: Uint32Array;
  maximumAugmentations?: number;
  maximumRelaxationRounds?: number;
  maximumRelabels?: number;
  mode?: GPUNetworkFlowProps['mode'];
  maximumCostPhases?: number;
  source?: number;
  sink?: number;
};

type FlowSceneResult = {
  flow: number[];
  cut: number[];
  summary: number[];
  exactCost: number[];
  invalid: number;
  converged: number;
  limited: number;
};

/** Includes compile, submission and readback; a GPU validation error must fail the spec. */
async function validateFlowExecution<Result>(
  device: Device,
  execute: () => Promise<Result>
): Promise<Result> {
  const nativeDevice = device.handle as GPUDevice;
  nativeDevice.pushErrorScope('validation');
  try {
    return await execute();
  } finally {
    try {
      await nativeDevice.queue.onSubmittedWorkDone();
    } finally {
      const error = await nativeDevice.popErrorScope();
      expect(error?.message, 'WebGPU validation').toBeUndefined();
    }
  }
}

async function runScene(
  device: Device,
  scene: Scene,
  solver: GPUNetworkFlowProps['solver'],
  checkFirstEncoding?: (result: FlowSceneResult) => void
): Promise<FlowSceneResult> {
  return validateFlowExecution(device, async () => {
    const harness = createOptimizationHarness(device);
    const output = {
      flow: harness.output('uint32', scene.capacities.length),
      cut: harness.output('uint32', scene.nodeCount),
      summary: harness.output('uint32', 8),
      exactCost: harness.output('uint32x2', 1),
      status: {
        invalidCount: harness.output('uint32', 1),
        converged: harness.output('uint32', 1),
        iterationCount: harness.output('uint32', 1),
        iterationLimitReached: harness.output('uint32', 1)
      }
    };
    const collect = (read: (view: GraphDataView) => number[]): FlowSceneResult => ({
      flow: read(output.flow),
      cut: read(output.cut),
      summary: read(output.summary),
      exactCost: read(output.exactCost),
      invalid: read(output.status.invalidCount)[0],
      converged: read(output.status.converged)[0],
      limited: read(output.status.iterationLimitReached)[0]
    });
    const offsets = new Uint32Array(scene.nodeCount + 1);
    const neighbors = new Uint32Array(scene.capacities.length);
    for (let edge = 0; edge < scene.capacities.length; edge++) {
      offsets[scene.edges[2 * edge] + 1]++;
      neighbors[edge] = scene.edges[2 * edge + 1];
    }
    for (let vertex = 1; vertex <= scene.nodeCount; vertex++)
      offsets[vertex] += offsets[vertex - 1];
    const read = await harness.run(
      new GPUNetworkFlow({
        nodeCount: scene.nodeCount,
        solver,
        mode: scene.mode,
        maximumCostPhases: scene.maximumCostPhases,
        edges: harness.input('uint32x2', scene.edges),
        capacities: harness.input('uint32', scene.capacities),
        costs: scene.costs ? harness.input('uint32', scene.costs) : undefined,
        parameters: harness.input(
          'uint32',
          Uint32Array.of(
            scene.source ?? 0,
            scene.sink ?? scene.nodeCount - 1,
            scene.required ?? 0xffffffff
          )
        ),
        topology: scene.topology
          ? {
              offsets: harness.input('uint32', scene.offsets ?? offsets),
              neighbors: harness.input('uint32', neighbors)
            }
          : undefined,
        maximumAugmentations: scene.maximumAugmentations,
        maximumRelabels: solver === 'reference' ? undefined : scene.maximumRelabels,
        maximumRelaxationRounds: solver === 'reference' ? undefined : scene.maximumRelaxationRounds,
        output
      }),
      checkFirstEncoding ? read => checkFirstEncoding(collect(read)) : undefined
    );
    return collect(read);
  });
}

for (const mode of [
  'maximum',
  'minimum-cost',
  'finite',
  'infeasible-finite',
  'zero-cost-cycle'
] as const) {
  it(`parallel residual CSR agrees with reference and CPU objectives (${mode})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scene: Scene =
      mode === 'zero-cost-cycle'
        ? {
            nodeCount: 4,
            // Parallel edges, antiparallel edges, a self-loop and a reachable zero-cost cycle.
            edges: Uint32Array.of(0, 0, 0, 1, 0, 1, 1, 0, 1, 2, 1, 3, 2, 1, 2, 3),
            capacities: Uint32Array.of(9, 1, 2, 1, 2, 1, 2, 2),
            costs: Uint32Array.of(0, 0, 0, 0, 0, 2, 0, 1),
            topology: true
          }
        : {
            nodeCount: 6,
            edges: Uint32Array.of(0, 1, 0, 2, 1, 3, 1, 4, 2, 3, 2, 4, 3, 5, 4, 5),
            capacities: new Uint32Array(8).fill(1),
            costs: mode === 'maximum' ? undefined : Uint32Array.of(0, 0, 1, 2, 2, 100, 0, 0),
            required: mode === 'finite' ? 1 : mode === 'infeasible-finite' ? 3 : undefined,
            topology: true
          };
    const parallel = await runScene(device, scene, 'parallel');
    const reference = await runScene(device, {...scene, topology: false}, 'reference');
    const cpu = computeNetworkFlowOnCPU(
      {
        edges: scene.edges,
        capacities: scene.capacities,
        costs: scene.costs,
        parameters: Uint32Array.of(
          scene.source ?? 0,
          scene.sink ?? scene.nodeCount - 1,
          scene.required ?? 0xffffffff
        )
      },
      {nodeCount: scene.nodeCount}
    );
    expect(parallel.summary.slice(0, 4)).toEqual(reference.summary.slice(0, 4));
    expect(parallel.summary.slice(0, 4)).toEqual(Array.from(cpu.summary.slice(0, 4)));
    expect(parallel.exactCost).toEqual(reference.exactCost);
    expect(parallel.exactCost).toEqual(Array.from(cpu.exactCost));
    expect(parallel.summary.slice(6)).toEqual(reference.summary.slice(6));
    expect([parallel.converged, parallel.limited, parallel.invalid]).toEqual([1, 0, 0]);
    expectNetworkFlowFeasible(scene, parallel.flow, parallel.summary[2]);
  });
}

it('a 100k-vertex unit grid matches the SciPy min cut through the automatic CSR route', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const edges = new Uint32Array(gridOracle.edgeCount * 2);
  let edge = 0;
  for (let source = 0; source < gridOracle.nodeCount; source++) {
    const row = Math.floor(source / gridOracle.columns);
    const column = source % gridOracle.columns;
    for (const destination of [
      ...(column + 1 < gridOracle.columns ? [source + 1] : []),
      ...(row + 1 < gridOracle.rows ? [source + gridOracle.columns] : [])
    ]) {
      edges[2 * edge] = source;
      edges[2 * edge + 1] = destination;
      edge++;
    }
  }
  const scene: Scene = {
    nodeCount: gridOracle.nodeCount,
    edges,
    capacities: new Uint32Array(gridOracle.edgeCount).fill(1),
    topology: true,
    maximumAugmentations: 2,
    maximumRelabels: 2,
    maximumRelaxationRounds: 2 * (gridOracle.rows + gridOracle.columns)
  };
  const result = await runScene(device, scene, 'auto');
  expect(result.summary[0]).toBe(1);
  expect(result.summary[2]).toBe(gridOracle.maximumFlow);
  expect(result.summary[6]).toBe(gridOracle.cutCapacity);
  expect(result.summary[7]).toBe(1);
  expect(result.cut.flatMap((reachable, vertex) => (reachable ? [vertex] : []))).toEqual(
    gridOracle.sourceReachable
  );
  expect([result.converged, result.limited, result.invalid]).toEqual([1, 0, 0]);
  expectNetworkFlowFeasible(scene, result.flow, result.summary[2]);
}, 120_000);

it.each<{name: string; budget: Partial<Scene>; expectedValue: number}>([
  {name: 'one augmentation', budget: {maximumAugmentations: 1}, expectedValue: 1},
  {name: 'one relaxation round', budget: {maximumRelaxationRounds: 1}, expectedValue: 0},
  {name: 'zero augmentations', budget: {maximumAugmentations: 0}, expectedValue: 0}
])('iteration budget publishes a feasible partial result ($name)', async ({
  budget,
  expectedValue
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene: Scene = {
    nodeCount: 4,
    edges: Uint32Array.of(0, 1, 0, 2, 1, 3, 2, 3),
    capacities: Uint32Array.of(1, 1, 1, 1),
    costs: new Uint32Array(4)
  };
  const result = await runScene(device, {...scene, ...budget}, 'parallel');
  expect([result.converged, result.limited, result.invalid]).toEqual([0, 1, 0]);
  expect(result.summary[2]).toBe(expectedValue);
  expect(result.summary[7]).toBe(0);
  expect(result.exactCost).toEqual([0, 0]);
  if (expectedValue === 0) expect(result.flow).toEqual([0, 0, 0, 0]);
  expectNetworkFlowFeasible(scene, result.flow, result.summary[2]);
});

it('keeps exact 64-bit cost words at the largest reference capacity and cost', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runScene(
    device,
    {
      nodeCount: 2,
      edges: Uint32Array.of(0, 1),
      capacities: Uint32Array.of(0xffffff),
      costs: Uint32Array.of(32767)
    },
    'parallel'
  );
  const expected = 0xffffffn * 32767n;
  expect(result.exactCost).toEqual([Number(expected & 0xffffffffn), Number(expected >> 32n)]);
  const actual = new Float32Array(Uint32Array.of(result.summary[5]).buffer)[0];
  expect(Math.abs(actual - Number(expected))).toBeLessThanOrEqual(Number(expected) * 2 ** -22);
});

it('rejects malformed CSR contents on GPU without publishing a cut certificate', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const offsets of [Uint32Array.of(0, 2, 1, 2), Uint32Array.of(0, 1, 1, 1)]) {
    const result = await runScene(
      device,
      {
        nodeCount: 3,
        edges: Uint32Array.of(0, 1, 1, 2),
        capacities: Uint32Array.of(1, 1),
        topology: true,
        offsets
      },
      'parallel'
    );
    expect(result.flow).toEqual([0, 0]);
    expect(result.summary).toEqual(new Array(8).fill(0));
    expect([result.invalid, result.converged, result.limited]).toEqual([1, 0, 0]);
  }
});

it('rewrites all state after per-frame capacities change on the same compiled graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const capacities = harness.input('uint32', Uint32Array.of(3, 3));
  const output = {
    flow: harness.output('uint32', 2),
    cut: harness.output('uint32', 3),
    summary: harness.output('uint32', 8),
    exactCost: harness.output('uint32x2', 1),
    status: {
      converged: harness.output('uint32', 1),
      iterationLimitReached: harness.output('uint32', 1)
    }
  };
  const read = await harness.run(
    new GPUNetworkFlow({
      solver: 'parallel',
      nodeCount: 3,
      edges: harness.input('uint32x2', Uint32Array.of(0, 1, 1, 2)),
      capacities,
      costs: harness.input('uint32', Uint32Array.of(1, 2)),
      parameters: harness.input('uint32', Uint32Array.of(0, 2, 0xffffffff)),
      maximumAugmentations: 1,
      output
    }),
    firstRead => {
      expect(firstRead(output.summary)[2]).toBe(3);
      expect(firstRead(output.exactCost)).toEqual([9, 0]);
      harness.rewrite(capacities, Uint32Array.of(0, 0));
    }
  );
  expect(read(output.flow)).toEqual([0, 0]);
  expect(read(output.exactCost)).toEqual([0, 0]);
  expect(read(output.summary).slice(0, 5)).toEqual([1, 1, 0, 0, 0]);
  expect(read(output.status.converged)).toEqual([1]);
  expect(read(output.status.iterationLimitReached)).toEqual([0]);
});

it('keeps the validated reference shape and output binding after caller props mutate', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const flow = harness.output('uint32', 1);
  const cut = harness.output('uint32', 2);
  const summary = harness.output('uint32', 8);
  const replacementFlow = harness.output('uint32', 1);
  const props: GPUNetworkFlowProps = {
    solver: 'reference',
    nodeCount: 2,
    maximumAugmentations: 1,
    edges: harness.input('uint32x2', Uint32Array.of(0, 1)),
    capacities: harness.input('uint32', Uint32Array.of(3)),
    parameters: harness.input('uint32', Uint32Array.of(0, 1, 0xffffffff)),
    output: {flow, cut, summary}
  };
  const contributor = new GPUNetworkFlow(props);
  props.nodeCount = 129;
  props.maximumAugmentations = 65537;
  props.output.flow = replacementFlow;
  const read = await harness.run(contributor);
  expect(read(flow)).toEqual([3]);
  expect(read(replacementFlow)).toEqual([0]);
  expect(read(cut)).toEqual([1, 0]);
  expect(read(summary).slice(0, 5)).toEqual([1, 1, 3, 0, 1]);
});

it('accepts a full unsigned capacity and rejects overflowing capacity totals', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runScene(
    device,
    {
      nodeCount: 2,
      edges: Uint32Array.of(0, 1),
      capacities: Uint32Array.of(0xffffffff),
      costs: Uint32Array.of(32767)
    },
    'parallel'
  );
  const expected = 0xffffffffn * 32767n;
  expect(result.flow).toEqual([0xffffffff]);
  expect(result.exactCost).toEqual([Number(expected & 0xffffffffn), Number(expected >> 32n)]);
  expect([result.converged, result.limited, result.invalid]).toEqual([1, 0, 0]);
  const invalid = await runScene(
    device,
    {
      nodeCount: 2,
      edges: Uint32Array.of(0, 1, 0, 1),
      capacities: Uint32Array.of(0x80000000, 0x80000000)
    },
    'parallel'
  );
  expect(invalid.flow).toEqual([0, 0]);
  expect([invalid.invalid, invalid.converged, invalid.limited]).toEqual([1, 0, 0]);
});

it('the 100k fan completes a cut of 99,998 in one parallel preflow phase', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const nodeCount = fanOracle.nodeCount;
  const pathCount = nodeCount - 2;
  const edges = new Uint32Array(pathCount * 4);
  for (let vertex = 1; vertex <= pathCount; vertex++) {
    edges[2 * (vertex - 1)] = 0;
    edges[2 * (vertex - 1) + 1] = vertex;
    edges[2 * (pathCount + vertex - 1)] = vertex;
    edges[2 * (pathCount + vertex - 1) + 1] = nodeCount - 1;
  }
  const scene: Scene = {
    nodeCount,
    edges,
    capacities: new Uint32Array(pathCount * 2).fill(1),
    maximumRelabels: 1,
    maximumRelaxationRounds: 3,
    topology: true
  };
  const result = await runScene(device, scene, 'auto');
  // Closed form: pathCount edge-disjoint paths and a source cut of exactly pathCount.
  expect(result.summary.slice(0, 5)).toEqual([1, 1, fanOracle.maximumFlow, 0, 1]);
  expect(result.summary.slice(6)).toEqual([fanOracle.cutCapacity, 1]);
  expect(result.cut.flatMap((reachable, vertex) => (reachable ? [vertex] : []))).toEqual(
    fanOracle.sourceReachable
  );
  expect(result.flow.every(amount => amount === 1)).toBe(true);
  expect([result.converged, result.limited, result.invalid]).toEqual([1, 0, 0]);
  expectNetworkFlowFeasible(scene, result.flow, fanOracle.maximumFlow);
}, 120_000);

for (const costed of [false, true]) {
  for (const nodeCount of [255, 256, 257, 511, 512, 513, 1025]) {
    it(`a ${nodeCount}-vertex ${costed ? 'costed' : 'preflow'} chain crosses boundaries with a nonzero source`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const vertices = [
        nodeCount - 1,
        ...Array.from({length: nodeCount - 2}, (_, index) => index + 1),
        0
      ];
      const edges = Uint32Array.from(
        vertices.slice(1).flatMap((vertex, index) => [vertices[index], vertex])
      );
      const scene: Scene = {
        nodeCount,
        edges,
        source: nodeCount - 1,
        sink: 0,
        capacities: new Uint32Array(nodeCount - 1).fill(1),
        costs: costed ? new Uint32Array(nodeCount - 1) : undefined,
        maximumAugmentations: 1,
        maximumRelabels: costed ? undefined : 1,
        maximumRelaxationRounds: nodeCount
      };
      const result = await runScene(device, scene, 'parallel');
      // One chain, one unit-capacity cut: every edge carries exactly one and every interior
      // vertex balances. Zero costs deliberately select SSP and its entire doubling schedule.
      expect(result.summary.slice(0, 5)).toEqual([1, 1, 1, 0, 1]);
      expect(result.summary.slice(6)).toEqual([1, 1]);
      expect(result.exactCost).toEqual([0, 0]);
      expect(result.flow).toEqual(new Array(nodeCount - 1).fill(1));
      expect([result.converged, result.limited, result.invalid]).toEqual([1, 0, 0]);
      expectNetworkFlowFeasible(scene, result.flow, 1);
    }, 120_000);
  }
}

it('the operation wrapper configures a completed 1,025-vertex chain with shared topology', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const nodeCount = 1025;
  const edges = Uint32Array.from(
    Array.from({length: nodeCount - 1}, (_, source) => [source, source + 1]).flat()
  );
  const offsets = Uint32Array.from({length: nodeCount + 1}, (_, vertex) =>
    Math.min(vertex, nodeCount - 1)
  );
  const result = await runOperationGraph(device, {
    inputs: {
      edges: {format: 'uint32x2', values: edges},
      capacities: {format: 'uint32', values: new Uint32Array(nodeCount - 1).fill(1)},
      parameters: {format: 'uint32', values: Uint32Array.of(0, nodeCount - 1, 0xffffffff)},
      offsets: {format: 'uint32', values: offsets},
      neighbors: {
        format: 'uint32',
        values: Uint32Array.from({length: nodeCount - 1}, (_, vertex) => vertex + 1)
      }
    },
    operations: [
      new GPUNetworkFlowOperation({
        nodeCount,
        edgeCount: nodeCount - 1,
        solver: 'parallel',
        maximumRelaxationRounds: nodeCount,
        maximumRelabels: 1,
        edges: 'edges',
        capacities: 'capacities',
        parameters: 'parameters',
        topology: {offsets: 'offsets', neighbors: 'neighbors'},
        output: {
          flow: 'flow',
          cut: 'cut',
          summary: 'summary',
          status: {converged: 'converged', iterationLimitReached: 'limited'}
        }
      })
    ],
    outputs: ['flow', 'summary', 'converged', 'limited']
  });
  expect(Array.from(result['summary'].slice(0, 4))).toEqual([1, 1, 1, 0]);
  expect(result['flow'].every(amount => amount === 1)).toBe(true);
  expect(Array.from(result['converged'])).toEqual([1]);
  expect(Array.from(result['limited'])).toEqual([0]);
}, 120_000);

it('reverse-arc rerouting and pointer doubling preserve vertex IDs around 256/512', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const vertices = [255, 256, 257, 511, 512, 0];
  const endpoints = [0, 1, 0, 2, 1, 3, 1, 4, 2, 3, 2, 4, 3, 5, 4, 5];
  const scene: Scene = {
    nodeCount: 513,
    source: 255,
    sink: 0,
    edges: Uint32Array.from(endpoints.map(vertex => vertices[vertex])),
    capacities: new Uint32Array(8).fill(1),
    costs: Uint32Array.of(0, 0, 1, 2, 2, 100, 0, 0),
    maximumAugmentations: 2,
    maximumRelaxationRounds: 6
  };
  const result = await runScene(device, scene, 'parallel');
  expect(result.flow).toEqual([1, 1, 0, 1, 1, 0, 1, 1]);
  expect(result.exactCost).toEqual([4, 0]);
  expect(result.summary.slice(0, 4)).toEqual([1, 1, 2, 0]);
  expect(result.summary.slice(6)).toEqual([2, 1]);
  expectNetworkFlowFeasible(scene, result.flow, 2);
});

it('multiple costed paths total UINT32_MAX and carry objective words exactly', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene: Scene = {
    nodeCount: 3,
    source: 2,
    sink: 0,
    edges: Uint32Array.of(2, 0, 2, 0),
    capacities: Uint32Array.of(0x80000000, 0x7fffffff),
    costs: Uint32Array.of(32767, 32766),
    maximumAugmentations: 2
  };
  const result = await runScene(device, scene, 'parallel');
  const expected = 0x80000000n * 32767n + 0x7fffffffn * 32766n;
  expect(result.exactCost).toEqual([Number(expected & 0xffffffffn), Number(expected >> 32n)]);
  expect(result.flow).toEqual([0x80000000, 0x7fffffff]);
  expect(result.summary[2]).toBe(0xffffffff);
  expect(result.summary.slice(6)).toEqual([0xffffffff, 1]);
  expectNetworkFlowFeasible(scene, result.flow, 0xffffffff);
});

it('preflow relabel exhaustion publishes a feasible positive partial batch', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene: Scene = {
    nodeCount: 5,
    edges: Uint32Array.of(0, 1, 0, 2, 1, 4, 2, 3, 3, 4),
    capacities: new Uint32Array(5).fill(1),
    maximumRelabels: 1,
    maximumRelaxationRounds: 5
  };
  const result = await runScene(device, scene, 'parallel');
  expect(result.flow).toEqual([1, 0, 1, 0, 0]);
  expect(result.summary.slice(0, 5)).toEqual([1, 0, 1, 0, 1]);
  expect([result.converged, result.limited, result.invalid]).toEqual([0, 1, 0]);
  expect(result.summary[7]).toBe(0);
  expectNetworkFlowFeasible(scene, result.flow, 1);
});

for (const costed of [false, true]) {
  it(`caller-owned storage at offset 260 survives valid -> invalid -> valid (${costed ? 'costed' : 'preflow'})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const storage = <Format extends 'uint32' | 'uint32x2'>(
      name: string,
      format: Format,
      length: number,
      values?: Uint32Array
    ) => {
      const componentCount = format === 'uint32x2' ? 2 : 1;
      const buffer = device.createBuffer({
        byteLength: 260 + Math.max(1, length) * componentCount * 4,
        usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
      });
      buffers.push(buffer);
      if (values) buffer.write(values, 260);
      const handle = graph.importBuffer(
        {id: name, byteLength: buffer.byteLength, usage: buffer.usage},
        buffer
      );
      return {buffer, view: graph.createDataView(handle, {format, length, byteOffset: 260})};
    };
    const edges = storage('edges', 'uint32x2', 2, Uint32Array.of(2, 1, 1, 0));
    const capacities = storage('capacities', 'uint32', 2, Uint32Array.of(3, 3));
    const parameters = storage('parameters', 'uint32', 3, Uint32Array.of(2, 0, 0xffffffff));
    const costs = costed ? storage('costs', 'uint32', 2, Uint32Array.of(1, 2)) : undefined;
    const flow = storage('flow', 'uint32', 2),
      cut = storage('cut', 'uint32', 3),
      summary = storage('summary', 'uint32', 8),
      exactCost = storage('exact', 'uint32x2', 1);
    const invalid = storage('invalid', 'uint32', 1),
      converged = storage('converged', 'uint32', 1),
      limited = storage('limited', 'uint32', 1);
    graph.add(
      new GPUNetworkFlow({
        nodeCount: 3,
        solver: 'parallel',
        edges: edges.view,
        capacities: capacities.view,
        parameters: parameters.view,
        costs: costs?.view,
        maximumAugmentations: 1,
        maximumRelabels: costed ? undefined : 1,
        output: {
          flow: flow.view,
          cut: cut.view,
          summary: summary.view,
          exactCost: exactCost.view,
          status: {
            invalidCount: invalid.view,
            converged: converged.view,
            iterationLimitReached: limited.view
          }
        }
      })
    );
    const compiled = graph.compile();
    const read = async (entry: {buffer: Buffer}, length: number) =>
      (await readUint32(entry.buffer)).slice(65, 65 + length);
    try {
      for (const source of [2, 99, 2]) {
        parameters.buffer.write(Uint32Array.of(source, 0, 0xffffffff), 260);
        submitGraph(device, compiled, undefined);
        const valid = source === 2;
        expect(await read(flow, 2)).toEqual(valid ? [3, 3] : [0, 0]);
        expect((await read(summary, 8)).slice(0, 4)).toEqual(valid ? [1, 1, 3, 0] : [0, 0, 0, 0]);
        expect(await read(exactCost, 2)).toEqual([valid && costed ? 9 : 0, 0]);
        expect(await read(invalid, 1)).toEqual([valid ? 0 : 1]);
        expect(await read(converged, 1)).toEqual([valid ? 1 : 0]);
        expect(await read(limited, 1)).toEqual([0]);
      }
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}

for (const scenario of ['fan', 'merge-complete', 'merge-limited'] as const) {
  it(`repeated preflow ${scenario} preserves conservation across workgroups`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const branchCount = 512;
    const source = scenario === 'fan' ? 513 : 514;
    const merge = 513;
    const endpoints: number[] = [];
    const capacities: number[] = [];
    if (scenario !== 'fan') {
      endpoints.push(source, 0);
      capacities.push(1);
    }
    for (let branch = 1; branch <= branchCount; branch++) {
      endpoints.push(source, branch);
      capacities.push(scenario === 'fan' ? 1 : 2);
    }
    for (let branch = 1; branch <= branchCount; branch++) {
      endpoints.push(branch, scenario === 'fan' ? 0 : merge);
      capacities.push(scenario === 'fan' ? 1 : 2);
    }
    if (scenario !== 'fan') {
      endpoints.push(merge, 0);
      capacities.push(7);
    }
    const scene: Scene = {
      nodeCount: source + 1,
      source,
      sink: 0,
      edges: Uint32Array.from(endpoints),
      capacities: Uint32Array.from(capacities),
      maximumRelabels: scenario === 'merge-complete' ? 2 : 1,
      maximumRelaxationRounds: scenario === 'fan' ? 3 : 5
    };
    const value = scenario === 'fan' ? 512 : scenario === 'merge-complete' ? 8 : 1;
    const completed = scenario !== 'merge-limited';
    const check = (result: FlowSceneResult) => {
      expect(result.summary.slice(0, 4)).toEqual([1, completed ? 1 : 0, value, 0]);
      expect([result.converged, result.limited, result.invalid]).toEqual([
        completed ? 1 : 0,
        completed ? 0 : 1,
        0
      ]);
      expect(result.summary[7]).toBe(completed ? 1 : 0);
      if (completed) expect(result.summary[6]).toBe(value);
      if (scenario === 'fan') expect(result.flow).toEqual(new Array(branchCount * 2).fill(1));
      else {
        // Independent cuts: the direct edge plus a seven-unit merge bottleneck; with one
        // relabel only the shorter direct path can carry flow.
        expect(result.flow[0]).toBe(1);
        expect(result.flow[result.flow.length - 1]).toBe(completed ? 7 : 0);
        if (!completed)
          expect(result.flow.slice(1)).toEqual(new Array(result.flow.length - 1).fill(0));
      }
      expectNetworkFlowFeasible(scene, result.flow, value);
    };
    // Compile once, encode twice. Repeating the test also changes which pooled allocations and
    // scheduling history precede each run without assuming deterministic edge-level tie choices.
    for (let repetition = 0; repetition < 3; repetition++) {
      check(await runScene(device, scene, 'parallel', check));
    }
  });
}

for (const costed of [false, true]) {
  for (const byteOffset of [0, 260]) {
    it(`caller-owned STORAGE-only ${costed ? 'costed' : 'preflow'} buffers work at offset ${byteOffset}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      await validateFlowExecution(device, async () => {
        const graph = new GPUCommandGraph(device);
        const buffers: Buffer[] = [];
        const storage = <Format extends 'uint32' | 'uint32x2'>(
          name: string,
          format: Format,
          length: number,
          values?: Uint32Array
        ) => {
          const componentCount = format === 'uint32x2' ? 2 : 1;
          const initial = new Uint32Array(byteOffset / 4 + length * componentCount);
          if (values) initial.set(values, byteOffset / 4);
          // Creation-time data mapping initializes inputs without granting COPY_DST.
          const buffer = device.createBuffer({data: initial, usage: Buffer.STORAGE});
          buffers.push(buffer);
          expect(buffer.usage).toBe(Buffer.STORAGE);
          const handle = graph.importBuffer(
            {id: name, byteLength: buffer.byteLength, usage: buffer.usage},
            buffer
          );
          return graph.createDataView(handle, {format, length, byteOffset});
        };
        const output = {
          flow: storage('flow', 'uint32', 2),
          cut: storage('cut', 'uint32', 3),
          summary: storage('summary', 'uint32', 8),
          exactCost: storage('exact-cost', 'uint32x2', 1),
          status: {
            invalidCount: storage('invalid', 'uint32', 1),
            converged: storage('converged', 'uint32', 1),
            iterationLimitReached: storage('limited', 'uint32', 1)
          }
        };
        graph.add(
          new GPUNetworkFlow({
            nodeCount: 3,
            solver: 'parallel',
            edges: storage('edges', 'uint32x2', 2, Uint32Array.of(2, 1, 1, 0)),
            capacities: storage('capacities', 'uint32', 2, Uint32Array.of(3, 3)),
            parameters: storage('parameters', 'uint32', 3, Uint32Array.of(2, 0, 0xffffffff)),
            costs: costed ? storage('costs', 'uint32', 2, Uint32Array.of(1, 2)) : undefined,
            maximumAugmentations: 1,
            maximumRelabels: costed ? undefined : 1,
            output
          })
        );
        const readbacks = new Map<GraphDataView, Buffer>();
        const {status, ...columns} = output;
        for (const [name, source] of Object.entries({...columns, ...status})) {
          const length = source.length * (source.format === 'uint32x2' ? 2 : 1);
          const buffer = device.createBuffer({
            byteLength: length * 4,
            usage: Buffer.STORAGE | Buffer.COPY_SRC
          });
          buffers.push(buffer);
          const handle = graph.importBuffer(
            {id: `readback-${name}`, byteLength: buffer.byteLength, usage: buffer.usage},
            buffer
          );
          const destination = graph.createDataView(handle, {format: 'uint32', length});
          graph.add(
            createNetworkFlowStorageReadbackNode(graph, `readback-${name}`, source, destination)
          );
          readbacks.set(source, buffer);
        }
        const compiled = graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          const read = (view: GraphDataView) => readUint32(readbacks.get(view)!);
          expect(await read(output.flow)).toEqual([3, 3]);
          expect(await read(output.cut)).toEqual([0, 0, 1]);
          expect((await read(output.summary)).slice(0, 5)).toEqual([1, 1, 3, 0, 1]);
          expect((await read(output.summary)).slice(6)).toEqual([3, 1]);
          expect(await read(output.exactCost)).toEqual([costed ? 9 : 0, 0]);
          expect(await read(output.status.invalidCount)).toEqual([0]);
          expect(await read(output.status.converged)).toEqual([1]);
          expect(await read(output.status.iterationLimitReached)).toEqual([0]);
        } finally {
          compiled.destroy();
          for (const buffer of buffers) buffer.destroy();
        }
      });
    });
  }
}

for (const fixture of primalDualOracle.scenes) {
  it(`primal-dual matches NetworkX network_simplex (${fixture.name})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scene: Scene = {
      ...fixture,
      edges: Uint32Array.from(fixture.edges),
      capacities: Uint32Array.from(fixture.capacities),
      costs: Uint32Array.from(fixture.costs),
      mode: 'primal-dual',
      maximumCostPhases: 12,
      maximumRelaxationRounds: fixture.nodeCount
    };
    const check = (result: FlowSceneResult) => {
      expect(result.summary.slice(0, 4)).toEqual([
        1,
        1,
        fixture.expectedValue,
        fixture.expectedUnserved
      ]);
      expect(result.exactCost).toEqual([fixture.expectedCost, 0]);
      expect([result.converged, result.limited, result.invalid]).toEqual([1, 0, 0]);
      expectNetworkFlowFeasible(scene, result.flow, fixture.expectedValue);
    };
    check(await runScene(device, scene, 'parallel', check));
    for (const solver of ['parallel', 'reference'] as const) {
      const legacy = await runScene(
        device,
        {
          ...scene,
          mode: undefined,
          maximumCostPhases: undefined
        },
        solver
      );
      expect(legacy.summary.slice(0, 4)).toEqual([
        1,
        1,
        fixture.expectedValue,
        fixture.expectedUnserved
      ]);
      expect(legacy.exactCost).toEqual([fixture.expectedCost, 0]);
      expect([legacy.converged, legacy.limited, legacy.invalid]).toEqual([1, 0, 0]);
      expectNetworkFlowFeasible(scene, legacy.flow, fixture.expectedValue);
    }
    // Independent small exact CPU route also solves the same integral problem.
    const expected = computeNetworkFlowOnCPU(
      {
        edges: scene.edges,
        capacities: scene.capacities,
        costs: scene.costs,
        parameters: Uint32Array.of(0, scene.nodeCount - 1, scene.required!)
      },
      {nodeCount: scene.nodeCount}
    );
    expect(expected.summary[2]).toBe(fixture.expectedValue);
    expect(Array.from(expected.exactCost)).toEqual([fixture.expectedCost, 0]);
  });
}

it('maximumCostPhases opts auto min-cost flow into one phase for a 100,000-edge fan', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const pathCount = 50_000;
  const nodeCount = pathCount + 2;
  const edges = new Uint32Array(pathCount * 4);
  const costs = new Uint32Array(pathCount * 2);
  for (let branch = 0; branch < pathCount; branch++) {
    edges[branch * 2] = 0;
    edges[branch * 2 + 1] = branch + 1;
    edges[(pathCount + branch) * 2] = branch + 1;
    edges[(pathCount + branch) * 2 + 1] = nodeCount - 1;
    costs[branch] = 2;
    costs[pathCount + branch] = 3;
  }
  const scene: Scene = {
    nodeCount,
    edges,
    capacities: new Uint32Array(pathCount * 2).fill(1),
    costs,
    maximumCostPhases: 1,
    maximumRelaxationRounds: 3
  };
  // maximumCostPhases explicitly opts into primal-dual while solver auto chooses parallel.
  // With neither new prop, the legacy default still uses successive shortest paths.
  const check = (result: FlowSceneResult) => {
    expect(result.flow).toEqual(new Array(pathCount * 2).fill(1));
    expect(result.summary.slice(0, 5)).toEqual([1, 1, pathCount, 0, 1]);
    expect(result.summary.slice(6)).toEqual([pathCount, 1]);
    expect(result.exactCost).toEqual([5 * pathCount, 0]);
    expect([result.converged, result.limited, result.invalid]).toEqual([1, 0, 0]);
    expectNetworkFlowFeasible(scene, result.flow, pathCount);
  };
  check(await runScene(device, scene, 'auto', check));
}, 120_000);

for (const costed of [false, true]) {
  it(`primal-dual repeats merge/return batches across workgroups (costed=${costed})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const branchCount = 512;
    const source = 514;
    const merge = 513;
    const edges = [source, 0];
    const capacities = [1];
    const costs = [1];
    for (let branch = 1; branch <= branchCount; branch++) {
      edges.push(source, branch, branch, merge);
      capacities.push(2, 2);
      costs.push(1, 2);
    }
    edges.push(merge, 0);
    capacities.push(7);
    costs.push(3);
    const scene: Scene = {
      nodeCount: 515,
      source,
      sink: 0,
      edges: Uint32Array.from(edges),
      capacities: Uint32Array.from(capacities),
      costs: Uint32Array.from(costs, cost => (costed ? cost : 0)),
      mode: 'primal-dual',
      maximumCostPhases: 2,
      maximumRelaxationRounds: 5
    };
    const check = (result: FlowSceneResult) => {
      expect(result.summary.slice(0, 5)).toEqual([1, 1, 8, 0, 2]);
      expect(result.summary.slice(6)).toEqual([8, 1]);
      expect(result.flow[0]).toBe(1);
      expect(result.flow[result.flow.length - 1]).toBe(7);
      expect(result.exactCost).toEqual([costed ? 43 : 0, 0]);
      expectNetworkFlowFeasible(scene, result.flow, 8);
    };
    for (let repetition = 0; repetition < 3; repetition++)
      check(await runScene(device, scene, 'parallel', check));
  });
}

it('one primal-dual phase carries UINT32_MAX with the exact unsigned 64-bit objective', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene: Scene = {
    nodeCount: 2,
    edges: Uint32Array.of(0, 1, 0, 1),
    capacities: Uint32Array.of(0x80000000, 0x7fffffff),
    costs: Uint32Array.of(32767, 32767),
    mode: 'primal-dual',
    maximumCostPhases: 1,
    maximumRelaxationRounds: 2
  };
  const result = await runScene(device, scene, 'parallel');
  const objective = 0xffffffffn * 32767n;
  expect(result.flow).toEqual([0x80000000, 0x7fffffff]);
  expect(result.summary.slice(0, 5)).toEqual([1, 1, 0xffffffff, 0, 1]);
  expect(result.exactCost).toEqual([Number(objective & 0xffffffffn), Number(objective >> 32n)]);
  expectNetworkFlowFeasible(scene, result.flow, 0xffffffff);
});

for (const maximumAugmentations of [undefined, 0, 1, 4]) {
  it(`legacy costed defaults preserve the 129-node result with augmentation budget ${maximumAugmentations ?? 'default'}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scene: Scene = {
      nodeCount: 129,
      edges: Uint32Array.of(0, 128, 0, 128, 0, 128, 0, 128),
      capacities: Uint32Array.of(1, 1, 1, 1),
      costs: Uint32Array.of(1, 2, 3, 4),
      required: 4,
      maximumAugmentations
    };
    // Omit mode and maximumCostPhases. The helper forwards these omissions to the contributor.
    const value = maximumAugmentations === undefined ? 4 : maximumAugmentations;
    const check = (result: FlowSceneResult) => {
      expect(result.summary.slice(0, 5)).toEqual([1, Number(value === 4), value, 4 - value, value]);
      expect(result.exactCost).toEqual([(value * (value + 1)) / 2, 0]);
      expect(result.flow).toEqual(
        Array.from(scene.capacities, (_, index) => Number(index < value))
      );
      expect([result.converged, result.limited, result.invalid]).toEqual([
        Number(value === 4),
        Number(value < 4),
        0
      ]);
      expectNetworkFlowFeasible(scene, result.flow, value);
    };
    check(await runScene(device, scene, 'auto', check));
  });
}

for (const nodeCount of [255, 256, 257, 511, 512, 513, 1025]) {
  it(`explicit primal-dual solves a ${nodeCount}-vertex unit-cost chain with nonzero source`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const vertices = [
      nodeCount - 1,
      ...Array.from({length: nodeCount - 2}, (_, index) => index + 1),
      0
    ];
    const scene: Scene = {
      nodeCount,
      source: nodeCount - 1,
      sink: 0,
      edges: Uint32Array.from(
        vertices.slice(1).flatMap((vertex, index) => [vertices[index], vertex])
      ),
      capacities: new Uint32Array(nodeCount - 1).fill(1),
      costs: new Uint32Array(nodeCount - 1).fill(1),
      mode: 'primal-dual',
      maximumCostPhases: 1,
      maximumRelaxationRounds: nodeCount
    };
    const check = (result: FlowSceneResult) => {
      // Closed form: one unit along each of the nodeCount - 1 edges, each costing one.
      expect(result.flow).toEqual(new Array(nodeCount - 1).fill(1));
      expect(result.exactCost).toEqual([nodeCount - 1, 0]);
      expect(result.summary.slice(0, 5)).toEqual([1, 1, 1, 0, 1]);
      expect(result.summary.slice(6)).toEqual([1, 1]);
      expect([result.converged, result.limited, result.invalid]).toEqual([1, 0, 0]);
      expectNetworkFlowFeasible(scene, result.flow, 1);
    };
    check(await runScene(device, scene, 'parallel', check));
  }, 120_000);
}

it('explicit primal-dual reverse cancellation crosses 256/512 vertex boundaries on repeated encodings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const vertices = [255, 256, 257, 511, 512, 0];
  const endpoints = [0, 1, 0, 2, 1, 3, 1, 4, 2, 3, 2, 4, 3, 5, 4, 5];
  const scene: Scene = {
    nodeCount: 513,
    source: 255,
    sink: 0,
    edges: Uint32Array.from(endpoints.map(vertex => vertices[vertex])),
    capacities: new Uint32Array(8).fill(1),
    costs: Uint32Array.of(0, 0, 1, 2, 2, 100, 0, 0),
    mode: 'primal-dual',
    maximumCostPhases: 2,
    maximumRelaxationRounds: 6
  };
  const check = (result: FlowSceneResult) => {
    // The optimal matching reroutes A->X to A->Y and B->X for total cost 2 + 2.
    expect(result.flow).toEqual([1, 1, 0, 1, 1, 0, 1, 1]);
    expect(result.summary.slice(0, 5)).toEqual([1, 1, 2, 0, 2]);
    expect(result.exactCost).toEqual([4, 0]);
    expect(result.summary.slice(6)).toEqual([2, 1]);
    expect([result.converged, result.limited, result.invalid]).toEqual([1, 0, 0]);
    expectNetworkFlowFeasible(scene, result.flow, 2);
  };
  for (let repetition = 0; repetition < 3; repetition++)
    check(await runScene(device, scene, 'parallel', check));
});

for (const fixture of movementOracle.flow) {
  it(`legacy defaults and primal-dual match the existing NetworkX objective (${fixture.name})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scene: Scene = {
      nodeCount: fixture.nodeCount,
      edges: Uint32Array.from(fixture.edges.flat()),
      capacities: Uint32Array.from(fixture.capacities),
      costs: Uint32Array.from(fixture.costs)
    };
    for (const [solver, mode] of [
      ['auto', undefined],
      ['parallel', undefined],
      ['parallel', 'primal-dual']
    ] as const) {
      const result = await runScene(device, {...scene, mode}, solver);
      expect(result.summary.slice(0, 4)).toEqual([1, 1, fixture.maximumFlow, 0]);
      expect(result.exactCost).toEqual([fixture.minimumCost, 0]);
      expect([result.converged, result.limited, result.invalid]).toEqual([1, 0, 0]);
      expectNetworkFlowFeasible(scene, result.flow, fixture.maximumFlow);
    }
  });
}

for (const solver of ['reference', 'parallel'] as const)
  for (const mode of solver === 'reference'
    ? [undefined]
    : (['successive-shortest-path', 'primal-dual'] as const))
    for (const required of [1, 0xffffffff])
      it(`independent integral enumeration checks ${solver} ${mode} request ${required}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const result = await runScene(
          device,
          {
            ...REVERSE_RESIDUAL_SCENE,
            edges: Uint32Array.from(REVERSE_RESIDUAL_SCENE.edges),
            capacities: Uint32Array.from(REVERSE_RESIDUAL_SCENE.capacities),
            costs: Uint32Array.from(REVERSE_RESIDUAL_SCENE.costs),
            mode,
            required,
            maximumAugmentations: 8,
            maximumRelaxationRounds: solver === 'parallel' ? 6 : undefined,
            maximumCostPhases: mode === 'primal-dual' ? 6 : undefined
          },
          solver
        );
        expectEnumeratedIntegralFlow(result, required);
      });
