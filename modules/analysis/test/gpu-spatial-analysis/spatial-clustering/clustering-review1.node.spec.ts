// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUDBSCANParameterValues,
  type GPUDBSCANSettings
} from '../../../src/gpu-spatial-analysis/spatial-clustering/spatial-clustering-parameters';
import {
  GPUOPTICS,
  OPTICS_PLANAR_DISTANCE_WGSL,
  type GPUOPTICSProps,
  type GPUOPTICSOutput
} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {getSpatialClusteringSharedWGSL} from '../../../src/gpu-spatial-analysis/spatial-clustering/spatial-clustering-kernels';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';

useCPUExecutorTestHooks();

it('preserves JavaScript typed-array bounds when packing DBSCAN parameters (B1)', () => {
  const actual = getGPUDBSCANParameterValues({
    bounds: Float32Array.of(0, 0, 12, 12) as unknown as GPUDBSCANSettings['bounds'],
    epsilon: 2,
    minimumSamples: 3
  });
  expect(Array.from(actual)).toEqual([0, 0, 12, 12, 2, 3, 0, 0]);
});

for (const epsilon of [2e19, 1e-23])
  it(`rejects unsupported squared-distance planar epsilon ${epsilon} (earlier-fix note)`, () => {
    expect(() =>
      getGPUDBSCANParameterValues({bounds: [0, 0, 12, 12], epsilon, minimumSamples: 2})
    ).toThrow(/GPUDBSCAN epsilon.*planar.*2\^-63.*2\^63/);
  });

it('enforces normal squared-distance limits in the production planar WGSL helper', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const parameters = createTransientView(graph, 'parameters', 'float32', 8);
    const output = createTransientView(graph, 'output', 'uint32', 2);
    const node = createWGSLKernelNode(graph, {
      id: 'planar-range',
      operation: 'review',
      invocationCount: 1,
      bindings: [
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        {name: 'output', view: output, type: 'u32', access: 'read_write'}
      ],
      declarations: getSpatialClusteringSharedWGSL(1, 1),
      body: `let lattice = readLattice();
        output[outputOffset] = u32(lattice.valid);
        output[outputOffset + 1u] = 0u;
        if (lattice.valid) { output[outputOffset + 1u] = u32(isWithinEpsilon(lattice, 0.0, 0.0, parameters[parametersOffset + 4u] * 2.0, 0.0)); }`
    });
    const memory = createCPUGraphMemory();
    for (const epsilon of [1e-23, 2 ** -63, 2 ** 63, 2e19]) {
      memory.write(parameters, Float32Array.of(0, 0, epsilon * 2, 0, epsilon, 2, 0, 0));
      executeCommandNodesOnCPU([node], memory);
      expect(Array.from(memory.readUint32(output))).toEqual([
        Number(epsilon >= 2 ** -63 && epsilon <= 2 ** 63),
        0
      ]);
    }
  } finally {
    device.destroy();
  }
});

function createOPTICS(
  graph: GPUCommandGraph,
  options: Partial<GPUOPTICSProps> = {},
  fields: readonly (keyof NonNullable<GPUOPTICSOutput['ports']>['status'])[] = ['reason']
) {
  let sequence = 0;
  const view = <Format extends 'float32x2' | 'float32' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `review-${sequence++}`, format, length);
  return new GPUOPTICS({
    positions: view('float32x2', 2),
    minimumSamples: 1,
    maximumDistance: 1,
    extractionDistance: 1,
    neighborIndex: 'grid',
    neighborCapacity: 2,
    output: {
      ordering: view('uint32', 2),
      coreDistances: view('float32', 2),
      reachability: view('float32', 2),
      predecessors: view('uint32', 2),
      labels: view('uint32', 2),
      ports: {status: Object.fromEntries(fields.map(name => [name, view('uint32', 1)]))}
    },
    ...options
  });
}

it('computes a finite native OPTICS distance when raw coordinate squares overflow', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const output = createTransientView(graph, 'distance-output', 'float32', 1);
    const node = createWGSLKernelNode(graph, {
      id: 'native-distance',
      operation: 'OPTICSReview',
      invocationCount: 1,
      declarations: OPTICS_PLANAR_DISTANCE_WGSL,
      bindings: [{name: 'output', view: output, type: 'f32', access: 'read_write'}],
      body: 'output[outputOffset] = getPointDistance(vec2<f32>(0.0), vec2<f32>(1180591620717411303424.0));'
    });
    const memory = createCPUGraphMemory();
    executeCommandNodesOnCPU([node], memory);
    const distance = memory.readFloat32(output)[0];
    // Two equal legs of 2^70: hypotenuse sqrt(2)*2^70, a normal finite float32.
    const expected = Math.fround(Math.SQRT2 * 2 ** 70);
    expect(Number.isFinite(distance)).toBe(true);
    expect(Math.abs(distance / expected - 1)).toBeLessThan(2e-7);
  } finally {
    device.destroy();
  }
});

it('keeps saturated OPTICS core and frontier radius squares finite', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const nodes = createOPTICS(graph, {
      minimumSamples: 2,
      maximumDistance: 3.4028234663852886e38
    }).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    for (const stage of ['core', 'reach']) {
      const setup = nodes.find(node => node.id.endsWith(`${stage}-neighbors-setup-parameters`))!;
      const record = getWGSLKernelNodeRecord(graph, setup.id)!;
      const parameters = record.bindings.find(binding => binding.name === 'parameters')!.view;
      executeCommandNodesOnCPU([setup], memory);
      const radius = memory.readFloat32(parameters)[4];
      expect(Number.isFinite(Math.fround(radius * radius)), stage).toBe(true);
      expect(radius, stage).toBeGreaterThan(0);
    }
  } finally {
    device.destroy();
  }
});

for (const temporal of [false, true]) {
  it(`keeps widened OPTICS ${temporal ? 'time' : 'distance'} radius finite at the float32 maximum (S3)`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const maximum = 3.4028234663852886e38; // IEEE-754 largest finite binary32.
      const model = createOPTICS(
        graph,
        temporal
          ? {
              times: createTransientView(graph, 'times', 'float32', 2),
              maximumTimeDistance: maximum,
              temporalContext: {
                encoding: 'float32-relative',
                epoch: 0n,
                unit: 'second',
                axis: {kind: 'irregular'}
              }
            }
          : {maximumDistance: maximum}
      );
      expect(() => model.getCommandNodes(graph)).not.toThrow();
    } finally {
      device.destroy();
    }
  });
}

for (const name of ['iterationCount', 'converged', 'reason', 'iterationLimitReached'] as const) {
  it(`binds OPTICS loss only when ${name} consumes it (S4)`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const nodes = createOPTICS(graph, {}, [name]).getCommandNodes(graph);
      const node = nodes.find(candidate => candidate.id.endsWith('-iteration-status'))!;
      const source = getGPUCommandNodeKernelSources([node])[0].source;
      expect(source.includes('var<storage, read> overflow:')).toBe(
        name === 'reason' || name === 'iterationLimitReached'
      );
    } finally {
      device.destroy();
    }
  });
}

it('rejects overlapping named OPTICS destinations at construction', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const model = createOPTICS(graph);
    const shared = createTransientView(graph, 'shared-status-word', 'uint32', 1);
    expect(
      () =>
        new GPUOPTICS({
          ...model.props,
          output: {
            ...model.props.output,
            ports: {status: {converged: shared, invalidCount: shared}}
          }
        })
    ).toThrow(
      /output.ports.status.converged and output.ports.status.invalidCount must not overlap/
    );
  } finally {
    device.destroy();
  }
});

it('keeps literal reason/limit outcomes when finalization consumes candidate loss (S4)', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const model = createOPTICS(graph, {}, [
      'candidateOverflow',
      'reason',
      'iterationLimitReached',
      'invalidCount'
    ]);
    const node = model
      .getCommandNodes(graph)
      .find(candidate => candidate.id.endsWith('-iteration-status'))!;
    const ports = model.props.output.ports!.status;
    const frontierStatus = getWGSLKernelNodeRecord(graph, node.id)!.bindings.find(
      binding => binding.name === 'status'
    )!.view;
    const memory = createCPUGraphMemory();
    for (const scene of [
      {status: [0, 0, 0], loss: 1, expected: [1, 0]},
      {status: [2, 0, 0], loss: 0, expected: [2, 1]},
      {status: [1, 1, 1], loss: 0, expected: [0, 0]}
    ]) {
      memory.write(frontierStatus, Uint32Array.from(scene.status));
      memory.write(ports.candidateOverflow!, Uint32Array.of(scene.loss));
      executeCommandNodesOnCPU([node], memory);
      expect([
        memory.readUint32(ports.reason!)[0],
        memory.readUint32(ports.iterationLimitReached!)[0]
      ]).toEqual(scene.expected);
      expect(Array.from(memory.readUint32(ports.invalidCount!))).toEqual([scene.status[2]]);
    }
  } finally {
    device.destroy();
  }
});
