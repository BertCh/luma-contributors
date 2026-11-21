// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeGeodesicPairsOnCPU} from '../../../src/gpu-spatial-analysis/geometry-measures/cpu-geodesic-pairs-operation';
import {CPUGeodesicPairsOperation} from '../../../src/gpu-spatial-analysis/geometry-measures/cpu-geodesic-pairs-operation';
import {computeLocalFrameProjectionOnCPU} from '../../../src/gpu-spatial-analysis/geometry-measures/cpu-local-frame-projection-operation';
import {createGPULocalFrame} from '../../../src/analysis-core/contracts/local-frame';
import {GPUGeodesicPairsOperation} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs-operation';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('the generated GPU inverse chooses the independently qualified model at two and three updates', () => {
  for (const budget of [2, 3]) {
    const device = createNullWebGPUDevice(),
      graph = new GPUCommandGraph(device);
    const operation = new GPUGeodesicPairsOperation({
      origins: 'origins',
      targets: 'targets',
      pairCount: 1,
      maximumIterations: budget,
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'},
      output: {
        distances: 'distance',
        status: {
          converged: 'converged',
          approximation: 'approximation',
          iterationLimitReached: 'limit'
        }
      }
    });
    const views = {
      origins: createTransientView(graph, 'origins', 'float32x2', 1),
      targets: createTransientView(graph, 'targets', 'float32x2', 1),
      ...Object.fromEntries(
        Object.entries(operation.outputs).map(([port, descriptor]) => [
          port,
          createTransientView(graph, port, descriptor.format, descriptor.length)
        ])
      )
    };
    const nodes = operation
      .createContributor(port => views[port as keyof typeof views] as never)
      .getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    memory.write(views.origins, new Float32Array([0, 20]));
    memory.write(views.targets, new Float32Array([170, 60]));
    try {
      executeCommandNodesOnCPU(nodes, memory);
      expect(
        ['converged', 'approximation', 'iterationLimitReached'].map(
          name => memory.readUint32(views[name as keyof typeof views])[0]
        )
      ).toEqual(budget === 2 ? [0, 1, 1] : [1, 0, 0]);
      const distance = memory.readFloat32(views['distances' as keyof typeof views])[0];
      expect(Number.isFinite(distance)).toBe(true);
      expect(Math.abs(distance - (budget === 2 ? 11073359.15 : 11091265.03))).toBeLessThan(
        budget === 2 ? 30 : 350
      );
    } finally {
      device.destroy();
    }
  }
});

it('captures geodesic budgets and nested ellipsoid values before lowering or CPU execution', async () => {
  const props = {
    origins: 'origins',
    targets: 'targets',
    pairCount: 1,
    maximumIterations: 1,
    spatialContext: {
      coordinateSpace: 'longitude-latitude' as const,
      metric: 'ellipsoidal' as const,
      ellipsoid: {semiMajorAxis: 6378137, flattening: 1 / 298.257223563}
    },
    output: {
      distances: 'distance',
      status: {
        converged: 'converged',
        approximation: 'approximation',
        iterationLimitReached: 'limit'
      }
    }
  };
  const gpu = new GPUGeodesicPairsOperation(props),
    cpu = new CPUGeodesicPairsOperation(props);
  props.maximumIterations = 100;
  props.spatialContext.ellipsoid.semiMajorAxis = 1;
  props.spatialContext.ellipsoid.flattening = 0;
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  const views = {
    origins: createTransientView(graph, 'origins', 'float32x2', 1),
    targets: createTransientView(graph, 'targets', 'float32x2', 1),
    distances: createTransientView(graph, 'distance', 'float32', 1),
    converged: createTransientView(graph, 'converged', 'uint32', 1),
    approximation: createTransientView(graph, 'approximation', 'uint32', 1),
    iterationLimitReached: createTransientView(graph, 'limit', 'uint32', 1)
  };
  try {
    const contributor = gpu.createContributor(
      port => views[port as keyof typeof views] as never
    ) as unknown as {maximumIterations: number; ellipsoid: {semiMajorAxis: number}};
    expect(contributor.maximumIterations).toBe(1);
    expect(contributor.ellipsoid.semiMajorAxis).toBe(6378137);
    const vector = (values: Float32Array) => ({
      format: 'float32x2' as const,
      length: 1,
      chunks: [{values, length: 1}],
      pack: () => values
    });
    const result = await cpu.execute({
      inputs: {
        origins: vector(new Float32Array([0, 30])),
        targets: vector(new Float32Array([100, -45]))
      }
    });
    expect(
      ['converged', 'approximation', 'limit'].map(name => (result[name] as Uint32Array)[0])
    ).toEqual([0, 1, 1]);
    expect(Math.abs((result['distance'] as Float32Array)[0] - 13052084.956326788)).toBeLessThan(2);
  } finally {
    device.destroy();
  }
});

for (const budget of [2, 3]) {
  it(`chooses the float32-qualified geodesic model after ${budget} iterations`, () => {
    const actual = computeGeodesicPairsOnCPU(
      {origins: new Float32Array([0, 20]), targets: new Float32Array([170, 60])},
      {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'},
      budget
    );
    expect([actual.converged[0], actual.approximation[0], actual.iterationLimitReached[0]]).toEqual(
      budget === 2 ? [0, 1, 1] : [1, 0, 0]
    );
    // Independent pyproj ellipsoid and authalic-sphere inverses (oracle.py).
    expect(Number.isFinite(actual.distances[0])).toBe(true);
    expect(Math.abs(actual.distances[0] - (budget === 2 ? 11073359.15 : 11091265.03))).toBeLessThan(
      2
    );
  });
}

it('honors one explicit inverse iteration in the public CPU operation', async () => {
  const operation = new CPUGeodesicPairsOperation({
    origins: 'origins',
    targets: 'targets',
    pairCount: 1,
    maximumIterations: 1,
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'},
    output: {
      distances: 'distances',
      status: {
        converged: 'converged',
        approximation: 'approximation',
        iterationLimitReached: 'iterationLimitReached'
      }
    }
  });
  const vector = (values: Float32Array) => ({
    format: 'float32x2' as const,
    length: 1,
    chunks: [{values, length: 1}],
    pack: () => values
  });
  const actual = await operation.execute({
    inputs: {
      origins: vector(new Float32Array([0, 30])),
      targets: vector(new Float32Array([100, -45]))
    }
  });
  expect(
    ['converged', 'approximation', 'iterationLimitReached'].map(
      name => (actual[name] as Uint32Array)[0]
    )
  ).toEqual([0, 1, 1]);
  // Independent spherical central angle on the mean-Earth sphere (Python oracle).
  expect(Math.abs((actual['distances'] as Float32Array)[0] - 13052084.96)).toBeLessThan(2);
});

it('ties complete ordinary geodesic statuses to the PROJ inverse distance', () => {
  const actual = computeGeodesicPairsOnCPU(
    {origins: new Float32Array([-73.75, 40.5]), targets: new Float32Array([2.25, 48.75])},
    {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'}
  );
  expect(Array.from(actual.converged)).toEqual([1]);
  expect(Array.from(actual.approximation)).toEqual([0]);
  expect(Array.from(actual.iterationLimitReached)).toEqual([0]);
  expect(Math.abs(actual.distances[0] - 5846818.164066021)).toBeLessThan(0.5);
});

it('matches direct PROJ local projections away from the origin', () => {
  for (const kind of ['azimuthal-equidistant', 'lambert-equal-area'] as const) {
    const actual = computeLocalFrameProjectionOnCPU(
      new Float32Array([0.5, 0.25, 1, 1]),
      createGPULocalFrame({origin: [0, 0], kind}),
      [-1, -1, 1, 1]
    );
    const expected =
      kind === 'azimuthal-equidistant'
        ? [55597.187279835605, 27799.122892627045, 111183.78889285575, 111200.72530193203]
        : [55659.174944973725, 27643.811936381957, 111305.47485636259, 110577.19525609486];
    actual.output.forEach((value, index) => {
      expect(Number.isFinite(value)).toBe(true);
      expect(Math.abs(value - expected[index])).toBeLessThan(0.02);
    });
  }
});
