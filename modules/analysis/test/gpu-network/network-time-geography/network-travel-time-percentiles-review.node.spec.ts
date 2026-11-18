// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  computeNetworkTravelTimePercentilesOnCPU,
  CPUNetworkTravelTimePercentilesOperation
} from '../../../src/gpu-network/network-time-geography/cpu-network-travel-time-percentiles-operation';
import {GPUNetworkTravelTimePercentilesOperation} from '../../../src/gpu-network/network-time-geography/gpu-network-travel-time-percentiles-operation';
import {GPUNetworkTravelTimePercentiles} from '../../../src/gpu-network/network-time-geography/gpu-network-travel-time-percentiles';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

const INPUT = {
  stateVertices: Uint32Array.of(0, 1),
  transitions: Uint32Array.of(0, 1),
  costs: Float32Array.of(5, 85),
  initialState: Uint32Array.of(0),
  window: Float32Array.of(0, 1),
  vertexCount: 2,
  departureCount: 2,
  percentiles: [50.00000001]
};
const PROPS = {
  id: 'percentile-review',
  stateCount: 2,
  transitionCount: 1,
  stateVertices: 'vertices',
  transitions: 'transitions',
  costs: 'costs',
  initialState: 'initial',
  window: 'window',
  vertexCount: 2,
  departureCount: 2,
  percentiles: INPUT.percentiles,
  output: {travelTimes: 'times', reachedCounts: 'reached', status: 'status'}
};

function getCPUInputs(): Record<string, GPUCPUVector> {
  const vector = (format: GPUCPUVector['format'], values: Uint32Array | Float32Array) => {
    const length = values.length / (format.endsWith('x2') ? 2 : 1);
    return {format, length, chunks: [{values, length}], pack: () => values};
  };
  return {
    stateVertices: vector('uint32', INPUT.stateVertices),
    transitions: vector('uint32x2', INPUT.transitions),
    costs: vector('float32x2', INPUT.costs),
    initialState: vector('uint32', INPUT.initialState),
    window: vector('float32', INPUT.window)
  };
}

function createGPUFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const views = {
    stateVertices: createTransientView(graph, 'vertices', 'uint32', 2),
    transitions: createTransientView(graph, 'transitions', 'uint32x2', 1),
    costs: createTransientView(graph, 'costs', 'float32x2', 1),
    initialState: createTransientView(graph, 'initial', 'uint32', 1),
    window: createTransientView(graph, 'window', 'float32', 2),
    travelTimes: createTransientView(graph, 'times', 'float32', 2),
    reachedCounts: createTransientView(graph, 'reached', 'uint32', 2),
    status: createTransientView(graph, 'status', 'uint32', 4)
  };
  return {
    device,
    graph,
    views,
    props: {
      id: PROPS.id,
      stateVertices: views.stateVertices,
      transitions: views.transitions,
      costs: views.costs,
      initialState: views.initialState,
      window: views.window,
      vertexCount: 2,
      departureCount: 2,
      percentiles: [50],
      output: {
        travelTimes: views.travelTimes,
        reachedCounts: views.reachedCounts,
        status: views.status
      }
    }
  };
}

it('honors a zero relaxation budget in the pure percentile function and CPU operation', async () => {
  const input = {...INPUT, maximumRelaxations: 0};
  const pure = computeNetworkTravelTimePercentilesOnCPU(input);
  expect(pure.travelTimes).toEqual(Float32Array.of(0, Infinity));
  expect(pure.reachedCounts).toEqual(Uint32Array.of(2, 0));
  expect(pure.status).toEqual(Uint32Array.of(1, 0, 1, 0));
  const operation = new CPUNetworkTravelTimePercentilesOperation({...PROPS, maximumRelaxations: 0});
  const result = await operation.execute({inputs: getCPUInputs()});
  expect(result['times']).toEqual(Float32Array.of(0, Infinity));
  expect(result['reached']).toEqual(Uint32Array.of(2, 0));
  expect(result['status']).toEqual(Uint32Array.of(1, 0, 1, 0));
});

it.each([
  false,
  true
])('retains a zero budget supplied as an inherited getter: %s', async inherited => {
  const budget = inherited
    ? Object.create({
        get maximumRelaxations() {
          return 0;
        }
      })
    : {maximumRelaxations: 0};
  const props = Object.assign(budget, PROPS);
  const operation = new CPUNetworkTravelTimePercentilesOperation(props);
  const actual = await operation.execute({inputs: getCPUInputs()});
  expect(actual['times']).toEqual(Float32Array.of(0, Infinity));
  expect(actual['status']).toEqual(Uint32Array.of(1, 0, 1, 0));
  const input = Object.assign(
    inherited
      ? Object.create({
          get maximumRelaxations() {
            return 0;
          }
        })
      : {maximumRelaxations: 0},
    INPUT
  );
  expect(computeNetworkTravelTimePercentilesOnCPU(input).travelTimes).toEqual(
    Float32Array.of(0, Infinity)
  );
  const fixture = createGPUFixture();
  try {
    const gpuOperation = new GPUNetworkTravelTimePercentilesOperation(props);
    const contributor = gpuOperation.createContributor(
      port => fixture.views[port as keyof typeof fixture.views] as never
    ) as GPUNetworkTravelTimePercentiles;
    expect(contributor.props.maximumRelaxations).toBe(0);
    const directProps = Object.assign(
      inherited
        ? Object.create({
            get maximumRelaxations() {
              return 0;
            }
          })
        : {maximumRelaxations: 0},
      fixture.props
    );
    const direct = new GPUNetworkTravelTimePercentiles(directProps);
    expect(direct.props.maximumRelaxations).toBe(0);
  } finally {
    fixture.device.destroy();
  }
});

it.each([
  50.00000001,
  NaN
])('snapshots CPU percentile settings before caller mutation to %s', async mutation => {
  const percentiles = [50];
  const operation = new CPUNetworkTravelTimePercentilesOperation({...PROPS, percentiles});
  expect((await operation.execute({inputs: getCPUInputs()}))['times']).toEqual(
    Float32Array.of(0, 5)
  );
  percentiles[0] = mutation;
  percentiles.push(100);
  expect((await operation.execute({inputs: getCPUInputs()}))['times']).toEqual(
    Float32Array.of(0, 5)
  );
});

for (const boundary of ['contributor', 'operation'] as const) {
  it.each([
    50.00000001,
    NaN
  ])(`snapshots GPU ${boundary} percentile settings before mutation to %s`, mutation => {
    const fixture = createGPUFixture();
    const percentiles = [50];
    const operation = new GPUNetworkTravelTimePercentilesOperation({...PROPS, percentiles});
    const direct = new GPUNetworkTravelTimePercentiles({...fixture.props, percentiles});
    percentiles[0] = mutation;
    percentiles.push(100);
    try {
      const contributor =
        boundary === 'contributor'
          ? direct
          : (operation.createContributor(
              port => fixture.views[port as keyof typeof fixture.views] as never
            ) as GPUNetworkTravelTimePercentiles);
      const source = getGPUCommandNodeKernelSources(contributor.getCommandNodes(fixture.graph)).at(
        -1
      )!.source;
      expect(source).toContain('const RANKS = array<u32, 1>(1u);');
    } finally {
      operation.destroy();
      fixture.device.destroy();
    }
  });
}

it('rejects a missing percentile in the pure function and CPU operation with a named error', () => {
  const percentiles = new Array<number>(1);
  expect(() => computeNetworkTravelTimePercentilesOnCPU({...INPUT, percentiles})).toThrow(
    /^network-travel-time-percentiles.*percentiles/
  );
  expect(() => new CPUNetworkTravelTimePercentilesOperation({...PROPS, percentiles})).toThrow(
    /^percentile-review.*percentiles/
  );
});

it('rejects a missing percentile in the GPU contributor and operation lowering with a named error', () => {
  const fixture = createGPUFixture();
  const percentiles = new Array<number>(1);
  try {
    expect(() => new GPUNetworkTravelTimePercentiles({...fixture.props, percentiles})).toThrow(
      /^percentile-review.*percentiles/
    );
    const operation = new GPUNetworkTravelTimePercentilesOperation({...PROPS, percentiles});
    expect(() =>
      operation.createContributor(
        port => fixture.views[port as keyof typeof fixture.views] as never
      )
    ).toThrow(/^percentile-review.*percentiles/);
  } finally {
    fixture.device.destroy();
  }
});
