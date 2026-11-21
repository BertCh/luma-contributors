// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  computeNetworkTravelTimePercentilesOnCPU,
  CPUNetworkTravelTimePercentilesOperation
} from '../../../src/gpu-network/network-time-geography/cpu-network-travel-time-percentiles-operation';
import {GPUNetworkTravelTimePercentilesOperation} from '../../../src/gpu-network/network-time-geography/gpu-network-travel-time-percentiles-operation';
import {GPUNetworkTravelTimePercentiles} from '../../../src/gpu-network/network-time-geography/gpu-network-travel-time-percentiles';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
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

it('rejects nested routing shapes before any graph mutation (NET-118)', () => {
  const fixture = createGPUFixture();
  const excessiveStates = createTransientView(fixture.graph, 'excessive-states', 'uint32', 257);
  const boundaryStates = createTransientView(fixture.graph, 'boundary-states', 'uint32', 256);
  const sampledCosts = createTransientView(fixture.graph, 'sampled-costs', 'float32x2', 2);
  const create = vi.spyOn(fixture.graph, 'createTransientBuffer');
  const add = vi.spyOn(fixture.graph, 'add');
  try {
    for (const [override, diagnostic] of [
      [{stateVertices: excessiveStates}, /stateVertices rows/],
      [{costModel: 'piecewise-arrival' as const, samplesPerTransition: 2}, /costs/],
      [{maximumRelaxations: 65537}, /maximumRelaxations/],
      [{costModel: 'unknown' as never}, /costModel/],
      [
        {
          costModel: 'piecewise-arrival' as const,
          samplesPerTransition: 2,
          waitingPolicy: 'unknown' as never,
          costs: sampledCosts
        },
        /waitingPolicy/
      ]
    ] as const) {
      expect(() => new GPUNetworkTravelTimePercentiles({...fixture.props, ...override})).toThrow(
        diagnostic
      );
      expect(create).not.toHaveBeenCalled();
      expect(add).not.toHaveBeenCalled();
    }
    const boundary = new GPUNetworkTravelTimePercentiles({
      ...fixture.props,
      stateVertices: boundaryStates,
      costModel: 'piecewise-arrival',
      samplesPerTransition: 2,
      costs: sampledCosts,
      maximumRelaxations: 65536
    });
    expect(create).not.toHaveBeenCalled();
    expect(boundary.getCommandNodes(fixture.graph)).toHaveLength(6);
  } finally {
    fixture.device.destroy();
  }
});

it('keeps the CPU percentile size contract independent of the GPU state cap (NET-118)', () => {
  // 257 states all map to vertex 0. The initial state's zero duration wins every departure.
  const result = computeNetworkTravelTimePercentilesOnCPU({
    ...INPUT,
    stateVertices: new Uint32Array(257),
    transitions: new Uint32Array(0),
    costs: new Float32Array(0),
    vertexCount: 1,
    percentiles: [100]
  });
  expect(Array.from(result.travelTimes)).toEqual([0]);
  expect(Array.from(result.reachedCounts)).toEqual([2]);
});

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
    expect(() => new GPUNetworkTravelTimePercentilesOperation({...PROPS, percentiles})).toThrow(
      /^percentile-review.*percentiles/
    );
  } finally {
    fixture.device.destroy();
  }
});

it.each([
  [{stateCount: 0}, /stateCount/],
  [{stateCount: 259}, /stateCount/],
  [{transitionCount: -1}, /transitionCount/],
  [{transitionCount: 4097}, /transitionCount/],
  [{maximumRelaxations: 65537}, /maximumRelaxations/],
  [{costModel: 'piecewise-arrival', samplesPerTransition: 1}, /samplesPerTransition/],
  [{costModel: 'piecewise-arrival', samplesPerTransition: 257}, /samplesPerTransition/],
  [{vertexCount: 0}, /vertexCount/],
  [{departureCount: 0}, /departureCount/],
  [{departureCount: 61}, /departureCount/],
  [{percentiles: []}, /percentiles/],
  [{percentiles: new Array(17).fill(50)}, /percentiles/],
  [{percentiles: [90, 50]}, /percentiles/],
  [{percentiles: [NaN]}, /percentiles/],
  [{percentiles: [0]}, /percentiles/],
  [{percentiles: [101]}, /percentiles/]
] as const)('validates GPU percentile declarations before lowering: %j', (override, diagnostic) => {
  expect(
    () => new GPUNetworkTravelTimePercentilesOperation({...PROPS, ...override} as typeof PROPS)
  ).toThrow(diagnostic);
});

it.each([
  {stateCount: 3, transitionCount: 1, diagnostic: /stateCount/},
  {stateCount: 2, transitionCount: 2, diagnostic: /transitionCount/}
])('checks declared percentile dimensions against resolved rows $stateCount/$transitionCount', ({
  diagnostic,
  ...shape
}) => {
  const fixture = createGPUFixture();
  try {
    const operation = new GPUNetworkTravelTimePercentilesOperation({...PROPS, ...shape});
    expect(() =>
      operation.createContributor(
        port => fixture.views[port as keyof typeof fixture.views] as never
      )
    ).toThrow(diagnostic);
  } finally {
    fixture.device.destroy();
  }
});

it('executes a CPU percentile operation above every GPU-only cap', async () => {
  // All 259 states map to vertex 0; self transitions cost zero. Every departure has travel time 0.
  const operation = new CPUNetworkTravelTimePercentilesOperation({
    ...PROPS,
    stateCount: 259,
    transitionCount: 4097,
    departureCount: 61,
    percentiles: new Array<number>(17).fill(100),
    vertexCount: 1,
    maximumRelaxations: 65537,
    costModel: 'piecewise-arrival',
    samplesPerTransition: 257
  });
  const vector = (
    format: GPUCPUVector['format'],
    values: Uint32Array | Float32Array
  ): GPUCPUVector => {
    const length = values.length / (format.endsWith('x2') ? 2 : 1);
    return {format, length, chunks: [{values, length}], pack: () => values};
  };
  const costs = Float32Array.from(
    {length: 4097 * 257 * 2},
    (_, index) => Math.floor(index / 2) % 257
  );
  const actual = await operation.execute({
    inputs: {
      stateVertices: vector('uint32', new Uint32Array(259)),
      transitions: vector('uint32x2', new Uint32Array(4097 * 2)),
      costs: vector('float32x2', costs),
      initialState: vector('uint32', Uint32Array.of(0)),
      window: vector('float32', Float32Array.of(0, 1))
    }
  });
  expect(actual['times']).toEqual(new Float32Array(17));
  expect(actual['reached']).toEqual(Uint32Array.of(61));
  expect(actual['status']).toEqual(new Uint32Array(122).fill(1));
});

it.each([
  [{costModel: 'bogus' as never, costs: Float32Array.of(5, 0)}, /costModel/],
  [
    {
      costModel: 'piecewise-arrival' as const,
      samplesPerTransition: 2,
      waitingPolicy: 'bogus' as never,
      costs: Float32Array.of(0, 5, 1, 6)
    },
    /waitingPolicy/
  ]
])('rejects invalid percentile routing options with valid cost rows: %j', (override, diagnostic) => {
  expect(() => computeNetworkTravelTimePercentilesOnCPU({...INPUT, ...override})).toThrow(
    diagnostic
  );
  for (const Operation of [
    CPUNetworkTravelTimePercentilesOperation,
    GPUNetworkTravelTimePercentilesOperation
  ]) {
    expect(
      () =>
        new Operation({
          ...PROPS,
          costModel: override.costModel,
          samplesPerTransition:
            'samplesPerTransition' in override ? override.samplesPerTransition : undefined,
          waitingPolicy: 'waitingPolicy' in override ? override.waitingPolicy : undefined
        })
    ).toThrow(diagnostic);
  }
});

it('accepts inclusive GPU percentile declaration boundaries', () => {
  expect(
    () =>
      new GPUNetworkTravelTimePercentilesOperation({
        ...PROPS,
        stateCount: 1,
        transitionCount: 0,
        vertexCount: 1,
        departureCount: 1,
        percentiles: [100],
        maximumRelaxations: 0
      })
  ).not.toThrow();
  expect(
    () =>
      new GPUNetworkTravelTimePercentilesOperation({
        ...PROPS,
        stateCount: 256,
        transitionCount: 4096,
        departureCount: 60,
        percentiles: new Array<number>(16).fill(100),
        maximumRelaxations: 65536,
        costModel: 'piecewise-arrival',
        samplesPerTransition: 256,
        waitingPolicy: 'wait-until-first-sample'
      })
  ).not.toThrow();
});

it.each([
  false,
  true
])('suppresses finite overestimates only when CPU percentile status is omitted (certificate=%s)', async certificate => {
  // Reverse transition order leaves state 2 at 10 after one sweep; its exact travel time is 2.
  const vector = (
    format: GPUCPUVector['format'],
    values: Uint32Array | Float32Array
  ): GPUCPUVector => {
    const length = values.length / (format.endsWith('x2') ? 2 : 1);
    return {format, length, chunks: [{values, length}], pack: () => values};
  };
  for (const maximumRelaxations of [1, 3]) {
    const operation = new CPUNetworkTravelTimePercentilesOperation({
      ...PROPS,
      stateCount: 3,
      transitionCount: 3,
      vertexCount: 3,
      departureCount: 1,
      percentiles: [100],
      maximumRelaxations,
      output: {
        travelTimes: 'times',
        reachedCounts: 'reached',
        ...(certificate ? {status: 'status'} : {})
      }
    });
    const result = await operation.execute({
      inputs: {
        stateVertices: vector('uint32', Uint32Array.of(0, 1, 2)),
        transitions: vector('uint32x2', Uint32Array.of(1, 2, 0, 1, 0, 2)),
        costs: vector('float32x2', Float32Array.of(1, 0, 1, 0, 10, 0)),
        initialState: vector('uint32', Uint32Array.of(0)),
        window: vector('float32', Float32Array.of(0, 1))
      }
    });
    expect(result['times']).toEqual(
      Float32Array.from(
        maximumRelaxations === 3 ? [0, 1, 2] : certificate ? [0, 1, 10] : [NaN, NaN, NaN]
      )
    );
    expect(result['reached']).toEqual(
      Uint32Array.from(maximumRelaxations === 1 && !certificate ? [0, 0, 0] : [1, 1, 1])
    );
    if (certificate)
      expect(result['status']).toEqual(Uint32Array.of(1, Number(maximumRelaxations === 3)));
  }
});

it('GPU percentile publication suppresses uncertified aggregates without a public certificate', () => {
  const fixture = createGPUFixture();
  try {
    const contributor = new GPUNetworkTravelTimePercentiles({
      ...fixture.props,
      output: {travelTimes: fixture.views.travelTimes, reachedCounts: fixture.views.reachedCounts}
    });
    const source = getGPUCommandNodeKernelSources(contributor.getCommandNodes(fixture.graph)).at(
      -1
    )!.source;
    expect(source).toContain('var complete = true;');
    expect(source).toContain('summaries[summariesOffset + departure *');
    expect(source).toContain('if (!complete)');
    expect(source).toContain('bitcast<f32>(0x7fc00000u | (index & 0u))');
  } finally {
    fixture.device.destroy();
  }
});

it.each([2, 3])('W5-04 estimates all declared routing stages for %i departures', departureCount => {
  const fixture = createGPUFixture();
  try {
    const contributor = new GPUNetworkTravelTimePercentiles({
      ...fixture.props,
      departureCount,
      output: {travelTimes: fixture.views.travelTimes}
    });
    // Preparation + elapsed solve and absolute publication per departure + percentiles.
    const expected = departureCount === 2 ? 6 : 8;
    expect(contributor.estimatedCommandNodeCount).toBe(expected);
    expect(contributor.getCommandNodes(fixture.graph)).toHaveLength(expected);
  } finally {
    fixture.device.destroy();
  }
});

it.each([
  3, 9, 60
])('B3 shares one elapsed and one absolute-publication pipeline across %i departures', departureCount => {
  const fixture = createGPUFixture();
  try {
    const contributor = new GPUNetworkTravelTimePercentiles({
      ...fixture.props,
      departureCount,
      output: {travelTimes: fixture.views.travelTimes}
    });
    const nodes = contributor.getCommandNodes(fixture.graph);
    expect(nodes).toHaveLength(2 + 2 * departureCount);
    for (const suffix of ['elapsed', 'publish-arrivals']) {
      const stage = nodes.filter(node => new RegExp(`-routing-\\d+-${suffix}$`).test(node.id));
      expect(stage).toHaveLength(departureCount);
      expect(getGPUCommandNodePipelineKeys(stage, {includeShaderLayout: true}).size).toBe(1);
    }
    // Preparation, elapsed solve, absolute publication and percentile publication.
    expect(getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}).size).toBe(4);
  } finally {
    fixture.device.destroy();
  }
});
