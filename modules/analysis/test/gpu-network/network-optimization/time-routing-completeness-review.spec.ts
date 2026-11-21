// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkTimeRouting} from '../../../src/gpu-network/network-optimization/gpu-network-time-routing';
import {GPUNetworkTravelTimePercentiles} from '../../../src/gpu-network/network-time-geography/gpu-network-travel-time-percentiles';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {RecipeTestFixture} from '../../gpu-spatial-analysis/recipes/recipe-harness';

it('piecewise identity self-loop never improves its departure (B3)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'piecewise-self-loop-review');
  try {
    const arrivals = fixture.output('arrivals', 'float32', 2);
    const elapsed = fixture.output('elapsed', 'float32', 2);
    const predecessors = fixture.output('predecessors', 'uint32', 2);
    const summary = fixture.output('summary', 'uint32', 4);
    fixture.graph.add(
      new GPUNetworkTimeRouting({
        stateVertices: fixture.input('vertices', Uint32Array.of(0, 1), 'uint32', 2),
        transitions: fixture.input('transitions', Uint32Array.of(0, 0, 0, 1), 'uint32x2', 2),
        costs: fixture.input('costs', Float32Array.of(0, 0, 41, 41, 0, 0, 41, 41), 'float32x2', 4),
        endpoints: fixture.input('endpoints', Uint32Array.of(0, 1), 'uint32', 2),
        departure: fixture.input('departure', Float32Array.of(1), 'float32', 1),
        costModel: 'piecewise-arrival',
        samplesPerTransition: 2,
        output: {
          arrivals: arrivals.view,
          elapsedTimes: elapsed.view,
          predecessorTransitions: predecessors.view,
          summary: summary.view
        }
      })
    );
    fixture.run();
    // A(t)=t cannot improve its self-loop, but the identity edge must reach state 1 at zero duration.
    expect(await fixture.readFloat32(elapsed, 2)).toEqual([0, 0]);
    expect(await fixture.readFloat32(arrivals, 2)).toEqual([1, 1]);
    expect(await fixture.readUint32(predecessors, 2)).toEqual([0xffffffff, 1]);
    expect(await fixture.readUint32(summary, 4)).toEqual([1, 1, 1, 0x3f800000]);
  } finally {
    fixture.destroy();
  }
});

it.each([
  false,
  true
])('percentiles suppress uncertified finite overestimates and retain certified incumbents (status=%s)', async certificate => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const maximumRelaxations of [1, 3]) {
    const fixture = new RecipeTestFixture(
      device,
      `percentile-completeness-${certificate}-${maximumRelaxations}`
    );
    try {
      const times = fixture.output('times', 'float32', 3);
      const reached = fixture.output('reached', 'uint32', 3);
      const status = certificate ? fixture.output('status', 'uint32', 2) : undefined;
      fixture.graph.add(
        new GPUNetworkTravelTimePercentiles({
          stateVertices: fixture.input('vertices', Uint32Array.of(0, 1, 2), 'uint32', 3),
          // First sweep leaves arrival 10 at vertex 2; the two-hop optimum has duration 2.
          transitions: fixture.input(
            'transitions',
            Uint32Array.of(1, 2, 0, 1, 0, 2),
            'uint32x2',
            3
          ),
          costs: fixture.input('costs', Float32Array.of(1, 0, 1, 0, 10, 0), 'float32x2', 3),
          initialState: fixture.input('initial', Uint32Array.of(0), 'uint32', 1),
          window: fixture.input('window', Float32Array.of(0, 1), 'float32', 2),
          vertexCount: 3,
          departureCount: 1,
          percentiles: [100],
          maximumRelaxations,
          output: {travelTimes: times.view, reachedCounts: reached.view, status: status?.view}
        })
      );
      fixture.run();
      expect(await fixture.readFloat32(times, 3)).toEqual(
        maximumRelaxations === 3 ? [0, 1, 2] : certificate ? [0, 1, 10] : [NaN, NaN, NaN]
      );
      expect(await fixture.readUint32(reached, 3)).toEqual(
        maximumRelaxations === 1 && !certificate ? [0, 0, 0] : [1, 1, 1]
      );
      if (status)
        expect(await fixture.readUint32(status, 2)).toEqual([1, Number(maximumRelaxations === 3)]);
    } finally {
      fixture.destroy();
    }
  }
});

it('FLT-12 preserves elapsed costs and route ordering on a large departure clock', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const direct of [false, true]) {
    const fixture = new RecipeTestFixture(device, `large-clock-${direct}`);
    try {
      const times = fixture.output('times', 'float32', 3);
      const reached = fixture.output('reached', 'uint32', 3);
      const status = fixture.output('status', 'uint32', 2);
      fixture.graph.add(
        new GPUNetworkTravelTimePercentiles({
          stateVertices: fixture.input('vertices', Uint32Array.of(0, 1, 2), 'uint32', 3),
          transitions: fixture.input(
            'transitions',
            direct ? Uint32Array.of(0, 1, 1, 2, 0, 2) : Uint32Array.of(0, 1, 1, 2),
            'uint32x2',
            direct ? 3 : 2
          ),
          costs: fixture.input(
            'costs',
            direct ? Float32Array.of(1, 0, 1, 0, 1.5, 0) : Float32Array.of(1, 0, 1, 0),
            'float32x2',
            direct ? 3 : 2
          ),
          initialState: fixture.input('initial', Uint32Array.of(0), 'uint32', 1),
          window: fixture.input('window', Float32Array.of(16_777_216, 0), 'float32', 2),
          vertexCount: 3,
          departureCount: 1,
          percentiles: [50],
          output: {travelTimes: times.view, reachedCounts: reached.view, status: status.view}
        })
      );
      fixture.run();
      // Two unit links cost 2; adding the 1.5 direct link changes the optimum to 1.5.
      expect(await fixture.readFloat32(times, 3)).toEqual([0, 1, direct ? 1.5 : 2]);
      expect(await fixture.readUint32(reached, 3)).toEqual([1, 1, 1]);
      expect(await fixture.readUint32(status, 2)).toEqual([1, 1]);
    } finally {
      fixture.destroy();
    }
  }
});

it('review B1 compares elapsed routes and publishes the direct-edge predecessor at 2**25', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'review-large-clock-predecessor');
  try {
    const stateVertices = fixture.input('vertices', Uint32Array.of(0, 1, 2), 'uint32', 3);
    const transitions = fixture.input(
      'transitions',
      Uint32Array.of(0, 1, 1, 2, 0, 2),
      'uint32x2',
      3
    );
    const costs = fixture.input('costs', Float32Array.of(1.5, 0, 1.5, 0, 2.5, 0), 'float32x2', 3);
    const arrivals = fixture.output('arrivals', 'float32', 3);
    const elapsed = fixture.output('elapsed', 'float32', 3);
    const predecessors = fixture.output('predecessors', 'uint32', 3);
    const summary = fixture.output('summary', 'uint32', 4);
    fixture.graph.add(
      new GPUNetworkTimeRouting({
        stateVertices,
        transitions,
        costs,
        endpoints: fixture.input('endpoints', Uint32Array.of(0, 2), 'uint32', 2),
        departure: fixture.input('departure', Float32Array.of(2 ** 25), 'float32', 1),
        output: {
          arrivals: arrivals.view,
          elapsedTimes: elapsed.view,
          predecessorTransitions: predecessors.view,
          summary: summary.view
        }
      })
    );
    const times = fixture.output('times', 'float32', 3);
    const reached = fixture.output('reached', 'uint32', 3);
    const status = fixture.output('status', 'uint32', 2);
    fixture.graph.add(
      new GPUNetworkTravelTimePercentiles({
        stateVertices,
        transitions,
        costs,
        initialState: fixture.input('initial', Uint32Array.of(0), 'uint32', 1),
        window: fixture.input('window', Float32Array.of(2 ** 25, 0), 'float32', 2),
        vertexCount: 3,
        departureCount: 1,
        percentiles: [50],
        output: {travelTimes: times.view, reachedCounts: reached.view, status: status.view}
      })
    );
    fixture.run();
    expect(await fixture.readFloat32(elapsed, 3)).toEqual([0, 1.5, 2.5]);
    expect(await fixture.readUint32(predecessors, 3)).toEqual([0xffffffff, 0, 2]);
    // Absolute clock values retain their public float32 representation (spacing four here).
    expect(await fixture.readFloat32(arrivals, 3)).toEqual([33554432, 33554432, 33554436]);
    expect((await fixture.readUint32(summary, 4)).slice(0, 3)).toEqual([1, 1, 2]);
    expect(await fixture.readFloat32(times, 3)).toEqual([0, 1.5, 2.5]);
    expect(await fixture.readUint32(reached, 3)).toEqual([1, 1, 1]);
    expect(await fixture.readUint32(status, 2)).toEqual([1, 1]);
  } finally {
    fixture.destroy();
  }
});
