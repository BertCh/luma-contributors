// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUTrajectoryKalman} from '../../../src/gpu-spatial-analysis/trajectory-estimation/gpu-trajectory-kalman';
import {RecipeTestFixture} from '../recipes/recipe-harness';

it.each([
  'isotropic',
  'full'
] as const)('R1 metadata %s isolated RTS uses nonzero copied process noise', async covarianceMode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `metadata-noise-${covarianceMode}`);
  try {
    const identity =
      covarianceMode === 'full' ? [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] : [1, 0, 0, 1];
    const status = fixture.output('status', 'uint32', 2);
    status.buffer.write(Uint32Array.of(1, 0));
    const smoothed = fixture.output('smoothed', 'float32x4', 2);
    const covariance = fixture.output(
      'smoothed-covariance',
      'float32x4',
      covarianceMode === 'full' ? 8 : 2
    );
    const contributor = new GPUTrajectoryKalman({
      covarianceMode,
      positions: fixture.input('positions', new Float32Array(4), 'float32x2', 2),
      times: fixture.input('times', Float32Array.of(0, 1), 'float32', 2),
      offsets: fixture.input('offsets', Uint32Array.of(0, 2), 'uint32', 2),
      parameters: fixture.input(
        'parameters',
        covarianceMode === 'full'
          ? Float32Array.of(1, 0, 1, 1, 0, 1, 1, 0, 1, 0)
          : Float32Array.of(1, 1, 1, 0),
        'float32',
        covarianceMode === 'full' ? 10 : 4
      ),
      output: {
        filteredState: fixture.input(
          'state',
          Float32Array.of(0, 0, 0, 0, 2, 0, 0, 0),
          'float32x4',
          2
        ),
        filteredCovariance: fixture.input(
          'covariance',
          Float32Array.from([...identity, ...identity]),
          'float32x4',
          covarianceMode === 'full' ? 8 : 2
        ),
        status: status.view,
        smoothedState: smoothed.view,
        smoothedCovariance: covariance.view
      }
    });
    const nodes = contributor.getCommandNodes(fixture.graph);
    for (const suffix of ['-initialize-local-metadata', '-smooth', '-publish-absolute'])
      fixture.graph.add(nodes.find(node => node.id.endsWith(suffix))!);
    fixture.run();
    // P=I, dt=1, q=1 => Ppred=[[7/3,3/2],[3/2,2]],
    // G=[[24,-18],[6,10]]/29; G*[2,0]=[48,12]/29.
    const values = await fixture.readFloat32(smoothed, 8);
    for (const value of values) expect(Number.isFinite(value)).toBe(true);
    expect(values[0]).toBeCloseTo(48 / 29, 5);
    expect(values[2]).toBeCloseTo(12 / 29, 5);
    expect(await fixture.readUint32(status, 2)).toEqual([1, 0]);
  } finally {
    fixture.destroy();
  }
});

it.each([
  'isotropic',
  'full'
] as const)('B3 %s preserves small measured positions after a distant first fix', async covarianceMode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `distant-first-${covarianceMode}`);
  try {
    const count = 10;
    const filtered = fixture.output('filtered', 'float32x4', count);
    const smoothed = fixture.output('smoothed', 'float32x4', count);
    fixture.graph.add(
      new GPUTrajectoryKalman({
        covarianceMode,
        positions: fixture.input(
          'positions',
          Float32Array.from(
            Array.from({length: count}, (_, row) => [row === 0 ? 500000 : 0.01, 0]).flat()
          ),
          'float32x2',
          count
        ),
        times: fixture.input(
          'times',
          Float32Array.from({length: count}, (_, row) => row * 3600),
          'float32',
          count
        ),
        offsets: fixture.input('offsets', Uint32Array.of(0, count), 'uint32', 2),
        parameters: fixture.input(
          'parameters',
          covarianceMode === 'full'
            ? Float32Array.of(0.01, 0, 0.01, 0.0001, 0, 0.0001, 1, 0, 1, 0)
            : Float32Array.of(0.01, 0.0001, 1, 0),
          'float32',
          covarianceMode === 'full' ? 10 : 4
        ),
        output: {
          filteredState: filtered.view,
          filteredCovariance: fixture.output(
            'covariance',
            'float32x4',
            covarianceMode === 'full' ? 40 : 10
          ).view,
          smoothedState: smoothed.view,
          smoothedCovariance: fixture.output(
            'smoothed-covariance',
            'float32x4',
            covarianceMode === 'full' ? 40 : 10
          ).view,
          status: fixture.output('status', 'uint32', count).view
        }
      })
    );
    fixture.run();
    // Nine repeated x=.01 measurements, r=1e-4, hour-long process increments dominate r:
    // the independent binary64 reference rounds the final estimate to f32(.01).
    for (const output of [filtered, smoothed]) {
      const values = await fixture.readFloat32(output, count * 4);
      expect(Number.isFinite(values[36])).toBe(true);
      expect(Math.abs(values[36] - 0.009999999776482582)).toBeLessThan(1e-6);
    }
  } finally {
    fixture.destroy();
  }
});

it.each([
  'isotropic',
  'full'
] as const)('FLT-09 %s predicts in a local frame through a nine-row gap', async covarianceMode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const origin of [0, 10_000_000]) {
    const fixture = new RecipeTestFixture(device, `kalman-local-${covarianceMode}-${origin}`);
    try {
      // K=(2/3,1/3) at row 1, so x(8)=2/3+7/3=3 and vx=1/3.
      // A second segment resets at a new fix; smoothing must never cross that boundary.
      const positions = Float32Array.from([
        origin,
        0,
        origin + 1,
        0,
        ...Array(14).fill(NaN),
        origin + 10,
        0,
        origin + 11,
        0
      ]);
      const count = 11;
      const covarianceRows = covarianceMode === 'full' ? count * 4 : count;
      const filtered = fixture.output('filtered', 'float32x4', count);
      const smoothed = fixture.output('smoothed', 'float32x4', count);
      const filteredCovariance = fixture.output('covariance', 'float32x4', covarianceRows);
      const smoothedCovariance = fixture.output('smoothed-covariance', 'float32x4', covarianceRows);
      const status = fixture.output('status', 'uint32', count);
      fixture.graph.add(
        new GPUTrajectoryKalman({
          covarianceMode,
          positions: fixture.input('positions', positions, 'float32x2', count),
          times: fixture.input(
            'times',
            Float32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 20, 21),
            'float32',
            count
          ),
          offsets: fixture.input('offsets', Uint32Array.of(0, count), 'uint32', 2),
          parameters: fixture.input(
            'parameters',
            covarianceMode === 'full'
              ? Float32Array.of(0, 0, 0, 1, 0, 1, 1, 0, 1, 2)
              : Float32Array.of(0, 1, 1, 2),
            'float32',
            covarianceMode === 'full' ? 10 : 4
          ),
          output: {
            filteredState: filtered.view,
            filteredCovariance: filteredCovariance.view,
            smoothedState: smoothed.view,
            smoothedCovariance: smoothedCovariance.view,
            status: status.view
          }
        })
      );
      fixture.run();
      for (const output of [filtered, smoothed]) {
        const values = await fixture.readFloat32(output, count * 4);
        for (const value of values) expect(Number.isFinite(value)).toBe(true);
        expect(values[32] - origin).toBeCloseTo(3, 5);
        expect(values[34]).toBeCloseTo(1 / 3, 6);
        // The first smoothed position is 1/3 in local coordinates. Absolute publication rounds once.
        expect(values[0]).toBeCloseTo(
          output === filtered ? origin : Math.fround(origin + 1 / 3),
          5
        );
      }
      expect(await fixture.readUint32(status, count)).toEqual([1, 0, 2, 2, 2, 2, 2, 2, 2, 1, 0]);
      // P1=[[2/3,1/3],[1/3,2/3]]. F(7)P1F(7)^T=[[38,5],[5,2/3]].
      const expectedCovariance =
        covarianceMode === 'full'
          ? [38, 0, 5, 0, 0, 38, 0, 5, 5, 0, 2 / 3, 0, 0, 5, 0, 2 / 3]
          : [38, 5, 5, 2 / 3];
      for (const output of [filteredCovariance, smoothedCovariance]) {
        const width = expectedCovariance.length;
        const values = await fixture.readFloat32(output, count * width);
        for (const [component, expected] of expectedCovariance.entries()) {
          expect(Number.isFinite(values[8 * width + component])).toBe(true);
          expect(values[8 * width + component]).toBeCloseTo(expected, 4);
        }
      }
    } finally {
      fixture.destroy();
    }
  }
});

it.each([
  'isotropic',
  'full'
] as const)('TRJ-107 %s smoother rejects a post-inversion overflow atomically', async covarianceMode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `smooth-finite-${covarianceMode}`);
  try {
    // Isolate the backward kernel with finite filtered rows. At delta=0, covariance=I,
    // noise=0, prediction and gain are I (determinant 1). Row 1's residual
    // 3e38 - (-3e38) overflows f32 after a successful inverse. It must keep its
    // filtered value; row 0 then consumes -3e38, not the rejected correction.
    // A second track is the finite control: gain I gives state 2 and covariance I.
    const states = Float32Array.of(
      0,
      0,
      0,
      0,
      -3e38,
      0,
      0,
      0,
      3e38,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      2,
      0,
      0,
      0
    );
    const identity =
      covarianceMode === 'full' ? [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] : [1, 0, 0, 1];
    const covariance = Float32Array.from(Array.from({length: 5}, () => identity).flat());
    const status = fixture.output('status', 'uint32', 5);
    status.buffer.write(Uint32Array.of(1, 2, 0, 1, 0));
    const smoothedState = fixture.output('smoothed', 'float32x4', 5);
    const smoothedCovariance = fixture.output(
      'smoothed-covariance',
      'float32x4',
      covariance.length / 4
    );
    const contributor = new GPUTrajectoryKalman({
      covarianceMode,
      positions: fixture.input('positions', new Float32Array(10), 'float32x2', 5),
      times: fixture.input('times', new Float32Array(5), 'float32', 5),
      offsets: fixture.input('offsets', Uint32Array.of(0, 3, 5), 'uint32', 3),
      parameters: fixture.input(
        'parameters',
        covarianceMode === 'full'
          ? Float32Array.of(0, 0, 0, 1, 0, 1, 1, 0, 1, 0)
          : Float32Array.of(0, 1, 1, 0),
        'float32',
        covarianceMode === 'full' ? 10 : 4
      ),
      output: {
        filteredState: fixture.input('filtered', states, 'float32x4', 5),
        filteredCovariance: fixture.input(
          'covariance',
          covariance,
          'float32x4',
          covariance.length / 4
        ),
        status: status.view,
        smoothedState: smoothedState.view,
        smoothedCovariance: smoothedCovariance.view
      }
    });
    const nodes = contributor.getCommandNodes(fixture.graph);
    for (const suffix of ['-initialize-local-metadata', '-smooth', '-publish-absolute'])
      fixture.graph.add(nodes.find(node => node.id.endsWith(suffix))!);
    fixture.run();
    expect(await fixture.readUint32(status, 5)).toEqual([1, 10, 0, 1, 0]);
    expect(await fixture.readFloat32(smoothedState, 20)).toEqual(
      Array.from(
        Float32Array.of(-3e38, 0, 0, 0, -3e38, 0, 0, 0, 3e38, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0)
      )
    );
    expect(await fixture.readFloat32(smoothedCovariance, covariance.length)).toEqual(
      Array.from(covariance)
    );
  } finally {
    fixture.destroy();
  }
});

it.each([
  'isotropic',
  'full'
] as const)('B4 %s smoother rejects covariance-only overflow before publishing either output', async covarianceMode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `covariance-only-${covarianceMode}`);
  try {
    // F=[[1,2],[0,1]], P=I, Q=0 => predicted=[[5,2],[2,1]], det=1,
    // gain=[[1,-2],[0,1]]. With next P=diag(1,1e38), corrected position variance
    // includes 4e38 and overflows f32. State correction is finite: [0,0,0,0] -> [1,0,0,0].
    // Rejecting covariance must therefore withhold that otherwise finite state correction too.
    const identity =
      covarianceMode === 'full' ? [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] : [1, 0, 0, 1];
    const next =
      covarianceMode === 'full'
        ? [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1e38, 0, 0, 0, 0, 1e38]
        : [1, 0, 0, 1e38];
    const covariance = Float32Array.from([...identity, ...next]);
    const status = fixture.output('status', 'uint32', 2);
    status.buffer.write(Uint32Array.of(1, 0));
    const state = fixture.output('smoothed', 'float32x4', 2);
    const uncertainty = fixture.output('smoothed-covariance', 'float32x4', covariance.length / 4);
    const contributor = new GPUTrajectoryKalman({
      covarianceMode,
      positions: fixture.input('positions', new Float32Array(4), 'float32x2', 2),
      times: fixture.input('times', Float32Array.of(0, 2), 'float32', 2),
      offsets: fixture.input('offsets', Uint32Array.of(0, 2), 'uint32', 2),
      parameters: fixture.input(
        'parameters',
        covarianceMode === 'full'
          ? Float32Array.of(0, 0, 0, 1, 0, 1, 1, 0, 1, 0)
          : Float32Array.of(0, 1, 1, 0),
        'float32',
        covarianceMode === 'full' ? 10 : 4
      ),
      output: {
        filteredState: fixture.input(
          'filtered',
          Float32Array.of(0, 0, 0, 0, 1, 0, 0, 0),
          'float32x4',
          2
        ),
        filteredCovariance: fixture.input(
          'covariance',
          covariance,
          'float32x4',
          covariance.length / 4
        ),
        status: status.view,
        smoothedState: state.view,
        smoothedCovariance: uncertainty.view
      }
    });
    const nodes = contributor.getCommandNodes(fixture.graph);
    for (const suffix of ['-initialize-local-metadata', '-smooth', '-publish-absolute'])
      fixture.graph.add(nodes.find(node => node.id.endsWith(suffix))!);
    fixture.run();
    expect(await fixture.readUint32(status, 2)).toEqual([9, 0]);
    expect(await fixture.readFloat32(state, 8)).toEqual([0, 0, 0, 0, 1, 0, 0, 0]);
    expect(await fixture.readFloat32(uncertainty, covariance.length)).toEqual(
      Array.from(covariance)
    );
  } finally {
    fixture.destroy();
  }
});

it.each([
  'isotropic',
  'full'
] as const)('TRJ-107 %s normal entry point falls back after a finite forward filter', async covarianceMode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `kalman-end-to-end-${covarianceMode}`);
  try {
    const positions = Float32Array.from(
      [-3, -3, 0, 1.25, 1.8, 2.25, 2.75, 2.75, 2.75, 2.75].flatMap(value => [value * 1e38, 0])
    );
    const filtered = fixture.output('filtered', 'float32x4', 10);
    const filteredCovariance = fixture.output(
      'filtered-covariance',
      'float32x4',
      covarianceMode === 'full' ? 40 : 10
    );
    const smoothed = fixture.output('smoothed', 'float32x4', 10);
    const smoothedCovariance = fixture.output(
      'smoothed-covariance',
      'float32x4',
      covarianceMode === 'full' ? 40 : 10
    );
    const status = fixture.output('status', 'uint32', 10);
    fixture.graph.add(
      new GPUTrajectoryKalman({
        covarianceMode,
        positions: fixture.input('positions', positions, 'float32x2', 10),
        times: fixture.input('times', new Float32Array(10), 'float32', 10),
        offsets: fixture.input('offsets', Uint32Array.of(0, 10), 'uint32', 2),
        parameters: fixture.input(
          'parameters',
          covarianceMode === 'full'
            ? Float32Array.of(0, 0, 0, 1, 0, 1, 1, 0, 1, 0)
            : Float32Array.of(0, 1, 1, 0),
          'float32',
          covarianceMode === 'full' ? 10 : 4
        ),
        output: {
          filteredState: filtered.view,
          filteredCovariance: filteredCovariance.view,
          smoothedState: smoothed.view,
          smoothedCovariance: smoothedCovariance.view,
          status: status.view
        }
      })
    );
    fixture.run();
    const forward = await fixture.readFloat32(filtered, 40);
    const backward = await fixture.readFloat32(smoothed, 40);
    const covarianceWidth = covarianceMode === 'full' ? 16 : 4;
    const forwardCovariance = await fixture.readFloat32(filteredCovariance, 10 * covarianceWidth);
    const backwardCovariance = await fixture.readFloat32(smoothedCovariance, 10 * covarianceWidth);
    for (const value of [...forward, ...backward, ...forwardCovariance, ...backwardCovariance])
      expect(Number.isFinite(value)).toBe(true);
    expect(await fixture.readUint32(status, 10)).toEqual([1, 8, 0, 0, 0, 0, 0, 0, 0, 0]);
    expect(backward.slice(0, 8)).toEqual(
      Array.from(Float32Array.of(-3e38, 0, 0, 0, -3e38, 0, 0, 0))
    );
    expect(backwardCovariance.slice(covarianceWidth, covarianceWidth * 2)).toEqual(
      forwardCovariance.slice(covarianceWidth, covarianceWidth * 2)
    );
    // Simultaneous independent unit-variance observations estimate one constant position.
    // The ten x coefficients sum to 10.3, hence the full-data mean is 1.03e38.
    for (let row = 2; row < 10; row++)
      expect(Math.abs(backward[row * 4] / 1.03e38 - 1)).toBeLessThan(2e-6);
  } finally {
    fixture.destroy();
  }
});
