// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeBrownianBridgeUtilizationOnCPU,
  CPUBrownianBridgeUtilizationOperation
} from '../../../src/gpu-spatial-analysis/trajectory-home-range/cpu-brownian-bridge-utilization-operation';
import {
  GPUBrownianBridgeUtilizationOperation,
  type BrownianBridgeUtilizationOperationProps
} from '../../../src/gpu-spatial-analysis/trajectory-home-range/gpu-brownian-bridge-utilization-operation';
import {getGPUBrownianBridgeUtilizationParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-home-range/gpu-brownian-bridge-utilization';
import {estimateWindowedBrownianMotionVariance} from '../../../src/gpu-spatial-analysis/trajectory-home-range/brownian-bridge-variance-estimation';

it('pads each track final row when handing windowed estimates to utilization', () => {
  const track = {x: [0, 1, 0], y: [0, 0, 0], times: [0, 1, 2]};
  const estimates = estimateWindowedBrownianMotionVariance(track, {locationError: 1});
  const motionVariances = new Float32Array(6);
  motionVariances.set(estimates, 0);
  motionVariances.set(estimates, 3);
  expect(motionVariances[2]).toBe(0);
  expect(motionVariances[5]).toBe(0);
  const result = computeBrownianBridgeUtilizationOnCPU(
    {
      positions: Float32Array.of(0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0),
      timestamps: Float32Array.of(0, 1, 2, 10, 11, 12),
      trackOffsets: Uint32Array.of(0, 3, 6),
      motionVariances,
      locationErrors: new Float32Array(6).fill(1),
      parameters: Float32Array.of(0, -0.5, 1, 0.5)
    },
    {gridSize: [1, 1], subStepCount: 1}
  );
  expect(result.density[0]).toBe(Math.fround(1 / Math.PI));
});

function getInput() {
  return {
    positions: new Float32Array(8),
    timestamps: new Float32Array([0, 2, 10, 11]),
    trackOffsets: new Uint32Array([0, 2, 2, 4]),
    motionVariances: new Float32Array([4, 999, 8, 999]),
    locationErrors: new Float32Array([2, 2, 2, 2]),
    parameters: getGPUBrownianBridgeUtilizationParameterValues({extent: [-3, -3, 3, 3]})
  };
}

it('clamps terminal offsets and evaluates a stationary unit-variance Gaussian', () => {
  const result = computeBrownianBridgeUtilizationOnCPU(
    {
      positions: new Float32Array(4),
      timestamps: new Float32Array([0, 1]),
      trackOffsets: new Uint32Array([0, 3]),
      motionVariances: new Float32Array([2, 2]),
      locationErrors: new Float32Array([1, 1]),
      parameters: new Float32Array([-1, -1, 1, 1])
    },
    {gridSize: [1, 1]}
  );
  // T*sigma²=2*delta²=2, so the variance is exactly one at every midpoint.
  expect(Array.from(result.density)).toEqual([Math.fround(1 / (2 * Math.PI))]);
});

it('accepts fewer than two fixes in the pure function and retains the graph minimum of two', () => {
  for (const rowCount of [0, 1]) {
    const result = computeBrownianBridgeUtilizationOnCPU(
      {
        positions: new Float32Array(rowCount * 2),
        timestamps: new Float32Array(rowCount),
        trackOffsets: new Uint32Array([0, rowCount]),
        motionVariances: new Float32Array(rowCount),
        locationErrors: new Float32Array(rowCount),
        parameters: new Float32Array([-1, -1, 1, 1])
      },
      {gridSize: [1, 1]}
    );
    expect(Array.from(result.density)).toEqual([0]);
    expect(
      () =>
        new CPUBrownianBridgeUtilizationOperation({
          positions: 'positions',
          timestamps: 'timestamps',
          trackOffsets: 'trackOffsets',
          motionVariances: 'motionVariances',
          locationErrors: 'locationErrors',
          parameters: 'parameters',
          rowCount,
          gridSize: [1, 1],
          output: {density: 'density'}
        })
    ).toThrow(/rowCount/);
  }
});

it('treats a NaN motion variance as an absent Gaussian contribution', () => {
  const result = computeBrownianBridgeUtilizationOnCPU(
    {
      positions: new Float32Array(4),
      timestamps: new Float32Array([0, 1]),
      trackOffsets: new Uint32Array([0, 2]),
      motionVariances: new Float32Array([NaN, 0]),
      locationErrors: new Float32Array([1, 1]),
      parameters: new Float32Array([-1, -1, 1, 1])
    },
    {gridSize: [1, 1], subStepCount: 1}
  );
  // The sole midpoint variance is NaN; variance > 0 is false, leaving zero density.
  expect(Array.from(result.density)).toEqual([0]);
});

it('reduces exactly to a stationary bivariate normal for every midpoint budget', () => {
  // On each segment T*sigma²=2*delta², so every midpoint variance is delta²=4.
  // The whole time mixture is therefore N(0,4I), regardless of durations or step count.
  for (const subStepCount of [1, 15, 257]) {
    const {density} = computeBrownianBridgeUtilizationOnCPU(getInput(), {
      gridSize: [3, 3],
      subStepCount
    });
    const expected = [
      Math.exp(-1),
      Math.exp(-0.5),
      Math.exp(-1),
      Math.exp(-0.5),
      1,
      Math.exp(-0.5),
      Math.exp(-1),
      Math.exp(-0.5),
      Math.exp(-1)
    ].map(value => Math.fround(value / (8 * Math.PI)));
    expect(Array.from(density)).toEqual(expected);
  }
});

it('keeps nonzero Gaussian tails beyond the GPU cutoff and preserves normalization across zero variance', () => {
  const input = getInput();
  const tail = computeBrownianBridgeUtilizationOnCPU(
    {
      ...input,
      parameters: getGPUBrownianBridgeUtilizationParameterValues({extent: [19, -1, 21, 1]})
    },
    {gridSize: [1, 1]}
  );
  expect(tail.density[0]).toBe(Math.fround(Math.exp(-50) / (8 * Math.PI)));
  expect(tail.density[0]).toBeGreaterThan(0);
  // The second segment has one time unit of zero variance, included in the three-unit denominator.
  const diluted = computeBrownianBridgeUtilizationOnCPU(
    {
      ...input,
      motionVariances: new Float32Array([4, 0, 0, 0]),
      locationErrors: new Float32Array([2, 2, 0, 0]),
      parameters: getGPUBrownianBridgeUtilizationParameterValues({extent: [-1, -1, 1, 1]})
    },
    {gridSize: [1, 1]}
  );
  expect(diluted.density[0]).toBe(Math.fround(1 / (12 * Math.PI)));
  expect(
    computeBrownianBridgeUtilizationOnCPU(
      {...input, timestamps: new Float32Array([0, 0, 1, 0])},
      {gridSize: [1, 1]}
    ).density[0]
  ).toBe(0);
});

it('validates packed arrays and operation construction', () => {
  const input = getInput();
  for (const argument of [
    'motionVariances',
    'locationErrors',
    'parameters',
    'timestamps',
    'positions'
  ] as const) {
    expect(() =>
      computeBrownianBridgeUtilizationOnCPU(
        {...input, [argument]: new Float32Array(1)},
        {gridSize: [1, 1]}
      )
    ).toThrow(new RegExp(argument));
  }
  const props: BrownianBridgeUtilizationOperationProps = {
    id: 'bridge-test',
    positions: 'positions',
    timestamps: 'timestamps',
    trackOffsets: 'trackOffsets',
    motionVariances: 'motionVariances',
    locationErrors: 'locationErrors',
    parameters: 'parameters',
    rowCount: 4,
    gridSize: [1, 1],
    output: {density: 'density'}
  };
  for (const Operation of [
    GPUBrownianBridgeUtilizationOperation,
    CPUBrownianBridgeUtilizationOperation
  ]) {
    expect(() => new Operation({...props, gridSize: [0, 1]})).toThrow(/^bridge-test .*length/);
    expect(() => new Operation({...props, output: {density: 'positions'}})).toThrow(
      /^bridge-test /
    );
  }
  expect(() => new CPUBrownianBridgeUtilizationOperation({...props, subStepCount: 0})).toThrow(
    /^bridge-test subStepCount/
  );
  expect(
    () => new CPUBrownianBridgeUtilizationOperation({...props, subStepCount: 257, rowCount: 20000})
  ).not.toThrow();
  expect(
    () =>
      new CPUBrownianBridgeUtilizationOperation({
        ...props,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
      })
  ).toThrow(/^bridge-test /);
});
