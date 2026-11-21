// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  type WGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPUSpatialPanelLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood';

// NumPy SVD / dense Gaussian information in panel-lag-pack.oracle.py, on original inputs.
const EXPECTED = {
  coefficients: [NaN, 0.0001003816807240593, 0.7665099072627592, 0],
  covariance: [
    NaN,
    NaN,
    NaN,
    NaN,
    1.138249149751649e-8,
    1.81206135840744e-8,
    NaN,
    1.81206135840744e-8,
    0.013339448408602577
  ],
  summary: [8.536850161543607e-8, 43.66466044666741],
  fitted: [
    0.0003674588388454959, 0.0006184130406556442, 0.0005180313599315849, 0.00041764967920752553,
    -0.0003674588388454959, -0.0006184130406556442, -0.0005180313599315849, -0.00041764967920752553
  ],
  residuals: [
    -0.00012331821384549592, -0.00013013179065564416, 0.0004585311400684151, 0.00031477219579247447,
    0.00012331821384549592, 0.00013013179065564416, -0.0004585311400684151, -0.00031477219579247447
  ]
};

export function expectPanelLagPackFit(
  result: Record<
    'coefficients' | 'covariance' | 'summary' | 'fitted' | 'residuals' | 'status',
    ArrayLike<number>
  >,
  scale: number
) {
  expect(Array.from(result.status)).toEqual([0]);
  expect(result.coefficients[0]).toBeNaN();
  for (const [slot, expected, tolerance] of [
    [1, EXPECTED.coefficients[1] * scale, 2e-9 * scale],
    [2, EXPECTED.coefficients[2], 1e-6],
    [3, 0, 0]
  ]) {
    expect(Number.isFinite(result.coefficients[slot])).toBe(true);
    expect(Math.abs(result.coefficients[slot] - expected)).toBeLessThanOrEqual(tolerance);
  }
  for (let row = 0; row < 3; row++)
    for (let column = 0; column < 3; column++) {
      const index = row * 3 + column;
      if (row === 0 || column === 0) expect(result.covariance[index]).toBeNaN();
      else {
        const factor = (row === 1 ? scale : 1) * (column === 1 ? scale : 1);
        const magnitude =
          Math.sqrt(EXPECTED.covariance[row * 3 + row] * EXPECTED.covariance[column * 3 + column]) *
          factor;
        expect(Number.isFinite(result.covariance[index])).toBe(true);
        expect(
          Math.abs(result.covariance[index] - EXPECTED.covariance[index] * factor)
        ).toBeLessThanOrEqual(1e-4 * magnitude);
      }
    }
  expect(Number.isFinite(result.summary[0])).toBe(true);
  expect(
    Math.abs(result.summary[0] / (EXPECTED.summary[0] * scale * scale) - 1)
  ).toBeLessThanOrEqual(2e-5);
  expect(Number.isFinite(result.summary[1])).toBe(true);
  expect(
    Math.abs(result.summary[1] - (EXPECTED.summary[1] - 8 * Math.log(scale)))
  ).toBeLessThanOrEqual(5e-5);
  expect(result.summary[2]).toBe(1);
  expect(result.summary[3]).toBe(0);
  expect(result.summary[4]).toBeGreaterThanOrEqual(0);
  expect(result.summary[4]).toBeLessThanOrEqual(5e-7);
  expect(result.summary[5]).toBe(0);
  for (const port of ['fitted', 'residuals'] as const) {
    expect(result[port].length).toBe(8);
    for (let row = 0; row < 8; row++) {
      expect(Number.isFinite(result[port][row])).toBe(true);
      expect(Math.abs(result[port][row] - EXPECTED[port][row] * scale)).toBeLessThanOrEqual(
        2e-8 * scale
      );
    }
  }
}

/** Hand oracle: y_it = effect_i +/- a_i*2^-12, a=[1,2,4,3], effects=+/-3000.
 * Each entity's two-period mean is its effect. On the four-cycle with weights .5,
 * every within lag is +/-2.5*2^-12. Averaging raw neighbors first loses the half step.
 */
export function createPanelLagPackInput(scale: number) {
  const steps = [1, 2, 4, 3, -1, -2, -4, -3];
  return {
    predictors: Float32Array.of(0, 1, 2, 3, 2, -2, 1, 4),
    response: Float32Array.from(
      steps,
      (step, row) => scale * ((row % 2 ? -3000 : 3000) + step * 2 ** -12) + 16
    ),
    weights: {
      offsets: Uint32Array.of(0, 2, 4, 6, 8),
      neighbors: Uint32Array.of(1, 3, 0, 2, 1, 3, 0, 2),
      weights: new Float32Array(8).fill(0.5)
    },
    withinResponse: steps.map(step => scale * step * 2 ** -12),
    withinLag: steps.map((_, row) => scale * (row < 4 ? 2.5 : -2.5) * 2 ** -12)
  };
}

/** Exact integer moment oracle on 4 entities and 8 periods. Each deviation column sums to zero
 * within each entity. The two large predictor squares sum to 2^25; the other thirty add 30.
 * A serial float32 walk drops those unit squares. Means are constant within each entity;
 * lag deviations and lag means use the same .5-weighted four-cycle as the full-fit fixture.
 */
export function createPanelLagMomentInput(effects: 'unit' | 'random') {
  const data = new Float32Array(32 * 8);
  for (let row = 0; row < 32; row++) {
    const entity = row % 4;
    const sign = row < 16 ? 1 : -1;
    const large = row === 0 || row === 16;
    const largePeriod = row < 4 || (row >= 16 && row < 20);
    data.set(
      [
        0,
        sign * (large ? 4096 : 1),
        large ? 0 : sign,
        sign * (largePeriod && entity % 2 ? 0.5 : 1),
        1,
        [8, -4, 2, -1][entity],
        [3, -2, 4, -8][entity],
        [-5, 3.5, -5, 3.5][entity]
      ],
      row * 8
    );
  }
  return {
    data,
    // Row-major moments: each predictor against all predictors, response, lag.
    deviation: effects === 'unit' ? [33554462, 30, 8220] : [0, 0, 0, 0, 0, 33554462, 30, 8220],
    cross: new Array(effects === 'unit' ? 3 : 8).fill(0),
    mean: effects === 'unit' ? [680, 384, -540] : [32, 40, -24, -24, 40, 680, 384, -540]
  };
}

/** Execute the actual contributor's moment section in isolation; only its outputs are exposed.
 * This retains the production reduction order instead of testing a duplicate implementation.
 */
export function createPanelLagMomentProbe(
  graph: GPUCommandGraph,
  record: WGSLKernelNodeRecord,
  output: GraphDataView,
  effects: 'unit' | 'random',
  data: GraphDataView
) {
  const count = effects === 'unit' ? 3 : 8;
  const start = record.body.indexOf('var deviationMoments:');
  const end = record.body.indexOf('var scale:array');
  expect(start).toBeGreaterThanOrEqual(0);
  expect(end).toBeGreaterThan(start);
  return createWGSLKernelNode(graph, {
    id: 'panel-moment-readback',
    operation: 'PanelMomentOracle',
    invocationCount: 1,
    bindings: [
      {name: 'data', view: data, type: 'f32', access: 'read'},
      {name: 'output', view: output, type: 'f32', access: 'read_write'}
    ],
    body: `${record.body.slice(start, end)}
for(var slot=0u;slot<${count}u;slot++){
 output[outputOffset+slot]=deviationMoments[slot];
 output[outputOffset+${count}u+slot]=crossMoments[slot];
 output[outputOffset+${count * 2}u+slot]=meanMoments[slot];
}`
  });
}

export function getPanelLagMomentRecord(
  graph: GPUCommandGraph,
  effects: 'unit' | 'random',
  tiled = false
) {
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `moment-${serial++}`, format, length);
  const nodes = new GPUSpatialPanelLagMaximumLikelihood({
    effects,
    ...(tiled ? {fitMode: 'tiled' as const, eigenvalues: view('float32', 4)} : {}),
    periodCount: 8,
    predictorCount: 1,
    predictors: view('float32', 32),
    response: view('float32', 32),
    weights: {
      offsets: view('uint32', 5),
      neighbors: view('uint32', 8),
      weights: view('float32', 8)
    },
    output: {
      coefficients: view('float32', 4),
      covariance: view('float32', 9),
      summary: view('float32', 6),
      status: view('uint32', 1),
      fitted: view('float32', 32),
      residuals: view('float32', 32)
    }
  }).getCommandNodes(graph);
  return getWGSLKernelNodeRecord(
    graph,
    nodes.find(node => node.id.endsWith(tiled ? '-moment-tiles' : '-fit'))!.id
  )!;
}

export function expectPanelLagMoments(
  actual: ArrayLike<number>,
  effects: 'unit' | 'random',
  tiled = false
) {
  const input = createPanelLagMomentInput(effects);
  const expected = tiled ? input.deviation : [...input.deviation, ...input.cross, ...input.mean];
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    expect(Number.isFinite(actual[index]), `finite moment ${index}`).toBe(true);
    // The large exact sum lies halfway between f32 values; a pairwise reduction may lose at
    // most two unit terms here. All other products and sums are exactly representable integers.
    expect(Math.abs(actual[index] - expected[index]), `moment ${index}`).toBeLessThanOrEqual(
      expected[index] === 33554462 ? 2 : 0
    );
  }
}
