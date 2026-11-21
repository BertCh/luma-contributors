// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {fp64arithmetic} from '@luma.gl/shaders';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  IS_FINITE_BITS_WGSL,
  IS_FINITE_VALUE_WGSL
} from '../../../src/analysis-core/kernels/wgsl-common';
import {getCholeskyWGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-regression-solve';
import {getSURGLSRefinementWGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/sur-gls-refinement-wgsl';
import {expect, it} from 'vitest';
import {GPUSeeminglyUnrelatedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {createRegressionTestContext} from './math-review-regression-test-utils';

// Hand-worked target: x-origin=[-2,-2,2,2,0], y=1+1.5*(x-origin)+u.
// Both residual vectors are orthogonal to [1,x-origin] and each other; Sigma=diag(.05,.05).
// This nonnegative zero-diagonal, row-standardized W was constructed independently by solving
// u'Wu=0 and (Wu)'Wu=tr(W'W)*u'u/5 for BOTH equations. Those are the KP moment equations at
// lambda=0. Rounding W to the uploaded float32 leaves moment errors < 1e-8.
// Thus lambda=0, beta=(1,1.5),(-2,.5) and structural residuals u are identified, rather than
// relying on an island graph (spatial SUR rejects islands) or a singular joint covariance.
const WEIGHTS = [
  0, 0.7302125096321106, 0.02981087751686573, 0.2399766445159912, 0, 0.0785946249961853,
  0.6892567276954651, 0.23214863240718842, 0.016570154577493668, 0.6715169548988342, 0,
  0.31191286444664, 0.6941668391227722, 0.03804992139339447, 0, 0.2677832543849945,
  0.7206304669380188, 0.009243836626410484, 0.13506284356117249, 0.13506284356117249
];
const TARGET_RESIDUALS = [0.25, -0.25, 0.25, -0.25, 0];
const SECOND_RESIDUALS = [0.25, -0.25, -0.25, 0.25, 0];

for (const origin of [0, 2 ** 24]) {
  it(`spatial SUR preserves structural residuals at predictor origin ${origin}`, async () => {
    const context = await createRegressionTestContext();
    try {
      const centered = [-2, -2, 2, 2, 0];
      const status = context.output('uint32', 1);
      const spatialStatus = context.output('uint32', 2);
      const parameters = context.output('float32', 2);
      const residuals = context.output('float32', 10);
      const residualCovariance = context.output('float32', 4);
      const coefficientCovariance = context.output('float32', 16);
      const coefficientOutput = context.output('float32', 4);
      // Require the kernels to write success and the near-zero spatial parameters explicitly.
      status.buffer.write(Uint32Array.of(0xffffffff));
      spatialStatus.buffer.write(Uint32Array.of(0xffffffff, 0xffffffff));
      parameters.buffer.write(Float32Array.of(NaN, NaN));
      context.graph.add(
        new GPUSeeminglyUnrelatedRegression({
          rowCount: 5,
          equationCount: 2,
          predictorCount: 1,
          predictors: context.input('float32', [
            ...centered.map(value => origin + value),
            ...centered
          ]),
          responses: context.input('float32', [
            ...centered.map((value, row) => 1 + 1.5 * value + TARGET_RESIDUALS[row]),
            ...centered.map((value, row) => -2 + 0.5 * value + SECOND_RESIDUALS[row])
          ]),
          spatial: {
            weights: {
              offsets: context.input('uint32', [0, 4, 8, 12, 16, 20]),
              neighbors: context.input(
                'uint32',
                [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]
              ),
              weights: context.input('float32', WEIGHTS)
            },
            output: {parameters: parameters.view, status: spatialStatus.view}
          },
          output: {
            coefficients: coefficientOutput.view,
            covariance: coefficientCovariance.view,
            residualCovariance: residualCovariance.view,
            residuals: residuals.view,
            status: status.view
          }
        })
      );
      context.run();
      // A failed fit must not masquerade as a residual cancellation reproduction.
      expect(Array.from(await readUint32(spatialStatus.buffer, 2))).toEqual([0, 0]);
      expect(Array.from(await readUint32(status.buffer, 1))).toEqual([0]);
      const parameterValues = await readFloat32(parameters.buffer, 2);
      expect(parameterValues).toHaveLength(2);
      for (const parameter of parameterValues) {
        expect(Number.isFinite(parameter)).toBe(true);
        expect(Math.abs(parameter)).toBeLessThan(1e-3);
      }
      const covariance = await readFloat32(residualCovariance.buffer, 4);
      expect(covariance).toHaveLength(4);
      [0.05, 0, 0, 0.05].forEach((expected, index) => {
        expect(Number.isFinite(covariance[index])).toBe(true);
        expect(Math.abs(covariance[index] - expected)).toBeLessThan(1e-3);
      });
      // Sigma=.05 I and sum centered x^2=16: slope variance=.003125,
      // structural intercept variance=.01+origin^2*.003125, covariance=-origin*.003125.
      const structuralCovariance = await readFloat32(coefficientCovariance.buffer, 16);
      expect(structuralCovariance).toHaveLength(16);
      const expectedCovariance = [
        0.01 + origin * origin * 0.003125,
        -origin * 0.003125,
        0,
        0,
        -origin * 0.003125,
        0.003125,
        0,
        0,
        0,
        0,
        0.01,
        0,
        0,
        0,
        0,
        0.003125
      ];
      expectedCovariance.forEach((expected, index) => {
        expect(Number.isFinite(structuralCovariance[index])).toBe(true);
        expect(
          Math.abs(structuralCovariance[index] - expected) / Math.max(1, Math.abs(expected)),
          `structural covariance slot ${index}, lambda=${parameterValues}, residual covariance=${covariance}`
        ).toBeLessThan(1e-3);
      });
      const values = await readFloat32(residuals.buffer, 10);
      expect(values).toHaveLength(10);
      [...TARGET_RESIDUALS, ...SECOND_RESIDUALS].forEach((expected, row) => {
        expect(Number.isFinite(values[row]), `residual ${row}`).toBe(true);
        expect(Math.abs(values[row] - expected), `residual ${row}`).toBeLessThan(1e-3);
      });
    } finally {
      context.destroy();
    }
  });
}

it('B1/S3 publishes small structural intercepts and nonzero cross-equation covariance', async () => {
  const context = await createRegressionTestContext();
  try {
    const origin = 2 ** 24;
    const centered = [0, 0, 0, 0, 4, 4, 4, 4];
    const firstErrors = [-1, 1, 1, -1, 1, 1, -1, -1];
    const secondErrors = [-1, 1, -1, 1, 1, 1, -1, -1];
    // Both errors are orthogonal to [1,x], u'Wu=0 on the successor ring,
    // and ||Wu||=||u||. KP lambda=0; Sigma=[[1,.5],[.5,1]].
    // Identical centered designs have sum (x-mean)^2=32. Thus cross-slope
    // covariance=.5/32, Cov(intercept1,slope2)=-(origin+2)*.5/32,
    // Cov(slope1,intercept2)=-2*.5/32, and the cross-intercept covariance
    // is .5*(1/8+(origin+2)*2/32). These are nonzero independent targets.
    const coefficients = context.output('float32', 4),
      covariance = context.output('float32', 16),
      residuals = context.output('float32', 16),
      status = context.output('uint32', 1),
      spatialStatus = context.output('uint32', 2),
      parameters = context.output('float32', 2);
    context.graph.add(
      new GPUSeeminglyUnrelatedRegression({
        rowCount: 8,
        equationCount: 2,
        predictorCount: 1,
        predictors: context.input('float32', [
          ...centered.map(value => origin + value),
          ...centered
        ]),
        responses: context.input('float32', [
          ...centered.map((value, row) => origin + value + 1 + firstErrors[row]),
          ...centered.map((value, row) => 2 + 2 * value + secondErrors[row])
        ]),
        spatial: {
          weights: {
            offsets: context.input('uint32', [0, 1, 2, 3, 4, 5, 6, 7, 8]),
            neighbors: context.input('uint32', [1, 2, 3, 4, 5, 6, 7, 0]),
            weights: context.input('float32', new Array(8).fill(1))
          },
          output: {parameters: parameters.view, status: spatialStatus.view}
        },
        output: {
          coefficients: coefficients.view,
          covariance: covariance.view,
          residualCovariance: context.output('float32', 4).view,
          residuals: residuals.view,
          status: status.view
        }
      })
    );
    context.run();
    expect(Array.from(await readUint32(status.buffer, 1))).toEqual([0]);
    expect(Array.from(await readUint32(spatialStatus.buffer, 2))).toEqual([0, 0]);
    const estimates = await readFloat32(coefficients.buffer, 4);
    const parameterValues = await readFloat32(parameters.buffer, 2);
    expect(estimates).toHaveLength(4);
    [1, 1, 2, 2].forEach((expected, index) => {
      expect(Number.isFinite(estimates[index])).toBe(true);
      expect(
        Math.abs(estimates[index] - expected),
        `structural coefficients=${estimates}, lambda=${parameterValues}`
      ).toBeLessThan(1e-3);
    });
    const errors = await readFloat32(residuals.buffer, 16);
    expect(errors).toHaveLength(16);
    [...firstErrors, ...secondErrors].forEach((expected, index) => {
      expect(Number.isFinite(errors[index])).toBe(true);
      expect(Math.abs(errors[index] - expected)).toBeLessThan(1e-3);
    });
    const matrix = await readFloat32(covariance.buffer, 16);
    expect(matrix).toHaveLength(16);
    for (const [index, expected] of [
      [2, 0.5 * (1 / 8 + ((origin + 2) * 2) / 32)],
      [3, (-(origin + 2) * 0.5) / 32],
      [6, (-2 * 0.5) / 32],
      [7, 0.5 / 32]
    ]) {
      expect(Number.isFinite(matrix[index])).toBe(true);
      expect(Math.abs(matrix[index] - expected) / Math.max(1, Math.abs(expected))).toBeLessThan(
        1e-3
      );
      const transpose = (index % 4) * 4 + Math.floor(index / 4);
      expect(Number.isFinite(matrix[transpose])).toBe(true);
      expect(Math.abs(matrix[transpose] - expected) / Math.max(1, Math.abs(expected))).toBeLessThan(
        1e-3
      );
    }
  } finally {
    context.destroy();
  }
});

it('B2 spatial SUR keeps the original-coordinate two-round stopping decision', async () => {
  const context = await createRegressionTestContext();
  try {
    const rows = 12;
    const predictors = Array.from({length: 24}, (_, index) => {
      const row = index % rows;
      return index < rows ? 100 + ((7 * row) % 11) : 300 + ((row * row + 3 * row) % 13);
    });
    const responses = Array.from({length: 24}, (_, index) => {
      const row = index % rows;
      return index < rows
        ? 2 + ((5 * row + 1) % 9)
        : -2 + ((5 * row + 3) % 9) + ((3 * row + 1) % 4);
    });
    const status = context.output('uint32', 1),
      count = context.output('uint32', 1);
    context.graph.add(
      new GPUSeeminglyUnrelatedRegression({
        rowCount: rows,
        equationCount: 2,
        predictorCount: 1,
        iteration: {maximumIterations: 2, tolerance: 0.02},
        predictors: context.input('float32', predictors),
        responses: context.input('float32', responses),
        spatial: {
          weights: {
            offsets: context.input(
              'uint32',
              Array.from({length: 13}, (_, row) => row)
            ),
            neighbors: context.input(
              'uint32',
              Array.from({length: 12}, (_, row) => (row + 1) % 12)
            ),
            weights: context.input('float32', new Array(12).fill(1))
          },
          output: {
            parameters: context.output('float32', 2).view,
            status: context.output('uint32', 2).view
          }
        },
        output: {
          coefficients: context.output('float32', 4).view,
          covariance: context.output('float32', 16).view,
          residualCovariance: context.output('float32', 4).view,
          residuals: context.output('float32', 24).view,
          status: status.view,
          iteration: {status: {iterationCount: count.view}}
        }
      })
    );
    context.run();
    // Independently solved in original filtered coordinates: two rounds exhaust the
    // .02 threshold (status 4), whereas the shifted-coordinate stopping test says 0.
    expect(Array.from(await readUint32(status.buffer, 1))).toEqual([4]);
    expect(Array.from(await readUint32(count.buffer, 1))).toEqual([2]);
  } finally {
    context.destroy();
  }
});

it('t1 SUR correction removes factorization ulps before a large origin amplifies them', async () => {
  const context = await createRegressionTestContext();
  try {
    const output = context.output('float32', 4);
    context.graph.add(
      createWGSLKernelNode(context.graph, {
        id: 'refine-sur-fixture',
        operation: 'SURRefinementFixture',
        invocationCount: 1,
        modules: [fp64arithmetic],
        bindings: [{name: 'output', view: output.view, type: 'f32', access: 'read_write'}],
        declarations: `${IS_FINITE_BITS_WGSL}\n${IS_FINITE_VALUE_WGSL}\n${getCholeskyWGSL(4)}\n${getSURGLSRefinementWGSL(4)}`,
        body: `var original = array<f32,16>(32.0,0.0,0.0,0.0,0.0,32.0,0.0,0.0,0.0,0.0,32.0,0.0,0.0,0.0,0.0,32.0);
var factor: array<f32,16>; for(var slot=0u;slot<16u;slot++){factor[slot]=original[slot];}
var right = array<f32,4>(32.0,32.0,64.0,64.0);
var scales = array<f32,4>(1.0,1.0,1.0,1.0);
var solution = array<f32,4>(1.0,1.0000003576278687,2.0,2.0);
_ = choleskyFactor_4(&factor);
refineSURSolution_4(&original,&right,&factor,&scales,&solution);
for(var slot=0u;slot<4u;slot++){output[outputOffset+slot]=solution[slot];}`
      })
    );
    context.run();
    const solution = await readFloat32(output.buffer, 4);
    // 32 I times (1,1,2,2) is exactly (32,32,64,64). Three slope ulps amplify to 6 at 2^24.
    expect(solution.every(Number.isFinite)).toBe(true);
    expect(solution).toEqual([1, 1, 2, 2]);
    expect(2 ** 24 - 2 ** 24 * solution[1] + 1).toBe(1);
  } finally {
    context.destroy();
  }
});
