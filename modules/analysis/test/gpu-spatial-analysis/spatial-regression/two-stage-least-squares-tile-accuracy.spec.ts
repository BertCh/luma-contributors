// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-two-stage-least-squares-parameters';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {createRegressionTestContext} from './math-review-regression-test-utils';

// Independent NumPy float64 calculation on these exact uploaded values:
// Z = column_stack([ones(11), x, W@y]); H = column_stack([ones(11), x, W@x])
// projected = H @ numpy.linalg.lstsq(H,Z,rcond=None)[0]
// normal = Z.T @ projected; beta = solve(normal, projected.T@y)
// u = y-Z@beta; sigma2 = u@u/11; covariance = sigma2*inv(normal).
// cond(normal)=32.688; this is an identified, well-conditioned model.
const COEFFICIENTS = [0.9333356054667526, 0.8844620156095567, 0.19249514331481565];
const VARIANCES = [0.7143115087011651, 0.10128349008256249, 0.20566462557008347];
const RESIDUALS = [
  1.161855424150506, 1.2784499505810984, -2.5108210354111997, 0.6057734910193928,
  -0.7226065914590507, -0.798507208343274, -0.664939845267714, 2.7009645206366515,
  -1.4711836679049792, -0.6456494325346783, 1.0666643945332472
];
const SIGMA_SQUARED = 2.0246974535836824;

// This varies regression row reduction tiles over ONE whole CSR. The public weights prop is
// GPUSpatialWeights, not a chunked CSR; chunked spatial weights would be a separate feature.
for (const tileRowCount of [11, 4]) {
  it(`2SLS ${tileRowCount}-row tiles match independent coefficients, variance and residuals`, async () => {
    const context = await createRegressionTestContext();
    try {
      const rowCount = 11;
      const neighbors: number[] = [];
      for (let row = 0; row < rowCount; row++) {
        neighbors.push(
          ...[(row + 1) % rowCount, (row + 3) % rowCount].sort((first, second) => first - second)
        );
      }
      const table = context.output('float32', 12);
      const summary = context.output('float32', GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH);
      const residuals = context.output('float32', rowCount);
      const status = context.output('uint32', 1);
      const contributor = new GPUSpatialTwoStageLeastSquares({
        predictorCount: 1,
        tileRowCount,
        predictors: context.input('float32', [-2, 1, 0, 3, -1, 2, -3, 1, 4, -2, 0]),
        response: context.input('float32', [1, 3, -1, 4, 0, 2, -2, 5, 3, -1, 2]),
        weights: {
          offsets: context.input(
            'uint32',
            Array.from({length: rowCount + 1}, (_, row) => row * 2)
          ),
          neighbors: context.input('uint32', neighbors),
          weights: context.input('float32', new Float32Array(rowCount * 2).fill(0.5))
        },
        output: {
          table: table.view,
          summary: summary.view,
          residuals: residuals.view,
          status: {code: status.view}
        }
      });
      expect(contributor.tileCount).toBe(tileRowCount === 11 ? 1 : 3);
      context.graph.add(contributor);
      context.run();
      expect(Array.from(await readUint32(status.buffer, 1))).toEqual([0]);
      const values = await readFloat32(table.buffer, 12);
      const check = (actual: number, expected: number) => {
        expect(Number.isFinite(actual)).toBe(true);
        expect(Math.abs(actual - expected)).toBeLessThan(2e-4);
      };
      COEFFICIENTS.forEach((expected, index) => check(values[index * 4], expected));
      VARIANCES.forEach((expected, index) => {
        expect(Number.isFinite(values[index * 4 + 1])).toBe(true);
        check(values[index * 4 + 1] ** 2, expected);
      });
      check(
        (await readFloat32(summary.buffer, GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH))[1],
        SIGMA_SQUARED
      );
      const errors = await readFloat32(residuals.buffer, rowCount);
      RESIDUALS.forEach((expected, row) => check(errors[row], expected));
    } finally {
      context.destroy();
    }
  });
}
