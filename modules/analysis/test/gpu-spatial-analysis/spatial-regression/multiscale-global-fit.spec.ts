// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createMultiscaleGlobalFit} from '../../../src/gpu-spatial-analysis/spatial-regression/multiscale-global-fit';
import {
  getRegressionBindings,
  getRegressionWGSL
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression';
import {readFloat32} from '../../utils/gpu-contributor-test-utils';
import {createRegressionTestContext, fitLeastSquares} from './math-review-regression-test-utils';

for (const scenario of ['ordinary', 'large-offset', 'empty', 'too-few', 'singular'] as const) {
  it(`MGWR tiled starting fit preserves the centered fit and zero fallback: ${scenario}`, async () => {
    const context = await createRegressionTestContext();
    const {graph, input, output} = context;
    try {
      // Partial final tile, a whole empty tile, leading/trailing padding, and nonfinite rows.
      const rowCount = 321;
      const predictors = Float32Array.from({length: rowCount * 2}, (_, index) => {
        const row = Math.floor(index / 2);
        const first = (row % 13) - 6;
        if (scenario === 'singular') return index % 2 ? 1 : first;
        return index % 2 ? ((row * 7) % 17) - 8 : first + (scenario === 'large-offset' ? 10000 : 0);
      });
      const response = Float32Array.from(
        {length: rowCount},
        (_, row) => 4 + 2 * predictors[row * 2] - 0.5 * predictors[row * 2 + 1] + (row % 5) * 0.125
      );
      const mask = Uint32Array.from({length: rowCount}, (_, row) =>
        Number(row >= 64 && row < rowCount - 1 && row % 11 !== 0)
      );
      if (scenario === 'empty') mask.fill(0);
      if (scenario === 'too-few') {
        mask.fill(0);
        mask[100] = 1;
        mask[200] = 1;
      }
      response[0] = NaN;
      response[128] = NaN;
      predictors[2 * 193] = Infinity;
      const positions = new Float32Array(rowCount * 2);
      positions[2 * 194] = NaN;
      const positionScalars = input('float32', positions);
      // Predictor, response, mask and output views have nonzero byte offsets; sentinels stay untouched.
      const sliceInput = (values: Float32Array | Uint32Array) => {
        const padded =
          values instanceof Uint32Array
            ? new Uint32Array(values.length + 4)
            : new Float32Array(values.length + 4);
        padded.fill(999);
        padded.set(values, 2);
        const source = input(values instanceof Uint32Array ? 'uint32' : 'float32', padded);
        return graph.createDataView(source.buffer, {
          format: source.format,
          byteOffset: source.byteOffset + 8,
          length: values.length
        });
      };
      const props = {
        positions: graph.createDataView(positionScalars.buffer, {
          format: 'float32x2' as const,
          length: rowCount
        }),
        predictors: sliceInput(predictors) as ReturnType<typeof input<'float32'>>,
        response: sliceInput(response) as ReturnType<typeof input<'float32'>>,
        mask: sliceInput(mask) as ReturnType<typeof input<'uint32'>>,
        parameters: input('float32', [1e-5, 10, 10, 10]),
        predictorCount: 2
      };
      const coefficients = output('float32', 7);
      coefficients.buffer.write(new Float32Array(7).fill(12345));
      const coefficientView = graph.createDataView(coefficients.view.buffer, {
        format: 'float32',
        byteOffset: coefficients.view.byteOffset + 8,
        length: 3
      });
      for (const node of createMultiscaleGlobalFit(graph, {
        id: 'starting-fit',
        rowCount,
        predictorCount: 2,
        bindings: getRegressionBindings(props).filter(binding => binding.name !== 'parameters'),
        declarations: getRegressionWGSL(props),
        output: coefficientView
      }))
        graph.add(node);
      context.run();
      const actual = Array.from(await readFloat32(coefficients.buffer, 7));
      expect(actual.slice(0, 2)).toEqual([12345, 12345]);
      expect(actual.slice(5)).toEqual([12345, 12345]);
      if (['empty', 'too-few', 'singular'].includes(scenario)) {
        expect(actual.slice(2, 5)).toEqual([0, 0, 0]);
      } else {
        const rows = Array.from({length: rowCount}, (_, row) => row).filter(
          row =>
            mask[row] !== 0 &&
            Number.isFinite(response[row]) &&
            Number.isFinite(predictors[2 * row]) &&
            Number.isFinite(positions[2 * row])
        );
        const expected = fitLeastSquares(
          rows.map(row => [1, predictors[2 * row], predictors[2 * row + 1]]),
          rows.map(row => response[row])
        ).estimate;
        for (let column = 0; column < 3; column++) {
          // The large-offset intercept cancels quantities around 20,000 in float32.
          expect(Math.abs(actual[column + 2] - expected[column])).toBeLessThan(
            scenario === 'large-offset' && column === 0 ? 0.02 : 2e-5
          );
        }
      }
    } finally {
      context.destroy();
    }
  });
}

for (const responseSign of [1, -1]) {
  it(`MGWR starting fit avoids centering overflow with a ${responseSign > 0 ? 'positive' : 'negative'} finite outlier`, async () => {
    const context = await createRegressionTestContext();
    const {graph, input, output} = context;
    try {
      const rowCount = 129;
      const predictors = Float32Array.from({length: rowCount}, (_, row) =>
        row === rowCount - 1 ? 0 : row % 2 ? 1 : -1
      );
      const response = new Float32Array(rowCount);
      response[rowCount - 1] = responseSign * 1e37;
      const positionScalars = input('float32', new Float32Array(rowCount * 2));
      const props = {
        positions: graph.createDataView(positionScalars.buffer, {
          format: 'float32x2' as const,
          length: rowCount
        }),
        predictors: input('float32', predictors),
        response: input('float32', response),
        parameters: input('float32', [1e-5, 10, 10]),
        predictorCount: 1
      };
      const coefficients = output('float32', 2);
      for (const node of createMultiscaleGlobalFit(graph, {
        id: 'outlier-fit',
        rowCount,
        predictorCount: 1,
        bindings: getRegressionBindings(props).filter(binding => binding.name !== 'parameters'),
        declarations: getRegressionWGSL(props),
        output: coefficients.view
      }))
        graph.add(node);
      context.run();
      const actual = await readFloat32(coefficients.buffer, 2);
      // sum(x) = sum(x*y) = 0, sum(x*x) = 128, and only the final response is nonzero.
      const expectedIntercept = response[rowCount - 1] / rowCount;
      expect(Number.isFinite(actual[0])).toBe(true);
      // Forming the intercept from a reference 129 times its size loses a few float32 bits.
      expect(Math.abs(actual[0] - expectedIntercept)).toBeLessThan(
        Math.abs(expectedIntercept) * 2e-5
      );
      expect(Math.abs(actual[1])).toBe(0);
    } finally {
      context.destroy();
    }
  });
}

for (const magnitude of [0, 2 ** -120, 2 ** 127]) {
  for (const responseSign of [1, -1]) {
    it(`MGWR starting fit centers opposite-sign responses: magnitude=${magnitude}, sign=${responseSign}`, async () => {
      const context = await createRegressionTestContext();
      const {graph, input, output} = context;
      try {
        const positionScalars = input('float32', new Float32Array(6));
        const props = {
          positions: graph.createDataView(positionScalars.buffer, {
            format: 'float32x2' as const,
            length: 3
          }),
          predictors: input('float32', [0.5, -0.5, 0]),
          response: input('float32', [-responseSign * magnitude, 0, responseSign * magnitude]),
          parameters: input('float32', [1e-5, 10, 10]),
          predictorCount: 1
        };
        const coefficients = output('float32', 2);
        for (const node of createMultiscaleGlobalFit(graph, {
          id: 'opposite-sign-fit',
          rowCount: 3,
          predictorCount: 1,
          bindings: getRegressionBindings(props).filter(binding => binding.name !== 'parameters'),
          declarations: getRegressionWGSL(props),
          output: coefficients.view
        }))
          graph.add(node);
        context.run();
        const actual = await readFloat32(coefficients.buffer, 2);
        // sum(x) = sum(y) = 0, sum(x*x) = 1/2, sum(x*y) = -sign*M/2.
        expect(Number.isFinite(actual[0])).toBe(true);
        expect(Number.isFinite(actual[1])).toBe(true);
        if (magnitude === 0) {
          expect(Math.abs(actual[0])).toBe(0);
          expect(Math.abs(actual[1])).toBe(0);
        } else {
          // The zero intercept is recovered by cancellation; measure error on the response scale.
          expect(Math.abs(actual[0] / magnitude)).toBeLessThan(2e-6);
          expect(Math.abs(actual[1] / magnitude + responseSign)).toBeLessThan(2e-6);
        }
      } finally {
        context.destroy();
      }
    });
  }
}
