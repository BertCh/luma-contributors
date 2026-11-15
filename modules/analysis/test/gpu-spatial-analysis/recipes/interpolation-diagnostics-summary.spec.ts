// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {addInterpolationDiagnosticsSummaryNodes} from '../../../src/gpu-spatial-analysis/recipes/interpolation-diagnostics-summary';
import {RecipeBuilder} from '../../../src/gpu-spatial-analysis/recipes/recipe-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {RecipeTestFixture} from './recipe-harness';

for (const rowCount of [0, 1, 257, 513]) {
  it(`parallel interpolation summary matches float64 metrics for ${rowCount} rows and live updates`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = new RecipeTestFixture(device, `diagnostic-summary-${rowCount}`);
    let compiled: ReturnType<typeof fixture.graph.compile> | undefined;
    try {
      // Nonzero offsets and hostile padding test both view indexing and the last partial group.
      const residualStorage = fixture.output('residual-storage', 'float32', rowCount + 2);
      const scoreStorage = fixture.output('score-storage', 'float32', rowCount + 2);
      const residuals = fixture.graph.createDataView(residualStorage.view.buffer, {
        format: 'float32',
        byteOffset: 4,
        length: rowCount
      });
      const standardizedResiduals = fixture.graph.createDataView(scoreStorage.view.buffer, {
        format: 'float32',
        byteOffset: 4,
        length: rowCount
      });
      const validCount = fixture.output('valid-count', 'uint32', 1);
      const standardizedCount = fixture.output('score-count', 'uint32', 1);
      const invalidCount = fixture.output('invalid-count', 'uint32', 1);
      const bias = fixture.output('bias', 'float32', 1);
      const meanAbsoluteError = fixture.output('absolute-error', 'float32', 1);
      const rootMeanSquaredError = fixture.output('root-error', 'float32', 1);
      addInterpolationDiagnosticsSummaryNodes(
        new RecipeBuilder(fixture.graph),
        'summary',
        {
          residuals,
          standardizedResiduals,
          validCount: validCount.view,
          standardizedCount: standardizedCount.view,
          bias: bias.view,
          meanAbsoluteError: meanAbsoluteError.view,
          rootMeanSquaredError: rootMeanSquaredError.view
        },
        invalidCount.view
      );
      compiled = fixture.graph.compile();
      const maximumFloat = Math.fround(3.4028234663852886e38);
      for (const mode of [
        'mixed',
        'missing',
        'zero',
        'large',
        'large-negative',
        'large-fractions',
        'large-cancellation',
        'cancellation'
      ] as const) {
        const values = Float32Array.from({length: rowCount}, (_, row) => {
          if (mode === 'missing') return row % 2 ? Infinity : NaN;
          if (mode === 'zero') return 0;
          if (mode === 'large') return maximumFloat;
          if (mode === 'large-negative') return -maximumFloat;
          if (mode === 'large-fractions') return maximumFloat / ((row % 3) + 1);
          if (mode === 'large-cancellation') return row % 2 ? -maximumFloat : maximumFloat;
          if (mode === 'cancellation') return row % 2 ? -1e30 : 1e30;
          return row % 11 === 0 ? NaN : ((row % 19) - 9) / 7;
        });
        const scores = Float32Array.from(values, (value, row) => (row % 3 ? value : NaN));
        residualStorage.buffer.write(new Float32Array([maximumFloat, ...values, maximumFloat]));
        scoreStorage.buffer.write(new Float32Array([maximumFloat, ...scores, maximumFloat]));
        submitGraph(device, compiled, undefined);
        const finiteValues = Array.from(values).filter(Number.isFinite);
        const finiteScores = Array.from(scores).filter(Number.isFinite);
        expect(await fixture.readUint32(validCount, 1), mode).toEqual([finiteValues.length]);
        expect(await fixture.readUint32(standardizedCount, 1), mode).toEqual([finiteScores.length]);
        expect(await fixture.readUint32(invalidCount, 1), mode).toEqual([
          rowCount - finiteValues.length
        ]);
        const actual = [
          (await fixture.readFloat32(bias, 1))[0],
          (await fixture.readFloat32(meanAbsoluteError, 1))[0],
          (await fixture.readFloat32(rootMeanSquaredError, 1))[0]
        ];
        if (!finiteValues.length) {
          expect(actual, mode).toEqual([NaN, NaN, NaN]);
          continue;
        }
        // Independent float64 oracle: sums and squares stay finite in JavaScript doubles.
        const expected = [
          finiteValues.reduce((total, value) => total + value, 0) / finiteValues.length,
          finiteValues.reduce((total, value) => total + Math.abs(value), 0) / finiteValues.length,
          Math.sqrt(
            finiteValues.reduce((total, value) => total + value * value, 0) / finiteValues.length
          )
        ];
        const magnitude = Math.max(...finiteValues.map(Math.abs));
        actual.forEach((value, index) => {
          expect(Number.isFinite(value), `${mode}: metric ${index}`).toBe(true);
          expect(Math.abs(value - expected[index]), `${mode}: metric ${index}`).toBeLessThanOrEqual(
            2e-6 * magnitude + 1e-6
          );
        });
      }
    } finally {
      compiled?.destroy();
      fixture.destroy();
    }
  });
}
