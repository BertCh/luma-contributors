// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSeeminglyUnrelatedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression';
import {InterpolationBenchmarkScope} from '../spatial-interpolation/interpolation-performance-harness';
import {requireTestDevice} from '../../utils/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {calculateSURReferenceWithQR} from './sur-qr-reference';

for (const [rows, equations, columns, supplied, residualScale] of [
  [513, 2, 1, false, 1],
  [513, 3, 4, true, 1],
  [1025, 4, 3, false, 1],
  [65536, 2, 7, true, 1],
  [2049, 2, 1, true, 1e19],
  [513, 2, 1, true, 1e20]
] as const) {
  it(`tiled SUR ${rows}/${equations}/${columns}/${residualScale} preserves QR outputs and replay status`, {
    timeout: 120000
  }, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scope = new InterpolationBenchmarkScope(device, `sur-${rows}-${equations}-${columns}`);
    const size = equations * (columns + 1);
    const predictors = Float32Array.from({length: rows * equations * columns}, (_, index) => {
      const row = Math.floor(index / columns) % rows;
      const equation = Math.floor(index / (rows * columns));
      return 10 + Math.sin(row * (0.17 + (index % columns) * 0.23) + equation * 0.31);
    });
    const initialResiduals = Float32Array.from(
      {length: rows * equations},
      (_, index) => 0.1 * Math.cos((index % rows) * (0.37 + Math.floor(index / rows) * 0.13))
    );
    const responses = Float32Array.from(initialResiduals, (error, index) => {
      let value = 2 + error;
      for (let column = 0; column < columns; column++)
        value += (predictors[index * columns + column] * (column + 1)) / 10;
      return value;
    });
    // A finite covariance can have an unnormalized sum beyond the f32 range.
    for (let index = 0; index < initialResiduals.length; index++)
      initialResiduals[index] *= residualScale;
    const mask = new Uint32Array(rows).fill(1);
    scope.graph.add(
      new GPUSeeminglyUnrelatedRegression({
        equationCount: equations,
        predictorCount: columns,
        rowCount: rows,
        predictors: scope.input('predictors', predictors, 'float32'),
        responses: scope.input('responses', responses, 'float32'),
        mask: scope.input('mask', mask, 'uint32'),
        initialResiduals: supplied
          ? scope.input('initialResiduals', initialResiduals, 'float32')
          : undefined,
        output: {
          coefficients: scope.output('coefficients', size, 'float32'),
          covariance: scope.output('covariance', size * size, 'float32'),
          residualCovariance: scope.output('residualCovariance', equations * equations, 'float32'),
          residuals: scope.output('residuals', rows * equations, 'float32'),
          status: scope.output('status', 1, 'uint32')
        }
      })
    );
    const compiled = scope.graph.compile();
    try {
      const scenarios =
        rows === 65536 || residualScale !== 1
          ? ['initial']
          : [
              'initial',
              'insufficient',
              'singular-design',
              ...(supplied ? ['singular-error'] : []),
              'changed',
              'initial'
            ];
      for (const scenario of scenarios) {
        const changedPredictors =
          scenario === 'singular-design' ? new Float32Array(predictors.length).fill(1) : predictors;
        const changedResponses = Float32Array.from(responses, value =>
          scenario === 'changed' ? value * 1.2 : value
        );
        const changedMask = mask.slice();
        const changedResiduals = initialResiduals.slice();
        if (scenario === 'insufficient') changedMask.fill(0, 2);
        if (scenario === 'singular-error') changedResiduals.fill(0);
        if (scenario === 'changed') {
          changedMask.fill(0, 0, 256); // An entirely excluded tile and a partially filled final tile.
          changedResponses[rows + 300] = NaN;
          if (supplied) changedResiduals[301] = NaN;
        }
        scope.initialWrites[0].buffer.write(changedPredictors);
        scope.initialWrites[1].buffer.write(changedResponses);
        scope.initialWrites[2].buffer.write(changedMask);
        if (supplied) scope.initialWrites[3].buffer.write(changedResiduals);
        submitGraph(device, compiled, undefined);
        const actual = await scope.read();
        const failure =
          scenario === 'insufficient'
            ? 1
            : scenario === 'singular-design'
              ? 2
              : scenario === 'singular-error'
                ? 3
                : 0;
        expect(actual['status']).toEqual([failure]);
        if (failure) {
          for (const key of ['coefficients', 'covariance', 'residuals'])
            expect(actual[key].every(Number.isNaN)).toBe(true);
          expect(
            actual['residualCovariance'].every(failure === 3 ? value => value === 0 : Number.isNaN)
          ).toBe(true);
          continue;
        }
        const included = Array.from({length: rows}, (_, row) => row).filter(
          row =>
            changedMask[row] !== 0 &&
            Array.from({length: equations}, (_, equation) => equation).every(
              equation =>
                Number.isFinite(changedResponses[equation * rows + row]) &&
                (!supplied || Number.isFinite(changedResiduals[equation * rows + row]))
            )
        );
        const selectedResponses = [],
          selectedPredictors = [],
          selectedResiduals = [];
        for (let equation = 0; equation < equations; equation++)
          for (const row of included) {
            selectedResponses.push(changedResponses[equation * rows + row]);
            selectedResiduals.push(changedResiduals[equation * rows + row]);
            for (let column = 0; column < columns; column++)
              selectedPredictors.push(
                changedPredictors[(equation * rows + row) * columns + column]
              );
          }
        const expected = calculateSURReferenceWithQR(
          selectedPredictors,
          selectedResponses,
          equations,
          columns,
          supplied ? selectedResiduals : undefined
        );
        for (const key of ['coefficients', 'covariance', 'residualCovariance'] as const) {
          actual[key].forEach((value, index) =>
            expect(
              Math.abs(value - expected[key][index]),
              `${scenario} ${key}[${index}]`
            ).toBeLessThanOrEqual(1e-4 * Math.max(1, Math.abs(expected[key][index])))
          );
        }
        const includedPositions = new Map(included.map((row, index) => [row, index]));
        actual['residuals'].forEach((value, index) => {
          const position = includedPositions.get(index % rows);
          if (position === undefined) expect(Number.isNaN(value)).toBe(true);
          else
            expect(
              Math.abs(
                value - expected.residuals[Math.floor(index / rows) * included.length + position]
              )
            ).toBeLessThanOrEqual(1e-4);
        });
      }
    } finally {
      compiled.destroy();
      scope.destroy();
    }
  });
}
