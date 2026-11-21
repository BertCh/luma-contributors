// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUSpatialPanelErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {createRegressionTestContext} from './math-review-regression-test-utils';

// W=[[0,1],[1,0]] has the independently known spectrum [+1,-1].
// The expanded I_T tensor W repeats both eigenvalues T times.
for (const periods of [8, 513]) {
  it(`SSR-118 entity and expanded spectra agree for N=2 T=${periods}`, async () => {
    const context = await createRegressionTestContext();
    try {
      const rows = 2 * periods;
      const predictors = Array.from({length: rows}, (_, row) => ((7 * row) % 19) - 9);
      const responses = predictors.map(
        (value, row) => 1 + 2 * value + ((row * row + 3 * row) % 11) - 5
      );
      const weights = {
        offsets: context.input('uint32', [0, 1, 2]),
        neighbors: context.input('uint32', [1, 0]),
        weights: context.input('float32', [1, 1])
      };
      const run = (expanded: boolean) => {
        const table = context.output('float32', 12),
          summary = context.output('float32', 22),
          status = context.output('uint32', 1),
          converged = context.output('uint32', 1);
        const eigenvalues = expanded
          ? Array.from({length: rows}, (_, index) => (index % 2 === 0 ? 1 : -1))
          : [1, -1];
        context.graph.add(
          new GPUSpatialPanelErrorMaximumLikelihood({
            id: expanded ? 'expanded' : 'entity',
            periodCount: periods,
            predictorCount: 1,
            predictors: context.input('float32', predictors),
            response: context.input('float32', responses),
            weights,
            logDeterminant: 'exact',
            eigenvalues: context.input('float32', eigenvalues),
            output: {
              table: table.view,
              summary: summary.view,
              status: {code: status.view, converged: converged.view}
            }
          })
        );
        return {table, summary, status};
      };
      const entity = run(false);
      // The historical NT<=1024 route remains comparable at T=8. For T=513,
      // qualification is specifically the entity-bounded route, not a new NT bound.
      const expanded = periods === 8 ? run(true) : undefined;
      context.run();
      expect(Array.from(await readUint32(entity.status.buffer, 1))).toEqual([0]);
      const entityTable = await readFloat32(entity.table.buffer, 12);
      const entitySummary = await readFloat32(entity.summary.buffer, 22);
      const expected =
        periods === 8
          ? {
              slope: 2.18674899,
              parameter: -0.24101298659907666,
              determinant: -0.47874112867459057,
              slopeError: 0.15130917991249115,
              parameterError: 0.1665082214863705
            }
          : {
              slope: 2.00667205,
              parameter: -0.04408877896547473,
              determinant: -0.9981503080059425,
              slopeError: 0.020039206599131606,
              parameterError: 0.022032628400138365
            };
      // NumPy filtered least squares and symmetric Gaussian information, entity.py.
      for (const [slot, value] of [
        [4, expected.slope],
        [5, expected.slopeError],
        [8, expected.parameter],
        [9, expected.parameterError]
      ]) {
        expect(Number.isFinite(entityTable[slot])).toBe(true);
        expect(Math.abs(entityTable[slot] - value)).toBeLessThan(2e-4);
      }
      expect(Number.isFinite(entitySummary[21])).toBe(true);
      expect(Math.abs(entitySummary[21] - expected.determinant)).toBeLessThan(3e-4);
      expect(entityTable).toHaveLength(12);
      expect(entitySummary).toHaveLength(22);
      for (const index of [4, 8]) expect(Number.isFinite(entityTable[index])).toBe(true);
      if (expanded) {
        expect(Array.from(await readUint32(expanded.status.buffer, 1))).toEqual([0]);
        const expandedTable = await readFloat32(expanded.table.buffer, 12);
        const expandedSummary = await readFloat32(expanded.summary.buffer, 22);
        expect(expandedTable).toHaveLength(12);
        expect(expandedSummary).toHaveLength(22);
        for (const index of [4, 5, 8, 9]) {
          expect(Number.isFinite(expandedTable[index])).toBe(true);
          expect(Math.abs(entityTable[index] - expandedTable[index])).toBeLessThan(1e-4);
        }
        for (const index of [0, 1, 2, 3]) {
          expect(Number.isFinite(entitySummary[index])).toBe(true);
          expect(Number.isFinite(expandedSummary[index])).toBe(true);
          expect(Math.abs(entitySummary[index] - expandedSummary[index])).toBeLessThan(1e-3);
        }
      }
    } finally {
      context.destroy();
    }
  });
}

it('SSR-118 rejects a spectrum length that matches neither entities nor expanded rows', async () => {
  const context = await createRegressionTestContext();
  try {
    expect(
      () =>
        new GPUSpatialPanelErrorMaximumLikelihood({
          periodCount: 3,
          predictorCount: 1,
          predictors: context.input('float32', [0, 1, 2, 3, 4, 5]),
          response: context.input('float32', [1, 4, 2, 5, 8, 3]),
          weights: {
            offsets: context.input('uint32', [0, 1, 2]),
            neighbors: context.input('uint32', [1, 0]),
            weights: context.input('float32', [1, 1])
          },
          logDeterminant: 'exact',
          eigenvalues: context.input('float32', [1, -1, 1]),
          output: {
            table: context.output('float32', 12).view,
            summary: context.output('float32', 22).view,
            status: {
              code: context.output('uint32', 1).view,
              converged: context.output('uint32', 1).view
            }
          }
        })
    ).toThrow(/eigenvalues/);
  } finally {
    context.destroy();
  }
});

it('SSR-118 repeated scalar entity spectrum is certified against the expanded CSR', async () => {
  const context = await createRegressionTestContext();
  try {
    const predictors = Array.from({length: 16}, (_, row) => ((7 * row) % 19) - 9);
    const table = context.output('float32', 12),
      summary = context.output('float32', 22),
      status = context.output('uint32', 1);
    context.graph.add(
      new GPUSpatialPanelErrorMaximumLikelihood({
        periodCount: 8,
        predictorCount: 1,
        maximumPower: 8,
        logDeterminant: 'exact',
        eigenvalues: context.input('float32', [0.25, 0.25]),
        predictors: context.input('float32', predictors),
        response: context.input(
          'float32',
          predictors.map((value, row) => 1 + 2 * value + ((row * row + 3 * row) % 11) - 5)
        ),
        weights: {
          offsets: context.input('uint32', [0, 1, 2]),
          neighbors: context.input('uint32', [0, 1]),
          weights: context.input('float32', [0.25, 0.25])
        },
        output: {table: table.view, summary: summary.view, status: {code: status.view}}
      })
    );
    context.run();
    expect(Array.from(await readUint32(status.buffer, 1))).toEqual([0]);
    const actual = await readFloat32(table.buffer, 12),
      values = await readFloat32(summary.buffer, 22);
    expect(Number.isFinite(actual[8])).toBe(true);
    expect(actual[8]).toBeCloseTo(-3.96, 5);
    expect(actual[9]).toBeNaN();
    expect(values[19]).toBe(0);
    expect(values[20]).toBe(1);
    expect(Number.isFinite(values[21])).toBe(true);
    expect(values[21]).toBeCloseTo(16 * Math.log(1.99), 4);
  } finally {
    context.destroy();
  }
});
