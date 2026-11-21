// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeSpatialWeightsTracesOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-weights-traces-operation';
import {computeSpatialLagImpactsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-lag-impacts-operation';
import {
  getSpatialExactLogDeterminant,
  getSpatialWeightsEigenvalues,
  getSymmetricEigenvalues
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-exact-log-determinant';
import {
  getSpatialLogDeterminantDerivativeFromTraces,
  getSpatialMaximumLikelihoodTraceOne,
  getSpatialMaximumLikelihoodTraceTwo
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-parameters';
import {
  getSpatialSeriesDerivativeTail,
  getSpatialSeriesLogDeterminantTail,
  getSpatialSeriesSecondTail,
  getSpatialSeriesTailModel
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-series-tail';
import {
  getSpatialLogDeterminantFromTraces,
  getSpatialLogDeterminantErrorBound,
  getSpatialLogDeterminantTruncationBound,
  getSpatialMultiplierTraceFromTraces,
  getSpatialMultiplierTotalFromPowerSums,
  getSpatialMultiplierTruncationBound
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-parameters';
import {expectRegressionRowsClose} from './regression-operation-test-utils';
import fixture from './likelihood-series-audit-fixture.json';

function expectClose(actual: number, expected: number, tolerance = 1e-11) {
  expect(Number.isFinite(actual)).toBe(true);
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(
    tolerance * Math.max(1, Math.abs(expected))
  );
}

for (const mode of ['exact', 'stochastic'] as const) {
  it(`audits every ${mode} trace output, including nonzero sampling errors, against dense NumPy`, () => {
    const result = computeSpatialWeightsTracesOnCPU(
      {
        weights: {
          offsets: Uint32Array.from(fixture.weights.offsets),
          neighbors: Uint32Array.from(fixture.weights.neighbors),
          weights: Float32Array.from(fixture.weights.weights)
        },
        parameters: Uint32Array.of(17)
      },
      {maximumPower: 8, mode, probeCount: 9, includeGramTraces: true}
    );
    for (const port of Object.keys(fixture.traces[mode]) as (keyof typeof result)[]) {
      expectRegressionRowsClose(result[port]!, fixture.traces[mode][port], 1e-7);
    }
    console.info(
      `${mode} trace maximum absolute differences from rounded dense oracle`,
      Object.fromEntries(
        Object.entries(fixture.traces[mode]).map(([port, values]) => [
          port,
          Math.max(
            ...values.map((value, row) =>
              Math.abs(result[port as keyof typeof result]![row] - Math.fround(value))
            )
          )
        ])
      )
    );
  });
}

it('audits eigenvalues, log determinants, derivatives and the actual truncation error near both bounds', () => {
  const eigenvalues = getSpatialWeightsEigenvalues(
    fixture.weights.offsets,
    fixture.weights.neighbors,
    fixture.weights.weights,
    8
  );
  eigenvalues.forEach((value, row) => expectClose(value, fixture.eigenvalues[row]));
  // Symmetric similar form of the three-row path; repeated eigenvalues are retained.
  const symmetric = Array.from({length: 8}, () => new Float64Array(8));
  symmetric[0][1] = symmetric[1][0] = 1;
  symmetric[2][3] = symmetric[3][2] = symmetric[3][4] = symmetric[4][3] = Math.SQRT1_2;
  symmetric[5][5] = 0.25;
  symmetric[6][6] = -0.5;
  getSymmetricEigenvalues(symmetric).forEach((value, row) =>
    expectClose(value, fixture.eigenvalues[row])
  );
  for (const reference of fixture.cases) {
    const {parameter, maximumPower} = reference;
    const traces = fixture.powerTraces.slice(0, maximumPower);
    const actual = getSpatialLogDeterminantFromTraces(parameter, traces);
    expectClose(actual, reference.truncated);
    expectClose(getSpatialExactLogDeterminant(parameter, eigenvalues), reference.exact);
    expectClose(getSpatialMaximumLikelihoodTraceOne(parameter, traces), reference.traceOne);
    expectClose(
      getSpatialLogDeterminantDerivativeFromTraces(parameter, traces),
      -reference.traceOne
    );
    expectClose(getSpatialMaximumLikelihoodTraceTwo(parameter, traces), reference.traceTwo);
    const bound = getSpatialLogDeterminantTruncationBound(parameter, maximumPower, 8);
    expect(Number.isFinite(bound)).toBe(true);
    expect(Math.abs(actual - reference.exact)).toBeLessThanOrEqual(bound + 2e-12);
    if (maximumPower === 128) {
      // At Q=128 the two dominant +/-1 modes give the exact geometric/alternating tail;
      // smaller eigenvalues contribute less than 2^-128. Log-tail stopping costs <1e-9.
      expectClose(
        getSpatialSeriesLogDeterminantTail(parameter, traces),
        reference.exact - actual,
        1e-9
      );
      expectClose(
        getSpatialSeriesDerivativeTail(parameter, traces),
        reference.derivativeTail,
        1e-10
      );
      expectClose(getSpatialSeriesSecondTail(parameter, traces), reference.secondTail, 1e-10);
    }
  }
  expect(getSpatialSeriesTailModel(fixture.powerTraces)).toEqual({
    level: 2,
    ratio: 1,
    alternatingLevel: 2,
    alternatingRatio: -1
  });
});

it('audits the complete finite impact series and multiplier remainder bounds against dense inverses', () => {
  const traces = fixture.traces.exact;
  const actual = computeSpatialLagImpactsOnCPU(
    {
      coefficients: Float32Array.of(99, 2, -3, 0.3),
      powerTraces: Float32Array.from(traces.powerTraces),
      powerSums: Float32Array.from(traces.powerSums),
      infinityNorm: Float32Array.of(1),
      status: Uint32Array.of(0)
    },
    {coefficientStride: 1, rhoIndex: 3, predictorCount: 2, rowCount: 8, maximumPower: 8}
  );
  expectRegressionRowsClose(actual.impacts, fixture.impacts, 3e-7);
  console.info(
    'impact maximum absolute difference from rounded dense oracle',
    Math.max(
      ...fixture.impacts.map((value, row) => Math.abs(actual.impacts[row] - Math.fround(value)))
    )
  );
  for (const reference of fixture.cases.filter(value => value.maximumPower <= 8)) {
    const {parameter, maximumPower} = reference;
    const direct =
      getSpatialMultiplierTraceFromTraces(
        parameter,
        fixture.powerTraces.slice(0, maximumPower),
        8
      ) / 8;
    const total =
      getSpatialMultiplierTotalFromPowerSums(
        parameter,
        traces.powerSums.slice(0, maximumPower),
        8
      ) / 8;
    const bound = getSpatialMultiplierTruncationBound(parameter, maximumPower, 8) / 8;
    expect(Number.isFinite(direct) && Number.isFinite(total)).toBe(true);
    expect(Math.abs(direct - reference.direct)).toBeLessThanOrEqual(bound + 1e-12);
    expect(Math.abs(total - reference.total)).toBeLessThanOrEqual(bound + 1e-12);
  }
});

it('uses the stated geometric remainder and correlated standard-error bounds, including scale', () => {
  // t=.25*2=.5, n=8, Q=3: log remainder=8*(1/16)/(4*1/2)=1/4;
  // multiplier remainder=8*(1/16)/(1/2)=1. The SE bound is
  // (1/2)*2 + (1/4)*3/2 + (1/8)*4/3 = 37/24.
  expectClose(getSpatialLogDeterminantTruncationBound(0.25, 3, 8, 2), 0.25);
  expectClose(getSpatialMultiplierTruncationBound(0.25, 3, 8, 2), 1);
  expectClose(getSpatialLogDeterminantErrorBound(0.25, [2, 3, 4], 2), 37 / 24);
  expect(getSpatialLogDeterminantTruncationBound(-0.5, 3, 8, 2)).toBe(Infinity);
  expect(getSpatialMultiplierTruncationBound(0.5, 3, 8, 2)).toBe(Infinity);
});
