// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {computeGeographicallyWeightedRegressionOnCPU} from './geographically-weighted-regression-oracle';
import {createGWRAuditScene, GWR_AUDIT_CASES} from './gwr-mathematical-audit-fixtures';

for (const fixture of GWR_AUDIT_CASES) {
  it(`matches independent raw-design SVD: ${fixture.kernel}/${fixture.bandwidthMode}/shift=${fixture.transformed}/golden=${'golden' in fixture}/refined=${'refinementPassCount' in fixture}`, () => {
    const actual = computeGeographicallyWeightedRegressionOnCPU({
      ...createGWRAuditScene(fixture.transformed),
      settings: {
        kernel: fixture.kernel,
        bandwidthMode: fixture.bandwidthMode,
        bandwidths: fixture.bandwidths
      },
      maximumBandwidthCount: 3,
      bandwidthSearch: 'golden' in fixture ? 'golden-section' : 'ladder',
      maximumSearchIterations: 1,
      refinementPassCount: 'refinementPassCount' in fixture ? fixture.refinementPassCount : 0
    });
    const ports = {
      coefficients: actual.coefficients,
      standardErrors: actual.standardErrors,
      tValues: actual.tValues,
      significant: actual.significant,
      localR2: actual.localR2,
      fitted: actual.fitted,
      residuals: actual.residuals,
      hatDiagonal: actual.hatDiagonal,
      localStatus: actual.localStatus,
      bandwidthScores: actual.bandwidthScores,
      candidateTable: actual.candidates.flatMap(candidate => [
        candidate.value,
        candidate.aicc,
        candidate.singularRowCount,
        candidate.residualSumOfSquares,
        candidate.traceOfHat,
        candidate.observationCount,
        candidate.includedRowCount
      ]),
      selectedBandwidth: [actual.selectedIndex, actual.selectedValue],
      summary: [
        actual.residualSumOfSquares,
        actual.traceOfHat,
        actual.aicc,
        actual.rSquared,
        actual.observationCount,
        Number(actual.hasValidCandidate),
        actual.singularRowCount,
        actual.sigmaSquared,
        actual.adjustedAlpha,
        actual.criticalT,
        actual.logLikelihood,
        actual.includedRowCount
      ]
    };
    const differences: Record<string, number> = {};
    for (const [port, values] of Object.entries(ports)) {
      const expected = fixture.expected[port as keyof typeof ports].slice(
        0,
        port === 'bandwidthScores' ? values.length : undefined
      );
      expect(values.length, port).toBe(expected.length);
      differences[port] = 0;
      for (let index = 0; index < values.length; index++) {
        if (Number.isNaN(expected[index])) expect(values[index], `${port}[${index}]`).toBeNaN();
        else {
          expect(Number.isFinite(values[index]), `${port}[${index}]`).toBe(true);
          differences[port] = Math.max(
            differences[port],
            Math.abs(values[index] - expected[index])
          );
          // Float64 Cholesky versus independent SVD; t quantiles use a gamma-ratio series.
          expect(Math.abs(values[index] - expected[index]), `${port}[${index}]`).toBeLessThan(1e-8);
        }
      }
    }
    console.info('GWR audit maximum absolute differences', differences);
  });
}

it('counts a distant rank-one island without changing the eight independent local fits', () => {
  const fixture = GWR_AUDIT_CASES[1];
  const scene = createGWRAuditScene();
  scene.mask[8] = 1;
  const actual = computeGeographicallyWeightedRegressionOnCPU({
    ...scene,
    settings: {kernel: 'bisquare', bandwidths: [3, 4]},
    maximumBandwidthCount: 3
  });
  expect(actual.localStatus).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 1]);
  expect(actual.candidates.map(candidate => candidate.singularRowCount)).toEqual([1, 1, NaN]);
  expect(actual.includedRowCount).toBe(9);
  expect(actual.observationCount).toBe(8);
  expect(actual.singularRowCount).toBe(1);
  expect(Math.abs(actual.aicc - fixture.expected.summary[2])).toBeLessThan(1e-10);
});
