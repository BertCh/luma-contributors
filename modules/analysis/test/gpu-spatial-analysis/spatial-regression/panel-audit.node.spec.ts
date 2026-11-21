// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeSpatialPanelLagMaximumLikelihoodOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-lag-maximum-likelihood-operation';
import {
  createPanelAuditInput,
  expectPanelAuditRows,
  getPanelAuditExpected,
  PANEL_AUDIT_EXPECTED
} from './panel-audit.test-utils';

for (const effects of ['unit', 'random'] as const) {
  for (const shifted of [false, true]) {
    it(`panel audit independent likelihood/information, ${effects}, affine=${shifted}`, () => {
      const expected = getPanelAuditExpected(effects, shifted);
      const result = computeSpatialPanelLagMaximumLikelihoodOnCPU(createPanelAuditInput(shifted), {
        periodCount: 4,
        predictorCount: 1,
        effects,
        hausman: effects === 'random',
        tolerance: 1e-10
      });
      expect(Array.from(result.status)).toEqual([0]);
      // Binary64 optimization followed by one float32 rounding; this bound includes SciPy's
      // independently profiled phi and the resulting prediction sensitivity to the mean.
      for (const port of ['coefficients', 'covariance', 'fitted', 'residuals'] as const)
        expectPanelAuditRows(result[port], expected[port], 3e-6);
      expectPanelAuditRows(result.summary.subarray(0, 2), expected.summary, 3e-6);
      expect(result.summary[2]).toBeGreaterThanOrEqual(1);
      expect(result.summary[3]).toBeLessThanOrEqual(1e-10);
      expect(result.summary[4]).toBeLessThanOrEqual(16 * Number.EPSILON);
      expect(result.summary[5]).toBe(0);
      if (effects === 'random') {
        expect(Array.from(result.hausman!.status)).toEqual([0]);
        expectPanelAuditRows(result.hausman!.tests, PANEL_AUDIT_EXPECTED.random.hausman, 3e-6);
      }
    });
  }
  it(`panel audit rejects an island without publishing a partial ${effects} fit`, () => {
    const result = computeSpatialPanelLagMaximumLikelihoodOnCPU(
      createPanelAuditInput(false, true),
      {
        periodCount: 4,
        predictorCount: 1,
        effects,
        hausman: effects === 'random'
      }
    );
    expect(Array.from(result.status)).toEqual([1]);
    for (const port of ['coefficients', 'covariance', 'summary', 'fitted', 'residuals'] as const)
      expect(Array.from(result[port]).every(Number.isNaN)).toBe(true);
    if (result.hausman) {
      expect(Array.from(result.hausman.status)).toEqual([1]);
      expect(Array.from(result.hausman.tests).every(Number.isNaN)).toBe(true);
    }
  });
}
