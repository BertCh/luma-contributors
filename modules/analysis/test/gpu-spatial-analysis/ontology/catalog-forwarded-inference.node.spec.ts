// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getGPUAnalysisCautionText,
  getGPUAnalysisSignificanceAttainability
} from '../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-interpretation';

it('uses the actual BH rank boundary rather than subtracting a tolerance', () => {
  // At P=99 the floor is .01. For m=12, .01*12/3=.04 exceeds alpha,
  // whereas .01*12/4=.03 is below it: three floor ties reject none; four suffice.
  const result = getGPUAnalysisSignificanceAttainability({
    correction: 'false-discovery-rate',
    permutationCount: 99,
    testCount: 12,
    significanceLevel: 0.039999999999
  });
  expect(result.attainable).toBe(true);
  expect(result.testsNeededAtMinimum).toBe(4);
  expect(result.reason).toContain('4 tests tied at the minimum would suffice');
});

it('does not promise a BH rejection when even its final rank exceeds the cutoff', () => {
  const result = getGPUAnalysisSignificanceAttainability({
    correction: 'false-discovery-rate',
    permutationCount: 99,
    testCount: 12,
    significanceLevel: 0.009999999999999998
  });
  expect(result.attainable).toBe(false);
  expect(result.testsNeededAtMinimum).toBeUndefined();
});

it('describes the doubled-tail floor at both attainable and unattainable cutoffs', () => {
  // Nineteen draws: doubled add-one minimum is 2/(19+1)=.10, not .05.
  for (const significanceLevel of [0.025, 0.1]) {
    const caution = getGPUAnalysisCautionText('pseudo-p-values', {
      permutationCount: 19,
      pValueConstruction: 'doubled-minimum-tail',
      significanceLevel
    });
    expect(caution.text).toContain('min(1, 2 / (permutations + 1))');
  }
  expect(getGPUAnalysisCautionText('pseudo-p-values').text).toContain(
    'smaller than 1 / (permutations + 1)'
  );
  expect(
    getGPUAnalysisCautionText('pseudo-p-values', {pValueConstruction: 'upper-tail-count'}).text
  ).toContain('smaller than 1 / (permutations + 1)');
});
