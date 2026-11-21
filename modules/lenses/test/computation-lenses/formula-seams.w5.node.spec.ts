// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, test} from 'vitest';
import {DYNAMICS_OPERATORS} from '../../src/gpu-spatial-analysis/computation-pack/operators/dynamics';
import {STATISTICS_SUMMARY_OPERATORS} from '../../src/gpu-spatial-analysis/computation-pack/operators/statistics-summaries';
import {STATISTICS_CLUSTERING_OPERATORS} from '../../src/gpu-spatial-analysis/computation-pack/operators/statistics-clustering';
import {expectFormulaBindingsExist} from '../gpu-spatial-analysis/computation-pack/formula-bindings';

test.each([
  [
    DYNAMICS_OPERATORS['GPUMantelTest'],
    [
      'modules/analysis/src/gpu-spatial-analysis/space-time-tests/gpu-mantel-test.ts',
      'modules/analysis/src/gpu-spatial-analysis/space-time-tests/mantel-inference-publication.ts'
    ]
  ],
  [
    STATISTICS_SUMMARY_OPERATORS['GPUNeighborhoodSummary'],
    ['modules/analysis/src/gpu-spatial-analysis/neighborhood-summary/gpu-neighborhood-summary.ts']
  ],
  [
    STATISTICS_CLUSTERING_OPERATORS['GPUSpatialKMeans'],
    ['modules/analysis/src/gpu-spatial-analysis/spatial-clustering/gpu-spatial-kmeans.ts']
  ]
] as const)('corrected formula %# retains exact kernel-symbol bindings', (entry, sources) => {
  if ('formula' in entry.formula) throw new Error('Expected a curated formula');
  expectFormulaBindingsExist(entry.formula, [...sources]);
});

test('W5-25: both Mantel formula forms condition inference on usable null draws', () => {
  const formula = DYNAMICS_OPERATORS['GPUMantelTest'].formula;
  if ('formula' in formula) throw new Error('Mantel requires a curated formula');
  expect(formula.ascii).toContain('N < 2 or either norm is zero');
  expect(formula.ascii).toContain('M usable null draws');
  expect(formula.ascii).toContain('(M + 1)');
  expect(formula.ascii).toContain('M = 0');
  expect(formula.ascii).toContain('observed pair count N is retained');
  expect(formula.tex).toContain('N<2');
  expect(formula.tex).toContain('M+1');
  expect(formula.tex).toContain('M=0');
  expect(formula.tex).toContain(String.raw`U&=\{s\in\{1,\ldots,A\}:r_s\text{ finite}\}`);
  expect(formula.tex).toContain(String.raw`A&=\text{attempted permutationCount}`);
  expect(formula.ascii).toContain('slots 1 through attempted permutationCount inclusive');
  expect(formula.ascii).toContain('By default, x = spatial distance, y = |t_i - t_j|');
  expect(formula.ascii).toContain('distanceTransform and timeTransform');
});

test('MRG-01: neighborhood formulas separate numeric and categorical eligibility and loss', () => {
  const formula = STATISTICS_SUMMARY_OPERATORS['GPUNeighborhoodSummary'].formula;
  if ('formula' in formula) throw new Error('Neighborhood summary requires a curated formula');
  expect(formula.tex).toContain('W_i^{\\mathrm{num}}');
  expect(formula.tex).toContain('W_i^{\\mathrm{cat}}');
  expect(formula.tex).toContain('c_s\\ne 0xffffffff');
  expect(formula.ascii).toContain('finite numeric value');
  expect(formula.ascii).toContain('category other than 0xffffffff');
  expect(formula.ascii).toContain('candidateOverflow');
  expect(formula.ascii).toContain('iterationLimitReached');
  expect(formula.ascii).toContain('converged');
  expect(formula.ascii).toContain('count too');
  expect(formula.ascii).toContain('overflow and candidateOverflow at matrix and chunk level');
  expect(formula.ascii).toContain('iterationLimitReached and converged at matrix level only');
  expect(formula.ascii).toContain(
    'The GPU rejects complete, requiredCount and chunk-local iteration/convergence certificates at construction'
  );
});

test('W5-26: scenario status meanings include geometry, decision and RTS fallback causes', () => {
  const readScenario = (name: string) =>
    readFileSync(
      new URL(
        `../../src/computation-lenses/scenarios/gpu-spatial-analysis/${name}.lens.ts`,
        import.meta.url
      ),
      'utf8'
    );
  const track = readScenario('track-similarity');
  expect(track).toContain('8 invalidGeometry');
  expect(track).toContain('16 invalidDecisionThreshold');
  expect(track).toContain('distance validity');
  expect(track).toContain('decision validity');
  const trajectory = readScenario('trajectory-coverage');
  expect(trajectory).toContain('8 RTS fallback');
  expect(trajectory).toContain('singular prediction');
  expect(trajectory).toContain('nonfinite backward state/covariance correction');
});
