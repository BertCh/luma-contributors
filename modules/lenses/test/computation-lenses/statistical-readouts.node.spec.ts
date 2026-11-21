// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {createGPURegressionModelResult} from '@luma-contributors/analysis/analysis-core';
import {decodeGPUAnalysisContiguityStatistics} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  computeOrdinaryLeastSquaresOnCPU,
  getGPUOrdinaryLeastSquaresModelResult,
  getGPUOrdinaryLeastSquaresParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  addModelResultReadouts,
  countSingularGWRRows
} from '../../../../examples/deck/spatial-analysis-explorer/modes/regression-result-readouts';
import {PAIR_STATISTICS_OPERATORS} from '../../src/gpu-dataframe/computation-pack/operators/pair-statistics';
import {GROUPS_JOINS_OPERATORS} from '../../src/gpu-dataframe/computation-pack/operators/groups-joins';
import {getQuantileLisaPValueLabel} from '../../../../examples/deck/kepler-compute/src/analysis-quantile-lisa-inference';
import {getLensScenario} from '../../src/computation-lenses/registry-implementation';
import {declareLensScenario} from './scenario-declaration.test-utils';
import '../../src/computation-lenses/scenarios/gpu-spatial-analysis/quantile-lisa.lens';
import '../../src/computation-lenses/scenarios/gpu-dataframe/ripley-distance-functions.lens';

test('regression readout callback replaces ridge evidence with each subsequent fit', () => {
  const values = new Map<string, string>();
  const controls = {
    addReadout: (label: string) => ({setValue: (value: string) => values.set(label, value)})
  };
  const readouts = addModelResultReadouts(controls, 'OLS', ['slope']);
  const input = {
    kind: 'ordinary-least-squares' as const,
    coefficients: [
      {
        name: 'slope',
        role: 'predictor' as const,
        estimate: 1.1,
        standardError: 0.6264982,
        statistic: 1.7558,
        pValue: 0.2212084,
        statisticKind: 't' as const
      }
    ],
    rowCount: 4,
    residualSumOfSquares: 2,
    standardErrorSigmaSquared: 'degrees-of-freedom' as const,
    statusCode: 0
  };
  readouts.update(
    createGPURegressionModelResult({
      ...input,
      regularization: {kind: 'ridge', lambda: 5},
      inferenceMethod: 'ridge-approximate'
    })
  );
  expect(values.get('OLS regularization')).toBe('ridge, λ 5.000');
  expect(values.get('OLS coefficient inference')).toBe('ridge-approximate');
  readouts.update(
    createGPURegressionModelResult({...input, inferenceMethod: 'classical-homoskedastic'})
  );
  expect(values.get('OLS regularization')).toBe('not reported');
  expect(values.get('OLS coefficient inference')).toBe('classical-homoskedastic');
  readouts.update(createGPURegressionModelResult(input));
  expect(values.get('OLS coefficient inference')).toBe('estimator convention (not reported)');
});

test('actual GWR row summary counts singular only, preserving OK, excluded and unknown codes', () => {
  // Producer codes: OK=0, SINGULAR=1, EXCLUDED=2; 99 is deliberately unknown.
  expect(countSingularGWRRows(Uint32Array.of(0, 1, 2, 99, 1))).toBe(2);
  expect(countSingularGWRRows(Uint32Array.of(0, 2, 99))).toBe(0);
});

test('Ripley metadata names both Hanisch denominator estimators and the rectangle condition', () => {
  const entry = PAIR_STATISTICS_OPERATORS['GPURipleyDistanceFunctions'];
  expect(entry.choices).toContain('hanischSearch');
  expect(entry.metadata?.['hanischSearchChoices']).toBe('capped, exact');
  expect(entry.metadata?.['hanischSearchExactCondition']).toBe('studyWindow: rectangle');
  expect(entry.metadata?.['hanischSearchCapped']).toContain('maximumDistance');
  expect(entry.metadata?.['hanischSearchExact']).toContain('boundary distance');
});

test('group moment formula distinguishes adaptive mean from caller-scale sum', () => {
  const formula = GROUPS_JOINS_OPERATORS['GPUGroupStatistics'].formula;
  if (!('ascii' in formula)) throw new Error('group statistics requires its moment formula');
  expect(formula.ascii).toContain('adaptive 96-bit total divided by the exact integer count');
  expect(formula.ascii).toContain('caller-scale sum is independent');
  expect(formula.ascii).toContain('M2=max(sum(d*d)-sum(d)^2/n,0)');
});

test('contiguity publication refuses independent final and candidate overflow and recovers', () => {
  // Three squares: one adjacent pair and one island, hence degrees [1,1,0], median=1.
  const values = {
    counts: Uint32Array.of(2, 0, 1, 0, 1, 1),
    median: Float32Array.of(1),
    validity: Uint32Array.of(1),
    weights: {status: {overflow: Uint32Array.of(0), candidateOverflow: Uint32Array.of(0)}}
  };
  const expected = {
    exact: true,
    overflow: false,
    candidateOverflow: false,
    slots: 2,
    islands: 1,
    minimum: 0,
    median: 1,
    maximum: 1
  };
  expect(decodeGPUAnalysisContiguityStatistics(values)).toEqual(expected);
  values.weights.status.overflow[0] = 1;
  expect(decodeGPUAnalysisContiguityStatistics(values)).toEqual({
    exact: false,
    overflow: true,
    candidateOverflow: false
  });
  values.weights.status.overflow[0] = 0;
  values.weights.status.candidateOverflow[0] = 1;
  expect(decodeGPUAnalysisContiguityStatistics(values)).toEqual({
    exact: false,
    overflow: false,
    candidateOverflow: true
  });
  values.weights.status.candidateOverflow[0] = 0;
  expect(decodeGPUAnalysisContiguityStatistics(values)).toEqual(expected);
});

test('numeric Quantile LISA labels identify the selected tail', () => {
  expect(getQuantileLisaPValueLabel('greater')).toBe('Quantile LISA p-value (greater)');
  expect(getQuantileLisaPValueLabel('two-sided')).toBe('Quantile LISA p-value (two-sided)');
  expect(getQuantileLisaPValueLabel('geoda-legacy')).toBe('Legacy pseudo p-value (geoda-legacy)');
});

test('assembled Quantile LISA record displays the configured alternative', async () => {
  const declaration = await declareLensScenario(
    getLensScenario('gpu-spatial-analysis.quantile-lisa')!
  );
  expect(declaration.record.choices.find(choice => choice.name === 'alternative')).toMatchObject({
    value: 'geoda-legacy',
    alternatives: ['greater', 'two-sided', 'geoda-legacy']
  });
});

test('actual packed OLS reader evidence reaches the same readout callback for ridge 2 then 0', () => {
  const values = new Map<string, string>();
  const readouts = addModelResultReadouts(
    {
      addReadout: label => ({
        setValue: value => {
          values.set(label, value);
        }
      })
    },
    'OLS',
    ['intercept', 'slope']
  );
  const result = (ridgeLambda: number) =>
    getGPUOrdinaryLeastSquaresModelResult(
      computeOrdinaryLeastSquaresOnCPU(
        {
          predictors: Float32Array.of(0, 1, 2, 3),
          response: Float32Array.of(1, 2, 4, 8),
          parameters: getGPUOrdinaryLeastSquaresParameterValues({ridgeLambda})
        },
        1
      )
    );
  const penalized = result(2);
  readouts.update(penalized);
  const penalizedLabel = values.get('OLS regularization');
  const ordinary = result(0);
  readouts.update(ordinary);
  expect(penalizedLabel).toBe('ridge, λ 2.000');
  expect(typeof penalized.inferenceMethod).toBe('string');
  expect(values.get('OLS regularization')).not.toBe('ridge, λ 2.000');
  expect(typeof ordinary.inferenceMethod).toBe('string');
  expect(ordinary.inferenceMethod).not.toBe(penalized.inferenceMethod);
});

test('assembled Ripley record identifies its configured Hanisch search and alternatives', async () => {
  const declaration = await declareLensScenario(
    getLensScenario('gpu-dataframe.ripley-distance-functions')!
  );
  expect(declaration.record.choices.find(choice => choice.name === 'hanischSearch')).toMatchObject({
    value: 'capped',
    alternatives: ['capped', 'exact']
  });
  expect(declaration.missingChoices).not.toContain('hanischSearch');
});
