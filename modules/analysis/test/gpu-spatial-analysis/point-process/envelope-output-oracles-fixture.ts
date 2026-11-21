// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import type {PointPatternEnvelopeOperationProps} from '../../../src/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope-operation';
import type {OperationGraphRows} from '../../utils/mixed-execution-test-utils';

export const ENVELOPE_INPUTS = {
  observed: {format: 'float32' as const, values: new Float32Array([2, 1, -1])},
  simulations: {
    format: 'float32' as const,
    values: new Float32Array([0, 0, 1, 1, 2, -2, 3, 1, 0, 2, 3, -1])
  },
  reference: {format: 'float32' as const, values: new Float32Array([0.25, 0.5, -0.25])},
  weights: {format: 'float32' as const, values: new Float32Array([1, 2, 0.5])}
};

export function getEnvelopeProps(
  statistic: PointPatternEnvelopeOperationProps['statistic'],
  withInterval = false
): PointPatternEnvelopeOperationProps {
  return {
    observed: 'observed',
    simulations: 'simulations',
    reference: 'reference',
    integrationWeights: 'weights',
    curveRowCount: 3,
    simulationCount: 4,
    nullModel: 'fixed external null',
    fittedParameters: 'fixed-external',
    statistic,
    rank: 2,
    significanceLevel: 0.4,
    output: {
      lower: 'lower',
      upper: 'upper',
      statistics: 'statistics',
      status: 'status',
      ...(withInterval ? {pValueInterval: 'interval'} : {})
    }
  };
}

export function getTiedEnvelopeProps(withInterval: boolean): PointPatternEnvelopeOperationProps {
  return {
    ...getEnvelopeProps('extreme-rank', withInterval),
    curveRowCount: 1,
    simulationCount: 5,
    rank: 1,
    significanceLevel: 0.34
  };
}
export const TIED_ENVELOPE_INPUTS = {
  observed: {format: 'float32' as const, values: new Float32Array([0])},
  simulations: {format: 'float32' as const, values: new Float32Array([0, 1, 2, 2, 2])},
  reference: {format: 'float32' as const, values: new Float32Array([0])},
  weights: {format: 'float32' as const, values: new Float32Array([1])}
};
export function expectTiedEnvelopeOracle(
  result: Record<string, OperationGraphRows>,
  withInterval: boolean
): void {
  expect(Object.keys(result).sort()).toEqual(
    ['lower', 'upper', 'statistics', 'status', ...(withInterval ? ['interval'] : [])].sort()
  );
  // Six exchangeable values: two 0s have rank 1.5, 1 has rank 3, three 2s rank 2.
  // The alpha budget is floor(.34*6)=2: critical rank 2 rejects both zeros together.
  expect(Array.from(result['lower'])).toEqual([1]);
  expect(Array.from(result['upper'])).toEqual([2]);
  expect(Array.from(result['statistics'])).toEqual([
    1.5,
    Math.fround(1 / 3),
    Math.fround(1 / 6),
    5
  ]);
  expect(Array.from(result['status'])).toEqual([0]);
  if (result['interval']) expect(Array.from(result['interval'])).toEqual([0, Math.fround(1 / 3)]);
}

export function getHalfRankEnvelopeCase(defaultLevel = false) {
  const simulations = defaultLevel
    ? [...Array.from({length: 19}, (_, index) => index + 1), 19]
    : [1, 2, 3, 3];
  const props = {
    ...getEnvelopeProps('extreme-rank', true),
    curveRowCount: 1,
    simulationCount: simulations.length,
    rank: 1,
    significanceLevel: defaultLevel ? 0.05 : 0.21
  };
  const inputs = {
    ...TIED_ENVELOPE_INPUTS,
    simulations: {format: 'float32' as const, values: new Float32Array(simulations)}
  };
  return {props, inputs};
}

export function expectHalfRankEnvelopeOracle(
  result: Record<string, OperationGraphRows>,
  defaultLevel = false
): void {
  // SciPy midranks: [1,2,3,1.5,1.5] (including observation first). Only rank1 is rejected.
  // At default .05 with 21 curves the same rank1 versus rank1.5 separation rejects just zero.
  expect(Object.keys(result).sort()).toEqual(
    ['lower', 'upper', 'statistics', 'status', 'interval'].sort()
  );
  expect(Array.from(result['lower'])).toEqual([1]);
  expect(Array.from(result['upper'])).toEqual([defaultLevel ? 19 : 3]);
  expect(Array.from(result['interval'])).toEqual([0, Math.fround(1 / (defaultLevel ? 21 : 5))]);
  expect(Array.from(result['status'])).toEqual([0]);
}

export function getHierarchicalRankEnvelopeCase(invalid = false) {
  const {props, inputs} = getHalfRankEnvelopeCase();
  const simulations = Float32Array.from([
    ...Array.from({length: 298}, (_, index) => index + 1),
    298
  ]);
  if (invalid) simulations[270] = NaN;
  return {
    props: {...props, simulationCount: 299, significanceLevel: 0.004},
    inputs: {...inputs, simulations: {format: 'float32' as const, values: simulations}}
  };
}

export function expectHierarchicalRankEnvelopeOracle(
  result: Record<string, OperationGraphRows>,
  invalid = false
): void {
  // 300 curves: observed0 has rank1; the two maximal298s have midrank1.5.
  // floor(.004*300)=1 rejects only observation, so retained bounds are [1,298].
  // The observed curve and optional invalid curve lie beyond the first 256-row reduction level.
  expect(Object.keys(result).sort()).toEqual([
    'interval',
    'lower',
    'statistics',
    'status',
    'upper'
  ]);
  expect(Array.from(result['status'])).toEqual([invalid ? 1 : 0]);
  if (invalid) {
    for (const name of ['lower', 'upper', 'interval'])
      expect(Array.from(result[name]).every(Number.isNaN)).toBe(true);
    expect(Number.isNaN(result['statistics'][0])).toBe(true);
    expect(Number.isNaN(result['statistics'][1])).toBe(true);
  } else {
    expect(Array.from(result['lower'])).toEqual([1]);
    expect(Array.from(result['upper'])).toEqual([298]);
    expect(Array.from(result['interval'])).toEqual([0, Math.fround(1 / 300)]);
    expect(Array.from(result['statistics'].slice(0, 2))).toEqual([1, Math.fround(1 / 300)]);
  }
  expect(Array.from(result['statistics'].slice(2))).toEqual([Math.fround(1 / 300), 299]);
}

export function expectEnvelopeOracle(
  result: Record<string, OperationGraphRows>,
  statistic: PointPatternEnvelopeOperationProps['statistic'],
  invalid = false,
  withInterval = false
): void {
  expect(Object.keys(result).sort()).toEqual(
    ['lower', 'upper', 'statistics', 'status', ...(withInterval ? ['interval'] : [])].sort()
  );
  expect(Array.from(result['status'])).toEqual([invalid ? 1 : 0]);
  if (invalid) {
    for (const name of ['lower', 'upper', ...(result['interval'] ? ['interval'] : [])])
      for (const value of result[name]) expect(Number.isNaN(value)).toBe(true);
    expect(Number.isNaN(result['statistics'][0])).toBe(true);
    expect(Number.isNaN(result['statistics'][1])).toBe(true);
    expect(Array.from(result['statistics'].slice(2))).toEqual([Math.fround(0.2), 4]);
    return;
  }
  // Hand-sorted simulation columns: [0,1,2,3], [0,1,2,3], [-2,-1,0,1].
  // DCLF weighted square sums are [1.34375,6.59375,8.09375,15.84375], observed 3.84375.
  // Max deviations are [1.25,1.75,2.75,2.5], observed 1.75: three exceed/tie in both modes.
  // Including observed, extreme midranks are [1,1,1,1,2.5]. Thus strict=4/5, inclusive=5/5;
  // alpha=.4 allows only two rejections, critical rank=1, so the envelope uses all column extrema.
  const ranked = statistic === 'extreme-rank';
  expect(Array.from(result['lower'])).toEqual(ranked ? [0, 0, -2] : [1, 1, -1]);
  expect(Array.from(result['upper'])).toEqual(ranked ? [3, 3, 1] : [2, 2, 0]);
  const expected = [
    ranked ? 2.5 : statistic === 'integrated-squared-deviation' ? 3.84375 : 1.75,
    ranked ? 1 : 0.8,
    0.2,
    4
  ];
  for (const [index, value] of expected.entries()) {
    expect(Number.isFinite(result['statistics'][index])).toBe(true);
    // Small exactly representable deviations; 2e-6 absolute permits float32 p-value division.
    expect(Math.abs(result['statistics'][index] - value)).toBeLessThanOrEqual(2e-6);
  }
  if (result['interval']) expect(Array.from(result['interval'])).toEqual([Math.fround(0.8), 1]);
}
