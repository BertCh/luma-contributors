// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computePointPatternEnvelopeOnCPU,
  CPUPointPatternEnvelopeOperation
} from '../../../src/gpu-spatial-analysis/point-process/cpu-point-pattern-envelope-operation';
import {
  GPUPointPatternEnvelopeOperation,
  type PointPatternEnvelopeOperationProps
} from '../../../src/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';

const PROPS: PointPatternEnvelopeOperationProps = {
  id: 'envelope-test',
  observed: 'observed',
  simulations: 'simulations',
  reference: 'reference',
  integrationWeights: 'weights',
  curveRowCount: 2,
  simulationCount: 4,
  nullModel: 'external exchangeable curves',
  fittedParameters: 'none',
  output: {lower: 'lower', upper: 'upper', statistics: 'statistics', status: 'status'}
};
const INPUT = {
  observed: Float32Array.of(2, 1),
  simulations: Float32Array.of(0, 0, 1, 2, 3, 1, 2, 3),
  reference: Float32Array.of(0, 0),
  integrationWeights: Float32Array.of(1, 2)
};

it('matches hand-computed weighted DCLF and maximum-deviation tests with moderate p-values', () => {
  // DCLF statistics: [0,9,11,22], observed 6. Maximum deviations: [0,2,3,3], observed 2.
  const integrated = computePointPatternEnvelopeOnCPU(INPUT, {
    ...PROPS,
    statistic: 'integrated-squared-deviation'
  });
  expect([...integrated.statistics]).toEqual([6, Math.fround(4 / 5), Math.fround(1 / 5), 4]);
  expect([...integrated.lower]).toEqual([0, 0]);
  expect([...integrated.upper]).toEqual([3, 3]);
  expect([...integrated.status]).toEqual([0]);
  const maximum = computePointPatternEnvelopeOnCPU(INPUT, PROPS);
  expect([...maximum.statistics]).toEqual([2, Math.fround(4 / 5), Math.fround(1 / 5), 4]);
  const inner = computePointPatternEnvelopeOnCPU(INPUT, {...PROPS, rank: 2});
  expect([...inner.lower]).toEqual([1, 1]);
  expect([...inner.upper]).toEqual([2, 2]);
});

it('matches independent pairwise midranks including ties and the observed curve', () => {
  // This oracle counts comparisons; the implementation sorts tied runs.
  const curves = [
    [0, 0],
    [1, 2],
    [3, 1],
    [2, 3],
    [2, 1]
  ];
  const ranks = curves.map(curve =>
    Math.min(
      ...curve.map((value, column) => {
        const less = curves.filter(other => other[column] < value).length;
        const greater = curves.filter(other => other[column] > value).length;
        const equal = curves.filter(other => other[column] === value).length;
        return Math.min(less, greater) + (equal + 1) / 2;
      })
    )
  );
  expect(ranks).toEqual([1, 2, 1, 1, 2.5]);
  const result = computePointPatternEnvelopeOnCPU(INPUT, {
    ...PROPS,
    statistic: 'extreme-rank',
    significanceLevel: 0.8
  });
  expect([...result.statistics]).toEqual([2.5, 1, Math.fround(1 / 5), 4]);
  expect([...result.pValueInterval]).toEqual([Math.fround(4 / 5), 1]);
  // At alpha=.8, rank 2 rejects three curves; rank 3 rejects all five and is inadmissible.
  expect([...result.lower]).toEqual([1, 1]);
  expect([...result.upper]).toEqual([2, 2]);
});

it('invalidates all inference when any curve is nonfinite or any deviation weight is nonpositive', () => {
  for (const input of [
    {...INPUT, integrationWeights: Float32Array.of(1, 0)},
    {...INPUT, simulations: Float32Array.of(NaN, 0, 1, 2, 3, 1, 2, 3)}
  ]) {
    const result = computePointPatternEnvelopeOnCPU(input, PROPS);
    expect([...result.status]).toEqual([1]);
    expect(result.lower.every(Number.isNaN)).toBe(true);
    expect(result.upper.every(Number.isNaN)).toBe(true);
    expect([...result.statistics].slice(0, 2).every(Number.isNaN)).toBe(true);
  }
  expect(
    computePointPatternEnvelopeOnCPU(
      {...INPUT, integrationWeights: Float32Array.of(0, NaN)},
      {...PROPS, statistic: 'extreme-rank'}
    ).status[0]
  ).toBe(0);
});

it('validates arrays, CPU construction and deferred GPU contributor construction', () => {
  expect(() =>
    computePointPatternEnvelopeOnCPU({...INPUT, observed: new Float32Array(0)}, PROPS)
  ).toThrow(/observed/);
  expect(() =>
    computePointPatternEnvelopeOnCPU({...INPUT, simulations: new Float32Array(2)}, PROPS)
  ).toThrow(/simulations/);
  expect(() =>
    computePointPatternEnvelopeOnCPU({...INPUT, reference: new Float32Array(1)}, PROPS)
  ).toThrow(/reference/);
  expect(() =>
    computePointPatternEnvelopeOnCPU({...INPUT, integrationWeights: new Float32Array(1)}, PROPS)
  ).toThrow(/integrationWeights/);
  expect(() => new CPUPointPatternEnvelopeOperation({...PROPS, rank: 3})).toThrow(
    /envelope-test.*rank/
  );
  expect(() => new GPUPointPatternEnvelopeOperation({...PROPS, curveRowCount: 0})).toThrow(
    /envelope-test.*length/
  );
  const inputs = {
    observed: {format: 'float32' as const, length: 2},
    simulations: {format: 'float32' as const, length: 8},
    reference: {format: 'float32' as const, length: 2},
    integrationWeights: {format: 'float32' as const, length: 2}
  };
  expect(() =>
    constructNetworkOperationContributor(
      new GPUPointPatternEnvelopeOperation({...PROPS, rank: 3}),
      inputs
    )
  ).toThrow(/envelope-test.*rank/);
  expect(() =>
    constructNetworkOperationContributor(
      new GPUPointPatternEnvelopeOperation({...PROPS, nullModel: ''}),
      inputs
    )
  ).toThrow(/envelope-test.*nullModel/);
});
