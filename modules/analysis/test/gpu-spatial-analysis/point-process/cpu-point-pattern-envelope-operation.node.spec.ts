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
import {roundPointPatternDyadicStatistic} from '../../../src/gpu-spatial-analysis/point-process/point-pattern-exact-statistic-cpu';

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
  // Independent ranks above: cutoff 2.5 rejects four of five curves (budget .8*5=4).
  // Only the observed [2,1] remains; an integer-only cutoff incorrectly retains [1,2].
  expect([...result.lower]).toEqual([2, 1]);
  expect([...result.upper]).toEqual([2, 1]);
});

it('invalidates all inference when any curve is nonfinite or any deviation weight is nonpositive', () => {
  for (const input of [
    {...INPUT, integrationWeights: Float32Array.of(1, 0)},
    {...INPUT, simulations: Float32Array.of(NaN, 0, 1, 2, 3, 1, 2, 3)}
  ]) {
    const result = computePointPatternEnvelopeOnCPU(input, {
      ...PROPS,
      statistic: 'integrated-squared-deviation'
    });
    expect([...result.status]).toEqual([1]);
    expect([...result.lower]).toEqual([NaN, NaN]);
    expect([...result.upper]).toEqual([NaN, NaN]);
    expect([...result.statistics]).toEqual([NaN, NaN, Math.fround(1 / 5), 4]);
    expect([...result.pValueInterval]).toEqual([NaN, NaN]);
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
    computePointPatternEnvelopeOnCPU(
      {...INPUT, integrationWeights: new Float32Array(1)},
      {...PROPS, statistic: 'integrated-squared-deviation'}
    )
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

it('TWN-41 and TWN-03 compare integrated statistics before rounding their displayed value', () => {
  // Exact statistics: observed 1 + 2^-26 (or 2^-54) > nineteen simulated values 1.
  // Thus zero exceedances and p=(0+1)/(19+1), even though displayed statistic rounds to 1.
  for (const exponent of [-13, -27]) {
    const result = computePointPatternEnvelopeOnCPU(
      {
        observed: Float32Array.of(1, 2 ** exponent),
        simulations: Float32Array.from({length: 38}, (_, index) => (index % 2 ? 0 : 1)),
        reference: Float32Array.of(0, 0),
        integrationWeights: Float32Array.of(1, 1)
      },
      {...PROPS, simulationCount: 19, statistic: 'integrated-squared-deviation'}
    );
    expect([...result.statistics]).toEqual([1, Math.fround(1 / 20), Math.fround(1 / 20), 19]);
    expect([...result.status]).toEqual([0]);
  }
});

it('TWN-03 retains exact integrated ties despite different ordinate order', () => {
  // Both curves have exactly 1 + 2^-54; all nineteen draws tie inclusively.
  const result = computePointPatternEnvelopeOnCPU(
    {
      observed: Float32Array.of(1, 2 ** -27),
      simulations: Float32Array.from({length: 38}, (_, index) => (index % 2 ? 1 : 2 ** -27)),
      reference: Float32Array.of(0, 0),
      integrationWeights: Float32Array.of(1, 1)
    },
    {...PROPS, simulationCount: 19, statistic: 'integrated-squared-deviation'}
  );
  expect([...result.statistics]).toEqual([1, 1, Math.fround(1 / 20), 19]);
  expect([...result.status]).toEqual([0]);
});

it('B2 pure CPU captures inherited significanceLevel once before resolving defaults', () => {
  let reads = 0;
  const options = {...PROPS, simulationCount: 39, statistic: 'extreme-rank' as const};
  Object.setPrototypeOf(options, {
    get significanceLevel() {
      reads++;
      return 0.01;
    }
  });
  const result = computePointPatternEnvelopeOnCPU(
    {
      observed: Float32Array.of(0),
      simulations: Float32Array.from({length: 39}, (_, index) => index + 1)
    },
    options
  );
  // The two most extreme curves cost 2/40, exceeding 0.01, so retain both endpoints.
  expect([...result.lower]).toEqual([0]);
  expect([...result.upper]).toEqual([39]);
  expect(reads).toBe(1);
});

it('B3 direct dyadic rounding preserves subnormal ties to even', () => {
  // Coefficient quantum is 2^-447; half the least subnormal is 2^-150 = 2^297 quanta.
  for (const [coefficient, expectedBits] of [
    [1n << 297n, 0],
    [(1n << 297n) + 1n, 1],
    [3n << 297n, 2]
  ] as const) {
    const result = Float32Array.of(roundPointPatternDyadicStatistic(coefficient));
    expect(new Uint32Array(result.buffer)[0]).toBe(expectedBits);
  }
});

for (const {name, observed, reference, weights, expectedBits} of [
  {
    name: 'above midpoint',
    observed: [1, 2 ** -12, 2 ** -31],
    reference: [0, 0, 0],
    weights: [1, 1, 1],
    expectedBits: 0x3f800001
  },
  {
    name: 'exact midpoint',
    observed: [1, 2 ** -12],
    reference: [0, 0],
    weights: [1, 1],
    expectedBits: 0x3f800000
  },
  {
    name: 'below overflow midpoint',
    observed: [2 ** 64, 2 ** 51],
    reference: [0, 1],
    weights: [1 - 2 ** -24, 2],
    expectedBits: 0x7f7fffff
  }
])
  it(`B3 rounds exact dyadic statistics once: ${name}`, () => {
    // First: 1+2^-24+2^-62 is above the midpoint. Second is the even midpoint.
    // Third: 2^128-2^103-2^53+2 is below the overflow midpoint 2^128-2^103.
    const result = computePointPatternEnvelopeOnCPU(
      {
        observed: Float32Array.from(observed),
        simulations: new Float32Array(observed.length * 2),
        reference: Float32Array.from(reference),
        integrationWeights: Float32Array.from(weights)
      },
      {...PROPS, simulationCount: 2, statistic: 'integrated-squared-deviation'}
    );
    expect(result.statistics.length).toBe(4);
    expect(Number.isFinite(result.statistics[0])).toBe(true);
    expect(new Uint32Array(result.statistics.buffer)[0]).toBe(expectedBits);
    expect([...result.status]).toEqual([0]);
  });

it('DEF-12 reads only inputs used by the selected statistic and infers exact replicate counts', () => {
  const input = {
    observed: Float32Array.of(3),
    simulations: Float32Array.of(1, 2),
    reference: Float32Array.of(0)
  };
  for (const integrationWeights of [undefined, Float32Array.of(0), Float32Array.of(1)]) {
    const result = computePointPatternEnvelopeOnCPU(
      {...input, integrationWeights},
      {...PROPS, simulationCount: undefined}
    );
    expect([...result.statistics]).toEqual([3, Math.fround(1 / 3), Math.fround(1 / 3), 2]);
    expect([...result.status]).toEqual([0]);
  }
  const rank = computePointPatternEnvelopeOnCPU(
    {observed: input.observed, simulations: input.simulations},
    {...PROPS, simulationCount: undefined, statistic: 'extreme-rank'}
  );
  expect([...rank.statistics]).toEqual([1, Math.fround(2 / 3), Math.fround(1 / 3), 2]);
  expect(() =>
    computePointPatternEnvelopeOnCPU(
      {...input, observed: Float32Array.of(3, 2), simulations: Float32Array.of(1, 2, 3)},
      {...PROPS, simulationCount: undefined}
    )
  ).toThrow(/simulationCount|simulations/);
});

it('DEF-12 operation bindings omit unused legacy ports on both backends', async () => {
  const {createSamplingCPUInputs} = await import('./validation-sampling-operation-test-utils');
  for (const statistic of ['extreme-rank', 'maximum-deviation'] as const) {
    const props = {
      ...PROPS,
      statistic,
      curveRowCount: 1,
      simulationCount: undefined,
      reference: statistic === 'extreme-rank' ? undefined : 'reference',
      integrationWeights: undefined
    };
    const gpu = new GPUPointPatternEnvelopeOperation(props);
    expect(gpu.inputs['integrationWeights']).toBeUndefined();
    const result = await new CPUPointPatternEnvelopeOperation(props).execute({
      inputs: createSamplingCPUInputs({
        observed: {format: 'float32', values: Float32Array.of(3)},
        simulations: {format: 'float32', values: Float32Array.of(1, 2)},
        ...(statistic === 'extreme-rank'
          ? {}
          : {reference: {format: 'float32' as const, values: Float32Array.of(0)}})
      })
    });
    expect(result['statistics']).toEqual(
      Float32Array.of(
        statistic === 'extreme-rank' ? 1 : 3,
        statistic === 'extreme-rank' ? 2 / 3 : 1 / 3,
        1 / 3,
        2
      )
    );
    expect(result['status']).toEqual(Uint32Array.of(0));
  }
});

it('B2 independent product and tiny-normal fixtures have zero CPU exceedances', () => {
  for (const [observed, simulated] of [
    [
      [1 + 2 ** -13, 1 - 2 ** -13],
      [1, 1]
    ],
    [
      [2 ** -40, 2 ** -53],
      [2 ** -40, 0]
    ]
  ]) {
    // Exact observed excesses: 2^-25 and 2^-106. Each p-value is (0+1)/(19+1).
    const result = computePointPatternEnvelopeOnCPU(
      {
        observed: Float32Array.from(observed),
        simulations: Float32Array.from(Array.from({length: 19}, () => simulated).flat()),
        reference: Float32Array.of(0, 0),
        integrationWeights: Float32Array.of(1, 1)
      },
      {...PROPS, simulationCount: 19, statistic: 'integrated-squared-deviation'}
    );
    expect([...result.statistics]).toEqual([
      Math.fround(observed[0] === 2 ** -40 ? 2 ** -80 : 2),
      Math.fround(1 / 20),
      Math.fround(1 / 20),
      19
    ]);
    expect([...result.status]).toEqual([0]);
  }
});
