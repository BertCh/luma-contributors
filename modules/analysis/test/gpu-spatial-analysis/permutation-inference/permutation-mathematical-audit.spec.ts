// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import type {GPULocalPermutationStatistic} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {
  createGlobalPermutationHarness,
  createLocalPermutationHarness,
  type GlobalPermutationReadback
} from './permutation-harness';
import {
  createAuditScene,
  PERMUTATION_AUDIT_EXPECTED,
  PERMUTATION_AUDIT_ALTERNATIVES
} from './permutation-mathematical-audit-fixture';
import {
  GPULocalPermutationTest,
  getGPUPermutationParameterValues
} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {
  createWGSLKernelNode,
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {readFloat32} from '../../utils/gpu-contributor-test-utils';

// Small dyadic/integer pair terms keep inclusive tie counts exact. Numerical
// statistics include f32 normalization, reduction and normal-CDF approximation:
// 2e-5 absolute allows these rounding errors; counts and histogram bins are exact.
function expectNumbers(
  actual: readonly number[],
  expected: readonly (number | null)[],
  context = ''
) {
  expect(actual).toHaveLength(expected.length);
  for (const [index, value] of expected.entries()) {
    if (value === null) expect(Number.isNaN(actual[index]), `index ${index}`).toBe(true);
    else {
      expect(Number.isFinite(actual[index]), `${context} index ${index}: ${actual[index]}`).toBe(
        true
      );
      expect(Math.abs(actual[index] - value), `${context} index ${index}`).toBeLessThanOrEqual(
        2e-5
      );
    }
  }
}

function expectGlobal(
  actual: GlobalPermutationReadback,
  expected: {
    summary: readonly number[];
    reference: readonly number[];
    histogram: readonly number[];
  }
) {
  expectNumbers(actual.results, expected.summary);
  for (const index of [2, 3, 10, 11]) expect(actual.results[index]).toBe(expected.summary[index]);
  expectNumbers(
    actual.referenceDistribution.slice(0, expected.reference.length),
    expected.reference
  );
  expect(actual.referenceDistribution.slice(expected.reference.length).every(Number.isNaN)).toBe(
    true
  );
  expect(actual.histogram).toEqual(expected.histogram);
  expect(actual.analysisCount).toBe(6);
  expect(actual.islandCount).toBe(1);
  expect(actual.invalidCount).toBe(0);
}

it('all global output ports match independent Python literals, including materialized batches and rescaling', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const statistic of ['moran', 'geary', 'getisOrdG', 'bivariateMoran', 'leesL'] as const) {
    for (const permutationBatchSize of [undefined, 4]) {
      for (const rescaled of [false, true]) {
        const scene = createAuditScene();
        if (rescaled) {
          scene.values = scene.values.map(
            value => 2 * value + (statistic === 'getisOrdG' ? 0 : 16)
          );
          scene.secondValues = scene.secondValues.map(value => 4 * value + 8);
        }
        const harness = createGlobalPermutationHarness(device, {
          scene,
          statistic,
          parameters: {seed: 3, permutationCount: 21},
          maximumPermutations: 24,
          histogramBins: 5,
          permutationBatchSize
        });
        try {
          expectGlobal(await harness.run(), PERMUTATION_AUDIT_EXPECTED.global[statistic]);
        } finally {
          harness.destroy();
        }
      }
    }
  }
});

it('global sequential summaries and every alternative match independent reference members', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const permutationBatchSize of [undefined, 4]) {
    const harness = createGlobalPermutationHarness(device, {
      scene: createAuditScene(),
      statistic: 'leesL',
      parameters: {seed: 3, permutationCount: 21},
      maximumPermutations: 24,
      histogramBins: 5,
      stopAfterExceedances: 3,
      permutationBatchSize
    });
    try {
      expectGlobal(await harness.run(), PERMUTATION_AUDIT_EXPECTED.sequentialGlobal);
    } finally {
      harness.destroy();
    }
  }
  for (const alternative of ['greater', 'lesser', 'directed', 'folded'] as const) {
    const harness = createGlobalPermutationHarness(device, {
      scene: createAuditScene(),
      statistic: 'leesL',
      alternative,
      parameters: {seed: 3, permutationCount: 21},
      maximumPermutations: 24,
      histogramBins: 5
    });
    const expected = PERMUTATION_AUDIT_EXPECTED.global.leesL;
    const summary: number[] = [...expected.summary];
    summary.splice(1, 2, ...PERMUTATION_AUDIT_ALTERNATIVES.globalAlternatives[alternative]);
    try {
      expectGlobal(await harness.run(), {...expected, summary});
    } finally {
      harness.destroy();
    }
  }
});

it('every local statistic writes oracle values, counts, validity diagnostics, Geary classes and bivariate quadrants', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const name of Object.keys(
    PERMUTATION_AUDIT_EXPECTED.local
  ) as (keyof typeof PERMUTATION_AUDIT_EXPECTED.local)[]) {
    const expected = PERMUTATION_AUDIT_EXPECTED.local[name];
    const scene = createAuditScene(name.includes('Join'));
    const bivariate = name.includes('Bivariate') || name === 'localLeeL';
    const statistic: GPULocalPermutationStatistic = name.startsWith('localBivariateJoinCount')
      ? 'localBivariateJoinCount'
      : (name as GPULocalPermutationStatistic);
    const harness = createLocalPermutationHarness(device, {
      scene: {...scene, secondValues: bivariate ? scene.secondValues : undefined},
      statistic,
      joinCase: name.endsWith('BJC') ? 'BJC' : name.endsWith('CLC') ? 'CLC' : undefined,
      parameters: {seed: 3, permutationCount: 21, significanceLevel: 0.5},
      maximumPermutations: 24,
      maximumNeighbors: 2,
      alternative: 'greater',
      rowTileSize: 3,
      replayRow: name.includes('Join') ? undefined : 0
    });
    try {
      const actual = await harness.run();
      expectNumbers(actual.observed, expected.observed);
      expect(actual.exceedances).toEqual(expected.exceedances);
      expectNumbers(actual.pseudoPValues, expected.probabilities);
      expectNumbers(actual.adjustedPValues, expected.probabilities);
      expect(actual.significant).toEqual(
        expected.probabilities.map(value => (value !== null && value <= 0.5 ? 1 : 0))
      );
      expect(actual.classes).toEqual(name === 'localGeary' ? [4, 4, 4, 1, 3, 3, 0, 0] : []);
      expect(actual.quadrants).toEqual(
        name === 'localBivariateMoran' ? [3, 3, 2, 1, 1, 1, 0, 0] : []
      );
      expect(actual.analysisCount).toBe(6);
      expect(actual.islandCount).toBe(1);
      expect(actual.overflow).toBe(0);
      expect(actual.invalidCount).toBe(0);
      if (actual.replay) {
        expectNumbers(actual.replay.nullDistribution, expected.replayFirstRow);
        const rawObserved: Partial<Record<GPULocalPermutationStatistic, number>> = {
          localMoran: 7.5,
          localGeary: 2,
          localG: 4,
          localGStar: 5,
          localBivariateMoran: 10,
          localLeeL: 12
        };
        expect(actual.replay.info[0]).toBe(rawObserved[statistic]);
        expect(actual.replay.info[1]).toBe(21);
        expect(actual.replay.infoBits[2]).toBe(0);
      }
    } finally {
      harness.destroy();
    }
  }
});

it('local sequential prefixes and alternative tails match the Python oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = {...createAuditScene(), secondValues: undefined};
  for (const alternative of ['two-sided', 'lesser', 'directed', 'folded', 'greater'] as const) {
    const sequential = alternative === 'greater';
    const expected =
      alternative === 'greater'
        ? PERMUTATION_AUDIT_EXPECTED.sequentialLocal
        : PERMUTATION_AUDIT_ALTERNATIVES.localAlternatives[alternative];
    const harness = createLocalPermutationHarness(device, {
      scene,
      statistic: 'localMoran',
      alternative,
      stopAfterExceedances: sequential ? 3 : undefined,
      parameters: {seed: 3, permutationCount: 21, significanceLevel: 0.5},
      maximumPermutations: 24,
      maximumNeighbors: 2
    });
    try {
      const actual = await harness.run();
      expect(actual.exceedances).toEqual(expected.exceedances);
      expectNumbers(actual.pseudoPValues, expected.probabilities);
      expectNumbers(actual.adjustedPValues, expected.probabilities);
      if (sequential) expect(actual.permutationCounts).toEqual(expected.counts);
    } finally {
      harness.destroy();
    }
  }
});

it('an optional copied p-value preserves the exact significance decision at 13/22', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createLocalPermutationHarness(device, {
    scene: {...createAuditScene(), secondValues: undefined},
    statistic: 'localMoran',
    alternative: 'greater',
    parameters: {seed: 3, permutationCount: 21, significanceLevel: Math.fround(13 / 22)},
    maximumPermutations: 21,
    maximumNeighbors: 2
  });
  try {
    const actual = await harness.run();
    expect(actual.exceedances[5]).toBe(12);
    // The oracle is the rational 13/22, not a required quotient bit pattern.
    // Shader division can use a rounded reciprocal and multiplication: these
    // produce the adjacent floats 0.5909090638160706 and 0.5909091234207153.
    // Three ulps at this magnitude allow that rounding; an incorrect count or
    // denominator changes the p-value by much more. The decision stays exact.
    const tolerance = 3 * 2 ** -24;
    for (const pValue of [actual.pseudoPValues[5], actual.adjustedPValues[5]]) {
      expect(Number.isFinite(pValue)).toBe(true);
      expect(Math.abs(pValue - 13 / 22)).toBeLessThanOrEqual(tolerance);
    }
    expect(actual.adjustedPValues[5]).toBe(actual.pseudoPValues[5]);
    expect(actual.significant).toEqual([1, 1, 1, 1, 1, 0, 0, 0]);
  } finally {
    harness.destroy();
  }
});

it('R1 rejects 9/22 above the exact stored cutoff with and without a copied p-value', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const withAdjustedPValues of [false, true]) {
    const harness = createLocalPermutationHarness(device, {
      scene: {...createAuditScene(), secondValues: undefined},
      statistic: 'localMoran',
      alternative: 'greater',
      parameters: {seed: 3, permutationCount: 21, significanceLevel: Math.fround(9 / 22)},
      maximumPermutations: 21,
      maximumNeighbors: 2,
      withAdjustedPValues
    });
    try {
      const actual = await harness.run();
      expect(actual.exceedances).toEqual([3, 8, 10, 3, 9, 12, 0xffffffff, 0xffffffff]);
      expect(actual.significant).toEqual([1, 0, 0, 1, 0, 0, 0, 0]);
      expectNumbers(actual.pseudoPValues, [
        4 / 22,
        9 / 22,
        11 / 22,
        4 / 22,
        10 / 22,
        13 / 22,
        null,
        null
      ]);
      if (withAdjustedPValues) expect(actual.adjustedPValues).toEqual(actual.pseudoPValues);
    } finally {
      harness.destroy();
    }
  }
});

it('R2 constant Moran and zero G denominators are not tested or replayed', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const statistic of [
    'localMoran',
    'localGeary',
    'localBivariateMoran',
    'localLeeL',
    'localG',
    'localGStar'
  ] as const) {
    const scene = createAuditScene();
    scene.values.fill(statistic === 'localG' || statistic === 'localGStar' ? 0 : 1);
    scene.secondValues.fill(1);
    const harness = createLocalPermutationHarness(device, {
      scene: {
        ...scene,
        secondValues:
          statistic === 'localBivariateMoran' || statistic === 'localLeeL'
            ? scene.secondValues
            : undefined
      },
      statistic,
      parameters: {seed: 3, permutationCount: 21},
      maximumPermutations: 21,
      maximumNeighbors: 2,
      replayRow: 0
    });
    try {
      const actual = await harness.run();
      expect(actual.exceedances).toEqual(new Array(8).fill(0xffffffff));
      expectNumbers(actual.observed, new Array(8).fill(null));
      expectNumbers(actual.pseudoPValues, new Array(8).fill(null));
      expectNumbers(actual.adjustedPValues, new Array(8).fill(null));
      expect(actual.significant).toEqual(new Array(8).fill(0));
      expect(actual.replay!.info[0]).toBeNaN();
      expect(actual.replay!.info[1]).toBe(0);
      expect(actual.replay!.nullDistribution).toEqual([]);
    } finally {
      harness.destroy();
    }
  }
});

it('R1 Geary classes share the exact ratio cutoff at 7/22', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const withAdjustedPValues of [false, true]) {
    const harness = createLocalPermutationHarness(device, {
      scene: {...createAuditScene(), secondValues: undefined},
      statistic: 'localGeary',
      alternative: 'greater',
      parameters: {seed: 3, permutationCount: 21, significanceLevel: Math.fround(7 / 22)},
      maximumPermutations: 21,
      maximumNeighbors: 2,
      withAdjustedPValues
    });
    try {
      const actual = await harness.run();
      // Exact f32(7/22)*22=6.9999998807907104 < 7, so row four is not significant.
      expect(actual.exceedances).toEqual([21, 14, 18, 5, 6, 9, 0xffffffff, 0xffffffff]);
      expect(actual.significant).toEqual([0, 0, 0, 1, 0, 0, 0, 0]);
      expect(actual.classes).toEqual([4, 4, 4, 1, 4, 4, 0, 0]);
    } finally {
      harness.destroy();
    }
  }
});

it('R3 an entirely undefined global reference has no histogram members or extrema', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createGlobalPermutationHarness(device, {
    scene: {
      values: Float32Array.of(1, 1),
      weights: {
        offsets: Uint32Array.of(0, 1, 2),
        neighbors: Uint32Array.of(1, 0),
        weights: Float32Array.of(1, 1)
      }
    },
    statistic: 'moran',
    parameters: {seed: 3, permutationCount: 3},
    maximumPermutations: 3,
    histogramBins: 2
  });
  try {
    const actual = await harness.run();
    expect(actual.referenceDistribution).toEqual([NaN, NaN, NaN]);
    expect(actual.histogram).toEqual([0, 0]);
    // Summary layout: observed,p,count,P,mean,sd,z,normal-p,min,max,islands,n.
    expect(actual.results.slice(0, 3)).toEqual([NaN, NaN, NaN]);
    expect(actual.results.slice(4, 10)).toEqual(new Array(6).fill(NaN));
    expect(actual.results[3]).toBe(3);
    expect(actual.results.slice(10)).toEqual([0, 2]);
  } finally {
    harness.destroy();
  }
});

it('R10 one compiled replay reports exactly 21, 3, 21 and zero members on resubmission', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createLocalPermutationHarness(device, {
    scene: {...createAuditScene(), secondValues: undefined},
    statistic: 'localMoran',
    parameters: {seed: 3, permutationCount: 21},
    maximumPermutations: 21,
    maximumNeighbors: 2,
    replayRow: 0
  });
  // Independent Philox/Python fixture. Replay exposes raw conditional pair terms.
  const expected = PERMUTATION_AUDIT_EXPECTED.local.localMoran.replayFirstRow;
  try {
    for (const permutationCount of [21, 3, 21]) {
      const actual = await harness.run({seed: 3, permutationCount});
      expect(actual.replay!.info[1]).toBe(permutationCount);
      expectNumbers(actual.replay!.nullDistribution, expected.slice(0, permutationCount));
    }
    const unavailable = await harness.run({seed: 3, permutationCount: 21}, {replayRow: 7});
    expect(unavailable.replay!.info[0]).toBeNaN();
    expect(unavailable.replay!.info[1]).toBe(0);
    expect(unavailable.replay!.nullDistribution).toEqual([]);
  } finally {
    harness.destroy();
  }
});

it('review 2 R2 defined rescaled bivariate statistics retain inference and three replay members', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const weights = {
    offsets: Uint32Array.of(0, 1, 2),
    neighbors: Uint32Array.of(1, 0),
    weights: Float32Array.of(1, 1)
  };
  // For x=[-a,a], y=[-b,b]: SSx=2a^2, SSy=2b^2, geometric mean=2ab,
  // raw cross=-ab and raw lag product=ab. All draws tie. The second/third
  // cases exercise positive/negative odd exponent sums; the last SS product
  // underflows. Scaling must occur before multiplying the sums of squares.
  for (const [firstMagnitude, secondMagnitude] of [
    [2 ** 34, 2 ** 34],
    [2 ** 34, 3 * 2 ** 33],
    [2 ** -50, 3 * 2 ** -51]
  ]) {
    for (const statistic of ['localBivariateMoran', 'localLeeL'] as const) {
      const harness = createLocalPermutationHarness(device, {
        scene: {
          weights,
          values: Float32Array.of(-firstMagnitude, firstMagnitude),
          secondValues: Float32Array.of(-secondMagnitude, secondMagnitude)
        },
        statistic,
        parameters: {seed: 3, permutationCount: 3},
        maximumPermutations: 3,
        maximumNeighbors: 1,
        replayRow: 0
      });
      try {
        const actual = await harness.run();
        expectNumbers(
          actual.observed,
          statistic === 'localBivariateMoran' ? [-0.5, -0.5] : [1, 1],
          `${statistic}, magnitudes ${firstMagnitude}/${secondMagnitude}, counts ${actual.exceedances}, replay count ${actual.replay!.info[1]}`
        );
        expect(actual.exceedances).toEqual([3, 3]);
        expectNumbers(actual.pseudoPValues, [1, 1]);
        expectNumbers(actual.adjustedPValues, [1, 1]);
        expect(actual.significant).toEqual([0, 0]);
        expect(actual.overflow).toBe(0);
        expect(actual.replay!.info[1]).toBe(3);
        const raw =
          (statistic === 'localBivariateMoran' ? -1 : 1) * firstMagnitude * secondMagnitude;
        expect(actual.replay!.info[0]).toBe(raw);
        expect(actual.replay!.nullDistribution).toEqual([raw, raw, raw]);
      } finally {
        harness.destroy();
      }
    }
  }
});

it('review 2 R2 local G preserves a small complement beside 2^24', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createLocalPermutationHarness(device, {
    scene: {
      weights: {
        offsets: Uint32Array.of(0, 1, 2),
        neighbors: Uint32Array.of(1, 0),
        weights: Float32Array.of(1, 1)
      },
      values: Float32Array.of(2 ** 24, 1)
    },
    statistic: 'localG',
    parameters: {seed: 3, permutationCount: 3},
    maximumPermutations: 3,
    maximumNeighbors: 1,
    replayRow: 0
  });
  try {
    const actual = await harness.run();
    // Each lag is the entire complement, so both statistics and all permutations equal one.
    expectNumbers(actual.observed, [1, 1]);
    expectNumbers(actual.pseudoPValues, [1, 1]);
    expectNumbers(actual.adjustedPValues, [1, 1]);
    expect(actual.exceedances).toEqual([3, 3]);
    expect(actual.significant).toEqual([0, 0]);
    expect(actual.overflow).toBe(0);
    expect(actual.replay!.info[1]).toBe(3);
    expect(actual.replay!.nullDistribution).toEqual([1, 1, 1]);
  } finally {
    harness.destroy();
  }
});

it('W5-12 local G retains eight unit observations beside a 2^24 donor', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createLocalPermutationHarness(device, {
    scene: {
      values: Float32Array.of(2 ** 25, 2 ** 24, 1, 1, 1, 1, 1, 1, 1, 1),
      weights: {
        offsets: Uint32Array.from({length: 11}, (_, row) => row),
        neighbors: Uint32Array.from({length: 10}, (_, row) => (row + 1) % 10),
        weights: new Float32Array(10).fill(1)
      }
    },
    statistic: 'localG',
    parameters: {seed: 12, permutationCount: 3},
    maximumPermutations: 3,
    maximumNeighbors: 1,
    replayRow: 0
  });
  try {
    const actual = await harness.run();
    // Exact denominator is 16777224, not 16777216. This dyadic result
    // is also obtained by the permitted f32 reciprocal-multiply route.
    expect(Number.isFinite(actual.observed[0])).toBe(true);
    expect(actual.observed[0]).toBe(0.9999995231628418);
    expect(actual.overflow).toBe(0);
    expect(actual.replay!.info[0]).toBe(16777216); // replay preserves its public raw-statistic units
    expect(actual.replay!.info[1]).toBe(3);
    expect(actual.replay!.nullDistribution).toHaveLength(3);
    for (const member of actual.replay!.nullDistribution) {
      expect(Number.isFinite(member)).toBe(true);
      expect([1, 16777216]).toContain(member); // the nine possible donors, without the focus
    }
    // Independent full-width Philox/Fisher-Yates for seed 12 selects donor rows [5,5,1].
    expect(actual.replay!.nullDistribution).toEqual([1, 1, 16777216]);
  } finally {
    harness.destroy();
  }
});

it('W5-12 local G publishes exact short and long dominant-focus denominators', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const longValues = new Float32Array(300002).fill(Math.fround(0.1));
  longValues[0] = 2 ** 25;
  longValues[1] = 2 ** 24;
  for (const {values, expectedDenominator, expectedObserved} of [
    {
      values: Float32Array.of(2 ** 25, 2 ** 24, 1, 1, 1, 1, 1, 1, 1, 1),
      expectedDenominator: 16777224,
      expectedObserved: 0.9999995231628418
    },
    {values: longValues, expectedDenominator: 16807216, expectedObserved: 0.9982150793075562}
  ]) {
    const rows = values.length;
    const rig = new WeightsRig(device);
    recordWGSLKernelNodes(rig.graph);
    const denominators = rig.output('float32', rows);
    const observed = rig.output('float32', rows);
    const contributor = new GPULocalPermutationTest({
      id: 'complement',
      statistic: 'localG',
      maximumPermutations: 3,
      parameters: rig.input(
        getGPUPermutationParameterValues({seed: 3, permutationCount: 3}),
        'uint32',
        4
      ),
      values: rig.input(values, 'float32', rows),
      weights: rig.uploadWeights({
        offsets: Array.from({length: rows + 1}, (_, row) => row),
        neighbors: Array.from({length: rows}, (_, row) => (row + 1) % rows),
        weights: new Array(rows).fill(1),
        distances: []
      }),
      output: {
        observed: observed.view,
        exceedances: rig.output('uint32', rows).view,
        pseudoPValues: rig.output('float32', rows).view,
        status: {overflow: rig.output('uint32', 1).view}
      }
    });
    try {
      rig.run({
        getCommandNodes(graph) {
          const nodes = contributor.getCommandNodes(graph);
          const normalization = getWGSLKernelNodeRecord(
            graph,
            'complement-statistic-validity'
          )!.bindings.find(binding => binding.name === 'normalization')!.view;
          return [
            ...nodes,
            createWGSLKernelNode(graph, {
              id: 'read-complement-denominator',
              operation: 'TestNormalizationProbe',
              invocationCount: rows,
              bindings: [
                {name: 'normalization', view: normalization, type: 'f32', access: 'read'},
                {name: 'denominators', view: denominators.view, type: 'f32', access: 'read_write'}
              ],
              body: 'denominators[denominatorsOffset + index] = normalization[normalizationOffset + index];'
            })
          ];
        }
      });
      expect((await readFloat32(denominators.buffer, rows))[0]).toBe(expectedDenominator);
      const actualObserved = (await readFloat32(observed.buffer, rows))[0];
      expect(Number.isFinite(actualObserved)).toBe(true);
      expect(actualObserved).toBe(expectedObserved);
    } finally {
      rig.destroy();
    }
  }
});

it('review 2 R6 General G with one positive value has no reference or histogram', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createGlobalPermutationHarness(device, {
    scene: {
      values: Float32Array.of(5, 0, 0),
      weights: {
        offsets: Uint32Array.of(0, 1, 2, 3),
        neighbors: Uint32Array.of(1, 2, 0),
        weights: Float32Array.of(1, 1, 1)
      }
    },
    statistic: 'getisOrdG',
    parameters: {seed: 3, permutationCount: 3},
    maximumPermutations: 3,
    histogramBins: 2
  });
  try {
    const actual = await harness.run();
    // Every cross-product of two different values is zero; the ratio is 0/0.
    expect(actual.results.slice(0, 3)).toEqual([NaN, NaN, NaN]);
    expect(actual.results[3]).toBe(3);
    expect(actual.results.slice(4, 10)).toEqual(new Array(6).fill(NaN));
    expect(actual.results.slice(10)).toEqual([0, 3]);
    expect(actual.referenceDistribution).toEqual([NaN, NaN, NaN]);
    expect(actual.histogram).toEqual([0, 0]);
    expect(actual.invalidCount).toBe(0);
  } finally {
    harness.destroy();
  }
});
