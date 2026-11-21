// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  GPUGlobalPermutationTest,
  GPULocalPermutationTest
} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {getPermutationPValue} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-alternative';
import {
  getGPUPermutationParameterValues,
  getGPUPermutationMetadata,
  type GPUPermutationSettings
} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-parameters';
import {
  computeGlobalPermutationOracle,
  computeLocalPermutationOracle,
  getOracleExceedance
} from './permutation-oracle';
import {createAuditScene} from './permutation-mathematical-audit-fixture';
import {readLocalPermutationReplay} from './permutation-harness';
import type {Buffer} from '@luma.gl/core';
import {isPermutationRatioSignificant} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-significance';
import {readPackedBuffer} from '../../../src/analysis-core/kernels/packed-readback';

const reciprocalWeights = {
  offsets: Uint32Array.of(0, 1, 2),
  neighbors: Uint32Array.of(1, 0),
  weights: Float32Array.of(1, 1)
};
const cycleWeights = {
  offsets: Uint32Array.of(0, 1, 2, 3),
  neighbors: Uint32Array.of(1, 2, 0),
  weights: Float32Array.of(1, 1, 1)
};

it('review 2 R2a local G and GStar reject the whole analyzed negative-input family', () => {
  for (const statistic of ['localG', 'localGStar'] as const) {
    const actual = computeLocalPermutationOracle({
      weights: cycleWeights,
      values: Float32Array.of(-1, 2, 3),
      statistic,
      seed: 3,
      permutationCount: 3,
      maximumNeighbors: 1,
      significanceLevel: 0.05
    });
    expect(actual.exceedances).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
    expect(actual.observed).toEqual([NaN, NaN, NaN]);
    expect(actual.pseudoPValues).toEqual([NaN, NaN, NaN]);
    expect(actual.overflow).toBe(1);
  }
});

it('review 2 R6 a single positive General G value has no denominator or reference', () => {
  const actual = computeGlobalPermutationOracle({
    weights: cycleWeights,
    values: Float32Array.of(5, 0, 0),
    statistic: 'getisOrdG',
    seed: 3,
    permutationCount: 3,
    stopAfterExceedances: 1
  });
  expect(actual.observed).toBeNaN();
  expect(actual.pseudoPValue).toBeNaN();
  expect(actual.exceedances).toBeNaN();
  expect(actual.simulated).toEqual([NaN, NaN, NaN]);
});

it('review 2 R6 an empty budget retains the defined observation but has no inference', () => {
  const actual = computeGlobalPermutationOracle({
    weights: reciprocalWeights,
    values: Float32Array.of(1, 2),
    statistic: 'moran',
    seed: 3,
    permutationCount: 0
  });
  expect(actual.observed).toBe(-1);
  expect(actual.simulated).toEqual([]);
  expect(actual.pseudoPValue).toBeNaN();
  expect(actual.exceedances).toBeNaN();
});

it('review 2 R2 reciprocal large scales and small complements have defined local inference', () => {
  // Closed form: SSx=SSy=2^69, xi*lag(y)=-2^68 and lag(x)*lag(y)=2^68.
  for (const statistic of ['localBivariateMoran', 'localLeeL', 'localG'] as const) {
    const values =
      statistic === 'localG' ? Float32Array.of(2 ** 24, 1) : Float32Array.of(-(2 ** 34), 2 ** 34);
    const actual = computeLocalPermutationOracle({
      weights: reciprocalWeights,
      values,
      secondValues: statistic === 'localG' ? undefined : values,
      statistic,
      seed: 3,
      permutationCount: 3,
      maximumNeighbors: 1,
      significanceLevel: 0.05
    });
    expect(actual.observed).toEqual(statistic === 'localBivariateMoran' ? [-0.5, -0.5] : [1, 1]);
    expect(actual.pseudoPValues).toEqual([1, 1]);
    expect(actual.exceedances).toEqual([3, 3]);
  }
});

it('review 2 R11 captures folded once and refuses wrapper omission', () => {
  let reads = 0;
  const result = getPermutationPValue(
    'folded',
    {
      greater: 21,
      lesser: 21,
      get folded() {
        return (++reads === 1 ? 21 : undefined) as number;
      }
    },
    21
  );
  expect(result).toEqual({exceedances: 21, pValue: 1});
  expect(reads).toBe(1);
  // @ts-expect-error Deliberate JavaScript omission.
  expect(() => getOracleExceedance('folded', 21, 21, 21)).toThrow(/folded/);
  const union: 'greater' | 'folded' = 'folded' as 'greater' | 'folded';
  // @ts-expect-error A count is required when the alternative can be folded.
  expect(() => getPermutationPValue(union, {greater: 21, lesser: 21}, 21)).toThrow(/folded/);
});

it('v3-1 reciprocal bivariate oracles cover positive and negative odd exponent sums', () => {
  // x=[-a,a], y=[-b,b] gives sqrt(SSx*SSy)=2ab: local I=-ab/(2ab), Lee L=2ab/(2ab).
  for (const [firstMagnitude, secondMagnitude] of [
    [2 ** 34, 3 * 2 ** 33],
    [2 ** -50, 3 * 2 ** -51]
  ]) {
    for (const statistic of ['localBivariateMoran', 'localLeeL'] as const) {
      const actual = computeLocalPermutationOracle({
        weights: reciprocalWeights,
        values: Float32Array.of(-firstMagnitude, firstMagnitude),
        secondValues: Float32Array.of(-secondMagnitude, secondMagnitude),
        statistic,
        seed: 3,
        permutationCount: 3,
        maximumNeighbors: 1,
        significanceLevel: 0.05
      });
      expect(actual.observed).toEqual(statistic === 'localBivariateMoran' ? [-0.5, -0.5] : [1, 1]);
      expect(actual.exceedances).toEqual([3, 3]);
      expect(actual.pseudoPValues).toEqual([1, 1]);
    }
  }
});

it('R1 ratio decisions agree with independent Python Fraction thresholds', () => {
  // permutation-mathematical-audit.oracle.py: exact Fraction(float32(level))*denominator.
  const denominators = [3, 22, 99, 1048577, 2147483647];
  const cases = [
    [2 ** -149, [0, 0, 0, 0, 0]],
    [0.05, [0, 1, 4, 52428, 107374183]],
    [9 / 22, [1, 8, 40, 428963, 878516031]],
    [13 / 22, [1, 12, 58, 619613, 1268967551]],
    [1 - 2 ** -24, [2, 21, 98, 1048576, 2147483519]]
  ] as const;
  for (const [level, thresholds] of cases) {
    for (const [index, denominator] of denominators.entries()) {
      const boundary = thresholds[index];
      for (const numerator of [1, boundary, boundary + 1, denominator]) {
        expect(isPermutationRatioSignificant(numerator, denominator, level)).toBe(
          numerator <= boundary
        );
      }
    }
  }
});

it('R12 metadata uses the level actually stored in the parameter buffer', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const contributor = new GPULocalPermutationTest({
    statistic: 'localMoran',
    values: view('values', 'float32', 2),
    parameters: view('parameters', 'uint32', 4),
    maximumPermutations: 21,
    weights: {
      offsets: view('offsets', 'uint32', 3),
      neighbors: view('neighbors', 'uint32', 2),
      weights: view('weights', 'float32', 2)
    },
    output: {
      exceedances: view('exceedances', 'uint32', 2),
      pseudoPValues: view('p-values', 'float32', 2),
      status: {overflow: view('overflow', 'uint32', 1)}
    }
  });
  expect(
    contributor.getInferenceMetadata({seed: 3, permutationCount: 21, significanceLevel: 0.05})
      .significanceLevel
  ).toBe(Math.fround(0.05));
  const reads = {seed: 0, permutationCount: 0, significanceLevel: 0};
  const metadata = contributor.getInferenceMetadata({
    get seed() {
      return ++reads.seed === 1 ? 3 : -1;
    },
    get permutationCount() {
      return ++reads.permutationCount === 1 ? 21 : 0;
    },
    get significanceLevel() {
      return ++reads.significanceLevel === 1 ? 0.05 : 1;
    }
  });
  expect(metadata).toMatchObject({
    seed: 3,
    requestedPermutationCount: 21,
    significanceLevel: Math.fround(0.05)
  });
  expect(reads).toEqual({seed: 1, permutationCount: 1, significanceLevel: 1});
  const global = new GPUGlobalPermutationTest({
    statistic: 'moran',
    values: contributor.props.values,
    weights: contributor.props.weights,
    parameters: contributor.props.parameters,
    maximumPermutations: 21,
    output: {summary: view('global-summary', 'float32', 12)}
  });
  for (const [describe, countField] of [
    [
      (settings: GPUPermutationSettings) => global.getInferenceMetadata(settings),
      'requestedPermutationCount'
    ],
    [getGPUPermutationMetadata, 'permutationCount']
  ] as const) {
    const otherReads = {seed: 0, permutationCount: 0, significanceLevel: 0};
    const other = describe({
      get seed() {
        otherReads.seed++;
        return 2 ** 40 + 3;
      },
      get permutationCount() {
        otherReads.permutationCount++;
        return 21;
      },
      get significanceLevel() {
        otherReads.significanceLevel++;
        return 0.05;
      }
    });
    expect(other).toMatchObject({seed: 2 ** 40 + 3, [countField]: 21});
    expect(otherReads).toEqual({seed: 1, permutationCount: 1, significanceLevel: 1});
  }
});

it('R1 compares the ratio to the exact stored float32 cutoff, not a rounded product', () => {
  // Exact dyadic product: f32(9/22)*22=8.999999940395355 < 9.
  const level = Math.fround(9 / 22);
  expect(level * 22).toBe(8.999999940395355);
  const actual = computeLocalPermutationOracle({
    ...createAuditScene(),
    secondValues: undefined,
    statistic: 'localMoran',
    seed: 3,
    permutationCount: 21,
    maximumNeighbors: 2,
    alternative: 'greater',
    significanceLevel: level
  });
  expect(actual.exceedances).toEqual([3, 8, 10, 3, 9, 12, 0xffffffff, 0xffffffff]);
  expect(actual.significant).toEqual([1, 0, 0, 1, 0, 0, 0, 0]);
});

for (const statistic of [
  'localMoran',
  'localGeary',
  'localBivariateMoran',
  'localLeeL',
  'localG',
  'localGStar'
] as const) {
  it(`R2 refuses undefined normalization for ${statistic}`, () => {
    const scene = createAuditScene();
    const bivariate = statistic === 'localBivariateMoran' || statistic === 'localLeeL';
    const actual = computeLocalPermutationOracle({
      ...scene,
      values: new Float32Array(8).fill(
        statistic === 'localG' || statistic === 'localGStar' ? 0 : 1
      ),
      secondValues: bivariate ? new Float32Array(8).fill(1) : undefined,
      statistic,
      seed: 3,
      permutationCount: 21,
      maximumNeighbors: 2,
      significanceLevel: 0.05
    });
    expect(actual.exceedances).toEqual(new Array(8).fill(0xffffffff));
    expect(actual.observed.every(Number.isNaN)).toBe(true);
    expect(actual.pseudoPValues.every(Number.isNaN)).toBe(true);
    expect(actual.significant).toEqual(new Array(8).fill(0));
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const view = <Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, name, format, length);
    const nodes = new GPULocalPermutationTest({
      statistic,
      values: view('values', 'float32', 8),
      secondValues: bivariate ? view('second', 'float32', 8) : undefined,
      parameters: view('parameters', 'uint32', 4),
      maximumPermutations: 21,
      maximumNeighbors: 2,
      weights: {
        offsets: view('offsets', 'uint32', 9),
        neighbors: view('neighbors', 'uint32', 12),
        weights: view('weights', 'float32', 12)
      },
      output: {
        exceedances: view('exceedances', 'uint32', 8),
        pseudoPValues: view('p', 'float32', 8),
        significant: view('significant', 'uint32', 8),
        observed: view('observed', 'float32', 8),
        classes: statistic === 'localGeary' ? view('classes', 'uint32', 8) : undefined,
        status: {overflow: view('overflow', 'uint32', 1)}
      }
    }).getCommandNodes(graph);
    const validityIndex = nodes.findIndex(node => node.id.endsWith('-statistic-validity'));
    expect(validityIndex).toBeGreaterThanOrEqual(0);
    expect(validityIndex).toBeLessThan(nodes.findIndex(node => node.id.endsWith('-permute')));
    const sources = getParsedKernelSources(nodes);
    if (bivariate) {
      const validitySource = [...sources.entries()].find(([name]) =>
        name.endsWith('-statistic-validity')
      )![1];
      // Metal can rewrite sqrt(a)*sqrt(b) as sqrt(a*b), so merely splitting
      // the roots does not protect the 2^69 inputs from overflowing at 2^138.
      expect(validitySource).toContain('fn getPermutationGeometricMean');
      expect(validitySource).toContain('frexp(');
      expect(validitySource).toContain('ldexp(');
    }
    for (const {source} of sources.configurations)
      expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
    if (statistic === 'localGeary')
      expect(
        [...sources.entries()].find(([name]) => name.endsWith('-geary-classes'))![1]
      ).toContain('fn isPermutationRatioSignificant');
  });
}

it('R6 global CPU inference and reference members are unavailable for constant Moran or negative General G', () => {
  const scene = createAuditScene();
  const negative = Float32Array.from(scene.values);
  negative[0] = -1;
  for (const [statistic, values] of [
    ['moran', new Float32Array(8).fill(1)],
    ['getisOrdG', negative]
  ] as const) {
    const actual = computeGlobalPermutationOracle({
      ...scene,
      statistic,
      values,
      seed: 3,
      permutationCount: 21,
      stopAfterExceedances: 3
    });
    expect(actual.observed).toBeNaN();
    expect(actual.pseudoPValue).toBeNaN();
    expect(actual.exceedances).toBeNaN();
    expect(actual.simulated).toHaveLength(21);
    expect(actual.simulated.every(Number.isNaN)).toBe(true);
    expect(actual.simulatedMean).toBeNaN();
    expect(actual.simulatedStandardDeviation).toBeNaN();
  }
});

it('R11 folded inference requires an explicit valid folded count', () => {
  // @ts-expect-error Deliberate runtime omission from a JavaScript caller.
  expect(() => getPermutationPValue('folded', {greater: 21, lesser: 21}, 21)).toThrow(/folded/);
  for (const folded of [-1, 22, 1.5, NaN])
    expect(() => getPermutationPValue('folded', {greater: 21, lesser: 21, folded}, 21)).toThrow(
      /folded/
    );
  expect(getPermutationPValue('folded', {greater: 21, lesser: 21, folded: 21}, 21)).toEqual({
    exceedances: 21,
    pValue: 1
  });
});

it.each([
  'moran',
  'getisOrdG'
] as const)('R3 global finalization filters non-finite references for %s before histogram publication', statistic => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const nodes = new GPUGlobalPermutationTest({
    statistic,
    values: view('values', 'float32', 2),
    weights: {
      offsets: view('offsets', 'uint32', 3),
      neighbors: view('neighbors', 'uint32', 2),
      weights: view('weights', 'float32', 2)
    },
    parameters: view('parameters', 'uint32', 4),
    maximumPermutations: 3,
    output: {
      summary: view('summary', 'float32', 12),
      referenceDistribution: view('reference', 'float32', 3),
      histogram: view('histogram', 'uint32', 2)
    }
  }).getCommandNodes(graph);
  const sources = getParsedKernelSources(nodes);
  for (const {source} of sources.configurations)
    expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
  if (statistic === 'getisOrdG')
    expect(
      [...sources.entries()].find(([name]) => name.endsWith('-reject-negative'))![1]
    ).toContain('positiveCount[positiveCountOffset] < 2u');
  const finalize = [...sources.entries()].find(
    ([name]) => name === 'global-permutation-test-finalize'
  )![1];
  expect(finalize).toContain('finiteReferenceCount');
  expect(finalize).toContain('if (isFiniteFloat(simulated) && !invalidInput)');
});

it('R12 refuses levels that round to endpoints and accepts the nearest float32 interior values', () => {
  for (const significanceLevel of [0.99999999, 1e-48])
    expect(() =>
      getGPUPermutationParameterValues({seed: 3, permutationCount: 21, significanceLevel})
    ).toThrow(/significanceLevel/);
  for (const [significanceLevel, bits] of [
    [1 - 2 ** -24, 0x3f7fffff],
    [2 ** -149, 1]
  ]) {
    expect(
      getGPUPermutationParameterValues({seed: 3, permutationCount: 21, significanceLevel})[3]
    ).toBe(bits);
  }
});

it('R10 replay readback follows this submission count rather than construction count', async () => {
  const fakeBuffer = (values: Float32Array) =>
    ({
      byteLength: values.byteLength,
      readAsync: async (byteOffset = 0, byteLength = values.byteLength - byteOffset) =>
        new Uint8Array(values.buffer, values.byteOffset + byteOffset, byteLength)
    }) as Buffer;
  const distribution = fakeBuffer(Float32Array.from({length: 21}, (_, index) => index + 1));
  for (const count of [21, 3, 21, 0]) {
    const actual = await readLocalPermutationReplay(
      distribution,
      fakeBuffer(Float32Array.of(7.5, count, 0))
    );
    expect(actual.nullDistribution).toEqual(Array.from({length: count}, (_, index) => index + 1));
    expect(actual.info).toEqual([7.5, count, 0]);
    expect(actual.infoBits).toEqual([
      0x40f00000,
      count === 21 ? 0x41a80000 : count === 3 ? 0x40400000 : 0,
      0
    ]);
  }
  expect(Array.from(await readPackedBuffer(distribution, 8, 'float32', 3))).toEqual([3, 4, 5]);
  await expect(readPackedBuffer(distribution, 0, 'float32', 22)).rejects.toThrow(
    /must fit 84 bytes/
  );
  const invalidLength = {byteLength: 84, readAsync: async () => new Uint8Array(84)} as Buffer;
  await expect(readPackedBuffer(invalidLength, 0, 'float32', 3)).rejects.toThrow(
    /returned 84 bytes; expected 12/
  );
});
