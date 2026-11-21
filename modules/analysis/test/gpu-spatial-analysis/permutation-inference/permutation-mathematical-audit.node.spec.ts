// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPULocalPermutationTest} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {
  getPermutationPValue,
  getSequentialPermutationPValue
} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-alternative';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  createAuditScene,
  PERMUTATION_AUDIT_EXPECTED,
  PERMUTATION_AUDIT_ALTERNATIVES
} from './permutation-mathematical-audit-fixture';
import {computeGlobalPermutationOracle, computeLocalPermutationOracle} from './permutation-oracle';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';

it('B3 the actual local-G denominator shader retains 300000 represented float32 tenths', () => {
  const rows = 300002;
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `long-complement-${serial++}`, format, length);
  try {
    const kernel = getGPUCommandNodeKernelSources(
      new GPULocalPermutationTest({
        statistic: 'localG',
        values: view('float32', rows),
        parameters: view('uint32', 4),
        maximumPermutations: 3,
        weights: {
          offsets: view('uint32', rows + 1),
          neighbors: view('uint32', rows),
          weights: view('float32', rows)
        },
        output: {
          exceedances: view('uint32', rows),
          pseudoPValues: view('float32', rows),
          status: {overflow: view('uint32', 1)}
        }
      }).getCommandNodes(graph)
    ).find(({id}) => id.endsWith('-statistic-validity'))!;
    const values = new Float32Array(rows).fill(Math.fround(0.1));
    values[0] = 2 ** 25;
    values[1] = 2 ** 24;
    const positions = Uint32Array.from({length: rows + 1}, (_, row) => row);
    const denominator = new Float32Array(rows);
    const totals = new Float32Array(32);
    totals[1] = 50361648;
    const memory: Record<string, Float32Array | Uint32Array> = {
      compactX: values,
      rowPositions: positions,
      normalization: denominator,
      totals,
      observedRaw: new Float32Array(rows)
    };
    // Interpret the producer's actual WGSL, limiting only the dispatch to row zero.
    // This keeps the long regression linear instead of executing irrelevant row invocations.
    // wgsl_reflect has no ldexp builtin. For these normal, bounded values,
    // multiplying by exp2(integer) is the same exact power-of-two scaling.
    const source =
      'fn probeLdexp(value: f32, power: i32) -> f32 { return value * exp2(f32(power)); }\n' +
      kernel.source
        .replace(/\bldexp\(/g, 'probeLdexp(')
        .replace(/_ = [^;]+;/g, '')
        .replace(
          /override (\w+): u32;/g,
          (_, name: string) =>
            `const ${name}: u32 = ${name === 'INVOCATION_COUNT' ? 1 : kernel.constants[name]}u;`
        );
    const bindings: Record<number, Record<number, Float32Array | Uint32Array>> = {0: {}};
    for (const binding of new WgslReflect(source).storage)
      bindings[binding.group][binding.binding] = memory[binding.name];
    new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', 1, bindings);
    // Python Fraction.from_float(float32(.1))*300000 + 2^24 rounds to 16807216.
    // The exact represented-input ratio rounds to .9982150793075562.
    expect(Number.isFinite(denominator[0])).toBe(true);
    expect(denominator[0]).toBe(16807216);
    expect(Math.fround(2 ** 24 / denominator[0])).toBe(0.9982150793075562);
  } finally {
    device.destroy();
  }
}, 120000);

it('W5-12 local G retains eight unit terms in its dominant-focus complement', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `complement-${serial++}`, format, length);
  try {
    const sources = getGPUCommandNodeKernelSources(
      new GPULocalPermutationTest({
        statistic: 'localG',
        values: view('float32', 10),
        parameters: view('uint32', 4),
        maximumPermutations: 3,
        weights: {
          offsets: view('uint32', 11),
          neighbors: view('uint32', 10),
          weights: view('float32', 10)
        },
        output: {
          exceedances: view('uint32', 10),
          pseudoPValues: view('float32', 10),
          status: {overflow: view('uint32', 1)}
        }
      }).getCommandNodes(graph)
    );
    expect(sources.find(({id}) => id.endsWith('-statistic-validity'))!.source).toContain(
      'getAdditionRoundingError'
    );
    const actual = computeLocalPermutationOracle({
      values: Float32Array.of(2 ** 25, 2 ** 24, 1, 1, 1, 1, 1, 1, 1, 1),
      weights: {
        offsets: Uint32Array.from({length: 11}, (_, row) => row),
        neighbors: Uint32Array.from({length: 10}, (_, row) => (row + 1) % 10),
        weights: new Float32Array(10).fill(1)
      },
      statistic: 'localG',
      seed: 3,
      permutationCount: 3,
      maximumNeighbors: 1,
      significanceLevel: 0.05
    });
    // Exact complement 2^24+8=16777224; f32(2^24/(2^24+8))=0.9999995231628418.
    expect(Number.isFinite(actual.observed[0])).toBe(true);
    expect(Math.fround(actual.observed[0])).toBe(0.9999995231628418);
  } finally {
    graph.device.destroy();
  }
});

it('W5-12 unrelated statistics do not scan every other value for every focus', () => {
  let reads = 0;
  const values = new Proxy(
    Float32Array.from({length: 100}, (_, row) => row + 1),
    {
      get(target, property) {
        if (typeof property === 'string' && /^\d+$/.test(property)) reads++;
        return Reflect.get(target, property, target);
      }
    }
  );
  computeLocalPermutationOracle({
    values,
    weights: {
      offsets: Uint32Array.from({length: 101}, (_, row) => row),
      neighbors: Uint32Array.from({length: 100}, (_, row) => (row + 1) % 100),
      weights: new Float32Array(100).fill(1)
    },
    statistic: 'localMoran',
    seed: 3,
    permutationCount: 1,
    maximumNeighbors: 1,
    significanceLevel: 0.05
  });
  // Compaction/centering are linear; a ten-read-per-row budget excludes the 9900-read complement.
  expect(reads).toBeLessThan(1000);
});

function expectNumbers(actual: readonly number[], expected: readonly (number | null)[]) {
  expect(actual).toHaveLength(expected.length);
  let maximumDifference = 0;
  for (const [index, value] of expected.entries()) {
    if (value === null) {
      expect(Number.isNaN(actual[index]), `row ${index}`).toBe(true);
    } else {
      expect(Number.isFinite(actual[index]), `row ${index}`).toBe(true);
      expect(Math.abs(actual[index] - value), `row ${index}`).toBeLessThanOrEqual(2e-6);
      maximumDifference = Math.max(maximumDifference, Math.abs(actual[index] - value));
    }
  }
  return maximumDifference;
}

it('all five global CPU twins match independent NumPy matrix formulas and Philox permutations', () => {
  const differences = {summary: 0, reference: 0};
  for (const statistic of ['moran', 'geary', 'getisOrdG', 'bivariateMoran', 'leesL'] as const) {
    const expected = PERMUTATION_AUDIT_EXPECTED.global[statistic];
    const actual = computeGlobalPermutationOracle({
      ...createAuditScene(),
      statistic,
      seed: 3,
      permutationCount: 21
    });
    differences.reference = Math.max(
      differences.reference,
      expectNumbers(actual.simulated, expected.reference)
    );
    differences.summary = Math.max(
      differences.summary,
      expectNumbers(
        [
          actual.observed,
          actual.pseudoPValue,
          actual.exceedances,
          21,
          actual.simulatedMean,
          actual.simulatedStandardDeviation,
          actual.zSimulated,
          actual.islandCount,
          actual.analysisCount
        ],
        [...expected.summary.slice(0, 7), 1, 6]
      )
    );
  }
  console.info('Global permutation audit maximum absolute differences', differences);
});

it('every local CPU statistic matches independent conditional draws including nonzero BJC and CLC', () => {
  const differences = {observed: 0, pseudoPValues: 0, adjustedPValues: 0};
  for (const name of Object.keys(
    PERMUTATION_AUDIT_EXPECTED.local
  ) as (keyof typeof PERMUTATION_AUDIT_EXPECTED.local)[]) {
    const expected = PERMUTATION_AUDIT_EXPECTED.local[name];
    const statistic = name.startsWith('localBivariateJoinCount')
      ? 'localBivariateJoinCount'
      : (name as Exclude<typeof name, 'localBivariateJoinCountBJC' | 'localBivariateJoinCountCLC'>);
    const actual = computeLocalPermutationOracle({
      ...createAuditScene(name.includes('Join')),
      secondValues:
        name.includes('Bivariate') || name === 'localLeeL'
          ? createAuditScene(name.includes('Join')).secondValues
          : undefined,
      statistic,
      joinCase: name.endsWith('BJC') ? 'BJC' : name.endsWith('CLC') ? 'CLC' : undefined,
      seed: 3,
      permutationCount: 21,
      significanceLevel: 0.5,
      maximumNeighbors: 2,
      alternative: 'greater'
    });
    differences.observed = Math.max(
      differences.observed,
      expectNumbers(actual.observed, expected.observed)
    );
    expect(actual.exceedances).toEqual(expected.exceedances);
    expect(actual.permutationCounts).toEqual(expected.counts);
    differences.pseudoPValues = Math.max(
      differences.pseudoPValues,
      expectNumbers(actual.pseudoPValues, expected.probabilities)
    );
    differences.adjustedPValues = Math.max(
      differences.adjustedPValues,
      expectNumbers(actual.adjustedPValues, expected.probabilities)
    );
    expect(actual.significant).toEqual(
      expected.probabilities.map(value => (value !== null && value <= 0.5 ? 1 : 0))
    );
    if (name === 'localGeary') expect(actual.classes).toEqual([4, 4, 4, 1, 3, 3, 0, 0]);
    if (name === 'localBivariateMoran') expect(actual.quadrants).toEqual([3, 3, 2, 1, 1, 1, 0, 0]);
    expect(actual.overflow).toBe(0);
    expect(actual.islandCount).toBe(1);
  }
  console.info(
    'Local permutation audit maximum absolute differences (integer outputs exact)',
    differences
  );
});

it('the global CPU twin applies its explicit sequential target to the reference and all summaries', () => {
  const expected = PERMUTATION_AUDIT_EXPECTED.sequentialGlobal;
  const actual = computeGlobalPermutationOracle({
    ...createAuditScene(),
    statistic: 'leesL',
    seed: 3,
    permutationCount: 21,
    stopAfterExceedances: 3
  });
  expectNumbers(actual.simulated, expected.reference);
  expectNumbers(
    [
      actual.pseudoPValue,
      actual.exceedances,
      actual.simulatedMean,
      actual.simulatedStandardDeviation,
      actual.zSimulated
    ],
    [
      expected.summary[1],
      expected.summary[2],
      expected.summary[4],
      expected.summary[5],
      expected.summary[6]
    ]
  );
});

it('local sequential CPU counts and all alternative tails match independent simulation literals', () => {
  for (const alternative of ['greater', 'two-sided', 'lesser', 'directed', 'folded'] as const) {
    const expected =
      alternative === 'greater'
        ? PERMUTATION_AUDIT_EXPECTED.sequentialLocal
        : PERMUTATION_AUDIT_ALTERNATIVES.localAlternatives[alternative];
    const actual = computeLocalPermutationOracle({
      ...createAuditScene(),
      secondValues: undefined,
      statistic: 'localMoran',
      seed: 3,
      permutationCount: 21,
      significanceLevel: 0.5,
      maximumNeighbors: 2,
      alternative,
      stopAfterExceedances: alternative === 'greater' ? 3 : undefined
    });
    expect(actual.exceedances).toEqual(expected.exceedances);
    expect(actual.permutationCounts).toEqual(expected.counts);
    expectNumbers(actual.pseudoPValues, expected.probabilities);
    expectNumbers(actual.adjustedPValues, expected.probabilities);
  }
});

it('tail helpers include inclusive ties, the observed reference member and the first stopping index', () => {
  // Hand count: observed=2, simulations=[0,2,2,3,5,1,2,4]. g=6, l=5;
  // mean=19/8, folded distance >=3/8 includes all eight simulations.
  const simulated = [0, 2, 2, 3, 5, 1, 2, 4];
  const counts = {greater: 6, lesser: 5, folded: 8};
  for (const [alternative, exceedances, probability] of [
    ['greater', 6, 7 / 9],
    ['lesser', 5, 6 / 9],
    ['two-sided', 5, 1],
    ['directed', 5, 6 / 9],
    ['folded', 8, 1]
  ] as const) {
    expect(getPermutationPValue(alternative, counts, 8)).toEqual({
      exceedances,
      pValue: probability
    });
  }
  expect(
    getSequentialPermutationPValue(
      'greater',
      simulated.map(value => ({greater: value >= 2, lesser: value <= 2})),
      3
    )
  ).toEqual({permutations: 4, exceedances: 3, stopped: true, pValue: 0.75});
});

it('requesting a copied adjusted p-value never overwrites the exact uncorrected significance decision', () => {
  // With correctly rounded division the stored p can equal this level, but
  // f32(level * 22)=12.999999046325684 < 13. The integer rule says 0.
  // Shader reciprocal multiplication may instead round the displayed p upward;
  // its bit pattern must not determine the uncorrected significance decision.
  const level = Math.fround(13 / 22);
  expect(Math.fround(level * 22) < 13).toBe(true);
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'audit-significance'});
  const view = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const contributor = new GPULocalPermutationTest({
    id: 'audit',
    statistic: 'localMoran',
    alternative: 'greater',
    maximumPermutations: 21,
    weights: {
      offsets: view('offsets', 'uint32', 9),
      neighbors: view('neighbors', 'uint32', 13),
      weights: view('weights', 'float32', 13)
    },
    values: view('values', 'float32', 8),
    parameters: view('parameters', 'uint32', 4),
    output: {
      exceedances: view('exceedances', 'uint32', 8),
      pseudoPValues: view('pValues', 'float32', 8),
      adjustedPValues: view('adjusted', 'float32', 8),
      significant: view('significant', 'uint32', 8),
      status: {overflow: view('overflow', 'uint32', 1)}
    }
  });
  const writers = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).filter(
    kernel => /significant\[significantOffset \+ index\] =/.test(kernel.source)
  );
  expect(writers.map(kernel => kernel.id)).toEqual(['audit-classify']);
});

it('the rational p-value oracle allows reciprocal rounding without changing the count-based decision', () => {
  // Independent IEEE float32 arithmetic: nearest division and a rounded
  // reciprocal product differ by one ulp for this rational, both legitimately
  // approximating 13/22. The headless assertion must allow either value.
  const level = Math.fround(13 / 22);
  const reciprocalProduct = Math.fround(13 * Math.fround(1 / 22));
  expect(level).toBe(0.5909090638160706);
  expect(reciprocalProduct).toBe(0.5909091234207153);
  expect(reciprocalProduct - level).toBe(2 ** -24);
  for (const pValue of [level, reciprocalProduct]) {
    expect(Number.isFinite(pValue)).toBe(true);
    expect(Math.abs(pValue - 13 / 22)).toBeLessThanOrEqual(3 * 2 ** -24);
  }
  // A rounded-p comparison depends on the division route; the integer rule does not.
  expect(level <= level).toBe(true);
  expect(reciprocalProduct <= level).toBe(false);
  expect(13 <= Math.fround(level * 22)).toBe(false);
});
