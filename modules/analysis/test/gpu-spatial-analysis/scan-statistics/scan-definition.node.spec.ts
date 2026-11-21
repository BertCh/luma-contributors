// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterAll, expect, test} from 'vitest';
import {computeScanOracle} from './scan-statistic-oracle';
import {
  createExpectationDefinitionFixture,
  createCylinderDefinitionFixture,
  createMaskedBaselineFixture,
  createScanDefinitionFixture
} from './scan-definition-fixtures';
import {
  computeExpectationBasedScanReference,
  computeReplicateMaximaReference,
  getExpectationBasedScore
} from './stx07-expectation-based-scan-reference';
import {
  getGPUSpatialScanStatisticParameterValues,
  getGPUScanRecurrenceInterval
} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-statistic-parameters';
import {getGPUExpectationBasedScanParameterValues} from '../../../src/gpu-spatial-analysis/scan-statistics/gpu-expectation-based-space-time-scan';
import {GPUExpectationBasedSpaceTimeScan} from '../../../src/gpu-spatial-analysis/scan-statistics/gpu-expectation-based-space-time-scan';
import {GPUSpatialScanStatistic} from '../../../src/gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';

function probeWindowLimits(
  model: 'poisson' | 'space-time-permutation',
  shape: 'circle' | 'nearest',
  nearestZoneSearch: 'indexed' | 'full-scan' = 'indexed',
  settings: {baseline?: Float32Array; fraction?: number; frameLimit?: number} = {}
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `review-window-${serial++}`, format, length);
  try {
    const spaceTime = model === 'space-time-permutation';
    const nodes = new GPUSpatialScanStatistic({
      model,
      nearestZoneSearch,
      positions: view('float32x2', 3),
      cases: view('uint32', spaceTime ? 6 : 3),
      ...(spaceTime ? {} : {baseline: view('float32', 3)}),
      timeBuckets: spaceTime ? 2 : 1,
      maximumWindowZones: 1,
      maximumPermutations: 1,
      maximumClusters: 1,
      parameters: view('uint32', 8),
      output: {
        clusterIndices: view('uint32', 4),
        clusterStatistics: view('float32', 8),
        statistics: view('float32', 2),
        summary: view('uint32', 4)
      }
    }).getCommandNodes(graph);
    const kernels = getGPUCommandNodeKernelSources(nodes);
    const centerBest = new Uint32Array(30);
    const baseline =
      settings.baseline ??
      (spaceTime ? Float32Array.of(5, 1, 5 / 6, 1 / 6, 25 / 6, 5 / 6) : Float32Array.of(1, 1, 1));
    const totals = new Uint32Array(17);
    totals.set([
      spaceTime ? 12 : 9,
      new Uint32Array(
        Float32Array.of(
          spaceTime
            ? 11.999999046325684
            : baseline.reduce((sum, value) => Math.fround(sum + value), 0)
        ).buffer
      )[0],
      0,
      0,
      0
    ]);
    const data: Record<string, ArrayBufferView> = {
      parameters: getGPUSpatialScanStatisticParameterValues({
        seed: 0,
        permutationCount: 1,
        maximumPopulationFraction: settings.fraction ?? (spaceTime ? 0.5 : 0.75),
        maximumWindowZones: settings.frameLimit,
        windowShape: shape
      }),
      neighbors: Uint32Array.of(0, 1, 1, 0, 2, 1),
      baseline,
      totals,
      centerBest,
      zoneTotals: Uint32Array.of(6, 1, 5),
      positions: Float32Array.of(-1, 0, 0, 0, 1, 0),
      distances: new Uint32Array(42),
      partials: new Uint32Array(30),
      destination: new Uint32Array(30),
      sortedKeysX: Uint32Array.of(0x407fffff, 0x80000000, 0xbf800000),
      sortedKeysY: Uint32Array.of(0x80000000, 0x80000000, 0x80000000),
      sortedRowsX: Uint32Array.of(0, 1, 2),
      sortedRowsY: Uint32Array.of(0, 1, 2),
      sortKeys: new Uint32Array(6),
      sortZones: new Uint32Array(6),
      population: spaceTime ? Uint32Array.of(6, 1, 5) : baseline
    };
    const execute = (kernel: (typeof kernels)[number]) => {
      const reflection = new WgslReflect(kernel.source);
      const bindings: Record<number, Record<number, ArrayBuffer>> = {};
      for (const storage of reflection.storage) {
        bindings[storage.group] ??= {};
        bindings[storage.group][storage.binding] = data[storage.name].buffer as ArrayBuffer;
      }
      for (const uniform of reflection.uniforms) {
        bindings[uniform.group] ??= {};
        bindings[uniform.group][uniform.binding] = new ArrayBuffer(uniform.size);
      }
      for (
        let invocation = 0;
        invocation < Number(kernel.constants['INVOCATION_COUNT']);
        invocation++
      ) {
        const source = kernel.source
          .replace(/^\s*_\s*=.*;\s*$/gm, '')
          .replace(/@workgroup_size\(\d+\)/, '@workgroup_size(1)')
          .replace(
            /let index = workgroupIndex \* \d+u \+ localInvocationIndex;/,
            `let index = ${invocation}u;`
          );
        new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
          reflection.entry.compute[0].name,
          1,
          bindings,
          {constants: kernel.constants}
        );
      }
    };
    if (!spaceTime)
      for (const kernel of kernels.filter(kernel => kernel.id.includes('-exact-population-'))) {
        if (kernel.id.includes('-merge-'))
          data['destination'] = kernel.source.includes('destinationOffset + 7u')
            ? totals
            : new Uint32Array(Number(kernel.constants['INVOCATION_COUNT']) * 10);
        execute(kernel);
        if (kernel.id.includes('-merge-')) data['partials'] = data['destination'];
      }
    execute(kernels.find(value => value.id.endsWith('-nearest'))!);
    execute(kernels.find(value => value.id.endsWith('-window-limit'))!);
    return Array.from(centerBest.slice(24));
  } finally {
    device.destroy();
  }
}

test('R1: space-time population boundary uses integer margins, not rounded expectations', () => {
  expect(
    probeWindowLimits('space-time-permutation', 'nearest').filter((_, index) => index % 2 === 0)
  ).toEqual([1, 1, 1]);
});

test('R12: fractional population circles are admitted at 100% on both discovery routes', () => {
  for (const route of ['indexed', 'full-scan'] as const) {
    expect
      .soft(
        probeWindowLimits('poisson', 'circle', route, {
          baseline: Float32Array.of(0.2, 0.2, 1),
          fraction: 1
        }),
        route
      )
      .toEqual([1, 1, 1, 1, 1, 1]);
  }
});

test('R13: explicit per-frame limits exclude capacity diagnostics for inadmissible larger circles', () => {
  // W5-03: an explicit limit equal to capacity is a model restriction; omission is storage only.
  for (const route of ['indexed', 'full-scan'] as const) {
    for (const [frameLimit, expected] of [
      [1, [1, 0, 1, 0, 1, 0]],
      [2, [1, 1, 1, 0, 1, 1]],
      [undefined, [1, 1, 1, 1, 1, 1]]
    ] as const) {
      expect
        .soft(
          probeWindowLimits('poisson', 'circle', route, {fraction: 1, frameLimit}),
          `${route} ${frameLimit}`
        )
        .toEqual(expected);
    }
  }
  for (const [frameMaximumWindowZones, overflow] of [
    [1, 0],
    [2, 2],
    [undefined, 3]
  ] as const) {
    const actual = computeScanOracle({
      positions: Float32Array.of(-1, 0, 0, 0, 1, 0),
      cases: Uint32Array.of(0, 9, 0),
      baseline: Float32Array.of(1, 1, 1),
      timeBuckets: 1,
      maximumWindowZones: 1,
      frameMaximumWindowZones,
      maximumPopulationFraction: 1,
      windowShape: 'circle',
      maximumClusters: 1,
      seed: 0,
      permutationCount: 1
    });
    expect(actual.overflow).toBe(overflow);
    expect(actual.clusters[0].truncatedByZoneCap).toBe(frameMaximumWindowZones === undefined);
  }
});

test('R15: space-time reference uses the packed 0.7 boundary and selects the 2000-case zone', () => {
  const actual = computeScanOracle({
    model: 'space-time-permutation',
    positions: Float32Array.of(-1, 0, 0, 0, 1, 0),
    cases: Uint32Array.of(5000, 2000, 1000, 0, 2000, 0),
    timeBuckets: 2,
    maximumWindowZones: 1,
    maximumPopulationFraction: 0.7,
    maximumTimeBuckets: 1,
    windowShape: 'circle',
    maximumClusters: 1,
    seed: 0,
    permutationCount: 1
  });
  expect(actual.clusters[0].center).toBe(2);
  expect(actual.clusters[0].firstBucket).toBe(0);
  expect(actual.clusters[0].observedCases).toBe(2000);
  expect(actual.clusters[0].expectedCases).toBe(1600);
  expect(actual.clusters[0].logLikelihoodRatio).toBeCloseTo(55.9657893, 6);
});

test('R20: distance-tail preflight refuses before scratch allocation and permits a smaller same-ID retry', () => {
  const device = createNullWebGPUDevice();
  const previousLimit = Object.getOwnPropertyDescriptor(
    device.limits,
    'maxStorageBufferBindingSize'
  )!;
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 2048});
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `preflight-${serial++}`, format, length);
  const props = {
    positions: view('float32x2', 32),
    cases: view('uint32', 32),
    baseline: view('float32', 32),
    maximumClusters: 1,
    maximumPermutations: 1,
    parameters: view('uint32', 8),
    output: {
      clusterIndices: view('uint32', 4),
      clusterStatistics: view('float32', 8),
      statistics: view('float32', 2),
      summary: view('uint32', 4)
    }
  };
  let allocations = 0;
  const createBuffer = graph.createTransientBuffer.bind(graph);
  graph.createTransientBuffer = descriptor => {
    allocations++;
    return createBuffer(descriptor);
  };
  try {
    expect(() =>
      new GPUSpatialScanStatistic({...props, maximumWindowZones: 15}).getCommandNodes(graph)
    ).toThrow(/distance scratch/);
    expect(allocations).toBe(0);
    expect(() =>
      // 32 * (listLength 4 + twelve exact boundary words) * 4 = 2048 bytes.
      new GPUSpatialScanStatistic({...props, maximumWindowZones: 3}).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', previousLimit);
    device.destroy();
  }
});

test('R17: each accessor-backed window setting is read once, validated and packed from the same value', () => {
  for (const [name, word] of [
    ['maximumWindowZones', 4],
    ['maximumTimeBuckets', 5],
    ['maximumWindowLength', 3]
  ] as const) {
    let reads = 0;
    const settings = {seed: 0, permutationCount: 1, replicateCount: 1};
    Object.defineProperty(settings, name, {
      get() {
        reads++;
        return reads <= (name === 'maximumWindowLength' ? 4 : 1) ? 3 : NaN;
      }
    });
    const values =
      name === 'maximumWindowLength'
        ? getGPUExpectationBasedScanParameterValues(settings)
        : getGPUSpatialScanStatisticParameterValues(settings);
    expect.soft(reads, name).toBe(1);
    expect.soft(values[word], name).toBe(3);
  }
});

test('R5: circle capacity loss includes the entire next distance tie', () => {
  for (const route of ['indexed', 'full-scan'] as const) {
    expect(probeWindowLimits('poisson', 'circle', route)).toEqual([1, 1, 1, 0, 1, 1]);
    expect(probeWindowLimits('poisson', 'nearest', route)).toEqual([1, 1, 1, 1, 1, 1]);
  }
  for (const windowShape of ['circle', 'nearest'] as const) {
    const actual = computeScanOracle({
      model: 'poisson',
      positions: Float32Array.of(-1, 0, 0, 0, 1, 0),
      cases: Uint32Array.of(0, 9, 0),
      baseline: Float32Array.of(1, 1, 1),
      timeBuckets: 1,
      maximumWindowZones: 1,
      permutationCount: 1,
      maximumClusters: 1,
      maximumPopulationFraction: 0.75,
      maximumTimeBuckets: 1,
      seed: 0,
      windowShape
    });
    expect(actual.overflow).toBe(windowShape === 'circle' ? 2 : 3);
    expect(actual.clusters[0].truncatedByZoneCap).toBe(windowShape === 'nearest');
  }
});

test('R9: exhaustive reference explicitly refuses 31 neighbors instead of silently scanning no subsets', () => {
  const scene = {
    positions: Float32Array.from({length: 62}, (_, index) => (index % 2 === 0 ? index / 2 : 0)),
    cases: Uint32Array.from({length: 31}, (_, row) => (row === 0 ? 2 : 1)),
    baseline: new Float32Array(31).fill(1),
    timeBuckets: 1,
    maximumNeighbors: 31,
    maximumWindowLength: 1
  };
  expect(() => computeExpectationBasedScanReference(scene, 1)).toThrow(/exhaustive.*16/);
  // Only zone 0 has C/B>1; adding any C=B zone lowers its score. GPU uses this closed form.
  expect(getExpectationBasedScore(2, 1)).toBeCloseTo(0.3862943611198906, 14);
});

test('R9: a near-equality positive maximum still selects its first winner', () => {
  const scene = {
    positions: Float32Array.of(0, 0),
    cases: Uint32Array.of(1000),
    baseline: Float32Array.of(999.99993896484375),
    timeBuckets: 1,
    maximumNeighbors: 1,
    maximumWindowLength: 1
  };
  const result = computeExpectationBasedScanReference(scene, 1);
  expect(result.best.center).toBe(0);
  // Decimal evaluation of C ln(C/B)+B-C with 60 digits of precision.
  expect(Math.abs(result.best.score - 1.8626452250221856e-12)).toBeLessThan(1e-25);
});

test('R11: invalid explicit windows and float32-underflowed population fractions are refused before packing', () => {
  for (const value of [NaN, Infinity, -1, 0]) {
    for (const name of ['maximumWindowZones', 'maximumTimeBuckets'] as const) {
      expect
        .soft(
          () =>
            getGPUSpatialScanStatisticParameterValues({
              seed: 0,
              permutationCount: 1,
              [name]: value
            }),
          name
        )
        .toThrow(new RegExp(name));
    }
    expect
      .soft(
        () =>
          getGPUExpectationBasedScanParameterValues({
            seed: 0,
            replicateCount: 1,
            maximumWindowLength: value
          }),
        'maximumWindowLength'
      )
      .toThrow(/maximumWindowLength/);
  }
  expect
    .soft(() =>
      getGPUSpatialScanStatisticParameterValues({
        seed: 0,
        permutationCount: 1,
        maximumPopulationFraction: 1e-50
      })
    )
    .toThrow(/maximumPopulationFraction/);
});

test('scan contributors capture validated counts, nested ports and contexts before caller mutation', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `capture-${serial++}`, format, length);
  try {
    const positions = view('float32x2', 8),
      cases = view('uint32', 8),
      baseline = view('float32', 8);
    const output = {
      clusterIndices: view('uint32', 4),
      clusterStatistics: view('float32', 8),
      statistics: view('float32', 6),
      summary: view('uint32', 4),
      status: {invalidCount: view('uint32', 1)}
    };
    const originalStatistics = output.statistics,
      originalInvalid = output.status.invalidCount;
    const context = {
      coordinateSpace: 'planar' as const,
      metric: 'native' as const,
      units: 'native' as const
    };
    const props = {
      positions,
      cases,
      baseline,
      maximumPermutations: 5,
      maximumClusters: 1,
      parameters: view('uint32', 8),
      output,
      spatialContext: context
    };
    const scan = new GPUSpatialScanStatistic(props);
    const expectationOutput = {
      clusterZones: view('uint32', 1),
      clusterStatistics: view('float32', 4),
      statistics: view('float32', 6),
      summary: view('uint32', 5)
    };
    const expectationProps = {
      positions,
      cases,
      baseline,
      timeBuckets: 1,
      maximumNeighbors: 1,
      maximumWindowLength: 1,
      maximumReplicates: 5,
      parameters: view('uint32', 4),
      output: expectationOutput
    };
    const expectation = new GPUExpectationBasedSpaceTimeScan(expectationProps);
    props.maximumPermutations = 99;
    output.statistics = view('float32', 100);
    output.status.invalidCount = view('uint32', 1);
    expectationProps.maximumReplicates = 99;
    expectationOutput.statistics = view('float32', 100);
    expect(scan.getInferenceMetadata({seed: 0, permutationCount: 99}).permutationCount).toBe(5);
    expect(expectation.getInferenceMetadata({seed: 0, replicateCount: 99}).permutationCount).toBe(
      5
    );
    expect(scan.props.output.statistics).toBe(originalStatistics);
    expect(scan.props.output.status?.invalidCount).toBe(originalInvalid);
    expect(Object.isFrozen(scan.props.spatialContext)).toBe(true);
  } finally {
    device.destroy();
  }
});

const maximumDifferences = {
  clusterStatistics: 0,
  statistics: 0,
  zoneStatistics: 0,
  expectationCluster: 0,
  expectationStatistics: 0,
  expectationZoneStatistics: 0
};
function recordDifferences(
  port: keyof typeof maximumDifferences,
  actual: ArrayLike<number>,
  expected: ArrayLike<number>
) {
  for (let row = 0; row < actual.length; row++) {
    expect(Number.isFinite(actual[row])).toBe(true);
    maximumDifferences[port] = Math.max(
      maximumDifferences[port],
      Math.abs(actual[row] - expected[row])
    );
  }
}
afterAll(() => console.info('Scan definition maximum absolute differences', maximumDifferences));

test('expectation scan excludes each non-positive or non-finite baseline cell before summing a window', () => {
  const {scene, expected} = createMaskedBaselineFixture();
  const actual = computeExpectationBasedScanReference(scene, 3);
  expect(actual.best.observed).toBe(expected.clusterStatistics[1]);
  expect(actual.best.expected).toBe(expected.clusterStatistics[2]);
  expect(actual.best.score).toBeCloseTo(expected.clusterStatistics[0], 12);
  expect(actual.best.windowLength).toBe(2);
  actual.centerScores.forEach((value, row) =>
    expect(value).toBeCloseTo(expected.zoneStatistics[row], 12)
  );
  computeReplicateMaximaReference(scene, 17, 3, 3).forEach((value, row) =>
    expect(value).toBeCloseTo(expected.statistics[row + 1], 12)
  );
});

for (const model of ['poisson', 'bernoulli', 'space-time-permutation'] as const) {
  for (const mode of ['retrospective', 'prospective'] as const) {
    test(`independent multi-bucket cylinder oracle: ${model} ${mode}`, () => {
      for (const transformed of [false, true]) {
        const {scene, expected} = createCylinderDefinitionFixture(model, mode, transformed);
        const actual = computeScanOracle(scene);
        expect(actual.clusters.length).toBe(expected.summary[0]);
        recordDifferences('statistics', actual.maxima, expected.statistics);
        recordDifferences(
          'zoneStatistics',
          actual.observed.map(value => value.ratio),
          expected.zoneStatistics
        );
        actual.maxima.forEach((value, row) =>
          expect(value).toBeCloseTo(expected.statistics[row], 10)
        );
        actual.observed.forEach((value, row) =>
          expect(value.ratio).toBeCloseTo(expected.zoneStatistics[row], 10)
        );
        actual.clusters.forEach((cluster, row) => {
          expect([
            cluster.center,
            cluster.zoneCount,
            cluster.firstBucket,
            cluster.lastBucket
          ]).toEqual(expected.indices.slice(4 * row, 4 * row + 4));
          const statistics = [
            cluster.logLikelihoodRatio,
            cluster.observedCases,
            cluster.expectedCases,
            cluster.pValue,
            cluster.radius,
            cluster.observedCases / cluster.expectedCases,
            Number(cluster.truncatedByZoneCap),
            0
          ];
          recordDifferences(
            'clusterStatistics',
            statistics,
            expected.clusterStatistics
              .slice(8 * row, 8 * row + 8)
              .map((value, word) => value * (transformed && word === 4 ? 4 : 1))
          );
          statistics.forEach((value, word) =>
            expect(value).toBeCloseTo(
              expected.clusterStatistics[8 * row + word] * (transformed && word === 4 ? 4 : 1),
              10
            )
          );
        });
      }
    });
  }
}

test('recurrence helper is the inverse alarm rate in the analysis interval unit', () => {
  expect(getGPUScanRecurrenceInterval(0.125, 2)).toBe(16);
  expect(getGPUScanRecurrenceInterval(0.25)).toBe(4);
});

test('large window limits saturate before uint32 packing instead of wrapping to smaller windows', () => {
  const scan = getGPUSpatialScanStatisticParameterValues({
    seed: 0,
    permutationCount: 1,
    maximumWindowZones: 2 ** 32 + 2,
    maximumTimeBuckets: 2 ** 32 + 1
  });
  // The shader clamps these requests to its compile-time capacity. UINT32_MAX
  // preserves an over-capacity request; modular conversion would request 2 and 1.
  expect(Array.from(scan.slice(4, 6))).toEqual([0xffffffff, 0xffffffff]);
});

test('large expectation-based window limits saturate before uint32 packing', () => {
  expect(
    getGPUExpectationBasedScanParameterValues({
      seed: 0,
      replicateCount: 1,
      maximumWindowLength: 2 ** 32 + 1
    })[3]
  ).toBe(0xffffffff);
});

test('expectation-based test reference agrees with independent exhaustive subsets and arrival-time Poisson draws', () => {
  for (const transformed of [false, true]) {
    const fixture = createExpectationDefinitionFixture(transformed);
    const result = computeExpectationBasedScanReference(fixture.scene, 3);
    expect(result.best.members).toEqual([1, 2]);
    expect([
      1,
      result.best.members.length,
      result.best.windowLength,
      result.best.center,
      3
    ]).toEqual(fixture.summary);
    expect([result.best.observed, result.best.expected]).toEqual([23, 8]);
    expect(result.best.score).toBeCloseTo(fixture.clusterStatistics[0], 12);
    result.centerScores.forEach((value, zone) =>
      expect(value).toBeCloseTo(fixture.zoneStatistics[zone], 12)
    );
    const maxima = computeReplicateMaximaReference(fixture.scene, 17, 3, 3);
    maxima.forEach((value, row) => expect(value).toBeCloseTo(fixture.statistics[row + 1], 12));
    expect((1 + maxima.filter(value => value >= result.best.score).length) / 4).toBe(0.25);
    recordDifferences(
      'expectationCluster',
      [result.best.score, result.best.observed, result.best.expected, 0.25],
      fixture.clusterStatistics
    );
    recordDifferences('expectationStatistics', maxima, fixture.statistics.slice(1, 4));
    recordDifferences('expectationZoneStatistics', result.centerScores, fixture.zoneStatistics);
  }
});

// There is no production CPU twin. This checks the existing test reference
// against independent SciPy literals, rather than treating it as its own oracle.
for (const model of ['poisson', 'bernoulli', 'space-time-permutation'] as const) {
  for (const direction of ['high', 'low', 'both'] as const) {
    test(`scan test reference against SciPy definition: ${model} ${direction}`, () => {
      for (const transformed of [false, true]) {
        const fixture = createScanDefinitionFixture(model, direction, transformed);
        const actual = computeScanOracle(fixture.scene);
        expect(actual.clusters.length).toBe(fixture.summary[0]);
        expect(actual.totalCases).toBe(fixture.summary[1]);
        expect(actual.overflow).toBe(fixture.summary[3]);
        recordDifferences('statistics', actual.maxima, fixture.statistics);
        recordDifferences(
          'zoneStatistics',
          actual.observed.map(value => value.ratio),
          fixture.zoneStatistics
        );
        actual.maxima.forEach(value => expect(value).toBeCloseTo(fixture.statistics[0], 12));
        actual.observed.forEach((value, zone) =>
          expect(value.ratio).toBeCloseTo(fixture.zoneStatistics[zone], 12)
        );
        actual.clusters.forEach((cluster, row) => {
          expect([
            cluster.center,
            cluster.zoneCount,
            cluster.firstBucket,
            cluster.lastBucket
          ]).toEqual(fixture.indices.slice(4 * row, 4 * row + 4));
          const values = [
            cluster.logLikelihoodRatio,
            cluster.observedCases,
            cluster.expectedCases,
            cluster.pValue,
            cluster.radius,
            cluster.observedCases / cluster.expectedCases,
            Number(cluster.truncatedByZoneCap),
            0
          ];
          recordDifferences(
            'clusterStatistics',
            values,
            fixture.clusterStatistics.slice(8 * row, 8 * row + 8)
          );
          values.forEach((value, word) =>
            expect(value).toBeCloseTo(fixture.clusterStatistics[8 * row + word], 12)
          );
        });
      }
    });
  }
}
