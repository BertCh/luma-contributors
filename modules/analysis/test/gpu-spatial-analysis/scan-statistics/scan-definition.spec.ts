// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {GPUSpatialScanStatistic} from '../../../src/gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic';
import {getGPUSpatialScanStatisticParameterValues} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-statistic-parameters';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {getScanBoundaryPopulationWGSL} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-statistic-nearest-zones';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createExpectationDefinitionFixture,
  createCylinderDefinitionFixture,
  createMaskedBaselineFixture,
  createScanDefinitionFixture
} from './scan-definition-fixtures';

for (const nearestZoneSearch of ['indexed', 'full-scan'] as const) {
  test(`W5-01: six half-population circles report capacity loss: ${nearestZoneSearch}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    try {
      const summary = rig.output('uint32', 4);
      rig.run(
        new GPUSpatialScanStatistic({
          nearestZoneSearch,
          positions: rig.input(
            Float32Array.of(0, 0, 1, 0, 2, 0, 10, 0, 11, 0, 12, 0),
            'float32x2',
            6
          ),
          cases: rig.input(Uint32Array.of(9, 0, 0, 0, 0, 0), 'uint32'),
          baseline: rig.input(Float32Array.of(1, 0.3, 0.3, 1, 0.3, 0.3), 'float32'),
          timeBuckets: 1,
          maximumWindowZones: 2,
          maximumClusters: 1,
          maximumPermutations: 1,
          parameters: rig.input(
            getGPUSpatialScanStatisticParameterValues({
              seed: 0,
              permutationCount: 1,
              maximumPopulationFraction: 0.5
            }),
            'uint32'
          ),
          output: {
            summary,
            clusterIndices: rig.output('uint32', 4),
            clusterStatistics: rig.output('float32', 8),
            statistics: rig.output('float32', 2)
          }
        })
      );
      // The disjoint triples have identical stored-f32 exposure sums. Each complete next
      // circle contains exactly half the total, but needs three slots at capacity two.
      expect((await rig.readUint(summary))[3]).toBe(6);
    } finally {
      rig.destroy();
    }
  });
}

test('R12: boundary accumulation rounds exact float32 sums once on the GPU', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [values, expected] of [
    [Float32Array.of(0.2, 0.2, 1), 1.399999976158142],
    [Float32Array.of(1, 2 ** -24, 2 ** -25), 1.0000001192092896]
  ] as const) {
    const rig = new AnalysisRig(device);
    try {
      const population = rig.input(values, 'float32'),
        result = rig.output('float32', 1);
      rig.run({
        getCommandNodes(graph) {
          return [
            createWGSLKernelNode(graph, {
              id: 'boundary-rounding',
              operation: 'scan-boundary-test',
              bindings: [
                {name: 'population', view: population, type: 'f32', access: 'read'},
                {name: 'result', view: result, type: 'f32', access: 'read_write'}
              ],
              invocationCount: 1,
              declarations: getScanBoundaryPopulationWGSL({
                view: population,
                integer: false,
                buckets: 1
              }),
              body: 'resetBoundaryPopulation(); addZoneBoundaryPopulation(0u); addZoneBoundaryPopulation(1u); addZoneBoundaryPopulation(2u); result[resultOffset] = readBoundaryPopulation();'
            })
          ];
        }
      });
      // Python Fraction plus IEEE round-to-nearest, including an inexact halfway-prefix sum.
      expect(await rig.readFloat(result)).toEqual([expected]);
    } finally {
      rig.destroy();
    }
  }
});

for (const nearestZoneSearch of ['indexed', 'full-scan'] as const) {
  for (const finding of ['B1', 'B2'] as const)
    test(`${finding}: exact conditional population on ${nearestZoneSearch}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new AnalysisRig(device);
      try {
        const populationBoundary = finding === 'B1';
        const zones = populationBoundary ? 8 : 3;
        const coordinates = populationBoundary ? [0, 0, 0, 0, 0, 0, 10, 10] : [0, 0, 0];
        const summary = rig.output('uint32', 4);
        const statistics = rig.output('float32', 2);
        rig.run(
          new GPUSpatialScanStatistic({
            nearestZoneSearch,
            positions: rig.input(
              Float32Array.from(coordinates.flatMap(value => [value, 0])),
              'float32x2',
              zones
            ),
            cases: rig.input(
              populationBoundary
                ? Uint32Array.of(64, 0, 0, 0, 0, 0, 0, 0)
                : Uint32Array.of(64, 0, 0),
              'uint32'
            ),
            baseline: rig.input(
              populationBoundary
                ? Float32Array.of(1, 0.8, 1, 0.8, 1, 0.8, 1, 0.8)
                : Float32Array.of(0.4, 0.1, 0.4),
              'float32'
            ),
            timeBuckets: 1,
            maximumWindowZones: populationBoundary ? 5 : 3,
            maximumClusters: 1,
            maximumPermutations: 1,
            parameters: rig.input(
              getGPUSpatialScanStatisticParameterValues({
                seed: 0,
                permutationCount: 1,
                maximumPopulationFraction: populationBoundary ? 0.75 : 1
              }),
              'uint32'
            ),
            output: {
              summary,
              statistics,
              clusterIndices: rig.output('uint32', 4),
              clusterStatistics: rig.output('float32', 8)
            }
          })
        );
        const counts = await rig.readUint(summary);
        expect(counts).toHaveLength(4);
        if (populationBoundary) expect(counts[3]).toBe(6);
        else {
          const maxima = await rig.readFloat(statistics);
          expect(maxima).toHaveLength(2);
          expect(maxima).toEqual([0, 0]);
          expect(counts[0]).toBe(0);
        }
      } finally {
        rig.destroy();
      }
    });
}

for (const nearestZoneSearch of ['indexed', 'full-scan'] as const) {
  for (const fixture of [
    {
      name: 'R12 fractional 100%',
      baseline: [0.2, 0.2, 1],
      frameLimit: undefined,
      overflow: 3,
      truncated: 1,
      score: 17.5131913415,
      expected: 1.2857142857
    },
    {
      name: 'R13 frame 1',
      baseline: [1, 1, 1],
      frameLimit: 1,
      overflow: 0,
      truncated: 0,
      score: 9.887510598,
      expected: 3
    },
    {
      name: 'R13 frame 2',
      baseline: [1, 1, 1],
      frameLimit: 2,
      overflow: 2,
      truncated: 0,
      score: 9.887510598,
      expected: 3
    },
    {
      name: 'R13 omitted frame',
      baseline: [1, 1, 1],
      frameLimit: undefined,
      overflow: 3,
      truncated: 1,
      score: 9.887510598,
      expected: 3
    }
  ]) {
    test(`${fixture.name}: ${nearestZoneSearch} respects complete-circle and frame limits`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new AnalysisRig(device);
      try {
        const clusterIndices = rig.output('uint32', 4),
          clusterStatistics = rig.output('float32', 8),
          summary = rig.output('uint32', 4);
        rig.run(
          new GPUSpatialScanStatistic({
            nearestZoneSearch,
            positions: rig.input(Float32Array.of(-1, 0, 0, 0, 1, 0), 'float32x2', 3),
            cases: rig.input(Uint32Array.of(0, 9, 0), 'uint32'),
            baseline: rig.input(Float32Array.from(fixture.baseline), 'float32'),
            maximumWindowZones: 1,
            maximumClusters: 1,
            maximumPermutations: 1,
            parameters: rig.input(
              getGPUSpatialScanStatisticParameterValues({
                seed: 0,
                permutationCount: 1,
                maximumPopulationFraction: 1,
                maximumWindowZones: fixture.frameLimit
              }),
              'uint32'
            ),
            output: {
              clusterIndices,
              clusterStatistics,
              summary,
              statistics: rig.output('float32', 2)
            }
          })
        );
        expect(await rig.readUint(clusterIndices)).toEqual([1, 1, 0, 0]);
        expect(await rig.readUint(summary)).toEqual([1, 9, 1, fixture.overflow]);
        const actual = await rig.readFloat(clusterStatistics);
        expect(actual[0]).toBeCloseTo(fixture.score, 4); // f32 normalization and logarithms.
        expect(actual[1]).toBe(9);
        expect(actual[2]).toBeCloseTo(fixture.expected, 5);
        expect(actual[6]).toBe(fixture.truncated);
      } finally {
        rig.destroy();
      }
    });
  }
}

test('R15: packed 0.7 excludes the 7000-case zone in the space-time scan', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const clusterIndices = rig.output('uint32', 4),
      clusterStatistics = rig.output('float32', 8);
    rig.run(
      new GPUSpatialScanStatistic({
        model: 'space-time-permutation',
        positions: rig.input(Float32Array.of(-1, 0, 0, 0, 1, 0), 'float32x2', 3),
        cases: rig.input(Uint32Array.of(5000, 2000, 1000, 0, 2000, 0), 'uint32'),
        timeBuckets: 2,
        maximumWindowZones: 1,
        maximumClusters: 1,
        maximumPermutations: 1,
        parameters: rig.input(
          getGPUSpatialScanStatisticParameterValues({
            seed: 0,
            permutationCount: 1,
            maximumPopulationFraction: 0.7,
            maximumTimeBuckets: 1
          }),
          'uint32'
        ),
        output: {
          clusterIndices,
          clusterStatistics,
          statistics: rig.output('float32', 2),
          summary: rig.output('uint32', 4)
        }
      })
    );
    expect(await rig.readUint(clusterIndices)).toEqual([2, 1, 0, 0]);
    const actual = await rig.readFloat(clusterStatistics);
    expect(actual[0]).toBeCloseTo(55.9657893, 2); // cancellation of two O(10^3) f32 log terms.
    expect(actual[1]).toBe(2000);
    expect(actual[2]).toBeCloseTo(1600, 2);
  } finally {
    rig.destroy();
  }
});

test('R1: the 50% space-time margin boundary retains the winning one-bucket cluster', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const clusterIndices = rig.output('uint32', 4),
      clusterStatistics = rig.output('float32', 8);
    rig.run(
      new GPUSpatialScanStatistic({
        model: 'space-time-permutation',
        positions: rig.input(Float32Array.of(-1, 0, 0, 0, 1, 0), 'float32x2', 3),
        cases: rig.input(Uint32Array.of(4, 2, 1, 0, 5, 0), 'uint32'),
        timeBuckets: 2,
        maximumWindowZones: 1,
        maximumPermutations: 1,
        maximumClusters: 1,
        parameters: rig.input(
          getGPUSpatialScanStatisticParameterValues({
            seed: 0,
            permutationCount: 1,
            maximumWindowZones: 1,
            maximumTimeBuckets: 1
          }),
          'uint32'
        ),
        output: {
          clusterIndices,
          clusterStatistics,
          statistics: rig.output('float32', 2),
          summary: rig.output('uint32', 4)
        }
      })
    );
    expect(await rig.readUint(clusterIndices)).toEqual([0, 1, 1, 1]);
    const actual = await rig.readFloat(clusterStatistics);
    // Conditional Poisson: 2 ln(2/1) + 10 ln(10/11), independently in decimal arithmetic.
    expect(actual[0]).toBeCloseTo(0.4331925631, 5);
    expect(actual[1]).toBe(2);
    // Cell expectations and their grand sum are float32; normalization can move E by one step.
    expect(actual[2]).toBeCloseTo(1, 6);
  } finally {
    rig.destroy();
  }
});

for (const nearestZoneSearch of ['indexed', 'full-scan'] as const) {
  for (const windowShape of ['circle', 'nearest'] as const) {
    test(`R5: ${windowShape} ${nearestZoneSearch} reports only admissible capacity-excluded windows`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new AnalysisRig(device);
      try {
        const clusterIndices = rig.output('uint32', 4),
          clusterStatistics = rig.output('float32', 8),
          summary = rig.output('uint32', 4);
        rig.run(
          new GPUSpatialScanStatistic({
            nearestZoneSearch,
            positions: rig.input(Float32Array.of(-1, 0, 0, 0, 1, 0), 'float32x2', 3),
            cases: rig.input(Uint32Array.of(0, 9, 0), 'uint32'),
            baseline: rig.input(Float32Array.of(1, 1, 1), 'float32'),
            maximumWindowZones: 1,
            maximumClusters: 1,
            maximumPermutations: 1,
            parameters: rig.input(
              getGPUSpatialScanStatisticParameterValues({
                seed: 0,
                permutationCount: 1,
                maximumPopulationFraction: 0.75,
                windowShape
              }),
              'uint32'
            ),
            output: {
              clusterIndices,
              clusterStatistics,
              summary,
              statistics: rig.output('float32', 2)
            }
          })
        );
        expect(await rig.readUint(clusterIndices)).toEqual([1, 1, 0, 0]);
        expect(await rig.readUint(summary)).toEqual([1, 9, 1, windowShape === 'circle' ? 2 : 3]);
        const actual = await rig.readFloat(clusterStatistics);
        expect(actual[0]).toBeCloseTo(9.887510598, 5); // 9 ln(9/3).
        expect(actual.slice(1, 3)).toEqual([9, 3]);
        expect(actual[6]).toBe(windowShape === 'circle' ? 0 : 1);
      } finally {
        rig.destroy();
      }
    });
  }
}

test('R9: 31-neighbor expectation scan agrees with the singleton closed form', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const clusterZones = rig.output('uint32', 31),
      clusterStatistics = rig.output('float32', 4);
    rig.run(
      new GPUExpectationBasedSpaceTimeScan({
        positions: rig.input(
          Float32Array.from({length: 62}, (_, index) => (index % 2 === 0 ? index / 2 : 0)),
          'float32x2',
          31
        ),
        cases: rig.input(
          Uint32Array.from({length: 31}, (_, row) => (row === 0 ? 2 : 1)),
          'uint32'
        ),
        baseline: rig.input(new Float32Array(31).fill(1), 'float32'),
        timeBuckets: 1,
        maximumNeighbors: 31,
        maximumWindowLength: 1,
        maximumReplicates: 1,
        parameters: rig.input(
          getGPUExpectationBasedScanParameterValues({seed: 0, replicateCount: 1}),
          'uint32'
        ),
        output: {
          clusterZones,
          clusterStatistics,
          statistics: rig.output('float32', 2),
          summary: rig.output('uint32', 5)
        }
      })
    );
    const members = await rig.readUint(clusterZones);
    expect(members[0]).toBe(0);
    expect(members.slice(1)).toEqual(new Array(30).fill(0xffffffff));
    const actual = await rig.readFloat(clusterStatistics);
    expect(actual[0]).toBeCloseTo(0.3862943611198906, 5);
    expect(actual.slice(1, 3)).toEqual([2, 1]);
  } finally {
    rig.destroy();
  }
});

test('expectation scan valid-cell baseline oracle covers every output', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {scene, expected} = createMaskedBaselineFixture();
  const rig = new AnalysisRig(device);
  try {
    const clusterZones = rig.output('uint32', 1),
      clusterStatistics = rig.output('float32', 4),
      statistics = rig.output('float32', 6),
      summary = rig.output('uint32', 5),
      zoneStatistics = rig.output('float32', 8);
    const constructionProps = {
      positions: rig.input(scene.positions, 'float32x2', 8),
      cases: rig.input(scene.cases, 'uint32'),
      baseline: rig.input(scene.baseline, 'float32'),
      timeBuckets: 3,
      maximumNeighbors: 1,
      maximumWindowLength: 3,
      maximumReplicates: 5,
      parameters: rig.input(
        getGPUExpectationBasedScanParameterValues({seed: 17, replicateCount: 3}),
        'uint32'
      ),
      output: {clusterZones, clusterStatistics, statistics, summary, zoneStatistics}
    };
    const contributor = new GPUExpectationBasedSpaceTimeScan(constructionProps);
    constructionProps.maximumReplicates = 1;
    constructionProps.maximumWindowLength = 1;
    constructionProps.output.summary = rig.output('uint32', 5);
    rig.run(contributor);
    expect(await rig.readUint(clusterZones)).toEqual(expected.clusterZones);
    expect(await rig.readUint(summary)).toEqual(expected.summary);
    for (const [view, reference] of [
      [clusterStatistics, expected.clusterStatistics],
      [statistics, expected.statistics],
      [zoneStatistics, expected.zoneStatistics]
    ] as const) {
      const actual = await rig.readFloat(view);
      reference.forEach((value, row) => {
        expect(Number.isFinite(actual[row])).toBe(true);
        expect(Math.abs(actual[row] - value)).toBeLessThan(2e-5);
      });
    }
  } finally {
    rig.destroy();
  }
});

for (const model of ['poisson', 'bernoulli', 'space-time-permutation'] as const) {
  for (const temporalMode of ['retrospective', 'prospective'] as const) {
    test(`conditional cylinder literals: ${model} ${temporalMode}, every output`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      for (const transformed of [false, true]) {
        const {scene, expected} = createCylinderDefinitionFixture(model, temporalMode, transformed);
        const rig = new AnalysisRig(device);
        try {
          const clusterIndices = rig.output('uint32', 16);
          const clusterStatistics = rig.output('float32', 32);
          const statistics = rig.output('float32', 6);
          const zoneStatistics = rig.output('float32', 8);
          const summary = rig.output('uint32', 4);
          const invalidCount = rig.output('uint32', 1);
          const constructionProps = {
            model,
            temporalMode,
            positions: rig.input(scene.positions, 'float32x2', 8),
            cases: rig.input(scene.cases, 'uint32'),
            ...(scene.baseline ? {baseline: rig.input(scene.baseline, 'float32')} : {}),
            ...(scene.controls ? {controls: rig.input(scene.controls, 'uint32')} : {}),
            timeBuckets: 4,
            maximumWindowZones: 8,
            maximumPermutations: 5,
            maximumClusters: 4,
            parameters: rig.input(getGPUSpatialScanStatisticParameterValues(scene), 'uint32'),
            output: {
              clusterIndices,
              clusterStatistics,
              statistics,
              zoneStatistics,
              summary,
              status: {invalidCount}
            }
          };
          const contributor = new GPUSpatialScanStatistic(constructionProps);
          constructionProps.maximumPermutations = 1;
          constructionProps.output.summary = rig.output('uint32', 4);
          constructionProps.output.status.invalidCount = rig.output('uint32', 1);
          rig.run(contributor);
          expect(await rig.readUint(clusterIndices)).toEqual(expected.indices);
          expect(await rig.readUint(summary)).toEqual(expected.summary);
          expect(await rig.readUint(invalidCount)).toEqual([0]);
          for (const [view, reference] of [
            [clusterStatistics, expected.clusterStatistics],
            [statistics, expected.statistics],
            [zoneStatistics, expected.zoneStatistics]
          ] as const) {
            const actual = await rig.readFloat(view);
            reference.forEach((value, row) => {
              const target =
                value * (view === clusterStatistics && transformed && row % 8 === 4 ? 4 : 1);
              expect(Number.isFinite(actual[row])).toBe(true);
              // Up to 48 cases; permits float32 likelihood/log rounding, not a changed window.
              expect(Math.abs(actual[row] - target)).toBeLessThan(8e-5);
            });
          }
        } finally {
          rig.destroy();
        }
      }
    });
  }
}
import {
  GPUExpectationBasedSpaceTimeScan,
  getGPUExpectationBasedScanParameterValues
} from '../../../src/gpu-spatial-analysis/scan-statistics/gpu-expectation-based-space-time-scan';

test('expectation-based definition oracle covers every output and independent Poisson replicate maxima', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const transformed of [false, true]) {
    const fixture = createExpectationDefinitionFixture(transformed);
    const {scene} = fixture;
    const rig = new AnalysisRig(device);
    try {
      const clusterZones = rig.output('uint32', 4);
      const clusterStatistics = rig.output('float32', 4);
      const statistics = rig.output('float32', 6);
      const summary = rig.output('uint32', 5);
      const zoneStatistics = rig.output('float32', 8);
      rig.run(
        new GPUExpectationBasedSpaceTimeScan({
          positions: rig.input(scene.positions, 'float32x2', 8),
          cases: rig.input(scene.cases, 'uint32'),
          baseline: rig.input(scene.baseline, 'float32'),
          timeBuckets: 3,
          maximumNeighbors: 4,
          maximumWindowLength: 3,
          maximumReplicates: 5,
          parameters: rig.input(
            getGPUExpectationBasedScanParameterValues({
              seed: 17,
              replicateCount: 3,
              maximumWindowLength: 2 ** 32 + 1
            }),
            'uint32'
          ),
          output: {clusterZones, clusterStatistics, statistics, summary, zoneStatistics}
        })
      );
      expect(Array.from(await rig.readUint(clusterZones))).toEqual(fixture.clusterZones);
      expect(Array.from(await rig.readUint(summary))).toEqual(fixture.summary);
      for (const [view, expected] of [
        [clusterStatistics, fixture.clusterStatistics],
        [statistics, fixture.statistics],
        [zoneStatistics, fixture.zoneStatistics]
      ] as const) {
        const actual = await rig.readFloat(view);
        for (let row = 0; row < expected.length; row++) {
          expect(Number.isFinite(actual[row])).toBe(true);
          // Three buckets and four neighbors: allow f32 logarithm and summation rounding.
          expect(Math.abs(actual[row] - expected[row])).toBeLessThan(2e-5);
        }
      }
    } finally {
      rig.destroy();
    }
  }
});

for (const model of ['poisson', 'bernoulli', 'space-time-permutation'] as const) {
  for (const rateDirection of ['high', 'low', 'both'] as const) {
    test(`scan definition oracle: ${model} ${rateDirection}, every output`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      for (const transformed of [false, true]) {
        const fixture = createScanDefinitionFixture(model, rateDirection, transformed);
        const {scene} = fixture;
        const rig = new AnalysisRig(device);
        try {
          const clusterIndices = rig.output('uint32', 40);
          const clusterStatistics = rig.output('float32', 80);
          const statistics = rig.output('float32', 6);
          const summary = rig.output('uint32', 4);
          const zoneStatistics = rig.output('float32', 8);
          const invalidCount = rig.output('uint32', 1);
          rig.run(
            new GPUSpatialScanStatistic({
              model,
              positions: rig.input(scene.positions, 'float32x2', 8),
              cases: rig.input(scene.cases, 'uint32'),
              ...(scene.baseline ? {baseline: rig.input(scene.baseline, 'float32')} : {}),
              ...(scene.controls ? {controls: rig.input(scene.controls, 'uint32')} : {}),
              timeBuckets: scene.timeBuckets,
              maximumWindowZones: scene.maximumWindowZones,
              maximumPermutations: 5,
              maximumClusters: 10,
              parameters: rig.input(
                getGPUSpatialScanStatisticParameterValues({
                  ...scene,
                  maximumWindowZones: 2 ** 32 + 2,
                  maximumTimeBuckets: 2 ** 32 + 1
                }),
                'uint32'
              ),
              output: {
                clusterIndices,
                clusterStatistics,
                statistics,
                summary,
                zoneStatistics,
                status: {invalidCount}
              }
            })
          );
          expect(Array.from(await rig.readUint(clusterIndices))).toEqual(fixture.indices);
          expect(Array.from(await rig.readUint(summary))).toEqual(fixture.summary);
          expect(Array.from(await rig.readUint(invalidCount))).toEqual([0]);
          for (const [view, expected] of [
            [clusterStatistics, fixture.clusterStatistics],
            [statistics, fixture.statistics],
            [zoneStatistics, fixture.zoneStatistics]
          ] as const) {
            const actual = await rig.readFloat(view);
            for (let row = 0; row < expected.length; row++) {
              expect(Number.isFinite(actual[row])).toBe(true);
              // Small bounded counts: 4e-6 absolute allows f32 log/series and summation rounding.
              expect(Math.abs(actual[row] - expected[row])).toBeLessThan(4e-6);
            }
          }
        } finally {
          rig.destroy();
        }
      }
    });
  }
}
