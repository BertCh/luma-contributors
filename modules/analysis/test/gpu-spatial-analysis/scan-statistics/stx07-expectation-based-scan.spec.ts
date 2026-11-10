// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {
  getGPUExpectationBasedScanParameterValues,
  getGPUScanRecurrenceInterval,
  getGPUSpatialScanStatisticParameterValues,
  GPUExpectationBasedSpaceTimeScan,
  GPUSpatialScanStatistic,
  GPU_EXPECTATION_BASED_SCAN_CLUSTER,
  GPU_EXPECTATION_BASED_SCAN_NO_ZONE,
  GPU_EXPECTATION_BASED_SCAN_STATISTIC_WORDS,
  GPU_EXPECTATION_BASED_SCAN_SUMMARY,
  GPU_EXPECTATION_BASED_SCAN_SUMMARY_LENGTH,
  GPU_SPATIAL_SCAN_STATISTIC_CLUSTER,
  GPU_SPATIAL_SCAN_STATISTIC_CLUSTER_INDEX,
  GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS,
  GPU_SPATIAL_SCAN_STATISTIC_STATISTIC_WORDS,
  GPU_SPATIAL_SCAN_STATISTIC_SUMMARY,
  GPU_SPATIAL_SCAN_STATISTIC_SUMMARY_LENGTH
} from '../../../src/gpu-spatial-analysis/scan-statistics/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {AnalysisRig, createSeededRandom} from '../catchment-accessibility/rig';
import {
  computeExpectationBasedScanReference,
  computeReplicateMaximaReference,
  type ExpectationBasedScanScene
} from './stx07-expectation-based-scan-reference';
import {getOracleLogLikelihoodRatio} from './scan-statistic-oracle';

function createGrid(side: number): Float32Array {
  const positions = new Float32Array(side * side * 2);
  for (let zone = 0; zone < side * side; zone++) {
    positions[zone * 2] = (zone % side) * 10;
    positions[zone * 2 + 1] = Math.floor(zone / side) * 10;
  }
  return positions;
}

async function runEB(
  scene: ExpectationBasedScanScene,
  options: {
    replicateCount: number;
    maximumReplicates?: number;
    batchReplicates?: number;
    windowLimit?: number;
    seed?: number;
  }
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  const zones = scene.positions.length / 2;
  const maximumReplicates = options.maximumReplicates ?? options.replicateCount;
  const clusterZones = rig.output('uint32', scene.maximumNeighbors);
  const clusterStatistics = rig.output('float32', GPU_EXPECTATION_BASED_SCAN_STATISTIC_WORDS);
  const statistics = rig.output('float32', maximumReplicates + 1);
  const summary = rig.output('uint32', GPU_EXPECTATION_BASED_SCAN_SUMMARY_LENGTH);
  const zoneStatistics = rig.output('float32', zones);
  rig.run(
    new GPUExpectationBasedSpaceTimeScan({
      positions: rig.input(scene.positions, 'float32x2', zones),
      cases: rig.input(scene.cases, 'uint32'),
      baseline: rig.input(scene.baseline, 'float32'),
      timeBuckets: scene.timeBuckets,
      maximumNeighbors: scene.maximumNeighbors,
      maximumWindowLength: scene.maximumWindowLength,
      maximumReplicates,
      ...(options.batchReplicates ? {batchReplicates: options.batchReplicates} : {}),
      parameters: rig.input(
        getGPUExpectationBasedScanParameterValues({
          seed: options.seed ?? 12345,
          replicateCount: options.replicateCount,
          maximumWindowLength: options.windowLimit
        }),
        'uint32'
      ),
      output: {clusterZones, clusterStatistics, statistics, summary, zoneStatistics}
    })
  );
  const result = {
    zones: await rig.readUint(clusterZones),
    cluster: await rig.readFloat(clusterStatistics),
    statistics: await rig.readFloat(statistics),
    summary: await rig.readUint(summary),
    zoneStatistics: await rig.readFloat(zoneStatistics)
  };
  rig.destroy();
  return result;
}

function drawPoissonish(random: () => number, mean: number): number {
  let limit = Math.exp(-mean);
  let product = random();
  let count = 0;
  while (product > limit) {
    product *= random();
    count++;
  }
  limit = 0;
  return count + limit;
}

function createScene(): ExpectationBasedScanScene {
  const random = createSeededRandom(4242);
  const side = 5;
  const timeBuckets = 6;
  const positions = createGrid(side);
  const zones = side * side;
  const cases = new Uint32Array(zones * timeBuckets);
  const baseline = new Float32Array(zones * timeBuckets);
  for (let zone = 0; zone < zones; zone++) {
    for (let bucket = 0; bucket < timeBuckets; bucket++) {
      const mean = 6 + (zone % 3);
      const column = zone % side;
      const row = Math.floor(zone / side);
      const raised = bucket >= 4 && column >= 1 && column <= 2 && row >= 1 && row <= 2;
      baseline[zone * timeBuckets + bucket] = mean;
      cases[zone * timeBuckets + bucket] = drawPoissonish(random, mean * (raised ? 3 : 1));
    }
  }
  return {positions, cases, baseline, timeBuckets, maximumNeighbors: 9, maximumWindowLength: 3};
}

describe('GPUExpectationBasedSpaceTimeScan', () => {
  it('finds the same best subset as exhaustive enumeration, per center and overall', async () => {
    const scene = createScene();
    const reference = computeExpectationBasedScanReference(scene, 3);
    const result = await runEB(scene, {replicateCount: 19});
    if (!result.zones) return;
    expect(result.statistics[0]).toBeCloseTo(reference.best.score, 2);
    result.zoneStatistics.forEach((score, zone) => {
      expect(Math.abs(score - reference.centerScores[zone])).toBeLessThan(
        2e-3 * Math.max(1, reference.centerScores[zone])
      );
    });
    const {found, memberCount, windowLength, centerZone} = GPU_EXPECTATION_BASED_SCAN_SUMMARY;
    expect(result.summary[found]).toBe(1);
    expect(result.summary[windowLength]).toBe(reference.best.windowLength);
    expect(result.summary[centerZone]).toBe(reference.best.center);
    const members = result.zones.slice(0, result.summary[memberCount]).sort((a, b) => a - b);
    expect(members).toEqual(reference.best.members);
    expect(
      result.zones
        .slice(result.summary[memberCount])
        .every(zone => zone === GPU_EXPECTATION_BASED_SCAN_NO_ZONE)
    ).toBe(true);
    const {observedCases, expectedCases, pValue, logLikelihoodRatio} =
      GPU_EXPECTATION_BASED_SCAN_CLUSTER;
    expect(result.cluster[observedCases]).toBeCloseTo(reference.best.observed, 3);
    expect(result.cluster[expectedCases]).toBeCloseTo(reference.best.expected, 3);
    expect(result.cluster[logLikelihoodRatio]).toBeCloseTo(reference.best.score, 2);
    expect(result.cluster[pValue]).toBeGreaterThan(0);
  });

  it('reproduces the Philox Poisson replicates of the binary64 reference, independent of the batch size', async () => {
    const scene = createScene();
    const replicateCount = 200;
    const whole = await runEB(scene, {replicateCount, maximumReplicates: replicateCount});
    const batched = await runEB(scene, {
      replicateCount,
      maximumReplicates: replicateCount,
      batchReplicates: 16
    });
    if (!whole.zones || !batched.zones) return;
    expect(batched.statistics).toEqual(whole.statistics);
    const reference = computeReplicateMaximaReference(scene, 12345, replicateCount, 3);
    let close = 0;
    for (let replicate = 0; replicate < replicateCount; replicate++) {
      if (
        Math.abs(whole.statistics[replicate + 1] - reference[replicate]) <
        1e-3 * Math.max(1, reference[replicate])
      ) {
        close++;
      }
    }
    // Replicate draws agree bit for bit except for f32 exp rounding in a tiny fraction of cells.
    expect(close).toBeGreaterThanOrEqual(Math.floor(replicateCount * 0.95));
    const exceedances = reference.reduce(
      (sum, maximum) => sum + (maximum >= whole.statistics[0] ? 1 : 0),
      0
    );
    const pValue = (1 + exceedances) / (replicateCount + 1);
    expect(
      Math.abs(whole.cluster[GPU_EXPECTATION_BASED_SCAN_CLUSTER.pValue] - pValue)
    ).toBeLessThan(3 / (replicateCount + 1));
  });

  it('computes the p-value from the replicate maxima on a scene with no signal', async () => {
    const random = createSeededRandom(909);
    const scene = createScene();
    for (let cell = 0; cell < scene.cases.length; cell++) {
      scene.cases[cell] = drawPoissonish(random, scene.baseline[cell]);
    }
    const replicateCount = 299;
    const result = await runEB(scene, {replicateCount, maximumReplicates: replicateCount});
    if (!result.zones) return;
    const reference = computeReplicateMaximaReference(scene, 12345, replicateCount, 3);
    const exceedances = reference.reduce(
      (sum, maximum) => sum + (maximum >= result.statistics[0] ? 1 : 0),
      0
    );
    expect(exceedances).toBeGreaterThan(0);
    const expected = (1 + exceedances) / (replicateCount + 1);
    expect(
      Math.abs(result.cluster[GPU_EXPECTATION_BASED_SCAN_CLUSTER.pValue] - expected)
    ).toBeLessThan(5 / (replicateCount + 1));
  });

  it('draws from the Poisson law: replicate means of F for one cell match the exact expectation, including PTRS', async () => {
    for (const mean of [4, 80]) {
      const positions = new Float32Array([0, 0]);
      const scene: ExpectationBasedScanScene = {
        positions,
        cases: new Uint32Array([0]),
        baseline: new Float32Array([mean]),
        timeBuckets: 1,
        maximumNeighbors: 1,
        maximumWindowLength: 1
      };
      const replicateCount = 4000;
      const result = await runEB(scene, {replicateCount, maximumReplicates: replicateCount});
      if (!result.zones) return;
      const replicates = result.statistics.slice(1);
      const observedMean = replicates.reduce((sum, value) => sum + value, 0) / replicateCount;
      let exactMean = 0;
      let exactVariance = 0;
      let probability = Math.exp(-mean);
      for (let count = 0; count < 400; count++) {
        if (count > 0) probability *= mean / count;
        const score = count > mean ? count * Math.log(count / mean) + mean - count : 0;
        exactMean += probability * score;
        exactVariance += probability * score * score;
      }
      exactVariance -= exactMean * exactMean;
      const standardError = Math.sqrt(exactVariance / replicateCount);
      expect(Math.abs(observedMean - exactMean)).toBeLessThan(4 * standardError + 1e-3);
    }
  });

  it('honors the per-frame window limit without recompiling', async () => {
    const scene = createScene();
    const reference = computeExpectationBasedScanReference(scene, 1);
    const result = await runEB(scene, {replicateCount: 19, windowLimit: 1});
    if (!result.zones) return;
    expect(result.summary[GPU_EXPECTATION_BASED_SCAN_SUMMARY.windowLength]).toBe(1);
    expect(result.statistics[0]).toBeCloseTo(reference.best.score, 2);
  });

  it('reports no cluster when nothing exceeds the baseline', async () => {
    const side = 3;
    const timeBuckets = 3;
    const zones = side * side;
    const scene: ExpectationBasedScanScene = {
      positions: createGrid(side),
      cases: new Uint32Array(zones * timeBuckets).fill(2),
      baseline: new Float32Array(zones * timeBuckets).fill(5),
      timeBuckets,
      maximumNeighbors: 4,
      maximumWindowLength: 2
    };
    const result = await runEB(scene, {replicateCount: 9});
    if (!result.zones) return;
    expect(result.summary[GPU_EXPECTATION_BASED_SCAN_SUMMARY.found]).toBe(0);
    expect(result.summary[GPU_EXPECTATION_BASED_SCAN_SUMMARY.memberCount]).toBe(0);
    expect(result.cluster).toEqual([0, 0, 0, 0]);
    expect(result.zones.every(zone => zone === GPU_EXPECTATION_BASED_SCAN_NO_ZONE)).toBe(true);
  });
});

describe('the space-time permutation scan cannot see a uniform rise, the expectation-based scan can', () => {
  // Every zone doubles in the last bucket. The permutation model's expectation of a cell is
  // (zone total)(bucket total)/C, which here equals the observed count everywhere.
  const side = 5;
  const timeBuckets = 6;
  const zones = side * side;
  const positions = createGrid(side);
  const baselineValue = 50;

  function createUniformRise(noisy: boolean) {
    const random = createSeededRandom(77);
    const cases = new Uint32Array(zones * timeBuckets);
    const baseline = new Float32Array(zones * timeBuckets).fill(baselineValue);
    for (let zone = 0; zone < zones; zone++) {
      for (let bucket = 0; bucket < timeBuckets; bucket++) {
        const mean = baselineValue * (bucket === timeBuckets - 1 ? 2 : 1);
        cases[zone * timeBuckets + bucket] = noisy ? drawPoissonish(random, mean) : mean;
      }
    }
    return {cases, baseline};
  }

  async function runPermutationScan(cases: Uint32Array) {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    const permutationCount = 99;
    const clusterIndices = rig.output('uint32', GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS);
    const clusterStatistics = rig.output('float32', GPU_SPATIAL_SCAN_STATISTIC_STATISTIC_WORDS);
    const statistics = rig.output('float32', permutationCount + 1);
    const summary = rig.output('uint32', GPU_SPATIAL_SCAN_STATISTIC_SUMMARY_LENGTH);
    rig.run(
      new GPUSpatialScanStatistic({
        positions: rig.input(positions, 'float32x2', zones),
        cases: rig.input(cases, 'uint32'),
        model: 'space-time-permutation',
        timeBuckets,
        maximumPermutations: permutationCount,
        maximumClusters: 1,
        parameters: rig.input(
          getGPUSpatialScanStatisticParameterValues({
            seed: 5,
            permutationCount,
            maximumTimeBuckets: 3
          }),
          'uint32'
        ),
        output: {clusterIndices, clusterStatistics, statistics, summary}
      })
    );
    const result = {
      cluster: await rig.readFloat(clusterStatistics),
      summary: await rig.readUint(summary),
      statistics: await rig.readFloat(statistics)
    };
    rig.destroy();
    return result;
  }

  it('an exact uniform rise: the permutation model has no positive window; the expectation-based scan flags it', async () => {
    const {cases, baseline} = createUniformRise(false);
    const permutation = await runPermutationScan(cases);
    const expectation = await runEB(
      {positions, cases, baseline, timeBuckets, maximumNeighbors: 25, maximumWindowLength: 3},
      {replicateCount: 99}
    );
    if (!permutation?.cluster || !expectation.zones) return;
    expect(permutation.summary[GPU_SPATIAL_SCAN_STATISTIC_SUMMARY.clusterCount]).toBe(0);
    expect(permutation.statistics[0]).toBeLessThan(1e-3);
    expect(expectation.summary[GPU_EXPECTATION_BASED_SCAN_SUMMARY.found]).toBe(1);
    expect(expectation.summary[GPU_EXPECTATION_BASED_SCAN_SUMMARY.memberCount]).toBe(25);
    expect(expectation.cluster[GPU_EXPECTATION_BASED_SCAN_CLUSTER.pValue]).toBeCloseTo(0.01, 6);
    // F of the whole grid in the last bucket: C ln(C/B) + B - C with C = 2B.
    const total = zones * baselineValue;
    expect(expectation.statistics[0]).toBeCloseTo(2 * total * Math.log(2) - total, 1);
  });

  it('a noisy uniform rise: the permutation p-value is not significant and the expectation-based one is', async () => {
    const {cases, baseline} = createUniformRise(true);
    const permutation = await runPermutationScan(cases);
    const expectation = await runEB(
      {positions, cases, baseline, timeBuckets, maximumNeighbors: 25, maximumWindowLength: 3},
      {replicateCount: 99}
    );
    if (!permutation?.cluster || !expectation.zones) return;
    if (permutation.summary[GPU_SPATIAL_SCAN_STATISTIC_SUMMARY.clusterCount] > 0) {
      expect(permutation.cluster[GPU_SPATIAL_SCAN_STATISTIC_CLUSTER.pValue]).toBeGreaterThan(0.05);
    }
    expect(expectation.cluster[GPU_EXPECTATION_BASED_SCAN_CLUSTER.pValue]).toBeLessThanOrEqual(
      0.02
    );
  });
});

describe('GPUSpatialScanStatistic prospective mode', () => {
  const side = 5;
  const timeBuckets = 8;
  const zones = side * side;

  function createScene() {
    const random = createSeededRandom(321);
    const positions = createGrid(side);
    const cases = new Uint32Array(zones * timeBuckets);
    const baseline = new Float32Array(zones * timeBuckets).fill(40);
    for (let zone = 0; zone < zones; zone++) {
      const column = zone % side;
      const row = Math.floor(zone / side);
      for (let bucket = 0; bucket < timeBuckets; bucket++) {
        let rate = 1;
        // A strong outbreak in the middle, a weaker one still under way at the end.
        if (column <= 1 && row <= 1 && bucket >= 2 && bucket <= 3) rate = 3;
        // Within the last three buckets but not ending at the last one: prospective windows skip it.
        if (column >= 3 && row <= 1 && bucket === timeBuckets - 2) rate = 3;
        if (column >= 3 && row >= 3 && bucket === timeBuckets - 1) rate = 2;
        cases[zone * timeBuckets + bucket] = drawPoissonish(random, 40 * rate);
      }
    }
    return {positions, cases, baseline};
  }

  async function runScan(temporalMode: 'retrospective' | 'prospective') {
    const {positions, cases, baseline} = createScene();
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    const permutationCount = 99;
    const clusterIndices = rig.output('uint32', 2 * GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS);
    const clusterStatistics = rig.output('float32', 2 * GPU_SPATIAL_SCAN_STATISTIC_STATISTIC_WORDS);
    const statistics = rig.output('float32', permutationCount + 1);
    const summary = rig.output('uint32', GPU_SPATIAL_SCAN_STATISTIC_SUMMARY_LENGTH);
    rig.run(
      new GPUSpatialScanStatistic({
        positions: rig.input(positions, 'float32x2', zones),
        cases: rig.input(cases, 'uint32'),
        baseline: rig.input(baseline, 'float32'),
        timeBuckets,
        maximumPermutations: permutationCount,
        maximumClusters: 2,
        temporalMode,
        parameters: rig.input(
          getGPUSpatialScanStatisticParameterValues({
            seed: 11,
            permutationCount,
            maximumWindowZones: 8,
            windowShape: 'nearest',
            maximumPopulationFraction: 1,
            maximumTimeBuckets: 3
          }),
          'uint32'
        ),
        output: {clusterIndices, clusterStatistics, statistics, summary}
      })
    );
    const result = {
      indices: await rig.readUint(clusterIndices),
      clusters: await rig.readFloat(clusterStatistics),
      statistics: await rig.readFloat(statistics)
    };
    rig.destroy();
    return {...result, positions, cases, baseline};
  }

  function bruteForceProspective(
    positions: Float32Array,
    cases: Uint32Array,
    baseline: Float32Array
  ): {llr: number; center: number; zoneCount: number; firstBucket: number} {
    const totalCases = cases.reduce((sum, value) => sum + value, 0);
    const totalBaseline = baseline.reduce((sum, value) => sum + value, 0);
    let best = {llr: 0, center: -1, zoneCount: 0, firstBucket: 0};
    for (let center = 0; center < zones; center++) {
      const order = Array.from({length: zones}, (_, zone) => zone).sort((left, right) => {
        const leftDistance =
          (positions[left * 2] - positions[center * 2]) ** 2 +
          (positions[left * 2 + 1] - positions[center * 2 + 1]) ** 2;
        const rightDistance =
          (positions[right * 2] - positions[center * 2]) ** 2 +
          (positions[right * 2 + 1] - positions[center * 2 + 1]) ** 2;
        return leftDistance - rightDistance || left - right;
      });
      for (let zoneCount = 1; zoneCount <= 8; zoneCount++) {
        for (
          let firstBucket = Math.max(0, timeBuckets - 3);
          firstBucket < timeBuckets;
          firstBucket++
        ) {
          let observed = 0;
          let expected = 0;
          for (const zone of order.slice(0, zoneCount)) {
            for (let bucket = firstBucket; bucket < timeBuckets; bucket++) {
              observed += cases[zone * timeBuckets + bucket];
              expected += baseline[zone * timeBuckets + bucket];
            }
          }
          const llr = getOracleLogLikelihoodRatio(observed, expected, totalCases, totalBaseline);
          if (llr > best.llr + 1e-9) {
            best = {llr, center, zoneCount, firstBucket};
          }
        }
      }
    }
    return best;
  }

  it('finds only windows ending at the last bucket and matches a brute-force prospective search', async () => {
    const retrospective = await runScan('retrospective');
    const prospective = await runScan('prospective');
    if (!retrospective?.indices || !prospective?.indices) return;
    const {firstBucket, lastBucket, center, zoneCount} = GPU_SPATIAL_SCAN_STATISTIC_CLUSTER_INDEX;
    // The strongest retrospective cluster is the middle outbreak.
    expect(retrospective.indices[lastBucket]).toBeLessThan(timeBuckets - 1);
    expect(prospective.indices[lastBucket]).toBe(timeBuckets - 1);
    // Whatever the mode, a window may not end at the skipped late outbreak.
    for (let cluster = 0; cluster < 2; cluster++) {
      const clusterLast =
        prospective.indices[cluster * GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS + lastBucket];
      const clusterRatio =
        prospective.clusters[cluster * GPU_SPATIAL_SCAN_STATISTIC_STATISTIC_WORDS];
      if (clusterRatio > 0) {
        expect(clusterLast).toBe(timeBuckets - 1);
      }
    }
    const reference = bruteForceProspective(
      prospective.positions,
      prospective.cases,
      prospective.baseline
    );
    expect(prospective.indices[center]).toBe(reference.center);
    expect(prospective.indices[zoneCount]).toBe(reference.zoneCount);
    expect(prospective.indices[firstBucket]).toBe(reference.firstBucket);
    expect(prospective.clusters[GPU_SPATIAL_SCAN_STATISTIC_CLUSTER.logLikelihoodRatio]).toBeCloseTo(
      reference.llr,
      1
    );
  });

  it('searches the same restricted windows in the replicates', async () => {
    const retrospective = await runScan('retrospective');
    const prospective = await runScan('prospective');
    if (!retrospective?.statistics || !prospective?.statistics) return;
    // The same streams over a subset of the windows can only lower each replicate's maximum.
    let lower = 0;
    for (let replicate = 1; replicate < retrospective.statistics.length; replicate++) {
      expect(prospective.statistics[replicate]).toBeLessThanOrEqual(
        retrospective.statistics[replicate] + 1e-4
      );
      if (prospective.statistics[replicate] < retrospective.statistics[replicate] - 1e-4) {
        lower++;
      }
    }
    expect(lower).toBeGreaterThan(0);
  });

  it('turns a p-value into a recurrence interval', () => {
    expect(getGPUScanRecurrenceInterval(0.01, 1)).toBe(100);
    expect(getGPUScanRecurrenceInterval(0.05, 7)).toBeCloseTo(140, 9);
    expect(() => getGPUScanRecurrenceInterval(0)).toThrow(/recurrence/);
  });
});
