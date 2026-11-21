// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ScanOracleInput} from './scan-statistic-oracle';
import type {ExpectationBasedScanScene} from './stx07-expectation-based-scan-reference';
import cylinders from './scan-cylinders.json';
import maskedBaseline from './scan-masked-baseline.json';

export function createMaskedBaselineFixture() {
  return {
    scene: {
      positions: Float32Array.from(Array.from({length: 8}, (_, row) => [row, 0]).flat()),
      cases: Uint32Array.from(
        [
          ...Array.from({length: 4}, () => [100, 6, 8]),
          [100, 100, 100],
          [2, 2, 2],
          [1, 1, 1],
          [2, 2, 5]
        ].flat()
      ),
      baseline: Float32Array.of(
        0,
        2,
        2,
        -2,
        2,
        2,
        Number.NaN,
        2,
        2,
        Number.POSITIVE_INFINITY,
        2,
        2,
        0,
        0,
        0,
        2,
        2,
        2,
        1,
        1,
        1,
        2,
        2,
        2
      ),
      timeBuckets: 3,
      maximumNeighbors: 1,
      maximumWindowLength: 3
    },
    expected: maskedBaseline
  };
}

export function createCylinderDefinitionFixture(
  model: NonNullable<ScanOracleInput['model']>,
  temporalMode: 'retrospective' | 'prospective',
  transformed = false
) {
  const cases = Uint32Array.from(
    [[6, 6, 0, 0], [6, 6, 0, 0], ...Array.from({length: 6}, () => [0, 0, 2, 2])].flat()
  );
  const scene: ScanOracleInput = {
    model,
    temporalMode,
    cases,
    positions: Float32Array.from(
      Array.from({length: 8}, (_, row) => [
        transformed ? 100 + 4 * row : row,
        transformed ? -20 : 0
      ]).flat()
    ),
    ...(model === 'poisson' ? {baseline: new Float32Array(32).fill(transformed ? 32 : 1)} : {}),
    ...(model === 'bernoulli' ? {controls: Uint32Array.from(cases, value => 10 - value)} : {}),
    timeBuckets: 4,
    maximumTimeBuckets: 2,
    maximumWindowZones: 8,
    maximumPopulationFraction: 0.5,
    windowShape: 'circle',
    maximumClusters: 4,
    seed: 17,
    permutationCount: 3
  };
  return {scene, expected: cylinders[`${model}:${temporalMode}`]};
}

export function createExpectationDefinitionFixture(transformed = false) {
  const scene: ExpectationBasedScanScene = {
    positions: Float32Array.from(
      [0, 1, 2, 3, 10, 11, 12, 13].flatMap(value => [
        transformed ? 100 + 4 * value : value,
        transformed ? -20 : 0
      ])
    ),
    cases: Uint32Array.of(
      2,
      2,
      2,
      2,
      5,
      8,
      2,
      4,
      6,
      2,
      1,
      1,
      3,
      3,
      3,
      3,
      6,
      9,
      1,
      1,
      1,
      100,
      100,
      100
    ),
    baseline: Float32Array.of(
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      3,
      3,
      3,
      3,
      3,
      3,
      1,
      1,
      1,
      0,
      0,
      0
    ),
    timeBuckets: 3,
    maximumNeighbors: 4,
    maximumWindowLength: 3
  };
  return {
    scene,
    clusterZones: [1, 2, 0xffffffff, 0xffffffff],
    clusterStatistics: [9.289211507734215, 23, 8, 0.25],
    statistics: [
      9.289211507734215, 0.9173105155479586, 0.23716427637242532, 1.6674938392734706, 0, 0
    ],
    zoneStatistics: [
      9.289211507734215, 9.289211507734215, 9.289211507734215, 9.289211507734215, 4.744360978112326,
      4.744360978112326, 4.744360978112326, 4.744360978112326
    ],
    summary: [1, 2, 2, 0, 3]
  };
}

// Literals from scan-definition.oracle.py (SciPy G test / 2). Replicate maxima
// are seed independent by the single-case and fixed-margin derivations there.
const POISSON = {
  high: 2.0794415416798357,
  low: 0.6931471805599453,
  highZones: [
    0.6931471805599453, 0.6931471805599453, 0.9808292530117262, 2.0794415416798357,
    0.9808292530117262, 0, 0, 0
  ],
  lowZones: [
    0.47000362924573563, 0.47000362924573563, 0.13353139262452257, 0, 0.13353139262452257,
    0.47000362924573563, 0.6931471805599453, 0.6931471805599453
  ],
  secondaryLow: 0.47000362924573563
};
const BERNOULLI = {
  high: 2.354372178183458,
  low: 0.7265052492518544,
  highZones: [
    0.7265052492518546, 0.7265052492518546, 1.0372992861055204, 2.354372178183458,
    1.0372992861055204, 0, 0, 0
  ],
  lowZones: [
    0.48983680538886615, 0.48983680538886615, 0.13820557168970615, 0, 0.13820557168970615,
    0.48983680538886615, 0.7265052492518544, 0.7265052492518544
  ],
  secondaryLow: 0.48983680538886615
};

export function createScanDefinitionFixture(
  model: NonNullable<ScanOracleInput['model']>,
  rateDirection: NonNullable<ScanOracleInput['rateDirection']>,
  transformed = false
) {
  const spaceTime = model === 'space-time-permutation';
  const cases = spaceTime
    ? Uint32Array.of(1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1)
    : Uint32Array.of(0, 0, 0, 1, 0, 0, 0, 0);
  const scene: ScanOracleInput = {
    model,
    rateDirection,
    positions: Float32Array.from(
      Array.from({length: 8}, (_, zone) => [
        transformed ? 100 + 4 * zone : zone,
        transformed ? -20 : 0
      ]).flat()
    ),
    cases,
    ...(model === 'poisson' ? {baseline: new Float32Array(8).fill(transformed ? 32 : 1)} : {}),
    ...(model === 'bernoulli' ? {controls: Uint32Array.of(2, 2, 2, 1, 2, 2, 2, 2)} : {}),
    timeBuckets: spaceTime ? 2 : 1,
    maximumTimeBuckets: 1,
    maximumWindowZones: spaceTime ? 1 : 8,
    maximumPopulationFraction: 0.5,
    windowShape: 'circle',
    maximumClusters: 10,
    seed: 17,
    permutationCount: 3
  };
  let indices: number[];
  let clusterStatistics: number[];
  let zoneStatistics: number[];
  let maximum: number;
  if (spaceTime) {
    const high = rateDirection === 'high';
    maximum = high ? 0.21019708015128524 : 0.5163081691005693;
    indices = Array.from({length: 8}, (_, zone) => [
      zone,
      1,
      high ? zone % 2 : 1 - (zone % 2),
      high ? zone % 2 : 1 - (zone % 2)
    ]).flat();
    clusterStatistics = Array.from({length: 8}, () => [
      maximum,
      high ? 1 : 0,
      0.5,
      1,
      0,
      high ? 2 : 0,
      1,
      0
    ]).flat();
    zoneStatistics = new Array(8).fill(maximum);
  } else {
    const scores = model === 'poisson' ? POISSON : BERNOULLI;
    const high = [scores.high, 1, 0.125, 1, 0, 8, 0, 0];
    const low = [scores.low, 0, 0.5, 1, transformed ? 8 : 2, 0, 0, 0];
    const secondaryLow = [scores.secondaryLow, 0, 0.375, 1, transformed ? 8 : 2, 0, 0, 0];
    indices =
      rateDirection === 'high'
        ? [3, 1, 0, 0]
        : rateDirection === 'low'
          ? [6, 4, 0, 0, 0, 3, 0, 0]
          : [3, 1, 0, 0, 6, 4, 0, 0];
    clusterStatistics =
      rateDirection === 'high'
        ? high
        : rateDirection === 'low'
          ? [...low, ...secondaryLow]
          : [...high, ...low];
    zoneStatistics =
      rateDirection === 'high'
        ? scores.highZones
        : rateDirection === 'low'
          ? scores.lowZones
          : scores.highZones.map((value, zone) => Math.max(value, scores.lowZones[zone]));
    maximum = rateDirection === 'low' ? scores.low : scores.high;
  }
  const summary = [indices.length / 4, spaceTime ? 8 : 1, 3, spaceTime ? 8 : 0];
  // Two unused clusters and two unused replicate rows must be cleared as well.
  return {
    scene,
    indices: [...indices, ...new Array(40 - indices.length).fill(0)],
    clusterStatistics: [...clusterStatistics, ...new Array(80 - clusterStatistics.length).fill(0)],
    zoneStatistics,
    statistics: [maximum, maximum, maximum, maximum, 0, 0],
    summary
  };
}
