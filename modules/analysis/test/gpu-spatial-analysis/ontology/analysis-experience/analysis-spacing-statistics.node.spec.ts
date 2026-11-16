// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import type {GPUAnalysisSpacingStatistics} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-experience-types';
import {GPU_EARTH_MEAN_RADIUS_METERS} from '../../../../src/analysis-core/contracts/earth';
import {
  GPU_ANALYSIS_DEFAULT_MAXIMUM_SAMPLE_SIZE,
  GPU_ANALYSIS_K_DISTANCE_QUANTILE_COUNT,
  computeGPUAnalysisSpacingStatistics as computeMaybeSpacingStatistics,
  getGPUAnalysisExtentSizes,
  getGPUAnalysisSamplingFactor
} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-spacing-statistics';

// The expected numbers marked ORACLE were produced by the Python scripts quoted at the end of this
// file (gen.py and oracle.py), run with /usr/bin/python3 (NumPy 2.0, SciPy 1.13): planar sets with
// scipy.spatial.cKDTree, geographic sets with a brute-force NumPy haversine on R = 6371008.8 m.
// The point sets come from the seeded generators below, which gen.py implements the same way
// (mulberry32 and Box-Muller), so no coordinates are pasted.

/** The statistics of input that has at least one valid point. */
function computeGPUAnalysisSpacingStatistics(
  coordinates: Float64Array | Float32Array | readonly number[],
  options: Parameters<typeof computeMaybeSpacingStatistics>[1]
): GPUAnalysisSpacingStatistics {
  const statistics = computeMaybeSpacingStatistics(coordinates, options);
  if (!statistics) {
    throw new Error('expected spacing statistics for at least one valid point');
  }
  return statistics;
}

/** mulberry32, a 32-bit seeded generator. */
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), state | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function createUniformPoints(seed: number, count: number, width = 1000, height = 1000): number[] {
  const random = createRandom(seed);
  const coordinates: number[] = [];
  for (let index = 0; index < count; index++) {
    const x = random() * width;
    const y = random() * height;
    coordinates.push(x, y);
  }
  return coordinates;
}

/** Four Gaussian clusters (sd 8) and 5 percent uniform noise over 1000 x 1000. */
function createClusteredPoints(seed: number, count: number): number[] {
  const random = createRandom(seed);
  const centers = [
    [200, 200],
    [800, 250],
    [500, 700],
    [250, 850]
  ];
  const normal = () => {
    const u = Math.max(random(), 1e-12);
    const v = random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const coordinates: number[] = [];
  for (let index = 0; index < count; index++) {
    if (index % 20 === 19) {
      const x = random() * 1000;
      const y = random() * 1000;
      coordinates.push(x, y);
    } else {
      const [centerX, centerY] = centers[index % 4];
      const x = centerX + 8 * normal();
      const y = centerY + 8 * normal();
      coordinates.push(x, y);
    }
  }
  return coordinates;
}

function createLatticePoints(): number[] {
  const coordinates: number[] = [];
  for (let row = 0; row < 30; row++) {
    for (let column = 0; column < 30; column++) {
      coordinates.push(10 * column, 10 * row);
    }
  }
  return coordinates;
}

/** 400 uniform points followed by two more copies of the first 100. */
function createDuplicatePoints(): number[] {
  const base = createUniformPoints(3, 400);
  return [...base, ...base.slice(0, 200), ...base.slice(0, 200)];
}

function createGeographicPoints(
  seed: number,
  count: number,
  longitudeMinimum: number,
  longitudeSpan: number,
  latitudeMinimum: number,
  latitudeSpan: number
): number[] {
  const random = createRandom(seed);
  const coordinates: number[] = [];
  for (let index = 0; index < count; index++) {
    let longitude = longitudeMinimum + longitudeSpan * random();
    const latitude = latitudeMinimum + latitudeSpan * random();
    if (longitude > 180) {
      longitude -= 360;
    }
    coordinates.push(longitude, latitude);
  }
  return coordinates;
}

// Brute force reference, O(n squared), independent of the grid hash.

const RADIANS_PER_DEGREE = Math.PI / 180;

function computeHaversine(
  longitudeA: number,
  latitudeA: number,
  longitudeB: number,
  latitudeB: number
): number {
  const latitudeRadiansA = latitudeA * RADIANS_PER_DEGREE;
  const latitudeRadiansB = latitudeB * RADIANS_PER_DEGREE;
  const halfLatitude = Math.sin((latitudeRadiansB - latitudeRadiansA) / 2);
  const halfLongitude = Math.sin(((longitudeB - longitudeA) * RADIANS_PER_DEGREE) / 2);
  const haversine =
    halfLatitude * halfLatitude +
    Math.cos(latitudeRadiansA) * Math.cos(latitudeRadiansB) * halfLongitude * halfLongitude;
  return 2 * GPU_EARTH_MEAN_RADIUS_METERS * Math.asin(Math.min(1, Math.sqrt(haversine)));
}

function computeBruteForceNeighbors(
  coordinates: readonly number[],
  model: 'planar' | 'geographic',
  ranks: readonly number[]
): {nearest: number[]; kth: Map<number, number[]>} {
  const count = coordinates.length / 2;
  const nearest: number[] = [];
  const kth = new Map<number, number[]>(ranks.map(rank => [rank, []]));
  for (let first = 0; first < count; first++) {
    const distances: number[] = [];
    for (let second = 0; second < count; second++) {
      if (second === first) {
        continue;
      }
      distances.push(
        model === 'planar'
          ? Math.hypot(
              coordinates[2 * first] - coordinates[2 * second],
              coordinates[2 * first + 1] - coordinates[2 * second + 1]
            )
          : computeHaversine(
              coordinates[2 * first],
              coordinates[2 * first + 1],
              coordinates[2 * second],
              coordinates[2 * second + 1]
            )
      );
    }
    distances.sort((a, b) => a - b);
    nearest.push(distances[0]);
    for (const rank of ranks) {
      kth.get(rank)?.push(distances[rank - 1]);
    }
  }
  for (const values of kth.values()) {
    values.sort((a, b) => a - b);
  }
  return {nearest, kth};
}

function reduceBruteForceQuantiles(sorted: readonly number[]): number[] {
  if (sorted.length <= GPU_ANALYSIS_K_DISTANCE_QUANTILE_COUNT) {
    return [...sorted];
  }
  return Array.from(
    {length: GPU_ANALYSIS_K_DISTANCE_QUANTILE_COUNT},
    (_, quantile) =>
      sorted[
        Math.round((quantile * (sorted.length - 1)) / (GPU_ANALYSIS_K_DISTANCE_QUANTILE_COUNT - 1))
      ]
  );
}

/** Fails when any number in the result, however deeply nested, is NaN. */
function expectNoNaN(value: unknown): void {
  if (typeof value === 'number') {
    expect(Number.isNaN(value)).toBe(false);
  } else if (value && typeof value === 'object') {
    Object.values(value).forEach(expectNoNaN);
  }
}

function expectClose(actual: number | undefined, expected: number, relativeTolerance = 1e-9): void {
  expect(actual).toBeDefined();
  if (actual === undefined) {
    return;
  }
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(
    relativeTolerance * Math.max(1, Math.abs(expected))
  );
}

// ORACLE values (see the scripts at the end of the file). `nearest` is [minimum, median, mean,
// maximum]; `kth` is [first, quarter, middle, last, sum] of the 3rd-neighbour quantile list of
// `kthCount` entries.
const ORACLE = {
  uniform: {
    n: 1500,
    nearest: [0.42004130901979136, 12.130534783693847, 12.993519812890716, 45.78487229149349],
    kthCount: 512,
    kth: [
      6.762087023187484, 19.28993370124454, 24.01197925949025, 51.048882636545414,
      12588.563916243724
    ],
    standardDistance: 405.32203911678783,
    medianFromCenter: 397.16239849455167,
    center: [491.94477307610214, 501.0213679516067]
  },
  clustered: {
    n: 1600,
    nearest: [0.019015047787379353, 0.7620352718116141, 3.6035334568121784, 153.2067984956858],
    kthCount: 512,
    kth: [
      0.30540522250046254, 1.074750925677896, 1.4242445130786132, 163.08111065166585,
      3216.4969704729624
    ],
    standardDistance: 367.4892339941968,
    medianFromCenter: 388.3230466416526,
    center: [451.1018680451244, 483.6122109973613]
  },
  lattice: {
    n: 900,
    nearest: [10.0, 10.0, 10.0, 10.0],
    kthCount: 512,
    kth: [10.0, 10.0, 10.0, 14.142135623730951, 5128.284271247461],
    standardDistance: 122.4064268465236,
    medianFromCenter: 120.20815280171308,
    center: [145.0, 145.0]
  },
  duplicates: {
    n: 600,
    nearest: [0.0, 1.579291674093144, 12.487583266221685, 89.74699624736897],
    kthCount: 512,
    kth: [
      1.8857242918759345, 17.835884497578384, 29.616989917144853, 109.46387010274701,
      16276.189672773013
    ],
    standardDistance: 403.23332001574136,
    medianFromCenter: 393.68096737047324,
    center: [515.209969568532, 495.44731951473904]
  },
  equator: {
    n: 300,
    nearest: [600.4361795763433, 3624.02511675212, 4201.7850012850695, 16347.508888349417],
    kthCount: 300,
    kth: [
      2617.3081439686684, 6407.121540512366, 8690.773166303854, 23901.170566195917,
      2620359.0613985043
    ],
    standardDistance: 69785.51719224115,
    medianFromCenter: 64961.198884409794,
    center: [10.901522057485689, -0.014234028972772279]
  },
  sixtyNorth: {
    n: 300,
    nearest: [52.36651894068164, 4396.262063571739, 4743.5443199711135, 18367.91960369888],
    kthCount: 300,
    kth: [
      3177.2740175578174, 6810.5145725995935, 8404.928052287865, 22299.02263193331,
      2672335.023300802
    ],
    standardDistance: 68906.06092813019,
    medianFromCenter: 59691.09269432818,
    center: [21.910802101166897, 59.98233723020577]
  },
  antimeridian: {
    n: 300,
    nearest: [117.79475493649512, 3962.579945320952, 4391.041229885398, 13550.698973919081],
    kthCount: 300,
    kth: [
      1016.3041052498087, 6409.048676801964, 8273.927901703282, 19960.829886150026,
      2539454.1077430863
    ],
    standardDistance: 70237.46185621448,
    medianFromCenter: 63355.72528264099,
    center: [179.98097705015985, 10.48274320266539]
  },
  pole: {
    n: 300,
    nearest: [162.74359835723467, 9963.715129628863, 11447.302818919346, 37517.34934106984],
    kthCount: 300,
    kth: [
      3843.8440044690374, 13460.715277426161, 20514.253614158682, 59029.237889238495,
      6396377.607949389
    ],
    standardDistance: 129906.29913379207,
    medianFromCenter: 110994.4900066279,
    center: [-53.36814449320127, 89.97457574210306]
  }
} as const;

type OracleEntry = {
  n: number;
  nearest: readonly number[];
  kthCount: number;
  kth: readonly number[];
  standardDistance: number;
  medianFromCenter: number;
  center: readonly number[];
};

function expectMatchesOracle(
  coordinates: number[],
  model: 'planar' | 'geographic',
  oracle: OracleEntry,
  tolerance: number
): void {
  const statistics = computeGPUAnalysisSpacingStatistics(coordinates, {coordinateModel: model});
  expect(statistics.pointCount).toBe(oracle.n);
  expect(statistics.sampled).toBe(false);
  expect(statistics.sampleSize).toBe(oracle.n);
  expect(statistics.unit).toBe(model === 'planar' ? 'coordinate-unit' : 'metre');
  const nearest = statistics.nearestNeighbor;
  expect(nearest).toBeDefined();
  expectClose(nearest?.minimum, oracle.nearest[0], tolerance);
  expectClose(nearest?.median, oracle.nearest[1], tolerance);
  expectClose(nearest?.mean, oracle.nearest[2], tolerance);
  expectClose(nearest?.maximum, oracle.nearest[3], tolerance);
  const distances = statistics.kDistance?.distances ?? [];
  expect(statistics.kDistance?.k).toBe(3);
  expect(distances.length).toBe(oracle.kthCount);
  expectClose(distances[0], oracle.kth[0], tolerance);
  expectClose(distances[Math.floor(distances.length / 4)], oracle.kth[1], tolerance);
  expectClose(distances[Math.floor(distances.length / 2)], oracle.kth[2], tolerance);
  expectClose(distances[distances.length - 1], oracle.kth[3], tolerance);
  expectClose(
    distances.reduce((sum, value) => sum + value, 0),
    oracle.kth[4],
    tolerance
  );
  for (let index = 1; index < distances.length; index++) {
    expect(distances[index]).toBeGreaterThanOrEqual(distances[index - 1]);
  }
  expectClose(statistics.standardDistance, oracle.standardDistance, tolerance);
  expectClose(statistics.medianDistanceFromCenter, oracle.medianFromCenter, tolerance);
  expectClose(statistics.center[0], oracle.center[0], tolerance);
  expectClose(statistics.center[1], oracle.center[1], tolerance);
}

describe('computeGPUAnalysisSpacingStatistics review regressions', () => {
  it.each([1e-200, 1e160, 1e307])('preserves planar distances at magnitude %s', magnitude => {
    const statistics = computeGPUAnalysisSpacingStatistics(
      [0, 0, magnitude, 0, 2 * magnitude, 0, 3 * magnitude, 0],
      {coordinateModel: 'planar'}
    );
    // /usr/bin/python3: cKDTree(np.array([[0,0],[1,0],[2,0],[3,0]])).query(..., k=4)
    // gives nearest [1,1,1,1], third [3,2,2,3]; NumPy RMS from the mean is 1.118033988749895.
    const nearest = statistics.nearestNeighbor!;
    for (const distance of [
      nearest.minimum,
      nearest.lowerQuartile,
      nearest.median,
      nearest.upperQuartile,
      nearest.mean,
      nearest.maximum,
      nearest.upperFence,
      nearest.nonZeroMedian!
    ]) {
      expect(distance / magnitude).toBeCloseTo(1, 12);
    }
    expect(statistics.standardDistance / magnitude).toBeCloseTo(1.118033988749895, 12);
    expect(statistics.kDistance?.distances).toHaveLength(4);
    for (const [index, expected] of [2, 2, 3, 3].entries()) {
      expect(statistics.kDistance!.distances[index] / magnitude).toBeCloseTo(expected, 12);
    }
    expectNoNaN(statistics);
  });

  it('preserves tiny separations beside a large constant coordinate', () => {
    const statistics = computeGPUAnalysisSpacingStatistics(
      [1e200, 0, 1e200, 1e-200, 1e200, 2e-200, 1e200, 3e-200],
      {coordinateModel: 'planar'}
    );
    expect(statistics.nearestNeighbor!.median / 1e-200).toBeCloseTo(1, 12);
    expect(statistics.standardDistance / 1e-200).toBeCloseTo(1.118033988749895, 12);
  });

  it('keeps a representable centre and spread when summing finite offsets overflows', () => {
    const statistics = computeGPUAnalysisSpacingStatistics(
      [-1e308, 0, 1e308, 0, -1e308, 0, 1e308, 0],
      {coordinateModel: 'planar'}
    );
    expect(statistics.center).toEqual([0, 0]);
    expect(statistics.standardDistance).toBe(1e308);
    expectNoNaN(statistics);
  });

  it('rescales the spread after aggregation when a single centre offset overflows', () => {
    const statistics = computeGPUAnalysisSpacingStatistics(
      [-1e308, 0, ...Array.from({length: 9}, () => [1e308, 0]).flat()],
      {coordinateModel: 'planar'}
    );
    // /usr/bin/python3: np.std(np.array([-1]+[1]*9)) gives 0.6000000000000001.
    expect(statistics.center[0] / 1e308).toBeCloseTo(0.8, 12);
    expect(statistics.standardDistance / 1e308).toBeCloseTo(0.6, 12);
    expect(statistics.medianDistanceFromCenter / 1e308).toBeCloseTo(0.2, 12);
    expectNoNaN(statistics);
  });

  it('computes the median over distinct locations regardless of multiplicity', () => {
    // /usr/bin/python3: points=np.array([[0,0],[3,0],[7,0],[15,0]]);
    // np.median(cKDTree(points).query(points,k=2)[0][:,1]) gives 3.5 from [3,3,4,8].
    const statistics = computeGPUAnalysisSpacingStatistics(
      [...Array.from({length: 10}, () => [0, 0]).flat(), 3, 0, 7, 0, 15, 0],
      {coordinateModel: 'planar'}
    );
    expect(statistics.nearestNeighbor?.median).toBe(0);
    expect(statistics.nearestNeighbor?.nonZeroMedian).toBe(3.5);
    expect(
      computeGPUAnalysisSpacingStatistics([0, 0, 0, 0], {coordinateModel: 'planar'}).nearestNeighbor
        ?.nonZeroMedian
    ).toBeUndefined();
    expect(
      computeGPUAnalysisSpacingStatistics([0, 0, -0, 0, 3, 0, 3, -0], {
        coordinateModel: 'planar'
      }).nearestNeighbor?.nonZeroMedian
    ).toBe(3);
  });

  it.each([
    0.001, 1, 1000
  ])('scales distinct-location spacing for forty repeated places by %s', scale => {
    // Reviewer c5's 8 by 5 grid, each location recorded ten times.
    // /usr/bin/python3: points=np.array([(i%8*100,i//8*100) for i in range(40)]);
    // np.median(cKDTree(points*scale).query(points*scale,k=2)[0][:,1]) = 100*scale.
    const locations = Array.from({length: 40}, (_, index) => [
      (index % 8) * 100 * scale,
      Math.floor(index / 8) * 100 * scale
    ]);
    const distinct = computeGPUAnalysisSpacingStatistics(locations.flat(), {
      coordinateModel: 'planar'
    });
    const repeated = computeGPUAnalysisSpacingStatistics(
      locations.flatMap(location => Array.from({length: 10}, () => location).flat()),
      {coordinateModel: 'planar'}
    );
    expect(repeated.duplicateLocationCount).toBe(360);
    expect(repeated.nearestNeighbor?.median).toBe(0);
    expect(repeated.nearestNeighbor?.nonZeroMedian).toBeCloseTo(100 * scale, 10);
    expect(repeated.nearestNeighbor?.nonZeroMedian).toBe(distinct.nearestNeighbor?.nonZeroMedian);
  });

  it.each([
    [40, 45],
    [60, 41]
  ])('computes distinct-location spacing within a sample of %s rows', (sampleSize, expectedMedian) => {
    // /usr/bin/python3, with NumPy and scipy.spatial.cKDTree:
    // def hash_index(index):
    //     value=index^0x9e3779b9
    //     value=((value^(value>>16))*0x85ebca6b)&0xffffffff
    //     value=((value^(value>>13))*0xc2b2ae35)&0xffffffff
    //     return value^(value>>16)
    // points=np.array([(i*i,0) for i in range(40)])
    // indices=sorted(range(400),key=hash_index)[:40]
    // sample=np.unique(np.repeat(points,10,axis=0)[indices],axis=0)
    // np.median(cKDTree(sample).query(sample,k=2)[0][:,1]) = 45 (29 distinct places),
    // With 60 sampled rows it is 41 (35 distinct places), compared with 38 for all 40 places.
    const coordinates = Array.from({length: 400}, (_, index) => [
      Math.floor(index / 10) ** 2,
      0
    ]).flat();
    const statistics = computeGPUAnalysisSpacingStatistics(coordinates, {
      coordinateModel: 'planar',
      maximumSampleSize: sampleSize
    });
    expect(statistics.sampled).toBe(true);
    expect(statistics.sampleSize).toBe(sampleSize);
    expect(statistics.nearestNeighbor?.nonZeroMedian).toBe(expectedMedian);
  });

  it('deduplicates geographic aliases before computing distinct-location spacing', () => {
    const statistics = computeGPUAnalysisSpacingStatistics(
      [180, 0, -180, 0, 180, 1, -180, 1, 0, 90, 123, 90],
      {coordinateModel: 'geographic'}
    );
    expect(statistics.duplicateLocationCount).toBe(3);
    expect(statistics.nearestNeighbor?.median).toBe(0);
    // Two equatorial locations are one meridian degree apart; the pole is much farther away.
    expect(statistics.nearestNeighbor?.nonZeroMedian).toBeCloseTo(
      GPU_EARTH_MEAN_RADIUS_METERS * RADIANS_PER_DEGREE,
      6
    );
  });

  it.each([
    [180, 0, -180, 0],
    [0, 0, 360, 0],
    [1e300, 90, -17, 90],
    [20, -90, 180, -90]
  ])('canonicalizes equivalent geographic locations %j', (...coordinates) => {
    const statistics = computeGPUAnalysisSpacingStatistics(coordinates, {
      coordinateModel: 'geographic'
    });
    expect(statistics.duplicateLocationCount).toBe(1);
    expect(statistics.nearestNeighbor).toMatchObject({minimum: 0, median: 0, maximum: 0});
    expect(statistics.standardDistance).toBe(0);
  });

  it('searches 20,000 alternating exact-pole points in well under a second', () => {
    const coordinates = Array.from({length: 20_000}, (_, index) => [
      (index * 360) / 20_000,
      index % 2 ? 90 : -90
    ]).flat();
    const start = performance.now();
    const statistics = computeGPUAnalysisSpacingStatistics(coordinates, {
      coordinateModel: 'geographic'
    });
    const elapsed = performance.now() - start;
    expect(elapsed).toBeLessThan(750);
    expect(statistics.duplicateLocationCount).toBe(19_998);
    expect(statistics.nearestNeighbor).toMatchObject({minimum: 0, median: 0, maximum: 0});
    expect(statistics.kDistance?.distances).toEqual(Array(512).fill(0));
  });
});

describe('computeGPUAnalysisSpacingStatistics against scipy and numpy oracles', () => {
  it('matches cKDTree on a uniform planar set of 1,500 points', () => {
    expectMatchesOracle(createUniformPoints(1, 1500), 'planar', ORACLE.uniform, 1e-9);
  });

  it('matches cKDTree on four tight clusters with noise (1,600 points)', () => {
    expectMatchesOracle(createClusteredPoints(2, 1600), 'planar', ORACLE.clustered, 1e-9);
  });

  it('matches cKDTree on a lattice with many equal distances', () => {
    const statistics = computeGPUAnalysisSpacingStatistics(createLatticePoints(), {
      coordinateModel: 'planar'
    });
    expectMatchesOracle(createLatticePoints(), 'planar', ORACLE.lattice, 1e-12);
    expect(statistics.nearestNeighbor).toMatchObject({
      minimum: 10,
      median: 10,
      mean: 10,
      maximum: 10
    });
    expect(statistics.duplicateLocationCount).toBe(0);
  });

  it('matches cKDTree on a set with repeated locations, and counts them', () => {
    const statistics = computeGPUAnalysisSpacingStatistics(createDuplicatePoints(), {
      coordinateModel: 'planar'
    });
    expectMatchesOracle(createDuplicatePoints(), 'planar', ORACLE.duplicates, 1e-9);
    expect(statistics.nearestNeighbor?.minimum).toBe(0);
    expect(statistics.duplicateLocationCount).toBe(200);
    expect(statistics.pointCount).toBe(600);
  });

  it('matches haversine at the equator', () => {
    expectMatchesOracle(
      createGeographicPoints(11, 300, 10, 2, -0.5, 1),
      'geographic',
      ORACLE.equator,
      1e-9
    );
  });

  it('matches haversine at 60 degrees north, where a degree of longitude is half as long', () => {
    expectMatchesOracle(
      createGeographicPoints(12, 300, 20, 4, 59.5, 1),
      'geographic',
      ORACLE.sixtyNorth,
      1e-9
    );
  });

  it('matches haversine across the antimeridian (longitudes 179 to 181 wrapped)', () => {
    const coordinates = createGeographicPoints(13, 300, 179, 2, 10, 1);
    expect(coordinates.some((value, index) => index % 2 === 0 && value > 179.5)).toBe(true);
    expect(coordinates.some((value, index) => index % 2 === 0 && value < -179.5)).toBe(true);
    expectMatchesOracle(coordinates, 'geographic', ORACLE.antimeridian, 1e-9);
  });

  it('matches haversine over the north pole, where every longitude meets', () => {
    expectMatchesOracle(
      createGeographicPoints(14, 300, -180, 360, 88, 2),
      'geographic',
      ORACLE.pole,
      1e-9
    );
  });
});

describe('computeGPUAnalysisSpacingStatistics against a brute-force implementation', () => {
  const cases: [string, number[], 'planar' | 'geographic'][] = [
    ['uniform planar', createUniformPoints(21, 2000), 'planar'],
    ['clustered planar', createClusteredPoints(22, 2000), 'planar'],
    [
      'planar with one far outlier',
      [...createUniformPoints(23, 1499, 100, 100), 1e6, -1e6],
      'planar'
    ],
    [
      'planar with a repeated location',
      [...createUniformPoints(24, 1000), ...Array.from({length: 200}, () => 500)],
      'planar'
    ],
    [
      'geographic, whole globe',
      createGeographicPoints(25, 2000, -180, 360, -90, 180),
      'geographic'
    ],
    [
      'geographic, high latitude strip',
      createGeographicPoints(26, 2000, -180, 360, 78, 12),
      'geographic'
    ],
    [
      'geographic, across the antimeridian',
      createGeographicPoints(27, 2000, 175, 10, -30, 40),
      'geographic'
    ],
    ['geographic, city sized', createGeographicPoints(28, 2000, -74, 0.4, 40.5, 0.3), 'geographic']
  ];
  for (const [name, coordinates, model] of cases) {
    it(`finds the same neighbour distances: ${name}`, () => {
      const ranks = [1, 3, 7];
      const reference = computeBruteForceNeighbors(coordinates, model, ranks);
      const sortedNearest = [...reference.nearest].sort((a, b) => a - b);
      const count = sortedNearest.length;
      const median =
        count % 2 === 1
          ? sortedNearest[(count - 1) / 2]
          : (sortedNearest[count / 2 - 1] + sortedNearest[count / 2]) / 2;
      const mean = reference.nearest.reduce((sum, value) => sum + value, 0) / count;
      for (const rank of ranks) {
        const statistics = computeGPUAnalysisSpacingStatistics(coordinates, {
          coordinateModel: model,
          neighborRank: rank
        });
        expect(statistics.pointCount).toBe(count);
        expect(statistics.nearestNeighbor?.minimum).toBeCloseTo(sortedNearest[0], 6);
        expectClose(statistics.nearestNeighbor?.maximum, sortedNearest[count - 1], 1e-9);
        expectClose(statistics.nearestNeighbor?.median, median, 1e-9);
        expectClose(statistics.nearestNeighbor?.mean, mean, 1e-9);
        const expected = reduceBruteForceQuantiles(reference.kth.get(rank) ?? []);
        expect(statistics.kDistance?.k).toBe(rank);
        expect(statistics.kDistance?.distances.length).toBe(expected.length);
        statistics.kDistance?.distances.forEach((distance, index) => {
          expectClose(distance, expected[index], 1e-9);
        });
      }
    });
  }
});

describe('computeGPUAnalysisSpacingStatistics geographic behaviour', () => {
  it('measures two points on either side of the antimeridian as close', () => {
    const statistics = computeGPUAnalysisSpacingStatistics([179.5, 0, -179.5, 0], {
      coordinateModel: 'geographic'
    });
    const oneDegree = (GPU_EARTH_MEAN_RADIUS_METERS * Math.PI) / 180;
    expectClose(statistics.nearestNeighbor?.minimum, oneDegree, 1e-9);
    // The mean centre is on the antimeridian, and each point is half a degree from it.
    expect(Math.abs(Math.abs(statistics.center[0]) - 180)).toBeLessThan(1e-9);
    expectClose(statistics.center[1], 0, 1e-9);
    expectClose(statistics.standardDistance, oneDegree / 2, 1e-9);
    // The extent is the arc across 180: west is greater than east.
    expect(statistics.extent).toEqual([179.5, 0, -179.5, 0]);
  });

  it('measures two points over the pole as close', () => {
    const statistics = computeGPUAnalysisSpacingStatistics([0, 89.99, 180, 89.99], {
      coordinateModel: 'geographic'
    });
    const expected = (GPU_EARTH_MEAN_RADIUS_METERS * 0.02 * Math.PI) / 180;
    expectClose(statistics.nearestNeighbor?.minimum, expected, 1e-9);
  });

  it('finds neighbours across grid cell borders at 85 degrees north', () => {
    // 1,000 points on a ring of constant latitude: neighbours are on the same parallel, where a
    // degree of longitude is 0.087 of a degree of arc. The nearest neighbour of each is the next
    // point east, at exactly one step.
    const coordinates: number[] = [];
    for (let index = 0; index < 1000; index++) {
      coordinates.push(-180 + (360 * index) / 1000, 85);
    }
    const statistics = computeGPUAnalysisSpacingStatistics(coordinates, {
      coordinateModel: 'geographic'
    });
    const step = computeHaversine(0, 85, 0.36, 85);
    expectClose(statistics.nearestNeighbor?.minimum, step, 1e-9);
    expectClose(statistics.nearestNeighbor?.maximum, step, 1e-9);
    expectClose(
      statistics.nearestNeighbor?.maximum,
      statistics.nearestNeighbor?.minimum ?? NaN,
      1e-9
    );
  });

  it('reports extent sizes in metres: arc along the middle parallel, meridian arc and their hypotenuse', () => {
    const statistics = computeGPUAnalysisSpacingStatistics([0, 0, 1, 0, 0, 1], {
      coordinateModel: 'geographic'
    });
    const oneDegree = (GPU_EARTH_MEAN_RADIUS_METERS * Math.PI) / 180;
    expectClose(statistics.extentWidth, oneDegree * Math.cos(0.5 * RADIANS_PER_DEGREE), 1e-12);
    expectClose(statistics.extentHeight, oneDegree, 1e-12);
    expectClose(
      statistics.extentDiagonal,
      Math.hypot(statistics.extentWidth, statistics.extentHeight),
      1e-12
    );
  });

  it('skips a row whose latitude is outside [-90, 90]', () => {
    const statistics = computeGPUAnalysisSpacingStatistics([0, 0, 1, 0, 5, 95, 6, -91], {
      coordinateModel: 'geographic'
    });
    expect(statistics.pointCount).toBe(2);
  });

  it('falls back to the mean longitude and latitude when the unit vectors cancel out', () => {
    const statistics = computeGPUAnalysisSpacingStatistics([0, 0, 180, 0], {
      coordinateModel: 'geographic'
    });
    expect(statistics.center).toEqual([90, 0]);
    expectClose(statistics.nearestNeighbor?.minimum, Math.PI * GPU_EARTH_MEAN_RADIUS_METERS, 1e-9);
  });
});

describe('computeGPUAnalysisSpacingStatistics degenerate inputs', () => {
  it('0 valid points: returns undefined, with no NaN anywhere', () => {
    for (const input of [[], new Float64Array(0), [Number.NaN, 1, 2, Number.POSITIVE_INFINITY]]) {
      expect(computeMaybeSpacingStatistics(input, {coordinateModel: 'planar'})).toBeUndefined();
      expect(computeMaybeSpacingStatistics(input, {coordinateModel: 'geographic'})).toBeUndefined();
    }
  });

  it('1 point: zero spread, no neighbour', () => {
    const statistics = computeGPUAnalysisSpacingStatistics([3, 4], {coordinateModel: 'planar'});
    expect(statistics.pointCount).toBe(1);
    expect(statistics.extent).toEqual([3, 4, 3, 4]);
    expect(statistics.center).toEqual([3, 4]);
    expect(statistics.extentWidth).toBe(0);
    expect(statistics.extentDiagonal).toBe(0);
    expect(statistics.standardDistance).toBe(0);
    expect(statistics.medianDistanceFromCenter).toBe(0);
    expect(statistics.nearestNeighbor).toBeUndefined();
    expect(statistics.kDistance).toBeUndefined();
    expect(statistics.duplicateLocationCount).toBe(0);
    expectNoNaN(statistics);
  });

  it('1 geographic point gives the same defined result', () => {
    const statistics = computeGPUAnalysisSpacingStatistics([-74, 40.7], {
      coordinateModel: 'geographic'
    });
    expect(statistics.pointCount).toBe(1);
    expect(statistics.standardDistance).toBeCloseTo(0, 6);
    expect(statistics.nearestNeighbor).toBeUndefined();
    expectNoNaN(statistics);
    expectClose(statistics.center[0], -74, 1e-9);
    expectClose(statistics.center[1], 40.7, 1e-9);
  });

  it('all points identical: zero spread, zero distances, duplicates counted', () => {
    for (const model of ['planar', 'geographic'] as const) {
      const statistics = computeGPUAnalysisSpacingStatistics(
        Array.from({length: 40}, (_, index) => (index % 2 === 0 ? 10 : 20)),
        {coordinateModel: model}
      );
      expect(statistics.pointCount).toBe(20);
      expect(statistics.extentWidth).toBe(0);
      expect(statistics.extentHeight).toBe(0);
      expect(statistics.standardDistance).toBeCloseTo(0, 6);
      expect(statistics.nearestNeighbor).toMatchObject({
        minimum: 0,
        median: 0,
        mean: 0,
        maximum: 0,
        upperFence: 0,
        aboveUpperFenceCount: 0
      });
      expect(statistics.kDistance?.distances.every(value => value === 0)).toBe(true);
      expect(statistics.duplicateLocationCount).toBe(19);
    }
  });

  it('collinear points on a horizontal, vertical and diagonal line', () => {
    const lines: [string, (index: number) => [number, number], number][] = [
      ['horizontal', index => [index, 5], 1],
      ['vertical', index => [5, index], 1],
      ['diagonal', index => [index, index], Math.SQRT2]
    ];
    for (const [, place, step] of lines) {
      const coordinates = Array.from({length: 500}, (_, index) => place(index)).flat();
      const statistics = computeGPUAnalysisSpacingStatistics(coordinates, {
        coordinateModel: 'planar'
      });
      expectClose(statistics.nearestNeighbor?.minimum, step, 1e-12);
      expectClose(statistics.nearestNeighbor?.maximum, step, 1e-12);
      // The third neighbour of an interior point is at 2 steps (one side has two, the other one);
      // the end points reach 3 steps.
      expectClose(statistics.kDistance?.distances[0] ?? NaN, 2 * step, 1e-12);
      expectClose(statistics.kDistance?.distances.at(-1) ?? NaN, 3 * step, 1e-12);
    }
  });

  it('fewer than k + 1 points: nearest neighbours exist, kDistance is absent', () => {
    const three = computeGPUAnalysisSpacingStatistics([0, 0, 1, 0, 0, 5], {
      coordinateModel: 'planar'
    });
    expect(three.nearestNeighbor).toMatchObject({
      minimum: 1,
      median: 1,
      mean: (1 + 1 + 5) / 3,
      maximum: 5
    });
    expect(three.kDistance).toBeUndefined();
    const four = computeGPUAnalysisSpacingStatistics([0, 0, 1, 0, 0, 5, 3, 3], {
      coordinateModel: 'planar'
    });
    expect(four.kDistance?.distances.length).toBe(4);
    const two = computeGPUAnalysisSpacingStatistics([0, 0, 3, 4], {coordinateModel: 'planar'});
    expect(two.nearestNeighbor).toMatchObject({minimum: 5, median: 5, mean: 5, maximum: 5});
    expect(two.kDistance).toBeUndefined();
    const rankOne = computeGPUAnalysisSpacingStatistics([0, 0, 3, 4], {
      coordinateModel: 'planar',
      neighborRank: 1
    });
    expect(rankOne.kDistance).toEqual({k: 1, distances: [5, 5]});
  });

  it('skips rows with a non-finite coordinate and does not count them', () => {
    const statistics = computeGPUAnalysisSpacingStatistics(
      [
        0,
        0,
        3,
        4,
        Number.NaN,
        1,
        2,
        Number.POSITIVE_INFINITY,
        Number.NEGATIVE_INFINITY,
        0,
        100,
        100
      ],
      {coordinateModel: 'planar'}
    );
    expect(statistics.pointCount).toBe(3);
    expect(statistics.extent).toEqual([0, 0, 100, 100]);
    expect(statistics.nearestNeighbor?.minimum).toBe(5);
    expect(
      computeMaybeSpacingStatistics([Number.NaN, 1, 2, Number.NaN], {coordinateModel: 'planar'})
    ).toBeUndefined();
  });

  it('treats -0 and 0 as one location', () => {
    const statistics = computeGPUAnalysisSpacingStatistics([0, 0, -0, -0, 1, 1], {
      coordinateModel: 'planar'
    });
    expect(statistics.duplicateLocationCount).toBe(1);
    expect(statistics.nearestNeighbor?.minimum).toBe(0);
  });

  it('accepts Float32Array and plain arrays', () => {
    const plain = [0, 0, 3, 4, 6, 8];
    const fromFloat32 = computeGPUAnalysisSpacingStatistics(new Float32Array(plain), {
      coordinateModel: 'planar'
    });
    const fromArray = computeGPUAnalysisSpacingStatistics(plain, {coordinateModel: 'planar'});
    expect(fromFloat32).toEqual(fromArray);
  });

  it('keeps precision for large coordinates with a small spread', () => {
    const statistics = computeGPUAnalysisSpacingStatistics(
      [1e9, 1e9, 1e9 + 1, 1e9, 1e9 + 2, 1e9 + 0.5, 1e9 + 40, 1e9 + 40],
      {coordinateModel: 'planar'}
    );
    expect(statistics.nearestNeighbor?.minimum).toBe(1);
    expectClose(statistics.center[0], 1e9 + 10.75, 1e-12);
  });

  it('does not take the square of a location repeated many times', () => {
    const coordinates: number[] = [];
    for (let index = 0; index < 60000; index++) {
      coordinates.push(7, 7);
    }
    coordinates.push(8, 7, 9, 9, 20, 20, 21, 20);
    const started = performance.now();
    const statistics = computeGPUAnalysisSpacingStatistics(coordinates, {
      coordinateModel: 'planar',
      maximumSampleSize: 70000
    });
    expect(performance.now() - started).toBeLessThan(3000);
    expect(statistics.duplicateLocationCount).toBe(59999);
    expect(statistics.nearestNeighbor?.minimum).toBe(0);
    expect(statistics.nearestNeighbor?.maximum).toBeGreaterThan(0);
  });
});

describe('computeGPUAnalysisSpacingStatistics validation', () => {
  it('throws for an odd coordinate length, naming the function and the argument', () => {
    expect(() =>
      computeGPUAnalysisSpacingStatistics([1, 2, 3], {coordinateModel: 'planar'})
    ).toThrow(/computeGPUAnalysisSpacingStatistics coordinates must hold x, y pairs/);
  });

  it('throws for an invalid neighborRank and maximumSampleSize at their limits', () => {
    const run = (options: {neighborRank?: number; maximumSampleSize?: number}) => () =>
      computeGPUAnalysisSpacingStatistics([0, 0, 1, 1], {coordinateModel: 'planar', ...options});
    expect(run({neighborRank: 0})).toThrow(/neighborRank must be an integer from 1 to 64; got 0/);
    expect(run({neighborRank: 65})).toThrow(/neighborRank/);
    expect(run({neighborRank: 1.5})).toThrow(/neighborRank/);
    expect(run({neighborRank: Number.NaN})).toThrow(/neighborRank/);
    expect(run({neighborRank: 1})).not.toThrow();
    expect(run({neighborRank: 64})).not.toThrow();
    expect(run({maximumSampleSize: 1})).toThrow(
      /maximumSampleSize must be an integer of at least 2; got 1/
    );
    expect(run({maximumSampleSize: 2.5})).toThrow(/maximumSampleSize/);
    expect(run({maximumSampleSize: 2})).not.toThrow();
  });

  it('supports the largest rank with just enough points', () => {
    const coordinates = Array.from({length: 65}, (_, index) => [index, 0]).flat();
    const statistics = computeGPUAnalysisSpacingStatistics(coordinates, {
      coordinateModel: 'planar',
      neighborRank: 64
    });
    expect(statistics.kDistance?.k).toBe(64);
    expect(statistics.kDistance?.distances.length).toBe(65);
    expect(statistics.kDistance?.distances.at(-1)).toBe(64);
  });
});

describe('computeGPUAnalysisSpacingStatistics k-distance reduction', () => {
  it('reduces 1,500 values to 512 ascending quantiles that keep the smallest and the largest', () => {
    const statistics = computeGPUAnalysisSpacingStatistics(createUniformPoints(1, 1500), {
      coordinateModel: 'planar'
    });
    expect(GPU_ANALYSIS_K_DISTANCE_QUANTILE_COUNT).toBe(512);
    expect(statistics.kDistance?.distances.length).toBe(512);
  });

  it('keeps exactly 512 values unreduced and reduces 513', () => {
    const exact = computeGPUAnalysisSpacingStatistics(createUniformPoints(5, 512), {
      coordinateModel: 'planar'
    });
    const above = computeGPUAnalysisSpacingStatistics(createUniformPoints(5, 513), {
      coordinateModel: 'planar'
    });
    expect(exact.kDistance?.distances.length).toBe(512);
    expect(above.kDistance?.distances.length).toBe(512);
  });
});

describe('computeGPUAnalysisSpacingStatistics sampling', () => {
  it('has a documented default limit', () => {
    expect(GPU_ANALYSIS_DEFAULT_MAXIMUM_SAMPLE_SIZE).toBe(100_000);
  });

  it('samples the rows with the smallest index hashes (reproduced by the Python oracle)', () => {
    // sample.py: sample(10, 5) = [1, 4, 6, 7, 8]; a fixed stride would take 0, 2, 4, 6, 8. Row i
    // sits at x = i * i, so the distances identify the rows: the sample is x = 1, 16, 36, 49, 64
    // with nearest-neighbour distances 15, 15, 13, 13, 15 and 3rd-neighbour distances 48, 33, 28,
    // 33, 48. The extent and the count use all ten rows.
    const coordinates: number[] = [];
    for (let row = 0; row < 10; row++) {
      coordinates.push(row * row, 0);
    }
    const statistics = computeGPUAnalysisSpacingStatistics(coordinates, {
      coordinateModel: 'planar',
      maximumSampleSize: 5
    });
    expect(statistics.sampled).toBe(true);
    expect(statistics.sampleSize).toBe(5);
    expect(statistics.pointCount).toBe(10);
    expect(statistics.nearestNeighbor).toMatchObject({
      minimum: 13,
      median: 15,
      mean: 14.2,
      maximum: 15
    });
    expect(statistics.kDistance?.distances).toEqual([28, 33, 33, 48, 48]);
    expect(statistics.extent).toEqual([0, 0, 81, 0]);
  });

  it('selects 100 of 1,000 rows exactly as the oracle does', () => {
    // sample.py: sample(1000, 100), the sorted row indices below. Rows sit at x = row, so the
    // expected nearest-neighbour statistics follow from the oracle's indices alone.
    const selected = [
      8, 25, 39, 41, 59, 72, 76, 80, 85, 93, 141, 158, 182, 196, 197, 202, 225, 228, 231, 240, 255,
      260, 283, 284, 289, 296, 319, 326, 348, 351, 356, 365, 367, 368, 374, 385, 391, 394, 402, 414,
      420, 424, 442, 452, 457, 460, 467, 480, 510, 513, 518, 527, 533, 547, 556, 557, 559, 574, 580,
      594, 596, 622, 625, 628, 632, 637, 638, 660, 661, 662, 670, 672, 678, 682, 690, 698, 718, 722,
      726, 739, 740, 746, 748, 766, 773, 799, 803, 804, 807, 821, 835, 876, 903, 923, 962, 964, 974,
      982, 983, 990
    ];
    const coordinates: number[] = [];
    for (let row = 0; row < 1000; row++) {
      coordinates.push(row, 0);
    }
    const statistics = computeGPUAnalysisSpacingStatistics(coordinates, {
      coordinateModel: 'planar',
      maximumSampleSize: 100
    });
    const nearest = selected.map((value, index) =>
      Math.min(
        index > 0 ? value - selected[index - 1] : Number.POSITIVE_INFINITY,
        index < selected.length - 1 ? selected[index + 1] - value : Number.POSITIVE_INFINITY
      )
    );
    expect(statistics.sampleSize).toBe(100);
    expect(statistics.nearestNeighbor?.maximum).toBe(Math.max(...nearest));
    expect(statistics.nearestNeighbor?.minimum).toBe(Math.min(...nearest));
    expectClose(
      statistics.nearestNeighbor?.mean,
      nearest.reduce((sum, value) => sum + value, 0) / 100,
      1e-12
    );
    expect(selected.reduce((sum, value) => sum + value, 0)).toBe(50130);
  });

  it('is deterministic', () => {
    const coordinates = createUniformPoints(31, 4000);
    const options = {coordinateModel: 'planar' as const, maximumSampleSize: 700};
    expect(computeGPUAnalysisSpacingStatistics(coordinates, options)).toEqual(
      computeGPUAnalysisSpacingStatistics(coordinates, options)
    );
  });

  it('computes extent, centre, spread and duplicates on every point, not the sample', () => {
    const coordinates = createUniformPoints(32, 4000);
    const full = computeGPUAnalysisSpacingStatistics(coordinates, {coordinateModel: 'planar'});
    const sampled = computeGPUAnalysisSpacingStatistics(coordinates, {
      coordinateModel: 'planar',
      maximumSampleSize: 400
    });
    expect(sampled.sampled).toBe(true);
    expect(sampled.pointCount).toBe(4000);
    expect(sampled.extent).toEqual(full.extent);
    expect(sampled.center).toEqual(full.center);
    expect(sampled.standardDistance).toBe(full.standardDistance);
    expect(sampled.medianDistanceFromCenter).toBe(full.medianDistanceFromCenter);
    expect(sampled.duplicateLocationCount).toBe(full.duplicateLocationCount);
  });

  it('is not sampled at exactly the limit and is sampled one above it', () => {
    const atLimit = computeGPUAnalysisSpacingStatistics(createUniformPoints(33, 300), {
      coordinateModel: 'planar',
      maximumSampleSize: 300
    });
    const above = computeGPUAnalysisSpacingStatistics(createUniformPoints(33, 301), {
      coordinateModel: 'planar',
      maximumSampleSize: 300
    });
    expect(atLimit.sampled).toBe(false);
    expect(above.sampled).toBe(true);
    expect(above.sampleSize).toBe(300);
  });

  it('shows the bias: sampled distances run high by about sqrt(n / m) for a uniform pattern', () => {
    const coordinates = createUniformPoints(34, 8000);
    const full = computeGPUAnalysisSpacingStatistics(coordinates, {coordinateModel: 'planar'});
    const sampled = computeGPUAnalysisSpacingStatistics(coordinates, {
      coordinateModel: 'planar',
      maximumSampleSize: 2000
    });
    expect(sampled.nearestNeighbor?.median ?? 0).toBeGreaterThan(
      full.nearestNeighbor?.median ?? Infinity
    );
    const factor = getGPUAnalysisSamplingFactor(8000, 2000);
    expect(factor).toBe(0.5);
    const corrected = (sampled.nearestNeighbor?.median ?? NaN) * factor;
    expect(Math.abs(corrected / (full.nearestNeighbor?.median ?? NaN) - 1)).toBeLessThan(0.08);
    const sampledKth = sampled.kDistance?.distances[255] ?? NaN;
    const fullKth = full.kDistance?.distances[255] ?? NaN;
    expect(Math.abs((sampledKth * factor) / fullKth - 1)).toBeLessThan(0.08);
  });

  it('getGPUAnalysisSamplingFactor is 1 without sampling and rejects impossible sizes', () => {
    expect(getGPUAnalysisSamplingFactor(500, 500)).toBe(1);
    expect(() => getGPUAnalysisSamplingFactor(500, 0)).toThrow(
      /sampleSize must be an integer from 1 to pointCount \(500\)/
    );
    expect(() => getGPUAnalysisSamplingFactor(500, 501)).toThrow(RangeError);
    expect(() => getGPUAnalysisSamplingFactor(500.5, 10)).toThrow(/integers/);
  });
});

describe('computeGPUAnalysisSpacingStatistics speed', () => {
  // Loose limits so that a slow machine does not fail; the measured times are in the doc comment of
  // GPU_ANALYSIS_DEFAULT_MAXIMUM_SAMPLE_SIZE.
  it('computes 20,000 points in well under a second', () => {
    const coordinates = createUniformPoints(41, 20000);
    const started = performance.now();
    const statistics = computeGPUAnalysisSpacingStatistics(coordinates, {
      coordinateModel: 'planar'
    });
    expect(performance.now() - started).toBeLessThan(1000);
    expect(statistics.sampled).toBe(false);
  });

  it('computes 200,000 points (sampled to the default limit) in under two seconds', () => {
    const coordinates = createUniformPoints(42, 200000);
    const started = performance.now();
    const statistics = computeGPUAnalysisSpacingStatistics(coordinates, {
      coordinateModel: 'planar'
    });
    expect(performance.now() - started).toBeLessThan(2000);
    expect(statistics.sampled).toBe(true);
    expect(statistics.sampleSize).toBe(GPU_ANALYSIS_DEFAULT_MAXIMUM_SAMPLE_SIZE);
  });

  it('keeps a far outlier among dense clusters exact and fast', () => {
    const coordinates = [...createClusteredPoints(43, 20000), 1e6, 1e6];
    const started = performance.now();
    const statistics = computeGPUAnalysisSpacingStatistics(coordinates, {
      coordinateModel: 'planar'
    });
    expect(performance.now() - started).toBeLessThan(2000);
    expect(statistics.nearestNeighbor?.maximum).toBeGreaterThan(1e5);
  });
});

describe('computeGPUAnalysisSpacingStatistics quartiles and the upper fence', () => {
  it('matches numpy percentile on the uniform and the clustered sets', () => {
    // quart.py: np.percentile(nn, [25, 75]); fence = min(max, q3 + 3 * (q3 - q1)); above = count(nn > fence).
    const uniform = computeGPUAnalysisSpacingStatistics(createUniformPoints(1, 1500), {
      coordinateModel: 'planar'
    });
    expectClose(uniform.nearestNeighbor?.lowerQuartile, 7.63850848257959, 1e-9);
    expectClose(uniform.nearestNeighbor?.upperQuartile, 17.381929212035754, 1e-9);
    expectClose(uniform.nearestNeighbor?.upperFence, 45.78487229149349, 1e-9);
    // The fence is capped at the maximum, so nothing lies above it.
    expect(uniform.nearestNeighbor?.upperFence).toBe(uniform.nearestNeighbor?.maximum);
    expect(uniform.nearestNeighbor?.aboveUpperFenceCount).toBe(0);
    const clustered = computeGPUAnalysisSpacingStatistics(createClusteredPoints(2, 1600), {
      coordinateModel: 'planar'
    });
    expectClose(clustered.nearestNeighbor?.lowerQuartile, 0.4777907912902482, 1e-9);
    expectClose(clustered.nearestNeighbor?.upperQuartile, 1.3178779079389908, 1e-9);
    expectClose(clustered.nearestNeighbor?.upperFence, 3.838139257885219, 1e-9);
    expect(clustered.nearestNeighbor?.aboveUpperFenceCount).toBe(111);
  });

  it('counts a single isolated point above the fence', () => {
    const coordinates = [...createUniformPoints(8, 200, 10, 10), 1e6, 1e6];
    const statistics = computeGPUAnalysisSpacingStatistics(coordinates, {
      coordinateModel: 'planar'
    });
    expect(statistics.nearestNeighbor?.aboveUpperFenceCount).toBe(1);
    expect(statistics.nearestNeighbor?.upperFence).toBeLessThan(5);
    expectClose(statistics.nearestNeighbor?.maximum, Math.hypot(1e6 - 10, 1e6 - 10), 1e-3);
  });

  it('keeps the minimum at 0 when the full set has a repeated location the sample misses', () => {
    const coordinates = [...createUniformPoints(9, 3000), 500, 500, 500, 500];
    // The pair of 500, 500 rows sits at the end; the sample of 100 almost surely holds neither.
    const statistics = computeGPUAnalysisSpacingStatistics(coordinates, {
      coordinateModel: 'planar',
      maximumSampleSize: 100
    });
    expect(statistics.sampled).toBe(true);
    expect(statistics.duplicateLocationCount).toBe(1);
    expect(statistics.nearestNeighbor?.minimum).toBe(0);
  });
});

describe('computeGPUAnalysisSpacingStatistics layouts that defeated the first projection axis', () => {
  const layouts: [string, (count: number) => number[]][] = [
    [
      'random points on longitudes 0 and 180 (a meridian great circle)',
      count => {
        const random = createRandom(7);
        const coordinates: number[] = [];
        for (let index = 0; index < count; index++) {
          coordinates.push(random() < 0.5 ? 0 : 180, random() * 180 - 90);
        }
        return coordinates;
      }
    ],
    [
      'random points on one meridian',
      count => {
        const random = createRandom(8);
        const coordinates: number[] = [];
        for (let index = 0; index < count; index++) {
          coordinates.push(30, random() * 180 - 90);
        }
        return coordinates;
      }
    ],
    [
      'evenly spaced points on a meridian great circle',
      count => {
        const coordinates: number[] = [];
        for (let index = 0; index < count; index++) {
          const angle = (2 * Math.PI * index) / count;
          coordinates.push(
            Math.cos(angle) >= 0 ? 0 : 180,
            (Math.asin(Math.sin(angle)) * 180) / Math.PI
          );
        }
        return coordinates;
      }
    ],
    [
      'random points on two meridians 120 degrees apart',
      count => {
        const random = createRandom(9);
        const coordinates: number[] = [];
        for (let index = 0; index < count; index++) {
          coordinates.push(random() < 0.5 ? 10 : -170, random() * 120 - 60);
        }
        return coordinates;
      }
    ],
    [
      'points along the equator',
      count => {
        const random = createRandom(10);
        const coordinates: number[] = [];
        for (let index = 0; index < count; index++) {
          coordinates.push(random() * 360 - 180, 0);
        }
        return coordinates;
      }
    ]
  ];
  for (const [name, create] of layouts) {
    it(`is exact against brute force: ${name}`, () => {
      const coordinates = create(1500);
      const reference = computeBruteForceNeighbors(coordinates, 'geographic', [3]);
      const statistics = computeGPUAnalysisSpacingStatistics(coordinates, {
        coordinateModel: 'geographic'
      });
      const sortedNearest = [...reference.nearest].sort((a, b) => a - b);
      expectClose(statistics.nearestNeighbor?.maximum, sortedNearest[1499], 1e-9);
      expectClose(
        statistics.nearestNeighbor?.median,
        (sortedNearest[749] + sortedNearest[750]) / 2,
        1e-9
      );
      const expected = reduceBruteForceQuantiles(reference.kth.get(3) ?? []);
      statistics.kDistance?.distances.forEach((distance, index) => {
        expectClose(distance, expected[index], 1e-9);
      });
    });
    it(`takes well under a second for 16,000 points: ${name}`, () => {
      const coordinates = create(16000);
      const started = performance.now();
      const statistics = computeGPUAnalysisSpacingStatistics(coordinates, {
        coordinateModel: 'geographic'
      });
      // Measured 15 to 35 ms; the failure before the fix was 34 s for 4,000 points.
      expect(performance.now() - started).toBeLessThan(1500);
      expect(statistics.pointCount).toBe(16000);
    });
  }
});

describe('computeGPUAnalysisSpacingStatistics extent across the antimeridian', () => {
  /** A 4 by 10 degree box at latitudes 60 to 70, on a 20 x 20 grid, centred on `centerLongitude`. */
  function createBox(centerLongitude: number): number[] {
    const coordinates: number[] = [];
    for (let column = 0; column < 20; column++) {
      for (let row = 0; row < 20; row++) {
        let longitude = centerLongitude - 2 + (4 * column) / 19;
        if (longitude > 180) {
          longitude -= 360;
        }
        coordinates.push(longitude, 60 + (10 * row) / 19);
      }
    }
    return coordinates;
  }

  it('gives identical boxes at longitude 0 and 180 the same width, height and diagonal', () => {
    const atZero = computeGPUAnalysisSpacingStatistics(createBox(0), {
      coordinateModel: 'geographic'
    });
    const atAntimeridian = computeGPUAnalysisSpacingStatistics(createBox(180), {
      coordinateModel: 'geographic'
    });
    expectClose(atAntimeridian.extentWidth, atZero.extentWidth, 1e-9);
    expectClose(atAntimeridian.extentHeight, atZero.extentHeight, 1e-9);
    expectClose(atAntimeridian.extentDiagonal, atZero.extentDiagonal, 1e-9);
    const expectedWidth =
      GPU_EARTH_MEAN_RADIUS_METERS * Math.cos(65 * RADIANS_PER_DEGREE) * 4 * RADIANS_PER_DEGREE;
    expectClose(atAntimeridian.extentWidth, expectedWidth, 1e-9);
    // About 188 km, not the 16,920 km of the plain longitude range.
    expect(atAntimeridian.extentWidth).toBeGreaterThan(180000);
    expect(atAntimeridian.extentWidth).toBeLessThan(195000);
  });

  it('reports west greater than east when the arc crosses 180, and plain ends otherwise', () => {
    const atAntimeridian = computeGPUAnalysisSpacingStatistics(createBox(180), {
      coordinateModel: 'geographic'
    });
    expectClose(atAntimeridian.extent[0], 178, 1e-12);
    expectClose(atAntimeridian.extent[2], -178, 1e-12);
    expect(atAntimeridian.extent[1]).toBe(60);
    expect(atAntimeridian.extent[3]).toBe(70);
    const atZero = computeGPUAnalysisSpacingStatistics(createBox(0), {
      coordinateModel: 'geographic'
    });
    expectClose(atZero.extent[0], -2, 1e-12);
    expectClose(atZero.extent[2], 2, 1e-12);
  });

  it('measures a 60 degree wide strip that crosses 180 and keeps source longitudes above 180', () => {
    const coordinates: number[] = [];
    for (let index = 0; index <= 60; index++) {
      let longitude = 150 + index;
      if (longitude > 180) {
        longitude -= 360;
      }
      coordinates.push(longitude, 60 + (index % 2));
    }
    const strip = computeGPUAnalysisSpacingStatistics(coordinates, {coordinateModel: 'geographic'});
    expectClose(
      strip.extentWidth,
      GPU_EARTH_MEAN_RADIUS_METERS * Math.cos(60.5 * RADIANS_PER_DEGREE) * 60 * RADIANS_PER_DEGREE,
      1e-9
    );
    // A dataset in 0..360 longitudes that does not wrap keeps its source ends.
    const east = computeGPUAnalysisSpacingStatistics([190, 0, 200, 1, 195, 0.5], {
      coordinateModel: 'geographic'
    });
    expect(east.extent).toEqual([190, 0, 200, 1]);
  });

  it('takes the smallest arc: -170, 0 and 170 degrees span 190 degrees, not 340', () => {
    const statistics = computeGPUAnalysisSpacingStatistics([-170, 0, 0, 0, 170, 0], {
      coordinateModel: 'geographic'
    });
    expectClose(
      statistics.extentWidth,
      GPU_EARTH_MEAN_RADIUS_METERS * 190 * RADIANS_PER_DEGREE,
      1e-9
    );
  });

  it('getGPUAnalysisExtentSizes measures any extent the same way', () => {
    const sizes = getGPUAnalysisExtentSizes([178, 60, -178, 70], 'geographic');
    const plain = getGPUAnalysisExtentSizes([-2, 60, 2, 70], 'geographic');
    expectClose(sizes.width, plain.width, 1e-12);
    expectClose(sizes.diagonal, Math.hypot(sizes.width, sizes.height), 1e-12);
    expect(getGPUAnalysisExtentSizes([0, 0, 30, 40], 'planar')).toEqual({
      width: 30,
      height: 40,
      diagonal: 50
    });
  });
});

describe('computeGPUAnalysisSpacingStatistics sampling of ordered files', () => {
  function gaussian(random: () => number): number {
    return Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
  }
  const count = 20000;
  const limit = 1000;
  // The factor is 20, so a pattern that looks the same everywhere gives sample / full = sqrt(20) = 4.47.
  const expectedRatio = Math.sqrt(count / limit);

  function ratioOfMedians(coordinates: number[]): number {
    const full = computeGPUAnalysisSpacingStatistics(coordinates, {
      coordinateModel: 'planar',
      maximumSampleSize: count
    });
    const sampled = computeGPUAnalysisSpacingStatistics(coordinates, {
      coordinateModel: 'planar',
      maximumSampleSize: limit
    });
    return (sampled.nearestNeighbor?.median ?? NaN) / (full.nearestNeighbor?.median ?? NaN);
  }

  it('20 vehicles interleaved row by row: the sample is not one vehicle (stride 20 was)', () => {
    const random = createRandom(1);
    const positions = Array.from({length: 20}, () => [random() * 1000, random() * 1000]);
    const coordinates: number[] = [];
    for (let step = 0; step < count / 20; step++) {
      for (const position of positions) {
        position[0] += gaussian(random) * 0.5;
        position[1] += gaussian(random) * 0.5;
        coordinates.push(position[0], position[1]);
      }
    }
    expect(ratioOfMedians(coordinates) / expectedRatio).toBeGreaterThan(0.8);
    expect(ratioOfMedians(coordinates) / expectedRatio).toBeLessThan(1.25);
  });

  it('clusters sorted by cluster with densities 1 to 100', () => {
    const random = createRandom(4);
    const coordinates: number[] = [];
    for (let cluster = 0; cluster < 10; cluster++) {
      const spread = 0.2 * 1.6 ** cluster;
      const centerX = random() * 1000;
      const centerY = random() * 1000;
      for (let index = 0; index < count / 10; index++) {
        coordinates.push(centerX + gaussian(random) * spread, centerY + gaussian(random) * spread);
      }
    }
    expect(ratioOfMedians(coordinates) / expectedRatio).toBeGreaterThan(0.8);
    expect(ratioOfMedians(coordinates) / expectedRatio).toBeLessThan(1.25);
  });

  it('one time-ordered track: the sample no longer invents a 20 times larger spacing', () => {
    const random = createRandom(2);
    let x = 0;
    let y = 0;
    const coordinates: number[] = [];
    for (let index = 0; index < count; index++) {
      x += 1 + gaussian(random) * 0.1;
      y += gaussian(random) * 0.2;
      coordinates.push(x, y);
    }
    // A track is a line: its spacing scales with the sampling fraction, not its square root, so the
    // ratio is larger than 4.47 (measured 7.2 for 5,000 of 100,000 rows) but nowhere near 20.
    const ratio = ratioOfMedians(coordinates);
    expect(ratio).toBeGreaterThan(expectedRatio);
    expect(ratio).toBeLessThan(12);
  });
});

/*
ORACLE SCRIPTS. Save as gen.py and oracle.py in one folder and run: /usr/bin/python3 oracle.py

----- gen.py -----
import math
import numpy as np
class Rng:
    def __init__(self, seed): self.a = seed & 0xFFFFFFFF
    def next(self):
        self.a = (self.a + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    def normal(self):
        u = max(self.next(), 1e-12); v = self.next()
        return math.sqrt(-2*math.log(u))*math.cos(2*math.pi*v)
def uniform(seed, n, w=1000.0, h=1000.0):
    r = Rng(seed); out = []
    for _ in range(n):
        x = r.next()*w; y = r.next()*h; out += [x, y]
    return np.array(out).reshape(-1, 2)
def clustered(seed, n):
    # 4 gaussian clusters (sd 8) at fixed centres + 5 percent uniform noise over 1000 x 1000
    r = Rng(seed); out = []
    centers = [(200,200),(800,250),(500,700),(250,850)]
    for i in range(n):
        if i % 20 == 19:
            out += [r.next()*1000, r.next()*1000]
        else:
            cx, cy = centers[i % 4]
            out += [cx + 8*r.normal(), cy + 8*r.normal()]
    return np.array(out).reshape(-1, 2)

----- oracle.py -----
import json, math
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from gen import Rng, uniform, clustered

R = 6371008.8
K = 3

def quantiles(sorted_values):
    n = len(sorted_values)
    if n <= 512:
        return list(sorted_values)
    return [sorted_values[round(q * (n - 1) / 511)] for q in range(512)]  # round(): see note below

def planar_oracle(points):
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=K + 1)
    nearest = distances[:, 1]
    kth = np.sort(distances[:, K])
    center = points.mean(axis=0)
    from_center = np.hypot(*(points - center).T)
    return summarize(nearest, kth, from_center, center, points)

def summarize(nearest, kth, from_center, center, points):
    q = quantiles(kth)
    return {
        'n': len(points),
        'nearest': [float(nearest.min()), float(np.median(nearest)), float(nearest.mean()), float(nearest.max())],
        'kthCount': len(q), 'kth': [float(q[0]), float(q[len(q)//4]), float(q[len(q)//2]), float(q[-1]), float(sum(q))],
        'standardDistance': float(math.sqrt((from_center ** 2).mean())),
        'medianFromCenter': float(np.median(from_center)),
        'center': [float(center[0]), float(center[1])],
    }

def haversine_matrix(lon, lat):
    lon = np.radians(lon); lat = np.radians(lat)
    dlat = lat[:, None] - lat[None, :]; dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def geographic_oracle(lonlat):
    lon, lat = lonlat[:, 0], lonlat[:, 1]
    d = haversine_matrix(lon, lat)
    d_sorted = np.sort(d, axis=1)  # column 0 is the point itself
    nearest = d_sorted[:, 1]; kth = np.sort(d_sorted[:, K])
    la, lo = np.radians(lat), np.radians(lon)
    v = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1).mean(axis=0)
    clon = math.atan2(v[1], v[0]); clat = math.atan2(v[2], math.hypot(v[0], v[1]))
    dl = lo - clon; dp = la - clat
    a = np.sin(dp / 2) ** 2 + np.cos(la) * math.cos(clat) * np.sin(dl / 2) ** 2
    from_center = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    out = summarize(nearest, kth, from_center, np.degrees([clon, clat]), lonlat)
    return out

def geo_set(seed, n, lon_min, lon_span, lat_min, lat_span):
    r = Rng(seed); out = []
    for _ in range(n):
        lon = lon_min + lon_span * r.next(); lat = lat_min + lat_span * r.next()
        if lon > 180: lon -= 360
        out += [lon, lat]
    return np.array(out).reshape(-1, 2)

def lattice():
    return np.array([[10.0 * i, 10.0 * j] for j in range(30) for i in range(30)])

def with_duplicates():
    base = uniform(3, 400)
    return np.vstack([base, base[:100], base[:100]])

def bandwidth(points):
    kde = gaussian_kde(points.T)  # scott factor n ** (-1/6) in two dimensions
    cov = np.cov(points.T)  # unbiased
    silverman = gaussian_kde(points.T, bw_method='silverman').factor
    return {'scottFactor': float(kde.factor), 'silvermanFactor': float(silverman),
            'bandwidth': float(kde.factor * math.sqrt((cov[0, 0] + cov[1, 1]) / 2))}

result = {
    'uniform': planar_oracle(uniform(1, 1500)),
    'clustered': planar_oracle(clustered(2, 1600)),
    'lattice': planar_oracle(lattice()),
    'duplicates': planar_oracle(with_duplicates()),
    'equator': geographic_oracle(geo_set(11, 300, 10, 2, -0.5, 1)),
    'sixtyNorth': geographic_oracle(geo_set(12, 300, 20, 4, 59.5, 1)),
    'antimeridian': geographic_oracle(geo_set(13, 300, 179, 2, 10, 1)),
    'pole': geographic_oracle(geo_set(14, 300, -180, 360, 88, 2)),
    'bandwidthUniform': bandwidth(uniform(1, 1500)),
    'bandwidthClustered': bandwidth(clustered(2, 1600)),
}
print(json.dumps(result, indent=1))
*/

/** A dense normal cloud, optionally mixed with a cloud whose spread is a thousand times larger. */
function createOutlierCloud(count: number, outlierCount: number, mixedDensity = false): number[] {
  const random = createRandom(87431);
  const normal = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
  const coordinates: number[] = [];
  for (let index = 0; index < count - outlierCount; index++) {
    const sparse = mixedDensity && index % 2 === 0;
    const spread = sparse ? 1000 : 1;
    coordinates.push((sparse ? 10000 : 0) + spread * normal(), spread * normal());
  }
  const outliers = [
    [1e6, 2e6],
    [-3e6, 2e6],
    [8e6, -7e6],
    [-9e6, 4e6]
  ];
  for (let index = 0; index < outlierCount; index++) {
    const [horizontal, vertical] = outliers[index % outliers.length];
    coordinates.push(horizontal + index * 100, vertical - index * 100);
  }
  return coordinates;
}

describe('robust spacing grid with outliers and unequal densities', () => {
  for (const coordinateModel of ['planar', 'geographic'] as const) {
    for (const layout of [1, 4, 100, 'two densities'] as const) {
      it(`matches brute force for ${coordinateModel}, ${layout}`, () => {
        const planar = createOutlierCloud(
          600,
          typeof layout === 'number' ? layout : 0,
          layout === 'two densities'
        );
        // A tiny geographic cloud around the antimeridian plus distant points; keep latitudes valid.
        const coordinates =
          coordinateModel === 'planar'
            ? planar
            : planar.map((value, index) =>
                index % 2 === 0 ? 179 + value * 1e-6 : 25 + value * 1e-6
              );
        for (const neighborRank of [1, 3, 7]) {
          const reference = computeBruteForceNeighbors(coordinates, coordinateModel, [
            neighborRank
          ]);
          const statistics = computeGPUAnalysisSpacingStatistics(coordinates, {
            coordinateModel,
            neighborRank
          });
          const nearest = reference.nearest.sort((left, right) => left - right);
          expectClose(statistics.nearestNeighbor?.minimum, nearest[0], 1e-8);
          expectClose(statistics.nearestNeighbor?.median, (nearest[299] + nearest[300]) / 2, 1e-8);
          expectClose(statistics.nearestNeighbor?.maximum, nearest[599], 1e-8);
          expectClose(
            statistics.nearestNeighbor?.mean,
            nearest.reduce((sum, distance) => sum + distance, 0) / nearest.length,
            1e-8
          );
          const expected = reduceBruteForceQuantiles(reference.kth.get(neighborRank) ?? []);
          expect(statistics.kDistance?.distances).toHaveLength(expected.length);
          statistics.kDistance?.distances.forEach((distance, index) =>
            expectClose(distance, expected[index], 1e-8)
          );
        }
      });
    }
  }

  it('computes a 20,000-point normal cloud with four distant points in under half a second', () => {
    const coordinates = createOutlierCloud(20000, 4);
    // Warm the same path before timing; the pre-fix search takes about ten seconds per call.
    computeGPUAnalysisSpacingStatistics(createOutlierCloud(1000, 4), {coordinateModel: 'planar'});
    const started = performance.now();
    const statistics = computeGPUAnalysisSpacingStatistics(coordinates, {
      coordinateModel: 'planar'
    });
    expect(performance.now() - started).toBeLessThan(500);
    expect(statistics.nearestNeighbor?.maximum).toBeGreaterThan(1e6);
    expect(statistics.pointCount).toBe(20000);
  });
});

describe('shared exact geographic extent conventions', () => {
  it('preserves source longitude bounds at the poles while deduplicating physical locations', () => {
    const spacing = computeGPUAnalysisSpacingStatistics([178, 90, -178, 90], {
      coordinateModel: 'geographic'
    });
    expect(spacing.extent).toEqual([178, 90, -178, 90]);
    expect(spacing.duplicateLocationCount).toBe(1);
  });

  it('keeps source 0-to-360 bounds when already shortest and resolves equal gaps deterministically', () => {
    expect(
      computeGPUAnalysisSpacingStatistics([150, 0, 250, 1], {coordinateModel: 'geographic'}).extent
    ).toEqual([150, 0, 250, 1]);
    for (const coordinates of [
      [-170, 0, -60, 1, 50, 2, 170, 3],
      [170, 3, 50, 2, -60, 1, -170, 0]
    ]) {
      expect(
        computeGPUAnalysisSpacingStatistics(coordinates, {coordinateModel: 'geographic'}).extent
      ).toEqual([170, 0, 50, 3]);
    }
  });
});
