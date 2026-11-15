// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  GPU_ANALYSIS_DEFAULT_RULES,
  type GPUAnalysisDataProfile,
  type GPUAnalysisSpacingStatistics
} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-experience-types';
import {
  GPU_ANALYSIS_KNEE_MINIMUM_BEND,
  GPU_ANALYSIS_KNEE_MINIMUM_RANGE_OVER_MEDIAN,
  listGPUAnalysisDefaultRules,
  resolveGPUAnalysisDefault
} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-default-rules';
import {
  computeGPUAnalysisSpacingStatistics as computeMaybeSpacingStatistics,
  getGPUAnalysisExtentSizes
} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-spacing-statistics';

// ORACLE values come from /usr/bin/python3 (NumPy 2.0, SciPy 1.13) with the scripts at the end of this
// file, which use the same seeded generators as the TypeScript below (mulberry32, Box-Muller):
// - nearest-neighbour and k-distance values: scipy.spatial.cKDTree;
// - bandwidths: scipy.stats.gaussian_kde(...).factor times sqrt(mean of the unbiased covariance
//   diagonal), which is the quantity GPUKernelDensity bandwidthRule computes for unit weights;
// - knee positions: the chord-distance rule on the 512-quantile 3rd-neighbour curve.
const ORACLE_UNIFORM = {
  maximumNearest: 45.78487229149349,
  medianNearest: 12.130534783693847,
  standardDistance: 405.32203911678783,
  scottFactor: 0.2955639465050328,
  silvermanFactor: 0.2955639465050328,
  bandwidth: 84.7386402685582,
  kneeBend: 0.2824005214452846
} as const;
const ORACLE_CLUSTERED = {
  bandwidth: 76.00556381373657,
  scottFactor: 0.2924017738212866,
  kneeBend: 0.8971992510770728,
  kneeValue: 6.845479091825402
} as const;
// Discrete bends of 100 * (i / 100) ** p on i = 0..100, from knee.py: p = 6 bends 0.58235 (value at
// the knee 11.7649), p = 10 bends 0.69673 (value 7.32668).
const POWER_SIX_BEND = 0.5823510000000001;
const POWER_TEN_BEND = 0.696733195274138;

function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), state | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function createUniformPoints(seed: number, count: number): number[] {
  const random = createRandom(seed);
  const coordinates: number[] = [];
  for (let index = 0; index < count; index++) {
    const x = random() * 1000;
    const y = random() * 1000;
    coordinates.push(x, y);
  }
  return coordinates;
}

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

/** Spacing statistics of input with at least one valid point. */
function computeGPUAnalysisSpacingStatistics(
  coordinates: readonly number[],
  options: Parameters<typeof computeMaybeSpacingStatistics>[1]
): GPUAnalysisSpacingStatistics {
  const statistics = computeMaybeSpacingStatistics(coordinates, options);
  if (!statistics) {
    throw new Error('expected spacing statistics for at least one valid point');
  }
  return statistics;
}

function createProfile(
  coordinates: readonly number[],
  model: 'planar' | 'geographic' = 'planar',
  overrides: Partial<GPUAnalysisDataProfile> = {}
): GPUAnalysisDataProfile {
  const spacing = computeMaybeSpacingStatistics(coordinates, {coordinateModel: model});
  return {
    kind: 'points',
    rowCount: spacing?.pointCount ?? 0,
    coordinates: {model, source: 'declared'},
    extent: spacing?.extent,
    columns: [],
    spacing,
    ...overrides
  };
}

const EMPTY_PROFILE: GPUAnalysisDataProfile = {kind: 'points', rowCount: 0, columns: []};

function withSpacing(
  spacing: GPUAnalysisSpacingStatistics,
  changes: Partial<GPUAnalysisSpacingStatistics>
): GPUAnalysisDataProfile {
  return {
    kind: 'points',
    rowCount: spacing.pointCount,
    extent: spacing.extent,
    columns: [],
    spacing: {...spacing, ...changes}
  };
}

function expectOneSentence(explanation: string): void {
  expect(explanation.length).toBeGreaterThan(20);
  expect(explanation.endsWith('.')).toBe(true);
  // Numbers such as 1.5 hold a dot, a sentence break is a dot followed by a space.
  expect(explanation.includes('. ')).toBe(false);
}

describe('resolveGPUAnalysisDefault review regressions', () => {
  it('declines a zero knee caused by duplicate locations', () => {
    const coordinates = [
      ...Array.from({length: 1000}, (_, index) => [
        Math.floor(index / 10) % 10,
        Math.floor(index / 100)
      ]).flat(),
      100,
      100
    ];
    const resolution = resolveGPUAnalysisDefault('k-distance-knee', createProfile(coordinates));
    expect(resolution.value).toBeUndefined();
    expect(resolution.explanation).toMatch(/duplicate locations/);
    expect(resolution.explanation).toMatch(/positive and finite/);
  });

  it('uses distinct-location spacing when the ordinary median is zero', () => {
    const profile = createProfile([
      ...Array.from({length: 10}, () => [0, 0]).flat(),
      3,
      0,
      7,
      0,
      15,
      0
    ]);
    const resolution = resolveGPUAnalysisDefault('median-nearest-neighbor-distance', profile, 2);
    // Distinct-location distances [3,3,4,8] and their median 3.5 are checked in the spacing spec.
    expect(resolution.value).toBe(7);
    expect(resolution.explanation).toMatch(
      /Median nearest-neighbour distance of distinct locations/
    );
    expect(resolution.explanation).toMatch(/plain median is 0 because of duplicate locations/);
    expect(resolution.explanation).toMatch(
      /each distinct location to its nearest other distinct location/
    );
  });

  it.each([
    0.001, 1, 1000
  ])('uses a data-derived radius for forty repeated places at scale %s', scale => {
    const locations = Array.from({length: 40}, (_, index) => [
      (index % 8) * 100 * scale,
      Math.floor(index / 8) * 100 * scale
    ]);
    const profile = createProfile(
      locations.flatMap(location => Array.from({length: 10}, () => location).flat())
    );
    const resolution = resolveGPUAnalysisDefault('median-nearest-neighbor-distance', profile);
    const distinctResolution = resolveGPUAnalysisDefault(
      'median-nearest-neighbor-distance',
      createProfile(locations.flat())
    );
    // Same cKDTree oracle as the spacing spec: the grid has spacing 100 * scale.
    expect(resolution.value).toBeCloseTo(100 * scale, 10);
    expect(resolution.value).toBe(distinctResolution.value);
    expect(resolution.explanation).toMatch(/distinct location/);
    expect(resolution.sampled).toBe(false);
  });

  it('explains that fallback spacing uses distinct locations of the sample', () => {
    const coordinates = Array.from({length: 400}, (_, index) => [
      Math.floor(index / 10) ** 2,
      0
    ]).flat();
    const profile = createProfile(coordinates, 'planar', {
      spacing: computeGPUAnalysisSpacingStatistics(coordinates, {
        coordinateModel: 'planar',
        maximumSampleSize: 60
      })
    });
    const resolution = resolveGPUAnalysisDefault('median-nearest-neighbor-distance', profile);
    // Independent hashed-sample cKDTree oracle is recorded in the spacing spec.
    expect(profile.spacing?.nearestNeighbor?.median).toBe(0);
    expect(resolution.value).toBe(41);
    expect(resolution.sampled).toBe(true);
    expect(resolution.explanation).toMatch(/distinct locations of the sample/);
  });

  it('checks the final scaled output of every distance rule for overflow and underflow', () => {
    const profile = createProfile([0, 0, 1000, 1000, 3000, 2000, 5000, 5000]);
    const rules = [
      'maximum-nearest-neighbor-distance',
      'median-nearest-neighbor-distance',
      'outlier-resistant-nearest-neighbor-distance',
      'mean-point-spacing',
      'extent-shorter-side-over-250',
      'extent-shorter-side-over-250-in-source-units',
      'extent-diagonal',
      'standard-distance',
      'normal-reference-bandwidth-scott',
      'normal-reference-bandwidth-silverman',
      'k-distance-knee'
    ] as const;
    for (const rule of rules) {
      const fixture =
        rule === 'k-distance-knee'
          ? {
              ...profile,
              spacing: {
                ...profile.spacing!,
                kDistance: {k: 3, distances: [2, 2, 2, 2, 2, 2, 3, 1000]}
              }
            }
          : profile;
      expect(resolveGPUAnalysisDefault(rule, fixture).value).toBeGreaterThan(1);
      const overflow = resolveGPUAnalysisDefault(rule, fixture, Number.MAX_VALUE);
      expect(overflow.value).toBeUndefined();
      expect(overflow.explanation).toMatch(/final scaled value must be positive and finite/);
      const smallSpacing = fixture.spacing!;
      const tinyFixture: GPUAnalysisDataProfile = {
        ...fixture,
        extent: [0, 0, 0.01, 0.01],
        spacing: {
          ...smallSpacing,
          standardDistance: 0.01,
          nearestNeighbor: {
            ...smallSpacing.nearestNeighbor!,
            maximum: 0.01,
            median: 0.01,
            upperFence: 0.01
          },
          kDistance: {k: 3, distances: [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 1]}
        }
      };
      expect(resolveGPUAnalysisDefault(rule, tinyFixture).value).toBeGreaterThan(0);
      const underflow = resolveGPUAnalysisDefault(rule, tinyFixture, Number.MIN_VALUE);
      expect(underflow.value).toBeUndefined();
      expect(underflow.explanation).toMatch(/final scaled value must be positive and finite/);
    }
  });

  it.each([
    1e-200, 1e160
  ])('keeps bandwidth and mean spacing in range at magnitude %s', magnitude => {
    const profile = createProfile([0, 0, magnitude, 0, 0, magnitude, magnitude, magnitude]);
    expect(resolveGPUAnalysisDefault('mean-point-spacing', profile).value! / magnitude).toBeCloseTo(
      0.5,
      12
    );
    for (const rule of [
      'normal-reference-bandwidth-scott',
      'normal-reference-bandwidth-silverman'
    ] as const) {
      const output = resolveGPUAnalysisDefault(rule, profile).value;
      expect(output).toBeGreaterThan(0);
      expect(Number.isFinite(output)).toBe(true);
    }
  });

  it('qualifies neighbour and island statements as describing the unscaled statistic', () => {
    const lattice = createProfile(
      Array.from({length: 100}, (_, index) => [index % 10, Math.floor(index / 10)]).flat()
    );
    for (const rule of [
      'maximum-nearest-neighbor-distance',
      'outlier-resistant-nearest-neighbor-distance'
    ] as const) {
      const resolution = resolveGPUAnalysisDefault(rule, lattice, 0.5);
      expect(resolution.value).toBe(0.5);
      expect(resolution.explanation).toMatch(/unscaled statistic/);
    }
    const profile = createProfile(
      [0, 100, 200, 300, 400, 500, 600, 705].flatMap(value => [value, 0])
    );
    const resolution = resolveGPUAnalysisDefault(
      'outlier-resistant-nearest-neighbor-distance',
      profile,
      1.1
    );
    expect(resolution.value).toBeCloseTo(110, 12);
    expect(resolution.explanation).toMatch(/1 point lies beyond the unscaled spacing fence/);
    expect(resolution.explanation).toMatch(/actual isolation depends on the final distance band/);
    expect(resolution.explanation).not.toMatch(/will have no neighbour/);
    const sampleProfile = {
      ...profile,
      spacing: {...profile.spacing!, sampled: true, pointCount: 80}
    };
    for (const rule of [
      'maximum-nearest-neighbor-distance',
      'outlier-resistant-nearest-neighbor-distance'
    ] as const) {
      expect(resolveGPUAnalysisDefault(rule, sampleProfile).explanation).toMatch(
        /1 (isolated )?sampled point/
      );
    }
  });

  it('describes a sample whose median is lower than the full-data median without a directional claim', () => {
    // These are the review probe's ten lowest hashed row indices among 0..99.
    const selected = new Set([8, 25, 39, 41, 59, 72, 76, 80, 85, 93]);
    let densePointIndex = 0;
    const coordinates = Array.from({length: 100}, (_, index) =>
      selected.has(index) ? [densePointIndex++ * 0.001, 0] : [100 + index * 100, 0]
    ).flat();
    const full = createProfile(coordinates);
    const sample = {
      ...full,
      spacing: computeGPUAnalysisSpacingStatistics(coordinates, {
        coordinateModel: 'planar',
        maximumSampleSize: 10
      })
    };
    expect(full.spacing?.nearestNeighbor?.median).toBe(100);
    expect(sample.spacing?.nearestNeighbor?.median).toBeCloseTo(0.001, 12);
    const resolution = resolveGPUAnalysisDefault('median-nearest-neighbor-distance', sample);
    expect(resolution.explanation).toContain(
      'Distances were computed within a sample of 10 of 100 points (10%). Sample statistics may differ substantially from full-data statistics; no correction is applied.'
    );
    expect(resolution.explanation).not.toMatch(/larger than the full set/);
  });

  it('describes the extent diagonal as a bound rather than an observed diameter', () => {
    const profile = createProfile([1, 0, -0.5, Math.sqrt(3) / 2, -0.5, -Math.sqrt(3) / 2]);
    const resolution = resolveGPUAnalysisDefault('extent-diagonal', profile);
    expect(resolution.value).toBeGreaterThan(Math.sqrt(3));
    expect(resolution.explanation).toMatch(/diagonal of the bounding extent, an upper bound/);
    expect(resolution.explanation).not.toMatch(/largest distance in the data/);
  });
});

describe('listGPUAnalysisDefaultRules', () => {
  it('lists every rule once, in contract order, with label, statistic and source', () => {
    const records = listGPUAnalysisDefaultRules();
    expect(records.map(record => record.id)).toEqual([...GPU_ANALYSIS_DEFAULT_RULES]);
    for (const record of records) {
      expect(record.label.length).toBeGreaterThan(3);
      expect(record.needs.length).toBeGreaterThan(3);
      expect(record.source.length).toBeGreaterThan(10);
    }
  });

  it('cites a primary source for the distance band and the cell size rules', () => {
    const byId = new Map(listGPUAnalysisDefaultRules().map(record => [record.id, record]));
    expect(byId.get('maximum-nearest-neighbor-distance')?.source).toMatch(/GeoDa/);
    expect(byId.get('extent-shorter-side-over-250')?.source).toMatch(/Esri/);
    expect(byId.get('k-distance-knee')?.source).toMatch(/Schubert/);
    expect(byId.get('median-nearest-neighbor-distance')?.source).toMatch(/Not a vendor default/);
  });
});

describe('resolveGPUAnalysisDefault on a uniform set of 1,500 points (oracle values)', () => {
  const coordinates = createUniformPoints(1, 1500);
  const profile = createProfile(coordinates);

  it('maximum nearest-neighbour distance equals the cKDTree maximum', () => {
    const resolution = resolveGPUAnalysisDefault('maximum-nearest-neighbor-distance', profile);
    expect(resolution.value).toBeCloseTo(ORACLE_UNIFORM.maximumNearest, 9);
    expect(resolution.unit).toBe('coordinate-unit');
    expect(resolution.factor).toBe(1);
    expect(resolution.sampled).toBe(false);
    expect(resolution.explanation).toBe(
      'Largest nearest-neighbour distance (45.8 coordinate units), at the unscaled statistic, every point has at least one neighbour.'
    );
    expectOneSentence(resolution.explanation);
  });

  it('median nearest-neighbour distance equals the cKDTree median', () => {
    const resolution = resolveGPUAnalysisDefault('median-nearest-neighbor-distance', profile);
    expect(resolution.value).toBeCloseTo(ORACLE_UNIFORM.medianNearest, 9);
    expectOneSentence(resolution.explanation);
  });

  it('standard distance equals the root mean squared distance from the centre', () => {
    const resolution = resolveGPUAnalysisDefault('standard-distance', profile);
    expect(resolution.value).toBeCloseTo(ORACLE_UNIFORM.standardDistance, 9);
    expectOneSentence(resolution.explanation);
  });

  it("Scott's and Silverman's rules give the scipy gaussian_kde bandwidth, and equal each other", () => {
    const scott = resolveGPUAnalysisDefault('normal-reference-bandwidth-scott', profile);
    const silverman = resolveGPUAnalysisDefault('normal-reference-bandwidth-silverman', profile);
    expect(scott.value).toBeCloseTo(ORACLE_UNIFORM.bandwidth, 9);
    expect(silverman.value).toBe(scott.value);
    expect(ORACLE_UNIFORM.silvermanFactor).toBe(ORACLE_UNIFORM.scottFactor);
    expect(scott.explanation).toMatch(/^Scott's rule bandwidth \(84\.7 coordinate units\)/);
    expect(silverman.explanation).toMatch(/^Silverman's rule bandwidth/);
    expectOneSentence(scott.explanation);
    expectOneSentence(silverman.explanation);
  });

  it('the bandwidth equals n^(-1/6) times the root mean unbiased variance, computed from the raw points', () => {
    const count = coordinates.length / 2;
    let meanX = 0;
    let meanY = 0;
    for (let index = 0; index < count; index++) {
      meanX += coordinates[2 * index] / count;
      meanY += coordinates[2 * index + 1] / count;
    }
    let spreadX = 0;
    let spreadY = 0;
    for (let index = 0; index < count; index++) {
      spreadX += (coordinates[2 * index] - meanX) ** 2;
      spreadY += (coordinates[2 * index + 1] - meanY) ** 2;
    }
    const expected =
      count ** (-1 / 6) * Math.sqrt((spreadX / (count - 1) + spreadY / (count - 1)) / 2);
    expect(count ** (-1 / 6)).toBeCloseTo(ORACLE_UNIFORM.scottFactor, 12);
    expect(
      resolveGPUAnalysisDefault('normal-reference-bandwidth-scott', profile).value
    ).toBeCloseTo(expected, 9);
  });

  it('a random pattern has no clear knee: the rule returns undefined and says why', () => {
    expect(ORACLE_UNIFORM.kneeBend).toBeLessThan(GPU_ANALYSIS_KNEE_MINIMUM_BEND);
    const resolution = resolveGPUAnalysisDefault('k-distance-knee', profile);
    expect(resolution.value).toBeUndefined();
    expect(resolution.explanation).toMatch(/no clear knee \(largest bend 0\.282, below 0\.65\)/);
    expect(resolution.explanation).toMatch(/3rd-neighbour/);
    expectOneSentence(resolution.explanation);
  });

  it('extent rules read the profile extent; the cell size is the shorter side over 250', () => {
    const extent = profile.extent ?? [0, 0, 0, 0];
    const cell = resolveGPUAnalysisDefault('extent-shorter-side-over-250', profile);
    expect(cell.value).toBeCloseTo(
      Math.min(extent[2] - extent[0], extent[3] - extent[1]) / 250,
      12
    );
    expect(cell.unit).toBe('coordinate-unit');
    expectOneSentence(cell.explanation);
    const edges = ['extent-west', 'extent-south', 'extent-east', 'extent-north'] as const;
    edges.forEach((rule, index) => {
      const resolution = resolveGPUAnalysisDefault(rule, profile);
      expect(resolution.value).toBe(extent[index]);
      expect(resolution.factor).toBe(1);
      expectOneSentence(resolution.explanation);
    });
    const diagonal = resolveGPUAnalysisDefault('extent-diagonal', profile);
    expect(diagonal.value).toBeCloseTo(Math.hypot(extent[2] - extent[0], extent[3] - extent[1]), 9);
  });

  it('multiplies the value by the factor and says so', () => {
    const base = resolveGPUAnalysisDefault('maximum-nearest-neighbor-distance', profile).value ?? 0;
    const scaled = resolveGPUAnalysisDefault('maximum-nearest-neighbor-distance', profile, 1.5);
    expect(scaled.value).toBeCloseTo(1.5 * base, 12);
    expect(scaled.factor).toBe(1.5);
    expect(scaled.explanation).toMatch(/times 1\.5 = 68\.7 coordinate units/);
    expectOneSentence(scaled.explanation);
    // An edge ignores the factor.
    expect(resolveGPUAnalysisDefault('extent-east', profile, 3).value).toBe(profile.extent?.[2]);
  });

  it('rejects a factor that is not a positive finite number', () => {
    for (const factor of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => resolveGPUAnalysisDefault('standard-distance', profile, factor)).toThrow(
        /resolveGPUAnalysisDefault factor must be a positive finite number/
      );
    }
  });
});

describe('resolveGPUAnalysisDefault on four clusters with noise (oracle values)', () => {
  const coordinates = createClusteredPoints(2, 1600);
  const profile = createProfile(coordinates);

  it('finds the knee of the 3rd-neighbour curve at the oracle value', () => {
    expect(ORACLE_CLUSTERED.kneeBend).toBeGreaterThan(GPU_ANALYSIS_KNEE_MINIMUM_BEND);
    const resolution = resolveGPUAnalysisDefault('k-distance-knee', profile);
    expect(resolution.value).toBeCloseTo(ORACLE_CLUSTERED.kneeValue, 9);
    expect(resolution.explanation).toMatch(
      /^Knee of the sorted 3rd-neighbour distance curve \(6\.85 coordinate units\)/
    );
    expectOneSentence(resolution.explanation);
  });

  it('gives the scipy bandwidth (the normal-reference rule is much wider than the clusters)', () => {
    const scott = resolveGPUAnalysisDefault('normal-reference-bandwidth-scott', profile);
    expect(scott.value).toBeCloseTo(ORACLE_CLUSTERED.bandwidth, 9);
    expect(scott.value).toBeGreaterThan(
      10 * (resolveGPUAnalysisDefault('median-nearest-neighbor-distance', profile).value ?? 0)
    );
  });
});

describe('resolveGPUAnalysisDefault knee threshold', () => {
  const baseSpacing = computeGPUAnalysisSpacingStatistics(createUniformPoints(9, 50), {
    coordinateModel: 'planar'
  });
  const curve = (power: number) =>
    Array.from({length: 101}, (_, index) => 100 * (index / 100) ** power);

  it('rejects a curve that bends 0.58 and accepts one that bends 0.70', () => {
    expect(POWER_SIX_BEND).toBeLessThan(GPU_ANALYSIS_KNEE_MINIMUM_BEND);
    expect(POWER_TEN_BEND).toBeGreaterThan(GPU_ANALYSIS_KNEE_MINIMUM_BEND);
    const rejected = resolveGPUAnalysisDefault(
      'k-distance-knee',
      withSpacing(baseSpacing, {kDistance: {k: 3, distances: curve(6)}})
    );
    expect(rejected.value).toBeUndefined();
    expect(rejected.explanation).toMatch(/largest bend 0\.582/);
    const accepted = resolveGPUAnalysisDefault(
      'k-distance-knee',
      withSpacing(baseSpacing, {kDistance: {k: 3, distances: curve(10)}})
    );
    expect(accepted.value).toBeCloseTo(7.32668, 5);
  });

  it('returns undefined for a straight, flat or too short curve and for a missing curve', () => {
    const straight = Array.from({length: 50}, (_, index) => index);
    const flat = Array.from({length: 50}, () => 4);
    for (const distances of [straight, flat, [1, 2, 9, 9, 9, 9, 9], []]) {
      const resolution = resolveGPUAnalysisDefault(
        'k-distance-knee',
        withSpacing(baseSpacing, {kDistance: {k: 3, distances}})
      );
      expect(resolution.value).toBeUndefined();
      expectOneSentence(resolution.explanation);
    }
    const missing = resolveGPUAnalysisDefault(
      'k-distance-knee',
      withSpacing(baseSpacing, {kDistance: undefined})
    );
    expect(missing.value).toBeUndefined();
    expect(missing.explanation).toMatch(/no curve of at least 8 points/);
  });

  it('accepts a curve of exactly 8 points with a sharp knee and applies the factor', () => {
    const distances = [1, 1, 1, 1, 1, 1, 1, 100];
    const resolution = resolveGPUAnalysisDefault(
      'k-distance-knee',
      withSpacing(baseSpacing, {kDistance: {k: 11, distances}}),
      2
    );
    // Normalised, the largest gap below the chord is at index 6 (x = 6/7, y = 0): value 1.
    expect(resolution.value).toBe(2);
    expect(resolution.explanation).toMatch(/11th-neighbour/);
  });
});

describe('resolveGPUAnalysisDefault when the profile lacks what a rule needs', () => {
  it('returns undefined with a sentence for every rule on an empty profile', () => {
    for (const rule of GPU_ANALYSIS_DEFAULT_RULES) {
      const resolution = resolveGPUAnalysisDefault(rule, EMPTY_PROFILE);
      expect(resolution.rule).toBe(rule);
      expect(resolution.value).toBeUndefined();
      expect(resolution.sampled).toBe(false);
      expectOneSentence(resolution.explanation);
    }
  });

  it('returns undefined for spacing statistics of 0 points and of 1 point', () => {
    for (const coordinates of [[], [5, 5]]) {
      const profile = createProfile(coordinates);
      for (const rule of [
        'maximum-nearest-neighbor-distance',
        'median-nearest-neighbor-distance',
        'k-distance-knee',
        'normal-reference-bandwidth-scott',
        'normal-reference-bandwidth-silverman',
        'standard-distance'
      ] as const) {
        const resolution = resolveGPUAnalysisDefault(rule, profile);
        expect(resolution.value).toBeUndefined();
        expectOneSentence(resolution.explanation);
      }
    }
  });

  it('returns undefined when all points share one location (a distance of 0 is no distance band)', () => {
    const profile = createProfile(
      Array.from({length: 60}, (_, index) => (index % 2 === 0 ? 3 : 4))
    );
    for (const rule of [
      'maximum-nearest-neighbor-distance',
      'median-nearest-neighbor-distance',
      'standard-distance',
      'normal-reference-bandwidth-scott',
      'extent-shorter-side-over-250',
      'extent-diagonal',
      'k-distance-knee'
    ] as const) {
      expect(resolveGPUAnalysisDefault(rule, profile).value).toBeUndefined();
    }
    expect(
      resolveGPUAnalysisDefault('maximum-nearest-neighbor-distance', profile).explanation
    ).toMatch(/share locations/);
    // The edges are still defined.
    expect(resolveGPUAnalysisDefault('extent-west', profile).value).toBe(3);
  });

  it('gives no cell size for collinear points (a side of 0) but gives the other rules', () => {
    const profile = createProfile(Array.from({length: 100}, (_, index) => [index, 2]).flat());
    expect(
      resolveGPUAnalysisDefault('extent-shorter-side-over-250', profile).value
    ).toBeUndefined();
    expect(resolveGPUAnalysisDefault('extent-diagonal', profile).value).toBe(99);
    expect(resolveGPUAnalysisDefault('maximum-nearest-neighbor-distance', profile).value).toBe(1);
  });

  it('uses the distinct-location median when most points repeat a location', () => {
    const coordinates = [...Array.from({length: 30}, () => 1), ...createUniformPoints(5, 5)];
    const profile = createProfile(coordinates);
    expect(profile.spacing?.nearestNeighbor?.median).toBe(0);
    const resolution = resolveGPUAnalysisDefault('median-nearest-neighbor-distance', profile);
    expect(resolution.value).toBe(profile.spacing?.nearestNeighbor?.nonZeroMedian);
    expect(resolution.value).toBeGreaterThan(0);
    expect(resolution.explanation).toMatch(/plain median is 0 because of duplicate locations/);
    expect(
      resolveGPUAnalysisDefault('maximum-nearest-neighbor-distance', profile).value
    ).toBeGreaterThan(0);
  });

  it('ignores a non-finite extent', () => {
    const profile: GPUAnalysisDataProfile = {...EMPTY_PROFILE, extent: [0, 0, Number.NaN, 10]};
    expect(resolveGPUAnalysisDefault('extent-west', profile).value).toBeUndefined();
    expect(
      resolveGPUAnalysisDefault('extent-shorter-side-over-250', profile).value
    ).toBeUndefined();
  });

  it('needs two points for the bandwidth: 2 points give a value, 1 does not', () => {
    const two = createProfile([0, 0, 6, 8]);
    // n = 2 points 10 apart: standard distance 5, so the unbiased variance sum is 2 * 25 * 2 / 1 / 2.
    const variance = (5 * 5 * 2) / (2 * 1);
    expect(resolveGPUAnalysisDefault('normal-reference-bandwidth-scott', two).value).toBeCloseTo(
      2 ** (-1 / 6) * Math.sqrt(variance),
      12
    );
    expect(
      resolveGPUAnalysisDefault('normal-reference-bandwidth-scott', createProfile([0, 0])).value
    ).toBeUndefined();
  });
});

describe('resolveGPUAnalysisDefault units, sampling and coordinate models', () => {
  it('reports metres for geographic spacing and degrees for the extent', () => {
    const coordinates = [-74, 40.7, -73.9, 40.8, -74.1, 40.6, -73.95, 40.65];
    const profile = createProfile(coordinates, 'geographic');
    const maximum = resolveGPUAnalysisDefault('maximum-nearest-neighbor-distance', profile);
    expect(maximum.unit).toBe('metre');
    expect(maximum.explanation).toMatch(/ (m|km)\)/);
    const west = resolveGPUAnalysisDefault('extent-west', profile);
    expect(west.unit).toBe('degree');
    expect(west.value).toBe(-74.1);
    expect(west.explanation).toMatch(/degrees/);
    const cell = resolveGPUAnalysisDefault('extent-shorter-side-over-250', profile);
    expect(cell.unit).toBe('metre');
    expect(cell.value).toBeCloseTo(
      Math.min(profile.spacing?.extentWidth ?? NaN, profile.spacing?.extentHeight ?? NaN) / 250,
      9
    );
    expect(cell.explanation).toMatch(/ (m|km)\)/);
    expect(resolveGPUAnalysisDefault('extent-diagonal', profile).unit).toBe('metre');
    expect(resolveGPUAnalysisDefault('normal-reference-bandwidth-scott', profile).unit).toBe(
      'metre'
    );
  });

  it('computes the diagonal of a planar and of a geographic extent without spacing statistics', () => {
    const planar: GPUAnalysisDataProfile = {
      ...EMPTY_PROFILE,
      coordinates: {model: 'planar', source: 'declared'},
      extent: [0, 0, 30, 40]
    };
    expect(resolveGPUAnalysisDefault('extent-diagonal', planar).value).toBe(50);
    const geographic: GPUAnalysisDataProfile = {
      ...planar,
      coordinates: {model: 'geographic', source: 'declared'}
    };
    // 30 by 40 degrees: width is an arc at middle latitude 20.
    const sizes = getGPUAnalysisExtentSizes([0, 0, 30, 40], 'geographic');
    expect(resolveGPUAnalysisDefault('extent-diagonal', geographic).value).toBeCloseTo(
      sizes.diagonal,
      6
    );
    expect(resolveGPUAnalysisDefault('extent-diagonal', geographic).unit).toBe('metre');
  });

  it('flags sampled neighbour statistics and states the sample fraction', () => {
    const spacing = computeGPUAnalysisSpacingStatistics(createUniformPoints(6, 3000), {
      coordinateModel: 'planar',
      maximumSampleSize: 600
    });
    const profile: GPUAnalysisDataProfile = {kind: 'points', rowCount: 3000, columns: [], spacing};
    const maximum = resolveGPUAnalysisDefault('maximum-nearest-neighbor-distance', profile);
    expect(maximum.sampled).toBe(true);
    expect(maximum.explanation).toMatch(
      /Distances were computed within a sample of 600 of 3,000 points \(20%\). Sample statistics may differ substantially from full-data statistics; no correction is applied/
    );
    expectOneSentence(
      maximum.explanation.split('. Distances were computed')[0] + (maximum.sampled ? '.' : '')
    );
    // Statistics computed on every point are not flagged.
    expect(resolveGPUAnalysisDefault('standard-distance', profile).sampled).toBe(false);
    expect(resolveGPUAnalysisDefault('normal-reference-bandwidth-scott', profile).sampled).toBe(
      false
    );
  });

  it('formats numbers with a fixed locale: 1,240 m', () => {
    const spacing = computeGPUAnalysisSpacingStatistics([0, 0, 1240, 0], {
      coordinateModel: 'planar'
    });
    const resolution = resolveGPUAnalysisDefault('maximum-nearest-neighbor-distance', {
      kind: 'points',
      rowCount: 2,
      columns: [],
      spacing: {...spacing, unit: 'metre', coordinateModel: 'geographic'}
    });
    expect(resolution.explanation).toBe(
      'Largest nearest-neighbour distance (1,240 m), at the unscaled statistic, every point has at least one neighbour.'
    );
    expect(resolution.value).toBe(1240);
  });

  it('uses the singular for one coordinate unit', () => {
    const spacing = computeGPUAnalysisSpacingStatistics([0, 0, 1, 0], {coordinateModel: 'planar'});
    const resolution = resolveGPUAnalysisDefault('maximum-nearest-neighbor-distance', {
      kind: 'points',
      rowCount: 2,
      columns: [],
      spacing
    });
    expect(resolution.explanation).toMatch(/\(1 coordinate unit\)/);
  });
});

describe('resolveGPUAnalysisDefault cell size is a ground distance', () => {
  const rule = 'extent-shorter-side-over-250';
  const oneDegree = (6371008.8 * Math.PI) / 180;

  it('geographic data with spacing: the shorter side in metres over 250', () => {
    // 2 degrees of longitude by 1 degree of latitude at the equator: the height is the shorter side.
    const profile = createProfile([0, 0, 2, 1, 1, 0.5], 'geographic');
    const resolution = resolveGPUAnalysisDefault(rule, profile);
    expect(resolution.unit).toBe('metre');
    expect(resolution.value).toBeCloseTo(oneDegree / 250, 6);
    expect(resolution.explanation).toMatch(/\(445 m\)/);
    expectOneSentence(resolution.explanation);
    expect(resolveGPUAnalysisDefault(rule, profile, 2).value).toBeCloseTo((2 * oneDegree) / 250, 6);
    // The edges of the same profile stay in degrees.
    expect(resolveGPUAnalysisDefault('extent-east', profile).unit).toBe('degree');
  });

  it('geographic data with only profile.extent gives metres from the great-circle sizes, never degrees', () => {
    const profile: GPUAnalysisDataProfile = {
      ...EMPTY_PROFILE,
      coordinates: {model: 'geographic', source: 'declared'},
      extent: [0, 0, 2, 1]
    };
    const resolution = resolveGPUAnalysisDefault(rule, profile);
    expect(resolution.unit).toBe('metre');
    expect(resolution.value).toBeCloseTo(
      Math.min(
        ...[getGPUAnalysisExtentSizes([0, 0, 2, 1], 'geographic')].map(sizes =>
          Math.min(sizes.width, sizes.height)
        )
      ) / 250,
      9
    );
    expectOneSentence(resolution.explanation);
  });

  it('geographic data with neither extent nor spacing returns undefined', () => {
    const profile: GPUAnalysisDataProfile = {
      ...EMPTY_PROFILE,
      coordinates: {model: 'geographic', source: 'declared'}
    };
    const resolution = resolveGPUAnalysisDefault(rule, profile);
    expect(resolution.value).toBeUndefined();
    expect(resolution.unit).toBe('metre');
    expect(resolution.explanation).toMatch(/needs an extent/);
    expectOneSentence(resolution.explanation);
  });

  it('planar data with spacing uses the spacing extent in coordinate units', () => {
    const profile = createProfile([0, 0, 500, 100, 250, 50]);
    const resolution = resolveGPUAnalysisDefault(rule, profile);
    expect(resolution.value).toBeCloseTo(100 / 250, 12);
    expect(resolution.unit).toBe('coordinate-unit');
  });

  it('planar data without spacing falls back to the profile extent', () => {
    const profile: GPUAnalysisDataProfile = {
      ...EMPTY_PROFILE,
      coordinates: {model: 'planar', source: 'declared'},
      extent: [0, 0, 1000, 500]
    };
    const resolution = resolveGPUAnalysisDefault(rule, profile);
    expect(resolution.value).toBe(2);
    expect(resolution.unit).toBe('coordinate-unit');
    expect(resolveGPUAnalysisDefault(rule, profile, 0.5).value).toBe(1);
  });

  it('a zero width or height gives undefined, not 0, with and without spacing, in both models', () => {
    for (const model of ['planar', 'geographic'] as const) {
      for (const coordinates of [
        [0, 5, 10, 5, 20, 5],
        [5, 0, 5, 10, 5, 20]
      ]) {
        const withSpacing = createProfile(coordinates, model);
        expect(resolveGPUAnalysisDefault(rule, withSpacing).value).toBeUndefined();
        const withoutSpacing = {...withSpacing, spacing: undefined};
        expect(resolveGPUAnalysisDefault(rule, withoutSpacing).value).toBeUndefined();
      }
    }
    const single = createProfile([3, 4]);
    expect(resolveGPUAnalysisDefault(rule, single).value).toBeUndefined();
  });

  it('records state the unit of every rule and that edges ignore the factor', () => {
    const byId = new Map(listGPUAnalysisDefaultRules().map(record => [record.id, record]));
    expect(byId.get(rule)?.unit).toMatch(/metres for geographic/);
    for (const id of ['extent-west', 'extent-south', 'extent-east', 'extent-north'] as const) {
      expect(byId.get(id)?.unit).toMatch(/degrees for geographic.*factor is ignored/i);
    }
    for (const record of byId.values()) {
      expect(record.unit.length).toBeGreaterThan(10);
    }
  });
});

describe('resolveGPUAnalysisDefault sampled clause', () => {
  it('nearest-neighbour rules and the knee state the sample size and uncertainty without asserting a bias direction', () => {
    const spacing = computeGPUAnalysisSpacingStatistics(createClusteredPoints(7, 3000), {
      coordinateModel: 'planar',
      maximumSampleSize: 900
    });
    const profile: GPUAnalysisDataProfile = {kind: 'points', rowCount: 3000, columns: [], spacing};
    for (const rule of [
      'maximum-nearest-neighbor-distance',
      'median-nearest-neighbor-distance',
      'k-distance-knee'
    ] as const) {
      const resolution = resolveGPUAnalysisDefault(rule, profile);
      if (resolution.value === undefined) {
        continue;
      }
      expect(resolution.sampled).toBe(true);
      expect(resolution.explanation).toMatch(
        /Distances were computed within a sample of 900 of 3,000 points \(30%\). Sample statistics may differ substantially from full-data statistics; no correction is applied/
      );
    }
    expect(resolveGPUAnalysisDefault('maximum-nearest-neighbor-distance', profile).sampled).toBe(
      true
    );
    const exact = createProfile(createClusteredPoints(7, 3000));
    expect(
      resolveGPUAnalysisDefault('maximum-nearest-neighbor-distance', exact).explanation
    ).not.toMatch(/sample/);
  });
});

// Generators of the knee experiment (knee2.py gauss_points and lattice_jitter).
function createGaussianPoints(
  seed: number,
  count: number,
  centers: readonly (readonly [number, number])[],
  spread: number,
  noiseFraction = 0
): number[] {
  const random = createRandom(seed);
  const normal = () => {
    const u = Math.max(random(), 1e-12);
    const v = random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const coordinates: number[] = [];
  for (let index = 0; index < count; index++) {
    if (noiseFraction > 0 && random() < noiseFraction) {
      const x = random() * 1000;
      const y = random() * 1000;
      coordinates.push(x, y);
    } else {
      const [centerX, centerY] = centers[index % centers.length];
      const x = centerX + spread * normal();
      const y = centerY + spread * normal();
      coordinates.push(x, y);
    }
  }
  return coordinates;
}

function createJitteredLattice(seed: number, jitter: number): number[] {
  const random = createRandom(seed);
  const normal = () => {
    const u = Math.max(random(), 1e-12);
    const v = random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const coordinates: number[] = [];
  for (let row = 0; row < 30; row++) {
    for (let column = 0; column < 30; column++) {
      const x = 10 * column + jitter * normal();
      const y = 10 * row + jitter * normal();
      coordinates.push(x, y);
    }
  }
  return coordinates;
}

function getRangeOverMedian(distances: readonly number[]): number {
  const sorted = [...distances].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  const median =
    sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  return (sorted[sorted.length - 1] - sorted[0]) / median;
}

describe('resolveGPUAnalysisDefault knee on realistic sets (knee3.py, scikit-learn DBSCAN at the knee)', () => {
  // knee3.py prints, per set: bend, range / median of the 512-quantile 3rd-neighbour curve, the
  // knee value and DBSCAN(eps = knee, min_samples = 4): clusters found and noise fraction.
  //   single Gaussian blob (2,000 points)   bend 0.776 ratio 14.87  knee 4.25  3 clusters, 3.3% noise
  //   two clean clusters (2,000 points)     bend 0.719 ratio 11.87  knee 4.99  8 clusters, 7.1% noise
  //   lattice, jitter 0.001 (900 points)    bend 0.995 ratio  0.41  knee 10.0  1 cluster,  0.0% noise
  //   four clusters + 15% noise (2,000)     bend 0.789 ratio 84.78  knee 6.28  5 clusters, 15.3% noise
  // Only the last is a knee: DBSCAN there recovers the noise fraction. The first two split clean
  // clusters and invent noise; the lattice's curve spans 0.4 of its median, so its 0.995 bend is
  // an artefact of scaling by a tiny range.
  const cases: [string, number[], number, number, number | undefined][] = [
    [
      'one Gaussian blob',
      createGaussianPoints(1, 2000, [[500, 500]], 20),
      0.7763399517999697,
      14.874672282579324,
      undefined
    ],
    [
      'two clean clusters',
      createGaussianPoints(
        2,
        2000,
        [
          [300, 500],
          [700, 500]
        ],
        20
      ),
      0.7185956450134738,
      11.870552482021697,
      undefined
    ],
    [
      'a lattice with jitter 0.001',
      createJitteredLattice(4, 0.001),
      0.9948520037776881,
      0.4143494752682914,
      undefined
    ],
    [
      'four clusters with 15 percent noise',
      createGaussianPoints(
        6,
        2000,
        [
          [200, 200],
          [800, 250],
          [500, 700],
          [250, 850]
        ],
        8,
        0.15
      ),
      0.7892214534891084,
      84.78125933370485,
      6.279982444122353
    ]
  ];
  for (const [name, coordinates, bend, ratio, kneeValue] of cases) {
    it(`${name}: ${kneeValue === undefined ? 'no knee' : 'a knee at the oracle value'}`, () => {
      const profile = createProfile(coordinates);
      const distances = profile.spacing?.kDistance?.distances ?? [];
      expect(getRangeOverMedian(distances)).toBeCloseTo(ratio, 6);
      const resolution = resolveGPUAnalysisDefault('k-distance-knee', profile);
      if (kneeValue === undefined) {
        expect(resolution.value).toBeUndefined();
        expect(resolution.explanation).toMatch(/rises by only|no clear knee/);
        // The bend alone would have accepted all three (0.72, 0.78, 0.995 against 0.65).
        expect(bend).toBeGreaterThan(GPU_ANALYSIS_KNEE_MINIMUM_BEND);
        expect(ratio).toBeLessThan(GPU_ANALYSIS_KNEE_MINIMUM_RANGE_OVER_MEDIAN);
      } else {
        expect(ratio).toBeGreaterThan(GPU_ANALYSIS_KNEE_MINIMUM_RANGE_OVER_MEDIAN);
        expect(resolution.value).toBeCloseTo(kneeValue, 9);
        expect(resolution.explanation).toMatch(
          /points whose 3rd-neighbour is farther than this are in sparser surroundings than most/
        );
        expect(resolution.explanation).not.toMatch(/likely noise/);
      }
      expectOneSentence(resolution.explanation);
    });
  }

  it('states the range-over-median limit in the explanation', () => {
    const profile = createProfile(createJitteredLattice(4, 0.001));
    const explanation = resolveGPUAnalysisDefault('k-distance-knee', profile).explanation;
    expect(explanation).toMatch(/rises by only 0\.414 times its median \(at least 30 is needed\)/);
    expect(explanation).toContain("does not meet this rule's range-to-median threshold");
    expect(explanation).not.toContain('its bend is not a knee');
  });
});

describe('resolveGPUAnalysisDefault outlier-resistant distance and honest maximum', () => {
  /** Ten tight clusters about 1 km across (100 points in 0.01 degrees) and, optionally, one far point. */
  function createClustersWithOutlier(withOutlier: boolean): number[] {
    const random = createRandom(11);
    const coordinates: number[] = [];
    for (let cluster = 0; cluster < 10; cluster++) {
      const centerLongitude = 10 + random() * 2;
      const centerLatitude = 50 + random() * 2;
      for (let index = 0; index < 100; index++) {
        coordinates.push(
          centerLongitude + (random() - 0.5) * 0.01,
          centerLatitude + (random() - 0.5) * 0.01
        );
      }
    }
    if (withOutlier) {
      coordinates.push(-70, -33);
    }
    return coordinates;
  }

  it('with one point 14,000 km away the maximum names the isolated point, the median and the alternative', () => {
    const profile = createProfile(createClustersWithOutlier(true), 'geographic');
    const maximum = resolveGPUAnalysisDefault('maximum-nearest-neighbor-distance', profile);
    expect(maximum.value).toBeGreaterThan(1e7);
    expect(maximum.explanation).toMatch(
      /set by \d+ isolated points? in the unscaled statistic \(the median is \d+(\.\d+)? m\)/
    );
    expect(maximum.explanation).toMatch(/outlier-resistant rule/);
    expect(maximum.explanation).not.toMatch(
      /at the unscaled statistic, every point has at least one neighbour/
    );
    expectOneSentence(
      maximum.explanation.split('. Distances were computed')[0] + (maximum.sampled ? '.' : '')
    );
    const resistant = resolveGPUAnalysisDefault(
      'outlier-resistant-nearest-neighbor-distance',
      profile
    );
    expect(resistant.value).toBeCloseTo(profile.spacing?.nearestNeighbor?.upperFence ?? NaN, 9);
    expect(resistant.value).toBeLessThan(5000);
    expect(resistant.explanation).toMatch(
      /\d+ points? (lies|lie) beyond the unscaled spacing fence/
    );
    expect(resistant.unit).toBe('metre');
    expectOneSentence(resistant.explanation);
  });

  it('without the outlier the maximum keeps its plain sentence and no point is beyond the fence', () => {
    const profile = createProfile(createClustersWithOutlier(false), 'geographic');
    const maximum = resolveGPUAnalysisDefault('maximum-nearest-neighbor-distance', profile);
    expect(profile.spacing?.nearestNeighbor?.aboveUpperFenceCount).toBeGreaterThanOrEqual(0);
    if (profile.spacing?.nearestNeighbor?.aboveUpperFenceCount === 0) {
      expect(maximum.explanation).toMatch(
        /at the unscaled statistic, every point has at least one neighbour/
      );
    }
    const uniform = createProfile(createUniformPoints(1, 1500));
    const resistant = resolveGPUAnalysisDefault(
      'outlier-resistant-nearest-neighbor-distance',
      uniform
    );
    expect(resistant.value).toBeCloseTo(45.78487229149349, 9);
    expect(resistant.explanation).toMatch(/no points lie beyond the unscaled spacing fence/);
  });

  it('counts several isolated points in the plural and scales by the factor', () => {
    const coordinates = [...createUniformPoints(12, 300), 1e5, 0, 2e5, 0, 3e5, 0];
    const profile = createProfile(coordinates);
    const resistant = resolveGPUAnalysisDefault(
      'outlier-resistant-nearest-neighbor-distance',
      profile,
      2
    );
    expect(resistant.explanation).toMatch(/times 2 = /);
    expect(resistant.value).toBeCloseTo(
      2 * (profile.spacing?.nearestNeighbor?.upperFence ?? NaN),
      9
    );
    const maximum = resolveGPUAnalysisDefault('maximum-nearest-neighbor-distance', profile);
    expect(maximum.explanation).toMatch(/set by \d+ isolated points/);
  });

  it('returns undefined without neighbour statistics', () => {
    for (const profile of [
      EMPTY_PROFILE,
      createProfile([1, 1]),
      createProfile([1, 1, 1, 1, 1, 1])
    ]) {
      expect(
        resolveGPUAnalysisDefault('outlier-resistant-nearest-neighbor-distance', profile).value
      ).toBeUndefined();
    }
  });

  it('says a sample was used for the fence', () => {
    const spacing = computeGPUAnalysisSpacingStatistics(createUniformPoints(6, 3000), {
      coordinateModel: 'planar',
      maximumSampleSize: 600
    });
    const resolution = resolveGPUAnalysisDefault('outlier-resistant-nearest-neighbor-distance', {
      kind: 'points',
      rowCount: 3000,
      columns: [],
      spacing
    });
    expect(resolution.sampled).toBe(true);
    expect(resolution.explanation).toMatch(/sample of 600 of 3,000 points/);
  });
});

describe('resolveGPUAnalysisDefault column rules', () => {
  const profile: GPUAnalysisDataProfile = {
    kind: 'points',
    rowCount: 1000,
    columns: [
      {
        name: 'price',
        type: 'number',
        rowCount: 1000,
        nullCount: 0,
        minimum: 10,
        maximum: 110,
        mean: 50,
        standardDeviation: 12
      },
      {name: 'label', type: 'category', rowCount: 1000, nullCount: 0, distinctCount: 4},
      {
        name: 'constant',
        type: 'number',
        rowCount: 1000,
        nullCount: 0,
        minimum: 5,
        maximum: 5,
        standardDeviation: 0
      },
      {name: 'bare', type: 'integer', rowCount: 1000, nullCount: 0},
      {
        name: 'sampledColumn',
        type: 'number',
        rowCount: 1000,
        nullCount: 0,
        minimum: 0,
        maximum: 4,
        standardDeviation: 1.5,
        sampleSize: 200
      }
    ]
  };

  it('returns the variance, standard deviation and range of the named column', () => {
    const variance = resolveGPUAnalysisDefault('column-variance', profile, 1, {column: 'price'});
    expect(variance.value).toBe(144);
    expect(variance.unit).toBe('column-unit');
    expect(variance.sampled).toBe(false);
    expect(variance.explanation).toBe(
      'Variance of column "price" (144 in squared units of the column).'
    );
    expect(
      resolveGPUAnalysisDefault('column-standard-deviation', profile, 1, {column: 'price'}).value
    ).toBe(12);
    const range = resolveGPUAnalysisDefault('column-range', profile, 2, {column: 'price'});
    expect(range.value).toBe(200);
    expect(range.factor).toBe(2);
    expect(range.explanation).toMatch(/times 2 = 200/);
    expectOneSentence(range.explanation);
  });

  it('returns undefined with the reason for a missing, unknown, non-numeric or lacking column', () => {
    const cases: [string, string | undefined, RegExp][] = [
      ['column-variance', undefined, /needs options\.column/],
      ['column-variance', '', /needs options\.column/],
      ['column-standard-deviation', 'missing', /no column named "missing"/],
      ['column-range', 'label', /type category, not a number or a time/],
      ['column-standard-deviation', 'constant', /no standard deviation above 0/],
      ['column-range', 'constant', /no minimum and maximum that differ/],
      ['column-standard-deviation', 'bare', /no standard deviation above 0/],
      ['column-range', 'bare', /no minimum and maximum that differ/]
    ];
    for (const [rule, column, pattern] of cases) {
      const resolution = resolveGPUAnalysisDefault(rule as 'column-range', profile, 1, {column});
      expect(resolution.value).toBeUndefined();
      expect(resolution.explanation).toMatch(pattern);
      expect(resolution.unit).toBe('column-unit');
      expectOneSentence(resolution.explanation);
    }
    expect(resolveGPUAnalysisDefault('column-range', profile).value).toBeUndefined();
  });

  it('marks statistics computed on fewer rows than the column has as sampled', () => {
    const resolution = resolveGPUAnalysisDefault('column-standard-deviation', profile, 1, {
      column: 'sampledColumn'
    });
    expect(resolution.sampled).toBe(true);
    expect(resolution.value).toBe(1.5);
    expect(resolution.explanation).toMatch(/from a sample of 200 of 1,000 rows/);
  });

  it('ignores options.column for the other rules', () => {
    const withColumn = resolveGPUAnalysisDefault(
      'extent-west',
      {...profile, extent: [1, 2, 3, 4]},
      1,
      {column: 'price'}
    );
    expect(withColumn.value).toBe(1);
  });
});

describe('resolveGPUAnalysisDefault mean point spacing', () => {
  it('planar: sqrt(width * height / count) of every point', () => {
    const coordinates = createUniformPoints(1, 1500);
    const profile = createProfile(coordinates);
    let minimumX = Infinity;
    let minimumY = Infinity;
    let maximumX = -Infinity;
    let maximumY = -Infinity;
    for (let index = 0; index < 1500; index++) {
      minimumX = Math.min(minimumX, coordinates[2 * index]);
      maximumX = Math.max(maximumX, coordinates[2 * index]);
      minimumY = Math.min(minimumY, coordinates[2 * index + 1]);
      maximumY = Math.max(maximumY, coordinates[2 * index + 1]);
    }
    const expected = Math.sqrt(((maximumX - minimumX) * (maximumY - minimumY)) / 1500);
    const resolution = resolveGPUAnalysisDefault('mean-point-spacing', profile);
    expect(resolution.value).toBeCloseTo(expected, 12);
    expect(resolution.unit).toBe('coordinate-unit');
    expect(resolution.explanation).toMatch(/holds one point on average/);
    expect(resolution.explanation).toMatch(/about 1 point on average/);
    const doubled = resolveGPUAnalysisDefault('mean-point-spacing', profile, 3);
    expect(doubled.value).toBeCloseTo(3 * expected, 12);
    expect(doubled.explanation).toMatch(/about 9 points on average/);
    expectOneSentence(resolution.explanation);
  });

  it('geographic: metres from the great-circle extent sizes', () => {
    const coordinates = createGeographicTestPoints();
    const profile = createProfile(coordinates, 'geographic');
    const sizes = getGPUAnalysisExtentSizes(profile.extent ?? [0, 0, 0, 0], 'geographic');
    const resolution = resolveGPUAnalysisDefault('mean-point-spacing', profile);
    expect(resolution.unit).toBe('metre');
    expect(resolution.value).toBeCloseTo(
      Math.sqrt((sizes.width * sizes.height) / (coordinates.length / 2)),
      6
    );
  });

  it('uses every point, not the sample', () => {
    const coordinates = createUniformPoints(5, 3000);
    const full = resolveGPUAnalysisDefault('mean-point-spacing', createProfile(coordinates));
    const spacing = computeGPUAnalysisSpacingStatistics(coordinates, {
      coordinateModel: 'planar',
      maximumSampleSize: 300
    });
    const sampled = resolveGPUAnalysisDefault('mean-point-spacing', {
      kind: 'points',
      rowCount: 3000,
      columns: [],
      spacing
    });
    expect(sampled.value).toBe(full.value);
    expect(sampled.sampled).toBe(false);
  });

  it('is undefined without spacing, for collinear points and for one point', () => {
    expect(resolveGPUAnalysisDefault('mean-point-spacing', EMPTY_PROFILE).value).toBeUndefined();
    expect(
      resolveGPUAnalysisDefault('mean-point-spacing', createProfile([0, 0, 5, 0, 9, 0])).value
    ).toBeUndefined();
    expect(
      resolveGPUAnalysisDefault('mean-point-spacing', createProfile([3, 3])).value
    ).toBeUndefined();
  });
});

function createGeographicTestPoints(): number[] {
  const random = createRandom(21);
  const coordinates: number[] = [];
  for (let index = 0; index < 500; index++) {
    coordinates.push(-74 + random() * 0.6, 40.5 + random() * 0.4);
  }
  return coordinates;
}

describe('resolveGPUAnalysisDefault extent rules use profile.extent (the geometry), not the anchors', () => {
  // A coastline of one feature has one anchor, so the spacing extent is a point, while the
  // geometry extent (all vertices) is 7 degrees by 1 degree.
  const anchorSpacing = computeGPUAnalysisSpacingStatistics([10, 50], {
    coordinateModel: 'geographic'
  });
  const lineProfile: GPUAnalysisDataProfile = {
    kind: 'lines',
    rowCount: 1,
    coordinates: {model: 'geographic', source: 'declared'},
    extent: [10, 50, 17, 51],
    columns: [],
    spacing: anchorSpacing
  };

  it('cell size and diagonal come from the geometry extent', () => {
    const sizes = getGPUAnalysisExtentSizes([10, 50, 17, 51], 'geographic');
    expect(anchorSpacing.extentWidth).toBe(0);
    const cell = resolveGPUAnalysisDefault('extent-shorter-side-over-250', lineProfile);
    expect(cell.value).toBeCloseTo(Math.min(sizes.width, sizes.height) / 250, 9);
    expect(cell.unit).toBe('metre');
    expect(resolveGPUAnalysisDefault('extent-diagonal', lineProfile).value).toBeCloseTo(
      sizes.diagonal,
      6
    );
  });

  it('falls back to the spacing extent only when profile.extent is absent', () => {
    const spacing = computeGPUAnalysisSpacingStatistics([0, 0, 2, 1, 1, 0.5], {
      coordinateModel: 'geographic'
    });
    const withoutExtent: GPUAnalysisDataProfile = {
      kind: 'points',
      rowCount: 3,
      columns: [],
      spacing
    };
    expect(
      resolveGPUAnalysisDefault('extent-shorter-side-over-250', withoutExtent).value
    ).toBeCloseTo(Math.min(spacing.extentWidth, spacing.extentHeight) / 250, 9);
    // With a wider geometry extent the geometry wins.
    const wider = resolveGPUAnalysisDefault('extent-shorter-side-over-250', {
      ...withoutExtent,
      coordinates: {model: 'geographic', source: 'declared'},
      extent: [0, 0, 20, 10]
    });
    expect(wider.value).toBeCloseTo(
      getGPUAnalysisExtentSizes([0, 0, 20, 10], 'geographic').height / 250,
      9
    );
  });

  it('mean point spacing takes its area from profile.extent and its count from the spacing', () => {
    const sizes = getGPUAnalysisExtentSizes([10, 50, 17, 51], 'geographic');
    const spacing = {...anchorSpacing, pointCount: 400};
    const resolution = resolveGPUAnalysisDefault('mean-point-spacing', {...lineProfile, spacing});
    expect(resolution.value).toBeCloseTo(Math.sqrt((sizes.width * sizes.height) / 400), 6);
  });

  it('the source-units rule gives degrees for geographic data and coordinate units for planar data', () => {
    const degrees = resolveGPUAnalysisDefault(
      'extent-shorter-side-over-250-in-source-units',
      lineProfile
    );
    expect(degrees.unit).toBe('degree');
    expect(degrees.value).toBeCloseTo(1 / 250, 12);
    expect(degrees.explanation).toMatch(/a degree is not a constant ground distance/);
    expect(
      resolveGPUAnalysisDefault('extent-shorter-side-over-250-in-source-units', lineProfile, 3)
        .value
    ).toBeCloseTo(3 / 250, 12);
    const planar: GPUAnalysisDataProfile = {
      ...EMPTY_PROFILE,
      coordinates: {model: 'planar', source: 'declared'},
      extent: [0, 0, 1000, 500]
    };
    const projected = resolveGPUAnalysisDefault(
      'extent-shorter-side-over-250-in-source-units',
      planar
    );
    expect(projected.unit).toBe('coordinate-unit');
    expect(projected.value).toBe(2);
    // A 4 degree wide box across 180 has a 4 degree side, not 356.
    const crossing = resolveGPUAnalysisDefault('extent-shorter-side-over-250-in-source-units', {
      ...EMPTY_PROFILE,
      coordinates: {model: 'geographic', source: 'declared'},
      extent: [178, 60, -178, 70]
    });
    expect(crossing.value).toBeCloseTo(4 / 250, 12);
  });

  it('the source-units rule is undefined for a degenerate or missing extent', () => {
    const flat: GPUAnalysisDataProfile = {...EMPTY_PROFILE, extent: [0, 5, 10, 5]};
    expect(
      resolveGPUAnalysisDefault('extent-shorter-side-over-250-in-source-units', flat).value
    ).toBeUndefined();
    expect(
      resolveGPUAnalysisDefault('extent-shorter-side-over-250-in-source-units', EMPTY_PROFILE).value
    ).toBeUndefined();
  });

  it('west and east edges of an antimeridian arc are the arc ends and say the data crosses 180', () => {
    const spacing = computeGPUAnalysisSpacingStatistics(
      Array.from({length: 40}, (_, index) => [index % 2 === 0 ? 178 : -178, 60 + index / 4]).flat(),
      {coordinateModel: 'geographic'}
    );
    const profile = createProfileFromSpacing(spacing);
    const west = resolveGPUAnalysisDefault('extent-west', profile);
    const east = resolveGPUAnalysisDefault('extent-east', profile);
    expect(west.value).toBe(178);
    expect(east.value).toBe(-178);
    expect(west.explanation).toMatch(/the data crosses the antimeridian/);
    expect(east.explanation).toMatch(/the data crosses the antimeridian/);
    expect(resolveGPUAnalysisDefault('extent-south', profile).explanation).not.toMatch(
      /antimeridian/
    );
    // Identical boxes at longitude 0 and 180 get the same cell size and diagonal.
    const box = (centerLongitude: number) => {
      const coordinates: number[] = [];
      for (let column = 0; column < 10; column++) {
        for (let row = 0; row < 10; row++) {
          let longitude = centerLongitude - 2 + (4 * column) / 9;
          if (longitude > 180) {
            longitude -= 360;
          }
          coordinates.push(longitude, 60 + (10 * row) / 9);
        }
      }
      return createProfile(coordinates, 'geographic');
    };
    for (const rule of [
      'extent-shorter-side-over-250',
      'extent-diagonal',
      'mean-point-spacing'
    ] as const) {
      const atZero = resolveGPUAnalysisDefault(rule, box(0));
      const atAntimeridian = resolveGPUAnalysisDefault(rule, box(180));
      expect(atAntimeridian.value).toBeCloseTo(atZero.value ?? NaN, 6);
    }
    // About 188 km wide, so the cell is under 1 km, not the 68 km of the plain range.
    expect(resolveGPUAnalysisDefault('extent-shorter-side-over-250', box(180)).value).toBeLessThan(
      1000
    );
  });
});

function createProfileFromSpacing(spacing: GPUAnalysisSpacingStatistics): GPUAnalysisDataProfile {
  return {
    kind: 'points',
    rowCount: spacing.pointCount,
    coordinates: {model: spacing.coordinateModel, source: 'declared'},
    extent: spacing.extent,
    columns: [],
    spacing
  };
}

describe('resolveGPUAnalysisDefault column sum', () => {
  const profile: GPUAnalysisDataProfile = {
    kind: 'attributes',
    rowCount: 5,
    // /usr/bin/python3: np.nansum([2.,-4.,np.nan,0.,10.]) = 8;
    // np.nanmean(...) = 2 and the non-null count is 4.
    columns: [{name: 'value', type: 'number', rowCount: 5, nullCount: 1, mean: 2}]
  };

  it('returns the sum from the mean and non-null count with the requested factor', () => {
    const resolution = resolveGPUAnalysisDefault('column-sum', profile, 3, {column: 'value'});
    expect(resolution.value).toBe(24);
    expect(resolution.unit).toBe('column-unit');
    expect(resolution.factor).toBe(3);
    expect(resolution.sampled).toBe(false);
    expect(resolution.explanation).toMatch(/mean 2 times 4 non-null rows/);
    expect(resolution.explanation).toMatch(/times 3 = 24/);
  });

  it.each([-2, 0])('allows a finite signed or zero sum from mean %s', mean => {
    const resolution = resolveGPUAnalysisDefault(
      'column-sum',
      {
        ...profile,
        columns: [{...profile.columns[0], mean}]
      },
      1,
      {column: 'value'}
    );
    expect(resolution.value).toBe(mean * 4);
  });

  it.each(['number', 'integer', 'time'] as const)('sums a non-null %s column', type => {
    const resolution = resolveGPUAnalysisDefault(
      'column-sum',
      {
        ...profile,
        columns: [{...profile.columns[0], type, nullCount: 0}]
      },
      1,
      {column: 'value'}
    );
    expect(resolution.value).toBe(10);
  });

  it.each([0, 5])('declines an empty or all-null column with %s rows and no mean', rowCount => {
    const resolution = resolveGPUAnalysisDefault(
      'column-sum',
      {
        ...profile,
        rowCount,
        columns: [{name: 'value', type: 'number', rowCount, nullCount: rowCount}]
      },
      1,
      {column: 'value'}
    );
    expect(resolution.value).toBeUndefined();
    expect(resolution.explanation).toMatch(/finite mean/);
  });

  it('estimates the full sum from the sampled mean and sampled non-null share', () => {
    // /usr/bin/python3: np.nanmean([0.,10.,20.,np.nan])*1000*(3/4) = 7500.
    const resolution = resolveGPUAnalysisDefault(
      'column-sum',
      {
        ...profile,
        rowCount: 1000,
        columns: [{...profile.columns[0], rowCount: 1000, sampleSize: 4, nullCount: 1, mean: 10}]
      },
      2,
      {column: 'value'}
    );
    expect(resolution.value).toBe(15000);
    expect(resolution.sampled).toBe(true);
    expect(resolution.explanation).toMatch(
      /mean 10 times 1,000 full rows times the sample non-null share 3\/4/
    );
    expect(resolution.explanation).toMatch(/estimates the full sum/);
    expect(resolution.explanation).toMatch(/times 2 = 15,000/);
  });

  it.each([
    {mean: undefined},
    {mean: NaN},
    {mean: Infinity},
    {rowCount: undefined},
    {nullCount: undefined},
    {nullCount: -1},
    {nullCount: 6},
    {sampleSize: 0},
    {sampleSize: 2, nullCount: 3}
  ])('declines missing or invalid mean/count statistics %j', overrides => {
    const column = {
      ...profile.columns[0],
      ...overrides
    } as GPUAnalysisDataProfile['columns'][number];
    const resolution = resolveGPUAnalysisDefault('column-sum', {...profile, columns: [column]}, 1, {
      column: 'value'
    });
    expect(resolution.value).toBeUndefined();
    expect(resolution.explanation).toMatch(/No sum of a column/);
  });

  it('requires a named numeric or time column', () => {
    expect(resolveGPUAnalysisDefault('column-sum', profile).value).toBeUndefined();
    expect(
      resolveGPUAnalysisDefault('column-sum', profile, 1, {column: 'missing'}).value
    ).toBeUndefined();
    expect(
      resolveGPUAnalysisDefault(
        'column-sum',
        {
          ...profile,
          columns: [{...profile.columns[0], type: 'category'}]
        },
        1,
        {column: 'value'}
      ).value
    ).toBeUndefined();
  });

  it('declines a sum that overflows after scaling', () => {
    const resolution = resolveGPUAnalysisDefault('column-sum', profile, Number.MAX_VALUE, {
      column: 'value'
    });
    expect(resolution.value).toBeUndefined();
    expect(resolution.explanation).toMatch(/final scaled value must be finite/);
  });
});

describe('listGPUAnalysisDefaultRules records of the added rules', () => {
  it('describes the source-units rule, mean point spacing and the column rules honestly', () => {
    const byId = new Map(listGPUAnalysisDefaultRules().map(record => [record.id, record]));
    expect(byId.get('extent-shorter-side-over-250-in-source-units')?.unit).toMatch(
      /parameters measured in coordinate units/
    );
    expect(byId.get('extent-shorter-side-over-250-in-source-units')?.unit).toMatch(
      /a degree is not a constant ground distance/
    );
    expect(byId.get('mean-point-spacing')?.source).toMatch(
      /Esri's Optimized Hot Spot Analysis instead/
    );
    expect(byId.get('outlier-resistant-nearest-neighbor-distance')?.source).toMatch(/Tukey/);
    for (const id of [
      'column-variance',
      'column-standard-deviation',
      'column-range',
      'column-sum'
    ] as const) {
      expect(byId.get(id)?.needs).toMatch(/options\.column/);
    }
    expect(byId.get('column-sum')?.needs).toMatch(/mean.*rowCount.*nullCount/);
  });
});

// knee3.py, quart.py and sample.py: see the scripts quoted below and in the spacing spec.

/*
ORACLE SCRIPTS. Save as gen.py, oracle.py and knee.py in one folder and run each with /usr/bin/python3.
The values above come from the outputs of oracle.py (bandwidthUniform, bandwidthClustered, and the
planar nearest-neighbour values) and knee.py (clustered2, uniform1, powers 6 and 10).

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

----- knee.py -----
import numpy as np
from scipy.spatial import cKDTree
from gen import *
def knee(sorted_vals):
    n=len(sorted_vals)
    q=np.array(sorted_vals) if n<=512 else np.array([sorted_vals[round(i*(n-1)/511)] for i in range(512)])
    m=len(q); x=np.arange(m)/(m-1); y=(q-q[0])/(q[-1]-q[0])
    dev=x-y; i=int(np.argmax(dev)); return dev[i], i/(m-1), q[i]
def curve(p,k=3):
    d,_=cKDTree(p).query(p,k=k+1); return np.sort(d[:,k])
sets={}
for s in range(1,9): sets['uniform%d'%s]=uniform(s,1500)
sets['uniform200']=uniform(9,200)
sets['uniform5000']=uniform(10,5000)
for s in range(1,5): sets['clustered%d'%s]=clustered(s,1600)
for name,p in sets.items(): print(name, '%.3f at %.3f value %.2f'%knee(curve(p)))
# lattice
lat=np.array([[10.0*i,10.0*j] for j in range(30) for i in range(30)])
print('lattice', knee(curve(lat)) if curve(lat)[-1]>curve(lat)[0] else 'flat')

----- knee2.py (functions used by knee3.py; its table is in the knee test) -----
import numpy as np, math
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from gen import Rng, uniform, clustered
def gauss_points(seed, n, centers, sd, noise_fraction=0.0, box=1000.0):
    r = Rng(seed); out = []
    for i in range(n):
        if noise_fraction > 0 and r.next() < noise_fraction:
            out.append([r.next()*box, r.next()*box])
        else:
            cx, cy = centers[i % len(centers)]
            out.append([cx + sd*r.normal(), cy + sd*r.normal()])
    return np.array(out)
def lattice_jitter(seed, side=30, jitter=0.001):
    r = Rng(seed)
    return np.array([[10.0*i + jitter*r.normal(), 10.0*j + jitter*r.normal()] for j in range(side) for i in range(side)])
def curve(p, k=3):
    d,_ = cKDTree(p).query(p, k=k+1); s = np.sort(d[:,k]); n = len(s)
    return s if n <= 512 else np.array([s[round(i*(n-1)/511)] for i in range(512)])
def knee(q):
    m=len(q); x=np.arange(m)/(m-1); y=(q-q[0])/(q[-1]-q[0]); dev=x-y; i=int(np.argmax(dev)); return float(dev[i]), float(q[i])
sets = {
 'single gaussian': gauss_points(1, 2000, [(500,500)], 20),
 'two clean clusters': gauss_points(2, 2000, [(300,500),(700,500)], 20),
 'three clean clusters': gauss_points(3, 3000, [(200,200),(800,300),(500,800)], 15),
 'lattice jitter 0.001': lattice_jitter(4),
 'lattice jitter 0.5': lattice_jitter(4, jitter=0.5),
 'uniform': uniform(5, 2000),
 'clusters 5% noise': clustered(2, 1600),
 'clusters 15% noise': gauss_points(6, 2000, [(200,200),(800,250),(500,700),(250,850)], 8, 0.15),
 'clusters 30% noise': gauss_points(7, 2000, [(200,200),(800,250),(500,700),(250,850)], 8, 0.30),
}
for name,p in sets.items():
    q = curve(p); bend, val = knee(q)
    ratio = (q[-1]-q[0])/np.median(q)
    lab = DBSCAN(eps=val, min_samples=4).fit_predict(p)
    nc = len(set(lab)-{-1}); noise = float((lab==-1).mean())
    print('%-22s bend %.3f range/median %6.2f  knee %.3g  clusters %d noise %.3f' % (name, bend, ratio, val, nc, noise))
print('--- clean gaussian sets by size')
for n in (300, 1000, 5000, 20000, 100000):
    for seed in (11, 12):
        p = gauss_points(seed, n, [(500,500)], 20)
        q = curve(p); bend, val = knee(q); print(n, seed, 'bend %.3f ratio %.1f'%(bend,(q[-1]-q[0])/np.median(q)))
for n in (1000, 20000):
    p = gauss_points(13, n, [(300,500),(700,500)], 20); q=curve(p); bend,val=knee(q); print('two', n, 'bend %.3f ratio %.1f'%(bend,(q[-1]-q[0])/np.median(q)))
for noise in (0.02, 0.05):
  for n in (1000, 20000):
    p = gauss_points(14, n, [(200,200),(800,250),(500,700),(250,850)], 8, noise); q=curve(p); bend,val=knee(q); lab=DBSCAN(eps=val,min_samples=4).fit_predict(p)
    print('noise',noise,n,'bend %.3f ratio %.1f noise %.3f clusters %d'%(bend,(q[-1]-q[0])/np.median(q),(lab==-1).mean(),len(set(lab)-{-1})))

----- knee3.py -----
exec(open('knee2.py').read().split("sets = {")[0])
import json
out={}
for name,p in {
 'singleGaussian': gauss_points(1, 2000, [(500,500)], 20),
 'twoClean': gauss_points(2, 2000, [(300,500),(700,500)], 20),
 'latticeJitter': lattice_jitter(4),
 'noise15': gauss_points(6, 2000, [(200,200),(800,250),(500,700),(250,850)], 8, 0.15),
}.items():
    q=curve(p); bend,val=knee(q); lab=DBSCAN(eps=val,min_samples=4).fit_predict(p)
    out[name]=dict(bend=bend,ratio=float((q[-1]-q[0])/np.median(q)),knee=val,clusters=len(set(lab)-{-1}),noise=float((lab==-1).mean()))
print(json.dumps(out,indent=1))

----- quart.py -----
import numpy as np
from scipy.spatial import cKDTree
from gen import uniform, clustered
def stats(p):
    d,_=cKDTree(p).query(p,k=2); nn=d[:,1]
    q1,q3=np.percentile(nn,[25,75]); fence=min(nn.max(), q3+3*(q3-q1))
    return dict(q1=float(q1),q3=float(q3),fence=float(fence),above=int((nn>fence).sum()))
print('uniform', stats(uniform(1,1500)))
print('clustered', stats(clustered(2,1600)))
*/
