// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import fixture from './clustering-fixture-oracles.json';
import {expectRelocationCenters} from './clustering-fixture-assertions';
import {
  computeKMeansOracleFromCenters,
  createSeparatedBlobScene,
  SKLEARN_RELOCATION_FIXTURES
} from './kmeans-oracle';
import {
  clusterGeographicPointsOracle,
  createGeographicBlobScene,
  ORACLE_NOISE
} from './spatial-clustering-oracle';

it('pins inertia and every relocation budget to the reproducible independent generator', () => {
  const positions = createSeparatedBlobScene(4, 60);
  const centers = [0, 60, 120, 180].flatMap(row => [positions[row * 2], positions[row * 2 + 1]]);
  const result = computeKMeansOracleFromCenters(positions, centers, 20);
  expect(Number.isFinite(result.inertia)).toBe(true);
  expect(result.inertia).toBeCloseTo(fixture.separatedInertia, 8);
  for (const scene of fixture.relocation) {
    const historical = SKLEARN_RELOCATION_FIXTURES.find(entry => entry.name === scene.name)!;
    expect(scene.positions).toEqual(historical.positions);
    for (const [budget, expected] of Object.entries(scene.steps)) {
      const actual = computeKMeansOracleFromCenters(
        Float32Array.from(scene.positions),
        scene.positions.slice(0, scene.k * 2),
        Number(budget)
      );
      expect(actual.labels).toEqual(expected.labels);
      expect(Number.isFinite(actual.inertia)).toBe(true);
      expect(actual.inertia).toBeCloseTo(expected.inertia, 8);
      expectRelocationCenters(actual.centers, expected.centers);
    }
  }
});

it('retains the historical stochastic initialization configurations alongside explicit centers (PTS-141 review)', () => {
  expect(fixture).toHaveProperty('separatedConfigurations');
  expect(
    fixture.separatedConfigurations.map(configuration => [
      configuration.initialization,
      configuration.nInit,
      configuration.randomState
    ])
  ).toEqual([
    ['explicit', 1, 0],
    ['k-means++', 10, 0],
    ['random', 10, 0]
  ]);
  for (const configuration of fixture.separatedConfigurations) {
    expect(Number.isFinite(configuration.inertia)).toBe(true);
    expect(configuration.inertia).toBeCloseTo(3707.8660708989237, 8);
  }
});

it('rejects missing relocation coordinates, including an empty output (S2)', () => {
  const expected = fixture.relocation[0].steps['1'].centers;
  expect(() => expectRelocationCenters(expected.slice(0, -1), expected)).toThrow();
  expect(() => expectRelocationCenters([], expected)).toThrow();
});

it('pins geographic DBSCAN partition and noise membership, beyond counts and sizes', () => {
  const actual = clusterGeographicPointsOracle(createGeographicBlobScene(), {
    bounds: [-180, -90, 180, 90],
    epsilon: 150,
    minimumSamples: 5
  });
  const groups = new Map<number, number[]>();
  const noise: number[] = [];
  actual.labels.forEach((label, row) => {
    if (label === ORACLE_NOISE) noise.push(row);
    else groups.set(label, [...(groups.get(label) ?? []), row]);
  });
  expect([...groups.values()].sort((first, second) => first[0] - second[0])).toEqual(
    fixture.geographic.clusters
  );
  expect(noise).toEqual(fixture.geographic.noise);
});
