// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  CURATED_RATE_CLUSTER_MAP_BINDING,
  publishRateClusterMap
} from '../../src/curated/bindings/curated-rate-cluster-map-binding';
import {prepareGpuRateClusterMap} from '../../src/curated/bindings/curated-rate-cluster-map-gpu';
import {makeLattice} from './curated-bindings-fixtures';

it('reports the rate-cluster pseudo p-value floor in each engine’s precision', () => {
  const areas = makeLattice(2, 2, () => ({cases: 25, population: 100}));
  const parameters = {
    casesProperty: 'cases',
    populationProperty: 'population',
    criterion: 'queen',
    coordinateSystem: 'planar',
    permutations: 999,
    seed: 1,
    significance: 0.05,
    multipleTesting: 'none'
  };
  const {rows, droppedCount, statuses, tiedAnchorCount, effectiveSignificance} =
    prepareGpuRateClusterMap({areas}, parameters);
  // Constant rates give zero prior variance and leave all four areas untested.
  const rates = rows.map(() => 0.25);
  const untested = rows.map(() => null);
  const computation = {
    rows,
    droppedCount,
    statuses,
    rawRates: rates,
    shrinkages: rows.map(() => 0),
    smoothedRates: rates,
    localI: untested,
    spatialLag: untested,
    pValues: untested,
    adjustedPValues: untested,
    neighborCounts: untested,
    clusters: rows.map(() => 'not-tested'),
    tiedAnchorCount,
    effectiveSignificance,
    globalRate: 0.25,
    priorVariance: 0
  };
  const gpu = publishRateClusterMap({areas}, parameters, [], computation, true);
  const cpu = CURATED_RATE_CLUSTER_MAP_BINDING.run!({areas}, parameters);
  for (const result of [gpu, cpu]) {
    expect(result.summary['areaCount']).toBe(4);
    const requestedSignificance = result.summary['requestedSignificance'];
    expect(Number.isFinite(requestedSignificance)).toBe(true);
    expect(requestedSignificance).toBe(parameters.significance);
  }
  // The add-one correction gives one count. The GPU doubles it and stores the quotient as Float32.
  expect(gpu.summary['minimumPseudoPValue']).toBe(Math.fround(2 / 1000));
  expect(cpu.summary['minimumPseudoPValue']).toBe(1 / 1000);
});
