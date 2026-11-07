// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '../runtime/dataset-registry';
import type {CatalogRunOutput} from './catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getFeatureIdentity,
  getPointPositions,
  getPolygonParts,
  makeCatalogTable,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeatureCollection
} from './catalog-geojson';
import {
  collectAllPositions,
  getFeatureAnchor,
  getFeatureGeometryText,
  getNormalTwoSidedPValue,
  makeSeededGenerator,
  readNumericProperty,
  addDroppedDiagnostic as addSharedDroppedDiagnostic
} from './catalog-platform3-b-common';
import {makeLocalFrame} from './catalog-local-frame';

export type Neighborhood = 'knn' | 'distance-band' | 'queen' | 'rook';

export type WeightsSettings = {
  neighborhood: Neighborhood;
  neighborCount: number;
  distanceBand: number;
  rowStandardize: boolean;
  coordinateSystem: CatalogCoordinateSystem;
};

export type SpatialWeights = {
  /** Neighbor feature indices (into the analysis set) per feature. */
  neighbors: number[][];
  weights: number[][];
  distances: number[][];
};

export type AnalysisSet = {
  /** Source feature index of every analysis row. */
  sourceIndex: number[];
  values: number[];
  weights: SpatialWeights;
  droppedCount: number;
};

/** One statistic block read back from a GPU contributor, replacing the CPU formulas. */
export type PrecomputedGlobalStatistic = {
  value: number;
  expected: number;
  variance: number;
  zScore: number;
  pValue: number;
};

/** Results computed elsewhere (the GPU engine) for the same analysis set; the CPU formulas are skipped. */
export type PrecomputedGlobalStatistics = {
  set: AnalysisSet;
  moran: PrecomputedGlobalStatistic;
  geary: PrecomputedGlobalStatistic;
};

export type PrecomputedLocalStatistics = {
  set: AnalysisSet;
  /** Local statistic per analysis row (local I or local C or Gi* z-score). */
  statistic: ArrayLike<number>;
  /** Spatial lag per analysis row (local Moran only), in standardized units. */
  spatialLag?: ArrayLike<number>;
  /** Pseudo or analytic p-value per analysis row; NaN means untested. */
  pValue?: ArrayLike<number>;
};

/** Pair work that the CPU reference accepts. */
const MAXIMUM_PAIR_WORK = 4e7;

function getVertexKey(position: readonly [number, number] | readonly number[]): string {
  return `${position[0]},${position[1]}`;
}

function buildContiguity(
  collection: CatalogFeatureCollection,
  rows: readonly number[],
  mode: 'queen' | 'rook'
): number[][] {
  const owners = new Map<string, Set<number>>();
  rows.forEach((sourceIndex, row) => {
    for (const polygon of getPolygonParts(collection.features[sourceIndex].geometry))
      for (const ring of polygon)
        for (let index = 0; index < ring.length - 1; index++) {
          const first = getVertexKey(ring[index]);
          const key =
            mode === 'queen' ? first : [first, getVertexKey(ring[index + 1])].sort().join('|');
          const set = owners.get(key) ?? new Set<number>();
          owners.set(key, set);
          set.add(row);
        }
  });
  const neighborSets = rows.map(() => new Set<number>());
  for (const set of owners.values()) {
    const members = [...set];
    for (const a of members) for (const b of members) if (a !== b) neighborSets[a].add(b);
  }
  return neighborSets.map(set => [...set].sort((a, b) => a - b));
}

/**
 * Builds the analysis set: features with a numeric value and a position, plus their neighbors.
 * Distance-based neighborhoods use anchors (the point, or a polygon centroid) in local meters.
 */
export function buildAnalysisSet(
  collection: CatalogFeatureCollection,
  valueProperty: string,
  settings: WeightsSettings
): AnalysisSet {
  const frame = makeLocalFrame(settings.coordinateSystem, collectAllPositions(collection), {
    shift: 'minimum'
  });
  const sourceIndex: number[] = [];
  const values: number[] = [];
  const anchors: [number, number][] = [];
  let droppedCount = 0;
  collection.features.forEach((feature, index) => {
    const value = valueProperty ? readNumericProperty(feature, valueProperty) : 0;
    const anchor = getFeatureAnchor(feature, frame);
    if (value === null || !anchor) {
      droppedCount++;
      return;
    }
    sourceIndex.push(index);
    values.push(value);
    anchors.push(anchor);
  });
  const count = sourceIndex.length;
  let neighbors: number[][];
  if (settings.neighborhood === 'queen' || settings.neighborhood === 'rook') {
    neighbors = buildContiguity(collection, sourceIndex, settings.neighborhood);
  } else {
    // Pair work exceeds the CPU reference capacity.
    if (count * count > MAXIMUM_PAIR_WORK)
      throw new KeplerAllocationError('capacity', {message: 'Work capacity exceeded'});
    neighbors = anchors.map((anchor, row) => {
      const candidates: {distance: number; index: number}[] = [];
      for (let other = 0; other < count; other++) {
        if (other === row) continue;
        const distance = Math.hypot(anchors[other][0] - anchor[0], anchors[other][1] - anchor[1]);
        if (settings.neighborhood === 'distance-band' && distance > settings.distanceBand) continue;
        candidates.push({distance, index: other});
      }
      candidates.sort((a, b) => a.distance - b.distance || a.index - b.index);
      const chosen =
        settings.neighborhood === 'knn' ? candidates.slice(0, settings.neighborCount) : candidates;
      return chosen.map(entry => entry.index);
    });
  }
  const distances = neighbors.map((list, row) =>
    list.map(other =>
      Math.hypot(anchors[other][0] - anchors[row][0], anchors[other][1] - anchors[row][1])
    )
  );
  const weights = neighbors.map(list =>
    list.map(() => (settings.rowStandardize && list.length ? 1 / list.length : 1))
  );
  return {sourceIndex, values, weights: {neighbors, weights, distances}, droppedCount};
}

export function addIslandDiagnostic(
  diagnostics: CatalogDiagnostic[],
  weights: SpatialWeights
): void {
  const islands = weights.neighbors.filter(list => !list.length).length;
  if (islands)
    addCatalogDiagnostic(
      diagnostics,
      'island-features',
      'warning',
      'Features without neighbors get a zero spatial lag; their local statistics are not meaningful.',
      islands
    );
}

export function getMeanAndDeviation(values: readonly number[]): {mean: number; deviation: number} {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return {mean, deviation: Math.sqrt(variance)};
}

// ---------------------------------------------------------------------------------------------
// Spatial weights table
// ---------------------------------------------------------------------------------------------

export function runSpatialWeights(
  collection: CatalogFeatureCollection,
  settings: WeightsSettings
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const set = buildAnalysisSet(collection, '', settings);
  addDroppedDiagnostic(diagnostics, set.droppedCount);
  addIslandDiagnostic(diagnostics, set.weights);
  const table = makeCatalogTable('weights', 'Spatial weights', {
    source_index: 'int32',
    source_id: 'string',
    neighbor_index: 'int32',
    neighbor_id: 'string',
    weight: 'float64',
    distance: 'float64'
  });
  set.weights.neighbors.forEach((list, row) =>
    list.forEach((other, position) =>
      appendCatalogRow(table, {
        source_index: set.sourceIndex[row],
        source_id: getFeatureIdentity(collection.features[set.sourceIndex[row]]),
        neighbor_index: set.sourceIndex[other],
        neighbor_id: getFeatureIdentity(collection.features[set.sourceIndex[other]]),
        weight: set.weights.weights[row][position],
        distance: set.weights.distances[row][position]
      })
    )
  );
  return {
    tables: [table],
    diagnostics,
    summary: {
      featureCount: set.sourceIndex.length,
      linkCount: table.rowCount,
      islandCount: set.weights.neighbors.filter(list => !list.length).length
    }
  };
}

const DROPPED_FEATURE_MESSAGE = 'Features without a numeric value or a position are skipped.';

function addDroppedDiagnostic(diagnostics: CatalogDiagnostic[], droppedCount: number): void {
  addSharedDroppedDiagnostic(diagnostics, droppedCount, DROPPED_FEATURE_MESSAGE);
}

// ---------------------------------------------------------------------------------------------
// Global statistics
// ---------------------------------------------------------------------------------------------

export function runGlobalSpatialStatistics(
  collection: CatalogFeatureCollection,
  parameters: {valueProperty: string} & WeightsSettings,
  precomputed?: PrecomputedGlobalStatistics
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const set =
    precomputed?.set ?? buildAnalysisSet(collection, parameters.valueProperty, parameters);
  addDroppedDiagnostic(diagnostics, set.droppedCount);
  addIslandDiagnostic(diagnostics, set.weights);
  const n = set.values.length;
  // Global statistics need at least four features with finite spread.
  if (n < 4) throw new Error('Too few features');
  const {mean, deviation} = getMeanAndDeviation(set.values);
  // Constant values have no spatial variation to test.
  if (deviation === 0) throw new Error('Constant values');
  const z = set.values.map(value => value - mean);
  const links = new Map<number, number>();
  let s0 = 0;
  let cross = 0;
  let differences = 0;
  set.weights.neighbors.forEach((list, row) =>
    list.forEach((other, position) => {
      const weight = set.weights.weights[row][position];
      links.set(row * n + other, weight);
      s0 += weight;
      cross += weight * z[row] * z[other];
      differences += weight * (set.values[row] - set.values[other]) ** 2;
    })
  );
  // No links at all; the neighborhood rule produced no neighbors.
  if (s0 === 0) throw new Error('No neighbors');
  let s1 = 0;
  const rowSums = new Float64Array(n);
  const columnSums = new Float64Array(n);
  for (const [key, weight] of links) {
    const row = Math.floor(key / n);
    const other = key % n;
    const reverseWeight = links.get(other * n + row);
    const reverse = reverseWeight ?? 0;
    s1 += 0.5 * (weight + reverse) ** 2;
    // S1 sums over all ordered pairs: a link without a reverse also adds the pair (other, row), where w is 0.
    if (reverseWeight === undefined) s1 += 0.5 * weight ** 2;
    rowSums[row] += weight;
    columnSums[other] += weight;
  }
  let s2 = 0;
  for (let index = 0; index < n; index++) s2 += (rowSums[index] + columnSums[index]) ** 2;
  const m2 = z.reduce((sum, value) => sum + value * value, 0) / n;
  const m4 = z.reduce((sum, value) => sum + value ** 4, 0) / n;
  const kurtosis = m4 / (m2 * m2);
  const moran = (n / s0) * (cross / z.reduce((sum, value) => sum + value * value, 0));
  const moranExpected = -1 / (n - 1);
  const moranVariance =
    (n * ((n * n - 3 * n + 3) * s1 - n * s2 + 3 * s0 * s0) -
      kurtosis * ((n * n - n) * s1 - 2 * n * s2 + 6 * s0 * s0)) /
      ((n - 1) * (n - 2) * (n - 3) * s0 * s0) -
    moranExpected ** 2;
  const geary =
    ((n - 1) * differences) / (2 * s0 * z.reduce((sum, value) => sum + value * value, 0));
  const gearyVariance = ((2 * s1 + s2) * (n - 1) - 4 * s0 * s0) / (2 * (n + 1) * s0 * s0);
  const table = makeCatalogTable('global', 'Global spatial statistics', {
    statistic: 'string',
    value: 'float64',
    expected: 'float64',
    variance: 'float64',
    z_score: 'float64',
    p_value: 'float64'
  });
  const moranBlock = precomputed?.moran ?? {
    value: moran,
    expected: moranExpected,
    variance: moranVariance,
    zScore: (moran - moranExpected) / Math.sqrt(moranVariance),
    pValue: getNormalTwoSidedPValue((moran - moranExpected) / Math.sqrt(moranVariance))
  };
  const gearyBlock = precomputed?.geary ?? {
    value: geary,
    expected: 1,
    variance: gearyVariance,
    zScore: (geary - 1) / Math.sqrt(gearyVariance),
    pValue: getNormalTwoSidedPValue((geary - 1) / Math.sqrt(gearyVariance))
  };
  for (const [statistic, block] of [
    ['morans-i', moranBlock],
    ['gearys-c', gearyBlock]
  ] as const)
    appendCatalogRow(table, {
      statistic,
      value: block.value,
      expected: block.expected,
      variance: block.variance,
      z_score: Number.isFinite(block.zScore) ? block.zScore : null,
      p_value: Number.isFinite(block.zScore) ? block.pValue : null
    });
  addCatalogDiagnostic(
    diagnostics,
    'analytic-inference',
    'info',
    "Moran's I uses the randomization variance and Geary's C the normality variance; both p-values are normal approximations."
  );
  return {tables: [table], diagnostics, summary: {featureCount: n, linkCount: links.size}};
}

// ---------------------------------------------------------------------------------------------
// Local statistics
// ---------------------------------------------------------------------------------------------

/** Conditional permutation: draws the same number of neighbor values from the other features. */
function permuteNeighborValues(
  row: number,
  count: number,
  values: readonly number[],
  random: () => number
): number[] {
  const picked: number[] = [];
  const used = new Set<number>([row]);
  while (picked.length < count && used.size < values.length) {
    const candidate = Math.floor(random() * values.length);
    if (used.has(candidate)) continue;
    used.add(candidate);
    picked.push(values[candidate]);
  }
  return picked;
}

function makeLocalOutputTable(
  id: string,
  label: string,
  extra: Record<string, 'string' | 'float64' | 'int32' | 'boolean'>
) {
  return makeCatalogTable(
    id,
    label,
    {
      source_index: 'int32',
      source_id: 'string',
      value: 'float64',
      neighbor_count: 'int32',
      ...extra,
      geometry: 'string'
    },
    'geometry'
  );
}

export function runLocalMoran(
  collection: CatalogFeatureCollection,
  parameters: {
    valueProperty: string;
    permutations: number;
    seed: number;
    significance: number;
  } & WeightsSettings,
  precomputed?: PrecomputedLocalStatistics
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const set =
    precomputed?.set ?? buildAnalysisSet(collection, parameters.valueProperty, parameters);
  addDroppedDiagnostic(diagnostics, set.droppedCount);
  addIslandDiagnostic(diagnostics, set.weights);
  const n = set.values.length;
  // Local statistics need at least four features.
  if (n < 4) throw new Error('Too few features');
  const {mean, deviation} = getMeanAndDeviation(set.values);
  // Constant values have no spatial variation to test.
  if (deviation === 0) throw new Error('Constant values');
  const z = set.values.map(value => (value - mean) / deviation);
  const random = makeSeededGenerator(parameters.seed);
  const table = makeLocalOutputTable('local-moran', 'Local Moran', {
    local_i: 'float64',
    spatial_lag: 'float64',
    p_value: 'float64',
    cluster: 'string'
  });
  set.weights.neighbors.forEach((list, row) => {
    const weights = set.weights.weights[row];
    const lag = precomputed?.spatialLag
      ? precomputed.spatialLag[row]
      : list.reduce((sum, other, position) => sum + weights[position] * z[other], 0);
    const localI = precomputed ? precomputed.statistic[row] : z[row] * lag;
    let pValue: number | null = null;
    if (precomputed?.pValue) {
      pValue = Number.isFinite(precomputed.pValue[row]) ? precomputed.pValue[row] : null;
    } else if (parameters.permutations > 0 && list.length) {
      let atLeast = 0;
      for (let draw = 0; draw < parameters.permutations; draw++) {
        const values = permuteNeighborValues(row, list.length, z, random);
        const permuted =
          z[row] * values.reduce((sum, value, position) => sum + weights[position] * value, 0);
        if (Math.abs(permuted) >= Math.abs(localI) - 1e-12) atLeast++;
      }
      pValue = (atLeast + 1) / (parameters.permutations + 1);
    }
    const quadrant = z[row] >= 0 ? (lag >= 0 ? 'HH' : 'HL') : lag >= 0 ? 'LH' : 'LL';
    appendCatalogRow(table, {
      source_index: set.sourceIndex[row],
      source_id: getFeatureIdentity(collection.features[set.sourceIndex[row]]),
      value: set.values[row],
      neighbor_count: list.length,
      local_i: localI,
      spatial_lag: lag,
      p_value: pValue,
      cluster:
        pValue === null
          ? 'not-tested'
          : pValue <= parameters.significance
            ? quadrant
            : 'not-significant',
      geometry: getFeatureGeometryText(collection.features[set.sourceIndex[row]])
    });
  });
  addCatalogDiagnostic(
    diagnostics,
    'pseudo-p-values',
    'info',
    'p-values are conditional-permutation pseudo p-values with no multiple-testing correction; see multiple-testing-adjustment.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {featureCount: n, permutations: parameters.permutations}
  };
}

export function runLocalGeary(
  collection: CatalogFeatureCollection,
  parameters: {
    valueProperty: string;
    permutations: number;
    seed: number;
  } & WeightsSettings,
  precomputed?: PrecomputedLocalStatistics
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const set =
    precomputed?.set ?? buildAnalysisSet(collection, parameters.valueProperty, parameters);
  addDroppedDiagnostic(diagnostics, set.droppedCount);
  addIslandDiagnostic(diagnostics, set.weights);
  const n = set.values.length;
  // Local statistics need at least four features.
  if (n < 4) throw new Error('Too few features');
  const {mean, deviation} = getMeanAndDeviation(set.values);
  // Constant values have no spatial variation to test.
  if (deviation === 0) throw new Error('Constant values');
  const z = set.values.map(value => (value - mean) / deviation);
  const random = makeSeededGenerator(parameters.seed);
  const table = makeLocalOutputTable('local-geary', 'Local Geary', {
    local_c: 'float64',
    p_value: 'float64'
  });
  set.weights.neighbors.forEach((list, row) => {
    const weights = set.weights.weights[row];
    const localC = precomputed
      ? precomputed.statistic[row]
      : list.reduce(
          (sum, other, position) => sum + weights[position] * (z[row] - z[other]) ** 2,
          0
        );
    let pValue: number | null = null;
    if (precomputed?.pValue) {
      pValue = Number.isFinite(precomputed.pValue[row]) ? precomputed.pValue[row] : null;
    } else if (parameters.permutations > 0 && list.length) {
      let atMost = 0;
      for (let draw = 0; draw < parameters.permutations; draw++) {
        const values = permuteNeighborValues(row, list.length, z, random);
        const permuted = values.reduce(
          (sum, value, position) => sum + weights[position] * (z[row] - value) ** 2,
          0
        );
        if (permuted <= localC + 1e-12) atMost++;
      }
      pValue = (atMost + 1) / (parameters.permutations + 1);
    }
    appendCatalogRow(table, {
      source_index: set.sourceIndex[row],
      source_id: getFeatureIdentity(collection.features[set.sourceIndex[row]]),
      value: set.values[row],
      neighbor_count: list.length,
      local_c: localC,
      p_value: pValue,
      geometry: getFeatureGeometryText(collection.features[set.sourceIndex[row]])
    });
  });
  addCatalogDiagnostic(
    diagnostics,
    'one-sided-pseudo-p',
    'info',
    'p-values are one-sided conditional-permutation pseudo p-values for small local C (similarity).'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {featureCount: n, permutations: parameters.permutations}
  };
}

export function runHotSpotGiStar(
  collection: CatalogFeatureCollection,
  parameters: {valueProperty: string} & Omit<WeightsSettings, 'rowStandardize'>,
  precomputed?: PrecomputedLocalStatistics
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const set =
    precomputed?.set ??
    buildAnalysisSet(collection, parameters.valueProperty, {
      ...parameters,
      rowStandardize: false
    });
  addDroppedDiagnostic(diagnostics, set.droppedCount);
  const n = set.values.length;
  // Hot-spot analysis needs at least four features.
  if (n < 4) throw new Error('Too few features');
  const {mean, deviation} = getMeanAndDeviation(set.values);
  // Constant values have no spatial variation to test.
  if (deviation === 0) throw new Error('Constant values');
  const table = makeLocalOutputTable('hotspots', 'Getis-Ord Gi*', {
    z_score: 'float64',
    p_value: 'float64',
    classification: 'string'
  });
  set.weights.neighbors.forEach((list, row) => {
    const weightSum = list.length + 1;
    const sum = list.reduce((total, other) => total + set.values[other], set.values[row]);
    const cpuZ =
      (sum - mean * weightSum) /
      (deviation * Math.sqrt((n * weightSum - weightSum * weightSum) / (n - 1)));
    // Islands keep the CPU formula: the contributor reports no z-score for a row without neighbors.
    const z =
      precomputed && list.length && Number.isFinite(precomputed.statistic[row])
        ? precomputed.statistic[row]
        : cpuZ;
    const magnitude = Math.abs(z);
    const level =
      magnitude >= 2.576 ? '99' : magnitude >= 1.96 ? '95' : magnitude >= 1.645 ? '90' : '';
    appendCatalogRow(table, {
      source_index: set.sourceIndex[row],
      source_id: getFeatureIdentity(collection.features[set.sourceIndex[row]]),
      value: set.values[row],
      neighbor_count: list.length,
      z_score: Number.isFinite(z) ? z : null,
      p_value: Number.isFinite(z) ? getNormalTwoSidedPValue(z) : null,
      classification: level ? `${z > 0 ? 'hot' : 'cold'}-${level}` : 'not-significant',
      geometry: getFeatureGeometryText(collection.features[set.sourceIndex[row]])
    });
  });
  addCatalogDiagnostic(
    diagnostics,
    'binary-weights-with-self',
    'info',
    'Gi* uses binary weights including the feature itself; p-values are normal approximations without multiple-testing correction.'
  );
  addIslandDiagnostic(diagnostics, set.weights);
  return {tables: [table], diagnostics, summary: {featureCount: n}};
}

// ---------------------------------------------------------------------------------------------
// Rates, multiple testing and Knox
// ---------------------------------------------------------------------------------------------

export function runEmpiricalBayesRates(
  collection: CatalogFeatureCollection,
  parameters: {numeratorProperty: string; denominatorProperty: string},
  /** Moments and per-row rates computed elsewhere (the GPU engine); the CPU formulas are skipped. */
  precomputed?: {
    globalRate: number;
    priorVariance: number;
    rawRates: ArrayLike<number>;
    smoothedRates: ArrayLike<number>;
  }
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const rows: {index: number; events: number; population: number}[] = [];
  let droppedCount = 0;
  collection.features.forEach((feature, index) => {
    const events = readNumericProperty(feature, parameters.numeratorProperty);
    const population = readNumericProperty(feature, parameters.denominatorProperty);
    if (events === null || population === null || population <= 0 || events < 0) {
      droppedCount++;
      return;
    }
    rows.push({index, events, population});
  });
  addDroppedDiagnostic(diagnostics, droppedCount);
  // Empirical Bayes shrinkage needs at least two usable areas.
  if (rows.length < 2) throw new Error('Too few features');
  const totalEvents = rows.reduce((sum, row) => sum + row.events, 0);
  const totalPopulation = rows.reduce((sum, row) => sum + row.population, 0);
  const globalRate = precomputed?.globalRate ?? totalEvents / totalPopulation;
  const meanPopulation = totalPopulation / rows.length;
  const weightedVariance =
    rows.reduce(
      (sum, row) => sum + row.population * (row.events / row.population - globalRate) ** 2,
      0
    ) / totalPopulation;
  const priorVariance =
    precomputed?.priorVariance ?? Math.max(0, weightedVariance - globalRate / meanPopulation);
  const table = makeCatalogTable(
    'rates',
    'Empirical Bayes rates',
    {
      source_index: 'int32',
      source_id: 'string',
      raw_rate: 'float64',
      shrinkage: 'float64',
      eb_rate: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  for (const [position, row] of rows.entries()) {
    const rawRate = precomputed ? precomputed.rawRates[position] : row.events / row.population;
    const shrinkage =
      priorVariance + globalRate / row.population > 0
        ? priorVariance / (priorVariance + globalRate / row.population)
        : 0;
    appendCatalogRow(table, {
      source_index: row.index,
      source_id: getFeatureIdentity(collection.features[row.index]),
      raw_rate: rawRate,
      shrinkage,
      eb_rate: precomputed
        ? precomputed.smoothedRates[position]
        : shrinkage * rawRate + (1 - shrinkage) * globalRate,
      geometry: getFeatureGeometryText(collection.features[row.index])
    });
  }
  addCatalogDiagnostic(
    diagnostics,
    'global-prior',
    'info',
    'Marshall global empirical Bayes: every area shrinks toward the overall rate; no spatial smoothing.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {areaCount: rows.length, globalRate, priorVariance}
  };
}

export type AdjustmentMethod = 'bonferroni' | 'holm' | 'benjamini-hochberg' | 'sidak';

export function runMultipleTestingAdjustment(
  collection: CatalogFeatureCollection,
  parameters: {pValueProperty: string; method: AdjustmentMethod; alpha: number},
  /** Adjusted p-values per valid row, computed elsewhere (the GPU engine); skips the CPU formulas. */
  precomputedAdjusted?: ArrayLike<number>
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const rows: {index: number; p: number}[] = [];
  let droppedCount = 0;
  collection.features.forEach((feature, index) => {
    const p = readNumericProperty(feature, parameters.pValueProperty);
    if (p === null || p < 0 || p > 1) {
      droppedCount++;
      return;
    }
    rows.push({index, p});
  });
  addDroppedDiagnostic(diagnostics, droppedCount);
  const m = rows.length;
  // No valid p-values to adjust.
  if (!m) throw new Error('No usable samples');
  const order = rows.map((_, position) => position).sort((a, b) => rows[a].p - rows[b].p);
  const adjusted = new Float64Array(m);
  if (precomputedAdjusted) adjusted.set(Array.from(precomputedAdjusted));
  else if (parameters.method === 'bonferroni')
    rows.forEach((row, position) => (adjusted[position] = Math.min(1, row.p * m)));
  else if (parameters.method === 'sidak')
    rows.forEach((row, position) => (adjusted[position] = 1 - (1 - row.p) ** m));
  else if (parameters.method === 'holm') {
    let running = 0;
    order.forEach((position, rank) => {
      running = Math.max(running, Math.min(1, (m - rank) * rows[position].p));
      adjusted[position] = running;
    });
  } else {
    let running = 1;
    for (let rank = m - 1; rank >= 0; rank--) {
      running = Math.min(running, (rows[order[rank]].p * m) / (rank + 1));
      adjusted[order[rank]] = Math.min(1, running);
    }
  }
  const table = makeCatalogTable(
    'adjusted',
    'Adjusted p-values',
    {
      source_index: 'int32',
      source_id: 'string',
      p_value: 'float64',
      adjusted_p_value: 'float64',
      rejected: 'boolean',
      geometry: 'string'
    },
    'geometry'
  );
  rows.forEach((row, position) =>
    appendCatalogRow(table, {
      source_index: row.index,
      source_id: getFeatureIdentity(collection.features[row.index]),
      p_value: row.p,
      adjusted_p_value: adjusted[position],
      rejected: adjusted[position] <= parameters.alpha,
      geometry: getFeatureGeometryText(collection.features[row.index])
    })
  );
  return {
    tables: [table],
    diagnostics,
    summary: {
      testCount: m,
      rejectedCount: adjusted.reduce(
        (count, value) => count + (value <= parameters.alpha ? 1 : 0),
        0
      )
    }
  };
}

export function runKnoxTest(
  points: CatalogFeatureCollection,
  parameters: {
    timeProperty: string;
    spatialThreshold: number;
    temporalThreshold: number;
    permutations: number;
    seed: number;
    coordinateSystem: CatalogCoordinateSystem;
  },
  /** Monte Carlo result computed elsewhere (the GPU engine); the CPU permutation loop is skipped. */
  precomputed?: {pValue: number}
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const frame = makeLocalFrame(parameters.coordinateSystem, collectAllPositions(points), {
    shift: 'minimum'
  });
  const x: number[] = [];
  const y: number[] = [];
  const times: number[] = [];
  let droppedCount = 0;
  for (const feature of points.features) {
    const position = getPointPositions(feature.geometry)[0];
    const time = readNumericProperty(feature, parameters.timeProperty);
    if (!position || time === null) {
      droppedCount++;
      continue;
    }
    const [px, py] = frame.forward(position);
    x.push(px);
    y.push(py);
    times.push(time);
  }
  addDroppedDiagnostic(diagnostics, droppedCount);
  const n = times.length;
  // The Knox test needs at least three events.
  if (n < 3) throw new Error('Too few features');
  const closePairs: [number, number][] = [];
  let temporalClose = 0;
  const spatialClose: [number, number][] = [];
  for (let first = 0; first < n; first++)
    for (let second = first + 1; second < n; second++) {
      if (Math.abs(times[first] - times[second]) <= parameters.temporalThreshold) temporalClose++;
      if (Math.hypot(x[first] - x[second], y[first] - y[second]) <= parameters.spatialThreshold)
        spatialClose.push([first, second]);
    }
  let observed = 0;
  for (const [first, second] of spatialClose)
    if (Math.abs(times[first] - times[second]) <= parameters.temporalThreshold) {
      observed++;
      closePairs.push([first, second]);
    }
  const pairCount = (n * (n - 1)) / 2;
  const expected = (spatialClose.length * temporalClose) / pairCount;
  const random = makeSeededGenerator(parameters.seed);
  let atLeast = 0;
  const shuffled = times.slice();
  for (let draw = 0; draw < (precomputed ? 0 : parameters.permutations); draw++) {
    for (let index = n - 1; index > 0; index--) {
      const swap = Math.floor(random() * (index + 1));
      [shuffled[index], shuffled[swap]] = [shuffled[swap], shuffled[index]];
    }
    let count = 0;
    for (const [first, second] of spatialClose)
      if (Math.abs(shuffled[first] - shuffled[second]) <= parameters.temporalThreshold) count++;
    if (count >= observed) atLeast++;
  }
  const table = makeCatalogTable('knox', 'Knox test', {
    event_count: 'int32',
    spatially_close_pairs: 'int32',
    temporally_close_pairs: 'int32',
    joint_close_pairs: 'int32',
    expected_joint_pairs: 'float64',
    ratio: 'float64',
    p_value: 'float64'
  });
  appendCatalogRow(table, {
    event_count: n,
    spatially_close_pairs: spatialClose.length,
    temporally_close_pairs: temporalClose,
    joint_close_pairs: observed,
    expected_joint_pairs: expected,
    ratio: expected > 0 ? observed / expected : null,
    p_value: precomputed ? precomputed.pValue : (atLeast + 1) / (parameters.permutations + 1)
  });
  return {tables: [table], diagnostics, summary: {eventCount: n, jointClosePairs: observed}};
}
