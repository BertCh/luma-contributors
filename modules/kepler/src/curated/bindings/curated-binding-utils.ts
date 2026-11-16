// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  CatalogParameterSpec,
  CatalogParameterValue
} from '../../analysis/catalog-binding-types';
import {
  addCatalogDiagnostic,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeature,
  type CatalogFeatureCollection
} from '../../analysis/catalog-geojson';
import {makeLocalFrame} from '../../analysis/catalog-local-frame';
import {
  collectAllPositions,
  getFeatureAnchor,
  makeSeededGenerator,
  readNumericProperty
} from '../../analysis/catalog-platform3-b-common';

// Shared pieces of the curated bindings. Everything here is CPU reference code; the curated
// bindings declare `engine: 'cpu-reference'` and a GPU path is future work.

/** Rows of an areal input the curated area bindings accept; the CPU contiguity search is quadratic in shared vertices. */
export const MAXIMUM_CURATED_AREA_FEATURES = 5000;

/** Vertices of an areal input: 5000 areas of about 40 vertices each. */
export const MAXIMUM_CURATED_AREA_VERTICES = 200000;

/**
 * Largest product of tested rows and conditional or global permutations the CPU engine runs. Each
 * unit is a handful of multiplications and a shuffle step, so 2e7 units take a few seconds.
 */
export const MAXIMUM_PERMUTATION_WORK = 2e7;

/** Fewest permutations the curated bindings accept: below 99 a pseudo p-value cannot fall under 0.01. */
export const MINIMUM_PERMUTATIONS = 99;

/** Most permutations accepted; a pseudo p-value resolution of 1e-4 is finer than the methods justify. */
export const MAXIMUM_PERMUTATIONS = 9999;

/** Four binary64 rounding steps cover log1p, division, expm1 and reciprocation at integer boundaries. */
const PERMUTATION_BOUNDARY_ROUNDING_FACTOR = 4 * Number.EPSILON;

/** Unit-less parameter values the bindings read. */
export type CuratedValues = Readonly<Record<string, CatalogParameterValue>>;

/** Name a thrown error after the binding so a message in a log or a panel says where it came from. */
export function makeCuratedError(bindingId: string, message: string): RangeError {
  return new RangeError(`${bindingId}: ${message}`);
}

/** A column-name parameter that must be filled. */
export function readColumnParameter(
  bindingId: string,
  values: CuratedValues,
  name: string
): string {
  const column = values[name];
  if (typeof column !== 'string' || !column.trim())
    throw makeCuratedError(bindingId, `parameter ${name} must name a column`);
  return column.trim();
}

/** A string parameter holding one column name, with the 64 character default limit widened for lists. */
export function makeColumnParameter(
  name: string,
  label: string,
  maximumLength = 64
): CatalogParameterSpec {
  return {name, label, kind: 'string', maximumLength, defaultValue: ''};
}

/** The permutation count parameter shared by the curated bindings. */
export function makePermutationParameter(defaultValue = 999): CatalogParameterSpec {
  return {
    name: 'permutations',
    label: 'Permutations',
    kind: 'integer',
    minimum: MINIMUM_PERMUTATIONS,
    maximum: MAXIMUM_PERMUTATIONS,
    defaultValue
  };
}

/** One areal feature with the numbers a rate analysis reads. */
export type RateRow = {
  /** Index of the feature in the input collection. */
  sourceIndex: number;
  feature: CatalogFeature;
  cases: number;
  population: number;
};

export type RateRows = {
  rows: RateRow[];
  droppedCount: number;
  /**
   * One entry per input feature: `analysed`, or the diagnostic code of the reason the feature was
   * dropped, so an output table can list every input row with the reason it has no result.
   */
  statuses: string[];
};

/**
 * Reads cases and population and keeps the usable rows. A row is dropped, and counted under its own
 * diagnostic code, when it has no geometry, when cases or population is missing, not a number or
 * not finite, when the population is zero or negative, or when cases are negative. Nothing is
 * dropped silently: every code appears in `diagnostics` with its count.
 */
export function readRateRows(
  collection: CatalogFeatureCollection,
  casesProperty: string,
  populationProperty: string,
  diagnostics: CatalogDiagnostic[]
): RateRows {
  const rows: RateRow[] = [];
  const statuses: string[] = [];
  let droppedCount = 0;
  const drop = (sourceIndex: number, code: string, message: string) => {
    droppedCount++;
    statuses[sourceIndex] = code;
    addCatalogDiagnostic(diagnostics, code, 'warning', message);
  };
  collection.features.forEach((feature, sourceIndex) => {
    statuses[sourceIndex] = 'analysed';
    if (!feature.geometry) {
      drop(
        sourceIndex,
        'missing-geometry',
        'Features without geometry are skipped: they have no neighbours.'
      );
      return;
    }
    const cases = readNumericProperty(feature, casesProperty);
    const population = readNumericProperty(feature, populationProperty);
    if (cases === null || population === null) {
      drop(
        sourceIndex,
        'missing-value',
        'Features whose cases or population is missing, not a number or not finite are skipped.'
      );
      return;
    }
    if (!(population > 0)) {
      drop(
        sourceIndex,
        'non-positive-population',
        'Features with a zero or negative population are skipped: their rate is undefined.'
      );
      return;
    }
    if (cases < 0) {
      drop(sourceIndex, 'negative-cases', 'Features with negative cases are skipped.');
      return;
    }
    rows.push({sourceIndex, feature, cases, population});
  });
  return {rows, droppedCount, statuses};
}

/** A collection holding the given features with their own properties replaced, geometry and id kept. */
export function makeDerivedCollection(
  features: readonly CatalogFeature[],
  getProperties: (index: number) => Record<string, unknown>
): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: features.map((feature, index) => ({
      id: feature.id,
      geometry: feature.geometry,
      properties: getProperties(index)
    }))
  };
}

/**
 * Moran's I of `values` over weights given as neighbour lists:
 * `I = (n / S0) sum_ij w_ij z_i z_j / sum_i z_i^2` with `z` the deviations from the mean
 * (Moran 1950; Cliff and Ord 1981). The same formula as the global statistic of
 * `runGlobalSpatialStatistics`, written as a function of values so a permutation loop can call it.
 */
export function computeMoransI(
  values: ArrayLike<number>,
  neighbors: readonly (readonly number[])[],
  weights: readonly (readonly number[])[]
): number {
  const count = values.length;
  let mean = 0;
  for (let row = 0; row < count; row++) mean += values[row];
  mean /= count;
  const deviations = new Float64Array(count);
  let sumOfSquares = 0;
  for (let row = 0; row < count; row++) {
    deviations[row] = values[row] - mean;
    sumOfSquares += deviations[row] * deviations[row];
  }
  return computeMoransIOfDeviations(deviations, sumOfSquares, neighbors, weights);
}

function computeMoransIOfDeviations(
  deviations: Float64Array,
  sumOfSquares: number,
  neighbors: readonly (readonly number[])[],
  weights: readonly (readonly number[])[]
): number {
  let weightSum = 0;
  let cross = 0;
  for (let row = 0; row < deviations.length; row++) {
    const list = neighbors[row];
    const rowWeights = weights[row];
    let lag = 0;
    for (let position = 0; position < list.length; position++) {
      lag += rowWeights[position] * deviations[list[position]];
      weightSum += rowWeights[position];
    }
    cross += deviations[row] * lag;
  }
  return (deviations.length / weightSum) * (cross / sumOfSquares);
}

export type MoranPermutationResult = {
  /** Moran's I of the values as given. */
  statistic: number;
  /** Mean of the permuted statistics. */
  simulatedMean: number;
  /** Permutations whose statistic is at least as far from `simulatedMean` as the observed one. */
  exceedanceCount: number;
  permutations: number;
  /** `(exceedanceCount + 1) / (permutations + 1)`: the folded two-sided pseudo p-value of esda. */
  pseudoPValue: number;
};

/**
 * Global permutation test of Moran's I. The values are shuffled over the rows with a seeded
 * Fisher-Yates shuffle, the statistic is recomputed for each shuffle, and the two-sided pseudo
 * p-value counts shuffles at least as far from the mean of the shuffled statistics as the observed
 * statistic (the folded test of esda `Moran`, `two_tailed=True`): `p = (m + 1) / (R + 1)`, so the
 * smallest value is `1 / (R + 1)`. The generator is `makeSeededGenerator`, so a seed fixes the result.
 */
export function runMoranPermutationTest(
  values: ArrayLike<number>,
  neighbors: readonly (readonly number[])[],
  weights: readonly (readonly number[])[],
  permutations: number,
  seed: number
): MoranPermutationResult {
  const count = values.length;
  let mean = 0;
  for (let row = 0; row < count; row++) mean += values[row];
  mean /= count;
  const deviations = new Float64Array(count);
  let sumOfSquares = 0;
  for (let row = 0; row < count; row++) {
    deviations[row] = values[row] - mean;
    sumOfSquares += deviations[row] * deviations[row];
  }
  const statistic = computeMoransIOfDeviations(deviations, sumOfSquares, neighbors, weights);
  const random = makeSeededGenerator(seed);
  const shuffled = Float64Array.from(deviations);
  const simulated = new Float64Array(permutations);
  for (let draw = 0; draw < permutations; draw++) {
    for (let index = count - 1; index > 0; index--) {
      const swap = Math.floor(random() * (index + 1));
      const held = shuffled[index];
      shuffled[index] = shuffled[swap];
      shuffled[swap] = held;
    }
    simulated[draw] = computeMoransIOfDeviations(shuffled, sumOfSquares, neighbors, weights);
  }
  let simulatedMean = 0;
  for (const value of simulated) simulatedMean += value;
  simulatedMean /= permutations;
  const observedDistance = Math.abs(statistic - simulatedMean);
  let exceedanceCount = 0;
  // A relative slack keeps a shuffle that reproduces the observed arrangement counted when its
  // distance differs from the observed one only by summation order.
  for (const value of simulated)
    if (Math.abs(value - simulatedMean) >= observedDistance - 1e-12 * Math.abs(observedDistance))
      exceedanceCount++;
  return {
    statistic,
    simulatedMean,
    exceedanceCount,
    permutations,
    pseudoPValue: (exceedanceCount + 1) / (permutations + 1)
  };
}

/** True when every value of the list is the same number. */
export function isConstantList(values: ArrayLike<number>): boolean {
  for (let index = 1; index < values.length; index++) if (values[index] !== values[0]) return false;
  return true;
}

/** Contiguity criteria the area bindings accept. */
export const CONTIGUITY_CRITERION_PARAMETER: CatalogParameterSpec = {
  name: 'criterion',
  label: 'Contiguity',
  kind: 'enum',
  options: ['queen', 'rook'],
  defaultValue: 'queen'
};

// ---------------------------------------------------------------------------------------------
// Permutation inference that does not depend on the order of the input rows
// ---------------------------------------------------------------------------------------------

/**
 * Anchor of every feature in the local frame the weights use (the point, or the area centroid):
 * the same call as `buildAnalysisSet`, so the values agree with the neighbour search. The frame
 * origin is the minimum over all positions, so it does not depend on row order.
 */
export function getFeatureAnchors(
  collection: CatalogFeatureCollection,
  coordinateSystem: CatalogCoordinateSystem
): ([number, number] | null)[] {
  const frame = makeLocalFrame(coordinateSystem, collectAllPositions(collection), {
    shift: 'minimum'
  });
  return collection.features.map(feature => getFeatureAnchor(feature, frame));
}

export type CanonicalOrder = {
  /** `order[rank]` is the row index holding that rank. */
  order: number[];
  /** `rank[row]` is the position of the row in the canonical order. */
  rank: number[];
  /** Rows whose anchor is exactly equal to that of another row; their relative order is the input order. */
  tiedCount: number;
};

/**
 * Canonical order of rows: by anchor x, then anchor y, ascending; rows with exactly equal anchors
 * keep their input order, and `tiedCount` says how many there are. Without coincident anchors a
 * permutation stream drawn in this order does not change when the rows are shuffled. Coincident
 * anchors can change inference throughout the dataset: their values also enter other rows' pools.
 */
export function getCanonicalOrder(anchors: readonly (readonly [number, number])[]): CanonicalOrder {
  const order = anchors
    .map((_, index) => index)
    .sort((a, b) => anchors[a][0] - anchors[b][0] || anchors[a][1] - anchors[b][1] || a - b);
  const rank = new Array<number>(anchors.length);
  order.forEach((row, position) => {
    rank[row] = position;
  });
  let tiedCount = 0;
  for (let position = 0; position < order.length; position++) {
    const here = anchors[order[position]];
    const before = position > 0 ? anchors[order[position - 1]] : undefined;
    const after = position < order.length - 1 ? anchors[order[position + 1]] : undefined;
    if (
      (before && before[0] === here[0] && before[1] === here[1]) ||
      (after && after[0] === here[0] && after[1] === here[1])
    )
      tiedCount++;
  }
  return {order, rank, tiedCount};
}

/** Generator seed of one area: a 32 bit mix of the user seed and the area's canonical rank. */
function getAreaSeed(seed: number, rank: number): number {
  let hash = Math.imul((seed ^ 0x9e3779b9) >>> 0, 0x85ebca6b) ^ Math.imul(rank + 1, 0xc2b2ae35);
  hash = Math.imul(hash ^ (hash >>> 16), 0x7feb352d);
  hash = Math.imul(hash ^ (hash >>> 15), 0x846ca68b);
  return (hash ^ (hash >>> 16)) >>> 0;
}

export type LocalPermutationResult = {
  /** Pseudo p-value per row, null for a row with no neighbour. */
  pValues: (number | null)[];
  tiedCount: number;
};

/**
 * Conditional permutation test of the local Moran statistic `z_i * sum_j w_ij z_j` with weights
 * standardised by row (the catalog scheme of `runLocalMoran`): each area keeps its own value, draws
 * as many values as it has neighbours from the other areas without replacement, and the pseudo
 * p-value is `(m + 1) / (R + 1)` with `m` the draws whose absolute statistic is at least the
 * observed absolute statistic (within 1e-12). Every area draws from its own generator, seeded by
 * the seed and the area's rank in the canonical order of its anchor (see `getCanonicalOrder`), and
 * the other areas are indexed by that rank, so shuffling rows with distinct anchors changes no
 * p-value. Coincident-anchor ordering can affect inference throughout the dataset.
 */
export function runConditionalLocalPermutationTest(
  deviations: readonly number[],
  neighbors: readonly (readonly number[])[],
  anchors: readonly (readonly [number, number])[],
  permutations: number,
  seed: number
): LocalPermutationResult {
  const count = deviations.length;
  const {order, rank, tiedCount} = getCanonicalOrder(anchors);
  const byRank = order.map(row => deviations[row]);
  const stamps = new Int32Array(count);
  let stamp = 0;
  const pValues = new Array<number | null>(count).fill(null);
  for (let row = 0; row < count; row++) {
    const neighborCount = neighbors[row].length;
    if (!neighborCount) continue;
    let lag = 0;
    for (const other of neighbors[row]) lag += deviations[other];
    const observed = Math.abs(deviations[row] * (lag / neighborCount));
    const ownRank = rank[row];
    const random = makeSeededGenerator(getAreaSeed(seed, ownRank));
    let atLeast = 0;
    for (let draw = 0; draw < permutations; draw++) {
      stamp++;
      stamps[ownRank] = stamp;
      let sum = 0;
      let picked = 0;
      while (picked < neighborCount) {
        const candidate = Math.floor(random() * count);
        if (stamps[candidate] === stamp) continue;
        stamps[candidate] = stamp;
        sum += byRank[candidate];
        picked++;
      }
      if (Math.abs(byRank[ownRank] * (sum / neighborCount)) >= observed - 1e-12) atLeast++;
    }
    pValues[row] = (atLeast + 1) / (permutations + 1);
  }
  return {pValues, tiedCount};
}

/**
 * `runMoranPermutationTest` with the rows put in the canonical order of their anchors first, so
 * the shuffle stream, and with it the pseudo p-value, does not depend on the order of the input rows.
 */
export function runCanonicalMoranPermutationTest(
  values: ArrayLike<number>,
  neighbors: readonly (readonly number[])[],
  weights: readonly (readonly number[])[],
  anchors: readonly (readonly [number, number])[],
  permutations: number,
  seed: number
): MoranPermutationResult & {tiedCount: number} {
  const {order, rank, tiedCount} = getCanonicalOrder(anchors);
  const result = runMoranPermutationTest(
    order.map(row => values[row]),
    order.map(row => neighbors[row].map(other => rank[other])),
    order.map(row => weights[row]),
    permutations,
    seed
  );
  return {...result, tiedCount};
}

/**
 * Whether any area could be flagged by a correction, given the permutation count. With `R`
 * permutations the smallest pseudo p-value is `pMin = pValueMultiplier / (R + 1)`
 * (multiplier 1 for absolute-statistic inference, 2 for doubled-tail inference). Bonferroni and Holm flag the
 * smallest p-value when `m pMin <= alpha`, Sidak when `1 - (1 - pMin)^m <= alpha`, and
 * Benjamini-Hochberg when `pMin <= alpha`, but it then needs at least `m pMin / alpha` areas
 * tied at the minimum (`requiredTiedMinimumCount`). `permutationsNeeded` is the smallest count
 * that makes the first flag possible, which can exceed the 9999 the bindings accept.
 */
export function getSignificanceAttainability(
  method: string,
  alpha: number,
  testedCount: number,
  permutations: number,
  pValueMultiplier = 1
): {attainable: boolean; permutationsNeeded: number; requiredTiedMinimumCount: number | null} {
  const pMin = pValueMultiplier / (permutations + 1);
  const pNeeded =
    method === 'bonferroni' || method === 'holm'
      ? alpha / testedCount
      : method === 'sidak'
        ? -Math.expm1(Math.log1p(-alpha) / testedCount)
        : alpha;
  const attainable = pMin <= pNeeded * (1 + 1e-12);
  const reciprocal = pValueMultiplier / pNeeded;
  const nearestInteger = Math.round(reciprocal);
  // Snap only within the floating-point evaluation error; a fixed absolute slack loses meaning
  // when a small significance level requires trillions of permutations.
  const minimumDrawCount =
    Math.abs(reciprocal - nearestInteger) <=
    PERMUTATION_BOUNDARY_ROUNDING_FACTOR * Math.max(1, reciprocal)
      ? nearestInteger
      : Math.ceil(reciprocal);
  return {
    attainable,
    permutationsNeeded: Math.max(1, minimumDrawCount - 1),
    requiredTiedMinimumCount:
      method === 'benjamini-hochberg'
        ? Math.max(1, Math.ceil((testedCount * pMin) / alpha - 1e-9))
        : null
  };
}
