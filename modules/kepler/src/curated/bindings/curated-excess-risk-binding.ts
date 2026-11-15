// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  CatalogOperationBinding,
  CatalogParameterSpec,
  CatalogRunOutput
} from '../../analysis/catalog-binding-types';
import {COORDINATE_SYSTEM_PARAMETER} from '../../analysis/catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getFeatureIdentity,
  makeCatalogTable,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic
} from '../../analysis/catalog-geojson';
import {
  PLATFORM3_CPU_CLEANUP,
  getFeatureGeometryText,
  makeInputSpec,
  makeNumberParameter,
  makeOutputSpec,
  readNumericProperty,
  type Inputs,
  type Values
} from '../../analysis/catalog-platform3-b-common';
import {
  buildAnalysisSet,
  runEmpiricalBayesRates
} from '../../analysis/catalog-platform3-b-statistics';
import {
  CONTIGUITY_CRITERION_PARAMETER,
  MAXIMUM_CURATED_AREA_FEATURES,
  MAXIMUM_CURATED_AREA_VERTICES,
  makeColumnParameter,
  makeCuratedError,
  makeDerivedCollection,
  readColumnParameter,
  readRateRows
} from './curated-binding-utils';

const BINDING_ID = 'curated-excess-risk';

/** Empirical Bayes shrinkage needs at least two usable areas; spatial pooling needs neighbours, so three is the floor. */
const MINIMUM_AREAS = 3;

/**
 * Areas with fewer expected cases than this have an unstable raw excess risk. One more or one fewer
 * case moves the ratio by 1 / expected, so at an expected count of 5 a single case is a 20 percent
 * change; 5 is also the usual minimum expected count for the normal approximations of Poisson
 * ratios (the rule of Cochran for chi-square tests).
 */
export const SMALL_EXPECTED_CASE_THRESHOLD = 5;

const PARAMETERS: readonly CatalogParameterSpec[] = [
  makeColumnParameter('casesProperty', 'Cases property'),
  makeColumnParameter('populationProperty', 'Population property'),
  CONTIGUITY_CRITERION_PARAMETER,
  makeNumberParameter(
    'rateScale',
    'Rate multiplier (for example 1000 for per thousand)',
    1000,
    1e-9,
    1e9
  ),
  {
    name: 'islandPolicy',
    label: 'Areas with no neighbour',
    kind: 'enum',
    options: ['undefined', 'own-rate'],
    defaultValue: 'undefined'
  },
  COORDINATE_SYSTEM_PARAMETER
];

function runExcessRisk(inputs: Inputs, values: Values): CatalogRunOutput {
  const areas = inputs['areas'];
  const diagnostics: CatalogDiagnostic[] = [];
  const casesProperty = readColumnParameter(BINDING_ID, values, 'casesProperty');
  const populationProperty = readColumnParameter(BINDING_ID, values, 'populationProperty');
  const rateScale = values['rateScale'] as number;
  const keepOwnRate = values['islandPolicy'] === 'own-rate';

  const {rows, droppedCount, statuses} = readRateRows(
    areas,
    casesProperty,
    populationProperty,
    diagnostics
  );
  if (rows.length < MINIMUM_AREAS)
    throw makeCuratedError(
      BINDING_ID,
      `${rows.length} usable areas; at least ${MINIMUM_AREAS} with geometry, finite cases and a positive population are needed`
    );

  // Expected cases under the overall rate (indirect standardisation with one stratum):
  // E_i = P_i * sum(O) / sum(P); excess risk is O_i / E_i, so the sum of expected equals the sum of observed.
  let totalCases = 0;
  let totalPopulation = 0;
  for (const row of rows) {
    totalCases += row.cases;
    totalPopulation += row.population;
  }
  const overallRate = totalCases / totalPopulation;
  if (!(totalCases > 0))
    addCatalogDiagnostic(
      diagnostics,
      'no-cases',
      'warning',
      'No usable area has a case, so the expected counts are zero and the excess risk is undefined everywhere.'
    );

  // Global empirical Bayes smoothed rate, by the catalog's own function.
  const rateCollection = makeDerivedCollection(
    rows.map(row => row.feature),
    index => ({cases: rows[index].cases, population: rows[index].population})
  );
  const smoothing = runEmpiricalBayesRates(rateCollection, {
    numeratorProperty: 'cases',
    denominatorProperty: 'population'
  });
  const rawRates = smoothing.tables[0].columns['raw_rate'] as number[];
  const smoothedRates = smoothing.tables[0].columns['eb_rate'] as number[];

  // Spatially smoothed rate: (E_i + sum of neighbours' E) / (P_i + sum of neighbours' P), the pooled
  // ratio of addSpatialRateRecipe, never an average of rates. The structure is binary.
  const set = buildAnalysisSet(rateCollection, '', {
    neighborhood: values['criterion'] as 'queen' | 'rook',
    neighborCount: 6,
    distanceBand: 1,
    rowStandardize: false,
    coordinateSystem: values['coordinateSystem'] as CatalogCoordinateSystem
  });
  const spatialRates = new Array<number | null>(rows.length).fill(null);
  const neighborCounts = new Array<number | null>(rows.length).fill(null);
  let islandCount = 0;
  set.sourceIndex.forEach((rowIndex, position) => {
    const neighbors = set.weights.neighbors[position];
    neighborCounts[rowIndex] = neighbors.length;
    if (!neighbors.length) {
      islandCount++;
      if (keepOwnRate) spatialRates[rowIndex] = rateScale * rawRates[rowIndex];
      return;
    }
    let pooledCases = rows[rowIndex].cases;
    let pooledPopulation = rows[rowIndex].population;
    for (const other of neighbors) {
      pooledCases += rows[set.sourceIndex[other]].cases;
      pooledPopulation += rows[set.sourceIndex[other]].population;
    }
    spatialRates[rowIndex] = (rateScale * pooledCases) / pooledPopulation;
  });
  if (islandCount)
    addCatalogDiagnostic(
      diagnostics,
      'island-features',
      'warning',
      keepOwnRate
        ? 'Areas with no contiguous neighbour keep their own rate as the spatial rate.'
        : 'Areas with no contiguous neighbour have no spatial rate (it is undefined for them).',
      islandCount
    );

  const rowByInputIndex = new Map(rows.map((row, index) => [row.sourceIndex, index]));
  const table = makeCatalogTable(
    'excess-risk',
    'Excess risk and smoothed rates',
    {
      source_index: 'int32',
      source_id: 'string',
      status: 'string',
      cases: 'float64',
      population: 'float64',
      expected_cases: 'float64',
      excess_risk: 'float64',
      smoothed_excess_risk: 'float64',
      raw_rate: 'float64',
      smoothed_rate: 'float64',
      spatial_rate: 'float64',
      neighbor_count: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
  let aboveExpectationCount = 0;
  let smallDenominatorAreaCount = 0;
  areas.features.forEach((feature, sourceIndex) => {
    const rowIndex = rowByInputIndex.get(sourceIndex);
    if (rowIndex === undefined) {
      appendCatalogRow(table, {
        source_index: sourceIndex,
        source_id: getFeatureIdentity(feature),
        status: statuses[sourceIndex],
        cases: readNumericProperty(feature, casesProperty),
        population: readNumericProperty(feature, populationProperty),
        geometry: getFeatureGeometryText(feature)
      });
      return;
    }
    const expected = rows[rowIndex].population * overallRate;
    const excessRisk = expected > 0 ? rows[rowIndex].cases / expected : null;
    if (excessRisk !== null && excessRisk > 1) aboveExpectationCount++;
    if (expected < SMALL_EXPECTED_CASE_THRESHOLD) smallDenominatorAreaCount++;
    appendCatalogRow(table, {
      source_index: sourceIndex,
      source_id: getFeatureIdentity(feature),
      status: statuses[sourceIndex],
      cases: rows[rowIndex].cases,
      population: rows[rowIndex].population,
      expected_cases: expected,
      excess_risk: excessRisk,
      // The empirical Bayes smoothed rate over the overall rate: shrunk toward 1 where the
      // population is small, so one case in a tiny area does not dominate the map.
      smoothed_excess_risk: overallRate > 0 ? smoothedRates[rowIndex] / overallRate : null,
      raw_rate: rateScale * rawRates[rowIndex],
      smoothed_rate: rateScale * smoothedRates[rowIndex],
      spatial_rate: spatialRates[rowIndex],
      neighbor_count: neighborCounts[rowIndex],
      geometry: getFeatureGeometryText(feature)
    });
  });
  if (smallDenominatorAreaCount)
    addCatalogDiagnostic(
      diagnostics,
      'small-denominators',
      'warning',
      `Areas with fewer than ${SMALL_EXPECTED_CASE_THRESHOLD} expected cases have an unstable raw excess_risk (one case moves it by more than 20 percent); map smoothed_excess_risk instead.`,
      smallDenominatorAreaCount
    );
  return {
    tables: [table],
    diagnostics,
    summary: {
      inputCount: areas.features.length,
      areaCount: rows.length,
      droppedCount,
      totalCases,
      totalPopulation,
      overallRate,
      aboveExpectationCount,
      islandCount,
      rateScale,
      // Run-time caution `small-denominators`: areas whose expected cases are below the threshold.
      smallDenominatorAreaCount,
      smallDenominatorThreshold: SMALL_EXPECTED_CASE_THRESHOLD
    }
  };
}

/**
 * Excess risk: expected cases under the overall rate, observed over expected, an empirical Bayes
 * smoothed rate and a rate pooled over contiguous neighbours.
 */
export const CURATED_EXCESS_RISK_BINDING: CatalogOperationBinding = {
  id: BINDING_ID,
  title: 'Excess risk and smoothed rates',
  domain: 'analysis',
  summary:
    'Computes the expected cases of each area from its population and the overall rate, the excess risk as observed over expected and its empirical Bayes smoothed form, the empirical Bayes smoothed rate, and the spatial rate pooled over the area and its contiguous neighbours.',
  capabilityIds: ['spatial-autocorrelation-and-inference'],
  contributorNames: [
    'GPUEmpiricalBayesRates',
    'GPUSpatialEmpiricalBayesRates',
    'GPUContiguityWeights'
  ],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [
    makeInputSpec(
      'areas',
      'Areas with cases and population',
      ['polygon'],
      MAXIMUM_CURATED_AREA_FEATURES,
      MAXIMUM_CURATED_AREA_VERTICES
    )
  ],
  parameters: PARAMETERS,
  outputs: [
    makeOutputSpec(
      'excess-risk',
      'Excess risk and smoothed rates',
      'polygon',
      ['source_index', 'source_id'],
      MAXIMUM_CURATED_AREA_FEATURES
    )
  ],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    'Every input area is listed in the output in input order; an area that cannot be used has a status naming why and empty results, and each reason is counted in a diagnostic: no geometry, a missing, non-numeric or non-finite value, a zero or negative population, or negative cases.',
    'Two excess-risk columns are published: excess_risk is the raw ratio of observed to expected cases and smoothed_excess_risk is the empirical Bayes smoothed rate over the overall rate; the smoothed column is the one to map by default. The raw ratio is unstable for small populations: an area with fewer than 5 expected cases changes by more than 20 percent per case, and the highest raw values of a map are often tiny areas driven by one case; the summary key smallDenominatorAreaCount counts such areas and a diagnostic names them.',
    'Expected cases are the population times the overall rate of the usable areas, so they add up to the observed cases; excess risk is observed over expected and is empty everywhere (with a diagnostic) when no usable area has a case.',
    'The smoothed rate is the global empirical Bayes rate of Marshall (1991) with the estimated prior variance clamped at zero; the spatial rate is the pooled ratio (cases of the area and its neighbours) over (their population), with binary weights; both are multiplied by the rate multiplier, as is the raw rate.',
    'An area with no contiguous neighbour among the usable areas has no spatial rate by default, or keeps its own rate with islandPolicy own-rate; it is counted in a diagnostic.',
    'Fewer than three usable areas throw before any work; at most 5000 areas.'
  ],
  limitation:
    'CPU reference engine only; the GPU path (GPUEmpiricalBayesRates, GPUSpatialEmpiricalBayesRates) is future work. Excess risk is a ratio with no confidence interval or Poisson test, and the spatially smoothed empirical Bayes rate of GPUSpatialEmpiricalBayesRates is not computed, only the pooled spatial rate of addSpatialRateRecipe.',
  run: (inputs, values) => runExcessRisk(inputs, values)
};
