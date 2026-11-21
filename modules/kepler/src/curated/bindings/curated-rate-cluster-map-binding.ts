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
  PLATFORM3_GPU_OR_CPU_CLEANUP,
  SEED_PARAMETER_SPEC,
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
  getMeanAndDeviation,
  runEmpiricalBayesRates,
  runMultipleTestingAdjustment,
  type AdjustmentMethod
} from '../../analysis/catalog-platform3-b-statistics';
import {KeplerAllocationError} from '../../runtime/dataset-registry';
import {
  CONTIGUITY_CRITERION_PARAMETER,
  MAXIMUM_CURATED_AREA_FEATURES,
  MAXIMUM_CURATED_AREA_VERTICES,
  MAXIMUM_PERMUTATION_WORK,
  isConstantList,
  makeColumnParameter,
  makeCuratedError,
  getFeatureAnchors,
  getSignificanceAttainability,
  MAXIMUM_PERMUTATIONS,
  makeDerivedCollection,
  makePermutationParameter,
  readColumnParameter,
  readRateRows,
  runConditionalLocalPermutationTest
} from './curated-binding-utils';

import {
  runGpuRateClusterMap,
  validateRateClusterRows,
  type RateClusterComputation
} from './curated-rate-cluster-map-gpu';

const BINDING_ID = 'curated-rate-cluster-map';

/** Local Moran and its permutation test need at least four areas (the catalog local statistics' own floor). */
const MINIMUM_AREAS = 4;

/** Multiple-testing choices; `none` leaves the pseudo p-values uncorrected. */
const MULTIPLE_TESTING_OPTIONS = [
  'benjamini-hochberg',
  'holm',
  'bonferroni',
  'sidak',
  'none'
] as const;

const PARAMETERS: readonly CatalogParameterSpec[] = [
  makeColumnParameter('casesProperty', 'Cases property'),
  makeColumnParameter('populationProperty', 'Population property'),
  CONTIGUITY_CRITERION_PARAMETER,
  makePermutationParameter(),
  SEED_PARAMETER_SPEC,
  makeNumberParameter('significance', 'Significance level', 0.05, 1e-9, 0.5),
  {
    name: 'multipleTesting',
    label: 'Multiple-testing correction',
    kind: 'enum',
    options: MULTIPLE_TESTING_OPTIONS,
    defaultValue: 'benjamini-hochberg'
  },
  COORDINATE_SYSTEM_PARAMETER
];

const COLUMN_TYPES = {
  source_index: 'int32',
  source_id: 'string',
  status: 'string',
  cases: 'float64',
  population: 'float64',
  raw_rate: 'float64',
  shrinkage: 'float64',
  smoothed_rate: 'float64',
  neighbor_count: 'int32',
  local_i: 'float64',
  spatial_lag: 'float64',
  p_value: 'float64',
  adjusted_p_value: 'float64',
  cluster: 'string',
  geometry: 'string'
} as const;

function runRateClusterMap(inputs: Inputs, values: Values): CatalogRunOutput {
  const areas = inputs['areas'];
  const diagnostics: CatalogDiagnostic[] = [];
  const casesProperty = readColumnParameter(BINDING_ID, values, 'casesProperty');
  const populationProperty = readColumnParameter(BINDING_ID, values, 'populationProperty');
  const permutations = values['permutations'] as number;
  const significance = values['significance'] as number;
  const multipleTesting = values['multipleTesting'] as (typeof MULTIPLE_TESTING_OPTIONS)[number];
  const coordinateSystem = values['coordinateSystem'] as CatalogCoordinateSystem;

  const {rows, droppedCount, statuses} = readRateRows(
    areas,
    casesProperty,
    populationProperty,
    diagnostics
  );
  validateRateClusterRows(rows, casesProperty, populationProperty);
  if (rows.length < MINIMUM_AREAS)
    throw makeCuratedError(
      BINDING_ID,
      `${rows.length} usable areas; at least ${MINIMUM_AREAS} with geometry, finite cases and a positive population are needed`
    );
  // Rows times permutations bounds the conditional permutation work of the local statistic.
  if (rows.length * permutations > MAXIMUM_PERMUTATION_WORK)
    throw new KeplerAllocationError('capacity', {
      message: `${BINDING_ID}: ${rows.length} areas times ${permutations} permutations exceeds the ${MAXIMUM_PERMUTATION_WORK} the CPU engine runs; lower permutations`
    });

  // Step 1: Marshall's global empirical Bayes estimate of the rate, with the prior variance
  // clamped at zero (the recipe's `clampPriorVariance` default), by the catalog's own function.
  const rateCollection = makeDerivedCollection(
    rows.map(row => row.feature),
    index => ({cases: rows[index].cases, population: rows[index].population})
  );
  const smoothing = runEmpiricalBayesRates(rateCollection, {
    numeratorProperty: 'cases',
    denominatorProperty: 'population'
  });
  const smoothingTable = smoothing.tables[0];
  const rawRates = smoothingTable.columns['raw_rate'] as number[];
  const shrinkages = smoothingTable.columns['shrinkage'] as number[];
  const smoothedRates = smoothingTable.columns['eb_rate'] as number[];

  // Step 2 and 3: row-standardised contiguity and a local Moran statistic of the smoothed rate,
  // `z_i * sum_j w_ij z_j` with `z` the standardised rate (the statistic of `runLocalMoran`), tested
  // by conditional permutation. The permutation stream is drawn per area in the canonical order of
  // the area anchors, so it does not depend on the order of the input rows.
  const hasSpread = !isConstantList(smoothedRates);
  const localI = new Array<number | null>(rows.length).fill(null);
  const spatialLag = new Array<number | null>(rows.length).fill(null);
  const pValues = new Array<number | null>(rows.length).fill(null);
  const quadrants = new Array<string>(rows.length).fill('not-tested');
  const neighborCounts = new Array<number | null>(rows.length).fill(null);
  let tiedAnchorCount = 0;
  if (hasSpread) {
    const smoothedCollection = makeDerivedCollection(
      rows.map(row => row.feature),
      index => ({smoothed_rate: smoothedRates[index]})
    );
    const set = buildAnalysisSet(smoothedCollection, 'smoothed_rate', {
      neighborhood: values['criterion'] as 'queen' | 'rook',
      neighborCount: 6,
      distanceBand: 1,
      rowStandardize: true,
      coordinateSystem
    });
    const {mean, deviation} = getMeanAndDeviation(set.values);
    const standardized = set.values.map(value => (value - mean) / deviation);
    const anchors = getFeatureAnchors(smoothedCollection, coordinateSystem);
    const neighbors = set.weights.neighbors;
    const test = runConditionalLocalPermutationTest(
      standardized,
      neighbors,
      set.sourceIndex.map(index => anchors[index] as [number, number]),
      permutations,
      values['seed'] as number
    );
    tiedAnchorCount = test.tiedCount;
    set.sourceIndex.forEach((rowIndex, position) => {
      const list = neighbors[position];
      const lag = list.reduce(
        (sum, other, slot) => sum + set.weights.weights[position][slot] * standardized[other],
        0
      );
      localI[rowIndex] = standardized[position] * lag;
      spatialLag[rowIndex] = lag;
      pValues[rowIndex] = test.pValues[position];
      quadrants[rowIndex] =
        list.length === 0
          ? 'not-tested'
          : standardized[position] >= 0
            ? lag >= 0
              ? 'HH'
              : 'HL'
            : lag >= 0
              ? 'LH'
              : 'LL';
      neighborCounts[rowIndex] = list.length;
    });
    if (tiedAnchorCount)
      addCatalogDiagnostic(
        diagnostics,
        'row-order-dependent-inference',
        'warning',
        'Coincident anchors retain input order in the permutation pools and streams, so reordering them can change inference throughout the dataset, including areas with unique anchors.',
        tiedAnchorCount
      );
    const islandCount = neighborCounts.filter(count => count === 0).length;
    if (islandCount)
      addCatalogDiagnostic(
        diagnostics,
        'island-features',
        'warning',
        'Areas with no contiguous neighbour are not tested: their cluster is not-tested and they are not part of the corrected family.',
        islandCount
      );
  } else {
    addCatalogDiagnostic(
      diagnostics,
      'constant-smoothed-rate',
      'warning',
      'The smoothed rates are all equal (the estimated prior variance is zero, so every area shrinks to the overall rate): there is no spatial variation to test and no area is tested.'
    );
  }

  // Step 4: Benjamini-Hochberg (or the chosen correction) over the tested family, which is every
  // area with a valid rate and at least one neighbour, as in the recipe.
  const testedRows = pValues
    .map((value, index) => (value === null ? -1 : index))
    .filter(index => index >= 0);
  const adjustedPValues = new Array<number | null>(rows.length).fill(null);
  if (testedRows.length) {
    if (multipleTesting === 'none') {
      for (const index of testedRows) adjustedPValues[index] = pValues[index];
    } else {
      const adjustment = runMultipleTestingAdjustment(
        makeDerivedCollection(
          testedRows.map(index => rows[index].feature),
          position => ({p: pValues[testedRows[position]]})
        ),
        {pValueProperty: 'p', method: multipleTesting as AdjustmentMethod, alpha: significance}
      );
      (adjustment.tables[0].columns['adjusted_p_value'] as number[]).forEach((value, position) => {
        adjustedPValues[testedRows[position]] = value;
      });
    }
  }
  const clusters = quadrants.map((quadrant, index) =>
    adjustedPValues[index] === null
      ? 'not-tested'
      : (adjustedPValues[index] as number) <= significance
        ? quadrant
        : 'not-significant'
  );

  return publishRateClusterMap(inputs, values, diagnostics, {
    rows,
    droppedCount,
    statuses,
    rawRates,
    shrinkages,
    smoothedRates,
    localI,
    spatialLag,
    pValues,
    adjustedPValues,
    neighborCounts,
    clusters,
    tiedAnchorCount,
    effectiveSignificance: significance,
    globalRate: Number(smoothing.summary['globalRate']),
    priorVariance: Number(smoothing.summary['priorVariance'])
  });
}

/** One publication contract for CPU reference and recipe readback. */
export function publishRateClusterMap(
  inputs: Inputs,
  values: Values,
  diagnostics: CatalogDiagnostic[],
  computation: RateClusterComputation,
  gpu = false
): CatalogRunOutput {
  const areas = inputs['areas'];
  const {
    rows,
    droppedCount,
    statuses,
    rawRates,
    shrinkages,
    smoothedRates,
    localI,
    spatialLag,
    pValues,
    adjustedPValues,
    neighborCounts,
    clusters,
    tiedAnchorCount,
    globalRate,
    priorVariance
  } = computation;
  const casesProperty = String(values['casesProperty']);
  const populationProperty = String(values['populationProperty']);
  const permutations = Number(values['permutations']);
  const significance = computation.effectiveSignificance;
  const multipleTesting = String(values['multipleTesting']);
  const testedRows = pValues.filter(value => value !== null);
  // Output: every input feature, in input order, with the reason when it was not analysed.
  const rowByInputIndex = new Map(rows.map((row, index) => [row.sourceIndex, index]));
  const table = makeCatalogTable('rate-clusters', 'Rate clusters', {...COLUMN_TYPES}, 'geometry');
  areas.features.forEach((feature, sourceIndex) => {
    const rowIndex = rowByInputIndex.get(sourceIndex);
    const analysed = rowIndex !== undefined;
    appendCatalogRow(table, {
      source_index: sourceIndex,
      source_id: getFeatureIdentity(feature),
      status: statuses[sourceIndex],
      cases: readNumericProperty(feature, casesProperty),
      population: readNumericProperty(feature, populationProperty),
      raw_rate: analysed ? rawRates[rowIndex] : null,
      shrinkage: analysed ? shrinkages[rowIndex] : null,
      smoothed_rate: analysed ? smoothedRates[rowIndex] : null,
      neighbor_count: analysed ? neighborCounts[rowIndex] : null,
      local_i: analysed ? localI[rowIndex] : null,
      spatial_lag: analysed ? spatialLag[rowIndex] : null,
      p_value: analysed ? pValues[rowIndex] : null,
      adjusted_p_value: analysed ? adjustedPValues[rowIndex] : null,
      cluster: analysed ? clusters[rowIndex] : 'not-tested',
      geometry: getFeatureGeometryText(feature)
    });
  });
  addCatalogDiagnostic(
    diagnostics,
    'pseudo-p-values',
    'info',
    `p-values are conditional-permutation pseudo p-values (${permutations} permutations, seeded, ${gpu ? 'doubling the smaller tail, using the public recipe and Philox stream' : 'counting permuted statistics whose absolute value is at least the observed one'}; the absolute-value and doubled-smaller-tail definitions can differ materially on skewed data; either may give the smaller p-value); ${
      multipleTesting === 'none'
        ? 'no correction was applied'
        : `${multipleTesting} correction over ${testedRows.length} tested areas`
    }. The cluster class is kept only when the corrected p-value is at most ${significance}.`
  );
  // Could any area be flagged at this permutation count? The smallest pseudo p-value is
  // 1 / (permutations + 1) on CPU and twice that on GPU, so correction can be out of reach.
  const minimumPValue = gpu ? Math.fround(2 / (permutations + 1)) : 1 / (permutations + 1);
  const attainability = getSignificanceAttainability(
    multipleTesting,
    significance,
    testedRows.length,
    permutations,
    gpu ? 2 : 1,
    gpu ? 'gpu' : 'cpu-reference'
  );
  const isAttainable = testedRows.length === 0 || attainability.attainable;
  if (!isAttainable)
    addCatalogDiagnostic(
      diagnostics,
      'significance-not-attainable',
      'warning',
      `With ${permutations} permutations the smallest pseudo p-value is ${minimumPValue.toPrecision(3)}, so ${multipleTesting} over ${testedRows.length} tested areas at ${significance} can never flag an area; about ${attainability.permutationsNeeded} permutations are needed${attainability.permutationsNeeded > MAXIMUM_PERMUTATIONS ? ` (more than the ${MAXIMUM_PERMUTATIONS} this binding accepts: use a less strict correction or a higher significance level)` : ''}.`
    );
  else if (
    attainability.sufficientMinimumPValueTieCount !== null &&
    Number.isFinite(attainability.sufficientMinimumPValueTieCount) &&
    attainability.sufficientMinimumPValueTieCount > 1
  )
    addCatalogDiagnostic(
      diagnostics,
      'significance-needs-ties',
      'info',
      `Benjamini-Hochberg selects the largest rank k satisfying p_(k) <= k * alpha / m and rejects ranks 1 through k. With ${permutations} permutations a sufficient scenario is at least ${attainability.sufficientMinimumPValueTieCount} of the ${testedRows.length} tested areas tied at the smallest possible pseudo p-value ${minimumPValue.toPrecision(3)}; other ranks can allow rejection without floor ties, depending on the cutoff.`
    );
  const countClass = (label: string) => clusters.filter(cluster => cluster === label).length;
  return {
    tables: [table],
    diagnostics,
    summary: {
      inputCount: areas.features.length,
      areaCount: rows.length,
      droppedCount,
      testedCount: testedRows.length,
      significantCount: countClass('HH') + countClass('LL') + countClass('HL') + countClass('LH'),
      highHighCount: countClass('HH'),
      lowLowCount: countClass('LL'),
      highLowCount: countClass('HL'),
      lowHighCount: countClass('LH'),
      globalRate,
      priorVariance,
      permutations,
      multipleTesting,
      rateAnalysis: 'smoothed',
      permutationAlternative: gpu ? 'two-sided-doubled-tail' : 'absolute-statistic',
      pValueConstruction: gpu ? 'doubled-minimum-tail' : 'upper-tail-count',
      localStatisticScaling: gpu ? 'sample-variance' : 'population-variance',
      spatialLagScaling: gpu ? 'centered-rate' : 'standardized-rate',
      minimumPseudoPValue: minimumPValue,
      requestedSignificance: Number(values['significance']),
      effectiveSignificance: significance,
      // Run-time caution `significance-not-attainable`: 1 when no area could be flagged.
      significanceNotAttainable: isAttainable ? 0 : 1,
      permutationsNeeded: testedRows.length ? attainability.permutationsNeeded : 0,
      // Run-time caution `row-order-dependent-inference`: areas whose centroid ties with another's.
      tiedAnchorCount
    }
  };
}

/**
 * Rate cluster map: empirical Bayes smoothing, queen or rook contiguity, a local Moran statistic of
 * the smoothed rate with a conditional permutation test, a multiple-testing correction over the
 * tested areas and the cluster class of each area.
 */
export const CURATED_RATE_CLUSTER_MAP_BINDING: CatalogOperationBinding = {
  id: BINDING_ID,
  title: 'Rate cluster map',
  domain: 'analysis',
  summary:
    'Smooths the rate of cases over population by empirical Bayes shrinkage, links areas that touch, tests the smoothed rate with a local Moran statistic and a conditional permutation test, corrects the p-values for the number of areas tested and classifies each area as a high or low cluster, an outlier or not significant.',
  capabilityIds: ['spatial-autocorrelation-and-inference'],
  contributorNames: [
    'GPUEmpiricalBayesRates',
    'GPUContiguityWeights',
    'GPUSpatialWeightsTransform',
    'GPULocalMoran',
    'GPULocalPermutationTest',
    'GPUMultipleTestingAdjustment'
  ],
  gpuContributorNames: [
    'GPUEmpiricalBayesRates',
    'GPUContiguityWeights',
    'GPUSpatialWeightsTransform',
    'GPULocalMoran',
    'GPULocalPermutationTest',
    'GPUMultipleTestingAdjustment'
  ],
  engine: 'gpu-or-cpu',
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
      'rate-clusters',
      'Rate clusters',
      'polygon',
      ['source_index', 'source_id'],
      MAXIMUM_CURATED_AREA_FEATURES
    )
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    'Every input area is listed in the output in input order; an area that cannot be analysed has a status naming why and the cluster not-tested, and each reason is counted in a diagnostic: no geometry, a missing, non-numeric or non-finite value, a zero or negative population, or negative cases.',
    'The smoothed rate is the global empirical Bayes rate of Marshall (1991) with the estimated prior variance clamped at zero; when it is zero every area shrinks to the overall rate, the smoothed rates are equal, no area is tested and a diagnostic says so.',
    'Neighbours are areas sharing a vertex (queen) or an edge (rook) among the usable areas, with weights standardised by row; an area with no neighbour is an island: it is not tested, is left out of the corrected family and is counted in a diagnostic.',
    'CPU reference p-value definition: for each area the area keeps its own standardised rate and draws as many rates as it has neighbours from the other areas without replacement; the pseudo p-value is (m + 1) / (permutations + 1), with m the draws whose absolute statistic is at least the observed absolute statistic. This absolute-value form is not the contributor GPULocalPermutationTest, which doubles the smaller tail: The definitions can differ materially on skewed data; either may give the smaller p-value.',
    'With distinct anchors the permutation stream does not depend on input row order: each area draws from a generator seeded by the seed and its rank when areas are sorted by anchor (centroid in the local frame) x, then y. Areas whose anchors coincide exactly keep their input order, which can affect inference throughout the dataset because their values enter other areas’ permutation pools; their count is in the summary key tiedAnchorCount and a diagnostic.',
    'The correction runs over the tested areas only and the cluster class is kept when the corrected p-value is at most the significance level. On the CPU reference with 999 permutations the smallest pseudo p-value is 0.001: Bonferroni and Holm over more than 50 tested areas, and Sidak over more than 51 tested areas, at 0.05 can never flag an area. Benjamini-Hochberg selects the largest rank k with p_(k) <= k * alpha / m and rejects ranks 1 through k; other ranks can allow rejection without floor ties, depending on the cutoff. The helper sufficientMinimumPValueTieCount verifies a sufficient floor-tie scenario in the selected engine arithmetic. The diagnostic and summary keys significanceNotAttainable and permutationsNeeded say when no area could be flagged.',
    'The GPU path dispatches addRateClusterMapRecipe with analyze=smoothed, clamped prior variance, row-standardized weights and two-sided (doubled-smaller-tail) permutation inference. CPU reference inference retains its absolute-statistic definition; selecting an engine can change p-values and classes. GPU local I uses the recipe sample-variance convention and its spatial_lag is in centered rate units; the CPU statistic uses population variance and a standardized-rate spatial_lag. Both preserve the class names and input identities.',
    'The GPU significance gate uses the requested significance rounded once to float32, exactly as packed into the recipe. The run summary records requestedSignificance and effectiveSignificance; diagnostics and published classes use effectiveSignificance. The CPU reference retains the requested binary64 threshold.',
    'GPU contiguity uses exact float32 local-frame vertex or edge keys. CPU tolerance-based contiguity can differ on nearly coincident vertices. GPU topology or permutation overflow rejects publication and uses the catalog fallback, with a diagnostic.',
    'Fewer than four usable areas throw before any work. The permutation work bound (areas times permutations above 20 million throws a capacity error) can only bind above 2000 areas, because permutations are at most 9999; at the default 999 permutations it would need more than the 5000 areas the binding accepts.'
  ],
  limitation:
    'Both engines analyse the smoothed rate, rather than the recipe default Assuncao-Reis rate. GPU inference uses doubled-smaller-tail p-values, sample-variance local I and a centered-rate spatial lag; CPU reference inference uses absolute-statistic p-values and a population-variance local I. Sidak correction runs on the CPU reference engine only. GPU compilation bounds are 64 neighbours per area and 16 directed slots per area on average; overflow falls back visibly to the CPU reference.',
  run: (inputs, values) => runRateClusterMap(inputs, values),
  async runGpu(context, inputs, values) {
    const {computation, diagnostics, dispatchedContributors} = await runGpuRateClusterMap(
      context,
      inputs,
      values
    );
    return {
      ...publishRateClusterMap(inputs, values, diagnostics, computation, true),
      dispatchedContributors
    };
  }
};
