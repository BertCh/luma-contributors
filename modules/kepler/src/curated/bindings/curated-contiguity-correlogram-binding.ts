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
  makeCatalogTable,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic
} from '../../analysis/catalog-geojson';
import {
  PLATFORM3_CPU_CLEANUP,
  SEED_PARAMETER_SPEC,
  makeInputSpec,
  makeNumberParameter,
  makeOutputSpec,
  readNumericProperty,
  type Inputs,
  type Values
} from '../../analysis/catalog-platform3-b-common';
import {
  buildAnalysisSet,
  runGlobalSpatialStatistics,
  type AnalysisSet,
  type PrecomputedGlobalStatistics
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
  makeDerivedCollection,
  makePermutationParameter,
  readColumnParameter,
  runCanonicalMoranPermutationTest
} from './curated-binding-utils';

const BINDING_ID = 'curated-contiguity-correlogram';

/** The global statistic needs at least four areas with a neighbour at the order (the catalog's own floor). */
const MINIMUM_AREAS_PER_ORDER = 4;

/**
 * Largest order. The rings of order k grow with k, so the permutation work grows with k too; 12
 * covers every lattice a 5000 area input can hold before the rings run out.
 */
const MAXIMUM_ORDER = 12;

const PARAMETERS: readonly CatalogParameterSpec[] = [
  makeColumnParameter('valueProperty', 'Value property'),
  makeNumberParameter(
    'maximumOrder',
    'Largest contiguity order',
    6,
    1,
    MAXIMUM_ORDER,
    undefined,
    'integer'
  ),
  CONTIGUITY_CRITERION_PARAMETER,
  makePermutationParameter(),
  SEED_PARAMETER_SPEC,
  makeNumberParameter('significance', 'Significance level', 0.05, 1e-9, 0.5),
  COORDINATE_SYSTEM_PARAMETER
];

/**
 * Neighbours at exactly `order` steps of a contiguity graph (libpysal `higher_order`: the rings of
 * a breadth-first search, with the area itself and every nearer area excluded). The graph is
 * undirected, so a ring is symmetric. Same construction as the `higherOrder` operation of
 * `computeSpatialWeightsAlgebraOnCPU` in the analysis package, which works on its CSR layout.
 */
export function computeContiguityRings(
  neighbors: readonly (readonly number[])[],
  maximumOrder: number
): number[][][] {
  const rowCount = neighbors.length;
  // rings[order - 1][row] is the sorted list of areas at exactly that many steps from the row.
  const rings: number[][][] = Array.from({length: maximumOrder}, () =>
    Array.from({length: rowCount}, () => [] as number[])
  );
  const stamp = new Int32Array(rowCount).fill(-1);
  for (let row = 0; row < rowCount; row++) {
    stamp[row] = row;
    let frontier = [row];
    for (let order = 1; order <= maximumOrder && frontier.length; order++) {
      const next: number[] = [];
      for (const middle of frontier)
        for (const other of neighbors[middle])
          if (stamp[other] !== row) {
            stamp[other] = row;
            next.push(other);
          }
      next.sort((a, b) => a - b);
      rings[order - 1][row] = next;
      frontier = next;
    }
  }
  return rings;
}

type OrderStatistics = {
  count: number;
  islandCount: number;
  linkCount: number;
  moransI: number | null;
  expected: number | null;
  variance: number | null;
  zScore: number | null;
  pValue: number | null;
  pseudoPValue: number | null;
};

function runContiguityCorrelogram(inputs: Inputs, values: Values): CatalogRunOutput {
  const areas = inputs['areas'];
  const diagnostics: CatalogDiagnostic[] = [];
  const valueProperty = readColumnParameter(BINDING_ID, values, 'valueProperty');
  const maximumOrder = values['maximumOrder'] as number;
  const permutations = values['permutations'] as number;
  const significance = values['significance'] as number;
  const coordinateSystem = values['coordinateSystem'] as CatalogCoordinateSystem;

  const features = [];
  const rowValues: number[] = [];
  let missingGeometryCount = 0;
  let missingValueCount = 0;
  for (const feature of areas.features) {
    if (!feature.geometry) {
      missingGeometryCount++;
      continue;
    }
    const value = readNumericProperty(feature, valueProperty);
    if (value === null) {
      missingValueCount++;
      continue;
    }
    features.push(feature);
    rowValues.push(value);
  }
  if (missingGeometryCount)
    addCatalogDiagnostic(
      diagnostics,
      'missing-geometry',
      'warning',
      'Features without geometry are skipped: they have no neighbours.',
      missingGeometryCount
    );
  if (missingValueCount)
    addCatalogDiagnostic(
      diagnostics,
      'missing-value',
      'warning',
      'Features whose value is missing, not a number or not finite are skipped.',
      missingValueCount
    );
  if (rowValues.length < MINIMUM_AREAS_PER_ORDER + 1)
    throw makeCuratedError(
      BINDING_ID,
      `${rowValues.length} usable areas; at least ${MINIMUM_AREAS_PER_ORDER + 1} with geometry and a finite ${valueProperty} are needed`
    );
  if (isConstantList(rowValues))
    throw makeCuratedError(
      BINDING_ID,
      `the values of ${valueProperty} are all equal, so Moran's I is undefined`
    );

  // Order one is the contiguity graph of the usable areas; higher orders are its breadth-first rings.
  const collection = makeDerivedCollection(features, index => ({value: rowValues[index]}));
  const baseSettings = {
    neighborhood: values['criterion'] as 'queen' | 'rook',
    neighborCount: 6,
    distanceBand: 1,
    rowStandardize: false,
    coordinateSystem
  };
  const baseSet = buildAnalysisSet(collection, 'value', baseSettings);
  const rings = computeContiguityRings(baseSet.weights.neighbors, maximumOrder);
  const linksPerOrder = rings.map(ring => ring.reduce((sum, list) => sum + list.length, 0));
  const permutationWork = linksPerOrder.reduce((sum, links) => sum + links, 0) * permutations;
  if (permutationWork > MAXIMUM_PERMUTATION_WORK)
    throw new KeplerAllocationError('capacity', {
      message: `${BINDING_ID}: ${permutationWork} link-permutation steps exceed the ${MAXIMUM_PERMUTATION_WORK} the CPU engine runs; lower permutations or maximumOrder`
    });

  const anchors = getFeatureAnchors(collection, coordinateSystem);
  let tiedAnchorCount = 0;
  const statistics: OrderStatistics[] = [];
  let emptyOrderCount = 0;
  for (let order = 1; order <= maximumOrder; order++) {
    const ring = rings[order - 1];
    // Rows with no neighbour at this order are islands of the order and leave its sample.
    const included: number[] = [];
    ring.forEach((list, row) => {
      if (list.length) included.push(row);
    });
    const empty: OrderStatistics = {
      count: included.length,
      islandCount: ring.length - included.length,
      linkCount: linksPerOrder[order - 1],
      moransI: null,
      expected: null,
      variance: null,
      zScore: null,
      pValue: null,
      pseudoPValue: null
    };
    const orderValues = included.map(row => rowValues[row]);
    if (included.length < MINIMUM_AREAS_PER_ORDER || isConstantList(orderValues)) {
      emptyOrderCount++;
      statistics.push(empty);
      continue;
    }
    const compactIndex = new Map(included.map((row, compact) => [row, compact]));
    const neighbors = included.map(row =>
      ring[row].map(other => compactIndex.get(other) as number)
    );
    const weights = neighbors.map(list => list.map(() => 1 / list.length));
    const set: AnalysisSet = {
      sourceIndex: included,
      values: orderValues,
      weights: {neighbors, weights, distances: neighbors.map(list => list.map(() => 0))},
      droppedCount: 0
    };
    // The analytic Moran block of the catalog's global statistic, computed from the ring weights.
    // The function accepts a ready analysis set through its `precomputed` argument; the statistic
    // blocks are left undefined there so that it computes them, which its code supports with `??`.
    const analytic = runGlobalSpatialStatistics(
      collection,
      {valueProperty: 'value', ...baseSettings, rowStandardize: true},
      {set, moran: undefined, geary: undefined} as unknown as PrecomputedGlobalStatistics
    ).tables[0];
    const permutation = runCanonicalMoranPermutationTest(
      orderValues,
      neighbors,
      weights,
      included.map(row => anchors[row] as [number, number]),
      permutations,
      values['seed'] as number
    );
    tiedAnchorCount = permutation.tiedCount;
    statistics.push({
      ...empty,
      moransI: analytic.columns['value'][0] as number,
      expected: analytic.columns['expected'][0] as number,
      variance: analytic.columns['variance'][0] as number,
      zScore: analytic.columns['z_score'][0] as number | null,
      pValue: analytic.columns['p_value'][0] as number | null,
      pseudoPValue: permutation.pseudoPValue
    });
  }
  const orderOneIslands = statistics[0].islandCount;
  if (orderOneIslands)
    addCatalogDiagnostic(
      diagnostics,
      'island-features',
      'warning',
      'Areas with no contiguous neighbour are left out of every order.',
      orderOneIslands
    );
  if (emptyOrderCount)
    addCatalogDiagnostic(
      diagnostics,
      'order-not-computed',
      'info',
      'Orders with fewer than four areas that have a neighbour at that order (the rings ran out) have no statistic.',
      emptyOrderCount
    );
  if (tiedAnchorCount)
    addCatalogDiagnostic(
      diagnostics,
      'row-order-dependent-inference',
      'warning',
      'Coincident anchors retain input order in permutation pools and streams, so reordering them can change inference throughout the dataset.',
      tiedAnchorCount
    );
  const table = makeCatalogTable('correlogram', 'Contiguity correlogram', {
    order: 'int32',
    area_count: 'int32',
    island_count: 'int32',
    link_count: 'int32',
    morans_i: 'float64',
    expected_i: 'float64',
    variance: 'float64',
    z_score: 'float64',
    p_value: 'float64',
    pseudo_p_value: 'float64',
    significant: 'boolean'
  });
  statistics.forEach((entry, index) =>
    appendCatalogRow(table, {
      order: index + 1,
      area_count: entry.count,
      island_count: entry.islandCount,
      link_count: entry.linkCount,
      morans_i: entry.moransI,
      expected_i: entry.expected,
      variance: entry.variance,
      z_score: entry.zScore,
      p_value: entry.pValue,
      pseudo_p_value: entry.pseudoPValue,
      significant: entry.pseudoPValue === null ? null : entry.pseudoPValue <= significance
    })
  );
  addCatalogDiagnostic(
    diagnostics,
    'analytic-and-permutation-inference',
    'info',
    `p_value is the normal approximation under the randomization variance; pseudo_p_value is the two-sided permutation pseudo p-value from ${permutations} seeded permutations of the values over the areas. Neither is corrected for the number of orders.`
  );
  return {
    tables: [table],
    diagnostics,
    summary: {
      areaCount: rowValues.length,
      maximumOrder,
      computedOrderCount: maximumOrder - emptyOrderCount,
      orderOneMoransI: statistics[0].moransI ?? 'not computed',
      // Run-time caution `row-order-dependent-inference`: areas whose centroid ties with another's.
      tiedAnchorCount,
      permutations
    }
  };
}

/**
 * Contiguity correlogram: global Moran's I of a value at contiguity orders 1 to a maximum, each
 * with an analytic p-value and a permutation pseudo p-value.
 */
export const CURATED_CONTIGUITY_CORRELOGRAM_BINDING: CatalogOperationBinding = {
  id: BINDING_ID,
  title: 'Contiguity correlogram',
  domain: 'analysis',
  summary:
    'Computes the global Moran statistic of a value for the areas at exactly one, two, up to a chosen number of steps of adjacency, with a permutation pseudo p-value for each order, so the table shows how association fades with distance in steps.',
  capabilityIds: ['spatial-autocorrelation-and-inference'],
  contributorNames: [
    'GPUContiguityWeights',
    'GPUSpatialWeightsAlgebra',
    'GPUSpatialWeightsTransform',
    'GPUGlobalSpatialStatistics'
  ],
  gpuContributorNames: [],
  engine: 'cpu-reference',
  inputs: [
    makeInputSpec(
      'areas',
      'Areas with a value',
      ['polygon'],
      MAXIMUM_CURATED_AREA_FEATURES,
      MAXIMUM_CURATED_AREA_VERTICES
    )
  ],
  parameters: PARAMETERS,
  outputs: [
    makeOutputSpec('correlogram', 'Contiguity correlogram', 'none', ['order'], MAXIMUM_ORDER)
  ],
  cleanup: PLATFORM3_CPU_CLEANUP,
  declaredBehavior: [
    'Features without geometry or with a missing, non-numeric or non-finite value are skipped and each reason is counted in a diagnostic; fewer than five usable areas, or a constant value, throw.',
    'Order one links areas that share a vertex (queen) or an edge (rook); order k links areas at exactly k steps of that graph, excluding the area itself and all nearer areas (libpysal higher_order), and each order is standardised by row.',
    'Areas with no neighbour at an order are islands of that order and leave its sample, so the count of areas can fall with the order; an order with fewer than four areas with a neighbour, or a constant value among them, has empty statistics and is counted in a diagnostic.',
    "Moran's I is (n / S0) sum w z z / sum z squared with the mean taken over the areas of the order; the analytic p-value is a normal approximation under the randomization variance; the pseudo p-value is two-sided, (m + 1) / (permutations + 1) with m the permutations at least as far from the mean of the permuted statistics as the observed one. The orders are tested separately: no correction is made for testing several orders. With distinct centroids the permutation stream does not depend on input order: areas are sorted by centroid x then y before shuffling. Coincident centroids keep input order and can affect inference throughout the dataset (summary key tiedAnchorCount).",
    'Permutation work above 20 million link steps throws a capacity error; at most 5000 areas and 12 orders.'
  ],
  limitation:
    'CPU reference engine only; the GPU path (GPUSpatialWeightsAlgebra and GPUGlobalSpatialStatistics) is future work. The analytic p-value uses the randomization variance only, not the normality variance that addContiguityOrderCorrelogramRecipe also reports; the permutation pseudo p-value is added. Orders are rings, not cumulative neighbourhoods, and the correlogram is not tested for trend.',
  run: (inputs, values) => runContiguityCorrelogram(inputs, values)
};
