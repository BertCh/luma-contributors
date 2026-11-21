// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUParameterBuffer, validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {
  addRateClusterMapRecipe,
  GPU_EMPIRICAL_BAYES_SUMMARY,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogGpuContext} from '../../analysis/catalog-binding-types';
import {
  addCatalogDiagnostic,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic
} from '../../analysis/catalog-geojson';
import {
  CatalogGpuUnsupportedError,
  withCatalogGpuSession
} from '../../analysis/catalog-gpu-session';
import type {Inputs, Values} from '../../analysis/catalog-platform3-b-common';
import {makeRecipeFrame, readRecipePolygons} from '../../analysis/catalog-platform5-recipe-common';
import {
  getCanonicalOrder,
  getFeatureAnchors,
  makeCuratedError,
  MAXIMUM_PERMUTATION_WORK,
  readColumnParameter,
  readRateRows,
  type RateRows
} from './curated-binding-utils';

const BINDING_ID = 'curated-rate-cluster-map';
/** Match the binding reference engine's local-inference floor. */
const MINIMUM_GPU_AREAS = 4;
/** The contributor's bounded per-invocation permutation pool. */
const MAXIMUM_NEIGHBORS = 64;
/** Bounded topology scratch; dense inputs must retry on the reference engine. */
const NEIGHBOR_SLOTS_PER_AREA = 16;
/** Shared vertices can emit repeated directed pairs before deduplication. */
const PAIRS_PER_NEIGHBOR_SLOT = 4;
/** Conservative upload bound; nonfinite GPU moments still reject publication. */
const MAXIMUM_RATE_INPUT = 1e20;

export type RateClusterComputation = RateRows & {
  rawRates: number[];
  shrinkages: number[];
  smoothedRates: number[];
  localI: (number | null)[];
  spatialLag: (number | null)[];
  pValues: (number | null)[];
  adjustedPValues: (number | null)[];
  neighborCounts: (number | null)[];
  clusters: string[];
  tiedAnchorCount: number;
  effectiveSignificance: number;
  globalRate: number;
  priorVariance: number;
};

/** Apply the reference engine's binomial-rate constraint before either engine runs. */
export function validateRateClusterRows(
  rows: RateRows['rows'],
  casesProperty: string,
  populationProperty: string
): void {
  for (const row of rows)
    validateRequirement(
      BINDING_ID,
      row.cases <= row.population,
      `casesProperty "${casesProperty}" at source row ${row.sourceIndex} must not exceed populationProperty "${populationProperty}" (${row.cases} > ${row.population}); correct the cases or population`
    );
}

/** CPU preparation only: role reads, canonical identity order and existing polygon packing. */
export function prepareGpuRateClusterMap(inputs: Inputs, values: Values) {
  if (values['multipleTesting'] === 'sidak')
    throw new CatalogGpuUnsupportedError(
      `${BINDING_ID}: Sidak correction is available on the CPU reference engine only`
    );
  const diagnostics: CatalogDiagnostic[] = [];
  const casesProperty = readColumnParameter(BINDING_ID, values, 'casesProperty');
  const populationProperty = readColumnParameter(BINDING_ID, values, 'populationProperty');
  const rateRows = readRateRows(inputs['areas'], casesProperty, populationProperty, diagnostics);
  validateRateClusterRows(rateRows.rows, casesProperty, populationProperty);
  if (rateRows.rows.length < MINIMUM_GPU_AREAS)
    throw makeCuratedError(
      BINDING_ID,
      `${rateRows.rows.length} usable areas; at least ${MINIMUM_GPU_AREAS} with geometry, finite cases and a positive population are needed`
    );
  validateRequirement(
    BINDING_ID,
    rateRows.rows.length * Number(values['permutations']) <= MAXIMUM_PERMUTATION_WORK,
    `areas times permutations must not exceed ${MAXIMUM_PERMUTATION_WORK}; lower permutations`
  );
  validateRequirement(
    BINDING_ID,
    rateRows.rows.every(
      row =>
        row.cases <= MAXIMUM_RATE_INPUT &&
        row.population <= MAXIMUM_RATE_INPUT &&
        Math.fround(row.population) > 0
    ),
    `cases and population must fit the GPU rate input limit ${MAXIMUM_RATE_INPUT} and population must remain positive in float32`
  );
  const coordinateSystem = values['coordinateSystem'] as CatalogCoordinateSystem;
  const collection = {
    type: 'FeatureCollection' as const,
    features: rateRows.rows.map(row => row.feature)
  };
  const anchors = getFeatureAnchors(collection, coordinateSystem);
  validateRequirement(
    BINDING_ID,
    anchors.every(anchor => anchor !== null && anchor.every(Number.isFinite)),
    'areas must have finite local-frame polygon anchors'
  );
  const canonical = getCanonicalOrder(anchors as [number, number][]);
  const rows = canonical.order.map(index => rateRows.rows[index]);
  const ordered = {type: 'FeatureCollection' as const, features: rows.map(row => row.feature)};
  const polygons = readRecipePolygons(ordered, makeRecipeFrame(coordinateSystem, [ordered]));
  validateRequirement(
    BINDING_ID,
    polygons.featureCount === rows.length && polygons.positions.every(Number.isFinite),
    'areas must pack one finite polygon feature per usable rate row'
  );
  return {
    rows,
    // This is the threshold the recipe packs and compares in float32.
    effectiveSignificance: Math.fround(Number(values['significance'])),
    droppedCount: rateRows.droppedCount,
    statuses: rateRows.statuses,
    diagnostics,
    tiedAnchorCount: canonical.tiedCount,
    polygons,
    events: Float32Array.from(rows, row => row.cases),
    populations: Float32Array.from(rows, row => row.population),
    neighborCapacity: rows.length * Math.min(NEIGHBOR_SLOTS_PER_AREA, rows.length - 1)
  };
}

export type RateClusterReadback = {
  rawRates: Float32Array;
  smoothedRates: Float32Array;
  summary: Float32Array;
  localI: Float32Array;
  spatialLag: Float32Array;
  pValues: Float32Array;
  adjustedPValues: Float32Array;
  quadrants: Uint32Array;
  offsets: Uint32Array;
  weightsOverflow: number;
  weightsCandidateOverflow: number;
  weightsRequiredCount: number;
  permutationOverflow: number;
};

/** Refuse incomplete topology or permutation output before constructing publication rows. */
export function readGpuRateClusterComputation(
  prepared: ReturnType<typeof prepareGpuRateClusterMap>,
  readback: RateClusterReadback
): RateClusterComputation {
  validateRequirement(
    BINDING_ID,
    readback.weightsCandidateOverflow === 0,
    'GPU contiguity candidate overflow: grow pair capacity; no incomplete map is published'
  );
  validateRequirement(
    BINDING_ID,
    readback.weightsOverflow === 0 && readback.weightsRequiredCount <= prepared.neighborCapacity,
    `GPU contiguity overflow: required ${readback.weightsRequiredCount} slots, capacity ${prepared.neighborCapacity}; no incomplete map is published`
  );
  validateRequirement(
    BINDING_ID,
    readback.permutationOverflow === 0,
    `GPU permutation overflow: at most ${MAXIMUM_NEIGHBORS} neighbours per area are supported; no incomplete map is published`
  );
  const rowCount = prepared.rows.length;
  for (const name of [
    'rawRates',
    'smoothedRates',
    'localI',
    'spatialLag',
    'pValues',
    'adjustedPValues',
    'quadrants'
  ] as const)
    validateRequirement(
      BINDING_ID,
      readback[name].length === rowCount,
      `${name} readback must contain ${rowCount} rows`
    );
  validateRequirement(
    BINDING_ID,
    readback.offsets.length === rowCount + 1 &&
      readback.summary.length === GPU_EMPIRICAL_BAYES_SUMMARY.length,
    'GPU offsets and empirical Bayes summary readbacks must match their declared lengths'
  );
  validateRequirement(
    BINDING_ID,
    readback.summary[GPU_EMPIRICAL_BAYES_SUMMARY.count] === rowCount &&
      readback.offsets[0] === 0 &&
      readback.offsets[rowCount] === readback.weightsRequiredCount,
    'GPU rate count and CSR offsets must cover every prepared row and required neighbour slot'
  );
  const globalRate = readback.summary[GPU_EMPIRICAL_BAYES_SUMMARY.pooledRate];
  const priorVariance = readback.summary[GPU_EMPIRICAL_BAYES_SUMMARY.clampedPriorVariance];
  validateRequirement(
    BINDING_ID,
    Number.isFinite(globalRate) &&
      Number.isFinite(priorVariance) &&
      readback.smoothedRates.every(Number.isFinite),
    'GPU rate moments must be finite; rescale cases and population before retrying'
  );
  const neighborCounts = Array.from(
    {length: rowCount},
    (_, row) => readback.offsets[row + 1] - readback.offsets[row]
  );
  // Moran moments exclude islands: their rates cannot create variation in the tested population.
  const connectedRates = readback.smoothedRates.filter((_, row) => neighborCounts[row] > 0);
  const hasSpread = priorVariance > 0 && new Set(connectedRates).size > 1;
  const tested = Array.from(
    readback.localI,
    (value, row) =>
      hasSpread &&
      neighborCounts[row] > 0 &&
      Number.isFinite(value) &&
      Number.isFinite(readback.pValues[row]) &&
      Number.isFinite(readback.adjustedPValues[row])
  );
  const getNullableValues = (values: Float32Array) =>
    Array.from(values, (value, row) => (tested[row] && Number.isFinite(value) ? value : null));
  const pValues = getNullableValues(readback.pValues);
  const adjustedPValues = getNullableValues(readback.adjustedPValues);
  const labels = ['not-significant', 'HH', 'LH', 'LL', 'HL'];
  return {
    rows: prepared.rows,
    droppedCount: prepared.droppedCount,
    statuses: prepared.statuses,
    tiedAnchorCount: prepared.tiedAnchorCount,
    effectiveSignificance: prepared.effectiveSignificance,
    globalRate,
    priorVariance,
    rawRates: Array.from(readback.rawRates),
    smoothedRates: Array.from(readback.smoothedRates),
    // The contributor exposes rates and prior moments, from which the shrinkage weight follows.
    shrinkages: Array.from(prepared.populations, population =>
      priorVariance > 0 ? priorVariance / (priorVariance + globalRate / population) : 0
    ),
    localI: getNullableValues(readback.localI),
    spatialLag: getNullableValues(readback.spatialLag),
    pValues,
    adjustedPValues,
    neighborCounts: hasSpread ? neighborCounts : new Array<number | null>(rowCount).fill(null),
    clusters: Array.from(readback.quadrants, (quadrant, row) =>
      !tested[row]
        ? 'not-tested'
        : readback.adjustedPValues[row] <= prepared.effectiveSignificance
          ? labels[quadrant]
          : 'not-significant'
    )
  };
}

/** Rate smoothing, topology, local statistics and inference come from the public GPU recipe. */
export async function runGpuRateClusterMap(
  context: CatalogGpuContext,
  inputs: Inputs,
  values: Values
) {
  const prepared = prepareGpuRateClusterMap(inputs, values);
  return withCatalogGpuSession(context.device, BINDING_ID, async session => {
    let parameters: GPUParameterBuffer<'float32'> | undefined;
    let permutationParameters: GPUParameterBuffer<'uint32'> | undefined;
    try {
      parameters = new GPUParameterBuffer(context.device, {
        id: `${BINDING_ID}-parameters`,
        format: 'float32',
        length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
      });
      permutationParameters = new GPUParameterBuffer(context.device, {
        id: `${BINDING_ID}-permutation-parameters`,
        format: 'uint32',
        length: GPU_PERMUTATION_PARAMETER_LENGTH
      });
      const rowCount = prepared.rows.length;
      const rawRates = session.output('raw-rates', 'float32', rowCount);
      const smoothedRates = session.output('smoothed-rates', 'float32', rowCount);
      const summary = session.output('rate-summary', 'float32', GPU_EMPIRICAL_BAYES_SUMMARY.length);
      const localI = session.output('local-i', 'float32', rowCount);
      const spatialLag = session.output('spatial-lag', 'float32', rowCount);
      const pValues = session.output('p-values', 'float32', rowCount);
      const adjustedPValues = session.output('adjusted-p-values', 'float32', rowCount);
      const quadrants = session.output('quadrants', 'uint32', rowCount);
      const offsets = session.output('weights-offsets', 'uint32', rowCount + 1);
      const weightsOverflow = session.output('weights-overflow', 'uint32', 1);
      const weightsCandidateOverflow = session.output('weights-candidate-overflow', 'uint32', 1);
      const weightsRequiredCount = session.output('weights-required', 'uint32', 1);
      const permutationOverflow = session.output('permutation-overflow', 'uint32', 1);
      const recipe = addRateClusterMapRecipe(session.graph, {
        id: BINDING_ID,
        events: session.input('events', prepared.events, 'float32'),
        populations: session.input('populations', prepared.populations, 'float32'),
        polygons: {
          kind: 'polygons',
          positions: session.input('positions', prepared.polygons.positions, 'float32x2'),
          featureOffsets: session.input(
            'feature-offsets',
            prepared.polygons.featureOffsets,
            'uint32'
          ),
          polygonOffsets: session.input(
            'polygon-offsets',
            prepared.polygons.polygonOffsets,
            'uint32'
          ),
          ringOffsets: session.input('ring-offsets', prepared.polygons.ringOffsets, 'uint32')
        },
        criterion: values['criterion'] as 'queen' | 'rook',
        snapTolerance: 0,
        neighborCapacity: prepared.neighborCapacity,
        pairCapacity: prepared.neighborCapacity * PAIRS_PER_NEIGHBOR_SLOT,
        analyze: 'smoothed',
        clampPriorVariance: true,
        multipleTesting: values['multipleTesting'] as
          | 'none'
          | 'benjamini-hochberg'
          | 'holm'
          | 'bonferroni',
        parameters,
        settings: {
          seed: Number(values['seed']),
          permutationCount: Number(values['permutations']),
          significanceLevel: prepared.effectiveSignificance
        },
        permutation: {
          parameters: permutationParameters,
          maximumPermutations: Number(values['permutations']),
          maximumNeighbors: MAXIMUM_NEIGHBORS,
          alternative: 'two-sided'
        },
        // Colors are graph intermediates; the host publishes the named class column.
        palette: session.input('palette', new Uint32Array(5), 'uint32'),
        output: {
          smoothedRates: smoothedRates.view,
          localI: localI.view,
          quadrants: quadrants.view,
          status: {
            weightsOverflow: weightsOverflow.view,
            weightsCandidateOverflow: weightsCandidateOverflow.view,
            weightsRequiredCount: weightsRequiredCount.view
          },
          permutation: {
            pseudoPValues: pValues.view,
            adjustedPValues: adjustedPValues.view,
            status: {overflow: permutationOverflow.view}
          }
        },
        scratch: {
          rawRates: rawRates.view,
          summary: summary.view,
          spatialLag: spatialLag.view,
          weights: {offsets: offsets.view}
        }
      });
      session.run();
      const computation = readGpuRateClusterComputation(prepared, {
        rawRates: await session.readFloat32(rawRates),
        smoothedRates: await session.readFloat32(smoothedRates),
        summary: await session.readFloat32(summary),
        localI: await session.readFloat32(localI),
        spatialLag: await session.readFloat32(spatialLag),
        pValues: await session.readFloat32(pValues),
        adjustedPValues: await session.readFloat32(adjustedPValues),
        quadrants: await session.readUint32(quadrants),
        offsets: await session.readUint32(offsets),
        weightsOverflow: await session.readScalar(weightsOverflow),
        weightsCandidateOverflow: await session.readScalar(weightsCandidateOverflow),
        weightsRequiredCount: await session.readScalar(weightsRequiredCount),
        permutationOverflow: await session.readScalar(permutationOverflow)
      });
      const diagnostics = prepared.diagnostics;
      addCatalogDiagnostic(
        diagnostics,
        'gpu-inference-method',
        'info',
        'Public addRateClusterMapRecipe: smoothed rates, sample-variance local I, centered-rate spatial lag and doubled-smaller-tail Philox permutation p-values. CPU reference uses population-variance local I and absolute-statistic p-values; classes can differ. Islands are excluded from GPU local-statistic moments.'
      );
      if (computation.neighborCounts.some(count => count === 0))
        addCatalogDiagnostic(
          diagnostics,
          'island-features',
          'warning',
          'Areas without contiguous neighbours are not tested and are excluded from the corrected family.',
          computation.neighborCounts.filter(count => count === 0).length
        );
      if (computation.pValues.every(value => value === null))
        addCatalogDiagnostic(
          diagnostics,
          'constant-smoothed-rate',
          'warning',
          'No varying smoothed rate with neighbours is available; no area is tested.'
        );
      if (prepared.tiedAnchorCount)
        addCatalogDiagnostic(
          diagnostics,
          'row-order-dependent-inference',
          'warning',
          'Coincident anchors retain input order in the canonical GPU permutation pools; reordering can change inference throughout the dataset.',
          prepared.tiedAnchorCount
        );
      const dispatchedContributors = recipe.contributors
        .map(contributor => contributor.constructor.name)
        .filter(name => name !== 'Object');
      // Multiple-testing adjustment is composed inside GPULocalPermutationTest.
      if (values['multipleTesting'] !== 'none')
        dispatchedContributors.push('GPUMultipleTestingAdjustment');
      return {computation, diagnostics, dispatchedContributors};
    } finally {
      permutationParameters?.destroy();
      parameters?.destroy();
    }
  });
}
