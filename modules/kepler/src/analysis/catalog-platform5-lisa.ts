// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUGlobalSpatialStatistics,
  GPULocalPermutationTest,
  GPU_GLOBAL_JOIN_COUNT_FIELD,
  GPU_GLOBAL_SPATIAL_STATISTIC_FIELD,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  getGPUPermutationParameterValues,
  type GPULocalPermutationStatistic,
  type GPUSpatialWeights
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogOperationBinding, CatalogRunOutput} from './catalog-binding-types';
import {appendCatalogRow, getFeatureIdentity, makeCatalogTable} from './catalog-geojson';
import {
  CatalogGpuUnsupportedError,
  withCatalogGpuSession,
  type CatalogGpuSession
} from './catalog-gpu-session';
import {
  guardContiguityInput,
  makeWeightsParameters,
  readWeightsSettings,
  requireValueProperty
} from './catalog-platform3-b-bindings';
import {
  PERMUTATION_PARAMETER_SPEC,
  SEED_PARAMETER_SPEC,
  getFeatureGeometryText,
  makeInputSpec,
  makeNumberParameter,
  makeOutputSpec,
  makeStringParameter,
  readNumericProperty,
  type Values
} from './catalog-platform3-b-common';
import {
  buildAnalysisSet,
  getMeanAndDeviation,
  type AnalysisSet
} from './catalog-platform3-b-statistics';
import {makeAnalysisCsr} from './catalog-platform5-statistics-gpu';

const GPU_ONLY_CLEANUP =
  'GPU only. The run creates storage buffers and compiled graphs and destroys them before the result returns; results are plain arrays and publication goes through native Sources.';

const MAXIMUM_FEATURES = 3000;
const MAXIMUM_PERMUTATION_NEIGHBORS = 32;

function uploadWeights(session: CatalogGpuSession, set: AnalysisSet): GPUSpatialWeights {
  const csr = makeAnalysisCsr(set);
  return {
    offsets: session.input('weights-offsets', csr.offsets, 'uint32'),
    neighbors: session.input('weights-neighbors', csr.neighbors, 'uint32'),
    weights: session.input('weights-values', csr.weights, 'float32'),
    information: {
      rowCount: set.values.length,
      transform: 'binary',
      symmetric: 'unknown'
    }
  };
}

const GLOBAL_TABLE_COLUMNS = {
  statistic: 'string',
  value: 'float64',
  expected: 'float64',
  variance: 'float64',
  z_score: 'float64',
  p_value: 'float64'
} as const;

const finiteOrNull = (value: number) => (Number.isFinite(value) ? value : null);

async function runGlobalJoinCounts(
  context: Parameters<NonNullable<CatalogOperationBinding['runGpu']>>[0],
  inputs: Parameters<NonNullable<CatalogOperationBinding['runGpu']>>[1],
  values: Values
): Promise<CatalogRunOutput> {
  const settings = readWeightsSettings(values, false);
  guardContiguityInput(inputs, settings);
  const set = buildAnalysisSet(inputs['features'], requireValueProperty(values), settings);
  // The statistics need a few features with spread and at least one link.
  if (set.values.length < 4 || !set.weights.neighbors.some(list => list.length))
    throw new CatalogGpuUnsupportedError('Too few features or links');
  const summary = await withCatalogGpuSession(
    context.device,
    'catalog-join-counts',
    async session => {
      const output = session.output(
        'summary',
        'float32',
        GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length
      );
      const joinCounts = session.output('join-counts', 'uint32', 3);
      session.graph.add(
        new GPUGlobalSpatialStatistics({
          id: 'catalog-join-counts',
          weights: uploadWeights(session, set),
          values: session.input('values', Float32Array.from(set.values), 'float32'),
          statistics: ['getisOrdG', 'joinCount'],
          islands: 'keep',
          output: {summary: output.view, joinCounts: joinCounts.view}
        })
      );
      session.run();
      return {
        summary: await session.readFloat32(output),
        joinCounts: await session.readUint32(joinCounts)
      };
    }
  );
  const table = makeCatalogTable('global', 'Global G and join counts', GLOBAL_TABLE_COLUMNS);
  const g = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.getisOrdG;
  const field = GPU_GLOBAL_SPATIAL_STATISTIC_FIELD;
  appendCatalogRow(table, {
    statistic: 'getis-ord-g',
    value: finiteOrNull(summary.summary[g + field.statistic]),
    expected: finiteOrNull(summary.summary[g + field.expected]),
    variance: finiteOrNull(summary.summary[g + field.varianceNormality]),
    z_score: finiteOrNull(summary.summary[g + field.zNormality]),
    p_value: finiteOrNull(summary.summary[g + field.pNormality])
  });
  const j = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.joinCount;
  const join = GPU_GLOBAL_JOIN_COUNT_FIELD;
  const [blackBlack, blackWhite, whiteWhite] = summary.joinCounts;
  // Ordered-pair counts are twice the joins for symmetric weights.
  appendCatalogRow(table, {
    statistic: 'join-count-black-black',
    value: blackBlack / 2,
    expected: finiteOrNull(summary.summary[j + join.expectedBlackBlack]),
    variance: finiteOrNull(summary.summary[j + join.varianceBlackBlack]),
    z_score: finiteOrNull(summary.summary[j + join.zBlackBlack]),
    p_value: finiteOrNull(summary.summary[j + join.pBlackBlack])
  });
  appendCatalogRow(table, {
    statistic: 'join-count-black-white',
    value: blackWhite / 2,
    expected: finiteOrNull(summary.summary[j + join.expectedBlackWhite]),
    variance: finiteOrNull(summary.summary[j + join.varianceBlackWhite]),
    z_score: finiteOrNull(summary.summary[j + join.zBlackWhite]),
    p_value: finiteOrNull(summary.summary[j + join.pBlackWhite])
  });
  appendCatalogRow(table, {
    statistic: 'join-count-white-white',
    value: whiteWhite / 2,
    expected: null,
    variance: null,
    z_score: null,
    p_value: null
  });
  return {
    tables: [table],
    diagnostics: [],
    summary: {featureCount: set.values.length},
    dispatchedContributors: ['GPUGlobalSpatialStatistics']
  };
}

const VARIANTS = {
  'local-getis-ord-g': {statistic: 'localG', binary: false, bivariate: false},
  'local-getis-ord-g-star': {statistic: 'localGStar', binary: false, bivariate: false},
  'local-join-count': {statistic: 'localJoinCount', binary: true, bivariate: false},
  'local-bivariate-moran': {statistic: 'localBivariateMoran', binary: false, bivariate: true}
} as const satisfies Record<
  string,
  {statistic: GPULocalPermutationStatistic; binary: boolean; bivariate: boolean}
>;

async function runLocalLisaVariants(
  context: Parameters<NonNullable<CatalogOperationBinding['runGpu']>>[0],
  inputs: Parameters<NonNullable<CatalogOperationBinding['runGpu']>>[1],
  values: Values
): Promise<CatalogRunOutput> {
  const variant = VARIANTS[values['variant'] as keyof typeof VARIANTS];
  const permutations = values['permutations'] as number;
  const significance = values['significance'] as number;
  // Local inference is a permutation test; at least one permutation is needed.
  if (permutations < 1) throw new CatalogGpuUnsupportedError('Permutations required');
  const settings = readWeightsSettings(values, !variant.binary);
  guardContiguityInput(inputs, settings);
  const collection = inputs['features'];
  const valueProperty = requireValueProperty(values);
  const set = buildAnalysisSet(collection, valueProperty, settings);
  const rowCount = set.values.length;
  if (rowCount < 4 || getMeanAndDeviation(set.values).deviation === 0)
    throw new CatalogGpuUnsupportedError('Too few or constant values');
  if (variant.binary && set.values.some(value => value !== 0 && value !== 1))
    throw new CatalogGpuUnsupportedError('Local join counts need a 0/1 property');
  if (set.weights.neighbors.some(list => list.length > MAXIMUM_PERMUTATION_NEIGHBORS))
    throw new CatalogGpuUnsupportedError('Too many neighbors for the permutation scratch');
  let secondValues: Float32Array | undefined;
  if (variant.bivariate) {
    const property = values['secondValueProperty'] as string;
    if (!property) throw new CatalogGpuUnsupportedError('Second property required');
    secondValues = Float32Array.from(set.sourceIndex, index => {
      const value = readNumericProperty(collection.features[index], property);
      return value ?? Number.NaN;
    });
    if (secondValues.some(value => !Number.isFinite(value)))
      throw new CatalogGpuUnsupportedError('Second property has missing values');
  }
  const result = await withCatalogGpuSession(
    context.device,
    'catalog-local-lisa',
    async session => {
      const observed = session.output('observed', 'float32', rowCount);
      const pseudo = session.output('pseudo-p-values', 'float32', rowCount);
      const exceedances = session.output('exceedances', 'uint32', rowCount);
      const overflow = session.output('overflow', 'uint32', 1);
      const invalidCount = session.output('invalid-count', 'uint32', 1);
      session.graph.add(
        new GPULocalPermutationTest({
          id: 'catalog-local-lisa',
          weights: uploadWeights(session, set),
          values: session.input('values', Float32Array.from(set.values), 'float32'),
          ...(secondValues ? {secondValues: session.input('second', secondValues, 'float32')} : {}),
          statistic: variant.statistic,
          alternative: 'two-sided',
          islands: 'keep',
          parameters: session.input(
            'parameters',
            getGPUPermutationParameterValues(
              {
                seed: values['seed'] as number,
                permutationCount: permutations,
                significanceLevel: significance
              },
              {maximumPermutations: permutations}
            ),
            'uint32'
          ),
          maximumPermutations: permutations,
          output: {
            exceedances: exceedances.view,
            pseudoPValues: pseudo.view,
            observed: observed.view,
            status: {overflow: overflow.view, invalidCount: invalidCount.view}
          }
        })
      );
      session.run();
      // G and G* are defined for non-negative data; the test then tests no row and also sets overflow.
      if ((await session.readScalar(invalidCount)) !== 0)
        throw new CatalogGpuUnsupportedError(
          'Getis-Ord local G requires non-negative values (the value property has negative values)'
        );
      if ((await session.readScalar(overflow)) !== 0)
        throw new CatalogGpuUnsupportedError('Permutation test incomplete');
      return {
        observed: await session.readFloat32(observed),
        pValues: await session.readFloat32(pseudo)
      };
    }
  );
  const table = makeCatalogTable(
    'local-lisa',
    'Local association statistics',
    {
      source_index: 'int32',
      source_id: 'string',
      value: 'float64',
      neighbor_count: 'int32',
      statistic: 'float64',
      p_value: 'float64',
      significant: 'boolean',
      geometry: 'string'
    },
    'geometry'
  );
  for (let row = 0; row < rowCount; row++) {
    const pValue = result.pValues[row];
    appendCatalogRow(table, {
      source_index: set.sourceIndex[row],
      source_id: getFeatureIdentity(collection.features[set.sourceIndex[row]]),
      value: set.values[row],
      neighbor_count: set.weights.neighbors[row].length,
      statistic: finiteOrNull(result.observed[row]),
      p_value: finiteOrNull(pValue),
      significant: Number.isFinite(pValue) && pValue <= significance,
      geometry: getFeatureGeometryText(collection.features[set.sourceIndex[row]])
    });
  }
  return {
    tables: [table],
    diagnostics: [],
    summary: {featureCount: rowCount, permutations},
    dispatchedContributors: ['GPULocalPermutationTest']
  };
}

const GLOBAL_JOIN_COUNTS_BINDING: CatalogOperationBinding = {
  id: 'global-join-counts-getis-ord',
  title: 'Global Getis-Ord G and join counts',
  domain: 'analysis',
  summary:
    'Getis-Ord General G and black/white join counts of a numeric property over binary weights, with analytic z-scores and p-values.',
  capabilityIds: ['spatial-autocorrelation-and-inference'],
  contributorNames: ['GPUGlobalSpatialStatistics'],
  gpuContributorNames: ['GPUGlobalSpatialStatistics'],
  engine: 'gpu',
  inputs: [makeInputSpec('features', 'Features', ['point', 'polygon'], MAXIMUM_FEATURES, 200000)],
  parameters: [
    makeStringParameter('valueProperty', 'Value property (non-zero is black)'),
    ...makeWeightsParameters()
  ],
  outputs: [makeOutputSpec('global', 'Global statistics', 'none', ['statistic'], 4)],
  cleanup: GPU_ONLY_CLEANUP,
  declaredBehavior: [
    'Weights are binary (the neighbor rule decides the joins); the neighbor lists come from the CPU neighbor builder (exact ties, polygon contiguity).',
    'Join counts treat non-zero values as black and report undirected joins (ordered-pair counts halved); knn weights are asymmetric, so prefer queen, rook or distance-band for join counts.',
    'General G reports the normality variance, z-score and normal p-value; islands stay in n.'
  ],
  limitation:
    'GPU only (no CPU engine; the specs check the join counts against an independent CPU count). Float32 statistics; the white-white join count has no analytic inference; at most 3,000 features.',
  runGpu: runGlobalJoinCounts
};

const LOCAL_LISA_BINDING: CatalogOperationBinding = {
  id: 'local-lisa-variants',
  title: 'Local association variants',
  domain: 'analysis',
  summary:
    'Local Getis-Ord G or G*, local join counts or local bivariate Moran with conditional-permutation pseudo p-values.',
  capabilityIds: ['spatial-autocorrelation-and-inference'],
  contributorNames: ['GPULocalPermutationTest'],
  gpuContributorNames: ['GPULocalPermutationTest'],
  engine: 'gpu',
  inputs: [makeInputSpec('features', 'Features', ['point', 'polygon'], MAXIMUM_FEATURES, 200000)],
  parameters: [
    {
      name: 'variant',
      label: 'Variant',
      kind: 'enum',
      options: Object.keys(VARIANTS),
      defaultValue: 'local-getis-ord-g-star'
    },
    makeStringParameter('valueProperty', 'Value property'),
    makeStringParameter('secondValueProperty', 'Second property (bivariate Moran)'),
    ...makeWeightsParameters(),
    {...PERMUTATION_PARAMETER_SPEC, minimum: 1, defaultValue: 199},
    SEED_PARAMETER_SPEC,
    makeNumberParameter('significance', 'Significance level', 0.05, 1e-6, 1)
  ],
  outputs: [
    makeOutputSpec(
      'local-lisa',
      'Local association',
      'none',
      ['source_index', 'source_id'],
      MAXIMUM_FEATURES
    )
  ],
  cleanup: GPU_ONLY_CLEANUP,
  declaredBehavior: [
    'G and G* and the bivariate Moran use row-standardized weights; local join counts use binary weights and need a 0/1 property.',
    'p-values are two-sided conditional-permutation pseudo p-values from GPULocalPermutationTest (a Philox stream keyed by seed), with no multiple-testing correction; rows with more than 32 neighbors fall back (the operation then fails: there is no CPU engine).',
    'Islands stay in the moments and are not tested (null p-value).',
    'Local G and G* need non-negative values; a negative value fails the operation with a CatalogGpuUnsupportedError (there is no CPU engine to fall back to).'
  ],
  limitation:
    'GPU only (no CPU engine; the specs check the statistics against independent CPU formulas). Float32 statistics; at most 3,000 features.',
  runGpu: runLocalLisaVariants
};

export const PLATFORM5_LISA_BINDINGS: readonly CatalogOperationBinding[] = [
  GLOBAL_JOIN_COUNTS_BINDING,
  LOCAL_LISA_BINDING
];
