// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUEnsembleScores,
  GPUGaussianPredictiveScores,
  GPURasterFalseDiscoveryRateMask,
  getGPURasterFalseDiscoveryRateMaskParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {withCatalogGpuSession} from './catalog-gpu-session';
import {makeInputSpec, makeNumberParameter, makeOutputSpec} from './catalog-platform3-b-common';
import {makeRecipeEnumParameter} from './catalog-platform5-recipe-common';
import {
  makeValidationBinding,
  makeValidationProperty,
  makeValidationPropertyList,
  makeValidationResult,
  MAXIMUM_VALIDATION_ROWS,
  readValidationColumn,
  readValidationNames,
  VALIDATION_BEHAVIOR
} from './catalog-validation-common';

const SCORE_BINDINGS = [false, true].map(ensemble => {
  const id = ensemble ? 'validation-ensemble-scores' : 'validation-gaussian-scores';
  const contributor = ensemble ? 'GPUEnsembleScores' : 'GPUGaussianPredictiveScores';
  return makeValidationBinding({
    id,
    title: ensemble ? 'Ensemble predictive scores' : 'Gaussian predictive scores',
    domain: 'analysis',
    summary: ensemble
      ? 'Scores member forecasts with per-site CRPS, whole-field energy and variogram scores.'
      : 'Scores predictive means and variances with Gaussian CRPS per site.',
    capabilityIds: ['coverage-local-prediction'],
    contributorNames: [contributor],
    inputs: [
      makeInputSpec(
        'points',
        'Observed forecasts',
        ['point'],
        MAXIMUM_VALIDATION_ROWS,
        MAXIMUM_VALIDATION_ROWS
      )
    ],
    parameters: [
      makeValidationProperty('observationProperty', 'Observed property', 'observed'),
      ...(ensemble
        ? [
            makeValidationPropertyList(
              'memberProperties',
              'Ensemble member properties (comma separated)',
              'member1,member2,member3',
              64
            ),
            {
              name: 'fair',
              label: 'Fair ensemble CRPS and energy score',
              kind: 'boolean' as const,
              defaultValue: false
            },
            makeNumberParameter('variogramOrder', 'Variogram score order', 0.5, 0.000001, 10)
          ]
        : [
            makeValidationProperty('meanProperty', 'Predictive mean property', 'predicted'),
            makeValidationProperty('varianceProperty', 'Predictive variance property', 'variance')
          ])
    ],
    outputs: [
      makeOutputSpec(id, 'Predictive scores', 'point', ['feature_index'], MAXIMUM_VALIDATION_ROWS),
      makeOutputSpec(`${id}-diagnostics`, 'Predictive scores diagnostics', 'none', ['metric'], 16)
    ],
    declaredBehavior: [
      ...VALIDATION_BEHAVIOR,
      'Lower CRPS means a better forecast. Negative or missing Gaussian variances produce null scores; zero variance is a point forecast.'
    ],
    limitation: ensemble
      ? 'At most 64 member columns. Variogram score uses both directions of consecutive input-row site pairs with equal weights; row order defines that pair list. All members must describe the same joint field.'
      : 'Gaussian predictive distributions only. Forecasts and observations must be independent of model training to measure predictive performance.',
    runGpu: async ({device}, inputs, values) =>
      withCatalogGpuSession(device, id, async session => {
        const collection = inputs['points'];
        const count = collection.features.length;
        const observations = session.input(
          'observations',
          readValidationColumn(collection, values['observationProperty'] as string),
          'float32'
        );
        const crps = session.output('crps', 'float32', count);
        const energy = session.output('energy', 'float32', 1);
        const variogram = session.output('variogram', 'float32', 1);
        let memberCount = 0;
        if (ensemble) {
          const names = readValidationNames(values['memberProperties'], 64);
          memberCount = names.length;
          const members = Float32Array.from(
            names.flatMap(name => Array.from(readValidationColumn(collection, name)))
          );
          const pairs = Uint32Array.from(
            Array.from({length: Math.max(0, count - 1)}, (_, index) => [
              index,
              index + 1,
              index + 1,
              index
            ]).flat()
          );
          session.graph.add(
            new GPUEnsembleScores({
              ensemble: session.input('ensemble', members, 'float32'),
              observations,
              memberCount,
              dimension: count,
              fair: values['fair'] as boolean,
              variogramOrder: values['variogramOrder'] as number,
              ...(pairs.length ? {variogramPairs: session.input('pairs', pairs, 'uint32x2')} : {}),
              output: {
                crps: crps.view,
                energyScore: energy.view,
                ...(pairs.length ? {variogramScore: variogram.view} : {})
              }
            })
          );
        } else
          session.graph.add(
            new GPUGaussianPredictiveScores({
              means: session.input(
                'means',
                readValidationColumn(collection, values['meanProperty'] as string),
                'float32'
              ),
              variances: session.input(
                'variances',
                readValidationColumn(collection, values['varianceProperty'] as string),
                'float32'
              ),
              observations,
              output: {crps: crps.view}
            })
          );
        session.run();
        const scores = await session.readFloat32(crps);
        const finiteScores = scores.filter(Number.isFinite);
        return makeValidationResult(
          id,
          'Predictive scores',
          collection,
          {crps: scores},
          {
            scored_rows: finiteScores.length,
            invalid_rows: count - finiteScores.length,
            mean_crps: finiteScores.length
              ? finiteScores.reduce((sum, value) => sum + value, 0) / finiteScores.length
              : NaN,
            ...(ensemble
              ? {
                  member_count: memberCount,
                  energy_score: (await session.readFloat32(energy))[0],
                  variogram_score: count > 1 ? (await session.readFloat32(variogram))[0] : NaN
                }
              : {})
          },
          [contributor]
        );
      })
  });
});

const FDR_BINDING = makeValidationBinding({
  id: 'validation-fdr',
  title: 'False discovery rate mask',
  domain: 'analysis',
  summary:
    'Adjusts a declared family of point p-values and publishes significance, adjusted p-values and family diagnostics.',
  capabilityIds: ['spatial-autocorrelation-and-inference'],
  contributorNames: ['GPURasterFalseDiscoveryRateMask'],
  inputs: [
    makeInputSpec(
      'points',
      'Tests (one point per test or raster cell center)',
      ['point'],
      MAXIMUM_VALIDATION_ROWS,
      MAXIMUM_VALIDATION_ROWS
    )
  ],
  parameters: [
    makeValidationProperty('pValueProperty', 'P-value property', 'pvalue'),
    makeNumberParameter('alpha', 'Significance alpha', 0.05, 0.000001, 1),
    makeRecipeEnumParameter(
      'method',
      'Multiple testing method',
      ['benjamini-hochberg', 'benjamini-yekutieli', 'bonferroni', 'holm'],
      'benjamini-yekutieli'
    )
  ],
  outputs: [
    makeOutputSpec(
      'validation-fdr',
      'False discovery rate mask',
      'point',
      ['feature_index'],
      MAXIMUM_VALIDATION_ROWS
    ),
    makeOutputSpec(
      'validation-fdr-diagnostics',
      'False discovery rate mask diagnostics',
      'none',
      ['metric'],
      16
    )
  ],
  declaredBehavior: [
    ...VALIDATION_BEHAVIOR,
    'Finite p-values in [0, 1] enter the family; invalid rows are untested, adjusted p-value null and significant = 0. The selected input is the full testing family.'
  ],
  limitation:
    'Raster cells must be supplied as point features at cell centers; this catalog operation does not read native raster tiles. Benjamini-Yekutieli is the default for dependent tests. Holm and Bonferroni control family-wise error.',
  runGpu: async ({device}, inputs, values) =>
    withCatalogGpuSession(device, 'catalog-validation-fdr', async session => {
      const count = inputs['points'].features.length;
      const significant = session.output('significant', 'uint32', count);
      const adjusted = session.output('adjusted', 'float32', count);
      const testedCount = session.output('tested-count', 'uint32', 1);
      const significantCount = session.output('significant-count', 'uint32', 1);
      const critical = session.output('critical', 'float32', 1);
      session.graph.add(
        new GPURasterFalseDiscoveryRateMask({
          cellCount: count,
          pValues: session.input(
            'pvalues',
            readValidationColumn(inputs['points'], values['pValueProperty'] as string),
            'float32'
          ),
          parameters: session.input(
            'parameters',
            getGPURasterFalseDiscoveryRateMaskParameterValues({alpha: values['alpha'] as number}),
            'float32'
          ),
          method: values['method'] as
            | 'benjamini-hochberg'
            | 'benjamini-yekutieli'
            | 'bonferroni'
            | 'holm',
          output: {
            significant: significant.view,
            adjustedPValues: adjusted.view,
            testedCount: testedCount.view,
            significantCount: significantCount.view,
            criticalPValue: critical.view
          }
        })
      );
      session.run();
      return makeValidationResult(
        'validation-fdr',
        'False discovery rate mask',
        inputs['points'],
        {
          adjusted_pvalue: await session.readFloat32(adjusted),
          significant: await session.readUint32(significant)
        },
        {
          tested_count: await session.readScalar(testedCount),
          significant_count: await session.readScalar(significantCount),
          critical_pvalue: (await session.readFloat32(critical))[0],
          alpha: values['alpha'] as number
        },
        ['GPURasterFalseDiscoveryRateMask']
      );
    })
});

export const VALIDATION_SCORE_BINDINGS = [...SCORE_BINDINGS, FDR_BINDING];
