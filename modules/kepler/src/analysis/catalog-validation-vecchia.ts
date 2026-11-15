// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  GPUVecchiaLogLikelihood,
  GPUVecchiaFisherScoring,
  GPUVecchiaSpaceTimeLogLikelihood,
  buildVecchiaConditioningSets,
  fitVecchiaCovarianceFisherScoring,
  maximizeVecchiaLogLikelihood,
  getVecchiaFisherEvaluation,
  getVecchiaLogLikelihoodResult,
  getGPUVecchiaLogLikelihoodParameterValues,
  getGPUVecchiaSpaceTimeLogLikelihoodParameterValues,
  getVecchiaNuggetFloor,
  VECCHIA_NUGGET_FLOOR_FRACTION,
  type GPUVecchiaCovariance,
  type VecchiaConditioningSets,
  type SpaceTimeVariogramModel
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {COORDINATE_SYSTEM_PARAMETER} from './catalog-binding-types';
import {withCatalogGpuSession} from './catalog-gpu-session';
import {makeInputSpec, makeNumberParameter, makeOutputSpec} from './catalog-platform3-b-common';
import {makeRecipeEnumParameter} from './catalog-platform5-recipe-common';
import {
  getValidationTemporalContext,
  makeValidationBinding,
  makeValidationProperty,
  makeValidationResult,
  MAXIMUM_VALIDATION_ROWS,
  readValidationColumn,
  readValidationPositions,
  readValidationTimes,
  VALIDATION_BEHAVIOR
} from './catalog-validation-common';
import {VALIDATION_TIME_PARAMETERS} from './catalog-validation-regression';

type VecchiaData = {
  positions: Float32Array;
  observations: Float32Array;
  neighborhoods: VecchiaConditioningSets;
  neighborCount: number;
  times?: Float32Array;
  temporalContext?: ReturnType<typeof getValidationTemporalContext>;
  anisotropy?: number;
};

export function makeValidationSpaceTimeCovariance(
  covariance: GPUVecchiaCovariance,
  anisotropy: number
): SpaceTimeVariogramModel {
  return {type: 'metric', joint: {...covariance, nugget: covariance.nugget ?? 0}, anisotropy};
}

/** Reject covariance candidates that become invalid in the contributor's Float32 parameter buffer. */
export function isValidationVecchiaCovarianceUsable(covariance: GPUVecchiaCovariance): boolean {
  const range = Math.fround(covariance.range);
  const sill = Math.fround(covariance.sill);
  const nugget = Math.fround(covariance.nugget ?? 0);
  const totalSill = Math.fround(sill + nugget);
  return (
    covariance.sill >= 0 &&
    (covariance.nugget ?? 0) >= 0 &&
    range > 0 &&
    sill >= 0 &&
    nugget >= 0 &&
    totalSill > 0 &&
    [range, sill, nugget, totalSill].every(Number.isFinite)
  );
}

/** Use the contributor's own packer for model-specific admissibility before allocating GPU resources. */
function getValidationVecchiaParameterValues(
  data: VecchiaData,
  covariance: GPUVecchiaCovariance,
  fisher: boolean
): Float32Array {
  return !fisher && data.times && data.temporalContext
    ? getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(
        makeValidationSpaceTimeCovariance(covariance, data.anisotropy ?? 1)
      )
    : getGPUVecchiaLogLikelihoodParameterValues(covariance);
}

/** Thin graph adapter shared with the ordinary-kriging variogram fitting UI. */
export async function evaluateValidationVecchia(
  device: Device,
  data: VecchiaData,
  covariance: GPUVecchiaCovariance,
  fisher = false
) {
  if (!isValidationVecchiaCovarianceUsable(covariance))
    throw new Error(
      'Vecchia evaluation requires a finite positive range and total sill at Float32 precision'
    );
  const parameterValues = getValidationVecchiaParameterValues(data, covariance, fisher);
  return withCatalogGpuSession(device, 'catalog-validation-vecchia', async session => {
    const count = data.observations.length / 4;
    const common = {
      positions: session.input('positions', data.positions, 'float32x2'),
      observations: session.input('observations', data.observations, 'float32x4'),
      neighborhoodOffsets: session.input('offsets', data.neighborhoods.offsets, 'uint32'),
      neighborhoods: session.input('neighbors', data.neighborhoods.neighbors, 'uint32'),
      neighborCount: data.neighborCount
    };
    const siteTerms = session.output('site-terms', 'float32', count * (fisher ? 10 : 1));
    const siteStatus = session.output('site-status', 'uint32', count);
    const status = session.output('status', 'uint32', 3);
    const totals = session.output('totals', 'float32', fisher ? 20 : 2);
    const conditionals = session.output('conditionals', 'float32x4', count);
    const conditionalStatus = session.output('conditional-status', 'uint32', count);
    if (fisher)
      session.graph.add(
        new GPUVecchiaFisherScoring({
          ...common,
          parameters: session.input('parameters', parameterValues, 'float32'),
          output: {
            siteTerms: siteTerms.view,
            siteStatus: siteStatus.view,
            totals: totals.view,
            status: status.view
          }
        })
      );
    else {
      const output = {
        siteTerms: siteTerms.view,
        siteStatus: siteStatus.view,
        logLikelihood: totals.view,
        status: status.view,
        conditionals: conditionals.view,
        conditionalStatus: conditionalStatus.view
      };
      if (data.times && data.temporalContext)
        session.graph.add(
          new GPUVecchiaSpaceTimeLogLikelihood({
            ...common,
            temporalContext: data.temporalContext,
            times: session.input('times', data.times, 'float32'),
            parameters: session.input('parameters', parameterValues, 'float32'),
            output
          })
        );
      else
        session.graph.add(
          new GPUVecchiaLogLikelihood({
            ...common,
            parameters: session.input('parameters', parameterValues, 'float32'),
            output
          })
        );
    }
    session.run();
    const totalValues = await session.readFloat32(totals);
    const statusValues = await session.readUint32(status);
    const evaluation = fisher
      ? getVecchiaFisherEvaluation(totalValues, statusValues)
      : {
          ...getVecchiaLogLikelihoodResult(totalValues, statusValues),
          score: [] as number[],
          information: [] as number[]
        };
    return {
      ...evaluation,
      siteTerms: await session.readFloat32(siteTerms),
      siteStatus: await session.readUint32(siteStatus),
      conditionals: fisher ? undefined : await session.readFloat32(conditionals),
      conditionalStatus: fisher ? undefined : await session.readUint32(conditionalStatus)
    };
  });
}

/** Uses the library optimizers; no fitting mathematics lives in the host adapter. */
export async function fitValidationVecchia(
  device: Device,
  data: VecchiaData,
  initial: GPUVecchiaCovariance,
  method: 'likelihood' | 'fisher',
  maximumIterations: number
) {
  if (!Number.isInteger(maximumIterations) || maximumIterations < 4 || maximumIterations > 128)
    throw new Error('Vecchia fitting maximumIterations must be 4 to 128');
  if (
    !(initial.sill > 0) ||
    !Number.isFinite(initial.sill) ||
    !(initial.range > 0) ||
    !Number.isFinite(initial.range)
  )
    throw new Error('Vecchia fitting initial sill and range must be positive and finite');
  const temporal = Boolean(data.times && data.temporalContext);
  const getCandidateNuggetFloor = (
    covariance: Pick<GPUVecchiaCovariance, 'model' | 'sill'>
  ): number => {
    const spatialFloor = getVecchiaNuggetFloor(covariance);
    // Temporal Matern requires n >= f * (s + n), hence n >= f / (1 - f) * s.
    return temporal ? spatialFloor / (1 - VECCHIA_NUGGET_FLOOR_FRACTION) : spatialFloor;
  };
  const isCandidateAdmissible = (covariance: GPUVecchiaCovariance): boolean => {
    if (!isValidationVecchiaCovarianceUsable(covariance)) return false;
    try {
      getValidationVecchiaParameterValues(data, covariance, method === 'fisher');
      return true;
    } catch {
      // A rejected parameter proposal is an optimizer penalty, not a failed fit run.
      return false;
    }
  };
  if (method === 'fisher') {
    const fit = await fitVecchiaCovarianceFisherScoring({
      initial,
      maximumIterations,
      evaluate: covariance =>
        isCandidateAdmissible(covariance)
          ? evaluateValidationVecchia(device, data, covariance, true)
          : {
              logLikelihood: NaN,
              score: [NaN, NaN, NaN],
              information: [NaN, NaN, NaN, NaN, NaN, NaN],
              failedSiteCount: 1
            }
    });
    return {
      covariance: fit.covariance,
      converged: fit.converged,
      budgetUsed: fit.iterationCount,
      failed: fit.failed
    };
  }
  const makeCovariance = (parameters: readonly number[]): GPUVecchiaCovariance => {
    const sill = Math.exp(parameters[0]);
    return {
      model: initial.model,
      sill,
      range: Math.exp(parameters[1]),
      nugget: Math.max(
        Math.exp(parameters[2]),
        getCandidateNuggetFloor({model: initial.model, sill})
      )
    };
  };
  const fit = await maximizeVecchiaLogLikelihood({
    initial: [
      Math.log(initial.sill),
      Math.log(initial.range),
      Math.log(Math.max(initial.nugget ?? 0, initial.sill * 1e-3))
    ],
    maximumEvaluations: maximumIterations,
    evaluate: async parameters => {
      const covariance = makeCovariance(parameters);
      if (!isCandidateAdmissible(covariance)) return NaN;
      return (await evaluateValidationVecchia(device, data, covariance)).logLikelihood;
    }
  });
  return {
    covariance: makeCovariance(fit.values),
    converged: fit.converged,
    budgetUsed: fit.evaluationCount,
    failed: !Number.isFinite(fit.logLikelihood)
  };
}

export const VALIDATION_COVARIANCE_PARAMETERS = [
  makeRecipeEnumParameter(
    'model',
    'Covariance model',
    ['exponential', 'matern-3/2'],
    'exponential'
  ),
  makeNumberParameter('sill', 'Partial sill', 1, 1e-6, 1e12),
  makeNumberParameter('nugget', 'Nugget', 0.05, 0, 1e12),
  makeNumberParameter('range', 'Effective range', 1000, 1e-6, 1e9, 'crs'),
  {
    ...makeNumberParameter('neighborCount', 'Conditioning neighbors', 8, 1, 32),
    kind: 'integer' as const
  }
];

export const VALIDATION_VECCHIA_BINDINGS = ['likelihood', 'fisher', 'space-time'].map(kind => {
  const spaceTime = kind === 'space-time';
  const fisher = kind === 'fisher';
  const id = `validation-vecchia-${kind}`;
  const contributor = spaceTime
    ? 'GPUVecchiaSpaceTimeLogLikelihood'
    : fisher
      ? 'GPUVecchiaFisherScoring'
      : 'GPUVecchiaLogLikelihood';
  return makeValidationBinding({
    id,
    title: spaceTime
      ? 'Space-time Vecchia covariance fit'
      : fisher
        ? 'Vecchia Fisher scoring covariance fit'
        : 'Vecchia log-likelihood covariance fit',
    domain: 'interpolation',
    summary:
      'Evaluates or fits a covariance with ordered nearest-neighbor conditional likelihoods, reporting every site status and total diagnostics.',
    capabilityIds: ['coverage-local-prediction'],
    contributorNames: [contributor],
    inputs: [
      makeInputSpec(
        'points',
        'Observations',
        ['point'],
        MAXIMUM_VALIDATION_ROWS,
        MAXIMUM_VALIDATION_ROWS
      )
    ],
    parameters: [
      makeValidationProperty('valueProperty', 'Observed property', 'observed'),
      makeNumberParameter('mean', 'Known response mean (subtract before fitting)', 0, -1e12, 1e12),
      makeValidationProperty(
        'errorVarianceProperty',
        'Observation error variance property (blank: zero)',
        ''
      ),
      ...VALIDATION_COVARIANCE_PARAMETERS,
      {
        name: 'fit',
        label: 'Fit covariance parameters',
        kind: 'boolean' as const,
        defaultValue: false
      },
      {
        ...makeNumberParameter(
          'maximumIterations',
          fisher ? 'Maximum Fisher iterations' : 'Maximum likelihood evaluations',
          fisher ? 8 : 32,
          4,
          128
        ),
        kind: 'integer' as const
      },
      ...(spaceTime
        ? [
            ...VALIDATION_TIME_PARAMETERS,
            makeNumberParameter(
              'anisotropy',
              'Space-time anisotropy (spatial units per second)',
              1,
              1e-6,
              1e9
            )
          ]
        : []),
      COORDINATE_SYSTEM_PARAMETER
    ],
    outputs: [
      makeOutputSpec(id, 'Vecchia sites', 'point', ['feature_index'], MAXIMUM_VALIDATION_ROWS),
      makeOutputSpec(`${id}-diagnostics`, 'Vecchia sites diagnostics', 'none', ['metric'], 32)
    ],
    declaredBehavior: [
      ...VALIDATION_BEHAVIOR,
      'The known mean is subtracted before Float32 rounding. Input order is the Vecchia ordering; conditioning sets are built once from earlier usable rows. A failed site invalidates the total likelihood. Site statuses: 0 used, 1 excluded, 2 solve failed, 3 invalid conditional variance.'
    ],
    limitation:
      'Exponential or Matern 3/2 covariance only. Space-time uses a metric joint covariance with fixed anisotropy. Fit uses the library Fisher scoring or Nelder-Mead loop with bounded work; convergence is reported, not assumed. This is an approximate likelihood for a supplied mean, not a trend estimator.',
    runGpu: async ({device}, inputs, values) => {
      const collection = inputs['points'];
      const [positions] = readValidationPositions([collection], values);
      const response = readValidationColumn(collection, values['valueProperty'] as string);
      const errorVariances = values['errorVarianceProperty']
        ? readValidationColumn(collection, values['errorVarianceProperty'] as string)
        : new Float32Array(response.length);
      const times = spaceTime ? readValidationTimes(collection, values) : undefined;
      const observations = new Float32Array(response.length * 4);
      const validity = new Uint32Array(response.length);
      collection.features.forEach((feature, index) => {
        const value = feature.properties?.[values['valueProperty'] as string];
        const residual =
          typeof value === 'number' ? Math.fround(value - (values['mean'] as number)) : NaN;
        validity[index] =
          Number.isFinite(residual) &&
          Number.isFinite(errorVariances[index]) &&
          errorVariances[index] >= 0 &&
          (!times || Number.isFinite(times[index]))
            ? 1
            : 0;
        observations.set([residual, errorVariances[index], 0, validity[index]], index * 4);
      });
      const neighborCount = values['neighborCount'] as number;
      const data: VecchiaData = {
        positions,
        observations,
        neighborCount,
        neighborhoods: buildVecchiaConditioningSets({
          positions,
          neighborCount,
          validity,
          ...(times ? {times, timeScale: values['anisotropy'] as number} : {})
        }),
        ...(times
          ? {
              times,
              temporalContext: getValidationTemporalContext(values),
              anisotropy: values['anisotropy'] as number
            }
          : {})
      };
      let covariance: GPUVecchiaCovariance = {
        model: values['model'] as 'exponential' | 'matern-3/2',
        sill: values['sill'] as number,
        nugget: values['nugget'] as number,
        range: values['range'] as number
      };
      let convergence = {converged: false, budgetUsed: 0, failed: false};
      if (values['fit']) {
        const fit = await fitValidationVecchia(
          device,
          data,
          covariance,
          fisher ? 'fisher' : 'likelihood',
          values['maximumIterations'] as number
        );
        covariance = fit.covariance;
        convergence = fit;
      }
      const evaluation = await evaluateValidationVecchia(device, data, covariance, fisher);
      const count = response.length;
      const columns: Record<string, ArrayLike<number>> = {
        site_log_likelihood: Float32Array.from(
          {length: count},
          (_, row) => evaluation.siteTerms[row * (fisher ? 10 : 1)]
        ),
        status: evaluation.siteStatus
      };
      if (fisher)
        [
          'score_log_sill',
          'score_log_range',
          'score_log_nugget',
          'information_00',
          'information_01',
          'information_02',
          'information_11',
          'information_12',
          'information_22'
        ].forEach((name, index) => {
          columns[name] = Float32Array.from(
            {length: count},
            (_, row) => evaluation.siteTerms[row * 10 + index + 1]
          );
        });
      else if (evaluation.conditionals) {
        const conditionals = evaluation.conditionals;
        ['conditional_mean', 'conditional_variance', 'pivot_ratio', 'retained_neighbors'].forEach(
          (name, index) => {
            columns[name] = Float32Array.from(
              {length: count},
              (_, row) => conditionals[row * 4 + index]
            );
          }
        );
        columns['conditional_status'] = evaluation.conditionalStatus!;
      }
      return makeValidationResult(
        id,
        'Vecchia sites',
        collection,
        columns,
        {
          log_likelihood: evaluation.logLikelihood,
          failed_sites: evaluation.failedSiteCount,
          excluded_sites: evaluation.excludedSiteCount,
          used_sites: evaluation.usedSiteCount,
          sill: covariance.sill,
          nugget: covariance.nugget ?? 0,
          range: covariance.range,
          ...(spaceTime
            ? {
                anisotropy: values['anisotropy'] as number,
                time_origin: values['timeOrigin'] as number
              }
            : {}),
          fit_requested: values['fit'] ? 1 : 0,
          converged: Number(convergence.converged),
          budget_used: convergence.budgetUsed,
          fit_failed: Number(convergence.failed),
          ...Object.fromEntries(evaluation.score.map((value, index) => [`score_${index}`, value])),
          ...Object.fromEntries(
            evaluation.information.map((value, index) => [`information_${index}`, value])
          )
        },
        [contributor]
      );
    }
  });
});
