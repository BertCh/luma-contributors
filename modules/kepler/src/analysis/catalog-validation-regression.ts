// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUGeographicallyTemporallyWeightedRegression,
  GPUMultiscaleGeographicallyTemporallyWeightedRegression,
  getGPUGTWRParameterValues,
  getGPUMGTWRParameterValues,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {COORDINATE_SYSTEM_PARAMETER} from './catalog-binding-types';
import {withCatalogGpuSession} from './catalog-gpu-session';
import {makeInputSpec, makeNumberParameter, makeOutputSpec} from './catalog-platform3-b-common';
import {makeRecipeEnumParameter} from './catalog-platform5-recipe-common';
import {
  getValidationTemporalContext,
  makeValidationBinding,
  makeValidationProperty,
  makeValidationPropertyList,
  makeValidationResult,
  MAXIMUM_VALIDATION_ROWS,
  readValidationColumn,
  readValidationMatrix,
  readValidationNames,
  readValidationPositions,
  readValidationTimes,
  VALIDATION_BEHAVIOR
} from './catalog-validation-common';

export const VALIDATION_TIME_PARAMETERS = [
  makeValidationProperty('timeProperty', 'Time property (seconds)', 'time'),
  {
    ...makeNumberParameter('timeOrigin', 'Time origin (Unix seconds)', 0, -8e12, 8e12),
    kind: 'integer' as const
  }
];

export const VALIDATION_REGRESSION_BINDINGS = [false, true].map(multiscale => {
  const id = multiscale ? 'regression-mgtwr' : 'regression-gtwr';
  const contributor = multiscale
    ? 'GPUMultiscaleGeographicallyTemporallyWeightedRegression'
    : 'GPUGeographicallyTemporallyWeightedRegression';
  return makeValidationBinding({
    id,
    title: multiscale
      ? 'Multiscale geographically temporally weighted regression'
      : 'Geographically temporally weighted regression',
    domain: 'analysis',
    summary:
      'Fits local space-time regressions and publishes coefficients, fitted values, residuals and numerical diagnostics.',
    contributorNames: [contributor],
    capabilityIds: ['expansion-regression'],
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
      makeValidationProperty('responseProperty', 'Response property', 'observed'),
      makeValidationPropertyList(
        'predictorProperties',
        'Predictor properties (comma separated)',
        'predictor',
        11
      ),
      ...VALIDATION_TIME_PARAMETERS,
      makeRecipeEnumParameter('kernel', 'Kernel', ['gaussian', 'bisquare'], 'gaussian'),
      makeNumberParameter('bandwidth', 'Bandwidth', 1000, 1e-6, 1e9, 'crs'),
      makeNumberParameter(
        'timeScale',
        'Time scale (squared spatial units per squared second)',
        1,
        0,
        1e12
      ),
      ...(multiscale
        ? [
            {
              name: 'componentBandwidths',
              label: 'Component bandwidths (intercept first)',
              kind: 'number-list' as const,
              defaultValue: [1000, 1000],
              minimum: 1e-6,
              maximum: 1e9,
              maximumLength: 12
            },
            {
              name: 'componentTimeScales',
              label: 'Component time scales (intercept first)',
              kind: 'number-list' as const,
              defaultValue: [1, 1],
              minimum: 0,
              maximum: 1e12,
              maximumLength: 12
            },
            {
              ...makeNumberParameter('iterationCount', 'Backfitting iterations', 4, 1, 32),
              kind: 'integer' as const
            }
          ]
        : []),
      COORDINATE_SYSTEM_PARAMETER
    ],
    outputs: [
      makeOutputSpec(
        id,
        'Space-time regression',
        'point',
        ['feature_index'],
        MAXIMUM_VALIDATION_ROWS
      ),
      makeOutputSpec(
        `${id}-diagnostics`,
        'Space-time regression diagnostics',
        'none',
        ['metric'],
        64
      )
    ],
    declaredBehavior: [
      ...VALIDATION_BEHAVIOR,
      'Times are numeric seconds; the integer origin is subtracted before Float32 rounding. Fixed bandwidths are used; singular and excluded rows retain their status.'
    ],
    limitation:
      'Exhaustive fixed-bandwidth fits. Automatic bandwidth search and adaptive kernels are not enabled. Multiscale convergence scores describe every requested backfitting iteration; they do not trigger early stopping.',
    runGpu: async ({device}, inputs, values) => {
      const collection = inputs['points'];
      const names = readValidationNames(values['predictorProperties']);
      if (names.includes('intercept'))
        throw new Error(
          `${id} predictor property "intercept" is reserved for the fitted intercept`
        );
      const count = collection.features.length;
      const [positions] = readValidationPositions([collection], values);
      const settings = {
        bandwidth: values['bandwidth'] as number,
        timeScale: values['timeScale'] as number
      };
      const componentBandwidths = values['componentBandwidths'] as readonly number[];
      const componentTimeScales = values['componentTimeScales'] as readonly number[];
      if (
        multiscale &&
        (componentBandwidths.length !== names.length + 1 ||
          componentTimeScales.length !== names.length + 1)
      )
        throw new Error(
          `${id} component bandwidths and time scales require one value per coefficient, intercept first`
        );
      return withCatalogGpuSession(device, id, async session => {
        const coefficients = session.output('coefficients', 'float32', count * (names.length + 1));
        const fitted = session.output('fitted', 'float32', count);
        const residuals = session.output('residuals', 'float32', count);
        const status = session.output('status', 'uint32', count);
        const summary = session.output(
          'summary',
          'float32',
          GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH
        );
        const scores = session.output(
          'scores',
          'float32',
          multiscale ? (values['iterationCount'] as number) : 1
        );
        const common = {
          id,
          positions: session.input('positions', positions, 'float32x2'),
          times: session.input('times', readValidationTimes(collection, values), 'float32'),
          temporalContext: getValidationTemporalContext(values),
          predictors: session.input(
            'predictors',
            readValidationMatrix(collection, names),
            'float32'
          ),
          predictorCount: names.length,
          response: session.input(
            'response',
            readValidationColumn(collection, values['responseProperty'] as string),
            'float32'
          ),
          kernel: values['kernel'] as 'gaussian' | 'bisquare'
        };
        if (multiscale)
          session.graph.add(
            new GPUMultiscaleGeographicallyTemporallyWeightedRegression({
              ...common,
              iterationCount: values['iterationCount'] as number,
              parameters: session.input(
                'parameters',
                getGPUMGTWRParameterValues({
                  initial: settings,
                  components: componentBandwidths.map((bandwidth, index) => ({
                    bandwidth,
                    timeScale: componentTimeScales[index]
                  }))
                }),
                'float32'
              ),
              output: {
                coefficients: coefficients.view,
                fitted: fitted.view,
                residuals: residuals.view,
                localStatus: status.view,
                scores: scores.view
              }
            })
          );
        else
          session.graph.add(
            new GPUGeographicallyTemporallyWeightedRegression({
              ...common,
              parameters: session.input(
                'parameters',
                getGPUGTWRParameterValues(settings),
                'float32'
              ),
              output: {
                coefficients: coefficients.view,
                fitted: fitted.view,
                residuals: residuals.view,
                localStatus: status.view,
                summary: summary.view,
                hatDiagonal: session.output('hat', 'float32', count).view
              }
            })
          );
        session.run();
        const coefficientValues = await session.readFloat32(coefficients);
        const statusValues = await session.readUint32(status);
        const metrics: Record<string, number> = {
          bandwidth: settings.bandwidth,
          time_scale: settings.timeScale,
          time_origin: values['timeOrigin'] as number,
          singular_rows: statusValues.filter(value => value === 1).length,
          excluded_rows: statusValues.filter(value => value === 2).length
        };
        if (multiscale)
          (await session.readFloat32(scores)).forEach((score, index) => {
            metrics[`backfit_score_${index + 1}`] = score;
          });
        else
          (await session.readFloat32(summary)).forEach((value, index) => {
            metrics[
              [
                'residual_sum_squares',
                'hat_trace',
                'aicc',
                'r_squared',
                'observation_count',
                'has_valid_candidate',
                'singular_count',
                'sigma_squared',
                'adjusted_alpha',
                'critical_t',
                'log_likelihood',
                'included_count'
              ][index] ?? `summary_${index}`
            ] = value;
          });
        return makeValidationResult(
          id,
          'Space-time regression',
          collection,
          {
            ...Object.fromEntries(
              ['intercept', ...names].map((name, column) => [
                `coef_${name}`,
                Float32Array.from(
                  {length: count},
                  (_, row) => coefficientValues[row * (names.length + 1) + column]
                )
              ])
            ),
            fitted: await session.readFloat32(fitted),
            residual: await session.readFloat32(residuals),
            status: statusValues
          },
          metrics,
          [contributor]
        );
      });
    }
  });
});
