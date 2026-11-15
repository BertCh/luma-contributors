// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUSpaceTimeKriging,
  getGPUSpaceTimeKrigingParameterValues,
  type GPUVecchiaCovariance
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
import {
  makeValidationSpaceTimeCovariance,
  VALIDATION_COVARIANCE_PARAMETERS
} from './catalog-validation-vecchia';

export const VALIDATION_KRIGING_BINDING = makeValidationBinding({
  id: 'interpolation-space-time-kriging',
  title: 'Space-time kriging',
  domain: 'interpolation',
  summary:
    'Predicts at explicitly supplied query points and times, with latent variance and solve diagnostics.',
  capabilityIds: ['coverage-local-prediction'],
  contributorNames: ['GPUSpaceTimeKriging'],
  inputs: ['points', 'queries'].map(name =>
    makeInputSpec(
      name,
      name === 'points' ? 'Observed samples' : 'Space-time queries',
      ['point'],
      MAXIMUM_VALIDATION_ROWS,
      MAXIMUM_VALIDATION_ROWS
    )
  ),
  parameters: [
    makeValidationProperty('valueProperty', 'Observed property', 'observed'),
    makeValidationProperty(
      'errorVarianceProperty',
      'Observation error variance property (blank: zero)',
      ''
    ),
    ...VALIDATION_TIME_PARAMETERS,
    ...VALIDATION_COVARIANCE_PARAMETERS,
    makeNumberParameter(
      'anisotropy',
      'Space-time anisotropy (spatial units per second)',
      1,
      1e-6,
      1e9
    ),
    makeRecipeEnumParameter('mode', 'Kriging mean mode', ['ordinary', 'simple'], 'ordinary'),
    makeNumberParameter('mean', 'Known mean (simple mode)', 0, -1e12, 1e12),
    COORDINATE_SYSTEM_PARAMETER
  ],
  outputs: [
    makeOutputSpec(
      'interpolation-space-time-kriging',
      'Space-time kriging',
      'point',
      ['feature_index'],
      MAXIMUM_VALIDATION_ROWS
    ),
    makeOutputSpec(
      'interpolation-space-time-kriging-diagnostics',
      'Space-time kriging diagnostics',
      'none',
      ['metric'],
      16
    )
  ],
  declaredBehavior: [
    ...VALIDATION_BEHAVIOR,
    'Samples and queries share numeric seconds and an integer time origin. Status 0 = success, 1 = invalid or insufficient samples, 2 = singular/unreliable solve. Variance is latent conditional model variance.'
  ],
  limitation:
    'Metric space-time covariance (exponential or Matern 3/2) and metric nearest neighbors only, at most 32. Query points and times are supplied explicitly; the ordinary spatial grid panel remains a separate workflow. Variance ignores parameter fitting uncertainty.',
  runGpu: async ({device}, inputs, values) => {
    const [positions, queries] = readValidationPositions(
      [inputs['points'], inputs['queries']],
      values
    );
    const response = readValidationColumn(inputs['points'], values['valueProperty'] as string);
    const errors = values['errorVarianceProperty']
      ? readValidationColumn(inputs['points'], values['errorVarianceProperty'] as string)
      : new Float32Array(response.length);
    const observations = Float32Array.from(
      Array.from(response, (value, index) => [
        value,
        errors[index],
        0,
        Number.isFinite(value) && Number.isFinite(errors[index]) && errors[index] >= 0 ? 1 : 0
      ]).flat()
    );
    const covariance: GPUVecchiaCovariance = {
      model: values['model'] as 'exponential' | 'matern-3/2',
      sill: values['sill'] as number,
      nugget: values['nugget'] as number,
      range: values['range'] as number
    };
    return withCatalogGpuSession(device, 'catalog-space-time-kriging', async session => {
      const count = inputs['queries'].features.length;
      const output = session.output('values', 'float32x4', count);
      const status = session.output('status', 'uint32', count);
      session.graph.add(
        new GPUSpaceTimeKriging({
          positions: session.input('positions', positions, 'float32x2'),
          queries: session.input('queries', queries, 'float32x2'),
          observations: session.input('observations', observations, 'float32x4'),
          times: session.input('times', readValidationTimes(inputs['points'], values), 'float32'),
          queryTimes: session.input(
            'query-times',
            readValidationTimes(inputs['queries'], values),
            'float32'
          ),
          temporalContext: getValidationTemporalContext(values),
          mode: values['mode'] as 'ordinary' | 'simple',
          neighborCount: values['neighborCount'] as number,
          parameters: session.input(
            'parameters',
            getGPUSpaceTimeKrigingParameterValues({
              covariance: makeValidationSpaceTimeCovariance(
                covariance,
                values['anisotropy'] as number
              ),
              mean: values['mean'] as number
            }),
            'float32'
          ),
          output: {values: output.view, status: status.view}
        })
      );
      session.run();
      const result = await session.readFloat32(output);
      const statuses = await session.readUint32(status);
      return makeValidationResult(
        'interpolation-space-time-kriging',
        'Space-time kriging',
        inputs['queries'],
        {
          ...Object.fromEntries(
            ['prediction', 'variance', 'pivot_ratio', 'retained_neighbors'].map((name, column) => [
              name,
              Float32Array.from({length: count}, (_, row) => result[row * 4 + column])
            ])
          ),
          status: statuses
        },
        {
          predicted_rows: statuses.filter(value => value === 0).length,
          invalid_rows: statuses.filter(value => value === 1).length,
          singular_rows: statuses.filter(value => value === 2).length,
          time_origin: values['timeOrigin'] as number,
          sill: covariance.sill,
          nugget: covariance.nugget ?? 0,
          range: covariance.range,
          anisotropy: values['anisotropy'] as number
        },
        ['GPUSpaceTimeKriging']
      );
    });
  }
});
