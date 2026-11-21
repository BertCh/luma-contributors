// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUAreaOfApplicability,
  GPUConformalPredictionIntervals,
  GPUGeoConformalPredictionIntervals,
  decodeConformalStatus,
  buildVecchiaPredictionNeighborhoods,
  getGPUConformalPredictionParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {COORDINATE_SYSTEM_PARAMETER} from './catalog-binding-types';
import {withCatalogGpuSession} from './catalog-gpu-session';
import {makeInputSpec, makeNumberParameter, makeOutputSpec} from './catalog-platform3-b-common';
import {
  makeValidationBinding,
  makeValidationProperty,
  makeValidationPropertyList,
  makeValidationResult,
  MAXIMUM_VALIDATION_ROWS,
  readValidationColumn,
  readValidationMatrix,
  readValidationNames,
  readValidationPositions,
  VALIDATION_BEHAVIOR
} from './catalog-validation-common';

/** Diagnostic counters shared by both bindings; raw row words are published unchanged. */
export function summarizeConformalStatuses(
  statuses: Uint32Array,
  geographic: boolean
): Record<string, number> {
  const summary = {
    unbounded_rows: 0,
    invalid_query_rows: 0,
    no_calibration_rows: 0,
    ...(geographic ? {neighbors_truncated_rows: 0, input_incomplete_rows: 0} : {})
  };
  for (const word of statuses) {
    const status = decodeConformalStatus(word);
    summary.unbounded_rows += Number(status.lowerUnbounded || status.upperUnbounded);
    summary.invalid_query_rows += Number(status.invalidQuery);
    summary.no_calibration_rows += Number(status.noCalibration);
    if (geographic) {
      summary.neighbors_truncated_rows! += Number(status.neighborsTruncated);
      summary.input_incomplete_rows! += Number(status.inputIncomplete);
    }
  }
  return summary;
}

const AREA_BINDING = makeValidationBinding({
  id: 'validation-area-of-applicability',
  title: 'Area of applicability',
  domain: 'analysis',
  summary:
    'Standardizes predictors against training rows and measures query dissimilarity and applicability.',
  capabilityIds: ['coverage-local-prediction'],
  contributorNames: ['GPUAreaOfApplicability'],
  inputs: ['training', 'queries'].map(name =>
    makeInputSpec(
      name,
      name === 'training' ? 'Training rows' : 'Query rows',
      ['point'],
      MAXIMUM_VALIDATION_ROWS,
      MAXIMUM_VALIDATION_ROWS
    )
  ),
  parameters: [
    makeValidationPropertyList(
      'predictorProperties',
      'Predictor properties (comma separated)',
      'predictor',
      16
    )
  ],
  outputs: [
    makeOutputSpec(
      'validation-area-of-applicability',
      'Area of applicability',
      'point',
      ['feature_index'],
      MAXIMUM_VALIDATION_ROWS
    ),
    makeOutputSpec(
      'validation-area-of-applicability-training',
      'Training dissimilarity',
      'point',
      ['feature_index'],
      MAXIMUM_VALIDATION_ROWS
    ),
    makeOutputSpec(
      'validation-area-of-applicability-diagnostics',
      'Area of applicability diagnostics',
      'none',
      ['metric'],
      16
    )
  ],
  declaredBehavior: [
    ...VALIDATION_BEHAVIOR,
    'Training rows are standardized using sample deviations. Threshold = min(Q3 + 1.5 IQR, maximum training dissimilarity); nonzero status reports degenerate training data.'
  ],
  limitation:
    'Equal variable weights and leave-one-out nearest training neighbors; spatial cross-validation folds and variable importance are not supplied. Applicability is predictor support, not calibrated predictive accuracy.',
  runGpu: async ({device}, inputs, values) =>
    withCatalogGpuSession(device, 'catalog-area-of-applicability', async session => {
      const names = readValidationNames(values['predictorProperties'], 16);
      const trainingCount = inputs['training'].features.length;
      const count = inputs['queries'].features.length;
      const trainDissimilarity = session.output('training-dissimilarity', 'float32', trainingCount);
      const queryDissimilarity = session.output('query-dissimilarity', 'float32', count);
      const applicable = session.output('applicable', 'uint32', count);
      const summary = session.output('summary', 'float32', 6);
      const status = session.output('status', 'uint32', 1);
      session.graph.add(
        new GPUAreaOfApplicability({
          training: session.input(
            'training',
            readValidationMatrix(inputs['training'], names),
            'float32'
          ),
          query: session.input(
            'queries',
            readValidationMatrix(inputs['queries'], names),
            'float32'
          ),
          predictorCount: names.length,
          output: {
            trainDissimilarity: trainDissimilarity.view,
            queryDissimilarity: queryDissimilarity.view,
            applicable: applicable.view,
            summary: summary.view,
            status: status.view
          }
        })
      );
      session.run();
      const summaryValues = await session.readFloat32(summary);
      const metrics = Object.fromEntries(
        [
          'mean_distance',
          'threshold',
          'usable_training_rows',
          'scored_training_rows',
          'first_quartile',
          'third_quartile'
        ].map((name, index) => [name, summaryValues[index]])
      );
      metrics['status'] = await session.readScalar(status);
      const result = makeValidationResult(
        'validation-area-of-applicability',
        'Area of applicability',
        inputs['queries'],
        {
          dissimilarity: await session.readFloat32(queryDissimilarity),
          applicable: await session.readUint32(applicable)
        },
        metrics,
        ['GPUAreaOfApplicability']
      );
      const trainingResult = makeValidationResult(
        'validation-area-of-applicability-training',
        'Training dissimilarity',
        inputs['training'],
        {dissimilarity: await session.readFloat32(trainDissimilarity)},
        {},
        []
      );
      result.tables.splice(1, 0, trainingResult.tables[0]);
      return result;
    })
});

const CONFORMAL_BINDINGS = [false, true].map(geographic => {
  const id = geographic ? 'validation-geo-conformal' : 'validation-conformal';
  const contributor = geographic
    ? 'GPUGeoConformalPredictionIntervals'
    : 'GPUConformalPredictionIntervals';
  return makeValidationBinding({
    id,
    title: geographic
      ? 'Geographic conformal prediction intervals'
      : 'Conformal prediction intervals',
    domain: 'analysis',
    summary:
      'Builds prediction intervals from explicitly held-out calibration observations and predictions.',
    capabilityIds: ['coverage-local-prediction'],
    contributorNames: [contributor],
    inputs: ['calibration', 'queries'].map(name =>
      makeInputSpec(
        name,
        name === 'calibration' ? 'Held-out calibration rows' : 'Query predictions',
        ['point'],
        MAXIMUM_VALIDATION_ROWS,
        MAXIMUM_VALIDATION_ROWS
      )
    ),
    parameters: [
      makeValidationProperty('observationProperty', 'Observed property', 'observed'),
      makeValidationProperty('predictionProperty', 'Prediction property', 'predicted'),
      makeValidationProperty('scaleProperty', 'Scale property (blank: unit scale)', ''),
      makeNumberParameter('miscoverage', 'Miscoverage alpha', 0.1, 0.000001, 0.999999),
      ...(geographic
        ? [
            {
              ...makeNumberParameter('neighborCount', 'Calibration neighbors', 64, 1, 256),
              kind: 'integer' as const
            },
            makeNumberParameter('bandwidth', 'Spatial kernel bandwidth', 1000, 1e-6, 1e9, 'crs'),
            makeNumberParameter('selfWeight', 'Query self weight', 1, 0, 1e6),
            COORDINATE_SYSTEM_PARAMETER
          ]
        : [])
    ],
    outputs: [
      makeOutputSpec(
        id,
        'Conformal intervals',
        'point',
        ['feature_index'],
        MAXIMUM_VALIDATION_ROWS
      ),
      makeOutputSpec(`${id}-diagnostics`, 'Conformal intervals diagnostics', 'none', ['metric'], 16)
    ],
    declaredBehavior: [
      ...VALIDATION_BEHAVIOR,
      'Calibration predictions must come from a model that did not train on these calibration rows. Optional positive scale values normalize residuals and scale query intervals. Status bits: 1 lower unbounded, 2 upper unbounded, 4 invalid query, 8 no calibration, 16 neighbors truncated.',
      ...(geographic
        ? [
            'Geographic status bit 32 input incomplete reports loss in upstream calibration weights. input_incomplete_rows counts these query rows; it is independent of neighbors_truncated_rows, which counts internal truncation (bit 16). A row can contribute to both counters.'
          ]
        : [])
    ],
    limitation: geographic
      ? 'Gaussian weights over at most 256 nearest usable calibration points. Local weighting does not supply a distribution-free spatial coverage guarantee; unbounded endpoints publish null with explicit status bits.'
      : 'Split holdout conformal only (one model). Exchangeability is required for marginal coverage. Unbounded endpoints publish null with explicit status bits; no in-sample fitting is performed.',
    runGpu: async ({device}, inputs, values) =>
      withCatalogGpuSession(device, id, async session => {
        const calibrationCollection = inputs['calibration'];
        const queryCollection = inputs['queries'];
        const observations = readValidationColumn(
          calibrationCollection,
          values['observationProperty'] as string
        );
        const predictions = readValidationColumn(
          calibrationCollection,
          values['predictionProperty'] as string
        );
        const scales = values['scaleProperty']
          ? readValidationColumn(calibrationCollection, values['scaleProperty'] as string)
          : undefined;
        const calibration = {
          observations: session.input('observations', observations, 'float32'),
          predictions: session.input('calibration-predictions', predictions, 'float32'),
          ...(scales ? {scales: session.input('calibration-scales', scales, 'float32')} : {})
        };
        const query = {
          predictions: session.input(
            'query-predictions',
            readValidationColumn(queryCollection, values['predictionProperty'] as string),
            'float32'
          ),
          ...(scales
            ? {
                scales: session.input(
                  'query-scales',
                  readValidationColumn(queryCollection, values['scaleProperty'] as string),
                  'float32'
                )
              }
            : {})
        };
        const intervals = session.output(
          'intervals',
          'float32',
          queryCollection.features.length * 2
        );
        const status = session.output('status', 'uint32', queryCollection.features.length);
        const calibrationCount = session.output('calibration-count', 'uint32', 1);
        const parameters = session.input(
          'parameters',
          getGPUConformalPredictionParameterValues({
            miscoverage: values['miscoverage'] as number,
            selfWeight: geographic ? (values['selfWeight'] as number) : 1
          }),
          'float32'
        );
        const usable = Uint32Array.from(observations, (value, index) =>
          Number.isFinite(value) &&
          Number.isFinite(predictions[index]) &&
          (!scales || (Number.isFinite(scales[index]) && scales[index] > 0))
            ? 1
            : 0
        );
        if (geographic) {
          const [positions, queries] = readValidationPositions(
            [calibrationCollection, queryCollection],
            values
          );
          const neighbors = buildVecchiaPredictionNeighborhoods({
            positions,
            queries,
            neighborCount: values['neighborCount'] as number,
            validity: usable
          });
          const weights = new Float32Array(neighbors.neighbors.length);
          for (let row = 0; row < queryCollection.features.length; row++)
            for (let entry = neighbors.offsets[row]; entry < neighbors.offsets[row + 1]; entry++) {
              const neighbor = neighbors.neighbors[entry];
              const distance = Math.hypot(
                queries[row * 2] - positions[neighbor * 2],
                queries[row * 2 + 1] - positions[neighbor * 2 + 1]
              );
              weights[entry] = Math.exp(-0.5 * (distance / (values['bandwidth'] as number)) ** 2);
            }
          session.graph.add(
            new GPUGeoConformalPredictionIntervals({
              calibration,
              query,
              parameters,
              neighbors: {
                offsets: session.input('offsets', neighbors.offsets, 'uint32'),
                indices: session.input('indices', neighbors.neighbors, 'uint32'),
                weights: session.input('weights', weights, 'float32')
              },
              output: {intervals: intervals.view, status: status.view}
            })
          );
        } else
          session.graph.add(
            new GPUConformalPredictionIntervals({
              calibration,
              query,
              parameters,
              output: {
                intervals: intervals.view,
                status: status.view,
                calibrationCount: calibrationCount.view
              }
            })
          );
        session.run();
        const endpoints = await session.readFloat32(intervals);
        const statuses = await session.readUint32(status);
        return makeValidationResult(
          id,
          'Conformal intervals',
          queryCollection,
          {
            lower: Float32Array.from({length: statuses.length}, (_, index) => endpoints[index * 2]),
            upper: Float32Array.from(
              {length: statuses.length},
              (_, index) => endpoints[index * 2 + 1]
            ),
            status: statuses
          },
          {
            calibration_count: geographic
              ? usable.reduce((sum, value) => sum + value, 0)
              : await session.readScalar(calibrationCount),
            miscoverage: values['miscoverage'] as number,
            ...summarizeConformalStatuses(statuses, geographic)
          },
          [contributor]
        );
      })
  });
});

export const VALIDATION_PREDICTION_BINDINGS = [AREA_BINDING, ...CONFORMAL_BINDINGS];
