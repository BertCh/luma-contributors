// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerDatasetProfileDeclarations} from '@luma-contributors/kepler';

const VALIDATION_SAMPLE_OPERATIONS = [
  'regression-gtwr',
  'regression-mgtwr',
  'interpolation-space-time-kriging',
  'validation-vecchia-likelihood',
  'validation-vecchia-fisher',
  'validation-vecchia-space-time',
  'validation-area-of-applicability',
  'validation-conformal',
  'validation-geo-conformal',
  'validation-ensemble-scores',
  'validation-gaussian-scores',
  'validation-fdr'
];

/** These authored synthetic coordinates are geographic, including forms without a CRS parameter. */
export function getCatalogValidationSampleDeclarations(
  operationId: string
): KeplerDatasetProfileDeclarations | undefined {
  return VALIDATION_SAMPLE_OPERATIONS.includes(operationId)
    ? {coordinateModel: 'geographic'}
    : undefined;
}

/** Reproducible city-scale examples shared by the twelve validation catalog forms. */
export function getCatalogValidationSampleInputs(
  operationId: string
): Record<string, string> | undefined {
  if (!VALIDATION_SAMPLE_OPERATIONS.includes(operationId)) return undefined;
  const makeRows = (query: boolean) =>
    JSON.stringify({
      type: 'FeatureCollection',
      features: Array.from({length: query ? 6 : 40}, (_, index) => {
        const predictor = Math.sin(index * 1.7) + index / 20;
        const predicted = 2 + 1.5 * predictor;
        const observed = predicted + Math.cos(index * 0.8) * 0.4;
        return {
          type: 'Feature',
          id: index,
          geometry: {
            type: 'Point',
            coordinates: [
              -87.65 + ((index % 8) + (query ? 0.3 : 0)) * 0.001,
              41.88 + (Math.floor(index / 8) + (query ? 0.3 : 0)) * 0.001
            ]
          },
          properties: {
            predictor,
            observed,
            predicted,
            variance: 0.2,
            time: index % 6,
            member1: predicted - 0.3,
            member2: predicted,
            member3: predicted + 0.3,
            pvalue: index < 4 ? 0.0001 * (index + 1) : (index + 1) / 41
          }
        };
      })
    });
  const training = makeRows(false);
  return {points: training, training, calibration: training, queries: makeRows(true)};
}
