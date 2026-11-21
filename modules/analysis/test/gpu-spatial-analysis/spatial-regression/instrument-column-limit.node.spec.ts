// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_MAXIMUM_CENTERED_INSTRUMENT_COLUMN_COUNT,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_MAXIMUM_PREDICTOR_COUNT,
  getGPUSpatialTwoStageLeastSquaresMaximumPredictorCount
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-two-stage-least-squares-parameters';

it('2SLS retains order-specific maxima within sixteen centered instrument columns', () => {
  expect(GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_MAXIMUM_CENTERED_INSTRUMENT_COLUMN_COUNT).toBe(16);
  expect(GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_MAXIMUM_PREDICTOR_COUNT).toBe(8);
  // Order one: floor(16 / 2) = 8. Order two: floor(16 / 3) = 5.
  expect(getGPUSpatialTwoStageLeastSquaresMaximumPredictorCount(1)).toBe(8);
  expect(getGPUSpatialTwoStageLeastSquaresMaximumPredictorCount(2)).toBe(5);
});
