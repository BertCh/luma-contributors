// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {computeRasterFractionalZonalDistributionOnCPU} from '../../../src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-distribution-operation';
import {expectLiteralRows} from '../../analysis-core/mixed-execution/output-oracles-test-utils';
import {createUnitDistributionInput} from './raster-fractional-zonal-distribution-operation-fixture';

it('weighted zonal CPU moments use the independent reliability-weight denominator', () => {
  const input = createUnitDistributionInput(new Float32Array([2, 6]), new Float32Array([0.5, NaN]));
  input.observationWeights = new Float32Array([1, 3]);
  const result = computeRasterFractionalZonalDistributionOnCPU(input, {
    grid: makeGPURasterMetadata({width: 2, height: 1}),
    zoneCount: 1,
    quantileMethod: 'exactextract',
    cellSizeMode: 'uniform'
  });
  // mean=5, M2=1*(2-5)^2+3*(6-5)^2=12; W=4, W2=10.
  // population=12/4=3; sample=12/(4-10/4)=8, effective n=16/10.
  expectLiteralRows(result.populationVariances, [3], 0);
  expectLiteralRows(result.sampleVariances, [8], 0);
  expectLiteralRows(result.weightSums, [4], 0);
  expectLiteralRows(result.squaredWeightSums, [10], 0);
  expectLiteralRows(result.quantileValidity, [1, 0], 0);
});
