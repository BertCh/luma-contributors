// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {GPURasterFractionalZonalDistributionOperation} from '../../../src/raster-analysis/raster-zonal-statistics/gpu-raster-fractional-zonal-distribution-operation';
import {
  expectLiteralRows,
  poisonOperation
} from '../../analysis-core/mixed-execution/output-oracles-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  DISTRIBUTION_PROPS,
  createUnitDistributionInput,
  getDistributionGraphInputs
} from './raster-fractional-zonal-distribution-operation-fixture';

it('weighted zonal operation writes independent literal variance, weight and quantile validity ports', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const input = createUnitDistributionInput(new Float32Array([2, 6]), new Float32Array([0.5, NaN]));
  input.observationWeights = new Float32Array([1, 3]);
  const rows = await runOperationGraph(device, {
    inputs: getDistributionGraphInputs(input),
    operations: [
      poisonOperation(
        new GPURasterFractionalZonalDistributionOperation({
          ...DISTRIBUTION_PROPS,
          grid: makeGPURasterMetadata({width: 2, height: 1}),
          probabilityCount: 2,
          quantileMethod: 'exactextract',
          cellSizeMode: 'uniform'
        })
      )
    ],
    outputs: Object.values(DISTRIBUTION_PROPS.output)
  });
  // W=4, W2=10, mean=5, weighted centered sum of squares=12.
  // Reliability correction divides by W-W2/W=1.5, not W-1.
  const expected = {
    means: [5],
    populationVariances: [3],
    sampleVariances: [8],
    effectiveSampleSizes: [1.6],
    weightSums: [4],
    squaredWeightSums: [10],
    coveredArea: [2],
    validArea: [2],
    quantiles: [4, NaN],
    quantileValidity: [1, 0],
    validity: [1],
    status: [0]
  };
  for (const [name, literals] of Object.entries(expected)) expectLiteralRows(rows[name], literals);
});
