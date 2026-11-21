// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUColumnQuantilesOperation} from '../../../src/dataframe-analysis/column-classification/gpu-column-quantiles-operation';
import {CPUColumnQuantilesOperation} from '../../../src/dataframe-analysis/column-classification/cpu-column-quantiles-operation';
import {constructRasterOperationContributor} from '../../utils/raster-operation-test-utils';
import {makeCPUInputRows} from '../../analysis-core/mixed-execution/output-oracles-test-utils';
import {getGPUColumnQuantilesParameterValues} from '../../../src/dataframe-analysis/column-classification/column-quantiles-parameters';

const props = {
  id: 'quantile-shape',
  values: 'values',
  parameters: 'parameters',
  rowCount: 2,
  quantileCount: 1,
  output: {quantiles: 'quantiles', validCount: 'count', filterBounds: 'bounds'}
};

it('both quantile forms reject a population longer than rowCount without outputMask', () => {
  expect(() =>
    constructRasterOperationContributor(new GPUColumnQuantilesOperation(props), {
      values: {format: 'float32', length: 3},
      parameters: {format: 'float32', length: 5}
    })
  ).toThrow('quantile-shape values length must equal rowCount');
  expect(() =>
    new CPUColumnQuantilesOperation(props).execute({
      inputs: {
        values: makeCPUInputRows('float32', new Float32Array([1, 3, 100])),
        parameters: makeCPUInputRows(
          'float32',
          getGPUColumnQuantilesParameterValues({quantiles: [0.5]})
        )
      }
    })
  ).toThrow('quantile-shape values length must equal rowCount');
});

it('both quantile constructors reject invalid rowCount without an outputMask', () => {
  for (const rowCount of [0, -1, 1.5, NaN, Infinity])
    for (const Operation of [GPUColumnQuantilesOperation, CPUColumnQuantilesOperation])
      expect(() => new Operation({...props, rowCount})).toThrow(/quantile-shape rowCount/);
});
