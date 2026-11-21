// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUColumnQuantilesOperation} from '../../../src/dataframe-analysis/column-classification/gpu-column-quantiles-operation';
import {getGPUColumnQuantilesParameterValues} from '../../../src/dataframe-analysis/column-classification/column-quantiles-parameters';
import {poisonOperation} from '../../analysis-core/mixed-execution/output-oracles-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('quantile optional filter ports use literal trimmed order statistics independently', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const optional of ['both', 'mask', 'bounds', 'neither']) {
    const output = {
      quantiles: 'quantiles',
      validCount: 'validCount',
      ...(['both', 'mask'].includes(optional) ? {outputMask: 'outputMask'} : {}),
      ...(['both', 'bounds'].includes(optional) ? {filterBounds: 'filterBounds'} : {})
    };
    const rows = await runOperationGraph(device, {
      inputs: {
        values: {format: 'float32', values: new Float32Array([9, -4, 0, 2, 8, 100, NaN, Infinity])},
        mask: {format: 'uint32', values: new Uint32Array([0, 1, 1, 1, 1, 0, 1, 1])},
        parameters: {
          format: 'float32',
          values: getGPUColumnQuantilesParameterValues({
            quantiles: [0.25, 0.75],
            interpolation: 'linear',
            filterRange: [0.25, 0.75]
          })
        }
      },
      operations: [
        poisonOperation(
          new GPUColumnQuantilesOperation({
            values: 'values',
            mask: 'mask',
            parameters: 'parameters',
            rowCount: 8,
            quantileCount: 2,
            output
          })
        )
      ],
      outputs: Object.values(output)
    });
    // Sorted sample [-4,0,2,8]: interpolated ranks .75,2.25 give [-1,3.5].
    // Filter uses floor(n*lower), ceil(n*upper)-1 = 1,2, hence [0,2], not quantiles.
    expect(Array.from(rows['quantiles'])).toEqual([-1, 3.5]);
    expect(Array.from(rows['validCount'])).toEqual([4]);
    if (output.outputMask) expect(Array.from(rows['outputMask'])).toEqual([0, 0, 1, 1, 0, 0, 0, 0]);
    if (output.filterBounds) expect(Array.from(rows['filterBounds'])).toEqual([0, 2]);
  }
});
