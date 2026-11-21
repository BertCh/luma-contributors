// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPURasterMedoidCompositeOperation} from '../../../src/raster-analysis/raster-time-series/gpu-raster-medoid-composite-operation';
import {
  expectLiteralRows,
  poisonOperation
} from '../../analysis-core/mixed-execution/output-oracles-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('medoid operation copies literal winning bands and publishes slice provenance and missing status', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const missing of [false, true]) {
    // Four Euclidean distance sums: 8.414214,10.236068,12.162278,6.812559 => slice 3.
    const rows = await runOperationGraph(device, {
      inputs: {
        values: {format: 'float32', values: new Float32Array([0, 0, 3, 0, 0, 4, 1, 1])},
        mask: {format: 'uint32', values: new Uint32Array(4).fill(missing ? 0 : 1)}
      },
      operations: [
        poisonOperation(
          new GPURasterMedoidCompositeOperation({
            cellCount: 1,
            sliceCount: 4,
            bandCount: 2,
            values: 'values',
            mask: 'mask',
            output: {composite: 'composite', sliceIndex: 'sliceIndex', status: 'status'}
          })
        )
      ],
      outputs: ['composite', 'sliceIndex', 'status']
    });
    expectLiteralRows(rows['composite'], missing ? [NaN, NaN] : [1, 1], 0);
    expectLiteralRows(rows['sliceIndex'], missing ? [0xffffffff] : [3], 0);
    expectLiteralRows(rows['status'], missing ? [1] : [0], 0);
  }
});
