// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPURasterMedoidCompositeOperation} from '../../../src/raster-analysis/raster-time-series/cpu-raster-medoid-composite-operation';
import {
  GPURasterMedoidCompositeOperation,
  type RasterMedoidCompositeOperationProps
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-medoid-composite-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const missing of [false, true]) {
  it(`compares every medoid output with missing=${missing} (tolerance zero)`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: RasterMedoidCompositeOperationProps = {
      cellCount: 1,
      sliceCount: 4,
      bandCount: 2,
      values: 'values',
      mask: 'mask',
      output: {composite: 'composite', sliceIndex: 'sliceIndex', status: 'status'}
    };
    const outputs = ['composite', 'sliceIndex', 'status'];
    const results = [];
    for (const Operation of [GPURasterMedoidCompositeOperation, CPURasterMedoidCompositeOperation])
      results.push(
        await runOperationGraph(device, {
          inputs: {
            values: {format: 'float32', values: new Float32Array([0, 0, 3, 0, 0, 4, 1, 1])},
            mask: {format: 'uint32', values: new Uint32Array(4).fill(missing ? 0 : 9)}
          },
          operations: [new Operation(props)],
          outputs
        })
      );
    for (const output of outputs)
      expect(Array.from(results[0][output])).toEqual(Array.from(results[1][output]));
  });
}
