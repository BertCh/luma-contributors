// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUPettittTestOperation} from '../../../src/gpu-spatial-analysis/change-points/cpu-pettitt-test-operation';
import {GPUPettittTestOperation} from '../../../src/gpu-spatial-analysis/change-points/gpu-pettitt-test-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const layout of ['cell-major', 'period-major'] as const) {
  for (const edgeCase of [false, true]) {
    it(`Pettitt graph twins agree in ${layout}, edge=${edgeCase}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const series = edgeCase
        ? [
            [NaN, NaN, NaN, NaN],
            [7, NaN, NaN, NaN],
            [2, 2, NaN, 2]
          ]
        : [
            [1, 1, 8, 8],
            [4, 1, 3, 2],
            [3, NaN, 7, 9]
          ];
      const values = Float32Array.from(
        layout === 'cell-major'
          ? series.flat()
          : Array.from({length: 4}, (_, slice) => series.map(row => row[slice])).flat()
      );
      const props = {
        values: 'values',
        layout,
        cellCount: 3,
        sliceCount: 4,
        output: {statistic: 'statistic', changeSlice: 'changeSlice', pValues: 'pValues'}
      };
      const results = [];
      for (const Operation of [GPUPettittTestOperation, CPUPettittTestOperation]) {
        results.push(
          await runOperationGraph(device, {
            inputs: {values: {format: 'float32', values}},
            operations: [new Operation(props)],
            outputs: Object.values(props.output)
          })
        );
      }
      for (const name of ['statistic', 'changeSlice'])
        expect(Array.from(results[0][name])).toEqual(Array.from(results[1][name]));
      // Integer outputs are exact; the exponential p-value allows 2e-6 absolute error.
      for (let cell = 0; cell < 3; cell++)
        expect(
          Math.abs(results[0]['pValues'][cell] - results[1]['pValues'][cell])
        ).toBeLessThanOrEqual(2e-6);
    });
  }
}
