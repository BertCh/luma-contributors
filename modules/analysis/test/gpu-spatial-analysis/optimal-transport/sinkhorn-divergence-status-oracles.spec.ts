// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSinkhornDivergenceOperation} from '../../../src/gpu-spatial-analysis/optimal-transport/gpu-sinkhorn-divergence-operation';
import {getGPUSinkhornDivergenceParameterValues} from '../../../src/gpu-spatial-analysis/optimal-transport/sinkhorn-divergence-parameters';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('Sinkhorn operation counts invalid masses in both distributions independently', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const dirty of [true, false]) {
    // Three invalid source cells (NaN, negative, infinity) plus two invalid target cells.
    // Replacing them with zero leaves a unit mass at the same cell in both distributions.
    const result = await runOperationGraph(device, {
      inputs: {
        source: {
          format: 'float32',
          values: Float32Array.from(dirty ? [1, NaN, -2, Infinity] : [1, 0, 0, 0])
        },
        target: {
          format: 'float32',
          values: Float32Array.from(dirty ? [1, -Infinity, 0, NaN] : [1, 0, 0, 0])
        },
        parameters: {
          format: 'float32',
          values: getGPUSinkhornDivergenceParameterValues({cellWidth: 1, cellHeight: 1, epsilon: 1})
        }
      },
      outputs: ['divergence', 'invalid'],
      operations: [
        new GPUSinkhornDivergenceOperation({
          grid: {width: 4, height: 1, pixelInterpretation: 'area', affine: [1, 0, 0, 0, -1, 0]},
          source: 'source',
          target: 'target',
          parameters: 'parameters',
          maximumIterations: 2,
          output: {divergence: 'divergence', status: {invalidCount: 'invalid'}}
        })
      ]
    });
    expect(Array.from(result['invalid'])).toEqual([dirty ? 5 : 0]);
    expect(result['divergence'][0]).toBe(0);
  }
});
