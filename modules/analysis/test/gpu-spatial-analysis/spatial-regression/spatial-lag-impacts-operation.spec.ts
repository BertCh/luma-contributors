// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {CPUSpatialLagImpactsOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-lag-impacts-operation';
import {GPUSpatialLagImpactsOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-impacts-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {expectRegressionRowsClose} from './regression-operation-test-utils';

for (const scene of ['ordinary', 'negative', 'boundary', 'failed', 'zero'] as const) {
  it(`impacts operations agree on ${scene} inputs (relative tolerance 3e-6)`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const outputs = [];
    for (const Operation of [GPUSpatialLagImpactsOperation, CPUSpatialLagImpactsOperation]) {
      outputs.push(
        await runOperationGraph(device, {
          inputs: {
            coefficients: {
              format: 'float32',
              values: Float32Array.of(
                9,
                99,
                2,
                99,
                -3,
                99,
                scene === 'boundary'
                  ? 0.5
                  : scene === 'zero'
                    ? 0
                    : scene === 'negative'
                      ? -0.25
                      : 0.25
              )
            },
            traces: {format: 'float32', values: Float32Array.of(0, 2, 0, 2, 0, 2, 0, 2)},
            sums: {format: 'float32', values: new Float32Array(8).fill(2)},
            norm: {format: 'float32', values: Float32Array.of(2)},
            status: {format: 'uint32', values: Uint32Array.of(scene === 'failed' ? 1 : 0)}
          },
          operations: [
            new Operation({
              coefficientStride: 2,
              rhoIndex: 3,
              predictorCount: 2,
              rowCount: 2,
              maximumPower: 8,
              coefficients: 'coefficients',
              traces: {powerTraces: 'traces', powerSums: 'sums', infinityNorm: 'norm'},
              status: 'status',
              output: {impacts: 'impacts'}
            })
          ],
          outputs: ['impacts']
        })
      );
    }
    expectRegressionRowsClose(outputs[0]['impacts'], outputs[1]['impacts'], 3e-6);
  });
}
