// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUSpatialWeightsTracesOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-weights-traces-operation';
import {GPUSpatialWeightsTracesOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces-operation';
import {getGPUSpatialWeightsTracesParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-parameters';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {expectRegressionRowsClose} from './regression-operation-test-utils';

it('runs the exact two-row swap through two separate CSR chunks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const names = ['traces', 'sums', 'errors', 'norm', 'approximation'];
  const outputs: Awaited<ReturnType<typeof runOperationGraph>>[] = [];
  for (const Operation of [GPUSpatialWeightsTracesOperation, CPUSpatialWeightsTracesOperation]) {
    outputs.push(
      await runOperationGraph(device, {
        inputs: {
          offsets0: {format: 'uint32', values: Uint32Array.of(0, 1)},
          neighbors0: {format: 'uint32', values: Uint32Array.of(1)},
          weights0: {format: 'float32', values: Float32Array.of(1)},
          offsets1: {format: 'uint32', values: Uint32Array.of(0, 1)},
          neighbors1: {format: 'uint32', values: Uint32Array.of(0)},
          weights1: {format: 'float32', values: Float32Array.of(1)}
        },
        operations: [
          new Operation({
            rowCount: 2,
            maximumPower: 2,
            mode: 'exact',
            weights: {
              rowCount: 2,
              chunks: [
                {
                  rowStart: 0,
                  rowCount: 1,
                  weights: {offsets: 'offsets0', neighbors: 'neighbors0', weights: 'weights0'}
                },
                {
                  rowStart: 1,
                  rowCount: 1,
                  weights: {offsets: 'offsets1', neighbors: 'neighbors1', weights: 'weights1'}
                }
              ]
            },
            output: {
              powerTraces: 'traces',
              powerSums: 'sums',
              powerTraceStandardErrors: 'errors',
              infinityNorm: 'norm',
              status: {approximation: 'approximation'}
            }
          })
        ],
        outputs: names
      })
    );
  }
  for (const result of outputs) {
    expect(Array.from(result['traces'])).toEqual([0, 2]);
    expect(Array.from(result['sums'])).toEqual([2, 2]);
    expect(Array.from(result['errors'])).toEqual([0, 0]);
    expect(Array.from(result['norm'])).toEqual([1]);
    expect(Array.from(result['approximation'])).toEqual([0]);
  }
  for (const name of names) expectRegressionRowsClose(outputs[0][name], outputs[1][name], 0);
});

for (const scene of ['exact', 'stochastic', 'single-probe', 'zero-links'] as const) {
  it(`trace operations agree on ${scene}: relative tolerance 2e-4, integer flags exact`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const outputs = [];
    const names = ['traces', 'errors', 'sums', 'gram', 'norm', 'approximation'];
    for (const Operation of [GPUSpatialWeightsTracesOperation, CPUSpatialWeightsTracesOperation]) {
      outputs.push(
        await runOperationGraph(device, {
          inputs: {
            offsets: {
              format: 'uint32',
              values: scene === 'zero-links' ? new Uint32Array(4) : Uint32Array.of(0, 3, 4, 5)
            },
            neighbors: {format: 'uint32', values: Uint32Array.of(0, 1, 99, 2, 0)},
            weights: {format: 'float32', values: Float32Array.of(1, 2, 999, 3, 4)},
            parameters: {
              format: 'uint32',
              values: getGPUSpatialWeightsTracesParameterValues({seed: 11})
            }
          },
          operations: [
            new Operation({
              rowCount: 3,
              maximumPower: 4,
              mode: scene === 'stochastic' || scene === 'single-probe' ? 'stochastic' : 'exact',
              probeCount: scene === 'single-probe' ? 1 : 8,
              probeBatchSize: 2,
              weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
              parameters: 'parameters',
              output: {
                powerTraces: 'traces',
                powerTraceStandardErrors: 'errors',
                powerSums: 'sums',
                gramTraces: 'gram',
                infinityNorm: 'norm',
                status: {approximation: 'approximation'}
              }
            })
          ],
          outputs: names
        })
      );
    }
    for (const name of names.slice(0, -1))
      expectRegressionRowsClose(outputs[0][name], outputs[1][name], 2e-4);
    expect(Array.from(outputs[0]['approximation'])).toEqual(
      Array.from(outputs[1]['approximation'])
    );
  });
}
