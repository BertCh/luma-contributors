// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {expect, it} from 'vitest';
import {CPUCostDistancePathOperation} from '../../../src/raster-analysis/cost-distance/cpu-cost-distance-path-operation';
import {
  GPUCostDistancePathOperation,
  type CostDistancePathOperationProps
} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance-path-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const capacity of [3, 2]) {
  it(`path operations agree on every output with capacity ${capacity} (integer tolerance zero)`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: CostDistancePathOperationProps = {
      grid: makeGPURasterMetadata({width: 3, height: 2}),
      backLinks: 'backLinks',
      target: 'target',
      capacity,
      output: {
        path: {
          ids: 'ids',
          status: {count: 'count', overflow: 'overflow', requiredCount: 'requiredCount'}
        },
        status: {iterationLimitReached: 'iterationLimitReached'}
      }
    };
    const outputs = ['ids', 'count', 'overflow', 'requiredCount', 'iterationLimitReached'];
    const results = [];
    for (const Operation of [GPUCostDistancePathOperation, CPUCostDistancePathOperation]) {
      results.push(
        await runOperationGraph(device, {
          inputs: {
            backLinks: {
              format: 'uint32',
              values: new Uint32Array([0, 16, 0xffffffff, 0xffffffff, 0xffffffff, 32])
            },
            target: {format: 'uint32', values: new Uint32Array([5])}
          },
          operations: [new Operation(props)],
          outputs
        })
      );
    }
    for (const output of outputs)
      expect(Array.from(results[0][output])).toEqual(Array.from(results[1][output]));
    expect(results[0]['count'][0]).toBe(capacity);
    expect(results[0]['requiredCount'][0]).toBe(3);
    expect(results[0]['overflow'][0]).toBe(Number(capacity < 3));
  });
}
