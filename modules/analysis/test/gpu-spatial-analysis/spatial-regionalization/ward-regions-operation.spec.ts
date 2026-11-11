// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUWardRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-ward-regions-operation';
import {GPUWardRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions-operation';
import {getGPUWardRegionsParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions';
import {getCSR} from './ward-reference';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const mergeSchedule of ['sequential', 'mutual-nearest'] as const) {
  for (const scene of ['ordinary', 'disconnected', 'invalid', 'no-merges'] as const) {
    for (const standardize of [false, true]) {
      it(`Ward graph twins agree: ${mergeSchedule}, ${scene}, standardize=${standardize}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const pattern = getCSR(
          scene === 'disconnected' ? [[1], [0], [3], [2]] : [[1], [0, 2], [1, 3], [2]]
        );
        const values = Float32Array.of(0, 5, 1, 5, 10, 5, 11, scene === 'invalid' ? NaN : 5);
        const targetRegionCount = scene === 'no-merges' ? 4 : scene === 'disconnected' ? 1 : 2;
        const props = {
          weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          rowCount: 4,
          values: 'values',
          parameters: 'parameters',
          columnCount: 2,
          standardize,
          mergeSchedule,
          statusLength: mergeSchedule === 'mutual-nearest' ? (5 as const) : (4 as const),
          targetRegionCount,
          maximumIterations: 4 - targetRegionCount,
          output: {labels: 'labels', objective: 'objective', status: 'status'}
        };
        const results = [];
        for (const Operation of [GPUWardRegionsOperation, CPUWardRegionsOperation]) {
          results.push(
            await runOperationGraph(device, {
              inputs: {
                offsets: {format: 'uint32', values: pattern.offsets},
                neighbors: {format: 'uint32', values: pattern.neighbors},
                weights: {format: 'float32', values: pattern.weights},
                values: {format: 'float32', values},
                parameters: {format: 'uint32', values: getGPUWardRegionsParameterValues({seed: 7})}
              },
              operations: [new Operation(props)],
              outputs: ['labels', 'objective', 'status']
            })
          );
        }
        expect(Array.from(results[0]['labels'])).toEqual(Array.from(results[1]['labels']));
        expect(Array.from(results[0]['status'])).toEqual(Array.from(results[1]['status']));
        // Every output: exact labels/status; objective allows 1e-4 relative (absolute floor 1e-4).
        expect(
          Math.abs(results[0]['objective'][0] - results[1]['objective'][0])
        ).toBeLessThanOrEqual(1e-4 * Math.max(1, Math.abs(results[1]['objective'][0])));
      });
    }
  }
}
